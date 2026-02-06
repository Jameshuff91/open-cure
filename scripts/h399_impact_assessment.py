#!/usr/bin/env python3
"""
h399 Impact Assessment: What happens if we move hierarchy/cv_pathway
checks BEFORE the rank>20 filter?

Calculate tier precision before and after the change.
"""

import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.production_predictor import (
    DrugRepurposingPredictor, ConfidenceTier,
    DISEASE_HIERARCHY_GROUPS,
)


def main():
    print("=" * 80)
    print("h399 Impact Assessment: Hierarchy Before Rank>20")
    print("=" * 80)

    predictor = DrugRepurposingPredictor()
    gt_data = predictor.ground_truth
    gt_diseases = [d for d in gt_data if len(gt_data[d]) > 0 and d in predictor.embeddings]

    # Collect current tier stats
    current_tiers = defaultdict(lambda: [0, 0])  # [total, gt]

    # Proposed change: hierarchy checks before rank>20
    proposed_tiers = defaultdict(lambda: [0, 0])  # [total, gt]

    # Track what changes
    promoted = 0
    promoted_gt = 0

    # Also check cancer_same_type + HIGH criteria
    cancer_promoted = 0
    cancer_promoted_gt = 0
    cancer_current_tiers = defaultdict(lambda: [0, 0])
    cancer_proposed_tiers = defaultdict(lambda: [0, 0])

    # Also check zero_precision_mismatch
    zpm_total = 0
    zpm_gt = 0

    HIERARCHY_GOLDEN_CATEGORIES = {'metabolic', 'neurological'}
    HIERARCHY_DEMOTE_TO_HIGH = {'thyroid'}
    HIERARCHY_DEMOTE_TO_MEDIUM = {'parkinsons', 'migraine'}

    for idx, disease_id in enumerate(gt_diseases):
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        gt_drugs = gt_data[disease_id]
        category = predictor.categorize_disease(disease_name)

        try:
            result = predictor.predict(disease_name, top_n=30, include_filtered=True)
        except Exception:
            continue

        for pred in result.predictions:
            is_gt = pred.drug_id in gt_drugs
            tier = pred.confidence_tier

            # Current tier
            current_tiers[tier.value][0] += 1
            if is_gt:
                current_tiers[tier.value][1] += 1

            # Proposed: Does this prediction get promoted?
            new_tier = tier  # Start with current

            # Check: Is this a rank>20 FILTER that would be rescued by hierarchy?
            if tier == ConfidenceTier.FILTER and pred.rank > 20:
                # Check hierarchy match
                if category in DISEASE_HIERARCHY_GROUPS and pred.drug_id:
                    _, same_group, matching_group = predictor._check_disease_hierarchy_match(
                        pred.drug_id, disease_name, category
                    )
                    if same_group:
                        # Assign based on hierarchy rules (same logic as _assign_confidence_tier)
                        if matching_group in HIERARCHY_DEMOTE_TO_MEDIUM:
                            new_tier = ConfidenceTier.MEDIUM
                        elif matching_group in HIERARCHY_DEMOTE_TO_HIGH:
                            new_tier = ConfidenceTier.HIGH
                        elif category in HIERARCHY_GOLDEN_CATEGORIES:
                            new_tier = ConfidenceTier.GOLDEN
                        else:
                            new_tier = ConfidenceTier.HIGH
                        promoted += 1
                        if is_gt:
                            promoted_gt += 1

                # Check cv_pathway_comprehensive
                if predictor._is_cv_complication(disease_name) and predictor._is_cv_pathway_comprehensive(pred.drug_name):
                    if new_tier == ConfidenceTier.FILTER:  # Only if not already promoted
                        new_tier = ConfidenceTier.HIGH
                        promoted += 1
                        if is_gt:
                            promoted_gt += 1

            proposed_tiers[new_tier.value][0] += 1
            if is_gt:
                proposed_tiers[new_tier.value][1] += 1

    # Print results
    print(f"\nPromoted predictions: {promoted}")
    print(f"Promoted GT hits: {promoted_gt}")

    print(f"\n{'Tier':<10} {'Current':>20} {'Proposed':>20} {'Delta':>10}")
    print(f"{'':10} {'Total':>8} {'GT':>5} {'Prec':>7} {'Total':>8} {'GT':>5} {'Prec':>7} {'ΔPrec':>10}")
    print("-" * 80)

    for tier_name in ['GOLDEN', 'HIGH', 'MEDIUM', 'LOW', 'FILTER']:
        ct, cg = current_tiers[tier_name]
        pt, pg = proposed_tiers[tier_name]
        c_prec = 100 * cg / ct if ct > 0 else 0
        p_prec = 100 * pg / pt if pt > 0 else 0
        delta = p_prec - c_prec
        print(f"{tier_name:<10} {ct:>8} {cg:>5} {c_prec:>6.1f}% {pt:>8} {pg:>5} {p_prec:>6.1f}% {delta:>+9.1f}pp")

    total_ct = sum(v[0] for v in current_tiers.values())
    total_cg = sum(v[1] for v in current_tiers.values())
    total_pt = sum(v[0] for v in proposed_tiers.values())
    total_pg = sum(v[1] for v in proposed_tiers.values())
    print(f"{'TOTAL':<10} {total_ct:>8} {total_cg:>5} {100*total_cg/total_ct:>6.1f}% {total_pt:>8} {total_pg:>5} {100*total_pg/total_pt:>6.1f}%")

    print(f"\n{'='*80}")
    print("CONCLUSION")
    print(f"{'='*80}")
    # Net impact
    for tier_name in ['GOLDEN', 'HIGH', 'MEDIUM']:
        ct, cg = current_tiers[tier_name]
        pt, pg = proposed_tiers[tier_name]
        c_prec = 100 * cg / ct if ct > 0 else 0
        p_prec = 100 * pg / pt if pt > 0 else 0
        delta = p_prec - c_prec
        if abs(delta) > 0.5:
            direction = "IMPROVES" if delta > 0 else "HURTS"
            print(f"  {tier_name}: {direction} by {abs(delta):.1f}pp ({c_prec:.1f}% → {p_prec:.1f}%)")

    print(f"\n{'='*80}")
    print("DONE")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
