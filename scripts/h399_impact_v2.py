#!/usr/bin/env python3
"""
h399 Impact v2: Hierarchy at rank>20 → cap at HIGH (never GOLDEN).
Also test cancer_same_type + HIGH criteria → promote to HIGH.
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
    print("h399 Impact v2: Hierarchy→HIGH-capped + cancer_same_type+HIGH→HIGH")
    print("=" * 80)

    predictor = DrugRepurposingPredictor()
    gt_data = predictor.ground_truth
    gt_diseases = [d for d in gt_data if len(gt_data[d]) > 0 and d in predictor.embeddings]

    HIERARCHY_GOLDEN_CATEGORIES = {'metabolic', 'neurological'}
    HIERARCHY_DEMOTE_TO_HIGH = {'thyroid'}
    HIERARCHY_DEMOTE_TO_MEDIUM = {'parkinsons', 'migraine'}

    # Test 3 scenarios
    scenarios = {
        'current': defaultdict(lambda: [0, 0]),
        'hierarchy_high_cap': defaultdict(lambda: [0, 0]),  # Hierarchy at rank>20 → max HIGH
        'hierarchy_high_cap+cancer_high': defaultdict(lambda: [0, 0]),  # + cancer_same_type w/ HIGH criteria
    }

    hierarchy_promoted = 0
    hierarchy_promoted_gt = 0
    cv_promoted = 0
    cv_promoted_gt = 0
    cancer_promoted = 0
    cancer_promoted_gt = 0

    for disease_id in gt_diseases:
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

            # Current
            scenarios['current'][tier.value][0] += 1
            if is_gt:
                scenarios['current'][tier.value][1] += 1

            # Scenario 1: hierarchy_high_cap
            new_tier_1 = tier
            if tier == ConfidenceTier.FILTER and pred.rank > 20:
                # Hierarchy override
                if category in DISEASE_HIERARCHY_GROUPS and pred.drug_id:
                    _, same_group, matching_group = predictor._check_disease_hierarchy_match(
                        pred.drug_id, disease_name, category
                    )
                    if same_group and matching_group not in HIERARCHY_DEMOTE_TO_MEDIUM:
                        # Cap at HIGH (never GOLDEN for rank>20)
                        new_tier_1 = ConfidenceTier.HIGH
                        hierarchy_promoted += 1
                        if is_gt:
                            hierarchy_promoted_gt += 1
                    elif same_group and matching_group in HIERARCHY_DEMOTE_TO_MEDIUM:
                        new_tier_1 = ConfidenceTier.MEDIUM

                # CV pathway-comprehensive
                if new_tier_1 == ConfidenceTier.FILTER:
                    if predictor._is_cv_complication(disease_name) and predictor._is_cv_pathway_comprehensive(pred.drug_name):
                        new_tier_1 = ConfidenceTier.HIGH
                        cv_promoted += 1
                        if is_gt:
                            cv_promoted_gt += 1

            scenarios['hierarchy_high_cap'][new_tier_1.value][0] += 1
            if is_gt:
                scenarios['hierarchy_high_cap'][new_tier_1.value][1] += 1

            # Scenario 2: + cancer promotion
            new_tier_2 = new_tier_1

            if category == 'cancer' and pred.drug_id:
                has_cancer_gt, same_type_match, _ = predictor._check_cancer_type_match(pred.drug_id, disease_name)
                if same_type_match and tier == ConfidenceTier.MEDIUM:
                    # Would this qualify for HIGH?
                    if (pred.train_frequency >= 15 and pred.mechanism_support) or \
                       (pred.rank <= 5 and pred.train_frequency >= 10 and pred.mechanism_support):
                        is_coherent = predictor._is_atc_coherent(pred.drug_name, category)
                        if is_coherent:
                            new_tier_2 = ConfidenceTier.HIGH
                            cancer_promoted += 1
                            if is_gt:
                                cancer_promoted_gt += 1

            scenarios['hierarchy_high_cap+cancer_high'][new_tier_2.value][0] += 1
            if is_gt:
                scenarios['hierarchy_high_cap+cancer_high'][new_tier_2.value][1] += 1

    # Print comparative results
    for name, tiers in scenarios.items():
        print(f"\n{'='*60}")
        print(f"Scenario: {name}")
        print(f"{'='*60}")
        print(f"{'Tier':<10} {'Total':>8} {'GT':>5} {'Prec':>7}")
        print("-" * 35)
        for tier_name in ['GOLDEN', 'HIGH', 'MEDIUM', 'LOW', 'FILTER']:
            t, g = tiers[tier_name]
            prec = 100 * g / t if t > 0 else 0
            print(f"{tier_name:<10} {t:>8} {g:>5} {prec:>6.1f}%")

    # Diff table
    print(f"\n{'='*80}")
    print("DIFF vs CURRENT")
    print(f"{'='*80}")
    print(f"{'Tier':<10} {'h_cap':>12} {'h_cap+cnc':>12}")
    print("-" * 40)
    for tier_name in ['GOLDEN', 'HIGH', 'MEDIUM', 'LOW', 'FILTER']:
        ct, cg = scenarios['current'][tier_name]
        c_prec = 100 * cg / ct if ct > 0 else 0

        h1t, h1g = scenarios['hierarchy_high_cap'][tier_name]
        h1_prec = 100 * h1g / h1t if h1t > 0 else 0

        h2t, h2g = scenarios['hierarchy_high_cap+cancer_high'][tier_name]
        h2_prec = 100 * h2g / h2t if h2t > 0 else 0

        d1 = h1_prec - c_prec
        d2 = h2_prec - c_prec
        print(f"{tier_name:<10} {d1:>+11.1f}pp {d2:>+11.1f}pp")

    print(f"\nHierarchy promoted: {hierarchy_promoted} (GT: {hierarchy_promoted_gt})")
    print(f"CV pathway promoted: {cv_promoted} (GT: {cv_promoted_gt})")
    print(f"Cancer promoted: {cancer_promoted} (GT: {cancer_promoted_gt})")


if __name__ == '__main__':
    main()
