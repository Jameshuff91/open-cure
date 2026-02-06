#!/usr/bin/env python3
"""
h399 Deep Dive: Analyze the most impactful rule interactions.

Focus on:
1. rank_too_low shadowing high-precision rules (hierarchy, cv_pathway)
2. cancer_same_type shadowing standard HIGH rules
3. zero_precision_mismatch shadowing MEDIUM rules
4. mechanism_specific shadowing MEDIUM rules
"""

import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.production_predictor import (
    DrugRepurposingPredictor, ConfidenceTier,
    CORTICOSTEROID_DRUGS, STATIN_NAMES, DISEASE_HIERARCHY_GROUPS,
)


def main():
    print("=" * 80)
    print("h399 Deep Dive: Targeted Interaction Analysis")
    print("=" * 80)

    predictor = DrugRepurposingPredictor()
    gt_data = predictor.ground_truth
    gt_diseases = [d for d in gt_data if len(gt_data[d]) > 0 and d in predictor.embeddings]

    # === ANALYSIS 1: rank>20 drugs that match hierarchy/cv_pathway rules ===
    print("\n" + "=" * 80)
    print("ANALYSIS 1: High-precision rules shadowed by rank_too_low (rank>20)")
    print("  Question: Should hierarchy/cv_pathway rules override rank>20 filter?")
    print("=" * 80)

    rank_shadow_stats = defaultdict(lambda: {'total': 0, 'gt_hits': 0, 'examples': []})

    # === ANALYSIS 2: cancer_same_type vs standard HIGH ===
    print("\n  (Also collecting cancer_same_type vs HIGH data)")
    cancer_vs_high = {'total': 0, 'gt_hits': 0, 'examples': []}

    # === ANALYSIS 3: zero_precision_mismatch with GT hits ===
    zpm_gt = {'total': 0, 'gt_hits': 0, 'examples': []}

    # === ANALYSIS 4: mechanism_specific shadowing MEDIUM ===
    mech_spec_med = {'total': 0, 'gt_hits': 0, 'examples': []}

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
            drug_lower = pred.drug_name.lower()
            disease_lower = disease_name.lower()

            # ANALYSIS 1: rank>20 drugs that match high-precision rules
            if pred.rank > 20:
                # Check hierarchy match
                if category in DISEASE_HIERARCHY_GROUPS and pred.drug_id:
                    _, same_group, matching_group = predictor._check_disease_hierarchy_match(
                        pred.drug_id, disease_name, category
                    )
                    if same_group:
                        key = f'hierarchy_{matching_group}'
                        rank_shadow_stats[key]['total'] += 1
                        if is_gt:
                            rank_shadow_stats[key]['gt_hits'] += 1
                            if len(rank_shadow_stats[key]['examples']) < 3:
                                rank_shadow_stats[key]['examples'].append(
                                    f"  {pred.drug_name} -> {disease_name} (rank={pred.rank})"
                                )

                # Check cv_pathway_comprehensive
                if predictor._is_cv_complication(disease_name) and predictor._is_cv_pathway_comprehensive(pred.drug_name):
                    rank_shadow_stats['cv_pathway_comprehensive']['total'] += 1
                    if is_gt:
                        rank_shadow_stats['cv_pathway_comprehensive']['gt_hits'] += 1
                        if len(rank_shadow_stats['cv_pathway_comprehensive']['examples']) < 3:
                            rank_shadow_stats['cv_pathway_comprehensive']['examples'].append(
                                f"  {pred.drug_name} -> {disease_name} (rank={pred.rank})"
                            )

                # Check comp_to_base
                if pred.drug_id:
                    is_comp_to_base, transferability, is_statin_cv = predictor._is_comp_to_base(
                        pred.drug_id, disease_name
                    )
                    if is_comp_to_base and (is_statin_cv or transferability >= 50):
                        key = 'comp_to_base_high' if not is_statin_cv else 'statin_cv_event'
                        rank_shadow_stats[key]['total'] += 1
                        if is_gt:
                            rank_shadow_stats[key]['gt_hits'] += 1
                            if len(rank_shadow_stats[key]['examples']) < 3:
                                rank_shadow_stats[key]['examples'].append(
                                    f"  {pred.drug_name} -> {disease_name} (rank={pred.rank})"
                                )

            # ANALYSIS 2: cancer_same_type that would qualify for standard HIGH
            if category == 'cancer' and pred.drug_id:
                has_cancer_gt, same_type_match, _ = predictor._check_cancer_type_match(pred.drug_id, disease_name)
                if same_type_match:
                    # Would this qualify for standard HIGH?
                    disease_tier = result.disease_tier
                    if (pred.train_frequency >= 15 and pred.mechanism_support) or \
                       (pred.rank <= 5 and pred.train_frequency >= 10 and pred.mechanism_support):
                        is_coherent = predictor._is_atc_coherent(pred.drug_name, category)
                        if is_coherent:
                            cancer_vs_high['total'] += 1
                            if is_gt:
                                cancer_vs_high['gt_hits'] += 1
                                if len(cancer_vs_high['examples']) < 5:
                                    cancer_vs_high['examples'].append(
                                        f"  {pred.drug_name} -> {disease_name} (freq={pred.train_frequency}, rank={pred.rank}, mech={pred.mechanism_support})"
                                    )

            # ANALYSIS 3: zero_precision_mismatch with GT hits
            if pred.drug_name and category:
                _, is_zero_prec, _ = predictor._check_atc_mismatch_rules(pred.drug_name, category)
                if is_zero_prec and is_gt:
                    zpm_gt['total'] += 1
                    zpm_gt['gt_hits'] += 1
                    if len(zpm_gt['examples']) < 10:
                        zpm_gt['examples'].append(
                            f"  {pred.drug_name} -> {disease_name} (cat={category}, tier={pred.confidence_tier.value})")

            # ANALYSIS 4: mechanism_specific that would qualify for MEDIUM
            if predictor._is_mechanism_specific_disease(disease_name):
                if (pred.train_frequency >= 5 and pred.mechanism_support) or pred.train_frequency >= 10:
                    mech_spec_med['total'] += 1
                    if is_gt:
                        mech_spec_med['gt_hits'] += 1
                        if len(mech_spec_med['examples']) < 5:
                            mech_spec_med['examples'].append(
                                f"  {pred.drug_name} -> {disease_name} (freq={pred.train_frequency}, mech={pred.mechanism_support})")

    # Print Analysis 1
    print("\nRule shadowed by rank>20 | Precision if rule had overridden rank>20:")
    print(f"{'Rule':<35} {'Total':>6} {'GT':>4} {'Prec':>7}")
    print("-" * 55)
    for rule, stats in sorted(rank_shadow_stats.items(), key=lambda x: x[1]['total'], reverse=True):
        prec = 100 * stats['gt_hits'] / stats['total'] if stats['total'] > 0 else 0
        print(f"{rule:<35} {stats['total']:>6} {stats['gt_hits']:>4} {prec:>6.1f}%")
        for ex in stats['examples']:
            print(ex)

    # Print Analysis 2
    print(f"\n{'='*80}")
    print("ANALYSIS 2: cancer_same_type shadowing standard HIGH")
    print(f"{'='*80}")
    prec = 100 * cancer_vs_high['gt_hits'] / cancer_vs_high['total'] if cancer_vs_high['total'] > 0 else 0
    print(f"  Count: {cancer_vs_high['total']}, GT: {cancer_vs_high['gt_hits']}, Precision: {prec:.1f}%")
    print(f"  Current: MEDIUM (cancer_same_type)")
    print(f"  Shadowed: HIGH (standard criteria met)")
    print(f"  Question: Should high-freq, mechanism-supported cancer same-type get HIGH instead of MEDIUM?")
    for ex in cancer_vs_high['examples']:
        print(ex)

    # Print Analysis 3
    print(f"\n{'='*80}")
    print("ANALYSIS 3: zero_precision_mismatch rules catching GT hits")
    print(f"{'='*80}")
    print(f"  GT hits caught by zero_precision_mismatch: {zpm_gt['total']}")
    print(f"  These are cases where the ATC mismatch rule incorrectly FILTERs a true positive:")
    for ex in zpm_gt['examples']:
        print(ex)

    # Print Analysis 4
    print(f"\n{'='*80}")
    print("ANALYSIS 4: mechanism_specific (LOW) shadowing MEDIUM criteria")
    print(f"{'='*80}")
    prec = 100 * mech_spec_med['gt_hits'] / mech_spec_med['total'] if mech_spec_med['total'] > 0 else 0
    print(f"  Count: {mech_spec_med['total']}, GT: {mech_spec_med['gt_hits']}, Precision: {prec:.1f}%")
    print(f"  Current: LOW (mechanism_specific cap)")
    print(f"  Shadowed: MEDIUM (meets standard MEDIUM criteria)")
    for ex in mech_spec_med['examples']:
        print(ex)

    print(f"\n{'='*80}")
    print("DONE")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
