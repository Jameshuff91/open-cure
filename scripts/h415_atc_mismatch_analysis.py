#!/usr/bin/env python3
"""
h415: Analyze which zero_precision_mismatch rules catch GT hits.
Determine if some rules should be removed or relaxed.
"""

import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.production_predictor import (
    DrugRepurposingPredictor, ConfidenceTier,
    ZERO_PRECISION_MISMATCHES,
)


def main():
    print("=" * 80)
    print("h415: Zero-Precision ATC Mismatch Analysis")
    print("=" * 80)

    predictor = DrugRepurposingPredictor()
    gt_data = predictor.ground_truth
    gt_diseases = [d for d in gt_data if len(gt_data[d]) > 0 and d in predictor.embeddings]

    print(f"\nZERO_PRECISION_MISMATCHES has {len(ZERO_PRECISION_MISMATCHES)} rules")
    print(f"GT diseases with embeddings: {len(gt_diseases)}")

    # Per-rule stats: (atc_l1, category) -> {total, gt_hits, examples}
    rule_stats = defaultdict(lambda: {'total': 0, 'gt_hits': 0, 'examples': []})

    # Overall stats
    total_filtered = 0
    total_gt_filtered = 0

    # ATC mapper
    from src.production_predictor import _get_atc_mapper
    mapper = _get_atc_mapper()

    for idx, disease_id in enumerate(gt_diseases):
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        gt_drugs = gt_data[disease_id]
        category = predictor.categorize_disease(disease_name)

        try:
            result = predictor.predict(disease_name, top_n=30, include_filtered=True)
        except Exception:
            continue

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx+1}/{len(gt_diseases)}...")

        for pred in result.predictions:
            # Check if zero_precision_mismatch fires
            if not pred.drug_name or not category:
                continue

            _, is_zero_prec, _ = predictor._check_atc_mismatch_rules(pred.drug_name, category)
            if not is_zero_prec:
                continue

            # Get the specific rule
            try:
                atc_codes = mapper.get_atc_codes(pred.drug_name)
                if not atc_codes or not atc_codes[0]:
                    continue
                atc_l1 = atc_codes[0][0]
            except Exception:
                continue

            key = (atc_l1, category)
            is_gt = pred.drug_id in gt_drugs

            rule_stats[key]['total'] += 1
            if is_gt:
                rule_stats[key]['gt_hits'] += 1
                if len(rule_stats[key]['examples']) < 5:
                    rule_stats[key]['examples'].append(
                        f"    {pred.drug_name} -> {disease_name} "
                        f"(rank={pred.rank}, freq={pred.train_frequency}, tier={pred.confidence_tier.value})"
                    )

            total_filtered += 1
            if is_gt:
                total_gt_filtered += 1

    # Print results
    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print(f"Total predictions filtered by zero_precision_mismatch: {total_filtered}")
    print(f"GT hits incorrectly filtered: {total_gt_filtered}")
    print(f"Overall precision of filtered predictions: {100*total_gt_filtered/max(1,total_filtered):.1f}%")

    # Per-rule breakdown
    print(f"\n{'Rule (ATC→Category)':<30} {'Total':>7} {'GT':>5} {'Prec':>7}")
    print("-" * 55)

    sorted_rules = sorted(rule_stats.items(), key=lambda x: x[1]['gt_hits'], reverse=True)

    problematic_rules = []
    for (atc_l1, cat), stats in sorted_rules:
        prec = 100 * stats['gt_hits'] / stats['total'] if stats['total'] > 0 else 0
        marker = " ***" if stats['gt_hits'] > 0 and prec > 5 else ""
        print(f"  ({atc_l1}, {cat}){'':<{20-len(cat)}} {stats['total']:>7} {stats['gt_hits']:>5} {prec:>6.1f}%{marker}")

        if stats['gt_hits'] > 0:
            for ex in stats['examples']:
                print(ex)

            if prec > 5:
                problematic_rules.append((atc_l1, cat, stats['total'], stats['gt_hits'], prec))

    # Summary of problematic rules
    if problematic_rules:
        print(f"\n{'='*80}")
        print(f"PROBLEMATIC RULES (>5% GT hit rate → should be relaxed/removed)")
        print(f"{'='*80}")
        total_recoverable = 0
        for atc_l1, cat, total, gt, prec in problematic_rules:
            print(f"  ({atc_l1}, {cat}): {gt} GT hits / {total} total = {prec:.1f}%")
            total_recoverable += gt
        print(f"\n  Total recoverable GT hits: {total_recoverable}")
        print(f"  Current zero_prec filter catches {total_filtered} predictions total")
        print(f"  Removing these rules would move {sum(r[2] for r in problematic_rules)} predictions from FILTER")
    else:
        print(f"\nNo rules with >5% GT hit rate found.")

    # Also check: what tier would these predictions get WITHOUT the mismatch filter?
    print(f"\n{'='*80}")
    print(f"TIER DISTRIBUTION IF ZERO_PRECISION_MISMATCH WERE REMOVED")
    print(f"{'='*80}")

    # Can't easily test this without modifying code, but we know:
    # These predictions are currently FILTER. Without the mismatch check,
    # they'd flow to standard tier rules. Let's check what standard tier they'd get.

    print("\n(Would need to test by temporarily removing the filter)")
    print("Based on h399 audit, these 495 predictions mostly match standard MEDIUM or LOW rules.")

    print(f"\n{'='*80}")
    print("DONE")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
