#!/usr/bin/env python3
"""
h811: Autoimmune RA Hierarchy GOLDEN→HIGH Demotion Analysis

Analyzes per-seed holdout for autoimmune_hierarchy_rheumatoid_arthritis
and tests demotion to HIGH.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from h393_holdout_tier_validation import (
    split_diseases,
    recompute_gt_structures,
    restore_gt_structures,
    evaluate_on_diseases,
)
from production_predictor import DrugRepurposingPredictor


def main() -> None:
    seeds = [42, 123, 456, 789, 2024]

    print("Loading predictor...")
    predictor = DrugRepurposingPredictor()

    gt_path = predictor.reference_dir / "expanded_ground_truth.json"
    with open(gt_path) as f:
        gt_data = json.load(f)

    all_diseases = [d for d in predictor.ground_truth if d in predictor.embeddings]
    print(f"Diseases: {len(all_diseases)}")

    # Collect per-seed data for RA and all GOLDEN hierarchy rules
    hierarchy_golden_rules = [
        'autoimmune_hierarchy_rheumatoid_arthritis',
        'cardiovascular_hierarchy_coronary',
        'cardiovascular_hierarchy_arrhythmia',
        'autoimmune_hierarchy_colitis',
        'infectious_hierarchy_uti',
    ]

    per_seed_ra = []
    per_seed_golden = []
    per_seed_rule_data = defaultdict(list)
    per_seed_rule_n = defaultdict(list)

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        train_set = set(train_ids)

        originals = recompute_gt_structures(predictor, train_set)
        holdout_result = evaluate_on_diseases(predictor, holdout_ids, gt_data)

        # Get RA-specific data
        for rule_name, stats in holdout_result['rule_precision'].items():
            if 'rheumatoid_arthritis' in rule_name:
                print(f"  RA: {stats['precision']:.1f}% ({stats['hits']}/{stats['total']})")
                per_seed_ra.append({
                    'seed': seed,
                    'precision': stats['precision'],
                    'hits': stats['hits'],
                    'total': stats['total'],
                })

            # Collect all hierarchy golden rules
            for hr in hierarchy_golden_rules:
                if rule_name == hr:
                    per_seed_rule_data[hr].append(stats['precision'])
                    per_seed_rule_n[hr].append(stats['total'])

        # GOLDEN tier
        golden = holdout_result['tier_precision'].get('GOLDEN', {})
        per_seed_golden.append({
            'seed': seed,
            'precision': golden.get('precision', 0),
            'hits': golden.get('hits', 0),
            'total': golden.get('total', 0),
        })
        print(f"  GOLDEN: {golden.get('precision', 0):.1f}% ({golden.get('hits', 0)}/{golden.get('total', 0)})")

        restore_gt_structures(predictor, originals)

    print("\n" + "=" * 70)
    print("PER-SEED RA BREAKDOWN")
    print("=" * 70)
    for entry in per_seed_ra:
        print(f"  Seed {entry['seed']}: {entry['precision']:.1f}% ({entry['hits']}/{entry['total']})")

    ra_precs = [e['precision'] for e in per_seed_ra]
    ra_ns = [e['total'] for e in per_seed_ra]
    print(f"\n  Mean: {np.mean(ra_precs):.1f}% ± {np.std(ra_precs):.1f}%")
    print(f"  Mean n: {np.mean(ra_ns):.1f}")
    print(f"  Range: {min(ra_precs):.1f}% - {max(ra_precs):.1f}%")

    print("\n" + "=" * 70)
    print("ALL GOLDEN HIERARCHY RULES (for comparison)")
    print("=" * 70)
    for hr in hierarchy_golden_rules:
        if hr in per_seed_rule_data:
            precs = per_seed_rule_data[hr]
            ns = per_seed_rule_n[hr]
            print(f"  {hr}: {np.mean(precs):.1f}% ± {np.std(precs):.1f}% (n={np.mean(ns):.1f}/seed)")

    print("\n" + "=" * 70)
    print("GOLDEN TIER PER-SEED")
    print("=" * 70)
    for entry in per_seed_golden:
        print(f"  Seed {entry['seed']}: {entry['precision']:.1f}% ({entry['hits']}/{entry['total']})")
    golden_precs = [e['precision'] for e in per_seed_golden]
    print(f"\n  Mean: {np.mean(golden_precs):.1f}% ± {np.std(golden_precs):.1f}%")

    # Now test demotion: what if RA was HIGH instead of GOLDEN?
    print("\n" + "=" * 70)
    print("DEMOTION SIMULATION: RA → HIGH")
    print("=" * 70)

    # Calculate impact: remove RA from GOLDEN, add to HIGH
    # For each seed, compute what GOLDEN and HIGH would be without/with RA
    for entry_g, entry_ra in zip(per_seed_golden, per_seed_ra):
        golden_hits = entry_g['hits']
        golden_total = entry_g['total']
        ra_hits = entry_ra['hits']
        ra_total = entry_ra['total']

        new_golden_hits = golden_hits - ra_hits
        new_golden_total = golden_total - ra_total
        new_golden_prec = (new_golden_hits / new_golden_total * 100) if new_golden_total > 0 else 0

        print(f"  Seed {entry_g['seed']}: GOLDEN {entry_g['precision']:.1f}% → {new_golden_prec:.1f}% "
              f"(removed {ra_total} RA preds, {ra_hits} hits)")

    # Aggregate
    new_golden_precs = []
    for entry_g, entry_ra in zip(per_seed_golden, per_seed_ra):
        gh = entry_g['hits'] - entry_ra['hits']
        gt = entry_g['total'] - entry_ra['total']
        new_golden_precs.append((gh / gt * 100) if gt > 0 else 0)

    print(f"\n  GOLDEN without RA: {np.mean(new_golden_precs):.1f}% ± {np.std(new_golden_precs):.1f}%")
    print(f"  GOLDEN with RA:    {np.mean(golden_precs):.1f}% ± {np.std(golden_precs):.1f}%")
    print(f"  Delta: {np.mean(new_golden_precs) - np.mean(golden_precs):+.1f}pp")

    # Save results
    results = {
        'per_seed_ra': per_seed_ra,
        'per_seed_golden': per_seed_golden,
        'ra_mean': round(float(np.mean(ra_precs)), 1),
        'ra_std': round(float(np.std(ra_precs)), 1),
        'golden_with_ra': round(float(np.mean(golden_precs)), 1),
        'golden_without_ra': round(float(np.mean(new_golden_precs)), 1),
        'golden_delta': round(float(np.mean(new_golden_precs) - np.mean(golden_precs)), 1),
    }

    with open('data/analysis/h811_ra_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to data/analysis/h811_ra_analysis.json")


if __name__ == "__main__":
    main()
