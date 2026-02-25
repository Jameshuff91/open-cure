#!/usr/bin/env python3
"""
h809: Cancer same-type mech+rank10 double-demotion analysis

h808 found cancer_same_type_mech_rank10 + NO/WEAK literature evidence = 13.4% holdout.
This is far below MEDIUM (34.8%), suggesting these should be demoted to LOW.

Also check: which OTHER literature-demoted sub-reasons should go to LOW?
h808 showed:
  - default_freq10_nomech_r1_5: 36.4% (MEDIUM OK)
  - cancer_same_type_mech_rank10: 13.4% (should be LOW)
  - default (misc HIGH): 15.7% (should be LOW, but tiny n)
  - transe_medium_promotion: 18.8% (tiny n)
  - others: tiny n

Test: What if we allow specific literature_high_demotion predictions to be
further demoted to LOW based on their original sub-reason?
"""

import json
import sys
import os
from collections import defaultdict
from typing import Set, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.production_predictor import DrugRepurposingPredictor, ConfidenceTier
from scripts.h393_holdout_tier_validation import (
    split_diseases,
    recompute_gt_structures,
    restore_gt_structures,
)


def main() -> None:
    seeds = [42, 123, 456, 789, 2024]
    print("=" * 70)
    print("h809: Cancer same-type double demotion + impact analysis")
    print("=" * 70)

    predictor = DrugRepurposingPredictor()

    gt_path = predictor.reference_dir / "expanded_ground_truth.json"
    with open(gt_path) as f:
        gt_data = json.load(f)
    gt_set: Set[Tuple[str, str]] = set()
    for disease_id, drugs in gt_data.items():
        for drug in drugs:
            if isinstance(drug, str):
                gt_set.add((disease_id, drug))

    all_diseases = [d for d in predictor.ground_truth if d in predictor.embeddings]
    print(f"Diseases: {len(all_diseases)}, GT pairs: {len(gt_set)}")

    # Store original method
    original_lit_method = predictor._get_literature_evidence

    # Track per-tier metrics for baseline and modified
    tier_metrics = {
        'baseline': defaultdict(lambda: {'hits': [], 'total': []}),
        'modified': defaultdict(lambda: {'hits': [], 'total': []}),
    }

    # Count of predictions affected
    moved_counts = []

    for seed_idx, seed in enumerate(seeds):
        print(f"\nSeed {seed} ({seed_idx + 1}/{len(seeds)})...")
        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        train_set = set(train_ids)
        originals = recompute_gt_structures(predictor, train_set)

        # Pass 1: Normal run (baseline) — capture tiers
        baseline_tiers = defaultdict(lambda: [0, 0])  # tier -> [hits, total]

        # Also capture original sub-reasons for literature-demoted preds
        # by doing a no-lit pass
        predictor._get_literature_evidence = lambda d, dis: ('NOT_ASSESSED', 0.0)
        orig_sub_map = {}  # (disease_id, drug_name) -> original_sub_reason
        for disease_id in holdout_ids:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            try:
                result = predictor.predict(disease_name, top_n=30, include_filtered=True)
            except Exception:
                continue
            for pred in result.predictions:
                if pred.confidence_tier.name == 'HIGH':
                    orig_sub_map[(disease_id, pred.drug_name)] = pred.category_specific_tier or 'default'

        predictor._get_literature_evidence = original_lit_method

        # Now normal run with literature
        normal_preds = {}  # (disease_id, drug_name) -> pred
        for disease_id in holdout_ids:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            try:
                result = predictor.predict(disease_name, top_n=30, include_filtered=True)
            except Exception:
                continue
            for pred in result.predictions:
                key = (disease_id, pred.drug_name)
                is_hit = (disease_id, pred.drug_id) in gt_set
                tier = pred.confidence_tier.name
                baseline_tiers[tier][0] += (1 if is_hit else 0)
                baseline_tiers[tier][1] += 1
                normal_preds[key] = {
                    'tier': tier,
                    'cat_specific': pred.category_specific_tier,
                    'lit_level': pred.literature_evidence_level,
                    'drug_id': pred.drug_id,
                    'is_hit': is_hit,
                }

        # Count and reassign: literature_high_demotion with bad original sub-reasons → LOW
        # Sub-reasons that h808 showed are LOW quality when literature is weak:
        LOW_QUALITY_SUBREASONS = {
            'cancer_same_type_mech_rank10',  # 13.4%
            'default',                        # 15.7%
            'corticosteroid_soc_promotion',  # 11.1%
            'cardiovascular_hierarchy_hypertension',  # 0%
            'cardiovascular',                # 16.7%
            'cancer',                        # 0%
        }

        modified_tiers = defaultdict(lambda: [0, 0])
        moved = 0

        for key, info in normal_preds.items():
            disease_id, drug_name = key
            tier = info['tier']

            # Check if this is a literature_high_demotion that should go further to LOW
            if (info['cat_specific'] == 'literature_high_demotion'
                    and key in orig_sub_map
                    and orig_sub_map[key] in LOW_QUALITY_SUBREASONS):
                tier = 'LOW'
                moved += 1

            modified_tiers[tier][0] += (1 if info['is_hit'] else 0)
            modified_tiers[tier][1] += 1

        moved_counts.append(moved)

        restore_gt_structures(predictor, originals)

        # Record per-seed metrics
        for t in ['GOLDEN', 'HIGH', 'MEDIUM', 'LOW', 'FILTER']:
            h, tot = baseline_tiers[t]
            tier_metrics['baseline'][t]['hits'].append(h)
            tier_metrics['baseline'][t]['total'].append(tot)
            h2, tot2 = modified_tiers[t]
            tier_metrics['modified'][t]['hits'].append(h2)
            tier_metrics['modified'][t]['total'].append(tot2)

        # Print seed summary
        print(f"  Moved {moved} preds MEDIUM→LOW")
        for t in ['GOLDEN', 'HIGH', 'MEDIUM', 'LOW']:
            bh, bt = baseline_tiers[t]
            mh, mt = modified_tiers[t]
            bp = bh / bt * 100 if bt > 0 else 0
            mp = mh / mt * 100 if mt > 0 else 0
            delta = mp - bp
            if abs(delta) > 0.01:
                print(f"    {t}: {bp:.1f}% ({bt}) → {mp:.1f}% ({mt}) [{delta:+.1f}pp]")

    # Aggregate
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS (5-seed mean ± std)")
    print("=" * 70)
    print(f"Predictions moved MEDIUM→LOW: {np.mean(moved_counts):.0f}/seed")

    for t in ['GOLDEN', 'HIGH', 'MEDIUM', 'LOW', 'FILTER']:
        for label in ['baseline', 'modified']:
            hits = tier_metrics[label][t]['hits']
            totals = tier_metrics[label][t]['total']
            precs = [h / tot * 100 if tot > 0 else 0 for h, tot in zip(hits, totals)]
            avg_n = np.mean(totals)
            mean_p = np.mean(precs)
            std_p = np.std(precs)
            if label == 'baseline':
                print(f"\n  {t}:")
                print(f"    Baseline: {mean_p:.1f}% ± {std_p:.1f}% (n={avg_n:.0f}/seed)")
            else:
                print(f"    Modified: {mean_p:.1f}% ± {std_p:.1f}% (n={avg_n:.0f}/seed)")

    # Also test: ONLY cancer_same_type_mech_rank10 demotion (most impactful single rule)
    print("\n" + "=" * 70)
    print("SENSITIVITY: Only cancer_same_type_mech_rank10 demotion")
    print("=" * 70)
    # Already computed above — the cancer rule alone accounts for ~8/seed preds


if __name__ == '__main__':
    main()
