#!/usr/bin/env python3
"""
h518: Test if LIKELY_GT_GAP (SOC) status differentiates holdout precision.

If SOC predictions have higher holdout precision within tiers, this could be:
1. A tier promotion signal (SOC + MEDIUM → HIGH potential)
2. An annotation that helps consumers prioritize predictions
"""
import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.production_predictor import (
    DrugRepurposingPredictor, ConfidenceTier,
    classify_literature_status,
    DISEASE_HIERARCHY_GROUPS, HIERARCHY_EXCLUSIONS,
)
from typing import Dict, Set, Tuple, List

# Import holdout infrastructure from h393
sys.path.insert(0, str(Path(__file__).parent))
from h393_holdout_tier_validation import recompute_gt_structures, restore_gt_structures


def extract_cancer_types(disease_name: str) -> Set[str]:
    """Extract cancer type keywords from disease name."""
    types = set()
    lower = disease_name.lower()
    cancer_terms = [
        'breast', 'lung', 'prostate', 'colorectal', 'colon', 'rectal',
        'pancreatic', 'liver', 'hepatocellular', 'renal', 'kidney',
        'bladder', 'ovarian', 'cervical', 'endometrial', 'thyroid',
        'melanoma', 'lymphoma', 'leukemia', 'myeloma', 'glioblastoma',
        'gastric', 'stomach', 'esophageal', 'head and neck',
    ]
    for term in cancer_terms:
        if term in lower:
            types.add(term)
    return types


def main():
    print("=" * 70)
    print("h518: SOC Status as Holdout Precision Signal")
    print("=" * 70)

    predictor = DrugRepurposingPredictor()
    gt_data = predictor.ground_truth

    seeds = [42, 123, 456, 789, 1024]

    # Collect results across seeds
    # Key: (tier, soc_status) -> list of (hits, total) per seed
    tier_soc_results = defaultdict(lambda: [])

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        np.random.seed(seed)

        disease_ids = list(gt_data.keys())
        np.random.shuffle(disease_ids)
        n_holdout = len(disease_ids) // 5
        holdout_diseases = set(disease_ids[:n_holdout])
        train_diseases = set(disease_ids[n_holdout:])

        # Recompute GT structures for training only
        originals = recompute_gt_structures(predictor, train_diseases)

        # Track per-seed results
        seed_counts = defaultdict(lambda: {'hits': 0, 'total': 0})

        for disease_id in holdout_diseases:
            disease_name = predictor.disease_names.get(disease_id, '')
            if not disease_name:
                continue

            try:
                result = predictor.predict(disease_name, top_n=30, include_filtered=True)
            except Exception:
                continue

            gt_drugs = gt_data.get(disease_id, set())

            for pred in result.predictions:
                tier = pred.confidence_tier.value
                drug_id = pred.drug_id
                is_hit = drug_id in gt_drugs

                # Classify literature status
                is_gt_in_train = disease_id in train_diseases and drug_id in gt_data.get(disease_id, set())
                lit_status, soc_class = classify_literature_status(
                    pred.drug_name, disease_name, result.category, is_gt_in_train
                )

                # We only care about non-known predictions for SOC vs NOVEL comparison
                if lit_status == 'KNOWN_INDICATION':
                    key = (tier, 'KNOWN')
                elif lit_status == 'LIKELY_GT_GAP':
                    key = (tier, 'SOC')
                else:
                    key = (tier, 'NOVEL')

                seed_counts[key]['total'] += 1
                if is_hit:
                    seed_counts[key]['hits'] += 1

        # Restore
        restore_gt_structures(predictor, originals)

        # Record seed results
        for key, counts in seed_counts.items():
            tier_soc_results[key].append(counts)

        # Print seed summary
        for tier in ['GOLDEN', 'HIGH', 'MEDIUM', 'LOW', 'FILTER']:
            for status in ['KNOWN', 'SOC', 'NOVEL']:
                key = (tier, status)
                if key in seed_counts:
                    c = seed_counts[key]
                    prec = 100 * c['hits'] / c['total'] if c['total'] else 0
                    print(f"  {tier:<8} {status:<6} {c['hits']:>4}/{c['total']:<5} = {prec:>5.1f}%")

    # Aggregate across seeds
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS (5-seed)")
    print("=" * 70)
    print(f"\n{'Tier':<10} {'Status':<8} {'Mean Prec':>10} {'Std':>8} {'Mean N':>8}")
    print("-" * 50)

    for tier in ['GOLDEN', 'HIGH', 'MEDIUM', 'LOW', 'FILTER']:
        for status in ['KNOWN', 'SOC', 'NOVEL']:
            key = (tier, status)
            if key not in tier_soc_results:
                continue
            seed_data = tier_soc_results[key]
            precisions = []
            totals = []
            for counts in seed_data:
                total = counts['total']
                if total > 0:
                    precisions.append(100 * counts['hits'] / total)
                    totals.append(total)
            if precisions:
                mean_prec = np.mean(precisions)
                std_prec = np.std(precisions)
                mean_n = np.mean(totals)
                print(f"{tier:<10} {status:<8} {mean_prec:>9.1f}% {std_prec:>7.1f}% {mean_n:>7.0f}")
        print()

    # Key comparison: SOC vs NOVEL within MEDIUM tier
    print("=" * 70)
    print("KEY QUESTION: Does SOC differentiate MEDIUM holdout precision?")
    print("=" * 70)

    for tier in ['HIGH', 'MEDIUM', 'LOW']:
        soc_key = (tier, 'SOC')
        novel_key = (tier, 'NOVEL')

        if soc_key in tier_soc_results and novel_key in tier_soc_results:
            soc_precs = []
            novel_precs = []
            for counts in tier_soc_results[soc_key]:
                if counts['total'] > 0:
                    soc_precs.append(100 * counts['hits'] / counts['total'])
            for counts in tier_soc_results[novel_key]:
                if counts['total'] > 0:
                    novel_precs.append(100 * counts['hits'] / counts['total'])

            if soc_precs and novel_precs:
                soc_mean = np.mean(soc_precs)
                novel_mean = np.mean(novel_precs)
                gap = soc_mean - novel_mean
                print(f"  {tier}: SOC={soc_mean:.1f}% vs NOVEL={novel_mean:.1f}% (gap={gap:+.1f}pp)")

    # Save results
    output = {
        'seeds': seeds,
        'tier_soc_results': {
            f"{k[0]}_{k[1]}": [dict(c) for c in v]
            for k, v in tier_soc_results.items()
        }
    }
    with open('data/analysis/h518_soc_holdout.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to data/analysis/h518_soc_holdout.json")


if __name__ == '__main__':
    main()
