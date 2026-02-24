#!/usr/bin/env python3
"""h757: Score gradient within default_freq10_nomech_r6_10.

Analyze whether a score threshold can split this sub-reason into
keepable MEDIUM vs demotable LOW predictions.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from production_predictor import DrugRepurposingPredictor


def split_diseases(all_diseases, seed, train_ratio=0.8):
    rng = np.random.RandomState(seed)
    shuffled = list(all_diseases)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def recompute_gt_structures(predictor, train_disease_ids):
    from collections import defaultdict as dd
    originals = {
        "drug_train_freq": dict(predictor.drug_train_freq),
        "drug_to_diseases": dict(predictor.drug_to_diseases),
        "drug_cancer_types": dict(predictor.drug_cancer_types),
        "drug_disease_groups": dict(predictor.drug_disease_groups),
    }
    new_freq = dd(int)
    new_d2d = dd(set)
    new_cancer = dd(set)
    new_groups = dd(lambda: dd(set))
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            for drug in predictor.ground_truth[disease_id]:
                new_freq[drug] += 1
                new_d2d[drug].add(disease_id)
    for drug, diseases in new_d2d.items():
        for d in diseases:
            cat = predictor._categorize_disease(d)
            new_groups[drug][cat].add(d)
            cancer_types = predictor._extract_cancer_types_for_disease(d)
            if cancer_types:
                new_cancer[drug].update(cancer_types)
    predictor.drug_train_freq = dict(new_freq)
    predictor.drug_to_diseases = {k: v for k, v in new_d2d.items()}
    predictor.drug_cancer_types = dict(new_cancer)
    predictor.drug_disease_groups = {k: dict(v) for k, v in new_groups.items()}
    return originals


def restore_gt_structures(predictor, originals):
    predictor.drug_train_freq = originals["drug_train_freq"]
    predictor.drug_to_diseases = originals["drug_to_diseases"]
    predictor.drug_cancer_types = originals["drug_cancer_types"]
    predictor.drug_disease_groups = originals["drug_disease_groups"]


def main():
    predictor = DrugRepurposingPredictor()

    # Load expanded GT
    with open('data/reference/expanded_ground_truth.json') as f:
        expanded_gt = json.load(f)
    expanded_gt_set = set()
    for disease, drugs in expanded_gt.items():
        for drug in drugs:
            expanded_gt_set.add((drug.lower(), disease.lower()))

    # Full-data analysis first
    print("=== FULL-DATA ANALYSIS ===")
    gt_diseases = list(predictor.ground_truth.keys())
    target_preds = []

    for disease_name in gt_diseases:
        result = predictor.predict(disease_name, top_n=30, include_filtered=True)
        if result:
            for p in result.predictions:
                if getattr(p, 'category_specific_tier', '') == 'default_freq10_nomech_r6_10':
                    target_preds.append({
                        'drug': p.drug_name,
                        'disease': disease_name,
                        'rank': p.rank,
                        'score': p.score,
                        'in_gt': (p.drug_name.lower(), disease_name.lower()) in expanded_gt_set
                    })

    print(f"Total default_freq10_nomech_r6_10 predictions: {len(target_preds)}")

    scores = [p['score'] for p in target_preds]
    print(f"Score range: {min(scores):.2f} - {max(scores):.2f}")
    print(f"Score mean: {np.mean(scores):.2f}, median: {np.median(scores):.2f}")

    # Score gradient
    print(f"\nFull-data by score bucket:")
    buckets = [(3.0, 4.0), (4.0, 5.0), (5.0, 6.0), (6.0, 8.0), (8.0, 999)]
    for lo, hi in buckets:
        b = [p for p in target_preds if lo <= p['score'] < hi]
        if b:
            hits = sum(1 for p in b if p['in_gt'])
            print(f"  Score {lo:.0f}-{hi:.0f}: {hits/len(b)*100:.1f}% ({hits}/{len(b)})")

    # Rank gradient
    print(f"\nFull-data by rank:")
    for r in range(6, 11):
        b = [p for p in target_preds if p['rank'] == r]
        if b:
            hits = sum(1 for p in b if p['in_gt'])
            print(f"  Rank {r}: {hits/len(b)*100:.1f}% ({hits}/{len(b)})")

    # 5-seed holdout
    print(f"\n=== 5-SEED HOLDOUT ===")
    seeds = [42, 123, 456, 789, 2024]

    score_splits = [
        (3.0, 4.5, "3.0-4.5"),
        (4.5, 6.0, "4.5-6.0"),
        (6.0, 999, "6.0+"),
        (3.0, 999, "ALL")
    ]
    bucket_precisions = {name: [] for _, _, name in score_splits}
    bucket_ns = {name: [] for _, _, name in score_splits}

    rank_splits = [(6, 8, "R6-7"), (8, 11, "R8-10"), (6, 11, "ALL")]
    rank_precisions = {name: [] for _, _, name in rank_splits}
    rank_ns = {name: [] for _, _, name in rank_splits}

    for seed_idx, seed in enumerate(seeds):
        train_diseases, holdout_diseases = split_diseases(gt_diseases, seed)
        holdout_set = set(holdout_diseases)
        train_set = set(train_diseases)

        originals = recompute_gt_structures(predictor, train_set)

        holdout_preds = []
        for disease_name in holdout_diseases:
            result = predictor.predict(disease_name, top_n=30, include_filtered=True)
            if result:
                for p in result.predictions:
                    if getattr(p, 'category_specific_tier', '') == 'default_freq10_nomech_r6_10':
                        holdout_preds.append({
                            'drug': p.drug_name,
                            'disease': disease_name,
                            'rank': p.rank,
                            'score': p.score,
                            'in_gt': (p.drug_name.lower(), disease_name.lower()) in expanded_gt_set
                        })

        restore_gt_structures(predictor, originals)

        print(f"\nSeed {seed}: {len(holdout_preds)} holdout preds for this sub-reason")

        for lo, hi, name in score_splits:
            b = [p for p in holdout_preds if lo <= p['score'] < hi]
            if b:
                hits = sum(1 for p in b if p['in_gt'])
                bucket_precisions[name].append(hits / len(b) * 100)
                bucket_ns[name].append(len(b))
            else:
                bucket_ns[name].append(0)

        for lo, hi, name in rank_splits:
            b = [p for p in holdout_preds if lo <= p['rank'] < hi]
            if b:
                hits = sum(1 for p in b if p['in_gt'])
                rank_precisions[name].append(hits / len(b) * 100)
                rank_ns[name].append(len(b))
            else:
                rank_ns[name].append(0)

    print(f"\n=== HOLDOUT RESULTS ===")
    print(f"\nBy score bucket:")
    for _, _, name in score_splits:
        precs = bucket_precisions[name]
        ns = bucket_ns[name]
        if precs:
            print(f"  {name}: {np.mean(precs):.1f}% ± {np.std(precs):.1f}% (n={np.mean(ns):.1f}/seed)")
        else:
            print(f"  {name}: no holdout predictions")

    print(f"\nBy rank bucket:")
    for _, _, name in rank_splits:
        precs = rank_precisions[name]
        ns = rank_ns[name]
        if precs:
            print(f"  {name}: {np.mean(precs):.1f}% ± {np.std(precs):.1f}% (n={np.mean(ns):.1f}/seed)")
        else:
            print(f"  {name}: no holdout predictions")

    # Top drugs in this sub-reason
    print(f"\n=== TOP DRUGS ===")
    drug_counts = defaultdict(int)
    drug_hits = defaultdict(int)
    for p in target_preds:
        drug_counts[p['drug']] += 1
        if p['in_gt']:
            drug_hits[p['drug']] += 1
    sorted_drugs = sorted(drug_counts.items(), key=lambda x: -x[1])
    print(f"Drug: count (hits/total = %)")
    for drug, count in sorted_drugs[:15]:
        hits = drug_hits[drug]
        print(f"  {drug}: {count} preds ({hits}/{count} = {hits/count*100:.0f}%)")


if __name__ == '__main__':
    main()
