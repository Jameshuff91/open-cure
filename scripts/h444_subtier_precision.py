#!/usr/bin/env python3
"""
h444: Sub-Tier Precision Reporting: Top-5 vs Top-10 vs Top-20 Within Tier

h443 showed kNN rank creates 11-16pp precision gaps within tiers on holdout.
For clinical presentation, report sub-tier precision: e.g., MEDIUM rank 1-5
may have ~35% precision, rank 6-10 ~25%, rank 11-20 ~15%.

Method:
1. For each tier, bucket predictions by kNN rank (1-5, 6-10, 11-15, 16-20)
2. Compute precision per bucket
3. Also compute cumulative (top-5, top-10, top-15, top-20)
4. Validate on 5-seed holdout
5. Check if gradient is monotonic (higher rank = higher precision)

Success criteria:
- Monotonic decline across rank buckets within each tier
- >10pp gap between rank 1-5 and rank 16-20 for HIGH/MEDIUM tiers
- Holds on holdout validation
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from production_predictor import (
    ConfidenceTier,
    DrugRepurposingPredictor,
    DISEASE_HIERARCHY_GROUPS,
    extract_cancer_types,
)


RANK_BUCKETS = [(1, 5), (6, 10), (11, 15), (16, 20)]
CUMULATIVE_TOPS = [5, 10, 15, 20]
TIER_ORDER = ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]
SEEDS = [42, 123, 456, 789, 2024]


def split_diseases(all_diseases: List[str], seed: int, train_ratio: float = 0.8) -> Tuple[List[str], List[str]]:
    rng = np.random.RandomState(seed)
    shuffled = list(all_diseases)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def recompute_gt_structures(predictor: DrugRepurposingPredictor, train_disease_ids: Set[str]) -> Dict:
    originals = {
        "drug_train_freq": dict(predictor.drug_train_freq),
        "drug_to_diseases": dict(predictor.drug_to_diseases),
        "drug_cancer_types": dict(predictor.drug_cancer_types),
        "drug_disease_groups": dict(predictor.drug_disease_groups),
        "train_diseases": list(predictor.train_diseases),
        "train_embeddings": predictor.train_embeddings.copy(),
        "train_disease_categories": dict(predictor.train_disease_categories),
    }

    new_freq: Dict[str, int] = defaultdict(int)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            for drug_id in predictor.ground_truth[disease_id]:
                new_freq[drug_id] += 1
    predictor.drug_train_freq = dict(new_freq)

    new_d2d: Dict[str, Set[str]] = defaultdict(set)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            for drug_id in predictor.ground_truth[disease_id]:
                new_d2d[drug_id].add(disease_name.lower())
    predictor.drug_to_diseases = dict(new_d2d)

    new_cancer: Dict[str, Set[str]] = defaultdict(set)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            cancer_types = extract_cancer_types(disease_name)
            if cancer_types:
                for drug_id in predictor.ground_truth[disease_id]:
                    new_cancer[drug_id].update(cancer_types)
    predictor.drug_cancer_types = dict(new_cancer)

    new_groups: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            category = predictor.categorize_disease(disease_name)
            if category in DISEASE_HIERARCHY_GROUPS:
                for group_name, keywords in DISEASE_HIERARCHY_GROUPS[category].items():
                    if any(kw in disease_name.lower() for kw in keywords):
                        for drug_id in predictor.ground_truth[disease_id]:
                            new_groups[drug_id].add((category, group_name))
    predictor.drug_disease_groups = dict(new_groups)

    predictor.train_diseases = [d for d in train_disease_ids if d in predictor.embeddings]
    predictor.train_embeddings = np.array(
        [predictor.embeddings[d] for d in predictor.train_diseases], dtype=np.float32
    )
    predictor.train_disease_categories = {}
    for d in predictor.train_diseases:
        name = predictor.disease_names.get(d, d)
        predictor.train_disease_categories[d] = predictor.categorize_disease(name)

    return originals


def restore_originals(predictor: DrugRepurposingPredictor, originals: Dict) -> None:
    predictor.drug_train_freq = originals["drug_train_freq"]
    predictor.drug_to_diseases = originals["drug_to_diseases"]
    predictor.drug_cancer_types = originals["drug_cancer_types"]
    predictor.drug_disease_groups = originals["drug_disease_groups"]
    predictor.train_diseases = originals["train_diseases"]
    predictor.train_embeddings = originals["train_embeddings"]
    predictor.train_disease_categories = originals["train_disease_categories"]


def collect_predictions(predictor: DrugRepurposingPredictor, diseases: List[str]) -> List[Dict]:
    """Collect all predictions for given diseases."""
    all_preds = []
    for disease_id in diseases:
        if disease_id not in predictor.embeddings:
            continue
        gt_drugs = predictor.ground_truth.get(disease_id, set())
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        result = predictor.predict(disease_name, k=20, top_n=30, include_filtered=True)

        for pred in result.predictions:
            all_preds.append({
                "disease_id": disease_id,
                "drug_id": pred.drug_id,
                "rank": pred.rank,
                "tier": pred.confidence_tier.value,
                "knn_score": pred.knn_score,
                "is_gt": pred.drug_id in gt_drugs,
                "transe_consilience": pred.transe_consilience,
            })
    return all_preds


def analyze_rank_buckets(preds: List[Dict]) -> Dict:
    """Analyze precision by rank bucket for predictions."""
    tier_preds: Dict[str, list] = defaultdict(list)
    for p in preds:
        tier_preds[p["tier"]].append(p)

    results = {}
    for tier_name in TIER_ORDER:
        if tier_name not in tier_preds:
            continue
        tier_ps = tier_preds[tier_name]
        n_total = len(tier_ps)
        gt_total = sum(1 for p in tier_ps if p["is_gt"])
        overall_prec = gt_total / n_total * 100 if n_total > 0 else 0

        # Bucket analysis
        buckets = {}
        for lo, hi in RANK_BUCKETS:
            bucket_ps = [p for p in tier_ps if lo <= p["rank"] <= hi]
            n = len(bucket_ps)
            gt = sum(1 for p in bucket_ps if p["is_gt"])
            prec = gt / n * 100 if n > 0 else 0
            buckets[f"rank_{lo}_{hi}"] = {
                "n": n, "gt": gt, "precision": round(prec, 2)
            }

        # Cumulative analysis
        cumulative = {}
        for top_n in CUMULATIVE_TOPS:
            cum_ps = [p for p in tier_ps if p["rank"] <= top_n]
            n = len(cum_ps)
            gt = sum(1 for p in cum_ps if p["is_gt"])
            prec = gt / n * 100 if n > 0 else 0
            cumulative[f"top_{top_n}"] = {
                "n": n, "gt": gt, "precision": round(prec, 2)
            }

        # TransE consilience split within each rank bucket
        transe_buckets = {}
        for lo, hi in RANK_BUCKETS:
            bucket_ps = [p for p in tier_ps if lo <= p["rank"] <= hi]
            with_t = [p for p in bucket_ps if p["transe_consilience"]]
            without_t = [p for p in bucket_ps if not p["transe_consilience"]]
            t_prec = sum(1 for p in with_t if p["is_gt"]) / len(with_t) * 100 if with_t else 0
            no_t_prec = sum(1 for p in without_t if p["is_gt"]) / len(without_t) * 100 if without_t else 0
            transe_buckets[f"rank_{lo}_{hi}"] = {
                "with_transe": {"n": len(with_t), "precision": round(t_prec, 2)},
                "without_transe": {"n": len(without_t), "precision": round(no_t_prec, 2)},
                "gap": round(t_prec - no_t_prec, 2),
            }

        results[tier_name] = {
            "n": n_total,
            "gt": gt_total,
            "overall_precision": round(overall_prec, 2),
            "rank_buckets": buckets,
            "cumulative": cumulative,
            "transe_rank_buckets": transe_buckets,
        }

    return results


def main():
    print("=" * 70)
    print("h444: Sub-Tier Precision Reporting: Top-5 vs Top-10 vs Top-20")
    print("=" * 70)

    print("\nLoading predictor...")
    t0 = time.time()
    predictor = DrugRepurposingPredictor()
    print(f"Loaded in {time.time() - t0:.1f}s")

    diseases_with_gt = [d for d in predictor.ground_truth if d in predictor.embeddings]
    print(f"Diseases with GT and embeddings: {len(diseases_with_gt)}")

    # ===== PART 1: Full-data rank bucket analysis =====
    print("\n" + "=" * 70)
    print("PART 1: Full-Data Rank Bucket Analysis")
    print("=" * 70)

    t0 = time.time()
    all_preds = collect_predictions(predictor, diseases_with_gt)
    print(f"Collected {len(all_preds)} predictions in {time.time() - t0:.1f}s")

    full_results = analyze_rank_buckets(all_preds)

    for tier_name in TIER_ORDER:
        if tier_name not in full_results:
            continue
        tr = full_results[tier_name]
        print(f"\n--- {tier_name} (n={tr['n']}, GT={tr['gt']}, prec={tr['overall_precision']:.1f}%) ---")

        print(f"  {'Bucket':<12} {'n':>6} {'GT':>5} {'Precision':>10}")
        for lo, hi in RANK_BUCKETS:
            b = tr['rank_buckets'][f'rank_{lo}_{hi}']
            print(f"  rank {lo:>2}-{hi:<2}  {b['n']:>6} {b['gt']:>5} {b['precision']:>9.1f}%")

        print(f"\n  {'Cumulative':<12} {'n':>6} {'GT':>5} {'Precision':>10}")
        for top_n in CUMULATIVE_TOPS:
            c = tr['cumulative'][f'top_{top_n}']
            print(f"  top-{top_n:<7} {c['n']:>6} {c['gt']:>5} {c['precision']:>9.1f}%")

        # Check monotonicity
        bucket_precs = [tr['rank_buckets'][f'rank_{lo}_{hi}']['precision'] for lo, hi in RANK_BUCKETS]
        is_monotonic = all(bucket_precs[i] >= bucket_precs[i + 1] for i in range(len(bucket_precs) - 1))
        gap_1_5_vs_16_20 = bucket_precs[0] - bucket_precs[-1]
        print(f"\n  Monotonic: {'YES' if is_monotonic else 'NO'} | Gap (rank 1-5 vs 16-20): {gap_1_5_vs_16_20:+.1f}pp")

    # ===== PART 2: TransE x Rank Bucket =====
    print("\n" + "=" * 70)
    print("PART 2: TransE Consilience x Rank Bucket (Full Data)")
    print("=" * 70)

    for tier_name in TIER_ORDER:
        if tier_name not in full_results:
            continue
        tr = full_results[tier_name]
        print(f"\n--- {tier_name} ---")
        print(f"  {'Bucket':<12} {'TransE':>8} {'No TransE':>10} {'Gap':>8}")
        for lo, hi in RANK_BUCKETS:
            tb = tr['transe_rank_buckets'][f'rank_{lo}_{hi}']
            t_str = f"{tb['with_transe']['precision']:.1f}% (n={tb['with_transe']['n']})"
            nt_str = f"{tb['without_transe']['precision']:.1f}% (n={tb['without_transe']['n']})"
            print(f"  rank {lo:>2}-{hi:<2}  {t_str:>15} {nt_str:>15} {tb['gap']:>+7.1f}pp")

    # ===== PART 3: Holdout Validation =====
    print("\n" + "=" * 70)
    print("PART 3: Holdout Validation (5-seed)")
    print("=" * 70)

    # Store per-seed results for averaging
    holdout_bucket_precs: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    # holdout_bucket_precs[tier][bucket_name] = [prec_seed1, prec_seed2, ...]
    holdout_cumulative_precs: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        train_diseases, holdout_diseases = split_diseases(diseases_with_gt, seed)
        train_set = set(train_diseases)

        originals = recompute_gt_structures(predictor, train_set)

        holdout_preds = collect_predictions(predictor, holdout_diseases)
        seed_results = analyze_rank_buckets(holdout_preds)

        restore_originals(predictor, originals)

        for tier_name in TIER_ORDER:
            if tier_name not in seed_results:
                continue
            tr = seed_results[tier_name]
            bucket_strs = []
            for lo, hi in RANK_BUCKETS:
                b = tr['rank_buckets'][f'rank_{lo}_{hi}']
                holdout_bucket_precs[tier_name][f'rank_{lo}_{hi}'].append(b['precision'])
                bucket_strs.append(f"{lo}-{hi}: {b['precision']:.1f}%")
            print(f"  {tier_name}: {' | '.join(bucket_strs)} (n={tr['n']})")

            for top_n in CUMULATIVE_TOPS:
                c = tr['cumulative'][f'top_{top_n}']
                holdout_cumulative_precs[tier_name][f'top_{top_n}'].append(c['precision'])

    # ===== Holdout Summary =====
    print("\n" + "=" * 70)
    print("PART 4: Holdout Summary (5-seed mean ± std)")
    print("=" * 70)

    summary_data = {}
    for tier_name in TIER_ORDER:
        if tier_name not in holdout_bucket_precs:
            continue

        print(f"\n--- {tier_name} ---")
        print(f"  {'Bucket':<12} {'Holdout Mean':>14} {'Std':>6} {'Full-Data':>10}")

        tier_summary = {"buckets": {}, "cumulative": {}}

        for lo, hi in RANK_BUCKETS:
            key = f'rank_{lo}_{hi}'
            vals = holdout_bucket_precs[tier_name][key]
            mean_v = np.mean(vals)
            std_v = np.std(vals)
            full_v = full_results[tier_name]['rank_buckets'][key]['precision'] if tier_name in full_results else 0
            print(f"  rank {lo:>2}-{hi:<2}  {mean_v:>10.1f}% ± {std_v:>4.1f} {full_v:>9.1f}%")
            tier_summary["buckets"][key] = {
                "holdout_mean": round(float(mean_v), 2),
                "holdout_std": round(float(std_v), 2),
                "full_data": full_v,
                "values": [round(float(v), 2) for v in vals],
            }

        print()
        for top_n in CUMULATIVE_TOPS:
            key = f'top_{top_n}'
            vals = holdout_cumulative_precs[tier_name][key]
            mean_v = np.mean(vals)
            std_v = np.std(vals)
            full_v = full_results[tier_name]['cumulative'][key]['precision'] if tier_name in full_results else 0
            print(f"  top-{top_n:<7} {mean_v:>10.1f}% ± {std_v:>4.1f} {full_v:>9.1f}%")
            tier_summary["cumulative"][key] = {
                "holdout_mean": round(float(mean_v), 2),
                "holdout_std": round(float(std_v), 2),
                "full_data": full_v,
                "values": [round(float(v), 2) for v in vals],
            }

        # Check monotonicity on holdout
        bucket_means = [np.mean(holdout_bucket_precs[tier_name][f'rank_{lo}_{hi}']) for lo, hi in RANK_BUCKETS]
        is_monotonic = all(bucket_means[i] >= bucket_means[i + 1] for i in range(len(bucket_means) - 1))
        gap = bucket_means[0] - bucket_means[-1]
        print(f"\n  Monotonic (holdout): {'YES' if is_monotonic else 'NO'} | Gap: {gap:+.1f}pp")
        tier_summary["holdout_monotonic"] = is_monotonic
        tier_summary["holdout_gap_1_5_vs_16_20"] = round(float(gap), 2)

        summary_data[tier_name] = tier_summary

    # Save results
    output = {
        "hypothesis": "h444",
        "title": "Sub-Tier Precision Reporting",
        "full_data": full_results,
        "holdout_summary": summary_data,
    }

    output_path = Path(__file__).parent.parent / "data" / "analysis" / "h444_subtier_precision.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
