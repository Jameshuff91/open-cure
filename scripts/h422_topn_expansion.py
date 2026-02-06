#!/usr/bin/env python3
"""
h422: Expand Top-N from 30 to 50 with Target Overlap Rescue

h417 showed target_overlap>=3 at rank 21-30 has 23.6% holdout precision.
This tests whether target overlap continues to be a useful signal at rank 31-50.

Key question: If we expand predictions to top-50, do drugs at rank 31-50
with target_overlap>=3 have meaningful precision?

Method:
1. Run predictions with top_n=50
2. Compute target_overlap for rank 31-50 predictions
3. Measure precision by rank bucket and overlap threshold
4. Run 5-seed holdout validation
5. Success: >20% holdout precision for overlap>=3 at rank 31-50
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from production_predictor import (
    DrugRepurposingPredictor,
    SELECTIVE_BOOST_CATEGORIES,
    SELECTIVE_BOOST_ALPHA,
)


def get_knn_scores(
    predictor: DrugRepurposingPredictor,
    disease_id: str,
    k: int = 20,
    top_n: int = 50,
) -> List[Tuple[str, int, float]]:
    """Get kNN drug scores for a disease, returning top_n ranked drugs."""
    if disease_id not in predictor.embeddings:
        return []

    test_emb = predictor.embeddings[disease_id].reshape(1, -1)
    category = predictor.categorize_disease(
        predictor.disease_names.get(disease_id, disease_id)
    )

    sims = cosine_similarity(test_emb, predictor.train_embeddings)[0]

    if category in SELECTIVE_BOOST_CATEGORIES:
        boosted = sims.copy()
        for i, td in enumerate(predictor.train_diseases):
            if predictor.train_disease_categories.get(td) == category:
                boosted[i] *= (1 + SELECTIVE_BOOST_ALPHA)
        top_k_idx = np.argsort(boosted)[-k:]
        working = boosted
    else:
        top_k_idx = np.argsort(sims)[-k:]
        working = sims

    drug_scores: Dict[str, float] = defaultdict(float)
    for idx in top_k_idx:
        nd = predictor.train_diseases[idx]
        ns = working[idx]
        if nd in predictor.ground_truth:
            for drug_id in predictor.ground_truth[nd]:
                if drug_id in predictor.embeddings:
                    drug_scores[drug_id] += ns

    sorted_drugs = sorted(drug_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [(drug_id, rank + 1, score) for rank, (drug_id, score) in enumerate(sorted_drugs)]


def main() -> None:
    print("=" * 70)
    print("h422: Expand Top-N from 30 to 50 with Target Overlap Rescue")
    print("=" * 70)

    print("\nLoading predictor...")
    t0 = time.time()
    predictor = DrugRepurposingPredictor()
    print(f"Loaded in {time.time() - t0:.1f}s")

    diseases = [d for d in predictor.ground_truth if d in predictor.embeddings]
    print(f"Diseases: {len(diseases)}")

    # ===== PART 1: Full-Data Analysis =====
    print(f"\n{'='*70}")
    print("PART 1: Full-Data Precision by Rank Bucket and Target Overlap")
    print(f"{'='*70}")

    # rank buckets: 1-5, 6-10, 11-20, 21-30, 31-40, 41-50
    rank_buckets = [(1, 5), (6, 10), (11, 20), (21, 30), (31, 40), (41, 50)]
    overlap_thresholds = [0, 1, 2, 3, 5]

    # Collect results: (rank_bucket, overlap_threshold) -> {hits, total}
    bucket_overlap_stats: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"hits": 0, "total": 0})
    )
    bucket_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"hits": 0, "total": 0})

    # Also track how many rank 31-50 predictions have overlap >= threshold
    rank31_50_by_overlap: Dict[int, int] = defaultdict(int)  # overlap_count -> n_predictions

    t1 = time.time()
    for i, d_id in enumerate(diseases):
        if d_id not in predictor.embeddings:
            continue

        gt_drugs = predictor.ground_truth.get(d_id, set())
        if not gt_drugs:
            continue

        preds = get_knn_scores(predictor, d_id, k=20, top_n=50)

        for drug_id, rank, score in preds:
            is_hit = drug_id in gt_drugs
            overlap = predictor._get_target_overlap_count(drug_id, d_id)

            # Find bucket
            for lo, hi in rank_buckets:
                if lo <= rank <= hi:
                    bucket_label = f"rank {lo}-{hi}"
                    bucket_stats[bucket_label]["total"] += 1
                    if is_hit:
                        bucket_stats[bucket_label]["hits"] += 1

                    for thresh in overlap_thresholds:
                        if overlap >= thresh:
                            label = f"overlap>={thresh}"
                            bucket_overlap_stats[bucket_label][label]["total"] += 1
                            if is_hit:
                                bucket_overlap_stats[bucket_label][label]["hits"] += 1

                    if 31 <= rank <= 50:
                        rank31_50_by_overlap[overlap] += 1
                    break

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(diseases)} ({time.time() - t1:.1f}s)")

    print(f"\nCompleted in {time.time() - t1:.1f}s")

    # Print results
    print(f"\n--- Precision by Rank Bucket (no overlap filter) ---")
    for lo, hi in rank_buckets:
        label = f"rank {lo}-{hi}"
        s = bucket_stats[label]
        prec = s["hits"] / s["total"] * 100 if s["total"] > 0 else 0
        print(f"  {label:15s}: {prec:5.1f}% ({s['hits']:4d}/{s['total']:5d})")

    print(f"\n--- Precision by Rank Bucket × Target Overlap ---")
    header = f"{'Bucket':15s}"
    for thresh in overlap_thresholds:
        header += f" | overlap>={thresh:d}"
    print(header)
    print("-" * len(header))

    for lo, hi in rank_buckets:
        label = f"rank {lo}-{hi}"
        row = f"{label:15s}"
        for thresh in overlap_thresholds:
            olabel = f"overlap>={thresh}"
            s = bucket_overlap_stats[label][olabel]
            prec = s["hits"] / s["total"] * 100 if s["total"] > 0 else 0
            n = s["total"]
            row += f" | {prec:5.1f}% n={n:4d}"
        print(row)

    # Coverage analysis: how many rank 31-50 predictions have overlap?
    print(f"\n--- Rank 31-50 Target Overlap Distribution ---")
    total_31_50 = sum(rank31_50_by_overlap.values())
    for overlap in sorted(rank31_50_by_overlap.keys()):
        n = rank31_50_by_overlap[overlap]
        pct = n / total_31_50 * 100 if total_31_50 > 0 else 0
        print(f"  overlap={overlap:2d}: {n:5d} predictions ({pct:.1f}%)")

    cumulative = 0
    print(f"\n--- Cumulative: rank 31-50 predictions with overlap >= threshold ---")
    for thresh in [1, 2, 3, 5, 10]:
        n = sum(v for k, v in rank31_50_by_overlap.items() if k >= thresh)
        pct = n / total_31_50 * 100 if total_31_50 > 0 else 0
        print(f"  overlap>={thresh:2d}: {n:5d} predictions ({pct:.1f}%)")

    # ===== PART 2: Holdout Validation =====
    print(f"\n{'='*70}")
    print("PART 2: Holdout Validation (5-seed)")
    print(f"{'='*70}")

    seeds = [42, 123, 456, 789, 2024]

    # Track per-seed results
    holdout_bucket_prec: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for seed_idx, seed in enumerate(seeds):
        print(f"\n--- Seed {seed} ({seed_idx+1}/{len(seeds)}) ---")

        rng = np.random.RandomState(seed)
        shuffled = list(diseases)
        rng.shuffle(shuffled)
        split_idx = int(len(shuffled) * 0.8)
        train_ids = shuffled[:split_idx]
        holdout_ids = shuffled[split_idx:]

        # Save originals
        orig = {
            "train_diseases": list(predictor.train_diseases),
            "train_embeddings": predictor.train_embeddings.copy(),
            "train_categories": dict(predictor.train_disease_categories),
            "drug_freq": dict(predictor.drug_train_freq),
        }

        # Rebuild from training
        predictor.train_diseases = [d for d in train_ids if d in predictor.embeddings]
        predictor.train_embeddings = np.array(
            [predictor.embeddings[d] for d in predictor.train_diseases], dtype=np.float32
        )
        predictor.train_disease_categories = {}
        for d in predictor.train_diseases:
            name = predictor.disease_names.get(d, d)
            predictor.train_disease_categories[d] = predictor.categorize_disease(name)

        new_freq: Dict[str, int] = defaultdict(int)
        for d_id in train_ids:
            if d_id in predictor.ground_truth:
                for drug_id in predictor.ground_truth[d_id]:
                    new_freq[drug_id] += 1
        predictor.drug_train_freq = dict(new_freq)

        # Evaluate holdout diseases
        seed_bucket_overlap: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: {"hits": 0, "total": 0})
        )

        for d_id in holdout_ids:
            if d_id not in predictor.embeddings:
                continue
            gt_drugs = predictor.ground_truth.get(d_id, set())
            if not gt_drugs:
                continue

            preds = get_knn_scores(predictor, d_id, k=20, top_n=50)

            for drug_id, rank, score in preds:
                is_hit = drug_id in gt_drugs
                overlap = predictor._get_target_overlap_count(drug_id, d_id)

                for lo, hi in rank_buckets:
                    if lo <= rank <= hi:
                        bucket = f"rank {lo}-{hi}"
                        for thresh in overlap_thresholds:
                            if overlap >= thresh:
                                olabel = f"overlap>={thresh}"
                                seed_bucket_overlap[bucket][olabel]["total"] += 1
                                if is_hit:
                                    seed_bucket_overlap[bucket][olabel]["hits"] += 1
                        break

        # Record precision for this seed
        for lo, hi in rank_buckets:
            bucket = f"rank {lo}-{hi}"
            for thresh in overlap_thresholds:
                olabel = f"overlap>={thresh}"
                s = seed_bucket_overlap[bucket][olabel]
                prec = s["hits"] / s["total"] * 100 if s["total"] > 0 else 0
                holdout_bucket_prec[bucket][olabel].append(prec)

        # Print seed results for key comparisons
        for bucket in ["rank 21-30", "rank 31-40", "rank 41-50"]:
            for olabel in ["overlap>=0", "overlap>=3"]:
                s = seed_bucket_overlap[bucket][olabel]
                prec = s["hits"] / s["total"] * 100 if s["total"] > 0 else 0
                print(f"  {bucket} {olabel}: {prec:.1f}% (n={s['total']})")

        # Restore
        predictor.train_diseases = orig["train_diseases"]
        predictor.train_embeddings = orig["train_embeddings"]
        predictor.train_disease_categories = orig["train_categories"]
        predictor.drug_train_freq = orig["drug_freq"]

    # Aggregate holdout results
    print(f"\n{'='*70}")
    print("AGGREGATE HOLDOUT (5-seed mean ± std)")
    print(f"{'='*70}")

    print(f"\n{'Bucket':15s} | {'All (>=0)':20s} | {'overlap>=1':20s} | {'overlap>=3':20s} | {'overlap>=5':20s}")
    print("-" * 100)
    for lo, hi in rank_buckets:
        bucket = f"rank {lo}-{hi}"
        row = f"{bucket:15s}"
        for thresh in [0, 1, 3, 5]:
            olabel = f"overlap>={thresh}"
            vals = holdout_bucket_prec[bucket][olabel]
            if vals:
                mean_v = np.mean(vals)
                std_v = np.std(vals)
                row += f" | {mean_v:5.1f}% ± {std_v:4.1f}%"
            else:
                row += f" | {'N/A':>14s}"
        print(row)

    # Key question: is rank 31-50 with overlap>=3 above 20%?
    print(f"\n{'='*70}")
    print("KEY QUESTION: Is rank 31-50 + overlap>=3 viable?")
    print(f"{'='*70}")

    for bucket in ["rank 31-40", "rank 41-50"]:
        for olabel in ["overlap>=3", "overlap>=5"]:
            vals = holdout_bucket_prec[bucket][olabel]
            if vals:
                mean_v = np.mean(vals)
                std_v = np.std(vals)
                print(f"  {bucket} {olabel}: {mean_v:.1f}% ± {std_v:.1f}% (n_per_seed avg: {np.mean([bucket_overlap_stats[bucket][olabel]['total']/len(diseases)*len(holdout_ids) for _ in seeds]):.0f})")
                if mean_v >= 20:
                    print(f"    ✓ ABOVE 20% threshold - viable for MEDIUM tier rescue")
                else:
                    print(f"    ✗ Below 20% threshold")

    # Save results
    output = {
        "hypothesis": "h422",
        "holdout_aggregate": {},
    }
    for lo, hi in rank_buckets:
        bucket = f"rank {lo}-{hi}"
        output["holdout_aggregate"][bucket] = {}
        for thresh in overlap_thresholds:
            olabel = f"overlap>={thresh}"
            vals = holdout_bucket_prec[bucket][olabel]
            if vals:
                output["holdout_aggregate"][bucket][olabel] = {
                    "mean": round(float(np.mean(vals)), 1),
                    "std": round(float(np.std(vals)), 1),
                }

    output_path = Path(__file__).parent.parent / "data" / "analysis" / "h422_topn_expansion.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
