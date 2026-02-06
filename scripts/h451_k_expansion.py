#!/usr/bin/env python3
"""
h451: k-Expansion for Rank Stabilization

h436 showed that SUBSAMPLING training diseases hurts (removes good neighbors).
This tests the OPPOSITE: using MULTIPLE k values (k=15,20,25,30) and
aggregating rankings via score averaging or Borda count.

Intuition: Drugs ranked highly across multiple k values are more stable
predictions. Expanding k includes more neighbors → more data → more stable.

Method:
1. For each disease, compute kNN scores at k=15,20,25,30
2. Aggregate via: (a) score normalization + average, (b) Borda count
3. Compare rank stability and R@30 vs single k=20
4. If stable, test on 5-seed holdout

Success criteria:
- Full→holdout rank-20 crossings reduced by >20% (from h437 baseline 4.1/disease)
- R@30 maintained or improved on holdout
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


def knn_scores_at_k(
    predictor: DrugRepurposingPredictor,
    disease_id: str,
    k: int,
) -> Dict[str, float]:
    """Compute kNN drug scores at a specific k value."""
    if disease_id not in predictor.embeddings:
        return {}

    test_emb = predictor.embeddings[disease_id].reshape(1, -1)
    category = predictor.categorize_disease(
        predictor.disease_names.get(disease_id, disease_id)
    )
    use_boost = category in SELECTIVE_BOOST_CATEGORIES

    sims = cosine_similarity(test_emb, predictor.train_embeddings)[0]

    if use_boost:
        boosted_sims = sims.copy()
        for i, train_d in enumerate(predictor.train_diseases):
            if predictor.train_disease_categories.get(train_d) == category:
                boosted_sims[i] *= (1 + SELECTIVE_BOOST_ALPHA)
        working_sims = boosted_sims
    else:
        working_sims = sims

    top_k_idx = np.argsort(working_sims)[-k:]

    drug_scores: Dict[str, float] = defaultdict(float)
    for idx in top_k_idx:
        neighbor_disease = predictor.train_diseases[idx]
        neighbor_sim = working_sims[idx]
        if neighbor_disease in predictor.ground_truth:
            for drug_id in predictor.ground_truth[neighbor_disease]:
                if drug_id in predictor.embeddings:
                    drug_scores[drug_id] += neighbor_sim

    return dict(drug_scores)


def multi_k_score_average(
    predictor: DrugRepurposingPredictor,
    disease_id: str,
    k_values: List[int],
) -> Dict[str, float]:
    """Average normalized scores across multiple k values."""
    all_scores: Dict[str, List[float]] = defaultdict(list)

    for k in k_values:
        scores = knn_scores_at_k(predictor, disease_id, k)
        if not scores:
            continue
        # Normalize scores to [0, 1] per k
        max_score = max(scores.values()) if scores else 1.0
        if max_score == 0:
            max_score = 1.0
        for drug_id, score in scores.items():
            all_scores[drug_id].append(score / max_score)

    # Average across k values (missing = 0)
    n_k = len(k_values)
    averaged = {}
    for drug_id, score_list in all_scores.items():
        # Pad with zeros for k values where drug didn't appear
        while len(score_list) < n_k:
            score_list.append(0.0)
        averaged[drug_id] = np.mean(score_list)

    return averaged


def multi_k_borda_count(
    predictor: DrugRepurposingPredictor,
    disease_id: str,
    k_values: List[int],
    top_n: int = 50,
) -> Dict[str, float]:
    """Borda count aggregation: rank drugs per k, sum inverse ranks."""
    all_ranks: Dict[str, float] = defaultdict(float)

    for k in k_values:
        scores = knn_scores_at_k(predictor, disease_id, k)
        if not scores:
            continue
        # Rank drugs (1 = best)
        sorted_drugs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for rank_idx, (drug_id, _) in enumerate(sorted_drugs[:top_n]):
            # Borda score: higher for better ranks
            all_ranks[drug_id] += (top_n - rank_idx) / top_n

    return dict(all_ranks)


def get_top_n(scores: Dict[str, float], n: int) -> List[str]:
    """Return top-n drug IDs by score."""
    return [d for d, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]]


def main() -> None:
    print("=" * 70)
    print("h451: k-Expansion for Rank Stabilization")
    print("=" * 70)

    print("\nLoading predictor...")
    t0 = time.time()
    predictor = DrugRepurposingPredictor()
    print(f"Loaded in {time.time() - t0:.1f}s")

    diseases = [d for d in predictor.ground_truth if d in predictor.embeddings]
    print(f"Diseases: {len(diseases)}")

    # Load GT for evaluation
    gt_path = predictor.reference_dir / "expanded_ground_truth.json"
    with open(gt_path) as f:
        gt_data = json.load(f)

    k_values = [15, 20, 25, 30]

    # ===== PART 1: Full-data comparison =====
    print(f"\n{'='*70}")
    print(f"PART 1: Full-Data Comparison (k=20 vs multi-k)")
    print(f"{'='*70}")

    methods = {
        "k=20 (standard)": lambda d: knn_scores_at_k(predictor, d, 20),
        "k=30": lambda d: knn_scores_at_k(predictor, d, 30),
        "multi-k avg [15,20,25,30]": lambda d: multi_k_score_average(predictor, d, k_values),
        "multi-k borda [15,20,25,30]": lambda d: multi_k_borda_count(predictor, d, k_values),
    }

    method_hits: Dict[str, Dict[str, int]] = {m: {"top20": 0, "top30": 0, "total_gt": 0} for m in methods}
    method_rankings: Dict[str, Dict[str, List[str]]] = {m: {} for m in methods}

    t1 = time.time()
    for i, d_id in enumerate(diseases):
        if d_id not in predictor.embeddings or d_id not in gt_data:
            continue

        gt_drugs = set()
        for entry in gt_data[d_id]:
            if isinstance(entry, str):
                gt_drugs.add(entry)
            elif isinstance(entry, dict):
                gt_drugs.add(entry.get("drug_id") or entry.get("drug", ""))
        if not gt_drugs:
            continue

        for method_name, method_fn in methods.items():
            scores = method_fn(d_id)
            top30 = get_top_n(scores, 30)
            top20 = top30[:20]

            method_hits[method_name]["top20"] += len(set(top20) & gt_drugs)
            method_hits[method_name]["top30"] += len(set(top30) & gt_drugs)
            method_hits[method_name]["total_gt"] += len(gt_drugs)
            method_rankings[method_name][d_id] = top30

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t1
            print(f"  {i+1}/{len(diseases)} ({elapsed:.1f}s)")

    print(f"\nCompleted in {time.time() - t1:.1f}s")

    print(f"\n--- Full-Data Recall ---")
    for method_name in methods:
        h = method_hits[method_name]
        r20 = h["top20"] / h["total_gt"] * 100
        r30 = h["top30"] / h["total_gt"] * 100
        print(f"  {method_name:35s}: R@20={r20:.2f}%, R@30={r30:.2f}%")

    # Compare rank overlap between k=20 and multi-k methods
    print(f"\n--- Rank Overlap (k=20 vs alternatives) ---")
    std_rankings = method_rankings["k=20 (standard)"]
    for method_name in methods:
        if method_name == "k=20 (standard)":
            continue
        alt_rankings = method_rankings[method_name]
        overlaps_20 = []
        overlaps_30 = []
        for d_id in std_rankings:
            if d_id in alt_rankings:
                std20 = set(std_rankings[d_id][:20])
                alt20 = set(alt_rankings[d_id][:20])
                std30 = set(std_rankings[d_id])
                alt30 = set(alt_rankings[d_id])
                if std20 and alt20:
                    overlaps_20.append(len(std20 & alt20) / 20)
                if std30 and alt30:
                    overlaps_30.append(len(std30 & alt30) / 30)
        if overlaps_20:
            print(f"  {method_name:35s}: top-20 overlap={np.mean(overlaps_20)*100:.1f}%, top-30 overlap={np.mean(overlaps_30)*100:.1f}%")

    # ===== PART 2: Holdout evaluation =====
    print(f"\n{'='*70}")
    print(f"PART 2: Holdout Evaluation (5-seed)")
    print(f"{'='*70}")

    seeds = [42, 123, 456, 789, 2024]
    holdout_r30: Dict[str, List[float]] = {m: [] for m in methods}

    for seed_idx, seed in enumerate(seeds):
        print(f"\n--- Seed {seed} ({seed_idx+1}/{len(seeds)}) ---")

        rng = np.random.RandomState(seed)
        shuffled = list(diseases)
        rng.shuffle(shuffled)
        split_idx = int(len(shuffled) * 0.8)
        train_ids = shuffled[:split_idx]
        holdout_ids = shuffled[split_idx:]

        # Save originals
        orig_train_diseases = list(predictor.train_diseases)
        orig_train_embeddings = predictor.train_embeddings.copy()
        orig_train_categories = dict(predictor.train_disease_categories)
        orig_drug_freq = dict(predictor.drug_train_freq)

        # Rebuild from training only
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

        # Evaluate each method on holdout
        method_seed_hits: Dict[str, Dict[str, int]] = {m: {"top30": 0, "total_gt": 0} for m in methods}

        for d_id in holdout_ids:
            if d_id not in predictor.embeddings or d_id not in gt_data:
                continue
            gt_drugs = set()
            for entry in gt_data[d_id]:
                if isinstance(entry, str):
                    gt_drugs.add(entry)
                elif isinstance(entry, dict):
                    gt_drugs.add(entry.get("drug_id") or entry.get("drug", ""))
            if not gt_drugs:
                continue

            for method_name, method_fn in methods.items():
                scores = method_fn(d_id)
                top30 = get_top_n(scores, 30)
                method_seed_hits[method_name]["top30"] += len(set(top30) & gt_drugs)
                method_seed_hits[method_name]["total_gt"] += len(gt_drugs)

        for method_name in methods:
            h = method_seed_hits[method_name]
            r30 = h["top30"] / h["total_gt"] * 100 if h["total_gt"] > 0 else 0
            holdout_r30[method_name].append(r30)

        for method_name in methods:
            print(f"  {method_name:35s}: R@30={holdout_r30[method_name][-1]:.2f}%")

        # Restore
        predictor.train_diseases = orig_train_diseases
        predictor.train_embeddings = orig_train_embeddings
        predictor.train_disease_categories = orig_train_categories
        predictor.drug_train_freq = orig_drug_freq

    # Aggregate holdout results
    print(f"\n{'='*70}")
    print(f"AGGREGATE HOLDOUT RESULTS (5-seed)")
    print(f"{'='*70}")
    for method_name in methods:
        vals = holdout_r30[method_name]
        print(f"  {method_name:35s}: {np.mean(vals):.2f}% ± {np.std(vals):.2f}%")

    baseline_mean = np.mean(holdout_r30["k=20 (standard)"])

    print(f"\n--- Deltas vs k=20 ---")
    for method_name in methods:
        if method_name == "k=20 (standard)":
            continue
        delta = np.mean(holdout_r30[method_name]) - baseline_mean
        print(f"  {method_name:35s}: {delta:+.2f}pp")

    # ===== PART 3: Full→Holdout rank drift =====
    print(f"\n{'='*70}")
    print(f"PART 3: Full→Holdout Rank Drift (seed=42)")
    print(f"{'='*70}")

    seed = 42
    rng = np.random.RandomState(seed)
    shuffled = list(diseases)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * 0.8)
    train_ids = shuffled[:split_idx]
    holdout_ids = shuffled[split_idx:]

    # Full-data rankings for holdout diseases
    full_rankings: Dict[str, Dict[str, List[str]]] = {m: {} for m in methods}
    for d_id in holdout_ids:
        if d_id not in predictor.embeddings:
            continue
        for method_name, method_fn in methods.items():
            scores = method_fn(d_id)
            full_rankings[method_name][d_id] = get_top_n(scores, 30)

    # Rebuild from training
    orig_train_diseases = list(predictor.train_diseases)
    orig_train_embeddings = predictor.train_embeddings.copy()
    orig_train_categories = dict(predictor.train_disease_categories)
    orig_drug_freq = dict(predictor.drug_train_freq)

    predictor.train_diseases = [d for d in train_ids if d in predictor.embeddings]
    predictor.train_embeddings = np.array(
        [predictor.embeddings[d] for d in predictor.train_diseases], dtype=np.float32
    )
    predictor.train_disease_categories = {}
    for d in predictor.train_diseases:
        name = predictor.disease_names.get(d, d)
        predictor.train_disease_categories[d] = predictor.categorize_disease(name)

    # Holdout rankings
    hold_rankings: Dict[str, Dict[str, List[str]]] = {m: {} for m in methods}
    for d_id in holdout_ids:
        if d_id not in predictor.embeddings:
            continue
        for method_name, method_fn in methods.items():
            scores = method_fn(d_id)
            hold_rankings[method_name][d_id] = get_top_n(scores, 30)

    # Restore
    predictor.train_diseases = orig_train_diseases
    predictor.train_embeddings = orig_train_embeddings
    predictor.train_disease_categories = orig_train_categories
    predictor.drug_train_freq = orig_drug_freq

    # Compare drift
    print(f"\n--- Rank Drift Metrics ---")
    for method_name in methods:
        crossings = []
        jaccards = []
        for d_id in full_rankings[method_name]:
            if d_id not in hold_rankings[method_name]:
                continue
            full_20 = set(full_rankings[method_name][d_id][:20])
            hold_20 = set(hold_rankings[method_name][d_id][:20])
            crossings.append(len(full_20 - hold_20))
            union = full_20 | hold_20
            if union:
                jaccards.append(len(full_20 & hold_20) / len(union))

        if crossings:
            print(f"  {method_name:35s}: crossings={np.mean(crossings):.2f}, Jaccard={np.mean(jaccards):.3f}")

    # Save results
    results = {
        "hypothesis": "h451",
        "title": "k-Expansion for Rank Stabilization",
        "k_values": k_values,
        "holdout_r30": {m: {"mean": round(np.mean(v), 2), "std": round(np.std(v), 2)} for m, v in holdout_r30.items()},
        "full_data_recall": {
            m: {
                "R@20": round(method_hits[m]["top20"] / method_hits[m]["total_gt"] * 100, 2),
                "R@30": round(method_hits[m]["top30"] / method_hits[m]["total_gt"] * 100, 2),
            }
            for m in methods
        },
    }

    output_path = Path(__file__).parent.parent / "data" / "analysis" / "h451_k_expansion.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Conclusion
    best_method = max(holdout_r30.items(), key=lambda x: np.mean(x[1]))
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")
    print(f"Best method on holdout: {best_method[0]} ({np.mean(best_method[1]):.2f}%)")
    best_delta = np.mean(best_method[1]) - baseline_mean
    if best_delta > 1.0:
        print(f"POSITIVE: {best_method[0]} improves R@30 by {best_delta:+.2f}pp")
    elif best_delta > -1.0:
        print(f"NEUTRAL: No method significantly improves over k=20")
    else:
        print(f"NEGATIVE: All alternatives worse than k=20")


if __name__ == "__main__":
    main()
