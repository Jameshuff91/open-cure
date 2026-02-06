#!/usr/bin/env python3
"""
h400: Deploy Category-Specific k Values

Re-validates h66 findings in the production pipeline context:
- Uses production predictor with h170 selective boosting
- 5-seed disease-holdout evaluation
- Tests h66's recommended k values vs default k=20

h66 recommended k values:
- k=5: dermatological, cardiovascular, psychiatric, respiratory
- k=10: autoimmune, gastrointestinal
- k=20: infectious, neurological (default)
- k=30: cancer, metabolic, other

Success criteria: Overall R@30 improvement > 1pp (p<0.05)
"""

import json
import sys
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


# h66 recommended k values
CATEGORY_K = {
    "dermatological": 5,
    "cardiovascular": 5,
    "psychiatric": 5,
    "respiratory": 5,
    "autoimmune": 10,
    "gastrointestinal": 10,
    "infectious": 20,
    "neurological": 20,
    "cancer": 30,
    "metabolic": 30,
    "other": 30,
}

DEFAULT_K = 20
SEEDS = [42, 123, 456, 789, 2024]


def evaluate_knn_hit_at_30(
    predictor: DrugRepurposingPredictor,
    test_disease_id: str,
    gt_drugs: Set[str],
    k: int,
) -> Tuple[bool, int]:
    """Evaluate if kNN achieves hit@30 for a single disease using production pipeline.

    Returns (hit, n_gt_in_top30)
    """
    if test_disease_id not in predictor.embeddings:
        return False, 0

    test_emb = predictor.embeddings[test_disease_id].reshape(1, -1)
    sims = cosine_similarity(test_emb, predictor.train_embeddings)[0]

    # h170: Apply selective category boost
    disease_name = predictor.disease_names.get(test_disease_id, test_disease_id)
    category = predictor.categorize_disease(disease_name)

    if category in SELECTIVE_BOOST_CATEGORIES:
        boosted_sims = sims.copy()
        for i, train_d in enumerate(predictor.train_diseases):
            if predictor.train_disease_categories.get(train_d) == category:
                boosted_sims[i] *= (1 + SELECTIVE_BOOST_ALPHA)
        top_k_idx = np.argsort(boosted_sims)[-k:]
        working_sims = boosted_sims
    else:
        top_k_idx = np.argsort(sims)[-k:]
        working_sims = sims

    # Aggregate drug scores from neighbors
    drug_scores: Dict[str, float] = defaultdict(float)
    for idx in top_k_idx:
        neighbor_disease = predictor.train_diseases[idx]
        neighbor_sim = working_sims[idx]
        for drug_id in predictor.ground_truth.get(neighbor_disease, set()):
            if drug_id in predictor.embeddings:
                drug_scores[drug_id] += neighbor_sim

    if not drug_scores:
        return False, 0

    sorted_drugs = sorted(drug_scores.items(), key=lambda x: x[1], reverse=True)
    top_30 = {d for d, _ in sorted_drugs[:30]}

    n_hits = len(top_30 & gt_drugs)
    return n_hits > 0, n_hits


def run_evaluation(
    predictor: DrugRepurposingPredictor,
    k_config: Dict[str, int],
    seed: int,
) -> Dict:
    """Run evaluation with given k configuration across all diseases."""
    all_diseases = [d for d in predictor.ground_truth if d in predictor.embeddings]

    # Split diseases
    rng = np.random.RandomState(seed)
    shuffled = list(all_diseases)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * 0.8)
    train_ids = set(shuffled[:split_idx])
    test_ids = shuffled[split_idx:]

    # Rebuild training index from split
    orig_train = predictor.train_diseases
    orig_emb = predictor.train_embeddings
    orig_cat = predictor.train_disease_categories
    orig_freq = dict(predictor.drug_train_freq)

    predictor.train_diseases = [d for d in train_ids if d in predictor.embeddings]
    predictor.train_embeddings = np.array(
        [predictor.embeddings[d] for d in predictor.train_diseases], dtype=np.float32
    )
    predictor.train_disease_categories = {
        d: predictor.categorize_disease(predictor.disease_names.get(d, d))
        for d in predictor.train_diseases
    }

    # Recompute training frequency from train diseases only
    new_freq = defaultdict(int)
    for d in train_ids:
        if d in predictor.ground_truth:
            for drug_id in predictor.ground_truth[d]:
                new_freq[drug_id] += 1
    predictor.drug_train_freq = dict(new_freq)

    # Evaluate on test diseases
    category_results = defaultdict(lambda: {"hits": 0, "total": 0})
    overall_hits = 0
    overall_total = 0

    for disease_id in test_ids:
        if disease_id not in predictor.embeddings:
            continue

        gt_drugs = {d for d in predictor.ground_truth.get(disease_id, set())
                    if d in predictor.embeddings}
        if not gt_drugs:
            continue

        disease_name = predictor.disease_names.get(disease_id, disease_id)
        category = predictor.categorize_disease(disease_name)
        k = k_config.get(category, DEFAULT_K)

        hit, n_hits = evaluate_knn_hit_at_30(predictor, disease_id, gt_drugs, k)

        category_results[category]["total"] += 1
        if hit:
            category_results[category]["hits"] += 1

        overall_total += 1
        if hit:
            overall_hits += 1

    # Restore original
    predictor.train_diseases = orig_train
    predictor.train_embeddings = orig_emb
    predictor.train_disease_categories = orig_cat
    predictor.drug_train_freq = orig_freq

    overall_r30 = overall_hits / overall_total * 100 if overall_total > 0 else 0

    cat_r30 = {}
    for cat, stats in category_results.items():
        r30 = stats["hits"] / stats["total"] * 100 if stats["total"] > 0 else 0
        cat_r30[cat] = {"r30": r30, "hits": stats["hits"], "total": stats["total"]}

    return {
        "overall_r30": overall_r30,
        "overall_hits": overall_hits,
        "overall_total": overall_total,
        "category_r30": cat_r30,
    }


def main() -> None:
    print("=" * 70)
    print("h400: Deploy Category-Specific k Values")
    print("=" * 70)

    print("\nLoading predictor...")
    predictor = DrugRepurposingPredictor()
    print(f"Diseases: {len(predictor.ground_truth)}")

    # Config 1: Default k=20 for all categories
    uniform_config = {cat: DEFAULT_K for cat in CATEGORY_K}

    # Config 2: h66 category-specific k values
    category_config = dict(CATEGORY_K)

    # Run both configs across 5 seeds
    uniform_results = []
    category_results = []

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\nSeed {seed} ({seed_idx+1}/{len(SEEDS)})...")

        print("  Uniform k=20...")
        u_result = run_evaluation(predictor, uniform_config, seed)
        uniform_results.append(u_result)

        print("  Category-specific k...")
        c_result = run_evaluation(predictor, category_config, seed)
        category_results.append(c_result)

        print(f"  Uniform R@30: {u_result['overall_r30']:.1f}%")
        print(f"  Category R@30: {c_result['overall_r30']:.1f}%")
        print(f"  Delta: {c_result['overall_r30'] - u_result['overall_r30']:+.1f}pp")

    # Aggregate
    u_r30s = [r["overall_r30"] for r in uniform_results]
    c_r30s = [r["overall_r30"] for r in category_results]
    deltas = [c - u for c, u in zip(c_r30s, u_r30s)]

    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS (5 seeds)")
    print("=" * 70)

    u_mean, u_std = np.mean(u_r30s), np.std(u_r30s)
    c_mean, c_std = np.mean(c_r30s), np.std(c_r30s)
    d_mean, d_std = np.mean(deltas), np.std(deltas)

    print(f"\nUniform k=20:     {u_mean:.1f}% ± {u_std:.1f}%")
    print(f"Category k:       {c_mean:.1f}% ± {c_std:.1f}%")
    print(f"Delta:            {d_mean:+.1f}pp ± {d_std:.1f}pp")

    # Paired t-test
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(c_r30s, u_r30s)
    print(f"Paired t-test:    t={t_stat:.2f}, p={p_value:.4f}")
    print(f"Significant (p<0.05): {'YES' if p_value < 0.05 else 'NO'}")

    # Per-category analysis
    print(f"\n--- PER-CATEGORY R@30 ---")
    print(f"{'Category':<18s} {'k=20':>8s} {'Cat-k':>8s} {'Delta':>8s} {'Cat k':>6s} {'N':>5s}")
    print("-" * 60)

    all_categories = set()
    for r in uniform_results + category_results:
        all_categories.update(r["category_r30"].keys())

    for cat in sorted(all_categories):
        u_cat_r30s = [r["category_r30"].get(cat, {}).get("r30", 0) for r in uniform_results]
        c_cat_r30s = [r["category_r30"].get(cat, {}).get("r30", 0) for r in category_results]
        u_cat_mean = np.mean(u_cat_r30s)
        c_cat_mean = np.mean(c_cat_r30s)
        delta = c_cat_mean - u_cat_mean
        cat_k = CATEGORY_K.get(cat, DEFAULT_K)
        n = int(np.mean([r["category_r30"].get(cat, {}).get("total", 0) for r in uniform_results]))
        print(f"  {cat:<16s} {u_cat_mean:7.1f}% {c_cat_mean:7.1f}% {delta:+7.1f}pp  k={cat_k:<3d} n={n}")

    # Save results
    output = {
        "uniform_k20": {"mean_r30": round(u_mean, 1), "std": round(u_std, 1), "per_seed": u_r30s},
        "category_k": {"mean_r30": round(c_mean, 1), "std": round(c_std, 1), "per_seed": c_r30s},
        "delta": {"mean": round(d_mean, 1), "std": round(d_std, 1), "per_seed": deltas},
        "t_stat": round(t_stat, 3),
        "p_value": round(p_value, 4),
        "category_k_values": CATEGORY_K,
    }

    output_path = Path(__file__).parent.parent / "data" / "analysis" / "h400_category_k.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
