#!/usr/bin/env python3
"""
h441: Drug-Level Embedding Stability for Within-Tier Ranking

For each drug prediction in the system, measure how STABLE the drug's ranking
is across different diseases. A drug that always ranks 2-5 is a more reliable
prediction than one that bounces between 1 and 20.

Key question: Does rank consistency predict GT hit rate within tiers?

Method:
1. Run predictions for all diseases
2. For each drug, collect all its ranks across diseases
3. Compute rank consistency metrics (mean rank, CV of rank, min/max spread)
4. Split predictions by consistency and measure GT hit rate
5. Compare within-tier precision for consistent vs inconsistent drugs

If this works, it provides a within-tier ranking boost that h445 showed
TransE cannot provide.
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


def get_all_predictions(
    predictor: DrugRepurposingPredictor,
    diseases: List[str],
    k: int = 20,
    top_n: int = 30,
) -> Dict[str, List[Tuple[str, int, float]]]:
    """
    Run predictions for all diseases.
    Returns: dict of disease_id -> [(drug_id, rank, score), ...]
    """
    all_preds: Dict[str, List[Tuple[str, int, float]]] = {}

    for disease_id in diseases:
        if disease_id not in predictor.embeddings:
            continue

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

        preds = []
        for rank, (drug_id, score) in enumerate(sorted_drugs, 1):
            preds.append((drug_id, rank, score))
        all_preds[disease_id] = preds

    return all_preds


def compute_drug_stability(
    all_preds: Dict[str, List[Tuple[str, int, float]]],
) -> Dict[str, Dict]:
    """
    For each drug, compute rank stability across all diseases.

    Returns: drug_id -> {
        'n_appearances': int,
        'mean_rank': float,
        'std_rank': float,
        'cv_rank': float (coefficient of variation),
        'min_rank': int,
        'max_rank': int,
        'rank_spread': int,
        'pct_top5': float,  # % of appearances in top 5
        'pct_top10': float, # % of appearances in top 10
    }
    """
    drug_ranks: Dict[str, List[int]] = defaultdict(list)

    for disease_id, preds in all_preds.items():
        for drug_id, rank, score in preds:
            drug_ranks[drug_id].append(rank)

    drug_stability = {}
    for drug_id, ranks in drug_ranks.items():
        ranks_arr = np.array(ranks)
        mean_rank = np.mean(ranks_arr)
        std_rank = np.std(ranks_arr)
        cv = std_rank / mean_rank if mean_rank > 0 else 0

        drug_stability[drug_id] = {
            "n_appearances": len(ranks),
            "mean_rank": round(float(mean_rank), 2),
            "std_rank": round(float(std_rank), 2),
            "cv_rank": round(float(cv), 3),
            "min_rank": int(np.min(ranks_arr)),
            "max_rank": int(np.max(ranks_arr)),
            "rank_spread": int(np.max(ranks_arr) - np.min(ranks_arr)),
            "pct_top5": round(float(np.mean(ranks_arr <= 5) * 100), 1),
            "pct_top10": round(float(np.mean(ranks_arr <= 10) * 100), 1),
        }

    return drug_stability


def main() -> None:
    print("=" * 70)
    print("h441: Drug-Level Embedding Stability for Within-Tier Ranking")
    print("=" * 70)

    print("\nLoading predictor...")
    t0 = time.time()
    predictor = DrugRepurposingPredictor()
    print(f"Loaded in {time.time() - t0:.1f}s")

    diseases = [d for d in predictor.ground_truth if d in predictor.embeddings]
    print(f"Diseases: {len(diseases)}")

    # Run all predictions
    print("\nRunning predictions for all diseases...")
    t1 = time.time()
    all_preds = get_all_predictions(predictor, diseases)
    print(f"Completed in {time.time() - t1:.1f}s, {len(all_preds)} diseases")

    # Compute drug stability
    print("\nComputing drug stability metrics...")
    stability = compute_drug_stability(all_preds)
    print(f"Unique drugs predicted: {len(stability)}")

    # Distribution of stability metrics
    all_n = [s["n_appearances"] for s in stability.values()]
    all_cv = [s["cv_rank"] for s in stability.values()]
    all_spread = [s["rank_spread"] for s in stability.values()]

    print(f"\n--- Drug Stability Distribution ---")
    print(f"Appearances: mean={np.mean(all_n):.1f}, median={np.median(all_n):.0f}, max={max(all_n)}")
    print(f"CV of rank: mean={np.mean(all_cv):.3f}, median={np.median(all_cv):.3f}")
    print(f"Rank spread: mean={np.mean(all_spread):.1f}, median={np.median(all_spread):.0f}")

    # ===== KEY ANALYSIS: Does stability predict GT hit rate? =====
    print(f"\n{'='*70}")
    print("ANALYSIS: Stability vs GT Hit Rate")
    print(f"{'='*70}")

    # For each prediction, classify by drug stability and check GT hit
    # Stability bins: low CV (stable) vs high CV (unstable)
    cv_median = np.median(all_cv)
    spread_median = np.median(all_spread)

    # Collect all predictions with GT status
    results_by_stability: Dict[str, Dict[str, int]] = defaultdict(lambda: {"hits": 0, "total": 0})
    results_by_rank_bucket: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"hits": 0, "total": 0})
    )

    for disease_id, preds in all_preds.items():
        gt_drugs = predictor.ground_truth.get(disease_id, set())

        for drug_id, rank, score in preds:
            is_hit = drug_id in gt_drugs
            drug_stab = stability.get(drug_id, {})
            cv = drug_stab.get("cv_rank", 0)
            spread = drug_stab.get("rank_spread", 0)
            n_app = drug_stab.get("n_appearances", 0)

            # By CV bucket
            if cv <= cv_median:
                bucket = "low_cv (stable)"
            else:
                bucket = "high_cv (unstable)"

            results_by_stability[bucket]["total"] += 1
            if is_hit:
                results_by_stability[bucket]["hits"] += 1

            # By rank bucket × stability
            if rank <= 5:
                rb = "rank 1-5"
            elif rank <= 10:
                rb = "rank 6-10"
            elif rank <= 20:
                rb = "rank 11-20"
            else:
                rb = "rank 21-30"

            results_by_rank_bucket[rb][bucket]["total"] += 1
            if is_hit:
                results_by_rank_bucket[rb][bucket]["hits"] += 1

    print(f"\n--- Overall Precision by Stability ---")
    for bucket, stats in sorted(results_by_stability.items()):
        prec = stats["hits"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {bucket:25s}: {prec:.1f}% ({stats['hits']}/{stats['total']})")

    print(f"\n--- Precision by Rank Bucket × Stability ---")
    for rb in ["rank 1-5", "rank 6-10", "rank 11-20", "rank 21-30"]:
        print(f"\n  {rb}:")
        for stab_bucket in ["low_cv (stable)", "high_cv (unstable)"]:
            stats = results_by_rank_bucket[rb][stab_bucket]
            prec = stats["hits"] / stats["total"] * 100 if stats["total"] > 0 else 0
            print(f"    {stab_bucket:25s}: {prec:.1f}% ({stats['hits']}/{stats['total']})")

    # More granular: split by quartile
    print(f"\n--- Precision by CV Quartile ---")
    cv_q25 = np.percentile(all_cv, 25)
    cv_q50 = np.percentile(all_cv, 50)
    cv_q75 = np.percentile(all_cv, 75)
    print(f"CV quartiles: Q1={cv_q25:.3f}, Q2={cv_q50:.3f}, Q3={cv_q75:.3f}")

    quartile_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"hits": 0, "total": 0})

    for disease_id, preds in all_preds.items():
        gt_drugs = predictor.ground_truth.get(disease_id, set())
        for drug_id, rank, score in preds:
            is_hit = drug_id in gt_drugs
            cv = stability.get(drug_id, {}).get("cv_rank", 0)

            if cv <= cv_q25:
                q = "Q1 (most stable)"
            elif cv <= cv_q50:
                q = "Q2"
            elif cv <= cv_q75:
                q = "Q3"
            else:
                q = "Q4 (least stable)"

            quartile_stats[q]["total"] += 1
            if is_hit:
                quartile_stats[q]["hits"] += 1

    for q in ["Q1 (most stable)", "Q2", "Q3", "Q4 (least stable)"]:
        stats = quartile_stats[q]
        prec = stats["hits"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {q:25s}: {prec:.1f}% ({stats['hits']}/{stats['total']})")

    # Also check: drug appearance count vs precision
    print(f"\n--- Precision by Drug Appearance Count ---")
    app_bins = [(1, 5), (6, 20), (21, 50), (51, 100), (101, 500)]
    app_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"hits": 0, "total": 0})

    for disease_id, preds in all_preds.items():
        gt_drugs = predictor.ground_truth.get(disease_id, set())
        for drug_id, rank, score in preds:
            is_hit = drug_id in gt_drugs
            n_app = stability.get(drug_id, {}).get("n_appearances", 0)
            for lo, hi in app_bins:
                if lo <= n_app <= hi:
                    label = f"appears {lo}-{hi}x"
                    app_stats[label]["total"] += 1
                    if is_hit:
                        app_stats[label]["hits"] += 1
                    break

    for lo, hi in app_bins:
        label = f"appears {lo}-{hi}x"
        stats = app_stats[label]
        prec = stats["hits"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {label:25s}: {prec:.1f}% ({stats['hits']}/{stats['total']})")

    # ===== HOLDOUT VALIDATION =====
    print(f"\n{'='*70}")
    print("HOLDOUT VALIDATION (5-seed)")
    print(f"{'='*70}")

    seeds = [42, 123, 456, 789, 2024]
    holdout_by_stability: Dict[str, List[float]] = {"low_cv": [], "high_cv": []}
    holdout_by_app: Dict[str, List[float]] = {"rare": [], "common": []}

    for seed_idx, seed in enumerate(seeds):
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

        # Run predictions on TRAIN diseases to compute stability
        train_preds = get_all_predictions(predictor, train_ids)
        train_stability = compute_drug_stability(train_preds)

        train_cvs = [s["cv_rank"] for s in train_stability.values()]
        train_cv_median = np.median(train_cvs) if train_cvs else 0.5

        train_apps = [s["n_appearances"] for s in train_stability.values()]
        train_app_median = np.median(train_apps) if train_apps else 10

        # Evaluate on holdout
        low_cv_hits, low_cv_total = 0, 0
        high_cv_hits, high_cv_total = 0, 0
        rare_hits, rare_total = 0, 0
        common_hits, common_total = 0, 0

        holdout_preds = get_all_predictions(predictor, holdout_ids)
        for disease_id, preds in holdout_preds.items():
            gt_drugs = predictor.ground_truth.get(disease_id, set())
            for drug_id, rank, score in preds:
                is_hit = drug_id in gt_drugs
                stab = train_stability.get(drug_id, {})
                cv = stab.get("cv_rank", 999)
                n_app = stab.get("n_appearances", 0)

                if cv <= train_cv_median:
                    low_cv_total += 1
                    if is_hit:
                        low_cv_hits += 1
                else:
                    high_cv_total += 1
                    if is_hit:
                        high_cv_hits += 1

                if n_app <= train_app_median:
                    rare_total += 1
                    if is_hit:
                        rare_hits += 1
                else:
                    common_total += 1
                    if is_hit:
                        common_hits += 1

        low_prec = low_cv_hits / low_cv_total * 100 if low_cv_total > 0 else 0
        high_prec = high_cv_hits / high_cv_total * 100 if high_cv_total > 0 else 0
        rare_prec = rare_hits / rare_total * 100 if rare_total > 0 else 0
        common_prec = common_hits / common_total * 100 if common_total > 0 else 0

        holdout_by_stability["low_cv"].append(low_prec)
        holdout_by_stability["high_cv"].append(high_prec)
        holdout_by_app["rare"].append(rare_prec)
        holdout_by_app["common"].append(common_prec)

        print(f"Seed {seed}: low_cv={low_prec:.1f}% vs high_cv={high_prec:.1f}%  |  rare={rare_prec:.1f}% vs common={common_prec:.1f}%")

        # Restore
        predictor.train_diseases = orig["train_diseases"]
        predictor.train_embeddings = orig["train_embeddings"]
        predictor.train_disease_categories = orig["train_categories"]
        predictor.drug_train_freq = orig["drug_freq"]

    print(f"\n--- Holdout Summary ---")
    for group, vals in holdout_by_stability.items():
        print(f"  {group:10s}: {np.mean(vals):.1f}% ± {np.std(vals):.1f}%")
    delta_cv = np.mean(holdout_by_stability["low_cv"]) - np.mean(holdout_by_stability["high_cv"])
    print(f"  Delta (low - high CV): {delta_cv:+.1f}pp")

    for group, vals in holdout_by_app.items():
        print(f"  {group:10s}: {np.mean(vals):.1f}% ± {np.std(vals):.1f}%")
    delta_app = np.mean(holdout_by_app["common"]) - np.mean(holdout_by_app["rare"])
    print(f"  Delta (common - rare): {delta_app:+.1f}pp")

    # Save results
    output = {
        "hypothesis": "h441",
        "full_data": {
            "cv_median": round(float(cv_median), 3),
            "spread_median": round(float(spread_median), 1),
        },
        "holdout": {
            "low_cv_mean": round(float(np.mean(holdout_by_stability["low_cv"])), 1),
            "high_cv_mean": round(float(np.mean(holdout_by_stability["high_cv"])), 1),
            "delta_cv_pp": round(float(delta_cv), 1),
            "common_mean": round(float(np.mean(holdout_by_app["common"])), 1),
            "rare_mean": round(float(np.mean(holdout_by_app["rare"])), 1),
            "delta_app_pp": round(float(delta_app), 1),
        },
    }
    output_path = Path(__file__).parent.parent / "data" / "analysis" / "h441_drug_stability.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Conclusion
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")
    if abs(delta_cv) > 5:
        print(f"SIGNIFICANT: CV stability splits precision by {delta_cv:+.1f}pp on holdout")
        if delta_cv > 0:
            print("Stable drugs (low CV) have HIGHER precision → can use as ranking boost")
        else:
            print("Stable drugs (low CV) have LOWER precision → high-CV drugs are better (surprising)")
    else:
        print(f"NOT SIGNIFICANT: CV stability gap is only {delta_cv:+.1f}pp on holdout")


if __name__ == "__main__":
    main()
