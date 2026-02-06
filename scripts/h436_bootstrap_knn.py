#!/usr/bin/env python3
"""
h436: kNN Bootstrap Ensemble for Rank Stability

h434/h437 showed kNN neighborhood instability (Jaccard 0.664, mean 4.1 drugs cross
rank-20 boundary) is the root cause of holdout degradation. The rank>20 filter
compensates for this instability.

Hypothesis: Bootstrap aggregation (bagging) of kNN neighborhoods will stabilize
drug rankings, reducing the number of drugs crossing the rank-20 boundary, and
potentially enabling rank>20 rescue.

Method:
1. For each disease, run kNN on B=10 bootstrap samples (80% of train diseases each)
2. Average drug scores across bootstraps
3. Measure rank stability vs standard kNN
4. If stable enough, test rank>20 rescue on holdout

Success criteria:
- Boundary crossings reduced by >50% (from 4.1 to <2.0 per disease)
- R@30 maintained or improved
- If rank stability achieved, holdout with rank 21-30 rescue doesn't degrade
"""

import json
import sys
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from production_predictor import (
    ConfidenceTier,
    DrugRepurposingPredictor,
    SELECTIVE_BOOST_CATEGORIES,
    SELECTIVE_BOOST_ALPHA,
    DISEASE_HIERARCHY_GROUPS,
    extract_cancer_types,
)


def bootstrap_knn_scores(
    predictor: DrugRepurposingPredictor,
    disease_id: str,
    k: int = 20,
    n_bootstraps: int = 10,
    subsample_ratio: float = 0.8,
    seed: int = 42,
) -> Dict[str, float]:
    """Run kNN on multiple bootstrap samples and average drug scores.

    Returns dict of drug_id -> averaged score across bootstraps.
    """
    rng = np.random.RandomState(seed)

    test_emb = predictor.embeddings[disease_id].reshape(1, -1)
    category = predictor.categorize_disease(
        predictor.disease_names.get(disease_id, disease_id)
    )
    use_boost = category in SELECTIVE_BOOST_CATEGORIES

    n_train = len(predictor.train_diseases)
    n_sample = int(n_train * subsample_ratio)

    all_drug_scores: Dict[str, List[float]] = defaultdict(list)

    for b in range(n_bootstraps):
        # Sample without replacement (subsampling, not full bootstrap)
        indices = rng.choice(n_train, size=n_sample, replace=False)
        sample_diseases = [predictor.train_diseases[i] for i in indices]
        sample_embeddings = predictor.train_embeddings[indices]

        # Compute similarities to this subsample
        sims = cosine_similarity(test_emb, sample_embeddings)[0]

        # Apply selective boost if needed
        if use_boost:
            boosted_sims = sims.copy()
            for i, train_d in enumerate(sample_diseases):
                if predictor.train_disease_categories.get(train_d) == category:
                    boosted_sims[i] *= (1 + SELECTIVE_BOOST_ALPHA)
            working_sims = boosted_sims
        else:
            working_sims = sims

        # Get top-k neighbors from this subsample
        top_k_idx = np.argsort(working_sims)[-k:]

        # Aggregate drug scores
        drug_scores: Dict[str, float] = defaultdict(float)
        for idx in top_k_idx:
            neighbor_disease = sample_diseases[idx]
            neighbor_sim = working_sims[idx]
            if neighbor_disease in predictor.ground_truth:
                for drug_id in predictor.ground_truth[neighbor_disease]:
                    if drug_id in predictor.embeddings:
                        drug_scores[drug_id] += neighbor_sim

        # Record this bootstrap's scores
        for drug_id, score in drug_scores.items():
            all_drug_scores[drug_id].append(score)

    # Average across bootstraps (drugs not present in a bootstrap get 0)
    averaged: Dict[str, float] = {}
    for drug_id, scores in all_drug_scores.items():
        # Pad with zeros for bootstraps where drug didn't appear
        while len(scores) < n_bootstraps:
            scores.append(0.0)
        averaged[drug_id] = np.mean(scores)

    return averaged


def standard_knn_scores(
    predictor: DrugRepurposingPredictor,
    disease_id: str,
    k: int = 20,
) -> Dict[str, float]:
    """Standard kNN scoring (current method)."""
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


def get_top_n_drugs(scores: Dict[str, float], n: int) -> List[str]:
    """Return top-n drug IDs sorted by score descending."""
    sorted_drugs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [drug_id for drug_id, _ in sorted_drugs[:n]]


def compute_boundary_crossings(
    standard_top30: List[str],
    bootstrap_top30: List[str],
    standard_scores: Dict[str, float],
    bootstrap_scores: Dict[str, float],
) -> Dict:
    """Compare rank-20 boundary crossings between two rankings."""
    std_top20 = set(standard_top30[:20])
    std_rank21_30 = set(standard_top30[20:30])

    boot_top20 = set(bootstrap_top30[:20])
    boot_rank21_30 = set(bootstrap_top30[20:30])

    # Drugs that move from 21-30 to 1-20 or vice versa
    promoted = std_rank21_30 & boot_top20  # Were 21-30, now top-20
    demoted = std_top20 & boot_rank21_30   # Were top-20, now 21-30

    return {
        "promoted": len(promoted),
        "demoted": len(demoted),
        "top20_overlap": len(std_top20 & boot_top20),
        "top30_overlap": len(set(standard_top30) & set(bootstrap_top30)),
    }


def main():
    print("=" * 70)
    print("h436: kNN Bootstrap Ensemble for Rank Stability")
    print("=" * 70)

    # Load predictor
    print("\nLoading predictor...")
    t0 = time.time()
    predictor = DrugRepurposingPredictor()
    print(f"Loaded in {time.time() - t0:.1f}s")

    # Get all diseases with GT and embeddings
    diseases_with_gt = [
        d for d in predictor.ground_truth
        if d in predictor.embeddings
    ]
    print(f"\nDiseases with GT and embeddings: {len(diseases_with_gt)}")

    # ===== PART 1: Compare standard vs bootstrap rank stability =====
    print("\n" + "=" * 70)
    print("PART 1: Standard vs Bootstrap Rank Stability (full data)")
    print("=" * 70)

    k = 20
    top_n = 30
    n_bootstraps = 10

    all_crossings = []
    all_gt_results = {"standard": {"hits_top20": 0, "hits_top30": 0, "total": 0},
                      "bootstrap": {"hits_top20": 0, "hits_top30": 0, "total": 0}}

    per_disease_results = []

    print(f"\nRunning on {len(diseases_with_gt)} diseases...")
    t0 = time.time()

    for i, disease_id in enumerate(diseases_with_gt):
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(diseases_with_gt) - i - 1) / rate
            print(f"  {i+1}/{len(diseases_with_gt)} ({elapsed:.1f}s elapsed, ~{remaining:.0f}s remaining)")

        gt_drugs = predictor.ground_truth.get(disease_id, set())
        if not gt_drugs:
            continue

        # Standard kNN
        std_scores = standard_knn_scores(predictor, disease_id, k=k)
        std_top30 = get_top_n_drugs(std_scores, top_n)

        # Bootstrap kNN
        boot_scores = bootstrap_knn_scores(
            predictor, disease_id, k=k,
            n_bootstraps=n_bootstraps, subsample_ratio=0.8, seed=42
        )
        boot_top30 = get_top_n_drugs(boot_scores, top_n)

        # Boundary crossings between standard and bootstrap
        crossings = compute_boundary_crossings(std_top30, boot_top30, std_scores, boot_scores)
        all_crossings.append(crossings)

        # GT hits
        std_hits_20 = len(set(std_top30[:20]) & gt_drugs)
        std_hits_30 = len(set(std_top30) & gt_drugs)
        boot_hits_20 = len(set(boot_top30[:20]) & gt_drugs)
        boot_hits_30 = len(set(boot_top30) & gt_drugs)

        all_gt_results["standard"]["hits_top20"] += std_hits_20
        all_gt_results["standard"]["hits_top30"] += std_hits_30
        all_gt_results["standard"]["total"] += len(gt_drugs)
        all_gt_results["bootstrap"]["hits_top20"] += boot_hits_20
        all_gt_results["bootstrap"]["hits_top30"] += boot_hits_30
        all_gt_results["bootstrap"]["total"] += len(gt_drugs)

        per_disease_results.append({
            "disease_id": disease_id,
            "disease_name": predictor.disease_names.get(disease_id, disease_id),
            "std_hits_30": std_hits_30,
            "boot_hits_30": boot_hits_30,
            "gt_size": len(gt_drugs),
            "top30_overlap": crossings["top30_overlap"],
            "top20_overlap": crossings["top20_overlap"],
            "promoted": crossings["promoted"],
            "demoted": crossings["demoted"],
        })

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s")

    # Aggregate results
    n_diseases = len(all_crossings)
    avg_promoted = np.mean([c["promoted"] for c in all_crossings])
    avg_demoted = np.mean([c["demoted"] for c in all_crossings])
    avg_top20_overlap = np.mean([c["top20_overlap"] for c in all_crossings])
    avg_top30_overlap = np.mean([c["top30_overlap"] for c in all_crossings])

    print(f"\n--- Rank Changes (Standard → Bootstrap) ---")
    print(f"Diseases analyzed: {n_diseases}")
    print(f"Mean top-20 overlap: {avg_top20_overlap:.1f}/20 ({avg_top20_overlap/20*100:.1f}%)")
    print(f"Mean top-30 overlap: {avg_top30_overlap:.1f}/30 ({avg_top30_overlap/30*100:.1f}%)")
    print(f"Mean drugs promoted (21-30 → top-20): {avg_promoted:.2f}")
    print(f"Mean drugs demoted (top-20 → 21-30): {avg_demoted:.2f}")
    print(f"Mean boundary crossings (promoted+demoted): {avg_promoted + avg_demoted:.2f}")

    std_r20 = all_gt_results["standard"]["hits_top20"] / all_gt_results["standard"]["total"]
    std_r30 = all_gt_results["standard"]["hits_top30"] / all_gt_results["standard"]["total"]
    boot_r20 = all_gt_results["bootstrap"]["hits_top20"] / all_gt_results["bootstrap"]["total"]
    boot_r30 = all_gt_results["bootstrap"]["hits_top30"] / all_gt_results["bootstrap"]["total"]

    print(f"\n--- Recall ---")
    print(f"Standard kNN:  R@20 = {std_r20*100:.2f}%, R@30 = {std_r30*100:.2f}%")
    print(f"Bootstrap kNN: R@20 = {boot_r20*100:.2f}%, R@30 = {boot_r30*100:.2f}%")
    print(f"Delta R@20: {(boot_r20 - std_r20)*100:+.2f}pp")
    print(f"Delta R@30: {(boot_r30 - std_r30)*100:+.2f}pp")

    # Per-disease analysis: where does bootstrap help/hurt?
    improved = sum(1 for r in per_disease_results if r["boot_hits_30"] > r["std_hits_30"])
    worse = sum(1 for r in per_disease_results if r["boot_hits_30"] < r["std_hits_30"])
    same = sum(1 for r in per_disease_results if r["boot_hits_30"] == r["std_hits_30"])

    print(f"\n--- Per-Disease Impact ---")
    print(f"Bootstrap improves: {improved} diseases")
    print(f"Bootstrap hurts:    {worse} diseases")
    print(f"No change:          {same} diseases")

    # ===== PART 2: Bootstrap stability under holdout =====
    print("\n" + "=" * 70)
    print("PART 2: Bootstrap Stability Under Holdout")
    print("=" * 70)
    print("(Testing if bootstrap reduces full→holdout rank changes)")

    seeds = [42, 123, 456, 789, 2024]
    holdout_results = {"standard": [], "bootstrap": []}
    boundary_crossing_comparison = {"standard": [], "bootstrap": []}

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")

        # Split diseases
        rng = np.random.RandomState(seed)
        shuffled = list(diseases_with_gt)
        rng.shuffle(shuffled)
        split_idx = int(len(shuffled) * 0.8)
        train_diseases = shuffled[:split_idx]
        holdout_diseases = shuffled[split_idx:]

        train_set = set(train_diseases)

        # Save original state
        orig_train_diseases = list(predictor.train_diseases)
        orig_train_embeddings = predictor.train_embeddings.copy()
        orig_train_categories = dict(predictor.train_disease_categories)
        orig_drug_freq = dict(predictor.drug_train_freq)
        orig_gt = dict(predictor.ground_truth)

        # Rebuild kNN index from training diseases only
        predictor.train_diseases = [d for d in train_diseases if d in predictor.embeddings]
        predictor.train_embeddings = np.array(
            [predictor.embeddings[d] for d in predictor.train_diseases], dtype=np.float32
        )
        predictor.train_disease_categories = {}
        for d in predictor.train_diseases:
            name = predictor.disease_names.get(d, d)
            predictor.train_disease_categories[d] = predictor.categorize_disease(name)

        # Recompute drug freq from training only
        new_freq: Dict[str, int] = defaultdict(int)
        for disease_id in train_set:
            if disease_id in predictor.ground_truth:
                for drug_id in predictor.ground_truth[disease_id]:
                    new_freq[drug_id] += 1
        predictor.drug_train_freq = dict(new_freq)

        # Evaluate on holdout
        seed_std_hits = 0
        seed_boot_hits = 0
        seed_total_gt = 0
        seed_std_boundary = []
        seed_boot_boundary = []

        for disease_id in holdout_diseases:
            gt_drugs = orig_gt.get(disease_id, set())
            if not gt_drugs or disease_id not in predictor.embeddings:
                continue

            # Standard kNN on holdout kNN index
            std_scores = standard_knn_scores(predictor, disease_id, k=k)
            std_top30 = get_top_n_drugs(std_scores, top_n)

            # Bootstrap kNN on holdout kNN index
            boot_scores = bootstrap_knn_scores(
                predictor, disease_id, k=k,
                n_bootstraps=n_bootstraps, subsample_ratio=0.8, seed=seed
            )
            boot_top30 = get_top_n_drugs(boot_scores, top_n)

            seed_std_hits += len(set(std_top30) & gt_drugs)
            seed_boot_hits += len(set(boot_top30) & gt_drugs)
            seed_total_gt += len(gt_drugs)

            # Also measure rank stability: how much do std vs boot differ?
            crossings = compute_boundary_crossings(std_top30, boot_top30, std_scores, boot_scores)
            seed_std_boundary.append(crossings["promoted"] + crossings["demoted"])

        std_r30_seed = seed_std_hits / seed_total_gt if seed_total_gt > 0 else 0
        boot_r30_seed = seed_boot_hits / seed_total_gt if seed_total_gt > 0 else 0

        holdout_results["standard"].append(std_r30_seed)
        holdout_results["bootstrap"].append(boot_r30_seed)

        print(f"  Holdout diseases: {len(holdout_diseases)}")
        print(f"  Standard R@30: {std_r30_seed*100:.2f}%")
        print(f"  Bootstrap R@30: {boot_r30_seed*100:.2f}%")
        print(f"  Delta: {(boot_r30_seed - std_r30_seed)*100:+.2f}pp")
        print(f"  Mean boundary crossings (std vs boot): {np.mean(seed_std_boundary):.2f}")

        # Restore original state
        predictor.train_diseases = orig_train_diseases
        predictor.train_embeddings = orig_train_embeddings
        predictor.train_disease_categories = orig_train_categories
        predictor.drug_train_freq = orig_drug_freq
        predictor.ground_truth = orig_gt

    print(f"\n--- Holdout Summary (5-seed) ---")
    std_mean = np.mean(holdout_results["standard"])
    std_std = np.std(holdout_results["standard"])
    boot_mean = np.mean(holdout_results["bootstrap"])
    boot_std = np.std(holdout_results["bootstrap"])

    print(f"Standard R@30: {std_mean*100:.2f}% +/- {std_std*100:.2f}%")
    print(f"Bootstrap R@30: {boot_mean*100:.2f}% +/- {boot_std*100:.2f}%")
    print(f"Delta: {(boot_mean - std_mean)*100:+.2f}pp")

    # ===== PART 3: Full→Holdout rank drift comparison =====
    print("\n" + "=" * 70)
    print("PART 3: Full→Holdout Rank Drift (Bootstrap vs Standard)")
    print("=" * 70)
    print("(Does bootstrap reduce the number of drugs that cross rank-20 boundary")
    print(" when going from full data to holdout?)")

    # Use seed=42 holdout split
    seed = 42
    rng = np.random.RandomState(seed)
    shuffled = list(diseases_with_gt)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * 0.8)
    train_diseases = shuffled[:split_idx]
    holdout_diseases = shuffled[split_idx:]
    train_set = set(train_diseases)

    # Step 1: Get FULL data rankings
    full_std_rankings = {}
    full_boot_rankings = {}

    print(f"\nComputing full-data rankings for holdout diseases...")
    for disease_id in holdout_diseases:
        if disease_id not in predictor.embeddings:
            continue
        std_scores = standard_knn_scores(predictor, disease_id, k=k)
        boot_scores = bootstrap_knn_scores(predictor, disease_id, k=k, n_bootstraps=n_bootstraps, seed=42)
        full_std_rankings[disease_id] = get_top_n_drugs(std_scores, top_n)
        full_boot_rankings[disease_id] = get_top_n_drugs(boot_scores, top_n)

    # Step 2: Rebuild kNN index from training only
    orig_train_diseases = list(predictor.train_diseases)
    orig_train_embeddings = predictor.train_embeddings.copy()
    orig_train_categories = dict(predictor.train_disease_categories)
    orig_drug_freq = dict(predictor.drug_train_freq)

    predictor.train_diseases = [d for d in train_diseases if d in predictor.embeddings]
    predictor.train_embeddings = np.array(
        [predictor.embeddings[d] for d in predictor.train_diseases], dtype=np.float32
    )
    predictor.train_disease_categories = {}
    for d in predictor.train_diseases:
        name = predictor.disease_names.get(d, d)
        predictor.train_disease_categories[d] = predictor.categorize_disease(name)

    # Step 3: Get HOLDOUT rankings
    holdout_std_rankings = {}
    holdout_boot_rankings = {}

    print(f"Computing holdout rankings...")
    for disease_id in holdout_diseases:
        if disease_id not in predictor.embeddings:
            continue
        std_scores = standard_knn_scores(predictor, disease_id, k=k)
        boot_scores = bootstrap_knn_scores(predictor, disease_id, k=k, n_bootstraps=n_bootstraps, seed=42)
        holdout_std_rankings[disease_id] = get_top_n_drugs(std_scores, top_n)
        holdout_boot_rankings[disease_id] = get_top_n_drugs(boot_scores, top_n)

    # Restore
    predictor.train_diseases = orig_train_diseases
    predictor.train_embeddings = orig_train_embeddings
    predictor.train_disease_categories = orig_train_categories
    predictor.drug_train_freq = orig_drug_freq

    # Step 4: Compare full→holdout drift for both methods
    std_crossings_full_to_holdout = []
    boot_crossings_full_to_holdout = []

    for disease_id in full_std_rankings:
        if disease_id not in holdout_std_rankings:
            continue

        # Standard: full vs holdout
        full_std_top20 = set(full_std_rankings[disease_id][:20])
        hold_std_top20 = set(holdout_std_rankings[disease_id][:20])
        std_cross = len(full_std_top20 - hold_std_top20)  # drugs that left top-20
        std_crossings_full_to_holdout.append(std_cross)

        # Bootstrap: full vs holdout
        full_boot_top20 = set(full_boot_rankings[disease_id][:20])
        hold_boot_top20 = set(holdout_boot_rankings[disease_id][:20])
        boot_cross = len(full_boot_top20 - hold_boot_top20)
        boot_crossings_full_to_holdout.append(boot_cross)

    print(f"\n--- Full→Holdout Rank Drift (seed=42) ---")
    print(f"Diseases compared: {len(std_crossings_full_to_holdout)}")
    print(f"Standard: Mean {np.mean(std_crossings_full_to_holdout):.2f} drugs leave top-20 (std: {np.std(std_crossings_full_to_holdout):.2f})")
    print(f"Bootstrap: Mean {np.mean(boot_crossings_full_to_holdout):.2f} drugs leave top-20 (std: {np.std(boot_crossings_full_to_holdout):.2f})")
    reduction = (1 - np.mean(boot_crossings_full_to_holdout) / np.mean(std_crossings_full_to_holdout)) * 100 if np.mean(std_crossings_full_to_holdout) > 0 else 0
    print(f"Reduction: {reduction:+.1f}%")

    # Jaccard overlap comparison
    std_jaccards = []
    boot_jaccards = []
    for disease_id in full_std_rankings:
        if disease_id not in holdout_std_rankings:
            continue

        full_std_set = set(full_std_rankings[disease_id][:20])
        hold_std_set = set(holdout_std_rankings[disease_id][:20])
        union = full_std_set | hold_std_set
        if union:
            std_jaccards.append(len(full_std_set & hold_std_set) / len(union))

        full_boot_set = set(full_boot_rankings[disease_id][:20])
        hold_boot_set = set(holdout_boot_rankings[disease_id][:20])
        union = full_boot_set | hold_boot_set
        if union:
            boot_jaccards.append(len(full_boot_set & hold_boot_set) / len(union))

    print(f"\nJaccard overlap (full→holdout top-20):")
    print(f"Standard: {np.mean(std_jaccards):.3f} ± {np.std(std_jaccards):.3f}")
    print(f"Bootstrap: {np.mean(boot_jaccards):.3f} ± {np.std(boot_jaccards):.3f}")

    # ===== Save results =====
    results = {
        "hypothesis": "h436",
        "title": "kNN Bootstrap Ensemble for Rank Stability",
        "params": {"k": k, "n_bootstraps": n_bootstraps, "subsample_ratio": 0.8},
        "part1_full_data": {
            "n_diseases": n_diseases,
            "mean_top20_overlap": round(float(avg_top20_overlap), 2),
            "mean_top30_overlap": round(float(avg_top30_overlap), 2),
            "mean_boundary_crossings": round(float(avg_promoted + avg_demoted), 2),
            "standard_R30": round(float(std_r30 * 100), 2),
            "bootstrap_R30": round(float(boot_r30 * 100), 2),
            "diseases_improved": improved,
            "diseases_worse": worse,
            "diseases_same": same,
        },
        "part2_holdout": {
            "standard_mean": round(float(std_mean * 100), 2),
            "standard_std": round(float(std_std * 100), 2),
            "bootstrap_mean": round(float(boot_mean * 100), 2),
            "bootstrap_std": round(float(boot_std * 100), 2),
            "delta_pp": round(float((boot_mean - std_mean) * 100), 2),
        },
        "part3_rank_drift": {
            "std_mean_crossings": round(float(np.mean(std_crossings_full_to_holdout)), 2),
            "boot_mean_crossings": round(float(np.mean(boot_crossings_full_to_holdout)), 2),
            "reduction_pct": round(float(reduction), 1),
            "std_jaccard": round(float(np.mean(std_jaccards)), 3),
            "boot_jaccard": round(float(np.mean(boot_jaccards)), 3),
        },
    }

    output_path = Path(__file__).parent.parent / "data" / "analysis" / "h436_bootstrap_knn.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # ===== Summary =====
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Bootstrap stabilizes rankings? Top-20 overlap: {avg_top20_overlap:.1f}/20")
    print(f"Full-data R@30: Standard {std_r30*100:.2f}% vs Bootstrap {boot_r30*100:.2f}%")
    print(f"Holdout R@30: Standard {std_mean*100:.2f}% ± {std_std*100:.2f}% vs Bootstrap {boot_mean*100:.2f}% ± {boot_std*100:.2f}%")
    print(f"Full→Holdout drift reduction: {reduction:+.1f}%")
    print(f"Jaccard improvement: {np.mean(std_jaccards):.3f} → {np.mean(boot_jaccards):.3f}")

    if reduction > 30:
        print("\n** Bootstrap SIGNIFICANTLY reduces rank drift. Proceed to rank>20 rescue test. **")
    elif reduction > 10:
        print("\n** Bootstrap MODERATELY reduces rank drift. Worth further investigation. **")
    else:
        print("\n** Bootstrap does NOT meaningfully reduce rank drift. Hypothesis invalidated. **")


if __name__ == "__main__":
    main()
