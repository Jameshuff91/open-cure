#!/usr/bin/env python3
"""
h442: kNN Score Margin as Confidence Signal

h436/h437 showed kNN neighborhoods are structurally unstable (Jaccard 0.664,
mean 4.1 drugs cross rank-20 boundary). Instead of trying to stabilize rankings,
use the MARGIN between rank-20 and rank-21 drug scores as a confidence signal.

A large margin means the ranking is robust even if neighborhoods change slightly.
A small margin means the drug at rank 20 vs 21 is essentially a coin flip.

Method:
1. For each disease, compute kNN scores and measure rank-20/21 margin
2. Split diseases into quartiles by margin
3. Compare tier precision for high-margin vs low-margin diseases
4. Validate on holdout

Success criteria:
- High-margin diseases have >10pp better precision than low-margin diseases
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
    DISEASE_HIERARCHY_GROUPS,
    extract_cancer_types,
)


def compute_margin(predictor: DrugRepurposingPredictor, disease_id: str, k: int = 20) -> Dict:
    """Compute kNN score margin and related metrics for a disease."""
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

    # Aggregate drug scores
    drug_scores: Dict[str, float] = defaultdict(float)
    for idx in top_k_idx:
        neighbor_disease = predictor.train_diseases[idx]
        neighbor_sim = working_sims[idx]
        if neighbor_disease in predictor.ground_truth:
            for drug_id in predictor.ground_truth[neighbor_disease]:
                if drug_id in predictor.embeddings:
                    drug_scores[drug_id] += neighbor_sim

    if len(drug_scores) < 25:
        return {"valid": False}

    sorted_drugs = sorted(drug_scores.items(), key=lambda x: x[1], reverse=True)
    scores = [s for _, s in sorted_drugs]

    # Score at rank 1, 20, 21, 30
    score_1 = scores[0]
    score_20 = scores[19] if len(scores) > 19 else 0
    score_21 = scores[20] if len(scores) > 20 else 0
    score_30 = scores[29] if len(scores) > 29 else 0

    # Normalized margin: (score_20 - score_21) / score_1
    norm_margin = (score_20 - score_21) / score_1 if score_1 > 0 else 0

    # Also compute broader margin metrics
    # Score dropoff ratio (how quickly scores decay)
    score_ratio_20_1 = score_20 / score_1 if score_1 > 0 else 0
    score_ratio_30_1 = score_30 / score_1 if score_1 > 0 else 0

    # Score entropy (how concentrated scores are)
    if score_1 > 0:
        top30_scores = np.array(scores[:30])
        top30_norm = top30_scores / top30_scores.sum()
        entropy = -np.sum(top30_norm * np.log(top30_norm + 1e-10))
    else:
        entropy = 0

    # Mean neighbor similarity (how similar the disease is to its neighbors)
    neighbor_sims = [working_sims[idx] for idx in top_k_idx]
    mean_neighbor_sim = np.mean(neighbor_sims)

    return {
        "valid": True,
        "norm_margin": float(norm_margin),
        "score_1": float(score_1),
        "score_20": float(score_20),
        "score_21": float(score_21),
        "score_ratio_20_1": float(score_ratio_20_1),
        "entropy": float(entropy),
        "mean_neighbor_sim": float(mean_neighbor_sim),
        "n_drugs": len(drug_scores),
    }


def split_diseases(all_diseases: List[str], seed: int, train_ratio: float = 0.8) -> Tuple[List[str], List[str]]:
    rng = np.random.RandomState(seed)
    shuffled = list(all_diseases)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def recompute_gt_structures(predictor: DrugRepurposingPredictor, train_set: Set[str]) -> Dict:
    """Recompute GT-derived structures from training diseases only."""
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
    for d in train_set:
        if d in predictor.ground_truth:
            for drug in predictor.ground_truth[d]:
                new_freq[drug] += 1
    predictor.drug_train_freq = dict(new_freq)

    new_d2d: Dict[str, Set[str]] = defaultdict(set)
    for d in train_set:
        if d in predictor.ground_truth:
            dn = predictor.disease_names.get(d, d)
            for drug in predictor.ground_truth[d]:
                new_d2d[drug].add(dn.lower())
    predictor.drug_to_diseases = dict(new_d2d)

    new_cancer: Dict[str, Set[str]] = defaultdict(set)
    for d in train_set:
        if d in predictor.ground_truth:
            dn = predictor.disease_names.get(d, d)
            ct = extract_cancer_types(dn)
            if ct:
                for drug in predictor.ground_truth[d]:
                    new_cancer[drug].update(ct)
    predictor.drug_cancer_types = dict(new_cancer)

    new_groups: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    for d in train_set:
        if d in predictor.ground_truth:
            dn = predictor.disease_names.get(d, d)
            cat = predictor.categorize_disease(dn)
            if cat in DISEASE_HIERARCHY_GROUPS:
                for gn, kws in DISEASE_HIERARCHY_GROUPS[cat].items():
                    if any(kw in dn.lower() for kw in kws):
                        for drug in predictor.ground_truth[d]:
                            new_groups[drug].add((cat, gn))
    predictor.drug_disease_groups = dict(new_groups)

    predictor.train_diseases = [d for d in train_set if d in predictor.embeddings]
    predictor.train_embeddings = np.array(
        [predictor.embeddings[d] for d in predictor.train_diseases], dtype=np.float32
    )
    predictor.train_disease_categories = {}
    for d in predictor.train_diseases:
        name = predictor.disease_names.get(d, d)
        predictor.train_disease_categories[d] = predictor.categorize_disease(name)

    return originals


def restore_originals(predictor: DrugRepurposingPredictor, originals: Dict) -> None:
    for key, val in originals.items():
        setattr(predictor, key, val)


def evaluate_margin_disease_set(
    predictor: DrugRepurposingPredictor,
    diseases: List[str],
    margin_quartiles: Dict[str, Tuple[float, float]],
) -> Dict:
    """Evaluate tier precision for diseases grouped by margin quartile."""
    # Compute margins
    disease_margins = {}
    for disease_id in diseases:
        if disease_id not in predictor.embeddings:
            continue
        m = compute_margin(predictor, disease_id)
        if m["valid"]:
            disease_margins[disease_id] = m["norm_margin"]

    # Assign to quartiles
    disease_quartiles: Dict[str, str] = {}
    for disease_id, margin in disease_margins.items():
        for q_name, (lo, hi) in margin_quartiles.items():
            if lo <= margin < hi:
                disease_quartiles[disease_id] = q_name
                break

    # Evaluate per quartile x tier
    results: Dict[str, Dict[str, Dict]] = defaultdict(lambda: defaultdict(lambda: {"gt": 0, "n": 0}))

    for disease_id, q_name in disease_quartiles.items():
        gt_drugs = predictor.ground_truth.get(disease_id, set())
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        result = predictor.predict(disease_name, k=20, top_n=30, include_filtered=True)

        for pred in result.predictions:
            tier = pred.confidence_tier.value
            results[q_name][tier]["n"] += 1
            if pred.drug_id in gt_drugs:
                results[q_name][tier]["gt"] += 1

    return dict(results)


def main():
    print("=" * 70)
    print("h442: kNN Score Margin as Confidence Signal")
    print("=" * 70)

    predictor = DrugRepurposingPredictor()

    diseases_with_gt = [
        d for d in predictor.ground_truth
        if d in predictor.embeddings
    ]
    print(f"Diseases with GT and embeddings: {len(diseases_with_gt)}")

    # ===== PART 1: Compute margins for all diseases =====
    print("\n" + "=" * 70)
    print("PART 1: Score Margin Distribution")
    print("=" * 70)

    margins = {}
    all_details = {}
    for disease_id in diseases_with_gt:
        m = compute_margin(predictor, disease_id)
        if m["valid"]:
            margins[disease_id] = m["norm_margin"]
            all_details[disease_id] = m

    margin_values = list(margins.values())
    print(f"\nDiseases with valid margins: {len(margin_values)}")
    print(f"Margin distribution:")
    print(f"  Min:    {np.min(margin_values):.6f}")
    print(f"  Q1:     {np.percentile(margin_values, 25):.6f}")
    print(f"  Median: {np.median(margin_values):.6f}")
    print(f"  Q3:     {np.percentile(margin_values, 75):.6f}")
    print(f"  Max:    {np.max(margin_values):.6f}")
    print(f"  Mean:   {np.mean(margin_values):.6f}")
    print(f"  Std:    {np.std(margin_values):.6f}")

    # Define quartile boundaries
    q1 = float(np.percentile(margin_values, 25))
    q2 = float(np.percentile(margin_values, 50))
    q3 = float(np.percentile(margin_values, 75))
    quartiles = {
        "Q1_low": (-1.0, q1),
        "Q2": (q1, q2),
        "Q3": (q2, q3),
        "Q4_high": (q3, 100.0),
    }
    print(f"\nQuartile boundaries: Q1<{q1:.6f}, Q2<{q2:.6f}, Q3<{q3:.6f}, Q4>={q3:.6f}")

    # ===== PART 2: Precision by margin quartile (full data) =====
    print("\n" + "=" * 70)
    print("PART 2: Tier Precision by Margin Quartile (Full Data)")
    print("=" * 70)

    full_results = evaluate_margin_disease_set(predictor, diseases_with_gt, quartiles)

    tier_order = ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]
    q_order = ["Q1_low", "Q2", "Q3", "Q4_high"]

    # Print table
    print(f"\n{'Tier':<10}", end="")
    for q in q_order:
        print(f"  {q:>12}", end="")
    print(f"  {'Q4-Q1 gap':>10}")

    for tier in tier_order:
        print(f"{tier:<10}", end="")
        precs = []
        for q in q_order:
            n = full_results.get(q, {}).get(tier, {}).get("n", 0)
            gt = full_results.get(q, {}).get(tier, {}).get("gt", 0)
            prec = gt / n * 100 if n > 0 else 0
            precs.append(prec)
            print(f"  {prec:>7.1f}%({n:>3})", end="")
        gap = precs[3] - precs[0] if len(precs) == 4 else 0
        print(f"  {gap:>+8.1f}pp")

    # ===== PART 3: Correlation analysis =====
    print("\n" + "=" * 70)
    print("PART 3: Margin Correlations")
    print("=" * 70)

    # For each disease, compute R@30 and compare with margin
    disease_r30 = {}
    for disease_id in margins:
        gt_drugs = predictor.ground_truth.get(disease_id, set())
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        result = predictor.predict(disease_name, k=20, top_n=30, include_filtered=True)
        hits = sum(1 for p in result.predictions if p.drug_id in gt_drugs)
        r30 = hits / len(gt_drugs) if gt_drugs else 0
        disease_r30[disease_id] = r30

    # Correlation
    common_ids = sorted(set(margins.keys()) & set(disease_r30.keys()))
    m_arr = np.array([margins[d] for d in common_ids])
    r_arr = np.array([disease_r30[d] for d in common_ids])
    correlation = np.corrcoef(m_arr, r_arr)[0, 1]
    print(f"\nMargin vs R@30 correlation: {correlation:.3f}")

    # Also check other metrics
    entropy_arr = np.array([all_details[d]["entropy"] for d in common_ids])
    neighbor_sim_arr = np.array([all_details[d]["mean_neighbor_sim"] for d in common_ids])
    n_drugs_arr = np.array([all_details[d]["n_drugs"] for d in common_ids])
    ratio_arr = np.array([all_details[d]["score_ratio_20_1"] for d in common_ids])

    print(f"Entropy vs R@30 correlation: {np.corrcoef(entropy_arr, r_arr)[0, 1]:.3f}")
    print(f"Mean neighbor sim vs R@30 correlation: {np.corrcoef(neighbor_sim_arr, r_arr)[0, 1]:.3f}")
    print(f"N_drugs vs R@30 correlation: {np.corrcoef(n_drugs_arr, r_arr)[0, 1]:.3f}")
    print(f"Score ratio (20/1) vs R@30 correlation: {np.corrcoef(ratio_arr, r_arr)[0, 1]:.3f}")

    # ===== PART 4: Holdout validation =====
    print("\n" + "=" * 70)
    print("PART 4: Holdout Validation (5-seed)")
    print("=" * 70)

    seeds = [42, 123, 456, 789, 2024]
    holdout_gaps_by_tier: Dict[str, List[float]] = defaultdict(list)

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        train_diseases, holdout_diseases = split_diseases(diseases_with_gt, seed)
        train_set = set(train_diseases)

        originals = recompute_gt_structures(predictor, train_set)

        # Compute margins for holdout diseases using holdout kNN index
        holdout_margins = {}
        for disease_id in holdout_diseases:
            m = compute_margin(predictor, disease_id)
            if m["valid"]:
                holdout_margins[disease_id] = m["norm_margin"]

        if len(holdout_margins) < 20:
            restore_originals(predictor, originals)
            continue

        # Use median split (more robust with smaller n)
        median_margin = float(np.median(list(holdout_margins.values())))
        high_margin_diseases = [d for d, m in holdout_margins.items() if m >= median_margin]
        low_margin_diseases = [d for d, m in holdout_margins.items() if m < median_margin]

        # Evaluate each group (single pass per disease)
        group_tiers: Dict[str, Dict[str, Dict]] = {
            "high": defaultdict(lambda: {"gt": 0, "n": 0}),
            "low": defaultdict(lambda: {"gt": 0, "n": 0}),
        }

        for label, disease_list in [("high", high_margin_diseases), ("low", low_margin_diseases)]:
            for disease_id in disease_list:
                gt_drugs = predictor.ground_truth.get(disease_id, set())
                disease_name = predictor.disease_names.get(disease_id, disease_id)
                result = predictor.predict(disease_name, k=20, top_n=30, include_filtered=True)
                for pred in result.predictions:
                    tier = pred.confidence_tier.value
                    group_tiers[label][tier]["n"] += 1
                    if pred.drug_id in gt_drugs:
                        group_tiers[label][tier]["gt"] += 1

        for tier in tier_order:
            h_n = group_tiers["high"][tier]["n"]
            h_gt = group_tiers["high"][tier]["gt"]
            l_n = group_tiers["low"][tier]["n"]
            l_gt = group_tiers["low"][tier]["gt"]
            h_prec = h_gt / h_n * 100 if h_n > 0 else 0
            l_prec = l_gt / l_n * 100 if l_n > 0 else 0
            print(f"  {tier} high-margin: {h_prec:.1f}% ({h_gt}/{h_n}) | low-margin: {l_prec:.1f}% ({l_gt}/{l_n}) | gap: {h_prec - l_prec:+.1f}pp")
            holdout_gaps_by_tier[tier].append(h_prec - l_prec)

        restore_originals(predictor, originals)

    print(f"\n--- Holdout Summary (5-seed mean ± std) ---")
    print(f"  {'Tier':<10} {'High-Low Margin Gap':>20}")
    for tier in tier_order:
        gaps = holdout_gaps_by_tier.get(tier, [])
        if gaps:
            print(f"  {tier:<10} {np.mean(gaps):>+8.1f} ± {np.std(gaps):>5.1f}pp")

    # Save results
    results = {
        "hypothesis": "h442",
        "margin_distribution": {
            "min": round(float(np.min(margin_values)), 6),
            "q1": round(q1, 6),
            "median": round(q2, 6),
            "q3": round(q3, 6),
            "max": round(float(np.max(margin_values)), 6),
            "mean": round(float(np.mean(margin_values)), 6),
        },
        "correlations": {
            "margin_vs_r30": round(float(correlation), 3),
            "entropy_vs_r30": round(float(np.corrcoef(entropy_arr, r_arr)[0, 1]), 3),
            "neighbor_sim_vs_r30": round(float(np.corrcoef(neighbor_sim_arr, r_arr)[0, 1]), 3),
            "n_drugs_vs_r30": round(float(np.corrcoef(n_drugs_arr, r_arr)[0, 1]), 3),
        },
        "holdout_gaps": {
            tier: {
                "mean": round(float(np.mean(gaps)), 2),
                "std": round(float(np.std(gaps)), 2),
                "values": [round(float(g), 2) for g in gaps],
            }
            for tier, gaps in holdout_gaps_by_tier.items()
        },
    }

    output_path = Path(__file__).parent.parent / "data" / "analysis" / "h442_score_margin.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
