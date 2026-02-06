#!/usr/bin/env python3
"""
h442 v2: Score Tie Density as Disease-Level Confidence Signal

Initial h442 found that rank-20/21 margin = 0 for >50% of diseases because
kNN scores are heavily tied (drugs sharing neighbors get identical scores).

Revised approach: Use the NUMBER of drugs tied at the rank-20 score value as
a measure of ranking instability. More ties = more arbitrary the rank-20 cutoff.

Also test: unique score count in top-30, score dispersion (max-min)/max,
and number of GT drugs per disease (as a confounder check).

Method:
1. For each disease, compute score tie metrics
2. Split by median of best metric
3. Compare tier precision
4. Validate on holdout
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
    DISEASE_HIERARCHY_GROUPS,
    extract_cancer_types,
)


def compute_score_metrics(predictor: DrugRepurposingPredictor, disease_id: str, k: int = 20) -> Dict:
    """Compute score distribution metrics for a disease."""
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
        nd = predictor.train_diseases[idx]
        ns = working_sims[idx]
        if nd in predictor.ground_truth:
            for drug_id in predictor.ground_truth[nd]:
                if drug_id in predictor.embeddings:
                    drug_scores[drug_id] += ns

    if len(drug_scores) < 30:
        return {"valid": False}

    sorted_drugs = sorted(drug_scores.items(), key=lambda x: x[1], reverse=True)
    scores = [s for _, s in sorted_drugs]

    # Round to avoid floating point noise
    scores_rounded = [round(s, 6) for s in scores]

    # Metric 1: Ties at rank 20 - how many drugs have the same score as rank 20?
    rank20_score = scores_rounded[19]
    ties_at_20 = sum(1 for s in scores_rounded if s == rank20_score)

    # Metric 2: Unique score count in top 30
    unique_top30 = len(set(scores_rounded[:30]))

    # Metric 3: Score dispersion (max - min) / max for top 30
    dispersion = (scores[0] - scores[29]) / scores[0] if scores[0] > 0 else 0

    # Metric 4: Score at rank 1 vs rank 20 ratio
    score_ratio = scores[19] / scores[0] if scores[0] > 0 else 0

    # Metric 5: Number of unique neighbors contributing (how many distinct diseases?)
    neighbor_count = 0
    for idx in top_k_idx:
        nd = predictor.train_diseases[idx]
        if nd in predictor.ground_truth and len(predictor.ground_truth[nd]) > 0:
            neighbor_count += 1

    # Metric 6: Total drugs (candidate pool size)
    n_drugs = len(drug_scores)

    # Metric 7: Max score (measures if disease has strong neighbors)
    max_score = scores[0]

    return {
        "valid": True,
        "ties_at_20": ties_at_20,
        "unique_top30": unique_top30,
        "dispersion": float(dispersion),
        "score_ratio": float(score_ratio),
        "neighbor_count": neighbor_count,
        "n_drugs": n_drugs,
        "max_score": float(max_score),
        "category": category,
    }


def split_diseases(all_diseases: List[str], seed: int, train_ratio: float = 0.8) -> Tuple[List[str], List[str]]:
    rng = np.random.RandomState(seed)
    shuffled = list(all_diseases)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def recompute_gt_structures(predictor: DrugRepurposingPredictor, train_set: Set[str]) -> Dict:
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


def main():
    print("=" * 70)
    print("h442 v2: Score Tie Density as Disease-Level Confidence Signal")
    print("=" * 70)

    predictor = DrugRepurposingPredictor()
    diseases = [d for d in predictor.ground_truth if d in predictor.embeddings]
    print(f"Diseases: {len(diseases)}")

    # ===== PART 1: Compute metrics =====
    print("\n" + "=" * 70)
    print("PART 1: Score Distribution Metrics")
    print("=" * 70)

    metrics = {}
    for d in diseases:
        m = compute_score_metrics(predictor, d)
        if m["valid"]:
            metrics[d] = m

    print(f"Valid diseases: {len(metrics)}")

    # Distribution of each metric
    for metric_name in ["ties_at_20", "unique_top30", "dispersion", "n_drugs", "max_score"]:
        vals = [m[metric_name] for m in metrics.values()]
        print(f"\n{metric_name}:")
        print(f"  Min={np.min(vals):.2f}, Q1={np.percentile(vals, 25):.2f}, "
              f"Median={np.median(vals):.2f}, Q3={np.percentile(vals, 75):.2f}, "
              f"Max={np.max(vals):.2f}")

    # ===== PART 2: Correlation with R@30 =====
    print("\n" + "=" * 70)
    print("PART 2: Correlation with Disease-Level R@30")
    print("=" * 70)

    disease_r30 = {}
    for d in metrics:
        gt_drugs = predictor.ground_truth.get(d, set())
        dn = predictor.disease_names.get(d, d)
        result = predictor.predict(dn, k=20, top_n=30, include_filtered=True)
        hits = sum(1 for p in result.predictions if p.drug_id in gt_drugs)
        disease_r30[d] = hits / len(gt_drugs) if gt_drugs else 0

    common = sorted(set(metrics.keys()) & set(disease_r30.keys()))
    r_arr = np.array([disease_r30[d] for d in common])

    for metric_name in ["ties_at_20", "unique_top30", "dispersion", "n_drugs", "max_score", "neighbor_count"]:
        m_arr = np.array([metrics[d][metric_name] for d in common])
        corr = np.corrcoef(m_arr, r_arr)[0, 1]
        print(f"{metric_name:>20} vs R@30: r = {corr:+.3f}")

    # Also correlate GT size (confounder check)
    gt_sizes = np.array([len(predictor.ground_truth.get(d, set())) for d in common])
    print(f"{'gt_size':>20} vs R@30: r = {np.corrcoef(gt_sizes, r_arr)[0, 1]:+.3f}")

    # ===== PART 3: Split by ties_at_20 and n_drugs =====
    print("\n" + "=" * 70)
    print("PART 3: Tier Precision by Score Metrics (Full Data)")
    print("=" * 70)

    tier_order = ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]

    # For each metric, split at median and compare precision
    for metric_name in ["ties_at_20", "unique_top30", "n_drugs", "max_score"]:
        vals = {d: metrics[d][metric_name] for d in metrics}
        median_val = float(np.median(list(vals.values())))

        # Handle ties at median by including in "high" group
        high_diseases = [d for d, v in vals.items() if v > median_val]
        low_diseases = [d for d, v in vals.items() if v <= median_val]

        # If imbalanced (>70/30 split), use quartiles instead
        if len(high_diseases) < len(metrics) * 0.2 or len(low_diseases) < len(metrics) * 0.2:
            q75 = float(np.percentile(list(vals.values()), 75))
            q25 = float(np.percentile(list(vals.values()), 25))
            high_diseases = [d for d, v in vals.items() if v >= q75]
            low_diseases = [d for d, v in vals.items() if v <= q25]
            split_type = "Q1 vs Q4"
        else:
            split_type = "median"

        print(f"\n--- {metric_name} ({split_type}, high={len(high_diseases)}, low={len(low_diseases)}) ---")

        # Compute per-tier precision for each group
        for label, disease_list in [("HIGH-metric", high_diseases), ("LOW-metric", low_diseases)]:
            tier_counts: Dict[str, Dict] = defaultdict(lambda: {"gt": 0, "n": 0})
            for d in disease_list:
                gt_drugs = predictor.ground_truth.get(d, set())
                dn = predictor.disease_names.get(d, d)
                result = predictor.predict(dn, k=20, top_n=30, include_filtered=True)
                for pred in result.predictions:
                    tier = pred.confidence_tier.value
                    tier_counts[tier]["n"] += 1
                    if pred.drug_id in gt_drugs:
                        tier_counts[tier]["gt"] += 1

            print(f"  {label}:")
            for tier in tier_order:
                n = tier_counts[tier]["n"]
                gt = tier_counts[tier]["gt"]
                prec = gt / n * 100 if n > 0 else 0
                print(f"    {tier}: {prec:.1f}% ({gt}/{n})")

    # ===== PART 4: Holdout validation with best metric =====
    print("\n" + "=" * 70)
    print("PART 4: Holdout Validation - n_drugs and unique_top30")
    print("=" * 70)

    seeds = [42, 123, 456, 789, 2024]
    holdout_gaps: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        train_diseases_list, holdout_diseases = split_diseases(diseases, seed)
        train_set = set(train_diseases_list)

        originals = recompute_gt_structures(predictor, train_set)

        # Compute metrics for holdout diseases (using holdout kNN index)
        holdout_metrics = {}
        for d in holdout_diseases:
            m = compute_score_metrics(predictor, d)
            if m["valid"]:
                holdout_metrics[d] = m

        for metric_name in ["n_drugs", "unique_top30", "ties_at_20"]:
            vals = {d: holdout_metrics[d][metric_name] for d in holdout_metrics}
            if not vals:
                continue
            median_val = float(np.median(list(vals.values())))

            high_d = [d for d, v in vals.items() if v > median_val]
            low_d = [d for d, v in vals.items() if v <= median_val]

            # If too imbalanced, use quartiles
            if len(high_d) < len(holdout_metrics) * 0.2 or len(low_d) < len(holdout_metrics) * 0.2:
                q75 = float(np.percentile(list(vals.values()), 75))
                q25 = float(np.percentile(list(vals.values()), 25))
                high_d = [d for d, v in vals.items() if v >= q75]
                low_d = [d for d, v in vals.items() if v <= q25]

            for tier in tier_order:
                h_gt, h_n, l_gt, l_n = 0, 0, 0, 0

                for d in high_d:
                    gt_drugs = predictor.ground_truth.get(d, set())
                    dn = predictor.disease_names.get(d, d)
                    result = predictor.predict(dn, k=20, top_n=30, include_filtered=True)
                    for pred in result.predictions:
                        if pred.confidence_tier.value == tier:
                            h_n += 1
                            if pred.drug_id in gt_drugs:
                                h_gt += 1

                for d in low_d:
                    gt_drugs = predictor.ground_truth.get(d, set())
                    dn = predictor.disease_names.get(d, d)
                    result = predictor.predict(dn, k=20, top_n=30, include_filtered=True)
                    for pred in result.predictions:
                        if pred.confidence_tier.value == tier:
                            l_n += 1
                            if pred.drug_id in gt_drugs:
                                l_gt += 1

                h_prec = h_gt / h_n * 100 if h_n > 0 else 0
                l_prec = l_gt / l_n * 100 if l_n > 0 else 0
                holdout_gaps[metric_name][tier].append(h_prec - l_prec)

            # Print summary for this seed
            print(f"  {metric_name}: high={len(high_d)}, low={len(low_d)}")

        restore_originals(predictor, originals)

    # Summary
    print(f"\n--- Holdout Summary (5-seed mean ± std) ---")
    for metric_name in ["n_drugs", "unique_top30", "ties_at_20"]:
        print(f"\n{metric_name}:")
        for tier in tier_order:
            gaps = holdout_gaps[metric_name].get(tier, [])
            if gaps:
                print(f"  {tier}: {np.mean(gaps):+.1f} ± {np.std(gaps):.1f}pp")

    # Save results
    results = {
        "hypothesis": "h442",
        "version": 2,
        "correlations": {},
        "holdout_gaps": {
            metric: {
                tier: {"mean": round(float(np.mean(g)), 2), "std": round(float(np.std(g)), 2)}
                for tier, g in tier_gaps.items()
            }
            for metric, tier_gaps in holdout_gaps.items()
        },
    }

    output_path = Path(__file__).parent.parent / "data" / "analysis" / "h442_score_margin_v2.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
