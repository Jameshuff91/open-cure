#!/usr/bin/env python3
"""
h445: TransE Score Distribution - Why Does TransE Hurt Global Ranking?

h443 found TransE boolean is NEGATIVE within tiers globally:
- MEDIUM: TransE=True 22.7% vs TransE=False 26.4% (-3.7pp)
- LOW: TransE=True 11.6% vs TransE=False 16.0% (-4.4pp)

But h405 found TransE is POSITIVE per-disease:
- MEDIUM + TransE top-30: 34.7% holdout (+13.6pp)

Hypothesis: The paradox arises because TransE consilience selects predictions
that are lower-ranked by kNN (i.e., rank 11-20 rather than rank 1-5).
Per-disease, these TransE predictions are still better than average.
But globally, they're systematically lower-ranked by kNN, so they appear
weaker when sorted across all diseases.

Method:
1. For TransE=True vs TransE=False predictions, compare average kNN rank
2. Break down per-disease: within a disease, does TransE=True have higher
   precision than the disease average?
3. Check if the global negative effect is a Simpson's paradox
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from production_predictor import DrugRepurposingPredictor


TIER_ORDER = ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]


def main():
    print("=" * 70)
    print("h445: TransE Paradox - Why Does TransE Hurt Global Ranking?")
    print("=" * 70)

    print("\nLoading predictor...")
    t0 = time.time()
    predictor = DrugRepurposingPredictor()
    print(f"Loaded in {time.time() - t0:.1f}s")

    diseases_with_gt = [d for d in predictor.ground_truth if d in predictor.embeddings]
    print(f"Diseases: {len(diseases_with_gt)}")

    # Collect all predictions
    all_preds = []
    for disease_id in diseases_with_gt:
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
                "transe": pred.transe_consilience,
            })

    print(f"Total predictions: {len(all_preds)}")

    # ===== PART 1: Average kNN Rank by TransE status =====
    print("\n" + "=" * 70)
    print("PART 1: Average kNN Rank by TransE Status (per tier)")
    print("=" * 70)

    tier_preds: Dict[str, list] = defaultdict(list)
    for p in all_preds:
        tier_preds[p["tier"]].append(p)

    print(f"\n  {'Tier':<10} {'TransE Avg Rank':>16} {'No-TransE Avg':>14} {'Delta':>8} {'TransE n':>10}")
    for tier in TIER_ORDER:
        if tier not in tier_preds:
            continue
        preds = tier_preds[tier]
        with_t = [p for p in preds if p["transe"]]
        without_t = [p for p in preds if not p["transe"]]
        avg_t = np.mean([p["rank"] for p in with_t]) if with_t else 0
        avg_nt = np.mean([p["rank"] for p in without_t]) if without_t else 0
        print(f"  {tier:<10} {avg_t:>14.1f} {avg_nt:>14.1f} {avg_t - avg_nt:>+7.1f} {len(with_t):>10}")

    # ===== PART 2: Per-Disease Simpson's Paradox Check =====
    print("\n" + "=" * 70)
    print("PART 2: Per-Disease TransE Precision (Simpson's Paradox Check)")
    print("=" * 70)

    # For each disease, compute TransE=True precision vs overall precision
    disease_preds: Dict[str, list] = defaultdict(list)
    for p in all_preds:
        disease_preds[p["disease_id"]].append(p)

    # Per-disease, per-tier analysis
    for tier in TIER_ORDER:
        disease_transe_wins = 0
        disease_transe_ties = 0
        disease_transe_losses = 0
        disease_transe_better_sum = 0.0
        disease_transe_worse_sum = 0.0
        n_diseases = 0

        per_disease_data = []

        for disease_id, dpreds in disease_preds.items():
            tier_dpreds = [p for p in dpreds if p["tier"] == tier]
            if len(tier_dpreds) < 3:
                continue

            with_t = [p for p in tier_dpreds if p["transe"]]
            without_t = [p for p in tier_dpreds if not p["transe"]]

            if not with_t or not without_t:
                continue

            t_prec = sum(1 for p in with_t if p["is_gt"]) / len(with_t) * 100
            nt_prec = sum(1 for p in without_t if p["is_gt"]) / len(without_t) * 100
            overall_prec = sum(1 for p in tier_dpreds if p["is_gt"]) / len(tier_dpreds) * 100

            n_diseases += 1
            gap = t_prec - nt_prec
            if gap > 0:
                disease_transe_wins += 1
                disease_transe_better_sum += gap
            elif gap < 0:
                disease_transe_losses += 1
                disease_transe_worse_sum += gap
            else:
                disease_transe_ties += 1

            per_disease_data.append({
                "disease_id": disease_id,
                "n_transe": len(with_t),
                "n_no_transe": len(without_t),
                "transe_prec": t_prec,
                "no_transe_prec": nt_prec,
                "gap": gap,
            })

        if n_diseases > 0:
            print(f"\n  --- {tier} (n={n_diseases} diseases with both TransE and non-TransE) ---")
            print(f"  TransE better: {disease_transe_wins} ({disease_transe_wins/n_diseases*100:.1f}%)")
            print(f"  TransE worse:  {disease_transe_losses} ({disease_transe_losses/n_diseases*100:.1f}%)")
            print(f"  Tied:          {disease_transe_ties}")
            avg_better = disease_transe_better_sum / disease_transe_wins if disease_transe_wins > 0 else 0
            avg_worse = disease_transe_worse_sum / disease_transe_losses if disease_transe_losses > 0 else 0
            print(f"  Avg gap when better: {avg_better:+.1f}pp")
            print(f"  Avg gap when worse:  {avg_worse:+.1f}pp")

    # ===== PART 3: TransE + kNN Rank interaction =====
    print("\n" + "=" * 70)
    print("PART 3: TransE Precision by kNN Rank Bucket (Is TransE additive to rank?)")
    print("=" * 70)

    # Key question: within rank 1-5, does TransE help?
    # This tells us if TransE adds info beyond rank
    rank_buckets = [(1, 5), (6, 10), (11, 20)]

    for tier in TIER_ORDER:
        if tier not in tier_preds:
            continue
        preds = tier_preds[tier]
        if len(preds) < 30:
            continue

        print(f"\n  --- {tier} ---")
        print(f"  {'Rank':<12} {'TransE Prec':>12} {'No-T Prec':>10} {'Gap':>8} {'T n':>5} {'NT n':>6}")
        for lo, hi in rank_buckets:
            bucket_ps = [p for p in preds if lo <= p["rank"] <= hi]
            with_t = [p for p in bucket_ps if p["transe"]]
            without_t = [p for p in bucket_ps if not p["transe"]]
            t_prec = sum(1 for p in with_t if p["is_gt"]) / len(with_t) * 100 if with_t else 0
            nt_prec = sum(1 for p in without_t if p["is_gt"]) / len(without_t) * 100 if without_t else 0
            print(f"  rank {lo:>2}-{hi:<2}  {t_prec:>10.1f}% {nt_prec:>10.1f}% {t_prec-nt_prec:>+7.1f} {len(with_t):>5} {len(without_t):>6}")

    # ===== PART 4: Disease Size Confound =====
    print("\n" + "=" * 70)
    print("PART 4: Disease GT Size Confound")
    print("=" * 70)
    print("(Do diseases with more GT drugs have more TransE consilience?)")

    # For each disease, check GT size vs fraction of TransE=True predictions
    disease_stats = []
    for disease_id, dpreds in disease_preds.items():
        gt_size = len(predictor.ground_truth.get(disease_id, set()))
        n_transe = sum(1 for p in dpreds if p["transe"])
        n_total = len(dpreds)
        frac_transe = n_transe / n_total if n_total > 0 else 0
        disease_prec = sum(1 for p in dpreds if p["is_gt"]) / n_total * 100 if n_total > 0 else 0
        disease_stats.append({
            "disease_id": disease_id,
            "gt_size": gt_size,
            "n_transe": n_transe,
            "n_total": n_total,
            "frac_transe": frac_transe,
            "precision": disease_prec,
        })

    gt_sizes = [s["gt_size"] for s in disease_stats]
    frac_transes = [s["frac_transe"] for s in disease_stats]
    precisions = [s["precision"] for s in disease_stats]

    corr_gt_transe = float(np.corrcoef(gt_sizes, frac_transes)[0, 1])
    corr_gt_prec = float(np.corrcoef(gt_sizes, precisions)[0, 1])
    corr_transe_prec = float(np.corrcoef(frac_transes, precisions)[0, 1])

    print(f"  GT size vs frac_TransE:  r = {corr_gt_transe:.3f}")
    print(f"  GT size vs precision:    r = {corr_gt_prec:.3f}")
    print(f"  frac_TransE vs precision: r = {corr_transe_prec:.3f}")

    # Quartile analysis
    sorted_by_gt = sorted(disease_stats, key=lambda x: x["gt_size"])
    q_size = len(sorted_by_gt) // 4
    quartiles = [
        ("Q1 (smallest GT)", sorted_by_gt[:q_size]),
        ("Q2", sorted_by_gt[q_size:2*q_size]),
        ("Q3", sorted_by_gt[2*q_size:3*q_size]),
        ("Q4 (largest GT)", sorted_by_gt[3*q_size:]),
    ]

    print(f"\n  {'Quartile':<20} {'Avg GT':>8} {'Avg TransE%':>12} {'Avg Prec':>10}")
    for name, group in quartiles:
        avg_gt = np.mean([s["gt_size"] for s in group])
        avg_frac = np.mean([s["frac_transe"] for s in group]) * 100
        avg_prec = np.mean([s["precision"] for s in group])
        print(f"  {name:<20} {avg_gt:>7.1f} {avg_frac:>10.1f}% {avg_prec:>9.1f}%")

    # ===== PART 5: Summary =====
    print("\n" + "=" * 70)
    print("PART 5: Summary & Diagnosis")
    print("=" * 70)

    # Compute final summary stats
    for tier in TIER_ORDER:
        if tier not in tier_preds:
            continue
        preds = tier_preds[tier]
        with_t = [p for p in preds if p["transe"]]
        without_t = [p for p in preds if not p["transe"]]

        # Key: per-disease precision for TransE=True preds
        # Group TransE=True by disease, compute per-disease precision
        disease_t_preds: Dict[str, list] = defaultdict(list)
        for p in with_t:
            disease_t_preds[p["disease_id"]].append(p)

        per_disease_precs = []
        for _, dpreds in disease_t_preds.items():
            prec = sum(1 for p in dpreds if p["is_gt"]) / len(dpreds) * 100
            per_disease_precs.append(prec)

        global_prec = sum(1 for p in with_t if p["is_gt"]) / len(with_t) * 100 if with_t else 0
        per_disease_avg = np.mean(per_disease_precs) if per_disease_precs else 0

        if with_t:
            print(f"\n  {tier}:")
            print(f"    Global TransE precision: {global_prec:.1f}% (n={len(with_t)})")
            print(f"    Per-disease avg TransE precision: {per_disease_avg:.1f}% (n={len(per_disease_precs)} diseases)")
            print(f"    Simpson's gap: {per_disease_avg - global_prec:+.1f}pp")

    # Save results
    results = {
        "hypothesis": "h445",
        "correlations": {
            "gt_size_vs_frac_transe": corr_gt_transe,
            "gt_size_vs_precision": corr_gt_prec,
            "frac_transe_vs_precision": corr_transe_prec,
        },
    }

    output_path = Path(__file__).parent.parent / "data" / "analysis" / "h445_transe_paradox.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
