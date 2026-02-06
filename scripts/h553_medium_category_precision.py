#!/usr/bin/env python3
"""
h553: MEDIUM Tier Precision by Category Analysis

Compute holdout precision for MEDIUM predictions broken down by disease_category.
Identify which categories are significantly below/above the MEDIUM average (~30%).
Assess whether category-specific demotion (MEDIUM→LOW) improves MEDIUM precision.

Uses the same 5-seed holdout framework from h393.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from production_predictor import (
    ConfidenceTier,
    DrugRepurposingPredictor,
    extract_cancer_types,
    DISEASE_HIERARCHY_GROUPS,
    HIERARCHY_EXCLUSIONS,
)

# Reuse h393 infrastructure
from h393_holdout_tier_validation import (
    split_diseases,
    recompute_gt_structures,
    restore_gt_structures,
)


def evaluate_medium_by_category(
    predictor: DrugRepurposingPredictor,
    disease_ids: List[str],
    gt_data: Dict[str, List[str]],
) -> Dict:
    """Evaluate MEDIUM tier precision by disease category.

    Returns per-category stats and per-sub-reason stats within MEDIUM.
    """
    gt_set = set()
    for disease_id, drugs in gt_data.items():
        for drug in drugs:
            if isinstance(drug, str):
                gt_set.add((disease_id, drug))
            elif isinstance(drug, dict):
                gt_set.add((disease_id, drug.get("drug_id") or drug.get("drug")))

    # Per-category stats for MEDIUM
    cat_stats: Dict[str, Dict] = defaultdict(lambda: {"hits": 0, "misses": 0})
    # Per-sub-reason stats for MEDIUM
    reason_stats: Dict[str, Dict] = defaultdict(lambda: {"hits": 0, "misses": 0})
    # Per-category × sub-reason
    cat_reason_stats: Dict[str, Dict] = defaultdict(lambda: {"hits": 0, "misses": 0})
    # Overall tier stats
    tier_stats: Dict[str, Dict] = defaultdict(lambda: {"hits": 0, "misses": 0})

    n_evaluated = 0

    for disease_id in disease_ids:
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        category = predictor.categorize_disease(disease_name)

        try:
            result = predictor.predict(disease_name, top_n=30, include_filtered=True)
        except Exception:
            continue

        n_evaluated += 1

        for pred in result.predictions:
            drug_id = pred.drug_id
            is_hit = (disease_id, drug_id) in gt_set
            tier = pred.confidence_tier.name
            reason = pred.category_specific_tier or "default"

            if is_hit:
                tier_stats[tier]["hits"] += 1
            else:
                tier_stats[tier]["misses"] += 1

            if tier == "MEDIUM":
                if is_hit:
                    cat_stats[category]["hits"] += 1
                    reason_stats[reason]["hits"] += 1
                    cat_reason_stats[f"{category}|{reason}"]["hits"] += 1
                else:
                    cat_stats[category]["misses"] += 1
                    reason_stats[reason]["misses"] += 1
                    cat_reason_stats[f"{category}|{reason}"]["misses"] += 1

    return {
        "n_diseases": n_evaluated,
        "category_stats": dict(cat_stats),
        "reason_stats": dict(reason_stats),
        "cat_reason_stats": dict(cat_reason_stats),
        "tier_stats": dict(tier_stats),
    }


def main() -> None:
    seeds = [42, 123, 456, 789, 2024]
    print("=" * 70)
    print("h553: MEDIUM Tier Precision by Category Analysis")
    print("=" * 70)

    print("\nLoading predictor...")
    predictor = DrugRepurposingPredictor()

    gt_path = predictor.reference_dir / "expanded_ground_truth.json"
    with open(gt_path) as f:
        gt_data = json.load(f)
    print(f"Loaded {len(gt_data)} GT diseases, {sum(len(v) for v in gt_data.values())} pairs")

    all_diseases = [d for d in predictor.ground_truth if d in predictor.embeddings]
    print(f"Diseases with GT + embeddings: {len(all_diseases)}")

    # Full-data baseline
    print("\n--- FULL-DATA BASELINE ---")
    full_result = evaluate_medium_by_category(predictor, all_diseases, gt_data)
    print(f"\nFull-data MEDIUM precision by category:")
    print(f"{'Category':<25s} {'Precision':>10s} {'Hits':>6s} {'Total':>7s}")
    print("-" * 55)
    for cat, stats in sorted(full_result["category_stats"].items(),
                              key=lambda x: x[1]["hits"]/(x[1]["hits"]+x[1]["misses"]) if (x[1]["hits"]+x[1]["misses"]) > 0 else 0,
                              reverse=True):
        total = stats["hits"] + stats["misses"]
        prec = stats["hits"] / total * 100 if total > 0 else 0
        print(f"  {cat:<25s} {prec:8.1f}%  {stats['hits']:>5d} / {total:>5d}")

    # Overall MEDIUM
    med_tier = full_result["tier_stats"].get("MEDIUM", {"hits": 0, "misses": 0})
    med_total = med_tier["hits"] + med_tier["misses"]
    med_prec = med_tier["hits"] / med_total * 100 if med_total > 0 else 0
    print(f"\n  {'OVERALL MEDIUM':<25s} {med_prec:8.1f}%  {med_tier['hits']:>5d} / {med_total:>5d}")

    print(f"\nFull-data MEDIUM precision by sub-reason:")
    print(f"{'Sub-reason':<40s} {'Precision':>10s} {'Hits':>6s} {'Total':>7s}")
    print("-" * 70)
    for reason, stats in sorted(full_result["reason_stats"].items(),
                                 key=lambda x: x[1]["hits"]+x[1]["misses"],
                                 reverse=True):
        total = stats["hits"] + stats["misses"]
        prec = stats["hits"] / total * 100 if total > 0 else 0
        print(f"  {reason:<40s} {prec:8.1f}%  {stats['hits']:>5d} / {total:>5d}")

    # Holdout evaluation
    print("\n" + "=" * 70)
    print("HOLDOUT EVALUATION (5 seeds)")
    print("=" * 70)

    all_cat_prec = defaultdict(list)  # category -> [precision per seed]
    all_cat_n = defaultdict(list)     # category -> [n per seed]
    all_reason_prec = defaultdict(list)
    all_reason_n = defaultdict(list)
    all_cat_reason_prec = defaultdict(list)
    all_cat_reason_n = defaultdict(list)
    all_tier_prec = defaultdict(list)

    for seed_idx, seed in enumerate(seeds):
        print(f"\nSeed {seed} ({seed_idx+1}/{len(seeds)})...")

        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        train_set = set(train_ids)

        originals = recompute_gt_structures(predictor, train_set)
        holdout_result = evaluate_medium_by_category(predictor, holdout_ids, gt_data)
        restore_gt_structures(predictor, originals)

        # Collect per-category MEDIUM holdout
        for cat, stats in holdout_result["category_stats"].items():
            total = stats["hits"] + stats["misses"]
            prec = stats["hits"] / total * 100 if total > 0 else 0
            all_cat_prec[cat].append(prec)
            all_cat_n[cat].append(total)

        # Collect per-reason MEDIUM holdout
        for reason, stats in holdout_result["reason_stats"].items():
            total = stats["hits"] + stats["misses"]
            prec = stats["hits"] / total * 100 if total > 0 else 0
            all_reason_prec[reason].append(prec)
            all_reason_n[reason].append(total)

        # Collect per cat×reason
        for cr, stats in holdout_result["cat_reason_stats"].items():
            total = stats["hits"] + stats["misses"]
            prec = stats["hits"] / total * 100 if total > 0 else 0
            all_cat_reason_prec[cr].append(prec)
            all_cat_reason_n[cr].append(total)

        # Tier-level
        for tier, stats in holdout_result["tier_stats"].items():
            total = stats["hits"] + stats["misses"]
            prec = stats["hits"] / total * 100 if total > 0 else 0
            all_tier_prec[tier].append(prec)

    # Print aggregate results
    print("\n" + "=" * 70)
    print("AGGREGATE HOLDOUT RESULTS")
    print("=" * 70)

    # Overall tier performance
    print("\n--- TIER PRECISION (holdout mean ± std) ---")
    for tier in ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]:
        if tier in all_tier_prec:
            mean = np.mean(all_tier_prec[tier])
            std = np.std(all_tier_prec[tier])
            print(f"  {tier:<10s}: {mean:5.1f}% ± {std:4.1f}%")

    # MEDIUM by category
    medium_avg = np.mean(all_tier_prec.get("MEDIUM", [30.0]))
    print(f"\n--- MEDIUM BY CATEGORY (holdout, sorted by precision) ---")
    print(f"{'Category':<25s} {'Holdout':>10s} {'±std':>6s} {'n/seed':>8s} {'Δ vs avg':>10s} {'Status'}")
    print("-" * 85)

    cat_rows = []
    for cat in sorted(all_cat_prec.keys()):
        if len(all_cat_prec[cat]) < 3:
            continue
        mean = np.mean(all_cat_prec[cat])
        std = np.std(all_cat_prec[cat])
        avg_n = np.mean(all_cat_n[cat])
        delta = mean - medium_avg
        cat_rows.append((cat, mean, std, avg_n, delta))

    cat_rows.sort(key=lambda x: x[1], reverse=True)
    for cat, mean, std, avg_n, delta in cat_rows:
        status = ""
        if avg_n < 5:
            status = "TINY-N"
        elif avg_n < 30:
            status = "SMALL-N"
        elif delta < -10:
            status = "CANDIDATE DEMOTE"
        elif delta < -5:
            status = "BELOW AVG"
        elif delta > 10:
            status = "ABOVE AVG"
        print(f"  {cat:<25s} {mean:8.1f}%  {std:4.1f}%  {avg_n:6.0f}   {delta:+8.1f}pp  {status}")

    # MEDIUM by sub-reason
    print(f"\n--- MEDIUM BY SUB-REASON (holdout, sorted by n) ---")
    print(f"{'Sub-reason':<40s} {'Holdout':>10s} {'±std':>6s} {'n/seed':>8s} {'Δ vs avg':>10s}")
    print("-" * 85)

    reason_rows = []
    for reason in sorted(all_reason_prec.keys()):
        if len(all_reason_prec[reason]) < 3:
            continue
        mean = np.mean(all_reason_prec[reason])
        std = np.std(all_reason_prec[reason])
        avg_n = np.mean(all_reason_n[reason])
        delta = mean - medium_avg
        reason_rows.append((reason, mean, std, avg_n, delta))

    reason_rows.sort(key=lambda x: x[3], reverse=True)
    for reason, mean, std, avg_n, delta in reason_rows:
        print(f"  {reason:<40s} {mean:8.1f}%  {std:4.1f}%  {avg_n:6.0f}   {delta:+8.1f}pp")

    # Cross-tab: category × reason (only for large cells)
    print(f"\n--- MEDIUM: CATEGORY × SUB-REASON (holdout, n/seed >= 10) ---")
    print(f"{'Category|Reason':<50s} {'Holdout':>10s} {'±std':>6s} {'n/seed':>8s} {'Δ vs avg':>10s}")
    print("-" * 90)

    cr_rows = []
    for cr in sorted(all_cat_reason_prec.keys()):
        if len(all_cat_reason_prec[cr]) < 3:
            continue
        avg_n = np.mean(all_cat_reason_n[cr])
        if avg_n < 10:
            continue
        mean = np.mean(all_cat_reason_prec[cr])
        std = np.std(all_cat_reason_prec[cr])
        delta = mean - medium_avg
        cr_rows.append((cr, mean, std, avg_n, delta))

    cr_rows.sort(key=lambda x: x[1])
    for cr, mean, std, avg_n, delta in cr_rows:
        print(f"  {cr:<50s} {mean:8.1f}%  {std:4.1f}%  {avg_n:6.0f}   {delta:+8.1f}pp")

    # Demotion simulation
    print(f"\n--- DEMOTION SIMULATION ---")
    print("If we demote categories below threshold to LOW, what happens to MEDIUM?")

    threshold_pps = [-15, -10, -5]
    for threshold in threshold_pps:
        demote_cats = [cat for cat, mean, std, avg_n, delta in cat_rows
                       if delta < threshold and avg_n >= 15]
        if not demote_cats:
            print(f"\n  Threshold: {threshold:+d}pp → No categories to demote")
            continue

        # Compute what MEDIUM would be without those categories
        remaining_hits = []
        remaining_total = []
        for seed_idx in range(len(seeds)):
            seed_hits = 0
            seed_total = 0
            for cat, mean, std, avg_n, delta in cat_rows:
                if cat in demote_cats:
                    continue
                if seed_idx < len(all_cat_prec.get(cat, [])):
                    cat_n = all_cat_n[cat][seed_idx]
                    cat_h = all_cat_prec[cat][seed_idx] * cat_n / 100
                    seed_hits += cat_h
                    seed_total += cat_n
            if seed_total > 0:
                remaining_hits.append(seed_hits / seed_total * 100)

        new_med_mean = np.mean(remaining_hits) if remaining_hits else 0
        demote_n = sum(np.mean(all_cat_n[c]) for c in demote_cats)
        print(f"\n  Threshold: {threshold:+d}pp → Demote: {', '.join(demote_cats)}")
        print(f"  Predictions moved: ~{demote_n:.0f}/seed")
        print(f"  New MEDIUM: {new_med_mean:.1f}% (was {medium_avg:.1f}%, Δ={new_med_mean-medium_avg:+.1f}pp)")

    # Save results
    output = {
        "medium_avg_holdout": round(float(medium_avg), 1),
        "categories": {},
        "reasons": {},
        "cross_tab": {},
    }
    for cat, mean, std, avg_n, delta in cat_rows:
        output["categories"][cat] = {
            "holdout_mean": round(mean, 1),
            "holdout_std": round(std, 1),
            "n_per_seed": round(avg_n, 0),
            "delta_vs_avg": round(delta, 1),
        }
    for reason, mean, std, avg_n, delta in reason_rows:
        output["reasons"][reason] = {
            "holdout_mean": round(mean, 1),
            "holdout_std": round(std, 1),
            "n_per_seed": round(avg_n, 0),
            "delta_vs_avg": round(delta, 1),
        }
    for cr, mean, std, avg_n, delta in cr_rows:
        output["cross_tab"][cr] = {
            "holdout_mean": round(mean, 1),
            "holdout_std": round(std, 1),
            "n_per_seed": round(avg_n, 0),
            "delta_vs_avg": round(delta, 1),
        }

    output_path = Path("data/analysis/h553_medium_category.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
