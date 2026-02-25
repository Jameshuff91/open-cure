#!/usr/bin/env python3
"""
h770: Holdout-Invisible Disease Analysis

Which diseases have zero holdout representation across all 5 seeds?
These diseases contribute to full-data precision but never to holdout precision,
causing systematic inflation in full-data estimates.

Analysis:
1. For each of 5 seeds, record which diseases land in holdout
2. Identify diseases that NEVER appear in holdout across all 5 seeds
3. Quantify the precision contribution of always-training diseases
4. Break down by tier and sub-reason
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
)

SEEDS = [42, 123, 456, 789, 2024]
TRAIN_RATIO = 0.8


def split_diseases(
    all_diseases: List[str], seed: int, train_ratio: float = 0.8
) -> Tuple[List[str], List[str]]:
    rng = np.random.RandomState(seed)
    shuffled = list(all_diseases)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def main():
    print("=" * 70)
    print("h770: Holdout-Invisible Disease Analysis")
    print("=" * 70)

    # Load predictor
    predictor = DrugRepurposingPredictor()

    # Load expanded GT
    gt_path = Path("data/reference/expanded_ground_truth.json")
    with open(gt_path) as f:
        expanded_gt = json.load(f)
    print(f"Expanded GT: {len(expanded_gt)} diseases")

    # Get all diseases in internal GT (what's used for splitting)
    internal_gt = predictor.ground_truth
    all_gt_diseases = sorted(internal_gt.keys())
    print(f"Internal GT: {len(all_gt_diseases)} diseases")

    # Track which diseases appear in holdout for each seed
    disease_holdout_count: Dict[str, int] = defaultdict(int)
    disease_holdout_seeds: Dict[str, List[int]] = defaultdict(list)

    for seed in SEEDS:
        train_diseases, holdout_diseases = split_diseases(all_gt_diseases, seed)
        for d in holdout_diseases:
            disease_holdout_count[d] += 1
            disease_holdout_seeds[d].append(seed)

    # Categorize diseases by holdout frequency
    never_holdout = []  # 0 seeds
    rare_holdout = []   # 1 seed
    sometimes_holdout = []  # 2-3 seeds
    often_holdout = []  # 4-5 seeds

    for d in all_gt_diseases:
        count = disease_holdout_count.get(d, 0)
        if count == 0:
            never_holdout.append(d)
        elif count == 1:
            rare_holdout.append(d)
        elif count <= 3:
            sometimes_holdout.append(d)
        else:
            often_holdout.append(d)

    print(f"\n--- Disease Holdout Frequency Distribution ---")
    print(f"Never in holdout (0/5 seeds): {len(never_holdout)} diseases ({100*len(never_holdout)/len(all_gt_diseases):.1f}%)")
    print(f"Rare holdout (1/5 seeds):     {len(rare_holdout)} diseases ({100*len(rare_holdout)/len(all_gt_diseases):.1f}%)")
    print(f"Sometimes (2-3/5 seeds):      {len(sometimes_holdout)} diseases ({100*len(sometimes_holdout)/len(all_gt_diseases):.1f}%)")
    print(f"Often (4-5/5 seeds):          {len(often_holdout)} diseases ({100*len(often_holdout)/len(all_gt_diseases):.1f}%)")

    # With 80/20 split and 5 seeds, expected appearance = 1 (20% * 5)
    # But it's random, so some diseases will be 0
    # Theoretical: P(never holdout in 5 seeds) = 0.8^5 = 32.8%
    print(f"\nTheoretical P(never holdout) = 0.8^5 = {0.8**5:.1%}")
    print(f"Actual: {len(never_holdout)/len(all_gt_diseases):.1%}")

    # Now generate predictions for ALL diseases and check which are holdout-invisible
    print("\n--- Generating Predictions for All Diseases ---")
    all_predictions = []
    disease_predictions: Dict[str, list] = defaultdict(list)

    for disease_id in all_gt_diseases:
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        result = predictor.predict(disease_name, top_n=30)
        for pred in result.predictions:
            all_predictions.append(pred)
            disease_predictions[disease_id].append(pred)

    print(f"Total predictions: {len(all_predictions)}")

    # Check which predictions come from holdout-invisible diseases
    never_holdout_set = set(never_holdout)

    # Per-tier analysis: how many predictions are from never-holdout diseases?
    tier_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {
        "total": 0, "never_holdout": 0,
        "gt_hit_total": 0, "gt_hit_never": 0
    })

    # Also track by sub-reason
    subreason_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {
        "total": 0, "never_holdout": 0,
        "gt_hit_total": 0, "gt_hit_never": 0
    })

    for disease_id, preds in disease_predictions.items():
        is_never = disease_id in never_holdout_set

        # Check GT hits using expanded GT
        disease_gt_drugs = set()
        if disease_id in expanded_gt:
            disease_gt_drugs = set(expanded_gt[disease_id])

        for pred in preds:
            tier = pred.confidence_tier.name if pred.confidence_tier else "UNKNOWN"
            sub_reason = getattr(pred, 'confidence_sub_reason', 'unknown') or 'unknown'

            # Check if this is a GT hit
            is_hit = pred.drug_id in disease_gt_drugs if disease_gt_drugs else False

            tier_stats[tier]["total"] += 1
            subreason_stats[sub_reason]["total"] += 1

            if is_never:
                tier_stats[tier]["never_holdout"] += 1
                subreason_stats[sub_reason]["never_holdout"] += 1

            if is_hit:
                tier_stats[tier]["gt_hit_total"] += 1
                subreason_stats[sub_reason]["gt_hit_total"] += 1
                if is_never:
                    tier_stats[tier]["gt_hit_never"] += 1
                    subreason_stats[sub_reason]["gt_hit_never"] += 1

    # Report tier-level analysis
    print("\n--- Per-Tier Holdout Invisibility ---")
    print(f"{'Tier':<12} {'Total':>6} {'Never-HO':>10} {'%':>6} {'GT Hits':>8} {'NH Hits':>8} {'FD Prec':>8} {'NH Prec':>8} {'Visible Prec':>12}")

    tier_order = ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]
    for tier in tier_order:
        if tier in tier_stats:
            s = tier_stats[tier]
            pct = 100 * s["never_holdout"] / s["total"] if s["total"] else 0
            fd_prec = 100 * s["gt_hit_total"] / s["total"] if s["total"] else 0
            nh_prec = 100 * s["gt_hit_never"] / s["never_holdout"] if s["never_holdout"] else 0
            visible = s["total"] - s["never_holdout"]
            visible_hits = s["gt_hit_total"] - s["gt_hit_never"]
            vis_prec = 100 * visible_hits / visible if visible else 0
            print(f"{tier:<12} {s['total']:>6} {s['never_holdout']:>10} {pct:>5.1f}% {s['gt_hit_total']:>8} {s['gt_hit_never']:>8} {fd_prec:>7.1f}% {nh_prec:>7.1f}% {vis_prec:>11.1f}%")

    # Inflation analysis: compare never-holdout precision vs holdout-visible precision
    print("\n--- Inflation Analysis ---")
    print("If never-holdout diseases have HIGHER precision than holdout-visible,")
    print("they inflate full-data estimates relative to holdout.\n")

    for tier in tier_order:
        if tier in tier_stats:
            s = tier_stats[tier]
            nh = s["never_holdout"]
            vis = s["total"] - nh
            nh_hits = s["gt_hit_never"]
            vis_hits = s["gt_hit_total"] - nh_hits

            nh_prec = 100 * nh_hits / nh if nh else 0
            vis_prec = 100 * vis_hits / vis if vis else 0
            inflation = nh_prec - vis_prec

            if nh > 0 and vis > 0:
                print(f"{tier:<12}: Never-HO {nh_prec:.1f}% ({nh_hits}/{nh}) vs Visible {vis_prec:.1f}% ({vis_hits}/{vis}) → Inflation: {inflation:+.1f}pp")

    # Top never-holdout diseases by prediction count
    print("\n--- Top 30 Never-Holdout Diseases by Prediction Count ---")
    never_holdout_pred_counts = []
    for d in never_holdout:
        name = predictor.disease_names.get(d, d)
        n_preds = len(disease_predictions.get(d, []))
        n_gt = len(internal_gt.get(d, {}))
        n_expanded_gt = len(expanded_gt.get(d, {}))

        # Count GT hits
        d_gt_drugs = set(expanded_gt.get(d, {}))
        hits = sum(1 for p in disease_predictions.get(d, []) if p.drug_id in d_gt_drugs)
        prec = 100 * hits / n_preds if n_preds else 0

        never_holdout_pred_counts.append((name, d, n_preds, n_gt, n_expanded_gt, hits, prec))

    never_holdout_pred_counts.sort(key=lambda x: x[2], reverse=True)
    print(f"{'Disease':<55} {'Preds':>5} {'iGT':>4} {'eGT':>4} {'Hits':>4} {'Prec':>6}")
    for name, did, n_preds, n_gt, n_egt, hits, prec in never_holdout_pred_counts[:30]:
        print(f"{name[:54]:<55} {n_preds:>5} {n_gt:>4} {n_egt:>4} {hits:>4} {prec:>5.1f}%")

    # Sub-reason analysis for never-holdout inflation
    print("\n--- Top 20 Sub-Reasons by Never-Holdout Inflation ---")
    inflation_data = []
    for sr, s in subreason_stats.items():
        nh = s["never_holdout"]
        vis = s["total"] - nh
        if nh >= 10 and vis >= 10:  # minimum sample
            nh_hits = s["gt_hit_never"]
            vis_hits = s["gt_hit_total"] - nh_hits
            nh_prec = 100 * nh_hits / nh
            vis_prec = 100 * vis_hits / vis
            inflation = nh_prec - vis_prec
            inflation_data.append((sr, s["total"], nh, vis, nh_prec, vis_prec, inflation))

    inflation_data.sort(key=lambda x: x[6], reverse=True)
    print(f"{'Sub-Reason':<40} {'Total':>5} {'NH':>5} {'Vis':>5} {'NH%':>6} {'Vis%':>6} {'Infl':>7}")
    for sr, total, nh, vis, nh_p, vis_p, infl in inflation_data[:20]:
        print(f"{sr[:39]:<40} {total:>5} {nh:>5} {vis:>5} {nh_p:>5.1f}% {vis_p:>5.1f}% {infl:>+6.1f}pp")

    # Category analysis of never-holdout diseases
    print("\n--- Never-Holdout Diseases by Category ---")
    cat_counts: Dict[str, int] = defaultdict(int)
    for d in never_holdout:
        cat = predictor.train_disease_categories.get(d, "unknown")
        cat_counts[cat] += 1

    all_cat_counts: Dict[str, int] = defaultdict(int)
    for d in all_gt_diseases:
        cat = predictor.train_disease_categories.get(d, "unknown")
        all_cat_counts[cat] += 1

    print(f"{'Category':<25} {'Never-HO':>10} {'Total':>8} {'%':>6}")
    for cat in sorted(all_cat_counts.keys()):
        nh = cat_counts.get(cat, 0)
        total = all_cat_counts[cat]
        pct = 100 * nh / total if total else 0
        print(f"{cat:<25} {nh:>10} {total:>8} {pct:>5.1f}%")

    # Save detailed results
    results = {
        "summary": {
            "total_diseases": len(all_gt_diseases),
            "never_holdout": len(never_holdout),
            "rare_holdout": len(rare_holdout),
            "sometimes_holdout": len(sometimes_holdout),
            "often_holdout": len(often_holdout),
            "theoretical_pct": 0.8**5,
            "actual_pct": len(never_holdout) / len(all_gt_diseases),
        },
        "tier_inflation": {},
        "never_holdout_diseases": [],
        "category_breakdown": {},
    }

    for tier in tier_order:
        if tier in tier_stats:
            s = tier_stats[tier]
            nh = s["never_holdout"]
            vis = s["total"] - nh
            nh_hits = s["gt_hit_never"]
            vis_hits = s["gt_hit_total"] - nh_hits
            results["tier_inflation"][tier] = {
                "total_preds": s["total"],
                "never_holdout_preds": nh,
                "never_holdout_pct": nh / s["total"] if s["total"] else 0,
                "never_holdout_precision": nh_hits / nh if nh else 0,
                "visible_precision": vis_hits / vis if vis else 0,
                "inflation_pp": (nh_hits / nh if nh else 0) - (vis_hits / vis if vis else 0),
                "full_data_precision": s["gt_hit_total"] / s["total"] if s["total"] else 0,
            }

    for name, did, n_preds, n_gt, n_egt, hits, prec in never_holdout_pred_counts:
        results["never_holdout_diseases"].append({
            "disease_id": did,
            "disease_name": name,
            "n_predictions": n_preds,
            "internal_gt": n_gt,
            "expanded_gt": n_egt,
            "gt_hits": hits,
            "precision": prec / 100,
        })

    for cat in sorted(all_cat_counts.keys()):
        results["category_breakdown"][cat] = {
            "never_holdout": cat_counts.get(cat, 0),
            "total": all_cat_counts[cat],
            "pct": cat_counts.get(cat, 0) / all_cat_counts[cat] if all_cat_counts[cat] else 0,
        }

    out_path = Path("data/analysis/h770_holdout_invisible_analysis.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
