#!/usr/bin/env python3
"""
h391: MEDIUM Tier Overlap Anomaly - Why Does Overlap Hurt MEDIUM?

h388 found MEDIUM tier predictions with target overlap have LOWER precision
(14.8%) than those without (20.7%). Counterintuitive.

Hypothesis: MEDIUM overlap predictions are enriched in broad-target drugs
(kinase inhibitors, etc.) that match many disease genes non-specifically.

Method:
1. Collect all MEDIUM predictions with/without target overlap
2. Analyze drug target counts (how many genes each drug targets)
3. Check if broad-target drugs are enriched in overlap group
4. Check if overlap drugs are concentrated in specific disease categories
5. Look at the specific drugs driving the anomaly
"""

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from production_predictor import DrugRepurposingPredictor


TIER_ORDER = ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]


def main():
    print("=" * 70)
    print("h391: MEDIUM Overlap Anomaly Investigation")
    print("=" * 70)

    print("\nLoading predictor...")
    t0 = time.time()
    predictor = DrugRepurposingPredictor()
    print(f"Loaded in {time.time() - t0:.1f}s")

    diseases_with_gt = [d for d in predictor.ground_truth if d in predictor.embeddings]
    print(f"Diseases: {len(diseases_with_gt)}")

    # Collect all predictions with overlap info
    all_preds = []
    for disease_id in diseases_with_gt:
        if disease_id not in predictor.embeddings:
            continue
        gt_drugs = predictor.ground_truth.get(disease_id, set())
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        category = predictor.categorize_disease(disease_name)
        result = predictor.predict(disease_name, k=20, top_n=30, include_filtered=True)

        for pred in result.predictions:
            overlap_count = predictor._get_target_overlap_count(pred.drug_id, disease_id)
            drug_targets = predictor.drug_targets.get(pred.drug_id, set())
            disease_genes = predictor.disease_genes.get(disease_id, set())

            all_preds.append({
                "disease_id": disease_id,
                "disease_name": disease_name,
                "drug_id": pred.drug_id,
                "drug_name": pred.drug_name,
                "rank": pred.rank,
                "tier": pred.confidence_tier.value,
                "knn_score": pred.knn_score,
                "is_gt": pred.drug_id in gt_drugs,
                "overlap_count": overlap_count,
                "has_overlap": overlap_count > 0,
                "drug_n_targets": len(drug_targets),
                "disease_n_genes": len(disease_genes),
                "category": category,
                "rule": pred.category_specific_tier or "default",
            })

    print(f"Total predictions: {len(all_preds)}")

    # ===== PART 1: Verify the anomaly =====
    print("\n" + "=" * 70)
    print("PART 1: Verify Overlap Anomaly Across All Tiers")
    print("=" * 70)

    for tier in TIER_ORDER:
        tier_ps = [p for p in all_preds if p["tier"] == tier]
        if not tier_ps:
            continue
        with_ov = [p for p in tier_ps if p["has_overlap"]]
        without_ov = [p for p in tier_ps if not p["has_overlap"]]
        ov_prec = sum(1 for p in with_ov if p["is_gt"]) / len(with_ov) * 100 if with_ov else 0
        no_ov_prec = sum(1 for p in without_ov if p["is_gt"]) / len(without_ov) * 100 if without_ov else 0
        print(f"  {tier:<10} Overlap: {ov_prec:.1f}% (n={len(with_ov)}) | No overlap: {no_ov_prec:.1f}% (n={len(without_ov)}) | Gap: {ov_prec-no_ov_prec:+.1f}pp")

    # ===== PART 2: Drug target count distribution =====
    print("\n" + "=" * 70)
    print("PART 2: Drug Target Count Distribution (MEDIUM tier)")
    print("=" * 70)

    medium_ps = [p for p in all_preds if p["tier"] == "MEDIUM"]
    ov_medium = [p for p in medium_ps if p["has_overlap"]]
    no_ov_medium = [p for p in medium_ps if not p["has_overlap"]]

    if ov_medium:
        ov_targets = [p["drug_n_targets"] for p in ov_medium]
        no_targets = [p["drug_n_targets"] for p in no_ov_medium]
        print(f"\n  With overlap (n={len(ov_medium)}):")
        print(f"    Drug target count: mean={np.mean(ov_targets):.1f}, median={np.median(ov_targets):.0f}, "
              f"Q1={np.percentile(ov_targets, 25):.0f}, Q3={np.percentile(ov_targets, 75):.0f}")
        print(f"    Precision: {sum(1 for p in ov_medium if p['is_gt'])/len(ov_medium)*100:.1f}%")

        print(f"\n  Without overlap (n={len(no_ov_medium)}):")
        print(f"    Drug target count: mean={np.mean(no_targets):.1f}, median={np.median(no_targets):.0f}, "
              f"Q1={np.percentile(no_targets, 25):.0f}, Q3={np.percentile(no_targets, 75):.0f}")
        print(f"    Precision: {sum(1 for p in no_ov_medium if p['is_gt'])/len(no_ov_medium)*100:.1f}%")

    # ===== PART 3: Broad vs narrow target drugs =====
    print("\n" + "=" * 70)
    print("PART 3: Broad vs Narrow Target Drugs in MEDIUM + Overlap")
    print("=" * 70)

    if ov_medium:
        # Split by target count quartiles
        target_counts = [p["drug_n_targets"] for p in ov_medium]
        median_targets = np.median(target_counts)
        broad = [p for p in ov_medium if p["drug_n_targets"] > median_targets]
        narrow = [p for p in ov_medium if p["drug_n_targets"] <= median_targets]

        broad_prec = sum(1 for p in broad if p["is_gt"]) / len(broad) * 100 if broad else 0
        narrow_prec = sum(1 for p in narrow if p["is_gt"]) / len(narrow) * 100 if narrow else 0

        print(f"  Broad (>{median_targets:.0f} targets): {broad_prec:.1f}% precision (n={len(broad)})")
        print(f"  Narrow (<={median_targets:.0f} targets): {narrow_prec:.1f}% precision (n={len(narrow)})")
        print(f"  Gap: {broad_prec - narrow_prec:+.1f}pp")

        # More detailed quartile breakdown
        quartiles_bounds = [0, np.percentile(target_counts, 25), np.percentile(target_counts, 50),
                           np.percentile(target_counts, 75), max(target_counts) + 1]
        print(f"\n  {'Target Count':<20} {'n':>6} {'GT':>5} {'Precision':>10}")
        for i in range(len(quartiles_bounds) - 1):
            lo = quartiles_bounds[i]
            hi = quartiles_bounds[i + 1]
            bucket = [p for p in ov_medium if lo <= p["drug_n_targets"] < hi]
            if bucket:
                prec = sum(1 for p in bucket if p["is_gt"]) / len(bucket) * 100
                print(f"  {lo:.0f}-{hi:.0f} targets  {len(bucket):>6} {sum(1 for p in bucket if p['is_gt']):>5} {prec:>9.1f}%")

    # ===== PART 4: Category breakdown =====
    print("\n" + "=" * 70)
    print("PART 4: MEDIUM + Overlap by Disease Category")
    print("=" * 70)

    cat_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"n": 0, "gt": 0})
    cat_no_ov: Dict[str, Dict[str, int]] = defaultdict(lambda: {"n": 0, "gt": 0})

    for p in medium_ps:
        cat = p["category"]
        if p["has_overlap"]:
            cat_stats[cat]["n"] += 1
            cat_stats[cat]["gt"] += int(p["is_gt"])
        else:
            cat_no_ov[cat]["n"] += 1
            cat_no_ov[cat]["gt"] += int(p["is_gt"])

    all_cats = sorted(set(list(cat_stats.keys()) + list(cat_no_ov.keys())))
    print(f"\n  {'Category':<25} {'OV Prec':>8} {'OV n':>5} {'No-OV Prec':>11} {'No-OV n':>7} {'Gap':>8}")
    for cat in all_cats:
        ov = cat_stats.get(cat, {"n": 0, "gt": 0})
        nov = cat_no_ov.get(cat, {"n": 0, "gt": 0})
        ov_prec = ov["gt"] / ov["n"] * 100 if ov["n"] > 0 else 0
        nov_prec = nov["gt"] / nov["n"] * 100 if nov["n"] > 0 else 0
        if ov["n"] >= 5 or nov["n"] >= 5:
            print(f"  {cat:<25} {ov_prec:>7.1f}% {ov['n']:>5} {nov_prec:>10.1f}% {nov['n']:>7} {ov_prec-nov_prec:>+7.1f}pp")

    # ===== PART 5: Specific drugs driving the anomaly =====
    print("\n" + "=" * 70)
    print("PART 5: Top Drugs in MEDIUM + Overlap (by prediction count)")
    print("=" * 70)

    drug_counts: Dict[str, Dict] = defaultdict(lambda: {"n": 0, "gt": 0, "targets": 0, "name": ""})
    for p in ov_medium:
        d = drug_counts[p["drug_id"]]
        d["n"] += 1
        d["gt"] += int(p["is_gt"])
        d["targets"] = p["drug_n_targets"]
        d["name"] = p["drug_name"]

    sorted_drugs = sorted(drug_counts.items(), key=lambda x: -x[1]["n"])
    print(f"\n  {'Drug':<35} {'n':>4} {'GT':>4} {'Prec':>6} {'Targets':>8}")
    for drug_id, stats in sorted_drugs[:25]:
        prec = stats["gt"] / stats["n"] * 100 if stats["n"] > 0 else 0
        print(f"  {stats['name'][:34]:<35} {stats['n']:>4} {stats['gt']:>4} {prec:>5.1f}% {stats['targets']:>8}")

    # ===== PART 6: Overlap count distribution =====
    print("\n" + "=" * 70)
    print("PART 6: Precision by Overlap Count (MEDIUM)")
    print("=" * 70)

    overlap_counts = Counter(p["overlap_count"] for p in ov_medium)
    print(f"\n  {'Overlap':>8} {'n':>6} {'GT':>5} {'Precision':>10}")
    for ov_count in sorted(overlap_counts.keys()):
        bucket = [p for p in ov_medium if p["overlap_count"] == ov_count]
        prec = sum(1 for p in bucket if p["is_gt"]) / len(bucket) * 100
        print(f"  {ov_count:>8} {len(bucket):>6} {sum(1 for p in bucket if p['is_gt']):>5} {prec:>9.1f}%")

    # ===== PART 7: Rule breakdown =====
    print("\n" + "=" * 70)
    print("PART 7: MEDIUM + Overlap by Assignment Rule")
    print("=" * 70)

    rule_stats: Dict[str, Dict] = defaultdict(lambda: {"n": 0, "gt": 0})
    for p in ov_medium:
        rule_stats[p["rule"]]["n"] += 1
        rule_stats[p["rule"]]["gt"] += int(p["is_gt"])

    sorted_rules = sorted(rule_stats.items(), key=lambda x: -x[1]["n"])
    print(f"\n  {'Rule':<45} {'n':>5} {'GT':>4} {'Precision':>10}")
    for rule, stats in sorted_rules:
        prec = stats["gt"] / stats["n"] * 100 if stats["n"] > 0 else 0
        print(f"  {rule[:44]:<45} {stats['n']:>5} {stats['gt']:>4} {prec:>9.1f}%")

    # Save summary
    output = {
        "hypothesis": "h391",
        "medium_overlap_n": len(ov_medium),
        "medium_no_overlap_n": len(no_ov_medium),
        "medium_overlap_prec": round(sum(1 for p in ov_medium if p["is_gt"]) / len(ov_medium) * 100, 2) if ov_medium else 0,
        "medium_no_overlap_prec": round(sum(1 for p in no_ov_medium if p["is_gt"]) / len(no_ov_medium) * 100, 2) if no_ov_medium else 0,
    }

    output_path = Path(__file__).parent.parent / "data" / "analysis" / "h391_medium_overlap.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
