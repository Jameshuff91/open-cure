#!/usr/bin/env python3
"""
h432: Consolidate Small Hierarchy Rules into Generic Category Groups

h402 found 21 rules too small to validate (1-2 diseases each), creating
code complexity without measurable benefit. Test whether we can consolidate
individual hierarchy groups (e.g., scleroderma, lupus, spondylitis) into
generic category-level rules (e.g., "autoimmune hierarchy match").

Method:
1. List all current hierarchy groups and their disease counts
2. For each category, compute precision at category level vs individual group level
3. If category-level precision is similar to individual group precision,
   consolidation is safe
4. Count how many rules could be removed
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from production_predictor import (
    DrugRepurposingPredictor,
    DISEASE_HIERARCHY_GROUPS,
)


def main():
    print("=" * 70)
    print("h432: Consolidate Small Hierarchy Rules")
    print("=" * 70)

    print("\nLoading predictor...")
    t0 = time.time()
    predictor = DrugRepurposingPredictor()
    print(f"Loaded in {time.time() - t0:.1f}s")

    diseases_with_gt = [d for d in predictor.ground_truth if d in predictor.embeddings]

    # ===== PART 1: Current hierarchy group inventory =====
    print("\n" + "=" * 70)
    print("PART 1: Hierarchy Group Inventory")
    print("=" * 70)

    # Count diseases per hierarchy group
    group_diseases: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    # group_diseases[category][group_name] = [disease_ids]

    for disease_id in diseases_with_gt:
        disease_name = predictor.disease_names.get(disease_id, disease_id).lower()
        category = predictor.categorize_disease(disease_name)
        if category in DISEASE_HIERARCHY_GROUPS:
            for group_name, keywords in DISEASE_HIERARCHY_GROUPS[category].items():
                if any(kw in disease_name for kw in keywords):
                    group_diseases[category][group_name].append(disease_id)

    total_groups = 0
    small_groups = 0  # <=2 diseases
    print(f"\n  {'Category':<20} {'Group':<30} {'Diseases':>8}")
    print(f"  {'-'*20} {'-'*30} {'-'*8}")

    for category in sorted(group_diseases.keys()):
        for group_name in sorted(group_diseases[category].keys()):
            diseases = group_diseases[category][group_name]
            total_groups += 1
            if len(diseases) <= 2:
                small_groups += 1
                marker = " *** SMALL"
            else:
                marker = ""
            print(f"  {category:<20} {group_name:<30} {len(diseases):>8}{marker}")

    print(f"\n  Total groups: {total_groups}")
    print(f"  Small (<=2 diseases): {small_groups} ({small_groups/total_groups*100:.0f}%)")

    # ===== PART 2: Precision at individual vs category level =====
    print("\n" + "=" * 70)
    print("PART 2: Individual Group vs Category-Level Precision")
    print("=" * 70)

    # Collect all predictions with hierarchy match info
    all_preds = []
    for disease_id in diseases_with_gt:
        if disease_id not in predictor.embeddings:
            continue
        gt_drugs = predictor.ground_truth.get(disease_id, set())
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        result = predictor.predict(disease_name, k=20, top_n=30, include_filtered=True)

        for pred in result.predictions:
            rule = pred.category_specific_tier or "default"
            all_preds.append({
                "disease_id": disease_id,
                "drug_id": pred.drug_id,
                "rank": pred.rank,
                "tier": pred.confidence_tier.value,
                "rule": rule,
                "is_gt": pred.drug_id in gt_drugs,
                "category": result.category,
            })

    print(f"\n  Total predictions: {len(all_preds)}")

    # Group by hierarchy rule
    hierarchy_preds = [p for p in all_preds if "hierarchy" in p["rule"]]
    print(f"  Hierarchy-matched predictions: {len(hierarchy_preds)}")

    # Individual group precision
    rule_stats: Dict[str, Dict] = defaultdict(lambda: {"n": 0, "gt": 0})
    for p in hierarchy_preds:
        rule_stats[p["rule"]]["n"] += 1
        rule_stats[p["rule"]]["gt"] += int(p["is_gt"])

    # Category-level precision (all hierarchy rules in same category combined)
    # Extract category from rule name: "autoimmune_hierarchy_lupus" → "autoimmune"
    cat_stats: Dict[str, Dict] = defaultdict(lambda: {"n": 0, "gt": 0})
    for p in hierarchy_preds:
        # Parse category from rule
        rule = p["rule"]
        for cat in DISEASE_HIERARCHY_GROUPS.keys():
            if rule.startswith(f"{cat}_hierarchy"):
                cat_stats[cat]["n"] += 1
                cat_stats[cat]["gt"] += int(p["is_gt"])
                break

    print(f"\n  {'Rule':<50} {'n':>5} {'GT':>4} {'Prec':>7}")
    print(f"  {'-'*50} {'-'*5} {'-'*4} {'-'*7}")

    sorted_rules = sorted(rule_stats.items(), key=lambda x: -x[1]["n"])
    for rule, stats in sorted_rules:
        prec = stats["gt"] / stats["n"] * 100 if stats["n"] > 0 else 0
        print(f"  {rule[:49]:<50} {stats['n']:>5} {stats['gt']:>4} {prec:>6.1f}%")

    print(f"\n  {'Category-Level Rule':<50} {'n':>5} {'GT':>4} {'Prec':>7}")
    print(f"  {'-'*50} {'-'*5} {'-'*4} {'-'*7}")
    for cat in sorted(cat_stats.keys()):
        stats = cat_stats[cat]
        prec = stats["gt"] / stats["n"] * 100 if stats["n"] > 0 else 0
        n_groups = len(group_diseases.get(cat, {}))
        print(f"  {cat} (all {n_groups} groups)<50 {stats['n']:>5} {stats['gt']:>4} {prec:>6.1f}%")

    # ===== PART 3: Consolidation analysis =====
    print("\n" + "=" * 70)
    print("PART 3: Consolidation Impact Analysis")
    print("=" * 70)

    # For each category, if we consolidated all small groups:
    # Would the category-level precision still meet tier threshold?
    print("\n  Question: Can we consolidate small groups (<=2 diseases) without losing precision?")
    print()

    for cat in sorted(group_diseases.keys()):
        groups = group_diseases[cat]
        small = {g: d for g, d in groups.items() if len(d) <= 2}
        large = {g: d for g, d in groups.items() if len(d) > 2}

        if not small:
            continue

        # Get predictions for small vs large groups
        small_group_rules = {f"{cat}_hierarchy_{g}" for g in small.keys()}
        large_group_rules = {f"{cat}_hierarchy_{g}" for g in large.keys()}

        small_preds = [p for p in hierarchy_preds if p["rule"] in small_group_rules]
        large_preds = [p for p in hierarchy_preds if p["rule"] in large_group_rules]

        small_prec = sum(1 for p in small_preds if p["is_gt"]) / len(small_preds) * 100 if small_preds else 0
        large_prec = sum(1 for p in large_preds if p["is_gt"]) / len(large_preds) * 100 if large_preds else 0

        print(f"  --- {cat} ---")
        print(f"    Large groups (>{2} diseases): {len(large)} groups, {len(large_preds)} preds, {large_prec:.1f}%")
        print(f"    Small groups (<=2 diseases): {len(small)} groups, {len(small_preds)} preds, {small_prec:.1f}%")
        for g in sorted(small.keys()):
            rule = f"{cat}_hierarchy_{g}"
            rs = rule_stats.get(rule, {"n": 0, "gt": 0})
            prec = rs["gt"] / rs["n"] * 100 if rs["n"] > 0 else 0
            print(f"      {g}: {len(small[g])} diseases, {rs['n']} preds, {prec:.1f}%")
        print()

    # ===== PART 4: Summary =====
    print("\n" + "=" * 70)
    print("PART 4: Summary")
    print("=" * 70)

    print(f"\n  Total hierarchy groups: {total_groups}")
    print(f"  Small (<=2 diseases): {small_groups}")
    print(f"  These small groups generate {sum(rule_stats[r]['n'] for r in rule_stats if any(f'_{g}' in r for cat, groups in group_diseases.items() for g, d in groups.items() if len(d) <= 2))} predictions")

    # Check if any small group has 0% precision
    zero_prec_groups = []
    for rule, stats in rule_stats.items():
        if stats["n"] > 0 and stats["gt"] == 0:
            # Check if this is a small group
            for cat, groups in group_diseases.items():
                for g, diseases in groups.items():
                    if rule == f"{cat}_hierarchy_{g}" and len(diseases) <= 2:
                        zero_prec_groups.append((rule, stats["n"]))

    if zero_prec_groups:
        print(f"\n  Zero-precision small groups:")
        for rule, n in zero_prec_groups:
            print(f"    {rule}: {n} predictions, 0% precision")

    # Save
    output = {
        "hypothesis": "h432",
        "total_groups": total_groups,
        "small_groups": small_groups,
    }
    output_path = Path(__file__).parent.parent / "data" / "analysis" / "h432_consolidation.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
