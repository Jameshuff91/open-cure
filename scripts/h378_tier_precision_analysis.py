#!/usr/bin/env python3
"""
h378: Tier Precision Analysis - Which Rules Hurt Precision?

Analyzes per-rule precision to identify rules that hurt overall precision.
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from production_predictor import DrugRepurposingPredictor, ConfidenceTier


def evaluate_rule_precision() -> Dict:
    """Evaluate precision for each category_specific_tier value."""
    print("Loading predictor...")
    predictor = DrugRepurposingPredictor()

    # Load expanded ground truth for validation
    gt_path = predictor.reference_dir / "expanded_ground_truth.json"
    with open(gt_path) as f:
        gt_data = json.load(f)

    # Build GT set for quick lookup
    gt_set = set()
    for disease_id, drugs in gt_data.items():
        for drug in drugs:
            if isinstance(drug, str):
                gt_set.add((disease_id, drug))
            elif isinstance(drug, dict):
                gt_set.add((disease_id, drug.get('drug_id') or drug.get('drug')))

    print(f"Loaded {len(gt_set)} GT pairs")

    # Track precision by rule source
    rule_stats: Dict[str, Dict] = defaultdict(lambda: {
        "tier": None,
        "hits": 0,
        "misses": 0,
        "examples": []
    })

    # Track tier-level stats
    tier_stats: Dict[str, Dict] = defaultdict(lambda: {"hits": 0, "misses": 0})

    # Generate predictions for sample of diseases
    diseases = list(predictor.disease_names.keys())
    print(f"Evaluating {min(200, len(diseases))} diseases...")

    for i, disease_id in enumerate(diseases[:200]):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/200 diseases...")

        disease_name = predictor.disease_names.get(disease_id, disease_id)

        try:
            result = predictor.predict(disease_name, top_n=30, include_filtered=True)
        except Exception as e:
            continue

        for pred in result.predictions:
            # Check if this prediction is in GT
            drug_id = pred.drug_id
            is_hit = (disease_id, drug_id) in gt_set

            # Get rule source
            rule = pred.category_specific_tier or "default"
            tier = pred.confidence_tier.name

            # Update rule stats
            rule_stats[rule]["tier"] = tier
            if is_hit:
                rule_stats[rule]["hits"] += 1
            else:
                rule_stats[rule]["misses"] += 1

            # Store example for debugging
            if len(rule_stats[rule]["examples"]) < 3:
                rule_stats[rule]["examples"].append({
                    "disease": disease_name,
                    "drug": pred.drug_name,
                    "is_hit": is_hit
                })

            # Update tier stats
            if is_hit:
                tier_stats[tier]["hits"] += 1
            else:
                tier_stats[tier]["misses"] += 1

    # Calculate precision for each rule
    rule_results = []
    for rule, stats in rule_stats.items():
        total = stats["hits"] + stats["misses"]
        precision = stats["hits"] / total if total > 0 else 0
        rule_results.append({
            "rule": rule,
            "tier": stats["tier"],
            "hits": stats["hits"],
            "misses": stats["misses"],
            "total": total,
            "precision": round(precision * 100, 1),
            "examples": stats["examples"]
        })

    # Calculate tier-level precision
    tier_results = {}
    for tier, stats in tier_stats.items():
        total = stats["hits"] + stats["misses"]
        precision = stats["hits"] / total if total > 0 else 0
        tier_results[tier] = {
            "hits": stats["hits"],
            "misses": stats["misses"],
            "total": total,
            "precision": round(precision * 100, 1)
        }

    # Sort rules by precision within tier
    rule_results.sort(key=lambda x: (x["tier"], -x["precision"]))

    return {
        "n_diseases": min(200, len(diseases)),
        "tier_summary": tier_results,
        "rule_details": rule_results
    }


def main():
    print("=" * 60)
    print("h378: Tier Precision Analysis")
    print("=" * 60)

    results = evaluate_rule_precision()

    # Save results
    output_path = Path(__file__).parent.parent / "data/analysis/h378_tier_precision.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("TIER-LEVEL PRECISION")
    print("=" * 60)

    tier_order = ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]
    for tier in tier_order:
        if tier in results["tier_summary"]:
            stats = results["tier_summary"][tier]
            print(f"  {tier}: {stats['precision']:.1f}% ({stats['hits']}/{stats['total']})")

    print("\n" + "=" * 60)
    print("RULE-LEVEL PRECISION (sorted by tier, then precision)")
    print("=" * 60)

    current_tier = None
    for rule_data in results["rule_details"]:
        if rule_data["total"] < 5:  # Skip rare rules
            continue

        tier = rule_data["tier"]
        if tier != current_tier:
            current_tier = tier
            tier_precision = results["tier_summary"].get(tier, {}).get("precision", 0)
            print(f"\n{tier} (tier avg: {tier_precision:.1f}%):")

        # Calculate delta from tier average
        tier_precision = results["tier_summary"].get(tier, {}).get("precision", 0)
        delta = rule_data["precision"] - tier_precision

        # Flag rules significantly below tier average
        flag = " ⚠️" if delta < -10 else ""

        print(f"  {rule_data['rule']}: {rule_data['precision']:.1f}% ({rule_data['hits']}/{rule_data['total']}) Δ={delta:+.1f}pp{flag}")

    # Identify problem rules
    print("\n" + "=" * 60)
    print("PROBLEM RULES (>10pp below tier average, n>=10)")
    print("=" * 60)

    problem_rules = []
    for rule_data in results["rule_details"]:
        if rule_data["total"] < 10:
            continue

        tier = rule_data["tier"]
        tier_precision = results["tier_summary"].get(tier, {}).get("precision", 0)
        delta = rule_data["precision"] - tier_precision

        if delta < -10:
            problem_rules.append({
                **rule_data,
                "delta": delta,
                "tier_avg": tier_precision
            })

    if problem_rules:
        problem_rules.sort(key=lambda x: x["delta"])
        for r in problem_rules:
            print(f"\n  {r['rule']} ({r['tier']})")
            print(f"    Rule precision: {r['precision']:.1f}%, Tier avg: {r['tier_avg']:.1f}%")
            print(f"    Delta: {r['delta']:+.1f}pp, Volume: {r['total']}")
            print(f"    Examples:")
            for ex in r["examples"]:
                status = "✓" if ex["is_hit"] else "✗"
                print(f"      {status} {ex['drug']} → {ex['disease']}")
    else:
        print("  No rules found >10pp below tier average")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
