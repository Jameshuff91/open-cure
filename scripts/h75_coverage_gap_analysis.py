#!/usr/bin/env python3
"""
h75: Coverage Gap Analysis Between Use Cases

Analyzes characteristics of diseases at different confidence tiers:
- What makes a disease achieve HIGH confidence (clinical tier)?
- What distinguishes MEDIUM from LOW?
- Can we predict which diseases will be high-confidence?
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_predictions():
    """Load production predictions with confidence."""
    path = Path("data/deliverables/drug_repurposing_predictions_with_confidence.json")
    with open(path) as f:
        return json.load(f)


def analyze_by_disease(predictions):
    """Aggregate predictions by disease and compute disease-level stats."""
    disease_stats = defaultdict(lambda: {
        "predictions": [],
        "confidence_probs": [],
        "categories": set(),
        "pool_sizes": [],
        "neighbors_with_gt": [],
        "known_count": 0,
        "novel_count": 0
    })

    for pred in predictions:
        disease = pred["disease_name"]
        stats = disease_stats[disease]
        stats["predictions"].append(pred)
        stats["confidence_probs"].append(pred["confidence_prob"])
        stats["categories"].add(pred.get("category", "unknown"))
        stats["pool_sizes"].append(pred.get("pool_size", 0))
        stats["neighbors_with_gt"].append(pred.get("neighbors_with_gt", 0))
        if pred.get("is_known_indication"):
            stats["known_count"] += 1
        else:
            stats["novel_count"] += 1

    # Compute aggregated metrics
    disease_summary = []
    for disease, stats in disease_stats.items():
        summary = {
            "disease_name": disease,
            "category": list(stats["categories"])[0] if len(stats["categories"]) == 1 else "mixed",
            "num_predictions": len(stats["predictions"]),
            "num_novel": stats["novel_count"],
            "num_known": stats["known_count"],
            "mean_confidence": np.mean(stats["confidence_probs"]),
            "max_confidence": max(stats["confidence_probs"]),
            "min_confidence": min(stats["confidence_probs"]),
            "pool_size": stats["pool_sizes"][0] if stats["pool_sizes"] else 0,
            "neighbors_with_gt": stats["neighbors_with_gt"][0] if stats["neighbors_with_gt"] else 0,
            "high_conf_count": sum(1 for p in stats["predictions"] if p.get("confidence_tier") == "HIGH"),
            "medium_conf_count": sum(1 for p in stats["predictions"] if p.get("confidence_tier") == "MEDIUM"),
            "low_conf_count": sum(1 for p in stats["predictions"] if p.get("confidence_tier") == "LOW")
        }
        # Determine disease tier based on best prediction
        if summary["max_confidence"] >= 0.8:
            summary["disease_tier"] = "CLINICAL"
        elif summary["max_confidence"] >= 0.5:
            summary["disease_tier"] = "VALIDATION"
        elif summary["max_confidence"] >= 0.3:
            summary["disease_tier"] = "DISCOVERY"
        else:
            summary["disease_tier"] = "BELOW_THRESHOLD"

        disease_summary.append(summary)

    return disease_summary


def analyze_tier_characteristics(disease_summary):
    """Analyze what distinguishes diseases at each tier."""
    tiers = defaultdict(list)
    for d in disease_summary:
        tiers[d["disease_tier"]].append(d)

    print("=" * 70)
    print("h75: Coverage Gap Analysis Between Use Cases")
    print("=" * 70)

    print(f"\n{'Tier':<15} {'Count':<8} {'Avg Pool':<10} {'Avg Neighbors':<12} {'Avg Confidence':<15}")
    print("-" * 60)

    tier_order = ["CLINICAL", "VALIDATION", "DISCOVERY", "BELOW_THRESHOLD"]
    tier_stats = {}

    for tier in tier_order:
        diseases = tiers.get(tier, [])
        if not diseases:
            continue

        avg_pool = np.mean([d["pool_size"] for d in diseases])
        avg_neighbors = np.mean([d["neighbors_with_gt"] for d in diseases])
        avg_conf = np.mean([d["mean_confidence"] for d in diseases])

        print(f"{tier:<15} {len(diseases):<8} {avg_pool:<10.1f} {avg_neighbors:<12.1f} {avg_conf:<15.1%}")

        tier_stats[tier] = {
            "count": len(diseases),
            "diseases": diseases,
            "avg_pool_size": avg_pool,
            "avg_neighbors_with_gt": avg_neighbors,
            "avg_confidence": avg_conf
        }

    return tier_stats


def analyze_category_distribution(tier_stats):
    """Analyze category distribution within each tier."""
    print("\n" + "=" * 70)
    print("CATEGORY DISTRIBUTION BY TIER")
    print("=" * 70)

    categories = set()
    for tier, stats in tier_stats.items():
        for d in stats["diseases"]:
            categories.add(d["category"])

    categories = sorted(categories)

    for tier in ["CLINICAL", "VALIDATION", "DISCOVERY"]:
        if tier not in tier_stats:
            continue
        print(f"\n{tier}:")
        print("-" * 40)

        cat_counts = defaultdict(int)
        for d in tier_stats[tier]["diseases"]:
            cat_counts[d["category"]] += 1

        for cat in sorted(cat_counts.keys(), key=lambda x: cat_counts[x], reverse=True):
            pct = cat_counts[cat] / tier_stats[tier]["count"] * 100
            print(f"  {cat}: {cat_counts[cat]} ({pct:.1f}%)")


def identify_predictors(tier_stats):
    """Identify what predicts high-confidence diseases."""
    print("\n" + "=" * 70)
    print("PREDICTORS OF HIGH-CONFIDENCE DISEASES")
    print("=" * 70)

    if "CLINICAL" not in tier_stats:
        print("No CLINICAL tier diseases found.")
        return {}

    clinical = tier_stats["CLINICAL"]["diseases"]
    other = []
    for tier in ["VALIDATION", "DISCOVERY"]:
        if tier in tier_stats:
            other.extend(tier_stats[tier]["diseases"])

    if not other:
        print("No comparison group found.")
        return {}

    # Compare characteristics
    features = [
        ("pool_size", "Pool Size"),
        ("neighbors_with_gt", "Neighbors with GT"),
        ("num_predictions", "Num Predictions"),
        ("num_known", "Known Indications")
    ]

    print(f"\n{'Feature':<25} {'Clinical':<15} {'Other':<15} {'Diff':<10}")
    print("-" * 65)

    predictors = {}
    for key, label in features:
        clinical_mean = np.mean([d[key] for d in clinical])
        other_mean = np.mean([d[key] for d in other])
        diff = clinical_mean - other_mean
        pct_diff = diff / other_mean * 100 if other_mean > 0 else 0

        print(f"{label:<25} {clinical_mean:<15.1f} {other_mean:<15.1f} {pct_diff:+.1f}%")
        predictors[key] = {
            "clinical_mean": clinical_mean,
            "other_mean": other_mean,
            "pct_difference": pct_diff
        }

    # Category enrichment
    print("\nCategory Enrichment in CLINICAL tier:")
    print("-" * 40)

    clinical_cats = defaultdict(int)
    other_cats = defaultdict(int)
    for d in clinical:
        clinical_cats[d["category"]] += 1
    for d in other:
        other_cats[d["category"]] += 1

    total_clinical = len(clinical)
    total_other = len(other)

    enrichments = []
    for cat in set(clinical_cats.keys()) | set(other_cats.keys()):
        clinical_pct = clinical_cats[cat] / total_clinical if total_clinical > 0 else 0
        other_pct = other_cats[cat] / total_other if total_other > 0 else 0
        if other_pct > 0:
            enrichment = clinical_pct / other_pct
        else:
            enrichment = float('inf') if clinical_pct > 0 else 1.0
        enrichments.append((cat, clinical_pct, other_pct, enrichment))

    enrichments.sort(key=lambda x: x[3], reverse=True)
    for cat, c_pct, o_pct, enrich in enrichments:
        if enrich >= 1.5 or enrich <= 0.67:  # >50% enriched or depleted
            marker = "↑" if enrich > 1 else "↓"
            print(f"  {marker} {cat}: {enrich:.2f}x (clinical: {c_pct:.1%}, other: {o_pct:.1%})")

    return predictors


def list_clinical_diseases(tier_stats):
    """List all diseases in the CLINICAL tier."""
    print("\n" + "=" * 70)
    print("CLINICAL TIER DISEASES (100% Precision)")
    print("=" * 70)

    if "CLINICAL" not in tier_stats:
        print("No CLINICAL tier diseases.")
        return

    diseases = sorted(tier_stats["CLINICAL"]["diseases"],
                     key=lambda x: x["max_confidence"], reverse=True)

    print(f"\n{'Disease':<35} {'Category':<15} {'Pool':<8} {'Neighbors':<10} {'Max Conf':<10}")
    print("-" * 78)

    for d in diseases[:20]:  # Top 20
        print(f"{d['disease_name'][:34]:<35} {d['category']:<15} "
              f"{d['pool_size']:<8} {d['neighbors_with_gt']:<10} {d['max_confidence']:.1%}")

    if len(diseases) > 20:
        print(f"\n... and {len(diseases) - 20} more")


def main():
    # Load data
    predictions = load_predictions()
    print(f"Loaded {len(predictions)} predictions")

    # Aggregate by disease
    disease_summary = analyze_by_disease(predictions)
    print(f"Aggregated to {len(disease_summary)} diseases")

    # Analyze tier characteristics
    tier_stats = analyze_tier_characteristics(disease_summary)

    # Category distribution
    analyze_category_distribution(tier_stats)

    # Identify predictors
    predictors = identify_predictors(tier_stats)

    # List clinical diseases
    list_clinical_diseases(tier_stats)

    # Save results
    output = {
        "hypothesis": "h75",
        "title": "Coverage Gap Analysis Between Use Cases",
        "date": "2026-01-31",
        "summary": {
            tier: {
                "count": stats["count"],
                "avg_pool_size": stats["avg_pool_size"],
                "avg_neighbors_with_gt": stats["avg_neighbors_with_gt"],
                "avg_confidence": stats["avg_confidence"]
            }
            for tier, stats in tier_stats.items()
        },
        "predictors": predictors,
        "clinical_diseases": [
            {
                "disease": d["disease_name"],
                "category": d["category"],
                "max_confidence": d["max_confidence"],
                "pool_size": d["pool_size"],
                "neighbors_with_gt": d["neighbors_with_gt"]
            }
            for d in sorted(tier_stats.get("CLINICAL", {"diseases": []})["diseases"],
                          key=lambda x: x["max_confidence"], reverse=True)
        ]
    }

    output_path = Path("data/analysis/h75_coverage_gap_analysis.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✓ Results saved to {output_path}")

    return tier_stats, predictors


if __name__ == "__main__":
    main()
