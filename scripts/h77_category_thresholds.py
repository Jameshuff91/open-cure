#!/usr/bin/env python3
"""
h77: Category-Specific Confidence Thresholds

Since autoimmune achieves clinical precision at current thresholds but cancer/metabolic don't,
test if category-specific thresholds can expand coverage while maintaining precision.
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


def calculate_per_category_precision_curve(predictions):
    """
    Calculate precision at each threshold for each category.

    Note: Since we don't have true labels in production predictions,
    we'll use is_known_indication as a proxy for "correct" predictions.
    This is a lower bound since novel predictions may also be correct.
    """
    # Group by category
    by_category = defaultdict(list)
    for pred in predictions:
        cat = pred.get("category", "unknown")
        by_category[cat].append(pred)

    # Thresholds to evaluate
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    category_curves = {}

    for cat, preds in by_category.items():
        curve = []
        for thresh in thresholds:
            # Filter to predictions above threshold
            above_thresh = [p for p in preds if p["confidence_prob"] >= thresh]

            if not above_thresh:
                continue

            # Calculate precision using known indications
            # This is a lower bound - some "novel" predictions may be correct
            n_total = len(above_thresh)
            n_known = sum(1 for p in above_thresh if p.get("is_known_indication", False))

            # Precision = known / total (lower bound)
            precision = n_known / n_total if n_total > 0 else 0

            curve.append({
                "threshold": thresh,
                "precision_lb": precision,  # Lower bound
                "n_predictions": n_total,
                "n_known": n_known,
                "n_novel": n_total - n_known
            })

        category_curves[cat] = curve

    return category_curves


def find_optimal_thresholds(category_curves, target_precision=0.90):
    """
    Find the minimum threshold to achieve target precision for each category.
    """
    results = {}

    for cat, curve in category_curves.items():
        # Sort by threshold ascending
        curve_sorted = sorted(curve, key=lambda x: x["threshold"])

        # Find first threshold that achieves target precision
        best = None
        for point in curve_sorted:
            if point["precision_lb"] >= target_precision and point["n_predictions"] >= 5:
                best = point
                break

        if best:
            results[cat] = {
                "threshold": best["threshold"],
                "precision": best["precision_lb"],
                "coverage": best["n_predictions"],
                "achieves_target": True
            }
        else:
            # Check max precision achieved
            if curve_sorted:
                max_prec = max(p["precision_lb"] for p in curve_sorted)
                results[cat] = {
                    "threshold": None,
                    "max_precision": max_prec,
                    "achieves_target": False
                }
            else:
                results[cat] = {
                    "threshold": None,
                    "max_precision": 0,
                    "achieves_target": False
                }

    return results


def compare_global_vs_category_thresholds(predictions, category_curves, target_precision=0.90):
    """
    Compare coverage using global threshold vs category-specific thresholds.
    """
    # Global approach: single threshold for all
    # We need the h68 data for this, but we can estimate from overall precision

    # Category-specific approach
    category_thresholds = find_optimal_thresholds(category_curves, target_precision)

    print("\n" + "=" * 70)
    print(f"CATEGORY-SPECIFIC THRESHOLDS FOR {target_precision:.0%} PRECISION")
    print("=" * 70)

    print(f"\n{'Category':<20} {'Threshold':<12} {'Precision':<12} {'Coverage':<12}")
    print("-" * 60)

    total_coverage = 0
    categories_that_achieve = []
    categories_that_dont = []

    for cat, result in sorted(category_thresholds.items(), key=lambda x: x[1].get("threshold") or 1.0):
        if result["achieves_target"]:
            print(f"{cat:<20} {result['threshold']:<12.1f} {result['precision']:<12.1%} {result['coverage']:<12}")
            total_coverage += result["coverage"]
            categories_that_achieve.append(cat)
        else:
            print(f"{cat:<20} {'N/A':<12} {result['max_precision']:<12.1%} {'0':<12}")
            categories_that_dont.append(cat)

    print("-" * 60)
    print(f"{'TOTAL':<20} {'':<12} {'':<12} {total_coverage:<12}")

    print(f"\nCategories achieving {target_precision:.0%}: {len(categories_that_achieve)}")
    print(f"Categories NOT achieving: {len(categories_that_dont)}")
    if categories_that_dont:
        print(f"  {', '.join(categories_that_dont)}")

    return category_thresholds, total_coverage


def main():
    print("=" * 70)
    print("h77: Category-Specific Confidence Thresholds")
    print("=" * 70)

    # Load data
    predictions = load_predictions()
    print(f"\nLoaded {len(predictions)} predictions")

    # Calculate per-category precision curves
    category_curves = calculate_per_category_precision_curve(predictions)

    print("\n" + "=" * 70)
    print("PRECISION CURVES BY CATEGORY (Known indication = correct)")
    print("=" * 70)

    for cat in sorted(category_curves.keys()):
        curve = category_curves[cat]
        if not curve:
            continue

        print(f"\n{cat.upper()}:")
        print(f"  {'Threshold':<10} {'Precision':<12} {'N':<10} {'Known':<10}")
        print("  " + "-" * 42)

        for point in curve:
            print(f"  {point['threshold']:<10.1f} {point['precision_lb']:<12.1%} "
                  f"{point['n_predictions']:<10} {point['n_known']:<10}")

    # Find optimal thresholds for different precision targets
    for target in [0.90, 0.80, 0.75]:
        thresholds, coverage = compare_global_vs_category_thresholds(
            predictions, category_curves, target
        )

    # Compare to global threshold (from h70)
    print("\n" + "=" * 70)
    print("COMPARISON: GLOBAL vs CATEGORY-SPECIFIC THRESHOLDS")
    print("=" * 70)

    print("""
From h70, global threshold for ~90% precision:
  - combined_avg @ 0.8: 100% precision, 5 diseases (150 predictions)

With category-specific thresholds at 90% target:""")

    thresholds_90, coverage_90 = compare_global_vs_category_thresholds(
        predictions, category_curves, 0.90
    )

    # Calculate coverage gain
    global_coverage = 150  # From h70 (5 diseases × ~30 predictions each)

    print(f"\n{'Approach':<30} {'Coverage (predictions)':<25}")
    print("-" * 55)
    print(f"{'Global (combined_avg @ 0.8)':<30} {global_coverage:<25}")
    print(f"{'Category-specific @ 90%':<30} {coverage_90:<25}")

    if coverage_90 > global_coverage:
        gain = coverage_90 / global_coverage
        print(f"\nCoverage gain: {gain:.1f}x ({coverage_90 - global_coverage:+} predictions)")
    else:
        print(f"\nNo coverage gain (category-specific is not better)")

    # Save results
    output = {
        "hypothesis": "h77",
        "title": "Category-Specific Confidence Thresholds",
        "date": "2026-01-31",
        "category_curves": {
            cat: curve for cat, curve in category_curves.items()
        },
        "thresholds_for_90_pct": thresholds_90,
        "comparison": {
            "global_approach": {
                "method": "combined_avg @ 0.8",
                "coverage_predictions": global_coverage,
                "precision": 1.0
            },
            "category_specific": {
                "coverage_predictions": coverage_90,
                "target_precision": 0.90
            }
        }
    }

    output_path = Path("data/analysis/h77_category_thresholds.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✓ Results saved to {output_path}")

    return output


if __name__ == "__main__":
    main()
