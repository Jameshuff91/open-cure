#!/usr/bin/env python3
"""
h71: Per-Category Calibration

Analyzes the h68 unified confidence results stratified by disease category.
This uses proper held-out evaluation (from h68) to determine per-category precision.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_h68_results():
    """Load h68 unified confidence evaluation results (with per-disease outcomes)."""
    # The full h68 results include per-disease results which we need to stratify
    # We need to re-run the evaluation with category tracking
    # For now, let's check if we have saved per-disease results

    results_path = Path("data/analysis/h68_unified_confidence_results.json")
    with open(results_path) as f:
        data = json.load(f)

    return data


def check_if_per_disease_results_available():
    """
    The h68 results file only contains aggregated metrics, not per-disease results.
    We need to re-run the evaluation to get per-disease category info.
    """
    results_path = Path("data/analysis/h68_unified_confidence_results.json")
    with open(results_path) as f:
        data = json.load(f)

    # Check if per-disease results are stored
    if "all_seed_results" in data:
        first_seed = data["all_seed_results"][0]
        # Check if results include disease-level data
        if "results" in first_seed:
            return True
        else:
            print("Per-disease results not stored in h68 output")
            return False
    return False


def main():
    print("=" * 70)
    print("h71: Per-Category Calibration Analysis")
    print("=" * 70)

    # Check if we have per-disease results
    if not check_if_per_disease_results_available():
        print("""

h68 output only contains aggregated metrics, not per-disease results.

To get per-category calibration, we need to either:
1. Re-run h68 evaluation with category tracking
2. Modify h68 script to save per-disease results

For now, let's use the production predictions which DO have category info,
and approximate calibration using the known-indication lower bound.

WARNING: This is an approximation. True calibration requires held-out GT.
        """)

        # Fall back to using category priors and confidence model separately
        analyze_with_approximation()
    else:
        print("Per-disease results available. Proceeding with proper calibration.")


def analyze_with_approximation():
    """
    Use production predictions to approximate per-category calibration.
    This is a lower bound since not all correct predictions are "known indications".
    """
    predictions_path = Path("data/deliverables/drug_repurposing_predictions_with_confidence.json")
    with open(predictions_path) as f:
        predictions = json.load(f)

    print(f"\nLoaded {len(predictions)} predictions")

    # Group by category
    by_category = defaultdict(list)
    for pred in predictions:
        cat = pred.get("category", "unknown")
        by_category[cat].append(pred)

    print(f"\n{'Category':<20} {'N Preds':<10} {'HIGH':<10} {'MEDIUM':<10} {'LOW':<10}")
    print("-" * 60)

    for cat in sorted(by_category.keys()):
        preds = by_category[cat]
        n_total = len(preds)
        n_high = sum(1 for p in preds if p.get("confidence_tier") == "HIGH")
        n_medium = sum(1 for p in preds if p.get("confidence_tier") == "MEDIUM")
        n_low = sum(1 for p in preds if p.get("confidence_tier") == "LOW")
        print(f"{cat:<20} {n_total:<10} {n_high:<10} {n_medium:<10} {n_low:<10}")

    # Check if we should run proper held-out evaluation
    print("\n" + "=" * 70)
    print("RECOMMENDATION: Run Proper Per-Category Calibration")
    print("=" * 70)
    print("""
To get TRUE per-category calibration, we should:

1. Modify evaluate_unified_confidence.py to save per-disease results
2. Re-run with category tracking
3. Compute calibration curves stratified by category

This would tell us, e.g.:
- Autoimmune @ 0.7 threshold: 95% precision, 15 diseases
- Cancer @ 0.7 threshold: 60% precision, 8 diseases
- etc.

From this, we can set category-specific thresholds to achieve uniform
precision (e.g., 90%) while maximizing per-category coverage.
    """)


if __name__ == "__main__":
    main()
