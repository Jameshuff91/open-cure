#!/usr/bin/env python3
"""
h70: Threshold Optimization by Use Case

Analyzes h68 calibration data to find optimal thresholds for:
1. Discovery Mode - Maximize coverage, tolerate ~50-60% precision
2. Validation Mode - Balanced ~75-80% precision with good coverage
3. Clinical Mode - Maximize precision >90%, accept low coverage
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_h68_results():
    """Load h68 unified confidence results."""
    results_path = Path("data/analysis/h68_unified_confidence_results.json")
    with open(results_path) as f:
        return json.load(f)


def aggregate_calibration_across_seeds(h68_results):
    """
    Aggregate calibration data across all seeds for each scoring method.
    Returns: {method: {threshold: {"precision": mean, "n": total_n, "precisions": [list], "ns": [list]}}}
    """
    methods = ["prob_h65", "prob_h52", "prob_category",
               "combined_avg", "combined_harmonic", "combined_max", "combined_min"]

    aggregated = {}

    for method in methods:
        aggregated[method] = defaultdict(lambda: {"precisions": [], "ns": []})

        for seed_result in h68_results["all_seed_results"]:
            if method not in seed_result["metrics"]:
                continue

            calibration = seed_result["metrics"][method].get("calibration", {})
            for threshold_str, data in calibration.items():
                threshold = float(threshold_str)
                aggregated[method][threshold]["precisions"].append(data["precision"])
                aggregated[method][threshold]["ns"].append(data["n"])

        # Compute summary stats
        for threshold in list(aggregated[method].keys()):
            precisions = aggregated[method][threshold]["precisions"]
            ns = aggregated[method][threshold]["ns"]
            if precisions:
                aggregated[method][threshold]["precision_mean"] = np.mean(precisions)
                aggregated[method][threshold]["precision_std"] = np.std(precisions)
                aggregated[method][threshold]["n_mean"] = np.mean(ns)
                aggregated[method][threshold]["n_total"] = sum(ns)

    return aggregated


def find_optimal_thresholds(aggregated, method, use_case_targets):
    """
    Find the optimal threshold for a given method and use case.

    use_case_targets = {
        "discovery": {"min_precision": 0.50, "priority": "coverage"},
        "validation": {"min_precision": 0.75, "priority": "balance"},
        "clinical": {"min_precision": 0.90, "priority": "precision"}
    }
    """
    thresholds = sorted(aggregated[method].keys())
    results = {}

    for use_case, target in use_case_targets.items():
        min_prec = target["min_precision"]
        priority = target["priority"]

        best_threshold = None
        best_score = -1

        for threshold in thresholds:
            data = aggregated[method][threshold]
            prec = data["precision_mean"]
            n = data["n_mean"]

            if prec < min_prec:
                continue

            # Score based on priority
            if priority == "coverage":
                # Maximize n (coverage)
                score = n
            elif priority == "precision":
                # Maximize precision
                score = prec
            else:  # balance
                # Balance: precision * sqrt(coverage)
                score = prec * np.sqrt(n)

            if score > best_score:
                best_score = score
                best_threshold = threshold

        if best_threshold is not None:
            data = aggregated[method][best_threshold]
            results[use_case] = {
                "threshold": best_threshold,
                "precision": data["precision_mean"],
                "precision_std": data["precision_std"],
                "coverage": data["n_mean"],
                "score": best_score
            }
        else:
            # No threshold meets requirement
            results[use_case] = None

    return results


def analyze_precision_coverage_tradeoffs(aggregated, method):
    """Generate full precision-coverage curve for a method."""
    thresholds = sorted(aggregated[method].keys())
    curve = []

    for threshold in thresholds:
        data = aggregated[method][threshold]
        curve.append({
            "threshold": threshold,
            "precision": data["precision_mean"],
            "precision_std": data["precision_std"],
            "coverage": data["n_mean"],
            "n_seeds": len(data["precisions"])
        })

    return curve


def main():
    print("=" * 70)
    print("h70: Threshold Optimization by Use Case")
    print("=" * 70)

    # Load data
    h68_results = load_h68_results()
    print(f"\nLoaded h68 results with {len(h68_results['all_seed_results'])} seeds")

    # Aggregate calibration across seeds
    aggregated = aggregate_calibration_across_seeds(h68_results)

    # Define use case targets
    use_case_targets = {
        "discovery": {"min_precision": 0.50, "priority": "coverage"},
        "validation": {"min_precision": 0.75, "priority": "balance"},
        "clinical": {"min_precision": 0.90, "priority": "precision"}
    }

    # Analyze each method
    methods_to_analyze = ["prob_h52", "combined_avg", "combined_harmonic", "combined_min"]

    all_results = {}

    for method in methods_to_analyze:
        print(f"\n{'='*60}")
        print(f"Method: {method}")
        print("=" * 60)

        # Print precision-coverage curve
        print("\nPrecision-Coverage Curve (averaged across 5 seeds):")
        print("-" * 60)
        print(f"{'Threshold':>10} {'Precision':>12} {'± Std':>8} {'Coverage':>10}")
        print("-" * 60)

        curve = analyze_precision_coverage_tradeoffs(aggregated, method)
        for point in curve:
            print(f"{point['threshold']:>10.1f} {point['precision']:>12.1%} "
                  f"±{point['precision_std']:>6.1%} {point['coverage']:>10.1f}")

        # Find optimal thresholds for each use case
        print("\nOptimal Thresholds by Use Case:")
        print("-" * 60)

        optima = find_optimal_thresholds(aggregated, method, use_case_targets)
        all_results[method] = optima

        for use_case, target in use_case_targets.items():
            result = optima[use_case]
            if result:
                print(f"\n{use_case.upper()} (min precision: {target['min_precision']:.0%}):")
                print(f"  Threshold: {result['threshold']}")
                print(f"  Precision: {result['precision']:.1%} ± {result['precision_std']:.1%}")
                print(f"  Coverage: {result['coverage']:.1f} diseases")
            else:
                print(f"\n{use_case.upper()}: No threshold meets {target['min_precision']:.0%} precision requirement")

    # Summary comparison across methods
    print("\n" + "=" * 70)
    print("SUMMARY: Best Method for Each Use Case")
    print("=" * 70)

    for use_case in ["discovery", "validation", "clinical"]:
        print(f"\n{use_case.upper()}:")
        print("-" * 60)

        candidates = []
        for method in methods_to_analyze:
            result = all_results[method].get(use_case)
            if result:
                candidates.append((method, result))

        if not candidates:
            print("  No method meets requirements")
            continue

        # Sort by appropriate criteria
        if use_case == "discovery":
            candidates.sort(key=lambda x: x[1]["coverage"], reverse=True)
        elif use_case == "clinical":
            candidates.sort(key=lambda x: x[1]["precision"], reverse=True)
        else:  # validation - balance
            candidates.sort(key=lambda x: x[1]["score"], reverse=True)

        for i, (method, result) in enumerate(candidates):
            marker = "★" if i == 0 else " "
            print(f"  {marker} {method}:")
            print(f"      Threshold: {result['threshold']}, "
                  f"Precision: {result['precision']:.1%}, "
                  f"Coverage: {result['coverage']:.1f}")

    # Production recommendations
    print("\n" + "=" * 70)
    print("PRODUCTION RECOMMENDATIONS")
    print("=" * 70)

    recommendations = {}

    # Discovery: prioritize coverage
    discovery_best = None
    best_coverage = 0
    for method in methods_to_analyze:
        result = all_results[method].get("discovery")
        if result and result["coverage"] > best_coverage:
            best_coverage = result["coverage"]
            discovery_best = (method, result)

    if discovery_best:
        method, result = discovery_best
        print(f"\nDISCOVERY MODE:")
        print(f"  Method: {method}")
        print(f"  Threshold: {result['threshold']}")
        print(f"  Expected precision: {result['precision']:.1%}")
        print(f"  Expected coverage: {result['coverage']:.1f} diseases")
        recommendations["discovery"] = {
            "method": method,
            "threshold": result["threshold"],
            "expected_precision": result["precision"],
            "expected_coverage": result["coverage"]
        }

    # Validation: balance
    validation_best = None
    best_score = 0
    for method in methods_to_analyze:
        result = all_results[method].get("validation")
        if result and result["score"] > best_score:
            best_score = result["score"]
            validation_best = (method, result)

    if validation_best:
        method, result = validation_best
        print(f"\nVALIDATION MODE:")
        print(f"  Method: {method}")
        print(f"  Threshold: {result['threshold']}")
        print(f"  Expected precision: {result['precision']:.1%}")
        print(f"  Expected coverage: {result['coverage']:.1f} diseases")
        recommendations["validation"] = {
            "method": method,
            "threshold": result["threshold"],
            "expected_precision": result["precision"],
            "expected_coverage": result["coverage"]
        }

    # Clinical: prioritize precision
    clinical_best = None
    best_prec = 0
    for method in methods_to_analyze:
        result = all_results[method].get("clinical")
        if result and result["precision"] > best_prec:
            best_prec = result["precision"]
            clinical_best = (method, result)

    if clinical_best:
        method, result = clinical_best
        print(f"\nCLINICAL MODE:")
        print(f"  Method: {method}")
        print(f"  Threshold: {result['threshold']}")
        print(f"  Expected precision: {result['precision']:.1%}")
        print(f"  Expected coverage: {result['coverage']:.1f} diseases")
        recommendations["clinical"] = {
            "method": method,
            "threshold": result["threshold"],
            "expected_precision": result["precision"],
            "expected_coverage": result["coverage"]
        }

    # Save results
    output = {
        "hypothesis": "h70",
        "title": "Threshold Optimization by Use Case",
        "date": "2026-01-31",
        "use_case_targets": use_case_targets,
        "per_method_results": {
            method: {
                "curve": analyze_precision_coverage_tradeoffs(aggregated, method),
                "optimal_thresholds": all_results[method]
            }
            for method in methods_to_analyze
        },
        "recommendations": recommendations
    }

    output_path = Path("data/analysis/h70_threshold_optimization.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✓ Results saved to {output_path}")

    return recommendations


if __name__ == "__main__":
    recommendations = main()
