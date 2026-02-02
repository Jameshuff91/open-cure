#!/usr/bin/env python3
"""
Statistical Significance Report for Honest Embedding Comparison.

Runs proper statistical tests on the original vs honest embedding comparison:
- Paired t-test (parametric)
- Wilcoxon signed-rank test (non-parametric alternative)
- Cohen's d effect size
- 95% bootstrap confidence intervals
- Mann-Whitney U test

Uses the 5-seed per-condition data from honest_embedding_comparison.json.
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> tuple[float, str]:
    """Calculate Cohen's d effect size for paired samples."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    d = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0.0

    # Interpretation
    abs_d = abs(d)
    if abs_d < 0.2:
        interp = "negligible"
    elif abs_d < 0.5:
        interp = "small"
    elif abs_d < 0.8:
        interp = "medium"
    else:
        interp = "large"

    return float(d), interp


def bootstrap_ci(
    data: np.ndarray,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42
) -> tuple[float, float]:
    """Calculate bootstrap confidence interval for the mean."""
    rng = np.random.RandomState(seed)
    n = len(data)
    bootstrap_means = np.array([
        np.mean(rng.choice(data, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = 1 - ci
    lower = float(np.percentile(bootstrap_means, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))
    return lower, upper


def bootstrap_diff_ci(
    group1: np.ndarray,
    group2: np.ndarray,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42
) -> tuple[float, float]:
    """Bootstrap CI for the difference of means (paired)."""
    rng = np.random.RandomState(seed)
    n = len(group1)
    diffs = group1 - group2
    bootstrap_diffs = np.array([
        np.mean(rng.choice(diffs, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = 1 - ci
    lower = float(np.percentile(bootstrap_diffs, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2)))
    return lower, upper


def main() -> None:
    input_path = ANALYSIS_DIR / "honest_embedding_comparison.json"

    if not input_path.exists():
        print(f"ERROR: {input_path} not found")
        print("Run compare_honest_embeddings.py first.")
        sys.exit(1)

    with open(input_path) as f:
        data = json.load(f)

    original_recalls = np.array(data["original"]["per_seed"])
    honest_recalls = np.array(data["honest"]["per_seed"])

    print("=" * 70)
    print("STATISTICAL SIGNIFICANCE REPORT")
    print("=" * 70)
    print()
    print("Data:")
    print(f"  Original (with treatment edges):  {original_recalls * 100}")
    print(f"  Honest (no treatment edges):      {honest_recalls * 100}")
    print()
    print(f"  Original mean: {np.mean(original_recalls)*100:.2f}%")
    print(f"  Honest mean:   {np.mean(honest_recalls)*100:.2f}%")
    print(f"  Difference:    {(np.mean(original_recalls) - np.mean(honest_recalls))*100:.2f} pp")
    print()

    # 1. Paired t-test
    t_stat, t_pvalue = stats.ttest_rel(original_recalls, honest_recalls)

    print("-" * 70)
    print("1. PAIRED T-TEST (parametric)")
    print("-" * 70)
    print(f"   t-statistic: {t_stat:.4f}")
    print(f"   p-value:     {t_pvalue:.6f}")
    print(f"   Significant at α=0.05: {'YES' if t_pvalue < 0.05 else 'NO'}")
    print()

    # 2. Wilcoxon signed-rank test (non-parametric)
    try:
        w_stat, w_pvalue = stats.wilcoxon(original_recalls, honest_recalls)
        wilcoxon_valid = True
    except ValueError as e:
        # Wilcoxon requires at least 5 samples and non-identical pairs
        wilcoxon_valid = False
        w_stat, w_pvalue = float("nan"), float("nan")
        print(f"   (Wilcoxon test failed: {e})")

    print("-" * 70)
    print("2. WILCOXON SIGNED-RANK TEST (non-parametric)")
    print("-" * 70)
    if wilcoxon_valid:
        print(f"   W-statistic: {w_stat:.4f}")
        print(f"   p-value:     {w_pvalue:.6f}")
        print(f"   Significant at α=0.05: {'YES' if w_pvalue < 0.05 else 'NO'}")
    else:
        print("   Test not applicable (insufficient non-zero differences)")
    print()

    # 3. Mann-Whitney U test (independent samples, for reference)
    u_stat, u_pvalue = stats.mannwhitneyu(
        original_recalls, honest_recalls, alternative="two-sided"
    )

    print("-" * 70)
    print("3. MANN-WHITNEY U TEST (independent samples, for reference)")
    print("-" * 70)
    print(f"   U-statistic: {u_stat:.4f}")
    print(f"   p-value:     {u_pvalue:.6f}")
    print(f"   Significant at α=0.05: {'YES' if u_pvalue < 0.05 else 'NO'}")
    print()

    # 4. Cohen's d effect size
    d, d_interp = cohens_d(original_recalls, honest_recalls)

    print("-" * 70)
    print("4. EFFECT SIZE (Cohen's d)")
    print("-" * 70)
    print(f"   Cohen's d:      {d:.4f}")
    print(f"   Interpretation: {d_interp}")
    print("   (|d| < 0.2: negligible, 0.2-0.5: small, 0.5-0.8: medium, >0.8: large)")
    print()

    # 5. Bootstrap confidence intervals
    orig_ci = bootstrap_ci(original_recalls)
    honest_ci = bootstrap_ci(honest_recalls)
    diff_ci = bootstrap_diff_ci(original_recalls, honest_recalls)

    print("-" * 70)
    print("5. BOOTSTRAP 95% CONFIDENCE INTERVALS (10,000 samples)")
    print("-" * 70)
    print(f"   Original mean: [{orig_ci[0]*100:.2f}%, {orig_ci[1]*100:.2f}%]")
    print(f"   Honest mean:   [{honest_ci[0]*100:.2f}%, {honest_ci[1]*100:.2f}%]")
    print(f"   Difference:    [{diff_ci[0]*100:.2f}, {diff_ci[1]*100:.2f}] pp")
    print(f"   CI excludes 0: {'YES' if (diff_ci[0] > 0 or diff_ci[1] < 0) else 'NO'}")
    print()

    # Summary
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    is_significant = t_pvalue < 0.05
    ci_excludes_zero = diff_ci[0] > 0 or diff_ci[1] < 0

    if is_significant and ci_excludes_zero:
        conclusion = (
            f"The {(np.mean(original_recalls) - np.mean(honest_recalls))*100:.1f} pp "
            f"difference between original ({np.mean(original_recalls)*100:.2f}%) and "
            f"honest ({np.mean(honest_recalls)*100:.2f}%) embeddings is "
            f"STATISTICALLY SIGNIFICANT (paired t-test p={t_pvalue:.4f}, "
            f"Cohen's d={d:.2f} [{d_interp}])."
        )
    else:
        conclusion = (
            f"The {(np.mean(original_recalls) - np.mean(honest_recalls))*100:.1f} pp "
            f"difference is NOT statistically significant at α=0.05 "
            f"(p={t_pvalue:.4f})."
        )

    print(f"\n{conclusion}\n")

    # Save results
    results: dict[str, Any] = {
        "analysis": "statistical_significance",
        "description": "Statistical tests comparing original vs honest (no-treatment) embeddings",
        "data": {
            "original_per_seed": list(original_recalls),
            "honest_per_seed": list(honest_recalls),
            "original_mean": float(np.mean(original_recalls)),
            "honest_mean": float(np.mean(honest_recalls)),
            "difference_pp": float((np.mean(original_recalls) - np.mean(honest_recalls)) * 100),
        },
        "tests": {
            "paired_t_test": {
                "t_statistic": float(t_stat),
                "p_value": float(t_pvalue),
                "significant_at_0.05": bool(t_pvalue < 0.05),
            },
            "wilcoxon_signed_rank": {
                "valid": wilcoxon_valid,
                "w_statistic": float(w_stat) if wilcoxon_valid else None,
                "p_value": float(w_pvalue) if wilcoxon_valid else None,
                "significant_at_0.05": bool(w_pvalue < 0.05) if wilcoxon_valid else None,
            },
            "mann_whitney_u": {
                "u_statistic": float(u_stat),
                "p_value": float(u_pvalue),
                "significant_at_0.05": bool(u_pvalue < 0.05),
            },
        },
        "effect_size": {
            "cohens_d": float(d),
            "interpretation": d_interp,
        },
        "confidence_intervals_95": {
            "original": [float(orig_ci[0]), float(orig_ci[1])],
            "honest": [float(honest_ci[0]), float(honest_ci[1])],
            "difference": [float(diff_ci[0]), float(diff_ci[1])],
            "ci_excludes_zero": ci_excludes_zero,
        },
        "conclusion": conclusion,
    }

    output_path = ANALYSIS_DIR / "statistical_significance.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
