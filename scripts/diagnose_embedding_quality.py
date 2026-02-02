#!/usr/bin/env python3
"""
Diagnose Embedding Quality.

Check for numerical issues in Node2Vec embeddings:
- NaN/Inf values
- Zero-norm vectors
- Extreme values that could cause overflow
- Disconnected nodes

Reports issues for both original and honest embeddings.
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"


def diagnose_embeddings(path: Path) -> dict[str, Any]:
    """Diagnose quality issues in embeddings file."""
    df = pd.read_csv(path)
    dim_cols = [c for c in df.columns if c.startswith("dim_")]
    embeddings = df[dim_cols].values.astype(np.float64)

    n_entities = len(df)
    n_dims = len(dim_cols)

    # Check for NaN values
    nan_mask = np.isnan(embeddings)
    nan_rows = np.where(nan_mask.any(axis=1))[0]
    nan_entities = df.iloc[nan_rows]["entity"].tolist() if len(nan_rows) > 0 else []

    # Check for Inf values
    inf_mask = np.isinf(embeddings)
    inf_rows = np.where(inf_mask.any(axis=1))[0]
    inf_entities = df.iloc[inf_rows]["entity"].tolist() if len(inf_rows) > 0 else []

    # Check for zero-norm vectors
    norms = np.linalg.norm(embeddings, axis=1)
    zero_norm_rows = np.where(norms == 0)[0]
    zero_norm_entities = df.iloc[zero_norm_rows]["entity"].tolist() if len(zero_norm_rows) > 0 else []

    # Check for near-zero norm vectors (potentially problematic)
    near_zero_threshold = 1e-10
    near_zero_rows = np.where((norms > 0) & (norms < near_zero_threshold))[0]
    near_zero_entities = df.iloc[near_zero_rows]["entity"].tolist() if len(near_zero_rows) > 0 else []

    # Check for extreme values
    max_val = np.nanmax(np.abs(embeddings))
    extreme_threshold = 1e6
    extreme_mask = np.abs(embeddings) > extreme_threshold
    extreme_rows = np.where(extreme_mask.any(axis=1))[0]
    extreme_entities = df.iloc[extreme_rows]["entity"].tolist() if len(extreme_rows) > 0 else []

    # Statistics
    valid_mask = ~nan_mask & ~inf_mask
    valid_embeddings = embeddings[valid_mask.all(axis=1)]

    if len(valid_embeddings) > 0:
        norm_stats = {
            "min": float(np.min(norms[norms > 0])) if np.any(norms > 0) else 0,
            "max": float(np.max(norms)),
            "mean": float(np.mean(norms)),
            "std": float(np.std(norms)),
        }
        value_stats = {
            "min": float(np.min(valid_embeddings)),
            "max": float(np.max(valid_embeddings)),
            "mean": float(np.mean(valid_embeddings)),
            "std": float(np.std(valid_embeddings)),
        }
    else:
        norm_stats = {"min": 0, "max": 0, "mean": 0, "std": 0}
        value_stats = {"min": 0, "max": 0, "mean": 0, "std": 0}

    # Count entities by type
    entity_types: dict[str, int] = {}
    for entity in df["entity"]:
        parts = entity.split("::")
        if len(parts) >= 1:
            etype = parts[0] if "::" not in entity else parts[0]
            entity_types[etype] = entity_types.get(etype, 0) + 1

    return {
        "path": str(path),
        "n_entities": n_entities,
        "n_dimensions": n_dims,
        "issues": {
            "nan_count": len(nan_rows),
            "nan_entities": nan_entities[:10],  # First 10
            "inf_count": len(inf_rows),
            "inf_entities": inf_entities[:10],
            "zero_norm_count": len(zero_norm_rows),
            "zero_norm_entities": zero_norm_entities[:10],
            "near_zero_norm_count": len(near_zero_rows),
            "near_zero_entities": near_zero_entities[:10],
            "extreme_value_count": len(extreme_rows),
            "extreme_entities": extreme_entities[:10],
        },
        "statistics": {
            "norms": norm_stats,
            "values": value_stats,
            "max_absolute_value": float(max_val),
        },
        "entity_type_counts": entity_types,
        "has_quality_issues": (
            len(nan_rows) > 0
            or len(inf_rows) > 0
            or len(zero_norm_rows) > 0
            or len(extreme_rows) > 0
        ),
    }


def main() -> None:
    print("=" * 70)
    print("EMBEDDING QUALITY DIAGNOSTICS")
    print("=" * 70)
    print()

    results: dict[str, Any] = {
        "analysis": "embedding_quality",
        "embeddings": {},
    }

    # Check both original and honest embeddings
    embedding_files = [
        ("original", EMBEDDINGS_DIR / "node2vec_256_named.csv"),
        ("honest", EMBEDDINGS_DIR / "node2vec_256_no_treatment.csv"),
    ]

    for label, path in embedding_files:
        if not path.exists():
            print(f"\n{label.upper()} EMBEDDINGS: NOT FOUND at {path}")
            continue

        print(f"\n{'-' * 70}")
        print(f"{label.upper()} EMBEDDINGS: {path.name}")
        print("-" * 70)

        report = diagnose_embeddings(path)
        results["embeddings"][label] = report

        print(f"  Total entities: {report['n_entities']:,}")
        print(f"  Dimensions: {report['n_dimensions']}")
        print()

        issues = report["issues"]
        print("  Quality Issues:")
        print(f"    NaN values:        {issues['nan_count']:,} entities")
        print(f"    Inf values:        {issues['inf_count']:,} entities")
        print(f"    Zero-norm vectors: {issues['zero_norm_count']:,} entities")
        print(f"    Near-zero norm:    {issues['near_zero_norm_count']:,} entities")
        print(f"    Extreme values:    {issues['extreme_value_count']:,} entities")
        print()

        stats = report["statistics"]
        print("  Embedding Statistics:")
        print(f"    Norm range: [{stats['norms']['min']:.4f}, {stats['norms']['max']:.4f}]")
        print(f"    Norm mean:  {stats['norms']['mean']:.4f} ± {stats['norms']['std']:.4f}")
        print(f"    Value range: [{stats['values']['min']:.4f}, {stats['values']['max']:.4f}]")
        print(f"    Max |value|: {stats['max_absolute_value']:.4f}")
        print()

        # Entity types
        print("  Entity Type Distribution:")
        for etype, count in sorted(report["entity_type_counts"].items(), key=lambda x: -x[1]):
            print(f"    {etype}: {count:,}")

        if report["has_quality_issues"]:
            print()
            print("  ⚠️  QUALITY ISSUES DETECTED")
            if issues["nan_count"] > 0:
                print(f"      NaN entities: {issues['nan_entities']}")
            if issues["inf_count"] > 0:
                print(f"      Inf entities: {issues['inf_entities']}")
            if issues["zero_norm_count"] > 0:
                print(f"      Zero-norm entities: {issues['zero_norm_entities']}")
        else:
            print("  ✓ No quality issues detected")

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    any_issues = False
    for label, report in results["embeddings"].items():
        if report["has_quality_issues"]:
            any_issues = True
            print(f"  {label}: ⚠️ Has quality issues")
        else:
            print(f"  {label}: ✓ Clean")

    if any_issues:
        results["recommendation"] = (
            "Filter out entities with NaN/Inf/zero-norm embeddings before evaluation. "
            "This may affect coverage metrics."
        )
        print(f"\nRecommendation: {results['recommendation']}")
    else:
        results["recommendation"] = (
            "No numerical issues detected. Embeddings are suitable for evaluation."
        )
        print(f"\n{results['recommendation']}")

    # Save
    output_path = ANALYSIS_DIR / "embedding_quality_report.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
