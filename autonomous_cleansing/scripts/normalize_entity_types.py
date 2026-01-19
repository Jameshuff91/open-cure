#!/usr/bin/env python3
"""
Normalize entity types in the Open Cure knowledge graph.

This script merges duplicate entity types (e.g., Gene/gene/protein, Drug/drug/Compound)
into consistent PascalCase naming.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import Counter


# Entity type mapping: source type -> normalized type
# Based on analysis of unified_nodes.csv
TYPE_MAPPING = {
    # Gene types
    "Gene": "Gene",
    "gene": "Gene",
    "gene/protein": "Gene",

    # Drug types
    "Drug": "Drug",
    "drug": "Drug",
    "Compound": "Drug",
    "compound": "Drug",

    # Disease types
    "Disease": "Disease",
    "disease": "Disease",

    # Biological Process types
    "BiologicalProcess": "BiologicalProcess",
    "biological_process": "BiologicalProcess",
    "bioprocess": "BiologicalProcess",

    # Molecular Function types
    "MolecularFunction": "MolecularFunction",
    "molecular_function": "MolecularFunction",
    "molfunc": "MolecularFunction",

    # Cellular Component types
    "CellularComponent": "CellularComponent",
    "cellular_component": "CellularComponent",
    "cellcomp": "CellularComponent",

    # Pathway types
    "Pathway": "Pathway",
    "pathway": "Pathway",

    # Anatomy types
    "Anatomy": "Anatomy",
    "anatomy": "Anatomy",

    # Side Effect types
    "SideEffect": "SideEffect",
    "side_effect": "SideEffect",
    "effect/phenotype": "Phenotype",

    # Other types (keep as-is but normalize case)
    "DrugClass": "DrugClass",
    "drugclass": "DrugClass",
    "Symptom": "Symptom",
    "symptom": "Symptom",
    "exposure": "Exposure",
    "Exposure": "Exposure",
    "Tax": "Taxonomy",
    "tax": "Taxonomy",
}


def normalize_types(input_path: Path, output_path: Path, log_path: Path) -> dict:
    """
    Normalize entity types in the nodes file.

    Args:
        input_path: Path to the input CSV file
        output_path: Path to save the normalized CSV
        log_path: Path to save the transformation log

    Returns:
        Dictionary with transformation statistics
    """
    print(f"Reading nodes from {input_path}...")
    df = pd.read_csv(input_path, low_memory=False)

    original_count = len(df)
    original_types = df["type"].value_counts().to_dict()

    print(f"Original node count: {original_count:,}")
    print(f"Original unique types: {len(original_types)}")

    # Track transformations
    transformations = []
    type_changes = Counter()

    # Apply normalization
    print("\nApplying type normalization...")

    def normalize_type(t: str) -> str:
        """Normalize a single type value."""
        if pd.isna(t):
            return t

        t_str = str(t).strip()
        if t_str in TYPE_MAPPING:
            normalized = TYPE_MAPPING[t_str]
            if t_str != normalized:
                type_changes[(t_str, normalized)] += 1
            return normalized
        else:
            # Warn about unmapped types
            print(f"  WARNING: Unmapped type '{t_str}'")
            return t_str

    df["type"] = df["type"].apply(normalize_type)

    # Verify counts
    final_count = len(df)
    final_types = df["type"].value_counts().to_dict()

    print(f"\nFinal node count: {final_count:,}")
    print(f"Final unique types: {len(final_types)}")

    # Validation
    if final_count != original_count:
        raise ValueError(f"Node count changed! {original_count} -> {final_count}")

    # Log transformations
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "task": "normalize_entity_types",
        "input_file": str(input_path),
        "output_file": str(output_path),
        "statistics": {
            "original_node_count": original_count,
            "final_node_count": final_count,
            "original_unique_types": len(original_types),
            "final_unique_types": len(final_types),
            "types_merged": len(original_types) - len(final_types)
        },
        "original_type_counts": original_types,
        "final_type_counts": final_types,
        "type_mapping_applied": TYPE_MAPPING,
        "transformations": [
            {"from": k[0], "to": k[1], "count": v}
            for k, v in sorted(type_changes.items(), key=lambda x: -x[1])
        ]
    }

    # Save log
    print(f"\nSaving transformation log to {log_path}...")
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)

    # Save normalized data
    print(f"Saving normalized nodes to {output_path}...")
    df.to_csv(output_path, index=False)

    # Print summary
    print("\n" + "=" * 60)
    print("NORMALIZATION SUMMARY")
    print("=" * 60)
    print(f"Types reduced from {len(original_types)} to {len(final_types)}")
    print("\nType transformations applied:")
    for (from_type, to_type), count in sorted(type_changes.items(), key=lambda x: -x[1]):
        print(f"  {from_type} -> {to_type}: {count:,} nodes")

    print("\nFinal type distribution:")
    for t, count in sorted(final_types.items(), key=lambda x: -x[1]):
        print(f"  {t}: {count:,}")

    print("=" * 60)

    return log_data


def validate_normalization(original_path: Path, normalized_path: Path) -> bool:
    """
    Validate that normalization didn't lose or corrupt data.

    Returns True if validation passes.
    """
    print("\nRunning validation checks...")

    df_orig = pd.read_csv(original_path, low_memory=False)
    df_norm = pd.read_csv(normalized_path, low_memory=False)

    checks_passed = True

    # Check 1: Row count
    if len(df_orig) != len(df_norm):
        print(f"  FAIL: Row count mismatch ({len(df_orig)} vs {len(df_norm)})")
        checks_passed = False
    else:
        print(f"  PASS: Row count unchanged ({len(df_orig):,})")

    # Check 2: ID uniqueness preserved
    if df_norm["id"].nunique() != len(df_norm):
        print("  FAIL: Duplicate IDs found in normalized data")
        checks_passed = False
    else:
        print("  PASS: All IDs unique")

    # Check 3: No new empty types
    empty_types = df_norm["type"].isna().sum()
    if empty_types > 0:
        print(f"  FAIL: {empty_types} nodes have empty types")
        checks_passed = False
    else:
        print("  PASS: No empty types")

    # Check 4: All types are in expected normalized set
    expected_types = set(TYPE_MAPPING.values())
    actual_types = set(df_norm["type"].unique())
    unexpected = actual_types - expected_types
    if unexpected:
        print(f"  WARNING: Unexpected types found: {unexpected}")
    else:
        print("  PASS: All types are expected normalized types")

    # Check 5: Sample spot check
    sample_ids = df_orig.sample(min(10, len(df_orig)))["id"].tolist()
    for node_id in sample_ids:
        orig_row = df_orig[df_orig["id"] == node_id].iloc[0]
        norm_row = df_norm[df_norm["id"] == node_id].iloc[0]
        if orig_row["name"] != norm_row["name"]:
            print(f"  FAIL: Name changed for {node_id}")
            checks_passed = False
            break
    else:
        print("  PASS: Spot check - names preserved")

    if checks_passed:
        print("\n✓ All validation checks passed!")
    else:
        print("\n✗ Some validation checks failed!")

    return checks_passed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Normalize entity types in knowledge graph")
    parser.add_argument("--input", type=str, default="data/processed/unified_nodes.csv",
                        help="Input nodes CSV file")
    parser.add_argument("--output", type=str, default="data/processed/unified_nodes_clean.csv",
                        help="Output normalized CSV file")
    parser.add_argument("--log", type=str, default="data/processed/validation/entity_type_normalization_log.json",
                        help="Transformation log file")
    parser.add_argument("--validate", action="store_true",
                        help="Run validation after normalization")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    log_path = Path(args.log)

    # Ensure directories exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Run normalization
    normalize_types(input_path, output_path, log_path)

    # Run validation if requested
    if args.validate:
        validate_normalization(input_path, output_path)
