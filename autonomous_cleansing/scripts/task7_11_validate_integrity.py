#!/usr/bin/env python3
"""
Task 7 & 11: Validate Edge References and Data Integrity

Task 7: Ensure edges reference valid nodes
Task 11: Run comprehensive integrity checks on cleaned data

Validates:
1. All node IDs are unique
2. All edges reference valid nodes (no orphaned edges)
3. No null/empty required fields
4. Spot-check samples
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import random

# Project paths
PROJECT_ROOT = Path("/Users/jimhuff/github/open-cure")
DATA_DIR = PROJECT_ROOT / "data" / "processed"
LOG_DIR = PROJECT_ROOT / "autonomous_cleansing"


def main():
    print("=" * 60)
    print("Tasks 7 & 11: Validate Data Integrity")
    print("=" * 60)

    # Load nodes
    print("\nLoading cleaned nodes...")
    nodes_df = pd.read_csv(DATA_DIR / "unified_nodes_clean.csv", low_memory=False)
    print(f"Loaded {len(nodes_df):,} nodes")

    # Get set of valid node IDs
    valid_node_ids = set(nodes_df['id'].values)

    # ===============================
    # Validation 1: Node ID uniqueness
    # ===============================
    print("\n--- Validation 1: Node ID Uniqueness ---")
    duplicate_ids = nodes_df[nodes_df.duplicated(subset=['id'], keep=False)]
    if len(duplicate_ids) > 0:
        print(f"ERROR: Found {len(duplicate_ids)} duplicate node IDs!")
        print(duplicate_ids[['id', 'type', 'name']].head(20).to_string())
        node_unique = False
    else:
        print(f"✓ All {len(nodes_df):,} node IDs are unique")
        node_unique = True

    # ===============================
    # Validation 2: Required fields not null
    # ===============================
    print("\n--- Validation 2: Required Fields ---")
    required_fields = ['id', 'type', 'name']
    fields_valid = True

    for field in required_fields:
        null_count = nodes_df[field].isna().sum()
        empty_count = (nodes_df[field] == '').sum()
        if null_count > 0 or empty_count > 0:
            print(f"WARNING: Field '{field}' has {null_count} null and {empty_count} empty values")
            fields_valid = False
        else:
            print(f"✓ Field '{field}' has no null/empty values")

    # ===============================
    # Validation 3: Edge node references
    # ===============================
    print("\n--- Validation 3: Edge Node References ---")

    # Process edges in chunks to avoid memory issues
    chunk_size = 500_000
    orphaned_edges = 0
    invalid_sources = []
    invalid_targets = []
    total_edges = 0

    for chunk in pd.read_csv(DATA_DIR / "unified_edges_clean.csv", chunksize=chunk_size, usecols=['source', 'target']):
        total_edges += len(chunk)

        # Check source references
        invalid_src = ~chunk['source'].isin(valid_node_ids)
        if invalid_src.any():
            orphaned_edges += invalid_src.sum()
            invalid_sources.extend(chunk[invalid_src]['source'].head(10).tolist())

        # Check target references
        invalid_tgt = ~chunk['target'].isin(valid_node_ids)
        if invalid_tgt.any():
            orphaned_edges += invalid_tgt.sum()
            invalid_targets.extend(chunk[invalid_tgt]['target'].head(10).tolist())

    if orphaned_edges > 0:
        print(f"WARNING: Found {orphaned_edges:,} orphaned edges")
        if invalid_sources:
            print(f"  Sample invalid sources: {invalid_sources[:5]}")
        if invalid_targets:
            print(f"  Sample invalid targets: {invalid_targets[:5]}")
        edges_valid = False
    else:
        print(f"✓ All {total_edges:,} edges reference valid nodes")
        edges_valid = True

    # ===============================
    # Validation 4: Spot checks
    # ===============================
    print("\n--- Validation 4: Spot Checks (50 Random Entities) ---")

    sample_nodes = nodes_df.sample(50)
    spot_checks = []

    for _, row in sample_nodes.iterrows():
        check = {
            'id': row['id'],
            'type': row['type'],
            'name': row['name'],
            'has_name': bool(row['name'] and not str(row['name']).startswith(('Gene::', 'Compound::', 'Disease::'))),
            'has_id': bool(row['id']),
            'valid': True
        }
        spot_checks.append(check)

    # Summary
    named_count = sum(1 for c in spot_checks if c['has_name'])
    print(f"Sample results:")
    print(f"  - With proper names: {named_count}/50")
    print(f"  - All have IDs: {sum(1 for c in spot_checks if c['has_id'])}/50")

    print("\nSample entities:")
    for check in spot_checks[:10]:
        status = "✓" if check['has_name'] else "○"
        print(f"  {status} [{check['type']}] {check['name'][:50]}...")

    # ===============================
    # Summary
    # ===============================
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    all_valid = node_unique and fields_valid and edges_valid

    print(f"\n  Node ID uniqueness: {'PASS' if node_unique else 'FAIL'}")
    print(f"  Required fields: {'PASS' if fields_valid else 'FAIL'}")
    print(f"  Edge references: {'PASS' if edges_valid else 'FAIL'}")
    print(f"\n  OVERALL: {'ALL CHECKS PASSED' if all_valid else 'SOME CHECKS FAILED'}")

    # Save validation results
    results = {
        "timestamp": datetime.now().isoformat(),
        "nodes_count": len(nodes_df),
        "edges_count": total_edges,
        "validations": {
            "node_id_unique": node_unique,
            "required_fields_valid": fields_valid,
            "edges_valid": edges_valid,
            "all_valid": all_valid
        },
        "statistics": {
            "orphaned_edges": orphaned_edges,
            "spot_check_named_ratio": named_count / 50
        },
        "spot_checks": spot_checks
    }

    results_file = DATA_DIR / "validation" / "integrity_check.json"
    results_file.parent.mkdir(exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {results_file}")

    # Update cleansing log
    log_file = LOG_DIR / "cleansing_log.json"
    if log_file.exists():
        with open(log_file) as f:
            log = json.load(f)
    else:
        log = {"version": "1.0", "transformations": []}

    log["transformations"].append({
        "task": "Tasks 7 & 11: Data Integrity Validation",
        "timestamp": datetime.now().isoformat(),
        "all_valid": all_valid,
        "node_unique": node_unique,
        "fields_valid": fields_valid,
        "edges_valid": edges_valid,
        "orphaned_edges": orphaned_edges,
        "total_nodes": len(nodes_df),
        "total_edges": total_edges
    })

    with open(log_file, 'w') as f:
        json.dump(log, f, indent=2)

    print("\n" + "=" * 60)
    print("Tasks 7 & 11 COMPLETE")
    print("=" * 60)

    return all_valid


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
