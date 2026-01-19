#!/usr/bin/env python3
"""
Task 6: Apply Relation Normalization to Edges

Applies the relation mapping created in Task 5 to normalize 153+ relation types
to ~99 standardized relations in the unified_edges.csv file.

Processes the 16M edge file in chunks for memory efficiency.
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from collections import Counter

# Project paths
PROJECT_ROOT = Path("/Users/jimhuff/github/open-cure")
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MAPPING_DIR = DATA_DIR / "entity_mappings"
LOG_DIR = PROJECT_ROOT / "autonomous_cleansing"


def load_relation_mapping():
    """Load relation mapping from JSON."""
    mapping_file = MAPPING_DIR / "relation_mappings.json"
    with open(mapping_file) as f:
        data = json.load(f)
    return data['mapping']


def main():
    print("=" * 60)
    print("Task 6: Apply Relation Normalization to Edges")
    print("=" * 60)

    # Load mapping
    relation_mapping = load_relation_mapping()
    print(f"Loaded {len(relation_mapping)} relation mappings")

    # Input/output files
    input_file = DATA_DIR / "unified_edges.csv"
    output_file = DATA_DIR / "unified_edges_clean.csv"

    # Process in chunks
    chunk_size = 500_000
    total_edges = 0
    normalized_count = 0
    unmapped_relations = Counter()
    relation_counts_before = Counter()
    relation_counts_after = Counter()

    print(f"\nProcessing {input_file} in chunks of {chunk_size:,}...")

    # First pass: count and identify unmapped relations
    first_chunk = True
    for i, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size)):
        total_edges += len(chunk)

        for rel in chunk['relation'].value_counts().items():
            rel_name, count = rel
            relation_counts_before[rel_name] += count

            if rel_name not in relation_mapping:
                unmapped_relations[rel_name] += count

        if (i + 1) % 10 == 0:
            print(f"  Scanned {total_edges:,} edges...")

    print(f"\nTotal edges: {total_edges:,}")
    print(f"Unique relations: {len(relation_counts_before)}")

    if unmapped_relations:
        print(f"\nWARNING: {len(unmapped_relations)} unmapped relations:")
        for rel, count in unmapped_relations.most_common(20):
            print(f"  {count:>10,}  {rel}")

    # Second pass: normalize and write
    print(f"\nNormalizing relations and writing to {output_file}...")

    first_chunk = True
    edges_written = 0

    for i, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size)):
        # Apply mapping
        def normalize_relation(rel):
            if rel in relation_mapping:
                return relation_mapping[rel]
            else:
                # Keep original if no mapping
                return rel

        chunk['relation_original'] = chunk['relation']
        chunk['relation'] = chunk['relation'].apply(normalize_relation)

        # Count after normalization
        for rel, count in chunk['relation'].value_counts().items():
            relation_counts_after[rel] += count

        # Write to output (create new file on first chunk, append after)
        if first_chunk:
            chunk.to_csv(output_file, index=False, mode='w')
            first_chunk = False
        else:
            chunk.to_csv(output_file, index=False, mode='a', header=False)

        edges_written += len(chunk)
        if (i + 1) % 5 == 0:
            print(f"  Written {edges_written:,} edges...")

    # Print statistics
    print("\n--- Normalization Statistics ---")
    print(f"  Total edges: {total_edges:,}")
    print(f"  Edges written: {edges_written:,}")
    print(f"  Relations before: {len(relation_counts_before)}")
    print(f"  Relations after: {len(relation_counts_after)}")

    print("\n--- Top 20 Normalized Relations ---")
    for rel, count in sorted(relation_counts_after.items(), key=lambda x: -x[1])[:20]:
        print(f"  {count:>10,}  {rel}")

    # Validate
    print("\n--- Validation ---")
    if edges_written == total_edges:
        print("✓ Edge count unchanged")
    else:
        print(f"ERROR: Edge count changed! {total_edges:,} → {edges_written:,}")
        return False

    # Update cleansing log
    log_file = LOG_DIR / "cleansing_log.json"
    if log_file.exists():
        with open(log_file) as f:
            log = json.load(f)
    else:
        log = {"version": "1.0", "transformations": []}

    log["transformations"].append({
        "task": "Task 6: Normalize Relations",
        "timestamp": datetime.now().isoformat(),
        "input_file": str(input_file),
        "output_file": str(output_file),
        "total_edges": total_edges,
        "relations_before": len(relation_counts_before),
        "relations_after": len(relation_counts_after),
        "unmapped_relations": len(unmapped_relations),
        "top_unmapped": list(unmapped_relations.most_common(10))
    })

    with open(log_file, 'w') as f:
        json.dump(log, f, indent=2)
    print("✓ Cleansing log updated")

    print("\n" + "=" * 60)
    print("Task 6 COMPLETE")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
