#!/usr/bin/env python3
"""
Task 3: Resolve Drug Names from DrugBank IDs

Replaces ID-only drug names (e.g., "Compound::DB02573") with actual drug names
(e.g., "Tobramycin") using the DrugBank lookup created in Task 2.

Updates the unified_nodes_clean.csv file with resolved names and adds
a drugbank_id column to preserve the original IDs.
"""

import pandas as pd
import json
import re
from pathlib import Path
from datetime import datetime

# Project paths
PROJECT_ROOT = Path("/Users/jimhuff/github/open-cure")
DATA_DIR = PROJECT_ROOT / "data" / "processed"
REF_DIR = PROJECT_ROOT / "data" / "reference"
LOG_DIR = PROJECT_ROOT / "autonomous_cleansing"


def load_drugbank_lookup():
    """Load DrugBank ID to name lookup."""
    lookup_file = REF_DIR / "drugbank_lookup.json"
    with open(lookup_file) as f:
        return json.load(f)


def main():
    print("=" * 60)
    print("Task 3: Resolve Drug Names from DrugBank IDs")
    print("=" * 60)

    # Load data
    input_file = DATA_DIR / "unified_nodes_clean.csv"
    output_file = DATA_DIR / "unified_nodes_clean.csv"  # Update in place

    print(f"\nReading {input_file}...")
    df = pd.read_csv(input_file, low_memory=False)
    original_count = len(df)

    # Load lookup
    drugbank_lookup = load_drugbank_lookup()
    print(f"Loaded {len(drugbank_lookup):,} DrugBank ID → Name mappings")

    # Get drugs
    drug_mask = df['type'] == 'Drug'
    drugs = df[drug_mask].copy()
    print(f"\nTotal drugs: {len(drugs):,}")

    # Add drugbank_id column if it doesn't exist
    if 'drugbank_id' not in df.columns:
        df['drugbank_id'] = ''

    # Pattern to extract DB IDs
    db_pattern = re.compile(r'DB\d{5}')

    # Track statistics
    already_named = 0
    resolved = 0
    unresolved = 0
    unresolved_ids = []

    print("\nResolving drug names...")
    for idx in drugs.index:
        name = str(df.at[idx, 'name'])
        node_id = str(df.at[idx, 'id'])

        # Check if name needs resolution (is ID-only)
        if name.startswith('Compound::') or db_pattern.match(name):
            # Extract DrugBank ID
            match = db_pattern.search(node_id)
            if not match:
                match = db_pattern.search(name)

            if match:
                db_id = match.group()
                df.at[idx, 'drugbank_id'] = db_id

                if db_id in drugbank_lookup:
                    df.at[idx, 'name'] = drugbank_lookup[db_id]
                    resolved += 1
                else:
                    unresolved += 1
                    if len(unresolved_ids) < 100:
                        unresolved_ids.append(db_id)
            else:
                unresolved += 1
        else:
            already_named += 1
            # Still try to extract DrugBank ID for reference
            match = db_pattern.search(node_id)
            if match:
                df.at[idx, 'drugbank_id'] = match.group()

    # Print statistics
    print("\n--- Resolution Statistics ---")
    print(f"  Already had proper names: {already_named:,}")
    print(f"  Successfully resolved: {resolved:,}")
    print(f"  Could not resolve: {unresolved:,}")
    if unresolved > 0:
        resolution_rate = resolved / (resolved + unresolved) * 100
        print(f"  Resolution rate: {resolution_rate:.1f}%")

    if unresolved_ids:
        print(f"\nSample unresolved IDs ({len(unresolved_ids)} shown):")
        for db_id in unresolved_ids[:20]:
            print(f"    {db_id}")

    # Validate
    print("\n--- Validation ---")
    final_count = len(df)
    print(f"Original count: {original_count:,}")
    print(f"Final count: {final_count:,}")

    if original_count != final_count:
        print("ERROR: Row count changed!")
        return False
    else:
        print("✓ Row count unchanged")

    # Spot check - show some resolved names
    print("\n--- Spot Check: Sample Resolved Drugs ---")
    resolved_drugs = df[(df['type'] == 'Drug') & (df['drugbank_id'] != '')]
    sample = resolved_drugs.sample(min(20, len(resolved_drugs)))
    for _, row in sample.iterrows():
        print(f"  {row['drugbank_id']} → {row['name']}")

    # Save updated data
    print(f"\n--- Saving to {output_file} ---")
    df.to_csv(output_file, index=False)
    print(f"Saved {final_count:,} nodes")

    # Update cleansing log
    log_file = LOG_DIR / "cleansing_log.json"
    if log_file.exists():
        with open(log_file) as f:
            log = json.load(f)
    else:
        log = {"version": "1.0", "transformations": []}

    log["transformations"].append({
        "task": "Task 3: Resolve Drug Names",
        "timestamp": datetime.now().isoformat(),
        "input_file": str(input_file),
        "output_file": str(output_file),
        "total_drugs": len(drugs),
        "already_named": already_named,
        "resolved": resolved,
        "unresolved": unresolved,
        "resolution_rate": resolved / max(resolved + unresolved, 1) * 100,
        "sample_unresolved": unresolved_ids[:20]
    })

    with open(log_file, 'w') as f:
        json.dump(log, f, indent=2)
    print("✓ Cleansing log updated")

    print("\n" + "=" * 60)
    print("Task 3 COMPLETE")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
