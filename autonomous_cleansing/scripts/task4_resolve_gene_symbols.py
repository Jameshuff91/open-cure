#!/usr/bin/env python3
"""
Task 4: Resolve Gene Symbols from NCBI IDs

Replaces numeric gene IDs (e.g., "Gene::2157") with HGNC symbols
(e.g., "EGF") using the NCBI gene lookup created in Task 2.

Updates the unified_nodes_clean.csv file with resolved symbols and adds
a gene_id column to preserve the original NCBI IDs.
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


def load_gene_lookup():
    """Load Gene ID to symbol lookup."""
    lookup_file = REF_DIR / "gene_lookup.json"
    with open(lookup_file) as f:
        return json.load(f)


def main():
    print("=" * 60)
    print("Task 4: Resolve Gene Symbols from NCBI IDs")
    print("=" * 60)

    # Load data
    input_file = DATA_DIR / "unified_nodes_clean.csv"
    output_file = DATA_DIR / "unified_nodes_clean.csv"  # Update in place

    print(f"\nReading {input_file}...")
    df = pd.read_csv(input_file, low_memory=False)
    original_count = len(df)

    # Load lookup
    gene_lookup = load_gene_lookup()
    print(f"Loaded {len(gene_lookup):,} Gene ID → Symbol mappings")

    # Get genes
    gene_mask = df['type'] == 'Gene'
    genes = df[gene_mask].copy()
    print(f"\nTotal genes: {len(genes):,}")

    # Add gene_id column if it doesn't exist
    if 'ncbi_gene_id' not in df.columns:
        df['ncbi_gene_id'] = ''

    # Pattern to extract numeric gene IDs
    gene_id_pattern = re.compile(r'Gene::(\d+)')

    # Track statistics
    already_named = 0
    resolved = 0
    unresolved = 0
    unresolved_ids = []

    print("\nResolving gene symbols...")
    for idx in genes.index:
        name = str(df.at[idx, 'name'])
        node_id = str(df.at[idx, 'id'])

        # Check if name needs resolution (is Gene::numeric)
        match = gene_id_pattern.search(name)
        if match:
            gene_id = match.group(1)
            df.at[idx, 'ncbi_gene_id'] = gene_id

            if gene_id in gene_lookup:
                gene_info = gene_lookup[gene_id]
                df.at[idx, 'name'] = gene_info['symbol']
                resolved += 1
            else:
                unresolved += 1
                if len(unresolved_ids) < 100:
                    unresolved_ids.append(gene_id)
        else:
            already_named += 1
            # Try to extract gene ID from the node ID for reference
            id_match = gene_id_pattern.search(node_id)
            if id_match:
                df.at[idx, 'ncbi_gene_id'] = id_match.group(1)
            else:
                # Try extracting just a number from the name if it looks like an ID
                if name.isdigit():
                    df.at[idx, 'ncbi_gene_id'] = name

    # Print statistics
    print("\n--- Resolution Statistics ---")
    print(f"  Already had proper symbols: {already_named:,}")
    print(f"  Successfully resolved: {resolved:,}")
    print(f"  Could not resolve: {unresolved:,}")
    if unresolved > 0:
        resolution_rate = resolved / (resolved + unresolved) * 100
        print(f"  Resolution rate: {resolution_rate:.1f}%")

    if unresolved_ids:
        print(f"\nSample unresolved IDs ({len(unresolved_ids)} shown):")
        for gene_id in unresolved_ids[:20]:
            print(f"    {gene_id}")

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

    # Spot check - show some resolved gene symbols
    print("\n--- Spot Check: Sample Resolved Genes ---")
    resolved_genes = df[(df['type'] == 'Gene') & (df['ncbi_gene_id'] != '')]
    sample = resolved_genes.sample(min(20, len(resolved_genes)))
    for _, row in sample.iterrows():
        print(f"  {row['ncbi_gene_id']} → {row['name']}")

    # Verify known genes
    print("\n--- Known Gene Verification ---")
    known_genes = {
        '2157': 'EGFR',  # EGF receptor
        '7157': 'TP53',  # Tumor protein p53
        '672': 'BRCA1',  # BRCA1
        '5290': 'PIK3CA',  # PI3K
        '1956': 'EGFR',  # EGFR (should match)
    }
    for gene_id, expected in known_genes.items():
        if gene_id in gene_lookup:
            actual = gene_lookup[gene_id]['symbol']
            status = "✓" if actual == expected else f"✗ (got {actual})"
            print(f"  Gene {gene_id}: expected {expected}, got {status}")
        else:
            print(f"  Gene {gene_id}: not in lookup")

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
        "task": "Task 4: Resolve Gene Symbols",
        "timestamp": datetime.now().isoformat(),
        "input_file": str(input_file),
        "output_file": str(output_file),
        "total_genes": len(genes),
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
    print("Task 4 COMPLETE")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
