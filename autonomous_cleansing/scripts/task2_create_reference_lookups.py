#!/usr/bin/env python3
"""
Task 2: Create Reference Lookups

Creates lookup dictionaries for drug names and gene symbols:
1. DrugBank ID → Drug Name (from Hetionet + PrimeKG)
2. Gene ID → Gene Symbol (from NCBI gene info)

Saves as JSON files in data/reference/ for fast loading.
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

def create_drugbank_lookup():
    """
    Create DrugBank ID to name mapping from Hetionet and PrimeKG data.
    These sources already have resolved drug names.
    """
    print("\n--- Creating DrugBank Lookup ---")

    df = pd.read_csv(DATA_DIR / "unified_nodes_clean.csv", low_memory=False)
    drugs = df[df['type'] == 'Drug']

    # Pattern to extract DB IDs
    db_pattern = re.compile(r'DB\d{5}')

    # Build mapping from sources that have proper names
    drugbank_lookup = {}
    sources_with_names = ['hetionet', 'primekg']

    for source in sources_with_names:
        source_drugs = drugs[drugs['source'] == source]
        for _, row in source_drugs.iterrows():
            # Extract DB ID from the ID field
            node_id = str(row['id'])
            match = db_pattern.search(node_id)
            if match:
                db_id = match.group()
                name = str(row['name'])
                # Only use if it's not an ID-only name
                if not name.startswith('Compound::') and not name.startswith('DB'):
                    if db_id not in drugbank_lookup:
                        drugbank_lookup[db_id] = name

    print(f"Found {len(drugbank_lookup):,} DrugBank ID → Name mappings")

    # Show sample
    print("\nSample mappings:")
    for i, (db_id, name) in enumerate(list(drugbank_lookup.items())[:10]):
        print(f"  {db_id} → {name}")

    # Save to JSON
    output_file = REF_DIR / "drugbank_lookup.json"
    with open(output_file, 'w') as f:
        json.dump(drugbank_lookup, f, indent=2)
    print(f"\nSaved to {output_file}")

    return drugbank_lookup


def create_gene_lookup():
    """
    Create Gene ID to Symbol mapping from NCBI gene info.
    """
    print("\n--- Creating Gene Lookup ---")

    gene_info_file = REF_DIR / "Homo_sapiens.gene_info"
    if not gene_info_file.exists():
        print(f"ERROR: {gene_info_file} not found!")
        print("Download with: wget https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz")
        return {}

    # Read NCBI gene info (tab-separated)
    gene_info = pd.read_csv(
        gene_info_file,
        sep='\t',
        comment='#',
        header=None,
        names=['tax_id', 'GeneID', 'Symbol', 'LocusTag', 'Synonyms', 'dbXrefs',
               'chromosome', 'map_location', 'description', 'type_of_gene',
               'Symbol_from_nomenclature_authority', 'Full_name_from_nomenclature_authority',
               'Nomenclature_status', 'Other_designations', 'Modification_date', 'Feature_type'],
        dtype={'GeneID': str}
    )

    print(f"Loaded {len(gene_info):,} gene entries")

    # Create lookup: GeneID → Symbol
    gene_lookup = {}
    for _, row in gene_info.iterrows():
        gene_id = str(row['GeneID'])
        symbol = str(row['Symbol'])
        if symbol != '-' and symbol != 'nan':
            gene_lookup[gene_id] = {
                'symbol': symbol,
                'description': str(row['description']) if pd.notna(row['description']) else '',
                'type': str(row['type_of_gene']) if pd.notna(row['type_of_gene']) else ''
            }

    print(f"Created {len(gene_lookup):,} Gene ID → Symbol mappings")

    # Show sample
    print("\nSample mappings:")
    for i, (gene_id, info) in enumerate(list(gene_lookup.items())[:10]):
        print(f"  {gene_id} → {info['symbol']} ({info['description'][:50]}...)")

    # Save to JSON
    output_file = REF_DIR / "gene_lookup.json"
    with open(output_file, 'w') as f:
        json.dump(gene_lookup, f, indent=2)
    print(f"\nSaved to {output_file}")

    return gene_lookup


def analyze_coverage():
    """
    Analyze how many drugs and genes can be resolved with current lookups.
    """
    print("\n--- Analyzing Coverage ---")

    # Load lookups
    with open(REF_DIR / "drugbank_lookup.json") as f:
        drug_lookup = json.load(f)
    with open(REF_DIR / "gene_lookup.json") as f:
        gene_lookup = json.load(f)

    # Load data
    df = pd.read_csv(DATA_DIR / "unified_nodes_clean.csv", low_memory=False)

    # Analyze drugs
    drugs = df[df['type'] == 'Drug']
    db_pattern = re.compile(r'DB\d{5}')

    drugs_need_resolution = 0
    drugs_can_resolve = 0
    drugs_no_db_id = 0

    for _, row in drugs.iterrows():
        name = str(row['name'])
        node_id = str(row['id'])

        # Check if name needs resolution (is ID-only)
        if name.startswith('Compound::') or db_pattern.match(name):
            drugs_need_resolution += 1
            # Check if we can resolve
            match = db_pattern.search(node_id)
            if match and match.group() in drug_lookup:
                drugs_can_resolve += 1
        elif not db_pattern.search(node_id):
            drugs_no_db_id += 1

    print(f"\nDrug Resolution Coverage:")
    print(f"  Total drugs: {len(drugs):,}")
    print(f"  Already have names: {len(drugs) - drugs_need_resolution:,}")
    print(f"  Need resolution: {drugs_need_resolution:,}")
    print(f"  Can resolve with lookup: {drugs_can_resolve:,}")
    print(f"  Resolution rate: {drugs_can_resolve/max(drugs_need_resolution,1)*100:.1f}%")

    # Analyze genes
    genes = df[df['type'] == 'Gene']
    gene_id_pattern = re.compile(r'Gene::(\d+)')

    genes_need_resolution = 0
    genes_can_resolve = 0

    for _, row in genes.iterrows():
        name = str(row['name'])

        # Check if name is a numeric gene ID
        if name.startswith('Gene::'):
            match = gene_id_pattern.search(name)
            if match:
                genes_need_resolution += 1
                gene_id = match.group(1)
                if gene_id in gene_lookup:
                    genes_can_resolve += 1

    print(f"\nGene Resolution Coverage:")
    print(f"  Total genes: {len(genes):,}")
    print(f"  Already have symbols: {len(genes) - genes_need_resolution:,}")
    print(f"  Need resolution: {genes_need_resolution:,}")
    print(f"  Can resolve with lookup: {genes_can_resolve:,}")
    print(f"  Resolution rate: {genes_can_resolve/max(genes_need_resolution,1)*100:.1f}%")


def main():
    print("=" * 60)
    print("Task 2: Create Reference Lookups")
    print("=" * 60)

    # Ensure reference directory exists
    REF_DIR.mkdir(parents=True, exist_ok=True)

    # Create lookups
    drug_lookup = create_drugbank_lookup()
    gene_lookup = create_gene_lookup()

    # Analyze coverage
    analyze_coverage()

    print("\n" + "=" * 60)
    print("Task 2 COMPLETE")
    print("=" * 60)

    # Summary
    print(f"\nReference files created in {REF_DIR}:")
    print(f"  - drugbank_lookup.json ({len(drug_lookup):,} entries)")
    print(f"  - gene_lookup.json ({len(gene_lookup):,} entries)")


if __name__ == "__main__":
    main()
