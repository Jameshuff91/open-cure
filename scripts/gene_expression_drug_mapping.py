#!/usr/bin/env python3
"""
Gene Expression → Drug Mapping Pipeline (h49)

Given a list of dysregulated genes, rank drugs by how many of those genes they target.
This enables going from gene expression data directly to drug candidates.
"""

import argparse
import json
import os
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import List, Dict, Set, Tuple

# DRKG edge types that represent drug-gene targeting relationships
DRUG_GENE_EDGE_TYPES = [
    # Direct targets
    'DRUGBANK::target::Compound:Gene',
    'DGIDB::INHIBITOR::Gene:Compound',
    'DGIDB::AGONIST::Gene:Compound',
    'DGIDB::ANTAGONIST::Gene:Compound',
    'DGIDB::ACTIVATOR::Gene:Compound',
    'DGIDB::BLOCKER::Gene:Compound',
    'DGIDB::MODULATOR::Gene:Compound',
    'DGIDB::BINDER::Gene:Compound',
    'DGIDB::CHANNEL BLOCKER::Gene:Compound',
    'DGIDB::ALLOSTERIC MODULATOR::Gene:Compound',
    'DGIDB::POSITIVE ALLOSTERIC MODULATOR::Gene:Compound',
    'DGIDB::ANTIBODY::Gene:Compound',
    'DGIDB::PARTIAL AGONIST::Gene:Compound',
    'DGIDB::OTHER::Gene:Compound',
    # Hetionet bindings
    'Hetionet::CbG::Compound:Gene',  # binding
    'Hetionet::CdG::Compound:Gene',  # downregulation
    'Hetionet::CuG::Compound:Gene',  # upregulation
    # GNBR
    'GNBR::B::Compound:Gene',   # binding
    'GNBR::N::Compound:Gene',   # inhibits
    'GNBR::A+::Compound:Gene',  # agonism
    'GNBR::A-::Compound:Gene',  # antagonism
    'GNBR::E::Compound:Gene',   # affects expression
    'GNBR::E+::Compound:Gene',  # increases expression
    'GNBR::E-::Compound:Gene',  # decreases expression
    'GNBR::Z::Compound:Gene',   # enzyme activity
    'GNBR::O::Compound:Gene',   # transport
    # IntAct
    'INTACT::ASSOCIATION::Compound:Gene',
    'INTACT::DIRECT INTERACTION::Compound:Gene',
    'INTACT::PHYSICAL ASSOCIATION::Compound:Gene',
    # bioarx
    'bioarx::DrugHumGen:Compound:Gene',
]


def load_drkg(drkg_path: str = 'data/raw/drkg/drkg.tsv') -> pd.DataFrame:
    """Load DRKG knowledge graph."""
    print(f"Loading DRKG from {drkg_path}...")
    df = pd.read_csv(drkg_path, sep='\t', header=None, names=['head', 'relation', 'tail'])
    print(f"  Loaded {len(df):,} edges")
    return df


def extract_drug_gene_mapping(drkg: pd.DataFrame) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Extract bidirectional drug-gene mappings from DRKG.
    
    Returns:
        drug_to_genes: dict mapping drug ID to set of gene IDs
        gene_to_drugs: dict mapping gene ID to set of drug IDs
    """
    drug_to_genes = defaultdict(set)
    gene_to_drugs = defaultdict(set)
    
    # Filter to relevant edge types
    mask = drkg['relation'].isin(DRUG_GENE_EDGE_TYPES)
    filtered = drkg[mask]
    print(f"  Found {len(filtered):,} drug-gene edges across {len(DRUG_GENE_EDGE_TYPES)} edge types")
    
    for _, row in filtered.iterrows():
        head, relation, tail = row['head'], row['relation'], row['tail']
        
        # Determine which is drug and which is gene based on entity type prefix
        if head.startswith('Compound::') and tail.startswith('Gene::'):
            drug, gene = head, tail
        elif head.startswith('Gene::') and tail.startswith('Compound::'):
            gene, drug = head, tail
        else:
            continue  # Skip malformed edges
            
        drug_to_genes[drug].add(gene)
        gene_to_drugs[gene].add(drug)
    
    print(f"  Extracted {len(drug_to_genes):,} drugs targeting {len(gene_to_drugs):,} genes")
    
    return dict(drug_to_genes), dict(gene_to_drugs)


def load_drug_names(drugbank_path: str = 'data/reference/drugbank_lookup.json') -> Dict[str, str]:
    """Load drug ID to name mapping."""
    if not os.path.exists(drugbank_path):
        print(f"  Warning: {drugbank_path} not found, using IDs only")
        return {}
    
    with open(drugbank_path) as f:
        lookup = json.load(f)
    
    # Create Compound::DB* -> name mapping
    name_map = {}
    for db_id, name_or_info in lookup.items():
        compound_id = f"Compound::{db_id}"
        # Handle both simple str format and dict format
        if isinstance(name_or_info, dict):
            name = name_or_info.get('name', db_id)
        else:
            name = name_or_info  # Simple ID: name format
        name_map[compound_id] = name
    
    print(f"  Loaded {len(name_map):,} drug names")
    return name_map


def rank_drugs_by_gene_coverage(
    query_genes: List[str],
    gene_to_drugs: Dict[str, Set[str]],
    drug_to_genes: Dict[str, Set[str]],
    drug_names: Dict[str, str] = None,
    min_overlap: int = 1
) -> pd.DataFrame:
    """
    Rank drugs by how many query genes they target.
    
    Args:
        query_genes: List of gene IDs (e.g., 'Gene::1234' or just '1234')
        gene_to_drugs: Mapping from gene to drugs
        drug_to_genes: Mapping from drug to genes
        drug_names: Optional drug ID to name mapping
        min_overlap: Minimum gene overlap to include a drug
        
    Returns:
        DataFrame with columns: drug_id, drug_name, n_genes_targeted, query_genes_targeted, total_targets
    """
    # Normalize gene IDs
    normalized_genes = set()
    for g in query_genes:
        if g.startswith('Gene::'):
            normalized_genes.add(g)
        else:
            normalized_genes.add(f'Gene::{g}')
    
    # Find all drugs that target any of the query genes
    drug_scores = defaultdict(lambda: {'genes_hit': set(), 'total_targets': 0})
    
    for gene in normalized_genes:
        if gene in gene_to_drugs:
            for drug in gene_to_drugs[gene]:
                drug_scores[drug]['genes_hit'].add(gene)
    
    # Build results
    results = []
    for drug, info in drug_scores.items():
        n_hit = len(info['genes_hit'])
        if n_hit >= min_overlap:
            total_targets = len(drug_to_genes.get(drug, set()))
            specificity = n_hit / total_targets if total_targets > 0 else 0
            
            results.append({
                'drug_id': drug,
                'drug_name': drug_names.get(drug, drug.replace('Compound::', '')) if drug_names else drug,
                'n_query_genes_targeted': n_hit,
                'query_genes_targeted': sorted([g.replace('Gene::', '') for g in info['genes_hit']]),
                'total_targets': total_targets,
                'coverage': n_hit / len(normalized_genes),  # fraction of query genes covered
                'specificity': specificity  # fraction of drug's targets that are in query
            })
    
    df = pd.DataFrame(results)
    if len(df) > 0:
        # Sort by coverage (primary), then by specificity (secondary)
        df = df.sort_values(['n_query_genes_targeted', 'specificity'], ascending=[False, False])
        df = df.reset_index(drop=True)
    
    return df


def validate_with_known_pair(
    disease: str,
    known_drug: str,
    gene_to_drugs: Dict[str, Set[str]],
    drug_to_genes: Dict[str, Set[str]],
    disease_genes: Dict[str, List[str]]
) -> Dict:
    """
    Validate by checking if a known drug ranks highly for its disease's genes.
    """
    if disease not in disease_genes:
        return {'status': 'no_genes', 'disease': disease}
    
    genes = disease_genes[disease]
    rankings = rank_drugs_by_gene_coverage(genes, gene_to_drugs, drug_to_genes)
    
    if known_drug in rankings['drug_id'].values:
        rank = rankings[rankings['drug_id'] == known_drug].index[0] + 1
        return {'status': 'found', 'rank': rank, 'total_drugs': len(rankings), 'n_genes': len(genes)}
    else:
        return {'status': 'not_found', 'total_drugs': len(rankings), 'n_genes': len(genes)}


def save_mapping(drug_to_genes: Dict, gene_to_drugs: Dict, output_dir: str = 'data/reference'):
    """Save the mappings for later use."""
    # Convert sets to lists for JSON serialization
    d2g = {k: list(v) for k, v in drug_to_genes.items()}
    g2d = {k: list(v) for k, v in gene_to_drugs.items()}
    
    d2g_path = os.path.join(output_dir, 'drug_to_genes_drkg.json')
    g2d_path = os.path.join(output_dir, 'gene_to_drugs_drkg.json')
    
    with open(d2g_path, 'w') as f:
        json.dump(d2g, f)
    print(f"  Saved drug→genes mapping to {d2g_path}")
    
    with open(g2d_path, 'w') as f:
        json.dump(g2d, f)
    print(f"  Saved gene→drugs mapping to {g2d_path}")


def load_disease_genes(path: str = 'data/reference/disease_genes.json') -> Dict[str, List[str]]:
    """Load disease-gene associations."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description='Gene Expression → Drug Mapping Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build the gene-drug mapping (first time only)
  python gene_expression_drug_mapping.py --build-mapping

  # Query with gene IDs
  python gene_expression_drug_mapping.py --genes "7124,7157,3458,1956"

  # Query with a MESH disease ID (uses DRKG disease-gene associations)
  python gene_expression_drug_mapping.py --disease "MESH:D011565"

  # Query from a file of genes
  python gene_expression_drug_mapping.py --gene-file my_dysregulated_genes.txt
        """
    )
    parser.add_argument('--genes', type=str, help='Comma-separated list of gene IDs (e.g., "1234,5678,9012")')
    parser.add_argument('--gene-file', type=str, help='File with one gene ID per line')
    parser.add_argument('--disease', type=str, help='MESH disease ID (e.g., "MESH:D011565" for psoriasis)')
    parser.add_argument('--build-mapping', action='store_true', help='Build and save the gene-drug mapping from DRKG')
    parser.add_argument('--output', type=str, default='data/analysis/gene_drug_ranking.csv', help='Output CSV path')
    parser.add_argument('--top-n', type=int, default=50, help='Number of top drugs to show')
    parser.add_argument('--min-overlap', type=int, default=1, help='Minimum gene overlap')
    
    args = parser.parse_args()
    
    # Load or build mapping
    d2g_path = 'data/reference/drug_to_genes_drkg.json'
    g2d_path = 'data/reference/gene_to_drugs_drkg.json'
    
    if args.build_mapping or not (os.path.exists(d2g_path) and os.path.exists(g2d_path)):
        print("Building gene-drug mapping from DRKG...")
        drkg = load_drkg()
        drug_to_genes, gene_to_drugs = extract_drug_gene_mapping(drkg)
        save_mapping(drug_to_genes, gene_to_drugs)
    else:
        print("Loading pre-built mapping...")
        with open(d2g_path) as f:
            drug_to_genes = {k: set(v) for k, v in json.load(f).items()}
        with open(g2d_path) as f:
            gene_to_drugs = {k: set(v) for k, v in json.load(f).items()}
        print(f"  Loaded {len(drug_to_genes):,} drugs, {len(gene_to_drugs):,} genes")
    
    # Load drug names
    drug_names = load_drug_names()
    
    # If no genes provided, just print stats and exit
    if not args.genes and not args.gene_file and not args.disease:
        print("\nMapping Statistics:")
        n_edges = sum(len(g) for g in drug_to_genes.values())
        print(f"  Total drug-gene edges: {n_edges:,}")
        print(f"  Unique drugs: {len(drug_to_genes):,}")
        print(f"  Unique genes: {len(gene_to_drugs):,}")
        
        # Show some example drugs with many targets
        target_counts = [(d, len(g)) for d, g in drug_to_genes.items()]
        target_counts.sort(key=lambda x: -x[1])
        print("\n  Top 10 drugs by target count:")
        for drug, count in target_counts[:10]:
            name = drug_names.get(drug, drug)
            print(f"    {name}: {count} targets")
        
        return
    
    # Parse query genes
    query_genes: List[str] = []
    if args.genes:
        query_genes = [g.strip() for g in args.genes.split(',')]
    elif args.gene_file:
        with open(args.gene_file) as f:
            query_genes = [line.strip() for line in f if line.strip()]
    elif args.disease:
        disease_genes_map = load_disease_genes()
        disease_id = args.disease
        # Try with and without MESH: prefix
        if disease_id not in disease_genes_map and not disease_id.startswith('MESH:'):
            disease_id = f'MESH:{disease_id}'
        if disease_id in disease_genes_map:
            query_genes = disease_genes_map[disease_id]
            print(f"Disease {args.disease}: {len(query_genes)} associated genes")
        else:
            print(f"Error: Disease {args.disease} not found in disease_genes.json")
            print("Available diseases: run with --list-diseases flag")
            return
    
    print(f"\nQuery: {len(query_genes)} genes")
    
    # Rank drugs
    rankings = rank_drugs_by_gene_coverage(
        query_genes, gene_to_drugs, drug_to_genes, drug_names,
        min_overlap=args.min_overlap
    )
    
    print(f"Found {len(rankings)} drugs targeting at least {args.min_overlap} query gene(s)")
    
    if len(rankings) > 0:
        print(f"\nTop {min(args.top_n, len(rankings))} drugs:")
        print("-" * 80)
        for i, row in rankings.head(args.top_n).iterrows():
            print(f"{i+1:3}. {row['drug_name']:<30} | {row['n_query_genes_targeted']:2} genes | "
                  f"coverage: {row['coverage']:.1%} | specificity: {row['specificity']:.1%}")
        
        # Save full results
        rankings.to_csv(args.output, index=False)
        print(f"\nFull results saved to {args.output}")


if __name__ == '__main__':
    main()
