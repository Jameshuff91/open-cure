#!/usr/bin/env python3
"""
h93: Direct Mechanism Traversal for Drug Repurposing

Traverses DRKG graph: Disease -> Associated Genes -> Drugs that Target Those Genes

This is a no-ML baseline that makes predictions purely based on graph structure:
- Finds all genes associated with a disease
- Finds all drugs that target those genes
- Ranks drugs by number of disease genes they target
"""

import json
import csv
from collections import defaultdict
from pathlib import Path


def load_mesh_mappings():
    """Load disease name to MESH ID mappings."""
    with open('data/reference/mesh_mappings_from_agents.json') as f:
        mesh_data = json.load(f)

    all_mesh = {}
    for k, v in mesh_data.items():
        if k != 'metadata' and isinstance(v, dict):
            all_mesh.update(v)
    return all_mesh


def load_drugbank_lookup():
    """Load DrugBank ID to drug name mappings."""
    try:
        with open('data/reference/drugbank_lookup.json') as f:
            return json.load(f)
    except:
        return {}


def build_disease_gene_index(edges_file='data/processed/unified_edges_clean.csv'):
    """Build index: MESH_ID -> set of gene IDs."""
    disease_genes = defaultdict(set)

    with open(edges_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['relation'] == 'ASSOCIATED_GENE':
                # source: drkg:Disease::MESH:D008113
                # target: drkg:Gene::1234
                source = row['source']
                target = row['target']

                if 'Disease::' in source and 'Gene::' in target:
                    # Extract MESH ID
                    mesh_id = source.replace('drkg:Disease::', '').replace('MESH:', '')
                    gene_id = target.replace('drkg:Gene::', '')
                    disease_genes[mesh_id].add(gene_id)

    return disease_genes


def build_gene_drug_index(edges_file='data/processed/unified_edges_clean.csv'):
    """Build index: Gene ID -> set of DrugBank IDs."""
    gene_drugs = defaultdict(set)

    with open(edges_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['relation'] == 'TARGETS':
                # source: drkg:Compound::DB00001
                # target: drkg:Gene::2147
                source = row['source']
                target = row['target']

                if 'Compound::' in source and 'Gene::' in target:
                    drug_id = source.replace('drkg:Compound::', '')
                    gene_id = target.replace('drkg:Gene::', '')
                    gene_drugs[gene_id].add(drug_id)

    return gene_drugs


def predict_drugs_for_disease(mesh_id, disease_genes, gene_drugs, top_n=30):
    """
    Predict drugs for a disease via mechanism traversal.

    Returns: List of (drug_id, score, num_genes_targeted) tuples
    """
    # Get genes associated with this disease
    genes = disease_genes.get(mesh_id, set())

    if not genes:
        return []

    # Count how many disease genes each drug targets
    drug_scores = defaultdict(int)
    drug_gene_targets = defaultdict(set)

    for gene in genes:
        drugs_targeting_gene = gene_drugs.get(gene, set())
        for drug in drugs_targeting_gene:
            drug_scores[drug] += 1
            drug_gene_targets[drug].add(gene)

    # Sort by number of genes targeted (descending)
    ranked_drugs = sorted(drug_scores.items(), key=lambda x: -x[1])

    # Return top N with scores
    results = []
    for drug_id, score in ranked_drugs[:top_n]:
        results.append({
            'drug_id': drug_id,
            'score': score,
            'genes_targeted': len(drug_gene_targets[drug_id]),
            'total_disease_genes': len(genes)
        })

    return results


def evaluate_on_benchmark(benchmark_file='data/analysis/zero_shot_benchmark.json'):
    """
    Evaluate mechanism traversal on zero-shot benchmark.
    """
    print("Loading DRKG indices...")
    disease_genes = build_disease_gene_index()
    gene_drugs = build_gene_drug_index()
    mesh_mappings = load_mesh_mappings()
    drugbank_lookup = load_drugbank_lookup()

    print(f"Diseases with gene associations: {len(disease_genes)}")
    print(f"Genes with drug targets: {len(gene_drugs)}")

    # Load benchmark
    with open(benchmark_file) as f:
        benchmark = json.load(f)

    diseases_in_drkg = benchmark['diseases_in_drkg']
    benchmark_diseases = benchmark['benchmark_diseases']

    # Create lookup for potential treatments
    disease_treatments = {}
    for entry in benchmark_diseases:
        disease_name = entry['disease'].lower()
        disease_treatments[disease_name] = [t.lower() for t in entry.get('potential_treatments', [])]

    results = []
    hits_at_30 = 0
    diseases_evaluated = 0

    for disease_name in diseases_in_drkg:
        disease_lower = disease_name.lower()
        mesh_id = mesh_mappings.get(disease_lower)

        if not mesh_id:
            continue

        if mesh_id not in disease_genes:
            continue

        # Get predictions
        predictions = predict_drugs_for_disease(mesh_id, disease_genes, gene_drugs, top_n=30)

        if not predictions:
            continue

        diseases_evaluated += 1

        # Get potential treatments for this disease
        potential_treatments = disease_treatments.get(disease_lower, [])

        # Check if any prediction matches potential treatment
        hit = False
        pred_drug_names = []

        for pred in predictions:
            drug_id = pred['drug_id']
            drug_name = drugbank_lookup.get(drug_id, drug_id).lower()
            pred_drug_names.append(drug_name)

            for treatment in potential_treatments:
                if treatment in drug_name or drug_name in treatment:
                    hit = True
                    break

        if hit:
            hits_at_30 += 1

        results.append({
            'disease': disease_name,
            'mesh_id': mesh_id,
            'num_genes': len(disease_genes[mesh_id]),
            'num_predictions': len(predictions),
            'hit_at_30': hit,
            'top_5_predictions': pred_drug_names[:5],
            'potential_treatments': potential_treatments
        })

    # Summary
    print(f"\n=== Mechanism Traversal Results ===")
    print(f"Diseases evaluated: {diseases_evaluated}")
    print(f"Hits@30: {hits_at_30}")
    if diseases_evaluated > 0:
        recall_at_30 = hits_at_30 / diseases_evaluated * 100
        print(f"Recall@30: {recall_at_30:.1f}%")

    print(f"\n=== Per-Disease Results ===")
    for r in results:
        hit_symbol = '✓' if r['hit_at_30'] else '✗'
        print(f"{hit_symbol} {r['disease']}: {r['num_genes']} genes -> {r['num_predictions']} drugs")
        print(f"    Predictions: {r['top_5_predictions'][:3]}")
        print(f"    Actual treatments: {r['potential_treatments'][:3]}")

    return {
        'diseases_evaluated': diseases_evaluated,
        'hits_at_30': hits_at_30,
        'recall_at_30': hits_at_30 / diseases_evaluated * 100 if diseases_evaluated > 0 else 0,
        'detailed_results': results
    }


if __name__ == '__main__':
    results = evaluate_on_benchmark()

    # Save results
    output_file = 'data/analysis/mechanism_traversal_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")
