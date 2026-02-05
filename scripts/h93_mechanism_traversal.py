#!/usr/bin/env python3
"""
h93: Direct Mechanism Traversal (No ML)

Pure graph reasoning approach:
1. For each disease, get associated genes
2. For each gene, get drugs that target it
3. Rank drugs by: number of disease genes targeted
4. Evaluate on standard benchmark
"""

import json
import random
from collections import defaultdict
import numpy as np

def load_data():
    """Load disease-gene, drug-target, and ground truth data."""
    with open('data/reference/disease_genes.json') as f:
        disease_genes = json.load(f)

    with open('data/reference/drug_targets.json') as f:
        drug_targets = json.load(f)

    with open('data/reference/expanded_ground_truth.json') as f:
        ground_truth = json.load(f)

    return disease_genes, drug_targets, ground_truth


def build_gene_to_drugs(drug_targets):
    """Build reverse index: gene -> list of drugs."""
    gene_to_drugs = defaultdict(list)
    for drug, genes in drug_targets.items():
        for gene in genes:
            gene_to_drugs[gene].append(drug)
    return dict(gene_to_drugs)


def predict_drugs_for_disease(disease_id, disease_genes, gene_to_drugs):
    """
    Predict drugs for a disease using mechanism traversal.

    Returns list of (drug_id, score) tuples, sorted by score descending.
    Score = number of disease genes the drug targets.
    """
    if disease_id not in disease_genes:
        return []

    genes = disease_genes[disease_id]

    # Count how many disease genes each drug targets
    drug_scores = defaultdict(int)
    for gene in genes:
        if gene in gene_to_drugs:
            for drug in gene_to_drugs[gene]:
                drug_scores[drug] += 1

    # Sort by score (number of genes targeted)
    ranked_drugs = sorted(drug_scores.items(), key=lambda x: -x[1])
    return ranked_drugs


def normalize_drug_id(drug_id):
    """Extract the core drug ID from various formats."""
    if '::' in drug_id:
        # Format: drkg:Compound::DB00001 or drkg:Compound::MESH:D000001
        parts = drug_id.split('::')
        return parts[-1]
    return drug_id


def normalize_disease_id(disease_id):
    """Extract MESH ID from various formats."""
    if '::' in disease_id:
        # Format: drkg:Disease::MESH:D001234
        parts = disease_id.split('::')
        return parts[-1]
    return disease_id


def evaluate_recall_at_k(predictions, ground_truth, k=30):
    """
    Evaluate Recall@k for a disease.

    predictions: list of (drug_id, score) tuples
    ground_truth: list of drug IDs that are correct
    k: cutoff for predictions

    Returns recall (0 to 1)
    """
    if not ground_truth:
        return None  # Can't evaluate without ground truth

    top_k_drugs = set(normalize_drug_id(d) for d, _ in predictions[:k])
    gt_drugs = set(normalize_drug_id(d) for d in ground_truth)

    hits = top_k_drugs & gt_drugs
    recall = len(hits) / len(gt_drugs)
    return recall


def disease_holdout_evaluation(disease_genes, gene_to_drugs, ground_truth,
                               test_fraction=0.2, seed=42, num_seeds=5):
    """
    Evaluate with disease holdout (honest evaluation).

    Returns mean and std of per-disease R@30 across seeds.
    """
    results_by_seed = []

    for s in range(num_seeds):
        random.seed(seed + s)

        # Get diseases that have both gene annotations and ground truth
        evaluable_diseases = []
        for gt_disease, gt_drugs in ground_truth.items():
            mesh_id = normalize_disease_id(gt_disease)
            if mesh_id in disease_genes and gt_drugs:
                evaluable_diseases.append((gt_disease, mesh_id))

        # Split into train/test by disease
        random.shuffle(evaluable_diseases)
        n_test = int(len(evaluable_diseases) * test_fraction)
        test_diseases = evaluable_diseases[:n_test]

        # Evaluate on test diseases
        recalls = []
        for gt_disease, mesh_id in test_diseases:
            predictions = predict_drugs_for_disease(mesh_id, disease_genes, gene_to_drugs)
            recall = evaluate_recall_at_k(predictions, ground_truth[gt_disease], k=30)
            if recall is not None:
                recalls.append(recall)

        if recalls:
            mean_recall = np.mean(recalls)
            results_by_seed.append(mean_recall)

    return np.mean(results_by_seed), np.std(results_by_seed), len(test_diseases)


def analyze_coverage(disease_genes, gene_to_drugs, ground_truth):
    """Analyze what fraction of GT drugs can be reached via mechanism."""
    total_gt_drugs = 0
    reachable_gt_drugs = 0
    diseases_with_reachable = 0
    diseases_evaluated = 0

    for gt_disease, gt_drugs in ground_truth.items():
        mesh_id = normalize_disease_id(gt_disease)
        if mesh_id not in disease_genes:
            continue

        diseases_evaluated += 1

        # Get all drugs reachable via mechanism for this disease
        predictions = predict_drugs_for_disease(mesh_id, disease_genes, gene_to_drugs)
        reachable = set(normalize_drug_id(d) for d, _ in predictions)

        # Count GT drugs that are reachable
        gt_normalized = set(normalize_drug_id(d) for d in gt_drugs)
        hits = reachable & gt_normalized

        total_gt_drugs += len(gt_normalized)
        reachable_gt_drugs += len(hits)

        if hits:
            diseases_with_reachable += 1

    return {
        'diseases_evaluated': diseases_evaluated,
        'diseases_with_reachable_gt': diseases_with_reachable,
        'total_gt_drugs': total_gt_drugs,
        'reachable_gt_drugs': reachable_gt_drugs,
        'coverage': reachable_gt_drugs / total_gt_drugs if total_gt_drugs > 0 else 0
    }


def main():
    print("h93: Direct Mechanism Traversal Evaluation")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    disease_genes, drug_targets, ground_truth = load_data()
    gene_to_drugs = build_gene_to_drugs(drug_targets)

    print(f"  Diseases with gene annotations: {len(disease_genes)}")
    print(f"  Drugs with target genes: {len(drug_targets)}")
    print(f"  Ground truth entries: {len(ground_truth)}")
    print(f"  Unique genes with drug targeting: {len(gene_to_drugs)}")

    # Coverage analysis
    print("\n" + "=" * 60)
    print("Coverage Analysis (theoretical ceiling)")
    print("=" * 60)
    coverage = analyze_coverage(disease_genes, gene_to_drugs, ground_truth)
    print(f"  Diseases evaluated: {coverage['diseases_evaluated']}")
    print(f"  Diseases with at least 1 reachable GT drug: {coverage['diseases_with_reachable_gt']} ({100*coverage['diseases_with_reachable_gt']/coverage['diseases_evaluated']:.1f}%)")
    print(f"  Total GT drugs: {coverage['total_gt_drugs']}")
    print(f"  Reachable GT drugs: {coverage['reachable_gt_drugs']} ({100*coverage['coverage']:.1f}%)")

    # Main evaluation
    print("\n" + "=" * 60)
    print("Disease Holdout Evaluation (5-seed)")
    print("=" * 60)

    mean_r30, std_r30, n_test = disease_holdout_evaluation(
        disease_genes, gene_to_drugs, ground_truth,
        test_fraction=0.2, num_seeds=5
    )

    print(f"\n  Per-Disease R@30: {100*mean_r30:.2f}% ± {100*std_r30:.2f}%")
    print(f"  Test diseases per seed: {n_test}")

    # Compare to baselines
    print("\n" + "=" * 60)
    print("Comparison to Baselines")
    print("=" * 60)
    print(f"  Mechanism Traversal: {100*mean_r30:.2f}% ± {100*std_r30:.2f}%")
    print(f"  kNN (no-treatment):  26.06% ± 3.84% (fair)")
    print(f"  kNN (with leakage):  36.59% ± 3.90%")
    print(f"  TxGNN:               6.7-14.5% (inductive)")

    # Sample predictions
    print("\n" + "=" * 60)
    print("Sample Predictions")
    print("=" * 60)

    sample_diseases = ['MESH:D003920', 'MESH:D010300', 'MESH:D006333']  # Diabetes, Parkinson's, Heart failure
    for mesh_id in sample_diseases:
        if mesh_id in disease_genes:
            preds = predict_drugs_for_disease(mesh_id, disease_genes, gene_to_drugs)
            print(f"\n{mesh_id}:")
            print(f"  Disease genes: {len(disease_genes.get(mesh_id, []))}")
            print(f"  Candidate drugs: {len(preds)}")
            if preds:
                print(f"  Top 5 predictions (score = #genes targeted):")
                for drug, score in preds[:5]:
                    print(f"    {drug}: {score} genes")

    return {
        'r30_mean': mean_r30,
        'r30_std': std_r30,
        'coverage': coverage
    }


if __name__ == '__main__':
    results = main()
