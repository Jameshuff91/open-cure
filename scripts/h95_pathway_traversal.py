#!/usr/bin/env python3
"""
h95: Pathway-Level Mechanism Traversal

PURPOSE:
    h93 showed direct gene targeting fails (3.5% R@30) because drug-disease
    gene overlap is rare. But drugs and diseases may share PATHWAYS without
    sharing exact genes.

APPROACH:
    1. For each disease, get its KEGG pathway memberships (via disease genes)
    2. For each drug, get its KEGG pathway memberships (via drug targets)
    3. Score drugs by number of shared pathways with disease
    4. Evaluate R@30 on disease holdout

SUCCESS CRITERIA:
    > 5% R@30 (beat h93's 3.5%) or demonstrate useful signal as kNN booster
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"

SEEDS = [42, 123, 456, 789, 1024]


def load_pathway_data() -> tuple:
    """Load disease-pathway and drug-pathway mappings."""
    with open(REFERENCE_DIR / "pathway" / "disease_pathways.json") as f:
        disease_pathways = json.load(f)
    with open(REFERENCE_DIR / "pathway" / "drug_pathways.json") as f:
        drug_pathways = json.load(f)
    return disease_pathways, drug_pathways


def load_ground_truth() -> Dict[str, Set[str]]:
    """Load expanded ground truth."""
    with open(REFERENCE_DIR / "expanded_ground_truth.json") as f:
        gt = json.load(f)
    return {k: set(v) for k, v in gt.items()}


def normalize_drug_id(drug_id: str) -> str:
    """Extract core drug ID from drkg format."""
    if "::" in drug_id:
        return drug_id.split("::")[-1]
    return drug_id


def normalize_disease_id(disease_id: str) -> str:
    """Extract MESH ID from drkg format."""
    if "::" in disease_id:
        return disease_id.split("::")[-1]
    return disease_id


def predict_drugs_for_disease(
    disease_id: str,
    disease_pathways: Dict[str, list],
    drug_pathways: Dict[str, list],
) -> list:
    """
    Predict drugs for a disease using pathway overlap.

    Returns list of (drug_id, score) tuples sorted by score descending.
    Score = number of shared KEGG pathways.
    """
    disease_mesh = normalize_disease_id(disease_id)

    if disease_mesh not in disease_pathways:
        return []

    disease_pathway_set = set(disease_pathways[disease_mesh])

    if not disease_pathway_set:
        return []

    # Score each drug by pathway overlap
    drug_scores = []
    for drug_id, drug_paths in drug_pathways.items():
        drug_pathway_set = set(drug_paths)
        overlap = disease_pathway_set & drug_pathway_set
        if overlap:
            drug_scores.append((drug_id, len(overlap)))

    # Sort by score
    drug_scores.sort(key=lambda x: -x[1])
    return drug_scores


def evaluate_recall_at_k(predictions: list, ground_truth: Set[str], k: int = 30) -> float:
    """
    Evaluate Recall@k.

    predictions: list of (drug_id, score) tuples
    ground_truth: set of drug IDs that are correct
    """
    if not ground_truth:
        return None

    top_k_drugs = set(normalize_drug_id(d) for d, _ in predictions[:k])
    gt_drugs = set(normalize_drug_id(d) for d in ground_truth)

    hits = top_k_drugs & gt_drugs
    return len(hits) / len(gt_drugs)


def disease_holdout_evaluation(
    disease_pathways: Dict[str, list],
    drug_pathways: Dict[str, list],
    ground_truth: Dict[str, Set[str]],
    test_fraction: float = 0.2,
    num_seeds: int = 5,
) -> tuple:
    """Evaluate with disease holdout."""
    results_by_seed = []

    for seed_idx, seed in enumerate(SEEDS[:num_seeds]):
        random.seed(seed)

        # Get diseases that have both pathway data and GT
        evaluable = []
        for gt_disease, gt_drugs in ground_truth.items():
            mesh_id = normalize_disease_id(gt_disease)
            if mesh_id in disease_pathways and gt_drugs:
                evaluable.append((gt_disease, mesh_id))

        random.shuffle(evaluable)
        n_test = int(len(evaluable) * test_fraction)
        test_diseases = evaluable[:n_test]

        # Evaluate
        recalls = []
        for gt_disease, mesh_id in test_diseases:
            preds = predict_drugs_for_disease(mesh_id, disease_pathways, drug_pathways)
            recall = evaluate_recall_at_k(preds, ground_truth[gt_disease])
            if recall is not None:
                recalls.append(recall)

        if recalls:
            results_by_seed.append(np.mean(recalls))

    return np.mean(results_by_seed), np.std(results_by_seed), len(test_diseases)


def analyze_coverage(
    disease_pathways: Dict[str, list],
    drug_pathways: Dict[str, list],
    ground_truth: Dict[str, Set[str]],
) -> dict:
    """Analyze what fraction of GT drugs can be reached via pathway overlap."""
    total_gt = 0
    reachable_gt = 0
    diseases_with_reachable = 0
    diseases_evaluated = 0

    for gt_disease, gt_drugs in ground_truth.items():
        mesh_id = normalize_disease_id(gt_disease)
        if mesh_id not in disease_pathways:
            continue

        diseases_evaluated += 1
        disease_pathway_set = set(disease_pathways[mesh_id])

        # Get all drugs reachable via pathway for this disease
        reachable = set()
        for drug_id, drug_paths in drug_pathways.items():
            if disease_pathway_set & set(drug_paths):
                reachable.add(drug_id)

        # Count GT drugs that are reachable
        gt_normalized = set(normalize_drug_id(d) for d in gt_drugs)
        hits = reachable & gt_normalized

        total_gt += len(gt_normalized)
        reachable_gt += len(hits)

        if hits:
            diseases_with_reachable += 1

    return {
        'diseases_evaluated': diseases_evaluated,
        'diseases_with_reachable_gt': diseases_with_reachable,
        'total_gt': total_gt,
        'reachable_gt': reachable_gt,
        'coverage': reachable_gt / total_gt if total_gt > 0 else 0,
    }


def main():
    print("h95: Pathway-Level Mechanism Traversal")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    disease_pathways, drug_pathways = load_pathway_data()
    ground_truth = load_ground_truth()

    print(f"  Diseases with pathway data: {len(disease_pathways)}")
    print(f"  Drugs with pathway data: {len(drug_pathways)}")
    print(f"  Ground truth diseases: {len(ground_truth)}")

    # Coverage analysis
    print("\n" + "=" * 60)
    print("Coverage Analysis (theoretical ceiling)")
    print("=" * 60)
    coverage = analyze_coverage(disease_pathways, drug_pathways, ground_truth)
    print(f"  Diseases evaluated: {coverage['diseases_evaluated']}")
    print(f"  Diseases with reachable GT: {coverage['diseases_with_reachable_gt']} ({100*coverage['diseases_with_reachable_gt']/coverage['diseases_evaluated']:.1f}%)")
    print(f"  GT drugs reachable: {coverage['reachable_gt']}/{coverage['total_gt']} ({100*coverage['coverage']:.1f}%)")

    # Main evaluation
    print("\n" + "=" * 60)
    print("Disease Holdout Evaluation (5-seed)")
    print("=" * 60)

    mean_r30, std_r30, n_test = disease_holdout_evaluation(
        disease_pathways, drug_pathways, ground_truth,
        test_fraction=0.2, num_seeds=5
    )

    print(f"\n  Per-Disease R@30: {100*mean_r30:.2f}% ± {100*std_r30:.2f}%")
    print(f"  Test diseases per seed: {n_test}")

    # Compare to baselines
    print("\n" + "=" * 60)
    print("Comparison to Baselines")
    print("=" * 60)
    print(f"  Pathway traversal:    {100*mean_r30:.2f}% ± {100*std_r30:.2f}%")
    print(f"  Gene traversal (h93): 3.53% ± 0.55%")
    print(f"  kNN (no-treatment):   26.06% ± 3.84% (fair)")

    improvement = mean_r30 - 0.0353
    print(f"\n  Improvement over h93: {100*improvement:+.2f} pp")

    # Success criteria
    print("\n" + "=" * 60)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 60)
    if mean_r30 > 0.05:
        print(f"  ✓ R@30 = {100*mean_r30:.2f}% > 5% threshold")
        print("  → VALIDATED: Pathway traversal beats gene traversal")
    else:
        print(f"  ✗ R@30 = {100*mean_r30:.2f}% < 5% threshold")
        print("  → INVALIDATED: Pathway traversal doesn't significantly improve")

    # Save results
    results_file = PROJECT_ROOT / "data" / "analysis" / "h95_pathway_traversal.json"
    with open(results_file, 'w') as f:
        json.dump({
            'r30_mean': float(mean_r30),
            'r30_std': float(std_r30),
            'coverage': coverage,
            'n_test': int(n_test),
            'success': bool(mean_r30 > 0.05),
        }, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
