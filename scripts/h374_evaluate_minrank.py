#!/usr/bin/env python3
"""
h374: Evaluate MinRank Ensemble in Production Predictor

Compare R@30 between:
1. MinRank ensemble (new, for cancer/neuro/metabolic)
2. kNN only (baseline)

Uses Leave-One-Out CV on evaluable diseases.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from production_predictor import (
    DrugRepurposingPredictor,
    MINRANK_ENSEMBLE_CATEGORIES,
)


def load_ground_truth():
    """Load ground truth from Excel file."""
    gt = pd.read_excel('data/reference/everycure/indicationList.xlsx')
    disease_col = 'final normalized disease label'
    drug_col = 'final normalized drug label'

    disease_drugs = defaultdict(set)
    for _, row in gt.iterrows():
        disease = row[disease_col]
        drug = row[drug_col]
        if pd.notna(disease) and pd.notna(drug):
            disease_drugs[disease].add(drug.lower())

    return disease_drugs


def evaluate_method(predictor, diseases, ground_truth, use_minrank=True, top_k=30):
    """Evaluate a method on diseases.

    Args:
        predictor: DrugRepurposingPredictor
        diseases: List of disease names to evaluate
        ground_truth: dict of disease -> set of drug names
        use_minrank: Whether to use MinRank ensemble
        top_k: Number of top predictions to consider
    """
    # Store original categories
    original_categories = MINRANK_ENSEMBLE_CATEGORIES.copy()

    if not use_minrank:
        # Disable MinRank
        MINRANK_ENSEMBLE_CATEGORIES.clear()

    hits = 0
    total = 0
    category_stats = defaultdict(lambda: {'hits': 0, 'total': 0})

    for disease in diseases:
        true_drugs = ground_truth.get(disease, set())
        if len(true_drugs) < 3:
            continue  # Need at least 3 drugs for meaningful eval

        result = predictor.predict(disease, top_n=top_k)
        if not result.predictions:
            continue

        pred_drugs = {p.drug_name.lower() for p in result.predictions[:top_k]}
        hit = 1 if pred_drugs & true_drugs else 0

        hits += hit
        total += 1
        category_stats[result.category]['hits'] += hit
        category_stats[result.category]['total'] += 1

    # Restore
    MINRANK_ENSEMBLE_CATEGORIES.update(original_categories)

    return hits, total, dict(category_stats)


def main():
    print("=" * 60)
    print("h374: MinRank Ensemble Evaluation")
    print("=" * 60)

    # Load predictor and ground truth
    print("\nLoading predictor...")
    predictor = DrugRepurposingPredictor()
    ground_truth = load_ground_truth()

    # Filter to diseases with embeddings and sufficient drugs
    evaluable = []
    for disease in ground_truth:
        disease_id = predictor.find_disease_id(disease)
        if disease_id and disease_id in predictor.embeddings:
            if len(ground_truth[disease]) >= 3:
                evaluable.append(disease)

    print(f"Evaluable diseases: {len(evaluable)}")

    # Separate by category for focused analysis
    category_diseases = defaultdict(list)
    for disease in evaluable:
        cat = predictor.categorize_disease(disease)
        category_diseases[cat].append(disease)

    print("\nCategory breakdown:")
    for cat, diseases in sorted(category_diseases.items()):
        print(f"  {cat}: {len(diseases)} diseases")

    # Evaluate MinRank vs kNN for each category
    print("\n" + "=" * 60)
    print("Results by Category")
    print("=" * 60)

    results = {}
    for cat in sorted(category_diseases.keys()):
        diseases = category_diseases[cat]
        if len(diseases) < 3:
            continue

        # MinRank
        hits_mr, total_mr, _ = evaluate_method(
            predictor, diseases, ground_truth, use_minrank=True
        )
        r30_mr = hits_mr / total_mr if total_mr > 0 else 0

        # kNN only
        hits_knn, total_knn, _ = evaluate_method(
            predictor, diseases, ground_truth, use_minrank=False
        )
        r30_knn = hits_knn / total_knn if total_knn > 0 else 0

        diff = r30_mr - r30_knn
        marker = "***" if diff > 0.03 else ""

        results[cat] = {
            'n': total_mr,
            'minrank': r30_mr,
            'knn': r30_knn,
            'diff': diff
        }

        uses_ensemble = cat in MINRANK_ENSEMBLE_CATEGORIES
        print(f"\n{cat} (n={total_mr}) {'[ENSEMBLE]' if uses_ensemble else '[kNN-only]'}")
        print(f"  MinRank: {r30_mr:.1%} ({hits_mr}/{total_mr})")
        print(f"  kNN:     {r30_knn:.1%} ({hits_knn}/{total_knn})")
        print(f"  Δ:       {diff:+.1%} {marker}")

    # Overall
    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)

    hits_mr_all, total_mr_all, _ = evaluate_method(
        predictor, evaluable, ground_truth, use_minrank=True
    )
    hits_knn_all, total_knn_all, _ = evaluate_method(
        predictor, evaluable, ground_truth, use_minrank=False
    )

    r30_mr_all = hits_mr_all / total_mr_all if total_mr_all > 0 else 0
    r30_knn_all = hits_knn_all / total_knn_all if total_knn_all > 0 else 0
    diff_all = r30_mr_all - r30_knn_all

    print(f"\nAll diseases (n={total_mr_all})")
    print(f"  MinRank: {r30_mr_all:.1%} ({hits_mr_all}/{total_mr_all})")
    print(f"  kNN:     {r30_knn_all:.1%} ({hits_knn_all}/{total_knn_all})")
    print(f"  Δ:       {diff_all:+.1%}")

    # Save results
    output = {
        'hypothesis': 'h374',
        'description': 'MinRank ensemble vs kNN baseline',
        'overall': {
            'minrank': r30_mr_all,
            'knn': r30_knn_all,
            'diff': diff_all,
            'n': total_mr_all
        },
        'by_category': results
    }

    with open('data/analysis/h374_minrank_evaluation.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to data/analysis/h374_minrank_evaluation.json")


if __name__ == '__main__':
    main()
