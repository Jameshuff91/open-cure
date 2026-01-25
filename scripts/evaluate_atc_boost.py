#!/usr/bin/env python3
"""
Evaluate ATC-based boosting strategies for drug repurposing predictions.

Similar to target overlap boost, this tests boosting predictions where
the drug's ATC class is relevant for the disease category.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from atc_features import ATCMapper, DISEASE_ATC_RELEVANCE


def load_ground_truth(gt_file: str = 'data/reference/expanded_ground_truth.json') -> Dict[str, List[str]]:
    """Load ground truth drug-disease pairs."""
    with open(gt_file) as f:
        data = json.load(f)

    gt = defaultdict(list)
    for pair in data:
        disease = pair.get('disease_name') or pair.get('indication_name', '')
        drug = pair.get('drug_name', '')
        if disease and drug:
            gt[disease.lower()].append(drug.lower())

    return dict(gt)


def load_predictions(pred_file: str = 'data/analysis/filtered_predictions.json') -> List[Dict]:
    """Load filtered predictions."""
    with open(pred_file) as f:
        data = json.load(f)
    return data.get('clean_predictions', [])


def evaluate_recall_at_k(
    predictions: List[Dict],
    ground_truth: Dict[str, List[str]],
    k: int = 30,
) -> Tuple[float, int, int]:
    """
    Evaluate per-drug Recall@K.

    Returns: (recall, hits, total_gt_drugs)
    """
    # Group predictions by disease
    by_disease: Dict[str, List[Dict]] = defaultdict(list)
    for pred in predictions:
        disease = pred['disease'].lower()
        by_disease[disease].append(pred)

    total_hits = 0
    total_gt_drugs = 0

    for disease, gt_drugs in ground_truth.items():
        if disease not in by_disease:
            continue

        # Sort by score descending
        disease_preds = sorted(by_disease[disease], key=lambda x: -x['score'])
        top_k_drugs = set(p['drug'].lower() for p in disease_preds[:k])

        for gt_drug in gt_drugs:
            total_gt_drugs += 1
            if gt_drug.lower() in top_k_drugs:
                total_hits += 1

    recall = total_hits / total_gt_drugs if total_gt_drugs > 0 else 0
    return recall, total_hits, total_gt_drugs


def apply_atc_boost(
    predictions: List[Dict],
    atc_mapper: ATCMapper,
    boost_factor: float = 0.1,
) -> List[Dict]:
    """
    Apply ATC-based boosting to predictions.

    If drug's ATC class is relevant for the disease, boost the score.
    """
    boosted = []
    for pred in predictions:
        new_pred = pred.copy()
        drug = pred['drug']
        disease = pred['disease']
        score = pred['score']

        mechanism_score = atc_mapper.get_mechanism_score(drug, disease)
        if mechanism_score > 0.5:
            # Drug ATC is relevant for disease
            new_pred['score'] = score * (1 + boost_factor)
            new_pred['atc_boosted'] = True
        else:
            new_pred['atc_boosted'] = False

        boosted.append(new_pred)

    return boosted


def main():
    print("Loading data...")
    predictions = load_predictions()
    ground_truth = load_ground_truth()
    atc_mapper = ATCMapper()

    print(f"Predictions: {len(predictions)}")
    print(f"Diseases with GT: {len(ground_truth)}")

    # Baseline evaluation
    print("\n" + "=" * 60)
    print("BASELINE (no ATC boost)")
    print("=" * 60)

    baseline_recall, baseline_hits, baseline_total = evaluate_recall_at_k(predictions, ground_truth)
    print(f"Recall@30: {baseline_recall:.2%} ({baseline_hits}/{baseline_total})")

    # Count how many drugs have ATC mappings
    drugs_with_atc = sum(1 for p in predictions if atc_mapper.get_atc_codes(p['drug']))
    print(f"Drugs with ATC mapping: {drugs_with_atc}/{len(predictions)} ({drugs_with_atc/len(predictions):.1%})")

    # Test different boost factors
    print("\n" + "=" * 60)
    print("ATC BOOST STRATEGIES")
    print("=" * 60)

    strategies = [
        ("boost_5%", 0.05),
        ("boost_10%", 0.10),
        ("boost_15%", 0.15),
        ("boost_20%", 0.20),
    ]

    best_strategy = None
    best_improvement = 0

    for name, boost in strategies:
        boosted = apply_atc_boost(predictions, atc_mapper, boost)
        recall, hits, total = evaluate_recall_at_k(boosted, ground_truth)
        improvement = recall - baseline_recall

        n_boosted = sum(1 for p in boosted if p.get('atc_boosted'))

        print(f"\n{name}:")
        print(f"  Recall@30: {recall:.2%} ({hits}/{total})")
        print(f"  Change: {improvement:+.2%}")
        print(f"  Predictions boosted: {n_boosted}")

        if improvement > best_improvement:
            best_improvement = improvement
            best_strategy = name

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if best_improvement > 0:
        print(f"Best strategy: {best_strategy} (+{best_improvement:.2%})")
    else:
        print("No improvement from ATC boosting")

    # Show examples of boosted predictions
    print("\n" + "=" * 60)
    print("EXAMPLES OF ATC-BOOSTED PREDICTIONS")
    print("=" * 60)

    boosted = apply_atc_boost(predictions, atc_mapper, 0.10)
    boosted_preds = [p for p in boosted if p.get('atc_boosted')]

    # Sort by score
    boosted_preds.sort(key=lambda x: -x['score'])

    for pred in boosted_preds[:20]:
        drug = pred['drug']
        disease = pred['disease']
        atc_codes = atc_mapper.get_atc_level1(drug)
        print(f"\n  {drug} -> {disease}")
        print(f"    Score: {pred['score']:.3f}")
        print(f"    ATC Level 1: {atc_codes}")


if __name__ == '__main__':
    main()
