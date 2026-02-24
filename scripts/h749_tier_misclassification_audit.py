#!/usr/bin/env python3
"""h749: Tier Misclassification Audit

Investigates rules whose holdout precision significantly differs from tier targets:
- neurological_hierarchy_epilepsy: GOLDEN at 20% holdout (should be much lower)
- gastrointestinal: HIGH at 90.4% holdout (GOLDEN candidate?)
- infectious_hierarchy_skin_infection: HIGH at 25% holdout
- autoimmune_hierarchy_lupus: HIGH at 40% holdout
- default_freq10_nomech_r6_10: MEDIUM at 34.3% holdout
- infectious_hierarchy_pneumonia: LOW at 45.2% holdout
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
from collections import defaultdict, Counter

# Load production predictor
from src.production_predictor import ProductionPredictor

def load_expanded_gt():
    """Load expanded ground truth."""
    gt_path = 'data/reference/expanded_ground_truth.json'
    with open(gt_path) as f:
        gt_data = json.load(f)
    # Build set of (drug, disease) pairs
    gt_pairs = set()
    for entry in gt_data:
        drug = entry.get('drug_name', entry.get('drug', ''))
        disease = entry.get('disease_name', entry.get('disease', ''))
        if drug and disease:
            gt_pairs.add((drug.lower(), disease.lower()))
    return gt_pairs

def analyze_rule_predictions(predictor, rule_name, predictions):
    """Analyze predictions for a specific rule."""
    rule_preds = [p for p in predictions if getattr(p, 'confidence_reason', '') == rule_name]

    if not rule_preds:
        # Try matching on sub_reason
        rule_preds = [p for p in predictions if getattr(p, 'confidence_sub_reason', '') == rule_name]

    if not rule_preds:
        print(f"  No predictions found for rule '{rule_name}'")
        return

    print(f"\n{'='*60}")
    print(f"Rule: {rule_name}")
    print(f"Count: {len(rule_preds)}")
    print(f"Tier: {rule_preds[0].confidence_tier if rule_preds else 'N/A'}")

    # Drug distribution
    drug_counts = Counter(p.drug_name for p in rule_preds)
    print(f"\nTop drugs:")
    for drug, count in drug_counts.most_common(10):
        print(f"  {drug}: {count}")

    # Disease distribution
    disease_counts = Counter(p.disease_name for p in rule_preds)
    print(f"\nTop diseases:")
    for disease, count in disease_counts.most_common(10):
        print(f"  {disease}: {count}")

    # Category distribution
    cat_counts = Counter(getattr(p, 'disease_category', 'unknown') for p in rule_preds)
    print(f"\nCategory distribution:")
    for cat, count in cat_counts.most_common():
        print(f"  {cat}: {count}")

    # CS fraction
    cs_drugs = {'prednisone', 'prednisolone', 'methylprednisolone', 'dexamethasone',
                'hydrocortisone', 'cortisone', 'betamethasone', 'triamcinolone',
                'budesonide', 'fluticasone', 'beclomethasone', 'mometasone',
                'fluocinolone', 'clobetasol'}
    cs_count = sum(1 for p in rule_preds if any(cs in p.drug_name.lower() for cs in cs_drugs))
    print(f"\nCS fraction: {cs_count}/{len(rule_preds)} ({100*cs_count/len(rule_preds):.1f}%)")

    # Rank distribution
    ranks = [getattr(p, 'knn_rank', 99) for p in rule_preds]
    if ranks:
        print(f"\nRank distribution: mean={np.mean(ranks):.1f}, median={np.median(ranks):.0f}")
        rank_bins = Counter()
        for r in ranks:
            if r <= 5: rank_bins['R1-5'] += 1
            elif r <= 10: rank_bins['R6-10'] += 1
            elif r <= 20: rank_bins['R11-20'] += 1
            else: rank_bins['R21+'] += 1
        for bin_name in ['R1-5', 'R6-10', 'R11-20', 'R21+']:
            print(f"  {bin_name}: {rank_bins.get(bin_name, 0)}")

    # Score distribution
    scores = [getattr(p, 'knn_score', 0) for p in rule_preds]
    if scores and max(scores) > 0:
        print(f"\nScore distribution: mean={np.mean(scores):.2f}, median={np.median(scores):.1f}, min={min(scores):.2f}, max={max(scores):.2f}")

    # Mechanism support
    mech_count = sum(1 for p in rule_preds if getattr(p, 'mechanism_support', False))
    print(f"\nMechanism support: {mech_count}/{len(rule_preds)} ({100*mech_count/len(rule_preds):.1f}%)")

    # TransE consilience
    transe_count = sum(1 for p in rule_preds if getattr(p, 'transe_consilience', False))
    print(f"TransE consilience: {transe_count}/{len(rule_preds)} ({100*transe_count/len(rule_preds):.1f}%)")

    return rule_preds


def run_holdout_analysis(predictor, rule_name, seeds=[42, 123, 456, 789, 1024]):
    """Run 5-seed holdout analysis for a specific rule."""
    from scripts.h393_holdout_tier_validation import run_holdout_evaluation
    # We'll use the existing evaluator
    pass


def main():
    print("Loading predictor...")
    predictor = ProductionPredictor()
    predictor.load_models()

    print("Running predictions...")
    predictions = predictor.predict_all()

    print(f"Total predictions: {len(predictions)}")

    # Count by confidence_reason
    reason_counts = Counter(getattr(p, 'confidence_reason', 'unknown') for p in predictions)

    # Target rules to investigate
    target_rules = [
        'neurological_hierarchy_epilepsy',
        'gastrointestinal',
        'infectious_hierarchy_skin_infection',
        'autoimmune_hierarchy_lupus',
        'default_freq10_nomech_r6_10',
        'infectious_hierarchy_pneumonia',
    ]

    for rule in target_rules:
        analyze_rule_predictions(predictor, rule, predictions)

    # Also look at ALL rules sorted by count to find any we missed
    print(f"\n{'='*60}")
    print("All confidence reasons by count:")
    for reason, count in reason_counts.most_common():
        tier = None
        for p in predictions:
            if getattr(p, 'confidence_reason', '') == reason:
                tier = p.confidence_tier
                break
        print(f"  {reason}: {count} ({tier})")


if __name__ == '__main__':
    main()
