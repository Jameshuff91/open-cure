#!/usr/bin/env python3
"""
Evaluate TxGNN predictions locally using saved predictions file.
No GPU needed - uses pre-computed predictions from txgnn_predictions_final.csv

This gives us the TRUE TxGNN performance on our ground truth.
"""

import json
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple

def load_ground_truth(filepath: str = 'data/reference/everycure_gt_for_txgnn.json') -> Dict:
    """Load Every Cure ground truth."""
    with open(filepath, 'r') as f:
        return json.load(f)

def load_drugbank_lookup(filepath: str = 'data/reference/drugbank_lookup.json') -> Dict[str, str]:
    """Load DrugBank ID -> name mapping and create reverse mapping."""
    with open(filepath, 'r') as f:
        drugbank = json.load(f)
    # Create name -> ID mapping (lowercase)
    name_to_id = {v.lower(): k for k, v in drugbank.items() if isinstance(v, str)}
    return name_to_id

def load_txgnn_predictions(filepath: str = 'data/reference/txgnn_predictions_final.csv') -> pd.DataFrame:
    """Load TxGNN predictions."""
    return pd.read_csv(filepath)

def evaluate_recall_at_k(
    gt: Dict,
    predictions: pd.DataFrame,
    drug_name_to_id: Dict[str, str],
    k: int = 30
) -> Tuple[float, Dict]:
    """
    Evaluate Recall@K for TxGNN predictions.

    Returns:
        recall_at_k: Overall recall
        details: Per-disease details
    """
    hits = 0
    total = 0
    all_ranks = []
    details = []

    diseases_evaluated = 0
    diseases_skipped = 0
    drugs_matched = 0
    drugs_unmatched = 0

    for disease_name, disease_data in gt.items():
        disease_lower = disease_name.lower()
        gt_drugs = [d['name'].lower() for d in disease_data['drugs']]

        # Get predictions for this disease
        disease_preds = predictions[predictions['disease_name'].str.lower() == disease_lower]

        if len(disease_preds) == 0:
            diseases_skipped += 1
            continue

        diseases_evaluated += 1

        # Get top K drug IDs
        top_k_drugs = set(disease_preds.head(k)['drug_id'].tolist())
        all_pred_drugs = disease_preds['drug_id'].tolist()

        for drug_name in gt_drugs:
            # Convert drug name to DrugBank ID
            drug_id = drug_name_to_id.get(drug_name)

            if drug_id is None:
                drugs_unmatched += 1
                continue

            drugs_matched += 1
            total += 1

            # Check if in top K
            if drug_id in top_k_drugs:
                hits += 1

            # Find rank
            if drug_id in all_pred_drugs:
                rank = all_pred_drugs.index(drug_id) + 1
            else:
                rank = len(all_pred_drugs) + 1

            all_ranks.append(rank)

            details.append({
                'disease': disease_name,
                'drug': drug_name,
                'drug_id': drug_id,
                'rank': rank,
                'hit_at_k': drug_id in top_k_drugs
            })

    recall_at_k = hits / total if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Diseases evaluated: {diseases_evaluated}")
    print(f"Diseases skipped: {diseases_skipped}")
    print(f"Drugs matched: {drugs_matched}")
    print(f"Drugs unmatched: {drugs_unmatched}")
    print(f"\nRecall@{k}: {recall_at_k:.1%} ({hits}/{total})")

    if all_ranks:
        print(f"Mean rank of GT drugs: {sum(all_ranks)/len(all_ranks):.0f}")
        sorted_ranks = sorted(all_ranks)
        print(f"Median rank: {sorted_ranks[len(sorted_ranks)//2]}")
        print(f"Best rank: {min(all_ranks)}")
        print(f"Worst rank: {max(all_ranks)}")

        # Distribution
        print(f"\nRank distribution:")
        for threshold in [10, 30, 50, 100, 500, 1000]:
            count = sum(1 for r in all_ranks if r <= threshold)
            print(f"  Top {threshold}: {count}/{len(all_ranks)} ({count/len(all_ranks)*100:.1f}%)")

    return recall_at_k, details


def analyze_by_disease_category(details: List[Dict]) -> None:
    """Analyze performance by disease."""
    by_disease = defaultdict(list)
    for d in details:
        by_disease[d['disease']].append(d)

    # Calculate per-disease recall
    disease_recalls = []
    for disease, drugs in by_disease.items():
        hits = sum(1 for d in drugs if d['hit_at_k'])
        recall = hits / len(drugs) if drugs else 0
        mean_rank = sum(d['rank'] for d in drugs) / len(drugs) if drugs else 0
        disease_recalls.append({
            'disease': disease,
            'recall_at_30': recall,
            'mean_rank': mean_rank,
            'n_drugs': len(drugs)
        })

    # Sort by recall
    disease_recalls.sort(key=lambda x: x['recall_at_30'], reverse=True)

    print(f"\n{'='*60}")
    print(f"TOP 20 DISEASES BY RECALL@30")
    print(f"{'='*60}")
    for dr in disease_recalls[:20]:
        print(f"{dr['disease'][:40]:<40} R@30={dr['recall_at_30']:.0%} (n={dr['n_drugs']}, mean_rank={dr['mean_rank']:.0f})")

    print(f"\n{'='*60}")
    print(f"BOTTOM 20 DISEASES BY RECALL@30")
    print(f"{'='*60}")
    for dr in disease_recalls[-20:]:
        print(f"{dr['disease'][:40]:<40} R@30={dr['recall_at_30']:.0%} (n={dr['n_drugs']}, mean_rank={dr['mean_rank']:.0f})")

    # How many diseases have >0% recall?
    diseases_with_hits = sum(1 for dr in disease_recalls if dr['recall_at_30'] > 0)
    print(f"\nDiseases with at least 1 hit in top 30: {diseases_with_hits}/{len(disease_recalls)}")


def main():
    print("Loading data...")
    gt = load_ground_truth()
    print(f"Ground truth: {len(gt)} diseases")

    drug_name_to_id = load_drugbank_lookup()
    print(f"Drug name mappings: {len(drug_name_to_id)}")

    predictions = load_txgnn_predictions()
    print(f"TxGNN predictions: {len(predictions)} rows")

    # Evaluate
    recall, details = evaluate_recall_at_k(gt, predictions, drug_name_to_id, k=30)

    # Analyze by disease
    analyze_by_disease_category(details)

    # Save detailed results
    output_file = 'data/reference/txgnn_local_evaluation.json'
    with open(output_file, 'w') as f:
        json.dump({
            'recall_at_30': recall,
            'details': details
        }, f, indent=2)
    print(f"\nDetailed results saved to {output_file}")


if __name__ == '__main__':
    main()
