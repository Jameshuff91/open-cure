#!/usr/bin/env python3
"""
h209: GT Coverage Analysis - Which Drug-Disease Pairs Are Blocking Predictions?

h207 showed that missing Rituximab in neighbor GT blocks prediction.
This analysis systematically identifies ALL cases where a known drug-disease pair
is NOT predicted because no kNN neighbor has it.

Key outputs:
1. For each GT pair NOT in top 30: count of kNN neighbors that have the drug
2. List of drugs with zero neighbor coverage (cannot be predicted)
3. Priority ranking for GT expansion by # blocked predictions
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, Tuple, List

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def load_data() -> Tuple[Dict, np.ndarray, List[str], Dict[str, str], Dict[str, str]]:
    """Load embeddings, ground truth, and mappings."""
    # Load embeddings
    embeddings_path = project_root / "data" / "embeddings" / "node2vec_256_named.csv"
    df = pd.read_csv(embeddings_path)
    dim_cols = [c for c in df.columns if c.startswith("dim_")]

    embeddings = {}
    for _, row in df.iterrows():
        entity = f"drkg:{row['entity']}"
        embeddings[entity] = row[dim_cols].values.astype(np.float32)

    # Load DrugBank lookup
    with open(project_root / "data" / "reference" / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)
    name_to_drug_id = {
        name.lower(): f"drkg:Compound::{db_id}"
        for db_id, name in id_to_name.items()
    }
    drug_id_to_name = {
        f"drkg:Compound::{db_id}": name
        for db_id, name in id_to_name.items()
    }

    # Load ground truth
    from production_predictor import DrugRepurposingPredictor
    predictor = DrugRepurposingPredictor(project_root)
    ground_truth = predictor.ground_truth
    disease_names = predictor.disease_names

    # Build training data structures
    train_diseases = [d for d in ground_truth if d in embeddings]
    train_embeddings = np.array(
        [embeddings[d] for d in train_diseases],
        dtype=np.float32
    )

    return ground_truth, embeddings, train_diseases, train_embeddings, drug_id_to_name, disease_names


def get_knn_neighbors(
    disease_id: str,
    embeddings: Dict,
    train_diseases: List[str],
    train_embeddings: np.ndarray,
    k: int = 20
) -> List[Tuple[str, float]]:
    """Get k nearest neighbor diseases with similarity scores."""
    if disease_id not in embeddings:
        return []

    test_emb = embeddings[disease_id].reshape(1, -1)
    sims = cosine_similarity(test_emb, train_embeddings)[0]
    top_k_idx = np.argsort(sims)[-k:][::-1]  # Descending order

    neighbors = []
    for idx in top_k_idx:
        neighbor_id = train_diseases[idx]
        if neighbor_id != disease_id:  # Exclude self
            neighbors.append((neighbor_id, float(sims[idx])))

    return neighbors[:k]


def get_knn_predictions(
    disease_id: str,
    embeddings: Dict,
    train_diseases: List[str],
    train_embeddings: np.ndarray,
    ground_truth: Dict,
    k: int = 20,
    top_n: int = 30
) -> Set[str]:
    """Get top-N predicted drugs for a disease using kNN."""
    if disease_id not in embeddings:
        return set()

    test_emb = embeddings[disease_id].reshape(1, -1)
    sims = cosine_similarity(test_emb, train_embeddings)[0]
    top_k_idx = np.argsort(sims)[-k:]

    # Aggregate drug scores
    drug_scores = defaultdict(float)
    for idx in top_k_idx:
        neighbor_disease = train_diseases[idx]
        neighbor_sim = sims[idx]
        for drug_id in ground_truth[neighbor_disease]:
            if drug_id in embeddings:
                drug_scores[drug_id] += neighbor_sim

    # Return top N drugs
    sorted_drugs = sorted(drug_scores.items(), key=lambda x: x[1], reverse=True)
    return set(drug_id for drug_id, _ in sorted_drugs[:top_n])


def analyze_blocked_predictions(
    ground_truth: Dict,
    embeddings: Dict,
    train_diseases: List[str],
    train_embeddings: np.ndarray,
    drug_id_to_name: Dict[str, str],
    disease_names: Dict[str, str],
    k: int = 20,
    top_n: int = 30
) -> Dict:
    """Analyze which GT pairs are blocked due to neighbor coverage gaps."""

    # Only analyze diseases that are in the training set AND have embeddings
    diseases_to_analyze = [d for d in ground_truth if d in embeddings]
    print(f"Analyzing {len(diseases_to_analyze)} diseases with embeddings")

    # Track blocked predictions
    blocked_pairs = []  # (disease_id, drug_id, neighbor_coverage)
    predicted_pairs = []  # (disease_id, drug_id) pairs that ARE predicted

    # Track drugs with issues
    drugs_blocked_count = defaultdict(int)  # drug_id -> count of blocked predictions
    drugs_total_gt = defaultdict(int)  # drug_id -> total GT pairs
    drugs_zero_coverage = defaultdict(int)  # drug_id -> count with 0 neighbors

    for i, disease_id in enumerate(diseases_to_analyze):
        if i % 100 == 0:
            print(f"  Processing disease {i+1}/{len(diseases_to_analyze)}")

        # Get top-30 predictions for this disease
        predicted_drugs = get_knn_predictions(
            disease_id, embeddings, train_diseases, train_embeddings,
            ground_truth, k=k, top_n=top_n
        )

        # Get kNN neighbors
        neighbors = get_knn_neighbors(
            disease_id, embeddings, train_diseases, train_embeddings, k=k
        )
        neighbor_ids = [n[0] for n in neighbors]

        # For each GT drug for this disease
        for drug_id in ground_truth[disease_id]:
            if drug_id not in embeddings:
                continue  # Drug has no embedding, skip (different issue - h206)

            drugs_total_gt[drug_id] += 1

            if drug_id in predicted_drugs:
                predicted_pairs.append((disease_id, drug_id))
            else:
                # Blocked - count neighbor coverage
                neighbor_coverage = sum(
                    1 for n_id in neighbor_ids if drug_id in ground_truth.get(n_id, set())
                )
                blocked_pairs.append((disease_id, drug_id, neighbor_coverage))
                drugs_blocked_count[drug_id] += 1
                if neighbor_coverage == 0:
                    drugs_zero_coverage[drug_id] += 1

    # Compute statistics
    total_gt_pairs = len(blocked_pairs) + len(predicted_pairs)
    prediction_rate = len(predicted_pairs) / total_gt_pairs * 100 if total_gt_pairs > 0 else 0

    # Identify drugs with most blocked predictions
    drugs_sorted_by_blocked = sorted(
        drugs_blocked_count.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Get drugs with highest zero-coverage count (most critical to add to GT)
    drugs_sorted_by_zero = sorted(
        drugs_zero_coverage.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Analyze coverage distribution
    coverage_distribution = defaultdict(int)
    for _, _, coverage in blocked_pairs:
        coverage_distribution[coverage] += 1

    # Build results
    results = {
        "summary": {
            "total_gt_pairs_analyzed": total_gt_pairs,
            "predicted_pairs": len(predicted_pairs),
            "blocked_pairs": len(blocked_pairs),
            "prediction_rate_pct": round(prediction_rate, 2),
            "unique_blocked_drugs": len(drugs_blocked_count),
            "zero_coverage_drugs": len([d for d, c in drugs_zero_coverage.items() if c > 0]),
        },
        "coverage_distribution": {
            f"coverage_{k}": v for k, v in sorted(coverage_distribution.items())
        },
        "top_blocked_drugs": [
            {
                "drug_id": drug_id,
                "drug_name": drug_id_to_name.get(drug_id, drug_id),
                "blocked_count": count,
                "total_gt": drugs_total_gt[drug_id],
                "blocked_pct": round(count / drugs_total_gt[drug_id] * 100, 1) if drugs_total_gt[drug_id] > 0 else 0,
                "zero_coverage_count": drugs_zero_coverage.get(drug_id, 0),
            }
            for drug_id, count in drugs_sorted_by_blocked[:50]
        ],
        "drugs_with_most_zero_coverage": [
            {
                "drug_id": drug_id,
                "drug_name": drug_id_to_name.get(drug_id, drug_id),
                "zero_coverage_count": count,
                "blocked_count": drugs_blocked_count.get(drug_id, 0),
                "total_gt": drugs_total_gt[drug_id],
            }
            for drug_id, count in drugs_sorted_by_zero[:30]
        ],
        "sample_blocked_pairs": [
            {
                "disease_id": d_id,
                "disease_name": disease_names.get(d_id, d_id),
                "drug_id": drug_id,
                "drug_name": drug_id_to_name.get(drug_id, drug_id),
                "neighbor_coverage": coverage,
            }
            for d_id, drug_id, coverage in blocked_pairs[:100]
        ],
    }

    return results


def main():
    print("h209: GT Coverage Analysis - Which Predictions Are Blocked?")
    print("=" * 60)

    # Load data
    print("\n1. Loading data...")
    ground_truth, embeddings, train_diseases, train_embeddings, drug_id_to_name, disease_names = load_data()
    print(f"   - {len(ground_truth)} diseases in GT")
    print(f"   - {len(embeddings)} entities with embeddings")
    print(f"   - {len(train_diseases)} training diseases")

    # Run analysis
    print("\n2. Analyzing blocked predictions...")
    results = analyze_blocked_predictions(
        ground_truth, embeddings, train_diseases, train_embeddings,
        drug_id_to_name, disease_names
    )

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    s = results["summary"]
    print(f"Total GT pairs analyzed: {s['total_gt_pairs_analyzed']}")
    print(f"Predicted pairs (in top 30): {s['predicted_pairs']} ({s['prediction_rate_pct']}%)")
    print(f"Blocked pairs (NOT in top 30): {s['blocked_pairs']}")
    print(f"Unique drugs with blocked predictions: {s['unique_blocked_drugs']}")
    print(f"Drugs with ZERO neighbor coverage: {s['zero_coverage_drugs']}")

    print("\n" + "-" * 60)
    print("COVERAGE DISTRIBUTION (blocked pairs by # neighbors with drug)")
    print("-" * 60)
    for k, v in sorted(results["coverage_distribution"].items()):
        coverage = int(k.split("_")[1])
        pct = v / s["blocked_pairs"] * 100 if s["blocked_pairs"] > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"  {coverage:2d} neighbors: {v:5d} ({pct:5.1f}%) {bar}")

    print("\n" + "-" * 60)
    print("TOP 20 DRUGS WITH MOST BLOCKED PREDICTIONS")
    print("-" * 60)
    print(f"{'Drug Name':<30} {'Blocked':>8} {'Zero Cov':>8} {'Total GT':>8} {'Block %':>8}")
    for d in results["top_blocked_drugs"][:20]:
        print(f"{d['drug_name'][:30]:<30} {d['blocked_count']:>8} {d['zero_coverage_count']:>8} {d['total_gt']:>8} {d['blocked_pct']:>7.1f}%")

    print("\n" + "-" * 60)
    print("TOP 20 DRUGS WITH MOST ZERO-COVERAGE BLOCKED PAIRS")
    print("-" * 60)
    print("(These drugs CANNOT be predicted for certain diseases due to no neighbor having them)")
    print(f"{'Drug Name':<30} {'Zero Cov':>8} {'Blocked':>8} {'Total GT':>8}")
    for d in results["drugs_with_most_zero_coverage"][:20]:
        print(f"{d['drug_name'][:30]:<30} {d['zero_coverage_count']:>8} {d['blocked_count']:>8} {d['total_gt']:>8}")

    # Save results
    output_path = project_root / "data" / "analysis" / "h209_gt_coverage_analysis.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    results = main()
