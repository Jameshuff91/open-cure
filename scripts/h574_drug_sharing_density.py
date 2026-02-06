#!/usr/bin/env python3
"""
h574: Drug-Sharing Density as Disease Quality Signal

h572 showed embeddings cluster by drug-sharing patterns, not disease category.
For each disease, compute the average number of drugs shared between it and
its k=20 neighbors (using GT). Higher drug-sharing density = better predictions.

IMPORTANT: This signal uses GT, so it IS potentially circular for kNN.
But if it correlates with precision AFTER controlling for GT size, it captures
something about the quality of the disease's embedding neighborhood.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from production_predictor import (
    DrugRepurposingPredictor,
)


def main():
    print("=" * 70)
    print("h574: Drug-Sharing Density as Disease Quality Signal")
    print("=" * 70)

    predictor = DrugRepurposingPredictor()
    gt_diseases = [d for d in predictor.ground_truth if d in predictor.embeddings]
    print(f"\nGT diseases with embeddings: {len(gt_diseases)}")

    # Step 1: For each disease, compute drug-sharing with k=20 kNN neighbors
    print("\n--- Step 1: Compute Drug-Sharing Density ---")

    # Build disease embeddings matrix
    disease_ids = list(predictor.embeddings.keys())
    embeddings_matrix = np.array([predictor.embeddings[d] for d in disease_ids], dtype=np.float32)
    disease_id_to_idx = {d: i for i, d in enumerate(disease_ids)}

    drug_sharing = {}  # disease_id -> mean drug overlap with k=20 neighbors

    for disease_id in gt_diseases:
        if disease_id not in disease_id_to_idx:
            continue

        idx = disease_id_to_idx[disease_id]
        disease_emb = embeddings_matrix[idx].reshape(1, -1)
        sims = cosine_similarity(disease_emb, embeddings_matrix)[0]

        # Get k=20 nearest (excluding self)
        top_indices = np.argsort(-sims)
        gt_drugs = set(predictor.ground_truth.get(disease_id, []))
        gt_size = len(gt_drugs)

        neighbor_overlaps = []
        count = 0
        for ni in top_indices:
            if ni == idx:
                continue
            if count >= 20:
                break
            neighbor_id = disease_ids[ni]
            neighbor_drugs = set(predictor.ground_truth.get(neighbor_id, []))
            overlap = len(gt_drugs & neighbor_drugs)
            neighbor_overlaps.append(overlap)
            count += 1

        mean_overlap = np.mean(neighbor_overlaps) if neighbor_overlaps else 0
        drug_sharing[disease_id] = {
            'mean_overlap': mean_overlap,
            'max_overlap': max(neighbor_overlaps) if neighbor_overlaps else 0,
            'gt_size': gt_size,
        }

    print(f"Computed for {len(drug_sharing)} diseases")

    # Distribution
    overlaps = [d['mean_overlap'] for d in drug_sharing.values()]
    print(f"Drug sharing: mean={np.mean(overlaps):.2f}, median={np.median(overlaps):.2f}, "
          f"max={np.max(overlaps):.2f}")

    # Step 2: Correlate with per-disease holdout precision
    print("\n--- Step 2: Correlate with Holdout Precision ---")

    with open('data/reference/disease_holdout_precision.json') as f:
        disease_precision = json.load(f)

    both = []
    for disease_id in drug_sharing:
        entry = disease_precision.get(disease_id)
        if entry and isinstance(entry, dict):
            prec = entry.get('holdout_precision')
            gt_size = entry.get('gt_size', 0)
            if prec is not None:
                both.append({
                    'drug_sharing': drug_sharing[disease_id]['mean_overlap'],
                    'precision': float(prec),
                    'gt_size': gt_size,
                })

    if both:
        sharing = np.array([b['drug_sharing'] for b in both])
        prec = np.array([b['precision'] for b in both])
        gt = np.array([b['gt_size'] for b in both], dtype=float)

        r_sharing_prec = np.corrcoef(sharing, prec)[0, 1]
        r_gt_prec = np.corrcoef(gt, prec)[0, 1]
        r_sharing_gt = np.corrcoef(sharing, gt)[0, 1]

        print(f"r(drug_sharing, holdout_precision): {r_sharing_prec:.3f}")
        print(f"r(GT_size, holdout_precision): {r_gt_prec:.3f}")
        print(f"r(drug_sharing, GT_size): {r_sharing_gt:.3f}")
        print(f"n diseases: {len(both)}")

        # Partial correlation: drug_sharing vs precision controlling for GT size
        # Using formula: r_xy.z = (r_xy - r_xz * r_yz) / sqrt((1-r_xz^2)(1-r_yz^2))
        r_xy = r_sharing_prec
        r_xz = r_sharing_gt
        r_yz = r_gt_prec
        denom = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
        if denom > 0:
            partial_r = (r_xy - r_xz * r_yz) / denom
            print(f"Partial r(drug_sharing, precision | GT_size): {partial_r:.3f}")
        else:
            print("Cannot compute partial correlation")

    # Step 3: Bin analysis
    print("\n--- Step 3: Precision by Drug-Sharing Bin ---")

    bins = [(0, 0.5), (0.5, 2), (2, 5), (5, 10), (10, 100)]
    for lo, hi in bins:
        bin_data = [b for b in both if lo <= b['drug_sharing'] < hi]
        if bin_data:
            mean_prec = np.mean([b['precision'] for b in bin_data])
            mean_gt = np.mean([b['gt_size'] for b in bin_data])
            print(f"  [{lo:.1f}, {hi:.1f}): precision={mean_prec:.1f}%, gt_size={mean_gt:.1f}, n={len(bin_data)}")

    # Save
    results = {
        "hypothesis": "h574",
        "r_drug_sharing_precision": round(float(r_sharing_prec), 3) if both else None,
        "r_gt_size_precision": round(float(r_gt_prec), 3) if both else None,
        "r_drug_sharing_gt_size": round(float(r_sharing_gt), 3) if both else None,
        "n_diseases": len(both),
    }
    with open("data/analysis/h574_output.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to data/analysis/h574_output.json")


if __name__ == "__main__":
    main()
