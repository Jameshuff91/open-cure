#!/usr/bin/env python3
"""
h108: Confidence Feature - Drug Training Frequency

PURPOSE:
    Drugs seen frequently in training (many GT indications) may generalize better.
    Novel or rare drugs in training may produce unreliable predictions.
    Simple count of training appearances as confidence proxy.

APPROACH:
    1. Count how many training diseases each drug appears in (GT frequency)
    2. For kNN predictions, record drug's training frequency
    3. Stratify: HIGH frequency drugs vs LOW frequency drugs
    4. Compare precision between frequent and rare drug predictions

SUCCESS CRITERIA:
    Frequent-drug predictions have 5%+ better precision.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, List

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from disease_name_matcher import DiseaseMatcher, load_mesh_mappings

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"

SEEDS = [42, 123, 456, 789, 1024]


def load_node2vec_embeddings() -> Dict[str, np.ndarray]:
    """Load Node2Vec embeddings."""
    embeddings_path = EMBEDDINGS_DIR / "node2vec_256_named.csv"
    df = pd.read_csv(embeddings_path)
    dim_cols = [c for c in df.columns if c.startswith("dim_")]
    embeddings: Dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        entity = f"drkg:{row['entity']}"
        embeddings[entity] = row[dim_cols].values.astype(np.float32)
    return embeddings


def load_drugbank_lookup() -> Dict[str, str]:
    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)
    name_to_id = {name.lower(): f"drkg:Compound::{db_id}" for db_id, name in id_to_name.items()}
    return name_to_id


def load_mesh_mappings_from_file() -> Dict[str, str]:
    mesh_path = REFERENCE_DIR / "mesh_mappings_from_agents.json"
    if not mesh_path.exists():
        return {}
    with open(mesh_path) as f:
        mesh_data = json.load(f)
    mesh_mappings: Dict[str, str] = {}
    for batch_data in mesh_data.values():
        if isinstance(batch_data, dict):
            for disease_name, mesh_id in batch_data.items():
                if mesh_id:
                    mesh_str = str(mesh_id)
                    if mesh_str.startswith("D") or mesh_str.startswith("C"):
                        mesh_mappings[disease_name.lower()] = f"drkg:Disease::MESH:{mesh_str}"
    return mesh_mappings


def load_ground_truth(
    mesh_mappings: Dict[str, str],
    name_to_drug_id: Dict[str, str],
) -> Dict[str, Set[str]]:
    df = pd.read_excel(REFERENCE_DIR / "everycure" / "indicationList.xlsx")
    fuzzy_mappings = load_mesh_mappings()
    matcher = DiseaseMatcher(fuzzy_mappings)

    gt: Dict[str, Set[str]] = defaultdict(set)
    for _, row in df.iterrows():
        disease = str(row.get("disease name", "")).strip()
        drug = str(row.get("final normalized drug label", "")).strip()
        if not disease or not drug:
            continue

        disease_id = matcher.get_mesh_id(disease)
        if not disease_id:
            disease_id = mesh_mappings.get(disease.lower())
        if not disease_id:
            continue

        drug_id = name_to_drug_id.get(drug.lower())
        if drug_id:
            gt[disease_id].add(drug_id)

    return dict(gt)


def compute_drug_frequencies(ground_truth: Dict[str, Set[str]]) -> Dict[str, int]:
    """Count how many diseases each drug treats in GT."""
    drug_freq: Dict[str, int] = defaultdict(int)
    for disease_id, drugs in ground_truth.items():
        for drug_id in drugs:
            drug_freq[drug_id] += 1
    return dict(drug_freq)


def run_knn_with_frequency(
    emb_dict: Dict[str, np.ndarray],
    train_gt: Dict[str, Set[str]],
    test_gt: Dict[str, Set[str]],
    drug_freq: Dict[str, int],  # Overall drug frequencies
    k: int = 20,
) -> List[Dict]:
    """Run kNN and record drug training frequency for each prediction."""
    train_disease_list = [d for d in train_gt if d in emb_dict]
    train_disease_embs = np.array([emb_dict[d] for d in train_disease_list], dtype=np.float32)

    # Compute drug frequency in TRAINING set only
    train_drug_freq: Dict[str, int] = defaultdict(int)
    for disease_id, drugs in train_gt.items():
        for drug_id in drugs:
            train_drug_freq[drug_id] += 1

    results = []

    for disease_id in test_gt:
        if disease_id not in emb_dict:
            continue
        gt_drugs = {d for d in test_gt[disease_id] if d in emb_dict}
        if not gt_drugs:
            continue

        # Find k nearest training diseases
        test_emb = emb_dict[disease_id].reshape(1, -1)
        sims = cosine_similarity(test_emb, train_disease_embs)[0]
        top_k_idx = np.argsort(sims)[-k:]

        # Count drug frequency
        drug_counts: Dict[str, float] = defaultdict(float)
        for idx in top_k_idx:
            neighbor_disease = train_disease_list[idx]
            neighbor_sim = sims[idx]
            for drug_id in train_gt[neighbor_disease]:
                if drug_id in emb_dict:
                    drug_counts[drug_id] += neighbor_sim

        if not drug_counts:
            continue

        # Get top 30
        sorted_drugs = sorted(drug_counts.items(), key=lambda x: x[1], reverse=True)[:30]

        for drug_id, score in sorted_drugs:
            # Get training frequency for this drug
            freq = train_drug_freq.get(drug_id, 0)
            is_hit = drug_id in gt_drugs

            results.append({
                'disease': disease_id,
                'drug': drug_id,
                'knn_score': score,
                'train_frequency': freq,
                'is_hit': is_hit,
            })

    return results


def main():
    print("h108: Drug Training Frequency as Confidence Feature")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    emb_dict = load_node2vec_embeddings()
    name_to_drug_id = load_drugbank_lookup()
    mesh_mappings = load_mesh_mappings_from_file()
    ground_truth = load_ground_truth(mesh_mappings, name_to_drug_id)

    print(f"  Embeddings: {len(emb_dict)}")
    print(f"  Ground truth diseases: {len(ground_truth)}")

    # Compute overall drug frequencies
    drug_freq = compute_drug_frequencies(ground_truth)
    print(f"  Unique drugs in GT: {len(drug_freq)}")
    freq_values = list(drug_freq.values())
    print(f"  Drug frequency stats: min={min(freq_values)}, max={max(freq_values)}, "
          f"mean={np.mean(freq_values):.1f}, median={np.median(freq_values):.1f}")

    # Multi-seed evaluation
    print("\n" + "=" * 70)
    print("Multi-Seed Evaluation (5 seeds)")
    print("=" * 70)

    all_results = []

    for seed in SEEDS:
        np.random.seed(seed)
        diseases = list(ground_truth.keys())
        np.random.shuffle(diseases)
        n_test = len(diseases) // 5
        test_diseases = set(diseases[:n_test])
        train_diseases = set(diseases[n_test:])

        train_gt = {d: ground_truth[d] for d in train_diseases}
        test_gt = {d: ground_truth[d] for d in test_diseases}

        seed_results = run_knn_with_frequency(
            emb_dict, train_gt, test_gt, drug_freq, k=20
        )
        all_results.extend(seed_results)
        print(f"  Seed {seed}: {len(seed_results)} predictions")

    # Analyze by drug training frequency
    print("\n" + "=" * 70)
    print("Precision by Drug Training Frequency")
    print("=" * 70)

    df = pd.DataFrame(all_results)
    print(f"\nTotal predictions: {len(df)}")
    print(f"Training frequency stats: min={df['train_frequency'].min()}, "
          f"max={df['train_frequency'].max()}, mean={df['train_frequency'].mean():.1f}")

    # Tertiles by frequency
    df_sorted = df.sort_values('train_frequency')
    n = len(df_sorted)
    low_freq = df_sorted.iloc[:n//3]  # Rare drugs
    mid_freq = df_sorted.iloc[n//3:2*n//3]  # Medium
    high_freq = df_sorted.iloc[2*n//3:]  # Frequent drugs

    def calc_precision(subset):
        return subset['is_hit'].sum() / len(subset) if len(subset) > 0 else 0

    prec_low = calc_precision(low_freq)
    prec_mid = calc_precision(mid_freq)
    prec_high = calc_precision(high_freq)

    print(f"\nLOW frequency (rare drugs, {len(low_freq)} predictions):")
    print(f"  Mean training frequency: {low_freq['train_frequency'].mean():.1f}")
    print(f"  Precision: {100*prec_low:.2f}%")

    print(f"\nMEDIUM frequency ({len(mid_freq)} predictions):")
    print(f"  Mean training frequency: {mid_freq['train_frequency'].mean():.1f}")
    print(f"  Precision: {100*prec_mid:.2f}%")

    print(f"\nHIGH frequency (common drugs, {len(high_freq)} predictions):")
    print(f"  Mean training frequency: {high_freq['train_frequency'].mean():.1f}")
    print(f"  Precision: {100*prec_high:.2f}%")

    # Key comparison
    print("\n" + "=" * 70)
    print("KEY COMPARISON")
    print("=" * 70)
    diff = prec_high - prec_low
    print(f"  HIGH frequency precision: {100*prec_high:.2f}%")
    print(f"  LOW frequency precision:  {100*prec_low:.2f}%")
    print(f"  Difference: {100*diff:+.2f} pp")

    # Correlation
    correlation = np.corrcoef(df['train_frequency'].values, df['is_hit'].values)[0, 1]
    print(f"\n  Correlation(train_frequency, is_hit): {correlation:.4f}")

    # Additional: Binary split at median
    median_freq = df['train_frequency'].median()
    above_median = df[df['train_frequency'] > median_freq]
    below_median = df[df['train_frequency'] <= median_freq]

    prec_above = calc_precision(above_median)
    prec_below = calc_precision(below_median)

    print(f"\n--- Binary split at median (frequency > {median_freq}) ---")
    print(f"  ABOVE median: {100*prec_above:.2f}% precision ({len(above_median)} predictions)")
    print(f"  BELOW median: {100*prec_below:.2f}% precision ({len(below_median)} predictions)")
    print(f"  Difference: {100*(prec_above - prec_below):+.2f} pp")

    # Success criteria check
    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 70)
    success = diff >= 0.05
    if success:
        print(f"  ✓ HIGH frequency precision is +{100*diff:.1f} pp better (>= 5 pp)")
        print("  → VALIDATED: Drug training frequency is a valid confidence signal")
    else:
        if diff > 0:
            print(f"  ~ HIGH frequency precision is +{100*diff:.1f} pp better (< 5 pp)")
            print("  → PARTIALLY VALIDATED: Some improvement but below threshold")
        else:
            print(f"  ✗ HIGH frequency precision is {100*diff:+.1f} pp vs LOW frequency")
            print("  → INVALIDATED: Drug training frequency doesn't improve precision")

    # Save results
    results = {
        'high_freq_precision': float(prec_high),
        'low_freq_precision': float(prec_low),
        'mid_freq_precision': float(prec_mid),
        'difference_pp': float(diff * 100),
        'correlation': float(correlation),
        'above_median_precision': float(prec_above),
        'below_median_precision': float(prec_below),
        'median_frequency': float(median_freq),
        'success': bool(success),
        'n_predictions': len(df),
    }

    results_file = PROJECT_ROOT / "data" / "analysis" / "h108_drug_frequency.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
