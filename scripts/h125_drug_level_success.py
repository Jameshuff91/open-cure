#!/usr/bin/env python3
"""
h125: Drug-Level Success Prediction

PURPOSE:
    Current confidence features operate at disease or prediction level.
    But some DRUGS may be inherently more reliable (e.g., well-studied drugs
    with clear mechanisms). A drug-level reliability score could improve precision.

METHODOLOGY:
    1. For each drug, compute historical hit rate using LOO-CV within training data
    2. Test correlation between drug hit rate and being a hit on test diseases
    3. Compare to existing signals (train_frequency, target_count)
    4. Check if drug-level signal is independent

SUCCESS CRITERIA:
    Drug-level hit rate achieves r > 0.15 with hits (independent of frequency)
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr, pearsonr
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from disease_name_matcher import DiseaseMatcher, load_mesh_mappings

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"

SEEDS = [42, 123, 456, 789, 1024]


def load_node2vec_embeddings() -> Dict[str, np.ndarray]:
    embeddings_path = EMBEDDINGS_DIR / "node2vec_256_named.csv"
    df = pd.read_csv(embeddings_path)
    dim_cols = [c for c in df.columns if c.startswith("dim_")]
    embeddings: Dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        entity = f"drkg:{row['entity']}"
        embeddings[entity] = row[dim_cols].values.astype(np.float32)
    return embeddings


def load_drugbank_lookup():
    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)
    name_to_id = {name.lower(): f"drkg:Compound::{db_id}" for db_id, name in id_to_name.items()}
    id_to_name_map = {f"drkg:Compound::{db_id}": name for db_id, name in id_to_name.items()}
    return name_to_id, id_to_name_map


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


def load_ground_truth(mesh_mappings, name_to_drug_id):
    df = pd.read_excel(REFERENCE_DIR / "everycure" / "indicationList.xlsx")
    fuzzy_mappings = load_mesh_mappings()
    matcher = DiseaseMatcher(fuzzy_mappings)

    gt: Dict[str, Set[str]] = defaultdict(set)
    disease_names: Dict[str, str] = {}

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

        disease_names[disease_id] = disease
        drug_id = name_to_drug_id.get(drug.lower())
        if drug_id:
            gt[disease_id].add(drug_id)

    return dict(gt), disease_names


def load_drug_targets() -> Dict[str, Set[str]]:
    targets_path = REFERENCE_DIR / "drug_targets.json"
    if not targets_path.exists():
        return {}
    with open(targets_path) as f:
        drug_targets = json.load(f)
    return {f"drkg:Compound::{k}": set(v) for k, v in drug_targets.items()}


def compute_drug_hit_rates_loo(train_gt, emb_dict, k=20):
    """
    Compute drug-level hit rates using leave-one-disease-out cross-validation.
    For each training disease, predict drugs using kNN on remaining training diseases,
    then track which drugs hit.
    """
    train_disease_list = [d for d in train_gt if d in emb_dict]
    if len(train_disease_list) < 2:
        return {}

    disease_embs = np.array([emb_dict[d] for d in train_disease_list], dtype=np.float32)
    sims = cosine_similarity(disease_embs)

    # Track predictions and hits per drug
    drug_predictions = defaultdict(int)  # times drug was in top 30
    drug_hits = defaultdict(int)  # times drug was in GT AND top 30

    for i, disease_id in enumerate(train_disease_list):
        gt_drugs = {d for d in train_gt[disease_id] if d in emb_dict}
        if not gt_drugs:
            continue

        # Get k nearest neighbors (excluding self)
        neighbor_sims = sims[i].copy()
        neighbor_sims[i] = -1  # Exclude self
        top_k_idx = np.argsort(neighbor_sims)[-k:]

        # Count drug recommendations from neighbors
        drug_counts = defaultdict(float)
        for idx in top_k_idx:
            neighbor_disease = train_disease_list[idx]
            neighbor_sim = neighbor_sims[idx]
            for drug_id in train_gt[neighbor_disease]:
                if drug_id in emb_dict:
                    drug_counts[drug_id] += neighbor_sim

        if not drug_counts:
            continue

        # Top 30 predictions
        sorted_drugs = sorted(drug_counts.items(), key=lambda x: x[1], reverse=True)[:30]

        for drug_id, _ in sorted_drugs:
            drug_predictions[drug_id] += 1
            if drug_id in gt_drugs:
                drug_hits[drug_id] += 1

    # Compute hit rates
    drug_hit_rates = {}
    for drug_id in drug_predictions:
        if drug_predictions[drug_id] >= 3:  # Require at least 3 predictions
            drug_hit_rates[drug_id] = drug_hits[drug_id] / drug_predictions[drug_id]

    return drug_hit_rates


def run_knn_with_drug_features(emb_dict, train_gt, test_gt, drug_hit_rates, drug_targets, k=20):
    """Run kNN and collect predictions with drug-level features."""
    train_disease_list = [d for d in train_gt if d in emb_dict]
    train_disease_embs = np.array([emb_dict[d] for d in train_disease_list], dtype=np.float32)

    # Compute drug training frequency
    drug_train_freq: Dict[str, int] = defaultdict(int)
    for disease_id, drugs in train_gt.items():
        for drug_id in drugs:
            drug_train_freq[drug_id] += 1

    results = []

    for disease_id in test_gt:
        if disease_id not in emb_dict:
            continue
        gt_drugs = {d for d in test_gt[disease_id] if d in emb_dict}
        if not gt_drugs:
            continue

        test_emb = emb_dict[disease_id].reshape(1, -1)
        sims = cosine_similarity(test_emb, train_disease_embs)[0]
        top_k_idx = np.argsort(sims)[-k:]

        drug_counts: Dict[str, float] = defaultdict(float)
        for idx in top_k_idx:
            neighbor_disease = train_disease_list[idx]
            neighbor_sim = sims[idx]
            for drug_id in train_gt[neighbor_disease]:
                if drug_id in emb_dict:
                    drug_counts[drug_id] += neighbor_sim

        if not drug_counts:
            continue

        sorted_drugs = sorted(drug_counts.items(), key=lambda x: x[1], reverse=True)[:30]

        for rank, (drug_id, score) in enumerate(sorted_drugs, 1):
            is_hit = drug_id in gt_drugs
            train_freq = drug_train_freq.get(drug_id, 0)
            n_targets = len(drug_targets.get(drug_id, set()))

            # Drug-level hit rate from training
            drug_hr = drug_hit_rates.get(drug_id, 0.0)

            results.append({
                'drug_id': drug_id,
                'disease_id': disease_id,
                'rank': rank,
                'train_freq': train_freq,
                'n_targets': n_targets,
                'drug_hit_rate': drug_hr,
                'is_hit': int(is_hit),
            })

    return results


def main():
    print("h125: Drug-Level Success Prediction")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    emb_dict = load_node2vec_embeddings()
    name_to_drug_id, drug_names = load_drugbank_lookup()
    mesh_mappings = load_mesh_mappings_from_file()
    ground_truth, disease_names = load_ground_truth(mesh_mappings, name_to_drug_id)
    drug_targets = load_drug_targets()

    print(f"  Embeddings: {len(emb_dict)}")
    print(f"  Ground truth diseases: {len(ground_truth)}")
    print(f"  Drugs with targets: {len(drug_targets)}")

    # Collect predictions
    print("\n" + "=" * 70)
    print("Collecting predictions across 5 seeds")
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

        # Compute drug hit rates from training data (LOO-CV)
        drug_hit_rates = compute_drug_hit_rates_loo(train_gt, emb_dict, k=20)
        print(f"  Seed {seed}: {len(drug_hit_rates)} drugs with hit rates")

        seed_results = run_knn_with_drug_features(
            emb_dict, train_gt, test_gt, drug_hit_rates, drug_targets, k=20
        )
        all_results.extend(seed_results)

    df = pd.DataFrame(all_results)
    print(f"\nTotal predictions: {len(df)}")
    print(f"Base hit rate: {df['is_hit'].mean()*100:.2f}%")

    # Filter to predictions with drug hit rate data
    df_with_hr = df[df['drug_hit_rate'] > 0]
    print(f"Predictions with drug hit rate: {len(df_with_hr)} ({len(df_with_hr)/len(df)*100:.1f}%)")

    # Correlation analysis
    print("\n" + "=" * 70)
    print("CORRELATION WITH HITS")
    print("=" * 70)

    # Drug hit rate
    hr_corr, hr_p = pointbiserialr(df_with_hr['is_hit'], df_with_hr['drug_hit_rate'])
    print(f"\n1. Drug Hit Rate (from training):")
    print(f"   r = {hr_corr:.3f}, p = {hr_p:.4f}")

    # Compare to train_frequency
    freq_corr, freq_p = pointbiserialr(df_with_hr['is_hit'], df_with_hr['train_freq'])
    print(f"\n2. Train Frequency:")
    print(f"   r = {freq_corr:.3f}, p = {freq_p:.4f}")

    # Independence check
    hr_freq_corr, _ = pearsonr(df_with_hr['drug_hit_rate'], df_with_hr['train_freq'])
    print(f"\n3. Independence (drug_hit_rate vs train_freq):")
    print(f"   r = {hr_freq_corr:.3f}")
    if abs(hr_freq_corr) < 0.3:
        print(f"   → INDEPENDENT (|r| < 0.3)")
    else:
        print(f"   → CORRELATED (|r| >= 0.3)")

    # Precision analysis
    print("\n" + "=" * 70)
    print("PRECISION BY DRUG HIT RATE")
    print("=" * 70)

    # Quantiles
    q1 = df_with_hr['drug_hit_rate'].quantile(0.33)
    q2 = df_with_hr['drug_hit_rate'].quantile(0.66)

    low_hr = df_with_hr[df_with_hr['drug_hit_rate'] <= q1]
    mid_hr = df_with_hr[(df_with_hr['drug_hit_rate'] > q1) & (df_with_hr['drug_hit_rate'] <= q2)]
    high_hr = df_with_hr[df_with_hr['drug_hit_rate'] > q2]

    print(f"\n  LOW hit rate (≤{q1*100:.1f}%):  {low_hr['is_hit'].mean()*100:5.2f}% precision (n={len(low_hr)})")
    print(f"  MID hit rate ({q1*100:.1f}%-{q2*100:.1f}%):  {mid_hr['is_hit'].mean()*100:5.2f}% precision (n={len(mid_hr)})")
    print(f"  HIGH hit rate (>{q2*100:.1f}%): {high_hr['is_hit'].mean()*100:5.2f}% precision (n={len(high_hr)})")

    high_low_diff = high_hr['is_hit'].mean() - low_hr['is_hit'].mean()
    print(f"\n  HIGH - LOW difference: {high_low_diff*100:+.2f} pp")

    # Top drugs by hit rate
    print("\n" + "=" * 70)
    print("TOP DRUGS BY HISTORICAL HIT RATE")
    print("=" * 70)

    drug_summary = df_with_hr.groupby('drug_id').agg({
        'drug_hit_rate': 'first',
        'train_freq': 'first',
        'is_hit': ['sum', 'count']
    }).reset_index()
    drug_summary.columns = ['drug_id', 'hit_rate', 'train_freq', 'test_hits', 'test_preds']
    drug_summary['test_hit_rate'] = drug_summary['test_hits'] / drug_summary['test_preds']
    drug_summary = drug_summary[drug_summary['test_preds'] >= 5].sort_values('hit_rate', ascending=False)

    print(f"\n{'Drug':<20} {'Train HR':>10} {'Test HR':>10} {'Freq':>6} {'N':>5}")
    print("-" * 55)
    for _, row in drug_summary.head(10).iterrows():
        drug_name = drug_names.get(row['drug_id'], row['drug_id'][-10:])[:18]
        print(f"{drug_name:<20} {row['hit_rate']*100:>9.1f}% {row['test_hit_rate']*100:>9.1f}% {row['train_freq']:>6} {row['test_preds']:>5}")

    # Success criteria check
    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 70)

    success = hr_corr > 0.15 and abs(hr_freq_corr) < 0.3

    print(f"\n  Target: r > 0.15 with hits AND independent of frequency")
    print(f"  Drug hit rate correlation: r = {hr_corr:.3f}")
    print(f"  Independence (with frequency): r = {hr_freq_corr:.3f}")

    if success:
        print(f"  → VALIDATED: Drug hit rate is independent signal (r = {hr_corr:.3f})")
    else:
        if hr_corr <= 0.15:
            print(f"  → INVALIDATED: Drug hit rate correlation {hr_corr:.3f} < 0.15 threshold")
        else:
            print(f"  → INVALIDATED: Drug hit rate is not independent of frequency")

    # Save results
    output = {
        'drug_hit_rate_correlation': float(hr_corr),
        'drug_hit_rate_p': float(hr_p),
        'train_freq_correlation': float(freq_corr),
        'independence_correlation': float(hr_freq_corr),
        'high_low_precision_diff': float(high_low_diff),
        'n_predictions_with_hr': int(len(df_with_hr)),
        'success': bool(success),
    }

    results_file = PROJECT_ROOT / "data" / "analysis" / "h125_drug_level_success.json"
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
