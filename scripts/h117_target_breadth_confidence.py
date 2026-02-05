#!/usr/bin/env python3
"""
h117: Confidence Feature - Drug Target Breadth

PURPOSE:
    h114 found number of targets correlates with drug frequency (ρ=0.24).
    Target breadth may be an independent confidence signal - drugs with more
    targets have more chances to hit disease pathways.

    Test if n_targets predicts hits independently of frequency.

SUCCESS CRITERIA:
    Target breadth provides 5+ pp precision difference (HIGH vs LOW targets).
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, List

import numpy as np
import pandas as pd
from scipy import stats
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
    id_to_name_full = {f"drkg:Compound::{db_id}": name for db_id, name in id_to_name.items()}
    return name_to_id, id_to_name_full


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


def run_knn_with_target_feature(
    emb_dict, train_gt, test_gt, drug_targets, k=20
) -> List[Dict]:
    """Run kNN and compute target breadth for each prediction."""
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

        sorted_drugs = sorted(drug_counts.items(), key=lambda x: x[1], reverse=True)[:30]

        for rank, (drug_id, score) in enumerate(sorted_drugs, 1):
            n_targets = len(drug_targets.get(drug_id, set()))
            train_freq = drug_train_freq.get(drug_id, 0)
            is_hit = drug_id in gt_drugs

            results.append({
                'disease': disease_id,
                'drug': drug_id,
                'n_targets': n_targets,
                'train_frequency': train_freq,
                'rank': rank,
                'is_hit': 1 if is_hit else 0,
            })

    return results


def main():
    print("h117: Confidence Feature - Drug Target Breadth")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    emb_dict = load_node2vec_embeddings()
    name_to_drug_id, id_to_name = load_drugbank_lookup()
    mesh_mappings = load_mesh_mappings_from_file()
    ground_truth, disease_names = load_ground_truth(mesh_mappings, name_to_drug_id)
    drug_targets = load_drug_targets()

    print(f"  Embeddings: {len(emb_dict)}")
    print(f"  Ground truth diseases: {len(ground_truth)}")
    print(f"  Drugs with target data: {len(drug_targets)}")

    # Collect predictions across seeds
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

        seed_results = run_knn_with_target_feature(
            emb_dict, train_gt, test_gt, drug_targets, k=20
        )
        all_results.extend(seed_results)
        print(f"  Seed {seed}: {len(seed_results)} predictions")

    df = pd.DataFrame(all_results)
    print(f"\nTotal predictions: {len(df)}")
    print(f"Base hit rate: {df['is_hit'].mean()*100:.2f}%")

    # === TARGET COVERAGE ===
    print("\n" + "=" * 70)
    print("TARGET DATA COVERAGE")
    print("=" * 70)

    has_targets = df[df['n_targets'] > 0]
    no_targets = df[df['n_targets'] == 0]
    print(f"  Predictions with target data: {len(has_targets)} ({len(has_targets)/len(df)*100:.1f}%)")
    print(f"  Predictions without target data: {len(no_targets)} ({len(no_targets)/len(df)*100:.1f}%)")

    # === PRECISION BY TARGET BREADTH ===
    print("\n" + "=" * 70)
    print("PRECISION BY TARGET BREADTH")
    print("=" * 70)

    # Split into terciles (among drugs with targets)
    target_vals = has_targets['n_targets']
    low_threshold = target_vals.quantile(0.33)
    high_threshold = target_vals.quantile(0.67)

    print(f"\nTarget distribution (drugs with target data):")
    print(f"  Median: {target_vals.median()}")
    print(f"  33rd percentile: {low_threshold}")
    print(f"  67th percentile: {high_threshold}")
    print(f"  Max: {target_vals.max()}")

    high_targets = has_targets[has_targets['n_targets'] >= high_threshold]
    medium_targets = has_targets[(has_targets['n_targets'] > low_threshold) & (has_targets['n_targets'] < high_threshold)]
    low_targets = has_targets[has_targets['n_targets'] <= low_threshold]

    print(f"\nPrecision by target breadth:")
    print(f"  HIGH targets (≥{int(high_threshold)}): {high_targets['is_hit'].mean()*100:.2f}% ({len(high_targets)} predictions)")
    print(f"  MEDIUM targets: {medium_targets['is_hit'].mean()*100:.2f}% ({len(medium_targets)} predictions)")
    print(f"  LOW targets (≤{int(low_threshold)}): {low_targets['is_hit'].mean()*100:.2f}% ({len(low_targets)} predictions)")
    print(f"  NO targets: {no_targets['is_hit'].mean()*100:.2f}% ({len(no_targets)} predictions)")

    high_prec = high_targets['is_hit'].mean() * 100
    low_prec = low_targets['is_hit'].mean() * 100
    diff = high_prec - low_prec

    print(f"\nDifference (HIGH - LOW): {diff:.2f} pp")

    # === CORRELATION ANALYSIS ===
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)

    # Correlation with hits
    r_hits, p_hits = stats.pointbiserialr(has_targets['n_targets'], has_targets['is_hit'])
    print(f"\nCorrelation with hits (point-biserial):")
    print(f"  n_targets vs is_hit: r = {r_hits:.4f} (p = {p_hits:.4e})")

    # Correlation with training frequency
    r_freq, p_freq = stats.spearmanr(has_targets['n_targets'], has_targets['train_frequency'])
    print(f"\nCorrelation with training frequency (Spearman):")
    print(f"  n_targets vs train_frequency: ρ = {r_freq:.4f} (p = {p_freq:.4e})")

    # === INDEPENDENCE CHECK ===
    print("\n" + "=" * 70)
    print("INDEPENDENCE FROM TRAINING FREQUENCY")
    print("=" * 70)

    # Check if n_targets predicts hits when controlling for frequency
    # Split by frequency, then check n_targets effect

    freq_median = has_targets['train_frequency'].median()
    high_freq = has_targets[has_targets['train_frequency'] > freq_median]
    low_freq = has_targets[has_targets['train_frequency'] <= freq_median]

    print(f"\nWithin HIGH-frequency drugs (freq > {freq_median}):")
    high_freq_high_targets = high_freq[high_freq['n_targets'] >= high_threshold]
    high_freq_low_targets = high_freq[high_freq['n_targets'] <= low_threshold]
    if len(high_freq_high_targets) > 0 and len(high_freq_low_targets) > 0:
        print(f"  HIGH targets: {high_freq_high_targets['is_hit'].mean()*100:.2f}% ({len(high_freq_high_targets)})")
        print(f"  LOW targets: {high_freq_low_targets['is_hit'].mean()*100:.2f}% ({len(high_freq_low_targets)})")
        print(f"  Δ: {(high_freq_high_targets['is_hit'].mean() - high_freq_low_targets['is_hit'].mean())*100:.2f} pp")

    print(f"\nWithin LOW-frequency drugs (freq ≤ {freq_median}):")
    low_freq_high_targets = low_freq[low_freq['n_targets'] >= high_threshold]
    low_freq_low_targets = low_freq[low_freq['n_targets'] <= low_threshold]
    if len(low_freq_high_targets) > 0 and len(low_freq_low_targets) > 0:
        print(f"  HIGH targets: {low_freq_high_targets['is_hit'].mean()*100:.2f}% ({len(low_freq_high_targets)})")
        print(f"  LOW targets: {low_freq_low_targets['is_hit'].mean()*100:.2f}% ({len(low_freq_low_targets)})")
        print(f"  Δ: {(low_freq_high_targets['is_hit'].mean() - low_freq_low_targets['is_hit'].mean())*100:.2f} pp")

    # === SUCCESS CRITERIA ===
    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 70)

    if diff >= 5.0:
        print(f"  ✓ Target breadth difference: {diff:.2f} pp (≥ 5 pp threshold)")
        print("  → VALIDATED: Target breadth is a valid confidence feature")
        success = True
    else:
        print(f"  ✗ Target breadth difference: {diff:.2f} pp (< 5 pp threshold)")
        print("  → INVALIDATED: Target breadth is too weak")
        success = False

    # Independence check
    print(f"\n  Correlation with frequency: ρ = {r_freq:.4f}")
    if abs(r_freq) < 0.3:
        print("  → Target breadth is INDEPENDENT of frequency (|ρ| < 0.3)")
    else:
        print("  → Target breadth is CORRELATED with frequency (|ρ| ≥ 0.3)")

    # Save results
    results = {
        'precision_high_targets': float(high_prec),
        'precision_low_targets': float(low_prec),
        'precision_diff': float(diff),
        'correlation_with_hits': float(r_hits),
        'correlation_with_frequency': float(r_freq),
        'success': bool(success),
        'high_threshold': float(high_threshold),
        'low_threshold': float(low_threshold),
        'n_predictions': len(df),
        'n_with_targets': len(has_targets),
    }

    results_file = PROJECT_ROOT / "data" / "analysis" / "h117_target_breadth_confidence.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
