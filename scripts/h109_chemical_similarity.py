#!/usr/bin/env python3
"""
h109: Confidence Feature - Chemical Fingerprint Similarity to Known Treatments

PURPOSE:
    h104 failed because ATC class is too coarse. Alternative: compute Tanimoto
    similarity between predicted drug and known treatments for similar diseases.
    High chemical similarity to proven drugs = more confidence.

APPROACH:
    1. Load drug fingerprints (Morgan fingerprints)
    2. For each kNN prediction, find drugs known to treat similar diseases
    3. Compute max Tanimoto similarity to any known treatment
    4. Stratify by similarity: HIGH vs LOW
    5. Compare precision between chemically-similar and dissimilar predictions

SUCCESS CRITERIA:
    Chemically-similar predictions have 5%+ better precision.
"""

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from disease_name_matcher import DiseaseMatcher, load_mesh_mappings

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
CHEMICAL_DIR = REFERENCE_DIR / "chemical"

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


def load_drugbank_lookup() -> tuple:
    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)
    name_to_id = {name.lower(): f"drkg:Compound::{db_id}" for db_id, name in id_to_name.items()}
    id_to_name_full = {f"drkg:Compound::{db_id}": name.lower() for db_id, name in id_to_name.items()}
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


def load_fingerprints() -> Dict[str, np.ndarray]:
    """Load drug fingerprints."""
    fp_path = CHEMICAL_DIR / "drug_fingerprints.pkl"
    with open(fp_path, 'rb') as f:
        fps = pickle.load(f)
    return fps


def tanimoto_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """Compute Tanimoto similarity between two fingerprints."""
    # For binary fingerprints
    intersection = np.sum(np.minimum(fp1, fp2))
    union = np.sum(np.maximum(fp1, fp2))
    if union == 0:
        return 0.0
    return intersection / union


def compute_max_similarity(
    drug_name: str,
    known_drugs: Set[str],
    fingerprints: Dict[str, np.ndarray],
) -> Optional[float]:
    """
    Compute max Tanimoto similarity between drug and any known treatment.
    """
    if drug_name not in fingerprints:
        return None

    drug_fp = fingerprints[drug_name]
    max_sim = 0.0

    for known_drug in known_drugs:
        if known_drug in fingerprints:
            sim = tanimoto_similarity(drug_fp, fingerprints[known_drug])
            max_sim = max(max_sim, sim)

    return max_sim if max_sim > 0 else None


def run_knn_with_chemical_sim(
    emb_dict: Dict[str, np.ndarray],
    train_gt: Dict[str, Set[str]],
    test_gt: Dict[str, Set[str]],
    id_to_name: Dict[str, str],
    fingerprints: Dict[str, np.ndarray],
    k: int = 20,
) -> List[Dict]:
    """Run kNN and compute chemical similarity for each prediction."""
    train_disease_list = [d for d in train_gt if d in emb_dict]
    train_disease_embs = np.array([emb_dict[d] for d in train_disease_list], dtype=np.float32)

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

        # Collect drugs that treat similar diseases
        similar_disease_drugs: Set[str] = set()
        for idx in top_k_idx:
            neighbor_disease = train_disease_list[idx]
            for drug_id in train_gt[neighbor_disease]:
                drug_name = id_to_name.get(drug_id)
                if drug_name:
                    similar_disease_drugs.add(drug_name)

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
            drug_name = id_to_name.get(drug_id)
            if not drug_name:
                continue

            # Remove self from similar drugs for fair comparison
            comparison_drugs = similar_disease_drugs - {drug_name}

            # Compute max similarity to known treatments
            max_sim = compute_max_similarity(drug_name, comparison_drugs, fingerprints)

            is_hit = drug_id in gt_drugs

            results.append({
                'disease': disease_id,
                'drug': drug_id,
                'drug_name': drug_name,
                'chemical_similarity': max_sim,
                'is_hit': is_hit,
            })

    return results


def main():
    print("h109: Chemical Fingerprint Similarity as Confidence Feature")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    emb_dict = load_node2vec_embeddings()
    name_to_drug_id, id_to_name = load_drugbank_lookup()
    mesh_mappings = load_mesh_mappings_from_file()
    ground_truth = load_ground_truth(mesh_mappings, name_to_drug_id)
    fingerprints = load_fingerprints()

    print(f"  Embeddings: {len(emb_dict)}")
    print(f"  Ground truth diseases: {len(ground_truth)}")
    print(f"  Drug fingerprints: {len(fingerprints)}")

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

        seed_results = run_knn_with_chemical_sim(
            emb_dict, train_gt, test_gt, id_to_name, fingerprints, k=20
        )
        all_results.extend(seed_results)
        print(f"  Seed {seed}: {len(seed_results)} predictions")

    # Analyze by chemical similarity
    print("\n" + "=" * 70)
    print("Precision by Chemical Similarity")
    print("=" * 70)

    df = pd.DataFrame(all_results)

    # Filter out predictions without fingerprint data
    with_fp = df[df['chemical_similarity'].notna()]
    without_fp = df[df['chemical_similarity'].isna()]

    print(f"\nPredictions with fingerprint data: {len(with_fp)} ({100*len(with_fp)/len(df):.1f}%)")
    print(f"Predictions without fingerprint data: {len(without_fp)}")

    if len(with_fp) == 0:
        print("No predictions with fingerprint data!")
        return

    # Tertiles by similarity
    df_sorted = with_fp.sort_values('chemical_similarity', ascending=False)
    n = len(df_sorted)
    high_sim = df_sorted.iloc[:n//3]
    mid_sim = df_sorted.iloc[n//3:2*n//3]
    low_sim = df_sorted.iloc[2*n//3:]

    def calc_precision(subset):
        return subset['is_hit'].sum() / len(subset) if len(subset) > 0 else 0

    prec_high = calc_precision(high_sim)
    prec_mid = calc_precision(mid_sim)
    prec_low = calc_precision(low_sim)

    print(f"\nHIGH similarity ({len(high_sim)} predictions):")
    print(f"  Mean similarity: {high_sim['chemical_similarity'].mean():.3f}")
    print(f"  Precision: {100*prec_high:.2f}%")

    print(f"\nMEDIUM similarity ({len(mid_sim)} predictions):")
    print(f"  Mean similarity: {mid_sim['chemical_similarity'].mean():.3f}")
    print(f"  Precision: {100*prec_mid:.2f}%")

    print(f"\nLOW similarity ({len(low_sim)} predictions):")
    print(f"  Mean similarity: {low_sim['chemical_similarity'].mean():.3f}")
    print(f"  Precision: {100*prec_low:.2f}%")

    # Key comparison
    print("\n" + "=" * 70)
    print("KEY COMPARISON")
    print("=" * 70)
    diff = prec_high - prec_low
    print(f"  HIGH similarity precision: {100*prec_high:.2f}%")
    print(f"  LOW similarity precision:  {100*prec_low:.2f}%")
    print(f"  Difference: {100*diff:+.2f} pp")

    # Correlation
    correlation = np.corrcoef(
        with_fp['chemical_similarity'].values,
        with_fp['is_hit'].values
    )[0, 1]
    print(f"\n  Correlation(chemical_similarity, is_hit): {correlation:.4f}")

    # Success criteria check
    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 70)
    success = diff >= 0.05
    if success:
        print(f"  ✓ HIGH similarity precision is +{100*diff:.1f} pp better (>= 5 pp)")
        print("  → VALIDATED: Chemical similarity is a valid confidence signal")
    else:
        if diff > 0:
            print(f"  ~ HIGH similarity precision is +{100*diff:.1f} pp better (< 5 pp)")
            print("  → PARTIALLY VALIDATED: Some improvement but below threshold")
        else:
            print(f"  ✗ HIGH similarity precision is {100*diff:+.1f} pp vs LOW similarity")
            print("  → INVALIDATED: Chemical similarity doesn't improve precision")

    # Save results
    results = {
        'high_sim_precision': float(prec_high),
        'mid_sim_precision': float(prec_mid),
        'low_sim_precision': float(prec_low),
        'difference_pp': float(diff * 100),
        'correlation': float(correlation),
        'success': bool(success),
        'n_with_fp': len(with_fp),
        'n_without_fp': len(without_fp),
        'pct_with_fp': float(100*len(with_fp)/len(df)),
    }

    results_file = PROJECT_ROOT / "data" / "analysis" / "h109_chemical_similarity.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
