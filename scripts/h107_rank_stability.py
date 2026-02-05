#!/usr/bin/env python3
"""
h107: Confidence Feature - Prediction Rank Stability Across Seeds

PURPOSE:
    If a drug ranks highly across multiple random seeds (train/test splits),
    it's more robust. High variance predictions may be unreliable.

APPROACH:
    1. Run kNN with 10 different seeds, record drug ranks for each disease
    2. Compute rank variance and mean rank for each drug-disease prediction
    3. Stratify by variance: LOW variance (stable) vs HIGH variance (unstable)
    4. Compare precision between stable and unstable predictions

SUCCESS CRITERIA:
    Stable predictions have 5%+ better precision.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from disease_name_matcher import DiseaseMatcher, load_mesh_mappings

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"

# Use 10 seeds for stability analysis
SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]


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


def run_knn_get_rankings(
    emb_dict: Dict[str, np.ndarray],
    train_gt: Dict[str, Set[str]],
    test_diseases: Set[str],
    all_gt: Dict[str, Set[str]],
    k: int = 20,
) -> Dict[Tuple[str, str], int]:
    """
    Run kNN and return drug rankings for all test diseases.
    Returns dict of (disease_id, drug_id) -> rank (1-indexed).
    """
    train_disease_list = [d for d in train_gt if d in emb_dict]
    train_disease_embs = np.array([emb_dict[d] for d in train_disease_list], dtype=np.float32)

    rankings: Dict[Tuple[str, str], int] = {}

    for disease_id in test_diseases:
        if disease_id not in emb_dict:
            continue
        if disease_id not in all_gt:
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

        # Sort by score and assign ranks
        sorted_drugs = sorted(drug_counts.items(), key=lambda x: x[1], reverse=True)
        for rank, (drug_id, _) in enumerate(sorted_drugs, 1):
            rankings[(disease_id, drug_id)] = rank

    return rankings


def main():
    print("h107: Prediction Rank Stability Across Seeds")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    emb_dict = load_node2vec_embeddings()
    name_to_drug_id = load_drugbank_lookup()
    mesh_mappings = load_mesh_mappings_from_file()
    ground_truth = load_ground_truth(mesh_mappings, name_to_drug_id)

    print(f"  Embeddings: {len(emb_dict)}")
    print(f"  Ground truth diseases: {len(ground_truth)}")

    # Collect rankings across all seeds
    print("\n" + "=" * 70)
    print(f"Running kNN with {len(SEEDS)} seeds to measure rank stability")
    print("=" * 70)

    # We need a consistent test set across seeds for fair comparison
    # Use first seed to define test diseases, then check stability of same diseases
    diseases = list(ground_truth.keys())

    # Track rankings per (disease, drug) across seeds
    all_rankings: Dict[Tuple[str, str], List[int]] = defaultdict(list)

    for seed_idx, seed in enumerate(SEEDS):
        np.random.seed(seed)
        shuffled = diseases.copy()
        np.random.shuffle(shuffled)
        n_test = len(shuffled) // 5
        test_diseases = set(shuffled[:n_test])
        train_diseases = set(shuffled[n_test:])

        train_gt = {d: ground_truth[d] for d in train_diseases if d in ground_truth}

        rankings = run_knn_get_rankings(
            emb_dict, train_gt, test_diseases, ground_truth, k=20
        )

        for key, rank in rankings.items():
            all_rankings[key].append(rank)

        print(f"  Seed {seed_idx+1}/{len(SEEDS)}: {len(rankings)} predictions")

    # Now analyze stability for predictions that appeared in at least 3 seeds
    print("\n" + "=" * 70)
    print("Analyzing Rank Stability")
    print("=" * 70)

    # Focus on top-30 predictions (most relevant for R@30)
    stability_data = []

    for (disease_id, drug_id), ranks in all_rankings.items():
        if len(ranks) < 3:
            continue  # Need at least 3 observations

        mean_rank = np.mean(ranks)
        std_rank = np.std(ranks)
        cv_rank = std_rank / mean_rank if mean_rank > 0 else 0  # Coefficient of variation
        min_rank = min(ranks)
        max_rank = max(ranks)

        # Only consider predictions that were top-30 in at least one seed
        if min_rank <= 30:
            # Check if this is a true positive
            is_hit = drug_id in ground_truth.get(disease_id, set())

            stability_data.append({
                'disease': disease_id,
                'drug': drug_id,
                'mean_rank': mean_rank,
                'std_rank': std_rank,
                'cv_rank': cv_rank,
                'min_rank': min_rank,
                'max_rank': max_rank,
                'rank_range': max_rank - min_rank,
                'n_seeds': len(ranks),
                'is_hit': is_hit,
            })

    print(f"\nPredictions with top-30 appearance: {len(stability_data)}")

    if not stability_data:
        print("No predictions to analyze!")
        return

    # Stratify by stability metrics
    # Use coefficient of variation (CV) as primary stability measure
    df = pd.DataFrame(stability_data)

    # Method 1: Tertiles by CV (lower CV = more stable)
    df_sorted = df.sort_values('cv_rank')
    n = len(df_sorted)
    tertile1 = df_sorted.iloc[:n//3]  # Most stable (low CV)
    tertile2 = df_sorted.iloc[n//3:2*n//3]  # Medium
    tertile3 = df_sorted.iloc[2*n//3:]  # Least stable (high CV)

    def calc_precision(subset):
        return subset['is_hit'].sum() / len(subset) if len(subset) > 0 else 0

    prec_stable = calc_precision(tertile1)
    prec_medium = calc_precision(tertile2)
    prec_unstable = calc_precision(tertile3)

    print(f"\n--- Stratification by Coefficient of Variation (CV) ---")
    print(f"\nSTABLE (low CV, {len(tertile1)} predictions):")
    print(f"  Mean CV: {tertile1['cv_rank'].mean():.3f}")
    print(f"  Mean rank range: {tertile1['rank_range'].mean():.1f}")
    print(f"  Precision: {100*prec_stable:.2f}%")

    print(f"\nMEDIUM stability ({len(tertile2)} predictions):")
    print(f"  Mean CV: {tertile2['cv_rank'].mean():.3f}")
    print(f"  Mean rank range: {tertile2['rank_range'].mean():.1f}")
    print(f"  Precision: {100*prec_medium:.2f}%")

    print(f"\nUNSTABLE (high CV, {len(tertile3)} predictions):")
    print(f"  Mean CV: {tertile3['cv_rank'].mean():.3f}")
    print(f"  Mean rank range: {tertile3['rank_range'].mean():.1f}")
    print(f"  Precision: {100*prec_unstable:.2f}%")

    # Key comparison
    print("\n" + "=" * 70)
    print("KEY COMPARISON")
    print("=" * 70)
    diff = prec_stable - prec_unstable
    print(f"  STABLE predictions precision: {100*prec_stable:.2f}%")
    print(f"  UNSTABLE predictions precision: {100*prec_unstable:.2f}%")
    print(f"  Difference: {100*diff:+.2f} pp")

    # Additional analysis: Mean rank as stability proxy
    print("\n--- Alternative: Stratification by Mean Rank ---")
    df_sorted_mr = df.sort_values('mean_rank')
    top_ranked = df_sorted_mr.iloc[:n//3]  # Best mean rank
    mid_ranked = df_sorted_mr.iloc[n//3:2*n//3]
    low_ranked = df_sorted_mr.iloc[2*n//3:]

    prec_top = calc_precision(top_ranked)
    prec_mid = calc_precision(mid_ranked)
    prec_low = calc_precision(low_ranked)

    print(f"\nTOP mean rank ({len(top_ranked)} predictions):")
    print(f"  Mean rank: {top_ranked['mean_rank'].mean():.1f}")
    print(f"  Precision: {100*prec_top:.2f}%")

    print(f"\nMID mean rank ({len(mid_ranked)} predictions):")
    print(f"  Mean rank: {mid_ranked['mean_rank'].mean():.1f}")
    print(f"  Precision: {100*prec_mid:.2f}%")

    print(f"\nLOW mean rank ({len(low_ranked)} predictions):")
    print(f"  Mean rank: {low_ranked['mean_rank'].mean():.1f}")
    print(f"  Precision: {100*prec_low:.2f}%")

    diff_mr = prec_top - prec_low
    print(f"\n  TOP vs LOW mean rank difference: {100*diff_mr:+.2f} pp")

    # Correlation analysis
    correlation_cv = np.corrcoef(df['cv_rank'].values, df['is_hit'].values)[0, 1]
    correlation_mr = np.corrcoef(df['mean_rank'].values, df['is_hit'].values)[0, 1]

    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)
    print(f"  Correlation(CV, is_hit): {correlation_cv:.4f}")
    print(f"  Correlation(mean_rank, is_hit): {correlation_mr:.4f}")

    # Success criteria check
    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 70)

    # CV-based stability
    success_cv = diff >= 0.05
    if success_cv:
        print(f"  ✓ STABLE (low CV) precision is +{100*diff:.1f} pp better (>= 5 pp)")
        print("  → VALIDATED: Rank stability (CV) is a valid confidence signal")
    else:
        if diff > 0:
            print(f"  ~ STABLE (low CV) precision is +{100*diff:.1f} pp better (< 5 pp)")
            print("  → PARTIALLY VALIDATED: Some improvement but below threshold")
        else:
            print(f"  ✗ STABLE precision is {100*diff:+.1f} pp vs UNSTABLE")
            print("  → INVALIDATED: Rank stability doesn't improve precision")

    # Mean rank as alternative signal
    success_mr = diff_mr >= 0.05
    if success_mr:
        print(f"\n  ✓ TOP mean rank precision is +{100*diff_mr:.1f} pp better (>= 5 pp)")
        print("  → Mean rank is also a valid confidence signal")
    else:
        print(f"\n  ~ Mean rank difference: {100*diff_mr:+.1f} pp")

    # Save results
    results = {
        'cv_stable_precision': float(prec_stable),
        'cv_unstable_precision': float(prec_unstable),
        'cv_difference_pp': float(diff * 100),
        'mr_top_precision': float(prec_top),
        'mr_low_precision': float(prec_low),
        'mr_difference_pp': float(diff_mr * 100),
        'correlation_cv_hit': float(correlation_cv),
        'correlation_mr_hit': float(correlation_mr),
        'success_cv': bool(success_cv),
        'success_mr': bool(success_mr),
        'n_predictions': len(stability_data),
        'n_seeds': len(SEEDS),
    }

    results_file = PROJECT_ROOT / "data" / "analysis" / "h107_rank_stability.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
