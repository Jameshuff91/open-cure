#!/usr/bin/env python3
"""
h123: Negative Confidence Signal - What Predicts Misses?

PURPOSE:
    h111 focused on what predicts HITS. But for precision, we also need
    to predict MISSES. If we can identify features that predict a drug
    will NOT work, we can filter them out and improve precision.

METHODOLOGY:
    1. Collect all predictions (hits and misses)
    2. Analyze features of MISSES vs HITS
    3. Identify features that negatively correlate with success
    4. Test if any feature predicts miss with >80% accuracy

SUCCESS CRITERIA:
    Identify feature(s) where presence predicts miss with >80% accuracy
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, List

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr, chi2_contingency
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from disease_name_matcher import DiseaseMatcher, load_mesh_mappings

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SEEDS = [42, 123, 456, 789, 1024]

# Load known problematic drug classes (from confidence_filter.py)
WITHDRAWN_DRUGS = {
    'pergolide', 'cisapride', 'valdecoxib', 'troglitazone', 'rofecoxib',
    'cerivastatin', 'terfenadine', 'astemizole', 'pemoline', 'dexfenfluramine'
}


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


def load_disease_genes() -> Dict[str, Set[str]]:
    genes_path = REFERENCE_DIR / "disease_genes.json"
    if not genes_path.exists():
        return {}
    with open(genes_path) as f:
        disease_genes = json.load(f)

    result = {}
    for k, v in disease_genes.items():
        gene_set = set(v)
        result[k] = gene_set
        if k.startswith('MESH:'):
            drkg_key = f"drkg:Disease::{k}"
            result[drkg_key] = gene_set
    return result


def load_drug_atc_classes() -> Dict[str, str]:
    """Load ATC class mappings for drugs."""
    edges_path = PROCESSED_DIR / "unified_edges_clean.csv"
    df = pd.read_csv(edges_path, usecols=['source', 'relation', 'target'])

    # Look for ATC-related edges
    atc_df = df[df['relation'].str.contains('ATC|DRUGBANK_CATEGORY', case=False, na=False)]

    drug_atc = {}
    for _, row in atc_df.iterrows():
        if 'Compound' in row['source']:
            drug_atc[row['source']] = row['target']

    return drug_atc


def run_knn_with_features(emb_dict, train_gt, test_gt, disease_names, drug_targets, disease_genes, drug_names, k=20):
    """Run kNN and collect all predictions with features."""
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
        max_score = sorted_drugs[0][1] if sorted_drugs else 1.0

        for rank, (drug_id, score) in enumerate(sorted_drugs, 1):
            drug_name = drug_names.get(drug_id, "unknown").lower()
            is_hit = drug_id in gt_drugs

            # Features
            targets = drug_targets.get(drug_id, set())
            dis_genes = disease_genes.get(disease_id, set())
            n_targets = len(targets)
            mechanism_support = 1 if len(targets & dis_genes) > 0 else 0
            train_freq = drug_train_freq.get(drug_id, 0)
            norm_score = score / max_score if max_score > 0 else 0

            # Negative signal candidates
            is_withdrawn = any(wd in drug_name for wd in WITHDRAWN_DRUGS)
            is_low_freq = train_freq <= 2
            has_no_targets = n_targets == 0
            no_mechanism = mechanism_support == 0
            low_knn_score = norm_score < 0.3
            high_rank = rank > 20

            results.append({
                'drug_id': drug_id,
                'drug_name': drug_name,
                'disease_id': disease_id,
                'rank': rank,
                'norm_score': norm_score,
                'n_targets': n_targets,
                'mechanism_support': mechanism_support,
                'train_freq': train_freq,
                'is_withdrawn': int(is_withdrawn),
                'is_low_freq': int(is_low_freq),
                'has_no_targets': int(has_no_targets),
                'no_mechanism': int(no_mechanism),
                'low_knn_score': int(low_knn_score),
                'high_rank': int(high_rank),
                'is_hit': int(is_hit),
            })

    return results


def main():
    print("h123: Negative Confidence Signal - What Predicts Misses?")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    emb_dict = load_node2vec_embeddings()
    name_to_drug_id, drug_names = load_drugbank_lookup()
    mesh_mappings = load_mesh_mappings_from_file()
    ground_truth, disease_names = load_ground_truth(mesh_mappings, name_to_drug_id)
    drug_targets = load_drug_targets()
    disease_genes = load_disease_genes()

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

        seed_results = run_knn_with_features(
            emb_dict, train_gt, test_gt, disease_names, drug_targets, disease_genes, drug_names, k=20
        )
        all_results.extend(seed_results)
        print(f"  Seed {seed}: {len(seed_results)} predictions")

    df = pd.DataFrame(all_results)
    print(f"\nTotal predictions: {len(df)}")
    print(f"Base hit rate: {df['is_hit'].mean()*100:.2f}%")
    print(f"Base miss rate: {(1 - df['is_hit'].mean())*100:.2f}%")

    # Analyze negative signals
    print("\n" + "=" * 70)
    print("NEGATIVE SIGNAL ANALYSIS")
    print("=" * 70)

    negative_features = [
        'is_withdrawn', 'is_low_freq', 'has_no_targets',
        'no_mechanism', 'low_knn_score', 'high_rank'
    ]

    feature_results = []

    for feature in negative_features:
        # Get predictions with this feature
        with_feature = df[df[feature] == 1]
        without_feature = df[df[feature] == 0]

        n_with = len(with_feature)
        n_without = len(without_feature)

        if n_with < 10:
            print(f"\n{feature}: Insufficient data (n={n_with})")
            continue

        hit_rate_with = with_feature['is_hit'].mean()
        hit_rate_without = without_feature['is_hit'].mean()
        miss_rate_with = 1 - hit_rate_with

        # Point-biserial correlation (negative = predicts miss)
        if df[feature].nunique() > 1:
            corr, p = pointbiserialr(df['is_hit'], df[feature])
        else:
            corr, p = 0, 1

        print(f"\n{feature}:")
        print(f"  n with feature: {n_with} ({n_with/len(df)*100:.1f}%)")
        print(f"  Hit rate WITH: {hit_rate_with*100:.2f}%")
        print(f"  Hit rate WITHOUT: {hit_rate_without*100:.2f}%")
        print(f"  Miss rate WITH: {miss_rate_with*100:.2f}%")
        print(f"  Correlation with hit: r = {corr:.3f} (p = {p:.4f})")

        if corr < 0:
            print(f"  → NEGATIVE SIGNAL (predicts miss)")

        feature_results.append({
            'feature': feature,
            'n_with': n_with,
            'hit_rate_with': hit_rate_with,
            'hit_rate_without': hit_rate_without,
            'miss_rate_with': miss_rate_with,
            'correlation': corr,
            'p_value': p,
            'predicts_miss_80': miss_rate_with > 0.80,
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: FEATURES THAT PREDICT MISS (>80% miss rate)")
    print("=" * 70)

    df_results = pd.DataFrame(feature_results)
    strong_negatives = df_results[df_results['miss_rate_with'] > 0.80]

    if len(strong_negatives) > 0:
        print(f"\n{'Feature':<20} {'Miss Rate':>10} {'N':>8} {'r':>8}")
        print("-" * 50)
        for _, row in strong_negatives.iterrows():
            print(f"{row['feature']:<20} {row['miss_rate_with']*100:>9.1f}% {row['n_with']:>8} {row['correlation']:>8.3f}")
    else:
        print("\nNo single feature achieves >80% miss prediction.")

    # Test combinations
    print("\n" + "=" * 70)
    print("COMBINATION ANALYSIS")
    print("=" * 70)

    # Try combinations of weak negative signals
    df['combo_low_freq_no_mech'] = ((df['is_low_freq'] == 1) & (df['no_mechanism'] == 1)).astype(int)
    df['combo_low_score_no_targets'] = ((df['low_knn_score'] == 1) & (df['has_no_targets'] == 1)).astype(int)
    df['combo_all_negatives'] = ((df['is_low_freq'] == 1) & (df['no_mechanism'] == 1) & (df['low_knn_score'] == 1)).astype(int)

    combos = ['combo_low_freq_no_mech', 'combo_low_score_no_targets', 'combo_all_negatives']

    for combo in combos:
        with_combo = df[df[combo] == 1]
        if len(with_combo) < 10:
            continue
        miss_rate = 1 - with_combo['is_hit'].mean()
        print(f"\n{combo}:")
        print(f"  n: {len(with_combo)} ({len(with_combo)/len(df)*100:.1f}%)")
        print(f"  Miss rate: {miss_rate*100:.1f}%")
        if miss_rate > 0.80:
            print(f"  → STRONG NEGATIVE COMBINATION!")

    # Success criteria check
    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 70)

    any_80 = (df_results['miss_rate_with'] > 0.80).any() if len(df_results) > 0 else False

    if any_80:
        print("  → VALIDATED: Found feature(s) with >80% miss prediction")
    else:
        # Check combos
        combo_80 = any(
            1 - df[df[c] == 1]['is_hit'].mean() > 0.80
            for c in combos
            if len(df[df[c] == 1]) >= 10
        )
        if combo_80:
            print("  → VALIDATED: Found combination(s) with >80% miss prediction")
        else:
            best_miss = df_results['miss_rate_with'].max() if len(df_results) > 0 else 0
            print(f"  → INVALIDATED: Best single feature miss rate = {best_miss*100:.1f}% (target: >80%)")

    # Save results
    output = {
        'feature_analysis': feature_results,
        'total_predictions': int(len(df)),
        'base_miss_rate': float(1 - df['is_hit'].mean()),
        'success': bool(any_80),
    }

    results_file = PROJECT_ROOT / "data" / "analysis" / "h123_negative_confidence.json"
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
