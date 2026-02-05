#!/usr/bin/env python3
"""
h99: Phenotype-Based Drug Transfer

PURPOSE:
    Use symptom/phenotype similarity instead of Node2Vec embedding similarity.
    Diseases with similar symptoms may respond to similar drugs even without
    shared genetic mechanisms or graph proximity.

METHODOLOGY:
    1. Extract disease-symptom edges from DRKG (PRESENTS_SYMPTOM)
    2. Compute symptom-based disease similarity (Jaccard)
    3. Use symptom-similar diseases to transfer drug recommendations
    4. Compare to Node2Vec-based kNN

SUCCESS CRITERIA:
    Symptom-based kNN achieves >30% R@30 (comparable to Node2Vec ~26%)
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from disease_name_matcher import DiseaseMatcher, load_mesh_mappings

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

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


def load_disease_symptoms() -> Dict[str, Set[str]]:
    """Load disease-symptom edges from DRKG."""
    edges_path = PROCESSED_DIR / "unified_edges_clean.csv"
    df = pd.read_csv(edges_path, usecols=['source', 'relation', 'target'])

    disease_symptoms = defaultdict(set)

    symptom_df = df[df['relation'] == 'PRESENTS_SYMPTOM']
    for _, row in symptom_df.iterrows():
        disease = row['source']
        symptom = row['target']
        disease_symptoms[disease].add(symptom)

    return dict(disease_symptoms)


def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def run_symptom_knn(train_gt, test_gt, disease_symptoms, emb_dict, k=20):
    """Run kNN using symptom similarity."""
    train_disease_list = [d for d in train_gt if d in disease_symptoms]
    if not train_disease_list:
        return [], 0

    results = []
    n_test_with_symptoms = 0

    for disease_id in test_gt:
        if disease_id not in disease_symptoms:
            continue

        test_symptoms = disease_symptoms[disease_id]
        gt_drugs = {d for d in test_gt[disease_id] if d in emb_dict}
        if not gt_drugs:
            continue

        n_test_with_symptoms += 1

        # Compute Jaccard similarity to all training diseases with symptoms
        sims = []
        for train_disease in train_disease_list:
            train_symptoms = disease_symptoms[train_disease]
            sim = jaccard_similarity(test_symptoms, train_symptoms)
            sims.append((train_disease, sim))

        # Get top k
        sims.sort(key=lambda x: x[1], reverse=True)
        top_k = sims[:k]

        # Count drug recommendations
        drug_counts = defaultdict(float)
        for neighbor_disease, sim in top_k:
            if sim > 0:  # Only count if there's any overlap
                for drug_id in train_gt[neighbor_disease]:
                    if drug_id in emb_dict:
                        drug_counts[drug_id] += sim

        if not drug_counts:
            continue

        sorted_drugs = sorted(drug_counts.items(), key=lambda x: x[1], reverse=True)[:30]
        hits = sum(1 for drug_id, _ in sorted_drugs if drug_id in gt_drugs)
        recall = hits / len(gt_drugs) if gt_drugs else 0

        results.append({
            'disease_id': disease_id,
            'n_gt_drugs': len(gt_drugs),
            'n_symptoms': len(test_symptoms),
            'max_sim': top_k[0][1] if top_k else 0,
            'hits': hits,
            'recall': recall,
        })

    return results, n_test_with_symptoms


def run_node2vec_knn(train_gt, test_gt, emb_dict, k=20):
    """Run kNN using Node2Vec similarity (baseline)."""
    train_disease_list = [d for d in train_gt if d in emb_dict]
    if not train_disease_list:
        return []

    train_disease_embs = np.array([emb_dict[d] for d in train_disease_list], dtype=np.float32)

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

        drug_counts = defaultdict(float)
        for idx in top_k_idx:
            neighbor_disease = train_disease_list[idx]
            neighbor_sim = sims[idx]
            for drug_id in train_gt[neighbor_disease]:
                if drug_id in emb_dict:
                    drug_counts[drug_id] += neighbor_sim

        if not drug_counts:
            continue

        sorted_drugs = sorted(drug_counts.items(), key=lambda x: x[1], reverse=True)[:30]
        hits = sum(1 for drug_id, _ in sorted_drugs if drug_id in gt_drugs)
        recall = hits / len(gt_drugs) if gt_drugs else 0

        results.append({
            'disease_id': disease_id,
            'n_gt_drugs': len(gt_drugs),
            'hits': hits,
            'recall': recall,
        })

    return results


def main():
    print("h99: Phenotype-Based Drug Transfer")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    emb_dict = load_node2vec_embeddings()
    name_to_drug_id = load_drugbank_lookup()
    mesh_mappings = load_mesh_mappings_from_file()
    ground_truth, disease_names = load_ground_truth(mesh_mappings, name_to_drug_id)
    disease_symptoms = load_disease_symptoms()

    print(f"  Embeddings: {len(emb_dict)}")
    print(f"  Ground truth diseases: {len(ground_truth)}")
    print(f"  Diseases with symptoms: {len(disease_symptoms)}")

    # Check overlap
    gt_with_symptoms = [d for d in ground_truth if d in disease_symptoms]
    print(f"  GT diseases with symptom data: {len(gt_with_symptoms)} ({len(gt_with_symptoms)/len(ground_truth)*100:.1f}%)")

    # Collect results
    print("\n" + "=" * 70)
    print("Running experiments across 5 seeds")
    print("=" * 70)

    all_symptom_results = []
    all_node2vec_results = []
    n_test_with_symptoms_total = 0

    for seed in SEEDS:
        np.random.seed(seed)
        diseases = list(ground_truth.keys())
        np.random.shuffle(diseases)
        n_test = len(diseases) // 5
        test_diseases = set(diseases[:n_test])
        train_diseases = set(diseases[n_test:])

        train_gt = {d: ground_truth[d] for d in train_diseases}
        test_gt = {d: ground_truth[d] for d in test_diseases}

        # Symptom-based kNN
        symptom_results, n_test = run_symptom_knn(train_gt, test_gt, disease_symptoms, emb_dict, k=20)
        all_symptom_results.extend(symptom_results)
        n_test_with_symptoms_total += n_test

        # Node2Vec-based kNN (on same diseases with symptoms)
        test_gt_filtered = {d: test_gt[d] for d in test_gt if d in disease_symptoms}
        node2vec_results = run_node2vec_knn(train_gt, test_gt_filtered, emb_dict, k=20)
        all_node2vec_results.extend(node2vec_results)

        print(f"  Seed {seed}: {n_test} test diseases with symptoms")

    # Summary statistics
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    df_symptom = pd.DataFrame(all_symptom_results)
    df_node2vec = pd.DataFrame(all_node2vec_results)

    symptom_recall = df_symptom['recall'].mean() * 100 if len(df_symptom) > 0 else 0
    node2vec_recall = df_node2vec['recall'].mean() * 100 if len(df_node2vec) > 0 else 0

    print(f"\nTest diseases with symptom data: {n_test_with_symptoms_total} (across 5 seeds)")
    print(f"Symptom-based kNN predictions: {len(df_symptom)}")
    print(f"Node2Vec kNN predictions: {len(df_node2vec)}")

    print(f"\n{'Method':<25} {'R@30':>10}")
    print("-" * 40)
    print(f"{'Symptom-based kNN':<25} {symptom_recall:>9.2f}%")
    print(f"{'Node2Vec kNN':<25} {node2vec_recall:>9.2f}%")

    diff = symptom_recall - node2vec_recall
    print(f"\nDifference: {diff:+.2f} pp")

    # Analyze symptom similarity distribution
    print("\n" + "=" * 70)
    print("SYMPTOM SIMILARITY ANALYSIS")
    print("=" * 70)

    if len(df_symptom) > 0:
        print(f"\nMax Jaccard similarity (nearest neighbor):")
        print(f"  Mean: {df_symptom['max_sim'].mean():.3f}")
        print(f"  Median: {df_symptom['max_sim'].median():.3f}")
        print(f"  Diseases with max_sim > 0.2: {(df_symptom['max_sim'] > 0.2).sum()} ({(df_symptom['max_sim'] > 0.2).mean()*100:.1f}%)")

    # Success criteria check
    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 70)

    success = symptom_recall > 30.0

    print(f"\n  Target: R@30 > 30% (comparable to Node2Vec ~26%)")
    print(f"  Symptom-based kNN: {symptom_recall:.2f}%")
    print(f"  Node2Vec baseline: {node2vec_recall:.2f}%")

    if success:
        print(f"  → VALIDATED: Symptom similarity achieves {symptom_recall:.2f}% R@30")
    else:
        if symptom_recall > 0.5 * node2vec_recall:
            print(f"  → INCONCLUSIVE: Symptom similarity achieves only {symptom_recall:.2f}% (below 30% but > 50% of Node2Vec)")
        else:
            print(f"  → INVALIDATED: Symptom similarity performs poorly ({symptom_recall:.2f}%)")

    # Save results
    output = {
        'symptom_recall': float(symptom_recall),
        'node2vec_recall': float(node2vec_recall),
        'difference': float(diff),
        'n_predictions_symptom': int(len(df_symptom)),
        'n_predictions_node2vec': int(len(df_node2vec)),
        'mean_max_jaccard': float(df_symptom['max_sim'].mean()) if len(df_symptom) > 0 else 0,
        'success': bool(success),
    }

    results_file = PROJECT_ROOT / "data" / "analysis" / "h99_phenotype_transfer.json"
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
