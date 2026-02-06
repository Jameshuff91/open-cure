#!/usr/bin/env python3
"""
h366: Target+kNN Ensemble for Cancer

Following h269, which showed that target overlap and kNN achieve equivalent
performance (65.8% vs 63.2%) but win on different diseases (7 vs 6 exclusive wins),
this experiment tests whether an ensemble can capture the complementary signals.
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def load_all_data():
    """Load all required data for cancer evaluation."""
    print("Loading data...")

    # MONDO to MESH mapping
    with open('data/reference/mondo_to_mesh.json') as f:
        mondo_to_mesh = json.load(f)

    # DRKG disease-gene associations
    disease_to_genes = defaultdict(set)
    with open('data/raw/drkg/drkg.tsv') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                head, rel, tail = parts[0], parts[1], parts[2]
                if 'Disease' in rel and 'Gene' in rel:
                    if head.startswith('Disease::MESH:') and tail.startswith('Gene::'):
                        mesh_id = head.replace('Disease::', '')
                        gene_id = tail.replace('Gene::', '')
                        disease_to_genes[mesh_id].add(gene_id)
                    elif tail.startswith('Disease::MESH:') and head.startswith('Gene::'):
                        mesh_id = tail.replace('Disease::', '')
                        gene_id = head.replace('Gene::', '')
                        disease_to_genes[mesh_id].add(gene_id)

    # Drug targets
    with open('data/reference/drug_targets.json') as f:
        drug_targets = json.load(f)
    with open('data/reference/drugbank_lookup.json') as f:
        drugbank = json.load(f)

    name_to_db = {}
    for db_id, info in drugbank.items():
        name = info if isinstance(info, str) else info.get('name', '')
        if name:
            name_to_db[name.lower()] = db_id

    # Node2Vec embeddings
    entity_to_emb = {}
    with open('data/embeddings/node2vec_256_no_treatment.csv') as f:
        f.readline()
        for line in f:
            parts = line.strip().split(',')
            entity = parts[0]
            emb = np.array([float(x) for x in parts[1:]])
            entity_to_emb[entity] = emb

    # Ground truth
    gt = pd.read_excel('data/reference/everycure/indicationList.xlsx')
    disease_col = 'final normalized disease label'
    disease_id_col = 'final normalized disease id'
    drug_col = 'final normalized drug label'

    cancer_terms = ['cancer', 'carcinoma', 'melanoma', 'leukemia', 'lymphoma',
                    'myeloma', 'sarcoma', 'tumor', 'neoplasm', 'glioma', 'blastoma']
    cancer_mask = gt[disease_col].str.lower().str.contains('|'.join(cancer_terms), na=False)
    cancer_gt = gt[cancer_mask]

    # Build cancer disease info
    cancer_diseases = {}
    for _, row in cancer_gt.iterrows():
        disease_name = row[disease_col]
        disease_id = row[disease_id_col]
        drug = row[drug_col]

        if pd.notna(disease_name) and pd.notna(disease_id):
            mesh_id = mondo_to_mesh.get(disease_id)
            if disease_name not in cancer_diseases:
                cancer_diseases[disease_name] = {
                    'mondo': disease_id, 'mesh': mesh_id, 'drugs': set(), 'genes': set()
                }
            if pd.notna(drug):
                cancer_diseases[disease_name]['drugs'].add(drug)
            if mesh_id and mesh_id in disease_to_genes:
                cancer_diseases[disease_name]['genes'] = disease_to_genes[mesh_id]

    # Filter evaluable
    evaluable = {d: info for d, info in cancer_diseases.items()
                 if len(info['drugs']) >= 3 and len(info['genes']) > 0}

    # Get drug embeddings
    drug_emb = {}
    for info in evaluable.values():
        for drug in info['drugs']:
            db_id = name_to_db.get(drug.lower())
            if db_id:
                for fmt in [f'Compound::{db_id}', db_id]:
                    if fmt in entity_to_emb:
                        drug_emb[drug] = entity_to_emb[fmt]
                        break

    # Get disease embeddings
    disease_emb = {}
    for d, info in evaluable.items():
        mesh_id = info.get('mesh')
        if mesh_id:
            for fmt in [f'Disease::{mesh_id}', mesh_id]:
                if fmt in entity_to_emb:
                    disease_emb[d] = entity_to_emb[fmt]
                    break

    # Filter to evaluable with embeddings
    evaluable_full = {d: info for d, info in evaluable.items() if d in disease_emb}

    # Pre-compute drug targets
    drug_target_sets = {}
    for info in evaluable.values():
        for drug in info['drugs']:
            db_id = name_to_db.get(drug.lower())
            if db_id and db_id in drug_targets:
                drug_target_sets[drug] = set(drug_targets[db_id])

    print(f"  Evaluable diseases: {len(evaluable_full)}")
    print(f"  Drugs with targets: {len(drug_target_sets)}")
    print(f"  Drugs with embeddings: {len(drug_emb)}")

    return evaluable_full, drug_target_sets, drug_emb, disease_emb


def get_target_scores(disease_info, drug_target_sets):
    """Get target overlap scores for all drugs."""
    disease_genes = disease_info['genes']
    scores = {}
    for drug, targets in drug_target_sets.items():
        scores[drug] = len(targets & disease_genes)
    return scores


def get_knn_scores(test_disease, test_emb, train_diseases, train_emb_matrix,
                   evaluable, drug_emb, k=20):
    """Get kNN-based scores for all drugs."""
    # Normalize test embedding
    test_norm = test_emb / (np.linalg.norm(test_emb) + 1e-10)

    # Find k nearest training diseases
    similarities = train_emb_matrix @ test_norm
    top_k_indices = np.argsort(-similarities)[:k]

    # Collect drugs from similar diseases
    drug_counts = defaultdict(float)
    for idx in top_k_indices:
        sim = similarities[idx]
        train_d = train_diseases[idx]
        for drug in evaluable[train_d]['drugs']:
            if drug in drug_emb:
                drug_counts[drug] += sim

    return dict(drug_counts)


def normalize_scores(scores):
    """Normalize scores to [0, 1] range using min-max scaling."""
    if not scores:
        return {}
    values = list(scores.values())
    min_val, max_val = min(values), max(values)
    if max_val == min_val:
        return {k: 0.5 for k in scores}
    return {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}


def rank_fusion(scores1, scores2, alpha=0.5):
    """Combine two score dictionaries using weighted rank fusion.

    Args:
        scores1: First score dictionary
        scores2: Second score dictionary
        alpha: Weight for first scores (0-1)

    Returns:
        Combined scores dictionary
    """
    # Get ranks for each method
    all_drugs = set(scores1.keys()) | set(scores2.keys())

    sorted1 = sorted(scores1.items(), key=lambda x: -x[1])
    sorted2 = sorted(scores2.items(), key=lambda x: -x[1])

    ranks1 = {d: i for i, (d, _) in enumerate(sorted1)}
    ranks2 = {d: i for i, (d, _) in enumerate(sorted2)}

    # Default rank for missing drugs is last
    max_rank1 = len(sorted1)
    max_rank2 = len(sorted2)

    # Combine ranks (lower is better, so we use negative)
    combined = {}
    for drug in all_drugs:
        r1 = ranks1.get(drug, max_rank1)
        r2 = ranks2.get(drug, max_rank2)
        # Convert to score (higher is better)
        combined[drug] = -(alpha * r1 + (1 - alpha) * r2)

    return combined


def score_fusion(scores1, scores2, alpha=0.5):
    """Combine two score dictionaries using weighted average of normalized scores.

    Args:
        scores1: First score dictionary
        scores2: Second score dictionary
        alpha: Weight for first scores (0-1)

    Returns:
        Combined scores dictionary
    """
    norm1 = normalize_scores(scores1)
    norm2 = normalize_scores(scores2)

    all_drugs = set(scores1.keys()) | set(scores2.keys())

    combined = {}
    for drug in all_drugs:
        s1 = norm1.get(drug, 0)
        s2 = norm2.get(drug, 0)
        combined[drug] = alpha * s1 + (1 - alpha) * s2

    return combined


def evaluate_method(method_name, get_scores_func, evaluable, disease_emb, drug_emb,
                    drug_target_sets, top_k=30):
    """Evaluate a scoring method using leave-one-out CV."""
    disease_list = list(evaluable.keys())
    emb_matrix = np.array([disease_emb[d] for d in disease_list])
    emb_matrix = emb_matrix / (np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-10)

    hits = 0
    for i, test_disease in enumerate(disease_list):
        disease_info = evaluable[test_disease]
        true_drugs = disease_info['drugs']

        # Get scores using provided function
        train_mask = np.ones(len(disease_list), dtype=bool)
        train_mask[i] = False
        train_emb = emb_matrix[train_mask]
        train_diseases = [d for j, d in enumerate(disease_list) if j != i]

        scores = get_scores_func(
            test_disease, disease_info, emb_matrix[i], train_diseases, train_emb,
            evaluable, drug_emb, drug_target_sets
        )

        # Rank and check hit
        sorted_drugs = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        top_drugs = {d for d, _ in sorted_drugs}

        if top_drugs & true_drugs:
            hits += 1

    recall = hits / len(disease_list)
    return recall, hits, len(disease_list)


def main():
    """Main evaluation function."""
    print("=" * 60)
    print("h366: Target+kNN Ensemble for Cancer")
    print("=" * 60)

    # Load data
    evaluable, drug_target_sets, drug_emb, disease_emb = load_all_data()
    disease_list = list(evaluable.keys())

    # Pre-compute embedding matrix
    emb_matrix = np.array([disease_emb[d] for d in disease_list])
    emb_matrix = emb_matrix / (np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-10)

    # Evaluate different methods with LOO CV
    print("\n" + "=" * 60)
    print("Evaluating methods with Leave-One-Out CV...")
    print("=" * 60)

    # Define scoring functions
    def target_only(test_d, info, test_emb, train_diseases, train_emb, evaluable, drug_emb, drug_target_sets):
        return get_target_scores(info, drug_target_sets)

    def knn_only(test_d, info, test_emb, train_diseases, train_emb, evaluable, drug_emb, drug_target_sets):
        return get_knn_scores(test_d, test_emb, train_diseases, train_emb, evaluable, drug_emb)

    def ensemble_score_05(test_d, info, test_emb, train_diseases, train_emb, evaluable, drug_emb, drug_target_sets):
        target = get_target_scores(info, drug_target_sets)
        knn = get_knn_scores(test_d, test_emb, train_diseases, train_emb, evaluable, drug_emb)
        return score_fusion(target, knn, alpha=0.5)

    def ensemble_score_03(test_d, info, test_emb, train_diseases, train_emb, evaluable, drug_emb, drug_target_sets):
        target = get_target_scores(info, drug_target_sets)
        knn = get_knn_scores(test_d, test_emb, train_diseases, train_emb, evaluable, drug_emb)
        return score_fusion(target, knn, alpha=0.3)

    def ensemble_score_07(test_d, info, test_emb, train_diseases, train_emb, evaluable, drug_emb, drug_target_sets):
        target = get_target_scores(info, drug_target_sets)
        knn = get_knn_scores(test_d, test_emb, train_diseases, train_emb, evaluable, drug_emb)
        return score_fusion(target, knn, alpha=0.7)

    def ensemble_rank_05(test_d, info, test_emb, train_diseases, train_emb, evaluable, drug_emb, drug_target_sets):
        target = get_target_scores(info, drug_target_sets)
        knn = get_knn_scores(test_d, test_emb, train_diseases, train_emb, evaluable, drug_emb)
        return rank_fusion(target, knn, alpha=0.5)

    def ensemble_rank_03(test_d, info, test_emb, train_diseases, train_emb, evaluable, drug_emb, drug_target_sets):
        target = get_target_scores(info, drug_target_sets)
        knn = get_knn_scores(test_d, test_emb, train_diseases, train_emb, evaluable, drug_emb)
        return rank_fusion(target, knn, alpha=0.3)

    def max_ensemble(test_d, info, test_emb, train_diseases, train_emb, evaluable, drug_emb, drug_target_sets):
        target = normalize_scores(get_target_scores(info, drug_target_sets))
        knn = normalize_scores(get_knn_scores(test_d, test_emb, train_diseases, train_emb, evaluable, drug_emb))
        all_drugs = set(target.keys()) | set(knn.keys())
        return {d: max(target.get(d, 0), knn.get(d, 0)) for d in all_drugs}

    methods = [
        ("Target Only", target_only),
        ("kNN Only", knn_only),
        ("Score Fusion α=0.3", ensemble_score_03),
        ("Score Fusion α=0.5", ensemble_score_05),
        ("Score Fusion α=0.7", ensemble_score_07),
        ("Rank Fusion α=0.3", ensemble_rank_03),
        ("Rank Fusion α=0.5", ensemble_rank_05),
        ("Max Ensemble", max_ensemble),
    ]

    results = []
    for name, func in methods:
        recall, hits, total = evaluate_method(
            name, func, evaluable, disease_emb, drug_emb, drug_target_sets
        )
        results.append((name, recall, hits, total))
        print(f"  {name}: {recall:.1%} ({hits}/{total} hits)")

    # Find best method
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    results.sort(key=lambda x: -x[1])
    for i, (name, recall, hits, total) in enumerate(results):
        marker = "***" if i == 0 else ""
        print(f"  {marker}{name}: {recall:.1%} ({hits}/{total}){marker}")

    best_name, best_recall, _, _ = results[0]
    target_recall = next(r for n, r, _, _ in results if n == "Target Only")
    knn_recall = next(r for n, r, _, _ in results if n == "kNN Only")

    print(f"\n  Target Only: {target_recall:.1%}")
    print(f"  kNN Only: {knn_recall:.1%}")
    print(f"  Best Ensemble: {best_name} at {best_recall:.1%}")

    improvement = best_recall - max(target_recall, knn_recall)
    print(f"\n  Improvement over best single method: {improvement:+.1%}")

    return results


if __name__ == '__main__':
    main()
