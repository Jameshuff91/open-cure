#!/usr/bin/env python3
"""
h269: Cancer-Specific Target-Based Scoring (v2)

Improved approach: Score drug-cancer pairs by overlap between drug targets and disease-specific genes from DRKG.

This is a more biologically meaningful approach than the generic cancer gene approach.
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def load_mondo_to_mesh():
    """Load MONDO to MESH mapping."""
    with open('data/reference/mondo_to_mesh.json') as f:
        return json.load(f)


def load_drkg_disease_genes():
    """Extract disease-gene associations from DRKG."""
    disease_to_genes = defaultdict(set)
    print('Loading DRKG disease-gene edges...')
    with open('data/raw/drkg/drkg.tsv') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                head, rel, tail = parts[0], parts[1], parts[2]

                # Check for gene-disease relationships
                if 'Disease' in rel and 'Gene' in rel:
                    if head.startswith('Disease::MESH:') and tail.startswith('Gene::'):
                        mesh_id = head.replace('Disease::', '')
                        gene_id = tail.replace('Gene::', '')
                        disease_to_genes[mesh_id].add(gene_id)
                    elif tail.startswith('Disease::MESH:') and head.startswith('Gene::'):
                        mesh_id = tail.replace('Disease::', '')
                        gene_id = head.replace('Gene::', '')
                        disease_to_genes[mesh_id].add(gene_id)

    print(f'  Loaded {len(disease_to_genes)} diseases with gene associations')
    return disease_to_genes


def load_drug_targets():
    """Load drug targets and create name-to-target mapping."""
    with open('data/reference/drug_targets.json') as f:
        drug_targets = json.load(f)

    with open('data/reference/drugbank_lookup.json') as f:
        drugbank = json.load(f)

    # Create name -> DrugBank ID mapping
    name_to_db = {}
    for db_id, info in drugbank.items():
        name = info if isinstance(info, str) else info.get('name', '')
        if name:
            name_to_db[name.lower()] = db_id

    return drug_targets, name_to_db


def get_drug_targets(drug_name, drug_targets, name_to_db):
    """Get target gene IDs for a drug."""
    db_id = name_to_db.get(drug_name.lower())
    if db_id and db_id in drug_targets:
        return set(drug_targets[db_id])
    return set()


def score_drug_disease_pair(drug_targets_set, disease_genes_set):
    """Score a drug-disease pair by target-gene overlap.

    Returns multiple scores:
    - overlap_count: number of shared genes
    - jaccard: Jaccard similarity
    - drug_coverage: fraction of drug targets that overlap
    - disease_coverage: fraction of disease genes that overlap
    """
    if not drug_targets_set or not disease_genes_set:
        return 0, 0.0, 0.0, 0.0

    overlap = drug_targets_set & disease_genes_set
    overlap_count = len(overlap)

    union = drug_targets_set | disease_genes_set
    jaccard = overlap_count / len(union) if union else 0.0

    drug_coverage = overlap_count / len(drug_targets_set)
    disease_coverage = overlap_count / len(disease_genes_set)

    return overlap_count, jaccard, drug_coverage, disease_coverage


def load_cancer_gt_with_drkg_genes():
    """Load cancer GT and enrich with DRKG disease genes."""
    # Load mappings
    mondo_to_mesh = load_mondo_to_mesh()
    disease_to_genes = load_drkg_disease_genes()

    # Load GT
    gt = pd.read_excel('data/reference/everycure/indicationList.xlsx')
    disease_col = 'final normalized disease label'
    disease_id_col = 'final normalized disease id'
    drug_col = 'final normalized drug label'

    # Get cancer diseases
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
                    'mondo': disease_id,
                    'mesh': mesh_id,
                    'drugs': set(),
                    'genes': set()
                }

            if pd.notna(drug):
                cancer_diseases[disease_name]['drugs'].add(drug)

            # Add DRKG genes if we have MESH mapping
            if mesh_id and mesh_id in disease_to_genes:
                cancer_diseases[disease_name]['genes'] = disease_to_genes[mesh_id]

    return cancer_diseases


def evaluate_target_overlap_scoring(seed=42, min_drugs=3, top_k=30):
    """Evaluate disease-specific target overlap scoring."""
    np.random.seed(seed)

    # Load data
    print("Loading cancer diseases with DRKG genes...")
    cancer_diseases = load_cancer_gt_with_drkg_genes()

    print("Loading drug targets...")
    drug_targets, name_to_db = load_drug_targets()

    # Filter to evaluable diseases (>= min_drugs drugs AND has genes)
    evaluable = {d: info for d, info in cancer_diseases.items()
                 if len(info['drugs']) >= min_drugs and len(info['genes']) > 0}
    print(f"\nEvaluable cancer diseases (>= {min_drugs} drugs, has genes): {len(evaluable)}")

    # Get all drugs from evaluable diseases
    all_drugs = set()
    for info in evaluable.values():
        all_drugs.update(info['drugs'])

    # Find drugs with target data
    drugs_with_targets = set()
    for drug in all_drugs:
        if get_drug_targets(drug, drug_targets, name_to_db):
            drugs_with_targets.add(drug)

    print(f"Cancer drugs with targets: {len(drugs_with_targets)}/{len(all_drugs)}")

    # 80/20 train/test split by disease
    disease_list = list(evaluable.keys())
    np.random.shuffle(disease_list)
    split_idx = int(0.8 * len(disease_list))
    train_diseases = set(disease_list[:split_idx])
    test_diseases = set(disease_list[split_idx:])

    print(f"Train diseases: {len(train_diseases)}, Test diseases: {len(test_diseases)}")

    # Pre-compute drug targets
    drug_target_sets = {}
    for drug in drugs_with_targets:
        drug_target_sets[drug] = get_drug_targets(drug, drug_targets, name_to_db)

    # Evaluate on test diseases
    hits = 0
    total_diseases = 0
    disease_results = []

    for disease in test_diseases:
        disease_info = evaluable[disease]
        true_drugs = disease_info['drugs']
        disease_genes = disease_info['genes']

        # Score all drugs by target-disease gene overlap
        drug_scores = []
        for drug, targets in drug_target_sets.items():
            overlap, jaccard, drug_cov, disease_cov = score_drug_disease_pair(
                targets, disease_genes
            )
            drug_scores.append({
                'drug': drug,
                'overlap': overlap,
                'jaccard': jaccard,
                'drug_coverage': drug_cov,
                'disease_coverage': disease_cov,
                'score': overlap  # Use raw overlap count for ranking
            })

        # Sort by score (descending)
        drug_scores.sort(key=lambda x: -x['score'])
        top_drugs = [d['drug'] for d in drug_scores[:top_k]]

        # Calculate hit
        true_positives = set(top_drugs) & true_drugs
        hit = 1 if len(true_positives) > 0 else 0
        hits += hit
        total_diseases += 1

        disease_results.append({
            'disease': disease,
            'true_drugs': len(true_drugs),
            'disease_genes': len(disease_genes),
            'hit': hit,
            'true_positives': list(true_positives)[:5],
            'top_drugs': [(d['drug'], d['overlap']) for d in drug_scores[:5]]
        })

    recall_at_k = hits / total_diseases if total_diseases > 0 else 0

    print(f"\n=== Results (Disease-Specific Target Overlap Scoring) ===")
    print(f"Test diseases: {total_diseases}")
    print(f"Hits: {hits}")
    print(f"Recall@{top_k}: {recall_at_k:.1%}")

    # Show some examples
    print("\nExample predictions:")
    for result in disease_results[:3]:
        print(f"\n  {result['disease']}:")
        print(f"    True drugs: {result['true_drugs']}, Disease genes: {result['disease_genes']}")
        print(f"    Hit: {'YES' if result['hit'] else 'NO'}")
        if result['true_positives']:
            print(f"    True positives: {result['true_positives']}")
        print(f"    Top predictions: {result['top_drugs'][:3]}")

    return recall_at_k, disease_results


def evaluate_knn_on_cancer(seed=42, min_drugs=3, top_k=30, k=20):
    """Evaluate kNN baseline on the same cancer diseases for fair comparison."""
    np.random.seed(seed)

    print("\n" + "=" * 60)
    print("kNN BASELINE on Cancer Diseases")
    print("=" * 60)

    # Load cancer diseases
    cancer_diseases = load_cancer_gt_with_drkg_genes()

    # Filter to evaluable (>= min_drugs, has genes - same filter as target approach)
    evaluable = {d: info for d, info in cancer_diseases.items()
                 if len(info['drugs']) >= min_drugs and len(info['genes']) > 0}

    # Load embeddings
    print("Loading embeddings...")
    emb_path = 'data/embeddings/node2vec_256_no_treatment.csv'

    entity_to_emb = {}
    with open(emb_path) as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(',')
            entity = parts[0]
            emb = np.array([float(x) for x in parts[1:]])
            entity_to_emb[entity] = emb

    print(f"  Loaded {len(entity_to_emb)} embeddings")

    # Load drugbank lookup
    with open('data/reference/drugbank_lookup.json') as f:
        drugbank = json.load(f)

    name_to_db = {}
    for db_id, info in drugbank.items():
        name = info if isinstance(info, str) else info.get('name', '')
        if name:
            name_to_db[name.lower()] = db_id

    # Build drug embedding lookup
    drug_emb = {}
    for drug_name in set().union(*[info['drugs'] for info in evaluable.values()]):
        db_id = name_to_db.get(drug_name.lower())
        if db_id:
            # Try various formats
            for fmt in [f'Compound::{db_id}', db_id]:
                if fmt in entity_to_emb:
                    drug_emb[drug_name] = entity_to_emb[fmt]
                    break

    print(f"  Drugs with embeddings: {len(drug_emb)}")

    # Build disease embedding lookup
    mondo_to_mesh = load_mondo_to_mesh()
    disease_emb = {}
    for disease_name, info in evaluable.items():
        mesh_id = info.get('mesh')
        if mesh_id:
            for fmt in [f'Disease::{mesh_id}', mesh_id]:
                if fmt in entity_to_emb:
                    disease_emb[disease_name] = entity_to_emb[fmt]
                    break

    print(f"  Diseases with embeddings: {len(disease_emb)}")

    # Filter evaluable to diseases with embeddings
    evaluable_with_emb = {d: info for d, info in evaluable.items()
                          if d in disease_emb}
    print(f"  Evaluable with embeddings: {len(evaluable_with_emb)}")

    if len(evaluable_with_emb) < 5:
        print("ERROR: Not enough diseases with embeddings for evaluation")
        return None, None

    # Split diseases
    disease_list = list(evaluable_with_emb.keys())
    np.random.shuffle(disease_list)
    split_idx = int(0.8 * len(disease_list))
    train_diseases = set(disease_list[:split_idx])
    test_diseases = set(disease_list[split_idx:])

    print(f"Train diseases: {len(train_diseases)}, Test diseases: {len(test_diseases)}")

    # Build training disease embeddings
    train_emb_matrix = []
    train_disease_list = []
    for d in train_diseases:
        if d in disease_emb:
            train_emb_matrix.append(disease_emb[d])
            train_disease_list.append(d)

    train_emb_matrix = np.array(train_emb_matrix)

    # Normalize
    norms = np.linalg.norm(train_emb_matrix, axis=1, keepdims=True)
    train_emb_matrix = train_emb_matrix / (norms + 1e-10)

    # Evaluate on test diseases
    hits = 0
    total = 0

    for disease in test_diseases:
        if disease not in disease_emb:
            continue

        true_drugs = evaluable_with_emb[disease]['drugs']

        # Get disease embedding
        d_emb = disease_emb[disease]
        d_emb = d_emb / (np.linalg.norm(d_emb) + 1e-10)

        # Find k nearest training diseases
        similarities = train_emb_matrix @ d_emb
        top_k_indices = np.argsort(-similarities)[:k]

        # Collect drugs from similar diseases
        drug_counts = defaultdict(float)
        for idx in top_k_indices:
            sim = similarities[idx]
            train_d = train_disease_list[idx]
            for drug in evaluable_with_emb[train_d]['drugs']:
                if drug in drug_emb:
                    drug_counts[drug] += sim

        # Rank drugs
        sorted_drugs = sorted(drug_counts.items(), key=lambda x: -x[1])[:top_k]
        top_drugs = [d for d, _ in sorted_drugs]

        hit = 1 if set(top_drugs) & true_drugs else 0
        hits += hit
        total += 1

    recall = hits / total if total > 0 else 0

    print(f"\n=== kNN Results ===")
    print(f"Test diseases: {total}")
    print(f"Hits: {hits}")
    print(f"Recall@{top_k}: {recall:.1%}")

    return recall, None


def main():
    """Main evaluation function."""
    print("=" * 60)
    print("h269: Cancer-Specific Target-Based Scoring (v2)")
    print("=" * 60)

    # Run evaluation with multiple seeds
    print("\n" + "=" * 60)
    print("Running Target Overlap evaluation with 5 seeds...")
    print("=" * 60)

    target_recalls = []
    for seed in [42, 123, 456, 789, 1234]:
        print(f"\n--- Seed {seed} ---")
        recall, results = evaluate_target_overlap_scoring(seed=seed)
        target_recalls.append(recall)

    print("\n" + "=" * 60)
    print("FINAL RESULTS - Target Overlap")
    print("=" * 60)
    print(f"Mean Recall@30: {np.mean(target_recalls):.1%} ± {np.std(target_recalls):.1%}")
    print(f"Individual: {[f'{r:.1%}' for r in target_recalls]}")

    # Run kNN baseline for comparison
    print("\n" + "=" * 60)
    print("Running kNN baseline on same diseases with 5 seeds...")
    print("=" * 60)

    knn_recalls = []
    for seed in [42, 123, 456, 789, 1234]:
        print(f"\n--- Seed {seed} ---")
        recall, _ = evaluate_knn_on_cancer(seed=seed)
        if recall is not None:
            knn_recalls.append(recall)

    if knn_recalls:
        print("\n" + "=" * 60)
        print("FINAL RESULTS - kNN Baseline")
        print("=" * 60)
        print(f"Mean Recall@30: {np.mean(knn_recalls):.1%} ± {np.std(knn_recalls):.1%}")
        print(f"Individual: {[f'{r:.1%}' for r in knn_recalls]}")

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"Target Overlap: {np.mean(target_recalls):.1%} ± {np.std(target_recalls):.1%}")
    if knn_recalls:
        print(f"kNN Baseline:   {np.mean(knn_recalls):.1%} ± {np.std(knn_recalls):.1%}")
        diff = np.mean(target_recalls) - np.mean(knn_recalls)
        print(f"Difference:     {diff:+.1%}")


if __name__ == '__main__':
    main()
