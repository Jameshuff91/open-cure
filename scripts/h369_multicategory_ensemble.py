#!/usr/bin/env python3
"""
h369: Apply Max Ensemble to Non-Cancer Categories

h366 showed max(target, kNN) improves cancer R@30 by 10.5 pp (63.2% → 76.3%).
This experiment tests whether this pattern generalizes to:
- Cardiovascular
- Autoimmune
- Neurological
- Metabolic
- Psychiatric
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Define disease category patterns
CATEGORIES = {
    'cardiovascular': ['heart', 'cardiac', 'hypertens', 'arrhythm', 'coronary', 'arterio',
                       'atheroscl', 'stroke', 'thrombo', 'vascular', 'angina', 'infarct', 'heart failure'],
    'autoimmune': ['arthritis', 'lupus', 'sclerosis', 'crohn', 'colitis', 'psoriasis',
                   'inflammatory bowel', 'sjogren', 'ankylosing', 'autoimmune', 'myasthenia'],
    'neurological': ['alzheimer', 'parkinson', 'epilep', 'seizure', 'neuropath', 'migraine',
                     'multiple sclerosis', 'huntington', 'dementia', 'neurodegen', 'amyotroph'],
    'metabolic': ['diabetes', 'obesity', 'metabolic', 'dyslipid', 'hyperlipid', 'hyperglycemia',
                  'hypercholesterol'],
    'psychiatric': ['depression', 'anxiety', 'schizophren', 'bipolar', 'psycho', 'ptsd',
                    'obsessive', 'panic disorder', 'adhd'],
    'cancer': ['cancer', 'carcinoma', 'melanoma', 'leukemia', 'lymphoma',
               'myeloma', 'sarcoma', 'tumor', 'neoplasm', 'glioma', 'blastoma']
}


def load_shared_data():
    """Load data shared across all categories."""
    print("Loading shared data...")

    # MONDO to MESH mapping
    with open('data/reference/mondo_to_mesh.json') as f:
        mondo_to_mesh = json.load(f)

    # DRKG disease-gene associations
    disease_to_genes = defaultdict(set)
    print("  Loading DRKG gene associations...")
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
    print(f"    {len(disease_to_genes)} diseases with gene associations")

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
    print("  Loading Node2Vec embeddings...")
    entity_to_emb = {}
    with open('data/embeddings/node2vec_256_no_treatment.csv') as f:
        f.readline()
        for line in f:
            parts = line.strip().split(',')
            entity = parts[0]
            emb = np.array([float(x) for x in parts[1:]])
            entity_to_emb[entity] = emb
    print(f"    {len(entity_to_emb)} entities with embeddings")

    # Ground truth
    gt = pd.read_excel('data/reference/everycure/indicationList.xlsx')

    return mondo_to_mesh, disease_to_genes, drug_targets, name_to_db, entity_to_emb, gt, drugbank


def load_category_data(category, terms, gt, mondo_to_mesh, disease_to_genes,
                       drug_targets, name_to_db, entity_to_emb):
    """Load data for a specific disease category."""
    disease_col = 'final normalized disease label'
    disease_id_col = 'final normalized disease id'
    drug_col = 'final normalized drug label'

    # Filter GT to category
    pattern = '|'.join(terms)
    cat_mask = gt[disease_col].str.lower().str.contains(pattern, na=False)
    cat_gt = gt[cat_mask]

    # Build category disease info
    cat_diseases = {}
    for _, row in cat_gt.iterrows():
        disease_name = row[disease_col]
        disease_id = row[disease_id_col]
        drug = row[drug_col]

        if pd.notna(disease_name) and pd.notna(disease_id):
            mesh_id = mondo_to_mesh.get(disease_id)
            if disease_name not in cat_diseases:
                cat_diseases[disease_name] = {
                    'mondo': disease_id, 'mesh': mesh_id, 'drugs': set(), 'genes': set()
                }
            if pd.notna(drug):
                cat_diseases[disease_name]['drugs'].add(drug)
            if mesh_id and mesh_id in disease_to_genes:
                cat_diseases[disease_name]['genes'] = disease_to_genes[mesh_id]

    # Filter evaluable: ≥3 drugs AND has gene associations
    evaluable = {d: info for d, info in cat_diseases.items()
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

    return evaluable_full, drug_target_sets, drug_emb, disease_emb


def get_target_scores(disease_info, drug_target_sets):
    """Get target overlap scores for all drugs."""
    disease_genes = disease_info['genes']
    scores = {}
    for drug, targets in drug_target_sets.items():
        overlap = len(targets & disease_genes)
        if overlap > 0:
            scores[drug] = overlap
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


def evaluate_category(category, evaluable, disease_emb, drug_emb, drug_target_sets, top_k=30):
    """Evaluate all methods on a category using leave-one-out CV."""
    if len(evaluable) < 5:
        return None

    disease_list = list(evaluable.keys())
    emb_matrix = np.array([disease_emb[d] for d in disease_list])
    emb_matrix = emb_matrix / (np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-10)

    results = {'target': 0, 'knn': 0, 'max_ensemble': 0}
    per_disease = []

    for i, test_disease in enumerate(disease_list):
        disease_info = evaluable[test_disease]
        true_drugs = disease_info['drugs']

        # Get train mask
        train_mask = np.ones(len(disease_list), dtype=bool)
        train_mask[i] = False
        train_emb = emb_matrix[train_mask]
        train_diseases = [d for j, d in enumerate(disease_list) if j != i]

        # Get scores from each method
        target_scores = get_target_scores(disease_info, drug_target_sets)
        knn_scores = get_knn_scores(test_disease, emb_matrix[i], train_diseases,
                                     train_emb, evaluable, drug_emb)

        # Normalize and create max ensemble
        target_norm = normalize_scores(target_scores)
        knn_norm = normalize_scores(knn_scores)
        all_drugs = set(target_norm.keys()) | set(knn_norm.keys())
        max_scores = {d: max(target_norm.get(d, 0), knn_norm.get(d, 0)) for d in all_drugs}

        # Evaluate each method
        target_hit = False
        knn_hit = False
        max_hit = False

        if target_scores:
            top_target = {d for d, _ in sorted(target_scores.items(), key=lambda x: -x[1])[:top_k]}
            target_hit = bool(top_target & true_drugs)

        if knn_scores:
            top_knn = {d for d, _ in sorted(knn_scores.items(), key=lambda x: -x[1])[:top_k]}
            knn_hit = bool(top_knn & true_drugs)

        if max_scores:
            top_max = {d for d, _ in sorted(max_scores.items(), key=lambda x: -x[1])[:top_k]}
            max_hit = bool(top_max & true_drugs)

        results['target'] += int(target_hit)
        results['knn'] += int(knn_hit)
        results['max_ensemble'] += int(max_hit)

        per_disease.append({
            'disease': test_disease,
            'target_hit': target_hit,
            'knn_hit': knn_hit,
            'max_hit': max_hit
        })

    n = len(disease_list)
    return {
        'n_diseases': n,
        'target_hits': results['target'],
        'target_recall': results['target'] / n,
        'knn_hits': results['knn'],
        'knn_recall': results['knn'] / n,
        'max_hits': results['max_ensemble'],
        'max_recall': results['max_ensemble'] / n,
        'per_disease': per_disease
    }


def main():
    """Main evaluation function."""
    print("=" * 70)
    print("h369: Apply Max Ensemble to Non-Cancer Categories")
    print("=" * 70)

    # Load shared data
    (mondo_to_mesh, disease_to_genes, drug_targets, name_to_db,
     entity_to_emb, gt, drugbank) = load_shared_data()

    # Evaluate each category
    print("\n" + "=" * 70)
    print("Evaluating categories...")
    print("=" * 70)

    all_results = {}

    for category, terms in CATEGORIES.items():
        print(f"\n--- {category.upper()} ---")

        # Load category-specific data
        evaluable, drug_target_sets, drug_emb, disease_emb = load_category_data(
            category, terms, gt, mondo_to_mesh, disease_to_genes,
            drug_targets, name_to_db, entity_to_emb
        )

        print(f"  Evaluable diseases: {len(evaluable)}")
        print(f"  Drugs with targets: {len(drug_target_sets)}")
        print(f"  Drugs with embeddings: {len(drug_emb)}")

        if len(evaluable) < 5:
            print(f"  SKIPPED: Not enough evaluable diseases (need ≥5)")
            continue

        # Evaluate
        results = evaluate_category(category, evaluable, disease_emb, drug_emb, drug_target_sets)

        if results:
            all_results[category] = results

            # Print results
            print(f"\n  Target Only:   {results['target_recall']:.1%} ({results['target_hits']}/{results['n_diseases']})")
            print(f"  kNN Only:      {results['knn_recall']:.1%} ({results['knn_hits']}/{results['n_diseases']})")
            print(f"  Max Ensemble:  {results['max_recall']:.1%} ({results['max_hits']}/{results['n_diseases']})")

            # Compute improvement
            best_single = max(results['target_recall'], results['knn_recall'])
            improvement = results['max_recall'] - best_single
            print(f"  Improvement:   {improvement:+.1%}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Category':<15} {'N':<5} {'Target':<10} {'kNN':<10} {'Ensemble':<10} {'Δ':<8}")
    print("-" * 58)

    categories_improved = 0
    total_improvement = 0

    for cat, res in sorted(all_results.items(), key=lambda x: -x[1]['max_recall']):
        best_single = max(res['target_recall'], res['knn_recall'])
        improvement = res['max_recall'] - best_single

        if improvement > 0.05:  # >5 pp improvement
            categories_improved += 1
        total_improvement += improvement

        marker = "✓" if improvement > 0.05 else ""
        print(f"{cat:<15} {res['n_diseases']:<5} {res['target_recall']:.1%}      {res['knn_recall']:.1%}      {res['max_recall']:.1%}      {improvement:+.1%} {marker}")

    print("-" * 58)
    print(f"\nCategories with >5 pp improvement: {categories_improved}/{len(all_results)}")
    print(f"Average improvement: {total_improvement/len(all_results):+.1%}")

    # Detailed analysis of best and worst categories
    print("\n" + "=" * 70)
    print("DETAILED ANALYSIS")
    print("=" * 70)

    for cat, res in all_results.items():
        # Analyze contingency
        both_hit = sum(1 for d in res['per_disease'] if d['target_hit'] and d['knn_hit'])
        target_only = sum(1 for d in res['per_disease'] if d['target_hit'] and not d['knn_hit'])
        knn_only = sum(1 for d in res['per_disease'] if not d['target_hit'] and d['knn_hit'])
        neither = sum(1 for d in res['per_disease'] if not d['target_hit'] and not d['knn_hit'])

        # Count how many ensemble gets that single methods miss
        ensemble_wins = sum(1 for d in res['per_disease']
                           if d['max_hit'] and not (d['target_hit'] and d['knn_hit']))
        ensemble_loses = sum(1 for d in res['per_disease']
                            if not d['max_hit'] and (d['target_hit'] or d['knn_hit']))

        print(f"\n{cat.upper()}:")
        print(f"  Both hit: {both_hit}, Target only: {target_only}, kNN only: {knn_only}, Neither: {neither}")
        print(f"  Ensemble captures Target-only wins: {target_only}/{target_only}")
        print(f"  Ensemble captures kNN-only wins: {knn_only}/{knn_only}")
        print(f"  Ensemble loses (single hit, ensemble miss): {ensemble_loses}")

    # Save results
    output = {
        'hypothesis': 'h369',
        'description': 'Max ensemble generalization to non-cancer categories',
        'results': {}
    }

    for cat, res in all_results.items():
        output['results'][cat] = {
            'n_diseases': res['n_diseases'],
            'target_recall': round(res['target_recall'], 3),
            'knn_recall': round(res['knn_recall'], 3),
            'max_recall': round(res['max_recall'], 3),
            'improvement': round(res['max_recall'] - max(res['target_recall'], res['knn_recall']), 3)
        }

    output['summary'] = {
        'categories_improved_5pp': categories_improved,
        'total_categories': len(all_results),
        'average_improvement': round(total_improvement/len(all_results), 3)
    }

    with open('data/analysis/h369_multicategory_ensemble.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to data/analysis/h369_multicategory_ensemble.json")

    return all_results


if __name__ == '__main__':
    main()
