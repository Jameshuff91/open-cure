#!/usr/bin/env python3
"""
h154: Cardiovascular Beta-Blocker Combined Criteria

h150 showed beta_blocker + rank<=10 achieves 25% for cardiovascular.
Test if adding mechanism support or higher frequency thresholds can push this over 30%.

From h150:
- beta_blocker + rank<=5: 22.2% (n=36)
- beta_blocker + rank<=10: 25.0% (n=40)
- beta_blocker overall: 21.7% (n=46)

TEST CRITERIA:
1. beta_blocker + rank<=10 + mechanism_support
2. beta_blocker + rank<=5
3. beta_blocker + rank<=10 + freq>=10
4. beta_blocker + rank<=5 + mechanism_support

SUCCESS: Find criteria achieving >30% precision
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
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"

SEEDS = [42, 123, 456, 789, 1024]

CARDIOVASCULAR_KEYWORDS = ['cardiovascular', 'heart', 'cardiac', 'hypertension', 'arrhythmia',
                           'atherosclerosis', 'coronary', 'angina', 'heart failure',
                           'cardiomyopathy', 'myocardial', 'atrial', 'ventricular']

BETA_BLOCKERS = ['metoprolol', 'atenolol', 'carvedilol', 'bisoprolol', 'propranolol',
                 'labetalol', 'nebivolol', 'nadolol', 'timolol', 'esmolol', 'sotalol']


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
        disease_name = str(row['disease name']).lower().strip()
        drug_name = str(row['final normalized drug label']).lower().strip()

        disease_id = mesh_mappings.get(disease_name)
        if not disease_id:
            disease_id = matcher.get_mesh_id(disease_name)
        if not disease_id:
            continue

        drug_id = name_to_drug_id.get(drug_name)
        if drug_id:
            gt[disease_id].add(drug_id)
            disease_names[disease_id] = disease_name

    return gt, disease_names


def load_drug_targets():
    targets_path = REFERENCE_DIR / "drug_targets.json"
    if not targets_path.exists():
        return {}
    with open(targets_path) as f:
        drug_targets = json.load(f)
    return {f"drkg:Compound::{k}": set(v) for k, v in drug_targets.items()}


def load_disease_genes():
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
            result[f"drkg:Disease::{k}"] = gene_set
    return result


def is_cardiovascular(disease_name: str) -> bool:
    disease_lower = disease_name.lower()
    return any(kw in disease_lower for kw in CARDIOVASCULAR_KEYWORDS)


def is_beta_blocker(drug_name: str) -> bool:
    drug_lower = drug_name.lower()
    return any(bb in drug_lower for bb in BETA_BLOCKERS)


def has_mechanism_support(drug_id: str, disease_id: str, drug_targets: Dict, disease_genes: Dict) -> bool:
    drug_genes = drug_targets.get(drug_id, set())
    dis_genes = disease_genes.get(disease_id, set())
    return len(drug_genes & dis_genes) > 0


def knn_predictions(disease_id, train_diseases, gt, embeddings, id_to_name, k=20):
    if disease_id not in embeddings:
        return []

    query_emb = embeddings[disease_id].reshape(1, -1)
    train_with_emb = [d for d in train_diseases if d in embeddings and d != disease_id]
    if not train_with_emb:
        return []

    train_embs = np.vstack([embeddings[d] for d in train_with_emb])
    sims = cosine_similarity(query_emb, train_embs)[0]
    top_idx = np.argsort(sims)[-k:][::-1]
    neighbors = [train_with_emb[i] for i in top_idx]
    neighbor_sims = [sims[i] for i in top_idx]

    drug_scores: Dict[str, float] = defaultdict(float)
    drug_freq: Dict[str, int] = defaultdict(int)
    for neighbor, sim in zip(neighbors, neighbor_sims):
        for drug in gt.get(neighbor, []):
            drug_scores[drug] += sim
            drug_freq[drug] += 1

    sorted_drugs = sorted(drug_scores.items(), key=lambda x: x[1], reverse=True)

    predictions = []
    for rank, (drug_id, score) in enumerate(sorted_drugs[:30], 1):
        drug_name = id_to_name.get(drug_id, drug_id)
        predictions.append({
            'drug_id': drug_id,
            'drug_name': drug_name,
            'rank': rank,
            'score': score,
            'freq': drug_freq[drug_id],
        })

    return predictions


def main():
    print("h154: Cardiovascular Beta-Blocker Combined Criteria")
    print("=" * 80)

    print("\nLoading data...")
    embeddings = load_node2vec_embeddings()
    mesh_mappings = load_mesh_mappings_from_file()
    name_to_drug_id, id_to_name = load_drugbank_lookup()
    gt, disease_names = load_ground_truth(mesh_mappings, name_to_drug_id)
    drug_targets = load_drug_targets()
    disease_genes = load_disease_genes()

    # Filter to cardiovascular diseases with embeddings
    cv_diseases = [d for d in gt.keys()
                   if d in embeddings and len(gt[d]) >= 1 and is_cardiovascular(disease_names[d])]
    print(f"  Cardiovascular diseases with GT and embeddings: {len(cv_diseases)}")

    # Get all other diseases for training
    all_diseases = [d for d in gt.keys() if d in embeddings and len(gt[d]) >= 1]

    # Collect predictions across seeds
    all_preds = []

    for seed in SEEDS:
        np.random.seed(seed)

        n_test = max(1, len(cv_diseases) // 5)
        test_diseases = set(np.random.choice(cv_diseases, n_test, replace=False))
        train_diseases = set(all_diseases) - test_diseases

        for disease_id in test_diseases:
            preds = knn_predictions(disease_id, train_diseases, gt, embeddings, id_to_name)
            gt_drugs = gt[disease_id]

            for p in preds:
                drug_is_bb = is_beta_blocker(p['drug_name'])
                mech = has_mechanism_support(p['drug_id'], disease_id, drug_targets, disease_genes)
                is_hit = p['drug_id'] in gt_drugs

                all_preds.append({
                    'disease': disease_names[disease_id],
                    'disease_id': disease_id,
                    'drug': p['drug_name'],
                    'drug_id': p['drug_id'],
                    'rank': p['rank'],
                    'score': p['score'],
                    'freq': p['freq'],
                    'is_beta_blocker': drug_is_bb,
                    'mechanism_support': mech,
                    'is_hit': is_hit,
                    'seed': seed,
                })

    df = pd.DataFrame(all_preds)

    print(f"\nTotal predictions: {len(df)}")
    print(f"Beta-blocker predictions: {df['is_beta_blocker'].sum()}")
    print(f"Base hit rate: {df['is_hit'].mean()*100:.2f}%")
    print(f"Beta-blocker hit rate: {df[df['is_beta_blocker']]['is_hit'].mean()*100:.2f}%")

    # Test criteria
    print("\n" + "=" * 80)
    print("CRITERIA TESTING")
    print("=" * 80)

    criteria = [
        ('beta_blocker (all)', lambda r: r['is_beta_blocker']),
        ('beta_blocker + rank<=20', lambda r: r['is_beta_blocker'] and r['rank'] <= 20),
        ('beta_blocker + rank<=15', lambda r: r['is_beta_blocker'] and r['rank'] <= 15),
        ('beta_blocker + rank<=10', lambda r: r['is_beta_blocker'] and r['rank'] <= 10),
        ('beta_blocker + rank<=5', lambda r: r['is_beta_blocker'] and r['rank'] <= 5),
        ('beta_blocker + mechanism', lambda r: r['is_beta_blocker'] and r['mechanism_support']),
        ('beta_blocker + rank<=10 + mechanism', lambda r: r['is_beta_blocker'] and r['rank'] <= 10 and r['mechanism_support']),
        ('beta_blocker + rank<=5 + mechanism', lambda r: r['is_beta_blocker'] and r['rank'] <= 5 and r['mechanism_support']),
        ('beta_blocker + rank<=10 + freq>=5', lambda r: r['is_beta_blocker'] and r['rank'] <= 10 and r['freq'] >= 5),
        ('beta_blocker + rank<=10 + freq>=10', lambda r: r['is_beta_blocker'] and r['rank'] <= 10 and r['freq'] >= 10),
        ('beta_blocker + rank<=5 + freq>=5', lambda r: r['is_beta_blocker'] and r['rank'] <= 5 and r['freq'] >= 5),
    ]

    print(f"\n{'Criteria':<50} {'N':>6} {'Hits':>6} {'Precision':>10}")
    print("-" * 80)

    results = []
    best_over_30 = None

    for name, condition in criteria:
        subset = df[df.apply(condition, axis=1)]
        n = len(subset)
        hits = subset['is_hit'].sum()
        precision = hits / n * 100 if n > 0 else 0.0

        star = " ***" if precision >= 30 else ""
        print(f"{name:<50} {n:>6} {hits:>6} {precision:>9.1f}%{star}")

        results.append({'criteria': name, 'n': n, 'hits': int(hits), 'precision': precision})

        if precision >= 30 and n >= 5 and (best_over_30 is None or n > best_over_30['n']):
            best_over_30 = {'criteria': name, 'n': n, 'precision': precision}

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if best_over_30:
        print(f"✓ SUCCESS: {best_over_30['criteria']} = {best_over_30['precision']:.1f}% (n={best_over_30['n']})")
    else:
        best = max(results, key=lambda x: x['precision'])
        print(f"✗ FAILED: Best criteria {best['criteria']} = {best['precision']:.1f}% (n={best['n']})")
        print("  No criteria achieved >30% precision with n>=5")

    # Save results
    output = {
        'n_cv_diseases': len(cv_diseases),
        'n_predictions': len(df),
        'n_beta_blocker_preds': int(df['is_beta_blocker'].sum()),
        'criteria_results': results,
        'best_over_30': best_over_30,
        'success': best_over_30 is not None,
    }

    output_file = ANALYSIS_DIR / "h154_cardiovascular_beta_blocker.json"
    with open(output_file, 'w') as f:
        def convert(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            return obj
        json.dump(output, f, indent=2, default=convert)

    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
