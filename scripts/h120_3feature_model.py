#!/usr/bin/env python3
"""
h120: 3-Feature Confidence Model (Remove Mechanism)

h118 showed mechanism_support only contributes +0.59 pp synergy.
Test if [train_frequency, tier_inv, norm_score] alone achieves >21% at top 10%.

This would remove the dependency on disease_genes.json and drug_targets.json,
simplifying the production pipeline.

SUCCESS CRITERIA: 3-feature achieves >21% precision at top 10%
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import cross_val_predict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from disease_name_matcher import DiseaseMatcher, load_mesh_mappings

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"

SEEDS = [42, 123, 456, 789, 1024]

# Category tiers from h71
TIER_1_CATEGORIES = {'autoimmune', 'dermatological', 'psychiatric', 'ophthalmic'}
TIER_2_CATEGORIES = {'cardiovascular', 'other', 'cancer'}

CATEGORY_KEYWORDS = {
    'autoimmune': ['autoimmune', 'lupus', 'rheumatoid', 'arthritis', 'scleroderma', 'myasthenia',
                   'multiple sclerosis', 'crohn', 'colitis', 'psoriasis', 'sjögren'],
    'infectious': ['infection', 'bacterial', 'viral', 'fungal', 'hiv', 'aids', 'hepatitis',
                   'tuberculosis', 'malaria', 'pneumonia', 'sepsis', 'meningitis'],
    'cancer': ['cancer', 'carcinoma', 'tumor', 'leukemia', 'lymphoma', 'melanoma',
               'neoplasm', 'oncology', 'sarcoma', 'myeloma'],
    'cardiovascular': ['cardiac', 'heart', 'coronary', 'hypertension', 'arrhythmia',
                       'atherosclerosis', 'stroke', 'vascular', 'myocardial', 'angina'],
    'neurological': ['neurological', 'alzheimer', 'parkinson', 'epilepsy', 'neuropathy',
                     'dementia', 'huntington', 'sclerosis', 'brain'],
    'metabolic': ['diabetes', 'metabolic', 'obesity', 'thyroid', 'hyperlipidemia',
                  'hypercholesterolemia', 'gout', 'porphyria'],
    'psychiatric': ['depression', 'anxiety', 'bipolar', 'schizophrenia', 'psychiatric',
                    'ptsd', 'ocd', 'adhd', 'psychosis'],
    'respiratory': ['respiratory', 'asthma', 'copd', 'pulmonary', 'lung', 'bronchitis',
                    'pneumonitis', 'fibrosis'],
    'gastrointestinal': ['gastrointestinal', 'gastric', 'intestinal', 'bowel', 'liver',
                         'hepatic', 'cirrhosis', 'pancreatitis', 'celiac'],
    'dermatological': ['skin', 'dermatitis', 'eczema', 'psoriasis', 'dermatological',
                       'acne', 'urticaria', 'vitiligo'],
    'ophthalmic': ['eye', 'retinal', 'glaucoma', 'macular', 'ophthalmic', 'uveitis',
                   'conjunctivitis', 'keratitis'],
    'hematological': ['anemia', 'leukemia', 'lymphoma', 'hemophilia', 'thrombocytopenia',
                      'neutropenia', 'hematological', 'myelodysplastic'],
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


def categorize_disease(disease_name: str) -> str:
    name_lower = disease_name.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in name_lower:
                return category
    return 'other'


def get_category_tier(category: str) -> int:
    if category in TIER_1_CATEGORIES:
        return 1
    elif category in TIER_2_CATEGORIES:
        return 2
    else:
        return 3


def run_knn_with_features(emb_dict, train_gt, test_gt, disease_names, k=20):
    """kNN WITHOUT mechanism support - only needs embeddings and GT."""
    train_disease_list = [d for d in train_gt if d in emb_dict]
    train_disease_embs = np.array([emb_dict[d] for d in train_disease_list], dtype=np.float32)

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

        disease_name = disease_names.get(disease_id, "")
        category = categorize_disease(disease_name)
        tier = get_category_tier(category)

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
            train_freq = drug_train_freq.get(drug_id, 0)
            norm_score = score / max_score if max_score > 0 else 0
            is_hit = drug_id in gt_drugs

            results.append({
                'train_frequency': train_freq,
                'tier_inv': 3 - tier,
                'norm_score': norm_score,
                'is_hit': 1 if is_hit else 0,
            })

    return results


def evaluate_ensemble(df, feature_cols):
    """Train and evaluate an ensemble with given features."""
    X = df[feature_cols].values
    y = df['is_hit'].values

    # Standardize
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    # Cross-validated predictions
    lr = LogisticRegression(max_iter=1000, class_weight='balanced')
    proba = cross_val_predict(lr, X_norm, y, cv=5, method='predict_proba')[:, 1]

    df_temp = df.copy()
    df_temp['score'] = proba

    # Evaluate at different thresholds
    df_sorted = df_temp.sort_values('score', ascending=False)
    n = len(df_sorted)

    top_10 = df_sorted.iloc[:n//10]
    top_20 = df_sorted.iloc[:n//5]
    top_33 = df_sorted.iloc[:n//3]

    return {
        'top_10_precision': top_10['is_hit'].mean() * 100,
        'top_20_precision': top_20['is_hit'].mean() * 100,
        'top_33_precision': top_33['is_hit'].mean() * 100,
        'n_features': len(feature_cols),
    }


def main():
    print("h120: 3-Feature Confidence Model (Remove Mechanism)")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    emb_dict = load_node2vec_embeddings()
    name_to_drug_id = load_drugbank_lookup()
    mesh_mappings = load_mesh_mappings_from_file()
    ground_truth, disease_names = load_ground_truth(mesh_mappings, name_to_drug_id)

    print(f"  Embeddings: {len(emb_dict)}")
    print(f"  Ground truth diseases: {len(ground_truth)}")

    # Collect predictions
    print("\n" + "=" * 70)
    print("Collecting predictions across 5 seeds (NO mechanism support)")
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
            emb_dict, train_gt, test_gt, disease_names, k=20
        )
        all_results.extend(seed_results)
        print(f"  Seed {seed}: {len(seed_results)} predictions")

    df = pd.DataFrame(all_results)
    print(f"\nTotal predictions: {len(df)}")
    print(f"Base hit rate: {df['is_hit'].mean()*100:.2f}%")

    # Evaluate 3-feature model
    features_3 = ['train_frequency', 'tier_inv', 'norm_score']

    print("\n" + "=" * 70)
    print("3-FEATURE MODEL EVALUATION")
    print("=" * 70)

    results_3 = evaluate_ensemble(df, features_3)
    print(f"\n   Features: {features_3}")
    print(f"   Top 10%: {results_3['top_10_precision']:.2f}%")
    print(f"   Top 20%: {results_3['top_20_precision']:.2f}%")
    print(f"   Top 33%: {results_3['top_33_precision']:.2f}%")

    # Compare to h118's 4-feature result (21.89%)
    h118_4feature_top10 = 21.89  # From h118 results

    print("\n" + "=" * 70)
    print("COMPARISON TO 4-FEATURE MODEL (from h118)")
    print("=" * 70)

    print(f"\n{'Model':<30} {'# Feat':>7} {'Top 10%':>10}")
    print("-" * 50)
    print(f"{'4-feature (with mechanism)':<30} {4:>7} {h118_4feature_top10:>9.2f}%")
    print(f"{'3-feature (no mechanism)':<30} {3:>7} {results_3['top_10_precision']:>9.2f}%")

    diff = results_3['top_10_precision'] - h118_4feature_top10
    print(f"\n   Difference: {diff:+.2f} pp")

    # Success criteria check
    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 70)

    target = 21.0  # Success criterion: >21% at top 10%
    top10 = results_3['top_10_precision']

    if top10 > target:
        print(f"  ✓ 3-feature model achieves {top10:.2f}% at top 10%")
        print(f"  ✓ Exceeds 21% threshold")
        if abs(diff) <= 1.0:
            print(f"  ✓ Within 1 pp of 4-feature model ({diff:+.2f} pp)")
            print(f"  → VALIDATED: 3-feature model is viable for production")
            success = True
        else:
            print(f"  ✗ More than 1 pp below 4-feature model ({diff:+.2f} pp)")
            print(f"  → PARTIAL: Exceeds 21% but loses precision vs 4-feature")
            success = False
    else:
        print(f"  ✗ 3-feature model achieves {top10:.2f}% at top 10%")
        print(f"  ✗ Below 21% threshold")
        print(f"  → INVALIDATED: 3-feature model insufficient")
        success = False

    # Pipeline simplification benefit
    print("\n" + "=" * 70)
    print("PIPELINE SIMPLIFICATION")
    print("=" * 70)

    print("""
   4-FEATURE MODEL REQUIRES:
   - drug_targets.json (DrugBank parsing)
   - disease_genes.json (Gene-Disease associations)
   - mechanism_support computation (target-gene overlap)

   3-FEATURE MODEL REQUIRES:
   - train_frequency (from training GT only)
   - tier_inv (keyword-based categorization)
   - norm_score (from kNN already computed)

   BENEFIT: No external gene/target data needed!
""")

    # Save results
    output = {
        '3_feature_results': results_3,
        'comparison_to_4feature': {
            '4feature_top10': h118_4feature_top10,
            '3feature_top10': results_3['top_10_precision'],
            'difference': float(diff),
        },
        'success': success,
    }

    results_file = PROJECT_ROOT / "data" / "analysis" / "h120_3feature_model.json"
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
