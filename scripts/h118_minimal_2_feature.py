#!/usr/bin/env python3
"""
h118: Minimal 2-Feature Confidence Score

Test if train_frequency + mechanism_support alone can match the 4-feature ensemble.
h111 showed these are the two strongest independent signals (r=0.187 and r=0.098).

SUCCESS CRITERIA: 2-feature model achieves >20% precision at top 10%
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


def compute_mechanism_support(drug_id, disease_id, drug_targets, disease_genes) -> int:
    drug_genes = drug_targets.get(drug_id, set())
    dis_genes = disease_genes.get(disease_id, set())
    return 1 if len(drug_genes & dis_genes) > 0 else 0


def run_knn_with_features(emb_dict, train_gt, test_gt, disease_names, drug_targets, disease_genes, k=20):
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
            mech_support = compute_mechanism_support(drug_id, disease_id, drug_targets, disease_genes)
            train_freq = drug_train_freq.get(drug_id, 0)
            norm_score = score / max_score if max_score > 0 else 0
            inv_rank = 1.0 / rank
            is_hit = drug_id in gt_drugs

            results.append({
                'mechanism_support': mech_support,
                'train_frequency': train_freq,
                'tier_inv': 3 - tier,
                'norm_score': norm_score,
                'inv_rank': inv_rank,
                'is_hit': 1 if is_hit else 0,
            })

    return results


def evaluate_ensemble(df, feature_cols, label=''):
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
    print("h118: Minimal 2-Feature Confidence Score")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    emb_dict = load_node2vec_embeddings()
    name_to_drug_id = load_drugbank_lookup()
    mesh_mappings = load_mesh_mappings_from_file()
    ground_truth, disease_names = load_ground_truth(mesh_mappings, name_to_drug_id)
    drug_targets = load_drug_targets()
    disease_genes = load_disease_genes()

    print(f"  Embeddings: {len(emb_dict)}")
    print(f"  Ground truth diseases: {len(ground_truth)}")

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
            emb_dict, train_gt, test_gt, disease_names, drug_targets, disease_genes, k=20
        )
        all_results.extend(seed_results)
        print(f"  Seed {seed}: {len(seed_results)} predictions")

    df = pd.DataFrame(all_results)
    print(f"\nTotal predictions: {len(df)}")
    print(f"Base hit rate: {df['is_hit'].mean()*100:.2f}%")

    # Define feature sets
    full_4_features = ['mechanism_support', 'train_frequency', 'tier_inv', 'norm_score']
    minimal_2_features = ['mechanism_support', 'train_frequency']
    freq_only = ['train_frequency']
    mech_only = ['mechanism_support']

    # Evaluate all variants
    print("\n" + "=" * 70)
    print("FEATURE SET COMPARISON")
    print("=" * 70)

    results = {}

    print("\n1. 4-FEATURE SIMPLIFIED (from h115)")
    full_4_results = evaluate_ensemble(df, full_4_features)
    results['full_4'] = full_4_results
    print(f"   Features: {full_4_features}")
    print(f"   Top 10%: {full_4_results['top_10_precision']:.2f}%")
    print(f"   Top 20%: {full_4_results['top_20_precision']:.2f}%")

    print("\n2. MINIMAL 2-FEATURE (mechanism + frequency)")
    min_2_results = evaluate_ensemble(df, minimal_2_features)
    results['minimal_2'] = min_2_results
    print(f"   Features: {minimal_2_features}")
    print(f"   Top 10%: {min_2_results['top_10_precision']:.2f}%")
    print(f"   Top 20%: {min_2_results['top_20_precision']:.2f}%")

    print("\n3. FREQUENCY ONLY (1 feature)")
    freq_results = evaluate_ensemble(df, freq_only)
    results['freq_only'] = freq_results
    print(f"   Features: {freq_only}")
    print(f"   Top 10%: {freq_results['top_10_precision']:.2f}%")
    print(f"   Top 20%: {freq_results['top_20_precision']:.2f}%")

    print("\n4. MECHANISM ONLY (1 feature)")
    mech_results = evaluate_ensemble(df, mech_only)
    results['mech_only'] = mech_results
    print(f"   Features: {mech_only}")
    print(f"   Top 10%: {mech_results['top_10_precision']:.2f}%")
    print(f"   Top 20%: {mech_results['top_20_precision']:.2f}%")

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)

    print(f"\n{'Model':<30} {'# Feat':>7} {'Top 10%':>10} {'Top 20%':>10}")
    print("-" * 60)
    print(f"{'4-feature (h115 best)':<30} {4:>7} {full_4_results['top_10_precision']:>9.2f}% {full_4_results['top_20_precision']:>9.2f}%")
    print(f"{'2-feature (mech + freq)':<30} {2:>7} {min_2_results['top_10_precision']:>9.2f}% {min_2_results['top_20_precision']:>9.2f}%")
    print(f"{'1-feature (frequency only)':<30} {1:>7} {freq_results['top_10_precision']:>9.2f}% {freq_results['top_20_precision']:>9.2f}%")
    print(f"{'1-feature (mechanism only)':<30} {1:>7} {mech_results['top_10_precision']:>9.2f}% {mech_results['top_20_precision']:>9.2f}%")

    # Success criteria check
    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 70)

    min_2_top10 = min_2_results['top_10_precision']
    full_4_top10 = full_4_results['top_10_precision']
    diff = min_2_top10 - full_4_top10

    target = 20.0  # Success criterion: >20% at top 10%
    within_2pp = abs(diff) <= 2.0

    if min_2_top10 > target:
        print(f"  ✓ 2-feature model achieves {min_2_top10:.2f}% at top 10%")
        print(f"  ✓ Exceeds 20% threshold")
        if within_2pp:
            print(f"  ✓ Within 2 pp of 4-feature model ({diff:+.2f} pp)")
            print(f"  → VALIDATED: Minimal 2-feature model is production-ready")
            success = True
        else:
            print(f"  ✗ More than 2 pp below 4-feature model ({diff:+.2f} pp)")
            print(f"  → PARTIAL: Exceeds 20% but loses precision vs 4-feature")
            success = False
    else:
        print(f"  ✗ 2-feature model achieves {min_2_top10:.2f}% at top 10%")
        print(f"  ✗ Below 20% threshold")
        print(f"  → INVALIDATED: 2-feature model insufficient")
        success = False

    # Feature importance analysis
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)

    freq_contribution = freq_results['top_10_precision']
    mech_contribution = mech_results['top_10_precision']
    combined_contribution = min_2_results['top_10_precision']

    # Check for synergy
    additive_expectation = (freq_contribution + mech_contribution) / 2
    synergy = combined_contribution - max(freq_contribution, mech_contribution)

    print(f"\n  Frequency alone: {freq_contribution:.2f}%")
    print(f"  Mechanism alone: {mech_contribution:.2f}%")
    print(f"  Combined: {combined_contribution:.2f}%")
    print(f"  Synergy: {synergy:+.2f} pp above best single feature")

    # Save results
    output = {
        'full_4_features': full_4_results,
        'minimal_2_features': min_2_results,
        'frequency_only': freq_results,
        'mechanism_only': mech_results,
        'success': success,
        'synergy': float(synergy),
    }

    results_file = PROJECT_ROOT / "data" / "analysis" / "h118_minimal_2_feature.json"
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
