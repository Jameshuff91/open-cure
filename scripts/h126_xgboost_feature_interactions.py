#!/usr/bin/env python3
"""
h126: XGBoost Feature Interaction Analysis

h119 showed XGBoost captures +2.07 pp from non-linear interactions.
Understanding WHICH interactions are valuable could inform feature engineering
and improve interpretability.

This script:
1. Trains XGBoost on 4 confidence features
2. Extracts SHAP values to understand feature importance
3. Computes SHAP interaction values
4. Identifies which feature combinations predict hits

SUCCESS CRITERIA: Identify at least one significant interaction (>10% contribution)
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("ERROR: XGBoost not available - cannot run this analysis")
    sys.exit(1)

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("ERROR: SHAP not available - install with: pip install shap")
    sys.exit(1)

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
            is_hit = drug_id in gt_drugs

            results.append({
                'mechanism_support': mech_support,
                'train_frequency': train_freq,
                'tier_inv': 3 - tier,
                'norm_score': norm_score,
                'is_hit': 1 if is_hit else 0,
            })

    return results


def main():
    print("h126: XGBoost Feature Interaction Analysis")
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

    # Collect predictions from single seed for speed
    print("\n" + "=" * 70)
    print("Collecting predictions (seed 42)")
    print("=" * 70)

    np.random.seed(42)
    diseases = list(ground_truth.keys())
    np.random.shuffle(diseases)
    n_test = len(diseases) // 5
    test_diseases = set(diseases[:n_test])
    train_diseases = set(diseases[n_test:])

    train_gt = {d: ground_truth[d] for d in train_diseases}
    test_gt = {d: ground_truth[d] for d in test_diseases}

    results = run_knn_with_features(
        emb_dict, train_gt, test_gt, disease_names, drug_targets, disease_genes, k=20
    )
    print(f"  Predictions: {len(results)}")

    df = pd.DataFrame(results)
    print(f"  Base hit rate: {df['is_hit'].mean()*100:.2f}%")

    # Prepare features
    features = ['mechanism_support', 'train_frequency', 'tier_inv', 'norm_score']
    X = df[features].values
    y = df['is_hit'].values

    # Train/test split for SHAP
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost model (shallow as in h119)
    print("\n" + "=" * 70)
    print("Training XGBoost model")
    print("=" * 70)

    model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=2,  # Shallow to enable interpretable interactions
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    # Feature importances from XGBoost
    print("\n" + "=" * 70)
    print("XGBoost Feature Importances (Gain)")
    print("=" * 70)

    importances = model.feature_importances_
    for name, imp in sorted(zip(features, importances), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {imp:.4f} ({imp*100:.1f}%)")

    # SHAP Analysis
    print("\n" + "=" * 70)
    print("SHAP Analysis")
    print("=" * 70)

    print("\nComputing SHAP values (this may take a moment)...")
    explainer = shap.TreeExplainer(model)

    # Use a sample for speed
    sample_size = min(500, len(X_test))
    X_sample = X_test[:sample_size]
    shap_values = explainer.shap_values(X_sample)

    # Mean absolute SHAP values (feature importance)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    print("\nMean Absolute SHAP Values:")
    total_shap = mean_abs_shap.sum()
    shap_importance = {}
    for name, val in sorted(zip(features, mean_abs_shap), key=lambda x: x[1], reverse=True):
        pct = val / total_shap * 100 if total_shap > 0 else 0
        print(f"  {name}: {val:.4f} ({pct:.1f}%)")
        shap_importance[name] = float(val)

    # SHAP Interaction Values
    print("\n" + "=" * 70)
    print("SHAP Interaction Values")
    print("=" * 70)

    print("\nComputing SHAP interaction values (this may take a while)...")
    # Limit sample for interaction computation (expensive)
    interaction_sample_size = min(200, sample_size)
    X_interaction = X_sample[:interaction_sample_size]
    shap_interaction_values = explainer.shap_interaction_values(X_interaction)

    # Average interaction matrix
    # Shape: (samples, features, features)
    mean_interactions = np.abs(shap_interaction_values).mean(axis=0)

    # Get off-diagonal elements (true interactions, not main effects)
    print("\nFeature Interactions (off-diagonal mean |SHAP|):")
    interaction_results = []
    n_features = len(features)
    total_interaction = 0
    for i in range(n_features):
        for j in range(i + 1, n_features):
            # Off-diagonal represents interaction between features i and j
            interaction_val = (mean_interactions[i, j] + mean_interactions[j, i]) / 2
            total_interaction += interaction_val
            interaction_results.append({
                'feature_1': features[i],
                'feature_2': features[j],
                'interaction': float(interaction_val)
            })

    interaction_results.sort(key=lambda x: x['interaction'], reverse=True)

    print("\nRanked Feature Interactions:")
    for item in interaction_results:
        pct = item['interaction'] / total_interaction * 100 if total_interaction > 0 else 0
        print(f"  {item['feature_1']} × {item['feature_2']}: {item['interaction']:.4f} ({pct:.1f}%)")

    # Main effects vs interactions
    main_effects = np.diag(mean_interactions).sum()
    total_effects = mean_interactions.sum()
    interaction_contribution = (total_effects - main_effects) / total_effects * 100 if total_effects > 0 else 0

    print(f"\nMain effects (diagonal): {main_effects:.4f}")
    print(f"Interaction effects (off-diagonal): {total_effects - main_effects:.4f}")
    print(f"Interaction contribution: {interaction_contribution:.1f}%")

    # Identify top interaction
    top_interaction = interaction_results[0] if interaction_results else None

    # Success criteria check
    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 70)

    success_threshold = 10.0  # At least 10% contribution from top interaction
    if top_interaction:
        top_pct = top_interaction['interaction'] / total_interaction * 100 if total_interaction > 0 else 0
        print(f"  Top interaction: {top_interaction['feature_1']} × {top_interaction['feature_2']}")
        print(f"  Contribution: {top_pct:.1f}% of interaction effects")

        if interaction_contribution >= 10:
            print(f"\n  ✓ Interaction effects contribute {interaction_contribution:.1f}% (≥10%)")
            if top_pct >= success_threshold:
                print(f"  ✓ Top interaction contributes {top_pct:.1f}% of interactions (≥{success_threshold}%)")
                print(f"  → VALIDATED: Significant interactions identified")
                success = True
            else:
                print(f"  ✗ Top interaction only {top_pct:.1f}% (spread across many interactions)")
                print(f"  → PARTIAL: Interactions exist but are diffuse")
                success = 'partial'
        else:
            print(f"  ✗ Interaction effects only {interaction_contribution:.1f}% (< 10%)")
            print(f"  → INVALIDATED: XGBoost gain comes from better main effect modeling, not interactions")
            success = False
    else:
        print("  ✗ No interactions computed")
        success = False

    # Interpret the findings
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    print("\n1. Feature Importances (XGBoost Gain):")
    print("   - This shows which features are most used in tree splits")

    print("\n2. SHAP Values:")
    print("   - These show which features most affect predictions")
    print("   - More actionable than XGBoost gain for understanding model behavior")

    print("\n3. SHAP Interactions:")
    print("   - Off-diagonal values show true synergies between features")
    print(f"   - If {interaction_contribution:.1f}% < 10%, XGBoost wins via better main effect modeling")
    print("   - Linear models may still be sufficient with the right features")

    # Save results
    output = {
        'xgboost_feature_importances': dict(zip(features, [float(x) for x in importances])),
        'shap_importance': shap_importance,
        'interactions': interaction_results,
        'main_effects': float(main_effects),
        'total_effects': float(total_effects),
        'interaction_contribution_pct': float(interaction_contribution),
        'top_interaction': top_interaction,
        'success': str(success),
        'interpretation': {
            'train_frequency': 'Drugs that treat many diseases in training are better predictions (polypharmacology)',
            'norm_score': 'Higher kNN similarity score = drug is common among similar diseases',
            'tier_inv': 'Some disease categories are more predictable than others',
            'mechanism_support': 'Target-gene overlap provides weak but independent signal',
        }
    }

    results_file = PROJECT_ROOT / "data" / "analysis" / "h126_xgboost_interactions.json"
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
