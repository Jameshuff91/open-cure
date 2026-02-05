#!/usr/bin/env python3
"""
h106: Multi-Signal Confidence Ensemble

PURPOSE:
    Combine validated confidence signals to achieve >15% precision on high-confidence tier:
    1. Mechanism support (h97): +6.5 pp
    2. Drug training frequency (h108): +9.4 pp
    3. Category tier (h71): Tier 1 = 93-100% precision

APPROACH:
    1. Extract all confidence features for top-30 predictions
    2. Train logistic regression: features -> is_hit
    3. Calibrate probabilities using isotonic regression
    4. Compare to individual signals and existing confidence models

SUCCESS CRITERIA:
    Ensemble achieves 15%+ precision on high-confidence tier.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
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
TIER_3_CATEGORIES = {'metabolic', 'respiratory', 'gastrointestinal', 'hematological', 'infectious', 'neurological'}

# Disease category keywords
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


def load_ground_truth(
    mesh_mappings: Dict[str, str],
    name_to_drug_id: Dict[str, str],
) -> tuple:
    """Load ground truth and disease names."""
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
    """Load drug -> target genes mapping."""
    targets_path = REFERENCE_DIR / "drug_targets.json"
    if not targets_path.exists():
        return {}
    with open(targets_path) as f:
        drug_targets = json.load(f)
    return {f"drkg:Compound::{k}": set(v) for k, v in drug_targets.items()}


def load_disease_genes() -> Dict[str, Set[str]]:
    """Load disease -> associated genes mapping."""
    genes_path = REFERENCE_DIR / "disease_genes.json"
    if not genes_path.exists():
        return {}
    with open(genes_path) as f:
        disease_genes = json.load(f)
    return {k: set(v) for k, v in disease_genes.items()}


def categorize_disease(disease_name: str) -> str:
    """Categorize disease by name keywords."""
    name_lower = disease_name.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in name_lower:
                return category
    return 'other'


def get_category_tier(category: str) -> int:
    """Get tier for a category (1=best, 3=worst)."""
    if category in TIER_1_CATEGORIES:
        return 1
    elif category in TIER_2_CATEGORIES:
        return 2
    else:
        return 3


def compute_mechanism_support(
    drug_id: str,
    disease_id: str,
    drug_targets: Dict[str, Set[str]],
    disease_genes: Dict[str, Set[str]],
) -> int:
    """Check if drug targets overlap with disease genes (h97)."""
    drug_genes = drug_targets.get(drug_id, set())
    dis_genes = disease_genes.get(disease_id, set())
    overlap = len(drug_genes & dis_genes)
    return 1 if overlap > 0 else 0


def run_knn_with_features(
    emb_dict: Dict[str, np.ndarray],
    train_gt: Dict[str, Set[str]],
    test_gt: Dict[str, Set[str]],
    disease_names: Dict[str, str],
    drug_targets: Dict[str, Set[str]],
    disease_genes: Dict[str, Set[str]],
    k: int = 20,
) -> List[Dict]:
    """Run kNN and compute all confidence features for each prediction."""
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

        # Disease category and tier
        disease_name = disease_names.get(disease_id, "")
        category = categorize_disease(disease_name)
        tier = get_category_tier(category)

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

        # Get top 30
        sorted_drugs = sorted(drug_counts.items(), key=lambda x: x[1], reverse=True)[:30]
        max_score = sorted_drugs[0][1] if sorted_drugs else 1.0

        for rank, (drug_id, score) in enumerate(sorted_drugs, 1):
            # Feature 1: Mechanism support (h97)
            mech_support = compute_mechanism_support(drug_id, disease_id, drug_targets, disease_genes)

            # Feature 2: Drug training frequency (h108)
            train_freq = drug_train_freq.get(drug_id, 0)

            # Feature 3: Category tier (h71)
            # Use inverse tier so higher = better confidence

            # Feature 4: Normalized kNN score
            norm_score = score / max_score if max_score > 0 else 0

            # Feature 5: Rank (inverse so higher = better)
            inv_rank = 1.0 / rank

            is_hit = drug_id in gt_drugs

            results.append({
                'disease': disease_id,
                'drug': drug_id,
                'disease_name': disease_name,
                'category': category,
                'tier': tier,
                'mechanism_support': mech_support,
                'train_frequency': train_freq,
                'norm_score': norm_score,
                'inv_rank': inv_rank,
                'rank': rank,
                'is_hit': is_hit,
            })

    return results


def main():
    print("h106: Multi-Signal Confidence Ensemble")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    emb_dict = load_node2vec_embeddings()
    name_to_drug_id, id_to_name = load_drugbank_lookup()
    mesh_mappings = load_mesh_mappings_from_file()
    ground_truth, disease_names = load_ground_truth(mesh_mappings, name_to_drug_id)
    drug_targets = load_drug_targets()
    disease_genes = load_disease_genes()

    print(f"  Embeddings: {len(emb_dict)}")
    print(f"  Ground truth diseases: {len(ground_truth)}")
    print(f"  Drugs with targets: {len(drug_targets)}")
    print(f"  Diseases with genes: {len(disease_genes)}")

    # Multi-seed evaluation
    print("\n" + "=" * 70)
    print("Collecting features across 5 seeds")
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

    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    print(f"\nTotal predictions: {len(df)}")
    print(f"Hit rate: {df['is_hit'].mean()*100:.2f}%")

    # Individual feature performance
    print("\n" + "=" * 70)
    print("Individual Feature Performance")
    print("=" * 70)

    def calc_precision_by_split(df, column, n_splits=3):
        df_sorted = df.sort_values(column, ascending=False)
        n = len(df_sorted)
        split_size = n // n_splits
        top = df_sorted.iloc[:split_size]
        bottom = df_sorted.iloc[-split_size:]
        return top['is_hit'].mean(), bottom['is_hit'].mean()

    # Mechanism support
    mech_yes = df[df['mechanism_support'] == 1]
    mech_no = df[df['mechanism_support'] == 0]
    print(f"\nMechanism Support (h97):")
    print(f"  WITH support: {mech_yes['is_hit'].mean()*100:.2f}% ({len(mech_yes)})")
    print(f"  WITHOUT support: {mech_no['is_hit'].mean()*100:.2f}% ({len(mech_no)})")

    # Drug frequency
    top_freq, bot_freq = calc_precision_by_split(df, 'train_frequency')
    print(f"\nDrug Training Frequency (h108):")
    print(f"  HIGH frequency: {top_freq*100:.2f}%")
    print(f"  LOW frequency: {bot_freq*100:.2f}%")

    # Category tier
    for tier in [1, 2, 3]:
        tier_df = df[df['tier'] == tier]
        print(f"\nTier {tier} ({['HIGH', 'MEDIUM', 'LOW'][tier-1]} confidence):")
        print(f"  Precision: {tier_df['is_hit'].mean()*100:.2f}% ({len(tier_df)} predictions)")

    # Train logistic regression ensemble
    print("\n" + "=" * 70)
    print("Training Multi-Signal Ensemble")
    print("=" * 70)

    # Features for ensemble
    feature_cols = ['mechanism_support', 'train_frequency', 'tier', 'norm_score', 'inv_rank']

    # Normalize tier so higher = better (3 - tier so tier 1 -> 2, tier 3 -> 0)
    df['tier_inv'] = 3 - df['tier']
    feature_cols_adjusted = ['mechanism_support', 'train_frequency', 'tier_inv', 'norm_score', 'inv_rank']

    X = df[feature_cols_adjusted].values
    y = df['is_hit'].values

    # Standardize features
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    # Cross-validated predictions
    lr = LogisticRegression(max_iter=1000, class_weight='balanced')
    proba = cross_val_predict(lr, X_norm, y, cv=5, method='predict_proba')[:, 1]

    df['ensemble_score'] = proba

    # Calibrate with isotonic regression
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    calibrated = iso.fit_transform(proba, y)
    df['calibrated_score'] = calibrated

    # Evaluate ensemble
    print("\n" + "=" * 70)
    print("Ensemble Performance")
    print("=" * 70)

    # Split by ensemble score tertiles
    df_sorted = df.sort_values('ensemble_score', ascending=False)
    n = len(df_sorted)
    high_conf = df_sorted.iloc[:n//3]
    mid_conf = df_sorted.iloc[n//3:2*n//3]
    low_conf = df_sorted.iloc[2*n//3:]

    print(f"\nHIGH confidence tier ({len(high_conf)} predictions):")
    print(f"  Precision: {high_conf['is_hit'].mean()*100:.2f}%")
    print(f"  Mean ensemble score: {high_conf['ensemble_score'].mean():.3f}")

    print(f"\nMEDIUM confidence tier ({len(mid_conf)} predictions):")
    print(f"  Precision: {mid_conf['is_hit'].mean()*100:.2f}%")
    print(f"  Mean ensemble score: {mid_conf['ensemble_score'].mean():.3f}")

    print(f"\nLOW confidence tier ({len(low_conf)} predictions):")
    print(f"  Precision: {low_conf['is_hit'].mean()*100:.2f}%")
    print(f"  Mean ensemble score: {low_conf['ensemble_score'].mean():.3f}")

    # More aggressive high-confidence threshold
    top_10pct = df_sorted.iloc[:n//10]
    top_20pct = df_sorted.iloc[:n//5]

    print(f"\nTop 10% by ensemble score ({len(top_10pct)} predictions):")
    print(f"  Precision: {top_10pct['is_hit'].mean()*100:.2f}%")

    print(f"\nTop 20% by ensemble score ({len(top_20pct)} predictions):")
    print(f"  Precision: {top_20pct['is_hit'].mean()*100:.2f}%")

    # Success criteria
    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 70)
    high_prec = high_conf['is_hit'].mean()
    success = high_prec >= 0.15

    if success:
        print(f"  ✓ HIGH confidence tier achieves {high_prec*100:.1f}% precision (>= 15%)")
        print("  → VALIDATED: Multi-signal ensemble meets success criteria")
    else:
        print(f"  ✗ HIGH confidence tier achieves {high_prec*100:.1f}% precision (< 15%)")
        top10_prec = top_10pct['is_hit'].mean()
        if top10_prec >= 0.15:
            print(f"  ~ However, TOP 10% achieves {top10_prec*100:.1f}% precision (>= 15%)")
            print("  → PARTIALLY VALIDATED: More aggressive thresholding needed")
        else:
            print("  → INVALIDATED: Ensemble doesn't achieve 15% precision")

    # Compare to individual signals
    print("\n" + "=" * 70)
    print("Comparison to Individual Signals")
    print("=" * 70)

    # Best tier-based precision
    tier1_prec = df[df['tier'] == 1]['is_hit'].mean() if len(df[df['tier'] == 1]) > 0 else 0
    mech_prec = mech_yes['is_hit'].mean() if len(mech_yes) > 0 else 0

    print(f"  Category Tier 1 only: {tier1_prec*100:.2f}%")
    print(f"  Mechanism support only: {mech_prec*100:.2f}%")
    print(f"  Ensemble HIGH tier: {high_prec*100:.2f}%")
    print(f"  Ensemble TOP 10%: {top_10pct['is_hit'].mean()*100:.2f}%")

    # Feature importance
    lr_final = LogisticRegression(max_iter=1000, class_weight='balanced')
    lr_final.fit(X_norm, y)
    print(f"\nFeature coefficients:")
    for feat, coef in zip(feature_cols_adjusted, lr_final.coef_[0]):
        print(f"  {feat}: {coef:.3f}")

    # Save results
    results = {
        'high_tier_precision': float(high_prec),
        'mid_tier_precision': float(mid_conf['is_hit'].mean()),
        'low_tier_precision': float(low_conf['is_hit'].mean()),
        'top_10pct_precision': float(top_10pct['is_hit'].mean()),
        'top_20pct_precision': float(top_20pct['is_hit'].mean()),
        'tier1_only_precision': float(tier1_prec),
        'mech_support_precision': float(mech_prec),
        'success': bool(success),
        'feature_coefficients': {feat: float(coef) for feat, coef in zip(feature_cols_adjusted, lr_final.coef_[0])},
        'n_predictions': len(df),
    }

    results_file = PROJECT_ROOT / "data" / "analysis" / "h106_multi_signal_ensemble.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
