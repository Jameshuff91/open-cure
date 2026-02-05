#!/usr/bin/env python3
"""
h116: Category Tier 2.0 - Disease-Specific Calibration

PURPOSE:
    h111 showed category tier is the weakest confidence signal (r=0.082 with hits).
    The current tier system is coarse: Tier 1 = autoimmune/dermatological/psychiatric/ophthalmic.
    Within Tier 3, some diseases may still be easy to predict.

    This hypothesis tests whether disease-specific historical hit rate
    (from training data) improves hit prediction beyond coarse category tiers.

METHODOLOGY:
    1. For each test fold, compute per-disease hit rates from TRAINING data only
    2. Use leave-one-disease-out within training to get unbiased estimates
    3. Apply these estimates to score test predictions
    4. Compare correlation with hits vs coarse category tier

SUCCESS CRITERIA:
    Disease-specific calibration achieves r > 0.15 with hits (vs 0.082 for tier)
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, pointbiserialr
from sklearn.metrics.pairwise import cosine_similarity

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


def compute_disease_hit_rate_cv(train_diseases, train_gt, emb_dict, k=20):
    """
    Compute hit rate for each training disease using leave-one-out cross-validation.
    This gives an unbiased estimate of how well kNN performs for each disease.
    """
    disease_list = [d for d in train_diseases if d in emb_dict]
    if len(disease_list) < 2:
        return {}

    disease_embs = np.array([emb_dict[d] for d in disease_list], dtype=np.float32)
    sims = cosine_similarity(disease_embs)

    hit_rates = {}

    for i, disease_id in enumerate(disease_list):
        gt_drugs = {d for d in train_gt[disease_id] if d in emb_dict}
        if not gt_drugs:
            hit_rates[disease_id] = 0.0
            continue

        # Get k nearest neighbors (excluding self)
        neighbor_sims = sims[i].copy()
        neighbor_sims[i] = -1  # Exclude self
        top_k_idx = np.argsort(neighbor_sims)[-k:]

        # Count drug recommendations from neighbors
        drug_counts = defaultdict(float)
        for idx in top_k_idx:
            neighbor_disease = disease_list[idx]
            neighbor_sim = neighbor_sims[idx]
            for drug_id in train_gt[neighbor_disease]:
                if drug_id in emb_dict:
                    drug_counts[drug_id] += neighbor_sim

        if not drug_counts:
            hit_rates[disease_id] = 0.0
            continue

        # Check hit rate @ 30
        sorted_drugs = sorted(drug_counts.items(), key=lambda x: x[1], reverse=True)[:30]
        hits = sum(1 for drug_id, _ in sorted_drugs if drug_id in gt_drugs)
        total = len(gt_drugs)
        hit_rates[disease_id] = hits / total if total > 0 else 0.0

    return hit_rates


def run_knn_with_calibration(emb_dict, train_gt, test_gt, disease_names, disease_hit_rates, k=20):
    """Run kNN and collect predictions with calibration features."""
    train_disease_list = [d for d in train_gt if d in emb_dict]
    train_disease_embs = np.array([emb_dict[d] for d in train_disease_list], dtype=np.float32)

    # Compute category-level average hit rate
    category_hit_rates = defaultdict(list)
    for disease_id, rate in disease_hit_rates.items():
        if disease_id in disease_names:
            cat = categorize_disease(disease_names[disease_id])
            category_hit_rates[cat].append(rate)

    category_avg = {cat: np.mean(rates) if rates else 0.0
                    for cat, rates in category_hit_rates.items()}

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

        # Get calibrated hit rate: disease-specific if available, else category average
        # For test diseases, use category average (no leakage)
        calibrated_rate = category_avg.get(category, 0.0)

        # kNN prediction
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

        for rank, (drug_id, score) in enumerate(sorted_drugs, 1):
            is_hit = drug_id in gt_drugs

            results.append({
                'disease_id': disease_id,
                'disease_name': disease_name,
                'category': category,
                'tier': tier,
                'tier_inv': 3 - tier,
                'calibrated_rate': calibrated_rate,
                'rank': rank,
                'is_hit': 1 if is_hit else 0,
            })

    return results


def main():
    print("h116: Category Tier 2.0 - Disease-Specific Calibration")
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

        # Compute hit rates from training data using LOO-CV
        disease_hit_rates = compute_disease_hit_rate_cv(train_diseases, train_gt, emb_dict, k=20)

        seed_results = run_knn_with_calibration(
            emb_dict, train_gt, test_gt, disease_names, disease_hit_rates, k=20
        )
        all_results.extend(seed_results)
        print(f"  Seed {seed}: {len(seed_results)} predictions")

    df = pd.DataFrame(all_results)
    print(f"\nTotal predictions: {len(df)}")
    print(f"Base hit rate: {df['is_hit'].mean()*100:.2f}%")

    # Correlation analysis
    print("\n" + "=" * 70)
    print("CORRELATION WITH HITS")
    print("=" * 70)

    # Tier inverse correlation
    tier_corr, tier_p = pointbiserialr(df['is_hit'], df['tier_inv'])
    print(f"\n1. Category Tier (inverted):")
    print(f"   r = {tier_corr:.3f}, p = {tier_p:.4f}")

    # Calibrated rate correlation
    cal_corr, cal_p = pointbiserialr(df['is_hit'], df['calibrated_rate'])
    print(f"\n2. Calibrated Category Hit Rate:")
    print(f"   r = {cal_corr:.3f}, p = {cal_p:.4f}")

    # Per-category analysis
    print("\n" + "=" * 70)
    print("PER-CATEGORY CALIBRATION ACCURACY")
    print("=" * 70)

    for category in sorted(df['category'].unique()):
        cat_df = df[df['category'] == category]
        actual_hit_rate = cat_df['is_hit'].mean()
        predicted_rate = cat_df['calibrated_rate'].iloc[0] if len(cat_df) > 0 else 0
        error = abs(actual_hit_rate - predicted_rate)
        n = len(cat_df)
        print(f"  {category:20s}: Actual={actual_hit_rate*100:5.1f}%, Predicted={predicted_rate*100:5.1f}%, Error={error*100:5.1f}% (n={n})")

    # Precision analysis
    print("\n" + "=" * 70)
    print("PRECISION BY CALIBRATED RATE")
    print("=" * 70)

    # Quantiles
    q1 = df['calibrated_rate'].quantile(0.33)
    q2 = df['calibrated_rate'].quantile(0.66)

    low_df = df[df['calibrated_rate'] <= q1]
    mid_df = df[(df['calibrated_rate'] > q1) & (df['calibrated_rate'] <= q2)]
    high_df = df[df['calibrated_rate'] > q2]

    print(f"\n  LOW calibration (≤{q1*100:.1f}%):  {low_df['is_hit'].mean()*100:5.2f}% precision (n={len(low_df)})")
    print(f"  MID calibration ({q1*100:.1f}%-{q2*100:.1f}%):  {mid_df['is_hit'].mean()*100:5.2f}% precision (n={len(mid_df)})")
    print(f"  HIGH calibration (>{q2*100:.1f}%): {high_df['is_hit'].mean()*100:5.2f}% precision (n={len(high_df)})")

    high_low_diff = high_df['is_hit'].mean() - low_df['is_hit'].mean()
    print(f"\n  HIGH - LOW difference: {high_low_diff*100:+.2f} pp")

    # Success criteria check
    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 70)

    success = cal_corr > 0.15

    print(f"\n  Target: r > 0.15 (vs 0.082 for tier)")
    print(f"  Tier correlation: r = {tier_corr:.3f}")
    print(f"  Calibrated rate correlation: r = {cal_corr:.3f}")

    if success:
        improvement = (cal_corr - tier_corr) / tier_corr * 100
        print(f"  → VALIDATED: Calibrated rate achieves r = {cal_corr:.3f} (>{tier_corr:.3f}, +{improvement:.0f}%)")
    else:
        print(f"  → INVALIDATED: Calibrated rate r = {cal_corr:.3f} does not exceed 0.15 threshold")

    # Save results
    output = {
        'tier_correlation': float(tier_corr),
        'tier_p_value': float(tier_p),
        'calibrated_correlation': float(cal_corr),
        'calibrated_p_value': float(cal_p),
        'high_low_precision_diff': float(high_low_diff),
        'success': bool(success),
    }

    results_file = PROJECT_ROOT / "data" / "analysis" / "h116_per_disease_calibration.json"
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
