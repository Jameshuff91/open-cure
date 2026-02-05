#!/usr/bin/env python3
"""
h130: Linear Model Calibration Analysis

h126 found that Linear model preferred predictions have 14.9% hit rate vs XGBoost-preferred 2.8%.
This suggests Linear model may be better calibrated even though XGBoost has higher precision at top-k.

This script analyzes:
1. WHY Linear is better at some predictions
2. Feature patterns that distinguish Linear-preferred vs XGBoost-preferred
3. Whether a hybrid approach could improve overall results

SUCCESS CRITERIA: Identify actionable pattern for model selection
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

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("ERROR: XGBoost required for this analysis")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from disease_name_matcher import DiseaseMatcher, load_mesh_mappings

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"

SEEDS = [42, 123, 456, 789, 1024]

# Category tiers
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
                'rank': rank,
                'is_hit': 1 if is_hit else 0,
                'category': category,
            })

    return results


def main():
    print("h130: Linear Model Calibration Analysis")
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

    # Train both models and get predictions
    print("\n" + "=" * 70)
    print("Training Models and Getting Predictions")
    print("=" * 70)

    features = ['mechanism_support', 'train_frequency', 'tier_inv', 'norm_score']
    X = df[features].values
    y = df['is_hit'].values

    # Standardize
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    # Get cross-validated predictions from both models
    print("\nTraining Logistic Regression...")
    linear_proba = cross_val_predict(
        LogisticRegression(max_iter=1000, class_weight='balanced'),
        X_norm, y, cv=5, method='predict_proba'
    )[:, 1]

    print("Training XGBoost...")
    xgb_proba = cross_val_predict(
        xgb.XGBClassifier(n_estimators=50, max_depth=2, learning_rate=0.1,
                          random_state=42, eval_metric='logloss'),
        X_norm, y, cv=5, method='predict_proba'
    )[:, 1]

    df['linear_score'] = linear_proba
    df['xgb_score'] = xgb_proba
    df['score_diff'] = df['linear_score'] - df['xgb_score']

    # 1. Analyze where models differ
    print("\n" + "=" * 70)
    print("1. Where Models Differ")
    print("=" * 70)

    # Quartiles of score difference
    q1, q3 = df['score_diff'].quantile([0.25, 0.75])
    linear_prefers = df[df['score_diff'] > q3]  # Top 25% where linear >> xgb
    xgb_prefers = df[df['score_diff'] < q1]      # Top 25% where xgb >> linear
    similar = df[(df['score_diff'] >= q1) & (df['score_diff'] <= q3)]

    print(f"\nLinear strongly prefers (top 25%): {len(linear_prefers)} predictions")
    print(f"  Hit rate: {linear_prefers['is_hit'].mean()*100:.2f}%")
    print(f"  Avg mechanism: {linear_prefers['mechanism_support'].mean():.3f}")
    print(f"  Avg frequency: {linear_prefers['train_frequency'].mean():.2f}")
    print(f"  Avg tier_inv: {linear_prefers['tier_inv'].mean():.2f}")
    print(f"  Avg norm_score: {linear_prefers['norm_score'].mean():.3f}")

    print(f"\nXGBoost strongly prefers (top 25%): {len(xgb_prefers)} predictions")
    print(f"  Hit rate: {xgb_prefers['is_hit'].mean()*100:.2f}%")
    print(f"  Avg mechanism: {xgb_prefers['mechanism_support'].mean():.3f}")
    print(f"  Avg frequency: {xgb_prefers['train_frequency'].mean():.2f}")
    print(f"  Avg tier_inv: {xgb_prefers['tier_inv'].mean():.2f}")
    print(f"  Avg norm_score: {xgb_prefers['norm_score'].mean():.3f}")

    print(f"\nBoth agree (middle 50%): {len(similar)} predictions")
    print(f"  Hit rate: {similar['is_hit'].mean()*100:.2f}%")

    # 2. Feature patterns
    print("\n" + "=" * 70)
    print("2. Feature Patterns")
    print("=" * 70)

    print("\nCorrelation of (linear_score - xgb_score) with features:")
    for feat in features + ['is_hit']:
        corr = df['score_diff'].corr(df[feat])
        print(f"  {feat:<20}: r = {corr:+.3f}")

    # 3. By category analysis
    print("\n" + "=" * 70)
    print("3. Category Analysis")
    print("=" * 70)

    print(f"\n{'Category':<20} {'Linear Pref HR':>15} {'XGB Pref HR':>15} {'Diff':>10}")
    print("-" * 65)

    category_results = {}
    for cat in df['category'].unique():
        cat_df = df[df['category'] == cat]
        cat_linear = cat_df[cat_df['score_diff'] > cat_df['score_diff'].quantile(0.75)]
        cat_xgb = cat_df[cat_df['score_diff'] < cat_df['score_diff'].quantile(0.25)]

        if len(cat_linear) > 0 and len(cat_xgb) > 0:
            linear_hr = cat_linear['is_hit'].mean() * 100
            xgb_hr = cat_xgb['is_hit'].mean() * 100
            diff = linear_hr - xgb_hr
            category_results[cat] = {
                'linear_hr': linear_hr,
                'xgb_hr': xgb_hr,
                'diff': diff,
                'n_linear': len(cat_linear),
                'n_xgb': len(cat_xgb)
            }
            print(f"{cat:<20} {linear_hr:>14.1f}% {xgb_hr:>14.1f}% {diff:>+9.1f}")

    # 4. Understand WHY linear is better for some predictions
    print("\n" + "=" * 70)
    print("4. WHY Linear is Better (Feature Analysis)")
    print("=" * 70)

    # Break down by feature combinations
    high_freq = df['train_frequency'] >= df['train_frequency'].quantile(0.75)
    has_mech = df['mechanism_support'] == 1
    tier1 = df['tier_inv'] == 2
    high_score = df['norm_score'] >= df['norm_score'].quantile(0.75)

    combos = [
        ('High freq + mechanism', high_freq & has_mech),
        ('High freq + no mechanism', high_freq & ~has_mech),
        ('Low freq + mechanism', ~high_freq & has_mech),
        ('Low freq + no mechanism', ~high_freq & ~has_mech),
        ('Tier 1 diseases', tier1),
        ('High kNN score', high_score),
    ]

    print(f"\n{'Condition':<30} {'Linear Win':>12} {'XGB Win':>12} {'Linear Better?':>15}")
    print("-" * 72)

    for name, mask in combos:
        subset = df[mask]
        if len(subset) > 100:
            # Which model has higher hit rate in this subset?
            linear_top = subset.nlargest(len(subset)//4, 'linear_score')
            xgb_top = subset.nlargest(len(subset)//4, 'xgb_score')
            linear_hr = linear_top['is_hit'].mean() * 100
            xgb_hr = xgb_top['is_hit'].mean() * 100
            better = "YES" if linear_hr > xgb_hr else "NO"
            print(f"{name:<30} {linear_hr:>11.1f}% {xgb_hr:>11.1f}% {better:>15}")

    # 5. Hybrid approach test
    print("\n" + "=" * 70)
    print("5. Hybrid Approach: XGBoost Ranking + Linear Calibration")
    print("=" * 70)

    # Strategy: rank by XGBoost but use Linear confidence for filtering
    # Take top 10% by XGBoost, then filter by Linear confidence

    n = len(df)
    n_top10 = n // 10

    # Pure XGBoost top 10%
    xgb_top10 = df.nlargest(n_top10, 'xgb_score')
    xgb_precision = xgb_top10['is_hit'].mean() * 100

    # Pure Linear top 10%
    linear_top10 = df.nlargest(n_top10, 'linear_score')
    linear_precision = linear_top10['is_hit'].mean() * 100

    # Hybrid: top 15% by XGBoost, then top 66% of those by Linear
    n_top15 = int(n * 0.15)
    xgb_top15 = df.nlargest(n_top15, 'xgb_score')
    hybrid_top10 = xgb_top15.nlargest(n_top10, 'linear_score')
    hybrid_precision = hybrid_top10['is_hit'].mean() * 100

    # Alternative: average of both scores
    df['avg_score'] = (df['linear_score'] + df['xgb_score']) / 2
    avg_top10 = df.nlargest(n_top10, 'avg_score')
    avg_precision = avg_top10['is_hit'].mean() * 100

    print(f"\nTop 10% precision comparison:")
    print(f"  XGBoost only:    {xgb_precision:.2f}%")
    print(f"  Linear only:     {linear_precision:.2f}%")
    print(f"  Hybrid (XGB->LR): {hybrid_precision:.2f}%")
    print(f"  Average scores:  {avg_precision:.2f}%")

    best_method = max([
        ('XGBoost', xgb_precision),
        ('Linear', linear_precision),
        ('Hybrid', hybrid_precision),
        ('Average', avg_precision),
    ], key=lambda x: x[1])

    # 6. Key insight
    print("\n" + "=" * 70)
    print("6. KEY INSIGHT: When to Trust Each Model")
    print("=" * 70)

    # What characterizes predictions where Linear > XGBoost?
    linear_wins = df[(df['linear_score'] > df['xgb_score']) & (df['is_hit'] == 1)]
    xgb_wins = df[(df['xgb_score'] > df['linear_score']) & (df['is_hit'] == 1)]

    print(f"\nHits where Linear scored higher: {len(linear_wins)}")
    print(f"  Avg frequency: {linear_wins['train_frequency'].mean():.2f}")
    print(f"  Avg mechanism: {linear_wins['mechanism_support'].mean():.3f}")
    print(f"  Avg tier_inv: {linear_wins['tier_inv'].mean():.2f}")

    print(f"\nHits where XGBoost scored higher: {len(xgb_wins)}")
    print(f"  Avg frequency: {xgb_wins['train_frequency'].mean():.2f}")
    print(f"  Avg mechanism: {xgb_wins['mechanism_support'].mean():.3f}")
    print(f"  Avg tier_inv: {xgb_wins['tier_inv'].mean():.2f}")

    # Success criteria
    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 70)

    # Did we find an actionable pattern?
    linear_better_categories = [c for c, v in category_results.items() if v['diff'] > 5]
    xgb_better_categories = [c for c, v in category_results.items() if v['diff'] < -5]

    if linear_better_categories or xgb_better_categories or hybrid_precision > max(xgb_precision, linear_precision):
        print("  VALIDATED: Found actionable patterns")
        if linear_better_categories:
            print(f"  Linear better for: {linear_better_categories}")
        if xgb_better_categories:
            print(f"  XGBoost better for: {xgb_better_categories}")
        if hybrid_precision > max(xgb_precision, linear_precision):
            print(f"  Hybrid approach improves by +{hybrid_precision - max(xgb_precision, linear_precision):.2f}%")
        success = True
    else:
        print("  INCONCLUSIVE: No clear actionable pattern for model selection")
        print(f"  Best method: {best_method[0]} at {best_method[1]:.2f}%")
        success = False

    # Save results
    output = {
        'linear_prefers': {
            'n': len(linear_prefers),
            'hit_rate': float(linear_prefers['is_hit'].mean() * 100),
            'avg_features': {
                'mechanism': float(linear_prefers['mechanism_support'].mean()),
                'frequency': float(linear_prefers['train_frequency'].mean()),
                'tier_inv': float(linear_prefers['tier_inv'].mean()),
                'norm_score': float(linear_prefers['norm_score'].mean()),
            }
        },
        'xgb_prefers': {
            'n': len(xgb_prefers),
            'hit_rate': float(xgb_prefers['is_hit'].mean() * 100),
            'avg_features': {
                'mechanism': float(xgb_prefers['mechanism_support'].mean()),
                'frequency': float(xgb_prefers['train_frequency'].mean()),
                'tier_inv': float(xgb_prefers['tier_inv'].mean()),
                'norm_score': float(xgb_prefers['norm_score'].mean()),
            }
        },
        'category_results': category_results,
        'precision_comparison': {
            'xgb_top10': xgb_precision,
            'linear_top10': linear_precision,
            'hybrid_top10': hybrid_precision,
            'avg_top10': avg_precision,
        },
        'correlations': {
            feat: float(df['score_diff'].corr(df[feat]))
            for feat in features + ['is_hit']
        },
        'success': success,
        'best_method': best_method,
    }

    results_file = ANALYSIS_DIR / "h130_linear_calibration.json"
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
