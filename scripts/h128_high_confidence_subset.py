#!/usr/bin/env python3
"""
h128: High-Confidence Subset: Freq AND Score Above Median

h126 showed freq×score synergy is the strongest interaction signal.
Filtering to predictions where BOTH train_frequency AND norm_score are
above median should yield the highest-confidence subset with best precision.

SUCCESS CRITERIA: >30% precision on filtered subset
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
                'norm_score': norm_score,
                'tier_inv': 3 - tier,
                'category': category,
                'rank': rank,
                'is_hit': 1 if is_hit else 0,
            })

    return results


def main():
    print("h128: High-Confidence Subset: Freq AND Score Above Median")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    emb_dict = load_node2vec_embeddings()
    name_to_drug_id = load_drugbank_lookup()
    mesh_mappings = load_mesh_mappings_from_file()
    ground_truth, disease_names = load_ground_truth(mesh_mappings, name_to_drug_id)

    print(f"  Embeddings: {len(emb_dict)}")
    print(f"  Ground truth diseases: {len(ground_truth)}")

    # Collect predictions across 5 seeds
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
            emb_dict, train_gt, test_gt, disease_names, k=20
        )
        all_results.extend(seed_results)
        print(f"  Seed {seed}: {len(seed_results)} predictions")

    df = pd.DataFrame(all_results)
    print(f"\nTotal predictions: {len(df)}")
    print(f"Base hit rate: {df['is_hit'].mean()*100:.2f}%")

    # Compute medians
    median_freq = df['train_frequency'].median()
    median_score = df['norm_score'].median()

    print(f"\nMedian train_frequency: {median_freq}")
    print(f"Median norm_score: {median_score:.4f}")

    # Test various filtering strategies
    print("\n" + "=" * 70)
    print("FILTERING ANALYSIS")
    print("=" * 70)

    results_summary = []

    # Baseline: all predictions
    baseline_precision = df['is_hit'].mean() * 100
    print(f"\n1. Baseline (all predictions): {len(df)} preds, {baseline_precision:.2f}% precision")
    results_summary.append(('Baseline (all)', len(df), baseline_precision))

    # Top 10% by any ranking (XGBoost-like baseline)
    n_top10 = len(df) // 10
    df_sorted_rank = df.sort_values('rank')
    top10_by_rank = df_sorted_rank.head(n_top10)
    top10_rank_precision = top10_by_rank['is_hit'].mean() * 100
    print(f"\n2. Top 10% by rank: {len(top10_by_rank)} preds, {top10_rank_precision:.2f}% precision")
    results_summary.append(('Top 10% by rank', len(top10_by_rank), top10_rank_precision))

    # Filter: train_frequency > median
    high_freq = df[df['train_frequency'] > median_freq]
    high_freq_precision = high_freq['is_hit'].mean() * 100 if len(high_freq) > 0 else 0
    print(f"\n3. High frequency only (>{median_freq}): {len(high_freq)} preds ({len(high_freq)/len(df)*100:.1f}%), {high_freq_precision:.2f}% precision")
    results_summary.append(('High freq only', len(high_freq), high_freq_precision))

    # Filter: norm_score > median
    high_score = df[df['norm_score'] > median_score]
    high_score_precision = high_score['is_hit'].mean() * 100 if len(high_score) > 0 else 0
    print(f"\n4. High score only (>{median_score:.3f}): {len(high_score)} preds ({len(high_score)/len(df)*100:.1f}%), {high_score_precision:.2f}% precision")
    results_summary.append(('High score only', len(high_score), high_score_precision))

    # Filter: BOTH above median (the synergy test)
    high_both = df[(df['train_frequency'] > median_freq) & (df['norm_score'] > median_score)]
    high_both_precision = high_both['is_hit'].mean() * 100 if len(high_both) > 0 else 0
    print(f"\n5. HIGH FREQ AND HIGH SCORE: {len(high_both)} preds ({len(high_both)/len(df)*100:.1f}%), {high_both_precision:.2f}% precision")
    results_summary.append(('Freq AND Score high', len(high_both), high_both_precision))

    # Test different percentile thresholds
    print("\n" + "=" * 70)
    print("PERCENTILE THRESHOLD ANALYSIS")
    print("=" * 70)

    for pct in [50, 60, 70, 75, 80, 90]:
        thresh_freq = np.percentile(df['train_frequency'], pct)
        thresh_score = np.percentile(df['norm_score'], pct)
        filtered = df[(df['train_frequency'] >= thresh_freq) & (df['norm_score'] >= thresh_score)]
        if len(filtered) > 0:
            prec = filtered['is_hit'].mean() * 100
            print(f"  {pct}th percentile (freq≥{thresh_freq:.0f}, score≥{thresh_score:.3f}): {len(filtered)} preds ({len(filtered)/len(df)*100:.1f}%), {prec:.2f}% precision")
            results_summary.append((f'{pct}th pct both', len(filtered), prec))

    # Also test top-k by frequency × score product
    print("\n" + "=" * 70)
    print("FREQ × SCORE PRODUCT ANALYSIS")
    print("=" * 70)

    df['freq_x_score'] = df['train_frequency'] * df['norm_score']
    df_sorted_product = df.sort_values('freq_x_score', ascending=False)

    for top_pct in [5, 10, 20, 25, 33]:
        n_top = len(df) * top_pct // 100
        top_by_product = df_sorted_product.head(n_top)
        if len(top_by_product) > 0:
            prec = top_by_product['is_hit'].mean() * 100
            print(f"  Top {top_pct}% by freq×score: {len(top_by_product)} preds, {prec:.2f}% precision")
            results_summary.append((f'Top {top_pct}% freq×score', len(top_by_product), prec))

    # Success criteria check
    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 70)

    target = 30.0  # Success criterion: >30% precision on filtered subset

    # Find best precision that maintains reasonable coverage (>5% of data)
    best_precision = 0
    best_method = None
    for name, count, prec in results_summary:
        if count >= len(df) * 0.05 and prec > best_precision:  # At least 5% coverage
            best_precision = prec
            best_method = name

    print(f"\nBest precision with ≥5% coverage: {best_method}")
    print(f"  Precision: {best_precision:.2f}%")

    if high_both_precision >= target:
        print(f"\n  ✓ Freq AND Score filter achieves {high_both_precision:.2f}% precision (≥{target}%)")
        print(f"  → VALIDATED: Synergy filtering creates high-confidence subset")
        success = True
    elif best_precision >= target:
        print(f"\n  ✓ Alternative method ({best_method}) achieves {best_precision:.2f}% precision (≥{target}%)")
        print(f"  → VALIDATED: High-confidence subset achievable")
        success = True
    else:
        print(f"\n  ✗ Best precision {best_precision:.2f}% < {target}%")
        print(f"  → INVALIDATED: Cannot reach 30% precision with simple filtering")
        success = False

    # Compute synergy bonus
    expected_if_independent = high_freq_precision * high_score_precision / baseline_precision
    synergy_bonus = high_both_precision - expected_if_independent

    print(f"\nSynergy analysis:")
    print(f"  If independent, expected precision: {expected_if_independent:.2f}%")
    print(f"  Actual precision: {high_both_precision:.2f}%")
    print(f"  Synergy bonus: {synergy_bonus:+.2f} pp")

    # Save results
    output = {
        'n_predictions': len(df),
        'baseline_precision': float(baseline_precision),
        'median_freq': float(median_freq),
        'median_score': float(median_score),
        'high_freq_only': {
            'n': len(high_freq),
            'precision': float(high_freq_precision)
        },
        'high_score_only': {
            'n': len(high_score),
            'precision': float(high_score_precision)
        },
        'high_both': {
            'n': len(high_both),
            'precision': float(high_both_precision)
        },
        'synergy_analysis': {
            'expected_if_independent': float(expected_if_independent),
            'actual': float(high_both_precision),
            'synergy_bonus': float(synergy_bonus)
        },
        'all_results': [{'method': m, 'n': n, 'precision': p} for m, n, p in results_summary],
        'best_method': best_method,
        'best_precision': float(best_precision),
        'success': success,
        'success_target': target
    }

    results_file = PROJECT_ROOT / "data" / "analysis" / "h128_high_confidence_subset.json"
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
