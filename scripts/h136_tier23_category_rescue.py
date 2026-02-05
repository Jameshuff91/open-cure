#!/usr/bin/env python3
"""
h136: Tier 2/3 Category Rescue: Can Different Filters Work?

h132 showed Tier 1 + freq + mech achieves 57.9% precision.
h137 showed Tier 2/3 have 13x lower drug overlap (0.005 vs 0.062 Jaccard).

Can we find category-specific filters that work for Tier 2/3?
- Cancer: maybe mechanism is more important than frequency?
- Cardiovascular: maybe specific drug classes work better?
- Infectious: known to have high model error

SUCCESS CRITERIA: Find criteria achieving >30% precision for at least one Tier 2/3 category
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

# Category tiers from h71
TIER_1_CATEGORIES = {'autoimmune', 'dermatological', 'psychiatric', 'ophthalmic'}
TIER_2_CATEGORIES = {'cardiovascular', 'other', 'cancer'}
TIER_3_CATEGORIES = {'infectious', 'neurological', 'metabolic', 'respiratory', 'gastrointestinal', 'hematological'}

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


def compute_mechanism_support(drug_id, disease_id, drug_targets, disease_genes) -> int:
    drug_genes = drug_targets.get(drug_id, set())
    dis_genes = disease_genes.get(disease_id, set())
    return 1 if len(drug_genes & dis_genes) > 0 else 0


def run_knn_with_features(emb_dict, train_gt, test_gt, disease_names, drug_targets, disease_genes,
                          id_to_name, k=20):
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
            drug_name = id_to_name.get(drug_id, drug_id)

            results.append({
                'drug_id': drug_id,
                'drug_name': drug_name,
                'disease_id': disease_id,
                'disease_name': disease_name,
                'mechanism_support': mech_support,
                'train_frequency': train_freq,
                'norm_score': norm_score,
                'rank': rank,
                'is_hit': 1 if is_hit else 0,
                'category': category,
            })

    return results


def main():
    print("h136: Tier 2/3 Category Rescue")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    emb_dict = load_node2vec_embeddings()
    name_to_drug_id, id_to_name = load_drugbank_lookup()
    mesh_mappings = load_mesh_mappings_from_file()
    ground_truth, disease_names = load_ground_truth(mesh_mappings, name_to_drug_id)
    drug_targets = load_drug_targets()
    disease_genes = load_disease_genes()

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
            emb_dict, train_gt, test_gt, disease_names, drug_targets, disease_genes, id_to_name, k=20
        )
        all_results.extend(seed_results)
        print(f"  Seed {seed}: {len(seed_results)} predictions")

    df = pd.DataFrame(all_results)
    print(f"\nTotal predictions: {len(df)}")

    # Analyze each non-Tier1 category
    print("\n" + "=" * 70)
    print("CATEGORY-SPECIFIC ANALYSIS")
    print("=" * 70)

    # Define categories to test
    test_categories = list(TIER_2_CATEGORIES) + list(TIER_3_CATEGORIES)
    test_categories = [c for c in test_categories if c != 'other']  # Skip 'other' - too heterogeneous

    category_results = {}

    for category in test_categories:
        cat_df = df[df['category'] == category]
        if len(cat_df) < 50:
            print(f"\n{category.upper()}: Skipped (only {len(cat_df)} predictions)")
            continue

        print(f"\n{category.upper()}: {len(cat_df)} predictions")
        print("-" * 50)

        base_precision = cat_df['is_hit'].mean() * 100
        print(f"  Baseline precision: {base_precision:.1f}%")

        results = {}

        # Test various filter combinations
        filters = [
            ('freq>=5', cat_df['train_frequency'] >= 5),
            ('freq>=10', cat_df['train_frequency'] >= 10),
            ('freq>=15', cat_df['train_frequency'] >= 15),
            ('mech', cat_df['mechanism_support'] == 1),
            ('rank<=5', cat_df['rank'] <= 5),
            ('rank<=10', cat_df['rank'] <= 10),
            ('score>0.5', cat_df['norm_score'] > 0.5),
            ('score>0.7', cat_df['norm_score'] > 0.7),
        ]

        # Test single filters
        for name, mask in filters:
            subset = cat_df[mask]
            if len(subset) >= 10:
                prec = subset['is_hit'].mean() * 100
                results[name] = {'n': len(subset), 'precision': prec}
                if prec >= 25:
                    print(f"  {name}: {prec:.1f}% ({len(subset)} preds)")

        # Test combinations
        combinations = [
            ('freq>=10 + mech', (cat_df['train_frequency'] >= 10) & (cat_df['mechanism_support'] == 1)),
            ('freq>=15 + mech', (cat_df['train_frequency'] >= 15) & (cat_df['mechanism_support'] == 1)),
            ('rank<=5 + freq>=10', (cat_df['rank'] <= 5) & (cat_df['train_frequency'] >= 10)),
            ('rank<=5 + mech', (cat_df['rank'] <= 5) & (cat_df['mechanism_support'] == 1)),
            ('rank<=10 + freq>=15 + mech', (cat_df['rank'] <= 10) & (cat_df['train_frequency'] >= 15) & (cat_df['mechanism_support'] == 1)),
            ('score>0.7 + mech', (cat_df['norm_score'] > 0.7) & (cat_df['mechanism_support'] == 1)),
            ('score>0.7 + freq>=10', (cat_df['norm_score'] > 0.7) & (cat_df['train_frequency'] >= 10)),
        ]

        for name, mask in combinations:
            subset = cat_df[mask]
            if len(subset) >= 10:
                prec = subset['is_hit'].mean() * 100
                results[name] = {'n': len(subset), 'precision': prec}
                if prec >= 25:
                    print(f"  {name}: {prec:.1f}% ({len(subset)} preds)")

        # Find best for this category
        if results:
            best = max(results.items(), key=lambda x: x[1]['precision'])
            results['best'] = best
            print(f"  BEST: {best[0]} = {best[1]['precision']:.1f}%")

            category_results[category] = {
                'n_predictions': len(cat_df),
                'base_precision': base_precision,
                'filters': results,
                'best_filter': best[0],
                'best_precision': best[1]['precision'],
            }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: BEST FILTERS BY CATEGORY")
    print("=" * 70)

    print(f"\n{'Category':<20} {'Best Filter':<30} {'Precision':>10} {'N':>8}")
    print("-" * 70)

    success = False
    best_category = None
    best_precision = 0

    for cat, data in sorted(category_results.items(), key=lambda x: -x[1]['best_precision']):
        print(f"{cat:<20} {data['best_filter']:<30} {data['best_precision']:>9.1f}% {data['filters'][data['best_filter']]['n']:>8}")
        if data['best_precision'] >= 30:
            success = True
            if data['best_precision'] > best_precision:
                best_precision = data['best_precision']
                best_category = cat

    # Success criteria
    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 70)

    if success:
        print(f"  VALIDATED: Found category with >30% precision")
        print(f"  Best: {best_category} at {best_precision:.1f}%")
    else:
        if category_results:
            overall_best = max(category_results.items(), key=lambda x: x[1]['best_precision'])
            print(f"  INVALIDATED: No Tier 2/3 category achieves >30% precision")
            print(f"  Best found: {overall_best[0]} at {overall_best[1]['best_precision']:.1f}%")
        else:
            print(f"  INVALIDATED: No category had enough data")

    # Save results
    output = {
        'category_results': category_results,
        'success': success,
        'best_category': best_category,
        'best_precision': best_precision if best_precision > 0 else None,
        'target': 30.0,
    }

    results_file = ANALYSIS_DIR / "h136_tier23_category_rescue.json"
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
