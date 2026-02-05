#!/usr/bin/env python3
"""
h132: High-Frequency Drug Mechanism Targeting

h126 found STRONG synergy (+4.90 pp) between frequency and mechanism.
High-freq drugs with mechanism support hit 21.6% vs 2.8% baseline.

This script identifies a "golden" subset of predictions with high precision
that could be prioritized for expert review in production.

SUCCESS CRITERIA: Identify subset with >25% precision
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
            drug_name = id_to_name.get(drug_id, drug_id)

            results.append({
                'drug_id': drug_id,
                'drug_name': drug_name,
                'disease_id': disease_id,
                'disease_name': disease_name,
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
    print("h132: High-Frequency Drug Mechanism Targeting")
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
    print(f"Base hit rate: {df['is_hit'].mean()*100:.2f}%")

    # Define "golden" prediction criteria
    print("\n" + "=" * 70)
    print("Defining Golden Prediction Criteria")
    print("=" * 70)

    # Define thresholds
    freq_thresholds = [5, 10, 15, 20, 25]

    print("\n1. HIGH FREQUENCY + MECHANISM SUPPORT")
    print("-" * 50)

    results = {}

    for freq_thresh in freq_thresholds:
        high_freq = df['train_frequency'] >= freq_thresh
        has_mech = df['mechanism_support'] == 1

        golden = df[high_freq & has_mech]
        if len(golden) > 0:
            precision = golden['is_hit'].mean() * 100
            n_hits = golden['is_hit'].sum()
            results[f'freq>={freq_thresh}_with_mech'] = {
                'n': len(golden),
                'n_hits': int(n_hits),
                'precision': precision,
                'pct_of_total': len(golden) / len(df) * 100,
            }
            print(f"  Freq >= {freq_thresh} + mechanism: {precision:.1f}% precision ({n_hits}/{len(golden)}, {len(golden)/len(df)*100:.1f}% of predictions)")

    # Test without mechanism requirement (high freq only)
    print("\n2. HIGH FREQUENCY ONLY (no mechanism requirement)")
    print("-" * 50)

    for freq_thresh in freq_thresholds:
        high_freq = df['train_frequency'] >= freq_thresh
        subset = df[high_freq]
        if len(subset) > 0:
            precision = subset['is_hit'].mean() * 100
            n_hits = subset['is_hit'].sum()
            results[f'freq>={freq_thresh}_no_mech'] = {
                'n': len(subset),
                'n_hits': int(n_hits),
                'precision': precision,
                'pct_of_total': len(subset) / len(df) * 100,
            }
            print(f"  Freq >= {freq_thresh}: {precision:.1f}% precision ({n_hits}/{len(subset)}, {len(subset)/len(df)*100:.1f}% of predictions)")

    # Test with tier 1 categories
    print("\n3. TIER 1 CATEGORIES + HIGH FREQUENCY + MECHANISM")
    print("-" * 50)

    tier1 = df['tier_inv'] == 2  # tier_inv = 3 - tier, so tier 1 = 2

    for freq_thresh in [5, 10, 15]:
        high_freq = df['train_frequency'] >= freq_thresh
        has_mech = df['mechanism_support'] == 1

        golden = df[tier1 & high_freq & has_mech]
        if len(golden) > 0:
            precision = golden['is_hit'].mean() * 100
            n_hits = golden['is_hit'].sum()
            results[f'tier1_freq>={freq_thresh}_mech'] = {
                'n': len(golden),
                'n_hits': int(n_hits),
                'precision': precision,
                'pct_of_total': len(golden) / len(df) * 100,
            }
            print(f"  Tier1 + Freq >= {freq_thresh} + mechanism: {precision:.1f}% precision ({n_hits}/{len(golden)}, {len(golden)/len(df)*100:.1f}% of predictions)")

    # Test with top rank constraint
    print("\n4. TOP RANK + HIGH FREQUENCY + MECHANISM")
    print("-" * 50)

    for rank_thresh in [5, 10]:
        for freq_thresh in [10, 15]:
            top_rank = df['rank'] <= rank_thresh
            high_freq = df['train_frequency'] >= freq_thresh
            has_mech = df['mechanism_support'] == 1

            golden = df[top_rank & high_freq & has_mech]
            if len(golden) > 0:
                precision = golden['is_hit'].mean() * 100
                n_hits = golden['is_hit'].sum()
                results[f'rank<={rank_thresh}_freq>={freq_thresh}_mech'] = {
                    'n': len(golden),
                    'n_hits': int(n_hits),
                    'precision': precision,
                    'pct_of_total': len(golden) / len(df) * 100,
                }
                print(f"  Rank <= {rank_thresh} + Freq >= {freq_thresh} + mechanism: {precision:.1f}% precision ({n_hits}/{len(golden)}, {len(golden)/len(df)*100:.1f}% of predictions)")

    # Find optimal "golden" criteria
    print("\n" + "=" * 70)
    print("OPTIMAL GOLDEN CRITERIA ANALYSIS")
    print("=" * 70)

    # Sort by precision (>25% target) and show best options
    high_precision = [(k, v) for k, v in results.items() if v['precision'] >= 25 and v['n'] >= 50]
    high_precision.sort(key=lambda x: -x[1]['precision'])

    print(f"\nCriteria achieving >25% precision with N>=50:")
    print(f"{'Criteria':<45} {'Precision':>10} {'N':>8} {'% Total':>10}")
    print("-" * 75)

    for criteria, stats in high_precision[:10]:
        print(f"{criteria:<45} {stats['precision']:>9.1f}% {stats['n']:>8} {stats['pct_of_total']:>9.1f}%")

    # Analyze best golden set
    best_criteria = high_precision[0] if high_precision else None

    if best_criteria:
        print(f"\n" + "=" * 70)
        print(f"BEST GOLDEN CRITERIA: {best_criteria[0]}")
        print("=" * 70)

        # Parse criteria
        criteria_name = best_criteria[0]
        if 'rank' in criteria_name:
            parts = criteria_name.split('_')
            rank_thresh = int(parts[0].replace('rank<=', ''))
            freq_thresh = int(parts[1].replace('freq>=', ''))
            has_mech = 'mech' in criteria_name
            has_tier = 'tier1' in criteria_name
        else:
            rank_thresh = 30  # default
            if 'tier1' in criteria_name:
                parts = criteria_name.split('_')
                freq_thresh = int(parts[1].replace('freq>=', ''))
                has_tier = True
            else:
                parts = criteria_name.split('_')
                freq_thresh = int(parts[0].replace('freq>=', ''))
                has_tier = False
            has_mech = 'mech' in criteria_name or 'with_mech' in criteria_name

        print(f"\nCriteria breakdown:")
        print(f"  - Frequency threshold: >= {freq_thresh}")
        print(f"  - Mechanism required: {has_mech}")
        print(f"  - Tier 1 only: {has_tier}")
        print(f"  - Rank threshold: <= {rank_thresh}")

        # Get sample golden predictions
        mask = (df['train_frequency'] >= freq_thresh)
        if has_mech:
            mask &= (df['mechanism_support'] == 1)
        if has_tier:
            mask &= (df['tier_inv'] == 2)
        if rank_thresh < 30:
            mask &= (df['rank'] <= rank_thresh)

        golden_df = df[mask].copy()
        golden_df['is_hit_str'] = golden_df['is_hit'].map({1: 'HIT', 0: 'miss'})

        print(f"\nSample golden predictions (showing hits first):")
        sample = golden_df.sort_values('is_hit', ascending=False).head(15)
        for _, row in sample.iterrows():
            print(f"  [{row['is_hit_str']:4}] {row['drug_name']:<25} -> {row['disease_name'][:30]:<30} (freq={row['train_frequency']:>3})")

    # Success criteria check
    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 70)

    # Find any criteria that achieves >25% with reasonable N
    success = False
    best_result = None

    for criteria, stats in results.items():
        if stats['precision'] >= 25 and stats['n'] >= 50:
            if best_result is None or stats['precision'] > best_result[1]['precision']:
                best_result = (criteria, stats)
            success = True

    if success and best_result:
        print(f"  VALIDATED: Found golden criteria with >25% precision")
        print(f"  Best: {best_result[0]}")
        print(f"  Precision: {best_result[1]['precision']:.1f}%")
        print(f"  N predictions: {best_result[1]['n']}")
        print(f"  Hits captured: {best_result[1]['n_hits']}")
    else:
        print(f"  INCONCLUSIVE: No single criteria achieves >25% precision with N>=50")
        # Show best option anyway
        if results:
            best = max(results.items(), key=lambda x: x[1]['precision'])
            print(f"  Best found: {best[0]} at {best[1]['precision']:.1f}% ({best[1]['n']} predictions)")

    # Save results
    output = {
        'all_criteria_results': results,
        'high_precision_criteria': [(k, v) for k, v in high_precision] if high_precision else [],
        'best_criteria': best_result if best_result else None,
        'success': success,
        'base_hit_rate': float(df['is_hit'].mean() * 100),
        'total_predictions': len(df),
    }

    results_file = ANALYSIS_DIR / "h132_golden_predictions.json"
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
