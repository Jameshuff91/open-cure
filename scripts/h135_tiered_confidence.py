#!/usr/bin/env python3
"""
h135: Production Tiered Confidence System

Combine findings from h123 (negative signals), h126 (XGBoost interactions),
h130 (Linear calibration), and h132 (golden criteria) into a unified
tiered confidence system.

TIERS:
- GOLDEN: Tier1 category + freq>=10 + mechanism (target ~57% precision)
- HIGH: freq>=15 + mechanism OR rank<=5 + freq>=10 (target ~30% precision)
- MEDIUM: freq>=5 + mechanism OR Linear-preferred (target ~15% precision)
- LOW: everything else not filtered
- FILTER: rank>20 OR no_targets OR (low_freq AND no_mechanism)

SUCCESS CRITERIA: Tier system provides >3x precision separation between GOLDEN and LOW
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
    print("Note: XGBoost not available")

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
            has_targets = drug_id in drug_targets and len(drug_targets.get(drug_id, set())) > 0

            results.append({
                'mechanism_support': mech_support,
                'train_frequency': train_freq,
                'tier_inv': 3 - tier,
                'norm_score': norm_score,
                'rank': rank,
                'is_hit': 1 if is_hit else 0,
                'category': category,
                'disease_tier': tier,
                'has_targets': 1 if has_targets else 0,
            })

    return results


def assign_tier(row):
    """Assign confidence tier based on combined criteria from h123, h126, h130, h132."""

    # FILTER tier (h123 negative signals)
    if row['rank'] > 20:
        return 'FILTER'
    if row['has_targets'] == 0:
        return 'FILTER'
    if row['train_frequency'] <= 2 and row['mechanism_support'] == 0:
        return 'FILTER'

    # GOLDEN tier (h132 - Tier1 + freq>=10 + mechanism)
    if row['disease_tier'] == 1 and row['train_frequency'] >= 10 and row['mechanism_support'] == 1:
        return 'GOLDEN'

    # HIGH tier (h132 variants)
    if row['train_frequency'] >= 15 and row['mechanism_support'] == 1:
        return 'HIGH'
    if row['rank'] <= 5 and row['train_frequency'] >= 10 and row['mechanism_support'] == 1:
        return 'HIGH'

    # MEDIUM tier
    if row['train_frequency'] >= 5 and row['mechanism_support'] == 1:
        return 'MEDIUM'
    if row['train_frequency'] >= 10:  # high freq without mechanism
        return 'MEDIUM'

    # LOW tier (everything else that passed filter)
    return 'LOW'


def main():
    print("h135: Production Tiered Confidence System")
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

    # Assign tiers
    print("\n" + "=" * 70)
    print("Assigning Confidence Tiers")
    print("=" * 70)

    df['confidence_tier'] = df.apply(assign_tier, axis=1)

    # Order for display
    tier_order = ['GOLDEN', 'HIGH', 'MEDIUM', 'LOW', 'FILTER']

    print(f"\n{'Tier':<10} {'Count':>10} {'% Total':>10} {'Precision':>12} {'Hits':>10}")
    print("-" * 55)

    tier_results = {}
    for tier in tier_order:
        tier_df = df[df['confidence_tier'] == tier]
        if len(tier_df) > 0:
            precision = tier_df['is_hit'].mean() * 100
            n_hits = tier_df['is_hit'].sum()
            pct_total = len(tier_df) / len(df) * 100
            tier_results[tier] = {
                'n': len(tier_df),
                'precision': precision,
                'n_hits': int(n_hits),
                'pct_total': pct_total,
            }
            print(f"{tier:<10} {len(tier_df):>10} {pct_total:>9.1f}% {precision:>11.1f}% {n_hits:>10}")

    # Calculate separation
    print("\n" + "=" * 70)
    print("Precision Separation Analysis")
    print("=" * 70)

    golden_prec = tier_results.get('GOLDEN', {}).get('precision', 0)
    high_prec = tier_results.get('HIGH', {}).get('precision', 0)
    medium_prec = tier_results.get('MEDIUM', {}).get('precision', 0)
    low_prec = tier_results.get('LOW', {}).get('precision', 0)
    filter_prec = tier_results.get('FILTER', {}).get('precision', 0)

    print(f"\nPrecision by tier:")
    print(f"  GOLDEN: {golden_prec:.1f}%")
    print(f"  HIGH:   {high_prec:.1f}%")
    print(f"  MEDIUM: {medium_prec:.1f}%")
    print(f"  LOW:    {low_prec:.1f}%")
    print(f"  FILTER: {filter_prec:.1f}%")

    # Separation ratios
    print(f"\nSeparation ratios:")
    if low_prec > 0:
        golden_low = golden_prec / low_prec
        high_low = high_prec / low_prec
        print(f"  GOLDEN/LOW:  {golden_low:.1f}x")
        print(f"  HIGH/LOW:    {high_low:.1f}x")
    if filter_prec > 0:
        golden_filter = golden_prec / filter_prec
        print(f"  GOLDEN/FILTER: {golden_filter:.1f}x")

    # Coverage analysis
    print("\n" + "=" * 70)
    print("Coverage Analysis")
    print("=" * 70)

    total_hits = df['is_hit'].sum()
    print(f"\nTotal hits: {total_hits}")

    cumulative_hits = 0
    print(f"\n{'Tier':<10} {'Hits':>10} {'Cumulative':>12} {'% of All Hits':>15}")
    print("-" * 50)

    for tier in ['GOLDEN', 'HIGH', 'MEDIUM', 'LOW', 'FILTER']:
        if tier in tier_results:
            hits = tier_results[tier]['n_hits']
            cumulative_hits += hits
            pct_hits = (cumulative_hits / total_hits) * 100
            print(f"{tier:<10} {hits:>10} {cumulative_hits:>12} {pct_hits:>14.1f}%")

    # Success criteria check
    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 70)

    # Target: >3x precision separation between GOLDEN and LOW
    if low_prec > 0:
        separation = golden_prec / low_prec
        target_sep = 3.0

        if separation >= target_sep:
            print(f"  VALIDATED: GOLDEN/LOW separation = {separation:.1f}x (>= {target_sep}x target)")
            success = True
        else:
            print(f"  INCONCLUSIVE: GOLDEN/LOW separation = {separation:.1f}x (< {target_sep}x target)")
            success = False
    else:
        print("  Could not calculate separation (LOW tier empty)")
        success = False

    # Additional validation: monotonic precision decrease
    precisions = [golden_prec, high_prec, medium_prec, low_prec, filter_prec]
    monotonic = all(precisions[i] >= precisions[i+1] for i in range(len(precisions)-1))
    print(f"\n  Monotonic decrease: {'YES' if monotonic else 'NO'}")

    # Summary
    print("\n" + "=" * 70)
    print("PRODUCTION TIER SYSTEM SUMMARY")
    print("=" * 70)

    print("""
TIER DEFINITIONS:
-----------------
GOLDEN:  Tier1 category + freq>=10 + mechanism support
         Target: ~55-60% precision
         Use: Highest confidence predictions for expert review

HIGH:    freq>=15 + mechanism OR rank<=5 + freq>=10 + mechanism
         Target: ~25-30% precision
         Use: Strong candidates, worth detailed investigation

MEDIUM:  freq>=5 + mechanism OR freq>=10
         Target: ~15-20% precision
         Use: Moderate confidence, needs additional validation

LOW:     Everything else passing filter
         Target: ~5-10% precision
         Use: Low confidence, bulk predictions

FILTER:  rank>20 OR no_targets OR (freq<=2 AND no_mechanism)
         Target: <5% precision
         Use: Excluded from production output
""")

    # Save results
    output = {
        'tier_results': tier_results,
        'separations': {
            'golden_low': float(golden_prec / low_prec) if low_prec > 0 else None,
            'high_low': float(high_prec / low_prec) if low_prec > 0 else None,
            'golden_filter': float(golden_prec / filter_prec) if filter_prec > 0 else None,
        },
        'monotonic_decrease': monotonic,
        'total_predictions': len(df),
        'total_hits': int(total_hits),
        'success': success,
        'tier_definitions': {
            'GOLDEN': 'Tier1 category + freq>=10 + mechanism',
            'HIGH': 'freq>=15 + mechanism OR rank<=5 + freq>=10 + mechanism',
            'MEDIUM': 'freq>=5 + mechanism OR freq>=10',
            'LOW': 'All else passing filter',
            'FILTER': 'rank>20 OR no_targets OR (freq<=2 AND no_mechanism)',
        }
    }

    results_file = ANALYSIS_DIR / "h135_tiered_confidence.json"
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
