#!/usr/bin/env python3
"""
h111: Confidence Feature Independence Analysis

PURPOSE:
    Determine if validated confidence signals are independent (uncorrelated):
    1. Mechanism support (h97): +6.5 pp
    2. Drug training frequency (h108): +9.4 pp
    3. Category tier (h71): Tier 1 = 93-100% precision
    4. Chemical similarity (h109): +8.81 pp
    5. kNN score/rank

QUESTION:
    Are these signals correlated or orthogonal? If independent, combining them
    should provide additive precision gains. If correlated, ensemble may not help.

SUCCESS CRITERIA:
    - Compute correlation matrix between all signals
    - For pairs with correlation < 0.3, check if combining provides additive gain
    - Document which signals are independent and worth combining
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, List

import numpy as np
import pandas as pd
from scipy import stats
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
    """Load disease -> associated genes mapping with both key formats."""
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


def load_chemical_similarity() -> Dict[str, float]:
    """Load chemical similarity scores from h109."""
    chem_path = PROJECT_ROOT / "data" / "analysis" / "h109_chemical_similarity.json"
    if not chem_path.exists():
        return {}
    with open(chem_path) as f:
        return json.load(f)


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


def compute_mechanism_support(drug_id, disease_id, drug_targets, disease_genes) -> int:
    """Check if drug targets overlap with disease genes (h97)."""
    drug_genes = drug_targets.get(drug_id, set())
    dis_genes = disease_genes.get(disease_id, set())
    overlap = len(drug_genes & dis_genes)
    return 1 if overlap > 0 else 0


def run_knn_with_all_features(
    emb_dict, train_gt, test_gt, disease_names, drug_targets, disease_genes, k=20
) -> List[Dict]:
    """Run kNN and compute ALL validated confidence features."""
    train_disease_list = [d for d in train_gt if d in emb_dict]
    train_disease_embs = np.array([emb_dict[d] for d in train_disease_list], dtype=np.float32)

    # Drug training frequency
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

        sorted_drugs = sorted(drug_counts.items(), key=lambda x: x[1], reverse=True)[:30]
        max_score = sorted_drugs[0][1] if sorted_drugs else 1.0

        for rank, (drug_id, score) in enumerate(sorted_drugs, 1):
            mech_support = compute_mechanism_support(drug_id, disease_id, drug_targets, disease_genes)
            train_freq = drug_train_freq.get(drug_id, 0)
            norm_score = score / max_score if max_score > 0 else 0
            inv_rank = 1.0 / rank
            is_hit = drug_id in gt_drugs

            results.append({
                'disease': disease_id,
                'drug': drug_id,
                'disease_name': disease_name,
                'category': category,
                # Signals to analyze
                'mechanism_support': mech_support,
                'train_frequency': train_freq,
                'tier_inv': 3 - tier,  # Higher = better
                'norm_score': norm_score,
                'inv_rank': inv_rank,
                'rank': rank,
                'is_hit': 1 if is_hit else 0,
            })

    return results


def main():
    print("h111: Confidence Feature Independence Analysis")
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

    # Collect all predictions across seeds
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

        seed_results = run_knn_with_all_features(
            emb_dict, train_gt, test_gt, disease_names, drug_targets, disease_genes, k=20
        )
        all_results.extend(seed_results)
        print(f"  Seed {seed}: {len(seed_results)} predictions")

    df = pd.DataFrame(all_results)
    print(f"\nTotal predictions: {len(df)}")
    print(f"Base hit rate: {df['is_hit'].mean()*100:.2f}%")

    # Define signals to analyze
    signals = ['mechanism_support', 'train_frequency', 'tier_inv', 'norm_score', 'inv_rank']
    signal_labels = {
        'mechanism_support': 'Mechanism (h97)',
        'train_frequency': 'Drug Freq (h108)',
        'tier_inv': 'Category Tier (h71)',
        'norm_score': 'kNN Score',
        'inv_rank': 'Inverse Rank',
    }

    # === CORRELATION ANALYSIS ===
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)

    print("\nPearson Correlation Matrix:")
    corr_matrix = df[signals].corr(method='pearson')
    print(corr_matrix.round(3).to_string())

    print("\n\nSpearman Correlation Matrix:")
    corr_spearman = df[signals].corr(method='spearman')
    print(corr_spearman.round(3).to_string())

    # Identify independent pairs (correlation < 0.3)
    print("\n" + "=" * 70)
    print("INDEPENDENT SIGNAL PAIRS (|r| < 0.3)")
    print("=" * 70)

    independent_pairs = []
    for i, s1 in enumerate(signals):
        for s2 in signals[i+1:]:
            r = corr_matrix.loc[s1, s2]
            if abs(r) < 0.3:
                independent_pairs.append((s1, s2, r))
                print(f"  {signal_labels[s1]} vs {signal_labels[s2]}: r = {r:.3f}")

    correlated_pairs = []
    for i, s1 in enumerate(signals):
        for s2 in signals[i+1:]:
            r = corr_matrix.loc[s1, s2]
            if abs(r) >= 0.3:
                correlated_pairs.append((s1, s2, r))

    print("\n" + "=" * 70)
    print("CORRELATED SIGNAL PAIRS (|r| >= 0.3)")
    print("=" * 70)
    for s1, s2, r in correlated_pairs:
        print(f"  {signal_labels[s1]} vs {signal_labels[s2]}: r = {r:.3f}")

    # === PRECISION BY SIGNAL ===
    print("\n" + "=" * 70)
    print("PRECISION BY INDIVIDUAL SIGNAL")
    print("=" * 70)

    signal_precision = {}
    for signal in signals:
        if signal == 'mechanism_support':
            # Binary signal
            high = df[df[signal] == 1]
            low = df[df[signal] == 0]
        else:
            # Continuous signal - split into tertiles
            high = df[df[signal] >= df[signal].quantile(0.67)]
            low = df[df[signal] <= df[signal].quantile(0.33)]

        high_prec = high['is_hit'].mean() * 100
        low_prec = low['is_hit'].mean() * 100
        diff = high_prec - low_prec

        signal_precision[signal] = {
            'high_precision': high_prec,
            'low_precision': low_prec,
            'difference': diff,
            'high_count': len(high),
            'low_count': len(low),
        }

        print(f"\n{signal_labels[signal]}:")
        print(f"  HIGH: {high_prec:.2f}% ({len(high)} predictions)")
        print(f"  LOW:  {low_prec:.2f}% ({len(low)} predictions)")
        print(f"  Δ:    +{diff:.2f} pp")

    # === ADDITIVE GAIN ANALYSIS ===
    print("\n" + "=" * 70)
    print("ADDITIVE GAIN ANALYSIS: Independent Pairs")
    print("=" * 70)

    additive_gains = []
    for s1, s2, r in independent_pairs:
        # Get HIGH for both signals
        if s1 == 'mechanism_support':
            high_s1 = df[s1] == 1
        else:
            high_s1 = df[s1] >= df[s1].quantile(0.67)

        if s2 == 'mechanism_support':
            high_s2 = df[s2] == 1
        else:
            high_s2 = df[s2] >= df[s2].quantile(0.67)

        both_high = df[high_s1 & high_s2]
        either_high = df[high_s1 | high_s2]
        neither_high = df[~high_s1 & ~high_s2]

        both_prec = both_high['is_hit'].mean() * 100 if len(both_high) > 0 else 0
        either_prec = either_high['is_hit'].mean() * 100 if len(either_high) > 0 else 0
        neither_prec = neither_high['is_hit'].mean() * 100 if len(neither_high) > 0 else 0

        # Expected additive gain (if independent)
        s1_lift = signal_precision[s1]['difference']
        s2_lift = signal_precision[s2]['difference']
        expected_combined = min(signal_precision[s1]['low_precision'] + s1_lift + s2_lift, 100)

        # Observed gain
        observed_combined = both_prec

        print(f"\n{signal_labels[s1]} + {signal_labels[s2]} (r = {r:.3f}):")
        print(f"  BOTH HIGH:    {both_prec:.2f}% ({len(both_high)} predictions)")
        print(f"  EITHER HIGH:  {either_prec:.2f}% ({len(either_high)} predictions)")
        print(f"  NEITHER HIGH: {neither_prec:.2f}% ({len(neither_high)} predictions)")
        print(f"  Expected if additive: ~{expected_combined:.1f}%")
        print(f"  Observed: {observed_combined:.2f}%")

        is_additive = observed_combined >= (signal_precision[s1]['high_precision'] + signal_precision[s2]['high_precision']) / 2
        additive_gains.append({
            'pair': f"{s1} + {s2}",
            'correlation': r,
            'both_high_precision': both_prec,
            'both_high_count': len(both_high),
            'is_additive': is_additive,
        })

        if is_additive:
            print(f"  → Combining provides additive gain")
        else:
            print(f"  → Combining is NOT additive (signals share information)")

    # === CORRELATION WITH is_hit ===
    print("\n" + "=" * 70)
    print("SIGNAL CORRELATION WITH HITS")
    print("=" * 70)

    hit_correlations = {}
    for signal in signals:
        r, p = stats.pointbiserialr(df[signal], df['is_hit'])
        hit_correlations[signal] = {'r': r, 'p': p}
        sig = '*' if p < 0.01 else ''
        print(f"  {signal_labels[signal]}: r = {r:.4f}{sig}")

    # === SUMMARY ===
    print("\n" + "=" * 70)
    print("SUMMARY: Signal Independence Analysis")
    print("=" * 70)

    # Sort by hit correlation
    sorted_signals = sorted(hit_correlations.items(), key=lambda x: abs(x[1]['r']), reverse=True)
    print("\nSignals ranked by hit correlation:")
    for signal, vals in sorted_signals:
        print(f"  {signal_labels[signal]}: r = {vals['r']:.4f}")

    print("\nKey findings:")
    n_independent = len(independent_pairs)
    n_correlated = len(correlated_pairs)
    print(f"  - {n_independent} independent pairs (|r| < 0.3)")
    print(f"  - {n_correlated} correlated pairs (|r| >= 0.3)")

    # Best independent combination
    if additive_gains:
        best_combo = max(additive_gains, key=lambda x: x['both_high_precision'])
        print(f"\n  Best combination: {best_combo['pair']}")
        print(f"    Precision: {best_combo['both_high_precision']:.2f}%")
        print(f"    Count: {best_combo['both_high_count']} predictions")

    # Save results
    results = {
        'pearson_correlations': corr_matrix.to_dict(),
        'spearman_correlations': corr_spearman.to_dict(),
        'signal_precision': signal_precision,
        'hit_correlations': {k: {'r': v['r'], 'p': v['p']} for k, v in hit_correlations.items()},
        'independent_pairs': [{'s1': s1, 's2': s2, 'r': float(r)} for s1, s2, r in independent_pairs],
        'correlated_pairs': [{'s1': s1, 's2': s2, 'r': float(r)} for s1, s2, r in correlated_pairs],
        'additive_gains': additive_gains,
        'n_predictions': len(df),
    }

    results_file = PROJECT_ROOT / "data" / "analysis" / "h111_independence_analysis.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
