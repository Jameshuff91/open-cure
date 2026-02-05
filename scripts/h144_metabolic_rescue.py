#!/usr/bin/env python3
"""
h144: Metabolic Disease Rescue Analysis

Metabolic diseases (diabetes, obesity, etc.) show 0 GOLDEN and 0 HIGH predictions
in the production export. This script investigates alternative confidence signals:

1. ATC class A10 (diabetes drugs) correlation with precision
2. Specific drug mechanisms (PPAR agonists, GLP-1, SGLT2, etc.)
3. Drug frequency patterns specific to metabolic drugs
4. Alternative rescue criteria

SUCCESS CRITERIA: Find criteria achieving >30% precision for metabolic diseases
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

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

# Metabolic disease keywords
METABOLIC_KEYWORDS = ['diabetes', 'metabolic', 'obesity', 'thyroid', 'hyperlipidemia',
                      'hypercholesterolemia', 'gout', 'porphyria', 'lipodystrophy',
                      'insulin resistance', 'hyperglycemia', 'type 2', 'type 1']

# Known metabolic drug classes and their patterns
METABOLIC_DRUG_CLASSES = {
    'biguanide': ['metformin'],
    'sulfonylurea': ['glimepiride', 'glyburide', 'glipizide', 'gliclazide'],
    'thiazolidinedione': ['pioglitazone', 'rosiglitazone'],  # PPAR agonists
    'dpp4_inhibitor': ['sitagliptin', 'saxagliptin', 'linagliptin', 'alogliptin'],
    'sglt2_inhibitor': ['empagliflozin', 'dapagliflozin', 'canagliflozin', 'ertugliflozin'],
    'glp1_agonist': ['liraglutide', 'semaglutide', 'dulaglutide', 'exenatide'],
    'insulin': ['insulin'],
    'statin': ['atorvastatin', 'simvastatin', 'rosuvastatin', 'pravastatin', 'lovastatin'],
    'fibrate': ['fenofibrate', 'gemfibrozil', 'bezafibrate'],
}

# ATC code mapping (A10 = diabetes)
ATC_A10_PATTERNS = ['A10', 'antidiabetic', 'insulin']


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


def is_metabolic_disease(disease_name: str) -> bool:
    name_lower = disease_name.lower()
    return any(kw in name_lower for kw in METABOLIC_KEYWORDS)


def classify_metabolic_drug(drug_name: str) -> str:
    """Classify drug into metabolic drug class."""
    drug_lower = drug_name.lower()
    for drug_class, patterns in METABOLIC_DRUG_CLASSES.items():
        for pattern in patterns:
            if pattern in drug_lower:
                return drug_class
    return 'other'


def compute_mechanism_support(drug_id, disease_id, drug_targets, disease_genes) -> int:
    drug_genes = drug_targets.get(drug_id, set())
    dis_genes = disease_genes.get(disease_id, set())
    return 1 if len(drug_genes & dis_genes) > 0 else 0


def run_metabolic_analysis(emb_dict, train_gt, test_gt, disease_names, drug_targets,
                           disease_genes, id_to_name, k=20):
    """Run kNN and collect metabolic disease predictions."""
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

        disease_name = disease_names.get(disease_id, "")
        if not is_metabolic_disease(disease_name):
            continue

        gt_drugs = {d for d in test_gt[disease_id] if d in emb_dict}
        if not gt_drugs:
            continue

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
            drug_class = classify_metabolic_drug(drug_name)
            has_targets = drug_id in drug_targets and len(drug_targets.get(drug_id, set())) > 0

            results.append({
                'drug_id': drug_id,
                'drug_name': drug_name,
                'drug_class': drug_class,
                'disease_id': disease_id,
                'disease_name': disease_name,
                'mechanism_support': mech_support,
                'train_frequency': train_freq,
                'norm_score': norm_score,
                'rank': rank,
                'is_hit': 1 if is_hit else 0,
                'has_targets': 1 if has_targets else 0,
            })

    return results


def main():
    print("h144: Metabolic Disease Rescue Analysis")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    emb_dict = load_node2vec_embeddings()
    name_to_drug_id, id_to_name = load_drugbank_lookup()
    mesh_mappings = load_mesh_mappings_from_file()
    ground_truth, disease_names = load_ground_truth(mesh_mappings, name_to_drug_id)
    drug_targets = load_drug_targets()
    disease_genes = load_disease_genes()

    # Count metabolic diseases
    metabolic_diseases = [d for d, name in disease_names.items() if is_metabolic_disease(name)]
    print(f"  Total diseases: {len(disease_names)}")
    print(f"  Metabolic diseases: {len(metabolic_diseases)}")

    # Collect predictions across seeds
    print("\n" + "=" * 70)
    print("Collecting metabolic predictions across 5 seeds")
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

        seed_results = run_metabolic_analysis(
            emb_dict, train_gt, test_gt, disease_names, drug_targets, disease_genes, id_to_name, k=20
        )
        all_results.extend(seed_results)
        print(f"  Seed {seed}: {len(seed_results)} metabolic predictions")

    df = pd.DataFrame(all_results)
    print(f"\nTotal metabolic predictions: {len(df)}")
    print(f"Base hit rate: {df['is_hit'].mean()*100:.2f}%")

    # Analyze by drug class
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Precision by Drug Class")
    print("=" * 70)

    print(f"\n{'Drug Class':<20} {'N':>8} {'Hits':>8} {'Precision':>12} {'Avg Freq':>10}")
    print("-" * 60)

    for drug_class in sorted(df['drug_class'].unique()):
        class_df = df[df['drug_class'] == drug_class]
        if len(class_df) > 0:
            precision = class_df['is_hit'].mean() * 100
            n_hits = class_df['is_hit'].sum()
            avg_freq = class_df['train_frequency'].mean()
            print(f"{drug_class:<20} {len(class_df):>8} {n_hits:>8} {precision:>11.1f}% {avg_freq:>10.1f}")

    # Analyze with mechanism support
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Drug Class + Mechanism Support")
    print("=" * 70)

    print(f"\n{'Criteria':<35} {'N':>8} {'Precision':>12}")
    print("-" * 58)

    for drug_class in ['biguanide', 'sulfonylurea', 'thiazolidinedione', 'dpp4_inhibitor',
                        'sglt2_inhibitor', 'glp1_agonist', 'insulin', 'statin']:
        class_df = df[df['drug_class'] == drug_class]
        if len(class_df) > 0:
            precision = class_df['is_hit'].mean() * 100
            print(f"{drug_class:<35} {len(class_df):>8} {precision:>11.1f}%")

        mech_df = class_df[class_df['mechanism_support'] == 1]
        if len(mech_df) > 5:
            precision_mech = mech_df['is_hit'].mean() * 100
            print(f"  + mechanism support                {len(mech_df):>8} {precision_mech:>11.1f}%")

    # Test rescue criteria
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Potential Rescue Criteria for Metabolic")
    print("=" * 70)

    rescue_criteria = [
        ("Base (all metabolic)", lambda x: True),
        ("rank <= 10", lambda x: x['rank'] <= 10),
        ("rank <= 5", lambda x: x['rank'] <= 5),
        ("freq >= 10", lambda x: x['train_frequency'] >= 10),
        ("freq >= 15", lambda x: x['train_frequency'] >= 15),
        ("mechanism", lambda x: x['mechanism_support'] == 1),
        ("rank<=10 + mech", lambda x: (x['rank'] <= 10) & (x['mechanism_support'] == 1)),
        ("rank<=5 + mech", lambda x: (x['rank'] <= 5) & (x['mechanism_support'] == 1)),
        ("freq>=10 + mech", lambda x: (x['train_frequency'] >= 10) & (x['mechanism_support'] == 1)),
        ("freq>=15 + mech", lambda x: (x['train_frequency'] >= 15) & (x['mechanism_support'] == 1)),
        ("rank<=10 + freq>=10 + mech", lambda x: (x['rank'] <= 10) & (x['train_frequency'] >= 10) & (x['mechanism_support'] == 1)),
        ("rank<=5 + freq>=10 + mech", lambda x: (x['rank'] <= 5) & (x['train_frequency'] >= 10) & (x['mechanism_support'] == 1)),
        # Drug class specific
        ("insulin class", lambda x: x['drug_class'] == 'insulin'),
        ("biguanide class", lambda x: x['drug_class'] == 'biguanide'),
        ("sulfonylurea class", lambda x: x['drug_class'] == 'sulfonylurea'),
        ("thiazolidinedione class", lambda x: x['drug_class'] == 'thiazolidinedione'),
        ("statin class", lambda x: x['drug_class'] == 'statin'),
        # Combined class + criteria
        ("insulin + rank<=10", lambda x: (x['drug_class'] == 'insulin') & (x['rank'] <= 10)),
        ("statin + rank<=10", lambda x: (x['drug_class'] == 'statin') & (x['rank'] <= 10)),
        ("statin + mech", lambda x: (x['drug_class'] == 'statin') & (x['mechanism_support'] == 1)),
    ]

    print(f"\n{'Criteria':<40} {'N':>8} {'Hits':>8} {'Precision':>12}")
    print("-" * 70)

    best_criteria = None
    best_precision = 0

    for name, criteria_fn in rescue_criteria:
        mask = df.apply(criteria_fn, axis=1)
        subset = df[mask]
        if len(subset) >= 10:  # Minimum sample size
            precision = subset['is_hit'].mean() * 100
            n_hits = subset['is_hit'].sum()
            print(f"{name:<40} {len(subset):>8} {n_hits:>8} {precision:>11.1f}%")

            if precision > best_precision and precision >= 30:
                best_precision = precision
                best_criteria = name

    # Check if any criteria achieves >30%
    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 70)

    if best_criteria and best_precision >= 30:
        print(f"\n  VALIDATED: Found rescue criteria achieving >30% precision")
        print(f"  Best: '{best_criteria}' at {best_precision:.1f}%")
    else:
        print(f"\n  INCONCLUSIVE: No criteria achieves >30% precision for metabolic diseases")
        print(f"  Best found: {best_precision:.1f}%" if best_precision > 0 else "  All criteria below threshold")

        # Additional analysis - what makes metabolic hard?
        print("\n" + "=" * 70)
        print("ROOT CAUSE ANALYSIS: Why are metabolic diseases hard?")
        print("=" * 70)

        # Compare to other categories
        print("\n1. Compare metabolic vs overall base rate:")
        print(f"   Metabolic base hit rate: {df['is_hit'].mean()*100:.1f}%")

        # Check drug frequency distribution
        print("\n2. Drug frequency distribution for metabolic predictions:")
        freq_bins = [(0, 5), (5, 10), (10, 15), (15, 20), (20, float('inf'))]
        for low, high in freq_bins:
            mask = (df['train_frequency'] >= low) & (df['train_frequency'] < high)
            subset = df[mask]
            if len(subset) > 0:
                precision = subset['is_hit'].mean() * 100
                print(f"   Freq {low}-{int(high) if high < 100 else '+'}: {len(subset)} predictions, {precision:.1f}% precision")

        # Check mechanism support distribution
        print("\n3. Mechanism support distribution:")
        mech_1 = df[df['mechanism_support'] == 1]
        mech_0 = df[df['mechanism_support'] == 0]
        print(f"   With mechanism: {len(mech_1)} ({len(mech_1)/len(df)*100:.1f}%), {mech_1['is_hit'].mean()*100:.1f}% precision")
        print(f"   No mechanism: {len(mech_0)} ({len(mech_0)/len(df)*100:.1f}%), {mech_0['is_hit'].mean()*100:.1f}% precision")

        # Check top-k distribution
        print("\n4. Top-rank distribution:")
        for k in [5, 10, 15, 20]:
            subset = df[df['rank'] <= k]
            if len(subset) > 0:
                precision = subset['is_hit'].mean() * 100
                print(f"   Rank <= {k}: {len(subset)} predictions, {precision:.1f}% precision")

    # Save results
    output = {
        'total_metabolic_predictions': len(df),
        'base_hit_rate': float(df['is_hit'].mean() * 100),
        'best_criteria': best_criteria,
        'best_precision': best_precision,
        'success': best_precision >= 30,
    }

    results_file = ANALYSIS_DIR / "h144_metabolic_rescue.json"
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
