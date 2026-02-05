#!/usr/bin/env python3
"""
h268: NSAIDs vs DMARDs for Autoimmune Subtypes

Analyze precision differences between NSAIDs and DMARDs for different
autoimmune disease subtypes (RA, lupus, psoriasis, MS, IBD, etc.)
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

# Autoimmune subtypes
AUTOIMMUNE_SUBTYPES = {
    'inflammatory_arthritis': ['rheumatoid arthritis', 'juvenile arthritis', 'ankylosing spondylitis',
                                'psoriatic arthritis', 'reactive arthritis', 'gout arthritis'],
    'connective_tissue': ['lupus', 'sle', 'systemic lupus', 'scleroderma', 'dermatomyositis',
                          'polymyositis', 'sjögren', 'vasculitis', 'mixed connective'],
    'skin': ['psoriasis', 'atopic dermatitis', 'vitiligo', 'alopecia areata', 'pemphigus',
             'pemphigoid', 'lichen planus'],
    'neuro_autoimmune': ['multiple sclerosis', 'myasthenia gravis', 'guillain-barre',
                         'cidp', 'transverse myelitis', 'neuromyelitis optica'],
    'ibd': ['crohn', 'ulcerative colitis', 'inflammatory bowel'],
    'other_autoimmune': ['autoimmune hepatitis', 'primary biliary', 'celiac',
                          'type 1 diabetes', 'hashimoto', 'graves', 'addison'],
}

# Drug class patterns
NSAID_DRUGS = ['ibuprofen', 'naproxen', 'diclofenac', 'indomethacin', 'celecoxib',
               'meloxicam', 'piroxicam', 'ketorolac', 'aspirin', 'sulindac', 'ketoprofen']

DMARD_DRUGS = ['methotrexate', 'sulfasalazine', 'hydroxychloroquine', 'leflunomide',
               'azathioprine', 'mycophenolate', 'cyclosporine', 'tacrolimus',
               'cyclophosphamide', 'tofacitinib', 'baricitinib']

CORTICOSTEROID_DRUGS = ['prednisone', 'prednisolone', 'methylprednisolone', 'dexamethasone',
                         'hydrocortisone', 'betamethasone', 'triamcinolone', 'budesonide']


def get_autoimmune_subtype(disease_name: str) -> str:
    """Get the autoimmune subtype for a disease."""
    disease_lower = disease_name.lower()
    for subtype, keywords in AUTOIMMUNE_SUBTYPES.items():
        for kw in keywords:
            if kw in disease_lower:
                return subtype
    return 'other_autoimmune'


def get_drug_class(drug_name: str) -> str:
    """Get the drug class for a drug name."""
    drug_lower = drug_name.lower()
    for nsaid in NSAID_DRUGS:
        if nsaid in drug_lower:
            return 'nsaid'
    for dmard in DMARD_DRUGS:
        if dmard in drug_lower:
            return 'dmard'
    for steroid in CORTICOSTEROID_DRUGS:
        if steroid in drug_lower:
            return 'corticosteroid'
    return 'other'


def load_node2vec_embeddings() -> Dict[str, np.ndarray]:
    """Load Node2Vec embeddings from CSV."""
    embeddings_path = EMBEDDINGS_DIR / "node2vec_256_named.csv"
    df = pd.read_csv(embeddings_path)
    dim_cols = [c for c in df.columns if c.startswith("dim_")]
    embeddings: Dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        entity = f"drkg:{row['entity']}"
        embeddings[entity] = row[dim_cols].values.astype(np.float32)
    return embeddings


def load_drugbank_lookup():
    """Load DrugBank ID to name mapping."""
    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)
    name_to_id = {name.lower(): f"drkg:Compound::{db_id}" for db_id, name in id_to_name.items()}
    id_to_name_full = {f"drkg:Compound::{db_id}": name for db_id, name in id_to_name.items()}
    return name_to_id, id_to_name_full


def load_mesh_mappings_from_file() -> Dict[str, str]:
    """Load MESH mappings."""
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
    """Load ground truth from Every Cure data."""
    df = pd.read_excel(REFERENCE_DIR / "everycure" / "indicationList.xlsx")
    fuzzy_mappings = load_mesh_mappings()
    matcher = DiseaseMatcher(fuzzy_mappings)

    gt: Dict[str, Set[str]] = defaultdict(set)
    disease_names: Dict[str, str] = {}

    for _, row in df.iterrows():
        disease_name = str(row['disease name']).lower().strip()
        drug_name = str(row['final normalized drug label']).lower().strip()

        disease_id = mesh_mappings.get(disease_name)
        if not disease_id:
            disease_id = matcher.get_mesh_id(disease_name)
        if not disease_id:
            continue

        drug_id = name_to_drug_id.get(drug_name)
        if not drug_id:
            continue

        gt[disease_id].add(drug_id)
        disease_names[disease_id] = disease_name

    return gt, disease_names


def analyze_autoimmune_subtypes():
    """Analyze NSAID vs DMARD precision by autoimmune subtype."""
    print("=" * 70)
    print("h268: NSAIDs vs DMARDs for Autoimmune Subtypes")
    print("=" * 70)
    print("\nLoading data...")

    # Load embeddings
    emb_dict = load_node2vec_embeddings()
    name_to_drug_id, id_to_name = load_drugbank_lookup()
    mesh_mappings = load_mesh_mappings_from_file()
    gt, disease_names = load_ground_truth(mesh_mappings, name_to_drug_id)

    print(f"Loaded {len(emb_dict)} embeddings, {len(gt)} diseases with GT")

    # Filter to autoimmune diseases
    autoimmune_diseases = {}
    for disease_id, drugs in gt.items():
        disease_name = disease_names.get(disease_id, '')
        # Check if it's autoimmune
        for subtype, keywords in AUTOIMMUNE_SUBTYPES.items():
            for kw in keywords:
                if kw in disease_name.lower():
                    autoimmune_diseases[disease_id] = {
                        'name': disease_name,
                        'subtype': subtype,
                        'drugs': drugs,
                    }
                    break

    print(f"Found {len(autoimmune_diseases)} autoimmune diseases with GT")

    # Count diseases by subtype
    subtype_counts = defaultdict(int)
    for data in autoimmune_diseases.values():
        subtype_counts[data['subtype']] += 1
    print("\nDiseases by subtype:")
    for subtype, count in sorted(subtype_counts.items(), key=lambda x: -x[1]):
        print(f"  {subtype:25}: {count}")

    # Analyze drug class precision by subtype
    # We need to run kNN predictions for each autoimmune disease and track
    # which drug classes hit GT
    print("\nAnalyzing drug class precision by subtype...")

    # Track precision by (subtype, drug_class)
    results = defaultdict(lambda: {'predictions': 0, 'hits': 0})

    # Get diseases with GT that have embeddings
    diseases_with_gt = [d for d in autoimmune_diseases.keys() if d in emb_dict]
    print(f"Autoimmune diseases with embeddings: {len(diseases_with_gt)}")

    if len(diseases_with_gt) < 10:
        print("WARNING: Very few autoimmune diseases with embeddings")
        return

    # For efficiency, use a single seed analysis
    np.random.seed(42)

    # Build training set (80%) and test set (20%)
    np.random.shuffle(diseases_with_gt)
    n_test = len(diseases_with_gt) // 5
    test_diseases = set(diseases_with_gt[:n_test])
    train_diseases = set(diseases_with_gt[n_test:])

    # Build training GT lookup
    train_gt = {}
    for d in train_diseases:
        train_gt[d] = autoimmune_diseases[d]['drugs']

    # Get embeddings
    test_disease_list = list(test_diseases)
    train_disease_list = list(train_diseases)

    test_emb = np.array([emb_dict[d] for d in test_disease_list])
    train_emb = np.array([emb_dict[d] for d in train_disease_list])

    # Compute disease similarities
    k = 20
    disease_sim = cosine_similarity(test_emb, train_emb)

    for i, test_disease in enumerate(test_disease_list):
        disease_data = autoimmune_diseases[test_disease]
        subtype = disease_data['subtype']
        test_gt = disease_data['drugs']

        # Get k nearest training diseases
        sims = disease_sim[i]
        top_k_idx = np.argsort(sims)[::-1][:k]

        # Collect drug scores from neighbors
        drug_scores = defaultdict(float)
        for j in top_k_idx:
            neighbor = train_disease_list[j]
            sim = sims[j]
            for drug in train_gt.get(neighbor, set()):
                drug_scores[drug] += sim

        if not drug_scores:
            continue

        # Rank drugs and get top 30
        ranked_drugs = sorted(drug_scores.items(), key=lambda x: -x[1])[:30]

        # Track predictions and hits by drug class
        for drug_id, _ in ranked_drugs:
            drug_name = id_to_name.get(drug_id, drug_id)
            drug_class = get_drug_class(drug_name)
            if drug_class in ['nsaid', 'dmard', 'corticosteroid']:
                key = (subtype, drug_class)
                results[key]['predictions'] += 1
                if drug_id in test_gt:
                    results[key]['hits'] += 1

    # Print results
    print("\n" + "=" * 70)
    print("PRECISION BY SUBTYPE AND DRUG CLASS")
    print("=" * 70)

    # Organize by subtype
    subtypes = set(k[0] for k in results.keys())
    drug_classes = ['nsaid', 'dmard', 'corticosteroid']

    print(f"\n{'Subtype':25} {'NSAID':>15} {'DMARD':>15} {'Steroid':>15}")
    print("-" * 75)

    for subtype in sorted(subtypes):
        row = [f"{subtype:25}"]
        for dc in drug_classes:
            key = (subtype, dc)
            if key in results and results[key]['predictions'] >= 3:
                prec = results[key]['hits'] / results[key]['predictions'] * 100
                n = results[key]['predictions']
                row.append(f"{prec:5.1f}% (n={n:2d})")
            else:
                row.append(f"{'--':>15}")
        print(" ".join(row))

    # Summary
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Find best drug class per subtype
    print("\nBest drug class per subtype (>30% precision, n>=3):")
    for subtype in sorted(subtypes):
        best = None
        best_prec = 0
        for dc in drug_classes:
            key = (subtype, dc)
            if key in results and results[key]['predictions'] >= 3:
                prec = results[key]['hits'] / results[key]['predictions'] * 100
                if prec > best_prec:
                    best_prec = prec
                    best = dc
        if best and best_prec >= 30:
            print(f"  {subtype:25} → {best:15} ({best_prec:.1f}%)")

    # Save results
    output = {
        'hypothesis': 'h268',
        'results': {str(k): v for k, v in results.items()},
        'subtype_counts': dict(subtype_counts),
    }

    output_path = PROJECT_ROOT / 'data' / 'analysis' / 'h268_autoimmune_subtype.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    analyze_autoimmune_subtypes()
