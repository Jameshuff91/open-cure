#!/usr/bin/env python3
"""
h160: Cancer Targeted Therapy Specificity

PURPOSE:
    Cancer drug predictions have low precision (~5-20%). Cancer is heterogeneous:
    - Imatinib (CML) vs Trastuzumab (breast cancer) have very different targets
    - May need cancer subtype + drug subtype matching

APPROACH:
    1. Stratify cancer by type: leukemia, lymphoma, solid tumor
    2. Stratify drugs: kinase inhibitors, mAbs, immunotherapy, chemotherapy
    3. Test precision for subtype combinations

SUCCESS CRITERIA:
    Find subtype combination achieving >30% precision.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from disease_name_matcher import DiseaseMatcher, load_mesh_mappings
from atc_features import ATCMapper

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"

SEEDS = [42, 123, 456, 789, 1024]

# Cancer type patterns
CANCER_TYPES = {
    'leukemia': ['leukemia', 'leukaemia', 'cml', 'cll', 'aml', 'all'],
    'lymphoma': ['lymphoma', 'hodgkin', 'non-hodgkin', 'nhl'],
    'myeloma': ['myeloma', 'plasmacytoma'],
    'breast_cancer': ['breast cancer', 'breast carcinoma', 'mammary'],
    'lung_cancer': ['lung cancer', 'lung carcinoma', 'nsclc', 'sclc', 'bronchogenic'],
    'colorectal_cancer': ['colorectal', 'colon cancer', 'rectal cancer'],
    'melanoma': ['melanoma'],
    'prostate_cancer': ['prostate cancer', 'prostate carcinoma'],
    'brain_tumor': ['glioma', 'glioblastoma', 'brain tumor', 'astrocytoma'],
    'renal_cancer': ['renal cell', 'kidney cancer'],
    'pancreatic_cancer': ['pancreatic cancer', 'pancreatic carcinoma'],
    'ovarian_cancer': ['ovarian cancer', 'ovarian carcinoma'],
    'hepatocellular': ['hepatocellular', 'liver cancer'],
}

# Drug class patterns based on common drug names and mechanisms
CANCER_DRUG_CLASSES = {
    'kinase_inhibitor': [
        'imatinib', 'erlotinib', 'gefitinib', 'sorafenib', 'sunitinib',
        'dasatinib', 'nilotinib', 'pazopanib', 'vemurafenib', 'crizotinib',
        'lapatinib', 'axitinib', 'regorafenib', 'ibrutinib', 'palbociclib',
        'osimertinib', 'ruxolitinib', 'lenvatinib', 'alectinib', 'brigatinib',
        # -nib suffix
    ],
    'monoclonal_antibody': [
        'rituximab', 'trastuzumab', 'bevacizumab', 'cetuximab', 'panitumumab',
        'alemtuzumab', 'obinutuzumab', 'ofatumumab', 'daratumumab', 'elotuzumab',
        'nivolumab', 'pembrolizumab', 'ipilimumab', 'atezolizumab', 'durvalumab',
        # -mab suffix
    ],
    'immunotherapy': [
        'nivolumab', 'pembrolizumab', 'ipilimumab', 'atezolizumab', 'durvalumab',
        'avelumab', 'cemiplimab', 'interferon', 'interleukin',
        # checkpoint inhibitors, cytokines
    ],
    'chemotherapy': [
        'cisplatin', 'carboplatin', 'oxaliplatin', 'doxorubicin', 'epirubicin',
        'cyclophosphamide', 'ifosfamide', 'methotrexate', 'fluorouracil',
        '5-fu', 'capecitabine', 'gemcitabine', 'paclitaxel', 'docetaxel',
        'vincristine', 'vinblastine', 'etoposide', 'irinotecan', 'topotecan',
        'bleomycin', 'mitomycin', 'temozolomide', 'dacarbazine', 'busulfan',
    ],
    'hormone_therapy': [
        'tamoxifen', 'letrozole', 'anastrozole', 'exemestane', 'fulvestrant',
        'enzalutamide', 'abiraterone', 'leuprolide', 'goserelin', 'bicalutamide',
    ],
    'targeted_other': [
        'bortezomib', 'carfilzomib', 'ixazomib',  # proteasome inhibitors
        'lenalidomide', 'pomalidomide', 'thalidomide',  # IMiDs
        'venetoclax', 'olaparib', 'rucaparib', 'niraparib',  # PARP inhibitors
    ],
}


def classify_cancer_type(disease_name: str) -> str:
    """Classify cancer into subtypes."""
    name_lower = disease_name.lower()

    for cancer_type, patterns in CANCER_TYPES.items():
        for pattern in patterns:
            if pattern in name_lower:
                return cancer_type

    # Generic categories
    if 'carcinoma' in name_lower:
        return 'other_carcinoma'
    if 'sarcoma' in name_lower:
        return 'sarcoma'
    if 'tumor' in name_lower or 'tumour' in name_lower:
        return 'other_tumor'
    if 'neoplasm' in name_lower or 'malignant' in name_lower:
        return 'other_neoplasm'

    return 'other_cancer'


def classify_drug_class(drug_name: str, atc_mapper: ATCMapper = None) -> str:
    """Classify cancer drug into therapeutic class."""
    name_lower = drug_name.lower()

    # Check patterns first
    for drug_class, patterns in CANCER_DRUG_CLASSES.items():
        for pattern in patterns:
            if pattern in name_lower:
                return drug_class

    # Use suffix patterns
    if name_lower.endswith('nib'):
        return 'kinase_inhibitor'
    if name_lower.endswith('mab'):
        return 'monoclonal_antibody'

    # Use ATC codes if available
    if atc_mapper:
        atc_l1 = atc_mapper.get_atc_level1(drug_name)
        atc_l4 = atc_mapper.get_atc_level4(drug_name)

        # L01X - Other antineoplastic agents
        for code in atc_l4:
            if code.startswith('L01XC'):  # Monoclonal antibodies
                return 'monoclonal_antibody'
            if code.startswith('L01XE'):  # Protein kinase inhibitors
                return 'kinase_inhibitor'
            if code.startswith('L01XX'):  # Other
                return 'targeted_other'

        # L01A-L01D - Cytotoxic agents
        for code in atc_l4:
            if code.startswith(('L01A', 'L01B', 'L01C', 'L01D')):
                return 'chemotherapy'

        # L02 - Endocrine therapy
        for code in atc_l1:
            if code == 'L' and any(c.startswith('L02') for c in atc_l4):
                return 'hormone_therapy'

    return 'unknown_class'


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


def load_drugbank_lookup() -> Tuple[Dict[str, str], Dict[str, str]]:
    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)
    name_to_id = {name.lower(): f"drkg:Compound::{db_id}" for db_id, name in id_to_name.items()}
    id_to_name_full = {f"drkg:Compound::{db_id}": name for db_id, name in id_to_name.items()}
    return name_to_id, id_to_name_full


def load_ground_truth(name_to_drug_id: Dict[str, str]) -> Tuple[Dict[str, Set[str]], Dict[str, str]]:
    """Load ground truth for cancer diseases only."""
    mesh_mappings = load_mesh_mappings()
    matcher = DiseaseMatcher(mesh_mappings)
    df = pd.read_excel(REFERENCE_DIR / "everycure" / "indicationList.xlsx")

    gt: Dict[str, Set[str]] = defaultdict(set)
    disease_names: Dict[str, str] = {}

    for _, row in df.iterrows():
        disease = str(row.get("disease name", "")).strip()
        drug = str(row.get("final normalized drug label", "")).strip()
        if not disease or not drug:
            continue

        disease_id = matcher.get_mesh_id(disease)
        if not disease_id:
            continue

        drug_id = name_to_drug_id.get(drug.lower())
        if drug_id:
            gt[disease_id].add(drug_id)
            disease_names[disease_id] = disease

    return dict(gt), disease_names


def is_cancer_disease(disease_name: str) -> bool:
    """Check if disease is a cancer."""
    name_lower = disease_name.lower()
    cancer_keywords = [
        'cancer', 'carcinoma', 'lymphoma', 'leukemia', 'leukaemia', 'tumor',
        'tumour', 'sarcoma', 'myeloma', 'melanoma', 'neoplasm', 'malignant',
        'glioma', 'glioblastoma', 'neuroblastoma', 'blastoma', 'oncologic'
    ]
    return any(kw in name_lower for kw in cancer_keywords)


def run_knn_for_cancer(
    emb_dict: Dict[str, np.ndarray],
    train_gt: Dict[str, Set[str]],
    test_gt: Dict[str, Set[str]],
    disease_names: Dict[str, str],
    id_to_name: Dict[str, str],
    atc_mapper: ATCMapper,
    k: int = 20,
) -> List[Dict]:
    """Run kNN for cancer diseases and classify predictions."""
    train_disease_list = [d for d in train_gt if d in emb_dict]
    train_disease_embs = np.array([emb_dict[d] for d in train_disease_list], dtype=np.float32)

    results = []

    for disease_id in test_gt:
        if disease_id not in emb_dict:
            continue

        disease_name = disease_names.get(disease_id, "")
        if not is_cancer_disease(disease_name):
            continue

        cancer_type = classify_cancer_type(disease_name)
        gt_drugs = {d for d in test_gt[disease_id] if d in emb_dict}
        if not gt_drugs:
            continue

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

        for rank, (drug_id, score) in enumerate(sorted_drugs, 1):
            drug_name = id_to_name.get(drug_id, "")
            drug_class = classify_drug_class(drug_name, atc_mapper)
            is_hit = drug_id in gt_drugs

            results.append({
                'disease': disease_id,
                'disease_name': disease_name,
                'cancer_type': cancer_type,
                'drug': drug_id,
                'drug_name': drug_name,
                'drug_class': drug_class,
                'rank': rank,
                'score': float(score),
                'is_hit': is_hit,
            })

    return results


def main():
    print("h160: Cancer Targeted Therapy Specificity")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    emb_dict = load_node2vec_embeddings()
    name_to_drug_id, id_to_name = load_drugbank_lookup()
    ground_truth, disease_names = load_ground_truth(name_to_drug_id)
    atc_mapper = ATCMapper()

    # Filter to cancer diseases
    cancer_diseases = {d: gt for d, gt in ground_truth.items()
                       if is_cancer_disease(disease_names.get(d, ""))}
    print(f"  Total diseases: {len(ground_truth)}")
    print(f"  Cancer diseases: {len(cancer_diseases)}")

    # Multi-seed evaluation
    print("\n" + "=" * 70)
    print("Multi-Seed Evaluation (5 seeds)")
    print("=" * 70)

    all_results = []

    for seed in SEEDS:
        np.random.seed(seed)
        diseases = list(cancer_diseases.keys())
        np.random.shuffle(diseases)
        n_test = len(diseases) // 5
        test_diseases = set(diseases[:n_test])
        train_diseases = set(diseases[n_test:])

        # Use all diseases for training (not just cancer)
        train_gt = {d: ground_truth[d] for d in ground_truth if d not in test_diseases}
        test_gt = {d: cancer_diseases[d] for d in test_diseases}

        seed_results = run_knn_for_cancer(
            emb_dict, train_gt, test_gt, disease_names, id_to_name, atc_mapper, k=20
        )
        all_results.extend(seed_results)
        print(f"  Seed {seed}: {len(seed_results)} cancer predictions")

    df = pd.DataFrame(all_results)

    # Overall cancer precision
    print("\n" + "=" * 70)
    print("Overall Cancer Precision")
    print("=" * 70)

    overall_prec = df['is_hit'].mean()
    print(f"\nAll cancer predictions: {100*overall_prec:.2f}% (n={len(df)})")

    # By cancer type
    print("\n" + "=" * 70)
    print("Precision by Cancer Type")
    print("=" * 70)

    print(f"\n{'Cancer Type':<20} {'N':<8} {'Precision':<12} {'Hits':<8}")
    print("-" * 50)

    cancer_type_stats = {}
    for cancer_type in sorted(df['cancer_type'].unique()):
        subset = df[df['cancer_type'] == cancer_type]
        prec = subset['is_hit'].mean()
        hits = subset['is_hit'].sum()
        cancer_type_stats[cancer_type] = {'n': len(subset), 'precision': prec, 'hits': hits}
        print(f"{cancer_type:<20} {len(subset):<8} {100*prec:>8.1f}%    {hits:<8}")

    # By drug class
    print("\n" + "=" * 70)
    print("Precision by Drug Class")
    print("=" * 70)

    print(f"\n{'Drug Class':<25} {'N':<8} {'Precision':<12} {'Hits':<8}")
    print("-" * 55)

    drug_class_stats = {}
    for drug_class in sorted(df['drug_class'].unique()):
        subset = df[df['drug_class'] == drug_class]
        prec = subset['is_hit'].mean()
        hits = subset['is_hit'].sum()
        drug_class_stats[drug_class] = {'n': len(subset), 'precision': prec, 'hits': hits}
        print(f"{drug_class:<25} {len(subset):<8} {100*prec:>8.1f}%    {hits:<8}")

    # Cross-tabulation: Cancer Type x Drug Class
    print("\n" + "=" * 70)
    print("Cross-Tabulation: Cancer Type x Drug Class Precision")
    print("=" * 70)

    # Find combinations with sufficient data
    combinations = []
    for cancer_type in df['cancer_type'].unique():
        for drug_class in df['drug_class'].unique():
            subset = df[(df['cancer_type'] == cancer_type) & (df['drug_class'] == drug_class)]
            if len(subset) >= 20:  # Minimum sample size
                prec = subset['is_hit'].mean()
                hits = subset['is_hit'].sum()
                combinations.append({
                    'cancer_type': cancer_type,
                    'drug_class': drug_class,
                    'n': len(subset),
                    'precision': prec,
                    'hits': hits,
                })

    combinations.sort(key=lambda x: -x['precision'])

    print(f"\n{'Cancer Type':<20} {'Drug Class':<20} {'N':<8} {'Precision':<12}")
    print("-" * 65)

    for combo in combinations[:20]:
        if combo['precision'] > 0:
            print(f"{combo['cancer_type']:<20} {combo['drug_class']:<20} {combo['n']:<8} {100*combo['precision']:>8.1f}%")

    # Success check: any combination >30%?
    print("\n" + "=" * 70)
    print("Success Check: Combinations > 30% Precision")
    print("=" * 70)

    high_prec = [c for c in combinations if c['precision'] >= 0.30]
    if high_prec:
        print(f"\n✓ Found {len(high_prec)} combinations with >30% precision:")
        for combo in high_prec:
            print(f"  {combo['cancer_type']} + {combo['drug_class']}: {100*combo['precision']:.1f}% (n={combo['n']})")
    else:
        print(f"\n✗ No combinations achieved >30% precision")
        best = max(combinations, key=lambda x: x['precision']) if combinations else None
        if best:
            print(f"  Best: {best['cancer_type']} + {best['drug_class']}: {100*best['precision']:.1f}%")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    # Best cancer types
    top_cancer = sorted(cancer_type_stats.items(), key=lambda x: -x[1]['precision'])[:5]
    print("\nTop 5 Cancer Types:")
    for ct, stats in top_cancer:
        if stats['n'] >= 50:
            print(f"  {ct}: {100*stats['precision']:.1f}% (n={stats['n']})")

    # Best drug classes
    top_drug = sorted(drug_class_stats.items(), key=lambda x: -x[1]['precision'])[:5]
    print("\nTop 5 Drug Classes:")
    for dc, stats in top_drug:
        if stats['n'] >= 50:
            print(f"  {dc}: {100*stats['precision']:.1f}% (n={stats['n']})")

    # Save results
    results_file = ANALYSIS_DIR / "h160_cancer_targeted_therapy.json"
    with open(results_file, 'w') as f:
        json.dump({
            'overall_precision': float(overall_prec),
            'n_predictions': int(len(df)),
            'cancer_type_stats': {k: {'n': int(v['n']), 'precision': float(v['precision']), 'hits': int(v['hits'])} for k, v in cancer_type_stats.items()},
            'drug_class_stats': {k: {'n': int(v['n']), 'precision': float(v['precision']), 'hits': int(v['hits'])} for k, v in drug_class_stats.items()},
            'top_combinations': [
                {'cancer_type': c['cancer_type'], 'drug_class': c['drug_class'], 'n': int(c['n']), 'precision': float(c['precision']), 'hits': int(c['hits'])} for c in combinations[:20]
            ],
            'high_precision_combos': [
                {'cancer_type': c['cancer_type'], 'drug_class': c['drug_class'], 'n': int(c['n']), 'precision': float(c['precision']), 'hits': int(c['hits'])} for c in high_prec
            ],
        }, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
