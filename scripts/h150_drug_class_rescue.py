#!/usr/bin/env python3
"""
h150: Drug Class Rescue for Other Categories

h144 showed drug class (statin) achieves 60% precision for metabolic.
Test if drug class-based rescue criteria work for other weak categories.

TARGET CATEGORIES:
- gastrointestinal: 0% base precision (worst)
- cancer: 4.9% base precision
- hematological: 2.6% base precision
- neurological: 2.0% base precision

DRUG CLASSES TO TEST:
- GI: PPIs (omeprazole, esomeprazole), H2 blockers, anti-TNF for IBD
- Cancer: Alkylating agents, antimetabolites, kinase inhibitors
- Hematological: EPO stimulants, colony factors, anticoagulants
- Neurological: Anti-epileptics, dopaminergics, cholinesterase inhibitors

SUCCESS CRITERIA: Find criteria achieving >30% precision for at least one category
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

# Category keywords for classification
CATEGORY_KEYWORDS = {
    'gastrointestinal': ['gastrointestinal', 'gastric', 'intestinal', 'bowel', 'liver',
                         'hepatic', 'cirrhosis', 'pancreatitis', 'celiac', 'crohn',
                         'colitis', 'gerd', 'ulcer', 'ibs', 'ibd'],
    'cancer': ['cancer', 'carcinoma', 'tumor', 'leukemia', 'lymphoma', 'melanoma',
               'neoplasm', 'sarcoma', 'myeloma', 'oncology'],
    'hematological': ['anemia', 'hemophilia', 'thrombocytopenia', 'neutropenia',
                      'hematological', 'myelodysplastic', 'blood disorder'],
    'neurological': ['neurological', 'alzheimer', 'parkinson', 'epilepsy', 'neuropathy',
                     'dementia', 'huntington', 'brain', 'seizure', 'stroke'],
}

# Drug classes by category
GI_DRUG_CLASSES = {
    'ppi': ['omeprazole', 'esomeprazole', 'pantoprazole', 'lansoprazole', 'rabeprazole', 'dexlansoprazole'],
    'h2_blocker': ['ranitidine', 'famotidine', 'cimetidine', 'nizatidine'],
    'anti_tnf': ['infliximab', 'adalimumab', 'certolizumab'],
    'aminosalicylate': ['mesalamine', 'sulfasalazine', 'balsalazide', 'olsalazine'],
    'antidiarrheal': ['loperamide', 'diphenoxylate'],
    'prokinetic': ['metoclopramide', 'domperidone'],
}

CANCER_DRUG_CLASSES = {
    'alkylating': ['cyclophosphamide', 'ifosfamide', 'melphalan', 'chlorambucil', 'busulfan'],
    'antimetabolite': ['methotrexate', 'fluorouracil', 'capecitabine', 'gemcitabine', 'cytarabine', 'pemetrexed'],
    'platinum': ['cisplatin', 'carboplatin', 'oxaliplatin'],
    'taxane': ['paclitaxel', 'docetaxel'],
    'kinase_inhibitor': ['imatinib', 'dasatinib', 'nilotinib', 'sunitinib', 'sorafenib', 'erlotinib', 'gefitinib'],
    'checkpoint': ['pembrolizumab', 'nivolumab', 'ipilimumab', 'atezolizumab'],
    'hormone': ['tamoxifen', 'letrozole', 'anastrozole', 'exemestane', 'fulvestrant'],
}

HEMATOLOGICAL_DRUG_CLASSES = {
    'epo': ['epoetin', 'darbepoetin'],
    'colony_factor': ['filgrastim', 'pegfilgrastim', 'sargramostim'],
    'anticoagulant': ['warfarin', 'heparin', 'enoxaparin', 'rivaroxaban', 'apixaban', 'dabigatran'],
    'antiplatelet': ['aspirin', 'clopidogrel', 'ticagrelor', 'prasugrel'],
    'iron': ['ferrous', 'iron'],
}

NEUROLOGICAL_DRUG_CLASSES = {
    'antiepileptic': ['valproate', 'carbamazepine', 'phenytoin', 'levetiracetam', 'lamotrigine', 'topiramate'],
    'dopaminergic': ['levodopa', 'carbidopa', 'pramipexole', 'ropinirole', 'rotigotine'],
    'cholinesterase': ['donepezil', 'rivastigmine', 'galantamine'],
    'nmda_antagonist': ['memantine'],
    'triptans': ['sumatriptan', 'rizatriptan', 'eletriptan', 'naratriptan'],
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
        disease_name = str(row['disease name']).lower().strip()
        drug_name = str(row['final normalized drug label']).lower().strip()

        # Get disease ID
        disease_id = mesh_mappings.get(disease_name)
        if not disease_id:
            disease_id = matcher.get_mesh_id(disease_name)
        if not disease_id:
            continue

        # Get drug ID
        drug_id = name_to_drug_id.get(drug_name)
        if drug_id:
            gt[disease_id].add(drug_id)
            disease_names[disease_id] = disease_name

    return gt, disease_names


def classify_disease(disease_name: str) -> str:
    disease_lower = disease_name.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in disease_lower for kw in keywords):
            return category
    return 'other'


def classify_drug(drug_name: str, category: str) -> str:
    """Classify drug by class for a given disease category."""
    drug_lower = drug_name.lower()

    if category == 'gastrointestinal':
        drug_classes = GI_DRUG_CLASSES
    elif category == 'cancer':
        drug_classes = CANCER_DRUG_CLASSES
    elif category == 'hematological':
        drug_classes = HEMATOLOGICAL_DRUG_CLASSES
    elif category == 'neurological':
        drug_classes = NEUROLOGICAL_DRUG_CLASSES
    else:
        return 'other'

    for class_name, drugs in drug_classes.items():
        if any(d in drug_lower for d in drugs):
            return class_name

    return 'other'


def knn_predictions(disease_id, train_diseases, gt, embeddings, id_to_name, k=20):
    """Generate kNN predictions for a disease."""
    if disease_id not in embeddings:
        return []

    query_emb = embeddings[disease_id].reshape(1, -1)

    # Get all training diseases with embeddings
    train_with_emb = [d for d in train_diseases if d in embeddings and d != disease_id]
    if not train_with_emb:
        return []

    # Compute similarities
    train_embs = np.vstack([embeddings[d] for d in train_with_emb])
    sims = cosine_similarity(query_emb, train_embs)[0]

    # Get top-k neighbors
    top_idx = np.argsort(sims)[-k:][::-1]
    neighbors = [train_with_emb[i] for i in top_idx]
    neighbor_sims = [sims[i] for i in top_idx]

    # Collect drug scores from neighbors
    drug_scores: Dict[str, float] = defaultdict(float)
    for neighbor, sim in zip(neighbors, neighbor_sims):
        for drug in gt.get(neighbor, []):
            drug_scores[drug] += sim

    # Sort by score
    sorted_drugs = sorted(drug_scores.items(), key=lambda x: x[1], reverse=True)

    predictions = []
    for rank, (drug_id, score) in enumerate(sorted_drugs[:30], 1):
        drug_name = id_to_name.get(drug_id, drug_id)
        predictions.append({
            'drug_id': drug_id,
            'drug_name': drug_name,
            'rank': rank,
            'score': score,
        })

    return predictions


def main():
    print("h150: Drug Class Rescue for Other Categories")
    print("=" * 70)

    print("\nLoading data...")
    embeddings = load_node2vec_embeddings()
    mesh_mappings = load_mesh_mappings_from_file()
    name_to_drug_id, id_to_name = load_drugbank_lookup()
    gt, disease_names = load_ground_truth(mesh_mappings, name_to_drug_id)

    # Filter to diseases with embeddings
    diseases = [d for d in gt.keys() if d in embeddings and len(gt[d]) >= 1]
    print(f"  Total diseases with GT and embeddings: {len(diseases)}")

    # Classify diseases by category
    category_diseases = defaultdict(list)
    for d in diseases:
        cat = classify_disease(disease_names[d])
        if cat in CATEGORY_KEYWORDS:
            category_diseases[cat].append(d)

    for cat, dlist in category_diseases.items():
        print(f"  {cat}: {len(dlist)} diseases")

    # Results collector
    results = {}

    # Test each category
    for target_cat in ['gastrointestinal', 'cancer', 'hematological', 'neurological']:
        print(f"\n{'='*70}")
        print(f"CATEGORY: {target_cat.upper()}")
        print("=" * 70)

        cat_diseases = category_diseases[target_cat]
        if len(cat_diseases) < 3:
            print(f"  Skipping - only {len(cat_diseases)} diseases")
            results[target_cat] = {'skipped': True, 'n_diseases': len(cat_diseases)}
            continue

        # Collect predictions across seeds
        all_preds = []

        for seed in SEEDS:
            np.random.seed(seed)

            # 80/20 train/test split
            n_test = max(1, len(cat_diseases) // 5)
            test_diseases = set(np.random.choice(cat_diseases, n_test, replace=False))
            train_diseases = set(cat_diseases) - test_diseases

            # Add other category diseases to training
            for other_cat, other_diseases in category_diseases.items():
                if other_cat != target_cat:
                    train_diseases.update(other_diseases)

            for disease in test_diseases:
                preds = knn_predictions(disease, train_diseases, gt, embeddings, id_to_name)
                gt_drugs = gt[disease]

                for p in preds:
                    drug_class = classify_drug(p['drug_name'], target_cat)
                    is_hit = p['drug_id'] in gt_drugs

                    all_preds.append({
                        'disease': disease_names[disease],
                        'drug': p['drug_name'],
                        'rank': p['rank'],
                        'score': p['score'],
                        'drug_class': drug_class,
                        'is_hit': is_hit,
                        'seed': seed,
                    })

        if not all_preds:
            print("  No predictions generated")
            results[target_cat] = {'skipped': True, 'reason': 'no predictions'}
            continue

        df = pd.DataFrame(all_preds)

        print(f"\nTotal predictions: {len(df)}")
        print(f"Base hit rate: {df['is_hit'].mean()*100:.2f}%")

        # Analyze by drug class
        print(f"\nPrecision by Drug Class:")
        print("-" * 60)
        print(f"{'Drug Class':<25} {'N':>6} {'Hits':>6} {'Precision':>10}")
        print("-" * 60)

        class_results = {}
        for drug_class in df['drug_class'].unique():
            subset = df[df['drug_class'] == drug_class]
            n = len(subset)
            hits = subset['is_hit'].sum()
            precision = hits / n * 100 if n > 0 else 0
            class_results[drug_class] = {'n': n, 'hits': int(hits), 'precision': precision}
            print(f"{drug_class:<25} {n:>6} {hits:>6} {precision:>9.1f}%")

        # Test combined criteria
        print(f"\nTesting Rescue Criteria:")
        print("-" * 70)

        criteria_results = []

        # Test rank + drug class combinations
        for drug_class in [c for c in df['drug_class'].unique() if c != 'other']:
            for max_rank in [5, 10, 15]:
                subset = df[(df['drug_class'] == drug_class) & (df['rank'] <= max_rank)]
                if len(subset) >= 5:  # Need minimum sample
                    precision = subset['is_hit'].mean() * 100
                    criteria = f"{drug_class} + rank<={max_rank}"
                    print(f"  {criteria:<40} N={len(subset):>4}  Precision: {precision:.1f}%")
                    criteria_results.append({
                        'criteria': criteria,
                        'n': len(subset),
                        'precision': precision,
                    })

        # Find best criteria
        if criteria_results:
            best = max(criteria_results, key=lambda x: x['precision'])
            print(f"\n  BEST: {best['criteria']} = {best['precision']:.1f}%")

            if best['precision'] >= 30:
                print(f"  ✓ SUCCESS: Found criteria achieving >30% precision!")
            else:
                print(f"  ✗ No criteria achieved >30% precision target")

        results[target_cat] = {
            'n_predictions': len(df),
            'base_precision': float(df['is_hit'].mean() * 100),
            'class_results': class_results,
            'criteria_results': criteria_results,
            'best_criteria': best['criteria'] if criteria_results else None,
            'best_precision': best['precision'] if criteria_results else 0,
            'success': best['precision'] >= 30 if criteria_results else False,
        }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    any_success = False
    for cat, res in results.items():
        if res.get('skipped'):
            print(f"{cat}: SKIPPED")
        else:
            best = res.get('best_criteria', 'N/A')
            prec = res.get('best_precision', 0)
            success = res.get('success', False)
            status = "✓ SUCCESS" if success else "✗ FAILED"
            print(f"{cat}: {best} = {prec:.1f}% {status}")
            if success:
                any_success = True

    print(f"\nOverall: {'VALIDATED' if any_success else 'INVALIDATED'} - " +
          f"{'Found' if any_success else 'No'} drug class criteria achieving >30% precision")

    # Save results
    output = {
        'category_results': results,
        'success': any_success,
    }

    results_file = ANALYSIS_DIR / "h150_drug_class_rescue.json"
    with open(results_file, 'w') as f:
        # Convert numpy types
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            return obj

        json.dump(output, f, indent=2, default=convert)

    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
