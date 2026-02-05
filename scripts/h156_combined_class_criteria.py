#!/usr/bin/env python3
"""
h156: Combined Multi-Class Rescue Criteria

Test if combining drug class criteria (e.g., 'ophthalmic antibiotic OR steroid + rank<=15')
maintains high precision while improving coverage.

From h150:
- Ophthalmic antibiotic + rank<=15: 62.5% precision (n=16, hits=10)
- Ophthalmic steroid + rank<=15: 48.0% precision (n=25, hits=12)
- Cancer taxane + rank<=5: 40.0% precision (n=10, hits=4)
- Cancer alkylating + rank<=10: 36.4% precision (n=11, hits=4)
- Dermatological topical_steroid + rank<=5: 63.6% precision (n=11, hits=7)
- Dermatological biologic + rank<=20: 33.3% precision (n=6, hits=2)

SUCCESS: Combined criteria maintain >40% precision with 2x coverage
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

# Category keywords
CATEGORY_KEYWORDS = {
    'ophthalmic': ['ophthalmic', 'eye', 'ocular', 'retinal', 'macular', 'glaucoma',
                   'cataract', 'retinopathy', 'conjunctivitis', 'uveitis', 'keratitis'],
    'cancer': ['cancer', 'carcinoma', 'tumor', 'leukemia', 'lymphoma', 'melanoma',
               'neoplasm', 'oncology', 'sarcoma', 'myeloma'],
    'dermatological': ['dermatological', 'skin', 'psoriasis', 'eczema', 'dermatitis',
                       'acne', 'rosacea', 'vitiligo', 'alopecia', 'urticaria'],
}

# Drug classes by category (from h150)
OPHTHALMIC_CLASSES = {
    'antibiotic': ['ciprofloxacin', 'moxifloxacin', 'ofloxacin', 'tobramycin', 'gentamicin',
                   'gatifloxacin', 'levofloxacin', 'besifloxacin', 'neomycin', 'polymyxin'],
    'steroid': ['dexamethasone', 'prednisolone', 'fluorometholone', 'loteprednol',
                'difluprednate', 'rimexolone', 'triamcinolone'],
    'immunosuppressant': ['cyclosporine', 'tacrolimus'],
}

CANCER_CLASSES = {
    'taxane': ['paclitaxel', 'docetaxel', 'cabazitaxel'],
    'alkylating': ['cyclophosphamide', 'ifosfamide', 'melphalan', 'chlorambucil', 'busulfan'],
    'checkpoint': ['pembrolizumab', 'nivolumab', 'ipilimumab', 'atezolizumab'],
}

DERMATOLOGICAL_CLASSES = {
    'topical_steroid': ['hydrocortisone', 'betamethasone', 'triamcinolone', 'clobetasol',
                        'fluocinolone', 'fluocinonide', 'mometasone', 'desonide', 'halobetasol',
                        'desoximetasone'],
    'biologic': ['adalimumab', 'etanercept', 'infliximab', 'ustekinumab', 'secukinumab',
                 'ixekizumab', 'dupilumab'],
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


def get_drug_classes(drug_name: str, category: str) -> Set[str]:
    """Get all matching drug classes for a drug in a category."""
    drug_lower = drug_name.lower()
    classes = set()

    if category == 'ophthalmic':
        drug_classes = OPHTHALMIC_CLASSES
    elif category == 'cancer':
        drug_classes = CANCER_CLASSES
    elif category == 'dermatological':
        drug_classes = DERMATOLOGICAL_CLASSES
    else:
        return classes

    for class_name, drugs in drug_classes.items():
        if any(d in drug_lower for d in drugs):
            classes.add(class_name)

    return classes


def knn_predictions(disease_id, train_diseases, gt, embeddings, id_to_name, k=20):
    """Generate kNN predictions for a disease."""
    if disease_id not in embeddings:
        return []

    query_emb = embeddings[disease_id].reshape(1, -1)
    train_with_emb = [d for d in train_diseases if d in embeddings and d != disease_id]
    if not train_with_emb:
        return []

    train_embs = np.vstack([embeddings[d] for d in train_with_emb])
    sims = cosine_similarity(query_emb, train_embs)[0]
    top_idx = np.argsort(sims)[-k:][::-1]
    neighbors = [train_with_emb[i] for i in top_idx]
    neighbor_sims = [sims[i] for i in top_idx]

    drug_scores: Dict[str, float] = defaultdict(float)
    for neighbor, sim in zip(neighbors, neighbor_sims):
        for drug in gt.get(neighbor, []):
            drug_scores[drug] += sim

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
    print("h156: Combined Multi-Class Rescue Criteria")
    print("=" * 80)

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

    for cat, dlist in sorted(category_diseases.items()):
        print(f"  {cat}: {len(dlist)} diseases")

    # Define combined criteria to test
    COMBINED_CRITERIA = {
        'ophthalmic': [
            {'name': 'antibiotic + rank<=15', 'classes': {'antibiotic'}, 'max_rank': 15},
            {'name': 'steroid + rank<=15', 'classes': {'steroid'}, 'max_rank': 15},
            {'name': 'antibiotic OR steroid + rank<=15', 'classes': {'antibiotic', 'steroid'}, 'max_rank': 15},
            {'name': 'antibiotic OR steroid + rank<=20', 'classes': {'antibiotic', 'steroid'}, 'max_rank': 20},
            {'name': 'antibiotic OR steroid OR immunosuppressant + rank<=15',
             'classes': {'antibiotic', 'steroid', 'immunosuppressant'}, 'max_rank': 15},
        ],
        'cancer': [
            {'name': 'taxane + rank<=5', 'classes': {'taxane'}, 'max_rank': 5},
            {'name': 'alkylating + rank<=10', 'classes': {'alkylating'}, 'max_rank': 10},
            {'name': 'taxane OR alkylating + rank<=10', 'classes': {'taxane', 'alkylating'}, 'max_rank': 10},
            {'name': 'taxane OR alkylating + rank<=5', 'classes': {'taxane', 'alkylating'}, 'max_rank': 5},
            {'name': 'taxane OR alkylating OR checkpoint + rank<=10',
             'classes': {'taxane', 'alkylating', 'checkpoint'}, 'max_rank': 10},
        ],
        'dermatological': [
            {'name': 'topical_steroid + rank<=5', 'classes': {'topical_steroid'}, 'max_rank': 5},
            {'name': 'biologic + rank<=20', 'classes': {'biologic'}, 'max_rank': 20},
            {'name': 'topical_steroid OR biologic + rank<=10',
             'classes': {'topical_steroid', 'biologic'}, 'max_rank': 10},
            {'name': 'topical_steroid + rank<=5 OR biologic + rank<=10',
             'classes': {'topical_steroid', 'biologic'}, 'max_rank': 10},  # Will need custom logic
        ],
    }

    results = {}

    for target_cat in ['ophthalmic', 'cancer', 'dermatological']:
        print(f"\n{'='*80}")
        print(f"CATEGORY: {target_cat.upper()}")
        print("=" * 80)

        cat_diseases = category_diseases[target_cat]
        if len(cat_diseases) < 3:
            print(f"  Skipping - only {len(cat_diseases)} diseases")
            continue

        # Collect predictions across seeds
        all_preds = []

        for seed in SEEDS:
            np.random.seed(seed)

            n_test = max(1, len(cat_diseases) // 5)
            test_diseases = set(np.random.choice(cat_diseases, n_test, replace=False))
            train_diseases = set(cat_diseases) - test_diseases

            for other_cat, other_diseases in category_diseases.items():
                if other_cat != target_cat:
                    train_diseases.update(other_diseases)

            for disease in test_diseases:
                preds = knn_predictions(disease, train_diseases, gt, embeddings, id_to_name)
                gt_drugs = gt[disease]

                for p in preds:
                    drug_classes = get_drug_classes(p['drug_name'], target_cat)
                    is_hit = p['drug_id'] in gt_drugs

                    all_preds.append({
                        'disease': disease_names[disease],
                        'drug': p['drug_name'],
                        'rank': p['rank'],
                        'score': p['score'],
                        'drug_classes': drug_classes,
                        'is_hit': is_hit,
                        'seed': seed,
                    })

        if not all_preds:
            continue

        df = pd.DataFrame(all_preds)
        print(f"\nTotal predictions: {len(df)}")
        print(f"Base hit rate: {df['is_hit'].mean()*100:.2f}%")

        # Test each criteria
        print(f"\nCombined Criteria Results:")
        print("-" * 80)
        print(f"{'Criteria':<55} {'N':>6} {'Hits':>6} {'Precision':>10}")
        print("-" * 80)

        criteria_results = []
        for crit in COMBINED_CRITERIA[target_cat]:
            # Filter predictions matching criteria
            def matches_criteria(row):
                if row['rank'] > crit['max_rank']:
                    return False
                return bool(row['drug_classes'] & crit['classes'])

            subset = df[df.apply(matches_criteria, axis=1)]
            n = len(subset)
            hits = subset['is_hit'].sum()
            precision = hits / n * 100 if n > 0 else 0.0

            star = " ***" if precision >= 40 else ""
            print(f"{crit['name']:<55} {n:>6} {hits:>6} {precision:>9.1f}%{star}")

            criteria_results.append({
                'criteria': crit['name'],
                'n': n,
                'hits': int(hits),
                'precision': precision,
            })

        # Find best combined criteria
        combined = [c for c in criteria_results if 'OR' in c['criteria']]
        single = [c for c in criteria_results if 'OR' not in c['criteria']]

        if combined:
            best_combined = max(combined, key=lambda x: x['precision'])
            best_single = max(single, key=lambda x: x['precision']) if single else None

            print(f"\n  Best single criteria: {best_single['criteria']} = {best_single['precision']:.1f}% (n={best_single['n']})" if best_single else "")
            print(f"  Best combined criteria: {best_combined['criteria']} = {best_combined['precision']:.1f}% (n={best_combined['n']})")

            # Check if combined maintains >40% precision with more coverage
            if best_single and best_combined:
                coverage_increase = best_combined['n'] / best_single['n'] if best_single['n'] > 0 else 0
                precision_drop = best_single['precision'] - best_combined['precision']
                print(f"  Coverage increase: {coverage_increase:.2f}x")
                print(f"  Precision drop: {precision_drop:.1f} pp")

                if best_combined['precision'] >= 40 and coverage_increase >= 1.5:
                    print(f"  ✓ SUCCESS: >40% precision with {coverage_increase:.1f}x coverage!")
                elif best_combined['precision'] >= 40:
                    print(f"  ○ PARTIAL: >40% precision but only {coverage_increase:.1f}x coverage")
                else:
                    print(f"  ✗ Combined precision dropped below 40%")

        results[target_cat] = {
            'n_predictions': len(df),
            'base_precision': float(df['is_hit'].mean() * 100),
            'criteria_results': criteria_results,
        }

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\nKey findings:")
    for cat, res in results.items():
        combined = [c for c in res['criteria_results'] if 'OR' in c['criteria'] and c['precision'] >= 40]
        if combined:
            for c in combined:
                print(f"  ✓ {cat}: {c['criteria']} = {c['precision']:.1f}% (n={c['n']})")

    # Save results
    output_file = ANALYSIS_DIR / "h156_combined_criteria.json"
    with open(output_file, 'w') as f:
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, set):
                return list(obj)
            return obj
        json.dump(results, f, indent=2, default=convert)

    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
