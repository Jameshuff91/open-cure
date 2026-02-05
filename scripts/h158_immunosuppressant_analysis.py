#!/usr/bin/env python3
"""
h158: Mechanism-Free Drug Classes - Expand Immunosuppression Pattern

h150 found corticosteroids achieve 48.6% precision for hematological diseases
WITHOUT requiring mechanism support. This hypothesis tests if other immunosuppressants
(cyclosporine, azathioprine, tacrolimus) show similar patterns.

SUCCESS: Find >30% precision for at least one immunosuppressant class
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, List

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

# Immunosuppressant drug classes
IMMUNOSUPPRESSANTS = {
    'corticosteroid': [
        'prednisone', 'prednisolone', 'methylprednisolone', 'dexamethasone',
        'hydrocortisone', 'betamethasone', 'triamcinolone', 'budesonide',
    ],
    'calcineurin_inhibitor': [
        'cyclosporine', 'tacrolimus',
    ],
    'antimetabolite': [
        'azathioprine', 'mycophenolate', 'methotrexate', 'leflunomide',
    ],
    'mtor_inhibitor': [
        'sirolimus', 'everolimus',
    ],
}

# Disease categories that use immunosuppressants
IMMUNOSUPPRESSANT_CATEGORIES = {
    'autoimmune': ['autoimmune', 'lupus', 'rheumatoid', 'arthritis', 'scleroderma', 'myasthenia',
                   'multiple sclerosis', 'crohn', 'colitis', 'psoriasis', 'sjögren', 'vasculitis'],
    'hematological': ['anemia', 'thrombocytopenia', 'leukemia', 'lymphoma', 'myeloma',
                      'myelodysplastic', 'hemolytic', 'aplastic', 'neutropenia', 'hemophilia'],
    'transplant': ['transplant', 'graft', 'rejection'],
    'inflammatory': ['inflammatory', 'inflammation', 'hepatitis', 'uveitis', 'myocarditis'],
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


def load_drug_targets() -> Dict[str, Set[str]]:
    """Load drug target information."""
    try:
        with open(REFERENCE_DIR / "drug_targets.json") as f:
            targets = json.load(f)
        return {k: set(v) for k, v in targets.items()}
    except Exception:
        return {}


def load_disease_genes() -> Dict[str, Set[str]]:
    """Load disease gene information."""
    try:
        with open(REFERENCE_DIR / "disease_genes.json") as f:
            genes = json.load(f)
        return {k: set(v) for k, v in genes.items()}
    except Exception:
        return {}


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


def classify_disease_category(disease_name: str) -> str:
    """Classify disease into immunosuppressant-relevant category."""
    disease_lower = disease_name.lower()
    for category, keywords in IMMUNOSUPPRESSANT_CATEGORIES.items():
        if any(kw in disease_lower for kw in keywords):
            return category
    return 'other'


def get_immunosuppressant_class(drug_name: str) -> str:
    """Get the immunosuppressant class for a drug."""
    drug_lower = drug_name.lower()
    for class_name, drugs in IMMUNOSUPPRESSANTS.items():
        if any(d in drug_lower for d in drugs):
            return class_name
    return None


def has_mechanism_support(drug_id: str, disease_id: str, drug_targets: Dict, disease_genes: Dict) -> bool:
    """Check if drug has direct target overlap with disease genes."""
    targets = drug_targets.get(drug_id, set())
    genes = disease_genes.get(disease_id, set())
    return len(targets & genes) > 0


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
    print("h158: Mechanism-Free Drug Classes - Expand Immunosuppression Pattern")
    print("=" * 80)

    print("\nLoading data...")
    embeddings = load_node2vec_embeddings()
    mesh_mappings = load_mesh_mappings_from_file()
    name_to_drug_id, id_to_name = load_drugbank_lookup()
    gt, disease_names = load_ground_truth(mesh_mappings, name_to_drug_id)
    drug_targets = load_drug_targets()
    disease_genes = load_disease_genes()

    # Filter to diseases with embeddings
    diseases = [d for d in gt.keys() if d in embeddings and len(gt[d]) >= 1]
    print(f"  Total diseases with GT and embeddings: {len(diseases)}")

    # Classify diseases
    category_diseases = defaultdict(list)
    for d in diseases:
        cat = classify_disease_category(disease_names[d])
        if cat != 'other':
            category_diseases[cat].append(d)

    print("\nDisease distribution:")
    for cat, dlist in sorted(category_diseases.items()):
        print(f"  {cat}: {len(dlist)} diseases")

    # Collect predictions across all categories and seeds
    all_preds = []

    for seed in SEEDS:
        np.random.seed(seed)

        for category in category_diseases:
            cat_diseases = category_diseases[category]
            if len(cat_diseases) < 3:
                continue

            n_test = max(1, len(cat_diseases) // 5)
            test_diseases = set(np.random.choice(cat_diseases, n_test, replace=False))
            train_diseases = set(diseases) - test_diseases

            for disease in test_diseases:
                preds = knn_predictions(disease, train_diseases, gt, embeddings, id_to_name)
                gt_drugs = gt[disease]

                for p in preds:
                    immuno_class = get_immunosuppressant_class(p['drug_name'])
                    mech_support = has_mechanism_support(p['drug_id'], disease, drug_targets, disease_genes)
                    is_hit = p['drug_id'] in gt_drugs

                    all_preds.append({
                        'disease': disease_names[disease],
                        'disease_category': category,
                        'drug': p['drug_name'],
                        'drug_id': p['drug_id'],
                        'rank': p['rank'],
                        'score': p['score'],
                        'immuno_class': immuno_class,
                        'mechanism_support': mech_support,
                        'is_hit': is_hit,
                        'seed': seed,
                    })

    if not all_preds:
        print("No predictions generated")
        return

    df = pd.DataFrame(all_preds)
    print(f"\nTotal predictions: {len(df)}")
    print(f"Overall hit rate: {df['is_hit'].mean()*100:.2f}%")

    # Analyze by immunosuppressant class
    print("\n" + "=" * 80)
    print("IMMUNOSUPPRESSANT CLASS ANALYSIS")
    print("=" * 80)

    results = {}

    for class_name in IMMUNOSUPPRESSANTS.keys():
        class_df = df[df['immuno_class'] == class_name]
        if len(class_df) == 0:
            continue

        print(f"\n{class_name.upper()}")
        print("-" * 60)

        # Base precision (no filters)
        n_base = len(class_df)
        hits_base = class_df['is_hit'].sum()
        precision_base = hits_base / n_base * 100 if n_base > 0 else 0

        # With mechanism support
        mech_df = class_df[class_df['mechanism_support']]
        n_mech = len(mech_df)
        hits_mech = mech_df['is_hit'].sum()
        precision_mech = hits_mech / n_mech * 100 if n_mech > 0 else 0

        # Without mechanism support
        no_mech_df = class_df[~class_df['mechanism_support']]
        n_no_mech = len(no_mech_df)
        hits_no_mech = no_mech_df['is_hit'].sum()
        precision_no_mech = hits_no_mech / n_no_mech * 100 if n_no_mech > 0 else 0

        # By rank threshold
        rank_results = []
        for max_rank in [5, 10, 15, 20]:
            rank_df = class_df[class_df['rank'] <= max_rank]
            n_rank = len(rank_df)
            hits_rank = rank_df['is_hit'].sum()
            precision_rank = hits_rank / n_rank * 100 if n_rank > 0 else 0
            rank_results.append({
                'max_rank': max_rank,
                'n': n_rank,
                'hits': int(hits_rank),
                'precision': precision_rank,
            })

        print(f"  Overall: {n_base} preds, {int(hits_base)} hits, {precision_base:.1f}% precision")
        print(f"  With mechanism: {n_mech} preds, {int(hits_mech)} hits, {precision_mech:.1f}% precision")
        print(f"  Without mechanism: {n_no_mech} preds, {int(hits_no_mech)} hits, {precision_no_mech:.1f}% precision")
        print(f"  Mechanism {'HELPS' if precision_mech > precision_no_mech else 'HURTS' if precision_mech < precision_no_mech else 'NEUTRAL'}: {precision_mech - precision_no_mech:+.1f} pp")

        print(f"\n  By rank threshold:")
        print(f"    {'Rank':<10} {'N':>6} {'Hits':>6} {'Precision':>10}")
        for r in rank_results:
            star = " ***" if r['precision'] >= 30 else ""
            print(f"    <=  {r['max_rank']:<6} {r['n']:>6} {r['hits']:>6} {r['precision']:>9.1f}%{star}")

        # By category
        print(f"\n  By disease category:")
        for cat in ['autoimmune', 'hematological', 'inflammatory', 'transplant']:
            cat_class_df = class_df[class_df['disease_category'] == cat]
            if len(cat_class_df) == 0:
                continue
            n_cat = len(cat_class_df)
            hits_cat = cat_class_df['is_hit'].sum()
            precision_cat = hits_cat / n_cat * 100
            star = " ***" if precision_cat >= 30 else ""
            print(f"    {cat:<15} {n_cat:>6} preds, {int(hits_cat):>4} hits, {precision_cat:>6.1f}%{star}")

        results[class_name] = {
            'total_n': n_base,
            'total_precision': precision_base,
            'with_mechanism_n': n_mech,
            'with_mechanism_precision': precision_mech,
            'without_mechanism_n': n_no_mech,
            'without_mechanism_precision': precision_no_mech,
            'mechanism_delta': precision_mech - precision_no_mech,
            'rank_results': rank_results,
        }

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: High-Precision Immunosuppressant Findings")
    print("=" * 80)

    successes = []
    for class_name, res in results.items():
        # Check if any criteria achieves >30%
        best_rank = max(res['rank_results'], key=lambda x: x['precision'])
        if best_rank['precision'] >= 30 and best_rank['n'] >= 5:
            successes.append({
                'class': class_name,
                'best_criteria': f"rank<={best_rank['max_rank']}",
                'precision': best_rank['precision'],
                'n': best_rank['n'],
                'mechanism_effect': res['mechanism_delta'],
            })

    if successes:
        print("\n✓ SUCCESS: Found high-precision immunosuppressant classes:")
        for s in sorted(successes, key=lambda x: -x['precision']):
            mech_note = "(mechanism HELPS)" if s['mechanism_effect'] > 0 else "(mechanism HURTS or neutral)"
            print(f"  - {s['class']} + {s['best_criteria']}: {s['precision']:.1f}% (n={s['n']}) {mech_note}")
    else:
        print("\n✗ No immunosuppressant class achieved >30% precision with n>=5")

    # Save results
    output_file = ANALYSIS_DIR / "h158_immunosuppressant_analysis.json"
    with open(output_file, 'w') as f:
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            return obj
        json.dump(results, f, indent=2, default=convert)

    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
