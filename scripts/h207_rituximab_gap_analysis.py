#!/usr/bin/env python3
"""
Hypothesis h207: Rituximab Prediction Gap Analysis.

PURPOSE:
    Investigate why Rituximab IS in the drug pool but NOT predicted for some CD20+ diseases:
    - Follicular lymphoma
    - Burkitt lymphoma
    - DLBCL

    Is it due to:
    1. kNN neighbors not having Rituximab in their GT?
    2. Rituximab score too low?
    3. Other drugs dominating?

APPROACH:
    1. Get kNN neighbors for each missing disease
    2. Check if neighbors have Rituximab in their GT
    3. Analyze which drugs ARE being recommended instead
    4. Compare with diseases WHERE Rituximab IS predicted
"""

import json
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"
DELIVERABLE_PATH = PROJECT_ROOT / "data" / "deliverables" / "drug_repurposing_predictions_with_confidence.xlsx"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"


def load_node2vec_embeddings():
    """Load node2vec embeddings."""
    df = pd.read_csv(EMBEDDINGS_DIR / "node2vec_256_named.csv")
    dim_cols = [c for c in df.columns if c.startswith("dim_")]
    embeddings = {}
    for _, row in df.iterrows():
        entity = f"drkg:{row['entity']}"
        embeddings[entity] = row[dim_cols].values.astype(np.float32)
    return embeddings


def load_drugbank_lookup():
    """Load drugbank name-to-ID mappings."""
    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)
    name_to_id = {name.lower(): f"drkg:Compound::{db_id}" for db_id, name in id_to_name.items()}
    id_to_name_drkg = {f"drkg:Compound::{db_id}": name for db_id, name in id_to_name.items()}
    return name_to_id, id_to_name_drkg


def load_mesh_mappings_from_file():
    """Load MESH mappings."""
    mesh_path = REFERENCE_DIR / "mesh_mappings_from_agents.json"
    if not mesh_path.exists():
        return {}
    with open(mesh_path) as f:
        mesh_data = json.load(f)
    mesh_mappings = {}
    for batch_data in mesh_data.values():
        if isinstance(batch_data, dict):
            for disease_name, mesh_id in batch_data.items():
                if mesh_id:
                    mesh_str = str(mesh_id)
                    if mesh_str.startswith("D") or mesh_str.startswith("C"):
                        mesh_mappings[disease_name.lower()] = f"drkg:Disease::MESH:{mesh_str}"
    return mesh_mappings


def load_ground_truth_by_name():
    """Load ground truth indexed by disease name."""
    df = pd.read_excel(REFERENCE_DIR / "everycure" / "indicationList.xlsx")
    gt_by_name = defaultdict(set)
    for _, row in df.iterrows():
        disease = str(row.get('final normalized disease label', '')).strip().lower()
        drug = str(row.get('final normalized drug label', '')).strip().lower()
        if disease and drug:
            gt_by_name[disease].add(drug)
    return dict(gt_by_name)


def main():
    print("=" * 70)
    print("h207: Rituximab Prediction Gap Analysis")
    print("=" * 70)
    print()

    # Target diseases where Rituximab SHOULD be predicted but isn't
    target_diseases = [
        'follicular lymphoma',
        'burkitt lymphoma',
        'diffuse large b cell lymphoma dlbcl',
    ]

    # Comparison diseases where Rituximab IS predicted
    comparison_diseases = [
        'small lymphocytic lymphoma',
        'mantle cell lymphoma',
        'lymphoplasmacytic lymphoma',
    ]

    # Load data
    print("Loading data...")
    emb_dict = load_node2vec_embeddings()
    name_to_drug_id, id_to_drug_name = load_drugbank_lookup()
    mesh_mappings = load_mesh_mappings_from_file()
    gt_by_name = load_ground_truth_by_name()

    # Load predictions
    pred_df = pd.read_excel(DELIVERABLE_PATH)
    print(f"Loaded {len(pred_df)} predictions")
    print(f"Embeddings: {len(emb_dict)} entities")

    # Get Rituximab info
    rituximab_id = name_to_drug_id.get('rituximab')
    print(f"\nRituximab ID: {rituximab_id}")
    print(f"Rituximab has embedding: {rituximab_id in emb_dict if rituximab_id else False}")

    # Load GT with disease IDs
    df = pd.read_excel(REFERENCE_DIR / "everycure" / "indicationList.xlsx")

    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from disease_name_matcher import load_mesh_mappings, DiseaseMatcher
    fuzzy_mappings = load_mesh_mappings()
    matcher = DiseaseMatcher(fuzzy_mappings)

    # Build GT by disease ID
    all_gt = defaultdict(set)
    disease_names = {}
    disease_name_to_id = {}

    for _, row in df.iterrows():
        disease = str(row.get('disease name', row.get('final normalized disease label', ''))).strip()
        drug = str(row.get('final normalized drug label', '')).strip()
        if not disease or not drug:
            continue

        disease_id = matcher.get_mesh_id(disease)
        if not disease_id:
            disease_id = mesh_mappings.get(disease.lower())
        if not disease_id:
            continue

        drug_id = name_to_drug_id.get(drug.lower())
        if drug_id:
            all_gt[disease_id].add(drug_id)
            disease_names[disease_id] = disease
            disease_name_to_id[disease.lower()] = disease_id

    print(f"GT loaded: {len(all_gt)} diseases with embeddings")

    # Get disease IDs for target diseases from predictions
    def find_disease_id_from_predictions(disease_name_pattern: str):
        """Find disease ID from prediction dataframe."""
        matches = pred_df[pred_df['disease_name'].str.lower().str.contains(disease_name_pattern, na=False)]
        if len(matches) > 0:
            return matches.iloc[0]['disease_id']
        return None

    # Analyze each target disease
    print("\n" + "=" * 70)
    print("ANALYSIS OF CD20+ DISEASES MISSING RITUXIMAB")
    print("=" * 70)

    findings = {}

    for disease_name in target_diseases:
        print(f"\n### {disease_name.upper()} ###")

        # Find in predictions
        disease_preds = pred_df[pred_df['disease_name'].str.lower().str.contains(disease_name.split()[0], na=False)]
        if len(disease_preds) == 0:
            print(f"  Disease not found in predictions")
            continue

        actual_disease_name = disease_preds.iloc[0]['disease_name']
        disease_id = disease_preds.iloc[0]['disease_id']
        print(f"  Disease ID: {disease_id}")
        print(f"  Full name in predictions: {actual_disease_name}")

        # Check if disease has embeddings
        if disease_id not in emb_dict:
            print(f"  ⚠️ Disease NOT in embeddings!")
            continue

        # Get kNN neighbors
        test_emb = emb_dict[disease_id].reshape(1, -1)

        # Get all other diseases with GT and embeddings
        train_diseases = [d for d in all_gt if d in emb_dict and d != disease_id]
        train_embs = np.array([emb_dict[d] for d in train_diseases], dtype=np.float32)

        sims = cosine_similarity(test_emb, train_embs)[0]
        top_k_idx = np.argsort(sims)[-20:]  # k=20
        top_k_sims = sims[top_k_idx]

        print(f"\n  Top 20 Nearest Neighbors (by Node2Vec similarity):")
        neighbor_drugs_all = set()
        neighbors_with_rituximab = []

        for i, idx in enumerate(reversed(top_k_idx)):  # Most similar first
            neighbor_id = train_diseases[idx]
            neighbor_name = disease_names.get(neighbor_id, neighbor_id)
            neighbor_sim = sims[idx]
            neighbor_drugs = all_gt.get(neighbor_id, set())

            has_rituximab = rituximab_id in neighbor_drugs
            rituximab_marker = " [HAS RITUXIMAB]" if has_rituximab else ""

            if has_rituximab:
                neighbors_with_rituximab.append(neighbor_name)

            neighbor_drugs_all.update(neighbor_drugs)

            print(f"    {i+1}. {neighbor_name[:50]}: sim={neighbor_sim:.3f}, {len(neighbor_drugs)} drugs{rituximab_marker}")

        # Check if Rituximab is in neighbor drug pool
        rituximab_in_pool = rituximab_id in neighbor_drugs_all
        print(f"\n  Rituximab in neighbor drug pool: {rituximab_in_pool}")
        print(f"  Neighbors with Rituximab: {len(neighbors_with_rituximab)}/20")
        if neighbors_with_rituximab:
            print(f"    - {', '.join(neighbors_with_rituximab[:5])}{'...' if len(neighbors_with_rituximab) > 5 else ''}")

        # Calculate what Rituximab's score WOULD be
        rituximab_score = 0.0
        if rituximab_id:
            for idx in top_k_idx:
                neighbor_id = train_diseases[idx]
                neighbor_sim = sims[idx]
                neighbor_drugs = all_gt.get(neighbor_id, set())
                if rituximab_id in neighbor_drugs:
                    rituximab_score += neighbor_sim

        print(f"\n  Rituximab weighted score: {rituximab_score:.4f}")

        # Compare with top predicted drugs
        print(f"\n  Top 5 predicted drugs (from prediction file):")
        for _, row in disease_preds.head(5).iterrows():
            known_marker = " [KNOWN]" if row['is_known_indication'] else ""
            print(f"    - {row['drug_name']}: score={row['knn_score']:.3f}{known_marker}")

        # Check if Rituximab's score would make top 30
        min_score = disease_preds.iloc[-1]['knn_score'] if len(disease_preds) > 0 else 0
        would_make_top30 = rituximab_score >= min_score

        print(f"\n  Min score in top 30: {min_score:.4f}")
        print(f"  Rituximab would make top 30: {would_make_top30}")

        findings[disease_name] = {
            'disease_id': disease_id,
            'neighbors_with_rituximab': len(neighbors_with_rituximab),
            'rituximab_in_neighbor_pool': rituximab_in_pool,
            'rituximab_score': float(rituximab_score),
            'min_top30_score': float(min_score),
            'would_make_top30': would_make_top30,
            'neighbor_list': neighbors_with_rituximab[:10],
        }

    # Compare with diseases where Rituximab IS predicted
    print("\n" + "=" * 70)
    print("COMPARISON: CD20+ DISEASES WHERE RITUXIMAB IS PREDICTED")
    print("=" * 70)

    for disease_name in comparison_diseases:
        print(f"\n### {disease_name.upper()} ###")

        disease_preds = pred_df[pred_df['disease_name'].str.lower().str.contains(disease_name.split()[0], na=False)]
        if len(disease_preds) == 0:
            continue

        disease_id = disease_preds.iloc[0]['disease_id']

        if disease_id not in emb_dict:
            print(f"  Disease NOT in embeddings!")
            continue

        # Get kNN neighbors
        test_emb = emb_dict[disease_id].reshape(1, -1)
        train_diseases = [d for d in all_gt if d in emb_dict and d != disease_id]
        train_embs = np.array([emb_dict[d] for d in train_diseases], dtype=np.float32)
        sims = cosine_similarity(test_emb, train_embs)[0]
        top_k_idx = np.argsort(sims)[-20:]

        neighbors_with_rituximab = []
        for idx in top_k_idx:
            neighbor_id = train_diseases[idx]
            neighbor_name = disease_names.get(neighbor_id, neighbor_id)
            neighbor_drugs = all_gt.get(neighbor_id, set())
            if rituximab_id in neighbor_drugs:
                neighbors_with_rituximab.append(neighbor_name)

        # Calculate Rituximab score
        rituximab_score = 0.0
        if rituximab_id:
            for idx in top_k_idx:
                neighbor_id = train_diseases[idx]
                neighbor_sim = sims[idx]
                if rituximab_id in all_gt.get(neighbor_id, set()):
                    rituximab_score += neighbor_sim

        print(f"  Neighbors with Rituximab: {len(neighbors_with_rituximab)}/20")
        print(f"  Rituximab score: {rituximab_score:.4f}")

        # Find Rituximab in predictions
        ritux_pred = disease_preds[disease_preds['drug_name'].str.lower().str.contains('rituximab', na=False)]
        if len(ritux_pred) > 0:
            print(f"  Rituximab rank: score={ritux_pred.iloc[0]['knn_score']:.4f}")

        findings[disease_name] = {
            'neighbors_with_rituximab': len(neighbors_with_rituximab),
            'rituximab_score': float(rituximab_score),
            'is_predicted': True,
        }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: ROOT CAUSE ANALYSIS")
    print("=" * 70)

    print("\n| Disease | Neighbors w/Rituximab | Score | Makes Top 30? |")
    print("|---------|----------------------|-------|---------------|")

    for disease, data in findings.items():
        n_neighbors = data.get('neighbors_with_rituximab', 0)
        score = data.get('rituximab_score', 0)
        makes_top30 = data.get('would_make_top30', data.get('is_predicted', False))
        status = "✓" if makes_top30 else "✗"
        print(f"| {disease[:25]:25} | {n_neighbors:20} | {score:.3f} | {status:13} |")

    # Determine root cause
    print("\n=== ROOT CAUSE ===")
    missing_diseases = [d for d in target_diseases if not findings.get(d, {}).get('would_make_top30', False)]

    for disease in missing_diseases:
        data = findings.get(disease, {})
        if data.get('neighbors_with_rituximab', 0) == 0:
            print(f"{disease}: No neighbors have Rituximab in GT (coverage gap)")
        elif data.get('rituximab_score', 0) < data.get('min_top30_score', 999):
            print(f"{disease}: Rituximab score too low ({data.get('rituximab_score', 0):.3f} < {data.get('min_top30_score', 0):.3f})")

    # Save findings
    output = {
        'hypothesis': 'h207',
        'title': 'Rituximab Prediction Gap Analysis',
        'findings': findings,
        'root_cause': 'kNN neighbors determine drug pool - diseases missing Rituximab have fewer/no neighbors with Rituximab in GT',
    }

    output_path = ANALYSIS_DIR / "h207_rituximab_gap_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved findings to: {output_path}")

    return output


if __name__ == "__main__":
    main()
