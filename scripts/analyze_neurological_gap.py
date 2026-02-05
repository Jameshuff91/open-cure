#!/usr/bin/env python3
"""
h168: Neurological Disease Performance Gap Analysis

Analyze why neurological diseases have the lowest MEDIUM-tier precision (26.1% vs 45%+ for most categories).
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, Tuple, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from disease_name_matcher import DiseaseMatcher, load_mesh_mappings
from production_predictor import CATEGORY_KEYWORDS

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"


def categorize_disease(disease_name: str) -> str:
    """Categorize disease by keywords."""
    disease_lower = disease_name.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in disease_lower for kw in keywords):
            return category
    return 'other'


def load_node2vec_embeddings(no_treatment: bool = True) -> Dict[str, np.ndarray]:
    """Load Node2Vec embeddings from CSV file."""
    if no_treatment:
        path = EMBEDDINGS_DIR / "node2vec_256_no_treatment.csv"
    else:
        path = EMBEDDINGS_DIR / "node2vec_256_named.csv"

    print(f"  Loading embeddings from: {path}")
    df = pd.read_csv(path)
    dim_cols = [c for c in df.columns if c.startswith("dim_")]
    embeddings: Dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        entity = f"drkg:{row['entity']}"
        embeddings[entity] = row[dim_cols].values.astype(np.float32)
    return embeddings


def load_mesh_mappings_from_file() -> Dict[str, str]:
    """Load MESH mappings from file."""
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


def load_drugbank_lookup() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Load DrugBank mappings."""
    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)
    name_to_id = {name.lower(): f"drkg:Compound::{db_id}" for db_id, name in id_to_name.items()}
    id_to_name_drkg = {f"drkg:Compound::{db_id}": name for db_id, name in id_to_name.items()}
    return name_to_id, id_to_name_drkg


def load_ground_truth(
    mesh_mappings: Dict[str, str],
    name_to_drug_id: Dict[str, str],
) -> Dict[str, Set[str]]:
    """Load ground truth with fuzzy matching."""
    df = pd.read_excel(REFERENCE_DIR / "everycure" / "indicationList.xlsx")
    fuzzy_mappings = load_mesh_mappings()
    matcher = DiseaseMatcher(fuzzy_mappings)

    gt_pairs: Dict[str, Set[str]] = defaultdict(set)
    disease_name_to_id: Dict[str, str] = {}

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
        drug_id = name_to_drug_id.get(drug.lower())
        if drug_id:
            gt_pairs[disease_id].add(drug_id)
            if disease_id not in disease_name_to_id:
                disease_name_to_id[disease_id] = disease

    return dict(gt_pairs), disease_name_to_id


def analyze_neurological_vs_others():
    """Compare neurological to high-performing categories."""

    print("Loading data...")
    embeddings = load_node2vec_embeddings(no_treatment=True)
    mesh_mappings = load_mesh_mappings_from_file()
    name_to_drug_id, drug_id_to_name = load_drugbank_lookup()
    gt, disease_id_to_name = load_ground_truth(mesh_mappings, name_to_drug_id)

    # Get disease embeddings (for diseases that are in GT and have embeddings)
    disease_embs = {}
    for disease_id in gt.keys():
        if disease_id in embeddings:
            disease_embs[disease_id] = embeddings[disease_id]

    print(f"\nDiseases in GT with embeddings: {len(disease_embs)}")
    print(f"Total diseases in GT: {len(gt)}")

    # Categorize diseases
    diseases_by_category = defaultdict(list)
    for disease_id in disease_embs:
        name = disease_id_to_name.get(disease_id, disease_id.split("::")[-1])
        category = categorize_disease(name)
        diseases_by_category[category].append(disease_id)

    print(f"\nDiseases by category (with embeddings):")
    for cat, diseases in sorted(diseases_by_category.items(), key=lambda x: -len(x[1])):
        print(f"  {cat}: {len(diseases)}")

    # Focus on neurological vs top performers
    categories_to_analyze = ['neurological', 'autoimmune', 'psychiatric', 'respiratory', 'cancer']

    # EMBEDDING ANALYSIS
    print("\n" + "="*60)
    print("EMBEDDING ANALYSIS")
    print("="*60)

    for category in categories_to_analyze:
        diseases = diseases_by_category[category]
        if len(diseases) < 3:
            print(f"\n{category.upper()}: Not enough diseases ({len(diseases)})")
            continue

        cat_embeddings = np.array([disease_embs[d] for d in diseases])
        norms = np.linalg.norm(cat_embeddings, axis=1, keepdims=True)
        normalized = cat_embeddings / (norms + 1e-10)
        sim_matrix = normalized @ normalized.T
        upper_tri = sim_matrix[np.triu_indices(len(diseases), k=1)]

        print(f"\n{category.upper()} ({len(diseases)} diseases):")
        print(f"  Intra-category similarity: mean={upper_tri.mean():.3f}, std={upper_tri.std():.3f}")

        # GT drug statistics
        all_drugs_in_cat = set()
        drug_counts = []
        for d in diseases:
            drugs = gt[d]
            all_drugs_in_cat.update(drugs)
            drug_counts.append(len(drugs))
        print(f"  Unique GT drugs: {len(all_drugs_in_cat)}")
        print(f"  GT drugs per disease: mean={np.mean(drug_counts):.1f}, std={np.std(drug_counts):.1f}")

    # kNN ANALYSIS FOR NEUROLOGICAL
    print("\n" + "="*60)
    print("kNN COVERAGE ANALYSIS BY CATEGORY")
    print("="*60)

    all_disease_ids = list(disease_embs.keys())
    all_embs = np.array([disease_embs[d] for d in all_disease_ids])
    all_norms = np.linalg.norm(all_embs, axis=1, keepdims=True)
    all_normalized = all_embs / (all_norms + 1e-10)
    disease_to_idx = {d: i for i, d in enumerate(all_disease_ids)}

    category_coverage = {}
    category_analysis_details = {}

    for category in categories_to_analyze:
        diseases = diseases_by_category[category]
        if len(diseases) < 3:
            continue

        coverages = []
        analysis_items = []

        for disease_id in diseases:
            disease_idx = disease_to_idx[disease_id]
            disease_vec = all_normalized[disease_idx:disease_idx+1]
            sims = (disease_vec @ all_normalized.T).flatten()

            # k=20 neighbors excluding self
            sorted_indices = np.argsort(sims)[::-1]
            top_k_indices = [i for i in sorted_indices if i != disease_idx][:20]

            gt_drugs = gt[disease_id]
            neighbor_drugs = set()
            neighbor_cats = []

            for idx in top_k_indices:
                neighbor = all_disease_ids[idx]
                neighbor_drugs.update(gt.get(neighbor, set()))
                neighbor_name = disease_id_to_name.get(neighbor, neighbor)
                neighbor_cats.append(categorize_disease(neighbor_name))

            if gt_drugs:
                coverage = len(gt_drugs & neighbor_drugs) / len(gt_drugs)
            else:
                coverage = 0

            coverages.append(coverage)

            # Get disease name
            disease_name = disease_id_to_name.get(disease_id, disease_id)

            analysis_items.append({
                'disease': disease_name,
                'disease_id': disease_id,
                'gt_drug_count': len(gt_drugs),
                'coverage': coverage,
                'neighbor_mean_sim': sims[top_k_indices].mean(),
                'same_cat_neighbors': neighbor_cats.count(category),
                'gt_drugs': [drug_id_to_name.get(d, d) for d in list(gt_drugs)[:5]]
            })

        category_coverage[category] = {
            'mean': np.mean(coverages),
            'std': np.std(coverages),
            'n_diseases': len(diseases)
        }
        category_analysis_details[category] = sorted(analysis_items, key=lambda x: x['coverage'])

        print(f"\n{category.upper()} kNN k=20 coverage:")
        print(f"  Mean: {np.mean(coverages)*100:.1f}%, Std: {np.std(coverages)*100:.1f}%")
        print(f"  Zero coverage diseases: {sum(1 for c in coverages if c == 0)} / {len(coverages)}")

    # Detailed neurological analysis
    print("\n" + "="*60)
    print("NEUROLOGICAL DISEASE BREAKDOWN")
    print("="*60)

    neuro_items = category_analysis_details.get('neurological', [])

    print("\n**LOWEST COVERAGE** (model fails):")
    for item in neuro_items[:10]:
        print(f"\n  {item['disease']}:")
        print(f"    GT drugs: {item['gt_drug_count']}, Coverage: {item['coverage']*100:.1f}%")
        print(f"    Neighbor sim: {item['neighbor_mean_sim']:.3f}, Same-cat neighbors: {item['same_cat_neighbors']}/20")
        print(f"    Sample GT: {item['gt_drugs']}")

    print("\n\n**HIGHEST COVERAGE** (model succeeds):")
    for item in neuro_items[-5:]:
        print(f"\n  {item['disease']}:")
        print(f"    GT drugs: {item['gt_drug_count']}, Coverage: {item['coverage']*100:.1f}%")
        print(f"    Neighbor sim: {item['neighbor_mean_sim']:.3f}, Same-cat neighbors: {item['same_cat_neighbors']}/20")
        print(f"    Sample GT: {item['gt_drugs']}")

    # DRUG FREQUENCY ANALYSIS
    print("\n" + "="*60)
    print("NEUROLOGICAL-SPECIFIC DRUG ANALYSIS")
    print("="*60)

    neuro_diseases = diseases_by_category['neurological']
    neuro_predicted_drugs = defaultdict(int)
    neuro_gt_drugs = defaultdict(int)

    for disease_id in neuro_diseases:
        disease_idx = disease_to_idx[disease_id]
        disease_vec = all_normalized[disease_idx:disease_idx+1]
        sims = (disease_vec @ all_normalized.T).flatten()
        sorted_indices = np.argsort(sims)[::-1]
        top_k_indices = [i for i in sorted_indices if i != disease_idx][:20]

        for idx in top_k_indices:
            neighbor = all_disease_ids[idx]
            for drug in gt.get(neighbor, set()):
                neuro_predicted_drugs[drug] += 1

        for drug in gt.get(disease_id, set()):
            neuro_gt_drugs[drug] += 1

    print("\nTop 20 drugs predicted for neurological (by kNN frequency):")
    sorted_predicted = sorted(neuro_predicted_drugs.items(), key=lambda x: -x[1])[:20]
    for drug_id, count in sorted_predicted:
        in_gt = neuro_gt_drugs.get(drug_id, 0)
        precision = in_gt / count if count > 0 else 0
        drug_name = drug_id_to_name.get(drug_id, drug_id)
        print(f"  {drug_name}: {count} predictions, {in_gt} GT hits, {precision*100:.1f}% precision")

    # Check if neurological drugs appear in neighbors
    print("\n\nNeurological-specific drugs in GT (how often they appear in kNN predictions):")
    neuro_specific = ['levodopa', 'carbidopa', 'valproic acid', 'carbamazepine',
                      'phenytoin', 'lamotrigine', 'topiramate', 'gabapentin',
                      'donepezil', 'memantine', 'pramipexole', 'ropinirole']

    for drug_name in neuro_specific:
        drug_id = name_to_drug_id.get(drug_name.lower())
        if drug_id:
            freq = neuro_predicted_drugs.get(drug_id, 0)
            gt_count = neuro_gt_drugs.get(drug_id, 0)
            print(f"  {drug_name}: {freq} kNN predictions, {gt_count} GT")

    # COMPARE TO HIGH PERFORMERS
    print("\n" + "="*60)
    print("CATEGORY COMPARISON SUMMARY")
    print("="*60)

    print("\n| Category | Mean Coverage | Zero-Coverage | Same-Cat Neighbors |")
    print("|----------|---------------|---------------|-------------------|")
    for category in categories_to_analyze:
        if category not in category_coverage:
            continue
        cov = category_coverage[category]
        items = category_analysis_details[category]
        zero_cov = sum(1 for it in items if it['coverage'] == 0)
        same_cat = np.mean([it['same_cat_neighbors'] for it in items])
        print(f"| {category:12} | {cov['mean']*100:10.1f}% | {zero_cov:7}/{cov['n_diseases']} | {same_cat:12.1f}/20 |")

    # Save results
    results = {
        'category_coverage': category_coverage,
        'neurological_details': neuro_items,
        'top_predicted_drugs': [(drug_id_to_name.get(d, d), c) for d, c in sorted_predicted],
    }

    output_path = ANALYSIS_DIR / "neurological_gap_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n\nResults saved to {output_path}")

    return results


if __name__ == '__main__':
    analyze_neurological_vs_others()
