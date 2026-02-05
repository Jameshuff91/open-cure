#!/usr/bin/env python3
"""
h170: Category-Aware kNN: Boost Same-Category Neighbor Weights

Test whether boosting weights for same-category neighbors improves precision
for isolated categories (neurological, respiratory).
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, Tuple, List

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from disease_name_matcher import DiseaseMatcher, load_mesh_mappings
from production_predictor import CATEGORY_KEYWORDS

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"

SEEDS = [42, 123, 456, 789, 1024]


def categorize_disease(disease_name: str) -> str:
    """Categorize disease by keywords."""
    disease_lower = disease_name.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in disease_lower for kw in keywords):
            return category
    return 'other'


def load_node2vec_embeddings() -> Dict[str, np.ndarray]:
    """Load Node2Vec embeddings from CSV file."""
    path = EMBEDDINGS_DIR / "node2vec_256_no_treatment.csv"
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
) -> Tuple[Dict[str, Set[str]], Dict[str, str]]:
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


def disease_level_split(
    gt_pairs: Dict[str, Set[str]],
    valid_entities: Set[str],
    test_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Split by disease (holdout) - no overlap."""
    valid_diseases = [d for d in gt_pairs if d in valid_entities]
    rng = np.random.RandomState(seed)
    rng.shuffle(valid_diseases)
    n_test = max(1, int(len(valid_diseases) * test_fraction))
    test_diseases = set(valid_diseases[:n_test])
    train_diseases = set(valid_diseases[n_test:])
    train_gt = {d: gt_pairs[d] for d in train_diseases}
    test_gt = {d: gt_pairs[d] for d in test_diseases}
    return train_gt, test_gt


def knn_predict_standard(
    test_disease: str,
    train_gt: Dict[str, Set[str]],
    disease_embs: Dict[str, np.ndarray],
    k: int = 20,
) -> List[Tuple[str, float]]:
    """Standard kNN: rank drugs by weighted frequency from k nearest training diseases."""
    test_emb = disease_embs[test_disease]
    test_norm = np.linalg.norm(test_emb)

    # Get training diseases with embeddings
    train_diseases = [d for d in train_gt if d in disease_embs]

    # Compute similarities
    sims = []
    for train_d in train_diseases:
        train_emb = disease_embs[train_d]
        train_norm = np.linalg.norm(train_emb)
        sim = np.dot(test_emb, train_emb) / (test_norm * train_norm + 1e-10)
        sims.append((train_d, sim))

    # Sort and take top k
    sims.sort(key=lambda x: -x[1])
    top_k = sims[:k]

    # Aggregate drug scores
    drug_scores: Dict[str, float] = defaultdict(float)
    for disease_id, sim in top_k:
        for drug in train_gt[disease_id]:
            drug_scores[drug] += sim

    return sorted(drug_scores.items(), key=lambda x: -x[1])


def knn_predict_category_weighted(
    test_disease: str,
    test_category: str,
    train_gt: Dict[str, Set[str]],
    disease_embs: Dict[str, np.ndarray],
    disease_id_to_name: Dict[str, str],
    k: int = 20,
    category_boost: float = 2.0,
) -> List[Tuple[str, float]]:
    """Category-weighted kNN: boost weights for same-category neighbors."""
    test_emb = disease_embs[test_disease]
    test_norm = np.linalg.norm(test_emb)

    # Get training diseases with embeddings
    train_diseases = [d for d in train_gt if d in disease_embs]

    # Compute similarities
    sims = []
    for train_d in train_diseases:
        train_emb = disease_embs[train_d]
        train_norm = np.linalg.norm(train_emb)
        sim = np.dot(test_emb, train_emb) / (test_norm * train_norm + 1e-10)
        sims.append((train_d, sim))

    # Sort and take top k
    sims.sort(key=lambda x: -x[1])
    top_k = sims[:k]

    # Aggregate drug scores with category boost
    drug_scores: Dict[str, float] = defaultdict(float)
    for disease_id, sim in top_k:
        disease_name = disease_id_to_name.get(disease_id, disease_id)
        neighbor_cat = categorize_disease(disease_name)

        # Apply category boost
        weight = sim * category_boost if neighbor_cat == test_category else sim

        for drug in train_gt[disease_id]:
            drug_scores[drug] += weight

    return sorted(drug_scores.items(), key=lambda x: -x[1])


def evaluate_recall_at_k(
    predictions: List[Tuple[str, float]],
    gt_drugs: Set[str],
    k: int = 30,
) -> float:
    """Compute Recall@k."""
    if not gt_drugs:
        return 0.0
    top_k_drugs = set(d for d, _ in predictions[:k])
    hits = len(top_k_drugs & gt_drugs)
    return hits / len(gt_drugs)


def run_evaluation():
    """Run multi-seed evaluation comparing standard vs category-weighted kNN."""
    print("Loading data...")
    embeddings = load_node2vec_embeddings()
    mesh_mappings = load_mesh_mappings_from_file()
    name_to_drug_id, drug_id_to_name = load_drugbank_lookup()
    gt, disease_id_to_name = load_ground_truth(mesh_mappings, name_to_drug_id)

    # Filter to diseases with embeddings
    disease_embs = {d: embeddings[d] for d in gt if d in embeddings}
    valid_entities = set(disease_embs.keys())

    print(f"Diseases with embeddings: {len(disease_embs)}")

    # Categories to analyze
    categories_to_test = ['neurological', 'respiratory', 'autoimmune', 'psychiatric', 'cancer']

    results_by_category = {cat: {'standard': [], 'weighted': []} for cat in categories_to_test}

    # Multi-seed evaluation
    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===")
        train_gt, test_gt = disease_level_split(gt, valid_entities, seed=seed)

        # Categorize test diseases
        test_by_category = defaultdict(list)
        for disease_id in test_gt:
            name = disease_id_to_name.get(disease_id, disease_id)
            cat = categorize_disease(name)
            test_by_category[cat].append(disease_id)

        for category in categories_to_test:
            cat_diseases = test_by_category.get(category, [])
            if not cat_diseases:
                continue

            cat_recalls_std = []
            cat_recalls_weighted = []

            for disease_id in cat_diseases:
                gt_drugs = test_gt[disease_id]

                # Standard kNN
                preds_std = knn_predict_standard(
                    disease_id, train_gt, disease_embs, k=20
                )
                recall_std = evaluate_recall_at_k(preds_std, gt_drugs, k=30)
                cat_recalls_std.append(recall_std)

                # Category-weighted kNN
                disease_name = disease_id_to_name.get(disease_id, disease_id)
                preds_weighted = knn_predict_category_weighted(
                    disease_id, category, train_gt, disease_embs,
                    disease_id_to_name, k=20, category_boost=2.0
                )
                recall_weighted = evaluate_recall_at_k(preds_weighted, gt_drugs, k=30)
                cat_recalls_weighted.append(recall_weighted)

            if cat_recalls_std:
                mean_std = np.mean(cat_recalls_std)
                mean_weighted = np.mean(cat_recalls_weighted)
                results_by_category[category]['standard'].append(mean_std)
                results_by_category[category]['weighted'].append(mean_weighted)
                print(f"  {category}: std={mean_std*100:.1f}%, weighted={mean_weighted*100:.1f}% (n={len(cat_recalls_std)})")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Category-Weighted kNN vs Standard kNN")
    print("="*70)
    print("\n| Category | Standard R@30 | Weighted R@30 | Delta | Significant? |")
    print("|----------|---------------|---------------|-------|--------------|")

    for category in categories_to_test:
        std_vals = results_by_category[category]['standard']
        weighted_vals = results_by_category[category]['weighted']

        if not std_vals:
            continue

        mean_std = np.mean(std_vals) * 100
        mean_weighted = np.mean(weighted_vals) * 100
        std_std = np.std(std_vals) * 100
        std_weighted = np.std(weighted_vals) * 100
        delta = mean_weighted - mean_std

        # Simple significance: delta > std error
        sig = "YES" if abs(delta) > (std_std + std_weighted) / 2 else "no"

        print(f"| {category:12} | {mean_std:5.1f}% ± {std_std:4.1f}% | {mean_weighted:5.1f}% ± {std_weighted:4.1f}% | {delta:+5.1f} pp | {sig:12} |")

    # Overall
    all_std = []
    all_weighted = []
    for cat in categories_to_test:
        all_std.extend(results_by_category[cat]['standard'])
        all_weighted.extend(results_by_category[cat]['weighted'])

    print(f"\nOverall mean: Standard={np.mean(all_std)*100:.1f}%, Weighted={np.mean(all_weighted)*100:.1f}%")
    print(f"Overall delta: {(np.mean(all_weighted) - np.mean(all_std))*100:+.1f} pp")

    # Save results
    output_path = ANALYSIS_DIR / "category_weighted_knn_results.json"
    results = {
        'results_by_category': {k: {'standard': v['standard'], 'weighted': v['weighted']}
                                for k, v in results_by_category.items()},
        'overall_standard': float(np.mean(all_std)),
        'overall_weighted': float(np.mean(all_weighted)),
    }
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    run_evaluation()
