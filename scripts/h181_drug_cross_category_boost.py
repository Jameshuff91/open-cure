#!/usr/bin/env python3
"""
h181: Drug-Level Cross-Category Transfer

PURPOSE:
    Test whether drugs that treat multiple disease categories should be
    boosted when predicting for related categories.

HYPOTHESIS:
    If a drug treats diseases in multiple categories, it might be a good
    repurposing candidate for related categories where it hasn't been tested.

APPROACH:
    1. Build drug→category mapping from GT
    2. For each test disease, boost kNN predictions for drugs that already
       treat related categories
    3. Compare R@30 vs baseline kNN

SUCCESS CRITERIA:
    >1pp improvement over baseline kNN (37% R@30)
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.disease_name_matcher import DiseaseMatcher, load_mesh_mappings
from src.disease_categorizer import categorize_disease

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"

SEEDS = [42, 123, 456, 789, 1024]
K_DEFAULT = 20

# Category relationships (from co-treatment analysis)
# Format: {category: [related_categories ordered by co-treatment %]}
RELATED_CATEGORIES = {
    'respiratory': ['infectious'],  # 41% of respiratory drugs also treat infectious
    'cardiovascular': ['neurological', 'metabolic'],  # 30%, 25%
    'neurological': ['cardiovascular', 'metabolic'],  # 39%, 22%
    'metabolic': ['cardiovascular', 'neurological'],  # 31%, 21%
    'infectious': ['respiratory'],  # 26%
    'autoimmune': ['infectious', 'dermatological', 'musculoskeletal'],  # 15%, 14%
    'cancer': ['autoimmune', 'hematological'],  # 5%, related therapeutics
    'hematological': ['cancer', 'cardiovascular'],
    'gastrointestinal': ['infectious', 'autoimmune'],
    'dermatological': ['autoimmune'],
    'ophthalmological': ['autoimmune'],
    'renal': ['cardiovascular', 'metabolic'],
    'musculoskeletal': ['autoimmune'],
    'psychiatric': ['neurological'],
    'endocrine': ['metabolic'],
}


def load_node2vec_embeddings() -> Dict[str, np.ndarray]:
    """Load Node2Vec embeddings."""
    df = pd.read_csv(EMBEDDINGS_DIR / "node2vec_256_named.csv")
    dim_cols = [c for c in df.columns if c.startswith("dim_")]
    embeddings: Dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        entity = f"drkg:{row['entity']}"
        embeddings[entity] = row[dim_cols].values.astype(np.float32)
    return embeddings


def load_drugbank_lookup() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Load DrugBank ID mappings."""
    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)
    name_to_id = {name.lower(): f"drkg:Compound::{db_id}" for db_id, name in id_to_name.items()}
    id_to_name_drkg = {f"drkg:Compound::{db_id}": name for db_id, name in id_to_name.items()}
    return name_to_id, id_to_name_drkg


def load_ground_truth(name_to_drug_id: Dict[str, str]) -> Tuple[Dict[str, Set[str]], Dict[str, str], Dict[str, str]]:
    """Load Every Cure ground truth."""
    mesh_mappings = load_mesh_mappings()
    matcher = DiseaseMatcher(mesh_mappings)

    df = pd.read_excel(REFERENCE_DIR / "everycure" / "indicationList.xlsx")

    gt: Dict[str, Set[str]] = defaultdict(set)
    disease_names: Dict[str, str] = {}
    disease_categories: Dict[str, str] = {}

    for _, row in df.iterrows():
        disease = str(row.get("disease name", "")).strip()
        drug = str(row.get("final normalized drug label", "")).strip()
        if not disease or not drug:
            continue

        disease_id = matcher.get_mesh_id(disease)
        if not disease_id:
            continue

        disease_names[disease_id] = disease
        cat = categorize_disease(disease)
        if cat:
            disease_categories[disease_id] = cat

        drug_id = name_to_drug_id.get(drug.lower())
        if drug_id:
            gt[disease_id].add(drug_id)

    return dict(gt), disease_names, disease_categories


def build_drug_categories(gt: Dict[str, Set[str]], disease_categories: Dict[str, str]) -> Dict[str, Set[str]]:
    """Build mapping of drug_id -> set of categories it treats."""
    drug_cats: Dict[str, Set[str]] = defaultdict(set)

    for disease_id, drugs in gt.items():
        cat = disease_categories.get(disease_id)
        if cat:
            for drug_id in drugs:
                drug_cats[drug_id].add(cat)

    return dict(drug_cats)


def knn_predict(
    disease_id: str,
    embeddings: Dict[str, np.ndarray],
    train_gt: Dict[str, Set[str]],
    k: int = K_DEFAULT,
    boost_drugs: Set[str] = None,
    boost_factor: float = 1.5
) -> List[Tuple[str, float]]:
    """
    kNN prediction with optional drug boosting.

    Returns list of (drug_id, score) sorted by score descending.
    """
    if disease_id not in embeddings:
        return []

    # Get embeddings for train diseases
    train_diseases = [d for d in train_gt if d in embeddings]
    train_embs = np.array([embeddings[d] for d in train_diseases], dtype=np.float32)

    # Compute similarities
    test_emb = embeddings[disease_id].reshape(1, -1)
    sims = cosine_similarity(test_emb, train_embs)[0]

    # Get k nearest
    top_k_idx = np.argsort(sims)[-k:]

    # Aggregate drug scores
    drug_scores: Dict[str, float] = defaultdict(float)
    for idx in top_k_idx:
        neighbor = train_diseases[idx]
        sim = sims[idx]
        for drug in train_gt[neighbor]:
            if drug in embeddings:
                drug_scores[drug] += sim

    # Apply boost if specified
    if boost_drugs:
        for drug in boost_drugs:
            if drug in drug_scores:
                drug_scores[drug] *= boost_factor

    # Sort by score
    return sorted(drug_scores.items(), key=lambda x: x[1], reverse=True)


def evaluate(
    method: str,
    gt: Dict[str, Set[str]],
    embeddings: Dict[str, np.ndarray],
    disease_categories: Dict[str, str],
    drug_categories: Dict[str, Set[str]],
    seed: int = 42,
    top_k: int = 30
) -> Tuple[float, int, int]:
    """
    Evaluate kNN with or without cross-category boost.

    Returns (R@30, hits, total).
    """
    np.random.seed(seed)
    diseases = list(gt.keys())
    np.random.shuffle(diseases)
    n_test = len(diseases) // 5
    test_diseases = set(diseases[:n_test])
    train_diseases = set(diseases[n_test:])

    train_gt = {d: gt[d] for d in train_diseases}

    # Rebuild drug_categories for train set only
    train_drug_cats = build_drug_categories(train_gt, disease_categories)

    hits = 0
    total = 0

    for disease_id in test_diseases:
        if disease_id not in embeddings:
            continue
        gt_drugs = {d for d in gt[disease_id] if d in embeddings}
        if not gt_drugs:
            continue

        # Determine boost drugs based on method
        if method == "baseline":
            boost_drugs = None
        elif method == "cross_category":
            # Get category of test disease
            test_cat = disease_categories.get(disease_id)
            if test_cat:
                related = RELATED_CATEGORIES.get(test_cat, [])
                # Boost drugs that treat related categories
                boost_drugs = set()
                for drug_id, drug_cats in train_drug_cats.items():
                    if drug_cats & set(related):
                        boost_drugs.add(drug_id)
            else:
                boost_drugs = None
        else:
            raise ValueError(f"Unknown method: {method}")

        # Get predictions
        predictions = knn_predict(
            disease_id, embeddings, train_gt,
            boost_drugs=boost_drugs, boost_factor=1.5
        )

        top_drugs = set(d for d, _ in predictions[:top_k])

        for gt_drug in gt_drugs:
            total += 1
            if gt_drug in top_drugs:
                hits += 1

    r30 = hits / total * 100 if total > 0 else 0.0
    return r30, hits, total


def main():
    print("h181: Drug-Level Cross-Category Transfer")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    embeddings = load_node2vec_embeddings()
    name_to_drug_id, id_to_name = load_drugbank_lookup()
    gt, disease_names, disease_categories = load_ground_truth(name_to_drug_id)
    drug_categories = build_drug_categories(gt, disease_categories)

    print(f"  Embeddings: {len(embeddings)}")
    print(f"  Ground truth diseases: {len(gt)}")
    print(f"  Diseases with categories: {len(disease_categories)}")
    print(f"  Drugs with category info: {len(drug_categories)}")

    # Drug category breadth
    breadth_counts: Dict[int, int] = defaultdict(int)
    for drug, cats in drug_categories.items():
        breadth_counts[len(cats)] += 1

    print("\n  Drug category breadth:")
    for b in sorted(breadth_counts.keys()):
        print(f"    {b} categories: {breadth_counts[b]} drugs")

    # Evaluate
    print("\n" + "=" * 70)
    print("Evaluating methods (5-seed mean)")
    print("=" * 70)

    methods = ["baseline", "cross_category"]
    results = {m: [] for m in methods}

    for seed in SEEDS:
        print(f"\n  Seed {seed}:")
        for method in methods:
            r30, hits, total = evaluate(
                method, gt, embeddings, disease_categories, drug_categories, seed=seed
            )
            results[method].append(r30)
            print(f"    {method}: {r30:.2f}% ({hits}/{total})")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for method in methods:
        mean = np.mean(results[method])
        std = np.std(results[method])
        print(f"\n{method}:")
        print(f"  R@30: {mean:.2f}% ± {std:.2f}%")

    # Improvement
    base_mean = np.mean(results["baseline"])
    boost_mean = np.mean(results["cross_category"])
    improvement = boost_mean - base_mean

    print(f"\nImprovement (cross_category - baseline): {improvement:+.2f} pp")

    # Save results
    output = {
        "hypothesis": "h181",
        "baseline_r30_mean": float(base_mean),
        "baseline_r30_std": float(np.std(results["baseline"])),
        "cross_category_r30_mean": float(boost_mean),
        "cross_category_r30_std": float(np.std(results["cross_category"])),
        "improvement_pp": float(improvement),
        "success": bool(improvement > 1.0),
        "per_seed_baseline": results["baseline"],
        "per_seed_cross_category": results["cross_category"],
    }

    output_file = ANALYSIS_DIR / "h181_drug_cross_category_boost.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    if improvement > 1.0:
        print("\n✓ SUCCESS: Cross-category boost improved R@30 by >1pp")
    elif improvement > 0:
        print(f"\n~ MARGINAL: Cross-category boost improved R@30 by {improvement:.2f}pp (below 1pp threshold)")
    else:
        print("\n✗ FAIL: Cross-category boost did not improve R@30")


if __name__ == "__main__":
    main()
