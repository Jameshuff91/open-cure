#!/usr/bin/env python3
"""
Hypothesis h66: Disease Category-Specific k Values.

PURPOSE:
    Different disease categories may benefit from different k values in kNN.
    - Rare diseases (GI, metabolic storage): Fewer relevant neighbors
    - Common diseases (autoimmune): Many relevant neighbors

EXPERIMENT:
    For each category, test k in [5, 10, 15, 20, 30, 50]
    Find optimal k per category
    Compare category-specific k vs global k=20

SUCCESS CRITERIA:
    >2 pp improvement on at least 3 categories
"""

import json
import sys
import time
from pathlib import Path
from collections import defaultdict
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
K_VALUES = [5, 10, 15, 20, 30, 50]

CATEGORY_KEYWORDS = {
    'autoimmune': ['rheumat', 'lupus', 'sclerosis', 'arthritis', 'crohn', 'colitis', 'psoria'],
    'cancer': ['cancer', 'carcinoma', 'lymphoma', 'leukemia', 'melanoma', 'tumor', 'neoplasm'],
    'infectious': ['infection', 'viral', 'bacterial', 'fungal', 'malaria', 'tuberculosis', 'hiv', 'hepatitis'],
    'metabolic': ['diabetes', 'obesity', 'hyperlipid', 'cholesterol', 'metabolic'],
    'cardiovascular': ['heart', 'cardiac', 'hypertension', 'coronary', 'vascular', 'stroke'],
    'neurological': ['parkinson', 'alzheimer', 'epilepsy', 'migraine', 'neuropath'],
    'respiratory': ['asthma', 'copd', 'pneumonia', 'bronch', 'pulmonary'],
    'dermatological': ['skin', 'dermat', 'eczema', 'acne', 'rash'],
    'psychiatric': ['depression', 'anxiety', 'bipolar', 'schizo', 'psychiatric'],
    'gastrointestinal': ['gastric', 'bowel', 'intestin', 'crohn', 'liver', 'hepat', 'esophag', 'colitis'],
}


def load_node2vec_embeddings() -> Dict[str, np.ndarray]:
    df = pd.read_csv(EMBEDDINGS_DIR / "node2vec_256_named.csv")
    dim_cols = [c for c in df.columns if c.startswith("dim_")]
    embeddings = {}
    for _, row in df.iterrows():
        entity = f"drkg:{row['entity']}"
        embeddings[entity] = row[dim_cols].values.astype(np.float32)
    return embeddings


def load_drugbank_lookup():
    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)
    name_to_id = {name.lower(): f"drkg:Compound::{db_id}" for db_id, name in id_to_name.items()}
    return name_to_id


def load_mesh_mappings_from_file():
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


def load_ground_truth(mesh_mappings, name_to_drug_id):
    df = pd.read_excel(REFERENCE_DIR / "everycure" / "indicationList.xlsx")
    fuzzy_mappings = load_mesh_mappings()
    matcher = DiseaseMatcher(fuzzy_mappings)

    gt_pairs = defaultdict(set)
    disease_names = {}

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
            disease_names[disease_id] = disease

    return dict(gt_pairs), disease_names


def categorize_disease(name: str) -> str:
    name_lower = name.lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in name_lower for kw in keywords):
            return cat
    return 'other'


def disease_level_split(gt_pairs, valid_entity_check, test_fraction=0.2, seed=42):
    valid_diseases = [d for d in gt_pairs if valid_entity_check(d)]
    rng = np.random.RandomState(seed)
    rng.shuffle(valid_diseases)
    n_test = max(1, int(len(valid_diseases) * test_fraction))
    test_diseases = set(valid_diseases[:n_test])
    train_diseases = set(valid_diseases[n_test:])
    train_gt = {d: gt_pairs[d] for d in train_diseases}
    test_gt = {d: gt_pairs[d] for d in test_diseases}
    return train_gt, test_gt


def knn_hit_at_30(
    disease_id: str,
    emb_dict: Dict[str, np.ndarray],
    train_gt: Dict[str, Set[str]],
    train_disease_list: List[str],
    train_disease_embs: np.ndarray,
    gt_drugs: Set[str],
    k: int,
) -> bool:
    """Check if kNN achieves hit@30 for a disease."""
    test_emb = emb_dict[disease_id].reshape(1, -1)
    sims = cosine_similarity(test_emb, train_disease_embs)[0]
    top_k_idx = np.argsort(sims)[-k:]

    drug_counts = defaultdict(float)
    for idx in top_k_idx:
        neighbor_disease = train_disease_list[idx]
        neighbor_sim = sims[idx]
        for drug_id in train_gt.get(neighbor_disease, set()):
            if drug_id in emb_dict:
                drug_counts[drug_id] += neighbor_sim

    if not drug_counts:
        return False

    sorted_drugs = sorted(drug_counts.items(), key=lambda x: x[1], reverse=True)
    top_30 = {d for d, _ in sorted_drugs[:30]}
    return len(top_30 & gt_drugs) > 0


def main():
    start_time = time.time()

    print("=" * 70)
    print("h66: DISEASE CATEGORY-SPECIFIC k VALUES")
    print("=" * 70)
    print()

    # Load data
    print("Loading data...")
    emb_dict = load_node2vec_embeddings()
    name_to_drug_id = load_drugbank_lookup()
    mesh_mappings = load_mesh_mappings_from_file()
    gt_pairs, disease_names = load_ground_truth(mesh_mappings, name_to_drug_id)
    print(f"  GT: {len(gt_pairs)} diseases")

    # Categorize all diseases
    disease_categories = {d: categorize_disease(disease_names.get(d, d)) for d in gt_pairs}

    # Multi-seed evaluation
    category_k_results = defaultdict(lambda: defaultdict(list))

    for seed in SEEDS:
        print(f"\nSeed {seed}...")
        train_gt, test_gt = disease_level_split(gt_pairs, lambda d: d in emb_dict, 0.2, seed)

        train_disease_list = [d for d in train_gt if d in emb_dict]
        train_disease_embs = np.array([emb_dict[d] for d in train_disease_list], dtype=np.float32)

        for disease_id in test_gt:
            if disease_id not in emb_dict:
                continue
            gt_drugs = {d for d in test_gt[disease_id] if d in emb_dict}
            if not gt_drugs:
                continue

            category = disease_categories.get(disease_id, 'other')

            for k in K_VALUES:
                hit = knn_hit_at_30(
                    disease_id, emb_dict, train_gt, train_disease_list, train_disease_embs, gt_drugs, k
                )
                category_k_results[category][k].append(1 if hit else 0)

    # Analyze results
    print("\n" + "=" * 70)
    print("RESULTS BY CATEGORY AND K VALUE")
    print("=" * 70)
    print()
    print(f"{'Category':<18} {'k=5':<10} {'k=10':<10} {'k=15':<10} {'k=20':<10} {'k=30':<10} {'k=50':<10} {'Best k':<10}")
    print("-" * 98)

    category_best_k = {}
    improvements = {}

    for category in sorted(category_k_results.keys()):
        k_hits = {k: np.mean(category_k_results[category][k]) for k in K_VALUES}
        best_k = max(k_hits, key=k_hits.get)
        category_best_k[category] = best_k

        # Calculate improvement over k=20
        k20_hit_rate = k_hits.get(20, 0)
        best_hit_rate = k_hits[best_k]
        improvement = (best_hit_rate - k20_hit_rate) * 100
        improvements[category] = improvement

        row = f"{category:<18}"
        for k in K_VALUES:
            hit_rate = k_hits.get(k, 0) * 100
            row += f"{hit_rate:.1f}%".ljust(10)
        row += f"k={best_k}"
        print(row)

        n_diseases = len(category_k_results[category][20])
        print(f"{'  (n=' + str(n_diseases) + ')':<18}" + " " * 60 + f"({improvement:+.1f} pp vs k=20)")

    print()
    print("=" * 70)
    print("OPTIMAL k BY CATEGORY")
    print("=" * 70)

    # Categories where best k != 20
    changed = {cat: k for cat, k in category_best_k.items() if k != 20}
    print(f"\nCategories where best k ≠ 20: {len(changed)}")
    for cat, k in sorted(changed.items(), key=lambda x: -improvements[x[0]]):
        print(f"  {cat}: k={k} ({improvements[cat]:+.1f} pp vs k=20)")

    # Categories with >2 pp improvement
    significant_improvements = {cat: imp for cat, imp in improvements.items() if imp > 2}
    print(f"\nCategories with >2 pp improvement: {len(significant_improvements)}")
    for cat, imp in sorted(significant_improvements.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {imp:+.1f} pp")

    print()
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if len(significant_improvements) >= 3:
        print(f"""
SUCCESS: {len(significant_improvements)} categories show >2 pp improvement with category-specific k.

RECOMMENDED CATEGORY-SPECIFIC k VALUES:""")
        for cat, k in sorted(category_best_k.items()):
            if k != 20:
                print(f"  {cat}: k={k}")
            else:
                print(f"  {cat}: k=20 (default)")
    else:
        print(f"""
RESULT: Only {len(significant_improvements)} categories show >2 pp improvement.
Category-specific k provides marginal benefit.

RECOMMENDATION: Keep global k=20 for simplicity.
""")

    # Save results
    elapsed = time.time() - start_time
    output = {
        'hypothesis': 'h66',
        'title': 'Disease Category-Specific k Values',
        'date': '2026-01-31',
        'category_results': {
            cat: {str(k): float(np.mean(category_k_results[cat][k]))
                  for k in K_VALUES}
            for cat in category_k_results
        },
        'best_k_per_category': {cat: int(k) for cat, k in category_best_k.items()},
        'improvements_vs_k20': {cat: float(imp) for cat, imp in improvements.items()},
        'significant_improvements': list(significant_improvements.keys()),
        'elapsed_seconds': elapsed,
    }

    output_path = ANALYSIS_DIR / "h66_category_specific_k_results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")
    print(f"Elapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
