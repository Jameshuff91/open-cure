#!/usr/bin/env python3
"""
h170: Category-Aware kNN with same-category weight boost

Hypothesis: Boosting weights for neighbors from the same disease category
could improve precision for isolated categories like neurological.

h168 showed neurological diseases have only ~1.2/20 same-category neighbors
vs 10.4/20 for cancer. This experiment tests whether boosting same-category
neighbor weights improves precision and coverage.
"""

import json
import numpy as np
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from production_predictor import CATEGORY_KEYWORDS

# Disease categories
CATEGORY_KEYWORDS_LOWER = {
    cat: [kw.lower() for kw in keywords]
    for cat, keywords in CATEGORY_KEYWORDS.items()
}


def categorize_disease(disease_name: str) -> str:
    """Categorize disease by name."""
    name_lower = disease_name.lower()
    for category, keywords in CATEGORY_KEYWORDS_LOWER.items():
        for kw in keywords:
            if kw in name_lower:
                return category
    return 'other'


def load_data():
    """Load embeddings, ground truth, and build category mappings."""
    import pandas as pd

    # Load embeddings from CSV (same as production_predictor)
    embeddings_path = PROJECT_ROOT / "data/embeddings/node2vec_256_named.csv"
    df_emb = pd.read_csv(embeddings_path)
    dim_cols = [c for c in df_emb.columns if c.startswith("dim_")]
    embeddings = {}
    for _, row in df_emb.iterrows():
        entity = f"drkg:{row['entity']}"
        embeddings[entity] = row[dim_cols].values.astype(np.float32)

    print(f"Loaded {len(embeddings)} embeddings")

    # Load DrugBank lookup for name->ID mapping
    import json
    with open(PROJECT_ROOT / "data/reference/drugbank_lookup.json") as f:
        id_to_name = json.load(f)
    name_to_drug_id = {
        name.lower(): f"drkg:Compound::{db_id}"
        for db_id, name in id_to_name.items()
    }

    # Load MESH mappings
    mesh_mappings = {}
    mesh_path = PROJECT_ROOT / "data/reference/mesh_mappings_from_agents.json"
    if mesh_path.exists():
        with open(mesh_path) as f:
            mesh_data = json.load(f)
        for batch_data in mesh_data.values():
            if isinstance(batch_data, dict):
                for disease_name, mesh_id in batch_data.items():
                    if mesh_id:
                        mesh_str = str(mesh_id)
                        if mesh_str.startswith("D") or mesh_str.startswith("C"):
                            mesh_mappings[disease_name.lower()] = f"drkg:Disease::MESH:{mesh_str}"

    # Load ground truth using disease name matcher
    from disease_name_matcher import DiseaseMatcher, load_mesh_mappings
    fuzzy_mappings = load_mesh_mappings()
    matcher = DiseaseMatcher(fuzzy_mappings)

    ground_truth = defaultdict(set)
    disease_id_to_name = {}

    # Load excel
    df = pd.read_excel(PROJECT_ROOT / "data/reference/everycure/indicationList.xlsx")
    for _, row in df.iterrows():
        disease = str(row.get("disease name", "")).strip()
        drug = str(row.get("final normalized drug label", "")).strip()
        if not disease or not drug:
            continue

        disease_id = matcher.get_mesh_id(disease)
        if not disease_id:
            disease_id = mesh_mappings.get(disease.lower())
        if not disease_id or disease_id not in embeddings:
            continue

        disease_id_to_name[disease_id] = disease
        drug_id = name_to_drug_id.get(drug.lower())
        if drug_id and drug_id in embeddings:
            ground_truth[disease_id].add(drug_id)

    print(f"Loaded ground truth for {len(ground_truth)} diseases")

    # Build drug to name mapping
    drug_id_to_name = {
        f"drkg:Compound::{db_id}": name
        for db_id, name in id_to_name.items()
    }

    return embeddings, dict(ground_truth), drug_id_to_name, disease_id_to_name


def evaluate_category_weighted_knn(
    embeddings: dict,
    ground_truth: dict,
    disease_id_to_name: dict,
    drug_id_to_name: dict,
    k: int = 20,
    alpha: float = 0.0,  # Same-category boost factor
    target_categories: list = None,
    seed: int = 42,
):
    """
    Evaluate category-weighted kNN.

    Args:
        alpha: Boost factor for same-category neighbors (0 = no boost, 1 = 2x weight)

    Returns:
        Dict with per-category metrics
    """
    np.random.seed(seed)

    # Get all diseases with ground truth
    all_diseases = [d for d in ground_truth if d in embeddings]

    # Split into train/test by disease
    np.random.shuffle(all_diseases)
    n_test = int(len(all_diseases) * 0.2)
    test_diseases = all_diseases[:n_test]
    train_diseases = all_diseases[n_test:]

    # Build training embeddings
    train_emb = np.array([embeddings[d] for d in train_diseases])

    # Build category lookup for training diseases
    train_categories = {}
    for d in train_diseases:
        name = disease_id_to_name.get(d, d)
        train_categories[d] = categorize_disease(name)

    # Evaluate by category
    results_by_cat = defaultdict(lambda: {'hits': 0, 'total': 0, 'recalls': [], 'coverages': []})

    for test_disease in test_diseases:
        test_name = disease_id_to_name.get(test_disease, test_disease)
        test_cat = categorize_disease(test_name)

        # Filter by target categories if specified
        if target_categories and test_cat not in target_categories:
            continue

        # Get kNN
        test_emb = embeddings[test_disease].reshape(1, -1)
        sims = cosine_similarity(test_emb, train_emb)[0]

        # Apply category boost
        if alpha > 0:
            boosted_sims = sims.copy()
            for i, train_d in enumerate(train_diseases):
                if train_categories[train_d] == test_cat:
                    boosted_sims[i] *= (1 + alpha)
            top_k_idx = np.argsort(boosted_sims)[-k:]
        else:
            top_k_idx = np.argsort(sims)[-k:]

        # Aggregate drug scores with potentially boosted similarities
        drug_scores = defaultdict(float)
        for idx in top_k_idx:
            neighbor_disease = train_diseases[idx]
            if alpha > 0:
                neighbor_sim = boosted_sims[idx]
            else:
                neighbor_sim = sims[idx]

            for drug_id in ground_truth[neighbor_disease]:
                if drug_id in embeddings:
                    drug_scores[drug_id] += neighbor_sim

        if not drug_scores:
            results_by_cat[test_cat]['total'] += 1
            results_by_cat[test_cat]['recalls'].append(0.0)
            results_by_cat[test_cat]['coverages'].append(0.0)
            continue

        # Get top 30 predictions
        sorted_drugs = sorted(drug_scores.items(), key=lambda x: x[1], reverse=True)
        top_30 = [d for d, _ in sorted_drugs[:30]]

        # Calculate recall@30
        gt_drugs = ground_truth[test_disease]
        hits = len(set(top_30) & gt_drugs)
        recall = hits / len(gt_drugs) if gt_drugs else 0

        # Calculate coverage (how many GT drugs are reachable in kNN pool)
        reachable_drugs = set()
        for idx in top_k_idx:
            neighbor_disease = train_diseases[idx]
            for drug_id in ground_truth[neighbor_disease]:
                if drug_id in embeddings:
                    reachable_drugs.add(drug_id)
        coverage = len(reachable_drugs & gt_drugs) / len(gt_drugs) if gt_drugs else 0

        results_by_cat[test_cat]['hits'] += hits
        results_by_cat[test_cat]['total'] += 1
        results_by_cat[test_cat]['recalls'].append(recall)
        results_by_cat[test_cat]['coverages'].append(coverage)

    return dict(results_by_cat)


def main():
    print("Loading data...")
    embeddings, ground_truth, drug_id_to_name, disease_id_to_name = load_data()
    print(f"Loaded {len(embeddings)} embeddings, {len(ground_truth)} diseases with GT")

    # Test different alpha values
    alphas = [0.0, 0.5, 1.0, 2.0, 3.0]

    # Focus on isolated categories identified in h168
    target_categories = ['neurological', 'respiratory', 'dermatological', 'gastrointestinal']

    # Run 5-seed evaluation
    seeds = [42, 123, 456, 789, 1011]

    print("\n" + "=" * 80)
    print("h170: Category-Weighted kNN Evaluation")
    print("=" * 80)

    # Collect results across seeds
    all_results = {alpha: defaultdict(lambda: {'recalls': [], 'coverages': []}) for alpha in alphas}

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        for alpha in alphas:
            results = evaluate_category_weighted_knn(
                embeddings, ground_truth, disease_id_to_name, drug_id_to_name,
                k=20, alpha=alpha, seed=seed
            )
            for cat, metrics in results.items():
                all_results[alpha][cat]['recalls'].extend(metrics['recalls'])
                all_results[alpha][cat]['coverages'].extend(metrics['coverages'])

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY: R@30 by Category and Alpha (5-seed means)")
    print("=" * 80)

    print(f"\n{'Category':<20} | " + " | ".join([f"α={a:.1f}" for a in alphas]) + " | Δ(best)")
    print("-" * (25 + 12 * len(alphas)))

    for cat in ['neurological', 'respiratory', 'dermatological', 'gastrointestinal', 'autoimmune', 'cancer', 'other']:
        row = f"{cat:<20} | "
        means = []
        for alpha in alphas:
            if cat in all_results[alpha] and all_results[alpha][cat]['recalls']:
                mean_r = np.mean(all_results[alpha][cat]['recalls']) * 100
                means.append(mean_r)
                row += f"{mean_r:>6.1f}% | "
            else:
                means.append(None)
                row += "   N/A | "

        # Calculate delta
        if means[0] is not None:
            best = max([m for m in means if m is not None])
            delta = best - means[0]  # vs baseline (alpha=0)
            row += f"{delta:+.1f}pp"
        print(row)

    # Overall analysis
    print("\n" + "=" * 80)
    print("COVERAGE ANALYSIS (5-seed means)")
    print("=" * 80)

    print(f"\n{'Category':<20} | " + " | ".join([f"α={a:.1f}" for a in alphas]) + " | Δ(best)")
    print("-" * (25 + 12 * len(alphas)))

    for cat in ['neurological', 'respiratory', 'dermatological', 'gastrointestinal']:
        row = f"{cat:<20} | "
        means = []
        for alpha in alphas:
            if cat in all_results[alpha] and all_results[alpha][cat]['coverages']:
                mean_cov = np.mean(all_results[alpha][cat]['coverages']) * 100
                means.append(mean_cov)
                row += f"{mean_cov:>6.1f}% | "
            else:
                means.append(None)
                row += "   N/A | "

        if means[0] is not None:
            best = max([m for m in means if m is not None])
            delta = best - means[0]
            row += f"{delta:+.1f}pp"
        print(row)

    # Per-category sample sizes
    print("\n" + "=" * 80)
    print("SAMPLE SIZES (total across 5 seeds)")
    print("=" * 80)
    for cat in sorted(all_results[0.0].keys()):
        n = len(all_results[0.0][cat]['recalls'])
        print(f"  {cat}: {n} disease-test instances")

    # Statistical analysis for neurological
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS: Neurological Category")
    print("=" * 80)

    if 'neurological' in all_results[0.0]:
        baseline_recalls = all_results[0.0]['neurological']['recalls']

        for alpha in [0.5, 1.0, 2.0, 3.0]:
            if 'neurological' in all_results[alpha]:
                boosted_recalls = all_results[alpha]['neurological']['recalls']

                # Paired t-test (same diseases across seeds)
                from scipy import stats
                if len(baseline_recalls) == len(boosted_recalls):
                    t_stat, p_val = stats.ttest_rel(boosted_recalls, baseline_recalls)
                    mean_baseline = np.mean(baseline_recalls) * 100
                    mean_boosted = np.mean(boosted_recalls) * 100
                    delta = mean_boosted - mean_baseline

                    print(f"\nα={alpha:.1f} vs baseline:")
                    print(f"  Mean R@30: {mean_boosted:.1f}% vs {mean_baseline:.1f}% (Δ={delta:+.1f}pp)")
                    print(f"  Paired t-test: t={t_stat:.3f}, p={p_val:.4f}")
                    if p_val < 0.05:
                        print(f"  *** SIGNIFICANT at p<0.05 ***")

    # Save results
    output = {
        'experiment': 'h170_category_weighted_knn',
        'method': 'Boost same-category neighbor weights by (1+alpha)',
        'k': 20,
        'seeds': seeds,
        'alphas_tested': alphas,
        'results': {}
    }

    for alpha in alphas:
        output['results'][f'alpha_{alpha}'] = {}
        for cat, metrics in all_results[alpha].items():
            output['results'][f'alpha_{alpha}'][cat] = {
                'mean_recall30': float(np.mean(metrics['recalls'])) if metrics['recalls'] else None,
                'std_recall30': float(np.std(metrics['recalls'])) if metrics['recalls'] else None,
                'mean_coverage': float(np.mean(metrics['coverages'])) if metrics['coverages'] else None,
                'n_samples': len(metrics['recalls'])
            }

    with open(PROJECT_ROOT / 'data/analysis/h170_category_weighted_knn.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to data/analysis/h170_category_weighted_knn.json")


if __name__ == "__main__":
    main()
