#!/usr/bin/env python3
"""
h105: Confidence Feature - Disease Coverage Strength

PURPOSE:
    kNN works by finding similar training diseases. If a test disease has
    MANY similar training diseases (high coverage), predictions should be
    more reliable. Quantify "coverage strength" as a confidence signal.

APPROACH:
    1. For each test disease, compute similarity to all training diseases
    2. Define coverage_strength = mean(top-k similarities)
    3. Stratify predictions by coverage strength (high/medium/low)
    4. Compare precision across strata

SUCCESS CRITERIA:
    High-coverage diseases have 10%+ better precision than low-coverage.
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

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"

SEEDS = [42, 123, 456, 789, 1024]


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


def load_drugbank_lookup() -> Dict[str, str]:
    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)
    name_to_id = {name.lower(): f"drkg:Compound::{db_id}" for db_id, name in id_to_name.items()}
    return name_to_id


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


def load_ground_truth(
    mesh_mappings: Dict[str, str],
    name_to_drug_id: Dict[str, str],
) -> Dict[str, Set[str]]:
    df = pd.read_excel(REFERENCE_DIR / "everycure" / "indicationList.xlsx")
    fuzzy_mappings = load_mesh_mappings()
    matcher = DiseaseMatcher(fuzzy_mappings)

    gt: Dict[str, Set[str]] = defaultdict(set)
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
            gt[disease_id].add(drug_id)

    return dict(gt)


def run_knn_with_coverage_analysis(
    emb_dict: Dict[str, np.ndarray],
    train_gt: Dict[str, Set[str]],
    test_gt: Dict[str, Set[str]],
    k: int = 20,
) -> List[Dict]:
    """Run kNN and return per-disease results with coverage metrics."""
    train_disease_list = [d for d in train_gt if d in emb_dict]
    train_disease_embs = np.array([emb_dict[d] for d in train_disease_list], dtype=np.float32)

    results = []

    for disease_id in test_gt:
        if disease_id not in emb_dict:
            continue
        gt_drugs = {d for d in test_gt[disease_id] if d in emb_dict}
        if not gt_drugs:
            continue

        # Compute similarities to all training diseases
        test_emb = emb_dict[disease_id].reshape(1, -1)
        sims = cosine_similarity(test_emb, train_disease_embs)[0]

        # Coverage strength metrics
        top_k_sims = np.sort(sims)[-k:]
        coverage_strength = float(np.mean(top_k_sims))  # mean similarity of k-nearest
        max_sim = float(np.max(sims))
        count_high_sim = int(np.sum(sims > 0.9))  # diseases with >0.9 similarity

        # Get top-k disease indices
        top_k_disease_idx = np.argsort(sims)[-k:]

        # Run kNN: count drug frequency among nearest diseases
        drug_counts: Dict[str, float] = defaultdict(float)
        for idx in top_k_disease_idx:
            neighbor_disease = train_disease_list[idx]
            neighbor_sim = sims[idx]
            for drug_id in train_gt[neighbor_disease]:
                if drug_id in emb_dict:
                    drug_counts[drug_id] += neighbor_sim

        # Get top 30 predictions
        if not drug_counts:
            continue
        sorted_drugs = sorted(drug_counts.items(), key=lambda x: x[1], reverse=True)
        top_30 = set(d for d, _ in sorted_drugs[:30])

        # Calculate hit rate
        hits = len(top_30 & gt_drugs)
        recall = hits / len(gt_drugs)

        results.append({
            'disease_id': disease_id,
            'coverage_strength': coverage_strength,
            'max_sim': max_sim,
            'count_high_sim': count_high_sim,
            'hits': hits,
            'total_gt': len(gt_drugs),
            'recall': recall,
        })

    return results


def analyze_coverage_precision(results: List[Dict]) -> Dict:
    """Analyze precision by coverage strength strata."""
    # Sort by coverage strength
    sorted_results = sorted(results, key=lambda x: x['coverage_strength'])

    # Split into tertiles (low/medium/high coverage)
    n = len(sorted_results)
    low = sorted_results[:n//3]
    medium = sorted_results[n//3:2*n//3]
    high = sorted_results[2*n//3:]

    def calc_precision(group):
        total_hits = sum(r['hits'] for r in group)
        total_preds = len(group) * 30  # 30 predictions per disease
        return total_hits / total_preds if total_preds > 0 else 0

    def calc_recall(group):
        total_hits = sum(r['hits'] for r in group)
        total_gt = sum(r['total_gt'] for r in group)
        return total_hits / total_gt if total_gt > 0 else 0

    return {
        'low': {
            'count': len(low),
            'mean_coverage': np.mean([r['coverage_strength'] for r in low]) if low else 0,
            'precision': calc_precision(low),
            'recall': calc_recall(low),
        },
        'medium': {
            'count': len(medium),
            'mean_coverage': np.mean([r['coverage_strength'] for r in medium]) if medium else 0,
            'precision': calc_precision(medium),
            'recall': calc_recall(medium),
        },
        'high': {
            'count': len(high),
            'mean_coverage': np.mean([r['coverage_strength'] for r in high]) if high else 0,
            'precision': calc_precision(high),
            'recall': calc_recall(high),
        },
    }


def main():
    print("h105: Disease Coverage Strength as Confidence Feature")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    emb_dict = load_node2vec_embeddings()
    name_to_drug_id = load_drugbank_lookup()
    mesh_mappings = load_mesh_mappings_from_file()
    ground_truth = load_ground_truth(mesh_mappings, name_to_drug_id)

    print(f"  Embeddings: {len(emb_dict)}")
    print(f"  Ground truth diseases: {len(ground_truth)}")

    # Multi-seed evaluation
    print("\n" + "=" * 70)
    print("Multi-Seed Evaluation (5 seeds)")
    print("=" * 70)

    all_results = []
    all_strata_analysis = []

    for seed in SEEDS:
        np.random.seed(seed)
        diseases = list(ground_truth.keys())
        np.random.shuffle(diseases)
        n_test = len(diseases) // 5
        test_diseases = set(diseases[:n_test])
        train_diseases = set(diseases[n_test:])

        train_gt = {d: ground_truth[d] for d in train_diseases}
        test_gt = {d: ground_truth[d] for d in test_diseases}

        seed_results = run_knn_with_coverage_analysis(emb_dict, train_gt, test_gt, k=20)
        all_results.extend(seed_results)

        strata = analyze_coverage_precision(seed_results)
        all_strata_analysis.append(strata)

    # Aggregate across seeds
    print("\nAggregate Strata Analysis (5 seeds):")
    print("-" * 60)

    for stratum in ['low', 'medium', 'high']:
        counts = [s[stratum]['count'] for s in all_strata_analysis]
        precisions = [s[stratum]['precision'] for s in all_strata_analysis]
        recalls = [s[stratum]['recall'] for s in all_strata_analysis]
        coverages = [s[stratum]['mean_coverage'] for s in all_strata_analysis]

        print(f"\n{stratum.upper()} Coverage:")
        print(f"  Diseases: {sum(counts)} total ({np.mean(counts):.0f} avg per seed)")
        print(f"  Mean coverage strength: {np.mean(coverages):.3f}")
        print(f"  Precision: {100*np.mean(precisions):.2f}% ± {100*np.std(precisions):.2f}%")
        print(f"  Recall@30: {100*np.mean(recalls):.2f}% ± {100*np.std(recalls):.2f}%")

    # Compare high vs low
    print("\n" + "=" * 70)
    print("KEY COMPARISON")
    print("=" * 70)
    high_prec = np.mean([s['high']['precision'] for s in all_strata_analysis])
    low_prec = np.mean([s['low']['precision'] for s in all_strata_analysis])
    diff = high_prec - low_prec

    print(f"  HIGH coverage precision: {100*high_prec:.2f}%")
    print(f"  LOW coverage precision:  {100*low_prec:.2f}%")
    print(f"  Difference: {100*diff:+.2f} pp")

    # Success criteria
    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 70)
    success = diff >= 0.10  # 10 pp threshold
    if success:
        print(f"  ✓ High-coverage precision is +{100*diff:.1f} pp better (>= 10 pp)")
        print("  → VALIDATED: Coverage strength is a valid confidence signal")
    else:
        print(f"  ✗ High-coverage precision is only +{100*diff:.1f} pp better (< 10 pp)")
        if diff > 0:
            print("  → PARTIALLY VALIDATED: Coverage strength helps but below threshold")
        else:
            print("  → INVALIDATED: Coverage strength doesn't improve precision")

    # Correlation analysis
    coverages = [r['coverage_strength'] for r in all_results]
    recalls = [r['recall'] for r in all_results]
    correlation = np.corrcoef(coverages, recalls)[0, 1]
    print(f"\n  Correlation(coverage, recall): {correlation:.3f}")

    # Save results
    results_file = PROJECT_ROOT / "data" / "analysis" / "h105_coverage_strength.json"
    with open(results_file, 'w') as f:
        json.dump({
            'high_precision': float(high_prec),
            'low_precision': float(low_prec),
            'difference_pp': float(diff * 100),
            'correlation': float(correlation),
            'success': bool(success),
            'strata_summary': {
                'high_mean_coverage': float(np.mean([s['high']['mean_coverage'] for s in all_strata_analysis])),
                'medium_mean_coverage': float(np.mean([s['medium']['mean_coverage'] for s in all_strata_analysis])),
                'low_mean_coverage': float(np.mean([s['low']['mean_coverage'] for s in all_strata_analysis])),
            }
        }, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
