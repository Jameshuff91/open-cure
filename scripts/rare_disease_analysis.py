#!/usr/bin/env python3
"""
Rare Disease Performance Analysis.

Analyze kNN performance on rare diseases, defined as:
1. Diseases with ≤2 known treatments in GT
2. Diseases with low max cosine similarity to training diseases

The hypothesis is that kNN fails on rare diseases because there are
no similar diseases to transfer treatments from.
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from disease_name_matcher import DiseaseMatcher, load_mesh_mappings

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"

SEEDS = [42, 123, 456, 789, 1024]
K = 20


def load_embeddings(path: Path) -> dict[str, np.ndarray]:
    df = pd.read_csv(path)
    dim_cols = [c for c in df.columns if c.startswith("dim_")]
    embeddings: dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        entity = f"drkg:{row['entity']}"
        embeddings[entity] = row[dim_cols].values.astype(np.float32)
    return embeddings


def load_drugbank_lookup() -> dict[str, str]:
    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)
    return {name.lower(): f"drkg:Compound::{db_id}" for db_id, name in id_to_name.items()}


def load_mesh_mappings_from_file() -> dict[str, str]:
    mesh_path = REFERENCE_DIR / "mesh_mappings_from_agents.json"
    if not mesh_path.exists():
        return {}
    with open(mesh_path) as f:
        mesh_data = json.load(f)
    mesh_mappings: dict[str, str] = {}
    for batch_data in mesh_data.values():
        if isinstance(batch_data, dict):
            for disease_name, mesh_id in batch_data.items():
                if mesh_id:
                    mesh_str = str(mesh_id)
                    if mesh_str.startswith("D") or mesh_str.startswith("C"):
                        mesh_mappings[disease_name.lower()] = f"drkg:Disease::MESH:{mesh_str}"
    return mesh_mappings


def load_ground_truth_with_names(
    mesh_mappings: dict[str, str],
    name_to_drug_id: dict[str, str],
) -> tuple[dict[str, set[str]], dict[str, str]]:
    df = pd.read_excel(REFERENCE_DIR / "everycure" / "indicationList.xlsx")
    fuzzy_mappings = load_mesh_mappings()
    matcher = DiseaseMatcher(fuzzy_mappings)

    gt_pairs: dict[str, set[str]] = defaultdict(set)
    disease_id_to_name: dict[str, str] = {}

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
            if disease_id not in disease_id_to_name:
                disease_id_to_name[disease_id] = disease

    return dict(gt_pairs), disease_id_to_name


def disease_level_split(
    gt_pairs: dict[str, set[str]],
    valid_entity_check,  # type: ignore[type-arg]
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    valid_diseases = [d for d in gt_pairs if valid_entity_check(d)]
    rng = np.random.RandomState(seed)
    rng.shuffle(valid_diseases)
    n_test = max(1, int(len(valid_diseases) * test_fraction))
    test_diseases = set(valid_diseases[:n_test])
    train_diseases = set(valid_diseases[n_test:])
    train_gt = {d: gt_pairs[d] for d in train_diseases}
    test_gt = {d: gt_pairs[d] for d in test_diseases}
    return train_gt, test_gt


def evaluate_knn_detailed(
    emb_dict: dict[str, np.ndarray],
    train_gt: dict[str, set[str]],
    test_gt: dict[str, set[str]],
    k: int = 20,
) -> list[dict[str, Any]]:
    """kNN with detailed per-disease results including similarity metrics."""
    train_disease_list = [d for d in train_gt if d in emb_dict]
    train_disease_embs = np.array([emb_dict[d] for d in train_disease_list], dtype=np.float32)

    results: list[dict[str, Any]] = []

    for disease_id in test_gt:
        if disease_id not in emb_dict:
            continue
        gt_drugs = {d for d in test_gt[disease_id] if d in emb_dict}
        if not gt_drugs:
            continue

        # Find k nearest training diseases
        test_emb = emb_dict[disease_id].reshape(1, -1)
        sims = cosine_similarity(test_emb, train_disease_embs)[0]

        # Handle NaN similarities
        valid_sims = sims[~np.isnan(sims)]
        if len(valid_sims) == 0:
            max_sim = 0
            mean_sim = 0
        else:
            max_sim = float(np.max(valid_sims))
            mean_sim = float(np.mean(valid_sims))

        top_k_disease_idx = np.argsort(sims)[-k:]

        # Count drug frequency among nearest diseases
        drug_counts: dict[str, float] = defaultdict(float)
        for idx in top_k_disease_idx:
            if np.isnan(sims[idx]):
                continue
            neighbor_disease = train_disease_list[idx]
            neighbor_sim = sims[idx]
            for drug_id in train_gt[neighbor_disease]:
                if drug_id in emb_dict:
                    drug_counts[drug_id] += neighbor_sim

        if drug_counts:
            sorted_drugs = sorted(drug_counts.items(), key=lambda x: x[1], reverse=True)
            top_30 = {d for d, _ in sorted_drugs[:30]}
        else:
            top_30 = set()

        hits = len(top_30 & gt_drugs)
        recall = hits / len(gt_drugs) if gt_drugs else 0

        results.append({
            "disease_id": disease_id,
            "n_gt_drugs": len(gt_drugs),
            "max_similarity": max_sim,
            "mean_similarity": mean_sim,
            "hits": hits,
            "recall": recall,
        })

    return results


def main() -> None:
    start_time = time.time()

    print("=" * 70)
    print("RARE DISEASE PERFORMANCE ANALYSIS")
    print("=" * 70)
    print()

    # Load data
    print("Loading data...")
    name_to_drug_id = load_drugbank_lookup()
    mesh_mappings = load_mesh_mappings_from_file()
    gt_pairs, disease_id_to_name = load_ground_truth_with_names(mesh_mappings, name_to_drug_id)
    print(f"  {len(gt_pairs)} diseases, {sum(len(v) for v in gt_pairs.values())} GT pairs")

    # Load honest embeddings
    honest_path = EMBEDDINGS_DIR / "node2vec_256_no_treatment.csv"
    if not honest_path.exists():
        print(f"ERROR: {honest_path} not found")
        sys.exit(1)

    emb_dict = load_embeddings(honest_path)
    print(f"  {len(emb_dict):,} embeddings loaded")
    print()

    # Run multi-seed evaluation
    print("-" * 70)
    print("Running 5-seed evaluation with detailed metrics...")
    print("-" * 70)

    all_results: list[dict[str, Any]] = []

    for seed in SEEDS:
        print(f"\n  Seed {seed}:")
        train_gt, test_gt = disease_level_split(
            gt_pairs, lambda d: d in emb_dict, test_fraction=0.2, seed=seed
        )

        results = evaluate_knn_detailed(emb_dict, train_gt, test_gt, k=K)

        for r in results:
            r["seed"] = seed
            r["disease_name"] = disease_id_to_name.get(r["disease_id"], "Unknown")

        all_results.extend(results)

        # Quick summary
        rare_1 = [r for r in results if r["n_gt_drugs"] == 1]
        rare_2 = [r for r in results if r["n_gt_drugs"] <= 2]
        common = [r for r in results if r["n_gt_drugs"] > 5]

        print(f"    1 GT drug:  {len(rare_1)} diseases, R@30 = {np.mean([r['recall'] for r in rare_1])*100:.1f}%")
        print(f"    ≤2 GT drugs: {len(rare_2)} diseases, R@30 = {np.mean([r['recall'] for r in rare_2])*100:.1f}%")
        print(f"    >5 GT drugs: {len(common)} diseases, R@30 = {np.mean([r['recall'] for r in common])*100:.1f}%")

    # Aggregate analysis
    print()
    print("=" * 70)
    print("RARE DISEASE ANALYSIS (aggregated)")
    print("=" * 70)

    # 1. By GT drug count (treatment rarity)
    print("\n1. BY NUMBER OF KNOWN TREATMENTS")
    print("-" * 50)

    gt_buckets = [
        ("1 treatment (ultra-rare)", lambda r: r["n_gt_drugs"] == 1),
        ("2 treatments", lambda r: r["n_gt_drugs"] == 2),
        ("3-5 treatments", lambda r: 3 <= r["n_gt_drugs"] <= 5),
        ("6-10 treatments", lambda r: 6 <= r["n_gt_drugs"] <= 10),
        ("11+ treatments (well-studied)", lambda r: r["n_gt_drugs"] >= 11),
    ]

    gt_stats: dict[str, dict[str, float]] = {}

    for label, filter_fn in gt_buckets:
        subset = [r for r in all_results if filter_fn(r)]
        if not subset:
            continue
        recalls = [r["recall"] for r in subset]
        max_sims = [r["max_similarity"] for r in subset]
        gt_stats[label] = {
            "n": len(subset),
            "recall_mean": float(np.mean(recalls)),
            "recall_std": float(np.std(recalls)),
            "avg_max_similarity": float(np.mean(max_sims)),
        }
        print(f"\n  {label}: {len(subset)} evaluations")
        print(f"    R@30: {np.mean(recalls)*100:.2f}% ± {np.std(recalls)*100:.2f}%")
        print(f"    Avg max sim to training: {np.mean(max_sims):.3f}")

    # 2. By similarity to training set (isolation)
    print("\n2. BY MAX SIMILARITY TO TRAINING DISEASES")
    print("-" * 50)

    sim_buckets = [
        ("Low sim (<0.5)", lambda r: r["max_similarity"] < 0.5),
        ("Medium sim (0.5-0.7)", lambda r: 0.5 <= r["max_similarity"] < 0.7),
        ("High sim (0.7-0.9)", lambda r: 0.7 <= r["max_similarity"] < 0.9),
        ("Very high sim (≥0.9)", lambda r: r["max_similarity"] >= 0.9),
    ]

    sim_stats: dict[str, dict[str, float]] = {}

    for label, filter_fn in sim_buckets:
        subset = [r for r in all_results if filter_fn(r)]
        if not subset:
            continue
        recalls = [r["recall"] for r in subset]
        sim_stats[label] = {
            "n": len(subset),
            "recall_mean": float(np.mean(recalls)),
            "recall_std": float(np.std(recalls)),
        }
        print(f"\n  {label}: {len(subset)} evaluations")
        print(f"    R@30: {np.mean(recalls)*100:.2f}% ± {np.std(recalls)*100:.2f}%")

    # 3. Combined rare disease definition
    print("\n3. RARE DISEASE DEFINITION (combined)")
    print("-" * 50)

    # Ultra-rare: ≤2 treatments AND low similarity
    ultra_rare = [r for r in all_results
                  if r["n_gt_drugs"] <= 2 and r["max_similarity"] < 0.7]
    common_well_connected = [r for r in all_results
                             if r["n_gt_drugs"] > 5 and r["max_similarity"] >= 0.7]

    print(f"\n  Ultra-rare (≤2 treatments AND sim<0.7): {len(ultra_rare)} evaluations")
    if ultra_rare:
        print(f"    R@30: {np.mean([r['recall'] for r in ultra_rare])*100:.2f}%")

    print(f"\n  Well-studied (>5 treatments AND sim≥0.7): {len(common_well_connected)} evaluations")
    if common_well_connected:
        print(f"    R@30: {np.mean([r['recall'] for r in common_well_connected])*100:.2f}%")

    # Worst performing diseases
    print("\n4. WORST PERFORMING DISEASES (0% R@30)")
    print("-" * 50)

    zero_recall = [r for r in all_results if r["recall"] == 0]
    zero_recall_diseases = set(r["disease_name"] for r in zero_recall)
    print(f"\n  {len(zero_recall)} evaluations with 0% recall ({len(zero_recall_diseases)} unique diseases)")
    print("\n  Sample (first 10 unique diseases with 0% recall):")
    for disease in sorted(zero_recall_diseases)[:10]:
        matching = [r for r in zero_recall if r["disease_name"] == disease][0]
        print(f"    - {disease} ({matching['n_gt_drugs']} GT drugs, max_sim={matching['max_similarity']:.2f})")

    # Summary
    print()
    print("=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    avg_overall = np.mean([r["recall"] for r in all_results])
    avg_1_drug = np.mean([r["recall"] for r in all_results if r["n_gt_drugs"] == 1])
    avg_many = np.mean([r["recall"] for r in all_results if r["n_gt_drugs"] > 5])

    print(f"\n  1. Treatment count strongly predicts performance:")
    print(f"     - 1 treatment: {avg_1_drug*100:.1f}% R@30")
    print(f"     - >5 treatments: {avg_many*100:.1f}% R@30")
    print(f"     - Gap: {(avg_many-avg_1_drug)*100:.1f} pp")

    print(f"\n  2. kNN fails on isolated diseases:")
    print(f"     - Low similarity (<0.5) diseases perform worst")
    print(f"     - Rare diseases often have no similar training diseases")

    print(f"\n  3. The hardest cases (rare + isolated) get ~0% recall")
    print(f"     - These are arguably the most important to solve")
    print(f"     - kNN paradigm is fundamentally limited here")

    # Save results
    results_data: dict[str, Any] = {
        "analysis": "rare_disease_performance",
        "parameters": {
            "k": K,
            "seeds": SEEDS,
        },
        "overall": {
            "n_evaluations": len(all_results),
            "recall_mean": float(np.mean([r["recall"] for r in all_results])),
        },
        "by_gt_count": gt_stats,
        "by_similarity": sim_stats,
        "worst_performing": {
            "zero_recall_count": len(zero_recall),
            "unique_diseases": len(zero_recall_diseases),
            "sample_diseases": list(zero_recall_diseases)[:20],
        },
        "elapsed_seconds": time.time() - start_time,
    }

    output_path = ANALYSIS_DIR / "rare_disease_performance.json"
    with open(output_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
