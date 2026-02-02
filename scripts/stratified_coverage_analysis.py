#!/usr/bin/env python3
"""
Stratified Coverage Analysis.

Split test diseases by coverage status and report R@30 separately:
- Has coverage: ≥1 GT drug appears in kNN pool
- No coverage: 0 GT drugs in kNN pool

Also stratify by:
- Disease category (autoimmune, cancer, metabolic, etc.)
- Number of GT drugs (1, 2-5, 6+)
- Rare vs common diseases
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
    """Load Node2Vec embeddings from CSV."""
    df = pd.read_csv(path)
    dim_cols = [c for c in df.columns if c.startswith("dim_")]
    embeddings: dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        entity = f"drkg:{row['entity']}"
        embeddings[entity] = row[dim_cols].values.astype(np.float32)
    return embeddings


def load_drugbank_lookup() -> tuple[dict[str, str], dict[str, str]]:
    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)
    name_to_id = {name.lower(): f"drkg:Compound::{db_id}" for db_id, name in id_to_name.items()}
    id_to_name_drkg = {f"drkg:Compound::{db_id}": name for db_id, name in id_to_name.items()}
    return name_to_id, id_to_name_drkg


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
    """Load GT and return disease name mapping."""
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


def evaluate_knn_stratified(
    emb_dict: dict[str, np.ndarray],
    train_gt: dict[str, set[str]],
    test_gt: dict[str, set[str]],
    k: int = 20,
) -> dict[str, Any]:
    """kNN with per-disease results and coverage tracking."""
    train_disease_list = [d for d in train_gt if d in emb_dict]
    train_disease_embs = np.array([emb_dict[d] for d in train_disease_list], dtype=np.float32)

    # Collect all drugs in training GT (the "kNN pool")
    train_drug_pool: set[str] = set()
    for drugs in train_gt.values():
        train_drug_pool.update(d for d in drugs if d in emb_dict)

    results: list[dict[str, Any]] = []

    for disease_id in test_gt:
        if disease_id not in emb_dict:
            continue
        gt_drugs = {d for d in test_gt[disease_id] if d in emb_dict}
        if not gt_drugs:
            continue

        # Check coverage: how many GT drugs are in the kNN pool?
        gt_in_pool = gt_drugs & train_drug_pool
        coverage = len(gt_in_pool) / len(gt_drugs) if gt_drugs else 0
        has_coverage = len(gt_in_pool) > 0

        # Find k nearest training diseases
        test_emb = emb_dict[disease_id].reshape(1, -1)
        sims = cosine_similarity(test_emb, train_disease_embs)[0]
        top_k_disease_idx = np.argsort(sims)[-k:]

        # Count drug frequency among nearest diseases (weighted by similarity)
        drug_counts: dict[str, float] = defaultdict(float)
        for idx in top_k_disease_idx:
            neighbor_disease = train_disease_list[idx]
            neighbor_sim = sims[idx]
            for drug_id in train_gt[neighbor_disease]:
                if drug_id in emb_dict:
                    drug_counts[drug_id] += neighbor_sim

        # Rank by weighted count, take top 30
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
            "n_gt_in_pool": len(gt_in_pool),
            "coverage": coverage,
            "has_coverage": has_coverage,
            "hits": hits,
            "recall": recall,
        })

    return {
        "per_disease": results,
        "train_pool_size": len(train_drug_pool),
    }


def categorize_disease(disease_name: str) -> str:
    """Categorize disease by name keywords."""
    name_lower = disease_name.lower()

    if any(kw in name_lower for kw in ["cancer", "carcinoma", "lymphoma", "leukemia", "tumor", "melanoma", "sarcoma", "myeloma"]):
        return "cancer"
    if any(kw in name_lower for kw in ["diabetes", "obesity", "hyperlipidemia", "metabolic"]):
        return "metabolic"
    if any(kw in name_lower for kw in ["arthritis", "lupus", "sclerosis", "crohn", "colitis", "psoriasis", "autoimmune"]):
        return "autoimmune"
    if any(kw in name_lower for kw in ["infection", "hiv", "hepatitis", "tuberculosis", "malaria", "sepsis"]):
        return "infectious"
    if any(kw in name_lower for kw in ["alzheimer", "parkinson", "epilepsy", "dementia", "neuropathy"]):
        return "neurological"
    if any(kw in name_lower for kw in ["heart", "cardiac", "hypertension", "coronary", "atrial"]):
        return "cardiovascular"
    if any(kw in name_lower for kw in ["asthma", "copd", "pulmonary", "respiratory"]):
        return "respiratory"
    if any(kw in name_lower for kw in ["depression", "anxiety", "schizophrenia", "bipolar"]):
        return "psychiatric"
    if any(kw in name_lower for kw in ["dermatitis", "eczema", "acne", "skin"]):
        return "dermatological"

    return "other"


def main() -> None:
    start_time = time.time()

    print("=" * 70)
    print("STRATIFIED COVERAGE ANALYSIS")
    print("=" * 70)
    print()

    # Load data
    print("Loading data...")
    name_to_drug_id, id_to_name = load_drugbank_lookup()
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

    # Run multi-seed evaluation with stratification
    print("-" * 70)
    print("Running 5-seed stratified evaluation...")
    print("-" * 70)

    all_results: list[dict[str, Any]] = []

    for seed in SEEDS:
        print(f"\n  Seed {seed}:")
        train_gt, test_gt = disease_level_split(
            gt_pairs, lambda d: d in emb_dict, test_fraction=0.2, seed=seed
        )

        result = evaluate_knn_stratified(emb_dict, train_gt, test_gt, k=K)

        # Add disease names and categories
        for r in result["per_disease"]:
            r["seed"] = seed
            r["disease_name"] = disease_id_to_name.get(r["disease_id"], "Unknown")
            r["category"] = categorize_disease(r["disease_name"])

        all_results.extend(result["per_disease"])

        # Quick summary
        has_cov = [r for r in result["per_disease"] if r["has_coverage"]]
        no_cov = [r for r in result["per_disease"] if not r["has_coverage"]]
        print(f"    With coverage: {len(has_cov)} diseases, "
              f"R@30 = {np.mean([r['recall'] for r in has_cov])*100:.1f}%")
        print(f"    No coverage: {len(no_cov)} diseases, "
              f"R@30 = {np.mean([r['recall'] for r in no_cov])*100 if no_cov else 0:.1f}%")

    # Aggregate analysis
    print()
    print("=" * 70)
    print("STRATIFIED RESULTS (aggregated across 5 seeds)")
    print("=" * 70)

    # 1. By coverage status
    print("\n1. BY COVERAGE STATUS")
    print("-" * 50)

    has_cov_all = [r for r in all_results if r["has_coverage"]]
    no_cov_all = [r for r in all_results if not r["has_coverage"]]

    has_cov_recall = [r["recall"] for r in has_cov_all]
    no_cov_recall = [r["recall"] for r in no_cov_all]

    print(f"  With coverage (≥1 GT drug in pool): {len(has_cov_all)} disease-evaluations")
    print(f"    R@30: {np.mean(has_cov_recall)*100:.2f}% ± {np.std(has_cov_recall)*100:.2f}%")
    print(f"    Median: {np.median(has_cov_recall)*100:.1f}%")

    print(f"\n  No coverage (0 GT drugs in pool): {len(no_cov_all)} disease-evaluations")
    print(f"    R@30: {np.mean(no_cov_recall)*100:.2f}% ± {np.std(no_cov_recall)*100:.2f}%")
    print(f"    Median: {np.median(no_cov_recall)*100:.1f}%")

    coverage_pct = len(has_cov_all) / len(all_results) * 100 if all_results else 0
    print(f"\n  Coverage rate: {coverage_pct:.1f}% of test diseases have kNN coverage")

    # 2. By number of GT drugs (rare vs common)
    print("\n2. BY GROUND TRUTH SIZE (proxy for disease rarity)")
    print("-" * 50)

    n_gt_buckets = [
        ("1 GT drug (rare)", lambda r: r["n_gt_drugs"] == 1),
        ("2-5 GT drugs", lambda r: 2 <= r["n_gt_drugs"] <= 5),
        ("6+ GT drugs (common)", lambda r: r["n_gt_drugs"] >= 6),
    ]

    for label, filter_fn in n_gt_buckets:
        subset = [r for r in all_results if filter_fn(r)]
        if not subset:
            continue
        recalls = [r["recall"] for r in subset]
        cov_rate = sum(1 for r in subset if r["has_coverage"]) / len(subset) * 100
        print(f"\n  {label}: {len(subset)} disease-evaluations")
        print(f"    R@30: {np.mean(recalls)*100:.2f}% ± {np.std(recalls)*100:.2f}%")
        print(f"    Coverage rate: {cov_rate:.1f}%")

    # 3. By disease category
    print("\n3. BY DISEASE CATEGORY")
    print("-" * 50)

    categories = sorted(set(r["category"] for r in all_results))
    category_stats: dict[str, dict[str, float]] = {}

    for cat in categories:
        subset = [r for r in all_results if r["category"] == cat]
        if len(subset) < 5:  # Skip tiny categories
            continue
        recalls = [r["recall"] for r in subset]
        cov_rate = sum(1 for r in subset if r["has_coverage"]) / len(subset) * 100
        category_stats[cat] = {
            "n": len(subset),
            "recall_mean": float(np.mean(recalls)),
            "recall_std": float(np.std(recalls)),
            "coverage_rate": cov_rate,
        }
        print(f"\n  {cat.capitalize()}: {len(subset)} disease-evaluations")
        print(f"    R@30: {np.mean(recalls)*100:.2f}% ± {np.std(recalls)*100:.2f}%")
        print(f"    Coverage rate: {cov_rate:.1f}%")

    # Overall
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    overall_recall = [r["recall"] for r in all_results]
    print(f"\n  Overall R@30: {np.mean(overall_recall)*100:.2f}% ± {np.std(overall_recall)*100:.2f}%")
    print(f"  With coverage: {np.mean(has_cov_recall)*100:.2f}%")
    print(f"  Without coverage: {np.mean(no_cov_recall)*100:.2f}%")
    print(f"\n  Key insight: {100-coverage_pct:.0f}% of diseases have ZERO coverage in kNN pool,")
    print(f"  contributing {np.mean(no_cov_recall)*100:.1f}% recall. The model is bimodal.")

    # Save results
    results: dict[str, Any] = {
        "analysis": "stratified_coverage",
        "parameters": {
            "k": K,
            "seeds": SEEDS,
        },
        "overall": {
            "n_evaluations": len(all_results),
            "recall_mean": float(np.mean(overall_recall)),
            "recall_std": float(np.std(overall_recall)),
        },
        "by_coverage": {
            "with_coverage": {
                "n": len(has_cov_all),
                "pct": len(has_cov_all) / len(all_results) * 100 if all_results else 0,
                "recall_mean": float(np.mean(has_cov_recall)) if has_cov_recall else 0,
                "recall_std": float(np.std(has_cov_recall)) if has_cov_recall else 0,
            },
            "no_coverage": {
                "n": len(no_cov_all),
                "pct": len(no_cov_all) / len(all_results) * 100 if all_results else 0,
                "recall_mean": float(np.mean(no_cov_recall)) if no_cov_recall else 0,
                "recall_std": float(np.std(no_cov_recall)) if no_cov_recall else 0,
            },
        },
        "by_gt_size": {
            "1_drug": {
                "n": len([r for r in all_results if r["n_gt_drugs"] == 1]),
                "recall_mean": float(np.mean([r["recall"] for r in all_results if r["n_gt_drugs"] == 1])) if [r for r in all_results if r["n_gt_drugs"] == 1] else 0,
            },
            "2_to_5_drugs": {
                "n": len([r for r in all_results if 2 <= r["n_gt_drugs"] <= 5]),
                "recall_mean": float(np.mean([r["recall"] for r in all_results if 2 <= r["n_gt_drugs"] <= 5])) if [r for r in all_results if 2 <= r["n_gt_drugs"] <= 5] else 0,
            },
            "6_plus_drugs": {
                "n": len([r for r in all_results if r["n_gt_drugs"] >= 6]),
                "recall_mean": float(np.mean([r["recall"] for r in all_results if r["n_gt_drugs"] >= 6])) if [r for r in all_results if r["n_gt_drugs"] >= 6] else 0,
            },
        },
        "by_category": category_stats,
        "elapsed_seconds": time.time() - start_time,
    }

    output_path = ANALYSIS_DIR / "stratified_metrics.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
