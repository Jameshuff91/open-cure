#!/usr/bin/env python3
"""
Precision and Cost Analysis.

Calculate precision metrics and expected cost per true positive for the kNN
drug repurposing model using honest (no-treatment) embeddings.

Metrics:
- Precision@K for K = 10, 20, 30, 50
- Expected cost per true positive (assuming $X validation cost)
- Break-even analysis vs random selection
- Disease-level analysis (which diseases have highest/lowest precision)
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

# Cost assumptions
VALIDATION_COST_PER_PREDICTION = 100_000  # $100K per drug candidate validation
K_VALUES = [10, 20, 30, 50, 100]
SEEDS = [42, 123, 456, 789, 1024]


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


def load_ground_truth(
    mesh_mappings: dict[str, str],
    name_to_drug_id: dict[str, str],
) -> dict[str, set[str]]:
    df = pd.read_excel(REFERENCE_DIR / "everycure" / "indicationList.xlsx")
    fuzzy_mappings = load_mesh_mappings()
    matcher = DiseaseMatcher(fuzzy_mappings)

    gt_pairs: dict[str, set[str]] = defaultdict(set)
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
    return dict(gt_pairs)


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


def evaluate_knn_with_precision(
    emb_dict: dict[str, np.ndarray],
    train_gt: dict[str, set[str]],
    test_gt: dict[str, set[str]],
    k_nn: int = 20,
    k_values: list[int] | None = None,
) -> dict[str, Any]:
    """kNN with detailed precision metrics at multiple K values."""
    if k_values is None:
        k_values = [10, 20, 30, 50]

    train_disease_list = [d for d in train_gt if d in emb_dict]
    train_disease_embs = np.array([emb_dict[d] for d in train_disease_list], dtype=np.float32)

    # Track per-disease results
    disease_results: list[dict[str, Any]] = []

    # Aggregate metrics
    precision_at_k: dict[int, list[float]] = {k: [] for k in k_values}
    recall_at_k: dict[int, list[float]] = {k: [] for k in k_values}

    for disease_id in test_gt:
        if disease_id not in emb_dict:
            continue
        gt_drugs = {d for d in test_gt[disease_id] if d in emb_dict}
        if not gt_drugs:
            continue

        # Find k nearest training diseases
        test_emb = emb_dict[disease_id].reshape(1, -1)
        sims = cosine_similarity(test_emb, train_disease_embs)[0]
        top_k_disease_idx = np.argsort(sims)[-k_nn:]

        # Count drug frequency among nearest diseases (weighted by similarity)
        drug_counts: dict[str, float] = defaultdict(float)
        for idx in top_k_disease_idx:
            neighbor_disease = train_disease_list[idx]
            neighbor_sim = sims[idx]
            for drug_id in train_gt[neighbor_disease]:
                if drug_id in emb_dict:
                    drug_counts[drug_id] += neighbor_sim

        # Rank drugs
        if drug_counts:
            sorted_drugs = sorted(drug_counts.items(), key=lambda x: x[1], reverse=True)
            ranked_drugs = [d for d, _ in sorted_drugs]
        else:
            ranked_drugs = []

        # Calculate precision and recall at each K
        disease_prec: dict[int, float] = {}
        disease_rec: dict[int, float] = {}

        for k in k_values:
            top_k = set(ranked_drugs[:k])
            hits = len(top_k & gt_drugs)
            prec = hits / k if k > 0 else 0
            rec = hits / len(gt_drugs) if gt_drugs else 0

            disease_prec[k] = prec
            disease_rec[k] = rec
            precision_at_k[k].append(prec)
            recall_at_k[k].append(rec)

        disease_results.append({
            "disease_id": disease_id,
            "n_gt_drugs": len(gt_drugs),
            "n_candidates": len(drug_counts),
            "precision": disease_prec,
            "recall": disease_rec,
        })

    # Aggregate
    agg_precision: dict[int, dict[str, float]] = {}
    agg_recall: dict[int, dict[str, float]] = {}

    for k in k_values:
        if precision_at_k[k]:
            agg_precision[k] = {
                "mean": float(np.mean(precision_at_k[k])),
                "std": float(np.std(precision_at_k[k])),
                "median": float(np.median(precision_at_k[k])),
            }
            agg_recall[k] = {
                "mean": float(np.mean(recall_at_k[k])),
                "std": float(np.std(recall_at_k[k])),
                "median": float(np.median(recall_at_k[k])),
            }
        else:
            agg_precision[k] = {"mean": 0, "std": 0, "median": 0}
            agg_recall[k] = {"mean": 0, "std": 0, "median": 0}

    return {
        "precision": agg_precision,
        "recall": agg_recall,
        "n_diseases": len(disease_results),
        "disease_results": disease_results,
    }


def calculate_cost_metrics(
    precision: dict[int, dict[str, float]],
    n_drugs_in_pool: int,
    n_gt_total: int,
) -> dict[str, Any]:
    """Calculate cost per true positive and break-even analysis."""
    cost_metrics: dict[str, Any] = {}

    # Random baseline precision
    random_precision = n_gt_total / n_drugs_in_pool if n_drugs_in_pool > 0 else 0

    for k, stats in precision.items():
        prec = stats["mean"]

        # Expected true positives per disease at K predictions
        expected_tp_per_disease = prec * k

        # Cost per true positive
        if expected_tp_per_disease > 0:
            cost_per_tp = VALIDATION_COST_PER_PREDICTION * k / expected_tp_per_disease
        else:
            cost_per_tp = float("inf")

        # Random baseline cost per TP
        if random_precision > 0:
            random_cost_per_tp = VALIDATION_COST_PER_PREDICTION / random_precision
        else:
            random_cost_per_tp = float("inf")

        # Cost reduction factor
        if random_cost_per_tp > 0 and random_cost_per_tp != float("inf"):
            cost_reduction = random_cost_per_tp / cost_per_tp if cost_per_tp > 0 else float("inf")
        else:
            cost_reduction = 1.0

        cost_metrics[k] = {
            "precision": prec,
            "expected_tp_per_disease_at_k": expected_tp_per_disease,
            "cost_per_tp": cost_per_tp,
            "random_precision": random_precision,
            "random_cost_per_tp": random_cost_per_tp,
            "cost_reduction_vs_random": cost_reduction,
        }

    return cost_metrics


def main() -> None:
    start_time = time.time()

    print("=" * 70)
    print("PRECISION AND COST ANALYSIS")
    print("=" * 70)
    print()
    print(f"Assumed validation cost per prediction: ${VALIDATION_COST_PER_PREDICTION:,}")
    print()

    # Load data
    print("Loading data...")
    name_to_drug_id, id_to_name = load_drugbank_lookup()
    mesh_mappings = load_mesh_mappings_from_file()
    gt_pairs = load_ground_truth(mesh_mappings, name_to_drug_id)
    print(f"  {len(gt_pairs)} diseases, {sum(len(v) for v in gt_pairs.values())} GT pairs")

    # Load honest embeddings
    honest_path = EMBEDDINGS_DIR / "node2vec_256_no_treatment.csv"
    if not honest_path.exists():
        print(f"ERROR: {honest_path} not found")
        sys.exit(1)

    emb_dict = load_embeddings(honest_path)
    print(f"  {len(emb_dict):,} embeddings loaded")

    # Count drugs in embedding pool
    n_drugs = len([e for e in emb_dict if "Compound::" in e])
    n_gt_total = sum(len(v) for v in gt_pairs.values())
    print(f"  {n_drugs:,} drugs in embedding space")
    print()

    # Run multi-seed evaluation with precision metrics
    print("-" * 70)
    print("Running 5-seed evaluation with precision@K metrics...")
    print("-" * 70)

    all_precision: dict[int, list[float]] = {k: [] for k in K_VALUES}
    all_recall: dict[int, list[float]] = {k: [] for k in K_VALUES}

    for seed in SEEDS:
        print(f"\n  Seed {seed}:")
        train_gt, test_gt = disease_level_split(
            gt_pairs, lambda d: d in emb_dict, test_fraction=0.2, seed=seed
        )

        result = evaluate_knn_with_precision(
            emb_dict, train_gt, test_gt,
            k_nn=20, k_values=K_VALUES
        )

        for k in K_VALUES:
            prec = result["precision"][k]["mean"]
            rec = result["recall"][k]["mean"]
            all_precision[k].append(prec)
            all_recall[k].append(rec)
            print(f"    P@{k}: {prec*100:.1f}%  R@{k}: {rec*100:.1f}%")

    # Aggregate across seeds
    print()
    print("-" * 70)
    print("AGGREGATE RESULTS (5-seed mean ± std)")
    print("-" * 70)

    agg_results: dict[str, Any] = {}

    print()
    print(f"  {'K':<6} {'Precision':<20} {'Recall':<20}")
    print("-" * 50)

    for k in K_VALUES:
        prec_mean = np.mean(all_precision[k])
        prec_std = np.std(all_precision[k])
        rec_mean = np.mean(all_recall[k])
        rec_std = np.std(all_recall[k])

        print(f"  {k:<6} {prec_mean*100:.2f}% ± {prec_std*100:.2f}%     {rec_mean*100:.2f}% ± {rec_std*100:.2f}%")

        agg_results[str(k)] = {
            "precision_mean": float(prec_mean),
            "precision_std": float(prec_std),
            "recall_mean": float(rec_mean),
            "recall_std": float(rec_std),
        }

    # Cost analysis
    print()
    print("-" * 70)
    print("COST ANALYSIS")
    print("-" * 70)
    print()

    # Build precision dict for cost calc
    precision_for_cost = {
        k: {"mean": np.mean(all_precision[k])}
        for k in K_VALUES
    }

    cost_metrics = calculate_cost_metrics(precision_for_cost, n_drugs, n_gt_total)

    print(f"  Validation cost assumption: ${VALIDATION_COST_PER_PREDICTION:,} per drug candidate")
    print(f"  Drug pool size: {n_drugs:,} drugs")
    print(f"  Random baseline precision: {cost_metrics[30]['random_precision']*100:.4f}%")
    print()
    print(f"  {'K':<6} {'Precision':<12} {'Cost/TP':<15} {'vs Random':<12}")
    print("-" * 50)

    for k in K_VALUES:
        metrics = cost_metrics[k]
        prec_pct = metrics["precision"] * 100
        cost = metrics["cost_per_tp"]
        reduction = metrics["cost_reduction_vs_random"]

        if cost == float("inf"):
            cost_str = "∞"
        else:
            cost_str = f"${cost/1e6:.2f}M"

        print(f"  {k:<6} {prec_pct:.2f}%       {cost_str:<15} {reduction:.1f}x better")

    # Summary
    print()
    print("=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    print()

    best_k = 30
    best_prec = np.mean(all_precision[best_k])
    best_cost = cost_metrics[best_k]["cost_per_tp"]
    random_cost = cost_metrics[best_k]["random_cost_per_tp"]

    print(f"  At K=30 (our standard evaluation point):")
    print(f"    - Precision: {best_prec*100:.1f}% of top-30 predictions are correct")
    print(f"    - Expected true positives: {best_prec * 30:.1f} per disease")
    print(f"    - Cost per true positive: ${best_cost/1e6:.2f}M (vs ${random_cost/1e6:.2f}M random)")
    print(f"    - Cost reduction: {cost_metrics[best_k]['cost_reduction_vs_random']:.1f}x better than random")
    print()
    print(f"  Interpretation: For every 30 predictions examined, ~{best_prec*30:.0f} are valid")
    print(f"  repurposing candidates (known treatments in ground truth).")

    # Save results
    results: dict[str, Any] = {
        "analysis": "precision_cost",
        "parameters": {
            "k_values": K_VALUES,
            "seeds": SEEDS,
            "k_nn": 20,
            "validation_cost_per_prediction": VALIDATION_COST_PER_PREDICTION,
        },
        "data_stats": {
            "n_diseases_in_gt": len(gt_pairs),
            "n_gt_pairs": sum(len(v) for v in gt_pairs.values()),
            "n_drugs_in_pool": n_drugs,
            "n_embeddings": len(emb_dict),
        },
        "metrics_per_k": agg_results,
        "cost_analysis": cost_metrics,
        "elapsed_seconds": time.time() - start_time,
    }

    output_path = ANALYSIS_DIR / "precision_cost.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
