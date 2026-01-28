#!/usr/bin/env python3
"""
Hypothesis h40: Multi-Seed Disease Holdout Stability Check.

PURPOSE:
    All previous results used seed=42 for disease-level holdout split.
    This script runs the same Node2Vec+XGBoost evaluation with 5 different
    seeds to establish mean +/- std R@30 and determine if the baseline
    is stable or a fluke of the particular split.

EXPERIMENT:
    - Seeds: 42, 123, 456, 789, 1024
    - Model: Node2Vec+XGBoost (concat features)
    - XGBoost params: default (n_estimators=100, max_depth=6, lr=0.1)
    - Also test with tuned XGBoost params from h38
    - Disease-level 80/20 holdout for each seed
"""

import json
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, Tuple, List

import numpy as np
import pandas as pd
import torch
from xgboost import XGBClassifier
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from disease_name_matcher import DiseaseMatcher, load_mesh_mappings

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
MODELS_DIR = PROJECT_ROOT / "models"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"

SEEDS = [42, 123, 456, 789, 1024]


def load_node2vec_embeddings() -> Dict[str, np.ndarray]:
    """Load Node2Vec 256-dim embeddings."""
    df = pd.read_csv(EMBEDDINGS_DIR / "node2vec_256_named.csv")
    dim_cols = [c for c in df.columns if c.startswith("dim_")]
    embeddings: Dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        entity = f"drkg:{row['entity']}"
        embeddings[entity] = row[dim_cols].values.astype(np.float32)
    return embeddings


def load_drugbank_lookup() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Load DrugBank name -> DRKG ID mappings."""
    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)
    name_to_id = {name.lower(): f"drkg:Compound::{db_id}" for db_id, name in id_to_name.items()}
    id_to_name_drkg = {f"drkg:Compound::{db_id}": name for db_id, name in id_to_name.items()}
    return name_to_id, id_to_name_drkg


def load_mesh_mappings_from_file() -> Dict[str, str]:
    """Load MESH mappings."""
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
    """Load ground truth as {disease_drkg_id: set of drug_drkg_ids}."""
    df = pd.read_excel(REFERENCE_DIR / "everycure" / "indicationList.xlsx")
    fuzzy_mappings = load_mesh_mappings()
    matcher = DiseaseMatcher(fuzzy_mappings)

    gt_pairs: Dict[str, Set[str]] = defaultdict(set)
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
    gt_pairs: Dict[str, Set[str]],
    valid_entity_check,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Split GT by disease into train/test."""
    valid_diseases = [d for d in gt_pairs if valid_entity_check(d)]
    rng = np.random.RandomState(seed)
    rng.shuffle(valid_diseases)

    n_test = max(1, int(len(valid_diseases) * test_fraction))
    test_diseases = set(valid_diseases[:n_test])
    train_diseases = set(valid_diseases[n_test:])

    train_gt = {d: gt_pairs[d] for d in train_diseases}
    test_gt = {d: gt_pairs[d] for d in test_diseases}
    return train_gt, test_gt


def features_concat(drug_emb: np.ndarray, disease_emb: np.ndarray) -> np.ndarray:
    return np.concatenate([drug_emb, disease_emb])


def build_training_data(
    emb_dict: Dict[str, np.ndarray],
    train_gt: Dict[str, Set[str]],
    neg_ratio: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build training data with concat features."""
    all_drugs = [e for e in emb_dict if "Compound::" in e]
    rng = np.random.RandomState(seed)

    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    for disease_id, drug_ids in train_gt.items():
        if disease_id not in emb_dict:
            continue
        disease_emb = emb_dict[disease_id]
        valid_drugs = [d for d in drug_ids if d in emb_dict]
        if not valid_drugs:
            continue

        for drug_id in valid_drugs:
            X_list.append(features_concat(emb_dict[drug_id], disease_emb))
            y_list.append(1)

        drugs_set = set(drug_ids)
        neg_pool = [d for d in all_drugs if d not in drugs_set]
        n_neg = min(len(valid_drugs) * neg_ratio, len(neg_pool))
        neg_samples = rng.choice(neg_pool, n_neg, replace=False)
        for neg_drug in neg_samples:
            X_list.append(features_concat(emb_dict[neg_drug], disease_emb))
            y_list.append(0)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)


def evaluate_recall(
    model,
    emb_dict: Dict[str, np.ndarray],
    test_gt: Dict[str, Set[str]],
    k: int = 30,
) -> Tuple[float, int, int, int]:
    """Evaluate R@K. Returns (recall, hits, total_gt, n_diseases)."""
    all_drugs = [e for e in emb_dict if "Compound::" in e]
    drug_embs = np.array([emb_dict[d] for d in all_drugs], dtype=np.float32)

    total_hits = 0
    total_gt = 0
    n_diseases = 0

    for disease_id in test_gt:
        if disease_id not in emb_dict:
            continue
        gt_drugs = {d for d in test_gt[disease_id] if d in emb_dict}
        if not gt_drugs:
            continue

        disease_emb = emb_dict[disease_id]
        n_drugs = len(all_drugs)
        disease_tiled = np.tile(disease_emb, (n_drugs, 1))
        X_batch = np.hstack([drug_embs, disease_tiled])

        scores = model.predict_proba(X_batch.astype(np.float32))[:, 1]
        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_set = {all_drugs[i] for i in top_k_idx}

        hits = len(top_k_set & gt_drugs)
        total_hits += hits
        total_gt += len(gt_drugs)
        n_diseases += 1

    recall = total_hits / total_gt if total_gt > 0 else 0
    return recall, total_hits, total_gt, n_diseases


def run_seed(
    emb_dict: Dict[str, np.ndarray],
    gt_pairs: Dict[str, Set[str]],
    seed: int,
    xgb_params: dict,
    label: str,
) -> dict:
    """Run one seed of the experiment."""
    train_gt, test_gt = disease_level_split(
        gt_pairs, lambda d: d in emb_dict, test_fraction=0.2, seed=seed
    )

    train_pairs = sum(len(v) for v in train_gt.values())
    test_pairs = sum(len(v) for v in test_gt.values())

    X_train, y_train = build_training_data(emb_dict, train_gt, neg_ratio=5, seed=seed)

    model = XGBClassifier(
        random_state=seed, n_jobs=-1, eval_metric="auc", verbosity=0,
        **xgb_params
    )
    model.fit(X_train, y_train)

    recall, hits, total_gt, n_diseases = evaluate_recall(model, emb_dict, test_gt, k=30)

    return {
        "seed": seed,
        "label": label,
        "recall_at_30": float(recall),
        "hits": hits,
        "total_gt": total_gt,
        "n_train_diseases": len(train_gt),
        "n_test_diseases": len(test_gt),
        "n_test_diseases_evaluated": n_diseases,
        "train_pairs": train_pairs,
        "test_pairs": test_pairs,
    }


def main():
    start_time = time.time()

    print("=" * 70)
    print("h40: MULTI-SEED DISEASE HOLDOUT STABILITY CHECK")
    print("=" * 70)
    print(f"Seeds: {SEEDS}")
    print()

    # Load data
    print("Loading data...")
    emb_dict = load_node2vec_embeddings()
    name_to_drug_id, _ = load_drugbank_lookup()
    mesh_mappings = load_mesh_mappings_from_file()
    gt_pairs = load_ground_truth(mesh_mappings, name_to_drug_id)
    n_diseases = len(gt_pairs)
    n_pairs = sum(len(v) for v in gt_pairs.values())
    print(f"  GT: {n_diseases} diseases, {n_pairs} pairs")
    print()

    # XGBoost configurations
    configs = {
        "default": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
        },
        "tuned_h38": {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.1,
            "min_child_weight": 1,
            "reg_alpha": 1.0,
            "reg_lambda": 1,
        },
    }

    all_results: List[dict] = []

    for config_name, xgb_params in configs.items():
        print("=" * 70)
        print(f"CONFIG: {config_name}")
        print(f"  Params: {xgb_params}")
        print("=" * 70)

        for seed in SEEDS:
            print(f"\n  Seed {seed}...", end=" ", flush=True)
            result = run_seed(emb_dict, gt_pairs, seed, xgb_params, config_name)
            print(f"R@30 = {result['recall_at_30']*100:.2f}% "
                  f"({result['hits']}/{result['total_gt']} hits, "
                  f"{result['n_test_diseases_evaluated']} diseases)")
            all_results.append(result)

        # Summary for this config
        config_results = [r for r in all_results if r["label"] == config_name]
        recalls = [r["recall_at_30"] for r in config_results]
        mean_r = np.mean(recalls)
        std_r = np.std(recalls)
        min_r = np.min(recalls)
        max_r = np.max(recalls)
        print(f"\n  {config_name} Summary:")
        print(f"    Mean R@30: {mean_r*100:.2f}% +/- {std_r*100:.2f}%")
        print(f"    Range: [{min_r*100:.2f}%, {max_r*100:.2f}%]")
        print()

    # Overall summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)

    for config_name in configs:
        config_results = [r for r in all_results if r["label"] == config_name]
        recalls = [r["recall_at_30"] for r in config_results]
        mean_r = np.mean(recalls)
        std_r = np.std(recalls)
        min_r = np.min(recalls)
        max_r = np.max(recalls)
        print(f"\n  {config_name}:")
        print(f"    Mean:  {mean_r*100:.2f}%")
        print(f"    Std:   {std_r*100:.2f}%")
        print(f"    Range: [{min_r*100:.2f}%, {max_r*100:.2f}%]")
        seed_strs = [f"{r['recall_at_30']*100:.2f}%" for r in config_results]
        print(f"    Seeds: {', '.join(seed_strs)}")

    print(f"\nElapsed: {elapsed:.0f}s")

    # Variance analysis
    default_recalls = [r["recall_at_30"] for r in all_results if r["label"] == "default"]
    tuned_recalls = [r["recall_at_30"] for r in all_results if r["label"] == "tuned_h38"]
    default_std = np.std(default_recalls) * 100
    tuned_std = np.std(tuned_recalls) * 100

    print("\n" + "=" * 70)
    print("STABILITY ASSESSMENT")
    print("=" * 70)
    if default_std < 2.0:
        print(f"  Default model variance ({default_std:.2f} pp) is LOW — results are stable")
    elif default_std < 5.0:
        print(f"  Default model variance ({default_std:.2f} pp) is MODERATE — single-seed results should be interpreted with caution")
    else:
        print(f"  Default model variance ({default_std:.2f} pp) is HIGH — single-seed experiments are unreliable")

    # Tuned vs default comparison
    default_mean = np.mean(default_recalls)
    tuned_mean = np.mean(tuned_recalls)
    improvement = (tuned_mean - default_mean) * 100
    print(f"\n  Tuned improvement over default: {improvement:+.2f} pp (mean)")
    if improvement > max(default_std, tuned_std):
        print("  Improvement EXCEEDS noise — tuned model is genuinely better")
    else:
        print("  Improvement is WITHIN noise — tuning gain may not be reliable")

    # Save results
    output = {
        "hypothesis": "h40",
        "title": "Multi-Seed Disease Holdout Stability Check",
        "date": "2026-01-27",
        "seeds": SEEDS,
        "configs": {k: v for k, v in configs.items()},
        "per_seed_results": all_results,
        "summary": {},
        "elapsed_seconds": elapsed,
    }

    for config_name in configs:
        config_results = [r for r in all_results if r["label"] == config_name]
        recalls = [r["recall_at_30"] for r in config_results]
        output["summary"][config_name] = {
            "mean_recall_at_30": float(np.mean(recalls)),
            "std_recall_at_30": float(np.std(recalls)),
            "min_recall_at_30": float(np.min(recalls)),
            "max_recall_at_30": float(np.max(recalls)),
            "all_recalls": [float(r) for r in recalls],
        }

    output_path = ANALYSIS_DIR / "h40_multi_seed_stability_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
