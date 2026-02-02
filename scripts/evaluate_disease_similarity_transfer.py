#!/usr/bin/env python3
"""
Hypothesis h39: Disease Similarity Transfer Learning.

PURPOSE:
    Test whether leveraging disease-disease similarity improves generalization
    to unseen diseases. Three approaches tested:

    A) Global model + similarity features: Add max/mean similarity to nearest
       training diseases as extra features.
    B) kNN drug transfer: For each test disease, rank drugs by how often they
       treat similar training diseases (no ML needed).
    C) Similarity-weighted local models: For each test disease, train a
       weighted XGBoost where training examples from similar diseases get
       higher sample weights.

EVALUATION:
    Multi-seed evaluation (seeds 42, 123, 456, 789, 1024) per h40 findings.
    Baseline: Tuned XGBoost mean 25.85% ± 4.06% R@30.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, Tuple, List, Optional

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from disease_name_matcher import DiseaseMatcher, load_mesh_mappings

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
MODELS_DIR = PROJECT_ROOT / "models"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"

SEEDS = [42, 123, 456, 789, 1024]


# ─── Data Loading (same as h40) ─────────────────────────────────────────────

def load_node2vec_embeddings(embeddings_path: Optional[Path] = None) -> Dict[str, np.ndarray]:
    """Load Node2Vec embeddings from CSV file.

    Args:
        embeddings_path: Path to embeddings CSV. If None, uses default node2vec_256_named.csv
    """
    if embeddings_path is None:
        embeddings_path = EMBEDDINGS_DIR / "node2vec_256_named.csv"

    print(f"  Loading embeddings from: {embeddings_path}")
    df = pd.read_csv(embeddings_path)
    dim_cols = [c for c in df.columns if c.startswith("dim_")]
    embeddings: Dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        entity = f"drkg:{row['entity']}"
        embeddings[entity] = row[dim_cols].values.astype(np.float32)
    return embeddings


def load_drugbank_lookup() -> Tuple[Dict[str, str], Dict[str, str]]:
    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)
    name_to_id = {name.lower(): f"drkg:Compound::{db_id}" for db_id, name in id_to_name.items()}
    id_to_name_drkg = {f"drkg:Compound::{db_id}": name for db_id, name in id_to_name.items()}
    return name_to_id, id_to_name_drkg


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


# ─── Approach A: Global Model + Similarity Features ──────────────────────────

def compute_disease_similarity_matrix(
    emb_dict: Dict[str, np.ndarray],
    disease_ids: List[str],
) -> np.ndarray:
    """Compute pairwise cosine similarity between diseases."""
    embs = np.array([emb_dict[d] for d in disease_ids], dtype=np.float32)
    return cosine_similarity(embs)


def build_training_data_with_sim_features(
    emb_dict: Dict[str, np.ndarray],
    train_gt: Dict[str, Set[str]],
    train_disease_list: List[str],
    train_sim_matrix: np.ndarray,
    neg_ratio: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build training data with similarity features added."""
    all_drugs = [e for e in emb_dict if "Compound::" in e]
    rng = np.random.RandomState(seed)
    disease_idx_map = {d: i for i, d in enumerate(train_disease_list)}

    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    for disease_id, drug_ids in train_gt.items():
        if disease_id not in emb_dict or disease_id not in disease_idx_map:
            continue
        disease_emb = emb_dict[disease_id]
        valid_drugs = [d for d in drug_ids if d in emb_dict]
        if not valid_drugs:
            continue

        # Similarity features for this disease relative to all other training diseases
        d_idx = disease_idx_map[disease_id]
        sims = train_sim_matrix[d_idx]
        # Exclude self-similarity
        other_sims = np.delete(sims, d_idx)
        sim_features = np.array([
            np.max(other_sims) if len(other_sims) > 0 else 0,
            np.mean(other_sims) if len(other_sims) > 0 else 0,
            np.median(other_sims) if len(other_sims) > 0 else 0,
        ], dtype=np.float32)

        for drug_id in valid_drugs:
            base = features_concat(emb_dict[drug_id], disease_emb)
            X_list.append(np.concatenate([base, sim_features]))
            y_list.append(1)

        drugs_set = set(drug_ids)
        neg_pool = [d for d in all_drugs if d not in drugs_set]
        n_neg = min(len(valid_drugs) * neg_ratio, len(neg_pool))
        neg_samples = rng.choice(neg_pool, n_neg, replace=False)
        for neg_drug in neg_samples:
            base = features_concat(emb_dict[neg_drug], disease_emb)
            X_list.append(np.concatenate([base, sim_features]))
            y_list.append(0)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)


def evaluate_approach_a(
    emb_dict: Dict[str, np.ndarray],
    train_gt: Dict[str, Set[str]],
    test_gt: Dict[str, Set[str]],
    xgb_params: dict,
    seed: int,
) -> float:
    """Approach A: Global model + similarity features."""
    train_disease_list = [d for d in train_gt if d in emb_dict]
    train_sim_matrix = compute_disease_similarity_matrix(emb_dict, train_disease_list)

    X_train, y_train = build_training_data_with_sim_features(
        emb_dict, train_gt, train_disease_list, train_sim_matrix,
        neg_ratio=5, seed=seed
    )

    model = XGBClassifier(random_state=seed, n_jobs=-1, eval_metric="auc", verbosity=0, **xgb_params)
    model.fit(X_train, y_train)

    # Evaluate
    all_drugs = [e for e in emb_dict if "Compound::" in e]
    drug_embs = np.array([emb_dict[d] for d in all_drugs], dtype=np.float32)
    train_disease_embs = np.array([emb_dict[d] for d in train_disease_list], dtype=np.float32)

    total_hits = 0
    total_gt = 0

    for disease_id in test_gt:
        if disease_id not in emb_dict:
            continue
        gt_drugs = {d for d in test_gt[disease_id] if d in emb_dict}
        if not gt_drugs:
            continue

        disease_emb = emb_dict[disease_id]
        n_drugs = len(all_drugs)
        disease_tiled = np.tile(disease_emb, (n_drugs, 1))
        base_features = np.hstack([drug_embs, disease_tiled])

        # Compute similarity features to training diseases
        test_emb = disease_emb.reshape(1, -1)
        sims = cosine_similarity(test_emb, train_disease_embs)[0]
        sim_features = np.array([np.max(sims), np.mean(sims), np.median(sims)], dtype=np.float32)
        sim_tiled = np.tile(sim_features, (n_drugs, 1))

        X_batch = np.hstack([base_features, sim_tiled]).astype(np.float32)
        scores = model.predict_proba(X_batch)[:, 1]

        top_k_idx = np.argpartition(scores, -30)[-30:]
        top_k_set = {all_drugs[i] for i in top_k_idx}
        hits = len(top_k_set & gt_drugs)
        total_hits += hits
        total_gt += len(gt_drugs)

    return total_hits / total_gt if total_gt > 0 else 0


# ─── Approach B: kNN Drug Transfer (No ML) ───────────────────────────────────

def evaluate_approach_b(
    emb_dict: Dict[str, np.ndarray],
    train_gt: Dict[str, Set[str]],
    test_gt: Dict[str, Set[str]],
    k_values: List[int],
) -> Dict[int, float]:
    """Approach B: For each test disease, rank drugs by frequency among k nearest training diseases."""
    train_disease_list = [d for d in train_gt if d in emb_dict]
    train_disease_embs = np.array([emb_dict[d] for d in train_disease_list], dtype=np.float32)
    all_drugs = [e for e in emb_dict if "Compound::" in e]
    drug_set_lookup = {d: emb_dict[d] for d in all_drugs}

    results: Dict[int, float] = {}

    for k in k_values:
        total_hits = 0
        total_gt = 0

        for disease_id in test_gt:
            if disease_id not in emb_dict:
                continue
            gt_drugs = {d for d in test_gt[disease_id] if d in emb_dict}
            if not gt_drugs:
                continue

            # Find k nearest training diseases
            test_emb = emb_dict[disease_id].reshape(1, -1)
            sims = cosine_similarity(test_emb, train_disease_embs)[0]
            top_k_disease_idx = np.argsort(sims)[-k:]

            # Count drug frequency among nearest diseases
            drug_counts: Dict[str, float] = defaultdict(float)
            for idx in top_k_disease_idx:
                neighbor_disease = train_disease_list[idx]
                neighbor_sim = sims[idx]
                for drug_id in train_gt[neighbor_disease]:
                    if drug_id in emb_dict:
                        drug_counts[drug_id] += neighbor_sim  # Weight by similarity

            # Rank by weighted count, take top 30
            if drug_counts:
                sorted_drugs = sorted(drug_counts.items(), key=lambda x: x[1], reverse=True)
                top_30 = {d for d, _ in sorted_drugs[:30]}
            else:
                top_30 = set()

            hits = len(top_30 & gt_drugs)
            total_hits += hits
            total_gt += len(gt_drugs)

        recall = total_hits / total_gt if total_gt > 0 else 0
        results[k] = recall

    return results


# ─── Approach C: Similarity-Weighted Local Models ─────────────────────────────

def evaluate_approach_c(
    emb_dict: Dict[str, np.ndarray],
    train_gt: Dict[str, Set[str]],
    test_gt: Dict[str, Set[str]],
    xgb_params: dict,
    seed: int,
    k_neighbors: int = 20,
    temperature: float = 5.0,
) -> float:
    """Approach C: For each test disease, train weighted XGBoost model."""
    all_drugs = [e for e in emb_dict if "Compound::" in e]
    drug_embs = np.array([emb_dict[d] for d in all_drugs], dtype=np.float32)
    train_disease_list = [d for d in train_gt if d in emb_dict]
    train_disease_embs = np.array([emb_dict[d] for d in train_disease_list], dtype=np.float32)
    rng = np.random.RandomState(seed)

    # Pre-build training data per disease (for efficiency)
    disease_training_data: Dict[str, Tuple[List[np.ndarray], List[int]]] = {}
    for disease_id in train_disease_list:
        drug_ids = train_gt[disease_id]
        disease_emb = emb_dict[disease_id]
        valid_drugs = [d for d in drug_ids if d in emb_dict]
        if not valid_drugs:
            continue

        X_list: List[np.ndarray] = []
        y_list: List[int] = []

        for drug_id in valid_drugs:
            X_list.append(features_concat(emb_dict[drug_id], disease_emb))
            y_list.append(1)

        drugs_set = set(drug_ids)
        neg_pool = [d for d in all_drugs if d not in drugs_set]
        n_neg = min(len(valid_drugs) * 5, len(neg_pool))
        neg_samples = rng.choice(neg_pool, n_neg, replace=False)
        for neg_drug in neg_samples:
            X_list.append(features_concat(emb_dict[neg_drug], disease_emb))
            y_list.append(0)

        disease_training_data[disease_id] = (X_list, y_list)

    total_hits = 0
    total_gt = 0

    for disease_id in test_gt:
        if disease_id not in emb_dict:
            continue
        gt_drugs = {d for d in test_gt[disease_id] if d in emb_dict}
        if not gt_drugs:
            continue

        # Find k nearest training diseases
        test_emb = emb_dict[disease_id].reshape(1, -1)
        sims = cosine_similarity(test_emb, train_disease_embs)[0]
        top_k_idx = np.argsort(sims)[-k_neighbors:]

        # Assemble weighted training data from nearest diseases
        X_all: List[np.ndarray] = []
        y_all: List[int] = []
        w_all: List[float] = []

        for idx in top_k_idx:
            neighbor_disease = train_disease_list[idx]
            sim = sims[idx]
            weight = np.exp(temperature * sim)  # Softmax-like weighting

            if neighbor_disease in disease_training_data:
                X_list, y_list = disease_training_data[neighbor_disease]
                X_all.extend(X_list)
                y_all.extend(y_list)
                w_all.extend([weight] * len(X_list))

        if len(X_all) < 10:
            # Not enough training data, skip
            total_gt += len(gt_drugs)
            continue

        X_train = np.array(X_all, dtype=np.float32)
        y_train = np.array(y_all, dtype=np.int32)
        w_train = np.array(w_all, dtype=np.float32)

        # Train local model
        model = XGBClassifier(
            n_estimators=50, max_depth=4, learning_rate=0.1,
            random_state=seed, n_jobs=-1, eval_metric="auc", verbosity=0,
        )
        model.fit(X_train, y_train, sample_weight=w_train)

        # Score all drugs for this test disease
        disease_emb = emb_dict[disease_id]
        n_drugs = len(all_drugs)
        disease_tiled = np.tile(disease_emb, (n_drugs, 1))
        X_batch = np.hstack([drug_embs, disease_tiled]).astype(np.float32)
        scores = model.predict_proba(X_batch)[:, 1]

        top_k_drug_idx = np.argpartition(scores, -30)[-30:]
        top_k_set = {all_drugs[i] for i in top_k_drug_idx}
        hits = len(top_k_set & gt_drugs)
        total_hits += hits
        total_gt += len(gt_drugs)

    return total_hits / total_gt if total_gt > 0 else 0


# ─── Baseline (same as h40) ──────────────────────────────────────────────────

def evaluate_baseline(
    emb_dict: Dict[str, np.ndarray],
    train_gt: Dict[str, Set[str]],
    test_gt: Dict[str, Set[str]],
    xgb_params: dict,
    seed: int,
) -> float:
    """Standard global model baseline."""
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
        n_neg = min(len(valid_drugs) * 5, len(neg_pool))
        neg_samples = rng.choice(neg_pool, n_neg, replace=False)
        for neg_drug in neg_samples:
            X_list.append(features_concat(emb_dict[neg_drug], disease_emb))
            y_list.append(0)

    X_train = np.array(X_list, dtype=np.float32)
    y_train = np.array(y_list, dtype=np.int32)

    model = XGBClassifier(random_state=seed, n_jobs=-1, eval_metric="auc", verbosity=0, **xgb_params)
    model.fit(X_train, y_train)

    drug_embs = np.array([emb_dict[d] for d in all_drugs], dtype=np.float32)
    total_hits = 0
    total_gt = 0

    for disease_id in test_gt:
        if disease_id not in emb_dict:
            continue
        gt_drugs = {d for d in test_gt[disease_id] if d in emb_dict}
        if not gt_drugs:
            continue
        disease_emb = emb_dict[disease_id]
        n_drugs = len(all_drugs)
        disease_tiled = np.tile(disease_emb, (n_drugs, 1))
        X_batch = np.hstack([drug_embs, disease_tiled]).astype(np.float32)
        scores = model.predict_proba(X_batch)[:, 1]
        top_k_idx = np.argpartition(scores, -30)[-30:]
        top_k_set = {all_drugs[i] for i in top_k_idx}
        hits = len(top_k_set & gt_drugs)
        total_hits += hits
        total_gt += len(gt_drugs)

    return total_hits / total_gt if total_gt > 0 else 0


# ─── Main ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate disease similarity transfer learning")
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=None,
        help="Path to custom embeddings CSV (default: node2vec_256_named.csv)"
    )
    parser.add_argument(
        "--knn-only",
        action="store_true",
        help="Only run kNN (approach B) evaluation, skip other approaches"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.time()

    print("=" * 70)
    print("h39: DISEASE SIMILARITY TRANSFER LEARNING")
    print("=" * 70)
    print()

    # Load data
    print("Loading data...")
    emb_dict = load_node2vec_embeddings(args.embeddings)
    name_to_drug_id, _ = load_drugbank_lookup()
    mesh_mappings = load_mesh_mappings_from_file()
    gt_pairs = load_ground_truth(mesh_mappings, name_to_drug_id)
    print(f"  GT: {len(gt_pairs)} diseases, {sum(len(v) for v in gt_pairs.values())} pairs")

    tuned_params = {
        "n_estimators": 500, "max_depth": 6, "learning_rate": 0.1,
        "min_child_weight": 1, "reg_alpha": 1.0, "reg_lambda": 1,
    }

    all_results: List[dict] = []

    # ─── First: quick single-seed test of all approaches ──────────────────
    print("\n" + "=" * 70)
    print("PHASE 1: SINGLE-SEED EXPLORATION (seed=42)")
    print("=" * 70)

    seed = 42
    train_gt, test_gt = disease_level_split(
        gt_pairs, lambda d: d in emb_dict, test_fraction=0.2, seed=seed
    )
    print(f"  Train: {len(train_gt)} diseases, Test: {len(test_gt)} diseases")

    # Baseline
    print("\n  [Baseline] Global model (tuned XGBoost)...", flush=True)
    baseline_r = evaluate_baseline(emb_dict, train_gt, test_gt, tuned_params, seed)
    print(f"    R@30 = {baseline_r*100:.2f}%")

    # Approach A: Global + sim features
    print("\n  [Approach A] Global model + similarity features...", flush=True)
    approach_a_r = evaluate_approach_a(emb_dict, train_gt, test_gt, tuned_params, seed)
    print(f"    R@30 = {approach_a_r*100:.2f}%")

    # Approach B: kNN drug transfer
    print("\n  [Approach B] kNN drug transfer (no ML)...")
    k_values = [3, 5, 10, 20, 50]
    approach_b_results = evaluate_approach_b(emb_dict, train_gt, test_gt, k_values)
    for k, r in approach_b_results.items():
        print(f"    k={k:3d}: R@30 = {r*100:.2f}%")

    # Approach C: Similarity-weighted local models
    print("\n  [Approach C] Similarity-weighted local models...")
    for k_n in [10, 20, 50]:
        for temp in [2.0, 5.0, 10.0]:
            print(f"    k={k_n}, temp={temp}...", end=" ", flush=True)
            approach_c_r = evaluate_approach_c(
                emb_dict, train_gt, test_gt, tuned_params, seed,
                k_neighbors=k_n, temperature=temp
            )
            print(f"R@30 = {approach_c_r*100:.2f}%")
            all_results.append({
                "approach": f"C_k{k_n}_t{temp}",
                "seed": seed,
                "recall_at_30": float(approach_c_r),
            })

    phase1_results = {
        "baseline": float(baseline_r),
        "approach_a": float(approach_a_r),
        "approach_b": {str(k): float(v) for k, v in approach_b_results.items()},
        "approach_c": {r["approach"]: r["recall_at_30"] for r in all_results if r["approach"].startswith("C_")},
    }

    # Identify best approaches from Phase 1
    best_approaches = []
    if approach_a_r > baseline_r:
        best_approaches.append(("approach_a", approach_a_r))
    best_b_k = max(approach_b_results, key=approach_b_results.get)
    best_b_r = approach_b_results[best_b_k]
    if best_b_r > baseline_r:
        best_approaches.append((f"approach_b_k{best_b_k}", best_b_r))
    c_results = [(r["approach"], r["recall_at_30"]) for r in all_results if r["approach"].startswith("C_")]
    if c_results:
        best_c = max(c_results, key=lambda x: x[1])
        if best_c[1] > baseline_r:
            best_approaches.append(best_c)

    print("\n" + "=" * 70)
    print("PHASE 1 SUMMARY (seed=42)")
    print("=" * 70)
    print(f"  Baseline:     {baseline_r*100:.2f}%")
    print(f"  Approach A:   {approach_a_r*100:.2f}% ({(approach_a_r-baseline_r)*100:+.2f} pp)")
    print(f"  Best B (k={best_b_k}): {best_b_r*100:.2f}% ({(best_b_r-baseline_r)*100:+.2f} pp)")
    if c_results:
        best_c = max(c_results, key=lambda x: x[1])
        print(f"  Best C ({best_c[0]}): {best_c[1]*100:.2f}% ({(best_c[1]-baseline_r)*100:+.2f} pp)")

    # ─── Phase 2: Multi-seed evaluation of promising approaches ───────────
    # Only do multi-seed if any approach beat baseline by >1 pp
    promising = [a for a in best_approaches if a[1] > baseline_r + 0.01]

    if promising:
        print("\n" + "=" * 70)
        print("PHASE 2: MULTI-SEED EVALUATION OF PROMISING APPROACHES")
        print("=" * 70)

        multi_seed_results: Dict[str, List[float]] = {"baseline": []}

        for approach_name, _ in promising:
            multi_seed_results[approach_name] = []

        for seed in SEEDS:
            print(f"\n  Seed {seed}:")
            train_gt, test_gt = disease_level_split(
                gt_pairs, lambda d: d in emb_dict, test_fraction=0.2, seed=seed
            )

            # Baseline
            r = evaluate_baseline(emb_dict, train_gt, test_gt, tuned_params, seed)
            multi_seed_results["baseline"].append(r)
            print(f"    Baseline: {r*100:.2f}%")

            for approach_name, _ in promising:
                if approach_name == "approach_a":
                    r = evaluate_approach_a(emb_dict, train_gt, test_gt, tuned_params, seed)
                elif approach_name.startswith("approach_b"):
                    k = int(approach_name.split("k")[1])
                    b_results = evaluate_approach_b(emb_dict, train_gt, test_gt, [k])
                    r = b_results[k]
                elif approach_name.startswith("C_"):
                    parts = approach_name.split("_")
                    k_n = int(parts[1][1:])
                    temp = float(parts[2][1:])
                    r = evaluate_approach_c(
                        emb_dict, train_gt, test_gt, tuned_params, seed,
                        k_neighbors=k_n, temperature=temp
                    )
                else:
                    continue
                multi_seed_results[approach_name].append(r)
                print(f"    {approach_name}: {r*100:.2f}%")

        # Multi-seed summary
        print("\n" + "=" * 70)
        print("MULTI-SEED SUMMARY")
        print("=" * 70)
        for name, recalls in multi_seed_results.items():
            mean_r = np.mean(recalls) * 100
            std_r = np.std(recalls) * 100
            print(f"  {name:30s}: {mean_r:.2f}% ± {std_r:.2f}%")
    else:
        print("\n  No approach beat baseline by >1 pp. Skipping multi-seed evaluation.")
        multi_seed_results = None

    # ─── Save Results ─────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    output = {
        "hypothesis": "h39",
        "title": "Disease Similarity Transfer Learning",
        "date": "2026-01-27",
        "phase1": phase1_results,
        "multi_seed": {k: [float(v) for v in vals] for k, vals in multi_seed_results.items()} if multi_seed_results else None,
        "elapsed_seconds": elapsed,
    }

    output_path = ANALYSIS_DIR / "h39_similarity_transfer_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")
    print(f"Elapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
