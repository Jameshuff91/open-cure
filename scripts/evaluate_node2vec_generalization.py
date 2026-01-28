#!/usr/bin/env python3
"""
Evaluate Node2Vec generalization to held-out diseases (Hypothesis h29).

PURPOSE:
    CLAUDE.md claims Node2Vec+XGBoost achieved 41.9% R@30 on held-out diseases,
    but code review shows train_with_node2vec.py uses PAIR-LEVEL split, not
    disease-level. This script verifies whether Node2Vec genuinely generalizes
    to unseen diseases using the same disease-level holdout methodology from h5.

EXPERIMENT DESIGN:
    1. Disease-level holdout: 80/20 split (seed=42, same as h5)
    2. Node2Vec embeddings: 256-dim from DRKG
    3. Feature variants tested:
       a. concat only (as in train_with_node2vec.py)
       b. concat + product + diff (as in h5 TransE evaluation)
    4. Also test: cosine similarity ranking (no ML model)
    5. Also test: existing Node2Vec model on held-out diseases (pair-level trained)
    6. Compare with TransE under identical conditions

BASELINE: GB+TransE existing model = 45.89% R@30 on held-out diseases
          GB+TransE retrained = 3-12% R@30 (h5 finding)
"""

import json
import pickle
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, Tuple, List, Optional

import numpy as np
import pandas as pd
import torch
from xgboost import XGBClassifier
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from disease_name_matcher import DiseaseMatcher, load_mesh_mappings

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
MODELS_DIR = PROJECT_ROOT / "models"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_node2vec_embeddings() -> Dict[str, np.ndarray]:
    """Load Node2Vec 256-dim embeddings from named CSV."""
    print("  Loading Node2Vec embeddings from CSV...")
    df = pd.read_csv(EMBEDDINGS_DIR / "node2vec_256_named.csv")
    dim_cols = [c for c in df.columns if c.startswith("dim_")]
    embeddings: Dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        entity = f"drkg:{row['entity']}"
        embeddings[entity] = row[dim_cols].values.astype(np.float32)
    print(f"  Loaded {len(embeddings)} entities, {len(dim_cols)} dims")
    return embeddings


def load_transe_embeddings() -> Tuple[np.ndarray, Dict[str, int]]:
    """Load TransE embeddings and entity2id."""
    print("  Loading TransE embeddings...")
    checkpoint = torch.load(MODELS_DIR / "transe.pt", map_location="cpu", weights_only=False)

    emb = None
    if "entity_embeddings" in checkpoint:
        emb = checkpoint["entity_embeddings"].numpy()
    elif "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
        for key in ["entity_embeddings.weight", "ent_embeddings.weight"]:
            if key in state:
                emb = state[key].numpy()
                break

    entity2id = checkpoint.get("entity2id", {})
    print(f"  Loaded {emb.shape if emb is not None else 'None'} embeddings, {len(entity2id)} entities")
    return emb, entity2id


def load_drugbank_lookup() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Load DrugBank name -> DRKG ID and ID -> name mappings."""
    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)
    name_to_id = {name.lower(): f"drkg:Compound::{db_id}" for db_id, name in id_to_name.items()}
    id_to_name_drkg = {f"drkg:Compound::{db_id}": name for db_id, name in id_to_name.items()}
    return name_to_id, id_to_name_drkg


def load_mesh_mappings_from_file() -> Dict[str, str]:
    """Load MESH mappings from agents file."""
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


# ─── Disease-Level Split ──────────────────────────────────────────────────────

def disease_level_split(
    gt_pairs: Dict[str, Set[str]],
    valid_entity_check: callable,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Split GT by disease into train/test. Only include diseases with embeddings."""
    valid_diseases = [d for d in gt_pairs if valid_entity_check(d)]
    rng = np.random.RandomState(seed)
    rng.shuffle(valid_diseases)

    n_test = max(1, int(len(valid_diseases) * test_fraction))
    test_diseases = set(valid_diseases[:n_test])
    train_diseases = set(valid_diseases[n_test:])

    train_gt = {d: gt_pairs[d] for d in train_diseases}
    test_gt = {d: gt_pairs[d] for d in test_diseases}
    return train_gt, test_gt


# ─── Feature Creation ────────────────────────────────────────────────────────

def features_concat(drug_emb: np.ndarray, disease_emb: np.ndarray) -> np.ndarray:
    """Concatenation only (as in train_with_node2vec.py)."""
    return np.concatenate([drug_emb, disease_emb])


def features_concat_product_diff(drug_emb: np.ndarray, disease_emb: np.ndarray) -> np.ndarray:
    """Concat + element-wise product + difference (as in h5 TransE evaluation)."""
    return np.concatenate([
        np.concatenate([drug_emb, disease_emb]),
        drug_emb * disease_emb,
        drug_emb - disease_emb,
    ])


# ─── Training ────────────────────────────────────────────────────────────────

def build_training_data_dict(
    emb_dict: Dict[str, np.ndarray],
    train_gt: Dict[str, Set[str]],
    feature_fn: callable,
    neg_ratio: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build training data from dict-based embeddings (Node2Vec)."""
    all_drugs = [e for e in emb_dict if "Compound::" in e]
    rng = np.random.RandomState(seed)

    X_list = []
    y_list = []

    for disease_id, drug_ids in train_gt.items():
        if disease_id not in emb_dict:
            continue
        disease_emb = emb_dict[disease_id]
        valid_drugs = [d for d in drug_ids if d in emb_dict]
        if not valid_drugs:
            continue

        # Positives
        for drug_id in valid_drugs:
            X_list.append(feature_fn(emb_dict[drug_id], disease_emb))
            y_list.append(1)

        # Negatives
        drugs_set = set(drug_ids)
        neg_pool = [d for d in all_drugs if d not in drugs_set]
        n_neg = min(len(valid_drugs) * neg_ratio, len(neg_pool))
        neg_samples = rng.choice(neg_pool, n_neg, replace=False)
        for neg_drug in neg_samples:
            X_list.append(feature_fn(emb_dict[neg_drug], disease_emb))
            y_list.append(0)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)


def build_training_data_array(
    embeddings: np.ndarray,
    entity2id: Dict[str, int],
    train_gt: Dict[str, Set[str]],
    feature_fn: callable,
    neg_ratio: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build training data from array-based embeddings (TransE)."""
    all_drugs = [e for e in entity2id if e.startswith("drkg:Compound::")]
    rng = np.random.RandomState(seed)

    X_list = []
    y_list = []

    for disease_id, drug_ids in train_gt.items():
        if disease_id not in entity2id:
            continue
        disease_emb = embeddings[entity2id[disease_id]]
        valid_drugs = [d for d in drug_ids if d in entity2id]
        if not valid_drugs:
            continue

        # Positives
        for drug_id in valid_drugs:
            drug_emb = embeddings[entity2id[drug_id]]
            X_list.append(feature_fn(drug_emb, disease_emb))
            y_list.append(1)

        # Negatives
        drugs_set = set(drug_ids)
        neg_pool = [d for d in all_drugs if d not in drugs_set]
        n_neg = min(len(valid_drugs) * neg_ratio, len(neg_pool))
        neg_samples = rng.choice(neg_pool, n_neg, replace=False)
        for neg_drug in neg_samples:
            drug_emb = embeddings[entity2id[neg_drug]]
            X_list.append(feature_fn(drug_emb, disease_emb))
            y_list.append(0)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_recall_dict(
    model,
    emb_dict: Dict[str, np.ndarray],
    test_gt: Dict[str, Set[str]],
    feature_fn: callable,
    k: int = 30,
) -> Tuple[float, List[Dict]]:
    """Evaluate R@K using dict-based embeddings (Node2Vec)."""
    all_drugs = [e for e in emb_dict if "Compound::" in e]
    drug_embs = np.array([emb_dict[d] for d in all_drugs], dtype=np.float32)

    per_disease = []
    total_hits = 0
    total_gt = 0

    for disease_id in tqdm(list(test_gt.keys()), desc=f"R@{k} eval"):
        if disease_id not in emb_dict:
            continue
        gt_drugs = {d for d in test_gt[disease_id] if d in emb_dict}
        if not gt_drugs:
            continue

        disease_emb = emb_dict[disease_id]

        # Batch feature creation
        n_drugs = len(all_drugs)
        disease_tiled = np.tile(disease_emb, (n_drugs, 1))

        if feature_fn == features_concat:
            X_batch = np.hstack([drug_embs, disease_tiled])
        else:
            X_batch = np.hstack([
                np.hstack([drug_embs, disease_tiled]),
                drug_embs * disease_tiled,
                drug_embs - disease_tiled,
            ])

        scores = model.predict_proba(X_batch.astype(np.float32))[:, 1]
        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_set = {all_drugs[i] for i in top_k_idx}

        hits = len(top_k_set & gt_drugs)
        total_hits += hits
        total_gt += len(gt_drugs)

        per_disease.append({
            "disease_id": disease_id,
            "gt_drugs": len(gt_drugs),
            "hits": hits,
            "recall": hits / len(gt_drugs),
        })

    recall = total_hits / total_gt if total_gt > 0 else 0
    return recall, per_disease


def evaluate_recall_array(
    model,
    embeddings: np.ndarray,
    entity2id: Dict[str, int],
    test_gt: Dict[str, Set[str]],
    feature_fn: callable,
    k: int = 30,
) -> Tuple[float, List[Dict]]:
    """Evaluate R@K using array-based embeddings (TransE)."""
    all_drugs = [e for e in entity2id if e.startswith("drkg:Compound::")]
    drug_indices = [entity2id[d] for d in all_drugs]
    drug_embs = embeddings[drug_indices]

    per_disease = []
    total_hits = 0
    total_gt = 0

    for disease_id in tqdm(list(test_gt.keys()), desc=f"R@{k} eval"):
        if disease_id not in entity2id:
            continue
        gt_drugs = {d for d in test_gt[disease_id] if d in entity2id}
        if not gt_drugs:
            continue

        disease_emb = embeddings[entity2id[disease_id]]
        n_drugs = len(all_drugs)
        disease_tiled = np.tile(disease_emb, (n_drugs, 1))

        if feature_fn == features_concat:
            X_batch = np.hstack([drug_embs, disease_tiled])
        else:
            X_batch = np.hstack([
                np.hstack([drug_embs, disease_tiled]),
                drug_embs * disease_tiled,
                drug_embs - disease_tiled,
            ])

        scores = model.predict_proba(X_batch.astype(np.float32))[:, 1]
        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_set = {all_drugs[i] for i in top_k_idx}

        hits = len(top_k_set & gt_drugs)
        total_hits += hits
        total_gt += len(gt_drugs)

        per_disease.append({
            "disease_id": disease_id,
            "gt_drugs": len(gt_drugs),
            "hits": hits,
            "recall": hits / len(gt_drugs),
        })

    recall = total_hits / total_gt if total_gt > 0 else 0
    return recall, per_disease


def evaluate_cosine_similarity(
    emb_dict: Dict[str, np.ndarray],
    test_gt: Dict[str, Set[str]],
    k: int = 30,
) -> Tuple[float, List[Dict]]:
    """Evaluate R@K using cosine similarity ranking (no ML model)."""
    all_drugs = [e for e in emb_dict if "Compound::" in e]
    drug_embs = np.array([emb_dict[d] for d in all_drugs], dtype=np.float32)

    per_disease = []
    total_hits = 0
    total_gt = 0

    for disease_id in tqdm(list(test_gt.keys()), desc="Cosine R@k eval"):
        if disease_id not in emb_dict:
            continue
        gt_drugs = {d for d in test_gt[disease_id] if d in emb_dict}
        if not gt_drugs:
            continue

        disease_emb = emb_dict[disease_id].reshape(1, -1)
        sims = cosine_similarity(disease_emb, drug_embs)[0]

        top_k_idx = np.argpartition(sims, -k)[-k:]
        top_k_set = {all_drugs[i] for i in top_k_idx}

        hits = len(top_k_set & gt_drugs)
        total_hits += hits
        total_gt += len(gt_drugs)

        per_disease.append({
            "disease_id": disease_id,
            "gt_drugs": len(gt_drugs),
            "hits": hits,
            "recall": hits / len(gt_drugs),
        })

    recall = total_hits / total_gt if total_gt > 0 else 0
    return recall, per_disease


def evaluate_cosine_array(
    embeddings: np.ndarray,
    entity2id: Dict[str, int],
    test_gt: Dict[str, Set[str]],
    k: int = 30,
) -> Tuple[float, List[Dict]]:
    """Evaluate R@K using cosine similarity for array-based embeddings (TransE)."""
    all_drugs = [e for e in entity2id if e.startswith("drkg:Compound::")]
    drug_indices = [entity2id[d] for d in all_drugs]
    drug_embs = embeddings[drug_indices]

    per_disease = []
    total_hits = 0
    total_gt = 0

    for disease_id in tqdm(list(test_gt.keys()), desc="Cosine R@k eval"):
        if disease_id not in entity2id:
            continue
        gt_drugs = {d for d in test_gt[disease_id] if d in entity2id}
        if not gt_drugs:
            continue

        disease_emb = embeddings[entity2id[disease_id]].reshape(1, -1)
        sims = cosine_similarity(disease_emb, drug_embs)[0]

        top_k_idx = np.argpartition(sims, -k)[-k:]
        top_k_set = {all_drugs[i] for i in top_k_idx}

        hits = len(top_k_set & gt_drugs)
        total_hits += hits
        total_gt += len(gt_drugs)

        per_disease.append({
            "disease_id": disease_id,
            "gt_drugs": len(gt_drugs),
            "hits": hits,
            "recall": hits / len(gt_drugs),
        })

    recall = total_hits / total_gt if total_gt > 0 else 0
    return recall, per_disease


# ─── Positive Controls ────────────────────────────────────────────────────────

def run_positive_controls_dict(
    model,
    emb_dict: Dict[str, np.ndarray],
    name_to_drug_id: Dict[str, str],
    feature_fn: callable,
) -> List[Dict]:
    """Check known drug-disease pairs rank using dict embeddings."""
    controls = [
        ("metformin", "drkg:Disease::MESH:D003924"),
        ("rituximab", "drkg:Disease::MESH:D009103"),
        ("imatinib", "drkg:Disease::MESH:D015464"),
        ("lisinopril", "drkg:Disease::MESH:D006973"),
    ]

    all_drugs = [e for e in emb_dict if "Compound::" in e]
    drug_embs = np.array([emb_dict[d] for d in all_drugs], dtype=np.float32)

    results = []
    for drug_name, disease_id in controls:
        drug_id = name_to_drug_id.get(drug_name)
        if not drug_id or drug_id not in emb_dict or disease_id not in emb_dict:
            results.append({"drug": drug_name, "disease_id": disease_id, "status": "missing"})
            continue

        disease_emb = emb_dict[disease_id]
        n_drugs = len(all_drugs)
        disease_tiled = np.tile(disease_emb, (n_drugs, 1))

        if feature_fn == features_concat:
            X_batch = np.hstack([drug_embs, disease_tiled])
        else:
            X_batch = np.hstack([
                np.hstack([drug_embs, disease_tiled]),
                drug_embs * disease_tiled,
                drug_embs - disease_tiled,
            ])

        scores = model.predict_proba(X_batch.astype(np.float32))[:, 1]

        drug_idx = all_drugs.index(drug_id) if drug_id in all_drugs else -1
        if drug_idx < 0:
            results.append({"drug": drug_name, "disease_id": disease_id, "status": "drug_not_found"})
            continue

        target_score = float(scores[drug_idx])
        rank = int((scores > target_score).sum() + 1)
        results.append({
            "drug": drug_name,
            "disease_id": disease_id,
            "rank": rank,
            "score": target_score,
            "hit_at_30": rank <= 30,
        })
    return results


# ─── Main Experiment ──────────────────────────────────────────────────────────

def main():
    start_time = time.time()

    print("=" * 70)
    print("h29: VERIFY NODE2VEC HELD-OUT DISEASE GENERALIZATION")
    print("=" * 70)
    print()
    print("Question: Does Node2Vec+XGBoost generalize to unseen diseases?")
    print("Background: GB+TransE collapses from 45.89% to 3-12% R@30 on")
    print("  disease-level holdout (h5). Node2Vec '41.9%' is UNVERIFIED.")
    print("Method: Same 80/20 disease split as h5, seed=42")
    print()

    # ─── Load Data ────────────────────────────────────────────────────────
    print("=" * 70)
    print("1. LOADING DATA")
    print("=" * 70)

    print("\n[1a] Node2Vec embeddings...")
    n2v_emb = load_node2vec_embeddings()
    n2v_compounds = sum(1 for e in n2v_emb if "Compound::" in e)
    n2v_diseases = sum(1 for e in n2v_emb if "Disease::" in e)
    print(f"  Compounds: {n2v_compounds}, Diseases: {n2v_diseases}")

    print("\n[1b] TransE embeddings...")
    transe_emb, entity2id = load_transe_embeddings()

    print("\n[1c] DrugBank lookup...")
    name_to_drug_id, id_to_drug_name = load_drugbank_lookup()
    mesh_mappings = load_mesh_mappings_from_file()
    print(f"  Drugs: {len(name_to_drug_id)}, MESH mappings: {len(mesh_mappings)}")

    print("\n[1d] Ground truth...")
    gt_pairs = load_ground_truth(mesh_mappings, name_to_drug_id)
    total_pairs = sum(len(v) for v in gt_pairs.values())
    print(f"  GT: {len(gt_pairs)} diseases, {total_pairs} pairs")

    # ─── Disease Split ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("2. DISEASE-LEVEL SPLIT (80/20, seed=42)")
    print("=" * 70)

    # Split using Node2Vec availability
    n2v_train_gt, n2v_test_gt = disease_level_split(
        gt_pairs, lambda d: d in n2v_emb, test_fraction=0.2, seed=42
    )
    n2v_train_pairs = sum(len(v) for v in n2v_train_gt.values())
    n2v_test_pairs = sum(len(v) for v in n2v_test_gt.values())
    print(f"\n  Node2Vec split:")
    print(f"    Train: {len(n2v_train_gt)} diseases, {n2v_train_pairs} pairs")
    print(f"    Test:  {len(n2v_test_gt)} diseases, {n2v_test_pairs} pairs")

    # Split using TransE availability (same seed for comparability)
    transe_train_gt, transe_test_gt = disease_level_split(
        gt_pairs, lambda d: d in entity2id, test_fraction=0.2, seed=42
    )
    transe_train_pairs = sum(len(v) for v in transe_train_gt.values())
    transe_test_pairs = sum(len(v) for v in transe_test_gt.values())
    print(f"\n  TransE split:")
    print(f"    Train: {len(transe_train_gt)} diseases, {transe_train_pairs} pairs")
    print(f"    Test:  {len(transe_test_gt)} diseases, {transe_test_pairs} pairs")

    # Check overlap — diseases in both splits
    n2v_test_diseases = set(n2v_test_gt.keys())
    transe_test_diseases = set(transe_test_gt.keys())
    common_test = n2v_test_diseases & transe_test_diseases
    print(f"\n  Common test diseases: {len(common_test)} "
          f"(of {len(n2v_test_diseases)} n2v, {len(transe_test_diseases)} transe)")

    # ─── Run Experiments ──────────────────────────────────────────────────
    results = {}

    # ──────────────────────────────────────────────────────────────────────
    # Experiment 1: Existing Node2Vec model (pair-level trained) on held-out diseases
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("3. EXPERIMENT 1: Existing Node2Vec Model (pair-level trained)")
    print("=" * 70)
    print("  This model was trained on ALL diseases with pair-level split.")
    print("  Evaluating on disease-level held-out test diseases...")

    existing_model_path = MODELS_DIR / "drug_repurposing_node2vec.pkl"
    if existing_model_path.exists():
        with open(existing_model_path, "rb") as f:
            existing_n2v_model = pickle.load(f)

        recall_existing, details_existing = evaluate_recall_dict(
            existing_n2v_model, n2v_emb, n2v_test_gt, features_concat, k=30
        )
        print(f"\n  Existing Node2Vec model R@30 (test diseases): {recall_existing*100:.2f}%")
        print(f"  Hits: {sum(d['hits'] for d in details_existing)}/{sum(d['gt_drugs'] for d in details_existing)}")

        results["existing_n2v_model"] = {
            "description": "Existing Node2Vec XGBoost (pair-level trained, evaluated on held-out diseases)",
            "recall_at_30": float(recall_existing),
            "diseases_evaluated": len(details_existing),
            "per_disease": details_existing,
        }
    else:
        print("  WARNING: Existing Node2Vec model not found.")
        results["existing_n2v_model"] = {"status": "model_not_found"}

    # ──────────────────────────────────────────────────────────────────────
    # Experiment 2: Node2Vec + XGBoost (concat only), disease-level holdout
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("4. EXPERIMENT 2: Node2Vec + XGBoost (concat) — Disease Holdout")
    print("=" * 70)

    print("  Building training data (concat features)...")
    X_n2v_concat, y_n2v_concat = build_training_data_dict(
        n2v_emb, n2v_train_gt, features_concat, neg_ratio=5, seed=42
    )
    print(f"  Training data: {X_n2v_concat.shape[0]} samples, {X_n2v_concat.shape[1]} features")
    print(f"  Pos/Neg: {(y_n2v_concat==1).sum()}/{(y_n2v_concat==0).sum()}")

    print("  Training XGBoost...")
    n2v_concat_model = XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=42, n_jobs=-1, eval_metric="auc", verbosity=0,
    )
    n2v_concat_model.fit(X_n2v_concat, y_n2v_concat)

    print("  Evaluating on held-out test diseases...")
    recall_n2v_concat, details_n2v_concat = evaluate_recall_dict(
        n2v_concat_model, n2v_emb, n2v_test_gt, features_concat, k=30
    )
    print(f"\n  Node2Vec+XGBoost (concat) R@30: {recall_n2v_concat*100:.2f}%")

    # Positive controls
    pc_n2v_concat = run_positive_controls_dict(
        n2v_concat_model, n2v_emb, name_to_drug_id, features_concat
    )
    print("  Positive controls:")
    for pc in pc_n2v_concat:
        if pc.get("rank"):
            print(f"    {pc['drug']}: rank={pc['rank']}, hit@30={pc.get('hit_at_30')}")

    results["n2v_concat_holdout"] = {
        "description": "Node2Vec + XGBoost (concat only), disease-level holdout",
        "recall_at_30": float(recall_n2v_concat),
        "diseases_evaluated": len(details_n2v_concat),
        "positive_controls": pc_n2v_concat,
        "per_disease": details_n2v_concat,
    }

    # ──────────────────────────────────────────────────────────────────────
    # Experiment 3: Node2Vec + XGBoost (concat+product+diff), disease holdout
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("5. EXPERIMENT 3: Node2Vec + XGBoost (concat+product+diff) — Disease Holdout")
    print("=" * 70)

    print("  Building training data (concat+product+diff features)...")
    X_n2v_cpd, y_n2v_cpd = build_training_data_dict(
        n2v_emb, n2v_train_gt, features_concat_product_diff, neg_ratio=5, seed=42
    )
    print(f"  Training data: {X_n2v_cpd.shape[0]} samples, {X_n2v_cpd.shape[1]} features")

    print("  Training XGBoost...")
    n2v_cpd_model = XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=42, n_jobs=-1, eval_metric="auc", verbosity=0,
    )
    n2v_cpd_model.fit(X_n2v_cpd, y_n2v_cpd)

    print("  Evaluating on held-out test diseases...")
    recall_n2v_cpd, details_n2v_cpd = evaluate_recall_dict(
        n2v_cpd_model, n2v_emb, n2v_test_gt, features_concat_product_diff, k=30
    )
    print(f"\n  Node2Vec+XGBoost (concat+prod+diff) R@30: {recall_n2v_cpd*100:.2f}%")

    pc_n2v_cpd = run_positive_controls_dict(
        n2v_cpd_model, n2v_emb, name_to_drug_id, features_concat_product_diff
    )
    print("  Positive controls:")
    for pc in pc_n2v_cpd:
        if pc.get("rank"):
            print(f"    {pc['drug']}: rank={pc['rank']}, hit@30={pc.get('hit_at_30')}")

    results["n2v_cpd_holdout"] = {
        "description": "Node2Vec + XGBoost (concat+product+diff), disease-level holdout",
        "recall_at_30": float(recall_n2v_cpd),
        "diseases_evaluated": len(details_n2v_cpd),
        "positive_controls": pc_n2v_cpd,
        "per_disease": details_n2v_cpd,
    }

    # ──────────────────────────────────────────────────────────────────────
    # Experiment 4: TransE + XGBoost (concat+product+diff), disease holdout
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("6. EXPERIMENT 4: TransE + XGBoost (concat+product+diff) — Disease Holdout")
    print("=" * 70)
    print("  (Reproducing h5 Strategy A for direct comparison)")

    print("  Building training data...")
    X_transe_cpd, y_transe_cpd = build_training_data_array(
        transe_emb, entity2id, transe_train_gt,
        features_concat_product_diff, neg_ratio=5, seed=42
    )
    print(f"  Training data: {X_transe_cpd.shape[0]} samples, {X_transe_cpd.shape[1]} features")

    print("  Training XGBoost...")
    transe_cpd_model = XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=42, n_jobs=-1, eval_metric="auc", verbosity=0,
    )
    transe_cpd_model.fit(X_transe_cpd, y_transe_cpd)

    print("  Evaluating on held-out test diseases...")
    recall_transe_cpd, details_transe_cpd = evaluate_recall_array(
        transe_cpd_model, transe_emb, entity2id, transe_test_gt,
        features_concat_product_diff, k=30
    )
    print(f"\n  TransE+XGBoost (concat+prod+diff) R@30: {recall_transe_cpd*100:.2f}%")

    results["transe_cpd_holdout"] = {
        "description": "TransE + XGBoost (concat+product+diff), disease-level holdout",
        "recall_at_30": float(recall_transe_cpd),
        "diseases_evaluated": len(details_transe_cpd),
        "per_disease": details_transe_cpd,
    }

    # ──────────────────────────────────────────────────────────────────────
    # Experiment 5: Node2Vec Cosine Similarity (no ML model)
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("7. EXPERIMENT 5: Node2Vec Cosine Similarity (No ML Model)")
    print("=" * 70)
    print("  Ranking drugs by cosine similarity to disease — no training needed")

    recall_n2v_cosine, details_n2v_cosine = evaluate_cosine_similarity(
        n2v_emb, n2v_test_gt, k=30
    )
    print(f"\n  Node2Vec Cosine Similarity R@30: {recall_n2v_cosine*100:.2f}%")

    results["n2v_cosine"] = {
        "description": "Node2Vec cosine similarity ranking (no ML model)",
        "recall_at_30": float(recall_n2v_cosine),
        "diseases_evaluated": len(details_n2v_cosine),
        "per_disease": details_n2v_cosine,
    }

    # ──────────────────────────────────────────────────────────────────────
    # Experiment 6: TransE Cosine Similarity (no ML model)
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("8. EXPERIMENT 6: TransE Cosine Similarity (No ML Model)")
    print("=" * 70)

    recall_transe_cosine, details_transe_cosine = evaluate_cosine_array(
        transe_emb, entity2id, transe_test_gt, k=30
    )
    print(f"\n  TransE Cosine Similarity R@30: {recall_transe_cosine*100:.2f}%")

    results["transe_cosine"] = {
        "description": "TransE cosine similarity ranking (no ML model)",
        "recall_at_30": float(recall_transe_cosine),
        "diseases_evaluated": len(details_transe_cosine),
        "per_disease": details_transe_cosine,
    }

    # ──────────────────────────────────────────────────────────────────────
    # Experiment 7: Existing GB+TransE model on held-out diseases
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("9. EXPERIMENT 7: Existing GB+TransE Model (pair-level trained)")
    print("=" * 70)

    with open(MODELS_DIR / "drug_repurposing_gb_enhanced.pkl", "rb") as f:
        existing_gb_model = pickle.load(f)

    recall_existing_gb, details_existing_gb = evaluate_recall_array(
        existing_gb_model, transe_emb, entity2id, transe_test_gt,
        features_concat_product_diff, k=30
    )
    print(f"\n  Existing GB+TransE model R@30 (test diseases): {recall_existing_gb*100:.2f}%")

    results["existing_gb_transe"] = {
        "description": "Existing GB+TransE (pair-level trained, evaluated on held-out diseases)",
        "recall_at_30": float(recall_existing_gb),
        "diseases_evaluated": len(details_existing_gb),
        "per_disease": details_existing_gb,
    }

    # ─── Summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("SUMMARY: NODE2VEC GENERALIZATION VERIFICATION (h29)")
    print("=" * 70)
    print(f"\n{'Experiment':<55} {'R@30':>8} {'Diseases':>10}")
    print("-" * 78)

    summary_rows = [
        ("Existing GB+TransE (pair-trained)", results.get("existing_gb_transe", {})),
        ("Existing Node2Vec (pair-trained)", results.get("existing_n2v_model", {})),
        ("Node2Vec+XGBoost concat (disease holdout)", results.get("n2v_concat_holdout", {})),
        ("Node2Vec+XGBoost cpd (disease holdout)", results.get("n2v_cpd_holdout", {})),
        ("TransE+XGBoost cpd (disease holdout)", results.get("transe_cpd_holdout", {})),
        ("Node2Vec Cosine (no ML)", results.get("n2v_cosine", {})),
        ("TransE Cosine (no ML)", results.get("transe_cosine", {})),
    ]

    for label, r in summary_rows:
        if "recall_at_30" in r:
            print(f"  {label:<55} {r['recall_at_30']*100:>7.2f}% {r.get('diseases_evaluated', '?'):>10}")
        else:
            print(f"  {label:<55} {'N/A':>8} {'N/A':>10}")

    print(f"\nElapsed time: {elapsed:.0f}s")

    # ─── Key Analysis ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("KEY ANALYSIS")
    print("=" * 70)

    n2v_concat_r = results.get("n2v_concat_holdout", {}).get("recall_at_30", 0)
    n2v_cpd_r = results.get("n2v_cpd_holdout", {}).get("recall_at_30", 0)
    transe_cpd_r = results.get("transe_cpd_holdout", {}).get("recall_at_30", 0)

    best_n2v = max(n2v_concat_r, n2v_cpd_r)
    print(f"\n  Best Node2Vec (disease holdout): {best_n2v*100:.2f}%")
    print(f"  TransE baseline (disease holdout): {transe_cpd_r*100:.2f}%")

    if best_n2v > 0.20:
        print("\n  RESULT: Node2Vec DOES generalize to unseen diseases!")
        if best_n2v > transe_cpd_r * 1.5:
            print("  Node2Vec significantly outperforms TransE under disease holdout.")
    elif best_n2v > transe_cpd_r * 1.5:
        print("\n  RESULT: Node2Vec generalizes BETTER than TransE, but both are weak.")
    else:
        print("\n  RESULT: Node2Vec also fails to generalize (same as TransE).")
        print("  The generalization problem is ARCHITECTURAL, not embedding-specific.")

    # ─── Save Results ─────────────────────────────────────────────────────
    # Strip per_disease details for compact JSON (keep summary)
    output = {
        "hypothesis": "h29",
        "title": "Verify Node2Vec Held-Out Disease Generalization",
        "date": "2026-01-27",
        "design": {
            "method": "Disease-level 80/20 holdout, seed=42",
            "n2v_train_diseases": len(n2v_train_gt),
            "n2v_test_diseases": len(n2v_test_gt),
            "transe_train_diseases": len(transe_train_gt),
            "transe_test_diseases": len(transe_test_gt),
            "common_test_diseases": len(common_test),
        },
        "results": {},
        "elapsed_seconds": elapsed,
    }

    for key, val in results.items():
        output["results"][key] = {
            k: v for k, v in val.items() if k != "per_disease"
        }
        if "per_disease" in val:
            output["results"][key]["diseases_evaluated"] = len(val["per_disease"])

    output_path = ANALYSIS_DIR / "h29_node2vec_generalization_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
