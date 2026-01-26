#!/usr/bin/env python3
"""
Evaluate GB model with target features on Every Cure ground truth.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Set
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REFERENCE_DIR = DATA_DIR / "reference"
MODELS_DIR = PROJECT_ROOT / "models"


def compute_target_features(
    drug_db_id: str,
    disease_mesh_id: str,
    drug_targets: Dict[str, Set[str]],
    disease_genes: Dict[str, Set[str]],
) -> np.ndarray:
    """Compute target-based features for a drug-disease pair."""
    drug_genes = drug_targets.get(drug_db_id, set())
    dis_genes = disease_genes.get(disease_mesh_id, set())

    overlap = drug_genes & dis_genes
    n_overlap = len(overlap)

    if len(drug_genes) > 0:
        frac = n_overlap / len(drug_genes)
    else:
        frac = 0.0

    return np.array([
        n_overlap,
        frac,
        1 if n_overlap > 0 else 0,
        len(drug_genes),
        len(dis_genes),
    ], dtype=np.float32)


def main():
    print("=" * 70)
    print("EVALUATING GB MODEL WITH TARGET FEATURES")
    print("=" * 70)

    # Load model
    print("\n1. Loading model with target features...")
    with open(MODELS_DIR / "drug_repurposing_gb_with_targets.pkl", "rb") as f:
        model_with_targets = pickle.load(f)

    # Also load baseline for comparison
    print("   Loading baseline model (no targets)...")
    with open(MODELS_DIR / "drug_repurposing_gb_enhanced.pkl", "rb") as f:
        model_baseline = pickle.load(f)

    # Load embeddings
    print("\n2. Loading TransE embeddings...")
    checkpoint = torch.load(MODELS_DIR / "transe.pt", map_location="cpu", weights_only=False)

    embeddings = None
    if "entity_embeddings" in checkpoint:
        embeddings = checkpoint["entity_embeddings"].numpy()
    elif "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
        for key in ["entity_embeddings.weight", "ent_embeddings.weight"]:
            if key in state:
                embeddings = state[key].numpy()
                break

    entity2id = checkpoint.get("entity2id", {})

    # Load target data
    print("\n3. Loading target data...")
    with open(REFERENCE_DIR / "drug_targets.json") as f:
        drug_targets_raw = json.load(f)
    drug_targets = {k: set(v) for k, v in drug_targets_raw.items()}

    with open(REFERENCE_DIR / "disease_genes.json") as f:
        disease_genes_raw = json.load(f)
    disease_genes = {k: set(v) for k, v in disease_genes_raw.items()}

    # Load mappings
    print("\n4. Loading mappings...")
    with open(REFERENCE_DIR / "mesh_mappings_from_agents.json") as f:
        mesh_data = json.load(f)

    mesh_mappings = {}
    for batch_data in mesh_data.values():
        if isinstance(batch_data, dict):
            for disease_name, mesh_id in batch_data.items():
                if mesh_id:
                    mesh_str = str(mesh_id)
                    if mesh_str.startswith("D"):
                        mesh_mappings[disease_name.lower()] = f"drkg:Disease::MESH:{mesh_str}"

    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)
    name_to_id = {name.lower(): f"drkg:Compound::{db_id}" for db_id, name in id_to_name.items()}

    with open(REFERENCE_DIR / "everycure_gt_for_txgnn.json") as f:
        gt = json.load(f)

    # Get all drug IDs
    all_drug_ids = [eid for eid in entity2id.keys() if "Compound" in eid]

    # Pre-compute drug embeddings
    valid_drug_ids = []
    valid_drug_indices = []
    for drug_id in all_drug_ids:
        drug_idx = entity2id.get(drug_id)
        if drug_idx is not None:
            valid_drug_ids.append(drug_id)
            valid_drug_indices.append(drug_idx)

    drug_embs = embeddings[valid_drug_indices]
    drug_id_to_local_idx = {did: i for i, did in enumerate(valid_drug_ids)}

    # Evaluate both models
    print("\n5. Evaluating...")

    results = {
        "baseline": {"hits": 0, "total": 0, "diseases_with_hits": 0},
        "with_targets": {"hits": 0, "total": 0, "diseases_with_hits": 0},
    }

    diseases_evaluated = 0

    for disease_name, disease_data in tqdm(gt.items(), desc="Diseases"):
        mesh_id = mesh_mappings.get(disease_name.lower())
        if not mesh_id:
            continue

        disease_idx = entity2id.get(mesh_id)
        if disease_idx is None:
            continue

        disease_emb = embeddings[disease_idx]
        diseases_evaluated += 1

        # Get mesh ID for target lookup
        mesh_id_short = mesh_id.split("MESH:")[-1]

        # Get GT drugs
        gt_local_indices = set()
        for drug_info in disease_data['drugs']:
            drug_name = drug_info['name'].lower()
            drug_id = name_to_id.get(drug_name)
            if drug_id and drug_id in drug_id_to_local_idx:
                gt_local_indices.add(drug_id_to_local_idx[drug_id])

        if not gt_local_indices:
            continue

        results["baseline"]["total"] += len(gt_local_indices)
        results["with_targets"]["total"] += len(gt_local_indices)

        n_drugs = len(drug_embs)
        disease_emb_tiled = np.tile(disease_emb, (n_drugs, 1))

        # Baseline features (no targets)
        concat_feats = np.hstack([drug_embs, disease_emb_tiled])
        product_feats = drug_embs * disease_emb_tiled
        diff_feats = drug_embs - disease_emb_tiled
        base_features = np.hstack([concat_feats, product_feats, diff_feats])

        # Score with baseline
        baseline_scores = model_baseline.predict_proba(base_features)[:, 1]
        baseline_top30 = set(np.argsort(baseline_scores)[-30:])
        baseline_hits = len(baseline_top30 & gt_local_indices)
        results["baseline"]["hits"] += baseline_hits
        if baseline_hits > 0:
            results["baseline"]["diseases_with_hits"] += 1

        # Add target features
        target_features = []
        for drug_id in valid_drug_ids:
            db_id = drug_id.split("::")[-1]
            tf = compute_target_features(db_id, f"MESH:{mesh_id_short}", drug_targets, disease_genes)
            target_features.append(tf)
        target_features = np.array(target_features)

        full_features = np.hstack([base_features, target_features])

        # Score with targets
        target_scores = model_with_targets.predict_proba(full_features)[:, 1]
        target_top30 = set(np.argsort(target_scores)[-30:])
        target_hits = len(target_top30 & gt_local_indices)
        results["with_targets"]["hits"] += target_hits
        if target_hits > 0:
            results["with_targets"]["diseases_with_hits"] += 1

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Diseases evaluated: {diseases_evaluated}")

    print("\n" + "-" * 40)
    print("BASELINE (no targets)")
    print("-" * 40)
    baseline_recall = results["baseline"]["hits"] / results["baseline"]["total"]
    print(f"Hits@30:       {results['baseline']['hits']} / {results['baseline']['total']}")
    print(f"Recall@30:     {baseline_recall:.1%}")
    print(f"Diseases w/hit: {results['baseline']['diseases_with_hits']}")

    print("\n" + "-" * 40)
    print("WITH TARGET FEATURES")
    print("-" * 40)
    target_recall = results["with_targets"]["hits"] / results["with_targets"]["total"]
    print(f"Hits@30:       {results['with_targets']['hits']} / {results['with_targets']['total']}")
    print(f"Recall@30:     {target_recall:.1%}")
    print(f"Diseases w/hit: {results['with_targets']['diseases_with_hits']}")

    print("\n" + "-" * 40)
    print("COMPARISON")
    print("-" * 40)
    diff = target_recall - baseline_recall
    print(f"Difference:    {diff:+.1%}")
    if diff > 0:
        print(f"IMPROVEMENT:   Target features help!")
    elif diff < 0:
        print(f"REGRESSION:    Target features hurt performance")
    else:
        print(f"NO CHANGE:     Target features have no effect")


if __name__ == "__main__":
    main()
