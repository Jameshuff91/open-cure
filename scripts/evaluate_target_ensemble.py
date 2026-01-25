#!/usr/bin/env python3
"""
Evaluate target-based ensemble: combine baseline GB scores with target overlap.

Instead of retraining, we boost scores when drug targets overlap with disease genes.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Set
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
MODELS_DIR = PROJECT_ROOT / "models"


def main():
    print("=" * 70)
    print("TARGET ENSEMBLE EVALUATION")
    print("=" * 70)

    # Load baseline model
    print("\n1. Loading baseline model...")
    with open(MODELS_DIR / "drug_repurposing_gb_enhanced.pkl", "rb") as f:
        model = pickle.load(f)

    # Load embeddings
    print("2. Loading embeddings...")
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
    print("3. Loading target data...")
    with open(REFERENCE_DIR / "drug_targets.json") as f:
        drug_targets_raw = json.load(f)
    drug_targets = {k: set(v) for k, v in drug_targets_raw.items()}

    with open(REFERENCE_DIR / "disease_genes.json") as f:
        disease_genes_raw = json.load(f)
    disease_genes = {k: set(v) for k, v in disease_genes_raw.items()}

    # Load mappings
    print("4. Loading mappings...")
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

    # Test different ensemble strategies
    strategies = {
        "baseline": lambda score, overlap, frac: score,
        "boost_if_overlap": lambda score, overlap, frac: score * 1.1 if overlap > 0 else score,
        "boost_by_frac": lambda score, overlap, frac: score * (1 + 0.2 * frac),
        "boost_by_overlap": lambda score, overlap, frac: score * (1 + 0.01 * min(overlap, 10)),
        "multiply_frac": lambda score, overlap, frac: score * (1 + frac),
        "add_bonus": lambda score, overlap, frac: min(1.0, score + 0.1 * frac),
    }

    results = {name: {"hits": 0, "total": 0} for name in strategies}

    print("\n5. Evaluating strategies...")

    for disease_name, disease_data in tqdm(gt.items(), desc="Diseases"):
        mesh_id = mesh_mappings.get(disease_name.lower())
        if not mesh_id:
            continue

        disease_idx = entity2id.get(mesh_id)
        if disease_idx is None:
            continue

        disease_emb = embeddings[disease_idx]
        mesh_id_short = mesh_id.split("MESH:")[-1]

        # Get disease genes
        dis_genes = disease_genes.get(f"MESH:{mesh_id_short}", set())

        # Get GT drugs
        gt_local_indices = set()
        for drug_info in disease_data['drugs']:
            drug_name = drug_info['name'].lower()
            drug_id = name_to_id.get(drug_name)
            if drug_id and drug_id in drug_id_to_local_idx:
                gt_local_indices.add(drug_id_to_local_idx[drug_id])

        if not gt_local_indices:
            continue

        for name in strategies:
            results[name]["total"] += len(gt_local_indices)

        # Score drugs
        n_drugs = len(drug_embs)
        disease_emb_tiled = np.tile(disease_emb, (n_drugs, 1))

        concat_feats = np.hstack([drug_embs, disease_emb_tiled])
        product_feats = drug_embs * disease_emb_tiled
        diff_feats = drug_embs - disease_emb_tiled
        base_features = np.hstack([concat_feats, product_feats, diff_feats])

        base_scores = model.predict_proba(base_features)[:, 1]

        # Compute target overlap for each drug
        overlaps = []
        fracs = []
        for drug_id in valid_drug_ids:
            db_id = drug_id.split("::")[-1]
            drug_genes = drug_targets.get(db_id, set())
            overlap = len(drug_genes & dis_genes)
            frac = overlap / len(drug_genes) if len(drug_genes) > 0 else 0
            overlaps.append(overlap)
            fracs.append(frac)

        overlaps = np.array(overlaps)
        fracs = np.array(fracs)

        # Evaluate each strategy
        for name, strategy_fn in strategies.items():
            scores = np.array([
                strategy_fn(base_scores[i], overlaps[i], fracs[i])
                for i in range(n_drugs)
            ])
            top_30 = set(np.argsort(scores)[-30:])
            hits = len(top_30 & gt_local_indices)
            results[name]["hits"] += hits

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    for name in strategies:
        total = results[name]["total"]
        hits = results[name]["hits"]
        recall = hits / total if total > 0 else 0
        diff = recall - results["baseline"]["hits"] / results["baseline"]["total"]
        diff_str = f"{diff:+.1%}" if name != "baseline" else ""
        print(f"{name:25} R@30: {recall:.1%} ({hits}/{total}) {diff_str}")


if __name__ == "__main__":
    main()
