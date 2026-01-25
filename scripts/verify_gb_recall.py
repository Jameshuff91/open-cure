#!/usr/bin/env python3
"""
Verify GB model Recall@30 on Every Cure ground truth.

This script re-runs the evaluation from scratch to confirm the 38.7% R@30 claim.
"""

import json
import pickle
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, List, Tuple
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REFERENCE_DIR = DATA_DIR / "reference"
MODELS_DIR = PROJECT_ROOT / "models"


def create_features(drug_emb: np.ndarray, disease_emb: np.ndarray) -> np.ndarray:
    """Create features from embeddings (must match training)."""
    concat = np.concatenate([drug_emb, disease_emb])
    product = drug_emb * disease_emb
    diff = drug_emb - disease_emb
    return np.concatenate([concat, product, diff])


def load_mesh_mappings() -> Dict[str, str]:
    """Load disease name -> MESH ID mappings from agents."""
    mesh_path = REFERENCE_DIR / "mesh_mappings_from_agents.json"
    if not mesh_path.exists():
        print(f"ERROR: MESH mappings not found: {mesh_path}")
        return {}

    with open(mesh_path) as f:
        data = json.load(f)

    mappings = {}
    for batch_name, batch_data in data.items():
        if not isinstance(batch_data, dict):
            continue
        for disease_name, mesh_id in batch_data.items():
            if mesh_id is None:
                continue
            mesh_str = str(mesh_id)
            if mesh_str.startswith("D"):
                drkg_id = f"drkg:Disease::MESH:{mesh_str}"
                mappings[disease_name.lower()] = drkg_id

    print(f"Loaded {len(mappings)} MESH mappings")
    return mappings


def load_drugbank_lookup() -> Dict[str, str]:
    """Load drug name -> DrugBank ID mapping."""
    lookup_path = REFERENCE_DIR / "drugbank_lookup.json"
    with open(lookup_path) as f:
        id_to_name = json.load(f)

    name_to_id = {}
    for db_id, name in id_to_name.items():
        name_to_id[name.lower()] = f"drkg:Compound::{db_id}"

    print(f"Loaded {len(name_to_id)} DrugBank mappings")
    return name_to_id


def load_ground_truth() -> Dict[str, List[str]]:
    """Load Every Cure ground truth."""
    gt_path = REFERENCE_DIR / "everycure_gt_for_txgnn.json"
    with open(gt_path) as f:
        gt = json.load(f)

    # Convert to disease_name -> [drug_names]
    result = {}
    for disease_name, disease_data in gt.items():
        drugs = [d['name'].lower() for d in disease_data['drugs']]
        result[disease_name.lower()] = drugs

    total_pairs = sum(len(drugs) for drugs in result.values())
    print(f"Loaded ground truth: {len(result)} diseases, {total_pairs} drug-disease pairs")
    return result


def evaluate_gb_model(
    model,
    embeddings: np.ndarray,
    entity2id: Dict[str, int],
    ground_truth: Dict[str, List[str]],
    mesh_mappings: Dict[str, str],
    drugbank_lookup: Dict[str, str],
    all_drug_ids: List[str],
    k: int = 30
) -> Dict:
    """Evaluate GB model on ground truth (VECTORIZED for speed)."""

    # Pre-compute drug embeddings matrix
    print("   Pre-computing drug embeddings...")
    valid_drug_ids = []
    valid_drug_indices = []
    for drug_id in all_drug_ids:
        drug_idx = entity2id.get(drug_id)
        if drug_idx is not None:
            valid_drug_ids.append(drug_id)
            valid_drug_indices.append(drug_idx)

    drug_embs = embeddings[valid_drug_indices]  # (num_drugs, 128)
    drug_id_to_local_idx = {did: i for i, did in enumerate(valid_drug_ids)}
    print(f"   Valid drugs for scoring: {len(valid_drug_ids)}")

    total_hits = 0
    total_gt_drugs = 0
    diseases_evaluated = 0
    diseases_skipped_no_mesh = 0
    diseases_skipped_no_embedding = 0
    diseases_with_hits = 0

    per_disease_results = []

    for disease_name, gt_drug_names in tqdm(ground_truth.items(), desc="Evaluating"):
        # Map disease to MESH
        mesh_id = mesh_mappings.get(disease_name)
        if not mesh_id:
            diseases_skipped_no_mesh += 1
            continue

        # Get disease embedding
        disease_idx = entity2id.get(mesh_id)
        if disease_idx is None:
            diseases_skipped_no_embedding += 1
            continue

        disease_emb = embeddings[disease_idx]
        diseases_evaluated += 1

        # Convert GT drug names to local indices
        gt_local_indices = set()
        for drug_name in gt_drug_names:
            drug_id = drugbank_lookup.get(drug_name)
            if drug_id and drug_id in drug_id_to_local_idx:
                gt_local_indices.add(drug_id_to_local_idx[drug_id])

        if not gt_local_indices:
            continue

        total_gt_drugs += len(gt_local_indices)

        # VECTORIZED: Create features for ALL drugs at once
        n_drugs = len(drug_embs)
        disease_emb_tiled = np.tile(disease_emb, (n_drugs, 1))  # (n_drugs, 128)
        concat_feats = np.hstack([drug_embs, disease_emb_tiled])  # (n_drugs, 256)
        product_feats = drug_embs * disease_emb_tiled  # (n_drugs, 128)
        diff_feats = drug_embs - disease_emb_tiled  # (n_drugs, 128)
        all_features = np.hstack([concat_feats, product_feats, diff_feats])  # (n_drugs, 512)

        # Score all drugs in one batch
        scores = model.predict_proba(all_features)[:, 1]

        # Get top k indices (argsort ascending, take last k)
        top_k_indices = set(np.argsort(scores)[-k:])

        # Count hits
        hits = len(top_k_indices & gt_local_indices)
        total_hits += hits

        if hits > 0:
            diseases_with_hits += 1

        per_disease_results.append({
            'disease': disease_name,
            'gt_drugs': len(gt_local_indices),
            'hits': hits,
            'recall': hits / len(gt_local_indices)
        })

    per_drug_recall = total_hits / total_gt_drugs if total_gt_drugs > 0 else 0

    return {
        'diseases_evaluated': diseases_evaluated,
        'diseases_skipped_no_mesh': diseases_skipped_no_mesh,
        'diseases_skipped_no_embedding': diseases_skipped_no_embedding,
        'diseases_with_hits': diseases_with_hits,
        'total_gt_drugs': total_gt_drugs,
        'total_hits': total_hits,
        'per_drug_recall_at_k': per_drug_recall,
        'per_disease_avg_recall': np.mean([r['recall'] for r in per_disease_results]) if per_disease_results else 0,
        'k': k,
        'per_disease_results': per_disease_results
    }


def main():
    print("=" * 70)
    print("VERIFICATION: GB Model Recall@30 on Every Cure Ground Truth")
    print("=" * 70)

    # Load TransE embeddings
    print("\n1. Loading TransE embeddings...")
    checkpoint = torch.load(MODELS_DIR / "transe.pt", map_location="cpu")

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
    print(f"   Embeddings shape: {embeddings.shape}")
    print(f"   Entities: {len(entity2id)}")

    # Get all drug IDs
    all_drug_ids = [eid for eid in entity2id.keys() if "Compound" in eid]
    print(f"   Total drugs in KG: {len(all_drug_ids)}")

    # Load mappings
    print("\n2. Loading mappings...")
    mesh_mappings = load_mesh_mappings()
    drugbank_lookup = load_drugbank_lookup()
    ground_truth = load_ground_truth()

    # Load model
    print("\n3. Loading GB model...")
    model_path = MODELS_DIR / "drug_repurposing_gb_enhanced.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"   Model: {model_path.name}")

    # Evaluate
    print("\n4. Evaluating (this may take a few minutes)...")
    results = evaluate_gb_model(
        model, embeddings, entity2id, ground_truth,
        mesh_mappings, drugbank_lookup, all_drug_ids, k=30
    )

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Diseases in ground truth:     {len(ground_truth)}")
    print(f"Diseases evaluated:           {results['diseases_evaluated']}")
    print(f"Diseases skipped (no MESH):   {results['diseases_skipped_no_mesh']}")
    print(f"Diseases skipped (no embed):  {results['diseases_skipped_no_embedding']}")
    print(f"Diseases with ≥1 hit:         {results['diseases_with_hits']}")
    print(f"")
    print(f"Total GT drug-disease pairs:  {results['total_gt_drugs']}")
    print(f"Hits in top 30:               {results['total_hits']}")
    print(f"")
    print(f"{'='*40}")
    print(f"PER-DRUG RECALL@30:           {results['per_drug_recall_at_k']:.1%}")
    print(f"PER-DISEASE AVG RECALL@30:    {results['per_disease_avg_recall']:.1%}")
    print(f"{'='*40}")

    # Compare to claimed 38.7%
    claimed = 0.387
    actual = results['per_drug_recall_at_k']
    diff = abs(actual - claimed) * 100

    print(f"\nClaimed R@30: {claimed:.1%}")
    print(f"Actual R@30:  {actual:.1%}")
    print(f"Difference:   {diff:.1f} percentage points")

    if diff < 1:
        print("\n✓ VERIFIED: Result matches within 1 percentage point")
    else:
        print(f"\n⚠ DISCREPANCY: Result differs by {diff:.1f} percentage points")

    # Save verification results
    output_path = DATA_DIR / "analysis" / "gb_recall_verification.json"
    with open(output_path, "w") as f:
        json.dump({
            'claimed_recall': claimed,
            'verified_recall': actual,
            'difference_pp': diff,
            'verified': diff < 1,
            'details': {k: v for k, v in results.items() if k != 'per_disease_results'}
        }, f, indent=2)
    print(f"\nSaved verification to: {output_path}")


if __name__ == "__main__":
    main()
