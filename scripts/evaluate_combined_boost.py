#!/usr/bin/env python3
"""
Evaluate combined boosting: target overlap + ATC mechanism relevance.

Tests if combining ATC-based boosting with target overlap provides
additional improvement over target overlap alone.
"""

import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Set, Callable

import numpy as np
import torch
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from atc_features import ATCMapper
    ATC_AVAILABLE = True
except ImportError:
    ATC_AVAILABLE = False
    print("Warning: ATC features not available")

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
MODELS_DIR = PROJECT_ROOT / "models"


def main() -> None:
    print("=" * 70)
    print("COMBINED BOOST EVALUATION (Target Overlap + ATC)")
    print("=" * 70)

    # Load ATC mapper if available
    atc_mapper = None
    if ATC_AVAILABLE:
        try:
            atc_mapper = ATCMapper()
            print("ATC mapper loaded successfully")
        except FileNotFoundError:
            print("Warning: ATC data file not found, skipping ATC features")

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

    mesh_mappings: Dict[str, str] = {}
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

    # Reverse mapping for ATC lookup
    id_to_drug_name = {f"drkg:Compound::{db_id}": name for db_id, name in id_to_name.items()}

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

    # Define strategies
    # Strategy signature: (base_score, target_overlap, target_frac, atc_score) -> boosted_score

    strategies: Dict[str, Callable] = {
        # Baseline
        "baseline": lambda s, o, f, a: s,

        # Target only (current best)
        "target_only": lambda s, o, f, a: s * (1 + 0.01 * min(o, 10)),

        # ATC only
        "atc_only_5%": lambda s, o, f, a: s * (1 + 0.05 * a) if a > 0.5 else s,
        "atc_only_10%": lambda s, o, f, a: s * (1 + 0.10 * a) if a > 0.5 else s,

        # Combined: Target + ATC additive
        "combined_add": lambda s, o, f, a: s * (1 + 0.01 * min(o, 10) + 0.05 * a),

        # Combined: Target primary, ATC secondary
        "combined_tiered": lambda s, o, f, a: s * (1 + 0.01 * min(o, 10)) * (1 + 0.03 * a),

        # Combined: Max boost
        "combined_max": lambda s, o, f, a: s * (1 + max(0.01 * min(o, 10), 0.05 * a)),

        # Combined: Only ATC if no target overlap
        "atc_fallback": lambda s, o, f, a: s * (1 + 0.01 * min(o, 10)) if o > 0 else s * (1 + 0.10 * a),
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
        gt_local_indices: Set[int] = set()
        for drug_info in disease_data['drugs']:
            drug_name_lower = drug_info['name'].lower()
            drug_id = name_to_id.get(drug_name_lower)
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

        # Compute target overlap and ATC scores for each drug
        overlaps = []
        fracs = []
        atc_scores = []

        for drug_id in valid_drug_ids:
            db_id = drug_id.split("::")[-1]

            # Target overlap
            drug_genes = drug_targets.get(db_id, set())
            overlap = len(drug_genes & dis_genes)
            frac = overlap / len(drug_genes) if len(drug_genes) > 0 else 0
            overlaps.append(overlap)
            fracs.append(frac)

            # ATC score
            if atc_mapper:
                drug_name = id_to_drug_name.get(drug_id, "")
                atc_score = atc_mapper.get_mechanism_score(drug_name, disease_name)
            else:
                atc_score = 0.0
            atc_scores.append(atc_score)

        overlaps_arr = np.array(overlaps)
        fracs_arr = np.array(fracs)
        atc_scores_arr = np.array(atc_scores)

        # Evaluate each strategy
        for name, strategy_fn in strategies.items():
            scores = np.array([
                strategy_fn(base_scores[i], overlaps_arr[i], fracs_arr[i], atc_scores_arr[i])
                for i in range(n_drugs)
            ])
            top_30 = set(np.argsort(scores)[-30:])
            hits = len(top_30 & gt_local_indices)
            results[name]["hits"] += hits

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    baseline_recall = results["baseline"]["hits"] / results["baseline"]["total"]

    print(f"\n{'Strategy':<25} {'R@30':>10} {'Hits/Total':>15} {'vs Baseline':>12} {'vs Target':>12}")
    print("-" * 75)

    target_recall = results["target_only"]["hits"] / results["target_only"]["total"]

    for name in strategies:
        total = results[name]["total"]
        hits = results[name]["hits"]
        recall = hits / total if total > 0 else 0
        vs_baseline = recall - baseline_recall
        vs_target = recall - target_recall
        vs_baseline_str = f"{vs_baseline:+.2%}" if name != "baseline" else "-"
        vs_target_str = f"{vs_target:+.2%}" if name != "target_only" else "-"
        print(f"{name:<25} {recall:>9.2%} {hits:>6}/{total:<6} {vs_baseline_str:>12} {vs_target_str:>12}")

    # ATC coverage stats
    if atc_mapper:
        print("\n" + "=" * 70)
        print("ATC COVERAGE STATS")
        print("=" * 70)
        drugs_with_atc = sum(1 for did in valid_drug_ids if atc_mapper.get_atc_codes(id_to_drug_name.get(did, "")))
        print(f"Drugs with ATC mapping: {drugs_with_atc}/{len(valid_drug_ids)} ({drugs_with_atc/len(valid_drug_ids):.1%})")


if __name__ == "__main__":
    main()
