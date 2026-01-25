#!/usr/bin/env python3
"""
Evaluate combined triple boosting: Target + ATC + Chemical Similarity.

Tests whether combining all three boost strategies provides additional
improvement over individual or pair-wise combinations.
"""

import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Set, Callable, Any

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

try:
    from chemical_features import DrugFingerprinter, compute_tanimoto_similarity
    CHEM_AVAILABLE = True
except ImportError:
    CHEM_AVAILABLE = False
    print("Warning: Chemical features not available")

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
MODELS_DIR = PROJECT_ROOT / "models"


def main() -> None:
    print("=" * 70)
    print("TRIPLE BOOST EVALUATION (Target + ATC + Chemical)")
    print("=" * 70)

    # Load ATC mapper
    atc_mapper = None
    if ATC_AVAILABLE:
        try:
            atc_mapper = ATCMapper()
            print("✓ ATC mapper loaded")
        except FileNotFoundError:
            print("✗ ATC data file not found")

    # Load fingerprinter
    fingerprinter = None
    if CHEM_AVAILABLE:
        try:
            fingerprinter = DrugFingerprinter(use_cache=True)
            print("✓ Chemical fingerprinter loaded")
        except Exception as e:
            print(f"✗ Fingerprinter error: {e}")

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
                    if mesh_str.startswith("D") or mesh_str.startswith("C"):
                        mesh_mappings[disease_name.lower()] = f"drkg:Disease::MESH:{mesh_str}"

    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)
    name_to_id = {name.lower(): f"drkg:Compound::{db_id}" for db_id, name in id_to_name.items()}

    # Reverse mapping for lookups
    id_to_drug_name = {f"drkg:Compound::{db_id}": name for db_id, name in id_to_name.items()}

    # Load ground truth (use everycure GT for consistent comparison with previous results)
    with open(REFERENCE_DIR / "everycure_gt_for_txgnn.json") as f:
        gt_raw = json.load(f)

    # Convert to same format as expanded_ground_truth
    # everycure format: {disease_name: {drugs: [{name: ...}]}}
    # needed format: {mesh_id: [drug_ids]}
    gt = {}
    for disease_name, disease_data in gt_raw.items():
        mesh_id = mesh_mappings.get(disease_name.lower())
        if mesh_id:
            drug_ids = []
            for drug_info in disease_data.get('drugs', []):
                drug_name_lower = drug_info['name'].lower()
                drug_id = name_to_id.get(drug_name_lower)
                if drug_id:
                    drug_ids.append(drug_id)
            if drug_ids:
                gt[mesh_id] = drug_ids

    # Get all drug IDs with embeddings
    all_drug_ids = [eid for eid in entity2id.keys() if "Compound" in eid]

    valid_drug_ids = []
    valid_drug_indices = []
    for drug_id in all_drug_ids:
        drug_idx = entity2id.get(drug_id)
        if drug_idx is not None:
            valid_drug_ids.append(drug_id)
            valid_drug_indices.append(drug_idx)

    drug_embs = embeddings[valid_drug_indices]
    drug_id_to_local_idx = {did: i for i, did in enumerate(valid_drug_ids)}

    print(f"   Drugs with embeddings: {len(valid_drug_ids)}")

    # Pre-compute chemical fingerprints availability
    drug_has_fp = {}
    if fingerprinter:
        for drug_id in valid_drug_ids:
            drug_name = id_to_drug_name.get(drug_id, "")
            fp = fingerprinter.get_fingerprint(drug_name, fetch_if_missing=False)
            drug_has_fp[drug_id] = fp is not None

        fp_count = sum(drug_has_fp.values())
        print(f"   Drugs with fingerprints: {fp_count}/{len(valid_drug_ids)} ({fp_count/len(valid_drug_ids):.1%})")

    # Define boost strategies
    # Params: (base_score, target_overlap, atc_score, chem_sim)
    strategies: Dict[str, Callable[[float, int, float, float], float]] = {
        # Baseline
        "baseline": lambda s, o, a, c: s,

        # Individual boosts
        "target_only": lambda s, o, a, c: s * (1 + 0.01 * min(o, 10)),
        "atc_only": lambda s, o, a, c: s * (1 + 0.05 * a),
        "chem_only": lambda s, o, a, c: s * 1.2 if c > 0.7 else s,

        # Pair-wise combinations
        "target+atc": lambda s, o, a, c: s * (1 + 0.01 * min(o, 10) + 0.05 * a),
        "target+chem": lambda s, o, a, c: s * (1 + 0.01 * min(o, 10)) * (1.2 if c > 0.7 else 1.0),
        "atc+chem": lambda s, o, a, c: s * (1 + 0.05 * a) * (1.2 if c > 0.7 else 1.0),

        # Triple combinations
        "triple_additive": lambda s, o, a, c: s * (1 + 0.01 * min(o, 10) + 0.05 * a + (0.2 if c > 0.7 else 0)),
        "triple_multiplicative": lambda s, o, a, c: s * (1 + 0.01 * min(o, 10)) * (1 + 0.05 * a) * (1.2 if c > 0.7 else 1.0),
        "triple_max": lambda s, o, a, c: s * (1 + max(0.01 * min(o, 10), 0.05 * a, 0.2 if c > 0.7 else 0)),

        # Alternative thresholds
        "triple_chem_0.5": lambda s, o, a, c: s * (1 + 0.01 * min(o, 10) + 0.05 * a + (0.1 if c > 0.5 else 0)),
        "triple_chem_scaled": lambda s, o, a, c: s * (1 + 0.01 * min(o, 10) + 0.05 * a + 0.1 * c),
    }

    results = {name: {"hits": 0, "total": 0} for name in strategies}

    print("\n5. Evaluating strategies...")
    diseases_evaluated = 0

    for mesh_full_id, gt_drugs in tqdm(gt.items(), desc="Diseases"):
        if not mesh_full_id.startswith("drkg:Disease::MESH:"):
            continue

        mesh_id = mesh_full_id.split("MESH:")[-1]

        disease_idx = entity2id.get(mesh_full_id)
        if disease_idx is None:
            continue

        disease_emb = embeddings[disease_idx]

        # Get disease genes
        dis_genes = disease_genes.get(f"MESH:{mesh_id}", set())

        # Get GT drugs as local indices
        gt_local_indices: Set[int] = set()
        gt_drug_names = []
        for drug_id in gt_drugs:
            if drug_id in drug_id_to_local_idx:
                gt_local_indices.add(drug_id_to_local_idx[drug_id])
                gt_drug_names.append(id_to_drug_name.get(drug_id, ""))

        if not gt_local_indices:
            continue

        diseases_evaluated += 1

        for name in strategies:
            results[name]["total"] += len(gt_local_indices)

        # Score all drugs
        n_drugs = len(drug_embs)
        disease_emb_tiled = np.tile(disease_emb, (n_drugs, 1))

        concat_feats = np.hstack([drug_embs, disease_emb_tiled])
        product_feats = drug_embs * disease_emb_tiled
        diff_feats = drug_embs - disease_emb_tiled
        base_features = np.hstack([concat_feats, product_feats, diff_feats])

        base_scores = model.predict_proba(base_features)[:, 1]

        # Pre-compute GT drug fingerprints for similarity comparison
        gt_fingerprints = []
        if fingerprinter:
            for drug_name in gt_drug_names:
                fp = fingerprinter.get_fingerprint(drug_name, fetch_if_missing=False)
                if fp is not None:
                    gt_fingerprints.append(fp)

        # Compute features for each drug
        overlaps = np.zeros(n_drugs)
        atc_scores = np.zeros(n_drugs)
        chem_sims = np.zeros(n_drugs)

        # Get disease name for ATC lookup (find reverse mapping)
        disease_name = ""
        for dn, mid in mesh_mappings.items():
            if mid == mesh_full_id:
                disease_name = dn
                break

        for i, drug_id in enumerate(valid_drug_ids):
            db_id = drug_id.split("::")[-1]
            drug_name = id_to_drug_name.get(drug_id, "")

            # Target overlap
            drug_genes = drug_targets.get(db_id, set())
            overlaps[i] = len(drug_genes & dis_genes)

            # ATC score
            if atc_mapper and disease_name:
                atc_scores[i] = atc_mapper.get_mechanism_score(drug_name, disease_name)

            # Chemical similarity (max to any GT drug)
            if fingerprinter and gt_fingerprints:
                query_fp = fingerprinter.get_fingerprint(drug_name, fetch_if_missing=False)
                if query_fp is not None:
                    max_sim = 0.0
                    for gt_fp in gt_fingerprints:
                        sim = compute_tanimoto_similarity(query_fp, gt_fp)
                        max_sim = max(max_sim, sim)
                    chem_sims[i] = max_sim

        # Evaluate each strategy
        for name, strategy_fn in strategies.items():
            boosted_scores = np.array([
                strategy_fn(base_scores[i], int(overlaps[i]), atc_scores[i], chem_sims[i])
                for i in range(n_drugs)
            ])
            top_30 = set(np.argsort(boosted_scores)[-30:])
            hits = len(top_30 & gt_local_indices)
            results[name]["hits"] += hits

    # Print results
    print(f"\nDiseases evaluated: {diseases_evaluated}")
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    baseline_recall = results["baseline"]["hits"] / results["baseline"]["total"] if results["baseline"]["total"] > 0 else 0
    target_recall = results["target_only"]["hits"] / results["target_only"]["total"] if results["target_only"]["total"] > 0 else 0

    print(f"\n{'Strategy':<25} {'R@30':>10} {'Hits/Total':>15} {'vs Base':>10} {'vs Target+ATC':>12}")
    print("-" * 75)

    target_atc_recall = results["target+atc"]["hits"] / results["target+atc"]["total"] if results["target+atc"]["total"] > 0 else 0

    # Sort by R@30 descending
    sorted_strategies = sorted(
        strategies.keys(),
        key=lambda n: results[n]["hits"] / results[n]["total"] if results[n]["total"] > 0 else 0,
        reverse=True
    )

    for name in sorted_strategies:
        total = results[name]["total"]
        hits = results[name]["hits"]
        recall = hits / total if total > 0 else 0
        vs_baseline = recall - baseline_recall
        vs_target_atc = recall - target_atc_recall

        vs_baseline_str = f"{vs_baseline:+.2%}" if name != "baseline" else "-"
        vs_ta_str = f"{vs_target_atc:+.2%}" if name != "target+atc" else "-"

        print(f"{name:<25} {recall:>9.2%} {hits:>6}/{total:<6} {vs_baseline_str:>10} {vs_ta_str:>12}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    best_strategy = sorted_strategies[0]
    best_recall = results[best_strategy]["hits"] / results[best_strategy]["total"]

    print(f"\nBest strategy: {best_strategy}")
    print(f"Best R@30: {best_recall:.2%}")
    print(f"Improvement over baseline: {best_recall - baseline_recall:+.2%}")
    print(f"Improvement over target+atc: {best_recall - target_atc_recall:+.2%}")

    # Feature coverage
    print("\n" + "=" * 70)
    print("FEATURE COVERAGE")
    print("=" * 70)

    if fingerprinter:
        fp_count = sum(drug_has_fp.values())
        print(f"Chemical fingerprints: {fp_count}/{len(valid_drug_ids)} ({fp_count/len(valid_drug_ids):.1%})")

    if atc_mapper:
        atc_count = sum(1 for did in valid_drug_ids if atc_mapper.get_atc_codes(id_to_drug_name.get(did, "")))
        print(f"ATC codes: {atc_count}/{len(valid_drug_ids)} ({atc_count/len(valid_drug_ids):.1%})")

    target_count = sum(1 for did in valid_drug_ids if drug_targets.get(did.split("::")[-1]))
    print(f"Target annotations: {target_count}/{len(valid_drug_ids)} ({target_count/len(valid_drug_ids):.1%})")


if __name__ == "__main__":
    main()
