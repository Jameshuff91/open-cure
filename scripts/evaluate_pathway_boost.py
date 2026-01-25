#!/usr/bin/env python3
"""
Evaluate pathway enrichment boosting for drug repurposing.

Tests whether boosting predictions based on drug-disease pathway overlap
improves Recall@30, and how it combines with existing boosts.
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

from pathway_features import PathwayEnrichment

try:
    from atc_features import ATCMapper
    ATC_AVAILABLE = True
except ImportError:
    ATC_AVAILABLE = False

try:
    from chemical_features import DrugFingerprinter, compute_tanimoto_similarity
    CHEM_AVAILABLE = True
except ImportError:
    CHEM_AVAILABLE = False

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
MODELS_DIR = PROJECT_ROOT / "models"


def main() -> None:
    print("=" * 70)
    print("PATHWAY BOOST EVALUATION")
    print("=" * 70)

    # Load pathway enrichment
    print("\n1. Loading pathway data...")
    pe = PathwayEnrichment()
    stats = pe.get_coverage_stats()
    print(f"   Drugs with pathways: {stats['drugs_with_pathways']}/{stats['total_drugs']}")
    print(f"   Diseases with pathways: {stats['diseases_with_pathways']}/{stats['total_diseases']}")

    # Load other feature modules
    atc_mapper = None
    if ATC_AVAILABLE:
        try:
            atc_mapper = ATCMapper()
            print("✓ ATC mapper loaded")
        except FileNotFoundError:
            print("✗ ATC data not found")

    fingerprinter = None
    if CHEM_AVAILABLE:
        try:
            fingerprinter = DrugFingerprinter(use_cache=True)
            print("✓ Chemical fingerprinter loaded")
        except Exception as e:
            print(f"✗ Fingerprinter error: {e}")

    # Load baseline model
    print("\n2. Loading baseline model...")
    with open(MODELS_DIR / "drug_repurposing_gb_enhanced.pkl", "rb") as f:
        model = pickle.load(f)

    # Load embeddings
    print("3. Loading embeddings...")
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
    print("4. Loading target data...")
    with open(REFERENCE_DIR / "drug_targets.json") as f:
        drug_targets_raw = json.load(f)
    drug_targets = {k: set(v) for k, v in drug_targets_raw.items()}

    with open(REFERENCE_DIR / "disease_genes.json") as f:
        disease_genes_raw = json.load(f)
    disease_genes = {k: set(v) for k, v in disease_genes_raw.items()}

    # Load mappings
    print("5. Loading mappings...")
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
    id_to_drug_name = {f"drkg:Compound::{db_id}": name for db_id, name in id_to_name.items()}

    # Load ground truth
    with open(REFERENCE_DIR / "everycure_gt_for_txgnn.json") as f:
        gt_raw = json.load(f)

    # Convert GT format
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

    # Get all drug IDs
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

    # Define boost strategies
    # Params: (base_score, target_overlap, atc_score, chem_sim, pathway_overlap, pathway_jaccard)
    strategies: Dict[str, Callable] = {
        # Baseline
        "baseline": lambda s, o, a, c, po, pj: s,

        # Individual boosts
        "target_only": lambda s, o, a, c, po, pj: s * (1 + 0.01 * min(o, 10)),
        "pathway_only": lambda s, o, a, c, po, pj: s * (1 + 0.01 * min(po, 10)),
        "pathway_jaccard": lambda s, o, a, c, po, pj: s * (1 + 0.2 * pj),

        # Pathway + target
        "target+pathway": lambda s, o, a, c, po, pj: s * (1 + 0.01 * min(o, 10) + 0.01 * min(po, 10)),
        "target+pathway_j": lambda s, o, a, c, po, pj: s * (1 + 0.01 * min(o, 10) + 0.2 * pj),

        # Current best (triple) for comparison
        "triple": lambda s, o, a, c, po, pj: s * (1 + 0.01 * min(o, 10)) * (1 + 0.05 * a) * (1.2 if c > 0.7 else 1.0),

        # Quadruple: Triple + Pathway
        "quad_additive": lambda s, o, a, c, po, pj: s * (1 + 0.01 * min(o, 10) + 0.05 * a + 0.01 * min(po, 10)) * (1.2 if c > 0.7 else 1.0),
        "quad_multiplicative": lambda s, o, a, c, po, pj: s * (1 + 0.01 * min(o, 10)) * (1 + 0.05 * a) * (1 + 0.01 * min(po, 10)) * (1.2 if c > 0.7 else 1.0),
        "quad_jaccard": lambda s, o, a, c, po, pj: s * (1 + 0.01 * min(o, 10)) * (1 + 0.05 * a) * (1 + 0.1 * pj) * (1.2 if c > 0.7 else 1.0),
    }

    results = {name: {"hits": 0, "total": 0} for name in strategies}

    print("\n6. Evaluating strategies...")
    diseases_evaluated = 0

    for mesh_full_id, gt_drugs in tqdm(gt.items(), desc="Diseases"):
        mesh_id = mesh_full_id.split("MESH:")[-1]

        disease_idx = entity2id.get(mesh_full_id)
        if disease_idx is None:
            continue

        disease_emb = embeddings[disease_idx]

        # Get disease genes and pathways
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

        # Get disease name for ATC lookup
        disease_name = ""
        for dn, mid in mesh_mappings.items():
            if mid == mesh_full_id:
                disease_name = dn
                break

        # Pre-compute GT fingerprints
        gt_fingerprints = []
        if fingerprinter:
            for drug_name in gt_drug_names:
                fp = fingerprinter.get_fingerprint(drug_name, fetch_if_missing=False)
                if fp is not None:
                    gt_fingerprints.append(fp)

        # Score all drugs
        n_drugs = len(drug_embs)
        disease_emb_tiled = np.tile(disease_emb, (n_drugs, 1))

        concat_feats = np.hstack([drug_embs, disease_emb_tiled])
        product_feats = drug_embs * disease_emb_tiled
        diff_feats = drug_embs - disease_emb_tiled
        base_features = np.hstack([concat_feats, product_feats, diff_feats])

        base_scores = model.predict_proba(base_features)[:, 1]

        # Compute features for each drug
        overlaps = np.zeros(n_drugs)
        atc_scores = np.zeros(n_drugs)
        chem_sims = np.zeros(n_drugs)
        pathway_overlaps = np.zeros(n_drugs)
        pathway_jaccards = np.zeros(n_drugs)

        for i, drug_id in enumerate(valid_drug_ids):
            db_id = drug_id.split("::")[-1]
            drug_name = id_to_drug_name.get(drug_id, "")

            # Target overlap
            drug_genes = drug_targets.get(db_id, set())
            overlaps[i] = len(drug_genes & dis_genes)

            # ATC score
            if atc_mapper and disease_name:
                atc_scores[i] = atc_mapper.get_mechanism_score(drug_name, disease_name)

            # Chemical similarity
            if fingerprinter and gt_fingerprints:
                query_fp = fingerprinter.get_fingerprint(drug_name, fetch_if_missing=False)
                if query_fp is not None:
                    max_sim = 0.0
                    for gt_fp in gt_fingerprints:
                        sim = compute_tanimoto_similarity(query_fp, gt_fp)
                        max_sim = max(max_sim, sim)
                    chem_sims[i] = max_sim

            # Pathway overlap
            po, pj, _ = pe.get_pathway_overlap(db_id, f"MESH:{mesh_id}")
            pathway_overlaps[i] = po
            pathway_jaccards[i] = pj

        # Evaluate each strategy
        for name, strategy_fn in strategies.items():
            boosted_scores = np.array([
                strategy_fn(
                    base_scores[i],
                    int(overlaps[i]),
                    atc_scores[i],
                    chem_sims[i],
                    int(pathway_overlaps[i]),
                    pathway_jaccards[i],
                )
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
    triple_recall = results["triple"]["hits"] / results["triple"]["total"] if results["triple"]["total"] > 0 else 0

    print(f"\n{'Strategy':<25} {'R@30':>10} {'Hits/Total':>15} {'vs Base':>10} {'vs Triple':>12}")
    print("-" * 75)

    # Sort by R@30
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
        vs_triple = recall - triple_recall

        vs_baseline_str = f"{vs_baseline:+.2%}" if name != "baseline" else "-"
        vs_triple_str = f"{vs_triple:+.2%}" if name != "triple" else "-"

        print(f"{name:<25} {recall:>9.2%} {hits:>6}/{total:<6} {vs_baseline_str:>10} {vs_triple_str:>12}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    best_strategy = sorted_strategies[0]
    best_recall = results[best_strategy]["hits"] / results[best_strategy]["total"]

    print(f"\nBest strategy: {best_strategy}")
    print(f"Best R@30: {best_recall:.2%}")
    print(f"Improvement over baseline: {best_recall - baseline_recall:+.2%}")
    print(f"Improvement over triple: {best_recall - triple_recall:+.2%}")


if __name__ == "__main__":
    main()
