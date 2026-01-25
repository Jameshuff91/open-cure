#!/usr/bin/env python3
"""
Evaluate chemical structure similarity boosting for drug repurposing.

Tests whether boosting predictions based on structural similarity to known
treatments improves Recall@30.
"""

import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from tqdm import tqdm

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chemical_features import DrugFingerprinter, compute_tanimoto_similarity


def load_resources():
    """Load all required data."""
    print("Loading resources...")

    # Ground truth
    with open('data/reference/expanded_ground_truth.json') as f:
        ground_truth = json.load(f)

    # DrugBank lookup
    with open('data/reference/drugbank_lookup.json') as f:
        drugbank_lookup = json.load(f)

    # MESH mappings
    with open('data/reference/mesh_mappings_from_agents.json') as f:
        mesh_mappings = json.load(f)

    # Load model
    with open('models/drug_repurposing_gb_enhanced.pkl', 'rb') as f:
        model = pickle.load(f)

    # Load embeddings
    import torch
    checkpoint = torch.load('models/transe.pt', map_location='cpu', weights_only=False)

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

    return ground_truth, drugbank_lookup, mesh_mappings, model, embeddings, entity2id


def get_drug_index(drug_id: str, entity2id: Dict) -> int:
    """Get embedding index for a drug (DrugBank ID)."""
    key = f"drkg:Compound::{drug_id}"
    return entity2id.get(key, -1)


def get_disease_index(mesh_id: str, entity2id: Dict) -> int:
    """Get embedding index for a disease (MESH ID)."""
    key = f"drkg:Disease::MESH:{mesh_id}"
    return entity2id.get(key, -1)


def flatten_mesh_mappings(mesh_mappings: Dict) -> Dict[str, str]:
    """Flatten mesh_mappings from batched format to disease_name -> mesh_id."""
    flat = {}
    for batch_name, batch_data in mesh_mappings.items():
        if isinstance(batch_data, dict):
            for disease_name, mesh_id in batch_data.items():
                if isinstance(mesh_id, str) and (mesh_id.startswith('D') or mesh_id.startswith('C')):
                    flat[disease_name] = mesh_id
    return flat


def evaluate_with_chemical_boost(
    ground_truth: Dict,
    drugbank_lookup: Dict,
    mesh_mappings: Dict,
    model,
    embeddings: np.ndarray,
    entity2id: Dict,
    fingerprinter: DrugFingerprinter,
    boost_strategies: Dict[str, callable],
) -> Dict[str, Dict]:
    """
    Evaluate different chemical similarity boosting strategies.

    Returns dict mapping strategy name to results.
    """
    results = {name: {'hits': 0, 'total': 0} for name in boost_strategies}

    # Flatten mesh mappings
    disease_to_mesh = flatten_mesh_mappings(mesh_mappings)
    print(f"Disease to MESH mappings: {len(disease_to_mesh)}")

    # Get drugs that have fingerprints
    drugs_with_fp = set()
    for drug_id, drug_name in drugbank_lookup.items():
        if fingerprinter.get_fingerprint(drug_name, fetch_if_missing=False) is not None:
            drugs_with_fp.add(drug_id)

    print(f"Drugs with fingerprints: {len(drugs_with_fp)}/{len(drugbank_lookup)}")

    # Evaluate each disease
    diseases_evaluated = 0

    for disease_name, mesh_id in tqdm(disease_to_mesh.items(), desc="Diseases"):
        # Build the full MESH key used in ground truth
        gt_key = f"drkg:Disease::MESH:{mesh_id}"

        # Get ground truth drugs for this disease
        gt_drugs = set(ground_truth.get(gt_key, []))
        if not gt_drugs:
            continue

        # Get disease embedding
        disease_idx = get_disease_index(mesh_id, entity2id)
        if disease_idx < 0:
            continue

        disease_emb = embeddings[disease_idx]

        # Extract DrugBank IDs from GT drugs (strip drkg:Compound:: prefix)
        gt_drugbank_ids = set()
        for gt_drug in gt_drugs:
            if gt_drug.startswith('drkg:Compound::DB'):
                db_id = gt_drug.replace('drkg:Compound::', '')
                gt_drugbank_ids.add(db_id)

        if not gt_drugbank_ids:
            continue

        # Get known treatment names (for fingerprint comparison)
        known_treatment_names = [
            drugbank_lookup.get(db_id, db_id) for db_id in gt_drugbank_ids
            if db_id in drugbank_lookup
        ]

        # Score all drugs
        drug_scores = {}

        for drug_id, drug_name in drugbank_lookup.items():
            # Get drug embedding
            drug_idx = get_drug_index(drug_id, entity2id)
            if drug_idx < 0:
                continue

            drug_emb = embeddings[drug_idx]

            # Create feature vector (concat + product + diff)
            concat_feats = np.concatenate([drug_emb, disease_emb])
            product_feats = drug_emb * disease_emb
            diff_feats = drug_emb - disease_emb
            features = np.hstack([concat_feats, product_feats, diff_feats]).reshape(1, -1)

            # Get base score
            base_score = model.predict_proba(features)[0, 1]

            # Get chemical similarity to known treatments
            max_sim = 0.0
            query_fp = fingerprinter.get_fingerprint(drug_name, fetch_if_missing=False)

            if query_fp is not None:
                for known_drug in known_treatment_names:
                    known_fp = fingerprinter.get_fingerprint(known_drug, fetch_if_missing=False)
                    if known_fp is not None:
                        sim = compute_tanimoto_similarity(query_fp, known_fp)
                        max_sim = max(max_sim, sim)

            drug_scores[drug_id] = {
                'base_score': base_score,
                'max_sim': max_sim,
                'has_fp': query_fp is not None,
            }

        # Apply boost strategies and evaluate
        for strategy_name, boost_fn in boost_strategies.items():
            # Apply boost
            boosted_scores = []
            for drug_id, data in drug_scores.items():
                boosted = boost_fn(data['base_score'], data['max_sim'], data['has_fp'])
                boosted_scores.append((drug_id, boosted))

            # Sort by boosted score
            boosted_scores.sort(key=lambda x: x[1], reverse=True)

            # Get top 30
            top30_ids = set(d[0] for d in boosted_scores[:30])

            # Count hits (compare against DrugBank IDs)
            hits = len(top30_ids & gt_drugbank_ids)
            results[strategy_name]['hits'] += hits
            results[strategy_name]['total'] += len(gt_drugbank_ids)

        diseases_evaluated += 1

    print(f"\nDiseases evaluated: {diseases_evaluated}")

    return results


def main():
    print("=" * 70)
    print("CHEMICAL STRUCTURE BOOST EVALUATION")
    print("=" * 70)

    # Load resources
    gt, drugbank, mesh_map, model, embeddings, entity2id = load_resources()

    # Initialize fingerprinter
    print("\nInitializing fingerprinter...")
    fingerprinter = DrugFingerprinter(use_cache=True)

    # Check coverage
    all_drugs = list(drugbank.values())
    stats = fingerprinter.get_coverage_stats(all_drugs)
    print(f"Current fingerprint coverage: {stats['fp_coverage']:.1%} ({stats['with_fingerprints']}/{stats['total_drugs']})")

    # If coverage is low, fetch more fingerprints
    if stats['fp_coverage'] < 0.1:
        print("\nCoverage is low. Fetching fingerprints for top drugs...")

        # Get drugs that appear most frequently in ground truth
        gt_drug_counts: Dict[str, int] = {}
        for mesh_id, drugs in gt.items():
            for drug_id in drugs:
                drug_name = drugbank.get(drug_id, drug_id)
                gt_drug_counts[drug_name] = gt_drug_counts.get(drug_name, 0) + 1

        # Get top 1000 most common GT drugs
        top_gt_drugs = sorted(gt_drug_counts.keys(), key=lambda x: gt_drug_counts[x], reverse=True)[:1000]

        # Also get a sample of other drugs
        other_drugs = [name for name in all_drugs if name not in gt_drug_counts][:1000]

        drugs_to_fetch = list(set(top_gt_drugs + other_drugs))
        print(f"Fetching fingerprints for {len(drugs_to_fetch)} drugs...")

        fingerprinter.precompute_fingerprints(drugs_to_fetch, fetch_missing=True)

        # Recheck coverage
        stats = fingerprinter.get_coverage_stats(all_drugs)
        print(f"Updated coverage: {stats['fp_coverage']:.1%}")

    # Define boost strategies
    def baseline(base_score, max_sim, has_fp):
        return base_score

    def boost_if_similar(base_score, max_sim, has_fp):
        # Boost by 10% if similar drug exists (Tanimoto > 0.5)
        if max_sim > 0.5:
            return base_score * 1.1
        return base_score

    def boost_by_similarity(base_score, max_sim, has_fp):
        # Boost proportionally to max similarity
        return base_score * (1 + 0.2 * max_sim)

    def boost_high_similarity(base_score, max_sim, has_fp):
        # Only boost if very similar (> 0.7)
        if max_sim > 0.7:
            return base_score * 1.2
        return base_score

    def additive_boost(base_score, max_sim, has_fp):
        # Add a fixed bonus for similar drugs
        if max_sim > 0.5:
            return base_score + 0.1
        return base_score

    strategies = {
        'baseline': baseline,
        'boost_if_similar_0.5': boost_if_similar,
        'boost_by_sim_0.2x': boost_by_similarity,
        'boost_high_sim_0.7': boost_high_similarity,
        'additive_0.1': additive_boost,
    }

    # Evaluate
    print("\nEvaluating strategies...")
    results = evaluate_with_chemical_boost(
        gt, drugbank, mesh_map, model, embeddings, entity2id,
        fingerprinter, strategies
    )

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"{'Strategy':<25} {'R@30':>10} {'Hits/Total':>15} {'vs Baseline':>12}")
    print("-" * 65)

    baseline_r30 = results['baseline']['hits'] / results['baseline']['total'] if results['baseline']['total'] > 0 else 0

    for name, data in results.items():
        r30 = data['hits'] / data['total'] if data['total'] > 0 else 0
        diff = r30 - baseline_r30

        diff_str = f"{diff:+.2%}" if name != 'baseline' else "-"

        print(f"{name:<25} {r30:>9.2%} {data['hits']:>6}/{data['total']:<8} {diff_str:>12}")


if __name__ == '__main__':
    main()
