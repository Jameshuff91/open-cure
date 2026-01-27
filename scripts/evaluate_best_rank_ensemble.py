#!/usr/bin/env python3
"""
Evaluate GB + TxGNN Best-Rank Ensemble (Hypothesis h1)

This script tests the hypothesis that taking min(GB_rank, TxGNN_rank) for each
drug-disease pair will improve Recall@30 beyond either model alone.

Baseline: GB model achieves 41.8% R@30
Target: >43% R@30 (>1.2% improvement)

Key insight from archive:
- TxGNN excels at storage diseases (83.3% R@30)
- GB is better overall (41.8% vs TxGNN's 14.5%)
- Previous ensemble achieved only 7.5% - likely due to mismatched disease sets
"""

import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
MODELS_DIR = PROJECT_ROOT / "models"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"

ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def create_features(drug_emb: np.ndarray, disease_emb: np.ndarray) -> np.ndarray:
    """Create features from embeddings for GB model."""
    concat = np.concatenate([drug_emb, disease_emb])
    product = drug_emb * disease_emb
    diff = drug_emb - disease_emb
    return np.concatenate([concat, product, diff])


def load_gb_model_and_embeddings():
    """Load GB model and TransE embeddings."""
    print("Loading GB model...")
    with open(MODELS_DIR / "drug_repurposing_gb_enhanced.pkl", "rb") as f:
        gb_model = pickle.load(f)

    print("Loading TransE embeddings...")
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
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Entities: {len(entity2id)}")

    return gb_model, embeddings, entity2id


def load_txgnn_predictions() -> pd.DataFrame:
    """Load TxGNN predictions with disease names and drug IDs."""
    print("Loading TxGNN predictions...")

    # Use final predictions file which has disease names
    txgnn_path = REFERENCE_DIR / "txgnn_predictions_final.csv"

    # Read carefully - some disease names have quotes/commas
    df = pd.read_csv(txgnn_path, on_bad_lines='skip')

    # Clean disease names
    df['disease_name'] = df['disease_name'].str.strip().str.lower()

    # Get unique diseases
    unique_diseases = df['disease_name'].nunique()
    print(f"  TxGNN diseases: {unique_diseases}")
    print(f"  TxGNN predictions: {len(df)}")

    return df


def load_disease_mappings() -> Dict[str, str]:
    """Load mappings from disease names to MESH IDs."""
    print("Loading disease mappings...")

    mappings = {}

    # Load MONDO to MESH mappings
    mondo_mesh_path = REFERENCE_DIR / "mondo_to_mesh.json"
    if mondo_mesh_path.exists():
        with open(mondo_mesh_path) as f:
            mondo_mesh = json.load(f)
        # This maps MONDO IDs to MESH IDs, we need disease names
        # For now, skip this complex mapping

    # Load mesh_mappings_from_agents
    mesh_agents_path = REFERENCE_DIR / "mesh_mappings_from_agents.json"
    if mesh_agents_path.exists():
        with open(mesh_agents_path) as f:
            mesh_data = json.load(f)
        for batch_data in mesh_data.values():
            if isinstance(batch_data, dict):
                for disease_name, mesh_id in batch_data.items():
                    if mesh_id:
                        mesh_str = str(mesh_id)
                        if mesh_str.startswith("D") or mesh_str.startswith("C"):
                            mappings[disease_name.lower().strip()] = f"drkg:Disease::MESH:{mesh_str}"

    # Load disease ontology mapping
    doid_path = REFERENCE_DIR / "disease_ontology_mapping.json"
    if doid_path.exists():
        with open(doid_path) as f:
            doid_mapping = json.load(f)
        for mesh_id, info in doid_mapping.items():
            if isinstance(info, dict) and 'name' in info:
                name = info['name'].lower().strip()
                mappings[name] = f"drkg:Disease::MESH:{mesh_id}"

    print(f"  Disease name -> MESH mappings: {len(mappings)}")
    return mappings


def load_ground_truth() -> Tuple[Dict[str, List[str]], Dict[str, str], Dict[str, str]]:
    """Load ground truth drug-disease pairs."""
    print("Loading ground truth...")

    # Load drugbank lookup for name->ID mapping
    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)
    name_to_id = {name.lower().strip(): f"drkg:Compound::{db_id}" for db_id, name in id_to_name.items()}

    # Load Every Cure ground truth
    with open(REFERENCE_DIR / "everycure_gt_for_txgnn.json") as f:
        gt_raw = json.load(f)

    # Load mesh mappings
    disease_mappings = load_disease_mappings()

    gt = {}  # mesh_id -> list of drugbank IDs

    for disease_name, disease_data in gt_raw.items():
        disease_lower = disease_name.lower().strip()
        mesh_id = disease_mappings.get(disease_lower)

        if mesh_id:
            drug_ids = []
            for drug_info in disease_data.get('drugs', []):
                drug_name_lower = drug_info['name'].lower().strip()
                drug_id = name_to_id.get(drug_name_lower)
                if drug_id:
                    drug_ids.append(drug_id)
            if drug_ids:
                gt[mesh_id] = drug_ids

    total_pairs = sum(len(drugs) for drugs in gt.values())
    print(f"  GT diseases with MESH mapping: {len(gt)}")
    print(f"  GT pairs: {total_pairs}")

    return gt, name_to_id, disease_mappings


def get_gb_predictions_for_disease(
    disease_id: str,
    gb_model,
    embeddings: np.ndarray,
    entity2id: Dict[str, int],
    drug_ids: List[str],
    drug_embs: np.ndarray,
) -> Dict[str, Tuple[float, int]]:
    """Get GB model predictions for a disease.

    Returns: dict mapping drug_id -> (score, rank)
    """
    disease_idx = entity2id.get(disease_id)
    if disease_idx is None:
        return {}

    disease_emb = embeddings[disease_idx]

    # Score all drugs
    scores = []
    for i, drug_id in enumerate(drug_ids):
        drug_emb = drug_embs[i]
        features = create_features(drug_emb, disease_emb).reshape(1, -1)
        score = gb_model.predict_proba(features)[0, 1]
        scores.append(score)

    scores = np.array(scores)

    # Rank drugs (higher score = better = lower rank)
    ranked_indices = np.argsort(-scores)  # descending

    results = {}
    for rank, idx in enumerate(ranked_indices, 1):
        drug_id = drug_ids[idx]
        results[drug_id] = (scores[idx], rank)

    return results


def evaluate_models(
    gt: Dict[str, List[str]],
    gb_model,
    embeddings: np.ndarray,
    entity2id: Dict[str, int],
    txgnn_df: pd.DataFrame,
    disease_mappings: Dict[str, str],
    name_to_drugbank: Dict[str, str],
    k: int = 30,
) -> Dict:
    """Evaluate GB, TxGNN, and ensemble on ground truth."""

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
    print(f"Valid drugs with embeddings: {len(valid_drug_ids)}")

    # Create reverse mapping: MESH ID -> disease names that map to it
    mesh_to_names = defaultdict(list)
    for name, mesh_id in disease_mappings.items():
        mesh_to_names[mesh_id].append(name)

    # Create TxGNN lookup: disease_name -> {drug_id -> rank}
    print("Building TxGNN lookup...")
    txgnn_lookup = defaultdict(dict)
    for _, row in txgnn_df.iterrows():
        disease_name = row['disease_name']
        drug_id = f"drkg:Compound::{row['drug_id']}"
        rank = int(row['rank'])
        txgnn_lookup[disease_name][drug_id] = rank

    # Evaluation metrics
    results = {
        'gb': {'hits': 0, 'total': 0, 'diseases': 0},
        'txgnn': {'hits': 0, 'total': 0, 'diseases': 0},
        'ensemble': {'hits': 0, 'total': 0, 'diseases': 0},
        'per_disease': [],
    }

    # Evaluate each disease in ground truth
    diseases_with_both = 0
    diseases_gb_only = 0
    diseases_txgnn_only = 0
    diseases_neither = 0

    for mesh_id, gt_drugs in tqdm(gt.items(), desc="Evaluating diseases"):
        # Check if disease has GB embedding
        has_gb = mesh_id in entity2id

        # Check if disease has TxGNN predictions
        # Need to find disease name that maps to this MESH ID
        txgnn_preds = {}
        disease_names_for_mesh = mesh_to_names.get(mesh_id, [])
        for name in disease_names_for_mesh:
            if name in txgnn_lookup:
                txgnn_preds = txgnn_lookup[name]
                break
        has_txgnn = len(txgnn_preds) > 0

        if not has_gb and not has_txgnn:
            diseases_neither += 1
            continue

        if has_gb and has_txgnn:
            diseases_with_both += 1
        elif has_gb:
            diseases_gb_only += 1
        else:
            diseases_txgnn_only += 1

        # Get GB predictions
        gb_preds = {}
        if has_gb:
            gb_preds = get_gb_predictions_for_disease(
                mesh_id, gb_model, embeddings, entity2id,
                valid_drug_ids, drug_embs
            )

        # Evaluate for each GT drug
        disease_result = {
            'mesh_id': mesh_id,
            'gt_drugs': len(gt_drugs),
            'gb_hits': 0,
            'txgnn_hits': 0,
            'ensemble_hits': 0,
        }

        for drug_id in gt_drugs:
            gb_rank = gb_preds.get(drug_id, (0, float('inf')))[1] if gb_preds else float('inf')
            txgnn_rank = txgnn_preds.get(drug_id, float('inf'))
            ensemble_rank = min(gb_rank, txgnn_rank)

            # GB evaluation
            if gb_preds and gb_rank <= k:
                results['gb']['hits'] += 1
                disease_result['gb_hits'] += 1
            if gb_preds:
                results['gb']['total'] += 1

            # TxGNN evaluation
            if txgnn_preds and txgnn_rank <= k:
                results['txgnn']['hits'] += 1
                disease_result['txgnn_hits'] += 1
            if txgnn_preds:
                results['txgnn']['total'] += 1

            # Ensemble evaluation (only where we have both)
            if gb_preds and txgnn_preds:
                if ensemble_rank <= k:
                    results['ensemble']['hits'] += 1
                    disease_result['ensemble_hits'] += 1
                results['ensemble']['total'] += 1

        if gb_preds:
            results['gb']['diseases'] += 1
        if txgnn_preds:
            results['txgnn']['diseases'] += 1
        if gb_preds and txgnn_preds:
            results['ensemble']['diseases'] += 1

        results['per_disease'].append(disease_result)

    # Add coverage stats
    results['coverage'] = {
        'both_models': diseases_with_both,
        'gb_only': diseases_gb_only,
        'txgnn_only': diseases_txgnn_only,
        'neither': diseases_neither,
    }

    return results


def main():
    print("=" * 70)
    print("GB + TxGNN Best-Rank Ensemble Evaluation (h1)")
    print("=" * 70)
    print()

    # Load models and data
    gb_model, embeddings, entity2id = load_gb_model_and_embeddings()
    txgnn_df = load_txgnn_predictions()
    gt, name_to_drugbank, disease_mappings = load_ground_truth()

    print()
    print("=" * 70)
    print("EVALUATION")
    print("=" * 70)
    print()

    results = evaluate_models(
        gt, gb_model, embeddings, entity2id,
        txgnn_df, disease_mappings, name_to_drugbank
    )

    # Calculate recall
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    print("Model Coverage:")
    print(f"  Diseases with both models: {results['coverage']['both_models']}")
    print(f"  Diseases with GB only: {results['coverage']['gb_only']}")
    print(f"  Diseases with TxGNN only: {results['coverage']['txgnn_only']}")
    print(f"  Diseases with neither: {results['coverage']['neither']}")
    print()

    gb_recall = results['gb']['hits'] / results['gb']['total'] if results['gb']['total'] > 0 else 0
    txgnn_recall = results['txgnn']['hits'] / results['txgnn']['total'] if results['txgnn']['total'] > 0 else 0
    ensemble_recall = results['ensemble']['hits'] / results['ensemble']['total'] if results['ensemble']['total'] > 0 else 0

    print("Recall@30:")
    print(f"  GB model:    {results['gb']['hits']:>5}/{results['gb']['total']:<5} = {gb_recall*100:>6.2f}%  ({results['gb']['diseases']} diseases)")
    print(f"  TxGNN:       {results['txgnn']['hits']:>5}/{results['txgnn']['total']:<5} = {txgnn_recall*100:>6.2f}%  ({results['txgnn']['diseases']} diseases)")
    print(f"  Ensemble:    {results['ensemble']['hits']:>5}/{results['ensemble']['total']:<5} = {ensemble_recall*100:>6.2f}%  ({results['ensemble']['diseases']} diseases)")
    print()

    # Ensemble improvement
    if results['ensemble']['total'] > 0:
        improvement = (ensemble_recall - gb_recall) * 100
        print(f"Ensemble improvement over GB (on overlapping diseases): {improvement:+.2f}pp")

    # Analyze where ensemble helps
    print()
    print("=" * 70)
    print("ENSEMBLE BENEFIT ANALYSIS")
    print("=" * 70)
    print()

    ensemble_helped = 0
    ensemble_same = 0
    gb_was_better = 0

    for d in results['per_disease']:
        if d['gb_hits'] > 0 or d['txgnn_hits'] > 0:
            if d['ensemble_hits'] > max(d['gb_hits'], d['txgnn_hits']):
                ensemble_helped += 1
            elif d['ensemble_hits'] == max(d['gb_hits'], d['txgnn_hits']):
                ensemble_same += 1
            else:
                gb_was_better += 1

    print(f"Diseases where ensemble found MORE drugs than either alone: {ensemble_helped}")
    print(f"Diseases where ensemble matched best single model: {ensemble_same}")
    print(f"Diseases where single model was better: {gb_was_better}")

    # Save results
    output = {
        'baseline_recall': gb_recall,
        'txgnn_recall': txgnn_recall,
        'ensemble_recall': ensemble_recall,
        'improvement_pp': (ensemble_recall - gb_recall) * 100 if results['ensemble']['total'] > 0 else 0,
        'coverage': results['coverage'],
        'gb_stats': {k: v for k, v in results['gb'].items() if k != 'per_disease'},
        'txgnn_stats': {k: v for k, v in results['txgnn'].items() if k != 'per_disease'},
        'ensemble_stats': {k: v for k, v in results['ensemble'].items() if k != 'per_disease'},
    }

    output_path = ANALYSIS_DIR / "h1_ensemble_results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print()
    print(f"Results saved to: {output_path}")

    # Final verdict
    print()
    print("=" * 70)
    print("HYPOTHESIS VERDICT")
    print("=" * 70)
    print()

    target_recall = 0.43  # Success criteria

    if ensemble_recall >= target_recall:
        print(f"✓ VALIDATED: Ensemble achieved {ensemble_recall*100:.2f}% R@30 (target: {target_recall*100}%)")
    else:
        print(f"✗ INVALIDATED: Ensemble achieved {ensemble_recall*100:.2f}% R@30 (target: {target_recall*100}%)")

    if ensemble_recall > gb_recall:
        print(f"  Ensemble IMPROVES over GB baseline by {(ensemble_recall-gb_recall)*100:.2f}pp")
    else:
        print(f"  Ensemble does NOT improve over GB baseline")

    return output


if __name__ == "__main__":
    main()
