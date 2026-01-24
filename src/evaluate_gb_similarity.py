#!/usr/bin/env python3
"""
Evaluate GB model with similarity features on Every Cure ground truth.

Computes per-drug Recall@30 across all diseases.

OPTIMIZED VERSION: Vectorized operations for 100x speedup.
"""

import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity

# Add src to path and import class for unpickling
sys.path.insert(0, str(Path(__file__).parent))
from train_gb_with_similarity import SimilarityFeatureComputer  # noqa: F401, E402

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REFERENCE_DIR = DATA_DIR / "reference"
MODELS_DIR = PROJECT_ROOT / "models"


def load_model():
    """Load the GB similarity model."""
    model_path = MODELS_DIR / "drug_repurposing_gb_similarity.pkl"
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['sim_computer'], data['feature_dim']


def load_transe():
    """Load TransE embeddings."""
    transe_path = MODELS_DIR / "transe.pt"
    checkpoint = torch.load(transe_path, map_location="cpu", weights_only=False)
    embeddings = checkpoint["model_state_dict"]["entity_embeddings.weight"].numpy()
    entity2id = checkpoint.get("entity2id", {})
    return embeddings, entity2id


def load_ground_truth():
    """Load expanded ground truth."""
    gt_path = REFERENCE_DIR / "expanded_ground_truth.json"
    with open(gt_path) as f:
        return json.load(f)


def create_features_batch(drug_embs: np.ndarray, disease_emb: np.ndarray) -> np.ndarray:
    """Create features for multiple drugs against one disease.

    Args:
        drug_embs: (n_drugs, emb_dim) array
        disease_emb: (emb_dim,) array

    Returns:
        (n_drugs, 512) feature array
    """
    n_drugs = drug_embs.shape[0]
    emb_dim = drug_embs.shape[1]

    # Broadcast disease embedding
    disease_embs = np.tile(disease_emb, (n_drugs, 1))

    # Compute features
    concat = np.hstack([drug_embs, disease_embs])  # (n, 256)
    product = drug_embs * disease_embs  # (n, 128)
    diff = drug_embs - disease_embs  # (n, 128)

    return np.hstack([concat, product, diff])


def compute_similarity_features_batch(
    drug_embs: np.ndarray,
    disease_emb: np.ndarray,
    known_drug_embs: np.ndarray | None,
    known_disease_embs_per_drug: Dict[int, np.ndarray],
    drug_indices: List[int],
) -> np.ndarray:
    """Compute similarity features for multiple drugs against one disease.

    Args:
        drug_embs: (n_drugs, emb_dim) drug embeddings
        disease_emb: (emb_dim,) disease embedding
        known_drug_embs: (n_known, emb_dim) embeddings of drugs known to treat this disease, or None
        known_disease_embs_per_drug: dict mapping drug index to its known disease embeddings
        drug_indices: list of drug entity indices

    Returns:
        (n_drugs, 4) similarity features
    """
    n_drugs = drug_embs.shape[0]
    sim_features = np.zeros((n_drugs, 4), dtype=np.float32)

    # Feature 1-2: Similarity to known treatments for this disease
    if known_drug_embs is not None and len(known_drug_embs) > 0:
        # Compute cosine similarity between all drugs and known treatments
        sims = cosine_similarity(drug_embs, known_drug_embs)  # (n_drugs, n_known)
        sim_features[:, 0] = sims.max(axis=1)  # max_sim_drug
        sim_features[:, 1] = sims.mean(axis=1)  # mean_sim_drug

    # Feature 3-4: Similarity to known indications for each drug
    disease_emb_2d = disease_emb.reshape(1, -1)
    for i, drug_idx in enumerate(drug_indices):
        if drug_idx in known_disease_embs_per_drug:
            known_diseases = known_disease_embs_per_drug[drug_idx]
            if len(known_diseases) > 0:
                sims = cosine_similarity(disease_emb_2d, known_diseases)[0]
                sim_features[i, 2] = np.max(sims)  # max_sim_disease
                sim_features[i, 3] = np.mean(sims)  # mean_sim_disease

    return sim_features


def evaluate_model(
    model,
    sim_computer,
    embeddings: np.ndarray,
    entity2id: Dict[str, int],
    ground_truth: Dict[str, List[str]],
    k: int = 30,
) -> Dict:
    """
    Evaluate model on ground truth (optimized version).
    """
    # Get all drugs
    all_drugs = [e for e in entity2id.keys() if e.startswith("drkg:Compound::")]
    drug_list = [d for d in all_drugs if d in entity2id]
    drug_indices = [entity2id[d] for d in drug_list]
    drug_embs = embeddings[drug_indices]  # (n_drugs, 128)

    logger.info(f"Total drugs: {len(drug_list)}")

    # Get diseases to evaluate
    diseases_to_eval = [d for d in ground_truth.keys() if d in entity2id]
    logger.info(f"Diseases to evaluate: {len(diseases_to_eval)}/{len(ground_truth)}")

    # Build drug-to-index mapping
    drug_to_local_idx = {d: i for i, d in enumerate(drug_list)}

    # Pre-extract known disease embeddings per drug from similarity computer
    known_disease_embs_per_drug = {}
    for drug_idx, disease_embs in sim_computer.drug_to_disease_embs.items():
        known_disease_embs_per_drug[drug_idx] = np.array(disease_embs)

    # Evaluate per disease
    total_gt_drugs = 0
    total_hits = 0
    per_disease_results = []

    for disease_entity in tqdm(diseases_to_eval, desc="Evaluating"):
        disease_idx = entity2id[disease_entity]
        disease_emb = embeddings[disease_idx]

        gt_drugs = set(ground_truth[disease_entity])
        gt_drugs_with_emb = gt_drugs & set(drug_list)

        if not gt_drugs_with_emb:
            continue

        # Get known drug embeddings for this disease
        known_drug_embs = None
        if disease_idx in sim_computer.disease_to_drug_embs:
            known_drug_embs = np.array(sim_computer.disease_to_drug_embs[disease_idx])

        # Compute base features (vectorized)
        base_features = create_features_batch(drug_embs, disease_emb)  # (n_drugs, 512)

        # Compute similarity features (partially vectorized)
        sim_features = compute_similarity_features_batch(
            drug_embs, disease_emb, known_drug_embs,
            known_disease_embs_per_drug, drug_indices
        )  # (n_drugs, 4)

        # Combine features
        features = np.hstack([base_features, sim_features])  # (n_drugs, 516)

        # Batch prediction
        probs = model.predict_proba(features)[:, 1]

        # Get top-k
        top_k_local_indices = np.argsort(probs)[-k:][::-1]
        top_k_drugs = set(drug_list[i] for i in top_k_local_indices)

        # Count hits
        hits = len(gt_drugs_with_emb & top_k_drugs)
        recall = hits / len(gt_drugs_with_emb) if gt_drugs_with_emb else 0

        total_gt_drugs += len(gt_drugs_with_emb)
        total_hits += hits

        per_disease_results.append({
            'disease': disease_entity,
            'gt_drugs': len(gt_drugs_with_emb),
            'hits': hits,
            'recall': recall,
        })

    # Compute overall per-drug Recall@k
    overall_recall = total_hits / total_gt_drugs if total_gt_drugs > 0 else 0

    logger.info(f"\nResults:")
    logger.info(f"  Diseases evaluated: {len(per_disease_results)}")
    logger.info(f"  Total GT drugs: {total_gt_drugs}")
    logger.info(f"  Total hits@{k}: {total_hits}")
    logger.info(f"  Per-drug Recall@{k}: {overall_recall*100:.1f}%")

    return {
        'diseases_evaluated': len(per_disease_results),
        'total_gt_drugs': total_gt_drugs,
        'total_hits': total_hits,
        f'recall_at_{k}': overall_recall,
        'per_disease_results': per_disease_results,
    }


def main():
    logger.info("=" * 70)
    logger.info("Evaluating GB Similarity Model (Optimized)")
    logger.info("=" * 70)

    # Load model
    logger.info("\n1. Loading model...")
    model, sim_computer, feature_dim = load_model()
    logger.info(f"   Feature dim: {feature_dim}")

    # Load embeddings
    logger.info("\n2. Loading embeddings...")
    embeddings, entity2id = load_transe()
    logger.info(f"   Embeddings shape: {embeddings.shape}")

    # Load ground truth
    logger.info("\n3. Loading ground truth...")
    ground_truth = load_ground_truth()
    logger.info(f"   Diseases: {len(ground_truth)}")

    # Evaluate
    logger.info("\n4. Evaluating...")
    results = evaluate_model(
        model, sim_computer, embeddings, entity2id, ground_truth, k=30
    )

    # Save results
    output_path = MODELS_DIR / "gb_similarity_evaluation.json"
    with open(output_path, 'w') as f:
        json.dump({
            'recall_at_30': results['recall_at_30'],
            'diseases_evaluated': results['diseases_evaluated'],
            'total_gt_drugs': results['total_gt_drugs'],
            'total_hits': results['total_hits'],
        }, f, indent=2)

    logger.success(f"\nSaved results to: {output_path}")

    return results


if __name__ == "__main__":
    main()
