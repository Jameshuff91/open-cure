#!/usr/bin/env python3
"""
Train GB with similarity features using PROPER disease-level train/test split.

Key difference from train_gb_with_similarity.py:
- Diseases are split 80/20 BEFORE building similarity lookup
- Similarity lookup built ONLY from training diseases
- Model trained on training diseases
- Evaluation on held-out test diseases

This prevents similarity feature leakage.
"""

import json
import pickle
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from loguru import logger

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REFERENCE_DIR = DATA_DIR / "reference"
MODELS_DIR = PROJECT_ROOT / "models"


class SimilarityFeatureComputer:
    """Computes similarity-based features for drug-disease pairs."""

    def __init__(self, embeddings: np.ndarray, entity2id: Dict[str, int]):
        self.embeddings = embeddings
        self.entity2id = entity2id
        self.disease_to_drug_embs: Dict[int, List[np.ndarray]] = {}
        self.drug_to_disease_embs: Dict[int, List[np.ndarray]] = {}

    def build_lookup(self, positive_pairs: List[Tuple[str, str]]):
        """Build lookup tables from positive training pairs."""
        for drug_id, disease_id in positive_pairs:
            drug_entity = self.entity2id.get(drug_id)
            disease_entity = self.entity2id.get(disease_id)

            if drug_entity is None or disease_entity is None:
                continue

            drug_emb = self.embeddings[drug_entity]
            disease_emb = self.embeddings[disease_entity]

            # Disease -> known drug embeddings
            if disease_entity not in self.disease_to_drug_embs:
                self.disease_to_drug_embs[disease_entity] = []
            self.disease_to_drug_embs[disease_entity].append(drug_emb)

            # Drug -> known disease embeddings
            if drug_entity not in self.drug_to_disease_embs:
                self.drug_to_disease_embs[drug_entity] = []
            self.drug_to_disease_embs[drug_entity].append(disease_emb)

        logger.info(f"Built similarity lookup: {len(self.disease_to_drug_embs)} diseases, "
                   f"{len(self.drug_to_disease_embs)} drugs")

    def compute_features(self, drug_entity: int, disease_entity: int,
                        exclude_self: bool = True) -> np.ndarray:
        """Compute 4 similarity features for a drug-disease pair."""
        drug_emb = self.embeddings[drug_entity].reshape(1, -1)
        disease_emb = self.embeddings[disease_entity].reshape(1, -1)

        features = np.zeros(4, dtype=np.float32)

        # Feature 1-2: Similarity to known treatments for this disease
        if disease_entity in self.disease_to_drug_embs:
            known_drug_embs = self.disease_to_drug_embs[disease_entity]

            if exclude_self:
                known_drug_embs = [e for e in known_drug_embs
                                  if not np.allclose(e, drug_emb[0])]

            if known_drug_embs:
                known_drugs = np.array(known_drug_embs)
                sims = cosine_similarity(drug_emb, known_drugs)[0]
                features[0] = np.max(sims)
                features[1] = np.mean(sims)

        # Feature 3-4: Similarity to known indications for this drug
        if drug_entity in self.drug_to_disease_embs:
            known_disease_embs = self.drug_to_disease_embs[drug_entity]

            if exclude_self:
                known_disease_embs = [e for e in known_disease_embs
                                     if not np.allclose(e, disease_emb[0])]

            if known_disease_embs:
                known_diseases = np.array(known_disease_embs)
                sims = cosine_similarity(disease_emb, known_diseases)[0]
                features[2] = np.max(sims)
                features[3] = np.mean(sims)

        return features


def load_transe_embeddings() -> Tuple[np.ndarray, Dict[str, int]]:
    """Load TransE embeddings and entity mapping."""
    transe_path = MODELS_DIR / "transe.pt"
    checkpoint = torch.load(transe_path, map_location="cpu", weights_only=False)
    embeddings = checkpoint["model_state_dict"]["entity_embeddings.weight"].numpy()
    entity2id = checkpoint.get("entity2id", {})
    logger.info(f"Loaded TransE embeddings: {embeddings.shape}")
    return embeddings, entity2id


def load_expanded_ground_truth() -> Dict[str, List[str]]:
    """Load expanded ground truth (disease entity -> drug entities)."""
    gt_path = REFERENCE_DIR / "expanded_ground_truth.json"
    with open(gt_path) as f:
        return json.load(f)


def create_features(
    drug_emb: np.ndarray,
    disease_emb: np.ndarray,
    sim_features: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Create features from drug and disease embeddings."""
    concat = np.concatenate([drug_emb, disease_emb])
    product = drug_emb * disease_emb
    diff = drug_emb - disease_emb
    base_features = np.concatenate([concat, product, diff])

    if sim_features is not None:
        return np.concatenate([base_features, sim_features])
    return base_features


def create_features_batch(drug_embs: np.ndarray, disease_emb: np.ndarray) -> np.ndarray:
    """Create features for multiple drugs against one disease (vectorized)."""
    n_drugs = drug_embs.shape[0]
    disease_embs = np.tile(disease_emb, (n_drugs, 1))
    concat = np.hstack([drug_embs, disease_embs])
    product = drug_embs * disease_embs
    diff = drug_embs - disease_embs
    return np.hstack([concat, product, diff])


def build_training_data(
    embeddings: np.ndarray,
    entity2id: Dict[str, int],
    train_diseases: Set[str],
    ground_truth: Dict[str, List[str]],
    sim_computer: SimilarityFeatureComputer,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Build training data from training diseases only."""

    emb_dim = embeddings.shape[1]

    # Build positive pairs from training diseases only
    positive_pairs: List[Tuple[str, str]] = []
    disease_to_drugs: Dict[str, Set[str]] = defaultdict(set)

    for disease_entity in train_diseases:
        if disease_entity not in ground_truth:
            continue
        if disease_entity not in entity2id:
            continue

        for drug_entity in ground_truth[disease_entity]:
            if drug_entity not in entity2id:
                continue
            positive_pairs.append((drug_entity, disease_entity))
            disease_to_drugs[disease_entity].add(drug_entity)

    logger.info(f"Training positive pairs: {len(positive_pairs)}")

    # Build hard negatives from training diseases
    all_drugs = set()
    for drugs in disease_to_drugs.values():
        all_drugs.update(drugs)

    negative_pairs: List[Tuple[str, str]] = []
    for disease, disease_drugs in disease_to_drugs.items():
        hard_negs = all_drugs - disease_drugs
        for drug in hard_negs:
            negative_pairs.append((drug, disease))

    logger.info(f"Hard negative pairs: {len(negative_pairs)}")

    # Subsample negatives (3:1 ratio)
    max_negatives = len(positive_pairs) * 3
    if len(negative_pairs) > max_negatives:
        np.random.seed(42)
        indices = np.random.choice(len(negative_pairs), max_negatives, replace=False)
        negative_pairs = [negative_pairs[i] for i in indices]
        logger.info(f"Subsampled to: {len(negative_pairs)}")

    # Create feature matrix
    all_pairs = positive_pairs + negative_pairs
    labels = [1] * len(positive_pairs) + [0] * len(negative_pairs)

    feature_dim = emb_dim * 4 + 4  # base + similarity features

    X = np.zeros((len(all_pairs), feature_dim), dtype=np.float32)

    logger.info(f"Creating features for {len(all_pairs)} pairs (dim={feature_dim})...")
    for i, (drug_id, disease_id) in enumerate(tqdm(all_pairs)):
        drug_idx = entity2id[drug_id]
        disease_idx = entity2id[disease_id]

        drug_emb = embeddings[drug_idx]
        disease_emb = embeddings[disease_idx]

        sim_features = sim_computer.compute_features(drug_idx, disease_idx, exclude_self=True)
        X[i] = create_features(drug_emb, disease_emb, sim_features)

    y = np.array(labels, dtype=np.int32)

    stats = {
        "positive_pairs": len(positive_pairs),
        "negative_pairs": len(negative_pairs),
        "training_diseases": len(disease_to_drugs),
        "unique_drugs": len(all_drugs),
        "feature_dim": feature_dim,
    }

    return X, y, stats


def evaluate_on_test_diseases(
    model,
    sim_computer: SimilarityFeatureComputer,
    embeddings: np.ndarray,
    entity2id: Dict[str, int],
    test_diseases: Set[str],
    ground_truth: Dict[str, List[str]],
    k: int = 30,
) -> Dict:
    """Evaluate model on held-out test diseases."""

    # Get all drugs
    all_drugs = [e for e in entity2id.keys() if e.startswith("drkg:Compound::")]
    drug_list = [d for d in all_drugs if d in entity2id]
    drug_indices = [entity2id[d] for d in drug_list]
    drug_embs = embeddings[drug_indices]

    logger.info(f"Total drugs: {len(drug_list)}")

    # Get test diseases with ground truth
    diseases_to_eval = [d for d in test_diseases if d in ground_truth and d in entity2id]
    logger.info(f"Test diseases to evaluate: {len(diseases_to_eval)}")

    # Pre-extract known disease embeddings per drug
    known_disease_embs_per_drug = {}
    for drug_idx, disease_embs_list in sim_computer.drug_to_disease_embs.items():
        known_disease_embs_per_drug[drug_idx] = np.array(disease_embs_list)

    total_gt_drugs = 0
    total_hits = 0
    per_disease_results = []

    for disease_entity in tqdm(diseases_to_eval, desc="Evaluating test diseases"):
        disease_idx = entity2id[disease_entity]
        disease_emb = embeddings[disease_idx]

        gt_drugs = set(ground_truth[disease_entity])
        gt_drugs_with_emb = gt_drugs & set(drug_list)

        if not gt_drugs_with_emb:
            continue

        # Get known drug embeddings for this disease (may be empty for test diseases!)
        known_drug_embs = None
        if disease_idx in sim_computer.disease_to_drug_embs:
            known_drug_embs = np.array(sim_computer.disease_to_drug_embs[disease_idx])

        # Compute base features (vectorized)
        base_features = create_features_batch(drug_embs, disease_emb)

        # Compute similarity features
        n_drugs = len(drug_list)
        sim_features = np.zeros((n_drugs, 4), dtype=np.float32)

        if known_drug_embs is not None and len(known_drug_embs) > 0:
            sims = cosine_similarity(drug_embs, known_drug_embs)
            sim_features[:, 0] = sims.max(axis=1)
            sim_features[:, 1] = sims.mean(axis=1)

        disease_emb_2d = disease_emb.reshape(1, -1)
        for i, drug_idx in enumerate(drug_indices):
            if drug_idx in known_disease_embs_per_drug:
                known_diseases = known_disease_embs_per_drug[drug_idx]
                if len(known_diseases) > 0:
                    sims = cosine_similarity(disease_emb_2d, known_diseases)[0]
                    sim_features[i, 2] = np.max(sims)
                    sim_features[i, 3] = np.mean(sims)

        # Combine features
        features = np.hstack([base_features, sim_features])

        # Batch prediction
        probs = model.predict_proba(features)[:, 1]

        # Get top-k
        top_k_indices = np.argsort(probs)[-k:][::-1]
        top_k_drugs = set(drug_list[i] for i in top_k_indices)

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

    overall_recall = total_hits / total_gt_drugs if total_gt_drugs > 0 else 0

    logger.info(f"\nTest Set Results:")
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
    """Main training pipeline with proper disease-level split."""
    logger.info("=" * 70)
    logger.info("Training GB with Similarity Features (Disease-Level Split)")
    logger.info("=" * 70)

    # Load data
    logger.info("\n1. Loading data...")
    embeddings, entity2id = load_transe_embeddings()
    ground_truth = load_expanded_ground_truth()

    # Get diseases with embeddings
    all_diseases = [d for d in ground_truth.keys() if d in entity2id]
    logger.info(f"Total diseases with embeddings: {len(all_diseases)}")

    # Split diseases 80/20
    logger.info("\n2. Splitting diseases 80/20...")
    np.random.seed(42)
    np.random.shuffle(all_diseases)
    split_idx = int(len(all_diseases) * 0.8)
    train_diseases = set(all_diseases[:split_idx])
    test_diseases = set(all_diseases[split_idx:])
    logger.info(f"Training diseases: {len(train_diseases)}")
    logger.info(f"Test diseases: {len(test_diseases)}")

    # Build similarity lookup from TRAINING diseases only
    logger.info("\n3. Building similarity lookup from training diseases...")
    sim_computer = SimilarityFeatureComputer(embeddings, entity2id)

    train_pairs = []
    for disease in train_diseases:
        if disease not in ground_truth:
            continue
        for drug in ground_truth[disease]:
            if drug in entity2id:
                train_pairs.append((drug, disease))

    sim_computer.build_lookup(train_pairs)

    # Build training data
    logger.info("\n4. Building training data...")
    X, y, stats = build_training_data(
        embeddings, entity2id, train_diseases, ground_truth, sim_computer
    )

    logger.info(f"\nTraining data stats:")
    for k_stat, v in stats.items():
        logger.info(f"  {k_stat}: {v}")

    # Train/val split (within training diseases)
    logger.info("\n5. Training model...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbose=1,
        n_iter_no_change=10,
    )

    model.fit(X_train, y_train)

    # Validation metrics
    logger.info("\n6. Validation metrics...")
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    auroc = roc_auc_score(y_val, y_pred_proba)
    auprc = average_precision_score(y_val, y_pred_proba)
    logger.info(f"Validation AUROC: {auroc:.4f}")
    logger.info(f"Validation AUPRC: {auprc:.4f}")

    # Feature importance
    logger.info("\n7. Feature importance...")
    emb_dim = embeddings.shape[1]
    importance = model.feature_importances_

    concat_imp = importance[:emb_dim * 2].sum()
    product_imp = importance[emb_dim * 2:emb_dim * 3].sum()
    diff_imp = importance[emb_dim * 3:emb_dim * 4].sum()
    sim_imp = importance[emb_dim * 4:].sum()

    total_imp = concat_imp + product_imp + diff_imp + sim_imp

    logger.info(f"Concat features:     {concat_imp/total_imp*100:.1f}%")
    logger.info(f"Product features:    {product_imp/total_imp*100:.1f}%")
    logger.info(f"Difference features: {diff_imp/total_imp*100:.1f}%")
    logger.info(f"Similarity features: {sim_imp/total_imp*100:.1f}%")

    # Evaluate on TEST diseases
    logger.info("\n8. Evaluating on held-out test diseases...")
    test_results = evaluate_on_test_diseases(
        model, sim_computer, embeddings, entity2id,
        test_diseases, ground_truth, k=30
    )

    # Save model
    model_path = MODELS_DIR / "drug_repurposing_gb_similarity_split.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model,
            "sim_computer": sim_computer,
            "feature_dim": stats["feature_dim"],
            "train_diseases": list(train_diseases),
            "test_diseases": list(test_diseases),
        }, f)
    logger.info(f"\nSaved model to: {model_path}")

    # Save metrics
    metrics = {
        "validation_auroc": float(auroc),
        "validation_auprc": float(auprc),
        "test_recall_at_30": test_results['recall_at_30'],
        "test_diseases_evaluated": test_results['diseases_evaluated'],
        "test_total_gt_drugs": test_results['total_gt_drugs'],
        "test_total_hits": test_results['total_hits'],
        "training_stats": stats,
        "feature_importance": {
            "concat": float(concat_imp / total_imp),
            "product": float(product_imp / total_imp),
            "difference": float(diff_imp / total_imp),
            "similarity": float(sim_imp / total_imp),
        },
    }

    metrics_path = MODELS_DIR / "gb_similarity_split_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.success("\nTraining complete!")
    logger.info(f"Test Recall@30: {test_results['recall_at_30']*100:.1f}%")

    return model, metrics, test_results


if __name__ == "__main__":
    main()
