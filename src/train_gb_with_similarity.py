#!/usr/bin/env python3
"""
Train Gradient Boosting classifier with similarity features.

This extends train_gb_enhanced.py by adding 4 new "guilt by association" features:
1. max_sim_drug: Max cosine similarity to drugs known to treat this disease
2. mean_sim_drug: Mean cosine similarity to known treatments
3. max_sim_disease: Max cosine similarity to diseases this drug treats
4. mean_sim_disease: Mean cosine similarity to known indications

Expected improvement: +2-4% Recall@30
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
        """Compute 4 similarity features for a drug-disease pair.

        Args:
            drug_entity: Drug entity index
            disease_entity: Disease entity index
            exclude_self: If True, exclude the query drug/disease from similarity computation
                         (important for training to avoid data leakage)

        Returns:
            4-element array: [max_sim_drug, mean_sim_drug, max_sim_disease, mean_sim_disease]
        """
        drug_emb = self.embeddings[drug_entity].reshape(1, -1)
        disease_emb = self.embeddings[disease_entity].reshape(1, -1)

        features = np.zeros(4, dtype=np.float32)

        # Feature 1-2: Similarity to known treatments for this disease
        if disease_entity in self.disease_to_drug_embs:
            known_drug_embs = self.disease_to_drug_embs[disease_entity]

            if exclude_self:
                # Exclude the query drug from comparison
                known_drug_embs = [e for e in known_drug_embs
                                  if not np.allclose(e, drug_emb[0])]

            if known_drug_embs:
                known_drugs = np.array(known_drug_embs)
                sims = cosine_similarity(drug_emb, known_drugs)[0]
                features[0] = np.max(sims)  # max_sim_drug
                features[1] = np.mean(sims)  # mean_sim_drug

        # Feature 3-4: Similarity to known indications for this drug
        if drug_entity in self.drug_to_disease_embs:
            known_disease_embs = self.drug_to_disease_embs[drug_entity]

            if exclude_self:
                # Exclude the query disease from comparison
                known_disease_embs = [e for e in known_disease_embs
                                     if not np.allclose(e, disease_emb[0])]

            if known_disease_embs:
                known_diseases = np.array(known_disease_embs)
                sims = cosine_similarity(disease_emb, known_diseases)[0]
                features[2] = np.max(sims)  # max_sim_disease
                features[3] = np.mean(sims)  # mean_sim_disease

        return features


def load_transe_embeddings() -> Tuple[np.ndarray, Dict[str, int]]:
    """Load TransE embeddings and entity mapping."""
    transe_path = MODELS_DIR / "transe.pt"
    checkpoint = torch.load(transe_path, map_location="cpu")

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
    """Create features from drug and disease embeddings.

    Base features (512D):
    - Concatenation: [drug_emb, disease_emb] (256D)
    - Element-wise product: drug_emb * disease_emb (128D)
    - Element-wise difference: drug_emb - disease_emb (128D)

    Similarity features (4D, optional):
    - max_sim_drug, mean_sim_drug, max_sim_disease, mean_sim_disease
    """
    concat = np.concatenate([drug_emb, disease_emb])
    product = drug_emb * disease_emb
    diff = drug_emb - disease_emb
    base_features = np.concatenate([concat, product, diff])

    if sim_features is not None:
        return np.concatenate([base_features, sim_features])
    return base_features


def build_training_data(
    embeddings: np.ndarray,
    entity2id: Dict[str, int],
    ground_truth: Dict[str, List[str]],
    use_similarity_features: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict, Optional[SimilarityFeatureComputer]]:
    """Build training data with hard negative mining and similarity features."""

    emb_dim = embeddings.shape[1]

    # Build positive pairs from ground truth
    positive_pairs: List[Tuple[str, str]] = []
    disease_to_drugs: Dict[str, Set[str]] = defaultdict(set)

    for disease_entity, drug_entities in ground_truth.items():
        if disease_entity not in entity2id:
            continue

        for drug_entity in drug_entities:
            if drug_entity not in entity2id:
                continue
            positive_pairs.append((drug_entity, disease_entity))
            disease_to_drugs[disease_entity].add(drug_entity)

    logger.info(f"Positive pairs: {len(positive_pairs)}")

    # Build similarity feature computer
    sim_computer = None
    if use_similarity_features:
        sim_computer = SimilarityFeatureComputer(embeddings, entity2id)
        sim_computer.build_lookup(positive_pairs)

    # Build hard negatives
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

    feature_dim = emb_dim * 4  # base features
    if use_similarity_features:
        feature_dim += 4  # similarity features

    X = np.zeros((len(all_pairs), feature_dim), dtype=np.float32)

    logger.info(f"Creating features for {len(all_pairs)} pairs (dim={feature_dim})...")
    for i, (drug_id, disease_id) in enumerate(tqdm(all_pairs)):
        drug_idx = entity2id[drug_id]
        disease_idx = entity2id[disease_id]

        drug_emb = embeddings[drug_idx]
        disease_emb = embeddings[disease_idx]

        if use_similarity_features:
            # Exclude self for training to avoid data leakage
            sim_features = sim_computer.compute_features(
                drug_idx, disease_idx, exclude_self=True
            )
            X[i] = create_features(drug_emb, disease_emb, sim_features)
        else:
            X[i] = create_features(drug_emb, disease_emb)

    y = np.array(labels, dtype=np.int32)

    stats = {
        "positive_pairs": len(positive_pairs),
        "negative_pairs": len(negative_pairs),
        "diseases": len(disease_to_drugs),
        "unique_drugs": len(all_drugs),
        "feature_dim": feature_dim,
        "use_similarity_features": use_similarity_features,
    }

    return X, y, stats, sim_computer


def main():
    """Main training pipeline."""
    logger.info("=" * 70)
    logger.info("Training GB with Similarity Features")
    logger.info("=" * 70)

    # Load data
    logger.info("\n1. Loading data...")
    embeddings, entity2id = load_transe_embeddings()
    ground_truth = load_expanded_ground_truth()

    # Build training data WITH similarity features
    logger.info("\n2. Building training data with similarity features...")
    X, y, stats, sim_computer = build_training_data(
        embeddings, entity2id, ground_truth, use_similarity_features=True
    )

    logger.info(f"\nTraining data stats:")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")

    # Train/test split
    logger.info("\n3. Training model...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train GB classifier
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbose=1,
        n_iter_no_change=10,  # Early stopping
    )

    model.fit(X_train, y_train)

    # Evaluate
    logger.info("\n4. Evaluating...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    auroc = roc_auc_score(y_test, y_pred_proba)
    auprc = average_precision_score(y_test, y_pred_proba)

    logger.info(f"Test AUROC: {auroc:.4f}")
    logger.info(f"Test AUPRC: {auprc:.4f}")

    # Feature importance analysis
    logger.info("\n5. Feature importance...")
    emb_dim = embeddings.shape[1]
    importance = model.feature_importances_

    concat_imp = importance[:emb_dim * 2].sum()
    product_imp = importance[emb_dim * 2:emb_dim * 3].sum()
    diff_imp = importance[emb_dim * 3:emb_dim * 4].sum()
    sim_imp = importance[emb_dim * 4:].sum() if len(importance) > emb_dim * 4 else 0

    total_imp = concat_imp + product_imp + diff_imp + sim_imp

    logger.info(f"Concat features:     {concat_imp/total_imp*100:.1f}%")
    logger.info(f"Product features:    {product_imp/total_imp*100:.1f}%")
    logger.info(f"Difference features: {diff_imp/total_imp*100:.1f}%")
    logger.info(f"Similarity features: {sim_imp/total_imp*100:.1f}%")

    # Individual similarity feature importance
    if len(importance) > emb_dim * 4:
        sim_names = ["max_sim_drug", "mean_sim_drug", "max_sim_disease", "mean_sim_disease"]
        for i, name in enumerate(sim_names):
            idx = emb_dim * 4 + i
            logger.info(f"  {name}: {importance[idx]/total_imp*100:.2f}%")

    # Save model and similarity computer
    model_path = MODELS_DIR / "drug_repurposing_gb_similarity.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model,
            "sim_computer": sim_computer,
            "feature_dim": stats["feature_dim"],
        }, f)
    logger.info(f"\nSaved model to: {model_path}")

    # Save metrics
    metrics = {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "training_stats": stats,
        "feature_importance": {
            "concat": float(concat_imp / total_imp),
            "product": float(product_imp / total_imp),
            "difference": float(diff_imp / total_imp),
            "similarity": float(sim_imp / total_imp),
        },
    }

    metrics_path = MODELS_DIR / "gb_similarity_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.success("\nTraining complete!")
    logger.info(f"New feature dimension: {stats['feature_dim']} (was 512)")

    return model, metrics


if __name__ == "__main__":
    main()
