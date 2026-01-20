#!/usr/bin/env python3
"""
Train Gradient Boosting classifier for drug repurposing with enhanced ground truth.

This script:
1. Loads TransE embeddings
2. Loads Every Cure indication data + our 18 CONFIRMED drugs
3. Creates features: concat + product + difference (512 dims)
4. Trains GB with hard negative mining
5. Evaluates on enhanced benchmark

Fix 4 from model_fix_experiments.md
"""

import json
import pickle
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
from loguru import logger

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REFERENCE_DIR = DATA_DIR / "reference"
MODELS_DIR = PROJECT_ROOT / "models"
EVAL_DIR = PROJECT_ROOT / "autonomous_evaluation"

# Disease name to MESH ID mapping (comprehensive)
_DISEASE_MESH_RAW = {
    # HIV
    "hiv infection": "drkg:Disease::MESH:D015658",
    "hiv infections": "drkg:Disease::MESH:D015658",
    "hiv": "drkg:Disease::MESH:D015658",
    "hiv1 infection": "drkg:Disease::MESH:D015658",
    "hiv-1 infection": "drkg:Disease::MESH:D015658",
    "human immunodeficiency virus infection": "drkg:Disease::MESH:D015658",
    # Hepatitis C
    "hepatitis c": "drkg:Disease::MESH:D006526",
    "chronic hepatitis c": "drkg:Disease::MESH:D006526",
    "hepatitis c virus infection": "drkg:Disease::MESH:D006526",
    # Tuberculosis
    "tuberculosis": "drkg:Disease::MESH:D014376",
    "pulmonary tuberculosis": "drkg:Disease::MESH:D014376",
    # Breast cancer
    "breast cancer": "drkg:Disease::MESH:D001943",
    "breast neoplasm": "drkg:Disease::MESH:D001943",
    "breast neoplasms": "drkg:Disease::MESH:D001943",
    "mammary cancer": "drkg:Disease::MESH:D001943",
    "metastatic breast cancer": "drkg:Disease::MESH:D001943",
    # Lung cancer
    "lung cancer": "drkg:Disease::MESH:D008175",
    "lung neoplasm": "drkg:Disease::MESH:D008175",
    "non-small cell lung cancer": "drkg:Disease::MESH:D008175",
    "non small cell lung cancer": "drkg:Disease::MESH:D008175",
    "nsclc": "drkg:Disease::MESH:D008175",
    "small cell lung cancer": "drkg:Disease::MESH:D008175",
    # Colorectal cancer
    "colorectal cancer": "drkg:Disease::MESH:D015179",
    "colon cancer": "drkg:Disease::MESH:D015179",
    "colorectal neoplasm": "drkg:Disease::MESH:D015179",
    # Hypertension
    "hypertension": "drkg:Disease::MESH:D006973",
    "high blood pressure": "drkg:Disease::MESH:D006973",
    "essential hypertension": "drkg:Disease::MESH:D006973",
    "arterial hypertension": "drkg:Disease::MESH:D006973",
    # Heart failure
    "heart failure": "drkg:Disease::MESH:D006333",
    "congestive heart failure": "drkg:Disease::MESH:D006333",
    "cardiac failure": "drkg:Disease::MESH:D006333",
    "chronic heart failure": "drkg:Disease::MESH:D006333",
    # Atrial fibrillation
    "atrial fibrillation": "drkg:Disease::MESH:D001281",
    "af": "drkg:Disease::MESH:D001281",
    # Epilepsy
    "epilepsy": "drkg:Disease::MESH:D004827",
    "seizures": "drkg:Disease::MESH:D004827",
    "seizure disorder": "drkg:Disease::MESH:D004827",
    "partial seizures": "drkg:Disease::MESH:D004827",
    # Parkinson disease
    "parkinson disease": "drkg:Disease::MESH:D010300",
    "parkinson's disease": "drkg:Disease::MESH:D010300",
    "parkinsons disease": "drkg:Disease::MESH:D010300",
    "parkinsonism": "drkg:Disease::MESH:D010300",
    # Alzheimer disease
    "alzheimer disease": "drkg:Disease::MESH:D000544",
    "alzheimer's disease": "drkg:Disease::MESH:D000544",
    "alzheimers disease": "drkg:Disease::MESH:D000544",
    # Rheumatoid arthritis
    "rheumatoid arthritis": "drkg:Disease::MESH:D001172",
    "ra": "drkg:Disease::MESH:D001172",
    # Multiple sclerosis
    "multiple sclerosis": "drkg:Disease::MESH:D009103",
    "ms": "drkg:Disease::MESH:D009103",
    "relapsing multiple sclerosis": "drkg:Disease::MESH:D009103",
    # Psoriasis
    "psoriasis": "drkg:Disease::MESH:D011565",
    "plaque psoriasis": "drkg:Disease::MESH:D011565",
    "chronic plaque psoriasis": "drkg:Disease::MESH:D011565",
    # Type 2 diabetes
    "type 2 diabetes": "drkg:Disease::MESH:D003924",
    "type 2 diabetes mellitus": "drkg:Disease::MESH:D003924",
    "type ii diabetes": "drkg:Disease::MESH:D003924",
    "type ii diabetes mellitus": "drkg:Disease::MESH:D003924",
    "noninsulin dependent diabetes mellitus type ii": "drkg:Disease::MESH:D003924",
    "diabetes mellitus type 2": "drkg:Disease::MESH:D003924",
    "t2dm": "drkg:Disease::MESH:D003924",
    "diabetes mellitus": "drkg:Disease::MESH:D003924",
    "diabetes": "drkg:Disease::MESH:D003924",
    # Obesity
    "obesity": "drkg:Disease::MESH:D009765",
    "morbid obesity": "drkg:Disease::MESH:D009765",
    # Asthma
    "asthma": "drkg:Disease::MESH:D001249",
    "bronchial asthma": "drkg:Disease::MESH:D001249",
    "allergic asthma": "drkg:Disease::MESH:D001249",
    # COPD
    "copd": "drkg:Disease::MESH:D029424",
    "chronic obstructive pulmonary disease": "drkg:Disease::MESH:D029424",
    "chronic bronchitis": "drkg:Disease::MESH:D029424",
    # Osteoporosis
    "osteoporosis": "drkg:Disease::MESH:D010024",
    "postmenopausal osteoporosis": "drkg:Disease::MESH:D010024",
    # Additional high-value diseases from Every Cure
    "myocardial infarction": "drkg:Disease::MESH:D009203",
    "stroke": "drkg:Disease::MESH:D020521",
    "ulcerative colitis": "drkg:Disease::MESH:D003093",
    "schizophrenia": "drkg:Disease::MESH:D012559",
    "major depressive disorder": "drkg:Disease::MESH:D003865",
    "depression": "drkg:Disease::MESH:D003865",
    "osteoarthritis": "drkg:Disease::MESH:D010003",
}

# Create lowercase lookup
DISEASE_MESH_MAP = {k.lower(): v for k, v in _DISEASE_MESH_RAW.items()}


def load_transe_embeddings() -> Tuple[torch.Tensor, Dict[str, int]]:
    """Load TransE embeddings and entity mapping."""
    transe_path = MODELS_DIR / "transe.pt"
    if not transe_path.exists():
        raise FileNotFoundError(f"TransE model not found: {transe_path}")

    checkpoint = torch.load(transe_path, map_location="cpu")

    # Extract embeddings
    embeddings = None
    if "entity_embeddings" in checkpoint:
        embeddings = checkpoint["entity_embeddings"]
    elif "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
        for key in ["entity_embeddings.weight", "ent_embeddings.weight"]:
            if key in state:
                embeddings = state[key]
                break

    if embeddings is None:
        raise ValueError("Could not find entity embeddings in checkpoint")

    entity2id = checkpoint.get("entity2id", {})

    logger.info(f"Loaded TransE embeddings: {embeddings.shape}")
    logger.info(f"Entity mapping: {len(entity2id)} entities")

    return embeddings, entity2id


def load_every_cure_data() -> Dict[str, List[str]]:
    """Load Every Cure indication data."""
    xlsx_path = REFERENCE_DIR / "everycure" / "indicationList.xlsx"
    if not xlsx_path.exists():
        logger.warning(f"Every Cure data not found: {xlsx_path}")
        return {}

    df = pd.read_excel(xlsx_path)
    logger.info(f"Loaded Every Cure data: {len(df)} rows")

    # Group by disease - use correct column names
    disease_drugs: Dict[str, List[str]] = defaultdict(list)
    for _, row in df.iterrows():
        disease = str(row.get("disease name", "")).strip()
        drug = str(row.get("final normalized drug label", "")).strip()
        if disease and drug:
            disease_drugs[disease].append(drug)

    return dict(disease_drugs)


def load_enhanced_ground_truth() -> Dict[str, List[dict]]:
    """Load our discovered CONFIRMED drugs."""
    gt_path = EVAL_DIR / "enhanced_ground_truth.json"
    if not gt_path.exists():
        logger.warning(f"Enhanced ground truth not found: {gt_path}")
        return {}

    with open(gt_path) as f:
        data = json.load(f)

    # Filter to CONFIRMED only
    confirmed_only = {}
    for disease, drugs in data.items():
        confirmed = [d for d in drugs if d["classification"] == "CONFIRMED"]
        if confirmed:
            confirmed_only[disease] = confirmed

    total_confirmed = sum(len(drugs) for drugs in confirmed_only.values())
    logger.info(f"Loaded enhanced ground truth: {total_confirmed} CONFIRMED drugs")

    return confirmed_only


def load_drugbank_lookup() -> Dict[str, str]:
    """Load DrugBank name to ID mapping (inverted: name.lower() -> drkg ID)."""
    lookup_path = REFERENCE_DIR / "drugbank_lookup.json"
    if not lookup_path.exists():
        return {}

    with open(lookup_path) as f:
        id_to_name = json.load(f)

    # Invert: name.lower() -> drkg:Compound::DB*
    name_to_id = {}
    for db_id, name in id_to_name.items():
        name_to_id[name.lower()] = f"drkg:Compound::{db_id}"

    logger.info(f"Loaded DrugBank lookup: {len(name_to_id)} drugs")
    return name_to_id


def create_features(
    drug_emb: np.ndarray,
    disease_emb: np.ndarray,
) -> np.ndarray:
    """Create features from drug and disease embeddings."""
    # Concat + product + difference (512 dims total for 128-dim embeddings)
    concat = np.concatenate([drug_emb, disease_emb])
    product = drug_emb * disease_emb
    diff = drug_emb - disease_emb
    return np.concatenate([concat, product, diff])


def build_training_data(
    embeddings: torch.Tensor,
    entity2id: Dict[str, int],
    disease_drugs: Dict[str, List[str]],
    enhanced_gt: Dict[str, List[dict]],
    drugbank_lookup: Dict[str, str],
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Build training data with hard negative mining."""

    # Convert embeddings to numpy
    emb_np = embeddings.numpy()
    emb_dim = emb_np.shape[1]

    # Build positive pairs
    positive_pairs: List[Tuple[str, str]] = []  # (drug_id, disease_mesh_id)
    disease_to_drugs: Dict[str, Set[str]] = defaultdict(set)

    # 1. Add Every Cure pairs
    for disease_name, drugs in disease_drugs.items():
        disease_mesh = DISEASE_MESH_MAP.get(disease_name.lower())
        if not disease_mesh or disease_mesh not in entity2id:
            continue

        for drug_name in drugs:
            # Try to map drug name to DrugBank ID
            drug_name_lower = drug_name.lower()
            drug_id = drugbank_lookup.get(drug_name_lower)

            if drug_id and drug_id in entity2id:
                positive_pairs.append((drug_id, disease_mesh))
                disease_to_drugs[disease_mesh].add(drug_id)

    logger.info(f"Every Cure positive pairs: {len(positive_pairs)}")

    # 2. Add enhanced ground truth (CONFIRMED only)
    enhanced_count = 0
    for disease_name, drugs in enhanced_gt.items():
        disease_mesh = DISEASE_MESH_MAP.get(disease_name.lower())
        if not disease_mesh or disease_mesh not in entity2id:
            logger.warning(f"Disease not mapped: {disease_name}")
            continue

        for drug in drugs:
            drug_id = drug["drug_id"]
            if drug_id in entity2id:
                positive_pairs.append((drug_id, disease_mesh))
                disease_to_drugs[disease_mesh].add(drug_id)
                enhanced_count += 1

    logger.info(f"Enhanced ground truth pairs added: {enhanced_count}")
    logger.info(f"Total positive pairs: {len(positive_pairs)}")

    # 3. Build hard negatives (drugs that treat OTHER diseases)
    all_drugs_with_indications = set()
    for drugs in disease_to_drugs.values():
        all_drugs_with_indications.update(drugs)

    negative_pairs: List[Tuple[str, str]] = []
    for disease_mesh, disease_drugs_set in disease_to_drugs.items():
        # Hard negatives: drugs that treat other diseases but NOT this one
        hard_negs = all_drugs_with_indications - disease_drugs_set
        for drug_id in hard_negs:
            negative_pairs.append((drug_id, disease_mesh))

    logger.info(f"Hard negative pairs: {len(negative_pairs)}")

    # Subsample negatives to balance (3:1 negative:positive ratio)
    max_negatives = len(positive_pairs) * 3
    if len(negative_pairs) > max_negatives:
        np.random.seed(42)
        indices = np.random.choice(len(negative_pairs), max_negatives, replace=False)
        negative_pairs = [negative_pairs[i] for i in indices]
        logger.info(f"Subsampled negatives to: {len(negative_pairs)}")

    # 4. Create feature matrix
    all_pairs = positive_pairs + negative_pairs
    labels = [1] * len(positive_pairs) + [0] * len(negative_pairs)

    feature_dim = emb_dim * 4  # concat (2x) + product (1x) + diff (1x)
    X = np.zeros((len(all_pairs), feature_dim), dtype=np.float32)

    logger.info(f"Creating features for {len(all_pairs)} pairs...")
    for i, (drug_id, disease_id) in enumerate(tqdm(all_pairs)):
        drug_idx = entity2id[drug_id]
        disease_idx = entity2id[disease_id]

        drug_emb = emb_np[drug_idx]
        disease_emb = emb_np[disease_idx]

        X[i] = create_features(drug_emb, disease_emb)

    y = np.array(labels, dtype=np.int32)

    stats = {
        "positive_pairs": len(positive_pairs),
        "negative_pairs": len(negative_pairs),
        "every_cure_pairs": len(positive_pairs) - enhanced_count,
        "enhanced_pairs": enhanced_count,
        "diseases": len(disease_to_drugs),
        "unique_drugs": len(all_drugs_with_indications),
    }

    return X, y, stats


def evaluate_model(
    model: GradientBoostingClassifier,
    embeddings: torch.Tensor,
    entity2id: Dict[str, int],
    disease_drugs: Dict[str, Set[str]],
    disease_mesh: str,
    all_drug_ids: List[str],
) -> Dict:
    """Evaluate model on a single disease."""
    emb_np = embeddings.numpy()

    disease_idx = entity2id.get(disease_mesh)
    if disease_idx is None:
        return {"error": f"Disease not found: {disease_mesh}"}

    disease_emb = emb_np[disease_idx]

    # Score all drugs
    scores = []
    valid_drugs = []

    for drug_id in all_drug_ids:
        drug_idx = entity2id.get(drug_id)
        if drug_idx is None:
            continue

        drug_emb = emb_np[drug_idx]
        features = create_features(drug_emb, disease_emb).reshape(1, -1)
        score = model.predict_proba(features)[0, 1]
        scores.append(score)
        valid_drugs.append(drug_id)

    # Rank drugs
    ranked_indices = np.argsort(scores)[::-1]
    ranked_drugs = [valid_drugs[i] for i in ranked_indices]

    # Calculate recall at various k
    known_drugs = disease_drugs.get(disease_mesh, set())

    def recall_at_k(k: int) -> float:
        if not known_drugs:
            return 0.0
        top_k = set(ranked_drugs[:k])
        found = len(top_k & known_drugs)
        return found / len(known_drugs)

    return {
        "recall@30": recall_at_k(30),
        "recall@50": recall_at_k(50),
        "recall@100": recall_at_k(100),
        "known_drugs": len(known_drugs),
        "found@30": int(recall_at_k(30) * len(known_drugs)),
    }


def main():
    """Main training pipeline."""
    logger.info("=" * 70)
    logger.info("Fix 4: Retrain GB with Enhanced Ground Truth")
    logger.info("=" * 70)

    # Load data
    logger.info("\n1. Loading data...")
    embeddings, entity2id = load_transe_embeddings()
    every_cure_data = load_every_cure_data()
    enhanced_gt = load_enhanced_ground_truth()
    drugbank_lookup = load_drugbank_lookup()

    # Build training data
    logger.info("\n2. Building training data...")
    X, y, stats = build_training_data(
        embeddings, entity2id, every_cure_data, enhanced_gt, drugbank_lookup
    )

    logger.info(f"\nTraining data stats:")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")

    # Train/test split
    logger.info("\n3. Training model...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train GB classifier (match original: 200 estimators, depth 6)
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbose=1,
    )

    model.fit(X_train, y_train)

    # Evaluate on test set
    logger.info("\n4. Evaluating on test set...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    auroc = roc_auc_score(y_test, y_pred_proba)
    auprc = average_precision_score(y_test, y_pred_proba)

    logger.info(f"Test AUROC: {auroc:.4f}")
    logger.info(f"Test AUPRC: {auprc:.4f}")

    # Feature importance
    logger.info("\n5. Feature importance analysis...")
    emb_dim = embeddings.shape[1]
    importance = model.feature_importances_

    # Split importance by feature type
    concat_imp = importance[:emb_dim * 2].sum()
    product_imp = importance[emb_dim * 2:emb_dim * 3].sum()
    diff_imp = importance[emb_dim * 3:].sum()

    total_imp = concat_imp + product_imp + diff_imp
    logger.info(f"Concat features: {concat_imp/total_imp*100:.1f}%")
    logger.info(f"Product features: {product_imp/total_imp*100:.1f}%")
    logger.info(f"Difference features: {diff_imp/total_imp*100:.1f}%")

    # Save model
    model_path = MODELS_DIR / "drug_repurposing_gb_enhanced.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"\nSaved model to: {model_path}")

    # Save metrics
    metrics = {
        "auroc": auroc,
        "auprc": auprc,
        "training_stats": stats,
        "feature_importance": {
            "concat": float(concat_imp / total_imp),
            "product": float(product_imp / total_imp),
            "difference": float(diff_imp / total_imp),
        },
    }

    metrics_path = MODELS_DIR / "gb_enhanced_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to: {metrics_path}")

    logger.success("\nTraining complete!")

    return model, metrics


if __name__ == "__main__":
    main()
