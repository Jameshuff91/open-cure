#!/usr/bin/env python3
"""
Train Gradient Boosting classifier with TxGNN prediction scores as features.

This script:
1. Loads TransE embeddings (original 512-dim features)
2. Loads TxGNN prediction scores from txgnn_predictions_final.csv
3. Creates enhanced features: original + TxGNN rank/score
4. Trains GB with hard negative mining
5. Evaluates on Every Cure ground truth (same as original GB)
6. Compares Recall@30: Original GB vs GB+TxGNN features

Author: Claude
"""

import json
import pickle
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional

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
REFERENCE_DIR = DATA_DIR / "reference"
MODELS_DIR = PROJECT_ROOT / "models"
ANALYSIS_DIR = DATA_DIR / "analysis"
EVAL_DIR = PROJECT_ROOT / "autonomous_evaluation"

ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# Disease name to MESH ID mapping (from original training script)
DISEASE_MESH_MAP = {
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

MESH_TO_NAME = {
    "drkg:Disease::MESH:D015658": "HIV infection",
    "drkg:Disease::MESH:D006526": "Hepatitis C",
    "drkg:Disease::MESH:D014376": "Tuberculosis",
    "drkg:Disease::MESH:D001943": "Breast cancer",
    "drkg:Disease::MESH:D008175": "Lung cancer",
    "drkg:Disease::MESH:D015179": "Colorectal cancer",
    "drkg:Disease::MESH:D006973": "Hypertension",
    "drkg:Disease::MESH:D006333": "Heart failure",
    "drkg:Disease::MESH:D001281": "Atrial fibrillation",
    "drkg:Disease::MESH:D004827": "Epilepsy",
    "drkg:Disease::MESH:D010300": "Parkinson disease",
    "drkg:Disease::MESH:D000544": "Alzheimer disease",
    "drkg:Disease::MESH:D001172": "Rheumatoid arthritis",
    "drkg:Disease::MESH:D009103": "Multiple sclerosis",
    "drkg:Disease::MESH:D011565": "Psoriasis",
    "drkg:Disease::MESH:D003924": "Type 2 diabetes",
    "drkg:Disease::MESH:D009765": "Obesity",
    "drkg:Disease::MESH:D001249": "Asthma",
    "drkg:Disease::MESH:D029424": "COPD",
    "drkg:Disease::MESH:D010024": "Osteoporosis",
}


class TxGNNFeatures:
    """Load and provide TxGNN prediction scores as features."""

    def __init__(self, predictions_path: Path):
        self.predictions_path = predictions_path
        # (disease_name_lower, drug_name_lower) -> (score, rank)
        self.scores: Dict[Tuple[str, str], Tuple[float, int]] = {}
        self.max_rank = 50  # Max rank in TxGNN predictions
        self.max_score = 0.0

    def load(self, drugbank_lookup: Dict[str, str]) -> "TxGNNFeatures":
        """Load TxGNN predictions and index by disease+drug name."""
        logger.info(f"Loading TxGNN predictions from {self.predictions_path}")

        # drugbank_lookup is DB_ID -> drug_name, create reverse
        db_id_to_name = drugbank_lookup

        df = pd.read_csv(self.predictions_path)
        logger.info(f"Loaded {len(df)} TxGNN predictions")

        self.max_score = df["score"].max()
        logger.info(f"Max TxGNN score: {self.max_score:.6f}")

        # Index by (disease_lower, drug_name_lower)
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Indexing TxGNN"):
            disease = row["disease_name"].lower()
            drug_id = row["drug_id"]  # DrugBank ID like DB00406
            score = row["score"]
            rank = row["rank"]

            # Map drug_id to drug name
            drug_name = db_id_to_name.get(drug_id)
            if drug_name:
                key = (disease, drug_name.lower())
                self.scores[key] = (score, rank)

        logger.info(f"Indexed {len(self.scores)} TxGNN (disease, drug) pairs")
        return self

    def get_features(
        self, disease_name: str, drug_name: str
    ) -> Tuple[float, float, float, float, float]:
        """
        Get TxGNN features for a drug-disease pair.

        Returns:
            - normalized_score: TxGNN score / max_score (0-1)
            - normalized_rank: 1 - (rank / max_rank), higher is better
            - in_top_10: 1.0 if rank <= 10, else 0.0
            - in_top_30: 1.0 if rank <= 30, else 0.0
            - in_top_50: 1.0 if rank <= 50, else 0.0
        """
        key = (disease_name.lower(), drug_name.lower())
        if key in self.scores:
            score, rank = self.scores[key]
            normalized_score = score / self.max_score if self.max_score > 0 else 0.0
            normalized_rank = 1.0 - (rank / self.max_rank)
            in_top_10 = 1.0 if rank <= 10 else 0.0
            in_top_30 = 1.0 if rank <= 30 else 0.0
            in_top_50 = 1.0 if rank <= 50 else 0.0
            return (normalized_score, normalized_rank, in_top_10, in_top_30, in_top_50)
        else:
            # No TxGNN prediction for this pair - use default values
            return (0.0, 0.0, 0.0, 0.0, 0.0)


def load_transe_embeddings() -> Tuple[torch.Tensor, Dict[str, int]]:
    """Load TransE embeddings and entity mapping."""
    transe_path = MODELS_DIR / "transe.pt"
    if not transe_path.exists():
        raise FileNotFoundError(f"TransE model not found: {transe_path}")

    checkpoint = torch.load(transe_path, map_location="cpu", weights_only=False)

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

    confirmed_only = {}
    for disease, drugs in data.items():
        confirmed = [d for d in drugs if d["classification"] == "CONFIRMED"]
        if confirmed:
            confirmed_only[disease] = confirmed

    total_confirmed = sum(len(drugs) for drugs in confirmed_only.values())
    logger.info(f"Loaded enhanced ground truth: {total_confirmed} CONFIRMED drugs")

    return confirmed_only


def load_drugbank_lookup() -> Dict[str, str]:
    """Load DrugBank ID to name mapping."""
    lookup_path = REFERENCE_DIR / "drugbank_lookup.json"
    if not lookup_path.exists():
        return {}

    with open(lookup_path) as f:
        id_to_name = json.load(f)

    logger.info(f"Loaded DrugBank lookup: {len(id_to_name)} drugs")
    return id_to_name


def create_base_features(drug_emb: np.ndarray, disease_emb: np.ndarray) -> np.ndarray:
    """Create base features from drug and disease embeddings (512 dims)."""
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
    txgnn_features: TxGNNFeatures,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Build training data with TxGNN features."""

    emb_np = embeddings.numpy()
    emb_dim = emb_np.shape[1]

    # Inverted lookup: drug_name_lower -> drkg_id
    name_to_drkg = {v.lower(): f"drkg:Compound::{k}" for k, v in drugbank_lookup.items()}

    positive_pairs: List[Tuple[str, str, str, str]] = []  # (drug_id, disease_mesh, drug_name, disease_name)
    disease_to_drugs: Dict[str, Set[str]] = defaultdict(set)

    # 1. Add Every Cure pairs
    for disease_name, drugs in disease_drugs.items():
        disease_mesh = DISEASE_MESH_MAP.get(disease_name.lower())
        if not disease_mesh or disease_mesh not in entity2id:
            continue

        for drug_name in drugs:
            drug_name_lower = drug_name.lower()
            drug_id = name_to_drkg.get(drug_name_lower)

            if drug_id and drug_id in entity2id:
                positive_pairs.append((drug_id, disease_mesh, drug_name, disease_name))
                disease_to_drugs[disease_mesh].add(drug_id)

    logger.info(f"Every Cure positive pairs: {len(positive_pairs)}")

    # 2. Add enhanced ground truth (CONFIRMED only)
    enhanced_count = 0
    for disease_name, drugs in enhanced_gt.items():
        disease_mesh = DISEASE_MESH_MAP.get(disease_name.lower())
        if not disease_mesh or disease_mesh not in entity2id:
            continue

        for drug in drugs:
            drug_id = drug["drug_id"]
            if drug_id in entity2id:
                # Get drug name from drug_id
                drug_name = drug.get("drug_name", "")
                positive_pairs.append((drug_id, disease_mesh, drug_name, disease_name))
                disease_to_drugs[disease_mesh].add(drug_id)
                enhanced_count += 1

    logger.info(f"Enhanced ground truth pairs added: {enhanced_count}")
    logger.info(f"Total positive pairs: {len(positive_pairs)}")

    # 3. Build hard negatives
    all_drugs_with_indications = set()
    drug_id_to_name: Dict[str, str] = {}
    for drug_id, disease_mesh, drug_name, disease_name in positive_pairs:
        all_drugs_with_indications.add(drug_id)
        if drug_name:
            drug_id_to_name[drug_id] = drug_name

    negative_pairs: List[Tuple[str, str, str, str]] = []
    for disease_mesh, disease_drugs_set in disease_to_drugs.items():
        disease_name = MESH_TO_NAME.get(disease_mesh, disease_mesh)
        hard_negs = all_drugs_with_indications - disease_drugs_set
        for drug_id in hard_negs:
            drug_name = drug_id_to_name.get(drug_id, "")
            negative_pairs.append((drug_id, disease_mesh, drug_name, disease_name))

    logger.info(f"Hard negative pairs: {len(negative_pairs)}")

    # Subsample negatives (3:1 ratio)
    max_negatives = len(positive_pairs) * 3
    if len(negative_pairs) > max_negatives:
        np.random.seed(42)
        indices = np.random.choice(len(negative_pairs), max_negatives, replace=False)
        negative_pairs = [negative_pairs[i] for i in indices]
        logger.info(f"Subsampled negatives to: {len(negative_pairs)}")

    # 4. Create feature matrix
    all_pairs = positive_pairs + negative_pairs
    labels = [1] * len(positive_pairs) + [0] * len(negative_pairs)

    # Original 512 dims + 5 TxGNN features
    txgnn_feature_dim = 5
    feature_dim = emb_dim * 4 + txgnn_feature_dim
    X = np.zeros((len(all_pairs), feature_dim), dtype=np.float32)

    txgnn_hit_count = 0
    logger.info(f"Creating features for {len(all_pairs)} pairs...")
    for i, (drug_id, disease_id, drug_name, disease_name) in enumerate(tqdm(all_pairs)):
        drug_idx = entity2id[drug_id]
        disease_idx = entity2id[disease_id]

        drug_emb = emb_np[drug_idx]
        disease_emb = emb_np[disease_idx]

        # Base features (512 dims)
        base_features = create_base_features(drug_emb, disease_emb)

        # TxGNN features (5 dims)
        txgnn_feats = txgnn_features.get_features(disease_name, drug_name)
        if txgnn_feats[0] > 0:  # Has TxGNN score
            txgnn_hit_count += 1

        # Combine
        X[i] = np.concatenate([base_features, np.array(txgnn_feats)])

    y = np.array(labels, dtype=np.int32)

    logger.info(f"TxGNN feature hits: {txgnn_hit_count}/{len(all_pairs)} ({100*txgnn_hit_count/len(all_pairs):.1f}%)")

    stats = {
        "positive_pairs": len(positive_pairs),
        "negative_pairs": len(negative_pairs),
        "every_cure_pairs": len(positive_pairs) - enhanced_count,
        "enhanced_pairs": enhanced_count,
        "diseases": len(disease_to_drugs),
        "unique_drugs": len(all_drugs_with_indications),
        "txgnn_feature_hits": txgnn_hit_count,
        "feature_dim": feature_dim,
    }

    return X, y, stats


def load_ground_truth(
    drugbank_lookup: Dict[str, str],
) -> Dict[str, Tuple[Set[str], List[str]]]:
    """Load ground truth for evaluation (same as original GB)."""
    name_to_drkg = {v.lower(): f"drkg:Compound::{k}" for k, v in drugbank_lookup.items()}

    df = pd.read_excel(REFERENCE_DIR / "everycure" / "indicationList.xlsx")

    # Returns {disease_mesh: (set of drug_ids, list of drug_names)}
    disease_drugs: Dict[str, Tuple[Set[str], List[str]]] = {}

    for _, row in df.iterrows():
        disease = str(row.get("disease name", "")).strip().lower()
        drug = str(row.get("final normalized drug label", "")).strip()

        disease_mesh = DISEASE_MESH_MAP.get(disease)
        drug_id = name_to_drkg.get(drug.lower())

        if disease_mesh and drug_id:
            if disease_mesh not in disease_drugs:
                disease_drugs[disease_mesh] = (set(), [])
            disease_drugs[disease_mesh][0].add(drug_id)
            disease_drugs[disease_mesh][1].append(drug)

    # Add enhanced ground truth
    gt_path = EVAL_DIR / "enhanced_ground_truth.json"
    if gt_path.exists():
        with open(gt_path) as f:
            enhanced = json.load(f)

        for disease_name, drugs in enhanced.items():
            disease_mesh = DISEASE_MESH_MAP.get(disease_name.lower())
            if not disease_mesh:
                continue

            if disease_mesh not in disease_drugs:
                disease_drugs[disease_mesh] = (set(), [])

            for drug in drugs:
                if drug["classification"] == "CONFIRMED":
                    disease_drugs[disease_mesh][0].add(drug["drug_id"])
                    disease_drugs[disease_mesh][1].append(drug.get("drug_name", ""))

    return disease_drugs


def evaluate_model(
    model: GradientBoostingClassifier,
    embeddings: np.ndarray,
    entity2id: Dict[str, int],
    ground_truth: Dict[str, Tuple[Set[str], List[str]]],
    all_drug_ids: List[str],
    drugbank_lookup: Dict[str, str],
    txgnn_features: TxGNNFeatures,
) -> Dict:
    """Evaluate model on all diseases."""
    results = {}

    # drkg_id -> drug_name
    drkg_to_name: Dict[str, str] = {}
    for db_id, name in drugbank_lookup.items():
        drkg_id = f"drkg:Compound::{db_id}"
        drkg_to_name[drkg_id] = name

    for disease_mesh, (known_drug_ids, known_drug_names) in tqdm(
        ground_truth.items(), desc="Evaluating"
    ):
        disease_name = MESH_TO_NAME.get(disease_mesh, disease_mesh)
        disease_idx = entity2id.get(disease_mesh)

        if disease_idx is None:
            continue

        disease_emb = embeddings[disease_idx]

        scores = []
        valid_drugs = []

        for drug_id in all_drug_ids:
            drug_idx = entity2id.get(drug_id)
            if drug_idx is None:
                continue

            drug_emb = embeddings[drug_idx]

            # Base features
            base_features = create_base_features(drug_emb, disease_emb)

            # TxGNN features
            drug_name = drkg_to_name.get(drug_id, "")
            txgnn_feats = txgnn_features.get_features(disease_name, drug_name)

            # Combine
            features = np.concatenate([base_features, np.array(txgnn_feats)]).reshape(
                1, -1
            )
            score = model.predict_proba(features)[0, 1]
            scores.append(score)
            valid_drugs.append(drug_id)

        # Rank drugs
        ranked_indices = np.argsort(scores)[::-1]
        ranked_drugs = [valid_drugs[i] for i in ranked_indices]

        def recall_at_k(k: int) -> Tuple[int, int, float]:
            top_k = set(ranked_drugs[:k])
            found = len(top_k & known_drug_ids)
            recall = found / len(known_drug_ids) if known_drug_ids else 0
            return found, len(known_drug_ids), recall

        r30 = recall_at_k(30)
        r50 = recall_at_k(50)
        r100 = recall_at_k(100)

        results[disease_name] = {
            "recall@30": r30[2],
            "recall@50": r50[2],
            "recall@100": r100[2],
            "found@30": r30[0],
            "found@50": r50[0],
            "found@100": r100[0],
            "total_known": r30[1],
        }

    return results


def main():
    """Main training pipeline."""
    logger.info("=" * 70)
    logger.info("Train GB Model with TxGNN Features")
    logger.info("=" * 70)

    np.random.seed(42)

    # Load data
    logger.info("\n1. Loading data...")
    embeddings, entity2id = load_transe_embeddings()
    every_cure_data = load_every_cure_data()
    enhanced_gt = load_enhanced_ground_truth()
    drugbank_lookup = load_drugbank_lookup()

    # Load TxGNN predictions
    logger.info("\n2. Loading TxGNN predictions...")
    txgnn_features = TxGNNFeatures(REFERENCE_DIR / "txgnn_predictions_final.csv")
    txgnn_features.load(drugbank_lookup)

    # Build training data
    logger.info("\n3. Building training data with TxGNN features...")
    X, y, stats = build_training_data(
        embeddings, entity2id, every_cure_data, enhanced_gt, drugbank_lookup, txgnn_features
    )

    logger.info(f"\nTraining data stats:")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")

    # Train/test split
    logger.info("\n4. Training model...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train GB classifier (same parameters as original)
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbose=1,
    )

    model.fit(X_train, y_train)

    # Evaluate on test set
    logger.info("\n5. Evaluating on test set...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    auroc = roc_auc_score(y_test, y_pred_proba)
    auprc = average_precision_score(y_test, y_pred_proba)

    logger.info(f"Test AUROC: {auroc:.4f}")
    logger.info(f"Test AUPRC: {auprc:.4f}")

    # Feature importance analysis
    logger.info("\n6. Feature importance analysis...")
    emb_dim = embeddings.shape[1]
    importance = model.feature_importances_

    # Split importance by feature type
    concat_imp = importance[: emb_dim * 2].sum()
    product_imp = importance[emb_dim * 2 : emb_dim * 3].sum()
    diff_imp = importance[emb_dim * 3 : emb_dim * 4].sum()
    txgnn_imp = importance[emb_dim * 4 :].sum()

    total_imp = concat_imp + product_imp + diff_imp + txgnn_imp
    logger.info(f"Feature importance breakdown:")
    logger.info(f"  Concat features: {concat_imp/total_imp*100:.1f}%")
    logger.info(f"  Product features: {product_imp/total_imp*100:.1f}%")
    logger.info(f"  Difference features: {diff_imp/total_imp*100:.1f}%")
    logger.info(f"  TxGNN features: {txgnn_imp/total_imp*100:.1f}%")

    # Detailed TxGNN feature importance
    txgnn_feature_names = [
        "txgnn_normalized_score",
        "txgnn_normalized_rank",
        "txgnn_in_top_10",
        "txgnn_in_top_30",
        "txgnn_in_top_50",
    ]
    logger.info(f"\n  TxGNN feature breakdown:")
    for i, name in enumerate(txgnn_feature_names):
        feat_imp = importance[emb_dim * 4 + i]
        logger.info(f"    {name}: {feat_imp/total_imp*100:.2f}%")

    # Evaluate on ground truth diseases
    logger.info("\n7. Evaluating Recall@30 on ground truth...")
    emb_np = embeddings.numpy()
    all_drug_ids = [eid for eid in entity2id.keys() if "Compound" in eid]
    ground_truth = load_ground_truth(drugbank_lookup)

    results = evaluate_model(
        model, emb_np, entity2id, ground_truth, all_drug_ids, drugbank_lookup, txgnn_features
    )

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS: GB Model with TxGNN Features")
    print("=" * 80)

    total_found_30 = 0
    total_known = 0

    print(
        f"\n{'Disease':<25} {'R@30':>8} {'R@50':>8} {'R@100':>8} {'Found/Known':>12}"
    )
    print("-" * 70)

    for disease in sorted(results.keys()):
        r = results[disease]
        print(
            f"{disease:<25} {r['recall@30']*100:>7.1f}% {r['recall@50']*100:>7.1f}% "
            f"{r['recall@100']*100:>7.1f}% {r['found@30']:>5}/{r['total_known']:<5}"
        )
        total_found_30 += r["found@30"]
        total_known += r["total_known"]

    print("-" * 70)
    agg_recall = total_found_30 / total_known if total_known > 0 else 0
    print(f"{'AGGREGATE':<25} {agg_recall*100:>7.1f}%")
    print(f"\nTotal: {total_found_30}/{total_known} drugs found in top 30")

    # Save model
    model_path = MODELS_DIR / "drug_repurposing_gb_txgnn_features.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"\nSaved model to: {model_path}")

    # Save results
    output = {
        "model_name": "GB with TxGNN Features",
        "test_auroc": auroc,
        "test_auprc": auprc,
        "aggregate_recall@30": agg_recall,
        "total_found@30": total_found_30,
        "total_known": total_known,
        "training_stats": stats,
        "feature_importance": {
            "concat": float(concat_imp / total_imp),
            "product": float(product_imp / total_imp),
            "difference": float(diff_imp / total_imp),
            "txgnn_total": float(txgnn_imp / total_imp),
            "txgnn_detailed": {
                name: float(importance[emb_dim * 4 + i] / total_imp)
                for i, name in enumerate(txgnn_feature_names)
            },
        },
        "by_disease": results,
    }

    output_path = ANALYSIS_DIR / "gb_txgnn_features_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved results to: {output_path}")

    # Compare with original GB model
    logger.info("\n" + "=" * 70)
    logger.info("Comparison with Original GB Enhanced Model")
    logger.info("=" * 70)

    original_metrics_path = MODELS_DIR / "gb_enhanced_evaluation.json"
    if original_metrics_path.exists():
        with open(original_metrics_path) as f:
            original_results = json.load(f)
        original_recall = original_results["aggregate_recall@30"]
        logger.info(f"Original GB Enhanced R@30: {original_recall*100:.1f}%")
        logger.info(f"GB + TxGNN Features R@30: {agg_recall*100:.1f}%")
        improvement = (agg_recall - original_recall) / original_recall * 100
        logger.info(f"Change: {improvement:+.1f}%")
    else:
        logger.warning("Original GB model results not found for comparison")

    logger.success("\nTraining complete!")

    return model, output


if __name__ == "__main__":
    main()
