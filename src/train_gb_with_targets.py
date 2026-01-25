#!/usr/bin/env python3
"""
Train GB model with drug-target features.

This adds target-based features to the embedding features:
- target_overlap: number of shared genes between drug targets and disease genes
- target_overlap_frac: fraction of drug targets that are disease genes
- has_target_overlap: binary indicator
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Set, List, Tuple
import numpy as np
import torch
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REFERENCE_DIR = DATA_DIR / "reference"
MODELS_DIR = PROJECT_ROOT / "models"


def load_target_data() -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Load drug-target and disease-gene mappings."""
    with open(REFERENCE_DIR / "drug_targets.json") as f:
        drug_targets_raw = json.load(f)
    drug_targets = {k: set(v) for k, v in drug_targets_raw.items()}

    with open(REFERENCE_DIR / "disease_genes.json") as f:
        disease_genes_raw = json.load(f)
    disease_genes = {k: set(v) for k, v in disease_genes_raw.items()}

    return drug_targets, disease_genes


def compute_target_features(
    drug_db_id: str,
    disease_mesh_id: str,
    drug_targets: Dict[str, Set[str]],
    disease_genes: Dict[str, Set[str]],
) -> np.ndarray:
    """Compute target-based features for a drug-disease pair."""
    drug_genes = drug_targets.get(drug_db_id, set())
    dis_genes = disease_genes.get(disease_mesh_id, set())

    overlap = drug_genes & dis_genes
    n_overlap = len(overlap)

    if len(drug_genes) > 0:
        frac = n_overlap / len(drug_genes)
    else:
        frac = 0.0

    return np.array([
        n_overlap,
        frac,
        1 if n_overlap > 0 else 0,
        len(drug_genes),
        len(dis_genes),
    ], dtype=np.float32)


def create_features_with_targets(
    drug_emb: np.ndarray,
    disease_emb: np.ndarray,
    target_features: np.ndarray,
) -> np.ndarray:
    """Create full feature vector including target features."""
    concat = np.concatenate([drug_emb, disease_emb])
    product = drug_emb * disease_emb
    diff = drug_emb - disease_emb
    return np.concatenate([concat, product, diff, target_features])


def main():
    print("=" * 70)
    print("TRAINING GB MODEL WITH TARGET FEATURES")
    print("=" * 70)

    # Load embeddings
    print("\n1. Loading TransE embeddings...")
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
    print(f"   Loaded {len(entity2id)} entities, embedding dim {embeddings.shape[1]}")

    # Load target data
    print("\n2. Loading target data...")
    drug_targets, disease_genes = load_target_data()
    print(f"   {len(drug_targets)} drugs with targets")
    print(f"   {len(disease_genes)} diseases with gene associations")

    # Load ground truth
    print("\n3. Loading ground truth...")
    with open(REFERENCE_DIR / "everycure_gt_for_txgnn.json") as f:
        gt = json.load(f)

    with open(REFERENCE_DIR / "mesh_mappings_from_agents.json") as f:
        mesh_data = json.load(f)

    mesh_mappings = {}
    for batch_data in mesh_data.values():
        if isinstance(batch_data, dict):
            for disease_name, mesh_id in batch_data.items():
                if mesh_id:
                    mesh_str = str(mesh_id)
                    if mesh_str.startswith("D"):
                        mesh_mappings[disease_name.lower()] = mesh_str

    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        drugbank_lookup = json.load(f)
    name_to_db = {name.lower(): db_id for db_id, name in drugbank_lookup.items()}

    # Build training data
    print("\n4. Building training data...")
    positive_pairs = []
    all_drug_ids = set()
    all_disease_ids = set()

    for disease_name, disease_data in gt.items():
        mesh_id = mesh_mappings.get(disease_name.lower())
        if not mesh_id:
            continue

        disease_drkg = f"drkg:Disease::MESH:{mesh_id}"
        if disease_drkg not in entity2id:
            continue

        for drug_info in disease_data['drugs']:
            drug_name = drug_info['name'].lower()
            db_id = name_to_db.get(drug_name)
            if not db_id:
                continue

            drug_drkg = f"drkg:Compound::{db_id}"
            if drug_drkg not in entity2id:
                continue

            positive_pairs.append((drug_drkg, disease_drkg, db_id, mesh_id))
            all_drug_ids.add(drug_drkg)
            all_disease_ids.add(disease_drkg)

    print(f"   Positive pairs: {len(positive_pairs)}")
    print(f"   Unique drugs: {len(all_drug_ids)}")
    print(f"   Unique diseases: {len(all_disease_ids)}")

    # Generate negative samples
    print("\n5. Generating negative samples...")
    np.random.seed(42)
    all_drugs_list = list(all_drug_ids)
    all_diseases_list = list(all_disease_ids)
    positive_set = {(d, dis) for d, dis, _, _ in positive_pairs}

    negative_pairs = []
    n_negatives = len(positive_pairs) * 3

    attempts = 0
    while len(negative_pairs) < n_negatives and attempts < n_negatives * 10:
        drug = np.random.choice(all_drugs_list)
        disease = np.random.choice(all_diseases_list)
        if (drug, disease) not in positive_set:
            # Extract IDs for target features
            db_id = drug.split("::")[-1]
            mesh_id = disease.split("MESH:")[-1]
            negative_pairs.append((drug, disease, db_id, mesh_id))
            positive_set.add((drug, disease))
        attempts += 1

    print(f"   Negative pairs: {len(negative_pairs)}")

    # Create feature matrix
    print("\n6. Creating feature matrix with target features...")
    X = []
    y = []

    for drug_drkg, disease_drkg, db_id, mesh_id in tqdm(positive_pairs, desc="Positives"):
        drug_idx = entity2id[drug_drkg]
        disease_idx = entity2id[disease_drkg]
        drug_emb = embeddings[drug_idx]
        disease_emb = embeddings[disease_idx]
        target_feats = compute_target_features(db_id, f"MESH:{mesh_id}", drug_targets, disease_genes)
        features = create_features_with_targets(drug_emb, disease_emb, target_feats)
        X.append(features)
        y.append(1)

    for drug_drkg, disease_drkg, db_id, mesh_id in tqdm(negative_pairs, desc="Negatives"):
        drug_idx = entity2id[drug_drkg]
        disease_idx = entity2id[disease_drkg]
        drug_emb = embeddings[drug_idx]
        disease_emb = embeddings[disease_idx]
        target_feats = compute_target_features(db_id, f"MESH:{mesh_id}", drug_targets, disease_genes)
        features = create_features_with_targets(drug_emb, disease_emb, target_feats)
        X.append(features)
        y.append(0)

    X = np.array(X)
    y = np.array(y)

    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Feature breakdown: 256 (concat) + 128 (product) + 128 (diff) + 5 (target) = {X.shape[1]}")

    # Train/test split
    print("\n7. Training model...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        verbose=1,
    )

    model.fit(X_train, y_train)

    # Evaluate
    print("\n8. Evaluating...")
    y_val_pred = model.predict_proba(X_val)[:, 1]
    auroc = roc_auc_score(y_val, y_val_pred)
    auprc = average_precision_score(y_val, y_val_pred)

    print(f"   Validation AUROC: {auroc:.3f}")
    print(f"   Validation AUPRC: {auprc:.3f}")

    # Feature importance
    print("\n9. Feature importance:")
    feature_names = (
        [f"concat_{i}" for i in range(256)] +
        [f"product_{i}" for i in range(128)] +
        [f"diff_{i}" for i in range(128)] +
        ["target_overlap", "target_overlap_frac", "has_target_overlap", "n_drug_targets", "n_disease_genes"]
    )

    importances = model.feature_importances_

    # Group by feature type
    concat_imp = importances[:256].sum()
    product_imp = importances[256:384].sum()
    diff_imp = importances[384:512].sum()
    target_imp = importances[512:].sum()

    print(f"   Concat features:  {concat_imp:.1%}")
    print(f"   Product features: {product_imp:.1%}")
    print(f"   Diff features:    {diff_imp:.1%}")
    print(f"   TARGET features:  {target_imp:.1%}")

    # Individual target feature importance
    print("\n   Target feature breakdown:")
    for i, name in enumerate(["target_overlap", "target_overlap_frac", "has_target_overlap", "n_drug_targets", "n_disease_genes"]):
        print(f"     {name}: {importances[512+i]:.3%}")

    # Save model
    model_path = MODELS_DIR / "drug_repurposing_gb_with_targets.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\n10. Saved model to {model_path}")

    # Save metadata
    metadata = {
        "n_positive_pairs": len(positive_pairs),
        "n_negative_pairs": len(negative_pairs),
        "feature_dim": X.shape[1],
        "val_auroc": auroc,
        "val_auprc": auprc,
        "feature_importance": {
            "concat": float(concat_imp),
            "product": float(product_imp),
            "diff": float(diff_imp),
            "target": float(target_imp),
        },
    }

    with open(MODELS_DIR / "drug_repurposing_gb_with_targets_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
