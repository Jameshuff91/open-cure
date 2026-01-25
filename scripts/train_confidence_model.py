#!/usr/bin/env python3
"""
Train the confidence calibration model.

Collects features from all ground truth predictions and trains a model
to predict which predictions will be in the top-30.
"""

import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Set, List
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from confidence_calibration import (
    ConfidenceCalibrator,
    PredictionFeatures,
    classify_drug_type,
    classify_disease_category,
    interpret_confidence,
)
from pathway_features import PathwayEnrichment
from chemical_features import DrugFingerprinter, compute_tanimoto_similarity
from atc_features import ATCMapper

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
MODELS_DIR = PROJECT_ROOT / "models"


def main() -> None:
    print("=" * 70)
    print("CONFIDENCE CALIBRATION TRAINING")
    print("=" * 70)

    # Load resources
    print("\n1. Loading resources...")

    with open(MODELS_DIR / "drug_repurposing_gb_enhanced.pkl", "rb") as f:
        model = pickle.load(f)

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

    with open(REFERENCE_DIR / "drug_targets.json") as f:
        drug_targets = {k: set(v) for k, v in json.load(f).items()}

    with open(REFERENCE_DIR / "disease_genes.json") as f:
        disease_genes = {k: set(v) for k, v in json.load(f).items()}

    with open(REFERENCE_DIR / "mesh_mappings_from_agents.json") as f:
        mesh_data = json.load(f)
    mesh_mappings = {}
    for batch_data in mesh_data.values():
        if isinstance(batch_data, dict):
            for disease_name, mesh_id in batch_data.items():
                if mesh_id and str(mesh_id).startswith(("D", "C")):
                    mesh_mappings[disease_name.lower()] = f"drkg:Disease::MESH:{mesh_id}"

    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)
    name_to_id = {name.lower(): f"drkg:Compound::{db_id}" for db_id, name in id_to_name.items()}
    id_to_drug_name = {f"drkg:Compound::{db_id}": name for db_id, name in id_to_name.items()}

    with open(REFERENCE_DIR / "everycure_gt_for_txgnn.json") as f:
        gt_raw = json.load(f)

    # Load feature modules
    print("   Loading pathway enrichment...")
    pe = PathwayEnrichment()
    print("   Loading fingerprinter...")
    fingerprinter = DrugFingerprinter(use_cache=True)
    print("   Loading ATC mapper...")
    atc_mapper = ATCMapper()

    # Build drug embeddings lookup
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

    # Collect training data
    print("\n2. Collecting training features...")

    all_features: List[np.ndarray] = []
    all_labels: List[int] = []
    all_metadata: List[Dict] = []  # For analysis

    diseases_evaluated = 0

    for disease_name, disease_data in tqdm(gt_raw.items(), desc="Diseases"):
        mesh_id = mesh_mappings.get(disease_name.lower())
        if not mesh_id:
            continue

        disease_idx = entity2id.get(mesh_id)
        if disease_idx is None:
            continue

        mesh_short = mesh_id.split("MESH:")[-1]
        disease_emb = embeddings[disease_idx]
        dis_genes = disease_genes.get(f"MESH:{mesh_short}", set())
        dis_cats = classify_disease_category(disease_name)

        # Get GT drugs
        gt_drug_ids = set()
        gt_drug_names = []
        for drug_info in disease_data.get('drugs', []):
            drug_name = drug_info['name']
            drug_id = name_to_id.get(drug_name.lower())
            if drug_id and drug_id in drug_id_to_local_idx:
                gt_drug_ids.add(drug_id)
                gt_drug_names.append(drug_name)

        if not gt_drug_ids:
            continue

        diseases_evaluated += 1

        # Get GT fingerprints
        gt_fps = []
        for dn in gt_drug_names:
            fp = fingerprinter.get_fingerprint(dn, fetch_if_missing=False)
            if fp is not None:
                gt_fps.append(fp)

        # Score all drugs
        n_drugs = len(drug_embs)
        disease_emb_tiled = np.tile(disease_emb, (n_drugs, 1))
        concat_feats = np.hstack([drug_embs, disease_emb_tiled])
        product_feats = drug_embs * disease_emb_tiled
        diff_feats = drug_embs - disease_emb_tiled
        base_features = np.hstack([concat_feats, product_feats, diff_feats])
        base_scores = model.predict_proba(base_features)[:, 1]

        # Compute features for each drug
        drug_features: List[PredictionFeatures] = []

        for i, drug_id in enumerate(valid_drug_ids):
            db_id = drug_id.split("::")[-1]
            drug_name = id_to_drug_name.get(drug_id, "")
            drug_types = classify_drug_type(drug_name)

            # Target overlap
            drug_genes = drug_targets.get(db_id, set())
            target_overlap = len(drug_genes & dis_genes)

            # ATC score
            atc_score = atc_mapper.get_mechanism_score(drug_name, disease_name)

            # Pathway overlap
            po, _, _ = pe.get_pathway_overlap(db_id, f"MESH:{mesh_short}")

            # Chemical similarity
            chem_sim = 0.0
            query_fp = fingerprinter.get_fingerprint(drug_name, fetch_if_missing=False)
            if query_fp is not None and gt_fps:
                for gt_fp in gt_fps:
                    sim = compute_tanimoto_similarity(query_fp, gt_fp)
                    chem_sim = max(chem_sim, sim)

            # Compute boosted score (quad additive)
            boost = 1 + 0.01 * min(target_overlap, 10) + 0.05 * atc_score + 0.01 * min(po, 10)
            if chem_sim > 0.7:
                boost *= 1.2
            boosted_score = base_scores[i] * boost

            # Create feature object
            feats = PredictionFeatures(
                base_score=float(base_scores[i]),
                target_overlap=int(target_overlap),
                atc_score=float(atc_score),
                chemical_sim=float(chem_sim),
                pathway_overlap=int(po),
                is_biologic=drug_types['is_biologic'],
                is_kinase_inhibitor=drug_types['is_kinase_inhibitor'],
                is_antibiotic=drug_types['is_antibiotic'],
                is_cancer=dis_cats['is_cancer'],
                is_infectious=dis_cats['is_infectious'],
                is_autoimmune=dis_cats['is_autoimmune'],
                has_fingerprint=query_fp is not None,
                has_targets=len(drug_genes) > 0,
                has_atc=len(atc_mapper.get_atc_codes(drug_name)) > 0,
                boosted_score=float(boosted_score),
            )
            drug_features.append(feats)

        # Get top-30 rankings
        boosted_scores = np.array([f.boosted_score for f in drug_features])
        rankings = np.argsort(boosted_scores)[::-1]
        top_30_set = set(rankings[:30])

        # Collect training samples from GT drugs only
        # (we know these are positive if they're in top-30, negative if not)
        for drug_id in gt_drug_ids:
            local_idx = drug_id_to_local_idx[drug_id]
            feats = drug_features[local_idx]
            label = 1 if local_idx in top_30_set else 0

            all_features.append(feats.to_array())
            all_labels.append(label)
            all_metadata.append({
                'disease': disease_name,
                'drug': id_to_drug_name.get(drug_id, ""),
                'rank': int(np.where(rankings == local_idx)[0][0]),
            })

    print(f"\nDiseases evaluated: {diseases_evaluated}")
    print(f"Training samples: {len(all_labels)}")
    print(f"Positive samples (in top-30): {sum(all_labels)} ({sum(all_labels)/len(all_labels):.1%})")

    # Convert to arrays
    X = np.array(all_features)
    y = np.array(all_labels)

    # Train model
    print("\n3. Training confidence model...")
    calibrator = ConfidenceCalibrator()
    metrics = calibrator.train(X, y, use_cross_val=True)

    # Print results
    print("\n" + "=" * 70)
    print("CALIBRATION RESULTS")
    print("=" * 70)

    print(f"\nBrier Score: {metrics['brier_score']:.4f} (lower is better)")
    print(f"AUROC: {metrics['auroc']:.4f}")
    print(f"AUPRC: {metrics['auprc']:.4f}")
    print(f"ECE: {metrics['ece']:.4f} (lower is better)")

    print("\nCalibration curve:")
    print(f"  {'Predicted':<12} {'Actual':<12}")
    print("  " + "-" * 24)
    for pred, true in zip(metrics['calibration_pred'], metrics['calibration_true']):
        print(f"  {pred:.2f}         {true:.2f}")

    # Feature importances
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCES")
    print("=" * 70)

    importances = calibrator.get_feature_importances()
    print(f"\n{'Feature':<25} {'Coefficient':>12}")
    print("-" * 40)
    for feat, coef in importances.items():
        direction = "+" if coef > 0 else ""
        print(f"{feat:<25} {direction}{coef:>11.4f}")

    # Save model
    print("\n4. Saving model...")
    model_path = MODELS_DIR / "confidence_calibrator.pkl"
    calibrator.save(model_path)
    print(f"   Saved to: {model_path}")

    # Test on some examples
    print("\n" + "=" * 70)
    print("EXAMPLE PREDICTIONS")
    print("=" * 70)

    # Get predictions for all training samples
    confidences = calibrator.predict_batch(X)

    # Show some high-confidence hits
    print("\nHigh-confidence HITS (correctly in top-30):")
    high_conf_hits = [
        (i, confidences[i], all_metadata[i])
        for i in range(len(y)) if y[i] == 1 and confidences[i] > 0.7
    ]
    high_conf_hits.sort(key=lambda x: x[1], reverse=True)
    for _, conf, meta in high_conf_hits[:5]:
        print(f"  {meta['drug'][:30]:<30} → {meta['disease'][:25]:<25} (conf={conf:.2f}, rank={meta['rank']})")

    # Show some low-confidence misses
    print("\nLow-confidence MISSES (correctly not in top-30):")
    low_conf_misses = [
        (i, confidences[i], all_metadata[i])
        for i in range(len(y)) if y[i] == 0 and confidences[i] < 0.3
    ]
    low_conf_misses.sort(key=lambda x: x[1])
    for _, conf, meta in low_conf_misses[:5]:
        print(f"  {meta['drug'][:30]:<30} → {meta['disease'][:25]:<25} (conf={conf:.2f}, rank={meta['rank']})")

    # Show calibration by confidence tier
    print("\n" + "=" * 70)
    print("CALIBRATION BY CONFIDENCE TIER")
    print("=" * 70)

    tiers = [
        ("Very High (≥0.8)", 0.8, 1.0),
        ("High (0.6-0.8)", 0.6, 0.8),
        ("Medium (0.4-0.6)", 0.4, 0.6),
        ("Low (0.2-0.4)", 0.2, 0.4),
        ("Very Low (<0.2)", 0.0, 0.2),
    ]

    print(f"\n{'Tier':<20} {'Samples':>10} {'Actual Hit%':>15} {'Pred Conf':>12}")
    print("-" * 60)

    for name, low, high in tiers:
        mask = (confidences >= low) & (confidences < high)
        if mask.sum() > 0:
            actual_rate = y[mask].mean()
            pred_rate = confidences[mask].mean()
            print(f"{name:<20} {mask.sum():>10} {actual_rate:>14.1%} {pred_rate:>11.2f}")


if __name__ == "__main__":
    main()
