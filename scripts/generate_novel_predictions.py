#!/usr/bin/env python3
"""
Generate novel drug-disease predictions with confidence scores.

Finds high-confidence predictions that are NOT in the ground truth,
representing potential new drug repurposing opportunities.
"""

import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Set, List
from dataclasses import dataclass

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


@dataclass
class NovelPrediction:
    """A novel drug-disease prediction."""
    drug_name: str
    disease_name: str
    confidence: float
    boosted_score: float
    base_score: float
    target_overlap: int
    chemical_sim: float
    atc_score: float
    drug_type: str
    disease_category: str

    def to_dict(self) -> Dict:
        return {
            'drug': self.drug_name,
            'disease': self.disease_name,
            'confidence': self.confidence,
            'confidence_tier': interpret_confidence(self.confidence),
            'boosted_score': self.boosted_score,
            'base_score': self.base_score,
            'target_overlap': self.target_overlap,
            'chemical_sim': self.chemical_sim,
            'atc_score': self.atc_score,
            'drug_type': self.drug_type,
            'disease_category': self.disease_category,
        }


def main() -> None:
    print("=" * 70)
    print("NOVEL PREDICTION GENERATION")
    print("=" * 70)

    # Load resources
    print("\n1. Loading resources...")

    # Load confidence calibrator
    calibrator = ConfidenceCalibrator(MODELS_DIR / "confidence_calibrator.pkl")
    print("   ✓ Confidence calibrator loaded")

    with open(MODELS_DIR / "drug_repurposing_gb_enhanced.pkl", "rb") as f:
        model = pickle.load(f)
    print("   ✓ GB model loaded")

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
    print("   ✓ Embeddings loaded")

    with open(REFERENCE_DIR / "drug_targets.json") as f:
        drug_targets = {k: set(v) for k, v in json.load(f).items()}

    with open(REFERENCE_DIR / "disease_genes.json") as f:
        disease_genes = {k: set(v) for k, v in json.load(f).items()}

    with open(REFERENCE_DIR / "mesh_mappings_from_agents.json") as f:
        mesh_data = json.load(f)
    mesh_mappings = {}
    mesh_to_name = {}
    for batch_data in mesh_data.values():
        if isinstance(batch_data, dict):
            for disease_name, mesh_id in batch_data.items():
                if mesh_id and str(mesh_id).startswith(("D", "C")):
                    full_id = f"drkg:Disease::MESH:{mesh_id}"
                    mesh_mappings[disease_name.lower()] = full_id
                    mesh_to_name[full_id] = disease_name

    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)
    name_to_id = {name.lower(): f"drkg:Compound::{db_id}" for db_id, name in id_to_name.items()}
    id_to_drug_name = {f"drkg:Compound::{db_id}": name for db_id, name in id_to_name.items()}

    with open(REFERENCE_DIR / "everycure_gt_for_txgnn.json") as f:
        gt_raw = json.load(f)

    # Build ground truth set for quick lookup
    gt_pairs: Set[tuple] = set()
    for disease_name, disease_data in gt_raw.items():
        for drug_info in disease_data.get('drugs', []):
            gt_pairs.add((drug_info['name'].lower(), disease_name.lower()))

    print(f"   Ground truth pairs: {len(gt_pairs)}")

    # Load feature modules
    print("   Loading feature modules...")
    pe = PathwayEnrichment()
    fingerprinter = DrugFingerprinter(use_cache=True)
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

    print(f"   Drugs: {len(valid_drug_ids)}")

    # Collect novel predictions
    print("\n2. Generating predictions...")

    novel_predictions: List[NovelPrediction] = []
    min_confidence = 0.7  # Only keep high-confidence predictions

    # Process each disease
    disease_list = list(mesh_to_name.items())

    for mesh_id, disease_name in tqdm(disease_list, desc="Diseases"):
        disease_idx = entity2id.get(mesh_id)
        if disease_idx is None:
            continue

        mesh_short = mesh_id.split("MESH:")[-1]
        disease_emb = embeddings[disease_idx]
        dis_genes = disease_genes.get(f"MESH:{mesh_short}", set())
        dis_cats = classify_disease_category(disease_name)

        # Get GT drugs for this disease (to exclude)
        gt_drug_names_lower = set()
        disease_data = gt_raw.get(disease_name, {})
        for drug_info in disease_data.get('drugs', []):
            gt_drug_names_lower.add(drug_info['name'].lower())

        # Score all drugs
        n_drugs = len(drug_embs)
        disease_emb_tiled = np.tile(disease_emb, (n_drugs, 1))
        concat_feats = np.hstack([drug_embs, disease_emb_tiled])
        product_feats = drug_embs * disease_emb_tiled
        diff_feats = drug_embs - disease_emb_tiled
        base_features_arr = np.hstack([concat_feats, product_feats, diff_feats])
        base_scores = model.predict_proba(base_features_arr)[:, 1]

        # Process top candidates (top 100 by base score to save time)
        top_indices = np.argsort(base_scores)[-100:]

        for local_idx in top_indices:
            drug_id = valid_drug_ids[local_idx]
            db_id = drug_id.split("::")[-1]
            drug_name = id_to_drug_name.get(drug_id, "")

            # Skip if in ground truth
            if drug_name.lower() in gt_drug_names_lower:
                continue

            drug_types = classify_drug_type(drug_name)

            # Target overlap
            drug_genes = drug_targets.get(db_id, set())
            target_overlap = len(drug_genes & dis_genes)

            # ATC score
            atc_score = atc_mapper.get_mechanism_score(drug_name, disease_name)

            # Pathway overlap
            po, _, _ = pe.get_pathway_overlap(db_id, f"MESH:{mesh_short}")

            # Chemical similarity (to GT drugs)
            chem_sim = 0.0
            query_fp = fingerprinter.get_fingerprint(drug_name, fetch_if_missing=False)
            if query_fp is not None:
                for gt_drug in gt_drug_names_lower:
                    gt_fp = fingerprinter.get_fingerprint(gt_drug, fetch_if_missing=False)
                    if gt_fp is not None:
                        sim = compute_tanimoto_similarity(query_fp, gt_fp)
                        chem_sim = max(chem_sim, sim)

            # Compute boosted score
            boost = 1 + 0.01 * min(target_overlap, 10) + 0.05 * atc_score + 0.01 * min(po, 10)
            if chem_sim > 0.7:
                boost *= 1.2
            boosted_score = base_scores[local_idx] * boost

            # Create feature object
            feats = PredictionFeatures(
                base_score=float(base_scores[local_idx]),
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

            # Get confidence
            confidence = calibrator.predict_confidence(feats)

            # Only keep high-confidence predictions
            if confidence >= min_confidence:
                # Determine drug type string
                if drug_types['is_biologic']:
                    drug_type_str = 'biologic'
                elif drug_types['is_kinase_inhibitor']:
                    drug_type_str = 'kinase_inhibitor'
                elif drug_types['is_antibiotic']:
                    drug_type_str = 'antibiotic'
                else:
                    drug_type_str = 'small_molecule'

                # Determine disease category string
                if dis_cats['is_cancer']:
                    dis_cat_str = 'cancer'
                elif dis_cats['is_infectious']:
                    dis_cat_str = 'infectious'
                elif dis_cats['is_autoimmune']:
                    dis_cat_str = 'autoimmune'
                else:
                    dis_cat_str = 'other'

                pred = NovelPrediction(
                    drug_name=drug_name,
                    disease_name=disease_name,
                    confidence=confidence,
                    boosted_score=boosted_score,
                    base_score=float(base_scores[local_idx]),
                    target_overlap=target_overlap,
                    chemical_sim=chem_sim,
                    atc_score=atc_score,
                    drug_type=drug_type_str,
                    disease_category=dis_cat_str,
                )
                novel_predictions.append(pred)

    # Sort by confidence
    novel_predictions.sort(key=lambda x: x.confidence, reverse=True)

    print(f"\nFound {len(novel_predictions)} novel predictions with confidence ≥ {min_confidence}")

    # Print top predictions
    print("\n" + "=" * 70)
    print("TOP 30 NOVEL PREDICTIONS")
    print("=" * 70)

    print(f"\n{'Drug':<30} {'Disease':<30} {'Conf':>6} {'Score':>7} {'Overlap':>8}")
    print("-" * 85)

    for pred in novel_predictions[:30]:
        print(f"{pred.drug_name[:28]:<30} {pred.disease_name[:28]:<30} {pred.confidence:>5.2f} {pred.boosted_score:>7.3f} {pred.target_overlap:>8}")

    # Stats by category
    print("\n" + "=" * 70)
    print("PREDICTIONS BY CONFIDENCE TIER")
    print("=" * 70)

    tiers = [
        ("Very High (≥0.9)", 0.9, 1.0),
        ("High (0.8-0.9)", 0.8, 0.9),
        ("Medium-High (0.7-0.8)", 0.7, 0.8),
    ]

    for name, low, high in tiers:
        count = len([p for p in novel_predictions if low <= p.confidence < high])
        print(f"{name}: {count}")

    # Stats by drug type
    print("\n" + "=" * 70)
    print("PREDICTIONS BY DRUG TYPE")
    print("=" * 70)

    drug_type_counts: Dict[str, int] = {}
    for pred in novel_predictions:
        drug_type_counts[pred.drug_type] = drug_type_counts.get(pred.drug_type, 0) + 1

    for dt, count in sorted(drug_type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{dt}: {count}")

    # Stats by disease category
    print("\n" + "=" * 70)
    print("PREDICTIONS BY DISEASE CATEGORY")
    print("=" * 70)

    dis_cat_counts: Dict[str, int] = {}
    for pred in novel_predictions:
        dis_cat_counts[pred.disease_category] = dis_cat_counts.get(pred.disease_category, 0) + 1

    for dc, count in sorted(dis_cat_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{dc}: {count}")

    # Save to file
    print("\n3. Saving predictions...")
    output_path = PROJECT_ROOT / "data" / "analysis" / "novel_predictions_with_confidence.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump([p.to_dict() for p in novel_predictions], f, indent=2)

    print(f"   Saved to: {output_path}")

    # Save top predictions for validation
    top_for_validation = [p.to_dict() for p in novel_predictions[:100]]
    validation_path = PROJECT_ROOT / "data" / "analysis" / "top_novel_for_validation.json"
    with open(validation_path, 'w') as f:
        json.dump(top_for_validation, f, indent=2)

    print(f"   Top 100 saved to: {validation_path}")


if __name__ == "__main__":
    main()
