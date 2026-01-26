#!/usr/bin/env python3
"""
Evaluate biologic penalty approach - penalize incompatible biologic-disease pairs.

Uses WHO INN naming convention ONLY (not known targets) to avoid circularity.
The -mab naming convention encodes target information in the drug name itself.
"""

import json
import pickle
import re
import sys
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
MODELS_DIR = PROJECT_ROOT / "models"


class TargetCategory(Enum):
    TUMOR = "tumor"
    IMMUNE = "immune"
    CARDIOVASCULAR = "cardiovascular"
    NERVOUS = "nervous"
    CYTOKINE = "cytokine"
    BONE = "bone"
    BACTERIAL = "bacterial"
    VIRAL = "viral"
    UNKNOWN = "unknown"


class DiseaseCategory(Enum):
    CANCER = "cancer"
    AUTOIMMUNE = "autoimmune"
    CARDIOVASCULAR = "cardiovascular"
    NEUROLOGICAL = "neurological"
    METABOLIC = "metabolic"
    INFECTIOUS = "infectious"
    BONE = "bone"
    OTHER = "other"


# Target infixes from WHO INN naming convention
TARGET_INFIXES = {
    "tu": TargetCategory.TUMOR,
    "tum": TargetCategory.TUMOR,
    "li": TargetCategory.IMMUNE,
    "lim": TargetCategory.IMMUNE,
    "ci": TargetCategory.CARDIOVASCULAR,
    "cir": TargetCategory.CARDIOVASCULAR,
    "ne": TargetCategory.NERVOUS,
    "ner": TargetCategory.NERVOUS,
    "ki": TargetCategory.CYTOKINE,
    "kin": TargetCategory.CYTOKINE,
    "os": TargetCategory.BONE,
    "so": TargetCategory.BONE,
    "ba": TargetCategory.BACTERIAL,
    "vi": TargetCategory.VIRAL,
}

# Disease category patterns
AUTOIMMUNE_PATTERNS = [
    r"multiple sclerosis", r"\bms\b", r"rheumatoid", r"lupus", r"psoriasis",
    r"crohn", r"ulcerative colitis", r"ankylosing spondylitis", r"sjögren",
    r"myasthenia", r"autoimmune", r"inflammatory bowel", r"atopic dermatitis",
    r"eczema", r"celiac", r"graves", r"hashimoto",
]

CANCER_PATTERNS = [
    r"cancer", r"carcinoma", r"melanoma", r"leukemia", r"lymphoma",
    r"sarcoma", r"tumor", r"myeloma", r"neoplasm", r"malignant",
    r"glioblastoma", r"neuroblastoma", r"adenocarcinoma",
]

CARDIOVASCULAR_PATTERNS = [
    r"heart failure", r"hypertension", r"coronary", r"myocardial",
    r"arrhythmia", r"atrial fibrillation", r"cardiomyopathy",
    r"atherosclerosis", r"thrombosis",
]

NEUROLOGICAL_PATTERNS = [
    r"parkinson", r"alzheimer", r"dementia", r"epilepsy", r"migraine",
    r"huntington", r"\bals\b", r"amyotrophic", r"neurodegenerat",
    r"neuropath",
]

BONE_PATTERNS = [
    r"osteoporosis", r"osteopenia", r"paget", r"bone loss",
]

INFECTIOUS_PATTERNS = [
    r"infection", r"sepsis", r"pneumonia", r"tuberculosis", r"hiv",
    r"hepatitis", r"influenza", r"covid", r"bacterial", r"viral",
]

# Compatibility matrix: which targets work for which diseases
COMPATIBILITY = {
    TargetCategory.TUMOR: [DiseaseCategory.CANCER],
    TargetCategory.IMMUNE: [DiseaseCategory.AUTOIMMUNE, DiseaseCategory.CANCER],
    TargetCategory.CYTOKINE: [DiseaseCategory.AUTOIMMUNE],
    TargetCategory.CARDIOVASCULAR: [DiseaseCategory.CARDIOVASCULAR],
    TargetCategory.NERVOUS: [DiseaseCategory.NEUROLOGICAL],
    TargetCategory.BONE: [DiseaseCategory.BONE],
    TargetCategory.BACTERIAL: [DiseaseCategory.INFECTIOUS],
    TargetCategory.VIRAL: [DiseaseCategory.INFECTIOUS],
    TargetCategory.UNKNOWN: [],  # Unknown = don't penalize (uncertain)
}


def create_features(drug_emb: np.ndarray, disease_emb: np.ndarray) -> np.ndarray:
    """Create features from embeddings (same as training)."""
    concat = np.concatenate([drug_emb, disease_emb])
    product = drug_emb * disease_emb
    diff = drug_emb - disease_emb
    return np.concatenate([concat, product, diff])


def create_features_batch(
    drug_embs: np.ndarray, disease_emb: np.ndarray
) -> np.ndarray:
    """Create features for all drugs against one disease (batch)."""
    n_drugs = drug_embs.shape[0]
    disease_emb_tiled = np.tile(disease_emb, (n_drugs, 1))

    concat = np.hstack([drug_embs, disease_emb_tiled])
    product = drug_embs * disease_emb_tiled
    diff = drug_embs - disease_emb_tiled
    return np.hstack([concat, product, diff])


def extract_target_from_name(drug: str) -> Optional[TargetCategory]:
    """Extract target category from -mab naming convention ONLY."""
    drug_lower = drug.lower()

    if not drug_lower.endswith("mab"):
        return None

    stem = drug_lower[:-3]  # Remove 'mab'

    # Check for infixes near the end
    for infix, target in TARGET_INFIXES.items():
        if infix in stem[-6:]:
            return target

    return TargetCategory.UNKNOWN


def categorize_disease(disease: str) -> DiseaseCategory:
    """Categorize a disease based on patterns."""
    disease_lower = disease.lower()

    for pattern in CANCER_PATTERNS:
        if re.search(pattern, disease_lower):
            return DiseaseCategory.CANCER

    for pattern in AUTOIMMUNE_PATTERNS:
        if re.search(pattern, disease_lower):
            return DiseaseCategory.AUTOIMMUNE

    for pattern in CARDIOVASCULAR_PATTERNS:
        if re.search(pattern, disease_lower):
            return DiseaseCategory.CARDIOVASCULAR

    for pattern in NEUROLOGICAL_PATTERNS:
        if re.search(pattern, disease_lower):
            return DiseaseCategory.NEUROLOGICAL

    for pattern in BONE_PATTERNS:
        if re.search(pattern, disease_lower):
            return DiseaseCategory.BONE

    for pattern in INFECTIOUS_PATTERNS:
        if re.search(pattern, disease_lower):
            return DiseaseCategory.INFECTIOUS

    return DiseaseCategory.OTHER


def is_compatible(drug: str, disease: str) -> tuple[bool, str]:
    """
    Check if a biologic is compatible with a disease using naming convention only.

    Returns: (is_compatible, reason)
    """
    target = extract_target_from_name(drug)

    if target is None:
        return True, "not_biologic"

    if target == TargetCategory.UNKNOWN:
        return True, "unknown_target"  # Don't penalize uncertain cases

    disease_cat = categorize_disease(disease)

    if disease_cat == DiseaseCategory.OTHER:
        return True, "unknown_disease"  # Don't penalize uncertain cases

    compatible_diseases = COMPATIBILITY.get(target, [])

    if disease_cat in compatible_diseases:
        return True, f"compatible:{target.value}->{disease_cat.value}"
    else:
        return False, f"incompatible:{target.value}->{disease_cat.value}"


def main() -> None:
    print("=" * 70)
    print("BIOLOGIC PENALTY EVALUATION")
    print("Uses WHO INN naming convention (non-circular)")
    print("=" * 70)
    sys.stdout.flush()

    # Load baseline model
    print("\n1. Loading baseline model...")
    sys.stdout.flush()
    with open(MODELS_DIR / "drug_repurposing_gb_enhanced.pkl", "rb") as f:
        model = pickle.load(f)

    # Load embeddings
    print("2. Loading embeddings...")
    sys.stdout.flush()
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

    if embeddings is None:
        print("ERROR: Could not load embeddings")
        return

    entity2id = checkpoint.get("entity2id", {})

    # Load mappings
    print("3. Loading mappings...")
    sys.stdout.flush()
    with open(REFERENCE_DIR / "mesh_mappings_from_agents.json") as f:
        mesh_data = json.load(f)

    mesh_mappings: dict[str, str] = {}
    mesh_to_name: dict[str, str] = {}
    for batch_data in mesh_data.values():
        if isinstance(batch_data, dict):
            for disease_name, mesh_id in batch_data.items():
                if mesh_id:
                    mesh_str = str(mesh_id)
                    if mesh_str.startswith("D") or mesh_str.startswith("C"):
                        full_id = f"drkg:Disease::MESH:{mesh_str}"
                        mesh_mappings[disease_name.lower()] = full_id
                        mesh_to_name[full_id] = disease_name

    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)
    name_to_id = {name.lower(): f"drkg:Compound::{db_id}" for db_id, name in id_to_name.items()}
    id_to_drug_name = {f"drkg:Compound::{db_id}": name for db_id, name in id_to_name.items()}

    # Load ground truth
    print("4. Loading ground truth...")
    sys.stdout.flush()
    with open(REFERENCE_DIR / "everycure_gt_for_txgnn.json") as f:
        gt_raw = json.load(f)

    # Convert GT format
    gt: dict[str, list[str]] = {}
    for disease_name, disease_data in gt_raw.items():
        mesh_id = mesh_mappings.get(disease_name.lower())
        if mesh_id:
            drug_ids = []
            for drug_info in disease_data.get('drugs', []):
                drug_name_lower = drug_info['name'].lower()
                drug_id = name_to_id.get(drug_name_lower)
                if drug_id:
                    drug_ids.append(drug_id)
            if drug_ids:
                gt[mesh_id] = drug_ids

    # Get all drug IDs
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

    # Pre-compute drug names and compatibility per disease category
    drug_names = [id_to_drug_name.get(did, "") for did in valid_drug_ids]
    drug_targets = [extract_target_from_name(name) for name in drug_names]

    print(f"\n   Diseases in GT: {len(gt)}")
    print(f"   Drugs available: {len(valid_drug_ids)}")
    sys.stdout.flush()

    # Count biologics
    biologic_count = sum(1 for t in drug_targets if t is not None)
    print(f"   Biologics (-mab): {biologic_count}")
    sys.stdout.flush()

    # Pre-compute all scores (batch)
    print("\n5. Computing all predictions (batch)...")
    sys.stdout.flush()

    # Store results per disease
    disease_scores: dict[str, np.ndarray] = {}

    for disease_id, true_drug_ids in tqdm(gt.items(), desc="Scoring diseases"):
        disease_idx = entity2id.get(disease_id)
        if disease_idx is None:
            continue

        disease_emb = embeddings[disease_idx]

        # Batch predict all drugs for this disease
        features = create_features_batch(drug_embs, disease_emb)
        scores = model.predict_proba(features)[:, 1]
        disease_scores[disease_id] = scores

    print(f"   Scored {len(disease_scores)} diseases")
    sys.stdout.flush()

    # Test different penalty factors
    print("\n" + "=" * 70)
    print("PENALTY FACTOR SWEEP")
    print("=" * 70)
    sys.stdout.flush()

    factors = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    best_factor = 1.0
    best_recall = 0.0
    results_by_factor: dict[float, dict] = {}

    for penalty_factor in factors:
        total_hits = 0
        total_possible = 0
        total_penalized = 0
        diseases_helped = 0
        diseases_hurt = 0

        for disease_id, true_drug_ids in gt.items():
            if disease_id not in disease_scores:
                continue

            disease_name = mesh_to_name.get(disease_id, disease_id)
            disease_cat = categorize_disease(disease_name)
            scores = disease_scores[disease_id].copy()

            # Count baseline hits
            ranked_baseline = np.argsort(-scores)[:30]
            top30_baseline = {valid_drug_ids[i] for i in ranked_baseline}
            hits_baseline = len(top30_baseline & set(true_drug_ids))

            # Apply penalties for incompatible biologics
            for i, (drug_id, target) in enumerate(zip(valid_drug_ids, drug_targets)):
                if target is None or target == TargetCategory.UNKNOWN:
                    continue
                if disease_cat == DiseaseCategory.OTHER:
                    continue

                compatible_diseases = COMPATIBILITY.get(target, [])
                if disease_cat not in compatible_diseases:
                    scores[i] *= penalty_factor
                    total_penalized += 1

            # Count penalized hits
            ranked_penalized = np.argsort(-scores)[:30]
            top30_penalized = {valid_drug_ids[i] for i in ranked_penalized}
            hits_penalized = len(top30_penalized & set(true_drug_ids))

            total_hits += hits_penalized
            total_possible += min(len(true_drug_ids), 30)

            if hits_penalized > hits_baseline:
                diseases_helped += 1
            elif hits_penalized < hits_baseline:
                diseases_hurt += 1

        recall = total_hits / total_possible if total_possible > 0 else 0

        results_by_factor[penalty_factor] = {
            'recall': recall,
            'biologics_penalized': total_penalized,
            'diseases_helped': diseases_helped,
            'diseases_hurt': diseases_hurt,
        }

        if recall > best_recall:
            best_recall = recall
            best_factor = penalty_factor

        print(f"Factor {penalty_factor:.1f}: R@30 = {recall*100:.2f}% "
              f"(helped={diseases_helped}, hurt={diseases_hurt})")
        sys.stdout.flush()

    # Get baseline (factor 1.0 = no penalty)
    baseline_recall = results_by_factor[1.0]['recall']

    print(f"\nBaseline R@30 (no penalty): {baseline_recall*100:.2f}%")
    print(f"Best R@30: {best_recall*100:.2f}% at factor {best_factor}")
    print(f"Delta: {(best_recall - baseline_recall)*100:+.2f}%")
    sys.stdout.flush()

    # Detailed analysis with best factor
    if best_factor < 1.0:
        print("\n" + "=" * 70)
        print(f"DETAILED ANALYSIS (factor={best_factor})")
        print("=" * 70)
        sys.stdout.flush()

        # Re-run with detailed tracking
        per_disease_results = []
        penalty_applications: list[dict] = []

        for disease_id, true_drug_ids in gt.items():
            if disease_id not in disease_scores:
                continue

            disease_name = mesh_to_name.get(disease_id, disease_id)
            disease_cat = categorize_disease(disease_name)
            scores_baseline = disease_scores[disease_id].copy()
            scores_penalized = scores_baseline.copy()

            # Apply penalties
            for i, (drug_id, target, drug_name) in enumerate(zip(valid_drug_ids, drug_targets, drug_names)):
                if target is None or target == TargetCategory.UNKNOWN:
                    continue
                if disease_cat == DiseaseCategory.OTHER:
                    continue

                compatible_diseases = COMPATIBILITY.get(target, [])
                if disease_cat not in compatible_diseases:
                    original_score = scores_baseline[i]
                    penalized_score = original_score * best_factor
                    scores_penalized[i] = penalized_score

                    penalty_applications.append({
                        'drug': drug_name,
                        'disease': disease_name,
                        'target': target.value,
                        'disease_cat': disease_cat.value,
                        'original_score': float(original_score),
                        'penalized_score': float(penalized_score),
                        'in_gt': drug_id in true_drug_ids,
                    })

            ranked_baseline = np.argsort(-scores_baseline)[:30]
            ranked_penalized = np.argsort(-scores_penalized)[:30]

            top30_baseline = {valid_drug_ids[i] for i in ranked_baseline}
            top30_penalized = {valid_drug_ids[i] for i in ranked_penalized}

            hits_baseline = len(top30_baseline & set(true_drug_ids))
            hits_penalized = len(top30_penalized & set(true_drug_ids))

            per_disease_results.append({
                'disease': disease_name,
                'disease_cat': disease_cat.value,
                'hits_baseline': hits_baseline,
                'hits_penalized': hits_penalized,
                'delta': hits_penalized - hits_baseline,
            })

        # Show penalty summary
        print(f"\nTotal penalty applications: {len(penalty_applications)}")

        # Group by target -> disease category
        by_mismatch: dict[str, int] = defaultdict(int)
        for p in penalty_applications:
            key = f"{p['target']} -> {p['disease_cat']}"
            by_mismatch[key] += 1

        print("\nPenalty breakdown (target -> disease category):")
        for key, count in sorted(by_mismatch.items(), key=lambda x: -x[1])[:10]:
            print(f"  {key}: {count}")

        # Any GT drugs penalized?
        gt_penalized = [p for p in penalty_applications if p['in_gt']]
        if gt_penalized:
            print(f"\n⚠️ WARNING: {len(gt_penalized)} ground truth drugs were penalized:")
            for p in gt_penalized[:5]:
                print(f"  {p['drug']} -> {p['disease']} ({p['target']} -> {p['disease_cat']})")
        else:
            print("\n✓ No ground truth drugs were penalized (good!)")

        # Show biggest improvements
        sorted_by_delta = sorted(per_disease_results, key=lambda x: x['delta'], reverse=True)

        print("\nTop 5 diseases HELPED:")
        for d in sorted_by_delta[:5]:
            if d['delta'] > 0:
                print(f"  {d['disease']} ({d['disease_cat']}): "
                      f"{d['hits_baseline']} → {d['hits_penalized']} (+{d['delta']})")

        print("\nTop 5 diseases HURT:")
        for d in sorted_by_delta[-5:]:
            if d['delta'] < 0:
                print(f"  {d['disease']} ({d['disease_cat']}): "
                      f"{d['hits_baseline']} → {d['hits_penalized']} ({d['delta']})")
    else:
        print("\nNo improvement from biologic penalty - baseline is best.")


if __name__ == "__main__":
    main()
