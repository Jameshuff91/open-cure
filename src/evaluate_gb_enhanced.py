#!/usr/bin/env python3
"""
Evaluate enhanced GB model on the benchmark.

Compares:
1. Original GB model (baseline)
2. Enhanced GB model (Fix 4)

Reports Recall@30, Recall@50, Recall@100 per disease.
"""

import json
import pickle
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from loguru import logger

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REFERENCE_DIR = DATA_DIR / "reference"
MODELS_DIR = PROJECT_ROOT / "models"
EVAL_DIR = PROJECT_ROOT / "autonomous_evaluation"

# Disease mappings (same as training)
DISEASE_MESH_MAP = {
    "hiv infection": "drkg:Disease::MESH:D015658",
    "hepatitis c": "drkg:Disease::MESH:D006526",
    "tuberculosis": "drkg:Disease::MESH:D014376",
    "breast cancer": "drkg:Disease::MESH:D001943",
    "lung cancer": "drkg:Disease::MESH:D008175",
    "colorectal cancer": "drkg:Disease::MESH:D015179",
    "hypertension": "drkg:Disease::MESH:D006973",
    "heart failure": "drkg:Disease::MESH:D006333",
    "congestive heart failure": "drkg:Disease::MESH:D006333",
    "atrial fibrillation": "drkg:Disease::MESH:D001281",
    "epilepsy": "drkg:Disease::MESH:D004827",
    "parkinson disease": "drkg:Disease::MESH:D010300",
    "parkinson's disease": "drkg:Disease::MESH:D010300",
    "alzheimer disease": "drkg:Disease::MESH:D000544",
    "alzheimer's disease": "drkg:Disease::MESH:D000544",
    "rheumatoid arthritis": "drkg:Disease::MESH:D001172",
    "multiple sclerosis": "drkg:Disease::MESH:D009103",
    "psoriasis": "drkg:Disease::MESH:D011565",
    "type 2 diabetes": "drkg:Disease::MESH:D003924",
    "type 2 diabetes mellitus": "drkg:Disease::MESH:D003924",
    "noninsulin dependent diabetes mellitus type ii": "drkg:Disease::MESH:D003924",
    "obesity": "drkg:Disease::MESH:D009765",
    "asthma": "drkg:Disease::MESH:D001249",
    "copd": "drkg:Disease::MESH:D029424",
    "chronic obstructive pulmonary disease": "drkg:Disease::MESH:D029424",
    "osteoporosis": "drkg:Disease::MESH:D010024",
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


def create_features(drug_emb: np.ndarray, disease_emb: np.ndarray) -> np.ndarray:
    """Create features from embeddings."""
    concat = np.concatenate([drug_emb, disease_emb])
    product = drug_emb * disease_emb
    diff = drug_emb - disease_emb
    return np.concatenate([concat, product, diff])


def load_ground_truth() -> Dict[str, Set[str]]:
    """Load combined ground truth (Every Cure + enhanced confirmed)."""
    # Load drugbank lookup (inverted)
    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)
    name_to_id = {name.lower(): f"drkg:Compound::{db_id}" for db_id, name in id_to_name.items()}

    # Load Every Cure data
    df = pd.read_excel(REFERENCE_DIR / "everycure" / "indicationList.xlsx")

    disease_drugs: Dict[str, Set[str]] = defaultdict(set)

    for _, row in df.iterrows():
        disease = str(row.get("disease name", "")).strip().lower()
        drug = str(row.get("final normalized drug label", "")).strip().lower()

        disease_mesh = DISEASE_MESH_MAP.get(disease)
        drug_id = name_to_id.get(drug)

        if disease_mesh and drug_id:
            disease_drugs[disease_mesh].add(drug_id)

    # Load enhanced ground truth (CONFIRMED only)
    with open(EVAL_DIR / "enhanced_ground_truth.json") as f:
        enhanced = json.load(f)

    for disease_name, drugs in enhanced.items():
        disease_mesh = DISEASE_MESH_MAP.get(disease_name.lower())
        if not disease_mesh:
            continue

        for drug in drugs:
            if drug["classification"] == "CONFIRMED":
                disease_drugs[disease_mesh].add(drug["drug_id"])

    return dict(disease_drugs)


def evaluate_model(
    model,
    embeddings: np.ndarray,
    entity2id: Dict[str, int],
    ground_truth: Dict[str, Set[str]],
    all_drug_ids: List[str],
) -> Dict:
    """Evaluate model on all diseases."""
    results = {}

    for disease_mesh, known_drugs in tqdm(ground_truth.items(), desc="Evaluating"):
        disease_name = MESH_TO_NAME.get(disease_mesh, disease_mesh)
        disease_idx = entity2id.get(disease_mesh)

        if disease_idx is None:
            continue

        disease_emb = embeddings[disease_idx]

        # Score all drugs
        scores = []
        valid_drugs = []

        for drug_id in all_drug_ids:
            drug_idx = entity2id.get(drug_id)
            if drug_idx is None:
                continue

            drug_emb = embeddings[drug_idx]
            features = create_features(drug_emb, disease_emb).reshape(1, -1)
            score = model.predict_proba(features)[0, 1]
            scores.append(score)
            valid_drugs.append(drug_id)

        # Rank drugs
        ranked_indices = np.argsort(scores)[::-1]
        ranked_drugs = [valid_drugs[i] for i in ranked_indices]

        # Calculate recall at various k
        def recall_at_k(k: int) -> tuple:
            top_k = set(ranked_drugs[:k])
            found = len(top_k & known_drugs)
            recall = found / len(known_drugs) if known_drugs else 0
            return found, len(known_drugs), recall

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
    logger.info("=" * 70)
    logger.info("Evaluating Enhanced GB Model (Fix 4)")
    logger.info("=" * 70)

    # Load TransE embeddings
    checkpoint = torch.load(MODELS_DIR / "transe.pt", map_location="cpu")
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
    logger.info(f"Loaded embeddings: {embeddings.shape}")

    # Get all drug IDs
    all_drug_ids = [eid for eid in entity2id.keys() if "Compound" in eid]
    logger.info(f"Total drugs: {len(all_drug_ids)}")

    # Load ground truth
    ground_truth = load_ground_truth()
    logger.info(f"Ground truth diseases: {len(ground_truth)}")

    # Load enhanced model
    with open(MODELS_DIR / "drug_repurposing_gb_enhanced.pkl", "rb") as f:
        enhanced_model = pickle.load(f)
    logger.info("Loaded enhanced GB model")

    # Evaluate
    logger.info("\nEvaluating enhanced model...")
    results = evaluate_model(enhanced_model, embeddings, entity2id, ground_truth, all_drug_ids)

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS: Enhanced GB Model (Fix 4)")
    print("=" * 80)

    total_found_30 = 0
    total_known = 0

    print(f"\n{'Disease':<25} {'R@30':>8} {'R@50':>8} {'R@100':>8} {'Found/Known':>12}")
    print("-" * 70)

    for disease in sorted(results.keys()):
        r = results[disease]
        print(f"{disease:<25} {r['recall@30']*100:>7.1f}% {r['recall@50']*100:>7.1f}% {r['recall@100']*100:>7.1f}% {r['found@30']:>5}/{r['total_known']:<5}")
        total_found_30 += r["found@30"]
        total_known += r["total_known"]

    print("-" * 70)
    agg_recall = total_found_30 / total_known if total_known > 0 else 0
    print(f"{'AGGREGATE':<25} {agg_recall*100:>7.1f}%")

    print(f"\nTotal: {total_found_30}/{total_known} drugs found in top 30")

    # Save results
    output = {
        "aggregate_recall@30": agg_recall,
        "total_found@30": total_found_30,
        "total_known": total_known,
        "by_disease": results,
    }

    output_path = MODELS_DIR / "gb_enhanced_evaluation.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
