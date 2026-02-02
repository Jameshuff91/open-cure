#!/usr/bin/env python3
"""
Export skin disease predictions for Ryland Mortlock collaboration.

Creates Excel files with:
1. All 330+ dermatological predictions from the main deliverable
2. Ichthyosis-specific predictions (generated fresh if not in deliverable)
3. Atopic dermatitis predictions (filtered from deliverable)
4. Psoriasis predictions (if available)

Output: data/exports/skin_disease_predictions.xlsx
        data/exports/ichthyosis_predictions.xlsx
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATA_DIR = PROJECT_ROOT / "data"
REFERENCE_DIR = DATA_DIR / "reference"
MODELS_DIR = PROJECT_ROOT / "models"
EXPORTS_DIR = DATA_DIR / "exports"

# Ichthyosis MESH IDs from our database
ICHTHYOSIS_DISEASES = {
    "ichthyosis vulgaris": "D016114",
    "lamellar ichthyosis": "D017490",
    "recessive x-linked ichthyosis": "D016113",
    "ichthyosis (general)": "D007057",
    "ichthyosis vulgaris (alt)": "D016112",
}


def load_knn_resources() -> Dict:
    """Load resources for kNN prediction."""
    # Node2Vec embeddings (from treatment-free training)
    n2v_path = MODELS_DIR / "node2vec_no_treatment_embeddings.pt"
    if n2v_path.exists():
        checkpoint = torch.load(n2v_path, map_location="cpu", weights_only=False)
        print(f"Loaded treatment-free embeddings from {n2v_path}")
    else:
        # Fallback to original TransE
        checkpoint = torch.load(MODELS_DIR / "transe.pt", map_location="cpu", weights_only=False)
        print("Using TransE embeddings (treatment-free not found)")

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

    # DrugBank lookup
    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        drugbank_lookup = json.load(f)

    # Ground truth for known indications
    gt_path = REFERENCE_DIR / "expanded_ground_truth.json"
    known_indications: Dict[str, set] = {}
    if gt_path.exists():
        with open(gt_path) as f:
            gt_data = json.load(f)
        for disease, drugs in gt_data.items():
            known_indications[disease] = set(drugs)

    return {
        "embeddings": embeddings,
        "entity2id": entity2id,
        "drugbank_lookup": drugbank_lookup,
        "known_indications": known_indications,
    }


def get_knn_predictions(
    disease_name: str,
    mesh_id: str,
    resources: Dict,
    k: int = 20,
    top_drugs: int = 30,
) -> List[Dict]:
    """
    Generate predictions using kNN collaborative filtering.

    Similar to the main evaluation pipeline but for specific diseases.
    """
    embeddings = resources["embeddings"]
    entity2id = resources["entity2id"]
    drugbank_lookup = resources["drugbank_lookup"]
    known_indications = resources["known_indications"]

    drkg_id = f"drkg:Disease::MESH:{mesh_id}"

    if drkg_id not in entity2id:
        print(f"  Warning: {disease_name} ({mesh_id}) not in embeddings")
        return []

    disease_idx = entity2id[drkg_id]
    disease_emb = embeddings[disease_idx]

    # Find all disease embeddings
    disease_ids = [eid for eid in entity2id.keys() if "Disease" in eid and eid != drkg_id]
    disease_indices = [entity2id[did] for did in disease_ids]
    disease_embs = embeddings[disease_indices]

    # Compute cosine similarities
    disease_emb_norm = disease_emb / (np.linalg.norm(disease_emb) + 1e-8)
    disease_embs_norm = disease_embs / (np.linalg.norm(disease_embs, axis=1, keepdims=True) + 1e-8)
    similarities = disease_embs_norm @ disease_emb_norm

    # Get k nearest neighbors
    top_k_idx = np.argsort(similarities)[-k:][::-1]
    neighbor_ids = [disease_ids[i] for i in top_k_idx]
    neighbor_sims = [similarities[i] for i in top_k_idx]

    # Collect drugs from neighbors (weighted by similarity)
    drug_scores: Dict[str, float] = {}
    for neighbor_id, sim in zip(neighbor_ids, neighbor_sims):
        neighbor_drugs = known_indications.get(neighbor_id, set())
        for drug in neighbor_drugs:
            drug_scores[drug] = drug_scores.get(drug, 0) + sim

    # Sort by score
    ranked_drugs = sorted(drug_scores.items(), key=lambda x: x[1], reverse=True)

    # Check which are known indications for this disease
    this_disease_known = known_indications.get(drkg_id, set())

    results = []
    for drug_id, score in ranked_drugs[:top_drugs]:
        # Extract DrugBank ID
        db_id = None
        if "::" in drug_id:
            parts = drug_id.split("::")
            for part in parts:
                if part.startswith("DB"):
                    db_id = part
                    break

        drug_name = drugbank_lookup.get(db_id, db_id) if db_id else drug_id
        is_known = drug_id in this_disease_known

        results.append({
            "disease_name": disease_name,
            "disease_id": drkg_id,
            "drug_name": drug_name,
            "drug_id": drug_id,
            "knn_score": float(score),
            "is_known_indication": is_known,
            "category": "dermatological",
        })

    return results


def main():
    print("=" * 70)
    print("SKIN DISEASE PREDICTIONS EXPORT")
    print("=" * 70)
    print()

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing deliverable
    deliverable_path = DATA_DIR / "deliverables" / "drug_repurposing_predictions_with_confidence.xlsx"
    print(f"Loading {deliverable_path}...")
    df = pd.read_excel(deliverable_path)
    print(f"Loaded {len(df)} predictions")
    print()

    # Filter dermatological predictions
    skin_df = df[df["category"] == "dermatological"].copy()
    print(f"Dermatological predictions: {len(skin_df)}")
    print(f"Diseases: {skin_df['disease_name'].nunique()}")
    print()

    # Export all skin predictions
    skin_output = EXPORTS_DIR / "skin_disease_predictions.xlsx"
    skin_df.to_excel(skin_output, index=False)
    print(f"Saved: {skin_output}")
    print()

    # Check for ichthyosis in existing predictions
    ichthyosis_in_df = skin_df[skin_df["disease_name"].str.contains("ichthy", case=False, na=False)]

    if len(ichthyosis_in_df) > 0:
        print(f"Found {len(ichthyosis_in_df)} ichthyosis predictions in deliverable")
        ichthyosis_df = ichthyosis_in_df
    else:
        print("Ichthyosis not in deliverable. Generating predictions...")
        print()

        # Load resources for kNN
        resources = load_knn_resources()

        # Generate predictions for each ichthyosis type
        all_predictions = []
        for disease_name, mesh_id in ICHTHYOSIS_DISEASES.items():
            print(f"  Generating predictions for {disease_name}...")
            preds = get_knn_predictions(disease_name, mesh_id, resources, k=20, top_drugs=30)
            all_predictions.extend(preds)
            print(f"    Got {len(preds)} predictions")

        if all_predictions:
            ichthyosis_df = pd.DataFrame(all_predictions)
            print()
            print(f"Generated {len(ichthyosis_df)} ichthyosis predictions")
        else:
            print("Could not generate ichthyosis predictions")
            ichthyosis_df = pd.DataFrame()

    # Export ichthyosis predictions
    if len(ichthyosis_df) > 0:
        ichthyosis_output = EXPORTS_DIR / "ichthyosis_predictions.xlsx"
        ichthyosis_df.to_excel(ichthyosis_output, index=False)
        print(f"Saved: {ichthyosis_output}")
    print()

    # Export atopic dermatitis (filtered)
    ad_df = skin_df[skin_df["disease_name"].str.contains("atopic", case=False, na=False)]
    if len(ad_df) > 0:
        ad_output = EXPORTS_DIR / "atopic_dermatitis_predictions.xlsx"
        ad_df.to_excel(ad_output, index=False)
        print(f"Saved: {ad_output} ({len(ad_df)} predictions)")

    # Summary
    print()
    print("=" * 70)
    print("EXPORT SUMMARY")
    print("=" * 70)
    print()
    print(f"  skin_disease_predictions.xlsx:       {len(skin_df)} predictions")
    print(f"  ichthyosis_predictions.xlsx:         {len(ichthyosis_df)} predictions")
    if len(ad_df) > 0:
        print(f"  atopic_dermatitis_predictions.xlsx:  {len(ad_df)} predictions")
    print()
    print(f"Export directory: {EXPORTS_DIR}")


if __name__ == "__main__":
    main()
