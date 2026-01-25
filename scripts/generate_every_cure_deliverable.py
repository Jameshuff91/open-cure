#!/usr/bin/env python3
"""
Generate Actionable Predictions for Every Cure.

This script produces a curated list of novel drug repurposing candidates
that are NOT already in the Every Cure ground truth.

Output: A deliverable JSON/CSV with high-confidence predictions for review.
"""

import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
import torch
from tqdm import tqdm

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from confidence_filter import filter_prediction, ConfidenceLevel

DATA_DIR = PROJECT_ROOT / "data"
REFERENCE_DIR = DATA_DIR / "reference"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = DATA_DIR / "deliverables"


@dataclass
class NovelPrediction:
    """A novel drug repurposing prediction."""
    drug: str
    disease: str
    score: float
    rank: int
    confidence: str
    drug_type: str
    reason: str


def load_resources():
    """Load all required resources."""
    print("Loading resources...")

    # TransE embeddings
    print("  Loading TransE embeddings...")
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

    # GB model
    print("  Loading GB model...")
    with open(MODELS_DIR / "drug_repurposing_gb_enhanced.pkl", "rb") as f:
        model = pickle.load(f)

    # MESH mappings
    print("  Loading MESH mappings...")
    with open(REFERENCE_DIR / "mesh_mappings_from_agents.json") as f:
        mesh_data = json.load(f)

    mesh_mappings = {}
    for batch_data in mesh_data.values():
        if isinstance(batch_data, dict):
            for disease_name, mesh_id in batch_data.items():
                if mesh_id:
                    mesh_str = str(mesh_id)
                    if mesh_str.startswith("D"):
                        mesh_mappings[disease_name.lower()] = f"drkg:Disease::MESH:{mesh_str}"

    # DrugBank lookup
    print("  Loading DrugBank lookup...")
    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)

    drugbank_id_to_name = id_to_name
    drugbank_name_to_id = {name.lower(): f"drkg:Compound::{db_id}" for db_id, name in id_to_name.items()}

    # Ground truth (to exclude known treatments)
    print("  Loading ground truth...")
    with open(REFERENCE_DIR / "everycure_gt_for_txgnn.json") as f:
        gt = json.load(f)

    # Build known treatments lookup
    known_treatments: Dict[str, Set[str]] = {}
    for disease_name, disease_data in gt.items():
        drugs = {d['name'].lower() for d in disease_data['drugs']}
        known_treatments[disease_name.lower()] = drugs

    # Get all drug IDs
    all_drug_ids = [eid for eid in entity2id.keys() if "Compound" in eid]

    print(f"  Loaded {len(embeddings)} embeddings, {len(mesh_mappings)} MESH mappings, {len(drugbank_name_to_id)} drugs")

    return {
        'embeddings': embeddings,
        'entity2id': entity2id,
        'model': model,
        'mesh_mappings': mesh_mappings,
        'drugbank_id_to_name': drugbank_id_to_name,
        'drugbank_name_to_id': drugbank_name_to_id,
        'known_treatments': known_treatments,
        'all_drug_ids': all_drug_ids,
        'ground_truth': gt,
    }


def get_drug_name_from_id(drug_id: str, drugbank_id_to_name: Dict[str, str]) -> str:
    """Convert drkg:Compound::DB00001 to drug name."""
    # Extract DrugBank ID from drkg format
    if "::" in drug_id:
        db_id = drug_id.split("::")[-1]
        return drugbank_id_to_name.get(db_id, db_id)
    return drug_id


def generate_predictions_for_disease(
    disease_name: str,
    resources: Dict,
    top_k: int = 50,
) -> List[NovelPrediction]:
    """Generate novel predictions for a single disease."""

    embeddings = resources['embeddings']
    entity2id = resources['entity2id']
    model = resources['model']
    mesh_mappings = resources['mesh_mappings']
    drugbank_id_to_name = resources['drugbank_id_to_name']
    known_treatments = resources['known_treatments']
    all_drug_ids = resources['all_drug_ids']

    # Get disease embedding
    mesh_id = mesh_mappings.get(disease_name.lower())
    if not mesh_id:
        return []

    disease_idx = entity2id.get(mesh_id)
    if disease_idx is None:
        return []

    disease_emb = embeddings[disease_idx]

    # Get known treatments for this disease
    known = known_treatments.get(disease_name.lower(), set())

    # Pre-compute valid drug embeddings
    valid_drug_ids = []
    valid_drug_indices = []
    for drug_id in all_drug_ids:
        drug_idx = entity2id.get(drug_id)
        if drug_idx is not None:
            valid_drug_ids.append(drug_id)
            valid_drug_indices.append(drug_idx)

    drug_embs = embeddings[valid_drug_indices]

    # Create features for all drugs (vectorized)
    n_drugs = len(drug_embs)
    disease_emb_tiled = np.tile(disease_emb, (n_drugs, 1))
    concat_feats = np.hstack([drug_embs, disease_emb_tiled])
    product_feats = drug_embs * disease_emb_tiled
    diff_feats = drug_embs - disease_emb_tiled
    all_features = np.hstack([concat_feats, product_feats, diff_feats])

    # Score all drugs
    scores = model.predict_proba(all_features)[:, 1]

    # Rank drugs
    ranked_indices = np.argsort(scores)[::-1]

    # Collect novel predictions
    novel_predictions = []
    for rank, idx in enumerate(ranked_indices):
        drug_id = valid_drug_ids[idx]
        drug_name = get_drug_name_from_id(drug_id, drugbank_id_to_name)
        score = scores[idx]

        # Skip if known treatment
        if drug_name.lower() in known:
            continue

        # Apply confidence filter
        filtered = filter_prediction(drug_name, disease_name, score)

        # Skip excluded predictions
        if filtered.confidence == ConfidenceLevel.EXCLUDED:
            continue

        novel_predictions.append(NovelPrediction(
            drug=drug_name,
            disease=disease_name,
            score=float(score),
            rank=len(novel_predictions) + 1,
            confidence=filtered.confidence.value,
            drug_type=filtered.drug_type or "unknown",
            reason=filtered.reason,
        ))

        if len(novel_predictions) >= top_k:
            break

    return novel_predictions


def generate_deliverable(
    resources: Dict,
    diseases: List[str] = None,
    top_k_per_disease: int = 30,
    min_score: float = 0.7,
) -> Dict:
    """Generate the full deliverable."""

    if diseases is None:
        # Use all diseases from ground truth that we can map
        diseases = list(resources['ground_truth'].keys())

    all_predictions = []
    diseases_processed = 0
    diseases_skipped = 0

    for disease_name in tqdm(diseases, desc="Generating predictions"):
        predictions = generate_predictions_for_disease(
            disease_name, resources, top_k=top_k_per_disease
        )

        if predictions:
            diseases_processed += 1
            # Filter by minimum score
            filtered = [p for p in predictions if p.score >= min_score]
            all_predictions.extend(filtered)
        else:
            diseases_skipped += 1

    # Sort by score descending
    all_predictions.sort(key=lambda x: x.score, reverse=True)

    # Compute statistics
    confidence_counts = {"high": 0, "medium": 0, "low": 0}
    drug_type_counts: Dict[str, int] = {}

    for pred in all_predictions:
        confidence_counts[pred.confidence] = confidence_counts.get(pred.confidence, 0) + 1
        drug_type_counts[pred.drug_type] = drug_type_counts.get(pred.drug_type, 0) + 1

    return {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "model": "drug_repurposing_gb_enhanced.pkl",
            "total_predictions": len(all_predictions),
            "diseases_processed": diseases_processed,
            "diseases_skipped": diseases_skipped,
            "min_score_threshold": min_score,
            "top_k_per_disease": top_k_per_disease,
        },
        "statistics": {
            "confidence_distribution": confidence_counts,
            "drug_type_distribution": drug_type_counts,
        },
        "predictions": [asdict(p) for p in all_predictions],
    }


def save_deliverable(deliverable: Dict, output_dir: Path):
    """Save deliverable in multiple formats."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d")

    # Save as JSON
    json_path = output_dir / f"every_cure_predictions_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(deliverable, f, indent=2)
    print(f"Saved JSON: {json_path}")

    # Save as CSV (predictions only)
    csv_path = output_dir / f"every_cure_predictions_{timestamp}.csv"
    with open(csv_path, "w") as f:
        f.write("rank,drug,disease,score,confidence,drug_type,reason\n")
        for i, pred in enumerate(deliverable["predictions"], 1):
            reason = pred['reason'].replace('"', '""')
            f.write(f'{i},"{pred["drug"]}","{pred["disease"]}",{pred["score"]:.4f},{pred["confidence"]},{pred["drug_type"]},"{reason}"\n')
    print(f"Saved CSV: {csv_path}")

    # Save summary report
    report_path = output_dir / f"every_cure_summary_{timestamp}.txt"
    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("EVERY CURE - NOVEL DRUG REPURPOSING PREDICTIONS\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Generated: {deliverable['metadata']['generated_at']}\n")
        f.write(f"Model: {deliverable['metadata']['model']}\n")
        f.write(f"Total predictions: {deliverable['metadata']['total_predictions']}\n")
        f.write(f"Diseases processed: {deliverable['metadata']['diseases_processed']}\n")
        f.write(f"Minimum score threshold: {deliverable['metadata']['min_score_threshold']}\n\n")

        f.write("CONFIDENCE DISTRIBUTION\n")
        f.write("-" * 40 + "\n")
        for conf, count in deliverable['statistics']['confidence_distribution'].items():
            f.write(f"  {conf}: {count}\n")
        f.write("\n")

        f.write("DRUG TYPE DISTRIBUTION\n")
        f.write("-" * 40 + "\n")
        for dtype, count in sorted(deliverable['statistics']['drug_type_distribution'].items(), key=lambda x: -x[1]):
            f.write(f"  {dtype}: {count}\n")
        f.write("\n")

        f.write("TOP 50 PREDICTIONS\n")
        f.write("-" * 70 + "\n")
        for pred in deliverable['predictions'][:50]:
            f.write(f"\n{pred['rank']}. {pred['drug']} → {pred['disease']}\n")
            f.write(f"   Score: {pred['score']:.3f} | Confidence: {pred['confidence']} | Type: {pred['drug_type']}\n")

    print(f"Saved report: {report_path}")


def main():
    print("=" * 70)
    print("GENERATING EVERY CURE DELIVERABLE")
    print("=" * 70)

    # Load resources
    resources = load_resources()

    # Generate deliverable
    print("\nGenerating predictions...")
    deliverable = generate_deliverable(
        resources,
        diseases=None,  # All diseases
        top_k_per_disease=30,
        min_score=0.8,  # High confidence threshold
    )

    # Save
    print("\nSaving deliverable...")
    save_deliverable(deliverable, OUTPUT_DIR)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total novel predictions: {deliverable['metadata']['total_predictions']}")
    print(f"Diseases with predictions: {deliverable['metadata']['diseases_processed']}")
    print(f"\nConfidence distribution:")
    for conf, count in deliverable['statistics']['confidence_distribution'].items():
        print(f"  {conf}: {count}")

    print("\nTop 10 predictions:")
    for pred in deliverable['predictions'][:10]:
        print(f"  {pred['score']:.3f} | {pred['drug']} → {pred['disease']} ({pred['confidence']})")


if __name__ == "__main__":
    main()
