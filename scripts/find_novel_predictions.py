#!/usr/bin/env python3
"""
Find novel drug repurposing predictions NOT in ground truth.

This script identifies high-scoring predictions that represent potential
new repurposing opportunities - drugs predicted to work for diseases where
they are not yet approved or known to be effective.

Output:
- Top novel predictions ranked by score + biological plausibility
- Filtered to avoid known false positive patterns
- Cross-referenced with target overlap data
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
OUTPUT_DIR = PROJECT_ROOT / "data" / "analysis"

# Target boost parameters
BOOST_MULTIPLIER = 0.01
MAX_OVERLAP = 10

# False positive patterns to exclude
FALSE_POSITIVE_PATTERNS = {
    "antibiotics_metabolic": {
        "drugs": ["gentamicin", "tobramycin", "amikacin", "streptomycin"],
        "diseases": ["diabetes", "metabolic", "obesity"],
    },
    "sympathomimetics_diabetes": {
        "drugs": ["pseudoephedrine", "phenylephrine", "ephedrine"],
        "diseases": ["diabetes", "glucose", "insulin"],
    },
    "alpha_blockers_heart_failure": {
        "drugs": ["doxazosin", "prazosin", "terazosin", "tamsulosin"],
        "diseases": ["heart failure", "cardiac failure", "cardiomyopathy"],
    },
    "diagnostic_agents": {
        "drugs": ["ioflupane", "technetium", "gadolinium", "fluorodeoxyglucose"],
        "diseases": [],  # exclude for all diseases
    },
    "tca_hypertension": {
        "drugs": ["protriptyline", "amitriptyline", "nortriptyline", "imipramine"],
        "diseases": ["hypertension", "blood pressure"],
    },
    "ppi_hypertension": {
        "drugs": ["pantoprazole", "omeprazole", "esomeprazole", "lansoprazole"],
        "diseases": ["hypertension", "blood pressure"],
    },
}


def is_false_positive_pattern(drug_name: str, disease_name: str) -> bool:
    """Check if drug-disease pair matches a known false positive pattern."""
    drug_lower = drug_name.lower()
    disease_lower = disease_name.lower()

    for _pattern_name, pattern in FALSE_POSITIVE_PATTERNS.items():
        drug_match = any(d in drug_lower for d in pattern["drugs"])
        disease_match = (
            not pattern["diseases"] or  # Empty means exclude for all
            any(d in disease_lower for d in pattern["diseases"])
        )
        if drug_match and disease_match:
            return True
    return False


def load_all_data() -> Tuple:
    """Load model, embeddings, ground truth, and target data."""
    print("Loading model...")
    with open(MODELS_DIR / "drug_repurposing_gb_enhanced.pkl", "rb") as f:
        model = pickle.load(f)

    print("Loading embeddings...")
    checkpoint = torch.load(
        MODELS_DIR / "transe.pt",
        map_location="cpu",
        weights_only=False
    )

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

    print("Loading reference data...")
    with open(REFERENCE_DIR / "drug_targets.json") as f:
        drug_targets_raw = json.load(f)
    drug_targets = {k: set(v) for k, v in drug_targets_raw.items()}

    with open(REFERENCE_DIR / "disease_genes.json") as f:
        disease_genes_raw = json.load(f)
    disease_genes = {k: set(v) for k, v in disease_genes_raw.items()}

    with open(REFERENCE_DIR / "mesh_mappings_from_agents.json") as f:
        mesh_data = json.load(f)

    mesh_mappings: Dict[str, str] = {}
    for batch_data in mesh_data.values():
        if isinstance(batch_data, dict):
            for disease_name, mesh_id in batch_data.items():
                if mesh_id:
                    mesh_str = str(mesh_id)
                    if mesh_str.startswith("D"):
                        mesh_mappings[disease_name.lower()] = f"drkg:Disease::MESH:{mesh_str}"

    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)

    # Load ground truth
    print("Loading ground truth...")
    with open(REFERENCE_DIR / "expanded_ground_truth.json") as f:
        ground_truth = json.load(f)

    # Build ground truth lookup: disease -> set of drug names
    gt_lookup: Dict[str, Set[str]] = defaultdict(set)
    for disease_key, drug_list in ground_truth.items():
        disease_name = disease_key.lower()
        for drug_name in drug_list:
            gt_lookup[disease_name].add(drug_name.lower())

    return (model, embeddings, entity2id, drug_targets, disease_genes,
            mesh_mappings, id_to_name, gt_lookup)


def predict_all_diseases(
    model,
    embeddings: np.ndarray,
    entity2id: Dict[str, int],
    drug_targets: Dict[str, Set[str]],
    disease_genes: Dict[str, Set[str]],
    mesh_mappings: Dict[str, str],
    id_to_name: Dict[str, str],
    gt_lookup: Dict[str, Set[str]],
    top_k: int = 50,
) -> List[Dict]:
    """Generate novel predictions for all diseases."""

    # Get all drug entities once
    all_drug_ids = [eid for eid in entity2id.keys() if "Compound" in eid]
    valid_drug_ids = []
    valid_drug_indices = []
    for drug_id in all_drug_ids:
        drug_idx = entity2id.get(drug_id)
        if drug_idx is not None:
            valid_drug_ids.append(drug_id)
            valid_drug_indices.append(drug_idx)

    drug_embs = embeddings[valid_drug_indices]
    n_drugs = len(drug_embs)

    # Build drug name lookup
    drug_names = []
    for drug_id in valid_drug_ids:
        db_id = drug_id.split("::")[-1]
        name = id_to_name.get(db_id, db_id)
        drug_names.append(name)

    all_novel_predictions = []

    for disease_name, mesh_id in tqdm(mesh_mappings.items(), desc="Processing diseases"):
        disease_idx = entity2id.get(mesh_id)
        if disease_idx is None:
            continue

        disease_emb = embeddings[disease_idx]
        mesh_id_short = mesh_id.split("MESH:")[-1]
        dis_genes = disease_genes.get(f"MESH:{mesh_id_short}", set())

        # Get GT drugs for this disease
        gt_drugs = gt_lookup.get(disease_name, set())

        # Compute features
        disease_emb_tiled = np.tile(disease_emb, (n_drugs, 1))
        concat_feats = np.hstack([drug_embs, disease_emb_tiled])
        product_feats = drug_embs * disease_emb_tiled
        diff_feats = drug_embs - disease_emb_tiled
        features = np.hstack([concat_feats, product_feats, diff_feats])

        # Get base scores
        base_scores = model.predict_proba(features)[:, 1]

        # Compute target overlap
        overlaps = []
        for drug_id in valid_drug_ids:
            db_id = drug_id.split("::")[-1]
            drug_genes = drug_targets.get(db_id, set())
            overlap = len(drug_genes & dis_genes)
            overlaps.append(overlap)
        overlaps = np.array(overlaps)

        # Apply boost
        scores = base_scores * (1 + BOOST_MULTIPLIER * np.minimum(overlaps, MAX_OVERLAP))

        # Get top-k predictions
        top_indices = np.argsort(scores)[-top_k:][::-1]

        for rank, idx in enumerate(top_indices, 1):
            drug_name = drug_names[idx]

            # Skip if in ground truth
            if drug_name.lower() in gt_drugs:
                continue

            # Skip known false positive patterns
            if is_false_positive_pattern(drug_name, disease_name):
                continue

            drug_id = valid_drug_ids[idx]
            db_id = drug_id.split("::")[-1]

            all_novel_predictions.append({
                "drug_name": drug_name,
                "drugbank_id": db_id,
                "disease_name": disease_name,
                "mesh_id": mesh_id_short,
                "score": float(scores[idx]),
                "base_score": float(base_scores[idx]),
                "target_overlap": int(overlaps[idx]),
                "rank_for_disease": rank,
                "has_biological_signal": overlaps[idx] > 0,
            })

    return all_novel_predictions


def rank_and_filter_predictions(predictions: List[Dict]) -> List[Dict]:
    """Rank predictions by multiple factors."""

    # Sort by composite score: score * (1 + overlap bonus)
    for pred in predictions:
        # Bonus for biological signal (target overlap)
        bio_bonus = 1 + 0.1 * min(pred["target_overlap"], 20)
        # Bonus for being in top-10 for the disease
        rank_bonus = 1.0 if pred["rank_for_disease"] <= 10 else 0.8
        pred["composite_score"] = pred["score"] * bio_bonus * rank_bonus

    # Sort by composite score
    predictions.sort(key=lambda x: x["composite_score"], reverse=True)

    # Remove duplicates (same drug, keep highest scoring disease)
    seen_drugs = set()
    unique_predictions = []
    for pred in predictions:
        drug_key = pred["drugbank_id"]
        if drug_key not in seen_drugs:
            seen_drugs.add(drug_key)
            unique_predictions.append(pred)

    return unique_predictions


def main():
    print("=" * 70)
    print("NOVEL DRUG REPURPOSING PREDICTION FINDER")
    print("=" * 70)

    # Load everything
    (model, embeddings, entity2id, drug_targets, disease_genes,
     mesh_mappings, id_to_name, gt_lookup) = load_all_data()

    print(f"\nData loaded:")
    print(f"  - Diseases: {len(mesh_mappings)}")
    print(f"  - Ground truth diseases: {len(gt_lookup)}")
    print(f"  - Drug targets: {len(drug_targets)}")
    print(f"  - Disease genes: {len(disease_genes)}")

    # Generate predictions
    print("\nGenerating predictions for all diseases...")
    all_predictions = predict_all_diseases(
        model, embeddings, entity2id, drug_targets, disease_genes,
        mesh_mappings, id_to_name, gt_lookup, top_k=50
    )

    print(f"\nTotal predictions (before filtering): {len(all_predictions)}")

    # Rank and filter
    ranked_predictions = rank_and_filter_predictions(all_predictions)

    print(f"Unique drugs with novel predictions: {len(ranked_predictions)}")

    # Get top 100 for detailed output
    top_100 = ranked_predictions[:100]

    # Display top predictions
    print("\n" + "=" * 70)
    print("TOP 50 NOVEL PREDICTIONS (not in ground truth)")
    print("=" * 70)
    print(f"{'Rank':<5} {'Drug':<25} {'Disease':<25} {'Score':<7} {'Overlap'}")
    print("-" * 70)

    for i, pred in enumerate(top_100[:50], 1):
        drug = pred["drug_name"][:24]
        disease = pred["disease_name"][:24]
        score = pred["score"]
        overlap = pred["target_overlap"]
        overlap_str = f"{overlap} genes" if overlap > 0 else "-"
        print(f"{i:<5} {drug:<25} {disease:<25} {score:.4f} {overlap_str}")

    # Summarize by disease category
    print("\n" + "=" * 70)
    print("NOVEL PREDICTIONS BY DISEASE CATEGORY")
    print("=" * 70)

    category_counts = defaultdict(int)
    for pred in ranked_predictions[:500]:
        disease = pred["disease_name"].lower()
        if "cancer" in disease or "carcinoma" in disease or "tumor" in disease:
            cat = "Cancer"
        elif "heart" in disease or "cardiac" in disease:
            cat = "Cardiovascular"
        elif "alzheimer" in disease or "parkinson" in disease or "neurolog" in disease:
            cat = "Neurological"
        elif "diabetes" in disease or "metabolic" in disease:
            cat = "Metabolic"
        elif "arthritis" in disease or "autoimmune" in disease:
            cat = "Autoimmune"
        else:
            cat = "Other"
        category_counts[cat] += 1

    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count} predictions")

    # High-confidence predictions (top score + biological signal)
    print("\n" + "=" * 70)
    print("HIGH-CONFIDENCE NOVEL PREDICTIONS (score > 0.9 + target overlap)")
    print("=" * 70)

    high_conf = [p for p in ranked_predictions
                 if p["score"] > 0.9 and p["target_overlap"] > 0]

    print(f"Found {len(high_conf)} high-confidence predictions\n")

    for i, pred in enumerate(high_conf[:20], 1):
        print(f"{i}. {pred['drug_name']} -> {pred['disease_name']}")
        print(f"   Score: {pred['score']:.4f}, Target overlap: {pred['target_overlap']} genes")

    # Convert numpy bools to Python bools for JSON serialization
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        elif isinstance(obj, (np.bool_, np.integer, np.floating)):
            return obj.item()
        return obj

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / "novel_predictions.json"
    output_data = clean_for_json({
        "total_predictions": len(all_predictions),
        "unique_drugs": len(ranked_predictions),
        "high_confidence": len(high_conf),
        "top_100": top_100,
        "high_confidence_predictions": high_conf[:50],
        "methodology": {
            "model": "GB Enhanced + Target Boost",
            "filters": [
                "Excluded ground truth drugs",
                "Excluded known false positive patterns",
                "Ranked by score * bio_bonus * rank_bonus"
            ],
            "false_positive_patterns": list(FALSE_POSITIVE_PATTERNS.keys()),
        }
    })

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n\nResults saved to: {output_file}")

    # Summary stats
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total novel predictions: {len(all_predictions)}")
    print(f"Unique drugs: {len(ranked_predictions)}")
    print(f"High-confidence (score > 0.9 + overlap): {len(high_conf)}")
    print(f"Predictions with target overlap: {sum(1 for p in ranked_predictions if p['target_overlap'] > 0)}")


if __name__ == "__main__":
    main()
