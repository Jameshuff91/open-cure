#!/usr/bin/env python3
"""
Generate CLEAN Actionable Predictions for Every Cure.

Filters to:
- Only FDA-approved drugs with proper names (no MESH/CHEMBL IDs)
- Novel predictions (not already known treatments)
- Excludes known harmful patterns
- High confidence threshold
"""

import json
import pickle
import sys
import re
from pathlib import Path
from typing import Dict, List, Set
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from confidence_filter import filter_prediction, ConfidenceLevel

DATA_DIR = PROJECT_ROOT / "data"
REFERENCE_DIR = DATA_DIR / "reference"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = DATA_DIR / "deliverables"


# Patterns for unmapped/experimental drugs to exclude
EXCLUDED_DRUG_PATTERNS = [
    r"^MESH:",
    r"^CHEMBL\d+",
    r"^zinc:",
    r"^CHEBI:",
    r"^DB\d{5}$",  # Raw DrugBank IDs
    r"^\d+$",  # Pure numbers
    r"^[A-Z]{2,3}\d{6}",  # Chemical registry IDs
    r"PHOSPHON",  # Chemical compound names
    r"DIHYDROXY.*PYRAN",  # Chemical names
    r"^N'-\[",  # Complex chemical names
    r"Cystein-S-Yl",  # Chemical intermediates
    r"tetraphosphate",  # Nucleotides
    r"LACTATE$",  # Metabolites
    r"^3-\(",  # Chemical names starting with position
    r"^5-Bromo",  # Chemical modifications
    r"^4,5-",  # Position numbers
]

# Known FDA-approved drugs we want to prioritize
# These are drugs we know have real clinical data
FDA_APPROVED_DRUGS = {
    # Cardiovascular
    "empagliflozin", "dapagliflozin", "canagliflozin",  # SGLT2i
    "losartan", "valsartan", "irbesartan",  # ARBs
    "lisinopril", "enalapril", "ramipril", "benazepril",  # ACEi
    "metoprolol", "carvedilol", "bisoprolol",  # Beta blockers
    "spironolactone", "eplerenone",  # MRAs
    "dantrolene",  # Validated for HF!

    # Diabetes
    "metformin", "glipizide", "glyburide",
    "sitagliptin", "linagliptin", "saxagliptin",  # DPP4i
    "liraglutide", "semaglutide", "dulaglutide",  # GLP1

    # Neurological
    "donepezil", "rivastigmine", "galantamine", "memantine",  # Alzheimer's
    "levodopa", "carbidopa", "pramipexole", "ropinirole",  # Parkinson's
    "gabapentin", "pregabalin", "lamotrigine", "levetiracetam",  # Seizure

    # Psychiatric
    "quetiapine", "olanzapine", "risperidone", "aripiprazole",
    "sertraline", "fluoxetine", "escitalopram", "venlafaxine",
    "lithium", "valproate",

    # Autoimmune/Anti-inflammatory
    "methotrexate", "hydroxychloroquine", "sulfasalazine",
    "prednisone", "prednisolone", "dexamethasone",
    "naproxen", "ibuprofen", "celecoxib",
    "colchicine",

    # Respiratory
    "formoterol", "salmeterol", "tiotropium",
    "budesonide", "fluticasone", "mometasone",
    "montelukast",

    # Oncology
    "paclitaxel", "docetaxel", "carboplatin", "cisplatin",
    "tamoxifen", "letrozole", "anastrozole",
    "imatinib", "nilotinib", "dasatinib",

    # Other
    "thiamine", "riboflavin", "pyridoxine",  # Vitamins
    "zinc", "selenium", "chromium",  # Minerals
    "omega-3", "fish oil", "dha",  # Supplements
}


@dataclass
class CleanPrediction:
    """A clean, actionable prediction."""
    rank: int
    drug: str
    disease: str
    score: float
    confidence: str
    drug_type: str
    is_fda_approved: bool
    reason: str


def is_valid_drug_name(name: str) -> bool:
    """Check if drug name is a proper name (not a chemical ID)."""
    for pattern in EXCLUDED_DRUG_PATTERNS:
        if re.search(pattern, name, re.IGNORECASE):
            return False
    # Must have at least one letter
    if not re.search(r"[a-zA-Z]", name):
        return False
    # Must not be too long (chemical names are verbose)
    if len(name) > 50:
        return False
    return True


def is_fda_approved(name: str) -> bool:
    """Check if drug is likely FDA approved."""
    return name.lower() in FDA_APPROVED_DRUGS


def load_resources():
    """Load all required resources."""
    print("Loading resources...")

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

    with open(MODELS_DIR / "drug_repurposing_gb_enhanced.pkl", "rb") as f:
        model = pickle.load(f)

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

    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)

    drugbank_id_to_name = id_to_name
    drugbank_name_to_id = {name.lower(): f"drkg:Compound::{db_id}" for db_id, name in id_to_name.items()}

    with open(REFERENCE_DIR / "everycure_gt_for_txgnn.json") as f:
        gt = json.load(f)

    known_treatments: Dict[str, Set[str]] = {}
    for disease_name, disease_data in gt.items():
        drugs = {d['name'].lower() for d in disease_data['drugs']}
        known_treatments[disease_name.lower()] = drugs

    all_drug_ids = [eid for eid in entity2id.keys() if "Compound" in eid]

    print(f"  Loaded {len(embeddings)} embeddings, {len(mesh_mappings)} MESH mappings")

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
    if "::" in drug_id:
        db_id = drug_id.split("::")[-1]
        return drugbank_id_to_name.get(db_id, db_id)
    return drug_id


def generate_clean_predictions(
    disease_name: str,
    resources: Dict,
    top_k: int = 20,
) -> List[CleanPrediction]:
    """Generate clean predictions for a single disease."""

    embeddings = resources['embeddings']
    entity2id = resources['entity2id']
    model = resources['model']
    mesh_mappings = resources['mesh_mappings']
    drugbank_id_to_name = resources['drugbank_id_to_name']
    known_treatments = resources['known_treatments']
    all_drug_ids = resources['all_drug_ids']

    mesh_id = mesh_mappings.get(disease_name.lower())
    if not mesh_id:
        return []

    disease_idx = entity2id.get(mesh_id)
    if disease_idx is None:
        return []

    disease_emb = embeddings[disease_idx]
    known = known_treatments.get(disease_name.lower(), set())

    # Pre-compute drug embeddings
    valid_drug_ids = []
    valid_drug_indices = []
    for drug_id in all_drug_ids:
        drug_idx = entity2id.get(drug_id)
        if drug_idx is not None:
            valid_drug_ids.append(drug_id)
            valid_drug_indices.append(drug_idx)

    drug_embs = embeddings[valid_drug_indices]

    # Vectorized scoring
    n_drugs = len(drug_embs)
    disease_emb_tiled = np.tile(disease_emb, (n_drugs, 1))
    concat_feats = np.hstack([drug_embs, disease_emb_tiled])
    product_feats = drug_embs * disease_emb_tiled
    diff_feats = drug_embs - disease_emb_tiled
    all_features = np.hstack([concat_feats, product_feats, diff_feats])

    scores = model.predict_proba(all_features)[:, 1]
    ranked_indices = np.argsort(scores)[::-1]

    predictions = []
    rank = 0
    for idx in ranked_indices:
        drug_id = valid_drug_ids[idx]
        drug_name = get_drug_name_from_id(drug_id, drugbank_id_to_name)
        score = scores[idx]

        # Skip known treatments
        if drug_name.lower() in known:
            continue

        # Skip invalid drug names
        if not is_valid_drug_name(drug_name):
            continue

        # Apply confidence filter
        filtered = filter_prediction(drug_name, disease_name, score)
        if filtered.confidence == ConfidenceLevel.EXCLUDED:
            continue

        rank += 1
        predictions.append(CleanPrediction(
            rank=rank,
            drug=drug_name,
            disease=disease_name,
            score=float(score),
            confidence=filtered.confidence.value,
            drug_type=filtered.drug_type or "unknown",
            is_fda_approved=is_fda_approved(drug_name),
            reason=filtered.reason,
        ))

        if rank >= top_k:
            break

    return predictions


def main():
    print("=" * 70)
    print("GENERATING CLEAN EVERY CURE DELIVERABLE")
    print("=" * 70)

    resources = load_resources()

    # Priority diseases for Every Cure
    priority_diseases = [
        # Major disease areas
        "heart failure", "parkinson disease", "alzheimer disease",
        "type 2 diabetes mellitus", "type 1 diabetes mellitus",
        "rheumatoid arthritis", "multiple sclerosis", "asthma",
        "breast cancer", "lung cancer", "colorectal cancer",
        "depression", "schizophrenia", "epilepsy",
        "chronic obstructive pulmonary disease", "hypertension",

        # Rare diseases with unmet need
        "amyotrophic lateral sclerosis", "huntington disease",
        "duchenne muscular dystrophy", "cystic fibrosis",
        "sickle cell disease", "thalassemia",
        "gaucher disease", "fabry disease",
        "phenylketonuria", "maple syrup urine disease",
    ]

    # Also include all diseases from ground truth
    all_diseases = list(set(priority_diseases + list(resources['ground_truth'].keys())))

    all_predictions = []
    fda_approved_predictions = []

    for disease_name in tqdm(all_diseases, desc="Generating predictions"):
        predictions = generate_clean_predictions(disease_name, resources, top_k=20)

        for pred in predictions:
            if pred.score >= 0.85:  # Higher threshold for clean deliverable
                all_predictions.append(pred)
                if pred.is_fda_approved:
                    fda_approved_predictions.append(pred)

    # Sort by score
    all_predictions.sort(key=lambda x: x.score, reverse=True)
    fda_approved_predictions.sort(key=lambda x: x.score, reverse=True)

    # De-duplicate (same drug-disease pair might appear if disease is in multiple names)
    seen = set()
    unique_predictions = []
    for pred in all_predictions:
        key = (pred.drug.lower(), pred.disease.lower())
        if key not in seen:
            seen.add(key)
            unique_predictions.append(pred)

    seen_fda = set()
    unique_fda = []
    for pred in fda_approved_predictions:
        key = (pred.drug.lower(), pred.disease.lower())
        if key not in seen_fda:
            seen_fda.add(key)
            unique_fda.append(pred)

    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d")

    # All predictions
    with open(OUTPUT_DIR / f"clean_predictions_{timestamp}.json", "w") as f:
        json.dump({
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_predictions": len(unique_predictions),
                "fda_approved_predictions": len(unique_fda),
                "min_score_threshold": 0.85,
            },
            "predictions": [asdict(p) for p in unique_predictions[:500]],  # Top 500
        }, f, indent=2)

    # FDA-approved only (highest quality)
    with open(OUTPUT_DIR / f"fda_approved_predictions_{timestamp}.json", "w") as f:
        json.dump({
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "description": "Novel predictions using FDA-approved drugs only",
                "total_predictions": len(unique_fda),
            },
            "predictions": [asdict(p) for p in unique_fda],
        }, f, indent=2)

    # Human-readable report
    with open(OUTPUT_DIR / f"clean_summary_{timestamp}.txt", "w") as f:
        f.write("=" * 70 + "\n")
        f.write("EVERY CURE - CLEAN DRUG REPURPOSING PREDICTIONS\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Total clean predictions: {len(unique_predictions)}\n")
        f.write(f"FDA-approved drug predictions: {len(unique_fda)}\n\n")

        f.write("=" * 70 + "\n")
        f.write("TOP 50 FDA-APPROVED DRUG PREDICTIONS\n")
        f.write("(These are the highest-quality, most actionable predictions)\n")
        f.write("=" * 70 + "\n\n")

        for i, pred in enumerate(unique_fda[:50], 1):
            f.write(f"{i}. {pred.drug} → {pred.disease}\n")
            f.write(f"   Score: {pred.score:.3f} | Confidence: {pred.confidence}\n\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("TOP 50 ALL PREDICTIONS (including non-FDA)\n")
        f.write("=" * 70 + "\n\n")

        for i, pred in enumerate(unique_predictions[:50], 1):
            fda_tag = " [FDA]" if pred.is_fda_approved else ""
            f.write(f"{i}. {pred.drug}{fda_tag} → {pred.disease}\n")
            f.write(f"   Score: {pred.score:.3f} | Confidence: {pred.confidence}\n\n")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total clean predictions: {len(unique_predictions)}")
    print(f"FDA-approved predictions: {len(unique_fda)}")
    print(f"\nFiles saved to: {OUTPUT_DIR}")

    print(f"\nTop 10 FDA-approved predictions:")
    for pred in unique_fda[:10]:
        print(f"  {pred.score:.3f} | {pred.drug} → {pred.disease}")


if __name__ == "__main__":
    main()
