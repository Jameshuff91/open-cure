#!/usr/bin/env python3
"""
Query drug repurposing predictions.

Examples:
    python scripts/query.py "parkinson disease"
    python scripts/query.py "heart failure" --top 20
    python scripts/query.py --drug "metformin"
    python scripts/query.py --drug "dantrolene" --top 10
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from confidence_filter import filter_prediction, ConfidenceLevel

DATA_DIR = PROJECT_ROOT / "data"
REFERENCE_DIR = DATA_DIR / "reference"
MODELS_DIR = PROJECT_ROOT / "models"

# Cache loaded resources
_RESOURCES: Optional[Dict] = None


def load_resources() -> Dict:
    """Load model and data (cached)."""
    global _RESOURCES
    if _RESOURCES is not None:
        return _RESOURCES

    print("Loading model and data...")

    # TransE embeddings
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

    # GB model
    with open(MODELS_DIR / "drug_repurposing_gb_enhanced.pkl", "rb") as f:
        model = pickle.load(f)

    # MESH mappings
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
    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)

    drugbank_id_to_name = id_to_name
    drugbank_name_to_id = {name.lower(): db_id for db_id, name in id_to_name.items()}

    # All drug/disease IDs
    all_drug_ids = [eid for eid in entity2id.keys() if "Compound" in eid]
    all_disease_ids = [eid for eid in entity2id.keys() if "Disease" in eid]

    # Pre-compute drug embeddings
    valid_drug_ids = []
    valid_drug_indices = []
    for drug_id in all_drug_ids:
        drug_idx = entity2id.get(drug_id)
        if drug_idx is not None:
            valid_drug_ids.append(drug_id)
            valid_drug_indices.append(drug_idx)

    drug_embs = embeddings[valid_drug_indices]

    _RESOURCES = {
        'embeddings': embeddings,
        'entity2id': entity2id,
        'model': model,
        'mesh_mappings': mesh_mappings,
        'drugbank_id_to_name': drugbank_id_to_name,
        'drugbank_name_to_id': drugbank_name_to_id,
        'all_drug_ids': all_drug_ids,
        'all_disease_ids': all_disease_ids,
        'valid_drug_ids': valid_drug_ids,
        'drug_embs': drug_embs,
    }

    print(f"Loaded {len(valid_drug_ids)} drugs, {len(mesh_mappings)} diseases\n")
    return _RESOURCES


def get_drug_name(drug_id: str, resources: Dict) -> str:
    """Convert drug ID to name."""
    if "::" in drug_id:
        db_id = drug_id.split("::")[-1]
        return resources['drugbank_id_to_name'].get(db_id, db_id)
    return drug_id


def query_disease(disease_name: str, top_k: int = 30, show_all: bool = False) -> List[Dict]:
    """Get top drug predictions for a disease."""
    resources = load_resources()

    # Find disease
    disease_lower = disease_name.lower()
    mesh_id = resources['mesh_mappings'].get(disease_lower)

    if not mesh_id:
        # Try partial match
        matches = [d for d in resources['mesh_mappings'].keys() if disease_lower in d]
        if matches:
            print(f"Disease '{disease_name}' not found. Did you mean:")
            for m in matches[:10]:
                print(f"  - {m}")
            return []
        else:
            print(f"Disease '{disease_name}' not found in database.")
            print("Try a different name or check data/reference/mesh_mappings_from_agents.json")
            return []

    disease_idx = resources['entity2id'].get(mesh_id)
    if disease_idx is None:
        print(f"Disease '{disease_name}' has no embedding. Cannot generate predictions.")
        return []

    disease_emb = resources['embeddings'][disease_idx]

    # Score all drugs (vectorized)
    drug_embs = resources['drug_embs']
    n_drugs = len(drug_embs)
    disease_emb_tiled = np.tile(disease_emb, (n_drugs, 1))

    concat_feats = np.hstack([drug_embs, disease_emb_tiled])
    product_feats = drug_embs * disease_emb_tiled
    diff_feats = drug_embs - disease_emb_tiled
    all_features = np.hstack([concat_feats, product_feats, diff_feats])

    scores = resources['model'].predict_proba(all_features)[:, 1]
    ranked_indices = np.argsort(scores)[::-1]

    # Collect results
    results = []
    for rank, idx in enumerate(ranked_indices, 1):
        drug_id = resources['valid_drug_ids'][idx]
        drug_name = get_drug_name(drug_id, resources)
        score = scores[idx]

        # Apply confidence filter
        filtered = filter_prediction(drug_name, disease_name, score)

        if not show_all and filtered.confidence == ConfidenceLevel.EXCLUDED:
            continue

        results.append({
            'rank': len(results) + 1,
            'drug': drug_name,
            'score': float(score),
            'confidence': filtered.confidence.value,
            'drug_type': filtered.drug_type or 'unknown',
        })

        if len(results) >= top_k:
            break

    return results


def query_drug(drug_name: str, top_k: int = 30) -> List[Dict]:
    """Get top disease predictions for a drug."""
    resources = load_resources()

    # Find drug
    drug_lower = drug_name.lower()
    db_id = resources['drugbank_name_to_id'].get(drug_lower)

    if not db_id:
        # Try partial match
        matches = [d for d in resources['drugbank_name_to_id'].keys() if drug_lower in d]
        if matches:
            print(f"Drug '{drug_name}' not found. Did you mean:")
            for m in matches[:10]:
                print(f"  - {m}")
            return []
        else:
            print(f"Drug '{drug_name}' not found in database.")
            return []

    drug_drkg_id = f"drkg:Compound::{db_id}"
    drug_idx = resources['entity2id'].get(drug_drkg_id)

    if drug_idx is None:
        print(f"Drug '{drug_name}' has no embedding. Cannot generate predictions.")
        return []

    drug_emb = resources['embeddings'][drug_idx]

    # Score all diseases
    results = []
    for disease_name, mesh_id in resources['mesh_mappings'].items():
        disease_idx = resources['entity2id'].get(mesh_id)
        if disease_idx is None:
            continue

        disease_emb = resources['embeddings'][disease_idx]

        # Create features
        concat = np.concatenate([drug_emb, disease_emb])
        product = drug_emb * disease_emb
        diff = drug_emb - disease_emb
        features = np.concatenate([concat, product, diff]).reshape(1, -1)

        score = resources['model'].predict_proba(features)[0, 1]

        results.append({
            'disease': disease_name,
            'score': float(score),
        })

    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)

    # Add ranks
    for i, r in enumerate(results[:top_k], 1):
        r['rank'] = i

    return results[:top_k]


def print_disease_results(disease: str, results: List[Dict]):
    """Pretty print disease query results."""
    print(f"Top predictions for: {disease}")
    print("=" * 60)
    print(f"{'Rank':<6} {'Drug':<30} {'Score':<8} {'Confidence'}")
    print("-" * 60)

    for r in results:
        conf_marker = ""
        if r['confidence'] == 'high':
            conf_marker = " *"
        elif r['confidence'] == 'low':
            conf_marker = " ?"

        print(f"{r['rank']:<6} {r['drug']:<30} {r['score']:.3f}    {r['confidence']}{conf_marker}")

    print("-" * 60)
    print("* = high confidence, ? = low confidence (use with caution)")


def print_drug_results(drug: str, results: List[Dict]):
    """Pretty print drug query results."""
    print(f"Top disease predictions for: {drug}")
    print("=" * 60)
    print(f"{'Rank':<6} {'Disease':<40} {'Score':<8}")
    print("-" * 60)

    for r in results:
        disease_display = r['disease'][:38] + '..' if len(r['disease']) > 40 else r['disease']
        print(f"{r['rank']:<6} {disease_display:<40} {r['score']:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Query drug repurposing predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/query.py "parkinson disease"
  python scripts/query.py "heart failure" --top 20
  python scripts/query.py --drug "metformin"
  python scripts/query.py --drug "dantrolene" --top 10
  python scripts/query.py "alzheimer" --json
        """
    )

    parser.add_argument("disease", nargs="?", help="Disease name to query")
    parser.add_argument("--drug", "-d", help="Query by drug instead of disease")
    parser.add_argument("--top", "-n", type=int, default=30, help="Number of results (default: 30)")
    parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    parser.add_argument("--all", "-a", action="store_true", help="Show all results (including excluded)")

    args = parser.parse_args()

    if not args.disease and not args.drug:
        parser.print_help()
        sys.exit(1)

    if args.drug:
        results = query_drug(args.drug, args.top)
        if results:
            if args.json:
                print(json.dumps(results, indent=2))
            else:
                print_drug_results(args.drug, results)
    else:
        results = query_disease(args.disease, args.top, args.all)
        if results:
            if args.json:
                print(json.dumps(results, indent=2))
            else:
                print_disease_results(args.disease, results)


if __name__ == "__main__":
    main()
