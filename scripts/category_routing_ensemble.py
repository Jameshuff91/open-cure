#!/usr/bin/env python3
"""
Category-Based Routing Ensemble
================================

Routes diseases to the best model (TxGNN or GB) based on their category.

Based on experimental findings:
- TxGNN excels: metabolic/storage (66.7%), psychiatric (28.6%), dermatological (25%), autoimmune (22.2%)
- GB excels: infectious (50%), cardiovascular (72%)

Strategy: Route to the model that performs best for each disease category.
"""

import ast
import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REFERENCE_DIR = DATA_DIR / "reference"
MODELS_DIR = PROJECT_ROOT / "models"
ANALYSIS_DIR = DATA_DIR / "analysis"

# Category to model routing based on experimental results
# TxGNN wins: metabolic, storage, psychiatric, dermatological, autoimmune
# GB wins: infectious, cardiovascular, respiratory, renal
TXGNN_CATEGORIES = {
    "metabolic",
    "storage",
    "psychiatric",
    "dermatological",
    "autoimmune",
    "neurological",  # TxGNN: 15.5% - mediocre but GB not clearly better
}

GB_CATEGORIES = {
    "infectious",
    "cardiovascular",
    "respiratory",
    "renal",
    "gastrointestinal",
}

# Keywords for categorizing diseases
CATEGORY_KEYWORDS = {
    "metabolic": [
        "diabetes", "hypercholesterolemia", "hyperlipidemia", "obesity",
        "gout", "thyroid", "hypothyroid", "hyperthyroid", "porphyria",
        "wilson", "hemochromatosis", "phenylketonuria"
    ],
    "storage": [
        "gaucher", "fabry", "hurler", "hunter", "niemann", "tay-sachs",
        "pompe", "mucopolysaccharidosis", "glycogen storage", "lysosomal"
    ],
    "psychiatric": [
        "bipolar", "schizophrenia", "depression", "anxiety", "ptsd",
        "adhd", "autism", "ocd", "phobia", "bulimia", "anorexia"
    ],
    "dermatological": [
        "psoriasis", "eczema", "dermatitis", "acne", "rosacea",
        "vitiligo", "alopecia", "pemphigus", "ichthyosis", "skin"
    ],
    "autoimmune": [
        "lupus", "rheumatoid", "arthritis", "multiple sclerosis", "ms",
        "crohn", "ulcerative colitis", "sjogren", "scleroderma",
        "myasthenia", "vasculitis", "hashimoto", "graves"
    ],
    "infectious": [
        "hiv", "hepatitis", "tuberculosis", "malaria", "covid",
        "influenza", "pneumonia", "sepsis", "meningitis", "infection"
    ],
    "cardiovascular": [
        "heart failure", "hypertension", "atrial fibrillation", "arrhythmia",
        "coronary", "angina", "cardiomyopathy", "atherosclerosis"
    ],
    "respiratory": [
        "asthma", "copd", "pulmonary", "lung", "bronchitis",
        "cystic fibrosis", "pulmonary fibrosis", "emphysema"
    ],
    "neurological": [
        "alzheimer", "parkinson", "epilepsy", "seizure", "migraine",
        "huntington", "als", "neuropathy", "dementia", "stroke"
    ],
    "renal": [
        "kidney", "renal", "nephritis", "nephropathy", "glomerulo"
    ],
    "gastrointestinal": [
        "gastric", "intestinal", "bowel", "colon", "liver", "hepatic",
        "pancreatitis", "cholestasis"
    ],
    "cancer": [
        "cancer", "carcinoma", "leukemia", "lymphoma", "melanoma",
        "sarcoma", "tumor", "neoplasm", "oncology", "myeloma"
    ],
}


def categorize_disease(disease_name: str) -> Optional[str]:
    """Categorize a disease based on keywords in its name."""
    disease_lower = disease_name.lower()

    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in disease_lower:
                return category

    return None


def get_routing_model(category: Optional[str]) -> str:
    """Determine which model to use based on category."""
    if category is None:
        return "best_rank"  # Fallback for uncategorized

    if category in TXGNN_CATEGORIES:
        return "txgnn"
    elif category in GB_CATEGORIES:
        return "gb"
    else:
        return "best_rank"  # Fallback for cancer and other categories


def create_features(drug_emb: np.ndarray, disease_emb: np.ndarray) -> np.ndarray:
    """Create features from embeddings for GB model."""
    concat = np.concatenate([drug_emb, disease_emb])
    product = drug_emb * disease_emb
    diff = drug_emb - disease_emb
    return np.concatenate([concat, product, diff])


def load_txgnn_results() -> pd.DataFrame:
    """Load TxGNN proper scoring results."""
    txgnn_path = REFERENCE_DIR / "txgnn_proper_scoring_results.csv"
    df = pd.read_csv(txgnn_path)
    logger.info(f"Loaded TxGNN results: {len(df)} diseases")
    return df


def parse_gt_ranks(gt_ranks_str: str) -> Dict[str, int]:
    """Parse the gt_ranks column from string to dict."""
    if pd.isna(gt_ranks_str):
        return {}
    try:
        return ast.literal_eval(gt_ranks_str)
    except (ValueError, SyntaxError):
        return {}


def load_gb_model_and_embeddings():
    """Load GB model and TransE embeddings."""
    with open(MODELS_DIR / "drug_repurposing_gb_enhanced.pkl", "rb") as f:
        gb_model = pickle.load(f)
    logger.info("Loaded GB enhanced model")

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
    logger.info(f"Loaded embeddings: {embeddings.shape}")

    return gb_model, embeddings, entity2id


def load_drugbank_lookup() -> Dict[str, str]:
    """Load DrugBank ID to name mapping."""
    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)
    name_to_id = {name.lower(): f"drkg:Compound::{db_id}" for db_id, name in id_to_name.items()}
    return name_to_id


# Disease name to MESH ID mappings
DISEASE_NAME_TO_MESH = {
    "hiv infection": "drkg:Disease::MESH:D015658",
    "hepatitis c": "drkg:Disease::MESH:D006526",
    "tuberculosis": "drkg:Disease::MESH:D014376",
    "breast cancer": "drkg:Disease::MESH:D001943",
    "lung cancer": "drkg:Disease::MESH:D008175",
    "colorectal cancer": "drkg:Disease::MESH:D015179",
    "hypertension": "drkg:Disease::MESH:D006973",
    "heart failure": "drkg:Disease::MESH:D006333",
    "atrial fibrillation": "drkg:Disease::MESH:D001281",
    "epilepsy": "drkg:Disease::MESH:D004827",
    "parkinson disease": "drkg:Disease::MESH:D010300",
    "alzheimer disease": "drkg:Disease::MESH:D000544",
    "rheumatoid arthritis": "drkg:Disease::MESH:D001172",
    "multiple sclerosis": "drkg:Disease::MESH:D009103",
    "psoriasis": "drkg:Disease::MESH:D011565",
    "type 2 diabetes mellitus": "drkg:Disease::MESH:D003924",
    "obesity": "drkg:Disease::MESH:D009765",
    "asthma": "drkg:Disease::MESH:D001249",
    "copd": "drkg:Disease::MESH:D029424",
    "osteoporosis": "drkg:Disease::MESH:D010024",
    "crohn disease": "drkg:Disease::MESH:D003424",
}


def get_gb_rankings_for_disease(
    disease_mesh: str,
    gb_model,
    embeddings: np.ndarray,
    entity2id: Dict[str, int],
    all_drug_ids: List[str],
    name_to_drugbank: Dict[str, str],
) -> Dict[str, int]:
    """Get GB model rankings for all drugs for a disease."""
    disease_idx = entity2id.get(disease_mesh)
    if disease_idx is None:
        return {}

    disease_emb = embeddings[disease_idx]
    scores = []
    drug_names = []

    drugbank_to_name = {v: k for k, v in name_to_drugbank.items()}

    for drug_id in all_drug_ids:
        drug_idx = entity2id.get(drug_id)
        if drug_idx is None:
            continue

        drug_emb = embeddings[drug_idx]
        features = create_features(drug_emb, disease_emb).reshape(1, -1)
        score = gb_model.predict_proba(features)[0, 1]
        scores.append(score)

        drug_name = drugbank_to_name.get(drug_id, drug_id.split("::")[-1]).lower()
        drug_names.append(drug_name)

    ranked_indices = np.argsort(scores)[::-1]

    rankings = {}
    for rank, idx in enumerate(ranked_indices, 1):
        drug_name = drug_names[idx]
        rankings[drug_name] = rank

    return rankings


def evaluate_routing_ensemble(
    txgnn_df: pd.DataFrame,
    gb_model,
    embeddings: np.ndarray,
    entity2id: Dict[str, int],
    all_drug_ids: List[str],
    name_to_drugbank: Dict[str, str],
    k: int = 30,
) -> Dict:
    """Evaluate the category-based routing ensemble."""

    results = {
        "txgnn_only": {"hits": 0, "total": 0, "diseases": []},
        "gb_only": {"hits": 0, "total": 0, "diseases": []},
        "best_rank": {"hits": 0, "total": 0, "diseases": []},
        "routing": {"hits": 0, "total": 0, "diseases": []},
    }

    routing_stats = {
        "txgnn_routed": 0,
        "gb_routed": 0,
        "best_rank_fallback": 0,
        "per_category": {},
    }

    # Process all TxGNN diseases
    for _, row in txgnn_df.iterrows():
        disease_name = row['disease'].lower()
        gt_ranks = parse_gt_ranks(row.get('gt_ranks', '{}'))

        if not gt_ranks:
            continue

        # Categorize disease
        category = categorize_disease(disease_name)
        routing_model = get_routing_model(category)

        # Track routing stats
        if category:
            if category not in routing_stats["per_category"]:
                routing_stats["per_category"][category] = {"count": 0, "hits": 0}
            routing_stats["per_category"][category]["count"] += 1

        # Get GB rankings if disease has MESH mapping
        mesh_id = DISEASE_NAME_TO_MESH.get(disease_name)
        gb_rankings = {}
        if mesh_id:
            gb_rankings = get_gb_rankings_for_disease(
                mesh_id, gb_model, embeddings, entity2id, all_drug_ids, name_to_drugbank
            )

        # Evaluate each GT drug
        for drug, txgnn_rank in gt_ranks.items():
            gb_rank = gb_rankings.get(drug)

            # TxGNN only
            txgnn_hit = txgnn_rank <= k
            results["txgnn_only"]["total"] += 1
            if txgnn_hit:
                results["txgnn_only"]["hits"] += 1

            # GB only (only if we have GB rankings)
            if gb_rankings:
                results["gb_only"]["total"] += 1
                gb_hit = gb_rank is not None and gb_rank <= k
                if gb_hit:
                    results["gb_only"]["hits"] += 1
            else:
                gb_hit = False

            # Best rank ensemble
            best_rank = min(txgnn_rank, gb_rank if gb_rank else float('inf'))
            best_rank_hit = best_rank <= k
            results["best_rank"]["total"] += 1
            if best_rank_hit:
                results["best_rank"]["hits"] += 1

            # Routing ensemble
            results["routing"]["total"] += 1
            if routing_model == "txgnn":
                routing_hit = txgnn_hit
                routing_stats["txgnn_routed"] += 1
            elif routing_model == "gb" and gb_rankings:
                routing_hit = gb_hit
                routing_stats["gb_routed"] += 1
            else:
                # Fallback to best rank
                routing_hit = best_rank_hit
                routing_stats["best_rank_fallback"] += 1

            if routing_hit:
                results["routing"]["hits"] += 1
                if category:
                    routing_stats["per_category"][category]["hits"] += 1

    return results, routing_stats


def main():
    logger.info("=" * 70)
    logger.info("Category-Based Routing Ensemble Evaluation")
    logger.info("=" * 70)

    # Load models and data
    txgnn_df = load_txgnn_results()
    gb_model, embeddings, entity2id = load_gb_model_and_embeddings()

    all_drug_ids = [eid for eid in entity2id.keys() if "Compound" in eid]
    logger.info(f"Total drugs in DRKG: {len(all_drug_ids)}")

    name_to_drugbank = load_drugbank_lookup()
    logger.info(f"DrugBank mappings: {len(name_to_drugbank)}")

    # Evaluate
    results, routing_stats = evaluate_routing_ensemble(
        txgnn_df, gb_model, embeddings, entity2id, all_drug_ids, name_to_drugbank
    )

    # Print results
    print("\n" + "=" * 80)
    print("CATEGORY-BASED ROUTING ENSEMBLE RESULTS")
    print("=" * 80)

    print("\n### Routing Strategy ###")
    print("TxGNN categories: metabolic, storage, psychiatric, dermatological, autoimmune, neurological")
    print("GB categories: infectious, cardiovascular, respiratory, renal, gastrointestinal")
    print("Fallback: best_rank for uncategorized diseases")

    print("\n### Overall Results ###")
    print(f"{'Model':<20} {'Hits@30':>10} {'Total':>10} {'Recall@30':>12}")
    print("-" * 55)

    for model_name, res in results.items():
        if res["total"] > 0:
            recall = res["hits"] / res["total"] * 100
            print(f"{model_name:<20} {res['hits']:>10} {res['total']:>10} {recall:>11.1f}%")

    print("\n### Routing Statistics ###")
    print(f"Drugs routed to TxGNN: {routing_stats['txgnn_routed']}")
    print(f"Drugs routed to GB: {routing_stats['gb_routed']}")
    print(f"Fallback to best_rank: {routing_stats['best_rank_fallback']}")

    print("\n### Per-Category Performance ###")
    print(f"{'Category':<25} {'Count':>8} {'Hits':>8} {'Recall':>10}")
    print("-" * 55)

    for cat, stats in sorted(routing_stats["per_category"].items(),
                              key=lambda x: x[1]["hits"]/max(x[1]["count"], 1),
                              reverse=True):
        recall = stats["hits"] / stats["count"] * 100 if stats["count"] > 0 else 0
        print(f"{cat:<25} {stats['count']:>8} {stats['hits']:>8} {recall:>9.1f}%")

    # Calculate improvement
    print("\n### Summary ###")
    txgnn_recall = results["txgnn_only"]["hits"] / results["txgnn_only"]["total"] * 100
    best_rank_recall = results["best_rank"]["hits"] / results["best_rank"]["total"] * 100
    routing_recall = results["routing"]["hits"] / results["routing"]["total"] * 100

    print(f"TxGNN alone: {txgnn_recall:.1f}% Recall@30")
    print(f"Best Rank ensemble: {best_rank_recall:.1f}% Recall@30")
    print(f"Category Routing ensemble: {routing_recall:.1f}% Recall@30")
    print(f"Routing vs TxGNN: {routing_recall - txgnn_recall:+.1f}%")
    print(f"Routing vs Best Rank: {routing_recall - best_rank_recall:+.1f}%")

    # Save results
    output = {
        "results": {k: {"hits": v["hits"], "total": v["total"],
                       "recall": v["hits"]/v["total"] if v["total"] > 0 else 0}
                   for k, v in results.items()},
        "routing_stats": routing_stats,
        "txgnn_categories": list(TXGNN_CATEGORIES),
        "gb_categories": list(GB_CATEGORIES),
    }

    output_path = ANALYSIS_DIR / "category_routing_results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
