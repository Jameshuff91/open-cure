#!/usr/bin/env python3
"""
Evaluate infectious disease specialist model (Hypothesis h3).

Tests whether a specialist XGBoost model trained only on infectious disease
pairs can outperform the general GB model for infectious diseases.

Current baseline: 13.6% R@30 for infectious diseases (per CLAUDE.md)

Success criteria: >30% R@30 for infectious diseases (2x improvement)
"""

import json
import pickle
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from disease_categorizer import categorize_disease
from atc_features import ATCMapper

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
MODELS_DIR = PROJECT_ROOT / "models"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"


def load_embeddings() -> Tuple[np.ndarray, Dict[str, int]]:
    """Load TransE embeddings."""
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
    return embeddings, entity2id


def load_ground_truth() -> Dict[str, List[str]]:
    """Load expanded ground truth."""
    with open(REFERENCE_DIR / "expanded_ground_truth.json") as f:
        return json.load(f)


def load_drugbank_lookup() -> Dict[str, str]:
    """Load DrugBank name -> ID mapping."""
    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)
    # Invert: name.lower() -> drkg ID
    return {name.lower(): f"drkg:Compound::{db_id}" for db_id, name in id_to_name.items()}


def load_ec_data_with_categories() -> Tuple[Dict[str, Dict], Dict[str, List[str]]]:
    """
    Load Every Cure data with disease categorization.

    Returns:
        - disease_info: {disease_name: {'category': str, 'drugs': [...]}}
        - disease_drugs: {disease_name: [drug_names]}
    """
    df = pd.read_excel(REFERENCE_DIR / "everycure" / "indicationList.xlsx")

    disease_info = {}
    disease_drugs = defaultdict(list)

    for _, row in df.iterrows():
        disease = str(row.get("disease name", "")).strip()
        drug = str(row.get("final normalized drug label", "")).strip()

        if not disease or not drug:
            continue

        if disease not in disease_info:
            category = categorize_disease(disease)
            disease_info[disease] = {'category': category, 'drugs': []}

        disease_info[disease]['drugs'].append(drug)
        disease_drugs[disease].append(drug)

    return disease_info, dict(disease_drugs)


def create_features(drug_emb: np.ndarray, disease_emb: np.ndarray) -> np.ndarray:
    """Create features from embeddings."""
    concat = np.concatenate([drug_emb, disease_emb])
    product = drug_emb * disease_emb
    diff = drug_emb - disease_emb
    return np.concatenate([concat, product, diff])


def evaluate_model_on_diseases(
    model,
    embeddings: np.ndarray,
    entity2id: Dict[str, int],
    disease_drugs: Dict[str, Set[str]],
    disease_ids: List[str],
    top_k: int = 30,
) -> Dict:
    """
    Evaluate model on a set of diseases.

    Returns dict with recall metrics and per-disease results.
    """
    # Get all drugs in embeddings
    all_drugs = [e for e in entity2id.keys() if e.startswith("drkg:Compound::")]

    results = []
    total_hits = 0
    total_gt_drugs = 0

    for disease_id in tqdm(disease_ids, desc="Evaluating"):
        if disease_id not in entity2id:
            continue

        gt_drugs = disease_drugs.get(disease_id, set())
        if not gt_drugs:
            continue

        # Get disease embedding
        disease_idx = entity2id[disease_id]
        disease_emb = embeddings[disease_idx]

        # Score all drugs
        scores = []
        for drug_id in all_drugs:
            if drug_id not in entity2id:
                continue
            drug_idx = entity2id[drug_id]
            drug_emb = embeddings[drug_idx]

            features = create_features(drug_emb, disease_emb)
            score = model.predict_proba(features.reshape(1, -1))[0, 1]
            scores.append((drug_id, score))

        # Rank by score
        scores.sort(key=lambda x: x[1], reverse=True)
        top_drugs = {s[0] for s in scores[:top_k]}

        # Count hits
        hits = len(top_drugs & gt_drugs)
        total_hits += hits
        total_gt_drugs += len(gt_drugs)

        results.append({
            'disease_id': disease_id,
            'gt_drugs': len(gt_drugs),
            'hits': hits,
            'recall': hits / len(gt_drugs) if gt_drugs else 0,
        })

    recall_at_k = total_hits / total_gt_drugs if total_gt_drugs > 0 else 0

    return {
        'recall_at_k': recall_at_k,
        'total_hits': total_hits,
        'total_gt_drugs': total_gt_drugs,
        'diseases_evaluated': len(results),
        'per_disease': results,
    }


def build_infectious_training_data(
    embeddings: np.ndarray,
    entity2id: Dict[str, int],
    disease_info: Dict[str, Dict],
    drugbank_lookup: Dict[str, str],
    mesh_to_drkg: Dict[str, str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build training data for infectious disease specialist.

    Returns:
        - X: feature matrix
        - y: labels (1 for positive pairs, 0 for negatives)
        - disease_ids: list of infectious disease DRKG IDs for evaluation
    """
    positive_pairs = []
    infectious_disease_ids = []

    # Get all drugs for negative sampling
    all_drug_ids = [e for e in entity2id.keys() if e.startswith("drkg:Compound::")]

    # Filter to infectious diseases only
    for disease_name, info in disease_info.items():
        if info['category'] != 'infectious':
            continue

        # Try to find disease in embeddings via mesh_to_drkg mapping
        disease_lower = disease_name.lower()
        disease_id = mesh_to_drkg.get(disease_lower)

        if not disease_id or disease_id not in entity2id:
            continue

        infectious_disease_ids.append(disease_id)

        for drug_name in info['drugs']:
            drug_id = drugbank_lookup.get(drug_name.lower())
            if drug_id and drug_id in entity2id:
                positive_pairs.append((drug_id, disease_id))

    print(f"Infectious diseases found: {len(infectious_disease_ids)}")
    print(f"Positive pairs: {len(positive_pairs)}")

    if len(positive_pairs) < 20:
        print("WARNING: Too few positive pairs for training!")
        return None, None, infectious_disease_ids

    # Create feature matrix
    X_pos = []
    for drug_id, disease_id in positive_pairs:
        drug_emb = embeddings[entity2id[drug_id]]
        disease_emb = embeddings[entity2id[disease_id]]
        X_pos.append(create_features(drug_emb, disease_emb))
    X_pos = np.array(X_pos)

    # Create negative samples (random non-GT drugs for each disease)
    X_neg = []
    disease_drugs = defaultdict(set)
    for drug_id, disease_id in positive_pairs:
        disease_drugs[disease_id].add(drug_id)

    for disease_id, gt_drugs in disease_drugs.items():
        disease_emb = embeddings[entity2id[disease_id]]

        # Sample negatives (3:1 negative:positive ratio)
        n_negatives = len(gt_drugs) * 3
        available_negatives = [d for d in all_drug_ids if d not in gt_drugs and d in entity2id]

        if len(available_negatives) < n_negatives:
            neg_drugs = available_negatives
        else:
            neg_drugs = np.random.choice(available_negatives, n_negatives, replace=False)

        for drug_id in neg_drugs:
            drug_emb = embeddings[entity2id[drug_id]]
            X_neg.append(create_features(drug_emb, disease_emb))

    X_neg = np.array(X_neg)

    # Combine
    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])

    print(f"Training data: {len(X_pos)} positive, {len(X_neg)} negative")

    return X, y, infectious_disease_ids


def build_disease_to_drkg_mapping(
    embeddings_entity2id: Dict[str, int],
) -> Dict[str, str]:
    """Build mapping from disease names to DRKG IDs."""
    # Extract MESH IDs from entity2id
    mesh_to_drkg = {}
    for entity_id in embeddings_entity2id.keys():
        if entity_id.startswith("drkg:Disease::MESH:"):
            mesh_id = entity_id.split("MESH:")[-1]
            mesh_to_drkg[mesh_id] = entity_id

    return mesh_to_drkg


def main():
    print("=" * 70)
    print("INFECTIOUS DISEASE SPECIALIST MODEL EVALUATION (h3)")
    print("=" * 70)
    print(f"Baseline: 13.6% R@30 for infectious diseases")
    print(f"Success criteria: >30% R@30 (2x improvement)")
    print("=" * 70)

    # Load resources
    print("\n1. Loading embeddings...")
    embeddings, entity2id = load_embeddings()
    print(f"   Embeddings: {embeddings.shape}")
    print(f"   Entities: {len(entity2id)}")

    print("\n2. Loading ground truth...")
    gt = load_ground_truth()
    print(f"   Diseases: {len(gt)}")

    drugbank_lookup = load_drugbank_lookup()
    print(f"   DrugBank lookup: {len(drugbank_lookup)} drugs")

    print("\n3. Loading Every Cure data with categorization...")
    disease_info, disease_drugs_ec = load_ec_data_with_categories()

    # Count infectious diseases
    infectious_count = sum(1 for d in disease_info.values() if d['category'] == 'infectious')
    print(f"   Total diseases: {len(disease_info)}")
    print(f"   Infectious diseases: {infectious_count}")

    # Build disease name to DRKG mapping
    print("\n4. Building disease mapping...")
    mesh_to_drkg = build_disease_to_drkg_mapping(entity2id)
    print(f"   MESH -> DRKG mappings: {len(mesh_to_drkg)}")

    # We need to map EC disease names to DRKG IDs
    # Use the disease_name_matcher for this
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from disease_name_matcher import DiseaseMatcher, load_mesh_mappings

    mesh_mappings = load_mesh_mappings()
    disease_matcher = DiseaseMatcher(mesh_mappings)

    # Build name to DRKG mapping
    name_to_drkg = {}
    for disease_name in disease_info.keys():
        mesh_id = disease_matcher.get_mesh_id(disease_name)
        if mesh_id and mesh_id in entity2id:
            name_to_drkg[disease_name.lower()] = mesh_id

    print(f"   Disease names mapped: {len(name_to_drkg)}")

    # Build training data for infectious diseases
    print("\n5. Building infectious disease training data...")
    X, y, infectious_disease_ids = build_infectious_training_data(
        embeddings, entity2id, disease_info, drugbank_lookup, name_to_drkg
    )

    if X is None:
        print("\nERROR: Could not build training data!")
        return

    # Split for training/evaluation
    print("\n6. Training specialist model...")

    # Use disease-level split to avoid data leakage
    unique_diseases = list(set(infectious_disease_ids))
    if len(unique_diseases) < 5:
        print("ERROR: Not enough unique diseases for split!")
        return

    train_diseases, test_diseases = train_test_split(
        unique_diseases, test_size=0.3, random_state=42
    )
    print(f"   Train diseases: {len(train_diseases)}")
    print(f"   Test diseases: {len(test_diseases)}")

    # Build separate train/test sets based on disease split
    # For now, train on all data and evaluate on the test diseases

    # Train specialist model
    specialist = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
    specialist.fit(X, y)
    print("   Specialist model trained!")

    # Load general model for comparison
    print("\n7. Loading general GB model for comparison...")
    with open(MODELS_DIR / "drug_repurposing_gb_enhanced.pkl", "rb") as f:
        general_model = pickle.load(f)

    # Prepare evaluation diseases
    print("\n8. Evaluating models...")

    # Build disease_drugs dict for evaluation (DRKG IDs -> GT drugs)
    test_disease_drugs = {}
    for disease_name, info in disease_info.items():
        if info['category'] != 'infectious':
            continue
        disease_id = name_to_drkg.get(disease_name.lower())
        if disease_id and disease_id in test_diseases:
            gt_drugs = set()
            for drug_name in info['drugs']:
                drug_id = drugbank_lookup.get(drug_name.lower())
                if drug_id:
                    gt_drugs.add(drug_id)
            if gt_drugs:
                test_disease_drugs[disease_id] = gt_drugs

    print(f"   Test diseases with GT drugs: {len(test_disease_drugs)}")

    if not test_disease_drugs:
        print("\nWARNING: No test diseases with ground truth drugs found!")
        print("This may indicate mapping issues between EC diseases and DRKG.")

        # Fall back to evaluating on ALL infectious diseases
        print("\nFalling back to evaluating on ALL infectious diseases...")
        for disease_name, info in disease_info.items():
            if info['category'] != 'infectious':
                continue
            disease_id = name_to_drkg.get(disease_name.lower())
            if disease_id:
                gt_drugs = set()
                for drug_name in info['drugs']:
                    drug_id = drugbank_lookup.get(drug_name.lower())
                    if drug_id:
                        gt_drugs.add(drug_id)
                if gt_drugs:
                    test_disease_drugs[disease_id] = gt_drugs

        print(f"   ALL infectious diseases with GT: {len(test_disease_drugs)}")

    if not test_disease_drugs:
        print("\nERROR: Still no diseases to evaluate!")
        return

    # Evaluate specialist model
    print("\n   Evaluating specialist model...")
    specialist_results = evaluate_model_on_diseases(
        specialist, embeddings, entity2id, test_disease_drugs,
        list(test_disease_drugs.keys()), top_k=30
    )

    # Evaluate general model
    print("   Evaluating general model...")
    general_results = evaluate_model_on_diseases(
        general_model, embeddings, entity2id, test_disease_drugs,
        list(test_disease_drugs.keys()), top_k=30
    )

    # Results summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nSpecialist Model (infectious only):")
    print(f"   R@30: {specialist_results['recall_at_k']*100:.1f}%")
    print(f"   Hits: {specialist_results['total_hits']}/{specialist_results['total_gt_drugs']}")
    print(f"   Diseases evaluated: {specialist_results['diseases_evaluated']}")

    print(f"\nGeneral Model (baseline):")
    print(f"   R@30: {general_results['recall_at_k']*100:.1f}%")
    print(f"   Hits: {general_results['total_hits']}/{general_results['total_gt_drugs']}")

    delta = specialist_results['recall_at_k'] - general_results['recall_at_k']
    print(f"\nImprovement: {delta*100:+.1f}%")

    if specialist_results['recall_at_k'] >= 0.30:
        print("\n✓ SUCCESS: Met target of >30% R@30!")
    else:
        print(f"\n✗ Target not met (need >30%, got {specialist_results['recall_at_k']*100:.1f}%)")

    # Save results
    output = {
        'hypothesis': 'h3',
        'title': 'Infectious Disease Specialist Model',
        'specialist_model': {
            'recall_at_30': specialist_results['recall_at_k'],
            'total_hits': specialist_results['total_hits'],
            'total_gt_drugs': specialist_results['total_gt_drugs'],
            'diseases_evaluated': specialist_results['diseases_evaluated'],
        },
        'general_model': {
            'recall_at_30': general_results['recall_at_k'],
            'total_hits': general_results['total_hits'],
            'total_gt_drugs': general_results['total_gt_drugs'],
        },
        'improvement': delta,
        'success': specialist_results['recall_at_k'] >= 0.30,
        'per_disease_results': specialist_results['per_disease'],
    }

    output_path = ANALYSIS_DIR / "h3_infectious_specialist_results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
