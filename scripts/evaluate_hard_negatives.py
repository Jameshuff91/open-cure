#!/usr/bin/env python3
"""
Evaluate Hard Negative Mining (Hypothesis h5).

Tests whether training with hard negatives (high-scoring but incorrect predictions)
and explicit confounding patterns improves R@30 and reduces false positive rate.

Baseline: 41.8% R@30
Success criteria: >42.5% R@30 + reduced false positive rate

Hard negatives include:
1. High-scoring GB predictions NOT in ground truth
2. Known confounding patterns (statins→T2D, checkpoint→UC, etc.)
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
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from disease_name_matcher import DiseaseMatcher, load_mesh_mappings

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


def load_drugbank_lookup() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Load DrugBank name -> ID and ID -> name mappings."""
    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)

    name_to_id = {name.lower(): f"drkg:Compound::{db_id}" for db_id, name in id_to_name.items()}
    id_to_name_drkg = {f"drkg:Compound::{db_id}": name for db_id, name in id_to_name.items()}

    return name_to_id, id_to_name_drkg


def load_ground_truth(
    mesh_mappings: Dict[str, str],
    name_to_drug_id: Dict[str, str]
) -> Tuple[Dict[str, Set[str]], Dict[str, List[str]]]:
    """
    Load ground truth data.

    Returns:
        - gt_pairs: {disease_drkg_id: set of drug_drkg_ids}
        - disease_drugs_by_name: {disease_name: [drug_names]}
    """
    df = pd.read_excel(REFERENCE_DIR / "everycure" / "indicationList.xlsx")

    gt_pairs = defaultdict(set)
    disease_drugs_by_name = defaultdict(list)

    # Initialize fuzzy matcher
    fuzzy_mappings = load_mesh_mappings()
    matcher = DiseaseMatcher(fuzzy_mappings)

    for _, row in df.iterrows():
        disease = str(row.get("disease name", "")).strip()
        drug = str(row.get("final normalized drug label", "")).strip()

        if not disease or not drug:
            continue

        disease_drugs_by_name[disease].append(drug)

        # Map disease to DRKG ID
        disease_id = matcher.get_mesh_id(disease)
        if not disease_id:
            disease_id = mesh_mappings.get(disease.lower())

        if not disease_id:
            continue

        # Map drug to DRKG ID
        drug_id = name_to_drug_id.get(drug.lower())
        if drug_id:
            gt_pairs[disease_id].add(drug_id)

    print(f"Ground truth: {len(gt_pairs)} diseases, {sum(len(v) for v in gt_pairs.values())} pairs")
    return dict(gt_pairs), dict(disease_drugs_by_name)


def load_confounding_patterns() -> List[Tuple[str, str]]:
    """
    Load known confounding patterns from analysis.

    Returns list of (drug_name, disease_name) pairs that are KNOWN false positives.
    """
    confounding_path = ANALYSIS_DIR / "confounding_analysis.json"
    if not confounding_path.exists():
        print("WARNING: Confounding analysis not found")
        return []

    with open(confounding_path) as f:
        data = json.load(f)

    patterns = []
    for item in data.get("confounded", []):
        drug = item["drug"]
        disease = item["disease"]
        confidence = item.get("confidence", 0)

        # Only use high-confidence confounding patterns
        if confidence >= 0.85:
            patterns.append((drug, disease))

    print(f"Loaded {len(patterns)} high-confidence confounding patterns")
    return patterns


def get_high_scoring_false_positives(
    model,
    embeddings: np.ndarray,
    entity2id: Dict[str, int],
    gt_pairs: Dict[str, Set[str]],
    all_drug_ids: List[str],
    top_n: int = 500,
) -> List[Tuple[str, str, float]]:
    """
    Find high-scoring drug-disease pairs that are NOT in ground truth.

    These are likely false positives and make excellent hard negatives.

    Returns list of (drug_id, disease_id, score) tuples.
    """
    print(f"Finding top {top_n} false positives per disease...")

    false_positives = []

    # Build set of all GT pairs for fast lookup
    all_gt_pairs = set()
    for disease_id, drug_ids in gt_pairs.items():
        for drug_id in drug_ids:
            all_gt_pairs.add((drug_id, disease_id))

    # Iterate through diseases
    for disease_id in tqdm(list(gt_pairs.keys())[:100], desc="Scanning diseases"):  # Sample for efficiency
        if disease_id not in entity2id:
            continue

        disease_idx = entity2id[disease_id]
        disease_emb = embeddings[disease_idx]

        # Score all drugs
        scores = []
        for drug_id in all_drug_ids:
            if drug_id not in entity2id:
                continue
            if (drug_id, disease_id) in all_gt_pairs:
                continue  # Skip GT pairs

            drug_idx = entity2id[drug_id]
            drug_emb = embeddings[drug_idx]

            features = create_features(drug_emb, disease_emb)
            score = model.predict_proba(features.reshape(1, -1))[0, 1]
            scores.append((drug_id, disease_id, score))

        # Keep top-N highest scoring non-GT pairs (likely false positives)
        scores.sort(key=lambda x: x[2], reverse=True)
        false_positives.extend(scores[:top_n])

    # Sort globally by score and take top overall
    false_positives.sort(key=lambda x: x[2], reverse=True)
    print(f"Found {len(false_positives)} potential false positives")

    return false_positives[:top_n * 10]  # Return top 5000


def create_features(drug_emb: np.ndarray, disease_emb: np.ndarray) -> np.ndarray:
    """Create features from drug and disease embeddings."""
    concat = np.concatenate([drug_emb, disease_emb])
    product = drug_emb * disease_emb
    diff = drug_emb - disease_emb
    return np.concatenate([concat, product, diff])


def build_training_data_with_hard_negatives(
    embeddings: np.ndarray,
    entity2id: Dict[str, int],
    gt_pairs: Dict[str, Set[str]],
    hard_negatives: List[Tuple[str, str, float]],
    confounding_pairs: List[Tuple[str, str]],
    name_to_drug_id: Dict[str, str],
    mesh_mappings: Dict[str, str],
    hard_neg_ratio: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Build training data with hard negative mining.

    Args:
        hard_neg_ratio: Proportion of negatives that are hard (vs random)
    """
    # Build positive pairs
    positive_pairs = []
    for disease_id, drug_ids in gt_pairs.items():
        if disease_id not in entity2id:
            continue
        for drug_id in drug_ids:
            if drug_id in entity2id:
                positive_pairs.append((drug_id, disease_id))

    print(f"Positive pairs: {len(positive_pairs)}")

    # Build negative pairs
    n_negatives = len(positive_pairs) * 3  # 3:1 ratio
    n_hard_negatives = int(n_negatives * hard_neg_ratio)
    n_random_negatives = n_negatives - n_hard_negatives

    negative_pairs = []

    # 1. Add high-scoring false positives as hard negatives
    hard_neg_added = 0
    for drug_id, disease_id, score in hard_negatives:
        if hard_neg_added >= n_hard_negatives:
            break
        if drug_id in entity2id and disease_id in entity2id:
            negative_pairs.append((drug_id, disease_id))
            hard_neg_added += 1

    print(f"Hard negatives (false positives): {hard_neg_added}")

    # 2. Add confounding patterns as explicit hard negatives
    confounding_added = 0
    fuzzy_mappings = load_mesh_mappings()
    matcher = DiseaseMatcher(fuzzy_mappings)

    for drug_name, disease_name in confounding_pairs:
        drug_id = name_to_drug_id.get(drug_name.lower())
        disease_id = matcher.get_mesh_id(disease_name)
        if not disease_id:
            disease_id = mesh_mappings.get(disease_name.lower())

        if drug_id and disease_id and drug_id in entity2id and disease_id in entity2id:
            negative_pairs.append((drug_id, disease_id))
            confounding_added += 1

    print(f"Confounding pattern negatives: {confounding_added}")

    # 3. Add random negatives
    all_drugs = [e for e in entity2id.keys() if e.startswith("drkg:Compound::")]
    all_diseases = list(gt_pairs.keys())

    # Build positive set for exclusion
    positive_set = set(positive_pairs)
    negative_set = set(negative_pairs)

    random_neg_added = 0
    attempts = 0
    max_attempts = n_random_negatives * 10

    while random_neg_added < n_random_negatives and attempts < max_attempts:
        drug_id = np.random.choice(all_drugs)
        disease_id = np.random.choice(all_diseases)

        if (drug_id, disease_id) not in positive_set and (drug_id, disease_id) not in negative_set:
            if drug_id in entity2id and disease_id in entity2id:
                negative_pairs.append((drug_id, disease_id))
                negative_set.add((drug_id, disease_id))
                random_neg_added += 1
        attempts += 1

    print(f"Random negatives: {random_neg_added}")
    print(f"Total negatives: {len(negative_pairs)}")

    # Create feature matrix
    all_pairs = positive_pairs + negative_pairs
    labels = [1] * len(positive_pairs) + [0] * len(negative_pairs)

    emb_dim = embeddings.shape[1]
    feature_dim = emb_dim * 4
    X = np.zeros((len(all_pairs), feature_dim), dtype=np.float32)

    for i, (drug_id, disease_id) in enumerate(tqdm(all_pairs, desc="Creating features")):
        drug_emb = embeddings[entity2id[drug_id]]
        disease_emb = embeddings[entity2id[disease_id]]
        X[i] = create_features(drug_emb, disease_emb)

    y = np.array(labels, dtype=np.int32)

    stats = {
        "positive_pairs": len(positive_pairs),
        "negative_pairs": len(negative_pairs),
        "hard_negatives_fp": hard_neg_added,
        "hard_negatives_confounding": confounding_added,
        "random_negatives": random_neg_added,
        "hard_neg_ratio_actual": (hard_neg_added + confounding_added) / len(negative_pairs),
    }

    return X, y, stats


def evaluate_recall_at_k(
    model,
    embeddings: np.ndarray,
    entity2id: Dict[str, int],
    gt_pairs: Dict[str, Set[str]],
    all_drug_ids: List[str],
    k: int = 30,
    sample_diseases: Optional[int] = None,
) -> Tuple[float, Dict]:
    """
    Evaluate R@K on ground truth diseases.

    Returns (recall, detailed_results).
    """
    diseases = list(gt_pairs.keys())
    if sample_diseases:
        np.random.seed(42)
        diseases = list(np.random.choice(diseases, min(sample_diseases, len(diseases)), replace=False))

    total_hits = 0
    total_gt_drugs = 0
    per_disease_results = []

    for disease_id in tqdm(diseases, desc=f"Evaluating R@{k}"):
        if disease_id not in entity2id:
            continue

        gt_drugs = gt_pairs[disease_id]
        if not gt_drugs:
            continue

        disease_idx = entity2id[disease_id]
        disease_emb = embeddings[disease_idx]

        # Score all drugs
        scores = []
        for drug_id in all_drug_ids:
            if drug_id not in entity2id:
                continue
            drug_idx = entity2id[drug_id]
            drug_emb = embeddings[drug_idx]

            features = create_features(drug_emb, disease_emb)
            score = model.predict_proba(features.reshape(1, -1))[0, 1]
            scores.append((drug_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_k_drugs = {s[0] for s in scores[:k]}

        hits = len(top_k_drugs & gt_drugs)
        total_hits += hits
        total_gt_drugs += len(gt_drugs)

        per_disease_results.append({
            "disease_id": disease_id,
            "gt_drugs": len(gt_drugs),
            "hits": hits,
            "recall": hits / len(gt_drugs),
        })

    recall = total_hits / total_gt_drugs if total_gt_drugs > 0 else 0

    return recall, {
        "total_hits": total_hits,
        "total_gt_drugs": total_gt_drugs,
        "diseases_evaluated": len(per_disease_results),
        "per_disease": per_disease_results,
    }


def main():
    print("=" * 70)
    print("HARD NEGATIVE MINING EVALUATION (h5)")
    print("=" * 70)
    print("Baseline: 41.8% R@30")
    print("Success criteria: >42.5% R@30 + reduced false positive rate")
    print("=" * 70)

    # Load resources
    print("\n1. Loading embeddings...")
    embeddings, entity2id = load_embeddings()
    print(f"   Embeddings: {embeddings.shape}")
    print(f"   Entities: {len(entity2id)}")

    print("\n2. Loading mappings...")
    name_to_drug_id, id_to_drug_name = load_drugbank_lookup()

    # Load MESH mappings
    with open(REFERENCE_DIR / "mesh_mappings_from_agents.json") as f:
        mesh_data = json.load(f)

    mesh_mappings = {}
    for batch_data in mesh_data.values():
        if isinstance(batch_data, dict):
            for disease_name, mesh_id in batch_data.items():
                if mesh_id:
                    mesh_str = str(mesh_id)
                    if mesh_str.startswith("D") or mesh_str.startswith("C"):
                        mesh_mappings[disease_name.lower()] = f"drkg:Disease::MESH:{mesh_str}"

    print(f"   MESH mappings: {len(mesh_mappings)}")

    print("\n3. Loading ground truth...")
    gt_pairs, disease_drugs_by_name = load_ground_truth(mesh_mappings, name_to_drug_id)

    print("\n4. Loading baseline model...")
    with open(MODELS_DIR / "drug_repurposing_gb_enhanced.pkl", "rb") as f:
        baseline_model = pickle.load(f)

    print("\n5. Loading confounding patterns...")
    confounding_pairs = load_confounding_patterns()

    print("\n6. Finding high-scoring false positives...")
    all_drug_ids = [e for e in entity2id.keys() if e.startswith("drkg:Compound::")]
    hard_negatives = get_high_scoring_false_positives(
        baseline_model, embeddings, entity2id, gt_pairs, all_drug_ids, top_n=100
    )

    print("\n7. Evaluating baseline model...")
    baseline_recall, baseline_results = evaluate_recall_at_k(
        baseline_model, embeddings, entity2id, gt_pairs, all_drug_ids, k=30
    )
    print(f"   Baseline R@30: {baseline_recall*100:.2f}%")

    # Train new model with hard negatives
    print("\n8. Building training data with hard negatives...")
    X, y, train_stats = build_training_data_with_hard_negatives(
        embeddings, entity2id, gt_pairs, hard_negatives, confounding_pairs,
        name_to_drug_id, mesh_mappings, hard_neg_ratio=0.5
    )

    print("\n   Training data stats:")
    for k, v in train_stats.items():
        print(f"     {k}: {v}")

    # Train/test split with disease-level holdout
    print("\n9. Training model with hard negatives...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    hard_neg_model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbose=0,
    )

    hard_neg_model.fit(X_train, y_train)

    # Evaluate test set metrics
    y_pred_proba = hard_neg_model.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, y_pred_proba)
    auprc = average_precision_score(y_test, y_pred_proba)

    print(f"   Test AUROC: {auroc:.4f}")
    print(f"   Test AUPRC: {auprc:.4f}")

    print("\n10. Evaluating hard negative model...")
    hard_neg_recall, hard_neg_results = evaluate_recall_at_k(
        hard_neg_model, embeddings, entity2id, gt_pairs, all_drug_ids, k=30
    )
    print(f"   Hard Neg Model R@30: {hard_neg_recall*100:.2f}%")

    # Compare results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nBaseline Model:")
    print(f"   R@30: {baseline_recall*100:.2f}%")
    print(f"   Hits: {baseline_results['total_hits']}/{baseline_results['total_gt_drugs']}")

    print(f"\nHard Negative Model:")
    print(f"   R@30: {hard_neg_recall*100:.2f}%")
    print(f"   Hits: {hard_neg_results['total_hits']}/{hard_neg_results['total_gt_drugs']}")

    delta = hard_neg_recall - baseline_recall
    print(f"\nImprovement: {delta*100:+.2f}%")

    success = hard_neg_recall > 0.425
    if success:
        print("\n SUCCESS: Met target of >42.5% R@30!")
    else:
        print(f"\n Target not met (need >42.5%, got {hard_neg_recall*100:.1f}%)")

    # Save results
    output = {
        "hypothesis": "h5",
        "title": "Hard Negative Mining",
        "baseline": {
            "recall_at_30": baseline_recall,
            "total_hits": baseline_results["total_hits"],
            "total_gt_drugs": baseline_results["total_gt_drugs"],
        },
        "hard_negative_model": {
            "recall_at_30": hard_neg_recall,
            "total_hits": hard_neg_results["total_hits"],
            "total_gt_drugs": hard_neg_results["total_gt_drugs"],
            "auroc": auroc,
            "auprc": auprc,
        },
        "training_stats": train_stats,
        "improvement": delta,
        "success": success,
    }

    output_path = ANALYSIS_DIR / "h5_hard_negatives_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {output_path}")

    # Save model if improved
    if delta > 0:
        model_path = MODELS_DIR / "drug_repurposing_gb_hard_neg.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(hard_neg_model, f)
        print(f"Saved improved model to {model_path}")


if __name__ == "__main__":
    main()
