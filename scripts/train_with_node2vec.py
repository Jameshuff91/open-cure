#!/usr/bin/env python3
"""
Train a drug repurposing model using Node2Vec embeddings from DRKG.
This follows the Every Cure approach: concatenate drug+disease embeddings and train XGBoost.
"""

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from collections import defaultdict
import random
import pickle
from pathlib import Path

def load_embeddings(path: str) -> dict[str, np.ndarray]:
    """Load embeddings from CSV into a dict mapping entity -> embedding vector."""
    df = pd.read_csv(path)

    # Add drkg: prefix to match ground truth format
    embeddings = {}
    for _, row in df.iterrows():
        entity = row['entity']
        # Convert: "Compound::DB00001" -> "drkg:Compound::DB00001"
        drkg_entity = f"drkg:{entity}"
        embedding = row[[c for c in df.columns if c.startswith('dim_')]].values.astype(np.float32)
        embeddings[drkg_entity] = embedding

    return embeddings


def load_ground_truth() -> dict:
    """Load ground truth disease->drugs mapping."""
    with open('data/reference/expanded_ground_truth.json') as f:
        return json.load(f)


def create_training_data(embeddings: dict, ground_truth: dict, negative_ratio: int = 5):
    """
    Create training pairs: (drug_embedding || disease_embedding) -> label

    For each disease:
    - Positive: drugs that treat the disease
    - Negative: random drugs that don't treat the disease
    """
    X = []
    y = []
    pairs = []  # (drug, disease) for tracking

    # Get all drugs in embeddings
    all_drugs = [e for e in embeddings if 'Compound::' in e]
    print(f"Total drugs in embeddings: {len(all_drugs)}")

    diseases_with_embeddings = 0
    drugs_with_embeddings = 0

    for disease, drugs in ground_truth.items():
        # Check if disease has embedding
        if disease not in embeddings:
            continue
        diseases_with_embeddings += 1
        disease_emb = embeddings[disease]

        # Get drugs that have embeddings
        valid_drugs = [d for d in drugs if d in embeddings]
        if not valid_drugs:
            continue
        drugs_with_embeddings += len(valid_drugs)

        # Positive examples
        for drug in valid_drugs:
            drug_emb = embeddings[drug]
            combined = np.concatenate([drug_emb, disease_emb])
            X.append(combined)
            y.append(1)
            pairs.append((drug, disease))

        # Negative examples: sample drugs that don't treat this disease
        drugs_set = set(drugs)
        negative_pool = [d for d in all_drugs if d not in drugs_set and d in embeddings]
        n_negatives = min(len(valid_drugs) * negative_ratio, len(negative_pool))

        for neg_drug in random.sample(negative_pool, n_negatives):
            neg_emb = embeddings[neg_drug]
            combined = np.concatenate([neg_emb, disease_emb])
            X.append(combined)
            y.append(0)
            pairs.append((neg_drug, disease))

    print(f"Diseases with embeddings: {diseases_with_embeddings}")
    print(f"Drug-disease positive pairs: {drugs_with_embeddings}")
    print(f"Total training samples: {len(X)} (positives: {sum(y)}, negatives: {len(y)-sum(y)})")

    return np.array(X), np.array(y), pairs


def train_model(X: np.ndarray, y: np.ndarray):
    """Train XGBoost classifier."""
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric='auc'
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )

    return model


def evaluate_recall_at_k(model, embeddings: dict, ground_truth: dict, k: int = 30):
    """
    Evaluate per-drug Recall@K.
    For each disease, rank all drugs and check if ground truth drugs are in top K.
    Uses batched predictions for speed.
    """
    all_drugs = [e for e in embeddings if 'Compound::' in e]
    drug_embeddings = np.array([embeddings[d] for d in all_drugs])  # Pre-compute drug matrix

    per_drug_recalls = []
    disease_results = []

    # Filter to diseases with embeddings
    valid_diseases = [(d, drugs) for d, drugs in ground_truth.items() if d in embeddings]
    print(f"Evaluating {len(valid_diseases)} diseases...")

    for i, (disease, true_drugs) in enumerate(valid_diseases):
        if (i + 1) % 500 == 0:
            print(f"  Progress: {i+1}/{len(valid_diseases)} diseases")

        disease_emb = embeddings[disease]
        true_drugs_set = set(d for d in true_drugs if d in embeddings)

        if not true_drugs_set:
            continue

        # BATCH: Create all drug-disease pairs at once
        disease_emb_repeated = np.tile(disease_emb, (len(all_drugs), 1))
        X_batch = np.hstack([drug_embeddings, disease_emb_repeated])

        # Single batch prediction
        probs = model.predict_proba(X_batch)[:, 1]

        # Get top K drugs
        top_k_indices = np.argsort(probs)[-k:][::-1]
        top_k_drugs = set(all_drugs[idx] for idx in top_k_indices)

        # Calculate recall
        hits = len(top_k_drugs & true_drugs_set)
        recall = hits / len(true_drugs_set)
        per_drug_recalls.append(recall)

        disease_results.append({
            'disease': disease,
            'n_true_drugs': len(true_drugs_set),
            'n_hits': hits,
            'recall': recall
        })

    mean_recall = np.mean(per_drug_recalls) if per_drug_recalls else 0
    return mean_recall, disease_results


def main():
    print("=" * 60)
    print("Training Drug Repurposing Model with Node2Vec Embeddings")
    print("=" * 60)

    # Load embeddings
    print("\n[1] Loading embeddings...")
    embeddings = load_embeddings('data/embeddings/node2vec_256_named.csv')
    print(f"Loaded {len(embeddings)} entity embeddings")

    # Count types
    compounds = sum(1 for e in embeddings if 'Compound::' in e)
    diseases = sum(1 for e in embeddings if 'Disease::' in e)
    print(f"  Compounds: {compounds}")
    print(f"  Diseases: {diseases}")

    # Load ground truth
    print("\n[2] Loading ground truth...")
    ground_truth = load_ground_truth()
    print(f"Loaded {len(ground_truth)} diseases with ground truth")

    # Create training data
    print("\n[3] Creating training data...")
    X, y, pairs = create_training_data(embeddings, ground_truth, negative_ratio=5)

    # Train model
    print("\n[4] Training XGBoost model...")
    model = train_model(X, y)

    # Evaluate
    print("\n[5] Evaluating Recall@30...")
    recall, results = evaluate_recall_at_k(model, embeddings, ground_truth, k=30)
    print(f"\n{'='*60}")
    print(f"Per-Drug Recall@30: {recall*100:.1f}%")
    print(f"{'='*60}")

    # Save model
    model_path = 'models/drug_repurposing_node2vec.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {model_path}")

    # Show some disease-level results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('recall', ascending=False)

    print("\nTop 10 performing diseases:")
    for _, row in results_df.head(10).iterrows():
        print(f"  {row['disease']}: {row['recall']*100:.1f}% ({row['n_hits']}/{row['n_true_drugs']})")

    print("\nWorst 10 performing diseases:")
    for _, row in results_df.tail(10).iterrows():
        print(f"  {row['disease']}: {row['recall']*100:.1f}% ({row['n_hits']}/{row['n_true_drugs']})")

    return model, recall


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    model, recall = main()
