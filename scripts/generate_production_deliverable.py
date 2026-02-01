#!/usr/bin/env python3
"""
Hypothesis h72: Production Deliverable with Confidence Tiers.

PURPOSE:
    Generate final drug repurposing predictions deliverable with confidence tiers.
    Uses h68 unified confidence scoring to tier predictions as HIGH/MEDIUM/LOW.

OUTPUT:
    Excel file with columns:
    - disease_name, disease_id
    - drug_name, drug_id
    - knn_score (raw similarity-weighted frequency)
    - confidence_prob (combined_avg from h68)
    - confidence_tier (HIGH/MEDIUM/LOW)
    - category
    - pool_size (number of drugs in kNN neighbor pool)

TIERS:
    - HIGH: confidence >= 0.7 (88% precision)
    - MEDIUM: 0.5 <= confidence < 0.7 (~70% precision)
    - LOW: confidence < 0.5 (exploratory)
"""

import json
import pickle
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, List

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from disease_name_matcher import DiseaseMatcher, load_mesh_mappings

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "data" / "deliverables"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"

# Category keywords
CATEGORY_KEYWORDS = {
    'autoimmune': ['rheumat', 'lupus', 'sclerosis', 'arthritis', 'crohn', 'colitis', 'psoria'],
    'cancer': ['cancer', 'carcinoma', 'lymphoma', 'leukemia', 'melanoma', 'tumor', 'neoplasm'],
    'infectious': ['infection', 'viral', 'bacterial', 'fungal', 'malaria', 'tuberculosis', 'hiv', 'hepatitis'],
    'metabolic': ['diabetes', 'obesity', 'hyperlipid', 'cholesterol', 'metabolic'],
    'cardiovascular': ['heart', 'cardiac', 'hypertension', 'coronary', 'vascular', 'stroke'],
    'neurological': ['parkinson', 'alzheimer', 'epilepsy', 'migraine', 'neuropath'],
    'respiratory': ['asthma', 'copd', 'pneumonia', 'bronch', 'pulmonary'],
    'dermatological': ['skin', 'dermat', 'eczema', 'acne', 'rash'],
    'psychiatric': ['depression', 'anxiety', 'bipolar', 'schizo', 'psychiatric'],
    'gastrointestinal': ['gastric', 'bowel', 'intestin', 'crohn', 'liver', 'hepat', 'esophag', 'colitis'],
    'hematological': ['anemia', 'leukemia', 'lymphoma', 'platelet', 'hemophilia', 'myeloma'],
    'renal': ['kidney', 'renal', 'nephro'],
    'musculoskeletal': ['arthritis', 'osteo', 'muscular', 'bone'],
    'ophthalmic': ['eye', 'retina', 'glaucoma', 'macular', 'optic'],
    'endocrine': ['thyroid', 'adrenal', 'pituitary', 'endocrine'],
}

CATEGORY_HIT_RATES = {
    'endocrine': 1.00,
    'autoimmune': 0.929,
    'dermatological': 0.882,
    'psychiatric': 0.833,
    'infectious': 0.75,
    'respiratory': 0.714,
    'cancer': 0.708,
    'ophthalmic': 0.667,
    'cardiovascular': 0.625,
    'neurological': 0.60,
    'other': 0.578,
    'metabolic': 0.545,
    'renal': 0.40,
    'musculoskeletal': 0.333,
    'hematological': 0.222,
    'gastrointestinal': 0.05
}


def load_node2vec_embeddings() -> Dict[str, np.ndarray]:
    df = pd.read_csv(EMBEDDINGS_DIR / "node2vec_256_named.csv")
    dim_cols = [c for c in df.columns if c.startswith("dim_")]
    embeddings = {}
    for _, row in df.iterrows():
        entity = f"drkg:{row['entity']}"
        embeddings[entity] = row[dim_cols].values.astype(np.float32)
    return embeddings


def load_drugbank_lookup():
    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)
    name_to_id = {name.lower(): f"drkg:Compound::{db_id}" for db_id, name in id_to_name.items()}
    id_to_name_drkg = {f"drkg:Compound::{db_id}": name for db_id, name in id_to_name.items()}
    return name_to_id, id_to_name_drkg


def load_mesh_mappings_from_file():
    mesh_path = REFERENCE_DIR / "mesh_mappings_from_agents.json"
    if not mesh_path.exists():
        return {}
    with open(mesh_path) as f:
        mesh_data = json.load(f)
    mesh_mappings = {}
    for batch_data in mesh_data.values():
        if isinstance(batch_data, dict):
            for disease_name, mesh_id in batch_data.items():
                if mesh_id:
                    mesh_str = str(mesh_id)
                    if mesh_str.startswith("D") or mesh_str.startswith("C"):
                        mesh_mappings[disease_name.lower()] = f"drkg:Disease::MESH:{mesh_str}"
    return mesh_mappings


def load_ground_truth(mesh_mappings, name_to_drug_id):
    df = pd.read_excel(REFERENCE_DIR / "everycure" / "indicationList.xlsx")
    fuzzy_mappings = load_mesh_mappings()
    matcher = DiseaseMatcher(fuzzy_mappings)

    gt_pairs = defaultdict(set)
    disease_names = {}

    for _, row in df.iterrows():
        disease = str(row.get("disease name", "")).strip()
        drug = str(row.get("final normalized drug label", "")).strip()
        if not disease or not drug:
            continue
        disease_id = matcher.get_mesh_id(disease)
        if not disease_id:
            disease_id = mesh_mappings.get(disease.lower())
        if not disease_id:
            continue
        drug_id = name_to_drug_id.get(drug.lower())
        if drug_id:
            gt_pairs[disease_id].add(drug_id)
            disease_names[disease_id] = disease

    return dict(gt_pairs), disease_names


def categorize_disease(name: str) -> str:
    name_lower = name.lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in name_lower for kw in keywords):
            return cat
    return 'other'


def knn_predict(
    disease_id: str,
    emb_dict: Dict[str, np.ndarray],
    all_gt: Dict[str, Set[str]],
    train_disease_list: List[str],
    train_disease_embs: np.ndarray,
    k: int = 20,
    top_n: int = 30,
) -> List[Dict]:
    """Get top-N drug predictions for a disease using kNN."""
    test_emb = emb_dict[disease_id].reshape(1, -1)
    sims = cosine_similarity(test_emb, train_disease_embs)[0]
    top_k_idx = np.argsort(sims)[-k:]
    top_k_sims = sims[top_k_idx]

    # Count drug frequency weighted by similarity
    drug_scores = defaultdict(float)
    for idx in top_k_idx:
        neighbor_disease = train_disease_list[idx]
        neighbor_sim = sims[idx]
        for drug_id in all_gt.get(neighbor_disease, set()):
            if drug_id in emb_dict:
                drug_scores[drug_id] += neighbor_sim

    if not drug_scores:
        return []

    sorted_drugs = sorted(drug_scores.items(), key=lambda x: x[1], reverse=True)
    predictions = []
    for drug_id, score in sorted_drugs[:top_n]:
        predictions.append({
            'drug_id': drug_id,
            'knn_score': score,
        })

    return predictions


def compute_confidence(
    disease_id: str,
    disease_name: str,
    emb_dict: Dict[str, np.ndarray],
    all_gt: Dict[str, Set[str]],
    train_disease_list: List[str],
    train_disease_embs: np.ndarray,
    h65_model,
    h65_features: List[str],
    h52_model,
    h52_features: List[str],
    k: int = 20,
) -> Dict:
    """Compute unified confidence for a disease."""
    test_emb = emb_dict[disease_id].reshape(1, -1)
    sims = cosine_similarity(test_emb, train_disease_embs)[0]
    top_k_idx = np.argsort(sims)[-k:]
    top_k_sims = sims[top_k_idx]

    # Compute features
    drug_pool = set()
    neighbors_with_gt = 0
    neighbor_gt_sizes = []
    for idx in top_k_idx:
        neighbor_disease = train_disease_list[idx]
        neighbor_drugs = all_gt.get(neighbor_disease, set())
        if neighbor_drugs:
            neighbors_with_gt += 1
            for d in neighbor_drugs:
                if d in emb_dict:
                    drug_pool.add(d)
        neighbor_gt_sizes.append(len(neighbor_drugs))

    category = categorize_disease(disease_name)

    # h65 features
    h65_feat = {
        'avg_sim': float(np.mean(top_k_sims)),
        'max_sim': float(np.max(top_k_sims)),
        'min_sim': float(np.min(top_k_sims)),
        'std_sim': float(np.std(top_k_sims)),
        'pool_size': len(drug_pool),
        'neighbors_w_gt': neighbors_with_gt,
        'gt_drugs': 0,
        'genes': 0,
    }
    for cat in ['autoimmune', 'cancer', 'cardiovascular', 'dermatological', 'endocrine',
                'gastrointestinal', 'infectious', 'metabolic', 'neurological', 'other',
                'psychiatric', 'respiratory']:
        h65_feat[f'cat_{cat}'] = 1 if category == cat else 0

    X_h65 = np.array([[h65_feat.get(f, 0) for f in h65_features]], dtype=np.float32)
    prob_h65 = float(h65_model.predict_proba(X_h65)[0, 1])

    # h52 features
    h52_feat = {
        'mean_sim': float(np.mean(top_k_sims)),
        'max_sim': float(np.max(top_k_sims)),
        'min_sim': float(np.min(top_k_sims)),
        'std_sim': float(np.std(top_k_sims)),
        'drug_pool_size': len(drug_pool),
        'mean_neighbor_gt': float(np.mean(neighbor_gt_sizes)) if neighbor_gt_sizes else 0,
        'max_neighbor_gt': float(np.max(neighbor_gt_sizes)) if neighbor_gt_sizes else 0,
    }
    for cat in ['autoimmune', 'cancer', 'cardiovascular', 'dermatological',
                'infectious', 'metabolic', 'neurological', 'other', 'respiratory']:
        h52_feat[f'cat_{cat}'] = 1 if category == cat else 0

    X_h52 = np.array([[h52_feat.get(f, 0) for f in h52_features]], dtype=np.float32)
    prob_h52 = float(h52_model.predict_proba(X_h52)[0, 1])

    # Category prior
    prob_cat = CATEGORY_HIT_RATES.get(category, 0.578)

    # Combined average
    combined_prob = np.mean([prob_h65, prob_h52, prob_cat])

    # Tier
    if combined_prob >= 0.7:
        tier = 'HIGH'
    elif combined_prob >= 0.5:
        tier = 'MEDIUM'
    else:
        tier = 'LOW'

    return {
        'prob_h65': prob_h65,
        'prob_h52': prob_h52,
        'prob_category': prob_cat,
        'confidence_prob': combined_prob,
        'confidence_tier': tier,
        'category': category,
        'pool_size': len(drug_pool),
        'neighbors_with_gt': neighbors_with_gt,
    }


def main():
    start_time = time.time()

    print("=" * 70)
    print("h72: PRODUCTION DELIVERABLE WITH CONFIDENCE TIERS")
    print("=" * 70)
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load models
    print("Loading models...")
    h65_model = joblib.load(MODELS_DIR / "disease_success_predictor.pkl")
    h65_features = joblib.load(MODELS_DIR / "disease_success_features.pkl")

    with open(MODELS_DIR / "meta_confidence_model.pkl", 'rb') as f:
        h52_data = pickle.load(f)
    h52_model = h52_data['model']
    h52_features = h52_data['feature_names']

    # Load data
    print("Loading data...")
    emb_dict = load_node2vec_embeddings()
    name_to_drug_id, id_to_drug_name = load_drugbank_lookup()
    mesh_mappings = load_mesh_mappings_from_file()
    all_gt, disease_names = load_ground_truth(mesh_mappings, name_to_drug_id)
    print(f"  GT: {len(all_gt)} diseases, {sum(len(v) for v in all_gt.values())} pairs")

    # Use all diseases as training (for production we use all available data)
    train_disease_list = [d for d in all_gt if d in emb_dict]
    train_disease_embs = np.array([emb_dict[d] for d in train_disease_list], dtype=np.float32)
    print(f"  Diseases with embeddings: {len(train_disease_list)}")

    # Generate predictions for all diseases
    print("\nGenerating predictions...")
    all_predictions = []
    tier_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}

    for i, disease_id in enumerate(train_disease_list):
        if (i + 1) % 100 == 0:
            print(f"  Processing disease {i+1}/{len(train_disease_list)}")

        disease_name = disease_names.get(disease_id, disease_id)
        gt_drugs = all_gt.get(disease_id, set())

        # Get predictions (exclude self from neighbors)
        other_diseases = [d for d in train_disease_list if d != disease_id]
        other_embs = np.array([emb_dict[d] for d in other_diseases], dtype=np.float32)

        predictions = knn_predict(
            disease_id, emb_dict, all_gt, other_diseases, other_embs,
            k=20, top_n=30
        )

        # Compute confidence
        confidence = compute_confidence(
            disease_id, disease_name, emb_dict, all_gt,
            other_diseases, other_embs,
            h65_model, h65_features, h52_model, h52_features, k=20
        )

        tier_counts[confidence['confidence_tier']] += 1

        # Add to results
        for pred in predictions:
            drug_name = id_to_drug_name.get(pred['drug_id'], pred['drug_id'])
            is_known = pred['drug_id'] in gt_drugs

            all_predictions.append({
                'disease_name': disease_name,
                'disease_id': disease_id,
                'drug_name': drug_name,
                'drug_id': pred['drug_id'],
                'knn_score': round(pred['knn_score'], 4),
                'confidence_prob': round(confidence['confidence_prob'], 3),
                'confidence_tier': confidence['confidence_tier'],
                'category': confidence['category'],
                'pool_size': confidence['pool_size'],
                'neighbors_with_gt': confidence['neighbors_with_gt'],
                'is_known_indication': is_known,
            })

    # Create DataFrame
    df = pd.DataFrame(all_predictions)
    print(f"\nTotal predictions: {len(df)}")
    print(f"Diseases: {df['disease_id'].nunique()}")
    print(f"Drugs: {df['drug_id'].nunique()}")

    # Tier summary
    print("\nDisease Tier Distribution:")
    for tier, count in sorted(tier_counts.items()):
        pct = count / len(train_disease_list) * 100
        print(f"  {tier}: {count} diseases ({pct:.1f}%)")

    # Summary by tier
    print("\nPredictions by Tier:")
    tier_summary = df.groupby('confidence_tier').agg({
        'disease_id': 'nunique',
        'drug_id': 'count',
        'is_known_indication': 'sum'
    }).reset_index()
    tier_summary.columns = ['Tier', 'Diseases', 'Predictions', 'Known Indications']
    print(tier_summary.to_string(index=False))

    # Novel predictions (not in GT)
    novel_df = df[~df['is_known_indication']].copy()
    print(f"\nNovel predictions (not in GT): {len(novel_df)}")

    # Save full deliverable
    output_path = OUTPUT_DIR / "drug_repurposing_predictions_with_confidence.xlsx"
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Full predictions
        df.to_excel(writer, sheet_name='All Predictions', index=False)

        # High confidence novel predictions
        high_novel = novel_df[novel_df['confidence_tier'] == 'HIGH'].sort_values(
            ['confidence_prob', 'knn_score'], ascending=[False, False]
        )
        high_novel.to_excel(writer, sheet_name='HIGH Confidence Novel', index=False)

        # Summary by disease
        disease_summary = df.groupby(['disease_name', 'confidence_tier', 'category']).agg({
            'drug_id': 'count',
            'is_known_indication': 'sum',
            'confidence_prob': 'first',
            'pool_size': 'first',
        }).reset_index()
        disease_summary.columns = ['Disease', 'Tier', 'Category', 'Predictions', 'Known', 'Confidence', 'Pool Size']
        disease_summary = disease_summary.sort_values(['Tier', 'Confidence'], ascending=[True, False])
        disease_summary.to_excel(writer, sheet_name='Disease Summary', index=False)

        # Tier summary
        tier_summary.to_excel(writer, sheet_name='Tier Summary', index=False)

    print(f"\nSaved to {output_path}")

    # Also save as JSON for programmatic use
    json_path = OUTPUT_DIR / "drug_repurposing_predictions_with_confidence.json"
    df.to_json(json_path, orient='records', indent=2)
    print(f"Saved JSON to {json_path}")

    # Print top high-confidence novel predictions for validation
    print("\n" + "=" * 70)
    print("TOP 20 HIGH-CONFIDENCE NOVEL PREDICTIONS")
    print("=" * 70)
    top_novel = high_novel.head(20)
    for _, row in top_novel.iterrows():
        print(f"  {row['drug_name']} -> {row['disease_name']}")
        print(f"    Confidence: {row['confidence_prob']:.3f}, Score: {row['knn_score']:.2f}, Category: {row['category']}")

    elapsed = time.time() - start_time
    print(f"\nElapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
