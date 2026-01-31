
"""
Meta-Confidence Prediction Helper

Usage:
    from meta_confidence_helper import predict_confidence, get_tier
    
    confidence = predict_confidence(disease_id, train_gt)
    tier = get_tier(confidence)
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

MODELS_DIR = Path(__file__).parent

# Load model on import
with open(MODELS_DIR / "meta_confidence_model.pkl", 'rb') as f:
    _model_data = pickle.load(f)
    _model = _model_data['model']
    _feature_names = _model_data['feature_names']
    _categories = _model_data['categories']

CATEGORIES = {
    'autoimmune': ['rheumat', 'lupus', 'sclerosis', 'arthritis', 'crohn', 'colitis', 'psoria'],
    'cancer': ['cancer', 'carcinoma', 'lymphoma', 'leukemia', 'melanoma', 'tumor', 'neoplasm'],
    'infectious': ['infection', 'viral', 'bacterial', 'fungal', 'malaria', 'tuberculosis', 'hiv', 'hepatitis'],
    'metabolic': ['diabetes', 'obesity', 'hyperlipid', 'cholesterol', 'metabolic'],
    'cardiovascular': ['heart', 'cardiac', 'hypertension', 'coronary', 'vascular', 'stroke'],
    'neurological': ['parkinson', 'alzheimer', 'epilepsy', 'migraine', 'neuropath'],
    'respiratory': ['asthma', 'copd', 'pneumonia', 'bronch', 'pulmonary'],
    'dermatological': ['skin', 'dermat', 'eczema', 'acne', 'psoria', 'rash'],
}

def categorize_disease(name):
    name_lower = name.lower()
    for cat, keywords in CATEGORIES.items():
        if any(kw in name_lower for kw in keywords):
            return cat
    return 'other'

def compute_features(disease_id, train_gt, emb_dict, disease_name="", k=20):
    """Compute features for meta-confidence prediction"""
    train_list = [d for d in train_gt if d in emb_dict]
    train_embs = np.array([emb_dict[d] for d in train_list], dtype=np.float32)
    
    test_emb = emb_dict[disease_id].reshape(1, -1)
    sims = cosine_similarity(test_emb, train_embs)[0]
    top_k_idx = np.argsort(sims)[-k:]
    
    features = {}
    top_k_sims = sims[top_k_idx]
    features['mean_sim'] = np.mean(top_k_sims)
    features['max_sim'] = np.max(top_k_sims)
    features['min_sim'] = np.min(top_k_sims)
    features['std_sim'] = np.std(top_k_sims)
    
    drug_pool = set()
    neighbor_gt_sizes = []
    for idx in top_k_idx:
        neighbor = train_list[idx]
        neighbor_drugs = train_gt.get(neighbor, set())
        drug_pool.update(neighbor_drugs)
        neighbor_gt_sizes.append(len(neighbor_drugs))
    
    features['drug_pool_size'] = len(drug_pool)
    features['mean_neighbor_gt'] = np.mean(neighbor_gt_sizes)
    features['max_neighbor_gt'] = np.max(neighbor_gt_sizes)
    features['category'] = categorize_disease(disease_name)
    
    return features

def predict_confidence(disease_id, train_gt, emb_dict, disease_name=""):
    """Predict probability of kNN achieving hit@30 for this disease"""
    features = compute_features(disease_id, train_gt, emb_dict, disease_name)
    
    feat_df = pd.DataFrame([features])
    cat_dummies = pd.get_dummies(feat_df['category'], prefix='cat')
    feat_df = pd.concat([feat_df.drop('category', axis=1), cat_dummies], axis=1)
    
    for col in _feature_names:
        if col not in feat_df.columns:
            feat_df[col] = 0
    feat_df = feat_df[_feature_names]
    
    return _model.predict_proba(feat_df.values)[0, 1]

def get_tier(confidence):
    """Get confidence tier (high/medium/low)"""
    if confidence >= 0.8:
        return 'high'
    elif confidence >= 0.5:
        return 'medium'
    else:
        return 'low'
