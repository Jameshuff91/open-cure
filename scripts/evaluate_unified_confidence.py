#!/usr/bin/env python3
"""
Hypothesis h68: Unified Confidence-Weighted Predictions for Production.

PURPOSE:
    Combine multiple confidence signals into a single production-ready score:
    1. h65 disease success predictor (RF) - predicts if kNN will hit@30
    2. h52 meta-confidence model (XGBoost) - predicts hit@30 probability
    3. h58/h59 category hit rates - prior probability by disease category

APPROACH:
    Three combination strategies tested:
    A) Simple average of all three confidence signals
    B) Weighted average (optimize weights on validation set)
    C) Stacked model (train meta-model to combine signals)

SUCCESS CRITERIA:
    - Combined confidence achieves >75% precision on HIGH tier (prob >= 0.7)
    - Improved calibration vs individual models

EVALUATION:
    Multi-seed cross-validation on held-out diseases.
"""

import json
import pickle
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, Tuple, List, Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from disease_name_matcher import DiseaseMatcher, load_mesh_mappings

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
MODELS_DIR = PROJECT_ROOT / "models"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"

SEEDS = [42, 123, 456, 789, 1024]

# Category hit rates from h58/h59
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

# Disease category keywords
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


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_node2vec_embeddings() -> Dict[str, np.ndarray]:
    df = pd.read_csv(EMBEDDINGS_DIR / "node2vec_256_named.csv")
    dim_cols = [c for c in df.columns if c.startswith("dim_")]
    embeddings = {}
    for _, row in df.iterrows():
        entity = f"drkg:{row['entity']}"
        embeddings[entity] = row[dim_cols].values.astype(np.float32)
    return embeddings


def load_drugbank_lookup() -> Tuple[Dict[str, str], Dict[str, str]]:
    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)
    name_to_id = {name.lower(): f"drkg:Compound::{db_id}" for db_id, name in id_to_name.items()}
    id_to_name_drkg = {f"drkg:Compound::{db_id}": name for db_id, name in id_to_name.items()}
    return name_to_id, id_to_name_drkg


def load_mesh_mappings_from_file() -> Dict[str, str]:
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


def load_ground_truth(
    mesh_mappings: Dict[str, str],
    name_to_drug_id: Dict[str, str],
) -> Tuple[Dict[str, Set[str]], Dict[str, str]]:
    """Load ground truth. Returns (disease_id -> drug_ids, disease_id -> disease_name)."""
    df = pd.read_excel(REFERENCE_DIR / "everycure" / "indicationList.xlsx")
    fuzzy_mappings = load_mesh_mappings()
    matcher = DiseaseMatcher(fuzzy_mappings)

    gt_pairs: Dict[str, Set[str]] = defaultdict(set)
    disease_names: Dict[str, str] = {}

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


def disease_level_split(
    gt_pairs: Dict[str, Set[str]],
    valid_entity_check,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    valid_diseases = [d for d in gt_pairs if valid_entity_check(d)]
    rng = np.random.RandomState(seed)
    rng.shuffle(valid_diseases)
    n_test = max(1, int(len(valid_diseases) * test_fraction))
    test_diseases = set(valid_diseases[:n_test])
    train_diseases = set(valid_diseases[n_test:])
    train_gt = {d: gt_pairs[d] for d in train_diseases}
    test_gt = {d: gt_pairs[d] for d in test_diseases}
    return train_gt, test_gt


# ─── Category Detection ───────────────────────────────────────────────────────

def categorize_disease(name: str) -> str:
    """Categorize disease by name using keyword matching."""
    name_lower = name.lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in name_lower for kw in keywords):
            return cat
    return 'other'


# ─── kNN Evaluation (from h39) ────────────────────────────────────────────────

def knn_evaluate_single_disease(
    disease_id: str,
    emb_dict: Dict[str, np.ndarray],
    train_gt: Dict[str, Set[str]],
    train_disease_list: List[str],
    train_disease_embs: np.ndarray,
    k: int = 20,
) -> Tuple[Set[str], Dict[str, float]]:
    """
    For a test disease, get kNN predictions and drug scores.
    Returns (top_30_drugs, drug_scores).
    """
    test_emb = emb_dict[disease_id].reshape(1, -1)
    sims = cosine_similarity(test_emb, train_disease_embs)[0]
    top_k_idx = np.argsort(sims)[-k:]

    drug_counts: Dict[str, float] = defaultdict(float)
    for idx in top_k_idx:
        neighbor_disease = train_disease_list[idx]
        neighbor_sim = sims[idx]
        for drug_id in train_gt[neighbor_disease]:
            if drug_id in emb_dict:
                drug_counts[drug_id] += neighbor_sim

    if drug_counts:
        sorted_drugs = sorted(drug_counts.items(), key=lambda x: x[1], reverse=True)
        top_30 = {d for d, _ in sorted_drugs[:30]}
    else:
        top_30 = set()

    return top_30, dict(drug_counts)


# ─── Confidence Feature Computation ───────────────────────────────────────────

def compute_h65_features(
    disease_id: str,
    disease_name: str,
    emb_dict: Dict[str, np.ndarray],
    train_gt: Dict[str, Set[str]],
    train_disease_list: List[str],
    train_disease_embs: np.ndarray,
    gt_drugs: Set[str],
    k: int = 20,
) -> Dict[str, float]:
    """Compute features used by h65 success predictor model."""
    test_emb = emb_dict[disease_id].reshape(1, -1)
    sims = cosine_similarity(test_emb, train_disease_embs)[0]
    top_k_idx = np.argsort(sims)[-k:]
    top_k_sims = sims[top_k_idx]

    # Count neighbors with GT and pool size
    drug_pool = set()
    neighbors_with_gt = 0
    for idx in top_k_idx:
        neighbor_disease = train_disease_list[idx]
        neighbor_drugs = train_gt.get(neighbor_disease, set())
        if neighbor_drugs:
            neighbors_with_gt += 1
            for d in neighbor_drugs:
                if d in emb_dict:
                    drug_pool.add(d)

    category = categorize_disease(disease_name)

    return {
        'avg_sim': float(np.mean(top_k_sims)),
        'max_sim': float(np.max(top_k_sims)),
        'min_sim': float(np.min(top_k_sims)),
        'std_sim': float(np.std(top_k_sims)),
        'pool_size': len(drug_pool),
        'neighbors_w_gt': neighbors_with_gt,
        'gt_drugs': len(gt_drugs),
        'genes': 0,  # Not available here, use 0
        'category': category,
    }


def compute_h52_features(
    disease_id: str,
    disease_name: str,
    emb_dict: Dict[str, np.ndarray],
    train_gt: Dict[str, Set[str]],
    train_disease_list: List[str],
    train_disease_embs: np.ndarray,
    k: int = 20,
) -> Dict[str, float]:
    """Compute features used by h52 meta-confidence model."""
    test_emb = emb_dict[disease_id].reshape(1, -1)
    sims = cosine_similarity(test_emb, train_disease_embs)[0]
    top_k_idx = np.argsort(sims)[-k:]
    top_k_sims = sims[top_k_idx]

    drug_pool = set()
    neighbor_gt_sizes = []
    for idx in top_k_idx:
        neighbor_disease = train_disease_list[idx]
        neighbor_drugs = train_gt.get(neighbor_disease, set())
        for d in neighbor_drugs:
            if d in emb_dict:
                drug_pool.add(d)
        neighbor_gt_sizes.append(len(neighbor_drugs))

    category = categorize_disease(disease_name)

    return {
        'mean_sim': float(np.mean(top_k_sims)),
        'max_sim': float(np.max(top_k_sims)),
        'min_sim': float(np.min(top_k_sims)),
        'std_sim': float(np.std(top_k_sims)),
        'drug_pool_size': len(drug_pool),
        'mean_neighbor_gt': float(np.mean(neighbor_gt_sizes)) if neighbor_gt_sizes else 0,
        'max_neighbor_gt': float(np.max(neighbor_gt_sizes)) if neighbor_gt_sizes else 0,
        'category': category,
    }


def predict_h65_prob(features: Dict[str, float], model, feature_names: List[str]) -> float:
    """Get h65 success probability."""
    # Create feature vector with one-hot encoding
    feat_dict = {k: v for k, v in features.items() if k != 'category'}
    category = features.get('category', 'other')

    # One-hot encode category
    for cat in ['autoimmune', 'cancer', 'cardiovascular', 'dermatological', 'endocrine',
                'gastrointestinal', 'infectious', 'metabolic', 'neurological', 'other',
                'psychiatric', 'respiratory']:
        feat_dict[f'cat_{cat}'] = 1 if category == cat else 0

    # Create feature vector in correct order
    X = np.array([[feat_dict.get(f, 0) for f in feature_names]], dtype=np.float32)
    return float(model.predict_proba(X)[0, 1])


def predict_h52_prob(features: Dict[str, float], model, feature_names: List[str]) -> float:
    """Get h52 meta-confidence probability."""
    feat_dict = {k: v for k, v in features.items() if k != 'category'}
    category = features.get('category', 'other')

    # One-hot encode category
    for cat in ['autoimmune', 'cancer', 'cardiovascular', 'dermatological',
                'infectious', 'metabolic', 'neurological', 'other', 'respiratory']:
        feat_dict[f'cat_{cat}'] = 1 if category == cat else 0

    X = np.array([[feat_dict.get(f, 0) for f in feature_names]], dtype=np.float32)
    return float(model.predict_proba(X)[0, 1])


def get_category_prior(category: str) -> float:
    """Get category-based prior probability."""
    return CATEGORY_HIT_RATES.get(category, 0.578)


# ─── Unified Confidence Functions ─────────────────────────────────────────────

def combine_simple_average(probs: List[float]) -> float:
    """Simple average of probabilities."""
    return np.mean(probs)


def combine_weighted_average(probs: List[float], weights: List[float]) -> float:
    """Weighted average of probabilities."""
    return np.average(probs, weights=weights)


def combine_harmonic_mean(probs: List[float]) -> float:
    """Harmonic mean - penalizes if any signal is low."""
    probs = [max(p, 0.001) for p in probs]  # Avoid division by zero
    return len(probs) / sum(1/p for p in probs)


def combine_max(probs: List[float]) -> float:
    """Maximum probability (optimistic)."""
    return max(probs)


def combine_min(probs: List[float]) -> float:
    """Minimum probability (pessimistic)."""
    return min(probs)


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_confidence_system(
    emb_dict: Dict[str, np.ndarray],
    train_gt: Dict[str, Set[str]],
    test_gt: Dict[str, Set[str]],
    disease_names: Dict[str, str],
    h65_model,
    h65_features: List[str],
    h52_model,
    h52_features: List[str],
) -> Dict:
    """Evaluate unified confidence scoring on test diseases."""
    train_disease_list = [d for d in train_gt if d in emb_dict]
    train_disease_embs = np.array([emb_dict[d] for d in train_disease_list], dtype=np.float32)

    results = []

    for disease_id in test_gt:
        if disease_id not in emb_dict:
            continue
        gt_drugs = {d for d in test_gt[disease_id] if d in emb_dict}
        if not gt_drugs:
            continue

        disease_name = disease_names.get(disease_id, disease_id)
        category = categorize_disease(disease_name)

        # Run kNN evaluation
        top_30, drug_scores = knn_evaluate_single_disease(
            disease_id, emb_dict, train_gt, train_disease_list, train_disease_embs, k=20
        )

        # Check if hit@30
        hit = 1 if len(top_30 & gt_drugs) > 0 else 0

        # Compute confidence signals
        h65_feats = compute_h65_features(
            disease_id, disease_name, emb_dict, train_gt,
            train_disease_list, train_disease_embs, gt_drugs, k=20
        )
        h52_feats = compute_h52_features(
            disease_id, disease_name, emb_dict, train_gt,
            train_disease_list, train_disease_embs, k=20
        )

        prob_h65 = predict_h65_prob(h65_feats, h65_model, h65_features)
        prob_h52 = predict_h52_prob(h52_feats, h52_model, h52_features)
        prob_cat = get_category_prior(category)

        # Compute combined scores
        probs = [prob_h65, prob_h52, prob_cat]

        results.append({
            'disease_id': disease_id,
            'disease_name': disease_name,
            'category': category,
            'hit': hit,
            'prob_h65': prob_h65,
            'prob_h52': prob_h52,
            'prob_category': prob_cat,
            'combined_avg': combine_simple_average(probs),
            'combined_harmonic': combine_harmonic_mean(probs),
            'combined_max': combine_max(probs),
            'combined_min': combine_min(probs),
            'pool_size': h65_feats['pool_size'],
            'neighbors_w_gt': h65_feats['neighbors_w_gt'],
        })

    return results


def compute_metrics(results: List[Dict], prob_column: str) -> Dict:
    """Compute precision/recall metrics for a probability column."""
    probs = np.array([r[prob_column] for r in results])
    hits = np.array([r['hit'] for r in results])

    if len(np.unique(hits)) < 2:
        return {'auc': 0, 'ap': 0, 'calibration': {}}

    auc = roc_auc_score(hits, probs)
    ap = average_precision_score(hits, probs)

    # Calibration
    thresholds = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
    calibration = {}
    for thresh in thresholds:
        mask = probs >= thresh
        if mask.sum() > 0:
            precision = hits[mask].mean()
            n = mask.sum()
            calibration[thresh] = {'precision': float(precision), 'n': int(n)}

    return {
        'auc': float(auc),
        'ap': float(ap),
        'calibration': calibration,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    start_time = time.time()

    print("=" * 70)
    print("h68: UNIFIED CONFIDENCE-WEIGHTED PREDICTIONS FOR PRODUCTION")
    print("=" * 70)
    print()

    # Load models
    print("Loading models...")
    h65_model = joblib.load(MODELS_DIR / "disease_success_predictor.pkl")
    h65_features = joblib.load(MODELS_DIR / "disease_success_features.pkl")
    print(f"  h65 model loaded: {len(h65_features)} features")

    with open(MODELS_DIR / "meta_confidence_model.pkl", 'rb') as f:
        h52_data = pickle.load(f)
    h52_model = h52_data['model']
    h52_features = h52_data['feature_names']
    print(f"  h52 model loaded: {len(h52_features)} features")

    # Load data
    print("\nLoading data...")
    emb_dict = load_node2vec_embeddings()
    name_to_drug_id, _ = load_drugbank_lookup()
    mesh_mappings = load_mesh_mappings_from_file()
    gt_pairs, disease_names = load_ground_truth(mesh_mappings, name_to_drug_id)
    print(f"  GT: {len(gt_pairs)} diseases, {sum(len(v) for v in gt_pairs.values())} pairs")

    # Multi-seed evaluation
    all_seed_results = []

    for seed in SEEDS:
        print(f"\n{'='*70}")
        print(f"SEED {seed}")
        print(f"{'='*70}")

        train_gt, test_gt = disease_level_split(
            gt_pairs, lambda d: d in emb_dict, test_fraction=0.2, seed=seed
        )
        print(f"  Train: {len(train_gt)} diseases, Test: {len(test_gt)} diseases")

        results = evaluate_confidence_system(
            emb_dict, train_gt, test_gt, disease_names,
            h65_model, h65_features, h52_model, h52_features
        )

        print(f"\n  Diseases evaluated: {len(results)}")
        print(f"  Overall hit rate: {np.mean([r['hit'] for r in results])*100:.1f}%")

        # Compute metrics for each signal
        prob_columns = ['prob_h65', 'prob_h52', 'prob_category',
                        'combined_avg', 'combined_harmonic', 'combined_max', 'combined_min']

        seed_metrics = {}
        for col in prob_columns:
            metrics = compute_metrics(results, col)
            seed_metrics[col] = metrics
            print(f"\n  {col}:")
            print(f"    AUC: {metrics['auc']:.3f}, AP: {metrics['ap']:.3f}")
            if metrics['calibration']:
                for thresh, cal in sorted(metrics['calibration'].items()):
                    print(f"    Threshold {thresh}: precision={cal['precision']:.1%} (n={cal['n']})")

        all_seed_results.append({
            'seed': seed,
            'results': results,
            'metrics': seed_metrics,
        })

    # Aggregate multi-seed results
    print("\n" + "=" * 70)
    print("MULTI-SEED SUMMARY")
    print("=" * 70)

    prob_columns = ['prob_h65', 'prob_h52', 'prob_category',
                    'combined_avg', 'combined_harmonic', 'combined_max', 'combined_min']

    summary = {}
    for col in prob_columns:
        aucs = [sr['metrics'][col]['auc'] for sr in all_seed_results]
        aps = [sr['metrics'][col]['ap'] for sr in all_seed_results]
        summary[col] = {
            'auc_mean': float(np.mean(aucs)),
            'auc_std': float(np.std(aucs)),
            'ap_mean': float(np.mean(aps)),
            'ap_std': float(np.std(aps)),
        }
        print(f"\n{col}:")
        print(f"  AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
        print(f"  AP:  {np.mean(aps):.3f} ± {np.std(aps):.3f}")

    # Aggregate calibration at key thresholds
    print("\n" + "=" * 70)
    print("CALIBRATION ANALYSIS (threshold=0.7)")
    print("=" * 70)

    for col in prob_columns:
        precisions = []
        ns = []
        for sr in all_seed_results:
            cal = sr['metrics'][col].get('calibration', {}).get(0.7, {})
            if cal:
                precisions.append(cal['precision'])
                ns.append(cal['n'])
        if precisions:
            print(f"\n{col}:")
            print(f"  Precision at 0.7: {np.mean(precisions):.1%} ± {np.std(precisions):.1%}")
            print(f"  Coverage: {np.mean(ns):.1f} diseases (avg)")

    # Save results
    elapsed = time.time() - start_time
    output = {
        'hypothesis': 'h68',
        'title': 'Unified Confidence-Weighted Predictions for Production',
        'date': '2026-01-31',
        'summary': summary,
        'all_seed_results': [
            {'seed': sr['seed'], 'metrics': sr['metrics']}
            for sr in all_seed_results
        ],
        'elapsed_seconds': elapsed,
    }

    output_path = ANALYSIS_DIR / "h68_unified_confidence_results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")
    print(f"Elapsed: {elapsed:.0f}s")

    # Production-ready model creation
    print("\n" + "=" * 70)
    print("PRODUCTION MODEL RECOMMENDATIONS")
    print("=" * 70)

    # Find best combination method
    best_method = max(summary.keys(), key=lambda k: summary[k]['ap_mean'])
    print(f"\nBest method by AP: {best_method}")
    print(f"  AP: {summary[best_method]['ap_mean']:.3f}")
    print(f"  AUC: {summary[best_method]['auc_mean']:.3f}")

    # Recommend thresholds for production tiering
    print("\nRecommended production tiers:")
    print("  HIGH confidence (prob >= 0.7): Surface predictions prominently")
    print("  MEDIUM confidence (0.5 <= prob < 0.7): Include with caveats")
    print("  LOW confidence (prob < 0.5): Flag as exploratory")


if __name__ == "__main__":
    main()
