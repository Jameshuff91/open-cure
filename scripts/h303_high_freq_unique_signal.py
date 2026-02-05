#!/usr/bin/env python3
"""
h303: High-Frequency Unique Drug Signal

PURPOSE:
    h289 found unique correct predictions have 25.02 avg training frequency
    vs 8.34 for incorrect. Test if high-frequency unique predictions can be
    boosted.

SUCCESS CRITERIA:
    High-frequency unique (train_freq > 20) has significantly higher precision
    than general unique predictions. If >10%, consider tier boost.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from disease_name_matcher import DiseaseMatcher, load_mesh_mappings
from atc_features import ATCMapper

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"

SEEDS = [42, 123, 456, 789, 1024]


def load_node2vec_embeddings() -> Dict[str, np.ndarray]:
    embeddings_path = EMBEDDINGS_DIR / "node2vec_256_named.csv"
    df = pd.read_csv(embeddings_path)
    dim_cols = [c for c in df.columns if c.startswith("dim_")]
    embeddings: Dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        entity = f"drkg:{row['entity']}"
        embeddings[entity] = row[dim_cols].values.astype(np.float32)
    return embeddings


def load_drugbank_lookup():
    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)
    name_to_id = {name.lower(): f"drkg:Compound::{db_id}" for db_id, name in id_to_name.items()}
    id_to_name_full = {f"drkg:Compound::{db_id}": name for db_id, name in id_to_name.items()}
    return name_to_id, id_to_name_full


def load_ground_truth(name_to_drug_id):
    mesh_mappings = load_mesh_mappings()
    matcher = DiseaseMatcher(mesh_mappings)

    df = pd.read_excel(REFERENCE_DIR / "everycure" / "indicationList.xlsx")

    gt: Dict[str, Set[str]] = defaultdict(set)
    disease_names: Dict[str, str] = {}

    for _, row in df.iterrows():
        disease = str(row.get("disease name", "")).strip()
        drug = str(row.get("final normalized drug label", "")).strip()
        if not disease or not drug:
            continue

        disease_id = matcher.get_mesh_id(disease)
        if not disease_id:
            continue

        disease_names[disease_id] = disease
        drug_id = name_to_drug_id.get(drug.lower())
        if drug_id:
            gt[disease_id].add(drug_id)

    return dict(gt), disease_names


def get_atc_class(atc_code: str) -> str:
    if atc_code and len(atc_code) >= 1:
        return atc_code[0]
    return "unknown"


def run_analysis(
    emb_dict, train_gt, test_gt, drug_atc, disease_names, id_to_name, k=20
) -> List[Dict]:
    """Run kNN and compute class uniqueness and frequency for each prediction."""
    train_disease_list = [d for d in train_gt if d in emb_dict]
    train_disease_embs = np.array([emb_dict[d] for d in train_disease_list], dtype=np.float32)

    # Drug training frequency
    drug_train_freq: Dict[str, int] = defaultdict(int)
    for disease_id, drugs in train_gt.items():
        for drug_id in drugs:
            drug_train_freq[drug_id] += 1

    # Build ATC class -> drugs mapping from training data
    atc_class_drugs: Dict[str, Set[str]] = defaultdict(set)
    for disease_id, drugs in train_gt.items():
        for drug_id in drugs:
            atc = drug_atc.get(drug_id, "")
            if atc:
                atc_class = get_atc_class(atc)
                atc_class_drugs[atc_class].add(drug_id)

    results = []

    for disease_id in test_gt:
        if disease_id not in emb_dict:
            continue
        gt_drugs = {d for d in test_gt[disease_id] if d in emb_dict}
        if not gt_drugs:
            continue

        disease_name = disease_names.get(disease_id, "")

        # Find k nearest training diseases
        test_emb = emb_dict[disease_id].reshape(1, -1)
        sims = cosine_similarity(test_emb, train_disease_embs)[0]
        top_k_idx = np.argsort(sims)[-k:]
        neighbor_diseases = set(train_disease_list[idx] for idx in top_k_idx)

        # Get drugs from neighbors
        neighbor_drugs = set()
        for neighbor in neighbor_diseases:
            neighbor_drugs.update(train_gt[neighbor])

        # Count drug frequency from neighbors (with weighting)
        drug_counts: Dict[str, float] = defaultdict(float)
        for idx in top_k_idx:
            neighbor_disease = train_disease_list[idx]
            neighbor_sim = sims[idx]
            for drug_id in train_gt[neighbor_disease]:
                if drug_id in emb_dict:
                    drug_counts[drug_id] += neighbor_sim

        if not drug_counts:
            continue

        sorted_drugs = sorted(drug_counts.items(), key=lambda x: x[1], reverse=True)[:30]
        max_score = sorted_drugs[0][1] if sorted_drugs else 1.0

        for rank, (drug_id, score) in enumerate(sorted_drugs, 1):
            atc = drug_atc.get(drug_id, "")
            atc_class = get_atc_class(atc) if atc else ""

            # Class uniqueness: is this the ONLY drug from its ATC class in neighbors?
            classmate_drugs = atc_class_drugs.get(atc_class, set())
            classmates_in_neighbors = classmate_drugs & neighbor_drugs
            n_classmates_in_neighbors = len(classmates_in_neighbors)
            is_class_unique = (n_classmates_in_neighbors <= 1)

            # Training frequency
            train_freq = drug_train_freq.get(drug_id, 0)

            is_hit = drug_id in gt_drugs
            drug_name = id_to_name.get(drug_id, drug_id.split("::")[-1])

            results.append({
                'disease': disease_id,
                'drug': drug_id,
                'drug_name': drug_name,
                'disease_name': disease_name,
                'atc_class': atc_class,
                'is_class_unique': is_class_unique,
                'n_classmates': n_classmates_in_neighbors,
                'train_frequency': train_freq,
                'knn_score': score,
                'norm_score': score / max_score if max_score > 0 else 0,
                'rank': rank,
                'is_hit': 1 if is_hit else 0,
            })

    return results


def main():
    print("h303: High-Frequency Unique Drug Signal")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    emb_dict = load_node2vec_embeddings()
    name_to_drug_id, id_to_name = load_drugbank_lookup()
    ground_truth, disease_names = load_ground_truth(name_to_drug_id)

    # Build drug_atc mapping using ATCMapper
    atc_mapper = ATCMapper()
    drug_atc = {}
    for drug_id, drug_name in id_to_name.items():
        codes = atc_mapper.get_atc_codes(drug_name)
        if codes:
            drug_atc[drug_id] = codes[0]

    print(f"  Embeddings: {len(emb_dict)}")
    print(f"  Ground truth diseases: {len(ground_truth)}")
    print(f"  Drugs with ATC codes: {len(drug_atc)}")

    # Collect predictions across seeds
    print("\n" + "=" * 70)
    print("Collecting predictions across 5 seeds")
    print("=" * 70)

    all_results = []

    for seed in SEEDS:
        np.random.seed(seed)
        diseases = list(ground_truth.keys())
        np.random.shuffle(diseases)
        n_test = len(diseases) // 5
        test_diseases = set(diseases[:n_test])
        train_diseases = set(diseases[n_test:])

        train_gt = {d: ground_truth[d] for d in train_diseases}
        test_gt = {d: ground_truth[d] for d in test_diseases}

        seed_results = run_analysis(
            emb_dict, train_gt, test_gt, drug_atc, disease_names, id_to_name, k=20
        )
        all_results.extend(seed_results)
        print(f"  Seed {seed}: {len(seed_results)} predictions")

    df = pd.DataFrame(all_results)
    print(f"\nTotal predictions: {len(df)}")
    print(f"Base hit rate: {df['is_hit'].mean()*100:.2f}%")

    # Filter to drugs with ATC
    has_atc = df[df['atc_class'] != ''].copy()
    print(f"With ATC data: {len(has_atc)} predictions")

    # Split by uniqueness
    unique = has_atc[has_atc['is_class_unique']]
    not_unique = has_atc[~has_atc['is_class_unique']]

    print(f"\n{'='*70}")
    print("BASELINE: Unique vs Not Unique Precision")
    print("=" * 70)

    unique_prec = unique['is_hit'].mean() * 100
    not_unique_prec = not_unique['is_hit'].mean() * 100
    print(f"\nUNIQUE: {len(unique)} predictions, {unique_prec:.2f}% precision")
    print(f"NOT UNIQUE: {len(not_unique)} predictions, {not_unique_prec:.2f}% precision")

    # Split unique by hits/misses
    unique_hits = unique[unique['is_hit'] == 1]
    unique_misses = unique[unique['is_hit'] == 0]

    print(f"\n{'='*70}")
    print("VERIFY: Train frequency difference (hits vs misses)")
    print("=" * 70)

    print(f"\nUnique CORRECT: avg train_freq = {unique_hits['train_frequency'].mean():.2f}")
    print(f"Unique INCORRECT: avg train_freq = {unique_misses['train_frequency'].mean():.2f}")
    print(f"Gap: {unique_hits['train_frequency'].mean() - unique_misses['train_frequency'].mean():.2f}")

    # === MAIN TEST: High-frequency unique drugs ===
    print(f"\n{'='*70}")
    print("MAIN TEST: High-Frequency Unique Predictions")
    print("=" * 70)

    # Test different thresholds
    thresholds = [5, 10, 15, 20, 25, 30, 40, 50]

    print(f"\n{'Threshold':<12} {'N Preds':>10} {'N Hits':>10} {'Precision':>12} {'Lift vs Baseline':>18}")
    print("-" * 65)

    results_by_threshold = []

    for thresh in thresholds:
        high_freq_unique = unique[unique['train_frequency'] >= thresh]
        if len(high_freq_unique) > 0:
            hits = high_freq_unique['is_hit'].sum()
            prec = high_freq_unique['is_hit'].mean() * 100
            lift = prec - unique_prec
            print(f"freq >= {thresh:<5} {len(high_freq_unique):>10} {int(hits):>10} {prec:>11.2f}% {lift:>+16.2f} pp")
            results_by_threshold.append({
                'threshold': thresh,
                'n_predictions': len(high_freq_unique),
                'n_hits': int(hits),
                'precision': prec,
                'lift_vs_baseline': lift,
            })
        else:
            print(f"freq >= {thresh:<5} {'N/A':>10} {'N/A':>10} {'N/A':>12} {'N/A':>18}")

    # === Inverse: Low-frequency unique ===
    print(f"\n{'='*70}")
    print("INVERSE: Low-Frequency Unique Predictions (sanity check)")
    print("=" * 70)

    for thresh in [2, 5, 10]:
        low_freq_unique = unique[unique['train_frequency'] <= thresh]
        if len(low_freq_unique) > 0:
            hits = low_freq_unique['is_hit'].sum()
            prec = low_freq_unique['is_hit'].mean() * 100
            lift = prec - unique_prec
            print(f"freq <= {thresh}: {len(low_freq_unique)} predictions, {prec:.2f}% precision ({lift:+.2f} pp vs baseline)")

    # === Compare with NOT unique at same threshold ===
    print(f"\n{'='*70}")
    print("COMPARISON: Same threshold for NOT unique drugs")
    print("=" * 70)

    print(f"\nBaseline NOT unique precision: {not_unique_prec:.2f}%")

    for thresh in [20, 25, 30]:
        high_freq_not_unique = not_unique[not_unique['train_frequency'] >= thresh]
        if len(high_freq_not_unique) > 0:
            prec = high_freq_not_unique['is_hit'].mean() * 100
            print(f"NOT unique freq >= {thresh}: {len(high_freq_not_unique)} predictions, {prec:.2f}% precision")

    # === Best unique hits analysis ===
    print(f"\n{'='*70}")
    print("TOP UNIQUE HITS: What drugs are unique AND correct?")
    print("=" * 70)

    top_unique_hits = unique_hits.sort_values(['train_frequency', 'rank'], ascending=[False, True]).head(20)

    print(f"\n{'Drug':<25} {'ATC':>5} {'Freq':>6} {'Rank':>5} {'Score':>7} {'Disease':<25}")
    print("-" * 80)

    for _, row in top_unique_hits.iterrows():
        print(f"{row['drug_name'][:24]:<25} {row['atc_class']:>5} {row['train_frequency']:>6} {row['rank']:>5} {row['norm_score']:>7.3f} {row['disease_name'][:24]:<25}")

    # === Analyze top drugs ===
    print(f"\n{'='*70}")
    print("DRUG CONCENTRATION: Which drugs dominate unique hits?")
    print("=" * 70)

    drug_hit_counts = unique_hits.groupby('drug_name').agg({
        'is_hit': 'sum',
        'train_frequency': 'first',
        'atc_class': 'first',
    }).reset_index()
    drug_hit_counts.columns = ['drug', 'hits', 'train_freq', 'atc']
    drug_hit_counts = drug_hit_counts.sort_values('hits', ascending=False).head(15)

    print(f"\n{'Drug':<30} {'Hits':>6} {'Freq':>6} {'ATC':>5}")
    print("-" * 50)
    for _, row in drug_hit_counts.iterrows():
        print(f"{row['drug'][:29]:<30} {int(row['hits']):>6} {int(row['train_freq']):>6} {row['atc']:>5}")

    # === Summary ===
    print(f"\n{'='*70}")
    print("SUMMARY: h303 Findings")
    print("=" * 70)

    # Find best threshold with meaningful lift
    best_thresh = None
    best_lift = 0
    best_prec = 0
    for r in results_by_threshold:
        if r['n_predictions'] >= 20 and r['lift_vs_baseline'] > best_lift:
            best_thresh = r['threshold']
            best_lift = r['lift_vs_baseline']
            best_prec = r['precision']

    if best_thresh:
        print(f"\nBest threshold: train_freq >= {best_thresh}")
        print(f"  Precision: {best_prec:.2f}%")
        print(f"  Lift vs unique baseline: {best_lift:+.2f} pp")

    # Conclusion
    print("\nCONCLUSION:")
    if best_prec > 10:
        print(f"  ✓ High-frequency unique predictions (freq >= {best_thresh}) achieve {best_prec:.1f}% precision")
        print(f"  ✓ This is {best_lift:+.1f} pp better than general unique predictions ({unique_prec:.1f}%)")
        print(f"  → CONSIDER: Add tier boost for high-frequency unique predictions")
    else:
        print("  ✗ High-frequency unique predictions do not achieve >10% precision")
        print("  → NOT RECOMMENDED: Tier boost would not be valuable")

    # Save results
    results = {
        'hypothesis': 'h303',
        'baseline_unique_precision': float(unique_prec),
        'baseline_not_unique_precision': float(not_unique_prec),
        'n_unique': int(len(unique)),
        'n_unique_hits': int(len(unique_hits)),
        'unique_hits_avg_train_freq': float(unique_hits['train_frequency'].mean()),
        'unique_misses_avg_train_freq': float(unique_misses['train_frequency'].mean()),
        'train_freq_gap': float(unique_hits['train_frequency'].mean() - unique_misses['train_frequency'].mean()),
        'results_by_threshold': results_by_threshold,
        'best_threshold': best_thresh,
        'best_precision': best_prec,
        'best_lift': best_lift,
    }

    results_file = ANALYSIS_DIR / "h303_high_freq_unique_signal.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
