#!/usr/bin/env python3
"""
h289: Why Does Class Uniqueness Hurt Precision?

PURPOSE:
    h193 showed class-unique predictions have LOWER precision (4.5-7.8%)
    than class-supported predictions (5.0-9.0%). Why?

    Hypotheses to test:
    1. Class-unique predictions have weaker kNN signal (lower scores)
    2. Class-unique drugs have lower training frequency (less data)
    3. Class-unique drugs are more mechanism-specific (narrow applicability)
    4. Class-unique correct predictions have different characteristics

SUCCESS CRITERIA:
    Identify distinguishing feature of unique correct vs incorrect predictions.
"""

import json
import sys
from collections import defaultdict, Counter
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


def load_atc_mapper():
    """Load ATCMapper for drug classification."""
    from atc_features import ATCMapper
    return ATCMapper()


def get_drug_atc_from_mapper(atc_mapper, drug_name: str) -> str:
    """Get ATC code for a drug using ATCMapper."""
    codes = atc_mapper.get_atc_codes(drug_name)
    if codes:
        return codes[0]  # Return first code
    return ""


def get_atc_class(atc_code: str) -> str:
    if atc_code and len(atc_code) >= 1:
        return atc_code[0]
    return "unknown"


def get_atc_l2(atc_code: str) -> str:
    """Get ATC level 2 (first 3 characters)."""
    if atc_code and len(atc_code) >= 3:
        return atc_code[:3]
    return "unknown"


def run_uniqueness_analysis(
    emb_dict, train_gt, test_gt, drug_atc, disease_names, id_to_name, k=20
) -> List[Dict]:
    """Run kNN and compute class uniqueness for each prediction."""
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

    # Build ATC L2 -> drugs mapping (finer-grained class)
    atc_l2_drugs: Dict[str, Set[str]] = defaultdict(set)
    for disease_id, drugs in train_gt.items():
        for drug_id in drugs:
            atc = drug_atc.get(drug_id, "")
            if atc:
                atc_l2 = get_atc_l2(atc)
                atc_l2_drugs[atc_l2].add(drug_id)

    # Build ATC -> diseases treated mapping
    atc_to_diseases: Dict[str, Set[str]] = defaultdict(set)
    for disease_id, drugs in train_gt.items():
        for drug_id in drugs:
            atc = drug_atc.get(drug_id, "")
            if atc:
                atc_class = get_atc_class(atc)
                atc_to_diseases[atc_class].add(disease_id)

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
        drug_neighbor_count: Dict[str, int] = defaultdict(int)  # How many neighbors have this drug
        for idx in top_k_idx:
            neighbor_disease = train_disease_list[idx]
            neighbor_sim = sims[idx]
            for drug_id in train_gt[neighbor_disease]:
                if drug_id in emb_dict:
                    drug_counts[drug_id] += neighbor_sim
                    drug_neighbor_count[drug_id] += 1

        if not drug_counts:
            continue

        sorted_drugs = sorted(drug_counts.items(), key=lambda x: x[1], reverse=True)[:30]
        max_score = sorted_drugs[0][1] if sorted_drugs else 1.0

        for rank, (drug_id, score) in enumerate(sorted_drugs, 1):
            atc = drug_atc.get(drug_id, "")
            atc_class = get_atc_class(atc) if atc else ""
            atc_l2 = get_atc_l2(atc) if atc else ""

            # Class uniqueness: is this the ONLY drug from its ATC class in neighbors?
            classmate_drugs = atc_class_drugs.get(atc_class, set())
            classmates_in_neighbors = classmate_drugs & neighbor_drugs
            n_classmates_in_neighbors = len(classmates_in_neighbors)
            is_class_unique = (n_classmates_in_neighbors <= 1)  # Only itself or none

            # L2 uniqueness (finer-grained)
            l2_classmate_drugs = atc_l2_drugs.get(atc_l2, set())
            l2_classmates_in_neighbors = l2_classmate_drugs & neighbor_drugs
            n_l2_classmates = len(l2_classmates_in_neighbors)

            # Training frequency
            train_freq = drug_train_freq.get(drug_id, 0)

            # Drug breadth (how many diseases does this drug treat in training?)
            drug_breadth = train_freq

            # Class breadth (how many diseases does this ATC class treat?)
            class_diseases = atc_to_diseases.get(atc_class, set())
            class_breadth = len(class_diseases)

            # Neighbor support (how many neighbors have this drug)
            neighbor_support = drug_neighbor_count.get(drug_id, 0)

            is_hit = drug_id in gt_drugs
            drug_name = id_to_name.get(drug_id, drug_id.split("::")[-1])

            results.append({
                'disease': disease_id,
                'drug': drug_id,
                'drug_name': drug_name,
                'disease_name': disease_name,
                'atc_class': atc_class,
                'atc_l2': atc_l2,
                'is_class_unique': is_class_unique,
                'n_classmates_in_neighbors': n_classmates_in_neighbors,
                'n_l2_classmates': n_l2_classmates,
                'train_frequency': train_freq,
                'drug_breadth': drug_breadth,
                'class_breadth': class_breadth,
                'neighbor_support': neighbor_support,
                'knn_score': score,
                'norm_score': score / max_score if max_score > 0 else 0,
                'rank': rank,
                'is_hit': 1 if is_hit else 0,
            })

    return results


def main():
    print("h289: Why Does Class Uniqueness Hurt Precision?")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    emb_dict = load_node2vec_embeddings()
    name_to_drug_id, id_to_name = load_drugbank_lookup()
    ground_truth, disease_names = load_ground_truth(name_to_drug_id)

    # Build drug_atc mapping using ATCMapper
    atc_mapper = load_atc_mapper()
    drug_atc = {}
    for drug_id, drug_name in id_to_name.items():
        atc_code = get_drug_atc_from_mapper(atc_mapper, drug_name)
        if atc_code:
            drug_atc[drug_id] = atc_code

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

        seed_results = run_uniqueness_analysis(
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
    print("VERIFY: Class Unique vs Not Unique Precision")
    print("=" * 70)

    print(f"\nClass UNIQUE: {len(unique)} predictions ({len(unique)/len(has_atc)*100:.1f}%)")
    print(f"Class NOT UNIQUE: {len(not_unique)} predictions ({len(not_unique)/len(has_atc)*100:.1f}%)")

    unique_prec = unique['is_hit'].mean() * 100
    not_unique_prec = not_unique['is_hit'].mean() * 100
    print(f"\nUNIQUE precision: {unique_prec:.2f}%")
    print(f"NOT UNIQUE precision: {not_unique_prec:.2f}%")
    print(f"Difference (unique - not unique): {unique_prec - not_unique_prec:.2f} pp")

    # === HYPOTHESIS 1: kNN Score ===
    print("\n" + "=" * 70)
    print("HYPOTHESIS 1: Do unique predictions have weaker kNN signal?")
    print("=" * 70)

    print(f"\nMean kNN score:")
    print(f"  UNIQUE: {unique['norm_score'].mean():.4f}")
    print(f"  NOT UNIQUE: {not_unique['norm_score'].mean():.4f}")

    unique_hits = unique[unique['is_hit'] == 1]
    not_unique_hits = not_unique[not_unique['is_hit'] == 1]
    unique_misses = unique[unique['is_hit'] == 0]
    not_unique_misses = not_unique[not_unique['is_hit'] == 0]

    print(f"\nMean kNN score (HITS ONLY):")
    print(f"  UNIQUE hits: {unique_hits['norm_score'].mean():.4f} (N={len(unique_hits)})")
    print(f"  NOT UNIQUE hits: {not_unique_hits['norm_score'].mean():.4f} (N={len(not_unique_hits)})")

    # === HYPOTHESIS 2: Training Frequency ===
    print("\n" + "=" * 70)
    print("HYPOTHESIS 2: Do unique drugs have lower training frequency?")
    print("=" * 70)

    print(f"\nMean train_frequency:")
    print(f"  UNIQUE: {unique['train_frequency'].mean():.2f}")
    print(f"  NOT UNIQUE: {not_unique['train_frequency'].mean():.2f}")

    print(f"\nMean train_frequency (HITS ONLY):")
    print(f"  UNIQUE hits: {unique_hits['train_frequency'].mean():.2f}")
    print(f"  NOT UNIQUE hits: {not_unique_hits['train_frequency'].mean():.2f}")

    # === HYPOTHESIS 3: Neighbor Support ===
    print("\n" + "=" * 70)
    print("HYPOTHESIS 3: Do unique predictions have lower neighbor support?")
    print("=" * 70)

    print(f"\nMean neighbor_support (# of kNN neighbors with this drug):")
    print(f"  UNIQUE: {unique['neighbor_support'].mean():.2f}")
    print(f"  NOT UNIQUE: {not_unique['neighbor_support'].mean():.2f}")

    print(f"\nMean neighbor_support (HITS ONLY):")
    print(f"  UNIQUE hits: {unique_hits['neighbor_support'].mean():.2f}")
    print(f"  NOT UNIQUE hits: {not_unique_hits['neighbor_support'].mean():.2f}")

    # === HYPOTHESIS 4: Class Breadth ===
    print("\n" + "=" * 70)
    print("HYPOTHESIS 4: Do unique drugs have narrower class breadth?")
    print("=" * 70)

    print(f"\nMean class_breadth (# diseases treated by ATC class):")
    print(f"  UNIQUE: {unique['class_breadth'].mean():.1f}")
    print(f"  NOT UNIQUE: {not_unique['class_breadth'].mean():.1f}")

    # === ANALYZE UNIQUE CORRECT vs UNIQUE INCORRECT ===
    print("\n" + "=" * 70)
    print("KEY QUESTION: What distinguishes unique CORRECT vs INCORRECT?")
    print("=" * 70)

    print(f"\nUnique correct (N={len(unique_hits)}):")
    for col in ['norm_score', 'train_frequency', 'neighbor_support', 'class_breadth', 'rank']:
        print(f"  Mean {col}: {unique_hits[col].mean():.3f}")

    print(f"\nUnique incorrect (N={len(unique_misses)}):")
    for col in ['norm_score', 'train_frequency', 'neighbor_support', 'class_breadth', 'rank']:
        print(f"  Mean {col}: {unique_misses[col].mean():.3f}")

    # Calculate delta
    print("\nDelta (correct - incorrect):")
    for col in ['norm_score', 'train_frequency', 'neighbor_support', 'class_breadth', 'rank']:
        delta = unique_hits[col].mean() - unique_misses[col].mean()
        print(f"  {col}: {delta:+.3f}")

    # === Classmates comparison ===
    print("\n" + "=" * 70)
    print("CLASSMATE EFFECT: # of classmates in neighbors")
    print("=" * 70)

    # Bin by number of classmates
    for n_classmates in [0, 1, 2, 3, 4, 5]:
        if n_classmates < 5:
            subset = has_atc[has_atc['n_classmates_in_neighbors'] == n_classmates]
        else:
            subset = has_atc[has_atc['n_classmates_in_neighbors'] >= 5]
        if len(subset) > 50:
            prec = subset['is_hit'].mean() * 100
            print(f"  {n_classmates}{'+'if n_classmates>=5 else ''} classmates: {prec:.1f}% precision (N={len(subset)})")

    # === Sample unique correct predictions ===
    print("\n" + "=" * 70)
    print("SAMPLE UNIQUE CORRECT PREDICTIONS (best rank first)")
    print("=" * 70)

    unique_hits_sorted = unique_hits.sort_values('rank').head(20)
    print(f"\n{'Drug':<25} {'ATC':>5} {'Rank':>5} {'Freq':>5} {'Support':>7} {'Disease':<30}")
    print("-" * 85)
    for _, row in unique_hits_sorted.iterrows():
        print(f"{row['drug_name'][:24]:<25} {row['atc_class']:>5} {row['rank']:>5} {row['train_frequency']:>5} {row['neighbor_support']:>7} {row['disease_name'][:29]:<30}")

    # === Summary ===
    print("\n" + "=" * 70)
    print("SUMMARY: Why Class Uniqueness Hurts Precision")
    print("=" * 70)

    findings = []

    # Check hypotheses
    if unique['norm_score'].mean() < not_unique['norm_score'].mean():
        diff = not_unique['norm_score'].mean() - unique['norm_score'].mean()
        findings.append(f"WEAK SIGNAL: Unique predictions have -{diff:.3f} lower kNN scores")

    if unique['train_frequency'].mean() < not_unique['train_frequency'].mean():
        diff = not_unique['train_frequency'].mean() - unique['train_frequency'].mean()
        findings.append(f"LOW FREQUENCY: Unique drugs have -{diff:.1f} lower training frequency")

    if unique['neighbor_support'].mean() < not_unique['neighbor_support'].mean():
        diff = not_unique['neighbor_support'].mean() - unique['neighbor_support'].mean()
        findings.append(f"LOW SUPPORT: Unique drugs have -{diff:.2f} fewer supporting neighbors")

    if unique['class_breadth'].mean() < not_unique['class_breadth'].mean():
        diff = not_unique['class_breadth'].mean() - unique['class_breadth'].mean()
        findings.append(f"NARROW CLASS: Unique drugs come from narrower ATC classes ({-diff:.0f} fewer diseases)")

    if findings:
        print("\nCONFIRMED EXPLANATIONS:")
        for f in findings:
            print(f"  ✓ {f}")

    # Key insight
    score_diff_unique = unique_hits['norm_score'].mean() - unique_misses['norm_score'].mean()
    score_diff_not_unique = not_unique_hits['norm_score'].mean() - not_unique_misses['norm_score'].mean()

    print(f"\nKEY FINDING:")
    print(f"  Score gap (hits - misses):")
    print(f"    UNIQUE: {score_diff_unique:.4f}")
    print(f"    NOT UNIQUE: {score_diff_not_unique:.4f}")

    if score_diff_unique < score_diff_not_unique:
        print("  → Unique predictions have LESS discriminative scores (harder to separate hits from misses)")

    # Save results
    results = {
        'hypothesis': 'h289',
        'unique_precision': float(unique_prec),
        'not_unique_precision': float(not_unique_prec),
        'precision_gap': float(unique_prec - not_unique_prec),
        'n_unique': int(len(unique)),
        'n_not_unique': int(len(not_unique)),
        'n_unique_hits': int(len(unique_hits)),
        'n_not_unique_hits': int(len(not_unique_hits)),
        'unique_mean_score': float(unique['norm_score'].mean()),
        'not_unique_mean_score': float(not_unique['norm_score'].mean()),
        'unique_mean_train_freq': float(unique['train_frequency'].mean()),
        'not_unique_mean_train_freq': float(not_unique['train_frequency'].mean()),
        'unique_mean_neighbor_support': float(unique['neighbor_support'].mean()),
        'not_unique_mean_neighbor_support': float(not_unique['neighbor_support'].mean()),
        'unique_mean_class_breadth': float(unique['class_breadth'].mean()),
        'not_unique_mean_class_breadth': float(not_unique['class_breadth'].mean()),
        'unique_hits_mean_score': float(unique_hits['norm_score'].mean()) if len(unique_hits) > 0 else None,
        'not_unique_hits_mean_score': float(not_unique_hits['norm_score'].mean()) if len(not_unique_hits) > 0 else None,
        'findings': findings,
    }

    results_file = ANALYSIS_DIR / "h289_class_uniqueness_analysis.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
