#!/usr/bin/env python3
"""
h104: Confidence Feature - Drug Class Coherence

PURPOSE:
    If a drug belongs to a class (ATC) where multiple members treat the same
    disease category, that drug is more likely a true positive.
    "Drug class coherence" = fraction of similar drugs that treat similar diseases.

APPROACH:
    1. For each kNN top-30 prediction, get drug's ATC class (level 3 = pharmacological subgroup)
    2. Count how many OTHER drugs in same ATC class treat diseases similar to target
    3. Compute coherence_score = count / total drugs in class
    4. Compare precision: high-coherence vs low-coherence predictions

SUCCESS CRITERIA:
    High-coherence predictions have 5%+ better precision.
"""

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, List

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from disease_name_matcher import DiseaseMatcher, load_mesh_mappings

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
GRAPHS_DIR = PROJECT_ROOT / "data" / "graphs"

SEEDS = [42, 123, 456, 789, 1024]


def load_node2vec_embeddings() -> Dict[str, np.ndarray]:
    """Load Node2Vec embeddings."""
    embeddings_path = EMBEDDINGS_DIR / "node2vec_256_named.csv"
    df = pd.read_csv(embeddings_path)
    dim_cols = [c for c in df.columns if c.startswith("dim_")]
    embeddings: Dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        entity = f"drkg:{row['entity']}"
        embeddings[entity] = row[dim_cols].values.astype(np.float32)
    return embeddings


def load_drugbank_lookup() -> Dict[str, str]:
    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)
    name_to_id = {name.lower(): f"drkg:Compound::{db_id}" for db_id, name in id_to_name.items()}
    return name_to_id


def load_mesh_mappings_from_file() -> Dict[str, str]:
    mesh_path = REFERENCE_DIR / "mesh_mappings_from_agents.json"
    if not mesh_path.exists():
        return {}
    with open(mesh_path) as f:
        mesh_data = json.load(f)
    mesh_mappings: Dict[str, str] = {}
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
) -> Dict[str, Set[str]]:
    df = pd.read_excel(REFERENCE_DIR / "everycure" / "indicationList.xlsx")
    fuzzy_mappings = load_mesh_mappings()
    matcher = DiseaseMatcher(fuzzy_mappings)

    gt: Dict[str, Set[str]] = defaultdict(set)
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
            gt[disease_id].add(drug_id)

    return dict(gt)


def load_atc_mappings() -> Dict[str, Set[str]]:
    """Load drug -> ATC class mappings from graph."""
    with open(GRAPHS_DIR / "unified_graph.gpickle", 'rb') as f:
        G = pickle.load(f)

    drug_to_atc = defaultdict(set)
    for u, v, data in G.edges(data=True):
        rel = data.get('relation', data.get('type', ''))
        if 'atc' in rel.lower() or 'ATC' in str(data).lower():
            if 'Compound' in u:
                drug = u.split('::')[-1] if '::' in u else u
                # Get ATC level 3 (e.g., J05AX from J05AX06)
                atc_full = v.split('::')[-1] if '::' in v else v
                if len(atc_full) >= 4:  # Level 3 is 4-5 chars
                    atc_l3 = atc_full[:4] if len(atc_full) == 4 else atc_full[:5]
                    drug_to_atc[drug].add(atc_l3)
            elif 'Compound' in v:
                drug = v.split('::')[-1] if '::' in v else v
                atc_full = u.split('::')[-1] if '::' in u else u
                if len(atc_full) >= 4:
                    atc_l3 = atc_full[:4] if len(atc_full) == 4 else atc_full[:5]
                    drug_to_atc[drug].add(atc_l3)

    return dict(drug_to_atc)


def build_atc_to_drugs(drug_to_atc: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    """Build reverse mapping: ATC class -> drugs in that class."""
    atc_to_drugs = defaultdict(set)
    for drug, atc_classes in drug_to_atc.items():
        for atc in atc_classes:
            atc_to_drugs[atc].add(drug)
    return dict(atc_to_drugs)


def compute_drug_coherence(
    drug_id: str,
    target_disease: str,
    drug_to_atc: Dict[str, Set[str]],
    atc_to_drugs: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
    emb_dict: Dict[str, np.ndarray],
    similarity_threshold: float = 0.7,
) -> float:
    """
    Compute coherence: what fraction of drugs in same ATC class treat similar diseases?

    Returns coherence score (0 to 1).
    """
    drug_core = drug_id.split('::')[-1] if '::' in drug_id else drug_id

    if drug_core not in drug_to_atc:
        return -1  # No ATC data

    # Get all drugs in same ATC class(es)
    class_drugs = set()
    for atc_class in drug_to_atc[drug_core]:
        class_drugs.update(atc_to_drugs.get(atc_class, set()))

    if len(class_drugs) <= 1:
        return -1  # Only this drug in class

    # Remove the drug itself
    class_drugs.discard(drug_core)

    # Count how many class drugs treat diseases similar to target
    if target_disease not in emb_dict:
        return -1

    target_emb = emb_dict[target_disease].reshape(1, -1)

    drugs_treating_similar = 0
    for class_drug in class_drugs:
        class_drug_drkg = f"drkg:Compound::{class_drug}"
        # Get diseases this class drug treats (from GT)
        for disease, drugs in ground_truth.items():
            if class_drug_drkg in drugs and disease in emb_dict:
                # Check if this disease is similar to target
                disease_emb = emb_dict[disease].reshape(1, -1)
                sim = cosine_similarity(target_emb, disease_emb)[0, 0]
                if sim >= similarity_threshold:
                    drugs_treating_similar += 1
                    break

    coherence = drugs_treating_similar / len(class_drugs)
    return coherence


def run_knn_with_coherence(
    emb_dict: Dict[str, np.ndarray],
    train_gt: Dict[str, Set[str]],
    test_gt: Dict[str, Set[str]],
    drug_to_atc: Dict[str, Set[str]],
    atc_to_drugs: Dict[str, Set[str]],
    k: int = 20,
) -> List[Dict]:
    """Run kNN and compute coherence for each prediction."""
    train_disease_list = [d for d in train_gt if d in emb_dict]
    train_disease_embs = np.array([emb_dict[d] for d in train_disease_list], dtype=np.float32)

    results = []

    for disease_id in test_gt:
        if disease_id not in emb_dict:
            continue
        gt_drugs = {d for d in test_gt[disease_id] if d in emb_dict}
        if not gt_drugs:
            continue

        # Find k nearest training diseases
        test_emb = emb_dict[disease_id].reshape(1, -1)
        sims = cosine_similarity(test_emb, train_disease_embs)[0]
        top_k_idx = np.argsort(sims)[-k:]

        # Count drug frequency
        drug_counts: Dict[str, float] = defaultdict(float)
        for idx in top_k_idx:
            neighbor_disease = train_disease_list[idx]
            neighbor_sim = sims[idx]
            for drug_id in train_gt[neighbor_disease]:
                if drug_id in emb_dict:
                    drug_counts[drug_id] += neighbor_sim

        if not drug_counts:
            continue

        # Get top 30
        sorted_drugs = sorted(drug_counts.items(), key=lambda x: x[1], reverse=True)[:30]

        for drug_id, score in sorted_drugs:
            coherence = compute_drug_coherence(
                drug_id, disease_id, drug_to_atc, atc_to_drugs, train_gt, emb_dict
            )

            is_hit = drug_id in gt_drugs

            results.append({
                'disease': disease_id,
                'drug': drug_id,
                'coherence': coherence,
                'is_hit': is_hit,
            })

    return results


def main():
    print("h104: Drug Class Coherence as Confidence Feature")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    emb_dict = load_node2vec_embeddings()
    name_to_drug_id = load_drugbank_lookup()
    mesh_mappings = load_mesh_mappings_from_file()
    ground_truth = load_ground_truth(mesh_mappings, name_to_drug_id)
    drug_to_atc = load_atc_mappings()
    atc_to_drugs = build_atc_to_drugs(drug_to_atc)

    print(f"  Embeddings: {len(emb_dict)}")
    print(f"  Ground truth diseases: {len(ground_truth)}")
    print(f"  Drugs with ATC codes: {len(drug_to_atc)}")
    print(f"  ATC classes: {len(atc_to_drugs)}")

    # Multi-seed evaluation
    print("\n" + "=" * 70)
    print("Multi-Seed Evaluation (5 seeds)")
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

        seed_results = run_knn_with_coherence(
            emb_dict, train_gt, test_gt, drug_to_atc, atc_to_drugs, k=20
        )
        all_results.extend(seed_results)
        print(f"  Seed {seed}: {len(seed_results)} predictions")

    # Analyze by coherence levels
    print("\n" + "=" * 70)
    print("Precision by Coherence Level")
    print("=" * 70)

    # Filter out predictions without ATC data
    with_atc = [r for r in all_results if r['coherence'] >= 0]
    without_atc = [r for r in all_results if r['coherence'] < 0]

    print(f"\nPredictions with ATC data: {len(with_atc)} ({100*len(with_atc)/len(all_results):.1f}%)")
    print(f"Predictions without ATC data: {len(without_atc)}")

    if with_atc:
        # Stratify by coherence
        coherences = [r['coherence'] for r in with_atc]
        hits = [r['is_hit'] for r in with_atc]

        # Tertiles
        sorted_results = sorted(with_atc, key=lambda x: x['coherence'])
        n = len(sorted_results)
        low = sorted_results[:n//3]
        medium = sorted_results[n//3:2*n//3]
        high = sorted_results[2*n//3:]

        def calc_precision(group):
            return sum(1 for r in group if r['is_hit']) / len(group) if group else 0

        low_prec = calc_precision(low)
        med_prec = calc_precision(medium)
        high_prec = calc_precision(high)

        print(f"\nLOW coherence ({n//3} predictions, mean {np.mean([r['coherence'] for r in low]):.3f}):")
        print(f"  Precision: {100*low_prec:.2f}%")

        print(f"\nMEDIUM coherence ({n//3} predictions, mean {np.mean([r['coherence'] for r in medium]):.3f}):")
        print(f"  Precision: {100*med_prec:.2f}%")

        print(f"\nHIGH coherence ({len(high)} predictions, mean {np.mean([r['coherence'] for r in high]):.3f}):")
        print(f"  Precision: {100*high_prec:.2f}%")

        # Key comparison
        print("\n" + "=" * 70)
        print("KEY COMPARISON")
        print("=" * 70)
        diff = high_prec - low_prec
        print(f"  HIGH coherence precision: {100*high_prec:.2f}%")
        print(f"  LOW coherence precision:  {100*low_prec:.2f}%")
        print(f"  Difference: {100*diff:+.2f} pp")

        # Success criteria
        success = diff >= 0.05
        print("\n" + "=" * 70)
        print("SUCCESS CRITERIA CHECK")
        print("=" * 70)
        if success:
            print(f"  ✓ High-coherence precision is +{100*diff:.1f} pp better (>= 5 pp)")
            print("  → VALIDATED: Drug class coherence is a valid confidence signal")
        else:
            print(f"  ✗ High-coherence precision is only +{100*diff:.1f} pp better (< 5 pp)")
            if diff > 0:
                print("  → PARTIALLY VALIDATED: Some improvement but below threshold")
            else:
                print("  → INVALIDATED: Drug class coherence doesn't improve precision")

        # Correlation
        correlation = np.corrcoef(coherences, hits)[0, 1]
        print(f"\n  Correlation(coherence, is_hit): {correlation:.3f}")

        # Save results
        results_file = PROJECT_ROOT / "data" / "analysis" / "h104_drug_class_coherence.json"
        with open(results_file, 'w') as f:
            json.dump({
                'high_precision': float(high_prec),
                'low_precision': float(low_prec),
                'medium_precision': float(med_prec),
                'difference_pp': float(diff * 100),
                'correlation': float(correlation),
                'success': bool(success),
                'n_with_atc': len(with_atc),
                'n_without_atc': len(without_atc),
            }, f, indent=2)
        print(f"\nResults saved to: {results_file}")

    else:
        print("No predictions with ATC data found!")


if __name__ == '__main__':
    main()
