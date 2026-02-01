#!/usr/bin/env python3
"""
h67: Drug Class (ATC) Boosting for kNN

Tests if boosting drugs in the same ATC class as kNN pool drugs improves predictions.

Approach:
1. Run vanilla kNN to get drug rankings
2. Identify dominant ATC classes in top-k pool drugs
3. Boost drugs matching those ATC classes
4. Evaluate if boosted ranking improves R@30
"""

import json
import pickle
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, Tuple, List

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from atc_features import ATCMapper

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"

SEEDS = [42, 123, 456, 789, 1024]


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


def load_mesh_mappings() -> Dict[str, str]:
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
    """Load ground truth."""
    from disease_name_matcher import DiseaseMatcher, load_mesh_mappings as load_fuzzy_mappings
    df = pd.read_excel(REFERENCE_DIR / "everycure" / "indicationList.xlsx")
    fuzzy_mappings = load_fuzzy_mappings()
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


def get_drug_atc_classes(
    drug_id: str,
    id_to_name: Dict[str, str],
    atc_mapper: ATCMapper
) -> List[str]:
    """Get ATC level 1 classes for a drug."""
    drug_name = id_to_name.get(drug_id, "")
    if not drug_name:
        return []
    return atc_mapper.get_atc_level1(drug_name)


def knn_evaluate_with_atc_boost(
    disease_id: str,
    emb_dict: Dict[str, np.ndarray],
    train_gt: Dict[str, Set[str]],
    train_disease_list: List[str],
    train_disease_embs: np.ndarray,
    id_to_name: Dict[str, str],
    atc_mapper: ATCMapper,
    k: int = 20,
    atc_boost: float = 0.5,
) -> Tuple[Set[str], Set[str]]:
    """
    Run kNN with optional ATC boosting.

    Returns (vanilla_top30, boosted_top30)
    """
    test_emb = emb_dict[disease_id].reshape(1, -1)
    sims = cosine_similarity(test_emb, train_disease_embs)[0]
    top_k_idx = np.argsort(sims)[-k:]

    # Collect drug scores from neighbors
    drug_counts: Dict[str, float] = defaultdict(float)
    for idx in top_k_idx:
        neighbor_disease = train_disease_list[idx]
        neighbor_sim = sims[idx]
        for drug_id in train_gt[neighbor_disease]:
            if drug_id in emb_dict:
                drug_counts[drug_id] += neighbor_sim

    if not drug_counts:
        return set(), set()

    # Vanilla ranking
    sorted_drugs = sorted(drug_counts.items(), key=lambda x: x[1], reverse=True)
    vanilla_top30 = {d for d, _ in sorted_drugs[:30]}

    # Identify dominant ATC classes in pool
    atc_class_counts = defaultdict(int)
    for drug_id, score in drug_counts.items():
        atc_classes = get_drug_atc_classes(drug_id, id_to_name, atc_mapper)
        for atc in atc_classes:
            atc_class_counts[atc] += 1

    # Find dominant classes (top 3 by frequency)
    if atc_class_counts:
        dominant_atc = [atc for atc, _ in sorted(atc_class_counts.items(),
                                                   key=lambda x: x[1], reverse=True)[:3]]
    else:
        dominant_atc = []

    # Boosted ranking
    boosted_scores = {}
    for drug_id, score in drug_counts.items():
        drug_atc = get_drug_atc_classes(drug_id, id_to_name, atc_mapper)
        boost = 1.0
        if drug_atc and dominant_atc:
            # Boost if drug's ATC matches any dominant class
            if any(atc in dominant_atc for atc in drug_atc):
                boost = 1.0 + atc_boost
        boosted_scores[drug_id] = score * boost

    sorted_boosted = sorted(boosted_scores.items(), key=lambda x: x[1], reverse=True)
    boosted_top30 = {d for d, _ in sorted_boosted[:30]}

    return vanilla_top30, boosted_top30


def evaluate_atc_boost(
    emb_dict: Dict[str, np.ndarray],
    train_gt: Dict[str, Set[str]],
    test_gt: Dict[str, Set[str]],
    id_to_name: Dict[str, str],
    atc_mapper: ATCMapper,
    atc_boost: float = 0.5,
) -> Dict:
    """Evaluate ATC boosting on test diseases."""
    train_disease_list = [d for d in train_gt if d in emb_dict]
    train_disease_embs = np.array([emb_dict[d] for d in train_disease_list], dtype=np.float32)

    vanilla_hits = 0
    boosted_hits = 0
    n_eval = 0

    for disease_id in test_gt:
        if disease_id not in emb_dict:
            continue
        gt_drugs = {d for d in test_gt[disease_id] if d in emb_dict}
        if not gt_drugs:
            continue

        vanilla_top30, boosted_top30 = knn_evaluate_with_atc_boost(
            disease_id, emb_dict, train_gt, train_disease_list, train_disease_embs,
            id_to_name, atc_mapper, k=20, atc_boost=atc_boost
        )

        n_eval += 1
        if vanilla_top30 & gt_drugs:
            vanilla_hits += 1
        if boosted_top30 & gt_drugs:
            boosted_hits += 1

    vanilla_recall = vanilla_hits / n_eval if n_eval > 0 else 0
    boosted_recall = boosted_hits / n_eval if n_eval > 0 else 0

    return {
        "n_eval": n_eval,
        "vanilla_hits": vanilla_hits,
        "boosted_hits": boosted_hits,
        "vanilla_recall": vanilla_recall,
        "boosted_recall": boosted_recall,
        "delta": boosted_recall - vanilla_recall,
    }


def main():
    start_time = time.time()

    print("=" * 70)
    print("h67: Drug Class (ATC) Boosting for kNN")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    emb_dict = load_node2vec_embeddings()
    name_to_drug_id, id_to_name = load_drugbank_lookup()
    mesh_mappings = load_mesh_mappings()
    gt_pairs, disease_names = load_ground_truth(mesh_mappings, name_to_drug_id)

    print(f"  Embeddings: {len(emb_dict)} entities")
    print(f"  GT: {len(gt_pairs)} diseases")

    # Load ATC mapper
    print("  Loading ATC mapper...")
    atc_mapper = ATCMapper()

    # Check ATC coverage for our drugs
    drugs_with_atc = 0
    drugs_without_atc = 0
    for drug_id in id_to_name:
        if get_drug_atc_classes(drug_id, id_to_name, atc_mapper):
            drugs_with_atc += 1
        else:
            drugs_without_atc += 1
    print(f"  Drugs with ATC: {drugs_with_atc} ({drugs_with_atc/(drugs_with_atc+drugs_without_atc):.1%})")

    # Test multiple boost values
    boost_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    all_results = {boost: [] for boost in boost_values}

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")

        train_gt, test_gt = disease_level_split(
            gt_pairs, lambda d: d in emb_dict, test_fraction=0.2, seed=seed
        )
        print(f"  Train: {len(train_gt)} diseases, Test: {len(test_gt)} diseases")

        for boost in boost_values:
            results = evaluate_atc_boost(
                emb_dict, train_gt, test_gt, id_to_name, atc_mapper, atc_boost=boost
            )
            all_results[boost].append(results)

            if boost == 0.0:
                print(f"  Vanilla: {results['vanilla_recall']:.1%}")
            else:
                print(f"  Boost={boost}: {results['boosted_recall']:.1%} "
                      f"(Δ={results['delta']*100:+.2f} pp)")

    # Summary
    print("\n" + "=" * 70)
    print("MULTI-SEED SUMMARY")
    print("=" * 70)

    print(f"\n{'Boost Value':<15} {'Mean R@30':<15} {'Std':<10} {'Δ vs Vanilla':<15}")
    print("-" * 55)

    vanilla_mean = np.mean([r['vanilla_recall'] for r in all_results[0.0]])

    for boost in boost_values:
        if boost == 0.0:
            recalls = [r['vanilla_recall'] for r in all_results[boost]]
        else:
            recalls = [r['boosted_recall'] for r in all_results[boost]]
        mean_recall = np.mean(recalls)
        std_recall = np.std(recalls)
        delta = mean_recall - vanilla_mean

        print(f"{boost:<15.2f} {mean_recall*100:<15.2f}% {std_recall*100:<10.2f}% "
              f"{delta*100:+.2f} pp")

    # Statistical test for best boost
    from scipy.stats import ttest_rel

    best_boost = None
    best_delta = 0
    for boost in boost_values[1:]:  # Skip vanilla
        boosted_recalls = [r['boosted_recall'] for r in all_results[boost]]
        vanilla_recalls = [r['vanilla_recall'] for r in all_results[0.0]]
        delta = np.mean(boosted_recalls) - np.mean(vanilla_recalls)
        if delta > best_delta:
            best_delta = delta
            best_boost = boost

    if best_boost:
        boosted_recalls = [r['boosted_recall'] for r in all_results[best_boost]]
        vanilla_recalls = [r['vanilla_recall'] for r in all_results[0.0]]
        t_stat, p_value = ttest_rel(boosted_recalls, vanilla_recalls)
        print(f"\nBest boost: {best_boost} (Δ={best_delta*100:+.2f} pp)")
        print(f"Paired t-test: t={t_stat:.2f}, p={p_value:.4f}")

        if p_value < 0.05 and best_delta > 0.01:
            print("✓ SIGNIFICANT improvement (p<0.05, Δ>1 pp)")
        else:
            print("✗ Not significant or improvement < 1 pp")

    # Save results
    output = {
        "hypothesis": "h67",
        "title": "Drug Class (ATC) Boosting for kNN",
        "date": "2026-01-31",
        "summary": {
            boost: {
                "mean_recall": float(np.mean([r['boosted_recall' if boost > 0 else 'vanilla_recall']
                                               for r in all_results[boost]])),
                "std_recall": float(np.std([r['boosted_recall' if boost > 0 else 'vanilla_recall']
                                             for r in all_results[boost]])),
            }
            for boost in boost_values
        },
        "vanilla_mean": float(vanilla_mean),
        "best_boost": best_boost,
        "best_delta": float(best_delta) if best_boost else None,
        "elapsed_seconds": time.time() - start_time
    }

    output_path = ANALYSIS_DIR / "h67_atc_knn_boost.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()
