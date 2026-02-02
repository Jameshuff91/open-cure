#!/usr/bin/env python3
"""
Compare kNN performance with original vs honest (no-treatment) embeddings.

PURPOSE:
    Test whether the 37% R@30 kNN performance is inflated by treatment edges
    in the Node2Vec embeddings. If disease similarity is partly derived from
    "these diseases share treatments," then removing treatment edges should
    significantly reduce performance.

METHODOLOGY:
    - Original embeddings: Node2Vec trained on full DRKG (including 64K treatment edges)
    - Honest embeddings: Node2Vec trained on DRKG minus treatment edges
    - Evaluate kNN (k=20) on both, using 5-seed disease-holdout evaluation

EXPECTED OUTCOMES:
    | Scenario          | Original | Honest | Interpretation                           |
    |-------------------|----------|--------|------------------------------------------|
    | Leakage confirmed | 37%      | 10-20% | Treatment edges were inflating score     |
    | No leakage        | 37%      | 35%+   | Similarity from indirect paths           |
    | Worse             | 37%      | <10%   | Severe leakage, TxGNN comparison invalid |
"""

import json
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, Tuple, List

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from disease_name_matcher import DiseaseMatcher, load_mesh_mappings

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"

SEEDS = [42, 123, 456, 789, 1024]
K = 20  # kNN parameter


def load_embeddings(path: Path) -> Dict[str, np.ndarray]:
    """Load Node2Vec embeddings from CSV."""
    df = pd.read_csv(path)
    dim_cols = [c for c in df.columns if c.startswith("dim_")]
    embeddings: Dict[str, np.ndarray] = {}
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

    gt_pairs: Dict[str, Set[str]] = defaultdict(set)
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
    return dict(gt_pairs)


def disease_level_split(
    gt_pairs: Dict[str, Set[str]],
    valid_entity_check,  # type: ignore[type-arg]
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


def evaluate_knn(
    emb_dict: Dict[str, np.ndarray],
    train_gt: Dict[str, Set[str]],
    test_gt: Dict[str, Set[str]],
    k: int = 20,
) -> float:
    """kNN drug transfer: rank drugs by frequency among k nearest training diseases."""
    train_disease_list = [d for d in train_gt if d in emb_dict]
    train_disease_embs = np.array([emb_dict[d] for d in train_disease_list], dtype=np.float32)

    total_hits = 0
    total_gt = 0

    for disease_id in test_gt:
        if disease_id not in emb_dict:
            continue
        gt_drugs = {d for d in test_gt[disease_id] if d in emb_dict}
        if not gt_drugs:
            continue

        # Find k nearest training diseases
        test_emb = emb_dict[disease_id].reshape(1, -1)
        sims = cosine_similarity(test_emb, train_disease_embs)[0]
        top_k_disease_idx = np.argsort(sims)[-k:]

        # Count drug frequency among nearest diseases (weighted by similarity)
        drug_counts: Dict[str, float] = defaultdict(float)
        for idx in top_k_disease_idx:
            neighbor_disease = train_disease_list[idx]
            neighbor_sim = sims[idx]
            for drug_id in train_gt[neighbor_disease]:
                if drug_id in emb_dict:
                    drug_counts[drug_id] += neighbor_sim

        # Rank by weighted count, take top 30
        if drug_counts:
            sorted_drugs = sorted(drug_counts.items(), key=lambda x: x[1], reverse=True)
            top_30 = {d for d, _ in sorted_drugs[:30]}
        else:
            top_30 = set()

        hits = len(top_30 & gt_drugs)
        total_hits += hits
        total_gt += len(gt_drugs)

    return total_hits / total_gt if total_gt > 0 else 0


def run_multi_seed_evaluation(
    emb_dict: Dict[str, np.ndarray],
    gt_pairs: Dict[str, Set[str]],
    label: str,
) -> Tuple[float, float, List[float]]:
    """Run 5-seed evaluation and return mean, std, and per-seed results."""
    recalls: List[float] = []

    for seed in SEEDS:
        train_gt, test_gt = disease_level_split(
            gt_pairs, lambda d: d in emb_dict, test_fraction=0.2, seed=seed
        )
        r = evaluate_knn(emb_dict, train_gt, test_gt, k=K)
        recalls.append(r)
        print(f"    Seed {seed}: {r*100:.2f}%")

    mean_r = float(np.mean(recalls))
    std_r = float(np.std(recalls))
    print(f"  {label}: {mean_r*100:.2f}% ± {std_r*100:.2f}%")

    return mean_r, std_r, recalls


def main() -> None:
    start_time = time.time()

    print("=" * 70)
    print("FAIR EMBEDDING COMPARISON: ORIGINAL vs HONEST (NO TREATMENT)")
    print("=" * 70)
    print()

    # Paths
    original_path = EMBEDDINGS_DIR / "node2vec_256_named.csv"
    honest_path = EMBEDDINGS_DIR / "node2vec_256_no_treatment.csv"

    if not honest_path.exists():
        print(f"ERROR: Honest embeddings not found at {honest_path}")
        print("Run train_node2vec_no_treatment.py first.")
        sys.exit(1)

    # Load ground truth
    print("Loading ground truth...")
    name_to_drug_id, _ = load_drugbank_lookup()
    mesh_mappings = load_mesh_mappings_from_file()
    gt_pairs = load_ground_truth(mesh_mappings, name_to_drug_id)
    print(f"  {len(gt_pairs)} diseases, {sum(len(v) for v in gt_pairs.values())} pairs")

    # Load embeddings
    print("\nLoading embeddings...")
    print(f"  Original: {original_path}")
    original_emb = load_embeddings(original_path)
    print(f"    {len(original_emb):,} entities")

    print(f"  Honest: {honest_path}")
    honest_emb = load_embeddings(honest_path)
    print(f"    {len(honest_emb):,} entities")

    # Check coverage difference
    original_diseases = {d for d in gt_pairs if d in original_emb}
    honest_diseases = {d for d in gt_pairs if d in honest_emb}
    missing_in_honest = original_diseases - honest_diseases

    print(f"\n  Disease coverage:")
    print(f"    Original: {len(original_diseases)} diseases in GT with embeddings")
    print(f"    Honest: {len(honest_diseases)} diseases in GT with embeddings")
    if missing_in_honest:
        print(f"    Missing in honest (disconnected after removing treatment edges): {len(missing_in_honest)}")

    # Evaluate original embeddings
    print("\n" + "=" * 70)
    print(f"ORIGINAL EMBEDDINGS (trained with treatment edges) - kNN k={K}")
    print("=" * 70)
    orig_mean, orig_std, orig_recalls = run_multi_seed_evaluation(
        original_emb, gt_pairs, "Original"
    )

    # Evaluate honest embeddings
    print("\n" + "=" * 70)
    print(f"HONEST EMBEDDINGS (no treatment edges) - kNN k={K}")
    print("=" * 70)
    honest_mean, honest_std, honest_recalls = run_multi_seed_evaluation(
        honest_emb, gt_pairs, "Honest"
    )

    # Calculate difference
    diff = honest_mean - orig_mean
    diff_pp = diff * 100

    # Determine interpretation
    if honest_mean >= orig_mean * 0.9:  # Within 10%
        interpretation = "NO SIGNIFICANT LEAKAGE - Similarity from indirect paths, fair comparison with TxGNN"
    elif honest_mean >= orig_mean * 0.5:  # 50-90% retained
        interpretation = "MODERATE LEAKAGE - Treatment edges contributed, but indirect similarity still substantial"
    else:  # <50% retained
        interpretation = "SEVERE LEAKAGE - Treatment edges dominated similarity, TxGNN comparison unfair"

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"  Original (with treatment edges):  {orig_mean*100:.2f}% ± {orig_std*100:.2f}%")
    print(f"  Honest (no treatment edges):      {honest_mean*100:.2f}% ± {honest_std*100:.2f}%")
    print()
    print(f"  Difference: {diff_pp:+.2f} percentage points")
    print(f"  Retention:  {100*honest_mean/orig_mean:.1f}% of original performance")
    print()
    print(f"  INTERPRETATION: {interpretation}")

    # Save results
    results = {
        "experiment": "honest_embedding_comparison",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "parameters": {
            "k": K,
            "seeds": SEEDS,
            "test_fraction": 0.2,
        },
        "original": {
            "path": str(original_path),
            "mean_r30": orig_mean,
            "std_r30": orig_std,
            "per_seed": orig_recalls,
            "n_entities": len(original_emb),
        },
        "honest": {
            "path": str(honest_path),
            "mean_r30": honest_mean,
            "std_r30": honest_std,
            "per_seed": honest_recalls,
            "n_entities": len(honest_emb),
        },
        "comparison": {
            "difference_pp": diff_pp,
            "retention_pct": 100 * honest_mean / orig_mean if orig_mean > 0 else 0,
            "interpretation": interpretation,
        },
        "elapsed_seconds": time.time() - start_time,
    }

    output_path = ANALYSIS_DIR / "honest_embedding_comparison.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
