#!/usr/bin/env python3
"""
Hypothesis h19: Disease Phenotype Similarity using HPO.

PURPOSE:
    Test whether Human Phenotype Ontology (HPO) phenotype similarity can
    improve drug repurposing predictions, either as:

    A) Standalone kNN with HPO similarity (replacing Node2Vec)
    B) Hybrid kNN combining HPO + Node2Vec similarity
    C) HPO-gated filtering on Node2Vec kNN predictions

The hypothesis is that diseases with similar phenotypes may respond to
similar treatments, providing external (non-DRKG) information that could
break the 37% R@30 ceiling.

PRECONDITIONS CHECKED:
    - 799 DRKG diseases have HPO phenotype mappings (22.9% coverage)
    - Mean phenotype Jaccard similarity is 0.036 (sparse but non-zero)
    - HPO annotations come from OMIM/Orphanet (external to DRKG)

EVALUATION:
    Multi-seed evaluation (seeds 42, 123, 456, 789, 1024) per h40 findings.
    Baseline: kNN k=20 with Node2Vec cosine = 37.04% ± 5.81% R@30.
"""

import json
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from disease_name_matcher import DiseaseMatcher, load_mesh_mappings

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"

SEEDS = [42, 123, 456, 789, 1024]
K_DEFAULT = 20


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_node2vec_embeddings() -> Dict[str, np.ndarray]:
    """Load Node2Vec embeddings."""
    df = pd.read_csv(EMBEDDINGS_DIR / "node2vec_256_named.csv")
    dim_cols = [c for c in df.columns if c.startswith("dim_")]
    embeddings: Dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        entity = f"drkg:{row['entity']}"
        embeddings[entity] = row[dim_cols].values.astype(np.float32)
    return embeddings


def load_hpo_phenotypes() -> Dict[str, Set[str]]:
    """Load HPO phenotype profiles for DRKG diseases."""
    with open(REFERENCE_DIR / "drkg_disease_phenotypes.json") as f:
        data = json.load(f)
    return {k: set(v) for k, v in data.items()}


def load_drugbank_lookup() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Load DrugBank ID mappings."""
    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)
    name_to_id = {name.lower(): f"drkg:Compound::{db_id}" for db_id, name in id_to_name.items()}
    id_to_name_drkg = {f"drkg:Compound::{db_id}": name for db_id, name in id_to_name.items()}
    return name_to_id, id_to_name_drkg


def load_mesh_mappings_from_file() -> Dict[str, str]:
    """Load manual MESH mappings."""
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
    """Load ground truth from Every Cure indications."""
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
    valid_entity_check,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Split diseases into train/test sets."""
    valid_diseases = [d for d in gt_pairs if valid_entity_check(d)]
    rng = np.random.RandomState(seed)
    rng.shuffle(valid_diseases)
    n_test = max(1, int(len(valid_diseases) * test_fraction))
    test_diseases = set(valid_diseases[:n_test])
    train_diseases = set(valid_diseases[n_test:])
    train_gt = {d: gt_pairs[d] for d in train_diseases}
    test_gt = {d: gt_pairs[d] for d in test_diseases}
    return train_gt, test_gt


# ─── Similarity Functions ────────────────────────────────────────────────────

def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def compute_hpo_similarity(
    disease1: str,
    disease2: str,
    phenotypes: Dict[str, Set[str]],
) -> float:
    """Compute HPO phenotype Jaccard similarity between diseases."""
    p1 = phenotypes.get(disease1, set())
    p2 = phenotypes.get(disease2, set())
    return jaccard_similarity(p1, p2)


def compute_node2vec_similarity(
    disease1: str,
    disease2: str,
    embeddings: Dict[str, np.ndarray],
) -> float:
    """Compute Node2Vec cosine similarity between diseases."""
    if disease1 not in embeddings or disease2 not in embeddings:
        return 0.0
    e1 = embeddings[disease1].reshape(1, -1)
    e2 = embeddings[disease2].reshape(1, -1)
    return float(cosine_similarity(e1, e2)[0, 0])


# ─── kNN Evaluation Functions ────────────────────────────────────────────────

def knn_evaluate(
    test_gt: Dict[str, Set[str]],
    train_gt: Dict[str, Set[str]],
    similarity_func,
    k: int = 20,
    embeddings: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[float, int, int, Dict[str, dict]]:
    """
    Evaluate kNN drug transfer using a given similarity function.

    Returns:
        recall: R@30 recall
        total_hits: number of correct predictions
        total_gt: total ground truth drugs
        per_disease_results: dict with per-disease breakdown
    """
    train_disease_list = list(train_gt.keys())

    # Filter to valid diseases (have embeddings if using Node2Vec)
    if embeddings is not None:
        train_disease_list = [d for d in train_disease_list if d in embeddings]

    total_hits = 0
    total_gt = 0
    per_disease_results: Dict[str, dict] = {}

    for disease_id, gt_drugs in test_gt.items():
        # Filter GT to valid drugs
        if embeddings is not None:
            gt_drugs = {d for d in gt_drugs if d in embeddings}
        if not gt_drugs:
            continue

        # Compute similarities to all training diseases
        sims = []
        for train_disease in train_disease_list:
            sim = similarity_func(disease_id, train_disease)
            sims.append((train_disease, sim))

        # Sort by similarity, get top k
        sims.sort(key=lambda x: x[1], reverse=True)
        top_k = sims[:k]

        # Count drug frequency among nearest diseases
        drug_counts: Dict[str, float] = defaultdict(float)
        for neighbor_disease, neighbor_sim in top_k:
            for drug_id in train_gt[neighbor_disease]:
                if embeddings is None or drug_id in embeddings:
                    drug_counts[drug_id] += neighbor_sim  # Weight by similarity

        # Rank by weighted count, take top 30
        if drug_counts:
            sorted_drugs = sorted(drug_counts.items(), key=lambda x: x[1], reverse=True)
            top_30 = {d for d, _ in sorted_drugs[:30]}
        else:
            top_30 = set()

        hits = len(top_30 & gt_drugs)
        total_hits += hits
        total_gt += len(gt_drugs)

        per_disease_results[disease_id] = {
            "gt_count": len(gt_drugs),
            "hits": hits,
            "recall": hits / len(gt_drugs) if gt_drugs else 0,
            "top_k_sims": [(d, s) for d, s in top_k[:3]],  # Top 3 for debugging
        }

    recall = total_hits / total_gt if total_gt > 0 else 0
    return recall, total_hits, total_gt, per_disease_results


def evaluate_hybrid_knn(
    test_gt: Dict[str, Set[str]],
    train_gt: Dict[str, Set[str]],
    phenotypes: Dict[str, Set[str]],
    embeddings: Dict[str, np.ndarray],
    k: int = 20,
    alpha: float = 0.5,
) -> Tuple[float, int, int]:
    """
    Evaluate hybrid kNN combining HPO + Node2Vec similarity.

    similarity = alpha * hpo_sim + (1 - alpha) * node2vec_sim
    """
    train_disease_list = [d for d in train_gt if d in embeddings]

    # Pre-compute Node2Vec similarity matrix for efficiency
    train_embs = np.array([embeddings[d] for d in train_disease_list], dtype=np.float32)

    total_hits = 0
    total_gt = 0

    for disease_id, gt_drugs in test_gt.items():
        if disease_id not in embeddings:
            continue
        gt_drugs = {d for d in gt_drugs if d in embeddings}
        if not gt_drugs:
            continue

        # Get Node2Vec similarities
        test_emb = embeddings[disease_id].reshape(1, -1)
        n2v_sims = cosine_similarity(test_emb, train_embs)[0]

        # Combine with HPO similarities
        combined_sims = []
        for i, train_disease in enumerate(train_disease_list):
            hpo_sim = compute_hpo_similarity(disease_id, train_disease, phenotypes)
            n2v_sim = n2v_sims[i]
            combined = alpha * hpo_sim + (1 - alpha) * n2v_sim
            combined_sims.append((train_disease, combined, hpo_sim, n2v_sim))

        # Sort by combined similarity
        combined_sims.sort(key=lambda x: x[1], reverse=True)
        top_k = combined_sims[:k]

        # Count drug frequency
        drug_counts: Dict[str, float] = defaultdict(float)
        for neighbor_disease, combined_sim, _, _ in top_k:
            for drug_id in train_gt[neighbor_disease]:
                if drug_id in embeddings:
                    drug_counts[drug_id] += combined_sim

        # Top 30
        if drug_counts:
            sorted_drugs = sorted(drug_counts.items(), key=lambda x: x[1], reverse=True)
            top_30 = {d for d, _ in sorted_drugs[:30]}
        else:
            top_30 = set()

        hits = len(top_30 & gt_drugs)
        total_hits += hits
        total_gt += len(gt_drugs)

    recall = total_hits / total_gt if total_gt > 0 else 0
    return recall, total_hits, total_gt


# ─── Main Evaluation ─────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Hypothesis h19: Disease Phenotype Similarity (HPO)")
    print("=" * 70)

    # Load data
    print("\n[1] Loading data...")
    embeddings = load_node2vec_embeddings()
    phenotypes = load_hpo_phenotypes()
    name_to_drug_id, _ = load_drugbank_lookup()
    mesh_mappings = load_mesh_mappings_from_file()
    gt_pairs = load_ground_truth(mesh_mappings, name_to_drug_id)

    print(f"    Embeddings: {len(embeddings)} entities")
    print(f"    Phenotypes: {len(phenotypes)} diseases with HPO data")
    print(f"    GT diseases: {len(gt_pairs)}")

    # Check overlap
    diseases_with_both = set(phenotypes.keys()) & set(gt_pairs.keys()) & set(embeddings.keys())
    print(f"    Diseases with GT + embeddings + HPO: {len(diseases_with_both)}")

    # Multi-seed evaluation
    results = {
        "node2vec_knn": [],
        "hpo_knn_all": [],
        "hpo_knn_subset": [],  # Only diseases with HPO
        "hybrid_alpha_0.3": [],
        "hybrid_alpha_0.5": [],
        "hybrid_alpha_0.7": [],
    }

    print("\n[2] Multi-seed evaluation...")
    for seed in SEEDS:
        print(f"\n  Seed {seed}:")

        # Split
        train_gt, test_gt = disease_level_split(
            gt_pairs,
            lambda d: d in embeddings,
            test_fraction=0.2,
            seed=seed,
        )

        # Filter test to diseases with embeddings
        test_gt_valid = {d: drugs for d, drugs in test_gt.items()
                        if d in embeddings and any(drug in embeddings for drug in drugs)}

        # Count coverage
        test_with_hpo = len([d for d in test_gt_valid if d in phenotypes])
        print(f"    Test diseases: {len(test_gt_valid)}, with HPO: {test_with_hpo} ({100*test_with_hpo/len(test_gt_valid):.1f}%)")

        # A) Baseline: Node2Vec kNN
        def n2v_sim(d1, d2):
            return compute_node2vec_similarity(d1, d2, embeddings)

        recall_n2v, hits_n2v, total_n2v, _ = knn_evaluate(
            test_gt_valid, train_gt, n2v_sim, k=K_DEFAULT, embeddings=embeddings
        )
        results["node2vec_knn"].append(recall_n2v)
        print(f"    Node2Vec kNN: {100*recall_n2v:.2f}% R@30 ({hits_n2v}/{total_n2v})")

        # B) HPO kNN (all diseases - fallback to 0 similarity for missing HPO)
        def hpo_sim_all(d1, d2):
            return compute_hpo_similarity(d1, d2, phenotypes)

        recall_hpo_all, hits_hpo_all, total_hpo_all, _ = knn_evaluate(
            test_gt_valid, train_gt, hpo_sim_all, k=K_DEFAULT, embeddings=embeddings
        )
        results["hpo_knn_all"].append(recall_hpo_all)
        print(f"    HPO kNN (all): {100*recall_hpo_all:.2f}% R@30 ({hits_hpo_all}/{total_hpo_all})")

        # C) HPO kNN (only diseases with HPO data)
        test_gt_hpo = {d: drugs for d, drugs in test_gt_valid.items() if d in phenotypes}
        train_gt_hpo = {d: drugs for d, drugs in train_gt.items() if d in phenotypes}

        if test_gt_hpo and train_gt_hpo:
            recall_hpo_sub, hits_hpo_sub, total_hpo_sub, _ = knn_evaluate(
                test_gt_hpo, train_gt_hpo, hpo_sim_all, k=K_DEFAULT, embeddings=embeddings
            )
            results["hpo_knn_subset"].append(recall_hpo_sub)
            print(f"    HPO kNN (subset): {100*recall_hpo_sub:.2f}% R@30 ({hits_hpo_sub}/{total_hpo_sub})")
        else:
            results["hpo_knn_subset"].append(0)

        # D) Hybrid kNN (various alpha values)
        for alpha in [0.3, 0.5, 0.7]:
            recall_hybrid, hits_hybrid, total_hybrid = evaluate_hybrid_knn(
                test_gt_valid, train_gt, phenotypes, embeddings, k=K_DEFAULT, alpha=alpha
            )
            results[f"hybrid_alpha_{alpha}"].append(recall_hybrid)
            print(f"    Hybrid α={alpha}: {100*recall_hybrid:.2f}% R@30 ({hits_hybrid}/{total_hybrid})")

    # Summary statistics
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (5-seed mean ± std)")
    print("=" * 70)

    summary = {}
    for method, recalls in results.items():
        mean_r = np.mean(recalls)
        std_r = np.std(recalls)
        summary[method] = {"mean": mean_r, "std": std_r, "values": recalls}
        print(f"{method:25s}: {100*mean_r:.2f}% ± {100*std_r:.2f}%")

    # Compare to baseline
    baseline_mean = summary["node2vec_knn"]["mean"]
    print("\n" + "-" * 70)
    print("DELTA vs Node2Vec baseline:")
    for method, stats in summary.items():
        if method != "node2vec_knn":
            delta = stats["mean"] - baseline_mean
            print(f"  {method}: {100*delta:+.2f} pp")

    # Statistical significance test (simple t-test approximation)
    print("\n" + "-" * 70)
    print("Analysis:")

    # Best hybrid
    best_hybrid = max(
        [(k, v["mean"]) for k, v in summary.items() if "hybrid" in k],
        key=lambda x: x[1]
    )
    print(f"  Best hybrid: {best_hybrid[0]} ({100*best_hybrid[1]:.2f}%)")

    # HPO-only analysis
    hpo_subset_mean = summary["hpo_knn_subset"]["mean"]
    print(f"  HPO-only (subset): {100*hpo_subset_mean:.2f}%")
    print(f"  Coverage: {len(diseases_with_both)}/{len(gt_pairs)} diseases ({100*len(diseases_with_both)/len(gt_pairs):.1f}%)")

    # Save results
    output = {
        "hypothesis": "h19",
        "title": "Disease Phenotype Similarity (HPO)",
        "baseline_metric": "37.04% R@30 (kNN k=20 Node2Vec)",
        "results": summary,
        "hpo_coverage": {
            "diseases_with_hpo": len(phenotypes),
            "diseases_with_gt_and_hpo": len(diseases_with_both),
            "total_gt_diseases": len(gt_pairs),
            "coverage_pct": len(diseases_with_both) / len(gt_pairs) * 100,
        },
        "findings": [],
        "conclusion": "",
    }

    # Determine conclusion
    best_result = max(summary.items(), key=lambda x: x[1]["mean"])
    if best_result[0] == "node2vec_knn":
        output["conclusion"] = "HPO phenotype similarity does NOT improve over Node2Vec baseline"
        output["status"] = "invalidated"
    elif best_result[1]["mean"] > baseline_mean + 0.02:  # >2 pp improvement
        output["conclusion"] = f"HPO improves R@30: {best_result[0]} achieves {100*best_result[1]['mean']:.2f}%"
        output["status"] = "validated"
    else:
        output["conclusion"] = "HPO provides marginal improvement, not exceeding noise"
        output["status"] = "inconclusive"

    output_path = ANALYSIS_DIR / "h19_hpo_phenotype_results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    print(f"\n{'='*70}")
    print(f"CONCLUSION: {output['conclusion']}")
    print(f"STATUS: {output['status']}")
    print(f"{'='*70}")

    return output


if __name__ == "__main__":
    main()
