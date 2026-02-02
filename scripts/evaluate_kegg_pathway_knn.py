#!/usr/bin/env python3
"""
KEGG Pathway kNN Evaluation - Inductive Disease Similarity.

PURPOSE:
    Evaluate drug repurposing using KEGG pathway-based disease similarity.
    This provides an INDUCTIVE evaluation (using only disease features, not
    graph embeddings) for fair comparison with TxGNN's zero-shot paradigm.

APPROACH:
    1. Compute Jaccard similarity between disease KEGG pathway sets
    2. Run kNN collaborative filtering (k=20)
    3. 5-seed evaluation for statistical validity

KEY DIFFERENCE FROM NODE2VEC:
    - Node2Vec embeddings are TRANSDUCTIVE (diseases present in graph during training)
    - KEGG pathways are INDUCTIVE (similarity from external biological knowledge)
    - Fair comparison to TxGNN which is also inductive

EXPECTED OUTCOME:
    - R@30 > 15% = competitive with TxGNN inductive
    - R@30 > 20% = breakthrough for feature-only approach
    - Dense data (47.4 pathways/disease avg) should outperform sparse HPO
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, Tuple, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from disease_name_matcher import DiseaseMatcher, load_mesh_mappings

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
PATHWAY_DIR = REFERENCE_DIR / "pathway"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"

SEEDS = [42, 123, 456, 789, 1024]
K_DEFAULT = 20


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_disease_pathways() -> Dict[str, Set[str]]:
    """Load KEGG pathway profiles for diseases."""
    with open(PATHWAY_DIR / "disease_pathways.json") as f:
        data = json.load(f)
    # Convert to drkg format and set
    result: Dict[str, Set[str]] = {}
    for mesh_id, pathways in data.items():
        # mesh_id is like "MESH:D005909", convert to "drkg:Disease::MESH:D005909"
        drkg_id = f"drkg:Disease::{mesh_id}"
        result[drkg_id] = set(pathways)
    return result


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
    valid_disease_check,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Split diseases into train/test sets."""
    valid_diseases = [d for d in gt_pairs if valid_disease_check(d)]
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


def compute_pathway_similarity(
    disease1: str,
    disease2: str,
    pathways: Dict[str, Set[str]],
) -> float:
    """Compute KEGG pathway Jaccard similarity between diseases."""
    p1 = pathways.get(disease1, set())
    p2 = pathways.get(disease2, set())
    return jaccard_similarity(p1, p2)


# ─── kNN Evaluation ──────────────────────────────────────────────────────────

def knn_evaluate(
    test_gt: Dict[str, Set[str]],
    train_gt: Dict[str, Set[str]],
    pathways: Dict[str, Set[str]],
    all_drugs: Set[str],
    k: int = 20,
) -> Tuple[float, int, int, Dict[str, dict]]:
    """
    Evaluate kNN drug transfer using KEGG pathway similarity.

    Returns:
        recall: R@30 recall
        total_hits: number of correct predictions
        total_gt: total ground truth drugs
        per_disease_results: dict with per-disease breakdown
    """
    train_disease_list = list(train_gt.keys())

    total_hits = 0
    total_gt = 0
    per_disease_results: Dict[str, dict] = {}

    for disease_id, gt_drugs in test_gt.items():
        # Filter GT to known drugs
        gt_drugs_valid = gt_drugs & all_drugs
        if not gt_drugs_valid:
            continue

        # Compute similarities to all training diseases
        sims = []
        for train_disease in train_disease_list:
            sim = compute_pathway_similarity(disease_id, train_disease, pathways)
            sims.append((train_disease, sim))

        # Sort by similarity, get top k
        sims.sort(key=lambda x: x[1], reverse=True)
        top_k = sims[:k]

        # Count drug frequency among nearest diseases
        drug_counts: Dict[str, float] = defaultdict(float)
        for neighbor_disease, neighbor_sim in top_k:
            if neighbor_sim > 0:  # Only count if some similarity
                for drug_id in train_gt[neighbor_disease]:
                    if drug_id in all_drugs:
                        drug_counts[drug_id] += neighbor_sim  # Weight by similarity

        # Rank by weighted count, take top 30
        if drug_counts:
            sorted_drugs = sorted(drug_counts.items(), key=lambda x: x[1], reverse=True)
            top_30 = {d for d, _ in sorted_drugs[:30]}
        else:
            top_30 = set()

        hits = len(top_30 & gt_drugs_valid)
        total_hits += hits
        total_gt += len(gt_drugs_valid)

        per_disease_results[disease_id] = {
            "gt_count": len(gt_drugs_valid),
            "hits": hits,
            "recall": hits / len(gt_drugs_valid) if gt_drugs_valid else 0,
            "top_k_sims": [(d, round(s, 4)) for d, s in top_k[:3]],
            "has_pathway": disease_id in pathways,
            "n_pathways": len(pathways.get(disease_id, set())),
        }

    recall = total_hits / total_gt if total_gt > 0 else 0
    return recall, total_hits, total_gt, per_disease_results


# ─── Main Evaluation ─────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("KEGG Pathway kNN Evaluation (Inductive)")
    print("=" * 70)
    print()
    print("PURPOSE: Fair comparison to TxGNN using only disease features")
    print("         (no graph embeddings = truly inductive)")
    print()

    # Load data
    print("[1] Loading data...")
    pathways = load_disease_pathways()
    name_to_drug_id, id_to_drug_name = load_drugbank_lookup()
    mesh_mappings = load_mesh_mappings_from_file()
    gt_pairs = load_ground_truth(mesh_mappings, name_to_drug_id)
    all_drugs = set(name_to_drug_id.values())

    print(f"    KEGG diseases: {len(pathways)}")
    print(f"    GT diseases: {len(gt_pairs)}")
    print(f"    All drugs: {len(all_drugs)}")

    # Analyze pathway density
    pathway_counts = [len(p) for p in pathways.values()]
    print(f"    Pathways per disease: mean={np.mean(pathway_counts):.1f}, "
          f"median={np.median(pathway_counts):.0f}, max={max(pathway_counts)}")

    # Check overlap
    diseases_with_both = set(pathways.keys()) & set(gt_pairs.keys())
    print(f"    Diseases with GT + KEGG: {len(diseases_with_both)} "
          f"({100*len(diseases_with_both)/len(gt_pairs):.1f}% coverage)")

    if len(diseases_with_both) < 50:
        print("\n[!] WARNING: Low overlap - results may be unreliable")
        print("    Continuing anyway for completeness...")

    # Multi-seed evaluation
    print("\n[2] Multi-seed evaluation...")
    results = {
        "kegg_knn_all": [],  # All diseases (0 similarity for missing KEGG)
        "kegg_knn_subset": [],  # Only diseases with KEGG data
    }

    all_per_disease: Dict[int, Dict] = {}

    for seed in SEEDS:
        print(f"\n  Seed {seed}:")

        # Split - use pathway coverage as validity check
        train_gt, test_gt = disease_level_split(
            gt_pairs,
            lambda d: d in pathways,  # Only diseases with KEGG data
            test_fraction=0.2,
            seed=seed,
        )

        # Filter test to diseases with pathway data
        test_gt_valid = {d: drugs for d, drugs in test_gt.items()
                        if d in pathways and any(drug in all_drugs for drug in drugs)}

        print(f"    Train: {len(train_gt)}, Test: {len(test_gt_valid)} diseases")

        # Run kNN evaluation
        recall, hits, total, per_disease = knn_evaluate(
            test_gt_valid, train_gt, pathways, all_drugs, k=K_DEFAULT
        )
        results["kegg_knn_subset"].append(recall)
        all_per_disease[seed] = per_disease
        print(f"    KEGG kNN (subset): {100*recall:.2f}% R@30 ({hits}/{total})")

        # Also evaluate on ALL diseases (sparse similarity for missing)
        train_gt_all, test_gt_all = disease_level_split(
            gt_pairs,
            lambda d: True,  # All diseases
            test_fraction=0.2,
            seed=seed,
        )
        test_gt_all_valid = {d: drugs for d, drugs in test_gt_all.items()
                           if any(drug in all_drugs for drug in drugs)}

        recall_all, hits_all, total_all, _ = knn_evaluate(
            test_gt_all_valid, train_gt_all, pathways, all_drugs, k=K_DEFAULT
        )
        results["kegg_knn_all"].append(recall_all)
        print(f"    KEGG kNN (all): {100*recall_all:.2f}% R@30 ({hits_all}/{total_all})")

    # Summary statistics
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (5-seed mean +/- std)")
    print("=" * 70)

    summary = {}
    for method, recalls in results.items():
        mean_r = np.mean(recalls)
        std_r = np.std(recalls)
        summary[method] = {"mean": mean_r, "std": std_r, "values": recalls}
        print(f"{method:25s}: {100*mean_r:.2f}% +/- {100*std_r:.2f}%")

    # Compare to baselines
    print("\n" + "-" * 70)
    print("COMPARISON TO BASELINES:")
    print("-" * 70)
    print()

    kegg_mean = summary["kegg_knn_subset"]["mean"]
    comparisons = [
        ("Node2Vec kNN (transductive)", 0.2606, "Honest evaluation w/o treatment edges"),
        ("TxGNN (inductive)", 0.145, "Zero-shot on unseen diseases"),
        ("HPO kNN (h19)", 0.142, "Phenotype similarity (sparse)"),
    ]

    for name, baseline, note in comparisons:
        delta = kegg_mean - baseline
        print(f"  vs {name}: {100*delta:+.2f} pp")
        print(f"     ({note})")
        print()

    # Interpretation
    print("-" * 70)
    print("INTERPRETATION:")
    print("-" * 70)

    kegg_mean = summary["kegg_knn_subset"]["mean"]
    if kegg_mean >= 0.20:
        interpretation = (
            f"BREAKTHROUGH: {100*kegg_mean:.1f}% R@30 using only KEGG pathways! "
            "This is a strong inductive baseline competitive with or exceeding "
            "TxGNN's zero-shot performance."
        )
        status = "breakthrough"
    elif kegg_mean >= 0.15:
        interpretation = (
            f"COMPETITIVE: {100*kegg_mean:.1f}% R@30 is on par with TxGNN's "
            "14.5% inductive performance. KEGG pathways provide meaningful "
            "disease similarity for drug repurposing."
        )
        status = "competitive"
    elif kegg_mean >= 0.10:
        interpretation = (
            f"MODERATE: {100*kegg_mean:.1f}% R@30 is below TxGNN but better than "
            "random. KEGG pathways capture some disease similarity signal."
        )
        status = "moderate"
    else:
        interpretation = (
            f"WEAK: {100*kegg_mean:.1f}% R@30 suggests KEGG pathway similarity "
            "alone is insufficient for drug repurposing. May need combination "
            "with other features."
        )
        status = "weak"

    print(f"\n  {interpretation}")
    print()

    # Analysis of per-disease performance
    print("-" * 70)
    print("PER-DISEASE ANALYSIS (last seed):")
    print("-" * 70)

    last_seed_results = all_per_disease[SEEDS[-1]]

    # Bin by pathway count
    bins = [(0, 50), (50, 100), (100, 200), (200, 500)]
    for low, high in bins:
        diseases_in_bin = [
            d for d, r in last_seed_results.items()
            if low <= r.get("n_pathways", 0) < high
        ]
        if diseases_in_bin:
            recalls = [last_seed_results[d]["recall"] for d in diseases_in_bin]
            print(f"  {low}-{high} pathways: {len(diseases_in_bin)} diseases, "
                  f"mean recall = {100*np.mean(recalls):.1f}%")

    # Save results
    output = {
        "analysis": "kegg_pathway_knn",
        "description": "Inductive kNN using KEGG pathway Jaccard similarity",
        "purpose": "Fair comparison to TxGNN (both inductive)",
        "params": {"k": K_DEFAULT, "seeds": SEEDS},
        "data": {
            "kegg_diseases": len(pathways),
            "gt_diseases": len(gt_pairs),
            "overlap": len(diseases_with_both),
            "coverage_pct": 100 * len(diseases_with_both) / len(gt_pairs),
            "mean_pathways_per_disease": float(np.mean(pathway_counts)),
        },
        "results": {
            k: {
                "mean": v["mean"],
                "std": v["std"],
                "values": v["values"],
            }
            for k, v in summary.items()
        },
        "comparisons": {
            "vs_node2vec_honest": kegg_mean - 0.2606,
            "vs_txgnn": kegg_mean - 0.145,
            "vs_hpo_h19": kegg_mean - 0.142,
        },
        "interpretation": interpretation,
        "status": status,
    }

    output_path = ANALYSIS_DIR / "kegg_pathway_knn_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nResults saved to {output_path}")

    print(f"\n{'='*70}")
    print(f"CONCLUSION: {interpretation}")
    print(f"STATUS: {status.upper()}")
    print(f"{'='*70}")

    return output


if __name__ == "__main__":
    main()
