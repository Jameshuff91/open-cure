#!/usr/bin/env python3
"""
Hypothesis h17: PPI Network Distance Features.

PURPOSE:
    Test whether protein-protein interaction (PPI) network distance from drug
    targets to disease genes can improve drug repurposing predictions.

    The hypothesis is that drugs targeting proteins close to disease genes
    in the PPI network are more likely to be effective treatments.

APPROACH:
    A) PPI-based disease similarity: Diseases with overlapping gene neighborhoods
    B) PPI-augmented kNN: Add PPI distance as additional similarity signal
    C) Direct PPI distance features: Min/mean distance as ML features

EVALUATION:
    Multi-seed evaluation (seeds 42, 123, 456, 789, 1024).
    Baseline: kNN k=20 with Node2Vec cosine = 37.04% ± 5.81% R@30.
"""

import json
import sys
from pathlib import Path
from collections import deque, defaultdict
from typing import Dict, List, Set, Tuple, Optional

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

def load_ppi_network() -> Dict[str, Set[str]]:
    """Load STRING PPI network (high confidence edges)."""
    with open(REFERENCE_DIR / "ppi" / "ppi_network_high_conf.json") as f:
        ppi = json.load(f)
    return {gene: set(neighbors) for gene, neighbors in ppi.items()}


def load_drug_targets() -> Dict[str, List[str]]:
    """Load drug target genes (Entrez IDs)."""
    with open(REFERENCE_DIR / "drug_targets.json") as f:
        return json.load(f)


def load_disease_genes() -> Dict[str, List[str]]:
    """Load disease-associated genes (Entrez IDs)."""
    with open(REFERENCE_DIR / "disease_genes.json") as f:
        return json.load(f)


def load_node2vec_embeddings() -> Dict[str, np.ndarray]:
    """Load Node2Vec embeddings."""
    df = pd.read_csv(EMBEDDINGS_DIR / "node2vec_256_named.csv")
    dim_cols = [c for c in df.columns if c.startswith("dim_")]
    embeddings: Dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        entity = f"drkg:{row['entity']}"
        embeddings[entity] = row[dim_cols].values.astype(np.float32)
    return embeddings


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


# ─── PPI Distance Functions ──────────────────────────────────────────────────

def bfs_distances(
    start_genes: List[str],
    ppi_neighbors: Dict[str, Set[str]],
    max_depth: int = 6,
) -> Dict[str, int]:
    """BFS from multiple start genes, return min distance to each reachable gene."""
    distances: Dict[str, int] = {}
    queue: deque[Tuple[str, int]] = deque()

    for gene in start_genes:
        if gene in ppi_neighbors and gene not in distances:
            distances[gene] = 0
            queue.append((gene, 0))

    while queue:
        gene, depth = queue.popleft()
        if depth >= max_depth:
            continue

        for neighbor in ppi_neighbors.get(gene, set()):
            if neighbor not in distances:
                distances[neighbor] = depth + 1
                queue.append((neighbor, depth + 1))

    return distances


def compute_ppi_distance(
    drug_targets: List[str],
    disease_genes: List[str],
    ppi_neighbors: Dict[str, Set[str]],
    max_depth: int = 6,
) -> Tuple[Optional[float], Optional[float]]:
    """Compute min and mean PPI distance from drug targets to disease genes."""
    if not drug_targets or not disease_genes:
        return None, None

    # BFS from drug targets
    target_distances = bfs_distances(drug_targets, ppi_neighbors, max_depth)

    # Find distances to disease genes
    distances = []
    for gene in disease_genes:
        if gene in target_distances:
            distances.append(target_distances[gene])

    if not distances:
        return None, None

    return min(distances), np.mean(distances)


def compute_disease_ppi_similarity(
    d1_genes: List[str],
    d2_genes: List[str],
    ppi_neighbors: Dict[str, Set[str]],
    max_depth: int = 2,
) -> float:
    """Compute similarity between diseases based on PPI neighborhood overlap."""
    if not d1_genes or not d2_genes:
        return 0.0

    # Get 1-hop and 2-hop neighborhoods
    def get_neighborhood(genes: List[str], depth: int) -> Set[str]:
        neighborhood = set(genes)
        for _ in range(depth):
            expanded = set()
            for gene in neighborhood:
                expanded.update(ppi_neighbors.get(gene, set()))
            neighborhood.update(expanded)
        return neighborhood

    n1 = get_neighborhood(d1_genes, max_depth)
    n2 = get_neighborhood(d2_genes, max_depth)

    if not n1 or not n2:
        return 0.0

    intersection = len(n1 & n2)
    union = len(n1 | n2)
    return intersection / union if union > 0 else 0.0


# ─── kNN Evaluation Functions ────────────────────────────────────────────────

def knn_node2vec(
    test_gt: Dict[str, Set[str]],
    train_gt: Dict[str, Set[str]],
    embeddings: Dict[str, np.ndarray],
    k: int = 20,
) -> Tuple[float, int, int]:
    """Standard Node2Vec kNN baseline."""
    train_disease_list = [d for d in train_gt if d in embeddings]
    train_embs = np.array([embeddings[d] for d in train_disease_list], dtype=np.float32)

    total_hits = 0
    total_gt = 0

    for disease_id, gt_drugs in test_gt.items():
        if disease_id not in embeddings:
            continue
        gt_drugs = {d for d in gt_drugs if d in embeddings}
        if not gt_drugs:
            continue

        # Node2Vec similarity
        test_emb = embeddings[disease_id].reshape(1, -1)
        sims = cosine_similarity(test_emb, train_embs)[0]

        # Top k neighbors
        top_k_idx = np.argsort(sims)[-k:]

        # Count drug frequency
        drug_counts: Dict[str, float] = defaultdict(float)
        for idx in top_k_idx:
            neighbor_sim = sims[idx]
            for drug_id in train_gt[train_disease_list[idx]]:
                if drug_id in embeddings:
                    drug_counts[drug_id] += neighbor_sim

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


def knn_ppi(
    test_gt: Dict[str, Set[str]],
    train_gt: Dict[str, Set[str]],
    disease_genes: Dict[str, List[str]],
    ppi_neighbors: Dict[str, Set[str]],
    embeddings: Dict[str, np.ndarray],
    k: int = 20,
) -> Tuple[float, int, int]:
    """kNN using PPI-based disease similarity."""
    train_disease_list = [d for d in train_gt if d in embeddings]

    # Get MESH IDs
    def drkg_to_mesh(drkg_id: str) -> str:
        return drkg_id.split("::")[-1] if "::" in drkg_id else drkg_id

    # Pre-compute disease genes for training diseases
    train_disease_genes: Dict[str, List[str]] = {}
    for d in train_disease_list:
        mesh_id = drkg_to_mesh(d)
        if mesh_id in disease_genes:
            train_disease_genes[d] = disease_genes[mesh_id]

    total_hits = 0
    total_gt = 0

    for disease_id, gt_drugs in test_gt.items():
        if disease_id not in embeddings:
            continue
        gt_drugs = {d for d in gt_drugs if d in embeddings}
        if not gt_drugs:
            continue

        test_mesh = drkg_to_mesh(disease_id)
        test_genes = disease_genes.get(test_mesh, [])

        # Compute PPI similarity to all training diseases
        sims = []
        for train_disease in train_disease_list:
            if train_disease in train_disease_genes:
                sim = compute_disease_ppi_similarity(
                    test_genes, train_disease_genes[train_disease],
                    ppi_neighbors, max_depth=2
                )
            else:
                sim = 0.0
            sims.append((train_disease, sim))

        # Sort by similarity
        sims.sort(key=lambda x: x[1], reverse=True)
        top_k = sims[:k]

        # Count drug frequency
        drug_counts: Dict[str, float] = defaultdict(float)
        for neighbor_disease, neighbor_sim in top_k:
            if neighbor_sim > 0:
                for drug_id in train_gt[neighbor_disease]:
                    if drug_id in embeddings:
                        drug_counts[drug_id] += neighbor_sim

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


def knn_hybrid(
    test_gt: Dict[str, Set[str]],
    train_gt: Dict[str, Set[str]],
    disease_genes: Dict[str, List[str]],
    ppi_neighbors: Dict[str, Set[str]],
    embeddings: Dict[str, np.ndarray],
    k: int = 20,
    alpha: float = 0.5,
) -> Tuple[float, int, int]:
    """Hybrid kNN: alpha * PPI + (1 - alpha) * Node2Vec."""
    train_disease_list = [d for d in train_gt if d in embeddings]
    train_embs = np.array([embeddings[d] for d in train_disease_list], dtype=np.float32)

    def drkg_to_mesh(drkg_id: str) -> str:
        return drkg_id.split("::")[-1] if "::" in drkg_id else drkg_id

    train_disease_genes: Dict[str, List[str]] = {}
    for d in train_disease_list:
        mesh_id = drkg_to_mesh(d)
        if mesh_id in disease_genes:
            train_disease_genes[d] = disease_genes[mesh_id]

    total_hits = 0
    total_gt = 0

    for disease_id, gt_drugs in test_gt.items():
        if disease_id not in embeddings:
            continue
        gt_drugs = {d for d in gt_drugs if d in embeddings}
        if not gt_drugs:
            continue

        # Node2Vec similarities
        test_emb = embeddings[disease_id].reshape(1, -1)
        n2v_sims = cosine_similarity(test_emb, train_embs)[0]

        # PPI similarities
        test_mesh = drkg_to_mesh(disease_id)
        test_genes = disease_genes.get(test_mesh, [])

        combined_sims = []
        for i, train_disease in enumerate(train_disease_list):
            n2v_sim = n2v_sims[i]

            if train_disease in train_disease_genes and test_genes:
                ppi_sim = compute_disease_ppi_similarity(
                    test_genes, train_disease_genes[train_disease],
                    ppi_neighbors, max_depth=2
                )
            else:
                ppi_sim = 0.0

            combined = alpha * ppi_sim + (1 - alpha) * n2v_sim
            combined_sims.append((train_disease, combined))

        # Sort by combined similarity
        combined_sims.sort(key=lambda x: x[1], reverse=True)
        top_k = combined_sims[:k]

        # Count drug frequency
        drug_counts: Dict[str, float] = defaultdict(float)
        for neighbor_disease, neighbor_sim in top_k:
            for drug_id in train_gt[neighbor_disease]:
                if drug_id in embeddings:
                    drug_counts[drug_id] += neighbor_sim

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
    print("Hypothesis h17: PPI Network Distance Features")
    print("=" * 70)

    # Load data
    print("\n[1] Loading data...")
    ppi_neighbors = load_ppi_network()
    drug_targets_data = load_drug_targets()
    disease_genes_data = load_disease_genes()
    embeddings = load_node2vec_embeddings()
    name_to_drug_id, _ = load_drugbank_lookup()
    mesh_mappings = load_mesh_mappings_from_file()
    gt_pairs = load_ground_truth(mesh_mappings, name_to_drug_id)

    print(f"    PPI network: {len(ppi_neighbors)} genes")
    print(f"    Drug targets: {len(drug_targets_data)} drugs")
    print(f"    Disease genes: {len(disease_genes_data)} diseases")
    print(f"    Embeddings: {len(embeddings)} entities")
    print(f"    GT diseases: {len(gt_pairs)}")

    # Coverage check
    def drkg_to_mesh(drkg_id: str) -> str:
        return drkg_id.split("::")[-1] if "::" in drkg_id else drkg_id

    gt_with_genes = sum(1 for d in gt_pairs if drkg_to_mesh(d) in disease_genes_data)
    print(f"    GT diseases with PPI genes: {gt_with_genes} ({100*gt_with_genes/len(gt_pairs):.1f}%)")

    # Multi-seed evaluation
    results = {
        "node2vec_knn": [],
        "ppi_knn": [],
        "hybrid_alpha_0.2": [],
        "hybrid_alpha_0.3": [],
        "hybrid_alpha_0.5": [],
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

        # Filter test to valid diseases
        test_gt_valid = {d: drugs for d, drugs in test_gt.items()
                        if d in embeddings and any(drug in embeddings for drug in drugs)}

        # Count coverage
        test_with_ppi = len([d for d in test_gt_valid
                           if drkg_to_mesh(d) in disease_genes_data])
        print(f"    Test diseases: {len(test_gt_valid)}, with PPI genes: {test_with_ppi}")

        # A) Node2Vec baseline
        recall_n2v, hits_n2v, total_n2v = knn_node2vec(
            test_gt_valid, train_gt, embeddings, k=K_DEFAULT
        )
        results["node2vec_knn"].append(recall_n2v)
        print(f"    Node2Vec kNN: {100*recall_n2v:.2f}% R@30 ({hits_n2v}/{total_n2v})")

        # B) PPI-only kNN
        recall_ppi, hits_ppi, total_ppi = knn_ppi(
            test_gt_valid, train_gt, disease_genes_data, ppi_neighbors,
            embeddings, k=K_DEFAULT
        )
        results["ppi_knn"].append(recall_ppi)
        print(f"    PPI kNN: {100*recall_ppi:.2f}% R@30 ({hits_ppi}/{total_ppi})")

        # C) Hybrid kNN
        for alpha in [0.2, 0.3, 0.5]:
            recall_hybrid, hits_hybrid, total_hybrid = knn_hybrid(
                test_gt_valid, train_gt, disease_genes_data, ppi_neighbors,
                embeddings, k=K_DEFAULT, alpha=alpha
            )
            results[f"hybrid_alpha_{alpha}"].append(recall_hybrid)
            print(f"    Hybrid α={alpha}: {100*recall_hybrid:.2f}% R@30 ({hits_hybrid}/{total_hybrid})")

    # Summary
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

    # Save results
    output = {
        "hypothesis": "h17",
        "title": "PPI Network Distance Features",
        "baseline_metric": "37.04% R@30 (kNN k=20 Node2Vec)",
        "results": summary,
        "ppi_coverage": {
            "gt_diseases_with_ppi_genes": gt_with_genes,
            "total_gt_diseases": len(gt_pairs),
            "coverage_pct": gt_with_genes / len(gt_pairs) * 100,
        },
    }

    # Determine conclusion
    best_result = max(summary.items(), key=lambda x: x[1]["mean"])
    if best_result[0] == "node2vec_knn":
        output["conclusion"] = "PPI distance does NOT improve over Node2Vec baseline"
        output["status"] = "invalidated"
    elif best_result[1]["mean"] > baseline_mean + 0.02:
        output["conclusion"] = f"PPI improves R@30: {best_result[0]} achieves {100*best_result[1]['mean']:.2f}%"
        output["status"] = "validated"
    else:
        output["conclusion"] = "PPI provides marginal improvement, not exceeding noise"
        output["status"] = "inconclusive"

    output_path = ANALYSIS_DIR / "h17_ppi_distance_results.json"
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
