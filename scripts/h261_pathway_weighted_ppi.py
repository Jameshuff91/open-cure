#!/usr/bin/env python3
"""
h261: Pathway-Weighted PPI Scoring

PURPOSE:
    Test whether weighting PPI-extended target overlap by pathway membership
    improves drug repurposing predictions.

    Hypothesis: If both a drug's targets and a disease's genes are in the
    same KEGG pathway, this is a stronger signal than just being connected
    via PPI edges.

APPROACH:
    1. Load PPI network and gene→pathway mappings
    2. For drug-disease pairs, compute:
       a) Raw PPI-extended overlap count
       b) Pathway-weighted overlap: 1.0 for same pathway, 0.5 otherwise
    3. Compare R@30 for both approaches

SUCCESS CRITERIA:
    >1pp improvement over raw PPI scoring (h96 found 6.49% for raw PPI)
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.disease_name_matcher import DiseaseMatcher, load_mesh_mappings

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"

SEEDS = [42, 123, 456, 789, 1024]


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


def load_gene_pathways() -> Dict[str, Set[str]]:
    """Load gene to KEGG pathway mappings."""
    with open(REFERENCE_DIR / "pathway" / "gene_pathways.json") as f:
        gene_pathways = json.load(f)
    return {gene: set(pathways) for gene, pathways in gene_pathways.items()}


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


def load_ground_truth(name_to_drug_id: Dict[str, str]) -> Tuple[Dict[str, Set[str]], Dict[str, str]]:
    """Load Every Cure ground truth."""
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


def get_ppi_extended_targets(
    drug_targets: Set[str],
    ppi_network: Dict[str, Set[str]],
    hops: int = 1
) -> Set[str]:
    """Extend drug targets through PPI network by N hops."""
    extended = set(drug_targets)
    current = set(drug_targets)

    for _ in range(hops):
        next_hop = set()
        for gene in current:
            if gene in ppi_network:
                next_hop.update(ppi_network[gene])
        extended.update(next_hop)
        current = next_hop

    return extended


def compute_raw_ppi_overlap(
    drug_targets_extended: Set[str],
    disease_genes: Set[str]
) -> int:
    """Compute raw overlap count."""
    return len(drug_targets_extended & disease_genes)


def compute_pathway_weighted_overlap(
    drug_targets_extended: Set[str],
    disease_genes: Set[str],
    gene_pathways: Dict[str, Set[str]],
    same_pathway_weight: float = 1.0,
    diff_pathway_weight: float = 0.5
) -> float:
    """
    Compute pathway-weighted overlap.

    For each overlapping gene:
    - If it shares a pathway with any drug target, weight = same_pathway_weight
    - Otherwise, weight = diff_pathway_weight
    """
    overlap = drug_targets_extended & disease_genes
    if not overlap:
        return 0.0

    # Get pathways for all original drug targets
    drug_pathways: Set[str] = set()
    for target in drug_targets_extended:
        if target in gene_pathways:
            drug_pathways.update(gene_pathways[target])

    weighted_sum = 0.0
    for gene in overlap:
        gene_paths = gene_pathways.get(gene, set())
        if gene_paths & drug_pathways:
            # Same pathway as drug target
            weighted_sum += same_pathway_weight
        else:
            # Connected via PPI but different pathway
            weighted_sum += diff_pathway_weight

    return weighted_sum


def evaluate_scoring(
    scoring_method: str,
    ground_truth: Dict[str, Set[str]],
    embeddings: Dict[str, np.ndarray],
    drug_targets: Dict[str, List[str]],
    disease_genes: Dict[str, List[str]],
    ppi_network: Dict[str, Set[str]],
    gene_pathways: Dict[str, Set[str]],
    seed: int = 42,
    top_k: int = 30
) -> float:
    """
    Evaluate a scoring method with disease holdout.

    Returns R@30.
    """
    np.random.seed(seed)
    diseases = list(ground_truth.keys())
    np.random.shuffle(diseases)
    n_test = len(diseases) // 5
    test_diseases = set(diseases[:n_test])
    train_diseases = set(diseases[n_test:])

    # Build train GT
    train_gt = {d: ground_truth[d] for d in train_diseases}

    # Get all drugs that appear in training GT
    train_drugs = set()
    for drugs in train_gt.values():
        train_drugs.update(drugs)

    # Filter to drugs with targets and embeddings
    candidate_drugs = [
        d for d in train_drugs
        if d in embeddings and d.split("::")[-1] in drug_targets
    ]

    hits = 0
    total = 0

    for disease_id in test_diseases:
        if disease_id not in embeddings:
            continue
        gt_drugs = ground_truth[disease_id] & set(candidate_drugs)
        if not gt_drugs:
            continue

        # Get disease genes
        dis_genes = set(disease_genes.get(disease_id.split("::")[-1], []))
        if not dis_genes:
            continue

        # Score all candidate drugs
        scores = []
        for drug_id in candidate_drugs:
            db_id = drug_id.split("::")[-1]
            targets = set(drug_targets.get(db_id, []))
            if not targets:
                continue

            # Extend targets via PPI
            extended = get_ppi_extended_targets(targets, ppi_network, hops=1)

            if scoring_method == "raw":
                score = compute_raw_ppi_overlap(extended, dis_genes)
            elif scoring_method == "pathway_weighted":
                score = compute_pathway_weighted_overlap(
                    extended, dis_genes, gene_pathways
                )
            else:
                raise ValueError(f"Unknown scoring method: {scoring_method}")

            scores.append((drug_id, score))

        # Rank by score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        top_drugs = set(d for d, _ in scores[:top_k])

        # Check hits
        for gt_drug in gt_drugs:
            total += 1
            if gt_drug in top_drugs:
                hits += 1

    return hits / total * 100 if total > 0 else 0.0


def main():
    print("h261: Pathway-Weighted PPI Scoring")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    embeddings = load_node2vec_embeddings()
    ppi_network = load_ppi_network()
    drug_targets = load_drug_targets()
    disease_genes = load_disease_genes()
    gene_pathways = load_gene_pathways()
    name_to_drug_id, id_to_name = load_drugbank_lookup()
    ground_truth, disease_names = load_ground_truth(name_to_drug_id)

    print(f"  Embeddings: {len(embeddings)}")
    print(f"  PPI network nodes: {len(ppi_network)}")
    print(f"  Drug targets: {len(drug_targets)}")
    print(f"  Disease genes: {len(disease_genes)}")
    print(f"  Gene pathways: {len(gene_pathways)}")
    print(f"  Ground truth diseases: {len(ground_truth)}")

    # Check pathway coverage
    ppi_genes = set()
    for gene, neighbors in ppi_network.items():
        ppi_genes.add(gene)
        ppi_genes.update(neighbors)

    genes_with_pathways = set(gene_pathways.keys())
    ppi_with_pathway = ppi_genes & genes_with_pathways
    print(f"\n  PPI genes with pathway info: {len(ppi_with_pathway)}/{len(ppi_genes)} ({len(ppi_with_pathway)/len(ppi_genes)*100:.1f}%)")

    # Evaluate both methods
    print("\n" + "=" * 70)
    print("Evaluating scoring methods (5-seed mean)")
    print("=" * 70)

    methods = ["raw", "pathway_weighted"]
    results = {m: [] for m in methods}

    for seed in SEEDS:
        print(f"\n  Seed {seed}:")
        for method in methods:
            r30 = evaluate_scoring(
                method, ground_truth, embeddings,
                drug_targets, disease_genes, ppi_network, gene_pathways,
                seed=seed
            )
            results[method].append(r30)
            print(f"    {method}: {r30:.2f}%")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for method in methods:
        mean = np.mean(results[method])
        std = np.std(results[method])
        print(f"\n{method}:")
        print(f"  R@30: {mean:.2f}% ± {std:.2f}%")
        print(f"  Per-seed: {[f'{r:.2f}' for r in results[method]]}")

    # Improvement
    raw_mean = np.mean(results["raw"])
    pw_mean = np.mean(results["pathway_weighted"])
    improvement = pw_mean - raw_mean

    print(f"\nImprovement (pathway_weighted - raw): {improvement:+.2f} pp")

    # Save results
    output = {
        "hypothesis": "h261",
        "raw_ppi_r30_mean": float(raw_mean),
        "raw_ppi_r30_std": float(np.std(results["raw"])),
        "pathway_weighted_r30_mean": float(pw_mean),
        "pathway_weighted_r30_std": float(np.std(results["pathway_weighted"])),
        "improvement_pp": float(improvement),
        "success": bool(improvement > 1.0),
        "per_seed_raw": results["raw"],
        "per_seed_pathway": results["pathway_weighted"],
    }

    output_file = ANALYSIS_DIR / "h261_pathway_weighted_ppi.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    if improvement > 1.0:
        print("\n✓ SUCCESS: Pathway weighting improved R@30 by >1pp")
    else:
        print("\n✗ FAIL: Pathway weighting did not improve R@30 by >1pp")


if __name__ == "__main__":
    main()
