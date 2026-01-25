#!/usr/bin/env python3
"""
Extract drug-target and disease-gene relationships from DRKG.

This creates lookup tables for:
1. drug → set of target genes
2. disease → set of associated genes

These can be used to compute target-based features for drug repurposing.
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set

PROJECT_ROOT = Path(__file__).parent.parent
DRKG_PATH = PROJECT_ROOT / "data" / "raw" / "drkg" / "drkg.tsv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "reference"


def extract_drug_targets() -> Dict[str, Set[str]]:
    """Extract drug → target gene mappings from DRKG."""

    # Relations that indicate drug-target interactions
    target_relations = {
        "DRUGBANK::target::Compound:Gene",
        "DRUGBANK::enzyme::Compound:Gene",
        "DGIDB::INHIBITOR::Gene:Compound",
        "DGIDB::AGONIST::Gene:Compound",
        "DGIDB::ANTAGONIST::Gene:Compound",
        "DGIDB::OTHER::Gene:Compound",
        "Hetionet::CdG::Compound:Gene",  # Compound downregulates Gene
        "Hetionet::CuG::Compound:Gene",  # Compound upregulates Gene
        "Hetionet::CbG::Compound:Gene",  # Compound binds Gene
    }

    drug_targets: Dict[str, Set[str]] = defaultdict(set)

    with open(DRKG_PATH) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue

            head, relation, tail = parts

            if relation in target_relations:
                # Determine which is drug and which is gene
                if head.startswith("Compound::"):
                    drug_id = head.replace("Compound::", "")
                    gene_id = tail.replace("Gene::", "")
                elif tail.startswith("Compound::"):
                    drug_id = tail.replace("Compound::", "")
                    gene_id = head.replace("Gene::", "")
                else:
                    continue

                drug_targets[drug_id].add(gene_id)

    return drug_targets


def extract_disease_genes() -> Dict[str, Set[str]]:
    """Extract disease → associated gene mappings from DRKG."""

    # Relations that indicate disease-gene associations
    gene_relations = {
        "GNBR::L::Gene:Disease",  # Associated
        "GNBR::J::Gene:Disease",  # Associated
        "GNBR::U::Gene:Disease",  # Associated
        "Hetionet::DaG::Disease:Gene",  # Disease associates Gene
        "Hetionet::DuG::Disease:Gene",  # Disease upregulates Gene
        "Hetionet::DdG::Disease:Gene",  # Disease downregulates Gene
    }

    disease_genes: Dict[str, Set[str]] = defaultdict(set)

    with open(DRKG_PATH) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue

            head, relation, tail = parts

            if relation in gene_relations:
                # Determine which is disease and which is gene
                if head.startswith("Disease::"):
                    disease_id = head.replace("Disease::", "")
                    gene_id = tail.replace("Gene::", "")
                elif tail.startswith("Disease::"):
                    disease_id = tail.replace("Disease::", "")
                    gene_id = head.replace("Gene::", "")
                else:
                    continue

                disease_genes[disease_id].add(gene_id)

    return disease_genes


def compute_target_overlap(
    drug_id: str,
    disease_id: str,
    drug_targets: Dict[str, Set[str]],
    disease_genes: Dict[str, Set[str]],
) -> Dict[str, float]:
    """
    Compute target-based features for a drug-disease pair.

    Returns:
        dict with features:
        - target_overlap: number of shared genes
        - target_overlap_frac: fraction of drug targets that are disease genes
        - has_target_overlap: binary indicator
    """
    drug_genes = drug_targets.get(drug_id, set())
    dis_genes = disease_genes.get(disease_id, set())

    overlap = drug_genes & dis_genes
    n_overlap = len(overlap)

    if len(drug_genes) > 0:
        frac = n_overlap / len(drug_genes)
    else:
        frac = 0.0

    return {
        "target_overlap": n_overlap,
        "target_overlap_frac": frac,
        "has_target_overlap": 1 if n_overlap > 0 else 0,
        "n_drug_targets": len(drug_genes),
        "n_disease_genes": len(dis_genes),
    }


def main():
    print("Extracting drug-target relationships...")
    drug_targets = extract_drug_targets()
    print(f"  Found targets for {len(drug_targets)} drugs")

    # Count total targets
    total_targets = sum(len(t) for t in drug_targets.values())
    print(f"  Total drug-target edges: {total_targets}")

    print("\nExtracting disease-gene relationships...")
    disease_genes = extract_disease_genes()
    print(f"  Found gene associations for {len(disease_genes)} diseases")

    total_genes = sum(len(g) for g in disease_genes.values())
    print(f"  Total disease-gene edges: {total_genes}")

    # Save to JSON (convert sets to lists for JSON serialization)
    drug_targets_json = {k: list(v) for k, v in drug_targets.items()}
    disease_genes_json = {k: list(v) for k, v in disease_genes.items()}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_DIR / "drug_targets.json", "w") as f:
        json.dump(drug_targets_json, f)
    print(f"\nSaved drug targets to {OUTPUT_DIR / 'drug_targets.json'}")

    with open(OUTPUT_DIR / "disease_genes.json", "w") as f:
        json.dump(disease_genes_json, f)
    print(f"Saved disease genes to {OUTPUT_DIR / 'disease_genes.json'}")

    # Print some statistics
    print("\nTop 10 drugs by number of targets:")
    for drug, targets in sorted(drug_targets.items(), key=lambda x: -len(x[1]))[:10]:
        print(f"  {drug}: {len(targets)} targets")

    print("\nTop 10 diseases by number of associated genes:")
    for disease, genes in sorted(disease_genes.items(), key=lambda x: -len(x[1]))[:10]:
        print(f"  {disease}: {len(genes)} genes")


if __name__ == "__main__":
    main()
