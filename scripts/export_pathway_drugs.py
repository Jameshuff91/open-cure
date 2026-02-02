#!/usr/bin/env python3
"""
Export drugs targeting specific pathways for Ryland Mortlock collaboration.

Creates Excel files with all drugs targeting genes in:
1. NF-kB signaling pathway (hsa04064) - for SLURP1 patients
2. ErbB/EGFR signaling pathway (hsa04012) - validation set

Output:
    data/exports/nfkb_pathway_drugs.xlsx
    data/exports/egfr_pathway_drugs.xlsx

These files enable the Gene -> Pathway -> Drug workflow for rare genetic diseases.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
PATHWAY_DIR = REFERENCE_DIR / "pathway"
EXPORTS_DIR = PROJECT_ROOT / "data" / "exports"

# Pathway configurations
PATHWAYS_TO_EXPORT = {
    "nfkb": {
        "id": "hsa04064",
        "name": "NF-kappa B signaling pathway",
        "use_case": "SLURP1 patients (rare skin disease)",
    },
    "egfr": {
        "id": "hsa04012",
        "name": "ErbB signaling pathway",
        "use_case": "Validation (includes EGFR inhibitors like erlotinib)",
    },
}

# Gene role descriptions for NF-kB pathway (key genes)
NFKB_GENE_ROLES = {
    "NFKB1": "NF-kB p105/p50 subunit - central transcription factor",
    "NFKB2": "NF-kB p100/p52 subunit - alternative pathway",
    "RELA": "p65/RelA subunit - main transactivation domain",
    "RELB": "RelB subunit - alternative pathway",
    "IKBKB": "IKK-beta - key kinase in canonical pathway",
    "IKBKG": "NEMO/IKK-gamma - regulatory subunit",
    "CHUK": "IKK-alpha - kinase in alternative pathway",
    "TNFRSF1A": "TNF receptor 1 - upstream activator",
    "TRAF6": "E3 ligase - signal transducer",
    "MYD88": "Adaptor protein - TLR/IL-1R signaling",
    "IRAK1": "IL-1R-associated kinase 1",
    "BCL2": "Anti-apoptotic protein",
    "TP53": "Tumor suppressor - crosstalk with NF-kB",
}


def load_gene_pathways() -> Dict[str, List[str]]:
    """Load gene ID -> pathway list mapping."""
    with open(PATHWAY_DIR / "gene_pathways.json") as f:
        return json.load(f)


def load_gene_to_drugs() -> Dict[str, List[str]]:
    """Load gene ID -> drug list mapping."""
    with open(REFERENCE_DIR / "gene_to_drugs_drkg.json") as f:
        data = json.load(f)
    # Convert keys from "Gene::1956" to "1956"
    return {k.replace("Gene::", ""): v for k, v in data.items()}


def load_gene_lookup() -> Dict[str, Dict]:
    """Load gene ID -> info mapping."""
    with open(REFERENCE_DIR / "gene_lookup.json") as f:
        return json.load(f)


def load_drugbank_lookup() -> Dict[str, str]:
    """Load DrugBank ID -> drug name mapping."""
    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        return json.load(f)


def extract_drugbank_id(compound_id: str) -> str:
    """Extract DrugBank ID from compound ID."""
    if "::DB" in compound_id:
        parts = compound_id.split("::")
        for part in parts:
            if part.startswith("DB"):
                return part
    return ""


def get_pathway_drugs(
    pathway_id: str,
    gene_pathways: Dict[str, List[str]],
    gene_to_drugs: Dict[str, List[str]],
    gene_lookup: Dict[str, Dict],
    drugbank_lookup: Dict[str, str],
    gene_roles: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Get all drugs targeting genes in a pathway.

    Returns DataFrame with:
    - drug_name, drugbank_id
    - target_gene, target_gene_id
    - n_pathway_targets (how many pathway genes this drug targets)
    - gene_role (if provided)
    """
    # Find genes in pathway
    genes_in_pathway = [g for g, pathways in gene_pathways.items() if pathway_id in pathways]

    # Collect drugs and their targets
    drug_targets: Dict[str, Set[str]] = defaultdict(set)

    for gene_id in genes_in_pathway:
        gene_symbol = gene_lookup.get(gene_id, {}).get("symbol", gene_id)
        drugs = gene_to_drugs.get(gene_id, [])

        for compound_id in drugs:
            db_id = extract_drugbank_id(compound_id)
            if db_id:  # Only include DrugBank drugs
                drug_targets[db_id].add(gene_symbol)

    # Build DataFrame
    records = []
    for db_id, targets in drug_targets.items():
        drug_name = drugbank_lookup.get(db_id, db_id)

        for target in sorted(targets):
            role = ""
            if gene_roles and target in gene_roles:
                role = gene_roles[target]

            records.append({
                "drug_name": drug_name,
                "drugbank_id": db_id,
                "target_gene": target,
                "gene_role": role,
                "n_pathway_targets": len(targets),
            })

    df = pd.DataFrame(records)

    # Sort by number of targets (drugs hitting more pathway genes first)
    if len(df) > 0:
        df = df.sort_values(
            by=["n_pathway_targets", "drug_name", "target_gene"],
            ascending=[False, True, True]
        )

    return df


def create_summary_sheet(
    df: pd.DataFrame,
    pathway_name: str,
    use_case: str,
    genes_in_pathway: int,
) -> pd.DataFrame:
    """Create a summary sheet for the export."""
    unique_drugs = df["drugbank_id"].nunique() if len(df) > 0 else 0
    unique_genes_targeted = df["target_gene"].nunique() if len(df) > 0 else 0

    summary_data = {
        "Metric": [
            "Pathway Name",
            "Pathway ID",
            "Use Case",
            "Genes in Pathway",
            "Genes with Drug Targets",
            "Unique DrugBank Drugs",
            "Total Drug-Gene Pairs",
        ],
        "Value": [
            pathway_name,
            df.attrs.get("pathway_id", ""),
            use_case,
            genes_in_pathway,
            unique_genes_targeted,
            unique_drugs,
            len(df),
        ],
    }

    return pd.DataFrame(summary_data)


def main():
    print("=" * 70)
    print("PATHWAY DRUGS EXPORT")
    print("=" * 70)
    print()

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    gene_pathways = load_gene_pathways()
    gene_to_drugs = load_gene_to_drugs()
    gene_lookup = load_gene_lookup()
    drugbank_lookup = load_drugbank_lookup()

    print(f"  Genes with pathway annotations: {len(gene_pathways)}")
    print(f"  Genes with drug targets: {len(gene_to_drugs)}")
    print(f"  DrugBank drugs: {len(drugbank_lookup)}")
    print()

    # Export each pathway
    for key, config in PATHWAYS_TO_EXPORT.items():
        pathway_id = config["id"]
        pathway_name = config["name"]
        use_case = config["use_case"]

        print(f"Exporting {pathway_name}...")

        # Get gene roles if available
        gene_roles = NFKB_GENE_ROLES if key == "nfkb" else None

        # Get pathway drugs
        df = get_pathway_drugs(
            pathway_id,
            gene_pathways,
            gene_to_drugs,
            gene_lookup,
            drugbank_lookup,
            gene_roles,
        )
        df.attrs["pathway_id"] = pathway_id

        # Count genes in pathway
        genes_in_pathway = sum(1 for g, p in gene_pathways.items() if pathway_id in p)

        print(f"  Genes in pathway: {genes_in_pathway}")
        print(f"  DrugBank drugs: {df['drugbank_id'].nunique()}")
        print(f"  Drug-gene pairs: {len(df)}")

        # Create summary
        summary_df = create_summary_sheet(df, pathway_name, use_case, genes_in_pathway)

        # Create drug-level summary (unique drugs with all their targets)
        drug_summary = df.groupby(["drug_name", "drugbank_id"]).agg({
            "target_gene": lambda x: ", ".join(sorted(x)),
            "n_pathway_targets": "first"
        }).reset_index()
        drug_summary.columns = ["drug_name", "drugbank_id", "target_genes", "n_targets"]
        drug_summary = drug_summary.sort_values("n_targets", ascending=False)

        # Export to Excel with multiple sheets
        output_path = EXPORTS_DIR / f"{key}_pathway_drugs.xlsx"

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            drug_summary.to_excel(writer, sheet_name="Drugs", index=False)
            df.to_excel(writer, sheet_name="Drug-Gene Pairs", index=False)

        print(f"  Saved: {output_path}")
        print()

    # Also export EGFR direct drugs (not pathway, just EGFR gene itself)
    print("Exporting EGFR direct drugs (validation)...")
    egfr_gene_id = None
    for gene_id, info in gene_lookup.items():
        if info.get("symbol") == "EGFR":
            egfr_gene_id = gene_id
            break

    if egfr_gene_id:
        egfr_drugs = gene_to_drugs.get(egfr_gene_id, [])
        egfr_records = []
        for compound_id in egfr_drugs:
            db_id = extract_drugbank_id(compound_id)
            if db_id:
                drug_name = drugbank_lookup.get(db_id, db_id)
                egfr_records.append({
                    "drug_name": drug_name,
                    "drugbank_id": db_id,
                    "compound_id": compound_id,
                })

        egfr_df = pd.DataFrame(egfr_records).drop_duplicates(subset=["drugbank_id"])
        egfr_df = egfr_df.sort_values("drug_name")

        egfr_output = EXPORTS_DIR / "egfr_direct_drugs.xlsx"
        egfr_df.to_excel(egfr_output, index=False)
        print(f"  EGFR direct drugs: {len(egfr_df)}")
        print(f"  Saved: {egfr_output}")

        # Verify erlotinib is present
        erlotinib_found = any(egfr_df["drug_name"].str.lower() == "erlotinib")
        print(f"  Erlotinib present: {erlotinib_found}")
    print()

    # Summary
    print("=" * 70)
    print("EXPORT COMPLETE")
    print("=" * 70)
    print()
    print(f"Export directory: {EXPORTS_DIR}")
    print()
    print("Files created:")
    print("  - nfkb_pathway_drugs.xlsx (for SLURP1 patients)")
    print("  - egfr_pathway_drugs.xlsx (ErbB pathway)")
    print("  - egfr_direct_drugs.xlsx (validation set)")


if __name__ == "__main__":
    main()
