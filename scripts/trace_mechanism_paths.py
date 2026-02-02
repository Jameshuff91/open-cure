#!/usr/bin/env python3
"""
Mechanism Tracing Pipeline.

PURPOSE:
    Transform black-box drug repurposing predictions into interpretable
    biological hypotheses by tracing Drug → Target → Pathway → Disease paths.

APPROACH:
    For each validated prediction:
    1. Get drug targets from drug_to_genes_drkg.json
    2. Get disease genes from disease_genes.json
    3. Find target-disease gene overlap (direct mechanism)
    4. Map to KEGG pathways via gene_pathways.json
    5. Generate biological hypothesis

PATH TYPES:
    1. DIRECT: Drug → Target Gene ← Disease Gene (same gene)
    2. PATHWAY: Drug → Gene → Pathway ← Gene ← Disease
    3. NETWORK: Drug → Gene → PPI → Gene ← Disease (future)

OUTPUT:
    - data/analysis/mechanism_tracings.json - All tracings
    - docs/mechanism_report.md - Human-readable report
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, List, Any, Optional

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
PATHWAY_DIR = REFERENCE_DIR / "pathway"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"
DOCS_DIR = PROJECT_ROOT / "docs"


# Key validated predictions
VALIDATED_PREDICTIONS = [
    {
        "drug": "Dantrolene",
        "disease": "Heart Failure",
        "mesh_id": "MESH:D006333",
        "evidence": "RCT P=0.034, 66% reduction in VT episodes",
        "expected_mechanism": "RYR2 inhibition prevents aberrant calcium release",
    },
    {
        "drug": "Lovastatin",
        "disease": "Multiple Myeloma",
        "mesh_id": "MESH:D009101",
        "evidence": "RCT: improved OS/PFS",
        "expected_mechanism": "HMG-CoA reductase inhibition affects cholesterol synthesis",
    },
    {
        "drug": "Rituximab",
        "disease": "Multiple Sclerosis",
        "mesh_id": "MESH:D009103",
        "evidence": "WHO Essential Medicine 2023",
        "expected_mechanism": "CD20 B-cell depletion reduces neuroinflammation",
    },
    {
        "drug": "Pitavastatin",
        "disease": "Rheumatoid Arthritis",
        "mesh_id": "MESH:D001172",
        "evidence": "Superior to MTX alone in trials",
        "expected_mechanism": "Anti-inflammatory via cholesterol pathway modulation",
    },
    {
        "drug": "Empagliflozin",
        "disease": "Parkinson's Disease",
        "mesh_id": "MESH:D010300",
        "evidence": "HR 0.80 in Korean observational study",
        "expected_mechanism": "SGLT2 inhibition may have neuroprotective effects",
    },
]


# KEGG pathway ID to name mapping (common pathways)
KEGG_PATHWAY_NAMES = {
    "hsa04020": "Calcium signaling pathway",
    "hsa04260": "Cardiac muscle contraction",
    "hsa04261": "Adrenergic signaling in cardiomyocytes",
    "hsa05410": "Hypertrophic cardiomyopathy",
    "hsa05414": "Dilated cardiomyopathy",
    "hsa04010": "MAPK signaling pathway",
    "hsa04151": "PI3K-Akt signaling pathway",
    "hsa04210": "Apoptosis",
    "hsa04630": "JAK-STAT signaling pathway",
    "hsa04064": "NF-kappa B signaling pathway",
    "hsa04062": "Chemokine signaling pathway",
    "hsa04660": "T cell receptor signaling",
    "hsa04662": "B cell receptor signaling",
    "hsa04670": "Leukocyte transendothelial migration",
    "hsa04612": "Antigen processing and presentation",
    "hsa04940": "Type I diabetes mellitus",
    "hsa04930": "Type II diabetes mellitus",
    "hsa05200": "Pathways in cancer",
    "hsa05202": "Transcriptional misregulation in cancer",
    "hsa05206": "MicroRNAs in cancer",
    "hsa04110": "Cell cycle",
    "hsa04115": "p53 signaling pathway",
    "hsa04350": "TGF-beta signaling pathway",
    "hsa05012": "Parkinson disease",
    "hsa05010": "Alzheimer disease",
    "hsa04728": "Dopaminergic synapse",
    "hsa04911": "Insulin secretion",
    "hsa04910": "Insulin signaling pathway",
    "hsa04922": "Glucagon signaling pathway",
    "hsa01100": "Metabolic pathways",
    "hsa00010": "Glycolysis / Gluconeogenesis",
    "hsa00020": "Citrate cycle (TCA cycle)",
    "hsa00100": "Steroid biosynthesis",
    "hsa00900": "Terpenoid backbone biosynthesis",
}


def get_pathway_name(pathway_id: str) -> str:
    """Get human-readable name for KEGG pathway."""
    return KEGG_PATHWAY_NAMES.get(pathway_id, pathway_id)


def load_drug_targets() -> Dict[str, Set[str]]:
    """Load drug -> target genes mapping."""
    with open(REFERENCE_DIR / "drug_to_genes_drkg.json") as f:
        data = json.load(f)

    # Parse to extract gene IDs
    result: Dict[str, Set[str]] = {}
    for drug_id, genes in data.items():
        gene_set: Set[str] = set()
        for gene in genes:
            # Extract numeric gene ID if present
            if gene.startswith("Gene::"):
                gene_id = gene.replace("Gene::", "")
                # Skip drugbank-specific IDs
                if not gene_id.startswith("drugbank:"):
                    gene_set.add(gene_id)
        if gene_set:
            result[drug_id] = gene_set

    return result


def load_disease_genes() -> Dict[str, Set[str]]:
    """Load disease -> associated genes mapping."""
    with open(REFERENCE_DIR / "disease_genes.json") as f:
        data = json.load(f)

    # Keys are like "MESH:D005909", values are lists of gene IDs
    result: Dict[str, Set[str]] = {}
    for mesh_id, genes in data.items():
        gene_set = {str(g) for g in genes if g}
        if gene_set:
            result[mesh_id] = gene_set

    return result


def load_gene_pathways() -> Dict[str, Set[str]]:
    """Load gene -> KEGG pathways mapping."""
    with open(PATHWAY_DIR / "gene_pathways.json") as f:
        data = json.load(f)
    return {k: set(v) for k, v in data.items()}


def load_drugbank_lookup() -> Dict[str, str]:
    """Load drug name -> DrugBank ID mapping."""
    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)

    name_to_id: Dict[str, str] = {}
    for db_id, name in id_to_name.items():
        name_to_id[name.lower()] = f"Compound::{db_id}"

    return name_to_id


def load_gene_info() -> Dict[str, str]:
    """Load gene ID -> symbol mapping from NCBI gene_info file."""
    gene_info_path = REFERENCE_DIR / "Homo_sapiens.gene_info"
    if not gene_info_path.exists():
        return {}

    gene_symbols: Dict[str, str] = {}
    with open(gene_info_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                gene_id = parts[1]
                symbol = parts[2]
                gene_symbols[gene_id] = symbol

    return gene_symbols


def trace_mechanism(
    drug_name: str,
    disease_mesh: str,
    drug_targets: Dict[str, Set[str]],
    disease_genes: Dict[str, Set[str]],
    gene_pathways: Dict[str, Set[str]],
    name_to_drug_id: Dict[str, str],
    gene_symbols: Dict[str, str],
) -> Dict[str, Any]:
    """
    Trace biological mechanism for a drug-disease prediction.

    Returns detailed mechanism breakdown.
    """
    result: Dict[str, Any] = {
        "drug": drug_name,
        "disease_mesh": disease_mesh,
        "found_drug": False,
        "found_disease_genes": False,
    }

    # Find drug
    drug_id = name_to_drug_id.get(drug_name.lower())
    if not drug_id:
        result["error"] = f"Drug '{drug_name}' not found in DrugBank"
        return result

    # Get drug targets
    targets = drug_targets.get(drug_id, set())
    if not targets:
        result["error"] = f"No targets found for {drug_id}"
        return result

    result["found_drug"] = True
    result["drug_id"] = drug_id
    result["drug_targets"] = list(targets)
    result["drug_target_symbols"] = [gene_symbols.get(t, t) for t in targets]
    result["n_targets"] = len(targets)

    # Get disease genes
    dg = disease_genes.get(disease_mesh, set())
    if not dg:
        result["error"] = f"No genes found for disease {disease_mesh}"
        return result

    result["found_disease_genes"] = True
    result["n_disease_genes"] = len(dg)

    # Direct mechanism: overlap between drug targets and disease genes
    overlap = targets & dg
    result["direct_overlap_genes"] = list(overlap)
    result["direct_overlap_symbols"] = [gene_symbols.get(g, g) for g in overlap]
    result["n_direct_overlap"] = len(overlap)

    if overlap:
        result["mechanism_type"] = "DIRECT"
        result["mechanism_description"] = (
            f"Drug directly targets {len(overlap)} disease-associated gene(s): "
            f"{', '.join(result['direct_overlap_symbols'][:5])}"
            + (f" and {len(overlap) - 5} more" if len(overlap) > 5 else "")
        )
    else:
        result["mechanism_type"] = "INDIRECT"

    # Pathway mechanism: shared pathways between drug targets and disease genes
    drug_pathways: Set[str] = set()
    for gene in targets:
        drug_pathways.update(gene_pathways.get(gene, set()))

    disease_pathways: Set[str] = set()
    for gene in dg:
        disease_pathways.update(gene_pathways.get(gene, set()))

    shared_pathways = drug_pathways & disease_pathways
    result["shared_pathways"] = [
        {"id": p, "name": get_pathway_name(p)}
        for p in sorted(shared_pathways)[:20]  # Top 20
    ]
    result["n_shared_pathways"] = len(shared_pathways)

    # Disease-relevant shared pathways (filter for interesting ones)
    relevant_keywords = [
        "signaling", "disease", "diabetes", "cancer", "apoptosis",
        "immune", "inflammation", "cardiac", "neuro", "parkinson",
    ]
    relevant_pathways = [
        {"id": p, "name": get_pathway_name(p)}
        for p in shared_pathways
        if any(kw in get_pathway_name(p).lower() for kw in relevant_keywords)
    ][:10]
    result["relevant_shared_pathways"] = relevant_pathways

    # Find bridge genes: drug targets in pathways shared with disease
    bridge_genes: Set[str] = set()
    for gene in targets:
        gene_paths = gene_pathways.get(gene, set())
        if gene_paths & disease_pathways:
            bridge_genes.add(gene)

    result["bridge_genes"] = list(bridge_genes)
    result["bridge_gene_symbols"] = [gene_symbols.get(g, g) for g in bridge_genes]
    result["n_bridge_genes"] = len(bridge_genes)

    # Generate hypothesis
    if overlap:
        hypothesis = (
            f"{drug_name} directly modulates {', '.join(result['direct_overlap_symbols'][:3])}, "
            f"which are implicated in {disease_mesh.replace('MESH:', '')} pathophysiology."
        )
    elif bridge_genes and shared_pathways:
        top_pathway = result["relevant_shared_pathways"][0]["name"] if result["relevant_shared_pathways"] else "shared metabolic pathways"
        hypothesis = (
            f"{drug_name} targets {', '.join(result['bridge_gene_symbols'][:3])}, "
            f"which participate in {top_pathway}, a pathway also dysregulated in the disease."
        )
    elif shared_pathways:
        top_pathway = result["relevant_shared_pathways"][0]["name"] if result["relevant_shared_pathways"] else get_pathway_name(list(shared_pathways)[0])
        hypothesis = (
            f"{drug_name} modulates genes in {top_pathway}, "
            f"which may intersect with disease mechanisms."
        )
    else:
        hypothesis = f"Mechanism unclear - no pathway overlap found between {drug_name} targets and disease genes."

    result["hypothesis"] = hypothesis

    return result


def generate_markdown_report(tracings: List[Dict[str, Any]], predictions: List[Dict]) -> str:
    """Generate human-readable markdown report."""
    lines = [
        "# Drug Repurposing Mechanism Tracings",
        "",
        "This report traces biological mechanisms for validated drug repurposing predictions.",
        "",
        "## Summary",
        "",
        f"- **Predictions analyzed:** {len(tracings)}",
        f"- **Direct mechanisms found:** {sum(1 for t in tracings if t.get('mechanism_type') == 'DIRECT')}",
        f"- **Pathway-based mechanisms:** {sum(1 for t in tracings if t.get('n_shared_pathways', 0) > 0)}",
        "",
        "---",
        "",
    ]

    for i, (tracing, pred) in enumerate(zip(tracings, predictions), 1):
        drug = tracing["drug"]
        disease = pred["disease"]

        lines.append(f"## {i}. {drug} → {disease}")
        lines.append("")

        if "error" in tracing:
            lines.append(f"**Error:** {tracing['error']}")
            lines.append("")
            continue

        lines.append(f"**Evidence:** {pred['evidence']}")
        lines.append("")

        # Drug targets
        targets = tracing.get("drug_target_symbols", [])[:10]
        lines.append(f"### Drug Targets ({tracing.get('n_targets', 0)} genes)")
        lines.append(f"Key targets: {', '.join(targets)}")
        lines.append("")

        # Disease genes
        lines.append(f"### Disease-Associated Genes ({tracing.get('n_disease_genes', 0)} genes)")
        lines.append("")

        # Direct overlap
        if tracing.get("n_direct_overlap", 0) > 0:
            overlap = tracing.get("direct_overlap_symbols", [])
            lines.append(f"### Direct Mechanism ✓")
            lines.append(f"**{tracing.get('n_direct_overlap', 0)} shared gene(s):** {', '.join(overlap[:10])}")
            lines.append("")

        # Shared pathways
        if tracing.get("n_shared_pathways", 0) > 0:
            lines.append(f"### Shared Pathways ({tracing.get('n_shared_pathways', 0)} total)")
            lines.append("")
            relevant = tracing.get("relevant_shared_pathways", [])
            if relevant:
                lines.append("**Disease-relevant pathways:**")
                for p in relevant[:5]:
                    lines.append(f"- {p['name']} ({p['id']})")
                lines.append("")

        # Bridge genes
        if tracing.get("n_bridge_genes", 0) > 0:
            bridge = tracing.get("bridge_gene_symbols", [])[:5]
            lines.append(f"### Bridge Genes")
            lines.append(f"Drug targets in disease-relevant pathways: {', '.join(bridge)}")
            lines.append("")

        # Hypothesis
        lines.append("### Biological Hypothesis")
        lines.append(f"> {tracing.get('hypothesis', 'Unknown')}")
        lines.append("")

        # Expected vs found
        expected = pred.get("expected_mechanism", "")
        if expected:
            lines.append(f"**Expected mechanism:** {expected}")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def main():
    print("=" * 70)
    print("MECHANISM TRACING PIPELINE")
    print("=" * 70)
    print()
    print("PURPOSE: Trace Drug → Target → Pathway → Disease paths")
    print("         to make predictions interpretable")
    print()

    # Load data
    print("[1] Loading data...")
    drug_targets = load_drug_targets()
    disease_genes = load_disease_genes()
    gene_pathways = load_gene_pathways()
    name_to_drug_id = load_drugbank_lookup()
    gene_symbols = load_gene_info()

    print(f"    Drugs with targets: {len(drug_targets)}")
    print(f"    Diseases with genes: {len(disease_genes)}")
    print(f"    Genes with pathways: {len(gene_pathways)}")
    print(f"    Gene symbols loaded: {len(gene_symbols)}")
    print()

    # Trace mechanisms
    print("[2] Tracing mechanisms...")
    print()

    tracings: List[Dict[str, Any]] = []

    for pred in VALIDATED_PREDICTIONS:
        drug = pred["drug"]
        mesh_id = pred["mesh_id"]

        print(f"  {drug} -> {pred['disease']}")

        tracing = trace_mechanism(
            drug, mesh_id,
            drug_targets, disease_genes, gene_pathways,
            name_to_drug_id, gene_symbols
        )

        if "error" in tracing:
            print(f"    [!] {tracing['error']}")
        else:
            print(f"    Targets: {tracing['n_targets']}")
            print(f"    Disease genes: {tracing['n_disease_genes']}")
            print(f"    Direct overlap: {tracing['n_direct_overlap']}")
            print(f"    Shared pathways: {tracing['n_shared_pathways']}")
            print(f"    Bridge genes: {tracing['n_bridge_genes']}")
            print(f"    Type: {tracing['mechanism_type']}")

        tracings.append(tracing)
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()

    success = [t for t in tracings if "error" not in t]
    direct = [t for t in success if t.get("mechanism_type") == "DIRECT"]
    with_pathways = [t for t in success if t.get("n_shared_pathways", 0) > 0]

    print(f"  Analyzed: {len(success)}/{len(tracings)}")
    print(f"  Direct mechanisms (gene overlap): {len(direct)}")
    print(f"  Pathway mechanisms: {len(with_pathways)}")
    print()

    if success:
        avg_overlap = sum(t.get("n_direct_overlap", 0) for t in success) / len(success)
        avg_pathways = sum(t.get("n_shared_pathways", 0) for t in success) / len(success)
        print(f"  Average direct overlap genes: {avg_overlap:.1f}")
        print(f"  Average shared pathways: {avg_pathways:.1f}")
        print()

    # Show hypotheses
    print("-" * 70)
    print("BIOLOGICAL HYPOTHESES")
    print("-" * 70)
    print()
    for pred, tracing in zip(VALIDATED_PREDICTIONS, tracings):
        if "error" not in tracing:
            print(f"  {pred['drug']} -> {pred['disease']}:")
            print(f"    {tracing['hypothesis']}")
            print()

    # Save JSON results
    output = {
        "analysis": "mechanism_tracing",
        "description": "Biological mechanism tracing for validated predictions",
        "n_predictions": len(VALIDATED_PREDICTIONS),
        "n_success": len(success),
        "n_direct_mechanisms": len(direct),
        "n_pathway_mechanisms": len(with_pathways),
        "tracings": tracings,
    }

    json_path = ANALYSIS_DIR / "mechanism_tracings.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"JSON saved to {json_path}")

    # Generate markdown report
    report = generate_markdown_report(tracings, VALIDATED_PREDICTIONS)
    md_path = DOCS_DIR / "mechanism_report.md"
    with open(md_path, "w") as f:
        f.write(report)
    print(f"Report saved to {md_path}")

    print(f"\n{'='*70}")
    print("CONCLUSION")
    print("=" * 70)
    print()
    print(f"Successfully traced mechanisms for {len(success)}/{len(tracings)} predictions.")
    if len(direct) > 0:
        print(f"{len(direct)} have DIRECT gene overlap - strongest mechanistic evidence.")
    if len(with_pathways) > 0:
        print(f"{len(with_pathways)} have shared pathways - pathway-based mechanism.")


if __name__ == "__main__":
    main()
