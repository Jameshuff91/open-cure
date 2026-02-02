#!/usr/bin/env python3
"""
Query drugs by gene or pathway for drug repurposing.

This script enables the Gene -> Pathway -> Drug workflow for rare genetic diseases.

Examples:
    python scripts/query_by_gene.py SLURP1                     # Direct gene lookup
    python scripts/query_by_gene.py EGFR --top 30              # Limit results
    python scripts/query_by_gene.py --pathway hsa04064         # NF-kB pathway drugs
    python scripts/query_by_gene.py SLURP1 --include-pathway   # Gene + pathway expansion
    python scripts/query_by_gene.py --list-pathways EGFR       # Show pathways for a gene
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
PATHWAY_DIR = REFERENCE_DIR / "pathway"

# KEGG pathway names
KEGG_PATHWAY_NAMES = {
    "hsa04064": "NF-kappa B signaling pathway",
    "hsa04010": "MAPK signaling pathway",
    "hsa04151": "PI3K-Akt signaling pathway",
    "hsa04012": "ErbB signaling pathway",
    "hsa04020": "Calcium signaling pathway",
    "hsa04630": "JAK-STAT signaling pathway",
    "hsa04660": "T cell receptor signaling",
    "hsa04662": "B cell receptor signaling",
    "hsa04210": "Apoptosis",
    "hsa04115": "p53 signaling pathway",
    "hsa04350": "TGF-beta signaling pathway",
    "hsa05200": "Pathways in cancer",
    "hsa05012": "Parkinson disease",
    "hsa05010": "Alzheimer disease",
    "hsa04910": "Insulin signaling pathway",
    "hsa04930": "Type II diabetes mellitus",
    "hsa04940": "Type I diabetes mellitus",
    "hsa04260": "Cardiac muscle contraction",
    "hsa04261": "Adrenergic signaling in cardiomyocytes",
    "hsa04062": "Chemokine signaling pathway",
    "hsa04728": "Dopaminergic synapse",
    "hsa01100": "Metabolic pathways",
    "hsa04080": "Neuroactive ligand-receptor interaction",
}


def load_gene_lookup() -> Tuple[Dict[str, str], Dict[str, Dict]]:
    """
    Load gene symbol -> ID and ID -> info mappings.
    Returns: (symbol_to_id, id_to_info)
    """
    with open(REFERENCE_DIR / "gene_lookup.json") as f:
        data = json.load(f)

    symbol_to_id: Dict[str, str] = {}
    id_to_info: Dict[str, Dict] = {}

    for gene_id, info in data.items():
        symbol = info.get("symbol", "")
        if symbol:
            symbol_to_id[symbol.upper()] = gene_id
            id_to_info[gene_id] = info

    return symbol_to_id, id_to_info


def load_gene_to_drugs() -> Dict[str, List[str]]:
    """Load gene ID -> drug list mapping."""
    with open(REFERENCE_DIR / "gene_to_drugs_drkg.json") as f:
        data = json.load(f)

    # Convert keys from "Gene::1956" to "1956"
    result: Dict[str, List[str]] = {}
    for key, drugs in data.items():
        gene_id = key.replace("Gene::", "")
        result[gene_id] = drugs

    return result


def load_drugbank_lookup() -> Dict[str, str]:
    """Load DrugBank ID -> drug name mapping."""
    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        return json.load(f)


def load_gene_pathways() -> Dict[str, List[str]]:
    """Load gene ID -> pathway list mapping."""
    with open(PATHWAY_DIR / "gene_pathways.json") as f:
        return json.load(f)


def get_pathway_name(pathway_id: str) -> str:
    """Get human-readable name for KEGG pathway."""
    return KEGG_PATHWAY_NAMES.get(pathway_id, pathway_id)


def extract_drugbank_id(compound_id: str) -> Optional[str]:
    """Extract DrugBank ID from compound ID like 'Compound::DB00001'."""
    if "::DB" in compound_id:
        parts = compound_id.split("::")
        for part in parts:
            if part.startswith("DB"):
                return part
    return None


def query_by_gene(
    gene_symbol: str,
    gene_to_drugs: Dict[str, List[str]],
    symbol_to_id: Dict[str, str],
    drugbank_lookup: Dict[str, str],
    top_k: int = 50,
) -> List[Dict]:
    """
    Get drugs targeting a specific gene.

    Returns list of dicts with: drug_name, drugbank_id, compound_id
    """
    gene_symbol_upper = gene_symbol.upper()
    gene_id = symbol_to_id.get(gene_symbol_upper)

    if not gene_id:
        print(f"Gene '{gene_symbol}' not found in database.")
        # Try partial match
        matches = [s for s in symbol_to_id.keys() if gene_symbol_upper in s]
        if matches:
            print("Did you mean:")
            for m in matches[:10]:
                print(f"  - {m}")
        return []

    drugs = gene_to_drugs.get(gene_id, [])
    if not drugs:
        print(f"No drugs found targeting gene {gene_symbol} (ID: {gene_id})")
        return []

    results: List[Dict] = []
    for compound_id in drugs:
        db_id = extract_drugbank_id(compound_id)
        drug_name = drugbank_lookup.get(db_id, "") if db_id else ""

        results.append({
            "drug_name": drug_name,
            "drugbank_id": db_id or "",
            "compound_id": compound_id,
            "source": "direct",
        })

    # Sort: DrugBank drugs with names first
    results.sort(key=lambda x: (not x["drug_name"], x["drug_name"] or x["compound_id"]))

    return results[:top_k]


def query_by_pathway(
    pathway_id: str,
    gene_pathways: Dict[str, List[str]],
    gene_to_drugs: Dict[str, List[str]],
    drugbank_lookup: Dict[str, str],
    id_to_info: Dict[str, Dict],
    top_k: int = 100,
) -> Tuple[List[Dict], List[str]]:
    """
    Get all drugs targeting genes in a pathway.

    Returns: (drug_list, gene_list)
    """
    # Find genes in this pathway
    genes_in_pathway = [g for g, pathways in gene_pathways.items() if pathway_id in pathways]

    if not genes_in_pathway:
        print(f"No genes found in pathway {pathway_id}")
        return [], []

    # Collect all drugs and track which gene they target
    drug_gene_map: Dict[str, Set[str]] = defaultdict(set)

    for gene_id in genes_in_pathway:
        drugs = gene_to_drugs.get(gene_id, [])
        gene_symbol = id_to_info.get(gene_id, {}).get("symbol", gene_id)
        for compound_id in drugs:
            drug_gene_map[compound_id].add(gene_symbol)

    # Build results
    results: List[Dict] = []
    for compound_id, target_genes in drug_gene_map.items():
        db_id = extract_drugbank_id(compound_id)
        drug_name = drugbank_lookup.get(db_id, "") if db_id else ""

        results.append({
            "drug_name": drug_name,
            "drugbank_id": db_id or "",
            "compound_id": compound_id,
            "target_genes": sorted(target_genes),
            "n_targets": len(target_genes),
            "source": "pathway",
        })

    # Sort by: has name, then by number of targets (more targets = more relevant)
    results.sort(key=lambda x: (not x["drug_name"], -x["n_targets"], x["drug_name"] or x["compound_id"]))

    gene_symbols = [id_to_info.get(g, {}).get("symbol", g) for g in genes_in_pathway]

    return results[:top_k], gene_symbols


def list_pathways_for_gene(
    gene_symbol: str,
    symbol_to_id: Dict[str, str],
    gene_pathways: Dict[str, List[str]],
) -> List[Dict]:
    """List all pathways containing a gene."""
    gene_symbol_upper = gene_symbol.upper()
    gene_id = symbol_to_id.get(gene_symbol_upper)

    if not gene_id:
        print(f"Gene '{gene_symbol}' not found.")
        return []

    pathways = gene_pathways.get(gene_id, [])
    results = [
        {"pathway_id": p, "pathway_name": get_pathway_name(p)}
        for p in pathways
    ]

    return results


def print_gene_results(gene: str, results: List[Dict]):
    """Pretty print gene query results."""
    db_results = [r for r in results if r["drugbank_id"]]
    other_results = [r for r in results if not r["drugbank_id"]]

    print(f"\nDrugs targeting {gene}")
    print("=" * 70)
    print(f"Total: {len(results)} ({len(db_results)} DrugBank, {len(other_results)} other)")
    print()

    if db_results:
        print("DrugBank Drugs:")
        print(f"{'Drug Name':<35} {'DrugBank ID':<15}")
        print("-" * 50)
        for r in db_results[:30]:
            print(f"{r['drug_name']:<35} {r['drugbank_id']:<15}")
        if len(db_results) > 30:
            print(f"... and {len(db_results) - 30} more")
    print()


def print_pathway_results(pathway_id: str, results: List[Dict], genes: List[str]):
    """Pretty print pathway query results."""
    db_results = [r for r in results if r["drugbank_id"]]
    pathway_name = get_pathway_name(pathway_id)

    print(f"\nDrugs targeting {pathway_name} ({pathway_id})")
    print("=" * 80)
    print(f"Genes in pathway: {len(genes)}")
    print(f"Total drugs: {len(results)} ({len(db_results)} DrugBank)")
    print()

    if db_results:
        print(f"{'Drug Name':<30} {'DrugBank ID':<12} {'Targets':<5} {'Target Genes (sample)'}")
        print("-" * 80)
        for r in db_results[:50]:
            genes_str = ", ".join(r["target_genes"][:3])
            if len(r["target_genes"]) > 3:
                genes_str += f" (+{len(r['target_genes']) - 3})"
            print(f"{r['drug_name']:<30} {r['drugbank_id']:<12} {r['n_targets']:<5} {genes_str}")
        if len(db_results) > 50:
            print(f"... and {len(db_results) - 50} more")
    print()


def print_pathway_list(gene: str, pathways: List[Dict]):
    """Print pathways for a gene."""
    print(f"\nPathways containing {gene}")
    print("=" * 60)
    print(f"Total: {len(pathways)}")
    print()
    for p in pathways:
        print(f"  {p['pathway_id']}: {p['pathway_name']}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Query drugs by gene or pathway",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/query_by_gene.py SLURP1                     # Direct gene lookup
  python scripts/query_by_gene.py EGFR --top 30              # Limit results
  python scripts/query_by_gene.py --pathway hsa04064         # NF-kB pathway drugs
  python scripts/query_by_gene.py SLURP1 --include-pathway   # Gene + pathway expansion
  python scripts/query_by_gene.py --list-pathways EGFR       # Show pathways for a gene
        """
    )

    parser.add_argument("gene", nargs="?", help="Gene symbol (e.g., SLURP1, EGFR)")
    parser.add_argument("--pathway", "-p", help="KEGG pathway ID (e.g., hsa04064 for NF-kB)")
    parser.add_argument("--include-pathway", action="store_true",
                        help="Include drugs from gene's pathways")
    parser.add_argument("--list-pathways", metavar="GENE",
                        help="List pathways containing this gene")
    parser.add_argument("--top", "-n", type=int, default=100,
                        help="Max results (default: 100)")
    parser.add_argument("--json", "-j", action="store_true",
                        help="Output as JSON")

    args = parser.parse_args()

    if not args.gene and not args.pathway and not args.list_pathways:
        parser.print_help()
        sys.exit(1)

    # Load data
    if not args.json:
        print("Loading data...")
    symbol_to_id, id_to_info = load_gene_lookup()
    gene_to_drugs = load_gene_to_drugs()
    drugbank_lookup = load_drugbank_lookup()
    gene_pathways = load_gene_pathways()
    if not args.json:
        print(f"Loaded {len(symbol_to_id)} genes, {len(drugbank_lookup)} drugs")

    # List pathways mode
    if args.list_pathways:
        pathways = list_pathways_for_gene(args.list_pathways, symbol_to_id, gene_pathways)
        if pathways:
            if args.json:
                print(json.dumps(pathways, indent=2))
            else:
                print_pathway_list(args.list_pathways, pathways)
        return

    # Pathway query mode
    if args.pathway:
        results, genes = query_by_pathway(
            args.pathway, gene_pathways, gene_to_drugs,
            drugbank_lookup, id_to_info, args.top
        )
        if results:
            if args.json:
                print(json.dumps({"pathway": args.pathway, "genes": genes, "drugs": results}, indent=2))
            else:
                print_pathway_results(args.pathway, results, genes)
        return

    # Gene query mode
    if args.gene:
        results = query_by_gene(
            args.gene, gene_to_drugs, symbol_to_id,
            drugbank_lookup, args.top
        )

        # Optionally expand to pathway drugs
        if args.include_pathway:
            gene_id = symbol_to_id.get(args.gene.upper())
            if gene_id:
                pathways = gene_pathways.get(gene_id, [])
                if not args.json:
                    print(f"\nExpanding to {len(pathways)} pathways...")
                all_pathway_drugs: Dict[str, Dict] = {}
                for pathway_id in pathways:
                    p_results, _ = query_by_pathway(
                        pathway_id, gene_pathways, gene_to_drugs,
                        drugbank_lookup, id_to_info, 9999
                    )
                    for r in p_results:
                        if r["compound_id"] not in all_pathway_drugs:
                            r["source"] = f"pathway:{pathway_id}"
                            all_pathway_drugs[r["compound_id"]] = r

                # Add pathway drugs not in direct results
                direct_ids = {r["compound_id"] for r in results}
                pathway_only = [r for cid, r in all_pathway_drugs.items() if cid not in direct_ids]
                pathway_only.sort(key=lambda x: (not x["drug_name"], x["drug_name"] or x["compound_id"]))

                if not args.json:
                    print(f"Found {len(pathway_only)} additional drugs via pathways")
                results.extend(pathway_only[:args.top - len(results)])

        if results:
            if args.json:
                print(json.dumps(results, indent=2))
            else:
                print_gene_results(args.gene, results)


if __name__ == "__main__":
    main()
