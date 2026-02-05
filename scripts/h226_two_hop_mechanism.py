#!/usr/bin/env python3
"""
h226: Two-Hop Mechanism Paths (Drug->Gene1->Gene2->Disease)

PURPOSE:
    h166 found that direct mechanism paths (drug->gene->disease) have:
    - 22.1% coverage (2,975/13,461 predictions)
    - 2.2x precision lift (13.58% vs 6.17%)

    This experiment extends to 2-hop paths through PPI:
    - Drug targets gene1, gene1 interacts with gene2, gene2 is disease-associated

    QUESTION: Does 2-hop extend coverage while maintaining precision?

APPROACH:
    1. Load 1-hop mechanism path coverage from h166
    2. For predictions WITHOUT 1-hop paths, check for 2-hop PPI paths
    3. Compare precision of 1-hop vs 2-hop paths
    4. Determine if 2-hop adds value or just noise
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, List, Tuple, Any

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
PPI_DIR = REFERENCE_DIR / "ppi"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"
DELIVERABLE_DIR = PROJECT_ROOT / "data" / "deliverables"


def load_ppi_network() -> Dict[str, Set[str]]:
    """Load PPI network (gene -> interacting genes)."""
    with open(PPI_DIR / "ppi_network_high_conf.json") as f:
        ppi = json.load(f)
    # Convert to sets for faster lookup
    return {k: set(v) for k, v in ppi.items()}


def load_drug_targets() -> Dict[str, Set[str]]:
    """Load drug -> target genes mapping."""
    with open(REFERENCE_DIR / "drug_to_genes_drkg.json") as f:
        data = json.load(f)

    result: Dict[str, Set[str]] = {}
    for drug_id, genes in data.items():
        gene_set: Set[str] = set()
        for gene in genes:
            if gene.startswith("Gene::"):
                gene_id = gene.replace("Gene::", "")
                if not gene_id.startswith("drugbank:"):
                    gene_set.add(gene_id)
        if gene_set:
            result[drug_id] = gene_set
    return result


def load_disease_genes() -> Dict[str, Set[str]]:
    """Load disease -> associated genes mapping."""
    with open(REFERENCE_DIR / "disease_genes.json") as f:
        data = json.load(f)
    return {k: set(str(g) for g in v if g) for k, v in data.items()}


def load_drugbank_lookup() -> Dict[str, str]:
    """Load drug name -> DrugBank ID mapping."""
    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)
    return {name.lower(): f"Compound::{db_id}" for db_id, name in id_to_name.items()}


def load_production_predictions() -> List[Dict]:
    """Load production predictions with ground truth status."""
    # Use JSON file (faster than xlsx)
    json_path = DELIVERABLE_DIR / "drug_repurposing_predictions_with_confidence.json"
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)

    raise FileNotFoundError("No production predictions found")


def find_1hop_paths(
    drug_targets: Set[str],
    disease_genes: Set[str]
) -> Set[str]:
    """Find direct drug->gene->disease connections (1-hop)."""
    return drug_targets & disease_genes


def find_2hop_paths(
    drug_targets: Set[str],
    disease_genes: Set[str],
    ppi: Dict[str, Set[str]]
) -> List[Tuple[str, str]]:
    """
    Find 2-hop paths: drug->gene1->gene2->disease.

    Returns list of (gene1, gene2) pairs where:
    - gene1 is a drug target
    - gene2 is disease-associated
    - gene1 and gene2 interact in PPI network
    """
    two_hop_paths = []

    for target in drug_targets:
        # Skip if this target is already a direct hit (1-hop)
        if target in disease_genes:
            continue

        # Get PPI neighbors of this drug target
        ppi_neighbors = ppi.get(target, set())

        # Find neighbors that are disease genes
        bridge_genes = ppi_neighbors & disease_genes

        for disease_gene in bridge_genes:
            two_hop_paths.append((target, disease_gene))

    return two_hop_paths


def analyze_predictions():
    """Main analysis: compare 1-hop vs 2-hop mechanism paths."""
    print("=" * 70)
    print("h226: TWO-HOP MECHANISM PATH ANALYSIS")
    print("=" * 70)
    print()

    # Load data
    print("[1] Loading data...")
    drug_targets = load_drug_targets()
    disease_genes = load_disease_genes()
    ppi = load_ppi_network()
    name_to_drug = load_drugbank_lookup()

    print(f"    Drugs with targets: {len(drug_targets)}")
    print(f"    Diseases with genes: {len(disease_genes)}")
    print(f"    Genes in PPI network: {len(ppi)}")
    print(f"    PPI edges: {sum(len(v) for v in ppi.values()) // 2}")
    print()

    # Load predictions
    print("[2] Loading production predictions...")
    try:
        predictions = load_production_predictions()
        print(f"    Loaded {len(predictions)} predictions")
    except Exception as e:
        print(f"    Error: {e}")
        print("    Using h166 sample data instead...")
        with open(ANALYSIS_DIR / "h166_mechanism_path_analysis.json") as f:
            h166_data = json.load(f)
        predictions = h166_data.get("sample_paths", [])
        print(f"    Using {len(predictions)} sample predictions")
    print()

    # Analyze each prediction
    print("[3] Analyzing mechanism paths...")

    results = {
        "no_path": [],       # Neither 1-hop nor 2-hop
        "one_hop_only": [],  # Has 1-hop path
        "two_hop_only": [],  # Has 2-hop but not 1-hop
        "both": [],          # Has both 1-hop and 2-hop
    }

    path_details = []

    for i, pred in enumerate(predictions):
        # Get drug and disease identifiers from JSON format
        drug_name = pred.get("drug_name") or pred.get("drug")
        disease_id_raw = pred.get("disease_id") or pred.get("disease_mesh")
        is_known = pred.get("is_known_indication", False) or pred.get("is_known", False)

        if not drug_name or not disease_id_raw:
            continue

        # Extract MESH ID from disease_id (e.g., "drkg:Disease::MESH:D006333" -> "MESH:D006333")
        if disease_id_raw.startswith("drkg:Disease::"):
            disease_mesh = disease_id_raw.replace("drkg:Disease::", "")
        else:
            disease_mesh = disease_id_raw

        # Get drug ID directly from prediction or map from name
        drug_id_raw = pred.get("drug_id")
        if drug_id_raw and drug_id_raw.startswith("drkg:Compound::"):
            drug_id = drug_id_raw.replace("drkg:", "")
        else:
            drug_id = name_to_drug.get(drug_name.lower())
            if not drug_id:
                # Try direct match
                for d_id in drug_targets.keys():
                    if drug_name.lower() in d_id.lower():
                        drug_id = d_id
                        break

        if not drug_id or drug_id not in drug_targets:
            continue

        # Get drug targets and disease genes
        d_targets = drug_targets[drug_id]
        d_genes = disease_genes.get(disease_mesh, set())

        if not d_genes:
            continue

        # Find paths
        one_hop = find_1hop_paths(d_targets, d_genes)
        two_hop = find_2hop_paths(d_targets, d_genes, ppi)

        has_1hop = len(one_hop) > 0
        has_2hop = len(two_hop) > 0

        detail = {
            "drug": drug_name,
            "disease": disease_mesh,
            "is_known": is_known,
            "n_1hop": len(one_hop),
            "n_2hop": len(two_hop),
            "one_hop_genes": list(one_hop)[:5],
            "two_hop_paths": two_hop[:5]
        }
        path_details.append(detail)

        if has_1hop and has_2hop:
            results["both"].append(detail)
        elif has_1hop:
            results["one_hop_only"].append(detail)
        elif has_2hop:
            results["two_hop_only"].append(detail)
        else:
            results["no_path"].append(detail)

        if (i + 1) % 1000 == 0:
            print(f"    Processed {i + 1} predictions...")

    print(f"    Total analyzed: {len(path_details)}")
    print()

    # Calculate statistics
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    total = len(path_details)

    # Coverage
    n_1hop = len(results["one_hop_only"]) + len(results["both"])
    n_2hop = len(results["two_hop_only"]) + len(results["both"])
    n_2hop_only = len(results["two_hop_only"])
    n_no_path = len(results["no_path"])

    print("COVERAGE:")
    print(f"  1-hop mechanism paths: {n_1hop}/{total} ({100*n_1hop/total:.1f}%)")
    print(f"  2-hop mechanism paths: {n_2hop}/{total} ({100*n_2hop/total:.1f}%)")
    print(f"  2-hop ONLY (no 1-hop): {n_2hop_only}/{total} ({100*n_2hop_only/total:.1f}%)")
    print(f"  No mechanism path: {n_no_path}/{total} ({100*n_no_path/total:.1f}%)")
    print()

    # Precision by path type
    def calc_precision(items):
        if not items:
            return 0.0, 0, 0
        known = sum(1 for p in items if p["is_known"])
        return 100 * known / len(items), known, len(items)

    p_no_path, k_no, t_no = calc_precision(results["no_path"])
    p_1hop_only, k_1h, t_1h = calc_precision(results["one_hop_only"])
    p_2hop_only, k_2h, t_2h = calc_precision(results["two_hop_only"])
    p_both, k_b, t_b = calc_precision(results["both"])

    # Combined 1-hop (includes both)
    all_1hop = results["one_hop_only"] + results["both"]
    p_1hop_all, k_1ha, t_1ha = calc_precision(all_1hop)

    print("PRECISION BY PATH TYPE:")
    print(f"  No path:        {p_no_path:.1f}% ({k_no}/{t_no})")
    print(f"  1-hop only:     {p_1hop_only:.1f}% ({k_1h}/{t_1h})")
    print(f"  2-hop only:     {p_2hop_only:.1f}% ({k_2h}/{t_2h})")
    print(f"  Both 1&2-hop:   {p_both:.1f}% ({k_b}/{t_b})")
    print()
    print(f"  All with 1-hop: {p_1hop_all:.1f}% ({k_1ha}/{t_1ha})")
    print()

    # The key question: does 2-hop add value?
    print("-" * 70)
    print("KEY QUESTION: Does 2-hop extend coverage while maintaining precision?")
    print("-" * 70)

    if n_2hop_only > 0:
        coverage_gain = 100 * n_2hop_only / total
        precision_ratio = p_2hop_only / p_1hop_all if p_1hop_all > 0 else 0

        print(f"\n  2-hop extends coverage by +{coverage_gain:.1f}% ({n_2hop_only} predictions)")
        print(f"  2-hop-only precision: {p_2hop_only:.1f}% vs 1-hop precision: {p_1hop_all:.1f}%")
        print(f"  Precision ratio (2-hop/1-hop): {precision_ratio:.2f}x")

        if precision_ratio >= 0.8:
            print("\n  VERDICT: 2-hop MAINTAINS precision (>=80% of 1-hop)")
            verdict = "MAINTAINS_PRECISION"
        elif precision_ratio >= 0.5:
            print("\n  VERDICT: 2-hop PARTIALLY maintains precision (50-80% of 1-hop)")
            verdict = "PARTIAL_VALUE"
        else:
            print("\n  VERDICT: 2-hop precision is POOR (<50% of 1-hop)")
            verdict = "LOW_VALUE"
    else:
        print("\n  No predictions have 2-hop-only paths!")
        verdict = "NO_2HOP_FOUND"
    print()

    # Analyze 2-hop path statistics
    if results["two_hop_only"]:
        print("2-HOP ONLY PATH STATISTICS:")
        n_2hop_counts = [d["n_2hop"] for d in results["two_hop_only"]]
        print(f"  Mean 2-hop paths per prediction: {sum(n_2hop_counts)/len(n_2hop_counts):.1f}")
        print(f"  Max 2-hop paths: {max(n_2hop_counts)}")
        print(f"  Median 2-hop paths: {sorted(n_2hop_counts)[len(n_2hop_counts)//2]}")
        print()

        # Precision by number of 2-hop paths
        print("PRECISION BY NUMBER OF 2-HOP PATHS:")
        by_count = defaultdict(list)
        for d in results["two_hop_only"]:
            count_bin = "1" if d["n_2hop"] == 1 else "2-5" if d["n_2hop"] <= 5 else "6-10" if d["n_2hop"] <= 10 else "10+"
            by_count[count_bin].append(d)

        for count_bin in ["1", "2-5", "6-10", "10+"]:
            if count_bin in by_count:
                items = by_count[count_bin]
                p, k, t = calc_precision(items)
                print(f"  {count_bin} paths: {p:.1f}% ({k}/{t})")
        print()

    # Sample 2-hop-only paths
    if results["two_hop_only"][:10]:
        print("SAMPLE 2-HOP ONLY PREDICTIONS:")
        for d in results["two_hop_only"][:10]:
            known_str = "✓" if d["is_known"] else ""
            paths_str = ", ".join(f"{p[0]}->{p[1]}" for p in d["two_hop_paths"][:2])
            print(f"  {d['drug']} -> {d['disease']}: {d['n_2hop']} paths {known_str}")
            print(f"    Example: {paths_str}")
        print()

    # Save results
    output = {
        "hypothesis": "h226",
        "title": "Two-Hop Mechanism Paths (Drug->Gene1->Gene2->Disease)",
        "summary": {
            "total_predictions": total,
            "coverage_1hop": n_1hop,
            "coverage_1hop_pct": round(100 * n_1hop / total, 2),
            "coverage_2hop_only": n_2hop_only,
            "coverage_2hop_only_pct": round(100 * n_2hop_only / total, 2),
            "coverage_any_path": n_1hop + n_2hop_only,
            "coverage_any_path_pct": round(100 * (n_1hop + n_2hop_only) / total, 2),
            "no_path": n_no_path,
            "no_path_pct": round(100 * n_no_path / total, 2),
        },
        "precision": {
            "no_path": round(p_no_path, 2),
            "1hop_only": round(p_1hop_only, 2),
            "2hop_only": round(p_2hop_only, 2),
            "both": round(p_both, 2),
            "all_1hop": round(p_1hop_all, 2),
        },
        "verdict": verdict,
        "details": {
            "no_path": len(results["no_path"]),
            "one_hop_only": len(results["one_hop_only"]),
            "two_hop_only": len(results["two_hop_only"]),
            "both": len(results["both"]),
        },
        "sample_2hop_only": results["two_hop_only"][:20]
    }

    output_path = ANALYSIS_DIR / "h226_two_hop_mechanism.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {output_path}")

    return output


if __name__ == "__main__":
    analyze_predictions()
