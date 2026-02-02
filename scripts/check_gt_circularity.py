#!/usr/bin/env python3
"""
Check Ground Truth Circularity.

For each test disease, check if GT drug-disease pairs from Every Cure
also exist as treatment edges in DRKG. This tells us:
1. What % of our "ground truth" was already in the training graph?
2. Are we just recovering known information from DRKG rather than predicting novel treatments?
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from disease_name_matcher import DiseaseMatcher, load_mesh_mappings

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "drkg"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"


def load_drkg_treatment_edges() -> set[tuple[str, str]]:
    """Load all treatment edges from DRKG (Compound treats Disease)."""
    drkg_path = RAW_DIR / "drkg.tsv"
    if not drkg_path.exists():
        print(f"ERROR: DRKG not found at {drkg_path}")
        sys.exit(1)

    treatment_edges: set[tuple[str, str]] = set()

    # Read DRKG and find treatment edges
    with open(drkg_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            head, rel, tail = parts

            # Check if this is a treatment edge
            if "treats" in rel.lower():
                # Normalize to match our format: drkg:Compound::DB00001, drkg:Disease::MESH:D000001
                drug_id = f"drkg:{head}" if not head.startswith("drkg:") else head
                disease_id = f"drkg:{tail}" if not tail.startswith("drkg:") else tail
                treatment_edges.add((drug_id, disease_id))

    return treatment_edges


def load_drugbank_lookup() -> dict[str, str]:
    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)
    return {name.lower(): f"drkg:Compound::{db_id}" for db_id, name in id_to_name.items()}


def load_mesh_mappings_from_file() -> dict[str, str]:
    mesh_path = REFERENCE_DIR / "mesh_mappings_from_agents.json"
    if not mesh_path.exists():
        return {}
    with open(mesh_path) as f:
        mesh_data = json.load(f)
    mesh_mappings: dict[str, str] = {}
    for batch_data in mesh_data.values():
        if isinstance(batch_data, dict):
            for disease_name, mesh_id in batch_data.items():
                if mesh_id:
                    mesh_str = str(mesh_id)
                    if mesh_str.startswith("D") or mesh_str.startswith("C"):
                        mesh_mappings[disease_name.lower()] = f"drkg:Disease::MESH:{mesh_str}"
    return mesh_mappings


def load_ground_truth_pairs(
    mesh_mappings: dict[str, str],
    name_to_drug_id: dict[str, str],
) -> list[tuple[str, str, str, str]]:
    """Load GT pairs with original names for reporting."""
    df = pd.read_excel(REFERENCE_DIR / "everycure" / "indicationList.xlsx")
    fuzzy_mappings = load_mesh_mappings()
    matcher = DiseaseMatcher(fuzzy_mappings)

    pairs: list[tuple[str, str, str, str]] = []  # (drug_id, disease_id, drug_name, disease_name)

    for _, row in df.iterrows():
        disease_name = str(row.get("disease name", "")).strip()
        drug_name = str(row.get("final normalized drug label", "")).strip()
        if not disease_name or not drug_name:
            continue

        disease_id = matcher.get_mesh_id(disease_name)
        if not disease_id:
            disease_id = mesh_mappings.get(disease_name.lower())
        if not disease_id:
            continue

        drug_id = name_to_drug_id.get(drug_name.lower())
        if not drug_id:
            continue

        pairs.append((drug_id, disease_id, drug_name, disease_name))

    return pairs


def main() -> None:
    print("=" * 70)
    print("GROUND TRUTH CIRCULARITY CHECK")
    print("=" * 70)
    print()
    print("Checking: What % of Every Cure GT pairs exist in DRKG treatment edges?")
    print()

    # Load DRKG treatment edges
    print("Loading DRKG treatment edges...")
    drkg_edges = load_drkg_treatment_edges()
    print(f"  {len(drkg_edges):,} treatment edges in DRKG")
    print()

    # Load Every Cure GT
    print("Loading Every Cure ground truth...")
    name_to_drug_id = load_drugbank_lookup()
    mesh_mappings = load_mesh_mappings_from_file()
    gt_pairs = load_ground_truth_pairs(mesh_mappings, name_to_drug_id)
    print(f"  {len(gt_pairs):,} GT pairs loaded")
    print()

    # Check overlap
    print("Checking overlap...")
    in_drkg: list[tuple[str, str, str, str]] = []
    not_in_drkg: list[tuple[str, str, str, str]] = []

    for drug_id, disease_id, drug_name, disease_name in gt_pairs:
        if (drug_id, disease_id) in drkg_edges:
            in_drkg.append((drug_id, disease_id, drug_name, disease_name))
        else:
            not_in_drkg.append((drug_id, disease_id, drug_name, disease_name))

    overlap_pct = len(in_drkg) / len(gt_pairs) * 100 if gt_pairs else 0

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"  Total GT pairs (mapped): {len(gt_pairs):,}")
    print(f"  In DRKG treatment edges: {len(in_drkg):,} ({overlap_pct:.1f}%)")
    print(f"  NOT in DRKG:             {len(not_in_drkg):,} ({100-overlap_pct:.1f}%)")
    print()

    # Per-disease breakdown
    disease_overlap: dict[str, dict[str, int]] = defaultdict(lambda: {"in_drkg": 0, "not_in_drkg": 0})
    for _, disease_id, _, _ in in_drkg:
        disease_overlap[disease_id]["in_drkg"] += 1
    for _, disease_id, _, _ in not_in_drkg:
        disease_overlap[disease_id]["not_in_drkg"] += 1

    # Diseases with 100% DRKG overlap (purely recovering known info)
    fully_circular = [d for d, counts in disease_overlap.items()
                      if counts["not_in_drkg"] == 0 and counts["in_drkg"] > 0]

    # Diseases with 0% DRKG overlap (truly novel predictions)
    fully_novel = [d for d, counts in disease_overlap.items()
                   if counts["in_drkg"] == 0 and counts["not_in_drkg"] > 0]

    print(f"  Disease-level analysis:")
    print(f"    Diseases with 100% DRKG overlap: {len(fully_circular)} (all GT is circular)")
    print(f"    Diseases with 0% DRKG overlap:   {len(fully_novel)} (all GT is novel)")
    print(f"    Mixed overlap:                   {len(disease_overlap) - len(fully_circular) - len(fully_novel)}")
    print()

    # Interpretation
    print("-" * 70)
    print("INTERPRETATION")
    print("-" * 70)
    print()

    if overlap_pct > 80:
        interpretation = (
            f"HIGH CIRCULARITY ({overlap_pct:.0f}%): Most of our 'ground truth' "
            "already exists in DRKG treatment edges. Our model may be largely "
            "recovering information it was trained on, not predicting novel treatments."
        )
    elif overlap_pct > 50:
        interpretation = (
            f"MODERATE CIRCULARITY ({overlap_pct:.0f}%): About half of GT pairs "
            "are in DRKG. Model performance is a mix of recall (known) and "
            "prediction (novel). This is acceptable but should be disclosed."
        )
    elif overlap_pct > 20:
        interpretation = (
            f"LOW CIRCULARITY ({overlap_pct:.0f}%): Most GT pairs are NOT in DRKG "
            "treatment edges. Our evaluation tests genuine prediction ability, "
            "though some recall of training data is present."
        )
    else:
        interpretation = (
            f"MINIMAL CIRCULARITY ({overlap_pct:.0f}%): Nearly all GT pairs are "
            "novel to DRKG. Our evaluation truly measures prediction, not recall."
        )

    print(f"  {interpretation}")
    print()

    # Sample of overlapping pairs
    print("-" * 70)
    print("SAMPLE: GT PAIRS IN DRKG (first 10)")
    print("-" * 70)
    for drug_id, disease_id, drug_name, disease_name in in_drkg[:10]:
        print(f"  {drug_name} -> {disease_name}")

    print()
    print("-" * 70)
    print("SAMPLE: GT PAIRS NOT IN DRKG (first 10)")
    print("-" * 70)
    for drug_id, disease_id, drug_name, disease_name in not_in_drkg[:10]:
        print(f"  {drug_name} -> {disease_name}")

    # Save results
    results: dict[str, Any] = {
        "analysis": "gt_circularity",
        "description": "Checks overlap between Every Cure GT and DRKG treatment edges",
        "counts": {
            "drkg_treatment_edges": len(drkg_edges),
            "gt_pairs_mapped": len(gt_pairs),
            "gt_in_drkg": len(in_drkg),
            "gt_not_in_drkg": len(not_in_drkg),
        },
        "percentages": {
            "overlap_pct": overlap_pct,
            "novel_pct": 100 - overlap_pct,
        },
        "disease_level": {
            "total_diseases": len(disease_overlap),
            "fully_circular": len(fully_circular),
            "fully_novel": len(fully_novel),
            "mixed": len(disease_overlap) - len(fully_circular) - len(fully_novel),
        },
        "interpretation": interpretation,
        "samples": {
            "in_drkg": [
                {"drug": d[2], "disease": d[3]}
                for d in in_drkg[:20]
            ],
            "not_in_drkg": [
                {"drug": d[2], "disease": d[3]}
                for d in not_in_drkg[:20]
            ],
        },
    }

    output_path = ANALYSIS_DIR / "gt_circularity.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
