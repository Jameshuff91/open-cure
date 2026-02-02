#!/usr/bin/env python3
"""
Analyze Disconnected Diseases.

For the diseases that lost embeddings after removing treatment edges:
1. List the disease names
2. Categorize them (rare, orphan, well-studied?)
3. Check what edges they DID have (only treatment edges?)
4. Are these the diseases we most want to help?
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
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"


def load_drkg_disease_edges() -> dict[str, list[tuple[str, str, str]]]:
    """Load all edges involving diseases from DRKG."""
    drkg_path = RAW_DIR / "drkg.tsv"
    if not drkg_path.exists():
        print(f"ERROR: DRKG not found at {drkg_path}")
        sys.exit(1)

    disease_edges: dict[str, list[tuple[str, str, str]]] = defaultdict(list)

    with open(drkg_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            head, rel, tail = parts

            # Check if head or tail is a disease
            if head.startswith("Disease::"):
                disease_edges[head].append((head, rel, tail))
            if tail.startswith("Disease::"):
                disease_edges[tail].append((head, rel, tail))

    return disease_edges


def load_embeddings_entities(path: Path) -> set[str]:
    """Load entity IDs from embeddings file (without drkg: prefix)."""
    df = pd.read_csv(path)
    return set(df["entity"].tolist())


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


def load_ground_truth_with_names(
    mesh_mappings: dict[str, str],
    name_to_drug_id: dict[str, str],
) -> tuple[dict[str, set[str]], dict[str, str]]:
    df = pd.read_excel(REFERENCE_DIR / "everycure" / "indicationList.xlsx")
    fuzzy_mappings = load_mesh_mappings()
    matcher = DiseaseMatcher(fuzzy_mappings)

    gt_pairs: dict[str, set[str]] = defaultdict(set)
    disease_id_to_name: dict[str, str] = {}

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
            if disease_id not in disease_id_to_name:
                disease_id_to_name[disease_id] = disease

    return dict(gt_pairs), disease_id_to_name


def categorize_disease(disease_name: str) -> str:
    """Categorize disease by name keywords."""
    name_lower = disease_name.lower()

    if any(kw in name_lower for kw in ["cancer", "carcinoma", "lymphoma", "leukemia", "tumor", "melanoma", "sarcoma", "myeloma"]):
        return "cancer"
    if any(kw in name_lower for kw in ["diabetes", "obesity", "hyperlipidemia", "metabolic"]):
        return "metabolic"
    if any(kw in name_lower for kw in ["arthritis", "lupus", "sclerosis", "crohn", "colitis", "psoriasis", "autoimmune"]):
        return "autoimmune"
    if any(kw in name_lower for kw in ["infection", "hiv", "hepatitis", "tuberculosis", "malaria", "sepsis"]):
        return "infectious"
    if any(kw in name_lower for kw in ["alzheimer", "parkinson", "epilepsy", "dementia", "neuropathy"]):
        return "neurological"
    if any(kw in name_lower for kw in ["heart", "cardiac", "hypertension", "coronary", "atrial"]):
        return "cardiovascular"
    if any(kw in name_lower for kw in ["asthma", "copd", "pulmonary", "respiratory"]):
        return "respiratory"
    if any(kw in name_lower for kw in ["depression", "anxiety", "schizophrenia", "bipolar"]):
        return "psychiatric"
    if any(kw in name_lower for kw in ["dermatitis", "eczema", "acne", "skin"]):
        return "dermatological"
    if any(kw in name_lower for kw in ["syndrome", "disease", "disorder"]):
        return "rare/genetic"

    return "other"


def main() -> None:
    print("=" * 70)
    print("DISCONNECTED DISEASES ANALYSIS")
    print("=" * 70)
    print()
    print("Analyzing diseases that lost embeddings after removing treatment edges")
    print()

    # Load embeddings from both files
    print("Loading embeddings...")
    original_path = EMBEDDINGS_DIR / "node2vec_256_named.csv"
    honest_path = EMBEDDINGS_DIR / "node2vec_256_no_treatment.csv"

    if not original_path.exists() or not honest_path.exists():
        print("ERROR: Embedding files not found")
        sys.exit(1)

    original_entities = load_embeddings_entities(original_path)
    honest_entities = load_embeddings_entities(honest_path)

    original_diseases = {e for e in original_entities if e.startswith("Disease::")}
    honest_diseases = {e for e in honest_entities if e.startswith("Disease::")}

    disconnected = original_diseases - honest_diseases
    print(f"  Original embeddings: {len(original_diseases):,} diseases")
    print(f"  Honest embeddings: {len(honest_diseases):,} diseases")
    print(f"  Disconnected: {len(disconnected):,} diseases")
    print()

    # Load GT to find which disconnected diseases are in our evaluation
    print("Loading ground truth...")
    name_to_drug_id = load_drugbank_lookup()
    mesh_mappings = load_mesh_mappings_from_file()
    gt_pairs, disease_id_to_name = load_ground_truth_with_names(mesh_mappings, name_to_drug_id)

    # Map GT disease IDs to DRKG format
    gt_diseases_drkg: set[str] = set()
    gt_id_to_name: dict[str, str] = {}
    for disease_id, name in disease_id_to_name.items():
        # disease_id is like "drkg:Disease::MESH:D000001", DRKG format is "Disease::MESH:D000001"
        drkg_id = disease_id.replace("drkg:", "")
        gt_diseases_drkg.add(drkg_id)
        gt_id_to_name[drkg_id] = name

    # Which disconnected diseases are in GT?
    disconnected_in_gt = disconnected & gt_diseases_drkg
    print(f"  Disconnected diseases in GT: {len(disconnected_in_gt)}")
    print()

    # Load DRKG edges to analyze what edges disconnected diseases had
    print("Loading DRKG edges...")
    disease_edges = load_drkg_disease_edges()

    # Analyze disconnected diseases
    print()
    print("=" * 70)
    print("DISCONNECTED DISEASES IN GROUND TRUTH")
    print("=" * 70)

    analysis_results: list[dict[str, Any]] = []

    for disease_id in sorted(disconnected_in_gt):
        disease_name = gt_id_to_name.get(disease_id, "Unknown")
        n_gt_drugs = len(gt_pairs.get(f"drkg:{disease_id}", set()))
        category = categorize_disease(disease_name)

        # Get edges for this disease
        edges = disease_edges.get(disease_id, [])

        # Categorize edges
        edge_types: dict[str, int] = defaultdict(int)
        treatment_edges = 0
        other_edges = 0

        for head, rel, tail in edges:
            if "treats" in rel.lower():
                treatment_edges += 1
            else:
                other_edges += 1

            # Extract relation type
            rel_type = rel.split("::")[0] if "::" in rel else rel
            edge_types[rel_type] += 1

        analysis_results.append({
            "disease_id": disease_id,
            "disease_name": disease_name,
            "n_gt_drugs": n_gt_drugs,
            "category": category,
            "total_edges": len(edges),
            "treatment_edges": treatment_edges,
            "other_edges": other_edges,
            "edge_types": dict(edge_types),
        })

    # Sort by number of GT drugs (descending) - most important diseases first
    analysis_results.sort(key=lambda x: x["n_gt_drugs"], reverse=True)

    # Print detailed results
    print(f"\n{len(analysis_results)} diseases in GT became disconnected:")
    print()

    for i, result in enumerate(analysis_results[:20]):
        print(f"  {i+1}. {result['disease_name']}")
        print(f"     GT drugs: {result['n_gt_drugs']}, Category: {result['category']}")
        print(f"     Edges: {result['total_edges']} total ({result['treatment_edges']} treatment, {result['other_edges']} other)")
        print()

    # Summary statistics
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_gt_drugs_lost = sum(r["n_gt_drugs"] for r in analysis_results)
    only_treatment = [r for r in analysis_results if r["other_edges"] == 0]
    had_other = [r for r in analysis_results if r["other_edges"] > 0]

    print(f"\n  Disconnected diseases: {len(analysis_results)}")
    print(f"  Total GT drugs affected: {total_gt_drugs_lost}")
    print()
    print(f"  Diseases with ONLY treatment edges: {len(only_treatment)} "
          f"({100*len(only_treatment)/len(analysis_results):.1f}%)")
    print(f"  Diseases with OTHER edges too: {len(had_other)} "
          f"({100*len(had_other)/len(analysis_results):.1f}%)")
    print()

    # Category breakdown
    print("  By category:")
    categories: dict[str, int] = defaultdict(int)
    for r in analysis_results:
        categories[r["category"]] += 1
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {count}")

    # Key insight
    print()
    print("-" * 70)
    print("KEY INSIGHT")
    print("-" * 70)
    print()

    if len(only_treatment) / len(analysis_results) > 0.8:
        insight = (
            "Most disconnected diseases had ONLY treatment edges in DRKG. "
            "These are diseases known primarily through drug relationships, "
            "not through biological mechanisms. Removing treatment edges "
            "correctly excludes them from honest evaluation."
        )
    else:
        insight = (
            "Many disconnected diseases had other edge types but still became "
            "disconnected. This suggests they were only weakly connected to "
            "the main graph through treatment edges. These may be rare or "
            "poorly-characterized diseases."
        )

    print(f"  {insight}")

    # Are these important diseases?
    print()
    print("-" * 70)
    print("ARE THESE IMPORTANT DISEASES?")
    print("-" * 70)
    print()

    high_gt = [r for r in analysis_results if r["n_gt_drugs"] >= 5]
    print(f"  Disconnected diseases with ≥5 GT drugs: {len(high_gt)}")
    for r in high_gt:
        print(f"    - {r['disease_name']} ({r['n_gt_drugs']} drugs)")

    if high_gt:
        print(f"\n  WARNING: We're losing {len(high_gt)} well-studied diseases from evaluation.")
    else:
        print("\n  ✓ No well-studied diseases (≥5 GT drugs) were lost.")

    # Save results
    results_data: dict[str, Any] = {
        "analysis": "disconnected_diseases",
        "description": "Analysis of diseases that lost embeddings after removing treatment edges",
        "summary": {
            "total_disconnected": len(disconnected),
            "disconnected_in_gt": len(disconnected_in_gt),
            "total_gt_drugs_lost": total_gt_drugs_lost,
            "only_treatment_edges": len(only_treatment),
            "had_other_edges": len(had_other),
        },
        "by_category": dict(categories),
        "diseases": analysis_results,
        "insight": insight,
    }

    output_path = ANALYSIS_DIR / "disconnected_diseases.json"
    with open(output_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
