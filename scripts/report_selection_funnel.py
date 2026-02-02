#!/usr/bin/env python3
"""
Report Selection Funnel.

Document the filtering at each step from total diseases in Every Cure GT
to the final evaluation set. This addresses selection bias concerns by
showing exactly what was dropped and why.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"


def load_every_cure_gt() -> pd.DataFrame:
    """Load raw Every Cure indication list."""
    return pd.read_excel(REFERENCE_DIR / "everycure" / "indicationList.xlsx")


def load_mesh_mappings() -> dict[str, str]:
    """Load all MESH mappings from agents."""
    mesh_path = REFERENCE_DIR / "mesh_mappings_from_agents.json"
    if not mesh_path.exists():
        return {}
    with open(mesh_path) as f:
        mesh_data = json.load(f)
    mappings: dict[str, str] = {}
    for batch_data in mesh_data.values():
        if isinstance(batch_data, dict):
            for disease_name, mesh_id in batch_data.items():
                if mesh_id:
                    mesh_str = str(mesh_id)
                    if mesh_str.startswith("D") or mesh_str.startswith("C"):
                        mappings[disease_name.lower()] = f"MESH:{mesh_str}"
    return mappings


def load_fuzzy_mappings() -> dict[str, str]:
    """Load fuzzy disease name mappings."""
    from disease_name_matcher import load_mesh_mappings
    return load_mesh_mappings()


def load_drugbank_lookup() -> dict[str, str]:
    """Load DrugBank name-to-ID mappings."""
    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)
    return {name.lower(): db_id for db_id, name in id_to_name.items()}


def load_embeddings(path: Path) -> set[str]:
    """Load entity IDs from embeddings file."""
    df = pd.read_csv(path)
    # Embeddings store entities like "Disease::MESH:D014141", "Compound::DB00001"
    return {row['entity'] for _, row in df.iterrows()}


def main() -> None:
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    print("=" * 70)
    print("SELECTION FUNNEL REPORT")
    print("=" * 70)
    print()
    print("Tracing data from Every Cure ground truth to final evaluation set")
    print()

    # Step 1: Load raw data
    df = load_every_cure_gt()
    step1_total_rows = len(df)
    step1_diseases = df["disease name"].dropna().str.strip().unique()
    step1_drugs = df["final normalized drug label"].dropna().str.strip().unique()

    print(f"STEP 1: Raw Every Cure Ground Truth")
    print(f"  Total rows: {step1_total_rows:,}")
    print(f"  Unique diseases: {len(step1_diseases):,}")
    print(f"  Unique drugs: {len(step1_drugs):,}")
    print()

    # Step 2: After filtering empty/invalid entries
    valid_rows = df[
        df["disease name"].notna()
        & df["final normalized drug label"].notna()
        & (df["disease name"].str.strip() != "")
        & (df["final normalized drug label"].str.strip() != "")
    ]
    step2_diseases = valid_rows["disease name"].str.strip().unique()
    step2_drugs = valid_rows["final normalized drug label"].str.strip().unique()

    print(f"STEP 2: After Removing Empty Entries")
    print(f"  Valid rows: {len(valid_rows):,}")
    print(f"  Unique diseases: {len(step2_diseases):,}")
    print(f"  Unique drugs: {len(step2_drugs):,}")
    print()

    # Step 3: Load all mapping sources
    mesh_mappings = load_mesh_mappings()
    fuzzy_mappings = load_fuzzy_mappings()
    drugbank_lookup = load_drugbank_lookup()

    # Try to map diseases
    diseases_with_mesh: set[str] = set()
    diseases_without_mesh: set[str] = set()
    mesh_id_map: dict[str, str] = {}

    for disease in step2_diseases:
        disease_lower = disease.lower()
        mesh_id = None

        # Try fuzzy mappings first
        for name_variant in [disease_lower, disease_lower.replace("'s", "s")]:
            if name_variant in fuzzy_mappings:
                mesh_id = fuzzy_mappings[name_variant]
                break

        # Try agent mappings
        if not mesh_id and disease_lower in mesh_mappings:
            mesh_id = mesh_mappings[disease_lower]

        if mesh_id:
            diseases_with_mesh.add(disease)
            mesh_id_map[disease] = mesh_id
        else:
            diseases_without_mesh.add(disease)

    print(f"STEP 3: After MESH ID Mapping")
    print(f"  Diseases with MESH ID: {len(diseases_with_mesh):,}")
    print(f"  Diseases without MESH ID: {len(diseases_without_mesh):,}")
    print(f"  Mapping rate: {100*len(diseases_with_mesh)/len(step2_diseases):.1f}%")
    print()

    # Step 4: Map drugs to DrugBank IDs
    drugs_with_drugbank: set[str] = set()
    drugs_without_drugbank: set[str] = set()

    for drug in step2_drugs:
        if drug.lower() in drugbank_lookup:
            drugs_with_drugbank.add(drug)
        else:
            drugs_without_drugbank.add(drug)

    print(f"STEP 4: Drug Mapping to DrugBank")
    print(f"  Drugs with DrugBank ID: {len(drugs_with_drugbank):,}")
    print(f"  Drugs without DrugBank ID: {len(drugs_without_drugbank):,}")
    print(f"  Mapping rate: {100*len(drugs_with_drugbank)/len(step2_drugs):.1f}%")
    print()

    # Step 5: Check embedding coverage
    original_emb_path = EMBEDDINGS_DIR / "node2vec_256_named.csv"
    honest_emb_path = EMBEDDINGS_DIR / "node2vec_256_no_treatment.csv"

    if not original_emb_path.exists() or not honest_emb_path.exists():
        print("ERROR: Embedding files not found")
        return

    original_entities = load_embeddings(original_emb_path)
    honest_entities = load_embeddings(honest_emb_path)

    # Check disease coverage
    diseases_in_original: set[str] = set()
    diseases_in_honest: set[str] = set()

    for disease in diseases_with_mesh:
        mesh_id = mesh_id_map[disease]
        # Normalize MESH ID format
        # Input could be: "MESH:D014141" or "drkg:Disease::MESH:D014141"
        if mesh_id.startswith("drkg:Disease::"):
            entity_id = mesh_id.replace("drkg:", "")  # -> "Disease::MESH:D014141"
        elif mesh_id.startswith("MESH:"):
            entity_id = f"Disease::{mesh_id}"  # -> "Disease::MESH:D014141"
        else:
            entity_id = f"Disease::MESH:{mesh_id}"  # Assume raw MESH ID

        if entity_id in original_entities:
            diseases_in_original.add(disease)
        if entity_id in honest_entities:
            diseases_in_honest.add(disease)

    print(f"STEP 5: Embedding Coverage (Diseases)")
    print(f"  Diseases in original embeddings: {len(diseases_in_original):,}")
    print(f"  Diseases in honest embeddings: {len(diseases_in_honest):,}")
    print(f"  Diseases lost (disconnected): {len(diseases_in_original) - len(diseases_in_honest):,}")
    print()

    # Check drug coverage
    drugs_in_original: set[str] = set()
    drugs_in_honest: set[str] = set()

    for drug in drugs_with_drugbank:
        db_id = drugbank_lookup[drug.lower()]
        # Format: "Compound::DB00001"
        entity_id = f"Compound::{db_id}"
        if entity_id in original_entities:
            drugs_in_original.add(drug)
        if entity_id in honest_entities:
            drugs_in_honest.add(drug)

    print(f"STEP 6: Embedding Coverage (Drugs)")
    print(f"  Drugs in original embeddings: {len(drugs_in_original):,}")
    print(f"  Drugs in honest embeddings: {len(drugs_in_honest):,}")
    print()

    # Step 7: Final evaluation set (diseases with at least 1 GT drug in embeddings)
    # Build GT pairs for diseases in honest embeddings
    gt_pairs: dict[str, set[str]] = defaultdict(set)

    for _, row in valid_rows.iterrows():
        disease = str(row["disease name"]).strip()
        drug = str(row["final normalized drug label"]).strip()

        if disease not in diseases_in_honest:
            continue
        if drug.lower() not in drugbank_lookup:
            continue

        db_id = drugbank_lookup[drug.lower()]
        drug_entity_id = f"Compound::{db_id}"
        if drug_entity_id not in honest_entities:
            continue

        mesh_id = mesh_id_map[disease]
        gt_pairs[disease].add(drug)

    diseases_with_gt_drugs = {d for d, drugs in gt_pairs.items() if len(drugs) > 0}

    print(f"STEP 7: Final Evaluation Set (Honest Embeddings)")
    print(f"  Diseases with ≥1 GT drug in embeddings: {len(diseases_with_gt_drugs):,}")
    print(f"  Total GT pairs evaluable: {sum(len(v) for v in gt_pairs.values()):,}")
    print()

    # Sample of dropped diseases at each stage
    dropped_at_mesh = list(diseases_without_mesh)[:20]
    dropped_at_embedding = list(diseases_in_original - diseases_in_honest)[:20]

    # Summary
    print("=" * 70)
    print("SELECTION FUNNEL SUMMARY")
    print("=" * 70)
    print()
    print(f"  {'Stage':<45} {'Diseases':>10} {'Drop %':>10}")
    print("-" * 70)
    print(f"  {'1. Raw Every Cure diseases':<45} {len(step1_diseases):>10,} {'-':>10}")
    print(f"  {'2. After removing empty entries':<45} {len(step2_diseases):>10,} {100*(1-len(step2_diseases)/len(step1_diseases)):>9.1f}%")
    print(f"  {'3. After MESH ID mapping':<45} {len(diseases_with_mesh):>10,} {100*(1-len(diseases_with_mesh)/len(step2_diseases)):>9.1f}%")
    print(f"  {'4. With embeddings (original)':<45} {len(diseases_in_original):>10,} {100*(1-len(diseases_in_original)/len(diseases_with_mesh)):>9.1f}%")
    print(f"  {'5. With embeddings (honest)':<45} {len(diseases_in_honest):>10,} {100*(1-len(diseases_in_honest)/len(diseases_in_original)):>9.1f}%")
    print(f"  {'6. With ≥1 GT drug in honest embeddings':<45} {len(diseases_with_gt_drugs):>10,} {100*(1-len(diseases_with_gt_drugs)/len(diseases_in_honest)):>9.1f}%")
    print()
    print(f"  Total attrition: {100*(1-len(diseases_with_gt_drugs)/len(step1_diseases)):.1f}% "
          f"({len(step1_diseases)} → {len(diseases_with_gt_drugs)})")
    print()

    # Save results
    results: dict[str, Any] = {
        "analysis": "selection_funnel",
        "description": "Documents filtering from Every Cure GT to final evaluation set",
        "funnel": {
            "step_1_raw_diseases": len(step1_diseases),
            "step_2_after_empty_removal": len(step2_diseases),
            "step_3_with_mesh_id": len(diseases_with_mesh),
            "step_4_in_original_embeddings": len(diseases_in_original),
            "step_5_in_honest_embeddings": len(diseases_in_honest),
            "step_6_final_with_gt_drugs": len(diseases_with_gt_drugs),
        },
        "drug_funnel": {
            "step_1_raw_drugs": len(step1_drugs),
            "step_2_after_empty_removal": len(step2_drugs),
            "step_3_with_drugbank_id": len(drugs_with_drugbank),
            "step_4_in_original_embeddings": len(drugs_in_original),
            "step_5_in_honest_embeddings": len(drugs_in_honest),
        },
        "gt_pairs": {
            "raw_rows": step1_total_rows,
            "valid_rows": len(valid_rows),
            "evaluable_pairs": sum(len(v) for v in gt_pairs.values()),
        },
        "dropped_diseases": {
            "no_mesh_mapping": dropped_at_mesh,
            "disconnected_after_treatment_removal": dropped_at_embedding,
            "total_without_mesh": len(diseases_without_mesh),
            "total_disconnected": len(diseases_in_original) - len(diseases_in_honest),
        },
        "retention_rates": {
            "mesh_mapping_rate": len(diseases_with_mesh) / len(step2_diseases),
            "embedding_coverage_original": len(diseases_in_original) / len(diseases_with_mesh),
            "embedding_coverage_honest": len(diseases_in_honest) / len(diseases_in_original),
            "final_retention": len(diseases_with_gt_drugs) / len(step1_diseases),
        },
    }

    output_path = ANALYSIS_DIR / "selection_funnel.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
