#!/usr/bin/env python3
"""
Task 1: Normalize Entity Types

Merges duplicate entity types into consistent PascalCase naming:
- Gene + gene/protein → Gene
- Drug + drug + Compound → Drug
- Disease + disease → Disease
- biological_process + BiologicalProcess → BiologicalProcess
- molecular_function + MolecularFunction → MolecularFunction
- cellular_component + CellularComponent → CellularComponent
- pathway + Pathway → Pathway
- anatomy + Anatomy → Anatomy
- effect/phenotype → Phenotype
- exposure → Exposure
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from collections import Counter

# Project paths
PROJECT_ROOT = Path("/Users/jimhuff/github/open-cure")
DATA_DIR = PROJECT_ROOT / "data" / "processed"
LOG_DIR = PROJECT_ROOT / "autonomous_cleansing"

# Type mapping: old_type → new_type
TYPE_MAPPING = {
    # Gene types
    "Gene": "Gene",
    "gene/protein": "Gene",

    # Drug types
    "Drug": "Drug",
    "drug": "Drug",
    "Compound": "Drug",

    # Disease types
    "Disease": "Disease",
    "disease": "Disease",

    # Biological process types
    "BiologicalProcess": "BiologicalProcess",
    "biological_process": "BiologicalProcess",

    # Molecular function types
    "MolecularFunction": "MolecularFunction",
    "molecular_function": "MolecularFunction",

    # Cellular component types
    "CellularComponent": "CellularComponent",
    "cellular_component": "CellularComponent",

    # Pathway types
    "Pathway": "Pathway",
    "pathway": "Pathway",

    # Anatomy types
    "Anatomy": "Anatomy",
    "anatomy": "Anatomy",

    # Other types - normalize to PascalCase
    "effect/phenotype": "Phenotype",
    "exposure": "Exposure",
    "SideEffect": "SideEffect",
    "DrugClass": "DrugClass",
    "Symptom": "Symptom",
    "Tax": "Taxonomy",
}


def load_cleansing_log() -> dict:
    """Load existing cleansing log or create new one."""
    log_path = LOG_DIR / "cleansing_log.json"
    if log_path.exists():
        with open(log_path) as f:
            return json.load(f)
    return {
        "version": "1.0",
        "created": datetime.now().isoformat(),
        "transformations": []
    }


def save_cleansing_log(log: dict):
    """Save cleansing log."""
    log_path = LOG_DIR / "cleansing_log.json"
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)


def main():
    print("=" * 60)
    print("Task 1: Normalize Entity Types")
    print("=" * 60)

    # Load data
    input_file = DATA_DIR / "unified_nodes.csv"
    output_file = DATA_DIR / "unified_nodes_clean.csv"

    print(f"\nReading {input_file}...")
    df = pd.read_csv(input_file, low_memory=False)
    original_count = len(df)
    print(f"Total nodes: {original_count:,}")

    # Get before stats
    print("\n--- Before Normalization ---")
    before_types = df['type'].value_counts()
    print(f"Unique entity types: {len(before_types)}")
    print("\nType distribution:")
    for type_name, count in before_types.items():
        print(f"  {type_name}: {count:,}")

    # Apply type mapping
    print("\n--- Applying Type Mapping ---")
    transformation_counts = Counter()

    def normalize_type(t):
        if t in TYPE_MAPPING:
            new_type = TYPE_MAPPING[t]
            if t != new_type:
                transformation_counts[f"{t} → {new_type}"] += 1
            return new_type
        else:
            # Unknown type - log it
            print(f"  WARNING: Unknown type '{t}' - keeping as-is")
            return t

    df['type'] = df['type'].apply(normalize_type)

    # Get after stats
    print("\n--- After Normalization ---")
    after_types = df['type'].value_counts()
    print(f"Unique entity types: {len(after_types)}")
    print("\nType distribution:")
    for type_name, count in after_types.items():
        print(f"  {type_name}: {count:,}")

    # Print transformation summary
    print("\n--- Transformation Summary ---")
    for transformation, count in sorted(transformation_counts.items(), key=lambda x: -x[1]):
        print(f"  {transformation}: {count:,}")

    # Validate
    print("\n--- Validation ---")
    final_count = len(df)
    print(f"Original count: {original_count:,}")
    print(f"Final count: {final_count:,}")

    if original_count != final_count:
        print("ERROR: Row count changed!")
        return False
    else:
        print("✓ Row count unchanged")

    # Check for case-insensitive duplicates
    types_lower = {t.lower() for t in after_types.index}
    if len(types_lower) != len(after_types):
        print("ERROR: Still have case-insensitive duplicate types!")
        return False
    else:
        print("✓ No case-insensitive duplicate types")

    # Check all types are PascalCase
    non_pascal = [t for t in after_types.index if not t[0].isupper()]
    if non_pascal:
        print(f"WARNING: Non-PascalCase types: {non_pascal}")
    else:
        print("✓ All types use PascalCase")

    # Save cleaned data
    print(f"\n--- Saving to {output_file} ---")
    df.to_csv(output_file, index=False)
    print(f"Saved {final_count:,} nodes")

    # Update cleansing log
    log = load_cleansing_log()
    log["transformations"].append({
        "task": "Task 1: Normalize Entity Types",
        "timestamp": datetime.now().isoformat(),
        "input_file": str(input_file),
        "output_file": str(output_file),
        "original_types": len(before_types),
        "final_types": len(after_types),
        "row_count": final_count,
        "type_mapping": TYPE_MAPPING,
        "transformations": dict(transformation_counts)
    })
    save_cleansing_log(log)
    print("✓ Cleansing log updated")

    print("\n" + "=" * 60)
    print("Task 1 COMPLETE")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
