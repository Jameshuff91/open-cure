#!/usr/bin/env python3
"""
h90: Create Zero-Shot Benchmark - Diseases with No FDA-Approved Treatments

This script identifies diseases that:
1. Are in the Every Cure indication list
2. Have ZERO FDA-approved treatments
3. May or may not exist in DRKG

These represent the TRUE zero-shot challenge: finding first treatments for untreated diseases.
"""

import json
from pathlib import Path
import pandas as pd

# Paths
DATA_DIR = Path("data/reference")
EVERYCURE_FILE = DATA_DIR / "everycure/indicationList.xlsx"
DRKG_DISEASES_FILE = DATA_DIR / "disease_ontology_mapping.json"
OUTPUT_FILE = Path("data/analysis/zero_shot_benchmark.json")


def normalize_disease_name(name: str) -> str:
    """Normalize disease name for matching."""
    if not name:
        return ""
    return name.lower().strip().replace("'", "").replace("-", " ").replace("_", " ")


def load_everycure_diseases():
    """Load diseases from Every Cure, categorized by FDA approval status."""
    df = pd.read_excel(EVERYCURE_FILE)
    print(f"Every Cure indication list: {len(df)} rows")

    disease_col = 'final normalized disease label'
    drug_col = 'final normalized drug label'

    # All unique diseases
    all_diseases = df[disease_col].dropna().unique().tolist()
    print(f"Unique diseases: {len(all_diseases)}")

    # Diseases with at least 1 FDA-approved drug
    fda_approved_pairs = df[df['FDA'] == 1.0]
    diseases_with_fda = set(fda_approved_pairs[disease_col].dropna().unique())
    print(f"Diseases with FDA-approved treatments: {len(diseases_with_fda)}")

    # Diseases with NO FDA-approved drugs
    diseases_without_fda = [d for d in all_diseases if d not in diseases_with_fda]
    print(f"Diseases with NO FDA-approved treatments: {len(diseases_without_fda)}")

    # For each no-FDA disease, get the potential repurposing candidates mentioned
    potential_treatments = {}
    for disease in diseases_without_fda:
        disease_rows = df[df[disease_col] == disease]
        drugs = disease_rows[drug_col].dropna().unique().tolist()
        potential_treatments[disease] = drugs

    return all_diseases, diseases_with_fda, diseases_without_fda, potential_treatments


def check_drkg_presence(diseases: list) -> tuple[list, list]:
    """Check which diseases have any presence in DRKG."""
    # Load DRKG disease mapping
    with open(DRKG_DISEASES_FILE) as f:
        mapping = json.load(f)

    # Get normalized DRKG disease names
    drkg_diseases_raw = set()
    name_to_doid = mapping.get('name_to_doid', {})
    doid_to_name = mapping.get('doid_to_name', {})

    # Collect all disease name variants
    for name in name_to_doid.keys():
        drkg_diseases_raw.add(normalize_disease_name(name))
    for name in doid_to_name.values():
        drkg_diseases_raw.add(normalize_disease_name(name))

    print(f"DRKG disease name variants: {len(drkg_diseases_raw)}")

    in_drkg = []
    not_in_drkg = []

    for disease in diseases:
        norm = normalize_disease_name(disease)
        # Exact match or substring match
        found = norm in drkg_diseases_raw
        if not found:
            # Try substring matching
            for drkg_name in drkg_diseases_raw:
                if norm in drkg_name or drkg_name in norm:
                    found = True
                    break

        if found:
            in_drkg.append(disease)
        else:
            not_in_drkg.append(disease)

    return in_drkg, not_in_drkg


def main():
    print("=" * 70)
    print("h90: Creating Zero-Shot Benchmark")
    print("    Diseases with NO FDA-Approved Treatments")
    print("=" * 70)
    print()

    # Load Every Cure data
    all_diseases, with_fda, without_fda, potential_treatments = load_everycure_diseases()

    print()

    # Check DRKG presence
    in_drkg, not_in_drkg = check_drkg_presence(without_fda)

    print(f"\nOf {len(without_fda)} diseases without FDA treatments:")
    print(f"  - In DRKG (can predict): {len(in_drkg)}")
    print(f"  - Not in DRKG (need literature): {len(not_in_drkg)}")

    # Build results
    results = {
        'benchmark_diseases': [
            {
                'disease': d,
                'normalized': normalize_disease_name(d),
                'in_drkg': d in in_drkg,
                'potential_treatments': potential_treatments.get(d, [])[:10]  # Top 10
            }
            for d in without_fda
        ],
        'diseases_in_drkg': in_drkg,
        'diseases_not_in_drkg': not_in_drkg,
        'stats': {
            'total_everycure_diseases': len(all_diseases),
            'diseases_with_fda_treatment': len(with_fda),
            'diseases_without_fda_treatment': len(without_fda),
            'benchmark_in_drkg': len(in_drkg),
            'benchmark_not_in_drkg': len(not_in_drkg)
        }
    }

    # Print sample
    print(f"\n{'=' * 70}")
    print("ZERO-SHOT BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Total Every Cure diseases:           {len(all_diseases)}")
    print(f"Diseases WITH FDA treatments:        {len(with_fda)}")
    print(f"Diseases WITHOUT FDA treatments:     {len(without_fda)} <- BENCHMARK")
    print(f"  - Present in DRKG:                 {len(in_drkg)}")
    print(f"  - Not in DRKG:                     {len(not_in_drkg)}")

    print(f"\nSample zero-treatment diseases IN DRKG (first 20):")
    for d in sorted(in_drkg)[:20]:
        n_potential = len(potential_treatments.get(d, []))
        print(f"  - {d} ({n_potential} potential drugs)")

    print(f"\nSample zero-treatment diseases NOT in DRKG (first 20):")
    for d in sorted(not_in_drkg)[:20]:
        n_potential = len(potential_treatments.get(d, []))
        print(f"  - {d} ({n_potential} potential drugs)")

    # Save benchmark
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nBenchmark saved to: {OUTPUT_FILE}")

    # Success criteria
    n_benchmark = len(without_fda)
    if n_benchmark >= 50:
        print(f"\n✓ SUCCESS: Benchmark has {n_benchmark} diseases (target: 50+)")
    else:
        print(f"\n✗ BELOW TARGET: Only {n_benchmark} diseases (target: 50+)")

    return results


if __name__ == "__main__":
    results = main()
