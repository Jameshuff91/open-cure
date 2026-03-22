"""h901: Analyze and resolve the MeSH mapping gap for Every Cure diseases.

Phase 1 (this script, autonomous): Programmatic resolution
- Exact substring matching against DRKG disease names
- Case/punctuation normalization
- Common suffix stripping (syndrome, disease, disorder)
- Plural/singular normalization
- Outputs candidates for manual/LLM review

Phase 2 (future session): LLM-assisted resolution for remaining gaps
Phase 3 (future session): Validation and re-evaluation

This script is READ-ONLY on existing data — safe to run alongside holdout evaluation.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent
REF_DIR = DATA_DIR / "data" / "reference"
OUTPUT_DIR = DATA_DIR / "data" / "analysis"


def normalize_disease_name(name: str) -> str:
    """Normalize disease name for fuzzy matching."""
    name = name.lower().strip()
    # Remove common suffixes that vary between databases
    suffixes = [
        ', unspecified', ', nos', ' nos', ' (disorder)',
        ' (disease)', ' (finding)', ', type 2', ', type 1',
        ', type ii', ', type i',
    ]
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[:-len(suffix)].strip()
    # Normalize whitespace and punctuation
    name = re.sub(r'\s+', ' ', name)
    name = re.sub(r"[''`]", "'", name)
    return name


def extract_mesh_id_from_drkg(entity: str) -> tuple[str, str]:
    """Extract MESH ID and build a lookup name from DRKG entity."""
    # Format: Disease::MESH:D001234 or Disease::MESH:C012345
    if 'Disease::MESH:' in entity:
        mesh_id = entity.split('Disease::MESH:')[1]
        return mesh_id, entity
    return '', entity


def build_drkg_disease_index(entities: list[str]) -> dict[str, str]:
    """Build normalized name → DRKG entity index from MeSH disease names."""
    # We need MeSH descriptor names — load from disease_ontology_mapping
    index: dict[str, str] = {}

    # Also build from the mesh_mappings file (name → MESH ID)
    mesh_path = REF_DIR / "mesh_mappings_from_agents.json"
    if mesh_path.exists():
        with open(mesh_path) as f:
            mesh_data = json.load(f)
        for batch in mesh_data.values():
            if isinstance(batch, dict):
                for name, mesh_id in batch.items():
                    if mesh_id:
                        normalized = normalize_disease_name(name)
                        drkg_id = f"drkg:Disease::MESH:{mesh_id}"
                        if drkg_id.replace('drkg:', '') in [str(e) for e in entities]:
                            index[normalized] = drkg_id

    return index


def try_programmatic_resolution(
    unmapped: list[str],
    drkg_entities: list[str],
    existing_mappings: dict[str, str],
) -> tuple[dict[str, str], list[str]]:
    """Try to resolve unmapped names programmatically.

    Returns:
        (resolved_dict, still_unmapped_list)
    """
    resolved: dict[str, str] = {}
    still_unmapped: list[str] = []

    # Build DRKG disease name lookup from entities
    drkg_disease_names: dict[str, str] = {}
    for entity in drkg_entities:
        entity_str = str(entity)
        if 'Disease::' in entity_str:
            # Extract the name part after Disease::
            parts = entity_str.split('Disease::')
            if len(parts) > 1:
                name_part = parts[1]
                # MESH IDs
                if name_part.startswith('MESH:'):
                    mesh_id = name_part
                    drkg_disease_names[mesh_id] = f"drkg:{entity_str}"
                # DOID IDs
                elif name_part.startswith('DOID:'):
                    drkg_disease_names[name_part] = f"drkg:{entity_str}"

    # Build reverse lookup: known names → MESH IDs from existing mappings
    name_to_mesh: dict[str, str] = {}
    for name, drkg_id in existing_mappings.items():
        name_to_mesh[normalize_disease_name(name)] = drkg_id

    # Strategy 1: Exact match after normalization
    for disease in unmapped:
        normalized = normalize_disease_name(disease)
        if normalized in name_to_mesh:
            resolved[disease] = name_to_mesh[normalized]
            continue

    # Strategy 2: Common disease name transformations
    transformations = [
        # "X disease" ↔ "X"
        (r'^(.+)\s+disease$', r'\1'),
        (r'^(.+)$', r'\1 disease'),
        # "X syndrome" ↔ "X"
        (r'^(.+)\s+syndrome$', r'\1'),
        # "X disorder" ↔ "X"
        (r'^(.+)\s+disorder$', r'\1'),
        # "X cancer" ↔ "X neoplasm" ↔ "X carcinoma"
        (r'^(.+)\s+cancer$', r'\1 neoplasm'),
        (r'^(.+)\s+cancer$', r'\1 carcinoma'),
        (r'^(.+)\s+neoplasm$', r'\1 cancer'),
        (r'^(.+)\s+carcinoma$', r'\1 cancer'),
        # Plural/singular
        (r'^(.+)s$', r'\1'),
        (r'^(.+)$', r'\1s'),
        # "acute X" ↔ "X"
        (r'^acute\s+(.+)$', r'\1'),
        (r'^chronic\s+(.+)$', r'\1'),
    ]

    remaining = [d for d in unmapped if d not in resolved]
    for disease in remaining:
        normalized = normalize_disease_name(disease)
        found = False
        for pattern, replacement in transformations:
            transformed = re.sub(pattern, replacement, normalized)
            if transformed != normalized and transformed in name_to_mesh:
                resolved[disease] = name_to_mesh[transformed]
                found = True
                break
        if not found:
            still_unmapped.append(disease)

    return resolved, still_unmapped


def analyze_unmapped_categories(unmapped: list[str]) -> dict[str, list[str]]:
    """Categorize unmapped diseases to prioritize resolution effort."""
    categories: dict[str, list[str]] = defaultdict(list)

    cancer_terms = ['cancer', 'carcinoma', 'lymphoma', 'leukemia', 'melanoma',
                    'sarcoma', 'myeloma', 'tumor', 'tumour', 'neoplasm', 'glioma',
                    'blastoma', 'mesothelioma']
    rare_terms = ['syndrome', 'congenital', 'hereditary', 'familial', 'genetic',
                  'neonatal', 'infantile', 'juvenile']
    infectious_terms = ['infection', 'infectious', 'viral', 'bacterial', 'fungal',
                        'parasitic', 'tuberculosis', 'hepatitis', 'hiv', 'malaria']
    neuro_terms = ['neuropathy', 'encephalopathy', 'epilepsy', 'seizure',
                   'dementia', 'ataxia', 'dystonia', 'palsy']

    for disease in unmapped:
        d_lower = disease.lower()
        if any(t in d_lower for t in cancer_terms):
            categories['cancer'].append(disease)
        elif any(t in d_lower for t in rare_terms):
            categories['rare_genetic'].append(disease)
        elif any(t in d_lower for t in infectious_terms):
            categories['infectious'].append(disease)
        elif any(t in d_lower for t in neuro_terms):
            categories['neurological'].append(disease)
        else:
            categories['other'].append(disease)

    return dict(categories)


def main() -> None:
    print("h901: MeSH Mapping Gap Analysis")
    print("=" * 60)

    # Load Every Cure diseases
    ec_path = REF_DIR / "everycure" / "indicationList.xlsx"
    df = pd.read_excel(ec_path)
    ec_diseases = sorted(set(df['final normalized disease label'].dropna().unique()))
    print(f"Unique Every Cure diseases: {len(ec_diseases)}")

    # Load existing mappings
    with open(REF_DIR / "mesh_mappings_from_agents.json") as f:
        mesh_data = json.load(f)
    existing_mappings: dict[str, str] = {}
    for batch in mesh_data.values():
        if isinstance(batch, dict):
            for name, mesh_id in batch.items():
                if mesh_id:
                    mesh_str = str(mesh_id)
                    if mesh_str.startswith("D") or mesh_str.startswith("C"):
                        existing_mappings[name.lower()] = f"drkg:Disease::MESH:{mesh_str}"

    mapped_count = sum(1 for d in ec_diseases if d.lower() in existing_mappings)
    unmapped = [d for d in ec_diseases if d.lower() not in existing_mappings]
    print(f"Currently mapped: {mapped_count} ({100*mapped_count/len(ec_diseases):.1f}%)")
    print(f"Currently unmapped: {len(unmapped)} ({100*len(unmapped)/len(ec_diseases):.1f}%)")

    # Load DRKG entities
    entities = np.load(
        DATA_DIR / "data" / "embeddings" / "node2vec_256_entities.npy",
        allow_pickle=True
    )
    drkg_diseases = [e for e in entities if 'Disease::' in str(e)]
    print(f"DRKG disease entities: {len(drkg_diseases)}")

    # Try programmatic resolution
    print("\n" + "=" * 60)
    print("Phase 1: Programmatic Resolution")
    print("=" * 60)

    resolved, still_unmapped = try_programmatic_resolution(
        unmapped, entities, existing_mappings
    )
    print(f"Programmatically resolved: {len(resolved)}")
    print(f"Still unmapped: {len(still_unmapped)}")

    if resolved:
        print("\nSample resolutions:")
        for name, drkg_id in list(resolved.items())[:20]:
            print(f"  {name} → {drkg_id}")

    # Categorize remaining unmapped
    print("\n" + "=" * 60)
    print("Unmapped Disease Categories (for LLM prioritization)")
    print("=" * 60)

    categories = analyze_unmapped_categories(still_unmapped)
    for cat, diseases in sorted(categories.items(), key=lambda x: -len(x[1])):
        print(f"\n{cat}: {len(diseases)} diseases")
        for d in diseases[:5]:
            print(f"  - {d}")
        if len(diseases) > 5:
            print(f"  ... and {len(diseases) - 5} more")

    # Count GT pairs for unmapped diseases
    ec_pairs_unmapped = len(df[df['final normalized disease label'].isin(still_unmapped)])
    ec_pairs_total = len(df)
    print(f"\n{'=' * 60}")
    print(f"Impact Assessment")
    print(f"{'=' * 60}")
    print(f"GT pairs behind unmapped diseases: {ec_pairs_unmapped} / {ec_pairs_total} "
          f"({100*ec_pairs_unmapped/ec_pairs_total:.1f}%)")
    print(f"If resolved, evaluable diseases could grow from {mapped_count} "
          f"to {mapped_count + len(resolved) + len(still_unmapped)}")

    # Save results
    output = {
        'summary': {
            'total_ec_diseases': len(ec_diseases),
            'currently_mapped': mapped_count,
            'programmatically_resolved': len(resolved),
            'still_unmapped': len(still_unmapped),
            'drkg_disease_entities': len(drkg_diseases),
            'gt_pairs_behind_unmapped': ec_pairs_unmapped,
            'gt_pairs_total': ec_pairs_total,
        },
        'resolved': {k: v for k, v in resolved.items()},
        'unmapped_by_category': {
            cat: diseases for cat, diseases in categories.items()
        },
        'unmapped_sample': still_unmapped[:200],
    }

    output_path = OUTPUT_DIR / "h901_mesh_mapping_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Save unmapped list for LLM batch resolution
    unmapped_path = OUTPUT_DIR / "h901_unmapped_diseases.txt"
    with open(unmapped_path, 'w') as f:
        for d in sorted(still_unmapped):
            f.write(d + '\n')
    print(f"Unmapped disease list saved to: {unmapped_path}")


if __name__ == '__main__':
    main()
