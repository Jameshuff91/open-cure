#!/usr/bin/env python3
"""
Hypothesis h206: Manual Rule Injection for Missing DRKG Drugs.

PURPOSE:
    h205 found Adcetris is missing from DRKG despite FDA approval for lymphoma.
    This script:
    1. Identifies ALL FDA-approved biologics in GT that are NOT in DRKG
    2. Creates a manual lookup table for missing drug→disease mappings
    3. Quantifies the impact (how many predictions are blocked)

APPROACH:
    1. Load GT drugs
    2. Identify biologics (mab/cept/ase suffixes)
    3. Check which have DRKG embeddings
    4. Output missing drugs with their indications
"""

import json
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"


def load_node2vec_embeddings():
    """Load node2vec embeddings."""
    df = pd.read_csv(EMBEDDINGS_DIR / "node2vec_256_named.csv")
    entities = set()
    for _, row in df.iterrows():
        entities.add(f"drkg:{row['entity']}")
    return entities


def load_drugbank_lookup():
    """Load drugbank name-to-ID mappings."""
    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)
    name_to_id = {name.lower(): f"drkg:Compound::{db_id}" for db_id, name in id_to_name.items()}
    return name_to_id


def is_biologic(drug_name: str) -> bool:
    """Check if drug is a biologic based on naming conventions."""
    drug_lower = drug_name.lower()
    biologic_suffixes = [
        'mab',   # monoclonal antibodies
        'cept',  # receptor fusion proteins (e.g., etanercept)
        'ase',   # enzymes (e.g., asparaginase)
        'tinib', # tyrosine kinase inhibitors (actually small molecules)
        'ximab', # chimeric mAbs
        'zumab', # humanized mAbs
        'umab',  # fully human mAbs
        'tuximab', # ADCs
    ]
    # More specific patterns for biologics
    if any(drug_lower.endswith(suffix) for suffix in ['mab', 'cept']):
        return True
    # Check for known biologics
    known_biologics = [
        'vedotin',  # ADC payload indicator (brentuximab vedotin, etc.)
        'emtansine',  # ADC payload
        'ozogamicin',  # ADC payload
        'ravtansine',  # ADC payload
    ]
    if any(kw in drug_lower for kw in known_biologics):
        return True
    return False


def main():
    print("=" * 70)
    print("h206: Missing DRKG Drugs Analysis")
    print("=" * 70)
    print()

    # Load data
    print("Loading data...")
    drkg_entities = load_node2vec_embeddings()
    name_to_id = load_drugbank_lookup()
    gt_df = pd.read_excel(REFERENCE_DIR / "everycure" / "indicationList.xlsx")

    print(f"DRKG entities with embeddings: {len(drkg_entities)}")
    print(f"DrugBank name mappings: {len(name_to_id)}")
    print(f"GT entries: {len(gt_df)}")

    # Get unique drugs from GT
    gt_drugs = gt_df['final normalized drug label'].dropna().unique()
    print(f"Unique drugs in GT: {len(gt_drugs)}")

    # Analyze drug coverage
    drugs_with_embedding = []
    drugs_without_embedding = []
    biologics_missing = []

    for drug in gt_drugs:
        drug_lower = drug.lower()
        drug_id = name_to_id.get(drug_lower)

        has_embedding = drug_id is not None and drug_id in drkg_entities
        is_bio = is_biologic(drug)

        drug_indications = gt_df[gt_df['final normalized drug label'].str.lower() == drug_lower]['final normalized disease label'].unique()

        if has_embedding:
            drugs_with_embedding.append(drug)
        else:
            drugs_without_embedding.append({
                'drug': drug,
                'is_biologic': is_bio,
                'n_indications': len(drug_indications),
                'indications': list(drug_indications)[:10],
                'drug_id_exists': drug_id is not None,
            })
            if is_bio:
                biologics_missing.append({
                    'drug': drug,
                    'n_indications': len(drug_indications),
                    'indications': list(drug_indications),
                })

    print(f"\n=== DRUG COVERAGE SUMMARY ===")
    print(f"Drugs WITH embeddings: {len(drugs_with_embedding)} ({len(drugs_with_embedding)/len(gt_drugs)*100:.1f}%)")
    print(f"Drugs WITHOUT embeddings: {len(drugs_without_embedding)} ({len(drugs_without_embedding)/len(gt_drugs)*100:.1f}%)")
    print(f"Biologics missing: {len(biologics_missing)}")

    # Sort missing biologics by impact (number of indications)
    biologics_missing.sort(key=lambda x: x['n_indications'], reverse=True)

    print(f"\n=== TOP 20 MISSING BIOLOGICS BY IMPACT ===")
    for i, bio in enumerate(biologics_missing[:20]):
        print(f"{i+1}. {bio['drug']} ({bio['n_indications']} indications)")
        print(f"    Indications: {', '.join(bio['indications'][:5])}{'...' if len(bio['indications']) > 5 else ''}")

    # Check specific missing drugs mentioned in h205
    print(f"\n=== SPECIFIC DRUGS FROM h205 ===")
    known_missing = ['adcetris', 'brentuximab vedotin', 'brentuximab']
    for drug_pattern in known_missing:
        found = [d for d in drugs_without_embedding if drug_pattern in d['drug'].lower()]
        if found:
            for d in found:
                print(f"  {d['drug']}: {d['n_indications']} indications")
                print(f"    {d['indications']}")

    # Aggregate missing drugs by indication count
    print(f"\n=== IMPACT ANALYSIS ===")
    total_blocked_pairs = sum(d['n_indications'] for d in drugs_without_embedding)
    biologic_blocked_pairs = sum(b['n_indications'] for b in biologics_missing)

    print(f"Total drug-disease pairs blocked by missing embeddings: {total_blocked_pairs}")
    print(f"Biologic drug-disease pairs blocked: {biologic_blocked_pairs}")

    # Group missing drugs by type
    print(f"\n=== MISSING DRUGS BY SUFFIX ===")
    suffix_counts = defaultdict(list)
    for drug_info in drugs_without_embedding:
        drug = drug_info['drug']
        if drug.lower().endswith('mab'):
            suffix_counts['mab'].append(drug)
        elif drug.lower().endswith('cept'):
            suffix_counts['cept'].append(drug)
        elif drug.lower().endswith('ase'):
            suffix_counts['ase'].append(drug)
        elif 'vedotin' in drug.lower() or 'emtansine' in drug.lower():
            suffix_counts['ADC'].append(drug)

    for suffix, drugs in sorted(suffix_counts.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  -{suffix}: {len(drugs)} drugs")
        for d in drugs[:5]:
            print(f"    • {d}")
        if len(drugs) > 5:
            print(f"    ... and {len(drugs) - 5} more")

    # Create manual rules lookup for top missing biologics
    print(f"\n=== GENERATING MANUAL RULES ===")
    manual_rules = []
    for bio in biologics_missing[:30]:  # Top 30 by impact
        rule = {
            'drug_name': bio['drug'],
            'indications': bio['indications'],
            'n_indications': bio['n_indications'],
            'rule_type': 'manual_injection',
            'source': 'every_cure_gt',
        }
        manual_rules.append(rule)

    print(f"Created {len(manual_rules)} manual rules for missing biologics")

    # Save findings
    output = {
        'hypothesis': 'h206',
        'title': 'Missing DRKG Drugs Analysis',
        'summary': {
            'total_gt_drugs': len(gt_drugs),
            'drugs_with_embeddings': len(drugs_with_embedding),
            'drugs_without_embeddings': len(drugs_without_embedding),
            'biologics_missing': len(biologics_missing),
            'total_blocked_pairs': total_blocked_pairs,
            'biologic_blocked_pairs': biologic_blocked_pairs,
        },
        'missing_biologics_top_20': biologics_missing[:20],
        'manual_rules': manual_rules,
        'suffix_counts': {k: len(v) for k, v in suffix_counts.items()},
    }

    output_path = ANALYSIS_DIR / "h206_missing_drkg_drugs.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved findings to: {output_path}")

    # Also save the manual rules as a separate lookup table
    rules_path = REFERENCE_DIR / "manual_drug_rules.json"
    with open(rules_path, 'w') as f:
        json.dump({'rules': manual_rules, 'version': '1.0', 'source': 'h206_analysis'}, f, indent=2)

    print(f"Saved manual rules to: {rules_path}")

    return output


if __name__ == "__main__":
    main()
