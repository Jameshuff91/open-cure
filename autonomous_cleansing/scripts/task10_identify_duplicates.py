#!/usr/bin/env python3
"""
Task 10: Identify potential duplicate entities.

Finds entities that may be duplicates based on:
1. Same name within same type (case-insensitive)
2. Same external ID (DrugBank ID, NCBI Gene ID)
3. Cross-source duplicates (same entity in drkg, hetionet, primekg)

Outputs:
- duplicate_candidates.json: Grouped duplicates with confidence scores
- duplicate_summary.md: Human-readable summary
"""

import pandas as pd
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import re


PROJECT_ROOT = Path("/Users/jimhuff/github/open-cure")
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = DATA_DIR / "validation"


def extract_base_id(entity_id: str) -> str | None:
    """Extract the base identifier from various ID formats."""
    # drkg:Gene::25802;1734 -> 25802
    # drkg:Compound::DB00145 -> DB00145
    # hetionet:7124 -> 7124
    # primekg:gene/protein:25802 -> 25802
    # primekg:drug:DB00145 -> DB00145

    patterns = [
        r'Gene::(\d+)',           # NCBI gene ID
        r'Compound::(DB\d+)',     # DrugBank ID
        r'Compound::hmdb:(HMDB\d+)',  # HMDB ID
        r'hetionet:(\d+)',        # Hetionet gene
        r'hetionet:(DB\d+)',      # Hetionet drug
        r'primekg:gene/protein:(\d+)',  # PrimeKG gene
        r'primekg:drug:(DB\d+)',  # PrimeKG drug
        r'::(C\d+)',              # MESH/other compound IDs
    ]

    for pattern in patterns:
        match = re.search(pattern, entity_id)
        if match:
            return match.group(1)
    return None


def calculate_confidence(group: list[dict]) -> float:
    """
    Calculate confidence that a group represents true duplicates.

    Factors:
    - Same external ID (high confidence)
    - Same name (medium confidence)
    - Different sources (increases confidence)
    """
    confidence = 0.0

    # Check for shared external IDs
    external_ids = set()
    for entity in group:
        base_id = extract_base_id(entity['id'])
        if base_id:
            external_ids.add(base_id)

    # If all entities share the same base ID, high confidence
    if len(external_ids) == 1:
        confidence = 0.95
    elif external_ids:
        # Some shared IDs
        confidence = 0.80
    else:
        # Only name matching
        confidence = 0.60

    # Boost confidence if from multiple sources
    sources = set(entity['source'] for entity in group)
    if len(sources) > 1:
        confidence = min(confidence + 0.05, 0.99)

    return round(confidence, 2)


def find_duplicates_by_name(df: pd.DataFrame) -> dict:
    """Find potential duplicates by matching names within same type."""

    df['name_normalized'] = df['name'].str.lower().str.strip()

    duplicates = defaultdict(list)

    # Group by type and normalized name
    for (entity_type, name), group in df.groupby(['type', 'name_normalized']):
        if len(group) > 1:
            key = f"{entity_type}::{name}"
            for _, row in group.iterrows():
                duplicates[key].append({
                    'id': row['id'],
                    'name': row['name'],
                    'type': row['type'],
                    'source': row['source'],
                    'drugbank_id': row.get('drugbank_id', ''),
                    'ncbi_gene_id': row.get('ncbi_gene_id', '')
                })

    return duplicates


def find_duplicates_by_external_id(df: pd.DataFrame) -> dict:
    """Find duplicates by shared external IDs (DrugBank, NCBI Gene)."""

    duplicates = defaultdict(list)

    # Group drugs by DrugBank ID
    drugs = df[df['type'] == 'Drug'].copy()
    drugs['drugbank_id'] = drugs['drugbank_id'].fillna('')

    for db_id, group in drugs.groupby('drugbank_id'):
        if db_id and len(group) > 1:
            key = f"DrugBank::{db_id}"
            for _, row in group.iterrows():
                duplicates[key].append({
                    'id': row['id'],
                    'name': row['name'],
                    'type': row['type'],
                    'source': row['source'],
                    'drugbank_id': str(db_id)
                })

    # Group genes by NCBI Gene ID
    genes = df[df['type'] == 'Gene'].copy()
    genes['ncbi_gene_id'] = genes['ncbi_gene_id'].fillna('')

    for gene_id, group in genes.groupby('ncbi_gene_id'):
        if gene_id and gene_id != '' and len(group) > 1:
            key = f"NCBI::{int(float(gene_id)) if isinstance(gene_id, (int, float)) else gene_id}"
            for _, row in group.iterrows():
                duplicates[key].append({
                    'id': row['id'],
                    'name': row['name'],
                    'type': row['type'],
                    'source': row['source'],
                    'ncbi_gene_id': str(gene_id)
                })

    return duplicates


def merge_duplicate_groups(name_dups: dict, id_dups: dict) -> list[dict]:
    """Merge and deduplicate the duplicate groups."""

    # Track which entity IDs we've already grouped
    processed_ids = set()
    merged_groups = []

    # Process external ID duplicates first (higher confidence)
    for key, entities in id_dups.items():
        entity_ids = frozenset(e['id'] for e in entities)
        if entity_ids not in processed_ids:
            processed_ids.add(entity_ids)
            confidence = calculate_confidence(entities)
            merged_groups.append({
                'match_key': key,
                'match_type': 'external_id',
                'confidence': confidence,
                'count': len(entities),
                'entities': entities
            })

    # Add name-based duplicates that weren't already covered
    for key, entities in name_dups.items():
        entity_ids = frozenset(e['id'] for e in entities)

        # Check if this group overlaps with an existing group
        already_covered = False
        for group in merged_groups:
            existing_ids = frozenset(e['id'] for e in group['entities'])
            if entity_ids & existing_ids:  # Any overlap
                already_covered = True
                break

        if not already_covered:
            confidence = calculate_confidence(entities)
            merged_groups.append({
                'match_key': key,
                'match_type': 'name',
                'confidence': confidence,
                'count': len(entities),
                'entities': entities
            })

    return sorted(merged_groups, key=lambda x: (-x['confidence'], -x['count']))


def generate_summary(groups: list[dict], total_nodes: int) -> str:
    """Generate a markdown summary of duplicate analysis."""

    total_groups = len(groups)
    total_duplicates = sum(g['count'] for g in groups)
    high_confidence = sum(1 for g in groups if g['confidence'] >= 0.9)
    medium_confidence = sum(1 for g in groups if 0.7 <= g['confidence'] < 0.9)
    low_confidence = sum(1 for g in groups if g['confidence'] < 0.7)

    # Count by type
    type_counts = defaultdict(int)
    for group in groups:
        entity_type = group['entities'][0]['type']
        type_counts[entity_type] += group['count']

    # Count by match type
    by_external_id = sum(1 for g in groups if g['match_type'] == 'external_id')
    by_name = sum(1 for g in groups if g['match_type'] == 'name')

    summary = f"""# Duplicate Entity Analysis

Generated: {datetime.now().isoformat()}

## Summary

| Metric | Value |
|--------|-------|
| Total nodes | {total_nodes:,} |
| Duplicate groups | {total_groups:,} |
| Total duplicated entities | {total_duplicates:,} |
| % of nodes in duplicate groups | {100*total_duplicates/total_nodes:.1f}% |

## Confidence Breakdown

| Confidence Level | Groups |
|-----------------|--------|
| High (≥0.90) | {high_confidence:,} |
| Medium (0.70-0.89) | {medium_confidence:,} |
| Low (<0.70) | {low_confidence:,} |

## Match Type

| Match Type | Groups |
|-----------|--------|
| External ID match | {by_external_id:,} |
| Name-only match | {by_name:,} |

## By Entity Type

| Type | Duplicated Entities |
|------|-------------------|
"""

    for entity_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        summary += f"| {entity_type} | {count:,} |\n"

    summary += f"""
## Top Duplicate Groups

### High Confidence (likely true duplicates)

"""

    high_conf_groups = [g for g in groups if g['confidence'] >= 0.9][:20]
    for i, group in enumerate(high_conf_groups, 1):
        summary += f"**{i}. {group['match_key']}** (confidence: {group['confidence']})\n"
        for entity in group['entities'][:5]:
            summary += f"   - `{entity['id']}` ({entity['source']})\n"
        if len(group['entities']) > 5:
            summary += f"   - ... and {len(group['entities']) - 5} more\n"
        summary += "\n"

    summary += """
## Recommendations

1. **High confidence duplicates**: Safe to merge automatically. Keep the most complete record.
2. **Medium confidence**: Review before merging. May be related but distinct entities.
3. **Low confidence**: Manual review needed. May be false positives.

### Merge Strategy

For true duplicates:
1. Create a canonical ID (prefer Hetionet > PrimeKG > DRKG for consistency)
2. Preserve all external IDs as aliases
3. Update edge references to point to canonical ID
4. Log the merge in transformation history

### Next Steps

1. Review this report
2. Implement merge logic for high-confidence groups
3. Create edge reference update script
4. Re-validate data integrity
"""

    return summary


def main():
    print("Loading cleaned nodes...")
    df = pd.read_csv(DATA_DIR / "unified_nodes_clean.csv", low_memory=False)
    total_nodes = len(df)
    print(f"Loaded {total_nodes:,} nodes")

    print("\nFinding duplicates by name...")
    name_duplicates = find_duplicates_by_name(df)
    print(f"Found {len(name_duplicates):,} name-based duplicate groups")

    print("\nFinding duplicates by external ID...")
    id_duplicates = find_duplicates_by_external_id(df)
    print(f"Found {len(id_duplicates):,} ID-based duplicate groups")

    print("\nMerging and analyzing duplicate groups...")
    merged_groups = merge_duplicate_groups(name_duplicates, id_duplicates)
    print(f"Total merged groups: {len(merged_groups):,}")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save JSON output
    output_json = OUTPUT_DIR / "duplicate_candidates.json"
    with open(output_json, 'w') as f:
        json.dump({
            'generated': datetime.now().isoformat(),
            'total_nodes': total_nodes,
            'total_groups': len(merged_groups),
            'total_duplicates': sum(g['count'] for g in merged_groups),
            'groups': merged_groups
        }, f, indent=2)
    print(f"\nSaved duplicate candidates to: {output_json}")

    # Generate and save summary
    summary = generate_summary(merged_groups, total_nodes)
    output_md = OUTPUT_DIR / "duplicate_summary.md"
    with open(output_md, 'w') as f:
        f.write(summary)
    print(f"Saved summary to: {output_md}")

    # Print quick stats
    print("\n" + "="*50)
    print("DUPLICATE ANALYSIS COMPLETE")
    print("="*50)
    high_conf = sum(1 for g in merged_groups if g['confidence'] >= 0.9)
    print(f"High confidence groups (≥0.9): {high_conf:,}")
    print(f"Total duplicated entities: {sum(g['count'] for g in merged_groups):,}")
    print(f"Percentage of total: {100*sum(g['count'] for g in merged_groups)/total_nodes:.1f}%")

    return merged_groups


if __name__ == "__main__":
    main()
