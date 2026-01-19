#!/usr/bin/env python3
"""
Generate statistics for Open Cure knowledge graph data.

This script generates comprehensive statistics about the nodes and edges,
used for tracking data quality before and after cleansing.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import Counter
import re


def analyze_nodes(nodes_path: Path) -> dict:
    """Analyze the nodes file and return statistics."""
    print(f"Reading nodes from {nodes_path}...")
    df = pd.read_csv(nodes_path, low_memory=False)

    stats = {
        "total_nodes": len(df),
        "columns": list(df.columns),
        "types": {},
        "type_summary": {
            "total_unique_types": df["type"].nunique(),
            "type_counts": df["type"].value_counts().to_dict()
        },
        "sources": df["source"].value_counts().to_dict() if "source" in df.columns else {},
        "coverage": {
            "has_name": int((~df["name"].isna() & (df["name"] != "")).sum()),
            "has_external_source": int((~df["external_source"].isna() & (df["external_source"] != "")).sum()) if "external_source" in df.columns else 0,
            "has_properties": int((~df["properties"].isna() & (df["properties"] != "")).sum()) if "properties" in df.columns else 0
        },
        "name_quality": analyze_name_quality(df)
    }

    # Calculate coverage percentages
    total = stats["total_nodes"]
    stats["coverage"]["pct_has_name"] = round(stats["coverage"]["has_name"] / total * 100, 2)
    stats["coverage"]["pct_has_external_source"] = round(stats["coverage"]["has_external_source"] / total * 100, 2) if "external_source" in df.columns else 0

    return stats


def analyze_name_quality(df: pd.DataFrame) -> dict:
    """Analyze the quality of entity names."""
    quality = {
        "id_only_patterns": {},
        "samples": {}
    }

    # Check for ID-only names (e.g., "Gene::2157", "Compound::DB02573")
    patterns = {
        "gene_id_only": r"^Gene::\d+$",
        "compound_drugbank": r"^Compound::DB\d+$",
        "disease_id_only": r"^Disease::\w+$"
    }

    for pattern_name, pattern in patterns.items():
        matches = df[df["name"].astype(str).str.match(pattern, na=False)]
        quality["id_only_patterns"][pattern_name] = len(matches)
        if len(matches) > 0:
            quality["samples"][pattern_name] = matches["name"].head(5).tolist()

    # Count drugs with ID-only names
    drug_mask = df["type"].str.lower().isin(["drug", "compound"])
    drugs = df[drug_mask]
    drugbank_id_pattern = r"DB\d{5}"
    drugs_with_id_names = drugs[drugs["name"].astype(str).str.contains(drugbank_id_pattern, na=False)]
    quality["drugs_with_id_only_names"] = len(drugs_with_id_names)
    quality["drugs_total"] = len(drugs)

    # Count genes with numeric-only names
    gene_mask = df["type"].str.lower().isin(["gene", "gene/protein"])
    genes = df[gene_mask]
    genes_with_numeric_names = genes[genes["name"].astype(str).str.match(r"^(Gene::)?\d+$", na=False)]
    quality["genes_with_numeric_only_names"] = len(genes_with_numeric_names)
    quality["genes_total"] = len(genes)

    return quality


def analyze_edges(edges_path: Path) -> dict:
    """Analyze the edges file and return statistics."""
    print(f"Reading edges from {edges_path}...")
    # Read in chunks for memory efficiency
    chunk_size = 1_000_000

    relation_counts: Counter = Counter()
    source_kg_counts: Counter = Counter()
    total_edges = 0

    for chunk in pd.read_csv(edges_path, chunksize=chunk_size, usecols=["relation", "source_kg"]):
        relation_counts.update(chunk["relation"].value_counts().to_dict())
        if "source_kg" in chunk.columns:
            source_kg_counts.update(chunk["source_kg"].value_counts().to_dict())
        total_edges += len(chunk)
        print(f"  Processed {total_edges:,} edges...")

    # Categorize relations by prefix
    relation_categories = categorize_relations(dict(relation_counts))

    stats = {
        "total_edges": total_edges,
        "relation_summary": {
            "total_unique_relations": len(relation_counts),
            "relation_counts": dict(relation_counts),
            "top_20_relations": dict(relation_counts.most_common(20))
        },
        "relation_categories": relation_categories,
        "source_kg_counts": dict(source_kg_counts)
    }

    return stats


def categorize_relations(relation_counts: dict) -> dict:
    """Categorize relations by their prefix/source."""
    categories = {
        "hetionet": {},
        "drugbank": {},
        "string": {},
        "intact": {},
        "gnbr": {},
        "bioarx": {},
        "other": {}
    }

    for relation, count in relation_counts.items():
        rel_lower = relation.lower()
        if rel_lower.startswith("hetionet::"):
            categories["hetionet"][relation] = count
        elif rel_lower.startswith("drugbank::"):
            categories["drugbank"][relation] = count
        elif rel_lower.startswith("string::"):
            categories["string"][relation] = count
        elif rel_lower.startswith("intact::"):
            categories["intact"][relation] = count
        elif rel_lower.startswith("gnbr::"):
            categories["gnbr"][relation] = count
        elif rel_lower.startswith("bioarx::"):
            categories["bioarx"][relation] = count
        else:
            categories["other"][relation] = count

    # Add counts per category
    summary = {}
    for cat, rels in categories.items():
        summary[cat] = {
            "count": len(rels),
            "total_edges": sum(rels.values()),
            "relations": rels
        }

    return summary


def generate_stats(data_dir: Path, output_path: Path, label: str = "before") -> dict:
    """Generate comprehensive statistics and save to JSON."""
    nodes_path = data_dir / "unified_nodes.csv"
    edges_path = data_dir / "unified_edges.csv"

    # Check for cleaned versions
    if label == "after":
        clean_nodes = data_dir / "unified_nodes_clean.csv"
        clean_edges = data_dir / "unified_edges_clean.csv"
        if clean_nodes.exists():
            nodes_path = clean_nodes
        if clean_edges.exists():
            edges_path = clean_edges

    stats = {
        "generated_at": datetime.now().isoformat(),
        "label": label,
        "files": {
            "nodes": str(nodes_path),
            "edges": str(edges_path)
        },
        "nodes": analyze_nodes(nodes_path),
        "edges": analyze_edges(edges_path)
    }

    # Save to JSON
    print(f"\nSaving statistics to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print_summary(stats)

    return stats


def print_summary(stats: dict) -> None:
    """Print a human-readable summary."""
    print("\n" + "=" * 60)
    print(f"DATA QUALITY STATISTICS ({stats['label'].upper()})")
    print("=" * 60)

    print(f"\nNODES: {stats['nodes']['total_nodes']:,}")
    print(f"  Unique entity types: {stats['nodes']['type_summary']['total_unique_types']}")
    print(f"  Types: {list(stats['nodes']['type_summary']['type_counts'].keys())}")

    print(f"\nEDGES: {stats['edges']['total_edges']:,}")
    print(f"  Unique relation types: {stats['edges']['relation_summary']['total_unique_relations']}")

    print(f"\nNAME QUALITY:")
    nq = stats['nodes']['name_quality']
    print(f"  Drugs with ID-only names: {nq.get('drugs_with_id_only_names', 0):,} / {nq.get('drugs_total', 0):,}")
    print(f"  Genes with numeric-only names: {nq.get('genes_with_numeric_only_names', 0):,} / {nq.get('genes_total', 0):,}")

    print(f"\nCOVERAGE:")
    cov = stats['nodes']['coverage']
    print(f"  Has name: {cov['pct_has_name']}%")
    print(f"  Has external_source: {cov['pct_has_external_source']}%")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate knowledge graph statistics")
    parser.add_argument("--data-dir", type=str, default="data/processed",
                        help="Directory containing the data files")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path")
    parser.add_argument("--label", type=str, default="before",
                        choices=["before", "after"],
                        help="Label for the statistics (before/after cleansing)")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = data_dir / "validation" / f"{args.label}_stats.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    generate_stats(data_dir, output_path, args.label)
