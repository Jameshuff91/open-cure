#!/usr/bin/env python3
"""
Build a unified knowledge graph from multiple biomedical sources.

This script combines DRKG, Hetionet, and PrimeKG into a unified graph format
that can be used for downstream drug repurposing analysis.

The unified graph uses a common schema:
- Nodes: (id, type, name, source, properties)
- Edges: (source_id, relation, target_id, source_kg, properties)
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
from loguru import logger
from tqdm import tqdm

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
GRAPH_DATA_DIR = PROJECT_ROOT / "data" / "graphs"

# Entity type mappings to unified schema
ENTITY_TYPE_MAP = {
    # DRKG types
    "Compound": "Drug",
    "Disease": "Disease",
    "Gene": "Gene",
    "Anatomy": "Anatomy",
    "Atc": "DrugClass",
    "Pharmacologic Class": "DrugClass",
    "Side Effect": "SideEffect",
    "Symptom": "Symptom",
    "Biological Process": "BiologicalProcess",
    "Cellular Component": "CellularComponent",
    "Molecular Function": "MolecularFunction",
    "Pathway": "Pathway",
    # Hetionet types
    "Compound": "Drug",
    "Anatomy": "Anatomy",
    "Gene": "Gene",
    "Disease": "Disease",
    "Symptom": "Symptom",
    "Side Effect": "SideEffect",
    "Biological Process": "BiologicalProcess",
    "Cellular Component": "CellularComponent",
    "Molecular Function": "MolecularFunction",
    "Pathway": "Pathway",
    "Pharmacologic Class": "DrugClass",
}


class UnifiedGraphBuilder:
    """Build a unified knowledge graph from multiple sources."""

    def __init__(self):
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[dict[str, Any]] = []
        self.stats = defaultdict(lambda: defaultdict(int))

    def load_drkg(self) -> bool:
        """Load DRKG (Drug Repurposing Knowledge Graph)."""
        drkg_dir = RAW_DATA_DIR / "drkg"
        drkg_file = drkg_dir / "drkg.tsv"

        if not drkg_file.exists():
            logger.warning(f"DRKG not found at {drkg_file}, skipping...")
            return False

        logger.info("Loading DRKG...")

        # DRKG format: head \t relation \t tail
        # Entity format: EntityType::DatabaseID:EntityID
        df = pd.read_csv(drkg_file, sep="\t", header=None, names=["head", "relation", "tail"])

        logger.info(f"  Loaded {len(df):,} edges from DRKG")

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing DRKG"):
            head = row["head"]
            relation = row["relation"]
            tail = row["tail"]

            # Parse entity format
            head_parts = head.split("::")
            tail_parts = tail.split("::")

            if len(head_parts) >= 2:
                head_type = head_parts[0]
                head_id = f"drkg:{head}"
            else:
                head_type = "Unknown"
                head_id = f"drkg:{head}"

            if len(tail_parts) >= 2:
                tail_type = tail_parts[0]
                tail_id = f"drkg:{tail}"
            else:
                tail_type = "Unknown"
                tail_id = f"drkg:{tail}"

            # Add nodes
            if head_id not in self.nodes:
                unified_type = ENTITY_TYPE_MAP.get(head_type, head_type)
                self.nodes[head_id] = {
                    "id": head_id,
                    "original_id": head,
                    "type": unified_type,
                    "name": head,
                    "source": "drkg",
                }
                self.stats["drkg"]["nodes"] += 1
                self.stats["drkg"][f"node_type_{unified_type}"] += 1

            if tail_id not in self.nodes:
                unified_type = ENTITY_TYPE_MAP.get(tail_type, tail_type)
                self.nodes[tail_id] = {
                    "id": tail_id,
                    "original_id": tail,
                    "type": unified_type,
                    "name": tail,
                    "source": "drkg",
                }
                self.stats["drkg"]["nodes"] += 1
                self.stats["drkg"][f"node_type_{unified_type}"] += 1

            # Add edge
            self.edges.append({
                "source": head_id,
                "relation": relation,
                "target": tail_id,
                "source_kg": "drkg",
            })
            self.stats["drkg"]["edges"] += 1

        logger.success(f"  DRKG: {self.stats['drkg']['nodes']:,} nodes, {self.stats['drkg']['edges']:,} edges")
        return True

    def load_hetionet(self) -> bool:
        """Load Hetionet."""
        hetionet_dir = RAW_DATA_DIR / "hetionet"
        hetionet_file = hetionet_dir / "hetionet-v1.0.json"

        if not hetionet_file.exists():
            logger.warning(f"Hetionet not found at {hetionet_file}, skipping...")
            return False

        logger.info("Loading Hetionet...")

        with open(hetionet_file) as f:
            data = json.load(f)

        # Process nodes
        for node in tqdm(data.get("nodes", []), desc="Processing Hetionet nodes"):
            node_id = f"hetionet:{node['identifier']}"
            node_type = node.get("kind", "Unknown")
            unified_type = ENTITY_TYPE_MAP.get(node_type, node_type)

            if node_id not in self.nodes:
                self.nodes[node_id] = {
                    "id": node_id,
                    "original_id": str(node["identifier"]),
                    "type": unified_type,
                    "name": node.get("name", str(node["identifier"])),
                    "source": "hetionet",
                    "properties": node.get("data", {}),
                }
                self.stats["hetionet"]["nodes"] += 1
                self.stats["hetionet"][f"node_type_{unified_type}"] += 1

        # Process edges
        # Hetionet edge format: source_id: [type, id], target_id: [type, id]
        for edge in tqdm(data.get("edges", []), desc="Processing Hetionet edges"):
            # Handle Hetionet's specific format
            source_info = edge.get("source_id", edge.get("source"))
            target_info = edge.get("target_id", edge.get("target"))

            # source_info is [type, identifier] in Hetionet
            if isinstance(source_info, list):
                source_id = f"hetionet:{source_info[1]}"
            else:
                source_id = f"hetionet:{source_info}"

            if isinstance(target_info, list):
                target_id = f"hetionet:{target_info[1]}"
            else:
                target_id = f"hetionet:{target_info}"

            relation = edge.get("kind", "related_to")

            self.edges.append({
                "source": source_id,
                "relation": relation,
                "target": target_id,
                "source_kg": "hetionet",
                "properties": edge.get("data", {}),
            })
            self.stats["hetionet"]["edges"] += 1

        logger.success(f"  Hetionet: {self.stats['hetionet']['nodes']:,} nodes, {self.stats['hetionet']['edges']:,} edges")
        return True

    def load_primekg(self) -> bool:
        """Load PrimeKG."""
        primekg_dir = RAW_DATA_DIR / "primekg"
        primekg_file = primekg_dir / "kg.csv"

        if not primekg_file.exists():
            logger.warning(f"PrimeKG not found at {primekg_file}, skipping...")
            return False

        logger.info("Loading PrimeKG...")

        # PrimeKG format: x_id, x_type, x_name, x_source, relation, y_id, y_type, y_name, y_source, display_relation
        df = pd.read_csv(primekg_file)

        logger.info(f"  Loaded {len(df):,} edges from PrimeKG")

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing PrimeKG"):
            # Source node
            source_id = f"primekg:{row['x_type']}:{row['x_id']}"
            source_type = ENTITY_TYPE_MAP.get(row["x_type"], row["x_type"])

            if source_id not in self.nodes:
                self.nodes[source_id] = {
                    "id": source_id,
                    "original_id": str(row["x_id"]),
                    "type": source_type,
                    "name": row.get("x_name", str(row["x_id"])),
                    "source": "primekg",
                    "external_source": row.get("x_source", ""),
                }
                self.stats["primekg"]["nodes"] += 1
                self.stats["primekg"][f"node_type_{source_type}"] += 1

            # Target node
            target_id = f"primekg:{row['y_type']}:{row['y_id']}"
            target_type = ENTITY_TYPE_MAP.get(row["y_type"], row["y_type"])

            if target_id not in self.nodes:
                self.nodes[target_id] = {
                    "id": target_id,
                    "original_id": str(row["y_id"]),
                    "type": target_type,
                    "name": row.get("y_name", str(row["y_id"])),
                    "source": "primekg",
                    "external_source": row.get("y_source", ""),
                }
                self.stats["primekg"]["nodes"] += 1
                self.stats["primekg"][f"node_type_{target_type}"] += 1

            # Edge
            self.edges.append({
                "source": source_id,
                "relation": row["relation"],
                "target": target_id,
                "source_kg": "primekg",
                "display_relation": row.get("display_relation", row["relation"]),
            })
            self.stats["primekg"]["edges"] += 1

        logger.success(f"  PrimeKG: {self.stats['primekg']['nodes']:,} nodes, {self.stats['primekg']['edges']:,} edges")
        return True

    def build_networkx_graph(self) -> nx.MultiDiGraph:
        """Convert to NetworkX graph."""
        logger.info("Building NetworkX graph...")

        G = nx.MultiDiGraph()

        # Add nodes
        for node_id, node_data in tqdm(self.nodes.items(), desc="Adding nodes"):
            G.add_node(node_id, **node_data)

        # Add edges
        for edge in tqdm(self.edges, desc="Adding edges"):
            G.add_edge(
                edge["source"],
                edge["target"],
                relation=edge["relation"],
                source_kg=edge["source_kg"],
            )

        logger.success(f"NetworkX graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
        return G

    def save(self, output_dir: Path | None = None):
        """Save the unified graph in multiple formats."""
        if output_dir is None:
            output_dir = PROCESSED_DATA_DIR

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save nodes as CSV
        logger.info("Saving nodes...")
        nodes_df = pd.DataFrame(self.nodes.values())
        nodes_df.to_csv(output_dir / "unified_nodes.csv", index=False)

        # Save edges as CSV
        logger.info("Saving edges...")
        edges_df = pd.DataFrame(self.edges)
        edges_df.to_csv(output_dir / "unified_edges.csv", index=False)

        # Save stats
        logger.info("Saving statistics...")
        with open(output_dir / "unified_stats.json", "w") as f:
            json.dump(dict(self.stats), f, indent=2, default=str)

        # Save NetworkX graph
        logger.info("Saving NetworkX graph...")
        GRAPH_DATA_DIR.mkdir(parents=True, exist_ok=True)
        G = self.build_networkx_graph()
        # Use pickle directly since nx.write_gpickle was removed in NetworkX 3.0
        import pickle
        with open(GRAPH_DATA_DIR / "unified_graph.gpickle", "wb") as f:
            pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)

        logger.success(f"Unified graph saved to {output_dir}")

    def print_summary(self):
        """Print summary statistics."""
        total_nodes = len(self.nodes)
        total_edges = len(self.edges)

        logger.info("\n" + "=" * 60)
        logger.info("UNIFIED GRAPH SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total nodes: {total_nodes:,}")
        logger.info(f"Total edges: {total_edges:,}")

        # Node types
        node_types = defaultdict(int)
        for node in self.nodes.values():
            node_types[node["type"]] += 1

        logger.info("\nNode types:")
        for node_type, count in sorted(node_types.items(), key=lambda x: -x[1]):
            logger.info(f"  {node_type}: {count:,}")

        # Edge relations
        relation_counts = defaultdict(int)
        for edge in self.edges:
            relation_counts[edge["relation"]] += 1

        logger.info(f"\nRelation types: {len(relation_counts):,}")
        logger.info("Top 10 relations:")
        for relation, count in sorted(relation_counts.items(), key=lambda x: -x[1])[:10]:
            logger.info(f"  {relation}: {count:,}")

        # Source breakdown
        logger.info("\nBy source:")
        for source, stats in self.stats.items():
            logger.info(f"  {source}: {stats.get('nodes', 0):,} nodes, {stats.get('edges', 0):,} edges")


def main():
    """Build the unified knowledge graph."""
    logger.info("Building Unified Knowledge Graph for Drug Repurposing")
    logger.info("=" * 60)

    builder = UnifiedGraphBuilder()

    # Load all available knowledge graphs
    builder.load_drkg()
    builder.load_hetionet()
    builder.load_primekg()

    # Print summary
    builder.print_summary()

    # Save
    builder.save()

    logger.info("\nNext steps:")
    logger.info("  1. Explore: jupyter lab notebooks/01_explore_graphs.ipynb")
    logger.info("  2. Train models: python src/models/train_link_prediction.py")


if __name__ == "__main__":
    main()
