#!/usr/bin/env python3
"""
Train Node2Vec embeddings on DRKG without treatment edges.

PURPOSE:
    Train Node2Vec on the filtered DRKG (no treatment edges) to enable
    fair comparison with TxGNN. The original embeddings may have encoded
    "similar diseases share treatments" information, which is circular
    for drug repurposing evaluation.

PARAMETERS:
    - dim: 256 (match original)
    - walk_length: 80
    - num_walks: 10
    - p: 1.0, q: 1.0 (balanced, no bias toward BFS/DFS)
    - window: 10

OUTPUT:
    data/embeddings/node2vec_256_no_treatment.csv

EXPECTED TIME: 30-60 minutes on M4 Pro
"""

import sys
import time
from pathlib import Path
from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd
from pecanpy.pecanpy import SparseOTF
from gensim.models import Word2Vec

PROJECT_ROOT = Path(__file__).parent.parent
DRKG_NO_TREATMENT = PROJECT_ROOT / "data" / "raw" / "drkg" / "drkg_no_treatment.tsv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "embeddings" / "node2vec_256_no_treatment.csv"
EDGE_LIST_PATH = PROJECT_ROOT / "data" / "raw" / "drkg" / "drkg_no_treatment.edgelist"

# Node2Vec parameters
DIMENSIONS = 256
WALK_LENGTH = 80
NUM_WALKS = 10
P = 1.0
Q = 1.0
WINDOW = 10
MIN_COUNT = 1
WORKERS = 8


def create_edge_list() -> Tuple[str, Dict[str, int], Dict[int, str]]:
    """
    Convert DRKG TSV to PecanPy edge list format.

    PecanPy expects: node1 node2 [weight] (space-separated)
    DRKG format: entity1 \t relation \t entity2

    We treat the graph as undirected and unweighted for Node2Vec.
    """
    print("Creating edge list for PecanPy...")

    node_to_id: Dict[str, int] = {}
    edges: Set[Tuple[int, int]] = set()

    with open(DRKG_NO_TREATMENT, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue

            h, _, t = parts[0], parts[1], parts[2]

            if h not in node_to_id:
                node_to_id[h] = len(node_to_id)
            if t not in node_to_id:
                node_to_id[t] = len(node_to_id)

            h_id, t_id = node_to_id[h], node_to_id[t]

            # Store edges in canonical order (smaller id first) to deduplicate
            if h_id < t_id:
                edges.add((h_id, t_id))
            else:
                edges.add((t_id, h_id))

    id_to_node = {v: k for k, v in node_to_id.items()}

    print(f"  Unique nodes: {len(node_to_id):,}")
    print(f"  Unique edges: {len(edges):,}")

    # Write edge list
    print(f"  Writing to: {EDGE_LIST_PATH}")
    with open(EDGE_LIST_PATH, "w") as f:
        for h_id, t_id in edges:
            f.write(f"{h_id} {t_id}\n")

    return str(EDGE_LIST_PATH), node_to_id, id_to_node


def train_node2vec(
    edge_list_path: str,
    id_to_node: Dict[int, str],
    num_nodes: int,
) -> Dict[str, np.ndarray]:
    """
    Train Node2Vec embeddings using PecanPy + Gensim Word2Vec.
    """
    print("\nInitializing PecanPy graph...")

    # Use SparseOTF - more memory efficient for large sparse graphs
    # OTF = On-The-Fly (computes transition probs lazily)
    g = SparseOTF(p=P, q=Q, workers=WORKERS, verbose=True)
    g.read_edg(edge_list_path, weighted=False, directed=False, delimiter=" ")

    print(f"\nGenerating random walks (num_walks={NUM_WALKS}, walk_length={WALK_LENGTH})...")
    start_time = time.time()
    walks = g.simulate_walks(num_walks=NUM_WALKS, walk_length=WALK_LENGTH)
    walk_time = time.time() - start_time
    print(f"  Generated {len(walks):,} walks in {walk_time:.1f}s")

    # Convert walks from node IDs to strings for Word2Vec
    print("\nConverting walks to string format...")
    walks_str = [[str(node) for node in walk] for walk in walks]

    print(f"\nTraining Word2Vec (dim={DIMENSIONS}, window={WINDOW})...")
    start_time = time.time()
    model = Word2Vec(
        sentences=walks_str,
        vector_size=DIMENSIONS,
        window=WINDOW,
        min_count=MIN_COUNT,
        sg=1,  # Skip-gram (standard for Node2Vec)
        workers=WORKERS,
        epochs=1,  # PecanPy already generates multiple walks
    )
    train_time = time.time() - start_time
    print(f"  Trained in {train_time:.1f}s")

    # Extract embeddings
    print("\nExtracting embeddings...")
    embeddings: Dict[str, np.ndarray] = {}
    missing = 0
    for node_id in range(num_nodes):
        node_id_str = str(node_id)
        if node_id_str in model.wv:
            node_name = id_to_node[node_id]
            embeddings[node_name] = model.wv[node_id_str]
        else:
            missing += 1

    print(f"  Extracted: {len(embeddings):,} embeddings")
    if missing > 0:
        print(f"  Missing (disconnected?): {missing:,}")

    return embeddings


def save_embeddings(embeddings: Dict[str, np.ndarray], output_path: Path) -> None:
    """Save embeddings in the same format as node2vec_256_named.csv."""
    print(f"\nSaving embeddings to: {output_path}")

    # Create DataFrame
    dim_cols = [f"dim_{i}" for i in range(DIMENSIONS)]
    rows = []
    for entity, vec in embeddings.items():
        row = {"entity": entity}
        row.update({f"dim_{i}": float(vec[i]) for i in range(DIMENSIONS)})
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df[["entity"] + dim_cols]
    df.to_csv(output_path, index=False)

    print(f"  Saved {len(df):,} embeddings")


def main() -> None:
    overall_start = time.time()

    print("=" * 70)
    print("TRAIN NODE2VEC ON DRKG (NO TREATMENT EDGES)")
    print("=" * 70)
    print()
    print(f"Input:  {DRKG_NO_TREATMENT}")
    print(f"Output: {OUTPUT_PATH}")
    print()
    print("Parameters:")
    print(f"  dim={DIMENSIONS}, walk_length={WALK_LENGTH}, num_walks={NUM_WALKS}")
    print(f"  p={P}, q={Q}, window={WINDOW}")
    print()

    if not DRKG_NO_TREATMENT.exists():
        print("ERROR: Filtered DRKG not found. Run filter_drkg_treatment_edges.py first.")
        sys.exit(1)

    # Step 1: Create edge list
    edge_list_path, node_to_id, id_to_node = create_edge_list()

    # Step 2: Train Node2Vec
    embeddings = train_node2vec(edge_list_path, id_to_node, len(node_to_id))

    # Step 3: Save embeddings
    save_embeddings(embeddings, OUTPUT_PATH)

    # Summary
    elapsed = time.time() - overall_start
    print()
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
