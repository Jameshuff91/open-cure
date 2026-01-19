#!/usr/bin/env python3
"""
Open Cure CLI - Command-line interface for drug repurposing predictions.

Usage:
    open-cure predict <disease> [--top-k=<n>] [--method=<m>]
    open-cure search <drug> [--disease=<d>]
    open-cure explain <drug> <disease>
    open-cure enrich <disease> [--max-papers=<n>]
    open-cure train [--epochs=<n>] [--models=<m>]
    open-cure status

Examples:
    open-cure predict "Castleman disease" --top-k=20
    open-cure search "sirolimus"
    open-cure explain "sirolimus" "Castleman disease"
    open-cure enrich "Castleman disease" --max-papers=50
"""

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
GRAPH_DIR = DATA_DIR / "graphs"
MODELS_DIR = PROJECT_ROOT / "models"


def cmd_predict(args: argparse.Namespace):
    """Find drug repurposing candidates for a disease."""
    disease = args.disease
    top_k = args.top_k
    method = args.method

    logger.info(f"Finding drug candidates for: {disease}")

    # Check if graph exists
    graph_path = GRAPH_DIR / "unified_graph.gpickle"
    if not graph_path.exists():
        logger.error("Unified graph not found. Run: python src/ingest/build_unified_graph.py")
        return 1

    # Try to find disease in graph
    try:
        from src.models.rare_disease import RareDiseaseRepurposer

        repurposer = RareDiseaseRepurposer()
        repurposer.load_graph(graph_path)

        # Find disease ID
        disease_id = _find_disease_id(disease)
        if not disease_id:
            logger.warning(f"Disease '{disease}' not found in graph. Searching by name...")
            # Search in processed nodes
            disease_id = _search_disease(disease)

        if not disease_id:
            logger.error(f"Could not find disease: {disease}")
            return 1

        logger.info(f"Found disease: {disease_id}")

        # Get candidates
        methods = [method] if method != "all" else None
        candidates = repurposer.find_candidates(disease_id, methods=methods, top_k=top_k)

        # Display results
        print(f"\n{'='*60}")
        print(f"TOP {top_k} DRUG REPURPOSING CANDIDATES FOR: {disease}")
        print(f"{'='*60}\n")

        for i, cand in enumerate(candidates, 1):
            print(f"{i:3}. {cand.drug_name}")
            print(f"     Score: {cand.score:.4f} | Method: {cand.method}")
            print(f"     {cand.explanation[:80]}...")
            print()

        return 0

    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.info("Install dependencies: pip install -r requirements.txt")
        return 1
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


def cmd_search(args: argparse.Namespace):
    """Search for a drug and its associations."""
    drug = args.drug
    disease_filter = args.disease

    logger.info(f"Searching for drug: {drug}")

    try:
        import pandas as pd

        nodes_path = PROCESSED_DIR / "unified_nodes.csv"
        edges_path = PROCESSED_DIR / "unified_edges.csv"

        if not nodes_path.exists():
            logger.error("Unified graph not built. Run: python src/ingest/build_unified_graph.py")
            return 1

        nodes = pd.read_csv(nodes_path)
        edges = pd.read_csv(edges_path)

        # Find drug
        drug_lower = drug.lower()
        drug_matches = nodes[
            (nodes["type"] == "Drug") &
            (nodes["name"].str.lower().str.contains(drug_lower, na=False))
        ]

        if drug_matches.empty:
            logger.warning(f"No drugs found matching: {drug}")
            return 1

        print(f"\n{'='*60}")
        print(f"SEARCH RESULTS FOR: {drug}")
        print(f"{'='*60}\n")

        for _, drug_row in drug_matches.iterrows():
            drug_id = drug_row["id"]
            drug_name = drug_row["name"]

            print(f"Drug: {drug_name}")
            print(f"  ID: {drug_id}")
            print(f"  Source: {drug_row.get('source', 'unknown')}")

            # Find associations
            drug_edges = edges[
                (edges["source"] == drug_id) | (edges["target"] == drug_id)
            ]

            # Group by relation type
            print(f"\n  Associations ({len(drug_edges)} total):")

            if disease_filter:
                # Filter to specific disease
                disease_lower = disease_filter.lower()
                for _, edge in drug_edges.iterrows():
                    other_id = edge["target"] if edge["source"] == drug_id else edge["source"]
                    other_node = nodes[nodes["id"] == other_id]
                    if not other_node.empty:
                        other_name = other_node.iloc[0]["name"]
                        if disease_lower in str(other_name).lower():
                            print(f"    - {edge['relation']} -> {other_name}")
            else:
                # Show summary by type
                for rel_type in drug_edges["relation"].unique()[:10]:
                    count = len(drug_edges[drug_edges["relation"] == rel_type])
                    print(f"    - {rel_type}: {count} associations")

            print()

        return 0

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


def cmd_explain(args: argparse.Namespace):
    """Explain why a drug might work for a disease."""
    drug = args.drug
    disease = args.disease

    logger.info(f"Explaining: {drug} -> {disease}")

    try:
        from src.models.explainer import PredictionExplainer

        graph_path = GRAPH_DIR / "unified_graph.gpickle"
        if not graph_path.exists():
            logger.error("Unified graph not found")
            return 1

        explainer = PredictionExplainer()
        explainer.load_graph(graph_path)

        # Find IDs
        drug_id = _search_entity(drug, "Drug")
        disease_id = _search_entity(disease, "Disease")

        if not drug_id or not disease_id:
            logger.error("Could not find drug or disease in graph")
            return 1

        # Generate explanation
        explanation = explainer.explain(drug_id, disease_id, use_llm=args.use_llm)

        print(f"\n{'='*60}")
        print(f"EXPLANATION: {drug} -> {disease}")
        print(f"{'='*60}\n")

        print(f"Confidence: {explanation.confidence:.2f}")
        print(f"\nPaths found: {len(explanation.paths)}")
        print(f"Shared gene targets: {len(explanation.shared_targets)}")
        print(f"Shared pathways: {len(explanation.shared_pathways)}")

        print(f"\n{explanation.natural_language}")

        if explanation.paths:
            print("\nTop paths:")
            for i, path in enumerate(explanation.paths[:3], 1):
                path_str = " -> ".join([f"{src}--[{rel}]-->{tgt}" for src, rel, tgt in path])
                print(f"  {i}. {path_str}")

        return 0

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


def cmd_enrich(args: argparse.Namespace):
    """Enrich knowledge graph with literature about a disease."""
    disease = args.disease
    max_papers = args.max_papers

    logger.info(f"Enriching knowledge for: {disease}")

    try:
        from src.models.llm_extractor import LiteratureKnowledgeEnricher

        enricher = LiteratureKnowledgeEnricher()

        # Load entity mappings if available
        nodes_path = PROCESSED_DIR / "unified_nodes.csv"
        if nodes_path.exists():
            enricher.load_entity_mappings(nodes_path)

        # Enrich
        relationships = enricher.enrich_for_disease(disease, max_papers=max_papers)

        print(f"\n{'='*60}")
        print(f"EXTRACTED RELATIONSHIPS FOR: {disease}")
        print(f"{'='*60}\n")

        print(f"Found {len(relationships)} relationships from literature\n")

        for i, rel in enumerate(relationships[:20], 1):
            resolved = "✓" if rel.drug_id else "?"
            print(f"{i:3}. [{resolved}] {rel.drug} --[{rel.relationship_type}]--> {rel.disease}")
            print(f"     Confidence: {rel.confidence:.2f}")
            print(f"     Evidence: {rel.evidence[:100]}...")
            print(f"     Source: {rel.source_paper}")
            print()

        return 0

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


def cmd_train(args: argparse.Namespace):
    """Train embedding and GNN models."""
    epochs = args.epochs
    models = args.models.split(",")

    logger.info(f"Training models: {models} for {epochs} epochs")

    try:
        # Check data exists
        if not (PROCESSED_DIR / "unified_nodes.csv").exists():
            logger.error("Data not processed. Run: python src/ingest/build_unified_graph.py")
            return 1

        from src.models.link_prediction import DrugDiseasePredictor

        predictor = DrugDiseasePredictor()
        predictor.train(PROCESSED_DIR, epochs=epochs)

        # Save model
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        predictor.save(MODELS_DIR / "predictor")

        logger.success(f"Model saved to {MODELS_DIR / 'predictor'}")
        return 0

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


def cmd_status(args: argparse.Namespace):
    """Check status of data and models."""
    print(f"\n{'='*60}")
    print("OPEN CURE STATUS")
    print(f"{'='*60}\n")

    # Check data
    print("DATA:")
    for kg in ["drkg", "hetionet", "primekg"]:
        kg_dir = DATA_DIR / "raw" / kg
        status = "✓ Downloaded" if kg_dir.exists() and any(kg_dir.iterdir()) else "✗ Not downloaded"
        print(f"  {kg}: {status}")

    # Check processed
    print("\nPROCESSED:")
    unified_nodes = PROCESSED_DIR / "unified_nodes.csv"
    unified_edges = PROCESSED_DIR / "unified_edges.csv"

    if unified_nodes.exists():
        import pandas as pd
        nodes = pd.read_csv(unified_nodes)
        edges = pd.read_csv(unified_edges)
        print(f"  Unified graph: ✓ {len(nodes):,} nodes, {len(edges):,} edges")
    else:
        print("  Unified graph: ✗ Not built")

    # Check graph
    print("\nGRAPH:")
    graph_file = GRAPH_DIR / "unified_graph.gpickle"
    if graph_file.exists():
        print(f"  NetworkX graph: ✓ {graph_file.stat().st_size / 1024 / 1024:.1f} MB")
    else:
        print("  NetworkX graph: ✗ Not built")

    # Check models
    print("\nMODELS:")
    model_dir = MODELS_DIR / "predictor"
    if model_dir.exists():
        print(f"  Trained predictor: ✓")
    else:
        print("  Trained predictor: ✗ Not trained")

    print("\nNEXT STEPS:")
    if not (DATA_DIR / "raw" / "drkg").exists():
        print("  1. python scripts/download_graphs.py --all")
    elif not unified_nodes.exists():
        print("  1. python src/ingest/build_unified_graph.py")
    elif not model_dir.exists():
        print("  1. open-cure train --epochs=50")
    else:
        print("  Ready to make predictions: open-cure predict <disease>")

    return 0


def _find_disease_id(disease_name: str) -> str | None:
    """Find disease ID by name."""
    try:
        import pandas as pd
        nodes = pd.read_csv(PROCESSED_DIR / "unified_nodes.csv")
        disease_lower = disease_name.lower()
        matches = nodes[
            (nodes["type"] == "Disease") &
            (nodes["name"].str.lower().str.contains(disease_lower, na=False))
        ]
        if not matches.empty:
            return matches.iloc[0]["id"]
    except Exception:
        pass
    return None


def _search_disease(disease_name: str) -> str | None:
    """Search for disease by partial name."""
    return _find_disease_id(disease_name)


def _search_entity(name: str, entity_type: str) -> str | None:
    """Search for entity by name and type."""
    try:
        import pandas as pd
        nodes = pd.read_csv(PROCESSED_DIR / "unified_nodes.csv")
        name_lower = name.lower()
        matches = nodes[
            (nodes["type"] == entity_type) &
            (nodes["name"].str.lower().str.contains(name_lower, na=False))
        ]
        if not matches.empty:
            return matches.iloc[0]["id"]
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Open Cure - Drug repurposing using AI and knowledge graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # predict
    predict_parser = subparsers.add_parser("predict", help="Find drug candidates for a disease")
    predict_parser.add_argument("disease", help="Disease name to find treatments for")
    predict_parser.add_argument("--top-k", type=int, default=20, help="Number of candidates")
    predict_parser.add_argument("--method", default="all", choices=["all", "similarity", "multi_hop"], help="Prediction method")

    # search
    search_parser = subparsers.add_parser("search", help="Search for a drug")
    search_parser.add_argument("drug", help="Drug name to search")
    search_parser.add_argument("--disease", help="Filter by disease")

    # explain
    explain_parser = subparsers.add_parser("explain", help="Explain drug-disease relationship")
    explain_parser.add_argument("drug", help="Drug name")
    explain_parser.add_argument("disease", help="Disease name")
    explain_parser.add_argument("--use-llm", action="store_true", help="Use LLM for explanation")

    # enrich
    enrich_parser = subparsers.add_parser("enrich", help="Enrich with literature")
    enrich_parser.add_argument("disease", help="Disease to enrich")
    enrich_parser.add_argument("--max-papers", type=int, default=50, help="Max papers to process")

    # train
    train_parser = subparsers.add_parser("train", help="Train models")
    train_parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    train_parser.add_argument("--models", default="transE,gnn", help="Models to train")

    # status
    subparsers.add_parser("status", help="Check system status")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    commands = {
        "predict": cmd_predict,
        "search": cmd_search,
        "explain": cmd_explain,
        "enrich": cmd_enrich,
        "train": cmd_train,
        "status": cmd_status,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
