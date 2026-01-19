#!/usr/bin/env python3
"""
Explainability module for drug repurposing predictions.

This module provides explanations for why a drug might work for a disease,
addressing one of the key limitations of embedding-based methods.

Approaches:
1. Path-based explanations: Find paths in the KG connecting drug to disease
2. Shared mechanism analysis: Identify common biological mechanisms
3. LLM-based natural language explanations
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO")


@dataclass
class PathExplanation:
    """A path-based explanation for a drug-disease prediction."""

    drug_id: str
    disease_id: str
    paths: list[list[tuple[str, str, str]]]  # List of (node, relation, node) sequences
    shared_targets: list[str]
    shared_pathways: list[str]
    confidence: float
    natural_language: str | None = None


class PredictionExplainer:
    """
    Generate explanations for drug-disease predictions.

    This addresses the "black box" problem of embedding-based methods
    by finding interpretable evidence for predictions.
    """

    def __init__(self, graph: nx.MultiDiGraph | None = None):
        self.graph = graph
        self.llm_client = None

    def load_graph(self, graph_path: Path) -> None:
        """Load the knowledge graph."""
        logger.info(f"Loading graph from {graph_path}")
        self.graph = nx.read_gpickle(graph_path)
        logger.info(f"Loaded graph with {self.graph.number_of_nodes():,} nodes")

    def find_paths(
        self,
        drug_id: str,
        disease_id: str,
        max_length: int = 4,
        max_paths: int = 10,
    ) -> list[list[tuple[str, str, str]]]:
        """Find paths connecting a drug to a disease in the knowledge graph."""
        if self.graph is None:
            raise ValueError("Graph not loaded. Call load_graph() first.")

        if drug_id not in self.graph or disease_id not in self.graph:
            return []

        paths = []
        try:
            # Find simple paths
            for path_nodes in nx.all_simple_paths(
                self.graph, drug_id, disease_id, cutoff=max_length
            ):
                if len(paths) >= max_paths:
                    break

                # Convert to (node, relation, node) format
                path_with_relations = []
                for i in range(len(path_nodes) - 1):
                    source = path_nodes[i]
                    target = path_nodes[i + 1]
                    # Get relation (take first if multiple)
                    edge_data = self.graph.get_edge_data(source, target)
                    if edge_data:
                        relation = list(edge_data.values())[0].get(
                            "relation", "related_to"
                        )
                    else:
                        relation = "related_to"
                    path_with_relations.append((source, relation, target))

                paths.append(path_with_relations)

        except nx.NetworkXNoPath:
            pass

        return paths

    def find_shared_targets(self, drug_id: str, disease_id: str) -> list[str]:
        """Find genes/proteins that both the drug and disease are connected to."""
        if self.graph is None:
            return []

        drug_targets = set()
        disease_genes = set()

        # Get drug targets (drug -> gene connections)
        if drug_id in self.graph:
            for neighbor in self.graph.neighbors(drug_id):
                node_type = self.graph.nodes[neighbor].get("type", "")
                if node_type in ["Gene", "Protein"]:
                    drug_targets.add(neighbor)

        # Get disease genes (disease -> gene connections)
        if disease_id in self.graph:
            for neighbor in self.graph.neighbors(disease_id):
                node_type = self.graph.nodes[neighbor].get("type", "")
                if node_type in ["Gene", "Protein"]:
                    disease_genes.add(neighbor)

            # Also check incoming edges
            for predecessor in self.graph.predecessors(disease_id):
                node_type = self.graph.nodes[predecessor].get("type", "")
                if node_type in ["Gene", "Protein"]:
                    disease_genes.add(predecessor)

        return list(drug_targets & disease_genes)

    def find_shared_pathways(self, drug_id: str, disease_id: str) -> list[str]:
        """Find biological pathways connected to both drug and disease."""
        if self.graph is None:
            return []

        drug_pathways = set()
        disease_pathways = set()

        def get_pathways(entity_id: str) -> set[str]:
            pathways = set()
            if entity_id not in self.graph:
                return pathways

            # Direct pathway connections
            for neighbor in self.graph.neighbors(entity_id):
                node_type = self.graph.nodes[neighbor].get("type", "")
                if node_type == "Pathway":
                    pathways.add(neighbor)

            # Pathways via genes (2-hop)
            for neighbor in self.graph.neighbors(entity_id):
                neighbor_type = self.graph.nodes[neighbor].get("type", "")
                if neighbor_type in ["Gene", "Protein"]:
                    for pathway in self.graph.neighbors(neighbor):
                        if self.graph.nodes[pathway].get("type") == "Pathway":
                            pathways.add(pathway)

            return pathways

        drug_pathways = get_pathways(drug_id)
        disease_pathways = get_pathways(disease_id)

        return list(drug_pathways & disease_pathways)

    def explain(
        self,
        drug_id: str,
        disease_id: str,
        use_llm: bool = False,
    ) -> PathExplanation:
        """Generate a comprehensive explanation for a drug-disease prediction."""
        # Find paths
        paths = self.find_paths(drug_id, disease_id)

        # Find shared mechanisms
        shared_targets = self.find_shared_targets(drug_id, disease_id)
        shared_pathways = self.find_shared_pathways(drug_id, disease_id)

        # Calculate confidence based on evidence
        confidence = self._calculate_confidence(paths, shared_targets, shared_pathways)

        # Generate natural language explanation
        natural_language = None
        if use_llm:
            natural_language = self._generate_llm_explanation(
                drug_id, disease_id, paths, shared_targets, shared_pathways
            )
        else:
            natural_language = self._generate_template_explanation(
                drug_id, disease_id, paths, shared_targets, shared_pathways
            )

        return PathExplanation(
            drug_id=drug_id,
            disease_id=disease_id,
            paths=paths,
            shared_targets=shared_targets,
            shared_pathways=shared_pathways,
            confidence=confidence,
            natural_language=natural_language,
        )

    def _calculate_confidence(
        self,
        paths: list[list[tuple[str, str, str]]],
        shared_targets: list[str],
        shared_pathways: list[str],
    ) -> float:
        """Calculate confidence score based on amount of evidence."""
        score = 0.0

        # More paths = higher confidence
        score += min(len(paths) * 0.1, 0.3)

        # Shared targets are strong evidence
        score += min(len(shared_targets) * 0.15, 0.4)

        # Shared pathways add moderate evidence
        score += min(len(shared_pathways) * 0.1, 0.3)

        return min(score, 1.0)

    def _generate_template_explanation(
        self,
        drug_id: str,
        disease_id: str,
        paths: list[list[tuple[str, str, str]]],
        shared_targets: list[str],
        shared_pathways: list[str],
    ) -> str:
        """Generate explanation using templates."""
        if self.graph is None:
            return "No graph loaded for explanation."

        drug_name = self.graph.nodes.get(drug_id, {}).get("name", drug_id)
        disease_name = self.graph.nodes.get(disease_id, {}).get("name", disease_id)

        explanation_parts = []

        if shared_targets:
            target_names = [
                self.graph.nodes.get(t, {}).get("name", t) for t in shared_targets[:3]
            ]
            explanation_parts.append(
                f"{drug_name} targets {', '.join(target_names)}, "
                f"which are also implicated in {disease_name}."
            )

        if shared_pathways:
            pathway_names = [
                self.graph.nodes.get(p, {}).get("name", p) for p in shared_pathways[:3]
            ]
            explanation_parts.append(
                f"Both {drug_name} and {disease_name} are connected to the "
                f"{', '.join(pathway_names)} pathway(s)."
            )

        if paths and not shared_targets and not shared_pathways:
            explanation_parts.append(
                f"Found {len(paths)} path(s) connecting {drug_name} to {disease_name} "
                f"in the knowledge graph."
            )

        if not explanation_parts:
            return f"No direct mechanistic evidence found connecting {drug_name} to {disease_name}."

        return " ".join(explanation_parts)

    def _generate_llm_explanation(
        self,
        drug_id: str,
        disease_id: str,
        paths: list[list[tuple[str, str, str]]],
        shared_targets: list[str],
        shared_pathways: list[str],
    ) -> str:
        """Generate natural language explanation using an LLM."""
        # This would use Claude or another LLM to generate a more sophisticated explanation
        # For now, fall back to template
        try:
            from anthropic import Anthropic

            if self.llm_client is None:
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if api_key:
                    self.llm_client = Anthropic(api_key=api_key)
                else:
                    return self._generate_template_explanation(
                        drug_id, disease_id, paths, shared_targets, shared_pathways
                    )

            if self.graph is None:
                return "No graph loaded."

            drug_name = self.graph.nodes.get(drug_id, {}).get("name", drug_id)
            disease_name = self.graph.nodes.get(disease_id, {}).get("name", disease_id)

            # Prepare evidence summary
            evidence = {
                "drug": drug_name,
                "disease": disease_name,
                "shared_targets": [
                    self.graph.nodes.get(t, {}).get("name", t) for t in shared_targets
                ],
                "shared_pathways": [
                    self.graph.nodes.get(p, {}).get("name", p) for p in shared_pathways
                ],
                "num_paths": len(paths),
            }

            prompt = f"""Based on the following evidence from a biomedical knowledge graph,
explain in 2-3 sentences why {drug_name} might be effective for treating {disease_name}.

Evidence:
- Shared gene/protein targets: {', '.join(evidence['shared_targets']) or 'None found'}
- Shared biological pathways: {', '.join(evidence['shared_pathways']) or 'None found'}
- Number of connecting paths in knowledge graph: {evidence['num_paths']}

Provide a clear, scientific explanation suitable for a researcher. If evidence is limited,
note that this is a hypothesis requiring validation."""

            response = self.llm_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )

            return response.content[0].text

        except Exception as e:
            logger.warning(f"LLM explanation failed: {e}")
            return self._generate_template_explanation(
                drug_id, disease_id, paths, shared_targets, shared_pathways
            )


def explain_prediction(
    drug_id: str,
    disease_id: str,
    graph_path: Path,
    use_llm: bool = False,
) -> PathExplanation:
    """Convenience function to explain a single prediction."""
    explainer = PredictionExplainer()
    explainer.load_graph(graph_path)
    return explainer.explain(drug_id, disease_id, use_llm=use_llm)
