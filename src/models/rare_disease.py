#!/usr/bin/env python3
"""
Specialized Module for Rare Disease Drug Repurposing.

Rare diseases present unique challenges for drug repurposing:
1. Data sparsity - few known treatments or even disease-gene associations
2. Limited literature - few papers to learn from
3. Heterogeneous phenotypes - same disease can present differently

This module implements specialized techniques:
1. Few-shot learning for diseases with minimal training data
2. Transfer learning from similar diseases
3. Phenotype-based matching across rare diseases
4. Multi-hop reasoning to find indirect drug-disease connections
"""

import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")


@dataclass
class RareDiseaseProfile:
    """Profile of a rare disease for matching."""

    disease_id: str
    name: str
    phenotypes: list[str]  # HPO terms
    genes: list[str]
    pathways: list[str]
    known_treatments: list[str]
    similar_diseases: list[tuple[str, float]]  # (disease_id, similarity)


@dataclass
class RepurposingCandidate:
    """A drug repurposing candidate for a rare disease."""

    drug_id: str
    drug_name: str
    disease_id: str
    disease_name: str
    score: float
    method: str  # which method found this candidate
    evidence: dict[str, Any]
    explanation: str


class DiseaseSimilarityCalculator:
    """
    Calculate similarity between diseases based on shared features.

    Uses multiple signals:
    - Shared phenotypes (HPO terms)
    - Shared genes
    - Shared pathways
    - Shared drug targets
    """

    def __init__(self, graph: nx.MultiDiGraph | None = None):
        self.graph = graph
        self.disease_profiles: dict[str, RareDiseaseProfile] = {}

    def load_graph(self, graph_path: Path):
        """Load the knowledge graph."""
        self.graph = nx.read_gpickle(graph_path)

    def build_disease_profile(self, disease_id: str) -> RareDiseaseProfile | None:
        """Build a comprehensive profile for a disease."""
        if self.graph is None or disease_id not in self.graph:
            return None

        disease_node = self.graph.nodes[disease_id]
        name = disease_node.get("name", disease_id)

        # Gather connected entities
        phenotypes = []
        genes = []
        pathways = []
        treatments = []

        for neighbor in self.graph.neighbors(disease_id):
            neighbor_type = self.graph.nodes[neighbor].get("type", "")
            if neighbor_type == "Phenotype":
                phenotypes.append(neighbor)
            elif neighbor_type in ["Gene", "Protein"]:
                genes.append(neighbor)
            elif neighbor_type == "Pathway":
                pathways.append(neighbor)
            elif neighbor_type == "Drug":
                # Check if it's a treatment edge
                edge_data = self.graph.get_edge_data(disease_id, neighbor)
                if edge_data:
                    for edge in edge_data.values():
                        rel = edge.get("relation", "")
                        if "treat" in rel.lower():
                            treatments.append(neighbor)

        # Also check incoming edges
        for predecessor in self.graph.predecessors(disease_id):
            pred_type = self.graph.nodes[predecessor].get("type", "")
            if pred_type in ["Gene", "Protein"]:
                genes.append(predecessor)
            elif pred_type == "Drug":
                edge_data = self.graph.get_edge_data(predecessor, disease_id)
                if edge_data:
                    for edge in edge_data.values():
                        rel = edge.get("relation", "")
                        if "treat" in rel.lower():
                            treatments.append(predecessor)

        profile = RareDiseaseProfile(
            disease_id=disease_id,
            name=name,
            phenotypes=list(set(phenotypes)),
            genes=list(set(genes)),
            pathways=list(set(pathways)),
            known_treatments=list(set(treatments)),
            similar_diseases=[],
        )

        self.disease_profiles[disease_id] = profile
        return profile

    def calculate_similarity(
        self,
        disease1_id: str,
        disease2_id: str,
        weights: dict[str, float] | None = None,
    ) -> float:
        """
        Calculate similarity between two diseases.

        Uses weighted Jaccard similarity across feature types.
        """
        if weights is None:
            weights = {
                "phenotypes": 0.3,
                "genes": 0.4,
                "pathways": 0.2,
                "treatments": 0.1,
            }

        # Get or build profiles
        if disease1_id not in self.disease_profiles:
            self.build_disease_profile(disease1_id)
        if disease2_id not in self.disease_profiles:
            self.build_disease_profile(disease2_id)

        profile1 = self.disease_profiles.get(disease1_id)
        profile2 = self.disease_profiles.get(disease2_id)

        if profile1 is None or profile2 is None:
            return 0.0

        def jaccard(set1: list[str], set2: list[str]) -> float:
            s1, s2 = set(set1), set(set2)
            if not s1 and not s2:
                return 0.0
            intersection = len(s1 & s2)
            union = len(s1 | s2)
            return intersection / union if union > 0 else 0.0

        similarity = (
            weights["phenotypes"] * jaccard(profile1.phenotypes, profile2.phenotypes)
            + weights["genes"] * jaccard(profile1.genes, profile2.genes)
            + weights["pathways"] * jaccard(profile1.pathways, profile2.pathways)
            + weights["treatments"]
            * jaccard(profile1.known_treatments, profile2.known_treatments)
        )

        return similarity

    def find_similar_diseases(
        self,
        disease_id: str,
        top_k: int = 10,
        min_similarity: float = 0.1,
    ) -> list[tuple[str, float]]:
        """Find diseases most similar to the given disease."""
        if self.graph is None:
            return []

        # Build profile for target disease
        target_profile = self.build_disease_profile(disease_id)
        if target_profile is None:
            return []

        # Find all diseases in graph
        all_diseases = [
            node_id
            for node_id, data in self.graph.nodes(data=True)
            if data.get("type") == "Disease" and node_id != disease_id
        ]

        # Calculate similarities
        similarities = []
        for other_disease in all_diseases:
            sim = self.calculate_similarity(disease_id, other_disease)
            if sim >= min_similarity:
                similarities.append((other_disease, sim))

        # Sort and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


class FewShotDrugPredictor(nn.Module):
    """
    Few-shot learning model for rare diseases with minimal training data.

    Uses prototypical networks: learns to compare disease embeddings
    to prototypes of diseases with known treatments.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dim: int = 128,
    ):
        super().__init__()

        # Encoder for disease features
        self.disease_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Encoder for drug features
        self.drug_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Metric network for comparing embeddings
        self.metric_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode_disease(self, disease_features: torch.Tensor) -> torch.Tensor:
        """Encode disease into embedding space."""
        return F.normalize(self.disease_encoder(disease_features), dim=-1)

    def encode_drug(self, drug_features: torch.Tensor) -> torch.Tensor:
        """Encode drug into embedding space."""
        return F.normalize(self.drug_encoder(drug_features), dim=-1)

    def forward(
        self,
        disease_features: torch.Tensor,
        drug_features: torch.Tensor,
    ) -> torch.Tensor:
        """Predict treatment probability."""
        disease_emb = self.encode_disease(disease_features)
        drug_emb = self.encode_drug(drug_features)

        combined = torch.cat([disease_emb, drug_emb], dim=-1)
        return torch.sigmoid(self.metric_network(combined))

    def transfer_from_similar(
        self,
        target_disease: torch.Tensor,
        similar_diseases: list[torch.Tensor],
        similar_treatments: list[list[torch.Tensor]],
        candidate_drug: torch.Tensor,
    ) -> float:
        """
        Transfer knowledge from similar diseases.

        Uses attention-weighted combination of predictions from similar diseases.
        """
        self.eval()
        with torch.no_grad():
            target_emb = self.encode_disease(target_disease)
            drug_emb = self.encode_drug(candidate_drug)

            # Calculate attention weights based on disease similarity
            attention_scores = []
            treatment_signals = []

            for i, similar_disease in enumerate(similar_diseases):
                similar_emb = self.encode_disease(similar_disease)

                # Cosine similarity as attention
                sim = F.cosine_similarity(target_emb, similar_emb, dim=-1)
                attention_scores.append(sim)

                # Check if drug treats similar disease
                treats_similar = 0.0
                for treatment in similar_treatments[i]:
                    treatment_emb = self.encode_drug(treatment)
                    if F.cosine_similarity(drug_emb, treatment_emb, dim=-1) > 0.8:
                        treats_similar = 1.0
                        break

                treatment_signals.append(treats_similar)

            # Softmax attention
            attention = F.softmax(torch.stack(attention_scores), dim=0)

            # Weighted treatment signal
            treatment_signals = torch.tensor(treatment_signals)
            score = (attention * treatment_signals).sum().item()

            return score


class MultiHopReasoner:
    """
    Multi-hop reasoning for finding indirect drug-disease connections.

    For rare diseases, direct drug-disease edges are often missing.
    This module finds paths like:
    - Drug -> Gene -> Disease
    - Drug -> Pathway -> Gene -> Disease
    - Drug -> treats SimilarDisease -> shares_pathway -> Disease
    """

    def __init__(self, graph: nx.MultiDiGraph | None = None):
        self.graph = graph

    def load_graph(self, graph_path: Path):
        """Load the knowledge graph."""
        self.graph = nx.read_gpickle(graph_path)

    def find_drug_disease_paths(
        self,
        drug_id: str,
        disease_id: str,
        max_hops: int = 4,
        max_paths: int = 20,
    ) -> list[list[tuple[str, str, str]]]:
        """
        Find all paths from drug to disease within max_hops.

        Returns list of paths, where each path is a list of (node, relation, node).
        """
        if self.graph is None:
            return []

        if drug_id not in self.graph or disease_id not in self.graph:
            return []

        paths = []
        try:
            for path_nodes in nx.all_simple_paths(
                self.graph, drug_id, disease_id, cutoff=max_hops
            ):
                if len(paths) >= max_paths:
                    break

                # Convert to (node, relation, node) format
                path_with_relations = []
                for i in range(len(path_nodes) - 1):
                    source = path_nodes[i]
                    target = path_nodes[i + 1]
                    edge_data = self.graph.get_edge_data(source, target)
                    if edge_data:
                        relation = list(edge_data.values())[0].get("relation", "connected")
                    else:
                        relation = "connected"
                    path_with_relations.append((source, relation, target))

                paths.append(path_with_relations)

        except nx.NetworkXNoPath:
            pass

        return paths

    def score_path(self, path: list[tuple[str, str, str]]) -> float:
        """
        Score a drug-disease path based on biological plausibility.

        Factors:
        - Path length (shorter is better)
        - Relation types (treatment-related relations score higher)
        - Intermediate node types (genes/pathways are meaningful)
        """
        if not path:
            return 0.0

        # Length penalty
        length_score = 1.0 / len(path)

        # Relation type scores
        relation_weights = {
            "treats": 1.0,
            "may_treat": 0.8,
            "targets": 0.7,
            "inhibits": 0.6,
            "activates": 0.6,
            "associated_with": 0.4,
            "interacts_with": 0.3,
        }

        relation_score = 0.0
        for _, relation, _ in path:
            rel_lower = relation.lower()
            for key, weight in relation_weights.items():
                if key in rel_lower:
                    relation_score += weight
                    break
            else:
                relation_score += 0.2  # Default for unknown relations

        relation_score /= len(path)

        # Node type bonus (paths through genes/pathways are more meaningful)
        node_type_bonus = 0.0
        if self.graph:
            for source, _, target in path:
                for node in [source, target]:
                    node_type = self.graph.nodes.get(node, {}).get("type", "")
                    if node_type in ["Gene", "Protein"]:
                        node_type_bonus += 0.1
                    elif node_type == "Pathway":
                        node_type_bonus += 0.15

        return length_score * 0.3 + relation_score * 0.5 + node_type_bonus * 0.2

    def find_repurposing_candidates(
        self,
        disease_id: str,
        max_candidates: int = 50,
    ) -> list[RepurposingCandidate]:
        """
        Find drug repurposing candidates for a disease using multi-hop reasoning.
        """
        if self.graph is None:
            return []

        # Find all drugs in the graph
        all_drugs = [
            node_id
            for node_id, data in self.graph.nodes(data=True)
            if data.get("type") == "Drug"
        ]

        candidates = []
        for drug_id in all_drugs:
            # Find paths from drug to disease
            paths = self.find_drug_disease_paths(drug_id, disease_id)

            if not paths:
                continue

            # Score best path
            best_path = max(paths, key=self.score_path)
            best_score = self.score_path(best_path)

            drug_name = self.graph.nodes[drug_id].get("name", drug_id)
            disease_name = self.graph.nodes[disease_id].get("name", disease_id)

            # Generate explanation
            path_description = " -> ".join(
                [f"{src} --[{rel}]--> {tgt}" for src, rel, tgt in best_path]
            )

            candidate = RepurposingCandidate(
                drug_id=drug_id,
                drug_name=drug_name,
                disease_id=disease_id,
                disease_name=disease_name,
                score=best_score,
                method="multi_hop_reasoning",
                evidence={
                    "paths": paths,
                    "best_path": best_path,
                    "num_paths": len(paths),
                },
                explanation=f"Found {len(paths)} path(s) connecting {drug_name} to {disease_name}. Best path: {path_description}",
            )
            candidates.append(candidate)

        # Sort by score
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[:max_candidates]


class RareDiseaseRepurposer:
    """
    Main interface for rare disease drug repurposing.

    Combines multiple approaches:
    1. Disease similarity matching
    2. Few-shot transfer learning
    3. Multi-hop reasoning
    4. LLM-based literature extraction
    """

    def __init__(self):
        self.similarity_calculator = DiseaseSimilarityCalculator()
        self.multi_hop_reasoner = MultiHopReasoner()
        self.few_shot_model: FewShotDrugPredictor | None = None

    def load_graph(self, graph_path: Path):
        """Load the knowledge graph for all components."""
        self.similarity_calculator.load_graph(graph_path)
        self.multi_hop_reasoner.load_graph(graph_path)

    def find_candidates(
        self,
        disease_id: str,
        methods: list[str] | None = None,
        top_k: int = 50,
    ) -> list[RepurposingCandidate]:
        """
        Find repurposing candidates using multiple methods.

        Args:
            disease_id: The disease to find treatments for
            methods: Which methods to use (default: all)
            top_k: Number of candidates to return

        Returns:
            Ranked list of repurposing candidates
        """
        if methods is None:
            methods = ["similarity", "multi_hop"]

        all_candidates: dict[str, RepurposingCandidate] = {}

        if "similarity" in methods:
            logger.info("Running similarity-based search...")
            sim_candidates = self._find_by_similarity(disease_id)
            for c in sim_candidates:
                key = f"{c.drug_id}_{c.disease_id}"
                if key not in all_candidates or c.score > all_candidates[key].score:
                    all_candidates[key] = c

        if "multi_hop" in methods:
            logger.info("Running multi-hop reasoning...")
            hop_candidates = self.multi_hop_reasoner.find_repurposing_candidates(
                disease_id, max_candidates=100
            )
            for c in hop_candidates:
                key = f"{c.drug_id}_{c.disease_id}"
                if key not in all_candidates:
                    all_candidates[key] = c
                else:
                    # Combine scores if found by multiple methods
                    existing = all_candidates[key]
                    combined_score = (existing.score + c.score) / 2
                    all_candidates[key] = RepurposingCandidate(
                        drug_id=c.drug_id,
                        drug_name=c.drug_name,
                        disease_id=c.disease_id,
                        disease_name=c.disease_name,
                        score=combined_score,
                        method="ensemble",
                        evidence={
                            "similarity": existing.evidence,
                            "multi_hop": c.evidence,
                        },
                        explanation=f"Found by multiple methods: {existing.explanation} | {c.explanation}",
                    )

        # Sort and return top k
        candidates = list(all_candidates.values())
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[:top_k]

    def _find_by_similarity(
        self,
        disease_id: str,
        top_k: int = 30,
    ) -> list[RepurposingCandidate]:
        """Find candidates by transferring from similar diseases."""
        candidates = []

        # Find similar diseases
        similar_diseases = self.similarity_calculator.find_similar_diseases(
            disease_id, top_k=10
        )

        if not similar_diseases:
            return candidates

        # Get treatments from similar diseases
        drug_scores: dict[str, list[tuple[str, float]]] = defaultdict(list)

        for similar_id, similarity in similar_diseases:
            profile = self.similarity_calculator.disease_profiles.get(similar_id)
            if profile and profile.known_treatments:
                for treatment_id in profile.known_treatments:
                    drug_scores[treatment_id].append((similar_id, similarity))

        # Create candidates
        graph = self.similarity_calculator.graph
        target_profile = self.similarity_calculator.disease_profiles.get(disease_id)

        for drug_id, sources in drug_scores.items():
            # Score based on cumulative similarity
            score = sum(sim for _, sim in sources) / len(sources)

            if graph and drug_id in graph:
                drug_name = graph.nodes[drug_id].get("name", drug_id)
            else:
                drug_name = drug_id

            disease_name = target_profile.name if target_profile else disease_id

            source_diseases = [
                graph.nodes[sid].get("name", sid) if graph and sid in graph else sid
                for sid, _ in sources
            ]

            candidate = RepurposingCandidate(
                drug_id=drug_id,
                drug_name=drug_name,
                disease_id=disease_id,
                disease_name=disease_name,
                score=score,
                method="similarity_transfer",
                evidence={
                    "source_diseases": sources,
                    "num_sources": len(sources),
                },
                explanation=f"{drug_name} treats similar diseases: {', '.join(source_diseases[:3])}",
            )
            candidates.append(candidate)

        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[:top_k]
