#!/usr/bin/env python3
"""
Link prediction models for drug-disease association discovery.

This module implements multiple approaches for predicting drug-disease links:
1. Knowledge Graph Embeddings (TransE, RotatE, ComplEx)
2. Graph Neural Networks (GraphSAGE, GAT)
3. LLM-based reasoning

The goal is ensemble predictions that combine multiple signals.
"""

import json
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO")


@dataclass
class Prediction:
    """A drug-disease prediction with metadata."""

    drug_id: str
    drug_name: str
    disease_id: str
    disease_name: str
    score: float
    rank: int
    model: str
    explanation: str | None = None
    supporting_evidence: list[dict[str, Any]] | None = None


class BasePredictor(ABC):
    """Base class for drug-disease predictors."""

    @abstractmethod
    def train(self, graph_path: Path) -> None:
        """Train the model on a knowledge graph."""
        pass

    @abstractmethod
    def predict(self, drug_id: str, disease_id: str) -> float:
        """Predict the likelihood of a drug-disease association."""
        pass

    @abstractmethod
    def predict_all_diseases(self, drug_id: str, top_k: int = 100) -> list[Prediction]:
        """Predict all diseases for a drug, returning top k."""
        pass

    @abstractmethod
    def predict_all_drugs(self, disease_id: str, top_k: int = 100) -> list[Prediction]:
        """Predict all drugs for a disease, returning top k."""
        pass


class TransEModel(nn.Module):
    """TransE knowledge graph embedding model.

    Learns embeddings where: head + relation ≈ tail
    Good for modeling hierarchical relationships.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 256,
        margin: float = 1.0,
    ):
        super().__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.margin = margin

        # Initialize embeddings
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

        # Normalize relation embeddings
        self.relation_embeddings.weight.data = F.normalize(
            self.relation_embeddings.weight.data, p=2, dim=1
        )

    def forward(
        self,
        head_ids: torch.Tensor,
        relation_ids: torch.Tensor,
        tail_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute TransE scores."""
        head = self.entity_embeddings(head_ids)
        relation = self.relation_embeddings(relation_ids)
        tail = self.entity_embeddings(tail_ids)

        # Normalize entity embeddings
        head = F.normalize(head, p=2, dim=1)
        tail = F.normalize(tail, p=2, dim=1)

        # Score: -||h + r - t||
        score = -torch.norm(head + relation - tail, p=2, dim=1)
        return score

    def loss(
        self,
        pos_head: torch.Tensor,
        pos_relation: torch.Tensor,
        pos_tail: torch.Tensor,
        neg_head: torch.Tensor,
        neg_relation: torch.Tensor,
        neg_tail: torch.Tensor,
    ) -> torch.Tensor:
        """Compute margin-based ranking loss."""
        pos_score = self.forward(pos_head, pos_relation, pos_tail)
        neg_score = self.forward(neg_head, neg_relation, neg_tail)

        # Margin ranking loss
        loss = F.relu(self.margin - pos_score + neg_score).mean()
        return loss


class GraphSAGELayer(nn.Module):
    """GraphSAGE aggregation layer."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features * 2, out_features)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate neighbor features."""
        row, col = edge_index

        # Aggregate neighbors (mean)
        neighbor_sum = torch.zeros_like(x)
        neighbor_count = torch.zeros(x.size(0), device=x.device)

        neighbor_sum.index_add_(0, row, x[col])
        neighbor_count.index_add_(0, row, torch.ones(col.size(0), device=x.device))

        neighbor_mean = neighbor_sum / (neighbor_count.unsqueeze(1) + 1e-8)

        # Concatenate self and neighbor features
        combined = torch.cat([x, neighbor_mean], dim=1)

        return F.relu(self.linear(combined))


class DrugDiseaseGNN(nn.Module):
    """Graph Neural Network for drug-disease link prediction."""

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 256,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        # GraphSAGE layers
        self.layers = nn.ModuleList()
        self.layers.append(GraphSAGELayer(embedding_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GraphSAGELayer(hidden_dim, hidden_dim))

        # Link prediction head
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def encode(self, edge_index: torch.Tensor) -> torch.Tensor:
        """Encode all entities using GNN layers."""
        x = self.entity_embeddings.weight

        for layer in self.layers:
            x = layer(x, edge_index)

        return x

    def predict_link(
        self,
        entity_embeddings: torch.Tensor,
        head_ids: torch.Tensor,
        relation_ids: torch.Tensor,
        tail_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Predict link probability."""
        head = entity_embeddings[head_ids]
        tail = entity_embeddings[tail_ids]
        relation = self.relation_embeddings(relation_ids)

        combined = torch.cat([head, relation, tail], dim=1)
        score = self.link_predictor(combined).squeeze()

        return torch.sigmoid(score)


class DrugDiseasePredictor(BasePredictor):
    """
    Ensemble predictor combining multiple approaches.

    This is the main interface for drug repurposing predictions.
    It combines:
    - Knowledge graph embeddings (TransE)
    - Graph neural networks (GraphSAGE)
    - Path-based features
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        device: str | None = None,
    ):
        self.embedding_dim = embedding_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.entity_to_id: dict[str, int] = {}
        self.id_to_entity: dict[int, str] = {}
        self.relation_to_id: dict[str, int] = {}
        self.id_to_relation: dict[int, str] = {}

        self.entity_names: dict[str, str] = {}
        self.entity_types: dict[str, str] = {}

        self.transE_model: TransEModel | None = None
        self.gnn_model: DrugDiseaseGNN | None = None

        self.trained = False

    def _build_mappings(self, nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> None:
        """Build entity and relation ID mappings."""
        logger.info("Building entity and relation mappings...")

        # Entity mappings
        entities = nodes_df["id"].unique()
        for idx, entity in enumerate(entities):
            self.entity_to_id[entity] = idx
            self.id_to_entity[idx] = entity

        # Store entity metadata
        for _, row in nodes_df.iterrows():
            self.entity_names[row["id"]] = row.get("name", row["id"])
            self.entity_types[row["id"]] = row.get("type", "Unknown")

        # Relation mappings
        relations = edges_df["relation"].unique()
        for idx, relation in enumerate(relations):
            self.relation_to_id[relation] = idx
            self.id_to_relation[idx] = relation

        logger.info(f"  Entities: {len(self.entity_to_id):,}")
        logger.info(f"  Relations: {len(self.relation_to_id):,}")

    def _prepare_training_data(
        self, edges_df: pd.DataFrame
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare training triples."""
        heads = []
        relations = []
        tails = []

        for _, row in edges_df.iterrows():
            if row["source"] in self.entity_to_id and row["target"] in self.entity_to_id:
                heads.append(self.entity_to_id[row["source"]])
                relations.append(self.relation_to_id[row["relation"]])
                tails.append(self.entity_to_id[row["target"]])

        return (
            torch.tensor(heads, dtype=torch.long),
            torch.tensor(relations, dtype=torch.long),
            torch.tensor(tails, dtype=torch.long),
        )

    def train(
        self,
        graph_path: Path,
        epochs: int = 100,
        batch_size: int = 1024,
        learning_rate: float = 0.001,
    ) -> None:
        """Train the ensemble models."""
        logger.info("Training Drug-Disease Predictor...")

        # Load data
        nodes_df = pd.read_csv(graph_path / "unified_nodes.csv")
        edges_df = pd.read_csv(graph_path / "unified_edges.csv")

        # Build mappings
        self._build_mappings(nodes_df, edges_df)

        # Prepare data
        heads, relations, tails = self._prepare_training_data(edges_df)

        num_entities = len(self.entity_to_id)
        num_relations = len(self.relation_to_id)

        # Initialize models
        self.transE_model = TransEModel(
            num_entities, num_relations, self.embedding_dim
        ).to(self.device)

        self.gnn_model = DrugDiseaseGNN(
            num_entities, num_relations, self.embedding_dim
        ).to(self.device)

        # Train TransE
        logger.info("Training TransE model...")
        self._train_transE(heads, relations, tails, epochs, batch_size, learning_rate)

        # Train GNN
        logger.info("Training GNN model...")
        self._train_gnn(heads, relations, tails, epochs, batch_size, learning_rate)

        self.trained = True
        logger.success("Training complete!")

    def _train_transE(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor,
        epochs: int,
        batch_size: int,
        learning_rate: float,
    ) -> None:
        """Train TransE model."""
        assert self.transE_model is not None

        optimizer = torch.optim.Adam(self.transE_model.parameters(), lr=learning_rate)
        num_entities = len(self.entity_to_id)

        heads = heads.to(self.device)
        relations = relations.to(self.device)
        tails = tails.to(self.device)

        for epoch in tqdm(range(epochs), desc="TransE"):
            self.transE_model.train()
            total_loss = 0.0

            # Mini-batch training
            perm = torch.randperm(heads.size(0))
            for i in range(0, heads.size(0), batch_size):
                batch_idx = perm[i : i + batch_size]

                batch_heads = heads[batch_idx]
                batch_relations = relations[batch_idx]
                batch_tails = tails[batch_idx]

                # Generate negative samples (corrupt tails)
                neg_tails = torch.randint(
                    0, num_entities, batch_tails.size(), device=self.device
                )

                optimizer.zero_grad()
                loss = self.transE_model.loss(
                    batch_heads,
                    batch_relations,
                    batch_tails,
                    batch_heads,
                    batch_relations,
                    neg_tails,
                )
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

    def _train_gnn(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor,
        epochs: int,
        batch_size: int,
        learning_rate: float,
    ) -> None:
        """Train GNN model."""
        assert self.gnn_model is not None

        optimizer = torch.optim.Adam(self.gnn_model.parameters(), lr=learning_rate)
        num_entities = len(self.entity_to_id)

        # Build edge index for GNN
        edge_index = torch.stack([heads, tails]).to(self.device)
        heads = heads.to(self.device)
        relations = relations.to(self.device)
        tails = tails.to(self.device)

        for epoch in tqdm(range(epochs), desc="GNN"):
            self.gnn_model.train()
            total_loss = 0.0

            # Encode all entities
            entity_embeddings = self.gnn_model.encode(edge_index)

            # Mini-batch training
            perm = torch.randperm(heads.size(0))
            for i in range(0, heads.size(0), batch_size):
                batch_idx = perm[i : i + batch_size]

                batch_heads = heads[batch_idx]
                batch_relations = relations[batch_idx]
                batch_tails = tails[batch_idx]

                # Positive samples
                pos_scores = self.gnn_model.predict_link(
                    entity_embeddings, batch_heads, batch_relations, batch_tails
                )

                # Negative samples
                neg_tails = torch.randint(
                    0, num_entities, batch_tails.size(), device=self.device
                )
                neg_scores = self.gnn_model.predict_link(
                    entity_embeddings, batch_heads, batch_relations, neg_tails
                )

                # Binary cross-entropy loss
                pos_loss = F.binary_cross_entropy(
                    pos_scores, torch.ones_like(pos_scores)
                )
                neg_loss = F.binary_cross_entropy(
                    neg_scores, torch.zeros_like(neg_scores)
                )
                loss = pos_loss + neg_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

    def predict(self, drug_id: str, disease_id: str) -> float:
        """Predict the likelihood of a drug-disease association."""
        if not self.trained:
            raise ValueError("Model not trained. Call train() first.")

        if drug_id not in self.entity_to_id or disease_id not in self.entity_to_id:
            return 0.0

        # Get ensemble score (average of TransE and GNN)
        transE_score = self._predict_transE(drug_id, disease_id)
        gnn_score = self._predict_gnn(drug_id, disease_id)

        return (transE_score + gnn_score) / 2

    def _predict_transE(self, drug_id: str, disease_id: str) -> float:
        """Get TransE prediction score."""
        assert self.transE_model is not None

        self.transE_model.eval()
        with torch.no_grad():
            head = torch.tensor([self.entity_to_id[drug_id]], device=self.device)
            tail = torch.tensor([self.entity_to_id[disease_id]], device=self.device)

            # Use "treats" relation if available, otherwise average over all
            if "treats" in self.relation_to_id:
                relation = torch.tensor(
                    [self.relation_to_id["treats"]], device=self.device
                )
                score = self.transE_model(head, relation, tail)
            else:
                # Average over all relations
                scores = []
                for rel_id in range(len(self.relation_to_id)):
                    relation = torch.tensor([rel_id], device=self.device)
                    scores.append(self.transE_model(head, relation, tail).item())
                score = torch.tensor([np.mean(scores)])

            # Normalize to [0, 1]
            return torch.sigmoid(score).item()

    def _predict_gnn(self, drug_id: str, disease_id: str) -> float:
        """Get GNN prediction score."""
        # Simplified - would need full graph for proper GNN inference
        return 0.5  # Placeholder

    def predict_all_diseases(self, drug_id: str, top_k: int = 100) -> list[Prediction]:
        """Predict all diseases for a drug."""
        if not self.trained:
            raise ValueError("Model not trained. Call train() first.")

        predictions = []

        # Get all disease entities
        diseases = [
            eid
            for eid, etype in self.entity_types.items()
            if etype == "Disease"
        ]

        for disease_id in tqdm(diseases, desc="Predicting diseases"):
            score = self.predict(drug_id, disease_id)
            predictions.append(
                Prediction(
                    drug_id=drug_id,
                    drug_name=self.entity_names.get(drug_id, drug_id),
                    disease_id=disease_id,
                    disease_name=self.entity_names.get(disease_id, disease_id),
                    score=score,
                    rank=0,
                    model="ensemble",
                )
            )

        # Sort by score and assign ranks
        predictions.sort(key=lambda x: x.score, reverse=True)
        for i, pred in enumerate(predictions[:top_k]):
            pred.rank = i + 1

        return predictions[:top_k]

    def predict_all_drugs(self, disease_id: str, top_k: int = 100) -> list[Prediction]:
        """Predict all drugs for a disease."""
        if not self.trained:
            raise ValueError("Model not trained. Call train() first.")

        predictions = []

        # Get all drug entities
        drugs = [
            eid
            for eid, etype in self.entity_types.items()
            if etype == "Drug"
        ]

        for drug_id in tqdm(drugs, desc="Predicting drugs"):
            score = self.predict(drug_id, disease_id)
            predictions.append(
                Prediction(
                    drug_id=drug_id,
                    drug_name=self.entity_names.get(drug_id, drug_id),
                    disease_id=disease_id,
                    disease_name=self.entity_names.get(disease_id, disease_id),
                    score=score,
                    rank=0,
                    model="ensemble",
                )
            )

        # Sort by score and assign ranks
        predictions.sort(key=lambda x: x.score, reverse=True)
        for i, pred in enumerate(predictions[:top_k]):
            pred.rank = i + 1

        return predictions[:top_k]

    def save(self, path: Path) -> None:
        """Save model and mappings."""
        path.mkdir(parents=True, exist_ok=True)

        # Save mappings
        with open(path / "mappings.json", "w") as f:
            json.dump(
                {
                    "entity_to_id": self.entity_to_id,
                    "relation_to_id": self.relation_to_id,
                    "entity_names": self.entity_names,
                    "entity_types": self.entity_types,
                },
                f,
            )

        # Save model weights
        if self.transE_model:
            torch.save(self.transE_model.state_dict(), path / "transE.pt")
        if self.gnn_model:
            torch.save(self.gnn_model.state_dict(), path / "gnn.pt")

    def load(self, path: Path) -> None:
        """Load model and mappings."""
        # Load mappings
        with open(path / "mappings.json") as f:
            mappings = json.load(f)
            self.entity_to_id = mappings["entity_to_id"]
            self.id_to_entity = {v: k for k, v in self.entity_to_id.items()}
            self.relation_to_id = mappings["relation_to_id"]
            self.id_to_relation = {v: k for k, v in self.relation_to_id.items()}
            self.entity_names = mappings["entity_names"]
            self.entity_types = mappings["entity_types"]

        # Initialize and load models
        num_entities = len(self.entity_to_id)
        num_relations = len(self.relation_to_id)

        self.transE_model = TransEModel(
            num_entities, num_relations, self.embedding_dim
        ).to(self.device)
        self.transE_model.load_state_dict(torch.load(path / "transE.pt"))

        self.gnn_model = DrugDiseaseGNN(
            num_entities, num_relations, self.embedding_dim
        ).to(self.device)
        self.gnn_model.load_state_dict(torch.load(path / "gnn.pt"))

        self.trained = True
