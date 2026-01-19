#!/usr/bin/env python3
"""
Knowledge Graph Embedding Models for Drug Repurposing.

Implements multiple embedding approaches:
- TransE: Translation-based (h + r ≈ t)
- RotatE: Rotation-based (h ∘ r ≈ t in complex space)
- ComplEx: Complex-valued embeddings with Hermitian products
- DistMult: Bilinear diagonal model

Each model learns vector representations of entities and relations
that can be used for link prediction (drug-disease associations).
"""

import math
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""

    num_entities: int
    num_relations: int
    embedding_dim: int = 256
    margin: float = 1.0
    regularization: float = 0.0001
    negative_samples: int = 10
    device: str = "cpu"


class BaseEmbeddingModel(nn.Module, ABC):
    """Base class for KG embedding models."""

    def __init__(self, config: EmbeddingConfig):
        super().__init__()
        self.config = config
        self.entity_embeddings = nn.Embedding(
            config.num_entities, config.embedding_dim
        )
        self.relation_embeddings = nn.Embedding(
            config.num_relations, config.embedding_dim
        )

    @abstractmethod
    def score(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        """Compute score for triples. Higher = more likely."""
        pass

    def forward(
        self,
        head_ids: torch.Tensor,
        relation_ids: torch.Tensor,
        tail_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass returning scores."""
        head = self.entity_embeddings(head_ids)
        relation = self.relation_embeddings(relation_ids)
        tail = self.entity_embeddings(tail_ids)
        return self.score(head, relation, tail)

    def get_entity_embedding(self, entity_id: int) -> torch.Tensor:
        """Get embedding for a single entity."""
        idx = torch.tensor([entity_id], device=self.config.device)
        return self.entity_embeddings(idx).squeeze(0)

    def get_relation_embedding(self, relation_id: int) -> torch.Tensor:
        """Get embedding for a single relation."""
        idx = torch.tensor([relation_id], device=self.config.device)
        return self.relation_embeddings(idx).squeeze(0)


class TransE(BaseEmbeddingModel):
    """
    TransE: Translating Embeddings for Modeling Multi-relational Data.

    Score function: -||h + r - t||
    Intuition: relation as translation from head to tail.

    Reference: Bordes et al., 2013
    """

    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)

        # Initialize embeddings
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

        # Normalize relation embeddings
        with torch.no_grad():
            self.relation_embeddings.weight.data = F.normalize(
                self.relation_embeddings.weight.data, p=2, dim=1
            )

    def score(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        # Normalize entities
        head = F.normalize(head, p=2, dim=-1)
        tail = F.normalize(tail, p=2, dim=-1)

        # Score: negative L2 distance
        return -torch.norm(head + relation - tail, p=2, dim=-1)

    def loss(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Margin-based ranking loss."""
        return F.relu(self.config.margin - pos_scores + neg_scores).mean()


class RotatE(BaseEmbeddingModel):
    """
    RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space.

    Score function: -||h ∘ r - t|| where ∘ is Hadamard product in complex space
    Intuition: relation as rotation in complex plane.

    Reference: Sun et al., 2019
    """

    def __init__(self, config: EmbeddingConfig):
        # RotatE uses complex embeddings, so we need 2x the dimension
        super().__init__(config)

        # Re-initialize with proper dimensions for complex numbers
        self.entity_embeddings = nn.Embedding(
            config.num_entities, config.embedding_dim * 2
        )
        # Relations are phase angles (embedding_dim values)
        self.relation_embeddings = nn.Embedding(
            config.num_relations, config.embedding_dim
        )

        self.epsilon = 2.0
        self.embedding_range = (
            self.config.margin + self.epsilon
        ) / config.embedding_dim

        # Initialize
        nn.init.uniform_(
            self.entity_embeddings.weight,
            -self.embedding_range,
            self.embedding_range,
        )
        nn.init.uniform_(
            self.relation_embeddings.weight,
            -math.pi,
            math.pi,
        )

    def score(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        # Split into real and imaginary parts
        dim = head.shape[-1] // 2
        head_re, head_im = head[..., :dim], head[..., dim:]
        tail_re, tail_im = tail[..., :dim], tail[..., dim:]

        # Relation as rotation (phase)
        phase = relation
        rel_re = torch.cos(phase)
        rel_im = torch.sin(phase)

        # Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        # h ∘ r
        rot_re = head_re * rel_re - head_im * rel_im
        rot_im = head_re * rel_im + head_im * rel_re

        # Distance to tail
        diff_re = rot_re - tail_re
        diff_im = rot_im - tail_im

        # L2 norm of complex difference
        score = torch.sqrt(diff_re**2 + diff_im**2 + 1e-8).sum(dim=-1)

        return -score

    def loss(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Self-adversarial negative sampling loss."""
        # Negative log-sigmoid loss
        pos_loss = -F.logsigmoid(pos_scores).mean()
        neg_loss = -F.logsigmoid(-neg_scores).mean()
        return pos_loss + neg_loss


class ComplEx(BaseEmbeddingModel):
    """
    ComplEx: Complex Embeddings for Simple Link Prediction.

    Score function: Re(<h, r, conj(t)>) - Hermitian inner product
    Intuition: asymmetric relations via complex conjugate.

    Reference: Trouillon et al., 2016
    """

    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)

        # Complex embeddings (real + imaginary)
        self.entity_embeddings = nn.Embedding(
            config.num_entities, config.embedding_dim * 2
        )
        self.relation_embeddings = nn.Embedding(
            config.num_relations, config.embedding_dim * 2
        )

        # Initialize
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def score(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        # Split into real and imaginary
        dim = head.shape[-1] // 2
        head_re, head_im = head[..., :dim], head[..., dim:]
        rel_re, rel_im = relation[..., :dim], relation[..., dim:]
        tail_re, tail_im = tail[..., :dim], tail[..., dim:]

        # Hermitian inner product: Re(<h, r, conj(t)>)
        # = Re((h_re + i*h_im) * (r_re + i*r_im) * (t_re - i*t_im))
        score = (
            (head_re * rel_re * tail_re).sum(dim=-1)
            + (head_re * rel_im * tail_im).sum(dim=-1)
            + (head_im * rel_re * tail_im).sum(dim=-1)
            - (head_im * rel_im * tail_re).sum(dim=-1)
        )

        return score

    def loss(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Binary cross-entropy loss with regularization."""
        pos_loss = F.softplus(-pos_scores).mean()
        neg_loss = F.softplus(neg_scores).mean()

        # L2 regularization
        reg = self.config.regularization * (
            self.entity_embeddings.weight.norm(p=2) ** 2
            + self.relation_embeddings.weight.norm(p=2) ** 2
        )

        return pos_loss + neg_loss + reg


class DistMult(BaseEmbeddingModel):
    """
    DistMult: Embedding Entities and Relations for Learning and Inference in KBs.

    Score function: <h, r, t> - trilinear dot product with diagonal relation matrix
    Intuition: symmetric bilinear model (relations are symmetric).

    Reference: Yang et al., 2015
    """

    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)

        # Initialize
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def score(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        # Trilinear dot product: sum(h * r * t)
        return (head * relation * tail).sum(dim=-1)

    def loss(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Binary cross-entropy loss with regularization."""
        pos_loss = F.softplus(-pos_scores).mean()
        neg_loss = F.softplus(neg_scores).mean()

        # L2 regularization
        reg = self.config.regularization * (
            self.entity_embeddings.weight.norm(p=2) ** 2
            + self.relation_embeddings.weight.norm(p=2) ** 2
        )

        return pos_loss + neg_loss + reg


class ConvE(nn.Module):
    """
    ConvE: Convolutional 2D Knowledge Graph Embeddings.

    Uses 2D convolution over reshaped embeddings.
    Better at modeling complex relationships.

    Reference: Dettmers et al., 2018
    """

    def __init__(self, config: EmbeddingConfig):
        super().__init__()
        self.config = config

        # Reshape dimensions for 2D convolution
        self.emb_dim1 = 20
        self.emb_dim2 = config.embedding_dim // self.emb_dim1

        self.entity_embeddings = nn.Embedding(
            config.num_entities, config.embedding_dim
        )
        self.relation_embeddings = nn.Embedding(
            config.num_relations, config.embedding_dim
        )

        # Convolution layers
        self.conv1 = nn.Conv2d(1, 32, (3, 3), padding=1)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm1d(config.embedding_dim)

        self.dropout = nn.Dropout(0.2)

        # Fully connected
        self.fc = nn.Linear(32 * self.emb_dim1 * self.emb_dim2, config.embedding_dim)

        # Initialize
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def forward(
        self,
        head_ids: torch.Tensor,
        relation_ids: torch.Tensor,
        tail_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        head = self.entity_embeddings(head_ids)
        relation = self.relation_embeddings(relation_ids)

        # Reshape to 2D
        batch_size = head.size(0)
        head = head.view(batch_size, 1, self.emb_dim1, self.emb_dim2)
        relation = relation.view(batch_size, 1, self.emb_dim1, self.emb_dim2)

        # Stack head and relation
        stacked = torch.cat([head, relation], dim=2)  # (batch, 1, 2*emb_dim1, emb_dim2)

        # Convolution
        x = self.bn0(stacked)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Flatten and FC
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Score against tail (or all entities)
        if tail_ids is not None:
            tail = self.entity_embeddings(tail_ids)
            score = (x * tail).sum(dim=-1)
        else:
            # Score against all entities
            score = torch.mm(x, self.entity_embeddings.weight.t())

        return score


ModelType = Literal["transE", "rotateE", "complEx", "distMult", "convE"]


def create_embedding_model(
    model_type: ModelType,
    config: EmbeddingConfig,
) -> nn.Module:
    """Factory function to create embedding models."""
    models = {
        "transE": TransE,
        "rotateE": RotatE,
        "complEx": ComplEx,
        "distMult": DistMult,
        "convE": ConvE,
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")

    return models[model_type](config)


class EmbeddingTrainer:
    """Trainer for KG embedding models."""

    def __init__(
        self,
        model: nn.Module,
        config: EmbeddingConfig,
        learning_rate: float = 0.001,
    ):
        self.model = model.to(config.device)
        self.config = config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def generate_negatives(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor,
        num_negatives: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate negative samples by corrupting tails."""
        batch_size = heads.size(0)
        neg_tails = torch.randint(
            0,
            self.config.num_entities,
            (batch_size * num_negatives,),
            device=self.config.device,
        )

        # Repeat positive heads and relations
        neg_heads = heads.repeat_interleave(num_negatives)
        neg_relations = relations.repeat_interleave(num_negatives)

        return neg_heads, neg_relations, neg_tails

    def train_step(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor,
    ) -> float:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        # Positive scores
        pos_scores = self.model(heads, relations, tails)

        # Generate negatives and compute scores
        neg_heads, neg_relations, neg_tails = self.generate_negatives(
            heads, relations, tails, self.config.negative_samples
        )
        neg_scores = self.model(neg_heads, neg_relations, neg_tails)

        # Reshape neg_scores to match batch
        neg_scores = neg_scores.view(-1, self.config.negative_samples).mean(dim=1)

        # Compute loss
        loss = self.model.loss(pos_scores, neg_scores)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor,
    ) -> dict[str, float]:
        """Evaluate model on test triples."""
        self.model.eval()
        with torch.no_grad():
            scores = self.model(heads, relations, tails)

        return {
            "mean_score": scores.mean().item(),
            "std_score": scores.std().item(),
        }
