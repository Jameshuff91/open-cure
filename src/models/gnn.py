#!/usr/bin/env python3
"""
Graph Neural Networks for Drug Repurposing.

Implements multiple GNN architectures:
- GAT: Graph Attention Network (learns attention over neighbors)
- GraphSAGE: Sample and Aggregate (inductive learning)
- GIN: Graph Isomorphism Network (most expressive)
- RGCN: Relational GCN (handles multiple edge types)

These models learn node representations that capture both local
and global graph structure for link prediction.
"""

import math
import sys
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for GAT."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat

        # Linear transformations for each head
        self.W = nn.Linear(in_features, out_features * num_heads, bias=False)

        # Attention parameters
        self.a_src = nn.Parameter(torch.zeros(num_heads, out_features))
        self.a_dst = nn.Parameter(torch.zeros(num_heads, out_features))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Edge indices [2, num_edges]
        Returns:
            Updated node features
        """
        num_nodes = x.size(0)
        row, col = edge_index

        # Linear transformation
        h = self.W(x).view(num_nodes, self.num_heads, self.out_features)

        # Compute attention scores
        # e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
        # Using additive decomposition: a^T [Wh_i || Wh_j] = a_src^T Wh_i + a_dst^T Wh_j
        attn_src = (h * self.a_src).sum(dim=-1)  # [num_nodes, num_heads]
        attn_dst = (h * self.a_dst).sum(dim=-1)

        # Attention for each edge
        attn = attn_src[row] + attn_dst[col]  # [num_edges, num_heads]
        attn = self.leaky_relu(attn)

        # Softmax over neighbors
        attn = self._sparse_softmax(attn, row, num_nodes)
        attn = self.dropout(attn)

        # Aggregate: weighted sum of neighbor features
        out = torch.zeros(num_nodes, self.num_heads, self.out_features, device=x.device)
        attn_expanded = attn.unsqueeze(-1)  # [num_edges, num_heads, 1]
        h_neighbors = h[col]  # [num_edges, num_heads, out_features]

        # Scatter add
        out.index_add_(0, row, attn_expanded * h_neighbors)

        if self.concat:
            out = out.view(num_nodes, self.num_heads * self.out_features)
        else:
            out = out.mean(dim=1)

        return out

    def _sparse_softmax(
        self,
        attn: torch.Tensor,
        index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """Compute softmax over neighbors for each node."""
        # Subtract max for numerical stability
        attn_max = torch.zeros(num_nodes, attn.size(1), device=attn.device)
        attn_max.scatter_reduce_(0, index.unsqueeze(1).expand_as(attn), attn, reduce="amax")
        attn = attn - attn_max[index]

        # Exp
        attn = torch.exp(attn)

        # Sum over neighbors
        attn_sum = torch.zeros(num_nodes, attn.size(1), device=attn.device)
        attn_sum.index_add_(0, index, attn)

        # Normalize
        return attn / (attn_sum[index] + 1e-8)


class GATLayer(nn.Module):
    """Single Graph Attention Network layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(
            in_features, out_features, num_heads, dropout, concat
        )
        self.norm = nn.LayerNorm(out_features * num_heads if concat else out_features)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        out = self.attention(x, edge_index)
        return self.norm(F.elu(out))


class GAT(nn.Module):
    """
    Graph Attention Network for drug-disease link prediction.

    Uses multi-head attention to learn which neighbors are most important
    for each node's representation.

    Reference: Veličković et al., 2018
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 256,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Entity embeddings
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        # GAT layers
        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(
            GATLayer(embedding_dim, hidden_dim, num_heads, dropout, concat=True)
        )

        # Middle layers
        for _ in range(num_layers - 2):
            self.layers.append(
                GATLayer(
                    hidden_dim * num_heads, hidden_dim, num_heads, dropout, concat=True
                )
            )

        # Final layer (no concat)
        if num_layers > 1:
            self.layers.append(
                GATLayer(
                    hidden_dim * num_heads, hidden_dim, num_heads, dropout, concat=False
                )
            )

        # Link prediction head
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def encode(self, edge_index: torch.Tensor) -> torch.Tensor:
        """Encode all entities using GAT layers."""
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

        combined = torch.cat([head, relation, tail], dim=-1)
        return self.link_predictor(combined).squeeze(-1)

    def forward(
        self,
        edge_index: torch.Tensor,
        head_ids: torch.Tensor,
        relation_ids: torch.Tensor,
        tail_ids: torch.Tensor,
    ) -> torch.Tensor:
        entity_embeddings = self.encode(edge_index)
        return self.predict_link(entity_embeddings, head_ids, relation_ids, tail_ids)


class GraphSAGELayer(nn.Module):
    """GraphSAGE aggregation layer with mean/max/LSTM aggregators."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        aggregator: str = "mean",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.aggregator = aggregator

        if aggregator == "mean":
            self.agg_linear = nn.Linear(in_features, out_features)
        elif aggregator == "max":
            self.agg_linear = nn.Linear(in_features, out_features)
        elif aggregator == "lstm":
            self.lstm = nn.LSTM(in_features, out_features, batch_first=True)
        else:
            raise ValueError(f"Unknown aggregator: {aggregator}")

        self.self_linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        num_nodes = x.size(0)

        # Self features
        self_out = self.self_linear(x)

        # Aggregate neighbors
        if self.aggregator == "mean":
            neighbor_sum = torch.zeros_like(x)
            neighbor_count = torch.zeros(num_nodes, 1, device=x.device)
            neighbor_sum.index_add_(0, row, x[col])
            neighbor_count.index_add_(0, row, torch.ones(col.size(0), 1, device=x.device))
            neighbor_mean = neighbor_sum / (neighbor_count + 1e-8)
            agg_out = self.agg_linear(neighbor_mean)
        elif self.aggregator == "max":
            # Simplified max pooling
            neighbor_out = torch.zeros_like(x)
            neighbor_out.scatter_reduce_(0, row.unsqueeze(1).expand_as(x[col]), x[col], reduce="amax")
            agg_out = self.agg_linear(neighbor_out)
        else:
            # LSTM aggregator would need neighbor ordering
            agg_out = self.agg_linear(x)  # Fallback to mean-like

        out = self_out + agg_out
        out = self.norm(F.relu(out))
        return self.dropout(out)


class GraphSAGE(nn.Module):
    """
    GraphSAGE: Inductive Representation Learning on Large Graphs.

    Learns to aggregate features from sampled neighbors.
    Good for inductive settings (new nodes).

    Reference: Hamilton et al., 2017
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 256,
        hidden_dim: int = 128,
        num_layers: int = 2,
        aggregator: str = "mean",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        self.layers = nn.ModuleList()
        self.layers.append(
            GraphSAGELayer(embedding_dim, hidden_dim, aggregator, dropout)
        )
        for _ in range(num_layers - 1):
            self.layers.append(
                GraphSAGELayer(hidden_dim, hidden_dim, aggregator, dropout)
            )

        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def encode(self, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.entity_embeddings.weight
        for layer in self.layers:
            x = layer(x, edge_index)
        return x

    def forward(
        self,
        edge_index: torch.Tensor,
        head_ids: torch.Tensor,
        relation_ids: torch.Tensor,
        tail_ids: torch.Tensor,
    ) -> torch.Tensor:
        entity_embeddings = self.encode(edge_index)
        head = entity_embeddings[head_ids]
        tail = entity_embeddings[tail_ids]
        relation = self.relation_embeddings(relation_ids)
        combined = torch.cat([head, relation, tail], dim=-1)
        return self.link_predictor(combined).squeeze(-1)


class GINLayer(nn.Module):
    """Graph Isomorphism Network layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        eps: float = 0.0,
        train_eps: bool = True,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features),
        )

        if train_eps:
            self.eps = nn.Parameter(torch.tensor([eps]))
        else:
            self.register_buffer("eps", torch.tensor([eps]))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        num_nodes = x.size(0)

        # Aggregate neighbors
        neighbor_sum = torch.zeros_like(x)
        neighbor_sum.index_add_(0, row, x[col])

        # (1 + eps) * x + neighbor_sum
        out = (1 + self.eps) * x + neighbor_sum
        return self.mlp(out)


class GIN(nn.Module):
    """
    Graph Isomorphism Network.

    Most expressive GNN under the Weisfeiler-Lehman test.
    Good for distinguishing graph structures.

    Reference: Xu et al., 2019
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 256,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        self.layers = nn.ModuleList()
        self.layers.append(GINLayer(embedding_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GINLayer(hidden_dim, hidden_dim))

        self.dropout = nn.Dropout(dropout)

        # Jumping knowledge: combine all layer outputs
        self.jk_linear = nn.Linear(hidden_dim * num_layers, hidden_dim)

        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def encode(self, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.entity_embeddings.weight
        layer_outputs = []

        for layer in self.layers:
            x = layer(x, edge_index)
            x = self.dropout(x)
            layer_outputs.append(x)

        # Jumping knowledge concatenation
        x = torch.cat(layer_outputs, dim=-1)
        x = self.jk_linear(x)
        return x

    def forward(
        self,
        edge_index: torch.Tensor,
        head_ids: torch.Tensor,
        relation_ids: torch.Tensor,
        tail_ids: torch.Tensor,
    ) -> torch.Tensor:
        entity_embeddings = self.encode(edge_index)
        head = entity_embeddings[head_ids]
        tail = entity_embeddings[tail_ids]
        relation = self.relation_embeddings(relation_ids)
        combined = torch.cat([head, relation, tail], dim=-1)
        return self.link_predictor(combined).squeeze(-1)


class RGCNLayer(nn.Module):
    """Relational Graph Convolutional Network layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_relations: int,
        num_bases: int | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations

        # Basis decomposition for parameter efficiency
        if num_bases is None:
            num_bases = min(num_relations, 30)
        self.num_bases = num_bases

        # Basis matrices
        self.bases = nn.Parameter(torch.zeros(num_bases, in_features, out_features))

        # Coefficients for each relation
        self.coefficients = nn.Parameter(torch.zeros(num_relations, num_bases))

        # Self-loop weight
        self.self_weight = nn.Linear(in_features, out_features, bias=False)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.bases)
        nn.init.xavier_uniform_(self.coefficients)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        num_nodes = x.size(0)
        out = self.self_weight(x)

        # Compute relation-specific weights via basis decomposition
        # W_r = sum_b(c_rb * B_b)
        weights = torch.einsum("rb,bio->rio", self.coefficients, self.bases)

        row, col = edge_index

        for r in range(self.num_relations):
            mask = edge_type == r
            if not mask.any():
                continue

            r_row = row[mask]
            r_col = col[mask]

            # Message passing for relation r
            msg = x[r_col] @ weights[r]

            # Aggregate
            out.index_add_(0, r_row, msg)

        return out


class RGCN(nn.Module):
    """
    Relational Graph Convolutional Network.

    Handles multiple edge types (relations) explicitly.
    Good for knowledge graphs with diverse relation types.

    Reference: Schlichtkrull et al., 2018
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 256,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_bases: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)

        self.layers = nn.ModuleList()
        self.layers.append(
            RGCNLayer(embedding_dim, hidden_dim, num_relations, num_bases)
        )
        for _ in range(num_layers - 1):
            self.layers.append(
                RGCNLayer(hidden_dim, hidden_dim, num_relations, num_bases)
            )

        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

        # Relation embeddings for link prediction
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def encode(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        x = self.entity_embeddings.weight

        for layer, norm in zip(self.layers, self.norms):
            x = layer(x, edge_index, edge_type)
            x = norm(F.relu(x))
            x = self.dropout(x)

        return x

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        head_ids: torch.Tensor,
        relation_ids: torch.Tensor,
        tail_ids: torch.Tensor,
    ) -> torch.Tensor:
        entity_embeddings = self.encode(edge_index, edge_type)
        head = entity_embeddings[head_ids]
        tail = entity_embeddings[tail_ids]
        relation = self.relation_embeddings(relation_ids)
        combined = torch.cat([head, relation, tail], dim=-1)
        return self.link_predictor(combined).squeeze(-1)


def create_gnn_model(
    model_type: str,
    num_entities: int,
    num_relations: int,
    **kwargs: Any,
) -> nn.Module:
    """Factory function to create GNN models."""
    models = {
        "gat": GAT,
        "graphsage": GraphSAGE,
        "gin": GIN,
        "rgcn": RGCN,
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")

    return models[model_type](num_entities, num_relations, **kwargs)
