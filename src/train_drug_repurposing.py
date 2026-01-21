#!/usr/bin/env python3
"""
Train heterogeneous GNN for drug repurposing.
Uses RGCN with AUPRC evaluation - comparable to TxGNN methodology.

This approach:
1. Treats drug-disease link prediction as a binary classification task
2. Uses AUPRC as the primary metric (same as TxGNN)
3. Leverages the full heterogeneous graph structure
4. Implements proper train/val/test splits on drug-disease edges
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)
from tqdm import tqdm
from loguru import logger

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Drug-disease relation types that indicate treatment/indication
# Updated for cleaned data format
INDICATION_RELATIONS = {
    # Cleaned data format (normalized)
    "INDICATED_FOR",
    "TREATS",
    "DRUG_TARGET_DISEASE",
    # Original format (for backwards compatibility)
    "DRUGBANK::treats::Compound:Disease",
    "indication",
    "treats",
    "Hetionet::CtD::Compound:Disease",
    "Hetionet::CpD::Compound:Disease",
    "palliates",
    "GNBR::T::Compound:Disease",
    "GNBR::Pa::Compound:Disease",
    "GNBR::J::Compound:Disease",
}

CONTRAINDICATION_RELATIONS = {
    # Cleaned data format
    "CONTRAINDICATED_FOR",
    # Original format
    "contraindication",
}

# Relations to exclude from message passing (we predict these)
TARGET_RELATIONS = INDICATION_RELATIONS | CONTRAINDICATION_RELATIONS


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.backends.mps.is_available():
        logger.info("Using MPS (Apple Silicon GPU)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        logger.info("Using CUDA GPU")
        return torch.device("cuda")
    else:
        logger.info("Using CPU")
        return torch.device("cpu")


class HeterogeneousGraph:
    """Load and process the heterogeneous knowledge graph."""

    def __init__(self, processed_dir: Path = PROCESSED_DIR):
        self.processed_dir = processed_dir

        # Entity mappings
        self.entity2id: Dict[str, int] = {}
        self.id2entity: Dict[int, str] = {}
        self.entity_types: Dict[int, str] = {}

        # Relation mappings
        self.relation2id: Dict[str, int] = {}
        self.id2relation: Dict[int, str] = {}

        # Graph structure
        self.edge_index: List[Tuple[int, int]] = []
        self.edge_type: List[int] = []

        # Drug-disease edges (for prediction)
        self.drug_disease_edges: List[Tuple[int, int, int, bool]] = []  # (drug, disease, rel, is_indication)

        # Node type sets
        self.drug_ids: Set[int] = set()
        self.disease_ids: Set[int] = set()

    def load(self) -> "HeterogeneousGraph":
        """Load the knowledge graph."""
        logger.info("Loading heterogeneous knowledge graph...")

        # Load nodes (use cleaned data if available)
        clean_nodes = self.processed_dir / "unified_nodes_clean.csv"
        original_nodes = self.processed_dir / "unified_nodes.csv"
        nodes_file = clean_nodes if clean_nodes.exists() else original_nodes
        logger.info(f"Using nodes file: {nodes_file.name}")
        nodes_df = pd.read_csv(nodes_file)
        for idx, row in tqdm(nodes_df.iterrows(), total=len(nodes_df), desc="Loading nodes"):
            entity_id = row["id"]
            self.entity2id[entity_id] = idx
            self.id2entity[idx] = entity_id
            node_type = str(row.get("type", "unknown")).lower()
            self.entity_types[idx] = node_type

            # Track drug and disease nodes
            if "drug" in node_type or "compound" in node_type:
                self.drug_ids.add(idx)
            elif "disease" in node_type:
                self.disease_ids.add(idx)

        logger.info(f"Loaded {len(self.entity2id):,} entities")
        logger.info(f"  Drugs: {len(self.drug_ids):,}")
        logger.info(f"  Diseases: {len(self.disease_ids):,}")

        # Load edges (use cleaned data if available)
        clean_edges = self.processed_dir / "unified_edges_clean.csv"
        original_edges = self.processed_dir / "unified_edges.csv"
        edges_file = clean_edges if clean_edges.exists() else original_edges
        logger.info(f"Using edges file: {edges_file.name}")
        edges_df = pd.read_csv(edges_file, low_memory=False)

        # Build relation mapping (excluding target relations from message passing)
        mp_relations = []  # message passing relations
        target_relations = []  # prediction target relations

        for rel in edges_df["relation"].unique():
            if rel in TARGET_RELATIONS:
                target_relations.append(rel)
            else:
                mp_relations.append(rel)

        # Map message passing relations
        for idx, rel in enumerate(mp_relations):
            self.relation2id[rel] = idx
            self.id2relation[idx] = rel

        logger.info(f"Message passing relations: {len(self.relation2id):,}")
        logger.info(f"Target relations (drug-disease): {len(target_relations)}")

        # Build graph
        skipped = 0
        drug_disease_count = 0

        for _, row in tqdm(edges_df.iterrows(), total=len(edges_df), desc="Loading edges"):
            head = row["source"]
            tail = row["target"]
            rel = row["relation"]

            if head not in self.entity2id or tail not in self.entity2id:
                skipped += 1
                continue

            h_id = self.entity2id[head]
            t_id = self.entity2id[tail]

            # Check if this is a drug-disease edge
            if rel in TARGET_RELATIONS:
                # Determine which is drug and which is disease
                if h_id in self.drug_ids and t_id in self.disease_ids:
                    drug_id, disease_id = h_id, t_id
                elif t_id in self.drug_ids and h_id in self.disease_ids:
                    drug_id, disease_id = t_id, h_id
                else:
                    # Not a valid drug-disease edge
                    continue

                is_indication = rel in INDICATION_RELATIONS
                self.drug_disease_edges.append((drug_id, disease_id, is_indication))
                drug_disease_count += 1
            else:
                # Regular edge for message passing
                if rel in self.relation2id:
                    r_id = self.relation2id[rel]
                    self.edge_index.append((h_id, t_id))
                    self.edge_type.append(r_id)
                    # Add reverse edge for undirected message passing
                    self.edge_index.append((t_id, h_id))
                    self.edge_type.append(r_id)

        if skipped > 0:
            logger.warning(f"Skipped {skipped:,} edges with missing entities")

        logger.success(f"Loaded {len(self.edge_index):,} message passing edges")
        logger.success(f"Loaded {len(self.drug_disease_edges):,} drug-disease edges for prediction")

        # Count indications vs contraindications
        indications = sum(1 for _, _, is_ind in self.drug_disease_edges if is_ind)
        contraindications = len(self.drug_disease_edges) - indications
        logger.info(f"  Indications: {indications:,}")
        logger.info(f"  Contraindications: {contraindications:,}")

        return self

    def get_edge_tensors(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get edge index and edge type tensors."""
        edge_index = torch.tensor(self.edge_index, dtype=torch.long, device=device).t()
        edge_type = torch.tensor(self.edge_type, dtype=torch.long, device=device)

        # Validate shapes match
        assert edge_index.shape[1] == edge_type.shape[0], \
            f"Edge index ({edge_index.shape[1]}) and edge type ({edge_type.shape[0]}) length mismatch!"

        logger.info(f"Edge tensors: index shape {edge_index.shape}, type shape {edge_type.shape}")
        return edge_index, edge_type

    def split_drug_disease_edges(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1
    ) -> Tuple[List, List, List]:
        """Split drug-disease edges into train/val/test."""
        # Shuffle edges
        edges = self.drug_disease_edges.copy()
        random.shuffle(edges)

        n = len(edges)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train = edges[:train_end]
        val = edges[train_end:val_end]
        test = edges[val_end:]

        logger.info(f"Drug-disease split: train={len(train):,}, val={len(val):,}, test={len(test):,}")
        return train, val, test

    @property
    def num_entities(self) -> int:
        return len(self.entity2id)

    @property
    def num_relations(self) -> int:
        return len(self.relation2id)


class DrugDiseaseDataset(Dataset):
    """Dataset for drug-disease link prediction with negative sampling."""

    def __init__(
        self,
        positive_edges: List[Tuple[int, int, bool]],
        all_drugs: Set[int],
        all_diseases: Set[int],
        positive_set: Set[Tuple[int, int]],
        num_negatives: int = 1,
    ):
        self.positive_edges = positive_edges
        self.all_drugs = list(all_drugs)
        self.all_diseases = list(all_diseases)
        self.positive_set = positive_set
        self.num_negatives = num_negatives

    def __len__(self) -> int:
        return len(self.positive_edges)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        drug, disease, is_indication = self.positive_edges[idx]

        # Positive sample
        samples = [(drug, disease, 1.0)]  # 1.0 = positive

        # Negative samples (corrupt disease)
        for _ in range(self.num_negatives):
            neg_disease = random.choice(self.all_diseases)
            while (drug, neg_disease) in self.positive_set:
                neg_disease = random.choice(self.all_diseases)
            samples.append((drug, neg_disease, 0.0))  # 0.0 = negative

        drugs = torch.tensor([s[0] for s in samples], dtype=torch.long)
        diseases = torch.tensor([s[1] for s in samples], dtype=torch.long)
        labels = torch.tensor([s[2] for s in samples], dtype=torch.float)

        return drugs, diseases, labels


class RGCNEncoder(nn.Module):
    """RGCN encoder for heterogeneous graphs."""

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_bases: int = 30,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Entity embeddings
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        nn.init.xavier_uniform_(self.entity_embeddings.weight)

        # RGCN layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer
        self.layers.append(
            RGCNLayer(embedding_dim, hidden_dim, num_relations, num_bases)
        )
        self.norms.append(nn.LayerNorm(hidden_dim))

        # Additional layers
        for _ in range(num_layers - 1):
            self.layers.append(
                RGCNLayer(hidden_dim, hidden_dim, num_relations, num_bases)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """Encode all entities."""
        x = self.entity_embeddings.weight

        for layer, norm in zip(self.layers, self.norms):
            x = layer(x, edge_index, edge_type)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)

        return x


class RGCNLayer(nn.Module):
    """Single RGCN layer with basis decomposition."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_relations: int,
        num_bases: int = 30,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations
        self.num_bases = min(num_bases, num_relations)

        # Basis matrices
        self.bases = nn.Parameter(
            torch.empty(self.num_bases, in_features, out_features)
        )

        # Coefficients for each relation
        self.coefficients = nn.Parameter(
            torch.empty(num_relations, self.num_bases)
        )

        # Self-loop
        self.self_weight = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))

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

        # Compute relation weights via basis decomposition
        # W_r = sum_b(c_rb * B_b)
        weights = torch.einsum("rb,bio->rio", self.coefficients, self.bases)

        row, col = edge_index[0], edge_index[1]

        # Aggregate messages per relation type
        for r in range(self.num_relations):
            mask = edge_type == r
            if not mask.any():
                continue

            r_row = row[mask]
            r_col = col[mask]

            # Message: neighbor features transformed by relation weight
            neighbor_feats = x[r_col]  # [num_edges_r, in_features]
            weight_r = weights[r]  # [in_features, out_features]
            msg = torch.mm(neighbor_feats, weight_r)  # [num_edges_r, out_features]

            # Normalize by node degree
            deg = torch.zeros(num_nodes, device=x.device)
            deg.scatter_add_(0, r_row, torch.ones(r_row.size(0), device=x.device))
            deg = deg.clamp(min=1)

            # Aggregate with normalization
            agg = torch.zeros(num_nodes, self.out_features, device=x.device)
            agg.index_add_(0, r_row, msg)
            agg = agg / deg.unsqueeze(1)

            out = out + agg

        return out + self.bias


class DrugRepurposingModel(nn.Module):
    """
    Drug repurposing model using RGCN encoder and MLP decoder.
    Predicts whether a drug treats a disease.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_bases: int = 30,
        dropout: float = 0.2,
    ):
        super().__init__()

        # RGCN encoder
        self.encoder = RGCNEncoder(
            num_entities=num_entities,
            num_relations=num_relations,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_bases=num_bases,
            dropout=dropout,
        )

        # Link prediction decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """Get entity embeddings."""
        return self.encoder(edge_index, edge_type)

    def decode(
        self,
        embeddings: torch.Tensor,
        drug_ids: torch.Tensor,
        disease_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Predict drug-disease scores."""
        drug_emb = embeddings[drug_ids]
        disease_emb = embeddings[disease_ids]
        combined = torch.cat([drug_emb, disease_emb], dim=-1)
        return self.decoder(combined).squeeze(-1)

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        drug_ids: torch.Tensor,
        disease_ids: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = self.encode(edge_index, edge_type)
        return self.decode(embeddings, drug_ids, disease_ids)


def evaluate(
    model: DrugRepurposingModel,
    embeddings: torch.Tensor,
    positive_edges: List[Tuple[int, int, bool]],
    all_drugs: Set[int],
    all_diseases: Set[int],
    positive_set: Set[Tuple[int, int]],
    device: torch.device,
    num_neg_samples: int = 100,
) -> Dict[str, float]:
    """
    Evaluate model with AUROC and AUPRC.

    For each positive edge, sample negative edges and compute metrics.
    """
    model.eval()

    all_scores = []
    all_labels = []

    all_drugs_list = list(all_drugs)
    all_diseases_list = list(all_diseases)

    with torch.no_grad():
        for drug, disease, _ in tqdm(positive_edges, desc="Evaluating"):
            # Positive
            drug_t = torch.tensor([drug], device=device)
            disease_t = torch.tensor([disease], device=device)
            pos_score = model.decode(embeddings, drug_t, disease_t).item()
            all_scores.append(pos_score)
            all_labels.append(1)

            # Negative samples
            neg_diseases = []
            attempts = 0
            while len(neg_diseases) < num_neg_samples and attempts < num_neg_samples * 10:
                neg_d = random.choice(all_diseases_list)
                if (drug, neg_d) not in positive_set and neg_d not in neg_diseases:
                    neg_diseases.append(neg_d)
                attempts += 1

            if neg_diseases:
                neg_drug_t = torch.tensor([drug] * len(neg_diseases), device=device)
                neg_disease_t = torch.tensor(neg_diseases, device=device)
                neg_scores = model.decode(embeddings, neg_drug_t, neg_disease_t).cpu().numpy()
                all_scores.extend(neg_scores.tolist())
                all_labels.extend([0] * len(neg_diseases))

    # Compute metrics
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # Apply sigmoid to get probabilities
    all_probs = 1 / (1 + np.exp(-all_scores))

    auroc = roc_auc_score(all_labels, all_probs)
    auprc = average_precision_score(all_labels, all_probs)

    # Compute precision at various recalls
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)

    metrics = {
        "AUROC": auroc,
        "AUPRC": auprc,
        "Precision@50%Recall": float(precision[np.argmin(np.abs(recall - 0.5))]),
    }

    model.train()
    return metrics


def train_model(
    graph: HeterogeneousGraph,
    train_edges: List[Tuple[int, int, bool]],
    val_edges: List[Tuple[int, int, bool]],
    test_edges: List[Tuple[int, int, bool]],
    device: torch.device,
    config: Dict,
) -> DrugRepurposingModel:
    """Train the drug repurposing model."""

    logger.info("=" * 60)
    logger.info("Training Drug Repurposing Model (RGCN)")
    logger.info("=" * 60)
    logger.info(f"Entities: {graph.num_entities:,}")
    logger.info(f"Relations: {graph.num_relations:,}")
    logger.info(f"Training edges: {len(train_edges):,}")
    logger.info(f"Config: {config}")

    # Build positive set for negative sampling
    positive_set = set((d, dis) for d, dis, _ in graph.drug_disease_edges)

    # Get graph tensors
    edge_index, edge_type = graph.get_edge_tensors(device)

    # Create model
    model = DrugRepurposingModel(
        num_entities=graph.num_entities,
        num_relations=graph.num_relations,
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_bases=config["num_bases"],
        dropout=config["dropout"],
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    # Create dataset and dataloader
    dataset = DrugDiseaseDataset(
        train_edges,
        graph.drug_ids,
        graph.disease_ids,
        positive_set,
        num_negatives=config["num_negatives"],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"],
    )

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    best_auprc = 0
    best_model_state = None
    patience_counter = 0

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        num_batches = 0

        # Encode once per epoch (expensive)
        with torch.no_grad():
            embeddings = model.encode(edge_index, edge_type)

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for drugs, diseases, labels in pbar:
            drugs = drugs.to(device).flatten()
            diseases = diseases.to(device).flatten()
            labels = labels.to(device).flatten()

            optimizer.zero_grad()

            # Decode
            scores = model.decode(embeddings, drugs, diseases)
            loss = criterion(scores, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        avg_loss = total_loss / num_batches

        # Evaluate
        if (epoch + 1) % config["eval_every"] == 0 or epoch == 0:
            # Re-encode with updated parameters
            with torch.no_grad():
                embeddings = model.encode(edge_index, edge_type)

            val_metrics = evaluate(
                model, embeddings, val_edges,
                graph.drug_ids, graph.disease_ids,
                positive_set, device,
                num_neg_samples=50,
            )

            logger.info(
                f"Epoch {epoch+1}: Loss={avg_loss:.4f}, "
                f"AUROC={val_metrics['AUROC']:.4f}, "
                f"AUPRC={val_metrics['AUPRC']:.4f}"
            )

            # Save best model
            if val_metrics["AUPRC"] > best_auprc:
                best_auprc = val_metrics["AUPRC"]
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
                logger.info(f"  New best AUPRC: {best_auprc:.4f}")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= config["patience"]:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final evaluation on test set
    logger.info("\n" + "=" * 60)
    logger.info("Final Evaluation on Test Set")
    logger.info("=" * 60)

    with torch.no_grad():
        embeddings = model.encode(edge_index, edge_type)

    test_metrics = evaluate(
        model, embeddings, test_edges,
        graph.drug_ids, graph.disease_ids,
        positive_set, device,
        num_neg_samples=100,
    )

    for k, v in test_metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    # Save model
    model_path = MODELS_DIR / "drug_repurposing_rgcn.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "entity2id": graph.entity2id,
        "relation2id": graph.relation2id,
        "config": config,
        "test_metrics": test_metrics,
    }, model_path)
    logger.success(f"Saved model to {model_path}")

    # Save metrics
    metrics_path = MODELS_DIR / "drug_repurposing_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")

    return model


def main():
    """Main training pipeline."""
    logger.info("=" * 60)
    logger.info("Drug Repurposing with Heterogeneous GNN (RGCN)")
    logger.info("Comparable to TxGNN methodology")
    logger.info("=" * 60)

    # Set seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Device
    device = get_device()

    # Load graph
    graph = HeterogeneousGraph().load()

    # Split drug-disease edges
    train_edges, val_edges, test_edges = graph.split_drug_disease_edges()

    # Training config
    config = {
        "embedding_dim": 128,
        "hidden_dim": 128,
        "num_layers": 2,
        "num_bases": 30,
        "dropout": 0.2,
        "batch_size": 512,
        "lr": 0.001,
        "weight_decay": 1e-5,
        "epochs": 100,
        "num_negatives": 5,
        "eval_every": 5,
        "patience": 10,
    }

    # Train
    model = train_model(
        graph, train_edges, val_edges, test_edges,
        device, config,
    )

    logger.success("\nTraining complete!")


if __name__ == "__main__":
    main()
