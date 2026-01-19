#!/usr/bin/env python3
"""
Train knowledge graph embeddings for drug repurposing.
Optimized for Apple Silicon (M1/M2/M3/M4) with MPS acceleration.
"""

import pickle
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from loguru import logger

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    """Get the best available device (MPS for Apple Silicon, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        logger.info("Using MPS (Apple Silicon GPU)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        logger.info("Using CUDA GPU")
        return torch.device("cuda")
    else:
        logger.info("Using CPU")
        return torch.device("cpu")


class KnowledgeGraph:
    """Load and process the unified knowledge graph."""

    def __init__(self, processed_dir: Path = PROCESSED_DIR):
        self.processed_dir = processed_dir
        self.entity2id: Dict[str, int] = {}
        self.id2entity: Dict[int, str] = {}
        self.relation2id: Dict[str, int] = {}
        self.id2relation: Dict[int, str] = {}
        self.triples: List[Tuple[int, int, int]] = []  # (head, relation, tail)
        self.entity_types: Dict[int, str] = {}

    def load(self) -> "KnowledgeGraph":
        """Load the knowledge graph from processed files."""
        logger.info("Loading knowledge graph...")

        # Load nodes
        nodes_df = pd.read_csv(self.processed_dir / "unified_nodes.csv")
        for idx, row in tqdm(nodes_df.iterrows(), total=len(nodes_df), desc="Loading nodes"):
            entity_id = row["id"]
            self.entity2id[entity_id] = idx
            self.id2entity[idx] = entity_id
            self.entity_types[idx] = row.get("type", "unknown")

        logger.info(f"Loaded {len(self.entity2id):,} entities")

        # Load edges
        edges_df = pd.read_csv(self.processed_dir / "unified_edges.csv")

        # Build relation mapping
        relations = edges_df["relation"].unique()
        for idx, rel in enumerate(relations):
            self.relation2id[rel] = idx
            self.id2relation[idx] = rel

        logger.info(f"Found {len(self.relation2id):,} relation types")

        # Build triples
        skipped = 0
        for _, row in tqdm(edges_df.iterrows(), total=len(edges_df), desc="Loading edges"):
            head = row["source"]
            tail = row["target"]
            rel = row["relation"]

            if head in self.entity2id and tail in self.entity2id:
                h_id = self.entity2id[head]
                r_id = self.relation2id[rel]
                t_id = self.entity2id[tail]
                self.triples.append((h_id, r_id, t_id))
            else:
                skipped += 1

        if skipped > 0:
            logger.warning(f"Skipped {skipped:,} edges with missing entities")

        logger.success(f"Loaded {len(self.triples):,} triples")
        return self

    def split(self, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[List, List, List]:
        """Split triples into train/val/test sets."""
        random.shuffle(self.triples)
        n = len(self.triples)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train = self.triples[:train_end]
        val = self.triples[train_end:val_end]
        test = self.triples[val_end:]

        logger.info(f"Split: train={len(train):,}, val={len(val):,}, test={len(test):,}")
        return train, val, test

    @property
    def num_entities(self) -> int:
        return len(self.entity2id)

    @property
    def num_relations(self) -> int:
        return len(self.relation2id)


class TripleDataset(Dataset):
    """Dataset for knowledge graph triples with negative sampling."""

    def __init__(self, triples: List[Tuple[int, int, int]], num_entities: int, num_negatives: int = 1):
        self.triples = triples
        self.num_entities = num_entities
        self.num_negatives = num_negatives

        # Build set for fast lookup
        self.triple_set = set(triples)

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        h, r, t = self.triples[idx]

        # Generate negative samples by corrupting head or tail
        negatives = []
        for _ in range(self.num_negatives):
            if random.random() < 0.5:
                # Corrupt head
                neg_h = random.randint(0, self.num_entities - 1)
                while (neg_h, r, t) in self.triple_set:
                    neg_h = random.randint(0, self.num_entities - 1)
                negatives.append((neg_h, r, t))
            else:
                # Corrupt tail
                neg_t = random.randint(0, self.num_entities - 1)
                while (h, r, neg_t) in self.triple_set:
                    neg_t = random.randint(0, self.num_entities - 1)
                negatives.append((h, r, neg_t))

        positive = torch.tensor([h, r, t], dtype=torch.long)
        negative = torch.tensor(negatives, dtype=torch.long)

        return positive, negative


class TransE(nn.Module):
    """TransE: Translation-based embeddings for knowledge graphs."""

    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int = 128, margin: float = 1.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.margin = margin

        # Initialize embeddings
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        # Initialize with uniform distribution
        nn.init.uniform_(self.entity_embeddings.weight, -6/np.sqrt(embedding_dim), 6/np.sqrt(embedding_dim))
        nn.init.uniform_(self.relation_embeddings.weight, -6/np.sqrt(embedding_dim), 6/np.sqrt(embedding_dim))

        # Normalize relation embeddings
        with torch.no_grad():
            self.relation_embeddings.weight.data = F.normalize(self.relation_embeddings.weight.data, p=2, dim=1)

    def score(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute TransE score: ||h + r - t||."""
        h_emb = self.entity_embeddings(h)
        r_emb = self.relation_embeddings(r)
        t_emb = self.entity_embeddings(t)

        # L2 distance
        return torch.norm(h_emb + r_emb - t_emb, p=2, dim=-1)

    def forward(self, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """Compute margin-based ranking loss."""
        # Positive scores
        pos_h, pos_r, pos_t = positive[:, 0], positive[:, 1], positive[:, 2]
        pos_score = self.score(pos_h, pos_r, pos_t)

        # Negative scores
        neg_h, neg_r, neg_t = negative[:, :, 0], negative[:, :, 1], negative[:, :, 2]
        batch_size, num_neg = neg_h.shape
        neg_score = self.score(
            neg_h.reshape(-1),
            neg_r.reshape(-1),
            neg_t.reshape(-1)
        ).reshape(batch_size, num_neg)

        # Margin ranking loss: max(0, margin + pos_score - neg_score)
        loss = torch.clamp(self.margin + pos_score.unsqueeze(1) - neg_score, min=0)
        return loss.mean()

    def normalize_entities(self):
        """Normalize entity embeddings (important for TransE)."""
        with torch.no_grad():
            self.entity_embeddings.weight.data = F.normalize(
                self.entity_embeddings.weight.data, p=2, dim=1
            )


class RotatE(nn.Module):
    """RotatE: Rotation-based embeddings in complex space."""

    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int = 128, margin: float = 6.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.epsilon = 2.0

        # Entity embeddings (complex: real + imaginary)
        self.entity_re = nn.Embedding(num_entities, embedding_dim)
        self.entity_im = nn.Embedding(num_entities, embedding_dim)

        # Relation embeddings (phase angles)
        self.relation_phase = nn.Embedding(num_relations, embedding_dim)

        # Initialize
        embedding_range = (self.margin + self.epsilon) / embedding_dim
        nn.init.uniform_(self.entity_re.weight, -embedding_range, embedding_range)
        nn.init.uniform_(self.entity_im.weight, -embedding_range, embedding_range)
        nn.init.uniform_(self.relation_phase.weight, -np.pi, np.pi)

    def score(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute RotatE score."""
        h_re = self.entity_re(h)
        h_im = self.entity_im(h)
        t_re = self.entity_re(t)
        t_im = self.entity_im(t)

        phase = self.relation_phase(r)
        r_re = torch.cos(phase)
        r_im = torch.sin(phase)

        # Complex multiplication: h * r
        hr_re = h_re * r_re - h_im * r_im
        hr_im = h_re * r_im + h_im * r_re

        # Distance to tail
        diff_re = hr_re - t_re
        diff_im = hr_im - t_im

        return torch.sqrt(diff_re**2 + diff_im**2 + 1e-8).sum(dim=-1)

    def forward(self, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """Compute margin-based ranking loss."""
        pos_h, pos_r, pos_t = positive[:, 0], positive[:, 1], positive[:, 2]
        pos_score = self.score(pos_h, pos_r, pos_t)

        neg_h, neg_r, neg_t = negative[:, :, 0], negative[:, :, 1], negative[:, :, 2]
        batch_size, num_neg = neg_h.shape
        neg_score = self.score(
            neg_h.reshape(-1),
            neg_r.reshape(-1),
            neg_t.reshape(-1)
        ).reshape(batch_size, num_neg)

        # Self-adversarial negative sampling loss
        pos_loss = -F.logsigmoid(self.margin - pos_score)
        neg_loss = -F.logsigmoid(neg_score - self.margin).mean(dim=1)

        return (pos_loss + neg_loss).mean()


def evaluate(model: nn.Module, triples: List[Tuple[int, int, int]],
             kg: KnowledgeGraph, device: torch.device,
             batch_size: int = 1000) -> Dict[str, float]:
    """Evaluate model with Mean Rank and Hits@K metrics."""
    model.eval()

    ranks = []
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_10 = 0

    # Sample for faster evaluation
    eval_triples = random.sample(triples, min(5000, len(triples)))

    with torch.no_grad():
        for h, r, t in tqdm(eval_triples, desc="Evaluating"):
            # Score all possible tails
            h_tensor = torch.tensor([h], device=device).expand(kg.num_entities)
            r_tensor = torch.tensor([r], device=device).expand(kg.num_entities)
            t_tensor = torch.arange(kg.num_entities, device=device)

            scores = model.score(h_tensor, r_tensor, t_tensor)

            # Get rank of true tail
            true_score = scores[t].item()
            rank = (scores < true_score).sum().item() + 1

            ranks.append(rank)
            if rank <= 1:
                hits_at_1 += 1
            if rank <= 3:
                hits_at_3 += 1
            if rank <= 10:
                hits_at_10 += 1

    n = len(ranks)
    metrics = {
        "MR": np.mean(ranks),
        "MRR": np.mean([1/r for r in ranks]),
        "Hits@1": hits_at_1 / n,
        "Hits@3": hits_at_3 / n,
        "Hits@10": hits_at_10 / n,
    }

    model.train()
    return metrics


def train_model(
    model_class: type,
    kg: KnowledgeGraph,
    train_triples: List,
    val_triples: List,
    device: torch.device,
    embedding_dim: int = 128,
    batch_size: int = 2048,
    epochs: int = 100,
    lr: float = 0.001,
    num_negatives: int = 5,
    model_name: str = "model"
) -> nn.Module:
    """Train a knowledge graph embedding model."""

    logger.info(f"\n{'='*60}")
    logger.info(f"Training {model_name}")
    logger.info(f"{'='*60}")
    logger.info(f"Entities: {kg.num_entities:,}, Relations: {kg.num_relations:,}")
    logger.info(f"Training triples: {len(train_triples):,}")
    logger.info(f"Embedding dim: {embedding_dim}, Batch size: {batch_size}")
    logger.info(f"Device: {device}")

    # Create model
    model = model_class(
        num_entities=kg.num_entities,
        num_relations=kg.num_relations,
        embedding_dim=embedding_dim
    ).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    # Create dataset and dataloader
    dataset = TripleDataset(train_triples, kg.num_entities, num_negatives)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_mrr = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for positive, negative in pbar:
            positive = positive.to(device)
            negative = negative.to(device)

            optimizer.zero_grad()
            loss = model(positive, negative)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Normalize entity embeddings for TransE
            if hasattr(model, 'normalize_entities'):
                model.normalize_entities()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / num_batches

        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            metrics = evaluate(model, val_triples, kg, device)
            logger.info(
                f"Epoch {epoch+1}: Loss={avg_loss:.4f}, "
                f"MRR={metrics['MRR']:.4f}, Hits@10={metrics['Hits@10']:.4f}"
            )

            scheduler.step(avg_loss)

            # Save best model
            if metrics['MRR'] > best_mrr:
                best_mrr = metrics['MRR']
                best_model_state = model.state_dict().copy()

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save model
    model_path = MODELS_DIR / f"{model_name}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'entity2id': kg.entity2id,
        'relation2id': kg.relation2id,
        'embedding_dim': embedding_dim,
        'num_entities': kg.num_entities,
        'num_relations': kg.num_relations,
    }, model_path)
    logger.success(f"Saved model to {model_path}")

    return model


def main():
    """Main training pipeline."""
    logger.info("="*60)
    logger.info("Drug Repurposing Knowledge Graph Embedding Training")
    logger.info("="*60)

    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Get device
    device = get_device()

    # Load knowledge graph
    kg = KnowledgeGraph().load()

    # Split data
    train_triples, val_triples, test_triples = kg.split()

    # Training config
    config = {
        "embedding_dim": 128,
        "batch_size": 4096,  # Larger batch for M4 Pro
        "epochs": 50,
        "lr": 0.001,
        "num_negatives": 10,
    }

    # Train TransE
    transe_model = train_model(
        TransE, kg, train_triples, val_triples, device,
        model_name="transe",
        **config
    )

    # Evaluate on test set
    logger.info("\nFinal evaluation on test set:")
    test_metrics = evaluate(transe_model, test_triples, kg, device)
    for k, v in test_metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    # Train RotatE
    config["epochs"] = 50
    rotate_model = train_model(
        RotatE, kg, train_triples, val_triples, device,
        model_name="rotate",
        **config
    )

    # Evaluate RotatE
    logger.info("\nRotatE Final evaluation on test set:")
    test_metrics = evaluate(rotate_model, test_triples, kg, device)
    for k, v in test_metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    # Save test metrics
    with open(MODELS_DIR / "training_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    logger.success("\nTraining complete!")
    logger.info(f"Models saved to: {MODELS_DIR}")


if __name__ == "__main__":
    main()
