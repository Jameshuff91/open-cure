#!/usr/bin/env python3
"""
h922-v2: Full GraphSAGE retrain on DRKG — replace Node2Vec embeddings.

Goal: push per-drug R@30 from ~37% (Node2Vec kNN) toward the 60% Oracle
ceiling by learning message-passing representations that random-walk
embeddings can't capture.

Design:
- Input: data/raw/drkg/drkg_no_treatment.tsv (~5.8M edges, 97k nodes, no
  treatment edges — prevents eval leakage).
- Model: 3-layer GraphSAGE (mean aggregator, 256-d output to match
  Node2Vec dim). Single-relation graph (edge types folded into
  neighbour lists — simpler than hetero, no perf gap for our scale).
- Training: unsupervised link prediction. Sample positive edges, corrupt
  to get negatives, minimize BCE on dot-product similarity. This is the
  direct analogue of Node2Vec's skip-gram objective but uses message
  passing instead of random walks.
- Runtime target: ~4-8 hrs on 1x RTX 4090 / A100.
- Output: data/embeddings/graphsage_256_entities.npy +
  data/embeddings/graphsage_256_embeddings.npy — drop-in replacements
  for Node2Vec, same format.

Install (on GPU instance):
    pip install torch torch-geometric torch-scatter torch-sparse tqdm numpy

Usage:
    python3 scripts/h922_v2_graphsage_retrain.py \\
        --drkg-tsv data/raw/drkg/drkg_no_treatment.tsv \\
        --out-dir data/embeddings \\
        --epochs 50 --batch 4096 --hidden 256 --num-neighbors 15,10

Evaluation happens on the LOCAL machine after pulling the embedding
files back: point production_predictor at the new embeddings and run
scripts/h393_holdout_tier_validation.py (see scripts/h922_v2_evaluate.py).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--drkg-tsv", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--num-neighbors", type=str, default="15,10",
                   help="Comma-separated fanout per layer")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-pct", type=float, default=0.05,
                   help="Fraction of edges held out for val/test AUC")
    p.add_argument("--early-stop-patience", type=int, default=5)
    return p.parse_args()


def load_drkg_edges(tsv_path: Path) -> Tuple[List[str], np.ndarray]:
    """Parse DRKG TSV into (node_ids, edge_index_src_dst).

    Returns entities in index order + a (2, num_edges) numpy array of
    int32 indices. Edges are bidirectionalised elsewhere.
    """
    print(f"Loading DRKG edges from {tsv_path} ...")
    node2id: dict = {}
    src_list: List[int] = []
    dst_list: List[int] = []
    n_lines = 0
    with open(tsv_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            h, _r, t = parts[0], parts[1], parts[2]
            if h not in node2id:
                node2id[h] = len(node2id)
            if t not in node2id:
                node2id[t] = len(node2id)
            src_list.append(node2id[h])
            dst_list.append(node2id[t])
            n_lines += 1
            if n_lines % 500_000 == 0:
                print(f"  {n_lines:>8,} edges / {len(node2id):>7,} nodes")
    print(f"  done: {n_lines:,} edges, {len(node2id):,} nodes")

    entities = [""] * len(node2id)
    for k, v in node2id.items():
        entities[v] = k
    edge_index = np.asarray([src_list, dst_list], dtype=np.int64)
    return entities, edge_index


def bidirectionalize(edge_index: np.ndarray) -> np.ndarray:
    rev = np.stack([edge_index[1], edge_index[0]])
    return np.concatenate([edge_index, rev], axis=1)


def split_edges(
    edge_index: np.ndarray, eval_pct: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = edge_index.shape[1]
    n_eval = int(n * eval_pct)
    perm = rng.permutation(n)
    eval_idx = perm[:n_eval]
    train_idx = perm[n_eval:]
    return edge_index[:, train_idx], edge_index[:, eval_idx]


def main() -> None:
    args = parse_args()
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.nn import SAGEConv
    from torch_geometric.loader import LinkNeighborLoader

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")

    entities, edge_index_np = load_drkg_edges(Path(args.drkg_tsv))
    num_nodes = len(entities)
    num_edges = edge_index_np.shape[1]

    # Split directed edges before bidirectionalisation to avoid leaks
    train_ei, eval_ei = split_edges(edge_index_np, args.eval_pct, args.seed)
    print(f"Train edges: {train_ei.shape[1]:,}; eval edges: {eval_ei.shape[1]:,}")

    train_ei_bi = bidirectionalize(train_ei)

    x = torch.empty(num_nodes, args.hidden)
    nn.init.xavier_normal_(x)

    data = Data(
        x=x,
        edge_index=torch.from_numpy(train_ei_bi).long(),
        num_nodes=num_nodes,
    )

    class GraphSAGEModel(nn.Module):
        def __init__(self, in_dim: int, hidden: int, layers: int = 3):
            super().__init__()
            self.convs = nn.ModuleList()
            d = in_dim
            for _ in range(layers):
                self.convs.append(SAGEConv(d, hidden, aggr="mean"))
                d = hidden
            self.layers = layers

        def forward(self, x, edge_index):
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                if i != self.layers - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=0.2, training=self.training)
            return x

    model = GraphSAGEModel(args.hidden, args.hidden, layers=3).to(device)
    x_init = data.x.to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + [x_init.requires_grad_(True)],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    num_neighbors = [int(n) for n in args.num_neighbors.split(",")]

    loader = LinkNeighborLoader(
        data=data,
        num_neighbors=num_neighbors,
        batch_size=args.batch,
        edge_label_index=torch.from_numpy(train_ei).long(),
        neg_sampling_ratio=1.0,
        shuffle=True,
        num_workers=0,
    )
    print(f"Batches per epoch: {len(loader):,}")

    def eval_auc() -> float:
        model.eval()
        with torch.no_grad():
            # Simple full-graph forward pass for final embeddings
            final = model(x_init, data.edge_index.to(device))
        rng = np.random.default_rng(args.seed + 1)
        n_eval = min(20_000, eval_ei.shape[1])
        pos_sample = rng.choice(eval_ei.shape[1], size=n_eval, replace=False)
        pos = eval_ei[:, pos_sample]
        neg_src = rng.integers(0, num_nodes, size=n_eval)
        neg_dst = rng.integers(0, num_nodes, size=n_eval)
        with torch.no_grad():
            ps = torch.from_numpy(pos[0]).to(device).long()
            pd = torch.from_numpy(pos[1]).to(device).long()
            ns = torch.from_numpy(neg_src).to(device).long()
            nd = torch.from_numpy(neg_dst).to(device).long()
            pos_scores = (final[ps] * final[pd]).sum(-1).sigmoid()
            neg_scores = (final[ns] * final[nd]).sum(-1).sigmoid()
        y = np.concatenate(
            [np.ones(n_eval), np.zeros(n_eval)]
        )
        s = np.concatenate(
            [pos_scores.cpu().numpy(), neg_scores.cpu().numpy()]
        )
        # AUROC by Mann-Whitney U
        order = np.argsort(s)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(s) + 1)
        r_pos = ranks[: n_eval].sum()
        auc = (r_pos - n_eval * (n_eval + 1) / 2) / (n_eval * n_eval)
        return float(auc)

    best_auc = -1.0
    patience_left = args.early_stop_patience
    best_state = None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        n_seen = 0
        for batch in loader:
            optimizer.zero_grad()
            batch = batch.to(device)
            h = model(x_init[batch.n_id], batch.edge_index)
            # edge_label_index points into the subgraph's local nodes
            src = h[batch.edge_label_index[0]]
            dst = h[batch.edge_label_index[1]]
            logit = (src * dst).sum(-1)
            label = batch.edge_label.float()
            loss = F.binary_cross_entropy_with_logits(logit, label)
            loss.backward()
            optimizer.step()
            bs = batch.edge_label.numel()
            total_loss += loss.item() * bs
            n_seen += bs
        avg_loss = total_loss / max(n_seen, 1)
        auc = eval_auc()
        dur = time.time() - t0
        print(
            f"Epoch {epoch:>3}/{args.epochs}  loss={avg_loss:.4f}  "
            f"val_auc={auc:.4f}  ({dur:.0f}s)"
        )
        if auc > best_auc:
            best_auc = auc
            patience_left = args.early_stop_patience
            best_state = {
                "model_state": model.state_dict(),
                "x_init": x_init.detach().cpu().clone(),
            }
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stop")
                break

    # Load best + compute final embeddings
    if best_state is not None:
        model.load_state_dict(best_state["model_state"])
        x_final_init = best_state["x_init"].to(device)
    else:
        x_final_init = x_init

    model.eval()
    with torch.no_grad():
        final = model(x_final_init, data.edge_index.to(device))
    final_np = final.cpu().numpy().astype(np.float32)

    ents_out = out_dir / "graphsage_256_entities.npy"
    embs_out = out_dir / "graphsage_256_embeddings.npy"
    np.save(ents_out, np.array(entities, dtype=object), allow_pickle=True)
    np.save(embs_out, final_np)
    print(f"Saved {ents_out}, {embs_out}; best val AUC={best_auc:.4f}")

    meta = {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "hidden_dim": args.hidden,
        "epochs_run": epoch,
        "best_val_auc": best_auc,
        "drkg_tsv": args.drkg_tsv,
    }
    import json
    (out_dir / "graphsage_256_meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
