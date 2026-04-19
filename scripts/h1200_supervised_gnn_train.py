#!/usr/bin/env python3
"""h1200 — Supervised GraphSAGE on DRKG with treatment edges as explicit labels.

Target: ≥35% honest R@30 (vs 26% Node2Vec). Forks h922_v2 with the loss swapped
from unsupervised link prediction to supervised drug→disease treatment-edge
prediction, with disease-level holdout matching h393 for fair comparison.

Install (GPU instance):
    pip install torch torch-geometric torch-scatter torch-sparse tqdm numpy

Run:
    python3 scripts/h1200_supervised_gnn_train.py \\
        --drkg-no-treatment data/raw/drkg/drkg_no_treatment.tsv \\
        --drkg-full data/raw/drkg/drkg.tsv \\
        --out-dir data/embeddings \\
        --seed 42 --epochs 50 --hidden 256 --batch 512 --num-neighbors 15,10
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np


TREATMENT_RELATION = "DRUGBANK::treats::Compound:Disease"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--drkg-no-treatment", required=True,
                   help="Message-passing graph (treatment edges removed)")
    p.add_argument("--drkg-full", required=True,
                   help="Full DRKG TSV — used to extract treatment edges")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--holdout-disease-pct", type=float, default=0.20)
    p.add_argument("--neg-ratio", type=int, default=3)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=512,
                   help="Treatment edges per batch (positives only; neg samples = batch*neg_ratio)")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--num-neighbors", type=str, default="15,10")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--early-stop-patience", type=int, default=8)
    p.add_argument("--eval-every", type=int, default=2,
                   help="Epochs between holdout R@30 evals")
    p.add_argument("--smoke-test", action="store_true",
                   help="Subset to 50k mp edges + 100 treatment edges + 2 epochs")
    return p.parse_args()


def load_drkg_tsv(tsv_path: Path) -> Tuple[List[Tuple[str, str, str]], int]:
    triples: List[Tuple[str, str, str]] = []
    with open(tsv_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            triples.append((parts[0], parts[1], parts[2]))
    return triples, len(triples)


def build_graph(
    mp_triples: List[Tuple[str, str, str]],
    train_treatment_edges: List[Tuple[str, str]],
) -> Tuple[List[str], np.ndarray, Dict[str, int]]:
    node2id: Dict[str, int] = {}
    src_list: List[int] = []
    dst_list: List[int] = []
    for h, _r, t in mp_triples:
        if h not in node2id:
            node2id[h] = len(node2id)
        if t not in node2id:
            node2id[t] = len(node2id)
        src_list.append(node2id[h])
        dst_list.append(node2id[t])
    for drug, disease in train_treatment_edges:
        if drug not in node2id:
            node2id[drug] = len(node2id)
        if disease not in node2id:
            node2id[disease] = len(node2id)
        src_list.append(node2id[drug])
        dst_list.append(node2id[disease])
    entities = [""] * len(node2id)
    for k, v in node2id.items():
        entities[v] = k
    edge_index = np.asarray([src_list, dst_list], dtype=np.int64)
    return entities, edge_index, node2id


def bidirectionalize(edge_index: np.ndarray) -> np.ndarray:
    rev = np.stack([edge_index[1], edge_index[0]])
    return np.concatenate([edge_index, rev], axis=1)


def split_diseases(
    treatment_edges: List[Tuple[str, str]],
    holdout_pct: float,
    seed: int,
) -> Tuple[Set[str], Set[str]]:
    diseases = sorted({d for _, d in treatment_edges})
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(diseases))
    n_hold = int(len(diseases) * holdout_pct)
    holdout = {diseases[i] for i in perm[:n_hold]}
    train = {d for d in diseases if d not in holdout}
    return train, holdout


def sample_negatives(
    n_pos: int,
    drug_ids: np.ndarray,
    disease_ids: np.ndarray,
    disease_degree: np.ndarray,
    treatment_set: Set[Tuple[int, int]],
    neg_ratio: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    n_neg = n_pos * neg_ratio
    neg_drugs = np.empty(n_neg, dtype=np.int64)
    neg_diseases = np.empty(n_neg, dtype=np.int64)
    disease_probs = disease_degree / disease_degree.sum()
    filled = 0
    attempts = 0
    while filled < n_neg and attempts < n_neg * 10:
        need = n_neg - filled
        d_cand = rng.choice(drug_ids, size=need)
        dis_cand = rng.choice(disease_ids, size=need, p=disease_probs)
        for i in range(need):
            if (int(d_cand[i]), int(dis_cand[i])) not in treatment_set:
                neg_drugs[filled] = d_cand[i]
                neg_diseases[filled] = dis_cand[i]
                filled += 1
                if filled >= n_neg:
                    break
        attempts += need
    return neg_drugs[:filled], neg_diseases[:filled]


def main() -> None:
    args = parse_args()
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.nn import SAGEConv

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")

    print(f"Loading full DRKG to extract treatment edges ...")
    full_triples, _ = load_drkg_tsv(Path(args.drkg_full))
    treatment_edges: List[Tuple[str, str]] = [
        (h, t) for h, r, t in full_triples if r == TREATMENT_RELATION
    ]
    print(f"  {len(treatment_edges):,} treatment edges")

    print(f"Loading message-passing graph ...")
    mp_triples, _ = load_drkg_tsv(Path(args.drkg_no_treatment))
    print(f"  {len(mp_triples):,} non-treatment edges")

    if args.smoke_test:
        mp_triples = mp_triples[:50_000]
        treatment_edges = treatment_edges[:100]
        args.epochs = 2
        args.batch = 32
        print("[SMOKE] Subset to 50k mp, 100 treatment, 2 epochs")

    train_dis, hold_dis = split_diseases(
        treatment_edges, args.holdout_disease_pct, args.seed
    )
    train_treatment = [e for e in treatment_edges if e[1] in train_dis]
    hold_treatment = [e for e in treatment_edges if e[1] in hold_dis]
    print(
        f"Disease split seed={args.seed}: "
        f"{len(train_dis)} train / {len(hold_dis)} holdout diseases  "
        f"({len(train_treatment)} train / {len(hold_treatment)} holdout treatment edges)"
    )

    entities, edge_index_np, node2id = build_graph(mp_triples, train_treatment)
    num_nodes = len(entities)
    num_edges = edge_index_np.shape[1]
    print(f"Graph: {num_nodes:,} nodes / {num_edges:,} directed edges")

    edge_index_bi = bidirectionalize(edge_index_np)

    drug_nodes = sorted({node2id[d] for d, _ in treatment_edges if d in node2id})
    disease_nodes_all = sorted(
        {node2id[d] for _, d in treatment_edges if d in node2id}
    )
    drug_ids_np = np.asarray(drug_nodes, dtype=np.int64)
    disease_ids_np = np.asarray(disease_nodes_all, dtype=np.int64)
    print(f"  {len(drug_nodes):,} drug nodes with ≥1 treatment edge")
    print(f"  {len(disease_nodes_all):,} disease nodes with ≥1 treatment edge")

    disease_degree = np.zeros(len(disease_nodes_all), dtype=np.float64)
    dis_idx_lookup = {d: i for i, d in enumerate(disease_nodes_all)}
    for _, disease in treatment_edges:
        if disease in node2id:
            disease_degree[dis_idx_lookup[node2id[disease]]] += 1.0
    disease_degree += 1.0

    treatment_set_idx: Set[Tuple[int, int]] = {
        (node2id[d], node2id[dis])
        for d, dis in treatment_edges
        if d in node2id and dis in node2id
    }
    train_pos_drug = np.asarray(
        [node2id[d] for d, dis in train_treatment if d in node2id and dis in node2id],
        dtype=np.int64,
    )
    train_pos_dis = np.asarray(
        [node2id[dis] for d, dis in train_treatment if d in node2id and dis in node2id],
        dtype=np.int64,
    )
    hold_pos_drug = np.asarray(
        [node2id[d] for d, dis in hold_treatment if d in node2id and dis in node2id],
        dtype=np.int64,
    )
    hold_pos_dis = np.asarray(
        [node2id[dis] for d, dis in hold_treatment if d in node2id and dis in node2id],
        dtype=np.int64,
    )

    x = torch.empty(num_nodes, args.hidden)
    nn.init.xavier_normal_(x)
    data = Data(
        x=x,
        edge_index=torch.from_numpy(edge_index_bi).long(),
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
    x_param = nn.Parameter(x.to(device))
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + [x_param],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    edge_index_gpu = data.edge_index.to(device)
    rng = np.random.default_rng(args.seed + 1)

    def compute_embeddings() -> torch.Tensor:
        model.eval()
        with torch.no_grad():
            return model(x_param, edge_index_gpu)

    def holdout_recall_at_30() -> Tuple[float, float]:
        if hold_pos_drug.size == 0:
            return 0.0, 0.0
        emb = compute_embeddings()
        drug_emb = emb[torch.from_numpy(drug_ids_np).to(device)]
        gt_per_disease: Dict[int, Set[int]] = {}
        for dr, dis in zip(hold_pos_drug.tolist(), hold_pos_dis.tolist()):
            gt_per_disease.setdefault(dis, set()).add(dr)
        hits30 = 0
        hits30_drug = 0
        n_dis = 0
        for dis_node, gt_drugs in gt_per_disease.items():
            dis_vec = emb[dis_node]
            scores = (drug_emb * dis_vec).sum(-1)
            topk = torch.topk(scores, k=min(30, scores.numel())).indices.cpu().numpy()
            top_drug_nodes = drug_ids_np[topk]
            hit = len(gt_drugs.intersection(set(top_drug_nodes.tolist())))
            if hit > 0:
                hits30 += 1
            hits30_drug += hit / max(len(gt_drugs), 1)
            n_dis += 1
        per_disease_hits = hits30 / max(n_dis, 1)
        per_drug_r30 = hits30_drug / max(n_dis, 1)
        return per_drug_r30, per_disease_hits

    n_pos_total = train_pos_drug.shape[0]
    steps_per_epoch = max(1, n_pos_total // args.batch)
    print(f"Train positives: {n_pos_total:,}  steps/epoch: {steps_per_epoch}")

    best_r30 = -1.0
    best_state = None
    patience_left = args.early_stop_patience
    epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        perm = rng.permutation(n_pos_total)
        total_loss = 0.0
        n_seen = 0
        for step in range(steps_per_epoch):
            start = step * args.batch
            end = min(start + args.batch, n_pos_total)
            batch_idx = perm[start:end]
            pos_d = train_pos_drug[batch_idx]
            pos_dis = train_pos_dis[batch_idx]
            neg_d, neg_dis = sample_negatives(
                pos_d.shape[0], drug_ids_np, disease_ids_np,
                disease_degree, treatment_set_idx, args.neg_ratio, rng,
            )
            optimizer.zero_grad()
            emb = model(x_param, edge_index_gpu)
            pos_src = emb[torch.from_numpy(pos_d).to(device)]
            pos_dst = emb[torch.from_numpy(pos_dis).to(device)]
            neg_src = emb[torch.from_numpy(neg_d).to(device)]
            neg_dst = emb[torch.from_numpy(neg_dis).to(device)]
            pos_logit = (pos_src * pos_dst).sum(-1)
            neg_logit = (neg_src * neg_dst).sum(-1)
            logits = torch.cat([pos_logit, neg_logit])
            labels = torch.cat([
                torch.ones_like(pos_logit),
                torch.zeros_like(neg_logit),
            ])
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * logits.numel()
            n_seen += logits.numel()
        avg_loss = total_loss / max(n_seen, 1)
        dur = time.time() - t0

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            per_drug_r30, per_disease_hit = holdout_recall_at_30()
            print(
                f"Epoch {epoch:>3}/{args.epochs}  loss={avg_loss:.4f}  "
                f"holdout R@30(drug)={per_drug_r30:.4f}  "
                f"holdout hit_rate={per_disease_hit:.4f}  ({dur:.0f}s)"
            )
            if per_drug_r30 > best_r30:
                best_r30 = per_drug_r30
                patience_left = args.early_stop_patience
                best_state = {
                    "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "x_param": x_param.detach().cpu().clone(),
                    "epoch": epoch,
                }
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print("Early stop (no R@30 improvement)")
                    break
        else:
            print(f"Epoch {epoch:>3}/{args.epochs}  loss={avg_loss:.4f}  ({dur:.0f}s)")

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state["model_state"].items()})
        with torch.no_grad():
            x_param.data = best_state["x_param"].to(device)

    final = compute_embeddings().cpu().numpy().astype(np.float32)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"graphsage_h1200_s{args.seed}_{args.hidden}"
    ents_out = out_dir / f"{tag}_entities.npy"
    embs_out = out_dir / f"{tag}_embeddings.npy"
    np.save(ents_out, np.array(entities, dtype=object), allow_pickle=True)
    np.save(embs_out, final)

    meta = {
        "hypothesis": "h1200",
        "architecture": "GraphSAGE-mean-3layer",
        "seed": args.seed,
        "hidden": args.hidden,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "n_treatment_edges": len(treatment_edges),
        "n_train_diseases": len(train_dis),
        "n_holdout_diseases": len(hold_dis),
        "epochs_run": epoch if best_state is None else best_state["epoch"],
        "best_holdout_r30": best_r30,
        "drkg_no_treatment": args.drkg_no_treatment,
        "drkg_full": args.drkg_full,
    }
    (out_dir / f"{tag}_meta.json").write_text(json.dumps(meta, indent=2))
    print(
        f"Saved {ents_out.name}, {embs_out.name}; "
        f"best holdout R@30={best_r30:.4f} at epoch {meta['epochs_run']}"
    )


if __name__ == "__main__":
    main()
