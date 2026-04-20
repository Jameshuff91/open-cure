#!/usr/bin/env python3
"""h1201 Phase E — score drug-disease reversal via LINCS × CREEDS.

For each (drug, disease) where drug has a LINCS signature and disease has a
CREEDS signature, compute reversal_score = -cosine(drug_sig, disease_sig).

Disease signature construction: build a sparse vector over LINCS gene space;
for each CREEDS up-gene set +1.0, down-gene set -1.0 (both weighted by
per-gene score if present). Multiple CREEDS signatures per disease → mean.

Evaluate per-drug R@30 on held-out DRKG treatment edges, matching h1199's
5-seed disease-level 80/20 split convention.

Outputs:
  data/analysis/h1201_lincs_creeds_benchmark.json  — full metric tables
  data/analysis/h1201_lincs_creeds_benchmark.md    — markdown summary
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMB_DIR = PROJECT_ROOT / "data" / "embeddings"
OUT_DIR = PROJECT_ROOT / "data" / "analysis"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="42,123,456,789,2024",
                   help="Comma-separated random seeds for disease holdout")
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--holdout-pct", type=float, default=0.20)
    p.add_argument("--human-only", action="store_true",
                   help="Keep only human CREEDS signatures")
    return p.parse_args()


def load_gene_symbol_to_entrez() -> Dict[str, str]:
    """Build symbol→entrez map from LINCS gene_info."""
    import gzip
    path = PROJECT_ROOT / "data" / "external" / "lincs" / "GSE92742_Broad_LINCS_gene_info.txt.gz"
    sym_to_entrez: Dict[str, str] = {}
    with gzip.open(path, "rt") as f:
        header = next(f).rstrip("\n").split("\t")
        eid = header.index("pr_gene_id")
        sym = header.index("pr_gene_symbol")
        for line in f:
            p = line.rstrip("\n").split("\t")
            s = p[sym].upper().strip()
            if s and s != "-666":
                sym_to_entrez[s] = p[eid]
    return sym_to_entrez


def build_disease_signatures(
    creeds_path: Path,
    do_to_mesh: Dict[str, str],
    drkg_mesh_to_id: Dict[str, str],
    gene_idx: Dict[str, int],
    sym_to_entrez: Dict[str, str],
    n_genes: int,
    human_only: bool,
) -> Dict[str, np.ndarray]:
    """Return {drkg_disease_id: signature_vector_length_n_genes}."""
    sigs = json.load(open(creeds_path))
    buckets: Dict[str, List[np.ndarray]] = defaultdict(list)
    n_kept = 0
    for s in sigs:
        if human_only and s.get("organism") != "human":
            continue
        do_id = s.get("do_id")
        if not do_id or do_id not in do_to_mesh:
            continue
        mesh = do_to_mesh[do_id]
        if mesh not in drkg_mesh_to_id:
            continue
        drkg = drkg_mesh_to_id[mesh]
        vec = np.zeros(n_genes, dtype=np.float32)
        for sym, score in s.get("up_genes", []):
            entrez = sym_to_entrez.get(sym.upper())
            if entrez and entrez in gene_idx:
                vec[gene_idx[entrez]] = float(score) if score else 1.0
        for sym, score in s.get("down_genes", []):
            entrez = sym_to_entrez.get(sym.upper())
            if entrez and entrez in gene_idx:
                vec[gene_idx[entrez]] = -(abs(float(score))) if score else -1.0
        nz = int((vec != 0).sum())
        if nz < 5:
            continue
        buckets[drkg].append(vec)
        n_kept += 1
    print(f"  CREEDS sigs kept: {n_kept}")
    out: Dict[str, np.ndarray] = {}
    for drkg, vecs in buckets.items():
        out[drkg] = np.mean(np.stack(vecs, axis=0), axis=0)
    return out


def load_drkg_treatment_edges() -> Tuple[Dict[str, Set[str]], Dict[str, str]]:
    dis_to_drugs: Dict[str, Set[str]] = {}
    mesh_to_dis: Dict[str, str] = {}
    with open(PROJECT_ROOT / "data" / "raw" / "drkg" / "drkg.tsv") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 3 or p[1] != "DRUGBANK::treats::Compound:Disease":
                continue
            dis = p[2]
            drug = p[0].split("::")[-1]
            dis_to_drugs.setdefault(dis, set()).add(drug)
            if "MESH:" in dis:
                mesh_to_dis[dis.split("MESH:")[-1]] = dis
    return dis_to_drugs, mesh_to_dis


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between a (n_a, d) and b (n_b, d)."""
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return an @ bn.T


def main() -> None:
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    print("Loading LINCS signatures ...")
    lincs_sigs = np.load(EMB_DIR / "lincs_signatures.npy")
    lincs_drugs = np.load(EMB_DIR / "lincs_drug_ids.npy", allow_pickle=True).astype(str)
    lincs_genes = np.load(EMB_DIR / "lincs_gene_ids.npy", allow_pickle=True).astype(str)
    gene_idx = {g: i for i, g in enumerate(lincs_genes)}
    drug_idx = {d: i for i, d in enumerate(lincs_drugs)}
    print(f"  {len(lincs_drugs):,} drugs × {len(lincs_genes):,} genes")

    print("Building symbol→entrez map ...")
    sym_to_entrez = load_gene_symbol_to_entrez()
    print(f"  {len(sym_to_entrez):,} gene symbols")

    print("Loading DO→MESH + DRKG treatment edges ...")
    do_to_mesh = json.load(open(PROJECT_ROOT / "data" / "reference" / "doid_to_mesh_mapping.json"))
    dis_to_drugs, mesh_to_dis = load_drkg_treatment_edges()

    print("Building disease signatures from CREEDS ...")
    creeds_path = PROJECT_ROOT / "data" / "external" / "creeds" / "disease_signatures-v1.0.json"
    disease_sigs = build_disease_signatures(
        creeds_path, do_to_mesh, mesh_to_dis, gene_idx, sym_to_entrez,
        len(lincs_genes), args.human_only,
    )
    print(f"  {len(disease_sigs)} diseases with signatures")

    evaluable: Dict[str, Set[str]] = {}
    for dis, drugs in dis_to_drugs.items():
        if dis not in disease_sigs:
            continue
        covered = drugs & set(lincs_drugs)
        if covered:
            evaluable[dis] = covered
    print(f"  {len(evaluable)} diseases with both CREEDS sig AND ≥1 LINCS-covered drug")
    print(f"  {sum(len(v) for v in evaluable.values())} treatment edges evaluable")

    dis_list = sorted(evaluable.keys())
    n_dis = len(dis_list)
    n_drugs = len(lincs_drugs)

    disease_mat = np.stack([disease_sigs[d] for d in dis_list], axis=0)
    sim = cosine_matrix(disease_mat, lincs_sigs)
    reversal = -sim

    per_seed_metrics: List[Dict[str, float]] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_dis)
        n_hold = int(n_dis * args.holdout_pct)
        hold_idx = perm[:n_hold]

        r30_per_drug_vals = []
        hits_at = {1: [], 5: [], 10: [], 30: [], 100: []}
        mrr_per_triple = []
        ap_per_triple = []
        auroc_per_triple = []

        for dis_i in hold_idx:
            dis = dis_list[dis_i]
            gt = evaluable[dis]
            scores = reversal[dis_i]
            order = np.argsort(-scores)
            ranked_drugs = [lincs_drugs[o] for o in order]
            ranked_is_gt = np.asarray([d in gt for d in ranked_drugs])

            top_k_hits = int(ranked_is_gt[: args.k].sum())
            r30_per_drug_vals.append(top_k_hits / max(len(gt), 1))
            for K in hits_at:
                hits_at[K].append(int(ranked_is_gt[:K].sum()))

            labels = ranked_is_gt.astype(int)
            if labels.sum() > 0 and labels.sum() < len(labels):
                ap_per_triple.append(
                    average_precision_score(labels, scores[order])
                )
                auroc_per_triple.append(
                    roc_auc_score(labels, scores[order])
                )
            first_hit = np.argmax(ranked_is_gt) if ranked_is_gt.any() else -1
            mrr_per_triple.append(1.0 / (first_hit + 1) if first_hit >= 0 else 0.0)

        m = {
            "seed": seed,
            "n_holdout": n_hold,
            "R@30_per_drug_mean": float(np.mean(r30_per_drug_vals)),
            "R@30_per_drug_std": float(np.std(r30_per_drug_vals)),
            "Hits@1_per_drug": float(np.mean([h / max(len(evaluable[dis_list[i]]), 1) for h, i in zip(hits_at[1], hold_idx)])),
            "Hits@5_per_drug": float(np.mean([h / max(len(evaluable[dis_list[i]]), 1) for h, i in zip(hits_at[5], hold_idx)])),
            "Hits@10_per_drug": float(np.mean([h / max(len(evaluable[dis_list[i]]), 1) for h, i in zip(hits_at[10], hold_idx)])),
            "Hits@30_per_drug": float(np.mean([h / max(len(evaluable[dis_list[i]]), 1) for h, i in zip(hits_at[30], hold_idx)])),
            "Hits@100_per_drug": float(np.mean([h / max(len(evaluable[dis_list[i]]), 1) for h, i in zip(hits_at[100], hold_idx)])),
            "MRR": float(np.mean(mrr_per_triple)),
            "AUPRC": float(np.mean(ap_per_triple)) if ap_per_triple else 0.0,
            "AUROC": float(np.mean(auroc_per_triple)) if auroc_per_triple else 0.0,
        }
        per_seed_metrics.append(m)
        print(f"  seed={seed}  n_hold={n_hold}  R@30={m['R@30_per_drug_mean']:.4f}  MRR={m['MRR']:.4f}  AUPRC={m['AUPRC']:.4f}  AUROC={m['AUROC']:.4f}")

    agg = {}
    for key in ["R@30_per_drug_mean", "MRR", "AUPRC", "AUROC",
                "Hits@1_per_drug", "Hits@5_per_drug", "Hits@10_per_drug",
                "Hits@30_per_drug", "Hits@100_per_drug"]:
        vals = [m[key] for m in per_seed_metrics]
        agg[key + "_mean"] = float(np.mean(vals))
        agg[key + "_std"] = float(np.std(vals))

    out_path_json = OUT_DIR / "h1201_lincs_creeds_benchmark.json"
    out_path_md = OUT_DIR / "h1201_lincs_creeds_benchmark.md"
    result = {
        "hypothesis": "h1201",
        "phase": "E (standalone LINCS×CREEDS reversal)",
        "n_drugs": int(n_drugs),
        "n_diseases_evaluable": int(n_dis),
        "n_treatment_edges_evaluable": int(sum(len(v) for v in evaluable.values())),
        "seeds": seeds,
        "holdout_pct": args.holdout_pct,
        "human_only": args.human_only,
        "per_seed": per_seed_metrics,
        "aggregate": agg,
    }
    with open(out_path_json, "w") as f:
        json.dump(result, f, indent=2)

    md = [
        "# h1201 Phase E — LINCS × CREEDS reversal-connectivity standalone\n",
        f"- Pool: **{n_dis} diseases / {sum(len(v) for v in evaluable.values())} treatment edges** (LINCS drug sig ∧ CREEDS disease sig ∧ DRKG treatment edge)",
        f"- LINCS drugs candidate pool: {n_drugs:,}",
        f"- Seeds: {seeds}  |  holdout disease pct: {args.holdout_pct}  |  human_only: {args.human_only}\n",
        "## Headline\n",
        "| Metric | Mean | Std |",
        "|---|---|---|",
        f"| R@30 per-drug | {agg['R@30_per_drug_mean_mean']*100:.2f}% | ±{agg['R@30_per_drug_mean_std']*100:.2f}% |",
        f"| MRR | {agg['MRR_mean']:.4f} | ±{agg['MRR_std']:.4f} |",
        f"| AUPRC | {agg['AUPRC_mean']:.4f} | ±{agg['AUPRC_std']:.4f} |",
        f"| AUROC | {agg['AUROC_mean']:.4f} | ±{agg['AUROC_std']:.4f} |",
        "\n## Hits@K per-drug\n",
        "| K | mean | std |",
        "|---|---|---|",
        f"| 1 | {agg['Hits@1_per_drug_mean']*100:.2f}% | ±{agg['Hits@1_per_drug_std']*100:.2f}% |",
        f"| 5 | {agg['Hits@5_per_drug_mean']*100:.2f}% | ±{agg['Hits@5_per_drug_std']*100:.2f}% |",
        f"| 10 | {agg['Hits@10_per_drug_mean']*100:.2f}% | ±{agg['Hits@10_per_drug_std']*100:.2f}% |",
        f"| 30 | {agg['Hits@30_per_drug_mean']*100:.2f}% | ±{agg['Hits@30_per_drug_std']*100:.2f}% |",
        f"| 100 | {agg['Hits@100_per_drug_mean']*100:.2f}% | ±{agg['Hits@100_per_drug_std']*100:.2f}% |",
        "\n## Comparison baselines\n",
        "| Approach | R@30 (on full 1011-disease pool, not this subset) |",
        "|---|---|",
        "| Node2Vec 256 (h1199) | 19.55% ± 1.18% |",
        "| h1215 ensemble (best DRKG) | 20.87% ± 0.91% |",
        "| h1200 supervised GNN (invalidated) | 11.47% (best variant) |",
        "\n**Ship gate:** ≥15% R@30 standalone on this subset → expand coverage. <15% → close h1201 inconclusive.",
    ]
    out_path_md.write_text("\n".join(md))
    print(f"\nWrote {out_path_json}")
    print(f"Wrote {out_path_md}")


if __name__ == "__main__":
    main()
