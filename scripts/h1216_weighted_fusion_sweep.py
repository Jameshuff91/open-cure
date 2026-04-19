#!/usr/bin/env python3
"""h1216: Weighted sim_mean / concat_l2 fusion sweep.

Follow-up to h1215. Sweeps w in {0.1, ..., 0.9} for the fraction of
Node2Vec similarity in the fused kNN scoring, plus w=0 (pure FastRP)
and w=1 (pure Node2Vec) as sanity endpoints.

Mathematical identity used here: for pre-L2-normalised halves a, b,
    v_w(x) = [sqrt(w)·a(x); sqrt(1-w)·b(x)]   (has unit norm)
    cos(v_w(x), v_w(y)) = w·cos(a(x),a(y)) + (1-w)·cos(b(x),b(y))

So weighted concat with (sqrt(w), sqrt(1-w)) on L2-normalised halves is
numerically identical to weighted sim_mean with weights (w, 1-w). We
build one weighted-concat lookup per w and reuse h1215's
`score_disease_single` (cheaper than rebuilding per-disease matrices
inside a fusion loop).

h1215 anchor (equal-weight concat_l2, w=0.5):
    R@30 20.87% ± 0.91%  MRR 0.0296  AUPRC 0.0642  AUROC 0.5851
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from clean_embedding_benchmark import (  # type: ignore[import-not-found]  # noqa: E402
    categorize,
    load_disease_names,
    load_embeddings,
    mean_std,
    split_diseases,
)
from h1215_fusion_benchmark import l2_normalise, score_disease_single  # type: ignore  # noqa: E402


def build_weighted_concat_lookup(
    lu_a: Dict[str, np.ndarray],
    lu_b: Dict[str, np.ndarray],
    weight_a: float,
) -> Dict[str, np.ndarray]:
    """Weighted L2-concat so cosine = weight_a*cos_a + (1-weight_a)*cos_b."""
    assert 0.0 <= weight_a <= 1.0
    keys = sorted(set(lu_a) & set(lu_b))
    mat_a = l2_normalise(np.stack([lu_a[k] for k in keys]))
    mat_b = l2_normalise(np.stack([lu_b[k] for k in keys]))
    sa = math.sqrt(weight_a)
    sb = math.sqrt(1.0 - weight_a)
    concat = np.concatenate([sa * mat_a, sb * mat_b], axis=1).astype(np.float32)
    return {k: concat[i] for i, k in enumerate(keys)}


def compute_metrics(
    weight_a: float,
    seed: int,
    lu_joint: Dict[str, np.ndarray],
    all_diseases: List[str],
    knn_gt: Dict[str, Set[str]],
    eval_gt: Dict[str, Set[str]],
    disease_names: Dict[str, str],
    k: int,
) -> Dict:
    train_ids, holdout_ids = split_diseases(all_diseases, seed)
    train_ids_ordered = [d for d in train_ids if d in lu_joint]
    train_emb = np.stack([lu_joint[d] for d in train_ids_ordered])
    train_gt = {d: (knn_gt[d] & set(lu_joint.keys())) for d in train_ids if d in knn_gt}

    candidate_drugs: Set[str] = set()
    for drugs in train_gt.values():
        candidate_drugs |= drugs

    all_holdout_gt_drugs: Set[str] = set()
    for did in holdout_ids:
        all_holdout_gt_drugs |= eval_gt.get(did, set())
    universe = (candidate_drugs | all_holdout_gt_drugs) & set(lu_joint.keys())
    cand_list = sorted(universe)
    cand_index = {d: i for i, d in enumerate(cand_list)}

    per_drug_r30: List[float] = []
    hits_drug: Dict[int, List[float]] = {K: [] for K in (1, 5, 10, 30, 100)}
    hits_triple: Dict[int, int] = {K: 0 for K in (1, 5, 10, 30, 100)}
    reciprocal_ranks: List[float] = []
    all_scores: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    per_cat_r30: Dict[str, List[float]] = defaultdict(list)

    n_triples = 0
    n_test_diseases = 0

    for disease_id in holdout_ids:
        if disease_id not in lu_joint:
            continue
        gt_drugs = eval_gt.get(disease_id, set()) & universe
        if not gt_drugs:
            continue
        n_test_diseases += 1
        cat = categorize(disease_names.get(disease_id, disease_id))

        drug_scores = score_disease_single(
            lu_joint[disease_id], train_emb, train_ids_ordered, train_gt, k
        )

        score_vec = np.zeros(len(cand_list), dtype=np.float32)
        for drug, sc in drug_scores.items():
            if drug in cand_index:
                score_vec[cand_index[drug]] = sc

        order = np.argsort(-score_vec, kind="stable")
        rank_of_drug: Dict[str, int] = {cand_list[idx]: r for r, idx in enumerate(order, start=1)}

        gt_ranks = [rank_of_drug[d] for d in gt_drugs if d in rank_of_drug]
        per_drug_r30.append(sum(1 for r in gt_ranks if r <= 30) / len(gt_drugs))
        per_cat_r30[cat].append(per_drug_r30[-1])
        for K in hits_drug:
            hits_drug[K].append(sum(1 for r in gt_ranks if r <= K) / len(gt_drugs))

        for d in gt_drugs:
            r = rank_of_drug.get(d)
            n_triples += 1
            if r is None:
                reciprocal_ranks.append(0.0)
                continue
            reciprocal_ranks.append(1.0 / r)
            for K in hits_triple:
                if r <= K:
                    hits_triple[K] += 1

        label_vec = np.zeros(len(cand_list), dtype=np.int8)
        for d in gt_drugs:
            if d in cand_index:
                label_vec[cand_index[d]] = 1
        all_scores.append(score_vec)
        all_labels.append(label_vec)

    scores_flat = np.concatenate(all_scores) if all_scores else np.zeros(0)
    labels_flat = np.concatenate(all_labels) if all_labels else np.zeros(0)
    auprc = float(average_precision_score(labels_flat, scores_flat)) if labels_flat.sum() > 0 else 0.0
    auroc = (
        float(roc_auc_score(labels_flat, scores_flat))
        if 0 < labels_flat.sum() < len(labels_flat)
        else 0.0
    )

    return {
        "weight_a": weight_a,
        "seed": seed,
        "n_test_diseases": n_test_diseases,
        "n_test_triples": n_triples,
        "per_drug_r30": float(np.mean(per_drug_r30)) if per_drug_r30 else 0.0,
        "hits_at_k_drug": {K: float(np.mean(v)) if v else 0.0 for K, v in hits_drug.items()},
        "hits_at_k_triple": {K: v / n_triples if n_triples else 0.0 for K, v in hits_triple.items()},
        "mrr_triple": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "auprc": auprc,
        "auroc": auroc,
        "per_category": {
            cat: {
                "n_diseases": len(per_cat_r30[cat]),
                "r30": float(np.mean(per_cat_r30[cat])) if per_cat_r30[cat] else 0.0,
            }
            for cat in per_cat_r30
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix-a", default="node2vec_256")
    ap.add_argument("--prefix-b", default="fastrp_256")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--seeds", type=str, default="42,123,456,789,2024")
    ap.add_argument(
        "--weights",
        type=str,
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help="comma-separated weight_a values",
    )
    ap.add_argument("--eval-gt", default="data/reference/expanded_ground_truth.json")
    ap.add_argument("--knn-gt", default="data/cache/ground_truth_cache.json")
    ap.add_argument("--output", default="data/analysis/h1216_weighted_fusion_sweep.json")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    weights = [float(w) for w in args.weights.split(",")]

    print("=" * 72)
    print(f"h1216: Weighted fusion sweep — {args.prefix_a} (w) + {args.prefix_b} (1-w)")
    print(f"       weights={weights}  seeds={seeds}  k={args.k}")
    print("=" * 72)

    t0 = time.time()
    print("Loading embeddings A...")
    lu_a, _, _ = load_embeddings(args.prefix_a)
    print(f"  {len(lu_a):,} entities")
    print("Loading embeddings B...")
    lu_b, _, _ = load_embeddings(args.prefix_b)
    print(f"  {len(lu_b):,} entities")

    with open(PROJECT_ROOT / args.eval_gt) as f:
        eval_gt_raw = json.load(f)
    eval_gt = {d: set(drugs) for d, drugs in eval_gt_raw.items()}

    with open(PROJECT_ROOT / args.knn_gt) as f:
        knn_raw = json.load(f)
    if isinstance(knn_raw, dict) and "ground_truth" in knn_raw:
        knn_raw = knn_raw["ground_truth"]
    knn_gt = {d: set(drugs) for d, drugs in knn_raw.items()}

    disease_names = load_disease_names()

    # Use intersection entity set (same as h1215) for apples-to-apples
    # comparison against the h1215 equal-weight baseline.
    # Match h1215 ordering exactly (dict iteration order of knn_gt, filtered
    # by intersection) so seed-based splits reproduce h1215's w=0.5 anchor.
    intersection_keys = set(lu_a) & set(lu_b)
    all_diseases = [d for d in knn_gt if d in intersection_keys]
    print(f"Disease universe (knn_gt ∩ A ∩ B): {len(all_diseases):,}")
    print(f"Setup complete in {time.time()-t0:.1f}s")

    per_weight: Dict[float, List[Dict]] = {w: [] for w in weights}

    for wi, w in enumerate(weights):
        print(f"\n=== WEIGHT_A = {w} ({wi+1}/{len(weights)}) ===")
        tw = time.time()
        lu_joint = build_weighted_concat_lookup(lu_a, lu_b, w)
        print(f"  joint lookup built: {len(lu_joint):,} entities  ({time.time()-tw:.1f}s)")
        for seed in seeds:
            ts = time.time()
            m = compute_metrics(
                weight_a=w,
                seed=seed,
                lu_joint=lu_joint,
                all_diseases=all_diseases,
                knn_gt=knn_gt,
                eval_gt=eval_gt,
                disease_names=disease_names,
                k=args.k,
            )
            m["elapsed_s"] = round(time.time() - ts, 1)
            per_weight[w].append(m)
            print(
                f"  seed={seed:5d}  R@30={m['per_drug_r30']*100:5.2f}%  "
                f"MRR={m['mrr_triple']:.4f}  AUPRC={m['auprc']:.4f}  "
                f"AUROC={m['auroc']:.4f}  ({m['elapsed_s']}s)"
            )

    # Aggregate
    print("\n" + "=" * 72)
    print("AGGREGATE (mean ± std across seeds, per weight)")
    print("=" * 72)
    agg: Dict[str, Dict] = {}
    for w in weights:
        rows = per_weight[w]
        r30 = mean_std([r["per_drug_r30"] for r in rows])
        mrr = mean_std([r["mrr_triple"] for r in rows])
        auprc = mean_std([r["auprc"] for r in rows])
        auroc = mean_std([r["auroc"] for r in rows])
        hits10 = mean_std([r["hits_at_k_drug"][10] for r in rows])
        hits30 = mean_std([r["hits_at_k_drug"][30] for r in rows])
        agg[str(w)] = {
            "r30": r30,
            "mrr": mrr,
            "auprc": auprc,
            "auroc": auroc,
            "hits10_drug": hits10,
            "hits30_drug": hits30,
        }
        print(
            f"  w_a={w:4.2f}  R@30={r30[0]*100:5.2f}%±{r30[1]*100:.2f}%  "
            f"MRR={mrr[0]:.4f}±{mrr[1]:.4f}  AUPRC={auprc[0]:.4f}±{auprc[1]:.4f}  "
            f"AUROC={auroc[0]:.4f}±{auroc[1]:.4f}"
        )

    # Per-metric optima
    print("\n" + "=" * 72)
    print("PER-METRIC OPTIMA")
    print("=" * 72)
    optima: Dict[str, Dict] = {}
    for metric in ("r30", "mrr", "auprc", "auroc", "hits10_drug", "hits30_drug"):
        best_w = max(weights, key=lambda x: agg[str(x)][metric][0])
        optima[metric] = {"weight_a": best_w, "value": agg[str(best_w)][metric]}
        v, s = agg[str(best_w)][metric]
        pretty = f"{v*100:5.2f}%±{s*100:.2f}%" if metric not in ("mrr", "auprc", "auroc") else f"{v:.4f}±{s:.4f}"
        print(f"  best {metric:14s}  @ w_a={best_w:4.2f}  = {pretty}")

    # Vs h1215 equal-weight (w=0.5) reference
    w05 = agg.get("0.5")
    if w05:
        print("\nDelta vs equal-weight w=0.5 anchor:")
        for metric in ("r30", "mrr", "auprc", "auroc"):
            best_w = optima[metric]["weight_a"]
            delta = agg[str(best_w)][metric][0] - w05[metric][0]
            print(
                f"  {metric:14s}  best w={best_w:4.2f} − w=0.5  = "
                f"{delta*100:+.3f} (raw {delta:+.4f})"
            )

    out_json = PROJECT_ROOT / args.output
    out_json.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "hypothesis": "h1216",
        "prefix_a": args.prefix_a,
        "prefix_b": args.prefix_b,
        "k": args.k,
        "seeds": seeds,
        "weights": weights,
        "n_diseases_eligible": len(all_diseases),
        "per_weight_per_seed": {str(w): per_weight[w] for w in weights},
        "aggregate": agg,
        "optima": optima,
    }
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out_json}")

    # Markdown summary
    out_md = out_json.with_suffix(".md")
    lines = []
    lines.append("# h1216: Weighted Fusion Sweep — Node2Vec (w) + FastRP (1−w)")
    lines.append("")
    lines.append(f"- Embeddings: `{args.prefix_a}` (weight_a) + `{args.prefix_b}` (1−weight_a)")
    lines.append(f"- k = {args.k}, seeds = {seeds}")
    lines.append(f"- Eligible diseases (∩): **{len(all_diseases):,}**")
    lines.append("- Identity: weighted concat with scales (√w, √(1−w)) ≡ weighted sim_mean with weights (w, 1−w).")
    lines.append("")
    lines.append("## Aggregate (mean ± std)")
    lines.append("")
    lines.append("| weight_a | R@30 | MRR | AUPRC | AUROC | Hits@10 | Hits@30 |")
    lines.append("|---|---|---|---|---|---|---|")
    for w in weights:
        a = agg[str(w)]
        lines.append(
            f"| {w:.2f} | "
            f"{a['r30'][0]*100:.2f}%±{a['r30'][1]*100:.2f}% | "
            f"{a['mrr'][0]:.4f}±{a['mrr'][1]:.4f} | "
            f"{a['auprc'][0]:.4f}±{a['auprc'][1]:.4f} | "
            f"{a['auroc'][0]:.4f}±{a['auroc'][1]:.4f} | "
            f"{a['hits10_drug'][0]*100:.2f}%±{a['hits10_drug'][1]*100:.2f}% | "
            f"{a['hits30_drug'][0]*100:.2f}%±{a['hits30_drug'][1]*100:.2f}% |"
        )
    lines.append("")
    lines.append("## Per-metric optima")
    lines.append("")
    lines.append("| Metric | Best weight_a | Value | Δ vs w=0.5 |")
    lines.append("|---|---|---|---|")
    for metric in ("r30", "mrr", "auprc", "auroc", "hits10_drug", "hits30_drug"):
        best_w = optima[metric]["weight_a"]
        v, s = agg[str(best_w)][metric]
        if metric in ("mrr", "auprc", "auroc"):
            val = f"{v:.4f}±{s:.4f}"
        else:
            val = f"{v*100:.2f}%±{s*100:.2f}%"
        if w05:
            delta = v - w05[metric][0]
            if metric in ("mrr", "auprc", "auroc"):
                dstr = f"{delta:+.4f}"
            else:
                dstr = f"{delta*100:+.2f}pp"
        else:
            dstr = "—"
        lines.append(f"| {metric} | {best_w:.2f} | {val} | {dstr} |")
    lines.append("")
    with open(out_md, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
