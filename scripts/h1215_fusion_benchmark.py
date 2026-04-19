#!/usr/bin/env python3
"""
h1215: FastRP + Node2Vec fusion benchmark.

Extends the h1199 clean embedding benchmark with three fusion modes for
combining two embeddings (Node2Vec + FastRP):

    concat_l2   — L2-normalise each half then concatenate (512-dim joint
                  space; cosine similarity on the joint vectors).
    sim_mean    — Compute cosine similarity independently in each
                  embedding, average, then run top-k neighbour aggregation
                  on the averaged similarity matrix.
    score_mean  — Run each embedding's kNN score aggregation independently,
                  z-normalise the resulting drug-score vectors per disease
                  per embedding, then average element-wise.

All modes are evaluated on the same disease universe as the Node2Vec
baseline (the 1,011-disease intersection of both embeddings' entity
vocabularies with knn_gt). Reports identical metrics to h1199: per-drug
R@30, Hits@K (per-drug + per-triple), MRR, AUPRC, AUROC.

Baseline anchor (from h1212):
    Node2Vec: R@30 19.55%±1.18%  MRR 0.0284  AUPRC 0.0569  AUROC 0.5766
    FastRP:   R@30 18.79%±0.92%  MRR 0.0267  AUPRC 0.0584  AUROC 0.5790
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

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

FUSION_MODES = ("concat_l2", "sim_mean", "score_mean", "node2vec", "fastrp")


def l2_normalise(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def build_concat_lookup(lu_a: Dict[str, np.ndarray], lu_b: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Return L2-concat lookup keyed by entities present in both."""
    keys = sorted(set(lu_a) & set(lu_b))
    mat_a = np.stack([lu_a[k] for k in keys])
    mat_b = np.stack([lu_b[k] for k in keys])
    mat_a = l2_normalise(mat_a)
    mat_b = l2_normalise(mat_b)
    concat = np.concatenate([mat_a, mat_b], axis=1).astype(np.float32)
    return {k: concat[i] for i, k in enumerate(keys)}


def score_disease_fusion(
    disease_id: str,
    train_ids_ordered: List[str],
    train_gt: Dict[str, Set[str]],
    lu_a: Dict[str, np.ndarray],
    lu_b: Dict[str, np.ndarray],
    k: int,
    mode: str,
) -> Dict[str, float]:
    """Aggregate drug scores for one disease under the chosen fusion mode."""
    emb_a = lu_a[disease_id]
    emb_b = lu_b[disease_id]
    train_emb_a = np.stack([lu_a[d] for d in train_ids_ordered])
    train_emb_b = np.stack([lu_b[d] for d in train_ids_ordered])

    if mode == "sim_mean":
        sims_a = cosine_similarity(emb_a[None, :], train_emb_a)[0]
        sims_b = cosine_similarity(emb_b[None, :], train_emb_b)[0]
        sims = 0.5 * (sims_a + sims_b)
        if k >= len(sims):
            top_idx = np.argsort(sims)
        else:
            top_idx = np.argpartition(sims, -k)[-k:]
        drug_scores: Dict[str, float] = defaultdict(float)
        for idx in top_idx:
            neighbour = train_ids_ordered[idx]
            for drug_id in train_gt.get(neighbour, ()):
                drug_scores[drug_id] += float(sims[idx])
        return dict(drug_scores)

    if mode == "score_mean":
        # Independent top-k per embedding, z-norm the resulting per-drug
        # scores, then average. Universe = union of drugs scored by either.
        def score_one(emb, train_emb) -> Dict[str, float]:
            sims = cosine_similarity(emb[None, :], train_emb)[0]
            if k >= len(sims):
                top_idx = np.argsort(sims)
            else:
                top_idx = np.argpartition(sims, -k)[-k:]
            ds: Dict[str, float] = defaultdict(float)
            for idx in top_idx:
                neighbour = train_ids_ordered[idx]
                for drug_id in train_gt.get(neighbour, ()):
                    ds[drug_id] += float(sims[idx])
            return dict(ds)

        ds_a = score_one(emb_a, train_emb_a)
        ds_b = score_one(emb_b, train_emb_b)
        drugs = set(ds_a) | set(ds_b)

        def z(vals: np.ndarray) -> np.ndarray:
            mu = float(vals.mean())
            sd = float(vals.std())
            return (vals - mu) / (sd if sd > 1e-9 else 1.0)

        if not drugs:
            return {}
        arr_a = np.array([ds_a.get(d, 0.0) for d in drugs], dtype=np.float32)
        arr_b = np.array([ds_b.get(d, 0.0) for d in drugs], dtype=np.float32)
        za = z(arr_a)
        zb = z(arr_b)
        combined = 0.5 * (za + zb)
        return {d: float(combined[i]) for i, d in enumerate(drugs)}

    raise ValueError(f"unsupported fusion mode: {mode}")


def score_disease_single(
    disease_emb: np.ndarray,
    train_emb: np.ndarray,
    train_ids_ordered: List[str],
    train_gt: Dict[str, Set[str]],
    k: int,
) -> Dict[str, float]:
    sims = cosine_similarity(disease_emb[None, :], train_emb)[0]
    if k >= len(sims):
        top_idx = np.argsort(sims)
    else:
        top_idx = np.argpartition(sims, -k)[-k:]
    drug_scores: Dict[str, float] = defaultdict(float)
    for idx in top_idx:
        neighbour = train_ids_ordered[idx]
        for drug_id in train_gt.get(neighbour, ()):
            drug_scores[drug_id] += float(sims[idx])
    return dict(drug_scores)


def compute_metrics_for_seed(
    holdout_ids: List[str],
    train_ids: List[str],
    lu_a: Dict[str, np.ndarray],
    lu_b: Dict[str, np.ndarray],
    lu_concat: Dict[str, np.ndarray],
    gt: Dict[str, Set[str]],
    train_gt: Dict[str, Set[str]],
    candidate_drugs: Set[str],
    k: int,
    mode: str,
    disease_names: Dict[str, str],
) -> Dict:
    """Compute h1199-identical metrics for one seed under `mode`."""

    # Choose the embedding lookup used to identify train-disease ordering
    # and the candidate-universe filter (drugs embeddable under the primary
    # lookup). For the fused modes we use lu_concat's coverage because a
    # disease/drug needs to be in BOTH single embeddings.
    lookup_for_universe = lu_concat if mode in ("concat_l2", "sim_mean", "score_mean") else (
        lu_a if mode == "node2vec" else lu_b
    )
    train_ids_ordered = [d for d in train_ids if d in lookup_for_universe]

    # Precompute train matrices where needed
    train_emb_concat = (
        np.stack([lu_concat[d] for d in train_ids_ordered]) if mode == "concat_l2" else None
    )
    train_emb_a = (
        np.stack([lu_a[d] for d in train_ids_ordered]) if mode == "node2vec" else None
    )
    train_emb_b = (
        np.stack([lu_b[d] for d in train_ids_ordered]) if mode == "fastrp" else None
    )
    # sim_mean and score_mean rebuild train matrices inside score_disease_fusion

    per_drug_r30: List[float] = []
    hits_drug: Dict[int, List[float]] = {K: [] for K in (1, 5, 10, 30, 100)}
    hits_triple: Dict[int, int] = {K: 0 for K in (1, 5, 10, 30, 100)}
    reciprocal_ranks: List[float] = []
    all_scores: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    per_cat_r30: Dict[str, List[float]] = defaultdict(list)

    n_triples = 0
    n_test_diseases = 0

    all_holdout_gt_drugs: Set[str] = set()
    for did in holdout_ids:
        all_holdout_gt_drugs |= gt.get(did, set())
    universe = candidate_drugs | all_holdout_gt_drugs
    universe = {d for d in universe if d in lookup_for_universe}
    cand_list = sorted(universe)
    cand_index = {d: i for i, d in enumerate(cand_list)}

    for disease_id in holdout_ids:
        if disease_id not in lookup_for_universe:
            continue
        gt_drugs = gt.get(disease_id, set())
        gt_drugs = gt_drugs & universe
        if not gt_drugs:
            continue

        n_test_diseases += 1
        cat = categorize(disease_names.get(disease_id, disease_id))

        if mode == "concat_l2":
            assert train_emb_concat is not None
            drug_scores = score_disease_single(
                lu_concat[disease_id], train_emb_concat, train_ids_ordered, train_gt, k
            )
        elif mode == "node2vec":
            assert train_emb_a is not None
            drug_scores = score_disease_single(
                lu_a[disease_id], train_emb_a, train_ids_ordered, train_gt, k
            )
        elif mode == "fastrp":
            assert train_emb_b is not None
            drug_scores = score_disease_single(
                lu_b[disease_id], train_emb_b, train_ids_ordered, train_gt, k
            )
        else:
            drug_scores = score_disease_fusion(
                disease_id=disease_id,
                train_ids_ordered=train_ids_ordered,
                train_gt=train_gt,
                lu_a=lu_a,
                lu_b=lu_b,
                k=k,
                mode=mode,
            )

        score_vec = np.zeros(len(cand_list), dtype=np.float32)
        for drug, sc in drug_scores.items():
            if drug in cand_index:
                score_vec[cand_index[drug]] = sc

        order = np.argsort(-score_vec, kind="stable")
        rank_of_drug: Dict[str, int] = {}
        for r, idx in enumerate(order, start=1):
            rank_of_drug[cand_list[idx]] = r

        gt_ranks = [rank_of_drug[d] for d in gt_drugs if d in rank_of_drug]
        hit_30 = sum(1 for r in gt_ranks if r <= 30)
        per_drug_r30.append(hit_30 / len(gt_drugs))
        per_cat_r30[cat].append(hit_30 / len(gt_drugs))
        for K in hits_drug:
            hit_k = sum(1 for r in gt_ranks if r <= K)
            hits_drug[K].append(hit_k / len(gt_drugs))

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
        "mode": mode,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix-a", default="node2vec_256")
    ap.add_argument("--prefix-b", default="fastrp_256")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--seeds", type=str, default="42,123,456,789,2024")
    ap.add_argument(
        "--modes",
        type=str,
        default="node2vec,fastrp,concat_l2,sim_mean,score_mean",
        help="comma-separated fusion modes",
    )
    ap.add_argument(
        "--eval-gt",
        default="data/reference/expanded_ground_truth.json",
    )
    ap.add_argument(
        "--knn-gt",
        default="data/cache/ground_truth_cache.json",
    )
    ap.add_argument("--output", default="data/analysis/h1215_fusion_benchmark.json")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        if m not in FUSION_MODES:
            raise SystemExit(f"unknown mode: {m} (choose from {FUSION_MODES})")

    print("=" * 72)
    print(f"h1215: Fusion benchmark — {args.prefix_a} + {args.prefix_b}")
    print(f"       modes={modes} k={args.k} seeds={seeds}")
    print("=" * 72)

    t0 = time.time()
    print("Loading embeddings A...")
    lu_a, _, _ = load_embeddings(args.prefix_a)
    print(f"  {len(lu_a):,} entities")
    print("Loading embeddings B...")
    lu_b, _, _ = load_embeddings(args.prefix_b)
    print(f"  {len(lu_b):,} entities")

    print("Building concat_l2 lookup...")
    lu_concat = build_concat_lookup(lu_a, lu_b)
    print(f"  {len(lu_concat):,} entities in intersection")

    with open(PROJECT_ROOT / args.eval_gt) as f:
        raw_eval_gt = json.load(f)
    eval_gt = {d: set(drugs) for d, drugs in raw_eval_gt.items()}
    print(f"eval_gt: {len(eval_gt):,} diseases, {sum(len(v) for v in eval_gt.values()):,} pairs")

    with open(PROJECT_ROOT / args.knn_gt) as f:
        raw_knn_gt = json.load(f)
    if isinstance(raw_knn_gt, dict) and "ground_truth" in raw_knn_gt:
        raw_knn_gt = raw_knn_gt["ground_truth"]
    knn_gt = {d: set(drugs) for d, drugs in raw_knn_gt.items()}
    print(f"knn_gt: {len(knn_gt):,} diseases")

    disease_names = load_disease_names()

    # Universe: diseases in knn_gt present in BOTH embeddings (concat coverage)
    all_diseases = [d for d in knn_gt if d in lu_concat]
    print(f"Disease universe (knn_gt ∩ A ∩ B): {len(all_diseases):,}")
    print(f"Setup complete in {time.time()-t0:.1f}s")

    per_mode_results: Dict[str, List[Dict]] = {m: [] for m in modes}

    for si, seed in enumerate(seeds):
        print(f"\n=== SEED {seed} ({si+1}/{len(seeds)}) ===")
        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        print(f"Train: {len(train_ids)}  Holdout: {len(holdout_ids)}")

        train_knn_gt = {
            d: (knn_gt[d] & set(lu_concat.keys())) for d in train_ids if d in knn_gt
        }
        candidate_drugs: Set[str] = set()
        for drugs in train_knn_gt.values():
            candidate_drugs |= drugs
        print(f"Candidate drug pool (train knn_gt): {len(candidate_drugs):,}")

        for mode in modes:
            tm0 = time.time()
            metrics = compute_metrics_for_seed(
                holdout_ids=holdout_ids,
                train_ids=train_ids,
                lu_a=lu_a,
                lu_b=lu_b,
                lu_concat=lu_concat,
                gt=eval_gt,
                train_gt=train_knn_gt,
                candidate_drugs=candidate_drugs,
                k=args.k,
                mode=mode,
                disease_names=disease_names,
            )
            metrics["seed"] = seed
            metrics["elapsed_s"] = round(time.time() - tm0, 1)
            per_mode_results[mode].append(metrics)
            print(
                f"  [{mode:12s}] R@30={metrics['per_drug_r30']*100:.2f}%  "
                f"MRR={metrics['mrr_triple']:.4f}  "
                f"AUPRC={metrics['auprc']:.4f}  "
                f"AUROC={metrics['auroc']:.4f}  "
                f"({metrics['elapsed_s']}s)"
            )

    # Aggregate
    print("\n" + "=" * 72)
    print("AGGREGATE (mean ± std across seeds)")
    print("=" * 72)
    agg: Dict[str, Dict] = {}
    for mode in modes:
        rows = per_mode_results[mode]
        r30 = mean_std([r["per_drug_r30"] for r in rows])
        mrr = mean_std([r["mrr_triple"] for r in rows])
        auprc = mean_std([r["auprc"] for r in rows])
        auroc = mean_std([r["auroc"] for r in rows])
        hits10_drug = mean_std([r["hits_at_k_drug"][10] for r in rows])
        hits30_drug = mean_std([r["hits_at_k_drug"][30] for r in rows])
        agg[mode] = {
            "r30": r30,
            "mrr": mrr,
            "auprc": auprc,
            "auroc": auroc,
            "hits10_drug": hits10_drug,
            "hits30_drug": hits30_drug,
        }
        print(
            f"  {mode:12s}  R@30={r30[0]*100:5.2f}%±{r30[1]*100:.2f}%  "
            f"MRR={mrr[0]:.4f}±{mrr[1]:.4f}  "
            f"AUPRC={auprc[0]:.4f}±{auprc[1]:.4f}  "
            f"AUROC={auroc[0]:.4f}±{auroc[1]:.4f}"
        )

    # Write JSON report
    out_json = PROJECT_ROOT / args.output
    out_json.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "hypothesis": "h1215",
        "prefix_a": args.prefix_a,
        "prefix_b": args.prefix_b,
        "k": args.k,
        "seeds": seeds,
        "n_diseases_eligible": len(all_diseases),
        "modes": modes,
        "per_seed_per_mode": per_mode_results,
        "aggregate": agg,
    }
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out_json}")

    # Write markdown summary
    out_md = out_json.with_suffix(".md")
    lines = []
    lines.append(f"# h1215: FastRP + Node2Vec Fusion Benchmark")
    lines.append("")
    lines.append(f"- Embeddings: `{args.prefix_a}` + `{args.prefix_b}`")
    lines.append(f"- k = {args.k}, seeds = {seeds}")
    lines.append(f"- Eligible diseases (∩): **{len(all_diseases):,}**")
    lines.append("")
    lines.append("## Aggregate (mean ± std)")
    lines.append("")
    lines.append("| Mode | R@30 | MRR | AUPRC | AUROC | Hits@10 drug | Hits@30 drug |")
    lines.append("|---|---|---|---|---|---|---|")
    for mode in modes:
        a = agg[mode]
        lines.append(
            f"| `{mode}` | "
            f"{a['r30'][0]*100:.2f}%±{a['r30'][1]*100:.2f}% | "
            f"{a['mrr'][0]:.4f}±{a['mrr'][1]:.4f} | "
            f"{a['auprc'][0]:.4f}±{a['auprc'][1]:.4f} | "
            f"{a['auroc'][0]:.4f}±{a['auroc'][1]:.4f} | "
            f"{a['hits10_drug'][0]*100:.2f}%±{a['hits10_drug'][1]*100:.2f}% | "
            f"{a['hits30_drug'][0]*100:.2f}%±{a['hits30_drug'][1]*100:.2f}% |"
        )
    lines.append("")
    lines.append("## Per-seed detail")
    lines.append("")
    for mode in modes:
        lines.append(f"### `{mode}`")
        lines.append("")
        lines.append("| Seed | n_dis | n_triples | R@30 | MRR | AUPRC | AUROC |")
        lines.append("|---|---|---|---|---|---|---|")
        for m in per_mode_results[mode]:
            lines.append(
                f"| {m['seed']} | {m['n_test_diseases']} | {m['n_test_triples']} | "
                f"{m['per_drug_r30']*100:.2f}% | {m['mrr_triple']:.4f} | "
                f"{m['auprc']:.4f} | {m['auroc']:.4f} |"
            )
        lines.append("")
    with open(out_md, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
