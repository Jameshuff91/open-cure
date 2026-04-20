#!/usr/bin/env python3
"""h1295: Per-disease audit of h1287's W035 vs W050 ΔR@30 lift.

h1287's 20-seed fine-grid sweep on SUBSET_D_GLOBAL found W035 beats W050 by
ΔR@30 = +0.174pp at p=0.001 (20 paired seeds) but is flat on per-disease
AUPRC (+0.00021 p=0.512). h1287 was INVALIDATED on the retrieval-recipe
promotion gate vs W040 (+0.034pp p=0.181), but the signed R@30 difference
vs W050 (the current canonical) is real.

This script decomposes that +0.174pp ΔR@30 per (seed × disease) row, mirroring
h1272's decomposition of h1269's AUPRC lift. The intent is to locate the
categories/n_gt-buckets/diseases where W035 genuinely out-ranks W050, so
that h1292 (R@30-targeted category-gated w) can be seeded from real
category priors rather than noisy retrospective sorting.

Outputs:
    data/analysis/h1295_w035_per_disease_audit.{json,md}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from clean_embedding_benchmark import (  # noqa: E402
    categorize,
    load_disease_names,
    load_embeddings,
    split_diseases,
)
from h1215_fusion_benchmark import build_concat_lookup, score_disease_single  # noqa: E402
from h1255_soft_blend_fusion import z_normalise  # noqa: E402

OUT_JSON = PROJECT_ROOT / "data/analysis/h1295_w035_per_disease_audit.json"
OUT_MD = PROJECT_ROOT / "data/analysis/h1295_w035_per_disease_audit.md"

W_A = 0.35  # target
W_B = 0.50  # canonical reference


def n_gt_bucket(n: int) -> str:
    if n <= 1:
        return "1"
    if n <= 5:
        return "2-5"
    if n <= 20:
        return "6-20"
    if n <= 50:
        return "21-50"
    return "51+"


def recall_at_30(score_vec: np.ndarray, label_vec: np.ndarray, n_gt: int) -> float:
    if n_gt == 0:
        return 0.0
    order = np.argsort(-score_vec, kind="stable")
    topk = order[:30]
    hits = int(label_vec[topk].sum())
    return hits / n_gt


DEFAULT_SEEDS = "42,123,456,789,2024,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix-a", default="node2vec_256")
    ap.add_argument("--prefix-b", default="fastrp_256")
    ap.add_argument("--seeds", type=str, default=DEFAULT_SEEDS)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--eval-gt", default="data/reference/expanded_ground_truth.json")
    ap.add_argument("--knn-gt", default="data/cache/ground_truth_cache.json")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    print("=" * 72)
    print(f"h1295: Per-disease W{int(W_A*100):03d} vs W{int(W_B*100):03d} ΔR@30 audit")
    print(f"  prefix_a={args.prefix_a}  prefix_b={args.prefix_b}  seeds={len(seeds)}")
    print("=" * 72)

    t0 = time.time()
    lu_a, _, _ = load_embeddings(args.prefix_a)
    lu_b, _, _ = load_embeddings(args.prefix_b)
    lu_concat = build_concat_lookup(lu_a, lu_b)

    with open(PROJECT_ROOT / args.eval_gt) as f:
        eval_gt = {d: set(v) for d, v in json.load(f).items()}
    with open(PROJECT_ROOT / args.knn_gt) as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "ground_truth" in raw:
        raw = raw["ground_truth"]
    knn_gt = {d: set(v) for d, v in raw.items()}

    disease_names = load_disease_names()
    intersection_keys = set(lu_a) & set(lu_b)
    all_diseases = [d for d in knn_gt if d in intersection_keys]
    print(f"Setup: {len(all_diseases):,} diseases ({time.time()-t0:.1f}s)")

    rows: List[Dict] = []
    for seed in seeds:
        ts = time.time()
        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        train_ids_ordered = [d for d in train_ids if d in lu_concat and d in lu_a]
        train_emb_a = np.stack([lu_a[d] for d in train_ids_ordered])
        train_emb_c = np.stack([lu_concat[d] for d in train_ids_ordered])
        train_gt = {
            d: (knn_gt[d] & set(lu_concat.keys()))
            for d in train_ids_ordered if d in knn_gt
        }
        cand_drugs: Set[str] = set()
        for d in train_ids_ordered:
            if d in knn_gt:
                cand_drugs |= (knn_gt[d] & set(lu_concat.keys()))
        all_hold_drugs: Set[str] = set()
        for did in holdout_ids:
            all_hold_drugs |= eval_gt.get(did, set())
        universe = (cand_drugs | all_hold_drugs) & set(lu_concat.keys()) & set(lu_a.keys())
        cand_list = sorted(universe)
        cand_index = {d: i for i, d in enumerate(cand_list)}
        n_cands = len(cand_list)

        n_processed = 0
        for did in holdout_ids:
            if did not in lu_concat or did not in lu_a:
                continue
            gt_drugs = eval_gt.get(did, set()) & universe
            if not gt_drugs:
                continue

            ds_a = score_disease_single(lu_a[did], train_emb_a, train_ids_ordered, train_gt, args.k)
            sv_a = np.zeros(n_cands, dtype=np.float32)
            for drug, sc in ds_a.items():
                if drug in cand_index:
                    sv_a[cand_index[drug]] = sc
            ds_c = score_disease_single(lu_concat[did], train_emb_c, train_ids_ordered, train_gt, args.k)
            sv_c = np.zeros(n_cands, dtype=np.float32)
            for drug, sc in ds_c.items():
                if drug in cand_index:
                    sv_c[cand_index[drug]] = sc

            z_a = z_normalise(sv_a)
            z_c = z_normalise(sv_c)
            score_w035 = W_A * z_a + (1.0 - W_A) * z_c
            score_w050 = W_B * z_a + (1.0 - W_B) * z_c

            label_vec = np.zeros(n_cands, dtype=np.int8)
            for d in gt_drugs:
                idx = cand_index.get(d)
                if idx is not None:
                    label_vec[idx] = 1
            n_gt = int(label_vec.sum())
            if n_gt == 0:
                continue

            r30_w035 = recall_at_30(score_w035, label_vec, n_gt)
            r30_w050 = recall_at_30(score_w050, label_vec, n_gt)

            rows.append({
                "seed": seed,
                "disease_id": did,
                "name": disease_names.get(did, did),
                "category": categorize(disease_names.get(did, did)),
                "n_gt_eval": n_gt,
                "r30_w035": r30_w035,
                "r30_w050": r30_w050,
                "delta_r30": r30_w035 - r30_w050,
            })
            n_processed += 1
        print(f"  seed {seed}: {n_processed} diseases ({time.time()-ts:.1f}s)")

    # ===== Aggregate =====
    print(f"\nTotal (seed, disease) rows: {len(rows)}")
    deltas = np.array([r["delta_r30"] for r in rows])
    n_pos = int(np.sum(deltas > 0))
    n_neg = int(np.sum(deltas < 0))
    n_zero = int(np.sum(deltas == 0))
    print(f"Aggregate mean Δ R@30 = {deltas.mean()*100:+.4f}pp  std = {deltas.std()*100:.4f}pp")
    print(
        f"  positive: {n_pos} ({n_pos / len(rows) * 100:.1f}%)  "
        f"negative: {n_neg} ({n_neg / len(rows) * 100:.1f}%)  "
        f"zero: {n_zero} ({n_zero / len(rows) * 100:.1f}%)"
    )

    nonzero_rows = [r for r in rows if r["delta_r30"] != 0.0]
    nz = np.array([r["delta_r30"] for r in nonzero_rows])
    print(
        f"Non-trivial rows only (Δ≠0, n={len(nonzero_rows)}): "
        f"mean Δ R@30 = {nz.mean()*100:+.4f}pp"
    )

    # Per-n_gt bucket
    print(f"\n--- Per-n_gt bucket Δ R@30 ---")
    by_bucket: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_bucket[n_gt_bucket(r["n_gt_eval"])].append(r)
    bucket_summary: Dict[str, Dict] = {}
    for b in ["1", "2-5", "6-20", "21-50", "51+"]:
        rs = by_bucket.get(b, [])
        if not rs:
            continue
        arr = np.array([r["delta_r30"] for r in rs])
        nz_count = int(np.sum(arr != 0))
        bucket_summary[b] = {
            "n_rows": int(len(arr)),
            "mean_delta_r30": float(arr.mean()),
            "std_delta_r30": float(arr.std()),
            "frac_nonzero": nz_count / len(arr) if len(arr) else 0.0,
            "frac_positive": float(np.mean(arr > 0)),
            "frac_negative": float(np.mean(arr < 0)),
            "nonzero_mean_delta": float(np.mean(arr[arr != 0])) if nz_count else 0.0,
        }
        print(
            f"  n_gt={b:6s}: n={len(arr):4d}  mean_Δ={arr.mean()*100:+.4f}pp  "
            f"nonzero={nz_count:3d}  nz_mean_Δ={bucket_summary[b]['nonzero_mean_delta']*100:+.4f}pp  "
            f"frac_+={np.mean(arr > 0) * 100:.1f}%  frac_-={np.mean(arr < 0) * 100:.1f}%"
        )

    # Per-category
    print(f"\n--- Per-category Δ R@30 ---")
    by_cat: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    cat_summary: Dict[str, Dict] = {}
    cats_sorted = sorted(by_cat.keys(), key=lambda c: float(np.mean([r["delta_r30"] for r in by_cat[c]])), reverse=True)
    for c in cats_sorted:
        rs = by_cat[c]
        arr = np.array([r["delta_r30"] for r in rs])
        nz_count = int(np.sum(arr != 0))
        cat_summary[c] = {
            "n_rows": int(len(arr)),
            "mean_delta_r30": float(arr.mean()),
            "std_delta_r30": float(arr.std()),
            "frac_nonzero": nz_count / len(arr) if len(arr) else 0.0,
            "frac_positive": float(np.mean(arr > 0)),
            "frac_negative": float(np.mean(arr < 0)),
            "nonzero_mean_delta": float(np.mean(arr[arr != 0])) if nz_count else 0.0,
        }
        print(
            f"  {c:18s}: n={len(arr):4d}  mean_Δ={arr.mean()*100:+.4f}pp  "
            f"nonzero={nz_count:3d}  nz_mean_Δ={cat_summary[c]['nonzero_mean_delta']*100:+.4f}pp  "
            f"frac_+={np.mean(arr > 0) * 100:.1f}%"
        )

    # Cross-tab: category × n_gt bucket
    print(f"\n--- Cross-tab (category × n_gt bucket) mean Δ R@30 (pp) ---")
    ct: Dict[str, Dict[str, float]] = {}
    for c in cats_sorted:
        ct[c] = {}
        for b in ["1", "2-5", "6-20", "21-50", "51+"]:
            sub = [r["delta_r30"] for r in by_cat[c] if n_gt_bucket(r["n_gt_eval"]) == b]
            if sub:
                ct[c][b] = float(np.mean(sub) * 100)
    hdr = f"  {'Category':<18}  {'1':>8}  {'2-5':>8}  {'6-20':>8}  {'21-50':>8}  {'51+':>8}"
    print(hdr)
    for c in cats_sorted:
        parts = [f"  {c:<18}"]
        for b in ["1", "2-5", "6-20", "21-50", "51+"]:
            v = ct[c].get(b)
            parts.append(f"{v:+7.3f}" if v is not None else "     —  ")
        print("  ".join(parts))

    # Per-disease aggregate
    by_disease: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_disease[r["disease_id"]].append(r)
    disease_agg = []
    for did, rs in by_disease.items():
        ds = [r["delta_r30"] for r in rs]
        disease_agg.append({
            "disease_id": did,
            "name": rs[0]["name"],
            "category": rs[0]["category"],
            "n_gt_eval_mean": float(np.mean([r["n_gt_eval"] for r in rs])),
            "n_seeds_seen": len(ds),
            "mean_delta_r30": float(np.mean(ds)),
            "std_delta_r30": float(np.std(ds)),
            "min_delta_r30": float(np.min(ds)),
            "max_delta_r30": float(np.max(ds)),
        })
    disease_agg_sorted = sorted(disease_agg, key=lambda r: r["mean_delta_r30"], reverse=True)
    print(f"\n--- TOP 15 best diseases (largest mean Δ R@30) ---")
    for r in disease_agg_sorted[:15]:
        print(
            f"  {r['mean_delta_r30']*100:+.3f}pp  n_seeds={r['n_seeds_seen']}  "
            f"n_gt={r['n_gt_eval_mean']:5.1f}  [{r['category']:15s}] {r['name'][:60]}"
        )
    print(f"\n--- BOTTOM 15 worst diseases (most negative mean Δ R@30) ---")
    for r in disease_agg_sorted[-15:][::-1]:
        print(
            f"  {r['mean_delta_r30']*100:+.3f}pp  n_seeds={r['n_seeds_seen']}  "
            f"n_gt={r['n_gt_eval_mean']:5.1f}  [{r['category']:15s}] {r['name'][:60]}"
        )

    # Persist JSON
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({
            "hypothesis": "h1295",
            "w_target": W_A,
            "w_reference": W_B,
            "seeds": seeds,
            "n_rows": len(rows),
            "aggregate": {
                "mean_delta_r30": float(deltas.mean()),
                "std_delta_r30": float(deltas.std()),
                "min_delta_r30": float(deltas.min()),
                "max_delta_r30": float(deltas.max()),
                "n_positive": n_pos,
                "n_negative": n_neg,
                "n_zero": n_zero,
                "nonzero_mean_delta_r30": float(nz.mean()) if len(nz) else 0.0,
            },
            "by_n_gt_bucket": bucket_summary,
            "by_category": cat_summary,
            "crosstab_cat_x_ngt_pp": ct,
            "top15_diseases": disease_agg_sorted[:15],
            "bottom15_diseases": disease_agg_sorted[-15:][::-1],
        }, f, indent=2)
    print(f"\nWrote {OUT_JSON.relative_to(PROJECT_ROOT)}")

    # Markdown
    md: List[str] = []
    md.append("# h1295 — Per-disease W035 vs W050 ΔR@30 audit (20 seeds, SUBSET_D_GLOBAL)\n\n")
    md.append(
        f"**Aggregate:** mean Δ R@30 = {deltas.mean()*100:+.4f}pp  "
        f"std = {deltas.std()*100:.4f}pp  "
        f"n = {len(rows)}  "
        f"({n_pos}+ / {n_neg}- / {n_zero}=0)\n\n"
    )
    md.append(
        f"**Non-zero rows only (Δ≠0, n={len(nonzero_rows)}):** "
        f"mean Δ R@30 = {nz.mean()*100:+.4f}pp\n\n"
    )

    md.append("## Per-n_gt bucket\n\n")
    md.append("| n_gt bucket | n_rows | mean Δ R@30 | nonzero mean Δ | frac+ | frac- |\n|---|---|---|---|---|---|\n")
    for b, v in bucket_summary.items():
        md.append(
            f"| {b} | {v['n_rows']} | {v['mean_delta_r30']*100:+.4f}pp | "
            f"{v['nonzero_mean_delta']*100:+.4f}pp | "
            f"{v['frac_positive']*100:.1f}% | {v['frac_negative']*100:.1f}% |\n"
        )

    md.append("\n## Per-category (sorted by mean Δ R@30)\n\n")
    md.append("| Category | n_rows | mean Δ R@30 | nonzero mean Δ | frac+ | frac- |\n|---|---|---|---|---|---|\n")
    for c in cats_sorted:
        v = cat_summary[c]
        md.append(
            f"| `{c}` | {v['n_rows']} | {v['mean_delta_r30']*100:+.4f}pp | "
            f"{v['nonzero_mean_delta']*100:+.4f}pp | "
            f"{v['frac_positive']*100:.1f}% | {v['frac_negative']*100:.1f}% |\n"
        )

    md.append("\n## Cross-tab (category × n_gt bucket) mean Δ R@30 (pp)\n\n")
    md.append("| Category | 1 | 2-5 | 6-20 | 21-50 | 51+ |\n|---|---|---|---|---|---|\n")
    for c in cats_sorted:
        row = [f"| `{c}` "]
        for b in ["1", "2-5", "6-20", "21-50", "51+"]:
            v = ct[c].get(b)
            row.append(f"| {v:+.3f} " if v is not None else "| — ")
        row.append("|\n")
        md.append("".join(row))

    md.append("\n## Top 15 best diseases\n\n| Mean Δ R@30 | n_seeds | n_gt_mean | Category | Name |\n|---|---|---|---|---|\n")
    for r in disease_agg_sorted[:15]:
        md.append(
            f"| {r['mean_delta_r30']*100:+.3f}pp | {r['n_seeds_seen']} | "
            f"{r['n_gt_eval_mean']:.1f} | `{r['category']}` | {r['name'][:80]} |\n"
        )
    md.append("\n## Bottom 15 worst diseases\n\n| Mean Δ R@30 | n_seeds | n_gt_mean | Category | Name |\n|---|---|---|---|---|\n")
    for r in disease_agg_sorted[-15:][::-1]:
        md.append(
            f"| {r['mean_delta_r30']*100:+.3f}pp | {r['n_seeds_seen']} | "
            f"{r['n_gt_eval_mean']:.1f} | `{r['category']}` | {r['name'][:80]} |\n"
        )

    with open(OUT_MD, "w") as f:
        f.write("".join(md))
    print(f"Wrote {OUT_MD.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
