#!/usr/bin/env python3
"""h1272: Per-disease audit of h1269 SUBSET_D_GLOBAL's +0.00343 per-disease
AUPRC lift.

The h1269 GLOBAL recipe blends every disease (n=200/seed) with
`score = 0.5 * z(n2v) + 0.5 * z(concat_l2)`. Aggregate Δ per-dis AUPRC =
+0.00343 (p=0.0022, 20 seeds), but per-seed Δ varies widely (+0.0114 best
seed; -0.0030 worst). Decompose to learn:

  1. Is the lift carried by a small minority (typical h1230 pattern)?
  2. Which categories disproportionately gain/lose under GLOBAL blending?
  3. Does the gain cluster on n_gt sub-strata or ATC L3 entropy buckets?
  4. Top 20 best / bottom 20 worst diseases — any GT pattern?

Outputs:
    data/analysis/h1272_global_per_disease_audit.{json,md}
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
from sklearn.metrics import average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from atc_features import ATCMapper  # noqa: E402
from clean_embedding_benchmark import (  # noqa: E402
    categorize,
    load_disease_names,
    load_embeddings,
    split_diseases,
)
from h1215_fusion_benchmark import build_concat_lookup, score_disease_single  # noqa: E402
from h1249_entropy_routed_benchmark import disease_atc_l3_entropy  # noqa: E402
from h1255_soft_blend_fusion import z_normalise  # noqa: E402

OUT_JSON = PROJECT_ROOT / "data/analysis/h1272_global_per_disease_audit.json"
OUT_MD = PROJECT_ROOT / "data/analysis/h1272_global_per_disease_audit.md"

BLEND_W = 0.5


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix-a", default="node2vec_256")
    ap.add_argument("--prefix-b", default="fastrp_256")
    ap.add_argument("--seeds", type=str, default="42,123,456,789,2024")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--eval-gt", default="data/reference/expanded_ground_truth.json")
    ap.add_argument("--knn-gt", default="data/cache/ground_truth_cache.json")
    ap.add_argument("--db-lookup", default="data/reference/drugbank_lookup.json")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    print("=" * 72)
    print(f"h1272: Per-disease audit of SUBSET_D_GLOBAL fusion lift")
    print(f"  prefix_a={args.prefix_a}  prefix_b={args.prefix_b}  seeds={seeds}")
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

    with open(PROJECT_ROOT / args.db_lookup) as f:
        db_lookup = json.load(f)
    mapper = ATCMapper()
    disease_names = load_disease_names()
    intersection_keys = set(lu_a) & set(lu_b)
    all_diseases = [d for d in knn_gt if d in intersection_keys]

    n_gt_per_disease: Dict[str, int] = {}
    entropy_per_disease: Dict[str, float] = {}
    for d in all_diseases:
        gt_drugs = list(eval_gt.get(d, set()))
        n_gt_per_disease[d] = len(gt_drugs)
        ent, _ = disease_atc_l3_entropy(gt_drugs, db_lookup, mapper)
        entropy_per_disease[d] = ent
    print(f"Setup: {len(all_diseases):,} diseases ({time.time()-t0:.1f}s)")

    # Per-disease across seeds: collect all (seed, disease) → Δ rows
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
            blended = BLEND_W * z_normalise(sv_a) + (1.0 - BLEND_W) * z_normalise(sv_c)

            label_vec = np.zeros(n_cands, dtype=np.int8)
            for d in gt_drugs:
                idx = cand_index.get(d)
                if idx is not None:
                    label_vec[idx] = 1
            n_pos = int(label_vec.sum())
            if n_pos == 0 or n_pos == n_cands:
                continue

            ap_concat = float(average_precision_score(label_vec, sv_c))
            ap_blend = float(average_precision_score(label_vec, blended))

            rows.append({
                "seed": seed,
                "disease_id": did,
                "name": disease_names.get(did, did),
                "category": categorize(disease_names.get(did, did)),
                "n_gt_eval": int(n_pos),
                "n_gt_full": int(n_gt_per_disease.get(did, 0)),
                "atc_l3_entropy": float(entropy_per_disease.get(did, 0.0)),
                "ap_concat": ap_concat,
                "ap_blend": ap_blend,
                "delta_ap": ap_blend - ap_concat,
            })
            n_processed += 1
        print(f"  seed {seed}: {n_processed} diseases ({time.time()-ts:.1f}s)")

    # ===== Decomposition =====
    print(f"\nTotal (seed, disease) rows: {len(rows)}")
    deltas = np.array([r["delta_ap"] for r in rows])
    print(f"Aggregate Δ per-dis AUPRC: mean = {deltas.mean():+.5f}  std = {deltas.std():.5f}  "
          f"min = {deltas.min():+.5f}  max = {deltas.max():+.5f}")
    n_positive = int(np.sum(deltas > 0))
    n_negative = int(np.sum(deltas < 0))
    n_zero = int(np.sum(deltas == 0))
    print(f"  positive: {n_positive} ({n_positive/len(rows)*100:.1f}%)  "
          f"negative: {n_negative} ({n_negative/len(rows)*100:.1f}%)  "
          f"zero: {n_zero} ({n_zero/len(rows)*100:.1f}%)")

    # By n_gt bucket
    print(f"\n--- Per-n_gt bucket Δ per-dis AUPRC ---")
    by_bucket: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        by_bucket[n_gt_bucket(r["n_gt_eval"])].append(r["delta_ap"])
    bucket_summary: Dict[str, Dict] = {}
    for b in ["1", "2-5", "6-20", "21-50", "51+"]:
        v = by_bucket.get(b, [])
        if not v:
            continue
        arr = np.array(v)
        bucket_summary[b] = {
            "n_rows": int(len(arr)),
            "mean_delta": float(arr.mean()),
            "std_delta": float(arr.std()),
            "frac_positive": float(np.mean(arr > 0)),
        }
        print(f"  n_gt={b:6s}: n={len(arr):4d}  mean_Δ={arr.mean():+.5f}  "
              f"std={arr.std():.5f}  frac_+={np.mean(arr>0)*100:.1f}%")

    # By category
    print(f"\n--- Per-category Δ per-dis AUPRC ---")
    by_cat: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r["delta_ap"])
    cat_summary: Dict[str, Dict] = {}
    cats_sorted = sorted(by_cat.keys(), key=lambda c: float(np.mean(by_cat[c])), reverse=True)
    for c in cats_sorted:
        v = np.array(by_cat[c])
        cat_summary[c] = {
            "n_rows": int(len(v)),
            "mean_delta": float(v.mean()),
            "std_delta": float(v.std()),
            "frac_positive": float(np.mean(v > 0)),
        }
        print(f"  {c:18s}: n={len(v):4d}  mean_Δ={v.mean():+.5f}  "
              f"std={v.std():.5f}  frac_+={np.mean(v>0)*100:.1f}%")

    # By ATC L3 entropy quartile
    print(f"\n--- Per-ATC-entropy quartile Δ per-dis AUPRC ---")
    ents = np.array([r["atc_l3_entropy"] for r in rows])
    qs = np.quantile(ents, [0.25, 0.5, 0.75])
    by_q: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        e = r["atc_l3_entropy"]
        if e < qs[0]:
            q = "Q1_low_ent"
        elif e < qs[1]:
            q = "Q2"
        elif e < qs[2]:
            q = "Q3"
        else:
            q = "Q4_high_ent"
        by_q[q].append(r["delta_ap"])
    quartile_summary: Dict[str, Dict] = {}
    for q in ["Q1_low_ent", "Q2", "Q3", "Q4_high_ent"]:
        v = np.array(by_q.get(q, []))
        if len(v) == 0:
            continue
        quartile_summary[q] = {
            "n_rows": int(len(v)),
            "mean_delta": float(v.mean()),
            "std_delta": float(v.std()),
            "frac_positive": float(np.mean(v > 0)),
        }
        print(f"  {q:15s}: n={len(v):4d}  mean_Δ={v.mean():+.5f}  "
              f"std={v.std():.5f}  frac_+={np.mean(v>0)*100:.1f}%")

    # Per-disease aggregate (mean delta across seeds), top/bottom 20
    by_disease: Dict[str, List[float]] = defaultdict(list)
    disease_meta: Dict[str, Dict] = {}
    for r in rows:
        by_disease[r["disease_id"]].append(r["delta_ap"])
        if r["disease_id"] not in disease_meta:
            disease_meta[r["disease_id"]] = {
                "name": r["name"],
                "category": r["category"],
                "n_gt_full": r["n_gt_full"],
                "atc_l3_entropy": r["atc_l3_entropy"],
            }
    disease_agg = []
    for did, vs in by_disease.items():
        meta = disease_meta[did]
        disease_agg.append({
            "disease_id": did,
            "name": meta["name"],
            "category": meta["category"],
            "n_gt_full": meta["n_gt_full"],
            "atc_l3_entropy": meta["atc_l3_entropy"],
            "n_seeds_seen": len(vs),
            "mean_delta": float(np.mean(vs)),
            "std_delta": float(np.std(vs)),
        })
    # Sort
    disease_agg_sorted = sorted(disease_agg, key=lambda r: r["mean_delta"], reverse=True)
    print(f"\n--- TOP 20 best diseases (largest mean Δ per-dis AUPRC) ---")
    for r in disease_agg_sorted[:20]:
        print(f"  {r['mean_delta']:+.4f}  n_seeds={r['n_seeds_seen']}  "
              f"n_gt={r['n_gt_full']:3d}  ent={r['atc_l3_entropy']:.2f}  "
              f"[{r['category']:15s}] {r['name'][:60]}")
    print(f"\n--- BOTTOM 20 worst diseases (most negative mean Δ per-dis AUPRC) ---")
    for r in disease_agg_sorted[-20:][::-1]:
        print(f"  {r['mean_delta']:+.4f}  n_seeds={r['n_seeds_seen']}  "
              f"n_gt={r['n_gt_full']:3d}  ent={r['atc_l3_entropy']:.2f}  "
              f"[{r['category']:15s}] {r['name'][:60]}")

    # Persist
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({
            "hypothesis": "h1272",
            "blend_w": BLEND_W,
            "seeds": seeds,
            "n_rows": len(rows),
            "aggregate": {
                "mean_delta": float(deltas.mean()),
                "std_delta": float(deltas.std()),
                "min_delta": float(deltas.min()),
                "max_delta": float(deltas.max()),
                "n_positive": n_positive,
                "n_negative": n_negative,
                "n_zero": n_zero,
            },
            "by_n_gt_bucket": bucket_summary,
            "by_category": cat_summary,
            "by_entropy_quartile": quartile_summary,
            "top20_diseases": disease_agg_sorted[:20],
            "bottom20_diseases": disease_agg_sorted[-20:][::-1],
        }, f, indent=2)
    print(f"\nWrote {OUT_JSON.relative_to(PROJECT_ROOT)}")

    # Markdown
    md: List[str] = []
    md.append("# h1272 — Per-disease audit of SUBSET_D_GLOBAL fusion lift\n\n")
    md.append(f"**Aggregate:** mean Δ per-dis AUPRC = {deltas.mean():+.5f}  ")
    md.append(f"std = {deltas.std():.5f}  n = {len(rows)}  ")
    md.append(f"({n_positive}+ / {n_negative}- / {n_zero}=0)\n\n")

    md.append("## Per-n_gt bucket\n\n| n_gt bucket | n_rows | mean Δ AUPRC | std | frac+ |\n|---|---|---|---|---|\n")
    for b, v in bucket_summary.items():
        md.append(f"| {b} | {v['n_rows']} | {v['mean_delta']:+.5f} | {v['std_delta']:.5f} | {v['frac_positive']*100:.1f}% |\n")

    md.append("\n## Per-category (sorted by mean Δ)\n\n| Category | n_rows | mean Δ AUPRC | std | frac+ |\n|---|---|---|---|---|\n")
    for c in cats_sorted:
        v = cat_summary[c]
        md.append(f"| `{c}` | {v['n_rows']} | {v['mean_delta']:+.5f} | {v['std_delta']:.5f} | {v['frac_positive']*100:.1f}% |\n")

    md.append("\n## Per-ATC-L3-entropy quartile\n\n| Quartile | n_rows | mean Δ AUPRC | std | frac+ |\n|---|---|---|---|---|\n")
    for q, v in quartile_summary.items():
        md.append(f"| {q} | {v['n_rows']} | {v['mean_delta']:+.5f} | {v['std_delta']:.5f} | {v['frac_positive']*100:.1f}% |\n")

    md.append("\n## Top 20 best diseases\n\n| Mean Δ | n_seeds | n_gt | ATC ent | Category | Name |\n|---|---|---|---|---|---|\n")
    for r in disease_agg_sorted[:20]:
        md.append(
            f"| {r['mean_delta']:+.4f} | {r['n_seeds_seen']} | {r['n_gt_full']} | "
            f"{r['atc_l3_entropy']:.2f} | `{r['category']}` | {r['name'][:80]} |\n"
        )
    md.append("\n## Bottom 20 worst diseases\n\n| Mean Δ | n_seeds | n_gt | ATC ent | Category | Name |\n|---|---|---|---|---|---|\n")
    for r in disease_agg_sorted[-20:][::-1]:
        md.append(
            f"| {r['mean_delta']:+.4f} | {r['n_seeds_seen']} | {r['n_gt_full']} | "
            f"{r['atc_l3_entropy']:.2f} | `{r['category']}` | {r['name'][:80]} |\n"
        )
    with open(OUT_MD, "w") as f:
        f.write("".join(md))
    print(f"Wrote {OUT_MD.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
