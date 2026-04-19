#!/usr/bin/env python3
"""h1218: Fusion gain decomposition — where does concat_l2 actually win?

h1215 established +1.32pp R@30 for concat_l2(Node2Vec, FastRP) over
Node2Vec alone. h1216 showed the linear-weight axis is flat. h1218 asks
the next question: WHICH diseases drive the gain? If gain concentrates
in a recognisable subset (e.g. diseases where N2V and FastRP disagree on
top-k neighbours), we have a per-disease routing feature that enables
adaptive fusion (h1226 precondition).

For each holdout disease across 5 seeds:
  - R@30 in Node2Vec-only kNN
  - R@30 in concat_l2 kNN
  - Δ R@30 = concat_l2 − node2vec
  - neighbour_jaccard = Jaccard(top-20 N2V train neighbours,
                                top-20 FastRP train neighbours)
  - neighbour_spearman = Spearman ρ of full similarity rankings

Output:
  - Per-disease table (seed, disease_id, category, Δ R@30, Jaccard, ρ)
  - Per-Jaccard-quartile mean Δ R@30
  - Per-category mean Δ R@30
  - Pearson correlation(Δ R@30, 1 − Jaccard)
  - List of categories where fusion REGRESSES (so we can gate)
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
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from clean_embedding_benchmark import (  # type: ignore[import-not-found]  # noqa: E402
    categorize,
    load_disease_names,
    load_embeddings,
    split_diseases,
)
from h1215_fusion_benchmark import build_concat_lookup, score_disease_single  # type: ignore  # noqa: E402


TOPN_FOR_JACCARD = 20


def per_disease_r30(
    lu: Dict[str, np.ndarray],
    holdout_ids: List[str],
    train_ids_ordered: List[str],
    train_gt: Dict[str, Set[str]],
    eval_gt: Dict[str, Set[str]],
    universe: Set[str],
    k: int,
) -> Dict[str, float]:
    """Return {disease_id: R@30} under a single embedding lookup."""
    train_emb = np.stack([lu[d] for d in train_ids_ordered])
    cand_list = sorted(universe & set(lu.keys()))
    cand_index = {d: i for i, d in enumerate(cand_list)}

    out: Dict[str, float] = {}
    for did in holdout_ids:
        if did not in lu:
            continue
        gt_drugs = eval_gt.get(did, set()) & universe
        if not gt_drugs:
            continue
        drug_scores = score_disease_single(lu[did], train_emb, train_ids_ordered, train_gt, k)
        score_vec = np.zeros(len(cand_list), dtype=np.float32)
        for drug, sc in drug_scores.items():
            if drug in cand_index:
                score_vec[cand_index[drug]] = sc
        order = np.argsort(-score_vec, kind="stable")
        rank = {cand_list[idx]: r for r, idx in enumerate(order, start=1)}
        hit = sum(1 for d in gt_drugs if rank.get(d, 10**9) <= 30) / len(gt_drugs)
        out[did] = hit
    return out


def top_k_train_ids(
    disease_emb: np.ndarray,
    train_emb: np.ndarray,
    train_ids_ordered: List[str],
    k: int,
) -> List[str]:
    sims = cosine_similarity(disease_emb[None, :], train_emb)[0]
    if k >= len(sims):
        order = np.argsort(-sims)[:k]
    else:
        idx = np.argpartition(-sims, k - 1)[:k]
        order = idx[np.argsort(-sims[idx])]
    return [train_ids_ordered[i] for i in order]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix-a", default="node2vec_256")
    ap.add_argument("--prefix-b", default="fastrp_256")
    ap.add_argument("--seeds", type=str, default="42,123,456,789,2024")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--eval-gt", default="data/reference/expanded_ground_truth.json")
    ap.add_argument("--knn-gt", default="data/cache/ground_truth_cache.json")
    ap.add_argument("--output", default="data/analysis/h1218_fusion_gain_decomposition.json")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    print("=" * 72)
    print(f"h1218: Fusion gain decomposition — {args.prefix_a} vs concat_l2({args.prefix_a},{args.prefix_b})")
    print(f"       seeds={seeds} k={args.k}")
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
    print(f"  {len(lu_concat):,} entities")

    with open(PROJECT_ROOT / args.eval_gt) as f:
        eval_raw = json.load(f)
    eval_gt = {d: set(v) for d, v in eval_raw.items()}
    with open(PROJECT_ROOT / args.knn_gt) as f:
        knn_raw = json.load(f)
    if isinstance(knn_raw, dict) and "ground_truth" in knn_raw:
        knn_raw = knn_raw["ground_truth"]
    knn_gt = {d: set(v) for d, v in knn_raw.items()}

    disease_names = load_disease_names()

    intersection_keys = set(lu_a) & set(lu_b)
    all_diseases = [d for d in knn_gt if d in intersection_keys]
    print(f"Disease universe (knn_gt ∩ A ∩ B): {len(all_diseases):,}")
    print(f"Setup complete in {time.time()-t0:.1f}s")

    # Per-disease records pooled across seeds
    # One row per (seed, disease_id)
    rows: List[Dict] = []

    for si, seed in enumerate(seeds):
        print(f"\n=== SEED {seed} ({si+1}/{len(seeds)}) ===")
        ts = time.time()
        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        train_ids_ordered = [d for d in train_ids if d in lu_concat]
        train_gt = {d: (knn_gt[d] & intersection_keys) for d in train_ids_ordered if d in knn_gt}
        cand_drugs: Set[str] = set()
        for vs in train_gt.values():
            cand_drugs |= vs
        all_hold_drugs: Set[str] = set()
        for did in holdout_ids:
            all_hold_drugs |= eval_gt.get(did, set())
        universe = (cand_drugs | all_hold_drugs) & intersection_keys

        # R@30 under three embeddings
        r30_n2v = per_disease_r30(lu_a, holdout_ids, train_ids_ordered, train_gt, eval_gt, universe, args.k)
        r30_frp = per_disease_r30(lu_b, holdout_ids, train_ids_ordered, train_gt, eval_gt, universe, args.k)
        r30_con = per_disease_r30(lu_concat, holdout_ids, train_ids_ordered, train_gt, eval_gt, universe, args.k)

        # Precompute train-embedding matrices once per seed
        train_emb_a = np.stack([lu_a[d] for d in train_ids_ordered])
        train_emb_b = np.stack([lu_b[d] for d in train_ids_ordered])

        # Per-holdout neighbour agreement
        for did in holdout_ids:
            if did not in r30_con:
                continue
            top_a = set(top_k_train_ids(lu_a[did], train_emb_a, train_ids_ordered, TOPN_FOR_JACCARD))
            top_b = set(top_k_train_ids(lu_b[did], train_emb_b, train_ids_ordered, TOPN_FOR_JACCARD))
            if top_a or top_b:
                jac = len(top_a & top_b) / len(top_a | top_b)
            else:
                jac = 0.0

            # Full Spearman ρ over train diseases (index access is
            # portable across SciPy versions with/without .statistic).
            sims_a = cosine_similarity(lu_a[did][None, :], train_emb_a)[0]
            sims_b = cosine_similarity(lu_b[did][None, :], train_emb_b)[0]
            rho_res = spearmanr(sims_a, sims_b, nan_policy="omit")
            rho = float(rho_res[0])  # type: ignore[index]

            rows.append({
                "seed": seed,
                "disease_id": did,
                "name": disease_names.get(did, did),
                "category": categorize(disease_names.get(did, did)),
                "n_gt_train_drugs": len(eval_gt.get(did, set()) & universe),
                "r30_node2vec": r30_n2v.get(did, 0.0),
                "r30_fastrp": r30_frp.get(did, 0.0),
                "r30_concat_l2": r30_con.get(did, 0.0),
                "delta_concat_minus_n2v": r30_con.get(did, 0.0) - r30_n2v.get(did, 0.0),
                "delta_concat_minus_frp": r30_con.get(did, 0.0) - r30_frp.get(did, 0.0),
                "jaccard_top20_neighbours": jac,
                "spearman_full_sim": rho,
            })
        print(f"  seed {seed} done in {time.time()-ts:.1f}s — {len(r30_con)} eval diseases")

    # Aggregate
    print("\n" + "=" * 72)
    print(f"Collected {len(rows):,} (seed, disease) rows.")
    print("=" * 72)

    # Overall correlations
    deltas = np.array([r["delta_concat_minus_n2v"] for r in rows])
    jacs = np.array([r["jaccard_top20_neighbours"] for r in rows])
    rhos = np.array([r["spearman_full_sim"] for r in rows])
    disagree = 1.0 - jacs  # higher = more disagreement

    pj = pearsonr(deltas, disagree)
    pr = pearsonr(deltas, -rhos)  # higher = more disagreement (lower rho)
    pj_r, pj_p = float(pj[0]), float(pj[1])  # type: ignore[index]
    pr_r, pr_p = float(pr[0]), float(pr[1])  # type: ignore[index]
    print(f"\nPearson(Δ R@30, 1−Jaccard)  = {pj_r:+.4f}  p={pj_p:.3g}  n={len(rows)}")
    print(f"Pearson(Δ R@30, −Spearman)  = {pr_r:+.4f}  p={pr_p:.3g}  n={len(rows)}")

    # Jaccard quartiles
    print("\n=== Per-Jaccard-quartile fusion gain (n={}) ===".format(len(rows)))
    qs = np.quantile(jacs, [0.25, 0.50, 0.75])
    buckets: Dict[str, List[float]] = {"Q1_low": [], "Q2": [], "Q3": [], "Q4_high": []}
    for d, j in zip(deltas, jacs):
        if j <= qs[0]:
            buckets["Q1_low"].append(float(d))
        elif j <= qs[1]:
            buckets["Q2"].append(float(d))
        elif j <= qs[2]:
            buckets["Q3"].append(float(d))
        else:
            buckets["Q4_high"].append(float(d))
    print(f"{'Bucket':<10}  {'n':>6}  mean Δ R@30    std")
    bucket_summary = {}
    for bk in ("Q1_low", "Q2", "Q3", "Q4_high"):
        vs = buckets[bk]
        mean = float(np.mean(vs)) if vs else 0.0
        std = float(np.std(vs)) if vs else 0.0
        bucket_summary[bk] = {"n": len(vs), "mean_delta": mean, "std": std}
        print(f"{bk:<10}  {len(vs):>6}  {mean*100:+7.3f}pp    {std*100:5.3f}pp")

    # Per-category gain
    print("\n=== Per-category fusion gain ===")
    cat_rows: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        cat_rows[r["category"]].append(r["delta_concat_minus_n2v"])
    print(f"{'Category':<20}  {'n':>6}  mean Δ R@30    std")
    cat_summary = {}
    for cat in sorted(cat_rows, key=lambda c: -np.mean(cat_rows[c])):
        vs = cat_rows[cat]
        mean = float(np.mean(vs))
        std = float(np.std(vs))
        cat_summary[cat] = {"n": len(vs), "mean_delta": mean, "std": std}
        flag = "  ←REGRESS" if mean < -0.005 else ""
        print(f"{cat:<20}  {len(vs):>6}  {mean*100:+7.3f}pp    {std*100:5.3f}pp{flag}")

    # Gain concentration: what fraction of diseases account for the total gain?
    abs_gain = sum(d for d in deltas if d > 0)
    abs_loss = sum(-d for d in deltas if d < 0)
    gainers = [r for r in rows if r["delta_concat_minus_n2v"] > 0]
    losers = [r for r in rows if r["delta_concat_minus_n2v"] < 0]
    neutral = [r for r in rows if r["delta_concat_minus_n2v"] == 0]
    print("\n=== Gain concentration ===")
    print(f"Rows gaining:  {len(gainers)} ({100*len(gainers)/len(rows):.1f}%)")
    print(f"Rows losing:   {len(losers)} ({100*len(losers)/len(rows):.1f}%)")
    print(f"Rows neutral:  {len(neutral)} ({100*len(neutral)/len(rows):.1f}%)")
    print(f"Σ positive Δ:  {abs_gain:+.3f} (pp×drug-frac)")
    print(f"Σ negative Δ:  {-abs_loss:+.3f}")
    print(f"Net gain per row: {(abs_gain-abs_loss)/len(rows)*100:+.3f}pp (matches h1215 +1.32pp expectation)")

    # Top gainers + top losers
    sorted_rows = sorted(rows, key=lambda r: -r["delta_concat_minus_n2v"])
    print("\nTop 10 gainers:")
    for r in sorted_rows[:10]:
        print(f"  {r['delta_concat_minus_n2v']*100:+6.2f}pp  jac={r['jaccard_top20_neighbours']:.3f}  "
              f"{r['category']:<15}  {r['name'][:50]}")
    print("\nTop 10 losers:")
    for r in sorted_rows[-10:]:
        print(f"  {r['delta_concat_minus_n2v']*100:+6.2f}pp  jac={r['jaccard_top20_neighbours']:.3f}  "
              f"{r['category']:<15}  {r['name'][:50]}")

    # Persist
    out_json = PROJECT_ROOT / args.output
    out_json.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "hypothesis": "h1218",
        "prefix_a": args.prefix_a,
        "prefix_b": args.prefix_b,
        "k": args.k,
        "seeds": seeds,
        "n_rows": len(rows),
        "correlation_delta_vs_disagreement": {
            "pearson_r_jaccard": pj_r,
            "pearson_p_jaccard": pj_p,
            "pearson_r_spearman": pr_r,
            "pearson_p_spearman": pr_p,
        },
        "jaccard_quartile_summary": bucket_summary,
        "category_summary": cat_summary,
        "rows": rows,
    }
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out_json}  ({len(rows):,} rows)")

    # Markdown
    out_md = out_json.with_suffix(".md")
    lines = []
    lines.append("# h1218: Fusion Gain Decomposition — per-disease concat_l2 vs node2vec")
    lines.append("")
    lines.append(f"- seeds = {seeds}, k = {args.k}")
    lines.append(f"- Rows (seed × holdout disease): **{len(rows):,}**")
    lines.append(f"- Pearson(Δ R@30, 1 − Jaccard top-20 train neighbours) = **{pj_r:+.4f}**  p={pj_p:.3g}")
    lines.append(f"- Pearson(Δ R@30, −Spearman of full train-sim rankings) = **{pr_r:+.4f}**  p={pr_p:.3g}")
    lines.append("")
    lines.append("## Fusion gain by neighbour-agreement quartile")
    lines.append("")
    lines.append("| Jaccard bucket | n | mean Δ R@30 | std |")
    lines.append("|---|---|---|---|")
    for bk in ("Q1_low", "Q2", "Q3", "Q4_high"):
        s = bucket_summary[bk]
        lines.append(f"| {bk} | {s['n']} | {s['mean_delta']*100:+.3f}pp | {s['std']*100:.3f}pp |")
    lines.append("")
    lines.append("## Per-category fusion gain")
    lines.append("")
    lines.append("| Category | n | mean Δ R@30 | std | note |")
    lines.append("|---|---|---|---|---|")
    for cat in sorted(cat_summary, key=lambda c: -cat_summary[c]["mean_delta"]):
        s = cat_summary[cat]
        note = "REGRESS" if s["mean_delta"] < -0.005 else ""
        lines.append(f"| {cat} | {s['n']} | {s['mean_delta']*100:+.3f}pp | {s['std']*100:.3f}pp | {note} |")
    lines.append("")
    lines.append("## Gain concentration")
    lines.append("")
    lines.append(f"- Rows gaining: {len(gainers)} ({100*len(gainers)/len(rows):.1f}%)")
    lines.append(f"- Rows losing: {len(losers)} ({100*len(losers)/len(rows):.1f}%)")
    lines.append(f"- Rows neutral: {len(neutral)} ({100*len(neutral)/len(rows):.1f}%)")
    lines.append(f"- Net mean Δ R@30 per row: **{(abs_gain-abs_loss)/len(rows)*100:+.3f}pp**")
    lines.append("")
    with open(out_md, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
