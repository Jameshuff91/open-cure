#!/usr/bin/env python3
"""h1281: Per-category soft-blend weight — fit w_cat on inner train fold,
apply on outer holdout.

Premise. h1275's 20-seed GLOBAL linear-weight sweep is flat on per-dis AUPRC
across w ∈ [0.3, 0.5] and collapses for w > 0.5. The flat plateau is the
AGGREGATE mean; h1272's per-disease audit found strong category
heterogeneity: endocrine +0.0247, dermatological +0.0156, neurological
+0.0132 vs cancer -0.0017, musculoskeletal -0.0005 per-row Δ per-dis AUPRC.
Different categories may have different-optimal w.

Design.

  For each outer seed:
    train_ids, holdout_ids = split_diseases(all_diseases, outer_seed)
    inner_train_ids, inner_fit_ids = split_diseases(train_ids,
                                                    outer_seed + 10007)

    # fit w_cat on inner fold using inner_train_ids as kNN basis
    per-disease AUPRC for each w ∈ {0.0, 0.25, 0.5, 0.75, 1.0} on inner_fit_ids
    aggregate by category; best_w[cat] = argmax mean Δ per-dis AUPRC vs w=0.5
    fallback w=0.5 for categories with < MIN_CAT_ROWS rows in inner fold

    # apply on outer holdout using FULL train_ids as kNN basis
    per holdout disease d: category = categorize(name(d)),
                           w = best_w.get(category, 0.5)
                           score = w*z(sv_a) + (1-w)*z(sv_c)

    Also compute W050 uniform baseline on outer holdout (same basis, w=0.5).

  Across 20 seeds: paired-t W_PER_CAT vs W050 on per-dis AUPRC, R@30,
                   per-dis AUROC.

Preregistered promotion gate.
  Promote if Δ per-dis AUPRC ≥ +0.001 at p < 0.05 AND no category regresses
  its outer-holdout mean Δ > -0.005 vs W050.

Outputs:
    data/analysis/h1281_per_category_weight.{json,md}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from clean_embedding_benchmark import (  # noqa: E402
    categorize,
    load_disease_names,
    load_embeddings,
    mean_std,
    split_diseases,
)
from h1215_fusion_benchmark import build_concat_lookup, score_disease_single  # noqa: E402
from h1249_entropy_routed_benchmark import paired_t  # noqa: E402
from h1255_soft_blend_fusion import z_normalise  # noqa: E402

OUT_JSON = PROJECT_ROOT / "data/analysis/h1281_per_category_weight.json"
OUT_MD = PROJECT_ROOT / "data/analysis/h1281_per_category_weight.md"

WEIGHTS: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
REFERENCE_W: float = 0.5
INNER_SEED_JITTER = 10007
MIN_CAT_ROWS = 5  # minimum inner-fit rows to trust a category-specific w
DEFAULT_SEEDS = "42,123,456,789,2024,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"


def _build_universe(
    *,
    basis_ids: Sequence[str],
    eval_ids: Sequence[str],
    knn_gt: Dict[str, Set[str]],
    eval_gt: Dict[str, Set[str]],
    lu_ref: Dict[str, np.ndarray],
) -> Tuple[List[str], Dict[str, int]]:
    """Universe of candidate drugs: union of train GT drugs + eval GT drugs."""
    cand: Set[str] = set()
    for d in basis_ids:
        if d in knn_gt:
            cand |= (knn_gt[d] & set(lu_ref.keys()))
    for d in eval_ids:
        cand |= eval_gt.get(d, set())
    cand &= set(lu_ref.keys())
    cand_list = sorted(cand)
    return cand_list, {d: i for i, d in enumerate(cand_list)}


def _score_and_metrics_per_disease(
    *,
    basis_ids: Sequence[str],
    eval_ids: Sequence[str],
    lu_a: Dict[str, np.ndarray],
    lu_concat: Dict[str, np.ndarray],
    knn_gt: Dict[str, Set[str]],
    eval_gt: Dict[str, Set[str]],
    k: int,
    weights: Sequence[float],
    disease_names: Dict[str, str],
) -> List[Dict]:
    """For each eval disease, compute per-disease AUPRC / AUROC / R@30 at each weight.

    Returns list of {disease_id, name, category, n_gt, per_w metrics}.
    """
    basis_ordered = [d for d in basis_ids if d in lu_concat and d in lu_a]
    basis_emb_a = np.stack([lu_a[d] for d in basis_ordered])
    basis_emb_c = np.stack([lu_concat[d] for d in basis_ordered])
    basis_gt = {
        d: (knn_gt[d] & set(lu_concat.keys()))
        for d in basis_ordered if d in knn_gt
    }
    cand_list, cand_index = _build_universe(
        basis_ids=basis_ordered,
        eval_ids=eval_ids,
        knn_gt=knn_gt,
        eval_gt=eval_gt,
        lu_ref=lu_concat,
    )
    n_cands = len(cand_list)

    out: List[Dict] = []
    for did in eval_ids:
        if did not in lu_concat or did not in lu_a:
            continue
        gt_drugs = eval_gt.get(did, set()) & set(cand_list)
        if not gt_drugs:
            continue

        ds_a = score_disease_single(lu_a[did], basis_emb_a, basis_ordered, basis_gt, k)
        sv_a = np.zeros(n_cands, dtype=np.float32)
        for drug, sc in ds_a.items():
            j = cand_index.get(drug)
            if j is not None:
                sv_a[j] = sc
        ds_c = score_disease_single(lu_concat[did], basis_emb_c, basis_ordered, basis_gt, k)
        sv_c = np.zeros(n_cands, dtype=np.float32)
        for drug, sc in ds_c.items():
            j = cand_index.get(drug)
            if j is not None:
                sv_c[j] = sc

        z_a = z_normalise(sv_a)
        z_c = z_normalise(sv_c)

        label_vec = np.zeros(n_cands, dtype=np.int8)
        for d in gt_drugs:
            j = cand_index.get(d)
            if j is not None:
                label_vec[j] = 1
        n_pos = int(label_vec.sum())
        if n_pos == 0 or n_pos == n_cands:
            continue

        name = disease_names.get(did, did)
        row = {
            "disease_id": did,
            "name": name,
            "category": categorize(name),
            "n_gt": n_pos,
            "per_w": {},
        }
        for w in weights:
            score = w * z_a + (1.0 - w) * z_c
            order = np.argsort(-score, kind="stable")
            rank_of_idx = {int(idx): r + 1 for r, idx in enumerate(order)}
            hits_30 = sum(
                1 for d in gt_drugs
                if cand_index.get(d) is not None
                and rank_of_idx.get(cand_index[d], 10**9) <= 30
            )
            r30 = hits_30 / n_pos
            ap = float(average_precision_score(label_vec, score))
            au = float(roc_auc_score(label_vec, score))
            row["per_w"][f"{w:.2f}"] = {"r30": r30, "auprc": ap, "auroc": au}
        out.append(row)
    return out


def fit_w_per_category(
    inner_rows: List[Dict],
    weights: Sequence[float],
    reference_w: float,
    min_rows: int,
) -> Tuple[Dict[str, float], Dict[str, Dict]]:
    """For each category, pick the w that maximises mean Δ per-dis AUPRC vs reference_w.

    Falls back to reference_w for categories with fewer than min_rows inner rows.
    """
    ref_key = f"{reference_w:.2f}"
    per_cat: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in inner_rows:
        cat = r["category"]
        ref_ap = r["per_w"][ref_key]["auprc"]
        for w in weights:
            dk = f"{w:.2f}"
            delta = r["per_w"][dk]["auprc"] - ref_ap
            per_cat[cat][dk].append(delta)

    best_w: Dict[str, float] = {}
    cat_fit_summary: Dict[str, Dict] = {}
    for cat, wdict in per_cat.items():
        first_w_deltas = list(wdict.values())[0]
        n_rows = len(first_w_deltas)
        if n_rows < min_rows:
            best_w[cat] = reference_w
            cat_fit_summary[cat] = {
                "n_rows": n_rows,
                "best_w": reference_w,
                "best_w_mean_delta_auprc": 0.0,
                "fallback": True,
                "means_by_w": {k: float(np.mean(v)) for k, v in wdict.items()},
            }
            continue
        means_by_w = {k: float(np.mean(v)) for k, v in wdict.items()}
        bw = max(weights, key=lambda w: means_by_w[f"{w:.2f}"])
        best_w[cat] = bw
        cat_fit_summary[cat] = {
            "n_rows": n_rows,
            "best_w": bw,
            "best_w_mean_delta_auprc": means_by_w[f"{bw:.2f}"],
            "fallback": False,
            "means_by_w": means_by_w,
        }
    return best_w, cat_fit_summary


def evaluate_outer_with_fitted(
    outer_rows: List[Dict],
    best_w: Dict[str, float],
    reference_w: float,
) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """Aggregate outer-holdout metrics under (a) per-category fitted w,
    (b) uniform reference_w, (c) each raw weight in the grid for diagnostics.

    Returns (aggregate, per_category_outer_summary).
    """
    modes_r30: Dict[str, List[float]] = defaultdict(list)
    modes_auprc: Dict[str, List[float]] = defaultdict(list)
    modes_auroc: Dict[str, List[float]] = defaultdict(list)

    ref_key = f"{reference_w:.2f}"

    per_cat_outer: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for row in outer_rows:
        cat = row["category"]
        w_cat = best_w.get(cat, reference_w)
        fitted_key = f"{w_cat:.2f}"
        ref_metrics = row["per_w"][ref_key]
        fitted_metrics = row["per_w"][fitted_key]

        modes_r30["W050"].append(ref_metrics["r30"])
        modes_auprc["W050"].append(ref_metrics["auprc"])
        modes_auroc["W050"].append(ref_metrics["auroc"])

        modes_r30["W_PER_CAT"].append(fitted_metrics["r30"])
        modes_auprc["W_PER_CAT"].append(fitted_metrics["auprc"])
        modes_auroc["W_PER_CAT"].append(fitted_metrics["auroc"])

        for wk, m in row["per_w"].items():
            if wk == ref_key:
                continue
            modes_r30[f"W{int(float(wk)*100):03d}"].append(m["r30"])
            modes_auprc[f"W{int(float(wk)*100):03d}"].append(m["auprc"])
            modes_auroc[f"W{int(float(wk)*100):03d}"].append(m["auroc"])

        per_cat_outer[cat]["delta_auprc"].append(
            fitted_metrics["auprc"] - ref_metrics["auprc"]
        )
        per_cat_outer[cat]["delta_r30"].append(
            fitted_metrics["r30"] - ref_metrics["r30"]
        )
        per_cat_outer[cat]["n_gt"].append(row["n_gt"])
        per_cat_outer[cat]["fitted_w"].append(w_cat)

    agg = {}
    for mode in modes_r30:
        agg[mode] = {
            "r30": float(np.mean(modes_r30[mode])),
            "auprc": float(np.mean(modes_auprc[mode])),
            "auroc": float(np.mean(modes_auroc[mode])),
            "n_diseases": len(modes_r30[mode]),
        }

    per_cat_summary = {}
    for cat, blocks in per_cat_outer.items():
        per_cat_summary[cat] = {
            "n_rows": len(blocks["delta_auprc"]),
            "mean_delta_auprc": float(np.mean(blocks["delta_auprc"])),
            "mean_delta_r30": float(np.mean(blocks["delta_r30"])),
            "fitted_w_mode": (
                float(max(set(blocks["fitted_w"]), key=blocks["fitted_w"].count))
                if blocks["fitted_w"] else reference_w
            ),
        }
    return agg, per_cat_summary


def run_seed(
    *,
    outer_seed: int,
    all_diseases: List[str],
    lu_a: Dict[str, np.ndarray],
    lu_concat: Dict[str, np.ndarray],
    knn_gt: Dict[str, Set[str]],
    eval_gt: Dict[str, Set[str]],
    disease_names: Dict[str, str],
    k: int,
) -> Dict:
    train_ids, holdout_ids = split_diseases(all_diseases, outer_seed)
    inner_seed = outer_seed + INNER_SEED_JITTER
    inner_train_ids, inner_fit_ids = split_diseases(train_ids, inner_seed)

    t_inner = time.time()
    inner_rows = _score_and_metrics_per_disease(
        basis_ids=inner_train_ids,
        eval_ids=inner_fit_ids,
        lu_a=lu_a,
        lu_concat=lu_concat,
        knn_gt=knn_gt,
        eval_gt=eval_gt,
        k=k,
        weights=WEIGHTS,
        disease_names=disease_names,
    )
    best_w, cat_fit = fit_w_per_category(inner_rows, WEIGHTS, REFERENCE_W, MIN_CAT_ROWS)
    t_inner_done = time.time() - t_inner

    t_outer = time.time()
    outer_rows = _score_and_metrics_per_disease(
        basis_ids=train_ids,
        eval_ids=holdout_ids,
        lu_a=lu_a,
        lu_concat=lu_concat,
        knn_gt=knn_gt,
        eval_gt=eval_gt,
        k=k,
        weights=WEIGHTS,
        disease_names=disease_names,
    )
    agg, per_cat_outer = evaluate_outer_with_fitted(outer_rows, best_w, REFERENCE_W)
    t_outer_done = time.time() - t_outer

    return {
        "seed": outer_seed,
        "n_inner_rows": len(inner_rows),
        "n_outer_rows": len(outer_rows),
        "best_w": best_w,
        "cat_fit_summary": cat_fit,
        "outer_aggregate": agg,
        "per_cat_outer": per_cat_outer,
        "timing": {"inner_s": t_inner_done, "outer_s": t_outer_done},
    }


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
    print(f"h1281: Per-category soft-blend weight (inner-fold fit, outer-fold eval)")
    print(f"  weights={WEIGHTS}  reference={REFERENCE_W}  seeds={len(seeds)}")
    print(f"  min_cat_rows={MIN_CAT_ROWS}")
    print("=" * 72)

    t0 = time.time()
    print("Loading embeddings...")
    lu_a, _, _ = load_embeddings(args.prefix_a)
    lu_b, _, _ = load_embeddings(args.prefix_b)
    lu_concat = build_concat_lookup(lu_a, lu_b)
    print(f"  A={len(lu_a):,}  B={len(lu_b):,}  concat={len(lu_concat):,}")

    disease_names = load_disease_names()
    with open(PROJECT_ROOT / args.eval_gt) as f:
        eval_gt = {d: set(v) for d, v in json.load(f).items()}
    with open(PROJECT_ROOT / args.knn_gt) as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "ground_truth" in raw:
        raw = raw["ground_truth"]
    knn_gt = {d: set(v) for d, v in raw.items()}

    intersection_keys = set(lu_a) & set(lu_b)
    all_diseases = [d for d in knn_gt if d in intersection_keys]
    print(f"Universe: {len(all_diseases):,} diseases  | setup {time.time()-t0:.1f}s")

    per_seed: List[Dict] = []
    for seed in seeds:
        ts = time.time()
        out = run_seed(
            outer_seed=seed,
            all_diseases=all_diseases,
            lu_a=lu_a,
            lu_concat=lu_concat,
            knn_gt=knn_gt,
            eval_gt=eval_gt,
            disease_names=disease_names,
            k=args.k,
        )
        per_seed.append(out)
        agg = out["outer_aggregate"]
        print(
            f"\n  seed {seed} ({time.time()-ts:.1f}s) "
            f"inner={out['n_inner_rows']} outer={out['n_outer_rows']}"
        )
        for mode in ("W050", "W_PER_CAT"):
            a = agg[mode]
            print(
                f"    {mode:10s} R@30={a['r30']*100:5.2f}% "
                f"AUPRC={a['auprc']:.4f} AUROC={a['auroc']:.4f}"
            )
        fit_w_counts: Dict[float, int] = defaultdict(int)
        for w in out["best_w"].values():
            fit_w_counts[w] += 1
        print(f"    fitted_w histogram: {dict(sorted(fit_w_counts.items()))}")

    # ---- Aggregate across seeds ----
    print("\n" + "=" * 72)
    print(f"AGGREGATE (mean ± std across {len(seeds)} seeds)")
    print("=" * 72)

    all_modes: List[str] = sorted(
        {m for s in per_seed for m in s["outer_aggregate"]},
        key=lambda m: (m != "W050", m != "W_PER_CAT", m),
    )
    agg_across = {}
    for m in all_modes:
        r30 = [s["outer_aggregate"][m]["r30"] for s in per_seed]
        ap = [s["outer_aggregate"][m]["auprc"] for s in per_seed]
        au = [s["outer_aggregate"][m]["auroc"] for s in per_seed]
        agg_across[m] = {
            "r30": mean_std(r30),
            "auprc": mean_std(ap),
            "auroc": mean_std(au),
        }
        print(
            f"  {m:10s}  R@30={agg_across[m]['r30'][0]*100:5.2f}%±{agg_across[m]['r30'][1]*100:.2f}%  "
            f"per-dis-AUPRC={agg_across[m]['auprc'][0]:.4f}±{agg_across[m]['auprc'][1]:.4f}  "
            f"per-dis-AUROC={agg_across[m]['auroc'][0]:.4f}±{agg_across[m]['auroc'][1]:.4f}"
        )

    # ---- Paired-t W_PER_CAT vs W050 ----
    d_r30 = [s["outer_aggregate"]["W_PER_CAT"]["r30"] - s["outer_aggregate"]["W050"]["r30"] for s in per_seed]
    d_ap = [s["outer_aggregate"]["W_PER_CAT"]["auprc"] - s["outer_aggregate"]["W050"]["auprc"] for s in per_seed]
    d_au = [s["outer_aggregate"]["W_PER_CAT"]["auroc"] - s["outer_aggregate"]["W050"]["auroc"] for s in per_seed]

    t_r30 = paired_t(d_r30)
    t_ap = paired_t(d_ap)
    t_au = paired_t(d_au)

    print("\n" + "=" * 72)
    print(f"PAIRED-T: W_PER_CAT vs W050 (n={len(seeds)} seeds)")
    print("=" * 72)
    print(f"  R@30           Δ={t_r30['mean']*100:+.4f}pp  p={t_r30['p_two_sided']:.4g}")
    print(f"  per-dis-AUPRC  Δ={t_ap['mean']:+.5f}  p={t_ap['p_two_sided']:.4g}")
    print(f"  per-dis-AUROC  Δ={t_au['mean']:+.5f}  p={t_au['p_two_sided']:.4g}")

    # ---- Per-category on outer holdout ----
    cat_outer_pool: Dict[str, List[float]] = defaultdict(list)
    cat_outer_pool_r30: Dict[str, List[float]] = defaultdict(list)
    cat_outer_pool_n: Dict[str, List[int]] = defaultdict(list)
    cat_outer_pool_w: Dict[str, List[float]] = defaultdict(list)
    for s in per_seed:
        for cat, stat in s["per_cat_outer"].items():
            cat_outer_pool[cat].append(stat["mean_delta_auprc"])
            cat_outer_pool_r30[cat].append(stat["mean_delta_r30"])
            cat_outer_pool_n[cat].append(stat["n_rows"])
            cat_outer_pool_w[cat].append(stat["fitted_w_mode"])

    print("\n" + "=" * 72)
    print("PER-CATEGORY (outer-holdout) summary (sorted by mean Δ per-dis AUPRC)")
    print("=" * 72)
    cat_rows: List[Tuple[str, int, float, float, float, float]] = []
    for cat, deltas in cat_outer_pool.items():
        rows = int(np.sum(cat_outer_pool_n[cat]))
        mean_ap = float(np.mean(deltas))
        std_ap = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0
        mean_r30 = float(np.mean(cat_outer_pool_r30[cat]))
        modal_w = (
            float(max(set(cat_outer_pool_w[cat]), key=cat_outer_pool_w[cat].count))
            if cat_outer_pool_w[cat] else REFERENCE_W
        )
        cat_rows.append((cat, rows, mean_ap, std_ap, mean_r30, modal_w))
    cat_rows.sort(key=lambda r: -r[2])
    worst_cat_delta = min((r[2] for r in cat_rows), default=0.0)
    for (cat, rows, mean_ap, std_ap, mean_r30, modal_w) in cat_rows:
        print(
            f"  {cat:16s} rows={rows:4d}  Δ-AUPRC={mean_ap:+.5f}±{std_ap:.5f}  "
            f"Δ-R@30={mean_r30*100:+.3f}pp  modal_w={modal_w:.2f}"
        )

    # ---- Preregistered promotion gate ----
    gate_auprc = t_ap["mean"] >= 0.001 and t_ap["p_two_sided"] < 0.05
    gate_no_regress = worst_cat_delta > -0.005
    decision = (
        "PROMOTE per-category w as new canonical recipe"
        if (gate_auprc and gate_no_regress)
        else "STAY with global W050"
    )
    print("\n" + "=" * 72)
    print(f"PROMOTION GATE")
    print(f"  Δ per-dis AUPRC ≥ +0.001 AND p<0.05 : "
          f"{'PASS' if gate_auprc else 'FAIL'} "
          f"(Δ={t_ap['mean']:+.5f}, p={t_ap['p_two_sided']:.3g})")
    print(f"  no category regresses > -0.005 Δ    : "
          f"{'PASS' if gate_no_regress else 'FAIL'} "
          f"(worst category Δ={worst_cat_delta:+.5f})")
    print(f"  DECISION: {decision}")
    print("=" * 72)

    # ---- Persist JSON ----
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({
            "hypothesis": "h1281",
            "weights": list(WEIGHTS),
            "reference_w": REFERENCE_W,
            "min_cat_rows": MIN_CAT_ROWS,
            "inner_seed_jitter": INNER_SEED_JITTER,
            "seeds": seeds,
            "n_diseases_universe": len(all_diseases),
            "aggregate": {m: {
                "r30": list(v["r30"]),
                "auprc": list(v["auprc"]),
                "auroc": list(v["auroc"]),
            } for m, v in agg_across.items()},
            "paired_t_per_cat_vs_w050": {
                "R@30": {"mean": t_r30["mean"], "p": t_r30["p_two_sided"]},
                "per_disease_AUPRC": {"mean": t_ap["mean"], "p": t_ap["p_two_sided"]},
                "per_disease_AUROC": {"mean": t_au["mean"], "p": t_au["p_two_sided"]},
            },
            "per_category_outer": {
                r[0]: {"n_rows": r[1], "mean_delta_auprc": r[2], "std_delta_auprc": r[3],
                       "mean_delta_r30": r[4], "modal_fitted_w": r[5]}
                for r in cat_rows
            },
            "per_seed": [
                {
                    "seed": s["seed"],
                    "n_inner_rows": s["n_inner_rows"],
                    "n_outer_rows": s["n_outer_rows"],
                    "best_w": s["best_w"],
                    "cat_fit_summary": s["cat_fit_summary"],
                    "outer_aggregate": s["outer_aggregate"],
                    "per_cat_outer": s["per_cat_outer"],
                    "timing": s["timing"],
                }
                for s in per_seed
            ],
            "gate": {
                "delta_auprc_gate": bool(gate_auprc),
                "no_regress_gate": bool(gate_no_regress),
                "decision": decision,
            },
        }, f, indent=2)
    print(f"\nWrote {OUT_JSON.relative_to(PROJECT_ROOT)}")

    # ---- Markdown ----
    md: List[str] = []
    md.append("# h1281 — Per-category soft-blend weight (inner-fit, outer-eval)\n\n")
    md.append(
        "**Premise.** h1275 locked global w=0.5 on the flat plateau. h1272 showed strong "
        "category heterogeneity in per-disease fusion lift. If each category has a different "
        "optimal w, a per-category sweep could unlock gains the global sweep misses.\n\n"
    )
    md.append(
        "**Design.** Inner 80/20 split on outer train fold → fit `best_w[cat]` (argmax mean "
        "Δ per-dis AUPRC vs w=0.5 on inner-fit rows, min 5 rows per category); apply "
        "fitted weights on outer holdout using full outer-train basis.\n\n"
    )
    md.append(f"Grid: w ∈ {list(WEIGHTS)}, reference w={REFERENCE_W}, seeds={len(seeds)}.\n\n")

    md.append(f"## Aggregate (mean ± std across {len(seeds)} seeds)\n\n")
    md.append("| Mode | R@30 | per-dis-AUPRC | per-dis-AUROC |\n|---|---|---|---|\n")
    for m in all_modes:
        a = agg_across[m]
        md.append(
            f"| `{m}` | {a['r30'][0]*100:.2f}%±{a['r30'][1]*100:.2f}% | "
            f"{a['auprc'][0]:.4f}±{a['auprc'][1]:.4f} | "
            f"{a['auroc'][0]:.4f}±{a['auroc'][1]:.4f} |\n"
        )

    md.append("\n## Paired-t: W_PER_CAT vs W050 (n seeds, outer holdout)\n\n")
    md.append("| Metric | Δ | p |\n|---|---|---|\n")
    md.append(f"| R@30 | {t_r30['mean']*100:+.4f}pp | {t_r30['p_two_sided']:.3g} |\n")
    md.append(f"| per-dis-AUPRC | {t_ap['mean']:+.5f} | {t_ap['p_two_sided']:.3g} |\n")
    md.append(f"| per-dis-AUROC | {t_au['mean']:+.5f} | {t_au['p_two_sided']:.3g} |\n")

    md.append("\n## Per-category outer-holdout summary (sorted by mean Δ per-dis AUPRC)\n\n")
    md.append("| Category | rows | mean Δ AUPRC | std | mean Δ R@30 | modal_w |\n")
    md.append("|---|---|---|---|---|---|\n")
    for (cat, rows, mean_ap, std_ap, mean_r30, modal_w) in cat_rows:
        md.append(
            f"| `{cat}` | {rows} | {mean_ap:+.5f} | {std_ap:.5f} | "
            f"{mean_r30*100:+.3f}pp | {modal_w:.2f} |\n"
        )

    md.append("\n## Preregistered promotion gate\n\n")
    md.append(f"- Δ per-dis AUPRC ≥ +0.001 AND p<0.05: **{'PASS' if gate_auprc else 'FAIL'}** "
              f"(Δ={t_ap['mean']:+.5f}, p={t_ap['p_two_sided']:.3g})\n")
    md.append(f"- No category regresses > -0.005 Δ AUPRC: **{'PASS' if gate_no_regress else 'FAIL'}** "
              f"(worst category Δ={worst_cat_delta:+.5f})\n")
    md.append(f"\n**Decision:** {decision}\n")

    OUT_MD.write_text("".join(md))
    print(f"Wrote {OUT_MD.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
