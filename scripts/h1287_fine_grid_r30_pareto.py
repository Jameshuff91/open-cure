#!/usr/bin/env python3
"""h1287: Fine-grid weight sweep extending h1275 to include w=0.25.

h1281's per-category outer-holdout run tested w ∈ {0.0, 0.25, 0.5, 0.75, 1.0}
as category-level knobs and, as a by-product, reported an aggregate W025 =
21.66%±1.12% R@30 vs W050 21.54% (+0.124pp, p=0.076). h1275's 20-seed GLOBAL
sweep covered w ∈ {0.3, 0.4, 0.5, 0.6, 0.7} and found W040 = 21.68%±1.15%
R@30 and W030 = 21.66%±1.11% R@30. The W025 datapoint suggests the R@30
Pareto plateau may extend below w=0.3.

This script extends h1275's 20-seed GLOBAL scaffold with two extra weights:
  w ∈ {0.25, 0.30, 0.35, 0.40, 0.45, 0.50}

Two paired comparisons per weight:
  (a) vs concat_l2_raw anchor
  (b) vs W050 (current canonical)
  (c) vs W040 (h1275's R@30 Pareto optimum)

Promotion gate:
  If any w beats W040 on ΔR@30 at p<0.05 AND Δ > +0.05pp,
  supersede h1283's dual-recipe retrieval weight.

Outputs:
    data/analysis/h1287_fine_grid_r30_pareto.{json,md}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from clean_embedding_benchmark import (  # noqa: E402
    load_embeddings,
    mean_std,
    split_diseases,
)
from h1215_fusion_benchmark import build_concat_lookup, score_disease_single  # noqa: E402
from h1249_entropy_routed_benchmark import paired_t  # noqa: E402
from h1255_soft_blend_fusion import z_normalise  # noqa: E402

OUT_JSON = PROJECT_ROOT / "data/analysis/h1287_fine_grid_r30_pareto.json"
OUT_MD = PROJECT_ROOT / "data/analysis/h1287_fine_grid_r30_pareto.md"

ANCHOR = "concat_l2_raw"
WEIGHTS: Tuple[float, ...] = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
REFERENCE_W = 0.5
R30_PARETO_W = 0.4  # h1275 R@30 Pareto optimum


def mode_name(w: float) -> str:
    return f"W{int(round(w * 100)):03d}"


def evaluate_seed(
    *,
    seed: int,
    lu_a: Dict[str, np.ndarray],
    lu_concat: Dict[str, np.ndarray],
    all_diseases: List[str],
    knn_gt: Dict[str, Set[str]],
    eval_gt: Dict[str, Set[str]],
    k: int,
) -> Dict:
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

    modes: Tuple[str, ...] = (ANCHOR,) + tuple(mode_name(w) for w in WEIGHTS)

    per_drug_r30: Dict[str, List[float]] = {m: [] for m in modes}
    per_disease_auprc: Dict[str, List[float]] = {m: [] for m in modes}
    per_disease_auroc: Dict[str, List[float]] = {m: [] for m in modes}
    pool_score: Dict[str, List[np.ndarray]] = {m: [] for m in modes}
    pool_label: Dict[str, List[np.ndarray]] = {m: [] for m in modes}
    n_eval = 0

    for did in holdout_ids:
        if did not in lu_concat or did not in lu_a:
            continue
        gt_drugs = eval_gt.get(did, set()) & universe
        if not gt_drugs:
            continue
        n_eval += 1

        ds_a = score_disease_single(lu_a[did], train_emb_a, train_ids_ordered, train_gt, k)
        sv_a = np.zeros(n_cands, dtype=np.float32)
        for drug, sc in ds_a.items():
            if drug in cand_index:
                sv_a[cand_index[drug]] = sc
        ds_c = score_disease_single(lu_concat[did], train_emb_c, train_ids_ordered, train_gt, k)
        sv_c = np.zeros(n_cands, dtype=np.float32)
        for drug, sc in ds_c.items():
            if drug in cand_index:
                sv_c[cand_index[drug]] = sc

        z_a = z_normalise(sv_a)
        z_c = z_normalise(sv_c)

        label_vec = np.zeros(n_cands, dtype=np.int8)
        for d in gt_drugs:
            idx = cand_index.get(d)
            if idx is not None:
                label_vec[idx] = 1
        n_pos = int(label_vec.sum())
        if n_pos == 0:
            continue

        score_per_mode: Dict[str, np.ndarray] = {ANCHOR: sv_c}
        for w in WEIGHTS:
            score_per_mode[mode_name(w)] = w * z_a + (1.0 - w) * z_c

        n_gt = len(gt_drugs)
        for mode, score_vec in score_per_mode.items():
            order = np.argsort(-score_vec, kind="stable")
            rank_of_idx = {int(idx): r + 1 for r, idx in enumerate(order)}
            hits_30 = sum(
                1 for d in gt_drugs
                if cand_index.get(d) is not None and rank_of_idx.get(cand_index[d], 10**9) <= 30
            )
            per_drug_r30[mode].append(hits_30 / n_gt)
            if 0 < n_pos < n_cands:
                per_disease_auprc[mode].append(
                    float(average_precision_score(label_vec, score_vec))
                )
                per_disease_auroc[mode].append(
                    float(roc_auc_score(label_vec, score_vec))
                )
            elif n_pos == n_cands:
                per_disease_auprc[mode].append(1.0)
            pool_score[mode].append(score_vec.copy())
            pool_label[mode].append(label_vec.copy())

    out: Dict[str, Dict] = {}
    for m in modes:
        sf = np.concatenate(pool_score[m]) if pool_score[m] else np.zeros(0)
        lf = np.concatenate(pool_label[m]) if pool_label[m] else np.zeros(0)
        pa = float(average_precision_score(lf, sf)) if lf.sum() > 0 else 0.0
        pr = (
            float(roc_auc_score(lf, sf))
            if 0 < lf.sum() < len(lf)
            else 0.0
        )
        out[m] = {
            "per_drug_r30": float(np.mean(per_drug_r30[m])) if per_drug_r30[m] else 0.0,
            "per_disease_auprc_mean": float(np.mean(per_disease_auprc[m])) if per_disease_auprc[m] else 0.0,
            "per_disease_auroc_mean": float(np.mean(per_disease_auroc[m])) if per_disease_auroc[m] else 0.0,
            "pooled_auprc": pa,
            "pooled_auroc": pr,
            "n_test_diseases": len(per_drug_r30[m]),
        }
    return {
        "seed": seed,
        "n_eval": n_eval,
        "modes": out,
    }


def paired_t_vs_ref(
    per_seed_results: List[Dict],
    mode: str,
    ref: str,
) -> Dict[str, Dict]:
    d_r30 = [r["modes"][mode]["per_drug_r30"] - r["modes"][ref]["per_drug_r30"] for r in per_seed_results]
    d_pd_ap = [r["modes"][mode]["per_disease_auprc_mean"] - r["modes"][ref]["per_disease_auprc_mean"] for r in per_seed_results]
    d_pd_auc = [r["modes"][mode]["per_disease_auroc_mean"] - r["modes"][ref]["per_disease_auroc_mean"] for r in per_seed_results]
    d_pool_ap = [r["modes"][mode]["pooled_auprc"] - r["modes"][ref]["pooled_auprc"] for r in per_seed_results]
    d_pool_auc = [r["modes"][mode]["pooled_auroc"] - r["modes"][ref]["pooled_auroc"] for r in per_seed_results]
    return {
        "R@30": {"mean": paired_t(d_r30)["mean"], "p": paired_t(d_r30)["p_two_sided"]},
        "per_disease_AUPRC": {"mean": paired_t(d_pd_ap)["mean"], "p": paired_t(d_pd_ap)["p_two_sided"]},
        "per_disease_AUROC": {"mean": paired_t(d_pd_auc)["mean"], "p": paired_t(d_pd_auc)["p_two_sided"]},
        "pooled_AUPRC": {"mean": paired_t(d_pool_ap)["mean"], "p": paired_t(d_pool_ap)["p_two_sided"]},
        "pooled_AUROC": {"mean": paired_t(d_pool_auc)["mean"], "p": paired_t(d_pool_auc)["p_two_sided"]},
    }


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
    ref_mode = mode_name(REFERENCE_W)
    pareto_mode = mode_name(R30_PARETO_W)
    mode_list = (ANCHOR,) + tuple(mode_name(w) for w in WEIGHTS)

    print("=" * 72)
    print(f"h1287: fine-grid weight sweep on SUBSET_D_GLOBAL")
    print(f"  weights={WEIGHTS}  ref={ref_mode}  R@30_pareto={pareto_mode}")
    print(f"  seeds={len(seeds)}")
    print("=" * 72)

    t0 = time.time()
    print("Loading embeddings...")
    lu_a, _, _ = load_embeddings(args.prefix_a)
    lu_b, _, _ = load_embeddings(args.prefix_b)
    lu_concat = build_concat_lookup(lu_a, lu_b)
    print(f"  A={len(lu_a):,}  B={len(lu_b):,}  concat={len(lu_concat):,}")

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

    per_seed_results: List[Dict] = []
    for seed in seeds:
        ts = time.time()
        seed_out = evaluate_seed(
            seed=seed, lu_a=lu_a, lu_concat=lu_concat,
            all_diseases=all_diseases, knn_gt=knn_gt, eval_gt=eval_gt,
            k=args.k,
        )
        per_seed_results.append(seed_out)
        print(f"\n  seed {seed} ({time.time()-ts:.1f}s) n_eval={seed_out['n_eval']}")
        for m in mode_list:
            mm = seed_out["modes"][m]
            print(
                f"    {m:6s} R@30={mm['per_drug_r30']*100:.2f}%  "
                f"per-dis-AUPRC={mm['per_disease_auprc_mean']:.4f}  "
                f"per-dis-AUROC={mm['per_disease_auroc_mean']:.4f}"
            )

    print("\n" + "=" * 72)
    print(f"AGGREGATE (mean ± std across {len(seeds)} seeds)")
    print("=" * 72)
    agg: Dict[str, Dict] = {}
    for m in mode_list:
        rows = [r["modes"][m] for r in per_seed_results]
        agg[m] = {
            "r30": mean_std([r["per_drug_r30"] for r in rows]),
            "per_disease_auprc": mean_std([r["per_disease_auprc_mean"] for r in rows]),
            "per_disease_auroc": mean_std([r["per_disease_auroc_mean"] for r in rows]),
            "pooled_auprc": mean_std([r["pooled_auprc"] for r in rows]),
            "pooled_auroc": mean_std([r["pooled_auroc"] for r in rows]),
        }
        print(
            f"  {m:6s}  R@30={agg[m]['r30'][0]*100:5.2f}%±{agg[m]['r30'][1]*100:.2f}%  "
            f"per-dis-AUPRC={agg[m]['per_disease_auprc'][0]:.4f}±{agg[m]['per_disease_auprc'][1]:.4f}  "
            f"per-dis-AUROC={agg[m]['per_disease_auroc'][0]:.4f}±{agg[m]['per_disease_auroc'][1]:.4f}"
        )

    print("\n" + "=" * 72)
    print(f"PAIRED-T vs {ANCHOR}  (n={len(seeds)} seeds)")
    print("=" * 72)
    paired_vs_anchor: Dict[str, Dict] = {}
    for w in WEIGHTS:
        m = mode_name(w)
        rt = paired_t_vs_ref(per_seed_results, m, ANCHOR)
        paired_vs_anchor[m] = rt
        print(
            f"\n  {m} (w={w}) vs {ANCHOR}:"
            f"\n    R@30           Δ={rt['R@30']['mean']*100:+.4f}pp  p={rt['R@30']['p']:.4g}"
            f"\n    per-dis-AUPRC  Δ={rt['per_disease_AUPRC']['mean']:+.5f}  p={rt['per_disease_AUPRC']['p']:.4g}"
            f"\n    per-dis-AUROC  Δ={rt['per_disease_AUROC']['mean']:+.5f}  p={rt['per_disease_AUROC']['p']:.4g}"
        )

    print("\n" + "=" * 72)
    print(f"PAIRED-T vs {ref_mode} (canonical w={REFERENCE_W}, n={len(seeds)})")
    print("=" * 72)
    paired_vs_ref: Dict[str, Dict] = {}
    for w in WEIGHTS:
        if w == REFERENCE_W:
            continue
        m = mode_name(w)
        rt = paired_t_vs_ref(per_seed_results, m, ref_mode)
        paired_vs_ref[m] = rt
        print(
            f"\n  {m} (w={w}) vs {ref_mode}:"
            f"\n    R@30           Δ={rt['R@30']['mean']*100:+.4f}pp  p={rt['R@30']['p']:.4g}"
            f"\n    per-dis-AUPRC  Δ={rt['per_disease_AUPRC']['mean']:+.5f}  p={rt['per_disease_AUPRC']['p']:.4g}"
            f"\n    per-dis-AUROC  Δ={rt['per_disease_AUROC']['mean']:+.5f}  p={rt['per_disease_AUROC']['p']:.4g}"
        )

    print("\n" + "=" * 72)
    print(f"PAIRED-T vs {pareto_mode} (h1275 R@30 Pareto w={R30_PARETO_W}, n={len(seeds)})")
    print("=" * 72)
    paired_vs_pareto: Dict[str, Dict] = {}
    for w in WEIGHTS:
        if w == R30_PARETO_W:
            continue
        m = mode_name(w)
        rt = paired_t_vs_ref(per_seed_results, m, pareto_mode)
        paired_vs_pareto[m] = rt
        print(
            f"\n  {m} (w={w}) vs {pareto_mode}:"
            f"\n    R@30           Δ={rt['R@30']['mean']*100:+.4f}pp  p={rt['R@30']['p']:.4g}"
            f"\n    per-dis-AUPRC  Δ={rt['per_disease_AUPRC']['mean']:+.5f}  p={rt['per_disease_AUPRC']['p']:.4g}"
        )

    # Promotion gate: R@30-retrieval recipe replacement
    print("\n" + "=" * 72)
    print(f"R@30 RETRIEVAL GATE: beat W040 (h1283 retrieval recipe) ΔR@30 > +0.05pp at p<0.05")
    print("=" * 72)
    promotion: Dict[str, Dict] = {}
    for w in WEIGHTS:
        if w == R30_PARETO_W:
            continue
        m = mode_name(w)
        rt = paired_vs_pareto[m]
        r = rt["R@30"]
        passes = (r["mean"] * 100 > 0.05) and (r["p"] < 0.05)
        promotion[m] = {
            "weight": w,
            "delta_r30_pp": r["mean"] * 100,
            "p": r["p"],
            "passes_gate": bool(passes),
            "decision": f"PROMOTE w={w} over W040 retrieval recipe" if passes else f"STAY with W040",
        }
        print(f"  {m} (w={w}): ΔR@30={r['mean']*100:+.4f}pp  p={r['p']:.4g}  →  {promotion[m]['decision']}")

    # Best-per-metric summary
    print("\n" + "=" * 72)
    print("BEST-W-PER-METRIC summary")
    print("=" * 72)
    best_per_metric: Dict[str, Dict] = {}
    for metric_key, metric_label in [
        ("r30", "R@30"),
        ("per_disease_auprc", "per-dis-AUPRC"),
        ("per_disease_auroc", "per-dis-AUROC"),
    ]:
        best_w = max(
            WEIGHTS,
            key=lambda w: agg[mode_name(w)][metric_key][0]
        )
        best_mean = agg[mode_name(best_w)][metric_key][0]
        best_per_metric[metric_label] = {"best_w": best_w, "mean": best_mean}
        print(f"  {metric_label:14s}  best w={best_w}  mean={best_mean:.4f}")

    # Persist JSON
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({
            "hypothesis": "h1287",
            "weights": list(WEIGHTS),
            "reference_w": REFERENCE_W,
            "pareto_w": R30_PARETO_W,
            "anchor": ANCHOR,
            "seeds": seeds,
            "n_diseases_universe": len(all_diseases),
            "aggregate": {m: {k: list(v) if isinstance(v, tuple) else v for k, v in d.items()} for m, d in agg.items()},
            "paired_t_vs_anchor": paired_vs_anchor,
            "paired_t_vs_reference": paired_vs_ref,
            "paired_t_vs_pareto": paired_vs_pareto,
            "promotion": promotion,
            "best_per_metric": best_per_metric,
            "per_seed_summaries": per_seed_results,
        }, f, indent=2)
    print(f"\nWrote {OUT_JSON.relative_to(PROJECT_ROOT)}")

    # Markdown
    md: List[str] = []
    md.append("# h1287 — Fine-grid R@30 Pareto around w=0.25–0.50 (20-seed GLOBAL soft-blend)\n\n")
    md.append("**Premise:** h1275's 20-seed global sweep covered w ∈ {0.3, 0.4, 0.5, 0.6, 0.7} and ")
    md.append("found the R@30 Pareto peak at W040 (21.68% ± 1.15%) with W030 only -0.02pp behind. ")
    md.append("h1281's outer-holdout run incidentally reported W025 = 21.66%±1.12% R@30 (+0.124pp vs ")
    md.append("W050, p=0.076), hinting the plateau may extend below w=0.3. This script runs a 20-seed ")
    md.append(f"fine-grid sweep on the locked GLOBAL subset with w ∈ {list(WEIGHTS)}.\n\n")
    md.append("`blended(d) = w * z(n2v_score_d) + (1 - w) * z(concat_l2_score_d)` on every disease.\n\n")

    md.append(f"## Aggregate (mean ± std across {len(seeds)} seeds)\n\n")
    md.append("| Mode | R@30 | per-dis-AUPRC | per-dis-AUROC | pooled-AUPRC | pooled-AUROC |\n")
    md.append("|---|---|---|---|---|---|\n")
    for m in mode_list:
        a = agg[m]
        md.append(
            f"| `{m}` | "
            f"{a['r30'][0]*100:.2f}%±{a['r30'][1]*100:.2f}% | "
            f"{a['per_disease_auprc'][0]:.4f}±{a['per_disease_auprc'][1]:.4f} | "
            f"{a['per_disease_auroc'][0]:.4f}±{a['per_disease_auroc'][1]:.4f} | "
            f"{a['pooled_auprc'][0]:.4f}±{a['pooled_auprc'][1]:.4f} | "
            f"{a['pooled_auroc'][0]:.4f}±{a['pooled_auroc'][1]:.4f} |\n"
        )

    md.append(f"\n## Paired-t vs `{ANCHOR}` (n={len(seeds)} seeds)\n\n")
    md.append("| Mode (w) | ΔR@30 (p) | Δper-dis-AUPRC (p) | Δper-dis-AUROC (p) |\n")
    md.append("|---|---|---|---|\n")
    for w in WEIGHTS:
        m = mode_name(w)
        rt = paired_vs_anchor[m]
        md.append(
            f"| `{m}` (w={w}) | "
            f"{rt['R@30']['mean']*100:+.4f}pp ({rt['R@30']['p']:.3g}) | "
            f"{rt['per_disease_AUPRC']['mean']:+.5f} ({rt['per_disease_AUPRC']['p']:.3g}) | "
            f"{rt['per_disease_AUROC']['mean']:+.5f} ({rt['per_disease_AUROC']['p']:.3g}) |\n"
        )

    md.append(f"\n## Paired-t vs `{ref_mode}` (canonical w={REFERENCE_W}, n={len(seeds)})\n\n")
    md.append("| Mode (w) | ΔR@30 (p) | Δper-dis-AUPRC (p) | Δper-dis-AUROC (p) |\n")
    md.append("|---|---|---|---|\n")
    for w in WEIGHTS:
        if w == REFERENCE_W:
            continue
        m = mode_name(w)
        rt = paired_vs_ref[m]
        md.append(
            f"| `{m}` (w={w}) | "
            f"{rt['R@30']['mean']*100:+.4f}pp ({rt['R@30']['p']:.3g}) | "
            f"{rt['per_disease_AUPRC']['mean']:+.5f} ({rt['per_disease_AUPRC']['p']:.3g}) | "
            f"{rt['per_disease_AUROC']['mean']:+.5f} ({rt['per_disease_AUROC']['p']:.3g}) |\n"
        )

    md.append(f"\n## Paired-t vs `{pareto_mode}` (h1275 R@30 Pareto w={R30_PARETO_W})\n\n")
    md.append("| Mode (w) | ΔR@30 (p) | Δper-dis-AUPRC (p) |\n")
    md.append("|---|---|---|\n")
    for w in WEIGHTS:
        if w == R30_PARETO_W:
            continue
        m = mode_name(w)
        rt = paired_vs_pareto[m]
        md.append(
            f"| `{m}` (w={w}) | "
            f"{rt['R@30']['mean']*100:+.4f}pp ({rt['R@30']['p']:.3g}) | "
            f"{rt['per_disease_AUPRC']['mean']:+.5f} ({rt['per_disease_AUPRC']['p']:.3g}) |\n"
        )

    md.append(f"\n## R@30 retrieval-recipe promotion gate (beat W040 ΔR@30 > +0.05pp at p<0.05)\n\n")
    md.append("| Mode (w) | ΔR@30 | p | Decision |\n|---|---|---|---|\n")
    for m, info in promotion.items():
        md.append(f"| `{m}` (w={info['weight']}) | {info['delta_r30_pp']:+.4f}pp | {info['p']:.3g} | **{info['decision']}** |\n")

    md.append(f"\n## Best-w per metric\n\n| Metric | Best w | Mean |\n|---|---|---|\n")
    for label, info in best_per_metric.items():
        md.append(f"| {label} | {info['best_w']} | {info['mean']:.4f} |\n")

    with open(OUT_MD, "w") as f:
        f.write("".join(md))
    print(f"Wrote {OUT_MD.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
