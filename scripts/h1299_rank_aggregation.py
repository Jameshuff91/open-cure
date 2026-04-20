#!/usr/bin/env python3
"""h1299: RRF + Borda rank-aggregation of Node2Vec + FastRP (+ concat_l2).

h1275/h1287 exhausted the linear weight axis on 2-way soft-blend. h1293
exhausted the linear stacking axis (TransE as third embedding regressed).
Both use continuous score combiners. Non-linear rank aggregation is the
remaining untested DRKG-only recall lever.

    RRF:   score_rrf(d) = Σ_i 1 / (k_rrf + rank_i(d))            (k_rrf=60)
    Borda: score_borda(d) = Σ_i (N_cands - rank_i(d))

Rank-aggregation is SCALE-INVARIANT (only order matters), which is the
exact property h1259 found fusion needs — pooled AUPRC/AUROC collapsed
under z-norm because of cross-disease score-scale heterogeneity.

Compared modes (20 seeds, SUBSET_D_GLOBAL):

    concat_l2_2way       anchor
    soft_blend_w050_2way canonical
    rrf_k60_n2v_fastrp   2-ranker RRF
    rrf_k60_n2v_concat   2-ranker RRF over N2V + concat_l2
    rrf_k60_3ranker      N2V + FastRP + concat_l2
    borda_n2v_fastrp     2-ranker Borda
    borda_3ranker        N2V + FastRP + concat_l2

Promotion gate:
    RRF/Borda beats soft_blend_w050_2way on ΔR@30 > +0.15pp at p<0.05
    AND Δper-dis AUPRC > +0.0005 at p<0.05.

Outputs:
    data/analysis/h1299_rank_aggregation.{json,md}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Set

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

OUT_JSON = PROJECT_ROOT / "data/analysis/h1299_rank_aggregation.json"
OUT_MD = PROJECT_ROOT / "data/analysis/h1299_rank_aggregation.md"

K_RRF = 60
BLEND_W = 0.5

MODES = (
    "concat_l2_2way",
    "soft_blend_w050_2way",
    "rrf_k60_n2v_fastrp",
    "rrf_k60_n2v_concat",
    "rrf_k60_3ranker",
    "borda_n2v_fastrp",
    "borda_3ranker",
)


def score_to_ranks(score_vec: np.ndarray) -> np.ndarray:
    """Return 1-indexed ranks among candidates with NON-ZERO scores.

    Candidates with score == 0 (i.e. not in any top-k training neighbour's GT)
    get rank 0 as a sentinel — the caller decides how to treat them in
    rank-aggregation. Rationale: a rank of N/2 for a candidate with no kNN
    signal is noise, not "mediocre"; excluding it from the rank-aggregation
    is closer to how RRF/Borda are applied in practice (top-K union lists).
    """
    ranks = np.zeros(len(score_vec), dtype=np.int64)  # 0 = unscored sentinel
    nonzero_idx = np.flatnonzero(score_vec > 0)
    if len(nonzero_idx) == 0:
        return ranks
    order_within = np.argsort(-score_vec[nonzero_idx], kind="stable")
    for r, pos in enumerate(order_within, start=1):
        ranks[nonzero_idx[pos]] = r
    return ranks


def rrf_score(ranks_list: List[np.ndarray], k: int = K_RRF) -> np.ndarray:
    """Reciprocal rank fusion. Unscored candidates (rank=0) contribute 0."""
    total = np.zeros(len(ranks_list[0]), dtype=np.float32)
    for r in ranks_list:
        contrib = np.where(r > 0, 1.0 / (k + r.astype(np.float32)), 0.0)
        total += contrib
    return total


def borda_score(ranks_list: List[np.ndarray], n_cands: int) -> np.ndarray:
    """Borda count. Unscored candidates (rank=0) contribute 0."""
    total = np.zeros(len(ranks_list[0]), dtype=np.float32)
    for r in ranks_list:
        contrib = np.where(r > 0, (n_cands - r).astype(np.float32), 0.0)
        total += contrib
    return total


def evaluate_seed(
    *,
    seed: int,
    lu_n2v: Dict[str, np.ndarray],
    lu_fastrp: Dict[str, np.ndarray],
    lu_concat: Dict[str, np.ndarray],
    all_diseases: List[str],
    knn_gt: Dict[str, Set[str]],
    eval_gt: Dict[str, Set[str]],
    k: int,
) -> Dict:
    train_ids, holdout_ids = split_diseases(all_diseases, seed)
    common = set(lu_n2v) & set(lu_fastrp) & set(lu_concat)
    train_ids_ordered = [d for d in train_ids if d in common]

    train_emb_n2v = np.stack([lu_n2v[d] for d in train_ids_ordered])
    train_emb_fastrp = np.stack([lu_fastrp[d] for d in train_ids_ordered])
    train_emb_concat = np.stack([lu_concat[d] for d in train_ids_ordered])

    train_gt = {d: (knn_gt[d] & common) for d in train_ids_ordered if d in knn_gt}

    cand_drugs: Set[str] = set()
    for d in train_ids_ordered:
        if d in knn_gt:
            cand_drugs |= (knn_gt[d] & common)
    all_hold_drugs: Set[str] = set()
    for did in holdout_ids:
        all_hold_drugs |= eval_gt.get(did, set())
    universe = (cand_drugs | all_hold_drugs) & common
    cand_list = sorted(universe)
    cand_index = {d: i for i, d in enumerate(cand_list)}
    n_cands = len(cand_list)

    per_drug_r30: Dict[str, List[float]] = {m: [] for m in MODES}
    per_disease_auprc: Dict[str, List[float]] = {m: [] for m in MODES}
    per_disease_auroc: Dict[str, List[float]] = {m: [] for m in MODES}
    pool_score: Dict[str, List[np.ndarray]] = {m: [] for m in MODES}
    pool_label: Dict[str, List[np.ndarray]] = {m: [] for m in MODES}
    n_eval = 0

    for did in holdout_ids:
        if did not in common:
            continue
        gt_drugs = eval_gt.get(did, set()) & universe
        if not gt_drugs:
            continue
        n_eval += 1

        def raw_scores(lu: Dict[str, np.ndarray], train_emb: np.ndarray) -> np.ndarray:
            ds = score_disease_single(lu[did], train_emb, train_ids_ordered, train_gt, k)
            sv = np.zeros(n_cands, dtype=np.float32)
            for drug, sc in ds.items():
                if drug in cand_index:
                    sv[cand_index[drug]] = sc
            return sv

        sv_n2v = raw_scores(lu_n2v, train_emb_n2v)
        sv_fastrp = raw_scores(lu_fastrp, train_emb_fastrp)
        sv_concat = raw_scores(lu_concat, train_emb_concat)

        ranks_n2v = score_to_ranks(sv_n2v)
        ranks_fastrp = score_to_ranks(sv_fastrp)
        ranks_concat = score_to_ranks(sv_concat)

        z_n2v = z_normalise(sv_n2v)
        z_concat = z_normalise(sv_concat)

        score_per_mode: Dict[str, np.ndarray] = {
            "concat_l2_2way": sv_concat,
            "soft_blend_w050_2way": BLEND_W * z_n2v + (1.0 - BLEND_W) * z_concat,
            "rrf_k60_n2v_fastrp": rrf_score([ranks_n2v, ranks_fastrp]),
            "rrf_k60_n2v_concat": rrf_score([ranks_n2v, ranks_concat]),
            "rrf_k60_3ranker": rrf_score([ranks_n2v, ranks_fastrp, ranks_concat]),
            "borda_n2v_fastrp": borda_score([ranks_n2v, ranks_fastrp], n_cands),
            "borda_3ranker": borda_score([ranks_n2v, ranks_fastrp, ranks_concat], n_cands),
        }

        label_vec = np.zeros(n_cands, dtype=np.int8)
        for d in gt_drugs:
            idx = cand_index.get(d)
            if idx is not None:
                label_vec[idx] = 1
        n_pos = int(label_vec.sum())
        if n_pos == 0:
            continue

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
                per_disease_auprc[mode].append(float(average_precision_score(label_vec, score_vec)))
                per_disease_auroc[mode].append(float(roc_auc_score(label_vec, score_vec)))
            elif n_pos == n_cands:
                per_disease_auprc[mode].append(1.0)
            pool_score[mode].append(score_vec.copy())
            pool_label[mode].append(label_vec.copy())

    out: Dict[str, Dict] = {}
    for m in MODES:
        sf = np.concatenate(pool_score[m]) if pool_score[m] else np.zeros(0)
        lf = np.concatenate(pool_label[m]) if pool_label[m] else np.zeros(0)
        pa = float(average_precision_score(lf, sf)) if lf.sum() > 0 else 0.0
        pr = float(roc_auc_score(lf, sf)) if 0 < lf.sum() < len(lf) else 0.0
        out[m] = {
            "per_drug_r30": float(np.mean(per_drug_r30[m])) if per_drug_r30[m] else 0.0,
            "per_disease_auprc_mean": float(np.mean(per_disease_auprc[m])) if per_disease_auprc[m] else 0.0,
            "per_disease_auroc_mean": float(np.mean(per_disease_auroc[m])) if per_disease_auroc[m] else 0.0,
            "pooled_auprc": pa,
            "pooled_auroc": pr,
            "n_test_diseases": len(per_drug_r30[m]),
        }
    return {"seed": seed, "n_eval": n_eval, "modes": out}


def paired_t_vs(results: List[Dict], mode: str, ref: str) -> Dict[str, Dict]:
    d_r30 = [r["modes"][mode]["per_drug_r30"] - r["modes"][ref]["per_drug_r30"] for r in results]
    d_pd_ap = [r["modes"][mode]["per_disease_auprc_mean"] - r["modes"][ref]["per_disease_auprc_mean"] for r in results]
    d_pd_auc = [r["modes"][mode]["per_disease_auroc_mean"] - r["modes"][ref]["per_disease_auroc_mean"] for r in results]
    return {
        "R@30": {"mean": paired_t(d_r30)["mean"], "p": paired_t(d_r30)["p_two_sided"]},
        "per_disease_AUPRC": {"mean": paired_t(d_pd_ap)["mean"], "p": paired_t(d_pd_ap)["p_two_sided"]},
        "per_disease_AUROC": {"mean": paired_t(d_pd_auc)["mean"], "p": paired_t(d_pd_auc)["p_two_sided"]},
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
    print("=" * 72)
    print(f"h1299: RRF + Borda rank-aggregation of N2V + FastRP")
    print(f"  seeds={len(seeds)}  k_rrf={K_RRF}")
    print("=" * 72)

    t0 = time.time()
    lu_n2v, _, _ = load_embeddings(args.prefix_a)
    lu_fastrp, _, _ = load_embeddings(args.prefix_b)
    lu_concat = build_concat_lookup(lu_n2v, lu_fastrp)

    with open(PROJECT_ROOT / args.eval_gt) as f:
        eval_gt = {d: set(v) for d, v in json.load(f).items()}
    with open(PROJECT_ROOT / args.knn_gt) as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "ground_truth" in raw:
        raw = raw["ground_truth"]
    knn_gt = {d: set(v) for d, v in raw.items()}

    intersection_keys = set(lu_n2v) & set(lu_fastrp)
    all_diseases = [d for d in knn_gt if d in intersection_keys]
    print(f"Universe: {len(all_diseases):,} diseases  | setup {time.time()-t0:.1f}s")

    per_seed_results: List[Dict] = []
    for seed in seeds:
        ts = time.time()
        seed_out = evaluate_seed(
            seed=seed, lu_n2v=lu_n2v, lu_fastrp=lu_fastrp, lu_concat=lu_concat,
            all_diseases=all_diseases, knn_gt=knn_gt, eval_gt=eval_gt, k=args.k,
        )
        per_seed_results.append(seed_out)
        print(f"\n  seed {seed} ({time.time()-ts:.1f}s) n_eval={seed_out['n_eval']}")
        for m in MODES:
            mm = seed_out["modes"][m]
            print(
                f"    {m:24s} R@30={mm['per_drug_r30']*100:.2f}%  "
                f"per-dis-AUPRC={mm['per_disease_auprc_mean']:.4f}  "
                f"per-dis-AUROC={mm['per_disease_auroc_mean']:.4f}"
            )

    # Aggregate
    print("\n" + "=" * 72)
    print(f"AGGREGATE (mean ± std, {len(seeds)} seeds)")
    print("=" * 72)
    agg: Dict[str, Dict] = {}
    for m in MODES:
        rows = [r["modes"][m] for r in per_seed_results]
        agg[m] = {
            "r30": mean_std([r["per_drug_r30"] for r in rows]),
            "per_disease_auprc": mean_std([r["per_disease_auprc_mean"] for r in rows]),
            "per_disease_auroc": mean_std([r["per_disease_auroc_mean"] for r in rows]),
            "pooled_auprc": mean_std([r["pooled_auprc"] for r in rows]),
            "pooled_auroc": mean_std([r["pooled_auroc"] for r in rows]),
        }
        print(
            f"  {m:24s}  R@30={agg[m]['r30'][0]*100:5.2f}%±{agg[m]['r30'][1]*100:.2f}%  "
            f"per-dis-AUPRC={agg[m]['per_disease_auprc'][0]:.4f}±{agg[m]['per_disease_auprc'][1]:.4f}  "
            f"per-dis-AUROC={agg[m]['per_disease_auroc'][0]:.4f}±{agg[m]['per_disease_auroc'][1]:.4f}"
        )

    # Paired-t vs anchor and vs canonical
    print("\n" + "=" * 72)
    print("PAIRED-T vs soft_blend_w050_2way (canonical)")
    print("=" * 72)
    paired_vs_canonical: Dict[str, Dict] = {}
    for m in MODES:
        if m == "soft_blend_w050_2way":
            continue
        rt = paired_t_vs(per_seed_results, m, "soft_blend_w050_2way")
        paired_vs_canonical[m] = rt
        print(
            f"\n  {m} vs canonical:"
            f"\n    R@30           Δ={rt['R@30']['mean']*100:+.4f}pp  p={rt['R@30']['p']:.4g}"
            f"\n    per-dis-AUPRC  Δ={rt['per_disease_AUPRC']['mean']:+.5f}  p={rt['per_disease_AUPRC']['p']:.4g}"
            f"\n    per-dis-AUROC  Δ={rt['per_disease_AUROC']['mean']:+.5f}  p={rt['per_disease_AUROC']['p']:.4g}"
        )

    print("\n" + "=" * 72)
    print("PAIRED-T vs concat_l2_2way (raw anchor)")
    print("=" * 72)
    paired_vs_anchor: Dict[str, Dict] = {}
    for m in MODES:
        if m == "concat_l2_2way":
            continue
        rt = paired_t_vs(per_seed_results, m, "concat_l2_2way")
        paired_vs_anchor[m] = rt
        print(
            f"\n  {m} vs anchor:"
            f"\n    R@30           Δ={rt['R@30']['mean']*100:+.4f}pp  p={rt['R@30']['p']:.4g}"
            f"\n    per-dis-AUPRC  Δ={rt['per_disease_AUPRC']['mean']:+.5f}  p={rt['per_disease_AUPRC']['p']:.4g}"
        )

    # Promotion gate
    print("\n" + "=" * 72)
    print("PROMOTION GATE: any RRF/Borda beats canonical at ΔR@30 > +0.15pp AND ΔAUPRC > +0.0005 both p<0.05")
    print("=" * 72)
    promotion: Dict[str, Dict] = {}
    rank_modes = [m for m in MODES if m.startswith("rrf") or m.startswith("borda")]
    for m in rank_modes:
        rt = paired_vs_canonical[m]
        r30_pass = rt["R@30"]["mean"] * 100 > 0.15 and rt["R@30"]["p"] < 0.05
        ap_pass = rt["per_disease_AUPRC"]["mean"] > 0.0005 and rt["per_disease_AUPRC"]["p"] < 0.05
        passes = r30_pass and ap_pass
        promotion[m] = {
            "delta_r30_pp": rt["R@30"]["mean"] * 100,
            "r30_p": rt["R@30"]["p"],
            "r30_pass": bool(r30_pass),
            "delta_auprc": rt["per_disease_AUPRC"]["mean"],
            "auprc_p": rt["per_disease_AUPRC"]["p"],
            "auprc_pass": bool(ap_pass),
            "passes_gate": bool(passes),
            "decision": f"PROMOTE {m} as new canonical" if passes else f"STAY with soft_blend_w050_2way",
        }
        print(f"  {m}:")
        print(f"    ΔR@30={rt['R@30']['mean']*100:+.4f}pp p={rt['R@30']['p']:.4g}  R@30_gate={'PASS' if r30_pass else 'FAIL'}")
        print(f"    ΔAUPRC={rt['per_disease_AUPRC']['mean']:+.5f} p={rt['per_disease_AUPRC']['p']:.4g}  AUPRC_gate={'PASS' if ap_pass else 'FAIL'}")
        print(f"    → {promotion[m]['decision']}")

    # JSON
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({
            "hypothesis": "h1299",
            "k_rrf": K_RRF,
            "seeds": seeds,
            "n_diseases_universe": len(all_diseases),
            "aggregate": {m: {k: list(v) if isinstance(v, tuple) else v for k, v in d.items()} for m, d in agg.items()},
            "paired_t_vs_canonical": paired_vs_canonical,
            "paired_t_vs_anchor": paired_vs_anchor,
            "promotion": promotion,
            "per_seed_summaries": per_seed_results,
        }, f, indent=2)
    print(f"\nWrote {OUT_JSON.relative_to(PROJECT_ROOT)}")

    # Markdown
    md: List[str] = []
    md.append("# h1299 — RRF + Borda rank-aggregation on N2V + FastRP (20-seed SUBSET_D_GLOBAL)\n\n")
    md.append(f"**k_rrf = {K_RRF}**, `N = {len(all_diseases):,}` diseases\n\n")
    md.append(f"## Aggregate (mean ± std across {len(seeds)} seeds)\n\n")
    md.append("| Mode | R@30 | per-dis-AUPRC | per-dis-AUROC | pooled-AUPRC | pooled-AUROC |\n")
    md.append("|---|---|---|---|---|---|\n")
    for m in MODES:
        a = agg[m]
        md.append(
            f"| `{m}` | "
            f"{a['r30'][0]*100:.2f}%±{a['r30'][1]*100:.2f}% | "
            f"{a['per_disease_auprc'][0]:.4f}±{a['per_disease_auprc'][1]:.4f} | "
            f"{a['per_disease_auroc'][0]:.4f}±{a['per_disease_auroc'][1]:.4f} | "
            f"{a['pooled_auprc'][0]:.4f}±{a['pooled_auprc'][1]:.4f} | "
            f"{a['pooled_auroc'][0]:.4f}±{a['pooled_auroc'][1]:.4f} |\n"
        )

    md.append(f"\n## Paired-t vs `soft_blend_w050_2way` (canonical, n={len(seeds)})\n\n")
    md.append("| Mode | ΔR@30 (p) | Δper-dis-AUPRC (p) | Δper-dis-AUROC (p) |\n|---|---|---|---|\n")
    for m in MODES:
        if m == "soft_blend_w050_2way":
            continue
        rt = paired_vs_canonical[m]
        md.append(
            f"| `{m}` | "
            f"{rt['R@30']['mean']*100:+.4f}pp ({rt['R@30']['p']:.3g}) | "
            f"{rt['per_disease_AUPRC']['mean']:+.5f} ({rt['per_disease_AUPRC']['p']:.3g}) | "
            f"{rt['per_disease_AUROC']['mean']:+.5f} ({rt['per_disease_AUROC']['p']:.3g}) |\n"
        )

    md.append("\n## Promotion gate\n\n")
    md.append("Pass if ΔR@30>+0.15pp AND ΔAUPRC>+0.0005 both at p<0.05 vs canonical soft_blend_w050_2way.\n\n")
    md.append("| Mode | ΔR@30 | R@30 pass | ΔAUPRC | AUPRC pass | Decision |\n|---|---|---|---|---|---|\n")
    for m, info in promotion.items():
        md.append(
            f"| `{m}` | "
            f"{info['delta_r30_pp']:+.4f}pp (p={info['r30_p']:.3g}) | "
            f"{'PASS' if info['r30_pass'] else 'FAIL'} | "
            f"{info['delta_auprc']:+.5f} (p={info['auprc_p']:.3g}) | "
            f"{'PASS' if info['auprc_pass'] else 'FAIL'} | "
            f"**{info['decision']}** |\n"
        )

    with open(OUT_MD, "w") as f:
        f.write("".join(md))
    print(f"Wrote {OUT_MD.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
