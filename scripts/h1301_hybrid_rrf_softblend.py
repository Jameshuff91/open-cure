#!/usr/bin/env python3
"""h1301: Hybrid RRF + soft-blend — adds soft_blend_w050 as a 4th ranker.

h1299 found `rrf_k60_3ranker` (N2V + FastRP + concat_l2) produces the best
DRKG-only per-dis AUPRC (0.1339) and per-dis AUROC (0.6429) of any fusion
mode tested, but is statistically tied with soft_blend_w050_2way on R@30
(Δ=-0.115pp p=0.201). The R@30 gap to soft-blend comes from soft-blend's
z-norm averaging preserving rank-30 boundary sharpness that rank-aggregation
tiebreaks away.

Hypothesis: adding soft_blend_w050 as a 4th ranker in RRF injects the top-30
sharpness back into rank-aggregation. The 4 rankers vote and the aggregate
gets both soft-blend's retrieval + RRF's ranking quality.

Caveat: soft-blend is a weighted sum of N2V + concat_l2 scores. Its ranks are
correlated with concat_l2 ranks. Adding it as a 4th ranker may over-weight
the N2V direction (concat_l2 already contains N2V via L2-concat) rather than
add orthogonal information.

Modes (20 seeds, SUBSET_D_GLOBAL):

    soft_blend_w050_2way     canonical
    rrf_k60_3ranker          h1299 best ranker (N2V + FastRP + concat_l2)
    rrf_k60_4ranker          candidate: N2V + FastRP + concat_l2 + soft_blend_w050
    rrf_k60_softblend_only   sanity check — should equal soft-blend exactly

Promotion gate (preregistered):
    4-ranker beats rrf_3ranker on ΔR@30 > +0.1pp at p<0.05 AND
    preserves ΔAUPRC ≥ 0 at p<0.05 vs rrf_3ranker.

Or equivalently: clears the h1299 preregistered dual-arm gate (ΔR@30 > +0.15pp
AND ΔAUPRC > +0.0005 at p<0.05 vs soft_blend_w050_2way canonical).

Outputs:
    data/analysis/h1301_hybrid_rrf_softblend.{json,md}
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
from h1299_rank_aggregation import rrf_score, score_to_ranks  # noqa: E402

OUT_JSON = PROJECT_ROOT / "data/analysis/h1301_hybrid_rrf_softblend.json"
OUT_MD = PROJECT_ROOT / "data/analysis/h1301_hybrid_rrf_softblend.md"

K_RRF = 60
BLEND_W = 0.5

MODES = (
    "soft_blend_w050_2way",
    "rrf_k60_3ranker",
    "rrf_k60_4ranker",
    "rrf_k60_softblend_only",
)


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

        z_n2v = z_normalise(sv_n2v)
        z_concat = z_normalise(sv_concat)
        sv_softblend = BLEND_W * z_n2v + (1.0 - BLEND_W) * z_concat

        ranks_n2v = score_to_ranks(sv_n2v)
        ranks_fastrp = score_to_ranks(sv_fastrp)
        ranks_concat = score_to_ranks(sv_concat)
        ranks_softblend = score_to_ranks(sv_softblend)

        score_per_mode: Dict[str, np.ndarray] = {
            "soft_blend_w050_2way": sv_softblend,
            "rrf_k60_3ranker": rrf_score([ranks_n2v, ranks_fastrp, ranks_concat]),
            "rrf_k60_4ranker": rrf_score([ranks_n2v, ranks_fastrp, ranks_concat, ranks_softblend]),
            "rrf_k60_softblend_only": rrf_score([ranks_softblend]),
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
    print(f"h1301: Hybrid RRF + soft-blend as 4th ranker  seeds={len(seeds)}")
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
        }
        print(
            f"  {m:24s}  R@30={agg[m]['r30'][0]*100:5.2f}%±{agg[m]['r30'][1]*100:.2f}%  "
            f"per-dis-AUPRC={agg[m]['per_disease_auprc'][0]:.4f}±{agg[m]['per_disease_auprc'][1]:.4f}  "
            f"per-dis-AUROC={agg[m]['per_disease_auroc'][0]:.4f}±{agg[m]['per_disease_auroc'][1]:.4f}"
        )

    # Paired-t comparisons
    comparisons = [
        ("rrf_k60_4ranker", "rrf_k60_3ranker", "4-ranker vs 3-ranker (R@30-recovery test)"),
        ("rrf_k60_4ranker", "soft_blend_w050_2way", "4-ranker vs soft-blend canonical (dual-gate)"),
        ("rrf_k60_softblend_only", "soft_blend_w050_2way", "sanity check — RRF(softblend) ≡ softblend"),
    ]
    print("\n" + "=" * 72)
    print("PAIRED-T COMPARISONS")
    print("=" * 72)
    paired: Dict[str, Dict] = {}
    for mode, ref, label in comparisons:
        rt = paired_t_vs(per_seed_results, mode, ref)
        paired[f"{mode}_vs_{ref}"] = rt
        print(
            f"\n  {label}:"
            f"\n    R@30           Δ={rt['R@30']['mean']*100:+.4f}pp  p={rt['R@30']['p']:.4g}"
            f"\n    per-dis-AUPRC  Δ={rt['per_disease_AUPRC']['mean']:+.5f}  p={rt['per_disease_AUPRC']['p']:.4g}"
            f"\n    per-dis-AUROC  Δ={rt['per_disease_AUROC']['mean']:+.5f}  p={rt['per_disease_AUROC']['p']:.4g}"
        )

    # Promotion gate
    print("\n" + "=" * 72)
    print("PROMOTION GATES")
    print("=" * 72)
    gate_a = paired["rrf_k60_4ranker_vs_rrf_k60_3ranker"]
    r30_a = gate_a["R@30"]["mean"] * 100 > 0.1 and gate_a["R@30"]["p"] < 0.05
    ap_nore_a = gate_a["per_disease_AUPRC"]["mean"] >= 0 and gate_a["per_disease_AUPRC"]["p"] < 0.05
    print(f"  GATE A: 4-ranker beats 3-ranker on ΔR@30 > +0.1pp p<0.05 AND preserves AUPRC (Δ≥0 p<0.05)")
    print(f"    R@30 Δ={gate_a['R@30']['mean']*100:+.4f}pp p={gate_a['R@30']['p']:.3g}  {'PASS' if r30_a else 'FAIL'}")
    print(f"    AUPRC Δ={gate_a['per_disease_AUPRC']['mean']:+.5f} p={gate_a['per_disease_AUPRC']['p']:.3g}  {'PASS' if ap_nore_a else 'FAIL'}")

    gate_b = paired["rrf_k60_4ranker_vs_soft_blend_w050_2way"]
    r30_b = gate_b["R@30"]["mean"] * 100 > 0.15 and gate_b["R@30"]["p"] < 0.05
    ap_b = gate_b["per_disease_AUPRC"]["mean"] > 0.0005 and gate_b["per_disease_AUPRC"]["p"] < 0.05
    dual_pass = r30_b and ap_b
    print(f"\n  GATE B: 4-ranker clears h1299 dual-arm gate vs soft-blend (ΔR@30>+0.15pp p<0.05 AND ΔAUPRC>+0.0005 p<0.05)")
    print(f"    R@30 Δ={gate_b['R@30']['mean']*100:+.4f}pp p={gate_b['R@30']['p']:.3g}  {'PASS' if r30_b else 'FAIL'}")
    print(f"    AUPRC Δ={gate_b['per_disease_AUPRC']['mean']:+.5f} p={gate_b['per_disease_AUPRC']['p']:.3g}  {'PASS' if ap_b else 'FAIL'}")
    print(f"    Dual-arm overall: {'PASS — 4-ranker is the new canonical' if dual_pass else 'FAIL — triple-recipe stays'}")

    # JSON
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({
            "hypothesis": "h1301",
            "k_rrf": K_RRF,
            "seeds": seeds,
            "n_diseases_universe": len(all_diseases),
            "aggregate": {m: {k: list(v) if isinstance(v, tuple) else v for k, v in d.items()} for m, d in agg.items()},
            "paired_t": paired,
            "gates": {
                "gate_a_vs_rrf_3ranker": {
                    "r30_pass": bool(r30_a),
                    "auprc_no_regress": bool(ap_nore_a),
                    "overall_pass": bool(r30_a and ap_nore_a),
                },
                "gate_b_vs_soft_blend_canonical": {
                    "r30_pass": bool(r30_b),
                    "auprc_pass": bool(ap_b),
                    "dual_arm_pass": bool(dual_pass),
                },
            },
            "per_seed_summaries": per_seed_results,
        }, f, indent=2)
    print(f"\nWrote {OUT_JSON.relative_to(PROJECT_ROOT)}")

    # MD
    md: List[str] = []
    md.append("# h1301 — Hybrid 4-ranker RRF with soft-blend as 4th voter (20-seed SUBSET_D_GLOBAL)\n\n")
    md.append(f"## Aggregate (mean ± std across {len(seeds)} seeds)\n\n")
    md.append("| Mode | R@30 | per-dis-AUPRC | per-dis-AUROC |\n|---|---|---|---|\n")
    for m in MODES:
        a = agg[m]
        md.append(
            f"| `{m}` | "
            f"{a['r30'][0]*100:.2f}%±{a['r30'][1]*100:.2f}% | "
            f"{a['per_disease_auprc'][0]:.4f}±{a['per_disease_auprc'][1]:.4f} | "
            f"{a['per_disease_auroc'][0]:.4f}±{a['per_disease_auroc'][1]:.4f} |\n"
        )
    md.append(f"\n## Paired-t comparisons\n\n")
    md.append("| Comparison | ΔR@30 (p) | Δper-dis-AUPRC (p) | Δper-dis-AUROC (p) |\n|---|---|---|---|\n")
    for mode, ref, label in comparisons:
        rt = paired[f"{mode}_vs_{ref}"]
        md.append(
            f"| {label} | "
            f"{rt['R@30']['mean']*100:+.4f}pp ({rt['R@30']['p']:.3g}) | "
            f"{rt['per_disease_AUPRC']['mean']:+.5f} ({rt['per_disease_AUPRC']['p']:.3g}) | "
            f"{rt['per_disease_AUROC']['mean']:+.5f} ({rt['per_disease_AUROC']['p']:.3g}) |\n"
        )
    md.append("\n## Gates\n\n")
    md.append(
        f"**Gate A** (4-ranker beats 3-ranker on ΔR@30>+0.1pp at p<0.05 AND AUPRC Δ≥0 at p<0.05): "
        f"R@30 {'PASS' if r30_a else 'FAIL'}, AUPRC-no-regress {'PASS' if ap_nore_a else 'FAIL'}\n\n"
        f"**Gate B** (4-ranker clears h1299 dual-arm gate vs soft-blend ΔR@30>+0.15pp AND ΔAUPRC>+0.0005 both p<0.05): "
        f"R@30 {'PASS' if r30_b else 'FAIL'}, AUPRC {'PASS' if ap_b else 'FAIL'}, "
        f"overall {'PASS — 4-ranker becomes the new single canonical' if dual_pass else 'FAIL — triple-recipe stays'}\n"
    )
    with open(OUT_MD, "w") as f:
        f.write("".join(md))
    print(f"Wrote {OUT_MD.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
