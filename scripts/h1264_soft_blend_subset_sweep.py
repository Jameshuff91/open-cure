#!/usr/bin/env python3
"""h1264: Soft-blend subset sweep — does h1259's per-disease AUPRC win
generalise beyond the (n_gt>=51 ∩ mid-entropy) subset?

h1259 validated soft_blend_w0.5 (per-disease z-norm of n2v + concat_l2)
ONLY on the n_gt>=51 mid-entropy "flipped" subset (~7-16 diseases per seed):
Δ per-dis AUPRC = +0.00054 p=0.013, Δ per-dis AUROC = +0.00059 p=0.0013.
Outside the subset, scores were concat_l2_znorm.

This script tests four progressively broader subsets of disease IDs eligible
for blending (everything outside still uses concat_l2_raw, NOT znorm — to
avoid h1255's z-norm pooled-AUPRC collapse on the global pool):

    SUBSET_A_FLIPPED   — h1259 baseline (n_gt>=51 ∩ mid-entropy)
    SUBSET_B_HIGHDENS  — all n_gt>=51
    SUBSET_C_MODHIGH   — all n_gt>=21
    SUBSET_D_GLOBAL    — every disease

Inside the subset:
    score(d) = 0.5 * z(n2v_score(d)) + 0.5 * z(concat_score(d))
Outside the subset:
    score(d) = concat_l2_raw_score(d)        ← rank-equivariant baseline

Note we mix raw and z-norm scores per disease (same disease only ever sees
ONE scoring rule), and per-disease AUPRC is rank-equivariant within disease,
so cross-disease pooled AUPRC may still be confounded — we therefore report
both pooled and per-disease metrics, and use per-disease as primary.

Promotion gate: any subset with per-dis AUPRC Δ ≥ +0.0005 at p < 0.05 (vs
concat_l2_raw anchor) becomes the new fusion default.

Outputs:
    data/analysis/h1264_soft_blend_subset_sweep.{json,md}
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

from atc_features import ATCMapper  # noqa: E402
from clean_embedding_benchmark import (  # noqa: E402
    load_embeddings,
    mean_std,
    split_diseases,
)
from h1215_fusion_benchmark import build_concat_lookup, score_disease_single  # noqa: E402
from h1249_entropy_routed_benchmark import (  # noqa: E402
    N_GT_HIGH_DENSITY,
    disease_atc_l3_entropy,
    paired_t,
)
from h1255_soft_blend_fusion import z_normalise  # noqa: E402

OUT_JSON = PROJECT_ROOT / "data/analysis/h1264_soft_blend_subset_sweep.json"
OUT_MD = PROJECT_ROOT / "data/analysis/h1264_soft_blend_subset_sweep.md"

ANCHOR = "concat_l2_raw"
SUBSETS = ("SUBSET_A_FLIPPED", "SUBSET_B_HIGHDENS", "SUBSET_C_MODHIGH", "SUBSET_D_GLOBAL")
BLEND_W = 0.5  # h1259 winning weight


def in_subset(name: str, n_gt: int, ent: float, cut_low: float, cut_high: float) -> bool:
    if name == "SUBSET_A_FLIPPED":
        return n_gt >= N_GT_HIGH_DENSITY and (cut_low <= ent < cut_high)
    if name == "SUBSET_B_HIGHDENS":
        return n_gt >= N_GT_HIGH_DENSITY
    if name == "SUBSET_C_MODHIGH":
        return n_gt >= 21
    if name == "SUBSET_D_GLOBAL":
        return True
    raise ValueError(name)


def evaluate_seed(
    *,
    seed: int,
    lu_a: Dict[str, np.ndarray],
    lu_concat: Dict[str, np.ndarray],
    all_diseases: List[str],
    knn_gt: Dict[str, Set[str]],
    eval_gt: Dict[str, Set[str]],
    n_gt_per_disease: Dict[str, int],
    entropy_per_disease: Dict[str, float],
    k: int,
) -> Dict:
    train_ids, holdout_ids = split_diseases(all_diseases, seed)

    # Train-side n_gt>=51 entropy tercile cuts (no holdout leakage)
    train_entropies = sorted(
        entropy_per_disease[d]
        for d in train_ids
        if n_gt_per_disease.get(d, 0) >= N_GT_HIGH_DENSITY
    )
    n_thd = len(train_entropies)
    if n_thd >= 3:
        cut_low = train_entropies[n_thd // 3]
        cut_high = train_entropies[2 * n_thd // 3]
    else:
        cut_low, cut_high = 3.18, 3.95

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

    modes = (ANCHOR,) + SUBSETS
    per_drug_r30: Dict[str, List[float]] = {m: [] for m in modes}
    per_disease_auprc: Dict[str, List[float]] = {m: [] for m in modes}
    per_disease_auroc: Dict[str, List[float]] = {m: [] for m in modes}
    pool_score: Dict[str, List[np.ndarray]] = {m: [] for m in modes}
    pool_label: Dict[str, List[np.ndarray]] = {m: [] for m in modes}
    in_subset_count: Dict[str, int] = {m: 0 for m in modes}
    n_eval = 0

    for did in holdout_ids:
        if did not in lu_concat or did not in lu_a:
            continue
        gt_drugs = eval_gt.get(did, set()) & universe
        if not gt_drugs:
            continue
        n_eval += 1

        # Score under both embeddings
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
        blended = BLEND_W * z_a + (1.0 - BLEND_W) * z_c

        n_gt_d = n_gt_per_disease.get(did, 0)
        ent = entropy_per_disease.get(did, 0.0)

        # Build label vector
        label_vec = np.zeros(n_cands, dtype=np.int8)
        for d in gt_drugs:
            idx = cand_index.get(d)
            if idx is not None:
                label_vec[idx] = 1
        n_pos = int(label_vec.sum())
        if n_pos == 0:
            continue

        # Anchor: concat_l2_raw on every disease
        score_per_mode: Dict[str, np.ndarray] = {ANCHOR: sv_c}
        for sub in SUBSETS:
            if in_subset(sub, n_gt_d, ent, cut_low, cut_high):
                score_per_mode[sub] = blended
                in_subset_count[sub] += 1
            else:
                score_per_mode[sub] = sv_c

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
            "in_subset_count": in_subset_count[m],
            "n_test_diseases": len(per_drug_r30[m]),
        }
    return {
        "seed": seed,
        "n_eval": n_eval,
        "cut_low": cut_low,
        "cut_high": cut_high,
        "modes": out,
    }


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
    print(f"h1264: Soft-blend subset sweep  prefix_a={args.prefix_a}  prefix_b={args.prefix_b}")
    print(f"  blend_w={BLEND_W}  subsets={SUBSETS}  seeds={seeds}")
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

    with open(PROJECT_ROOT / args.db_lookup) as f:
        db_lookup = json.load(f)
    mapper = ATCMapper()
    intersection_keys = set(lu_a) & set(lu_b)
    all_diseases = [d for d in knn_gt if d in intersection_keys]
    print(f"Universe: {len(all_diseases):,} diseases")

    print("Pre-computing per-disease n_gt + ATC L3 entropy...")
    n_gt_per_disease: Dict[str, int] = {}
    entropy_per_disease: Dict[str, float] = {}
    for d in all_diseases:
        gt_drugs = list(eval_gt.get(d, set()))
        n_gt_per_disease[d] = len(gt_drugs)
        ent, _ = disease_atc_l3_entropy(gt_drugs, db_lookup, mapper)
        entropy_per_disease[d] = ent
    print(f"  setup {time.time()-t0:.1f}s")

    per_seed_results: List[Dict] = []
    for seed in seeds:
        ts = time.time()
        seed_out = evaluate_seed(
            seed=seed, lu_a=lu_a, lu_concat=lu_concat,
            all_diseases=all_diseases, knn_gt=knn_gt, eval_gt=eval_gt,
            n_gt_per_disease=n_gt_per_disease, entropy_per_disease=entropy_per_disease,
            k=args.k,
        )
        per_seed_results.append(seed_out)
        print(f"\n  seed {seed} ({time.time()-ts:.1f}s) n_eval={seed_out['n_eval']}  "
              f"cuts=({seed_out['cut_low']:.3f},{seed_out['cut_high']:.3f})")
        for m in (ANCHOR,) + SUBSETS:
            mm = seed_out["modes"][m]
            print(
                f"    {m:20s} R@30={mm['per_drug_r30']*100:.2f}%  "
                f"per-dis-AUPRC={mm['per_disease_auprc_mean']:.4f}  "
                f"per-dis-AUROC={mm['per_disease_auroc_mean']:.4f}  "
                f"in_subset={mm['in_subset_count']}/{mm['n_test_diseases']}"
            )

    # Aggregate
    print("\n" + "=" * 72)
    print("AGGREGATE (mean ± std across 5 seeds)")
    print("=" * 72)
    agg: Dict[str, Dict] = {}
    for m in (ANCHOR,) + SUBSETS:
        rows = [r["modes"][m] for r in per_seed_results]
        agg[m] = {
            "r30": mean_std([r["per_drug_r30"] for r in rows]),
            "per_disease_auprc": mean_std([r["per_disease_auprc_mean"] for r in rows]),
            "per_disease_auroc": mean_std([r["per_disease_auroc_mean"] for r in rows]),
            "pooled_auprc": mean_std([r["pooled_auprc"] for r in rows]),
            "pooled_auroc": mean_std([r["pooled_auroc"] for r in rows]),
            "in_subset_mean": float(np.mean([r["in_subset_count"] for r in rows])),
        }
        print(
            f"  {m:20s}  R@30={agg[m]['r30'][0]*100:5.2f}%±{agg[m]['r30'][1]*100:.2f}%  "
            f"per-dis-AUPRC={agg[m]['per_disease_auprc'][0]:.4f}±{agg[m]['per_disease_auprc'][1]:.4f}  "
            f"per-dis-AUROC={agg[m]['per_disease_auroc'][0]:.4f}±{agg[m]['per_disease_auroc'][1]:.4f}  "
            f"|  in_subset_mean={agg[m]['in_subset_mean']:.1f}"
        )

    # Paired-t per-seed vs anchor
    print("\n" + "=" * 72)
    print(f"PAIRED-T per-seed (n=5) vs {ANCHOR}")
    print("=" * 72)
    paired_results: Dict[str, Dict] = {}
    for sub in SUBSETS:
        d_r30 = [r["modes"][sub]["per_drug_r30"] - r["modes"][ANCHOR]["per_drug_r30"] for r in per_seed_results]
        d_pd_auprc = [r["modes"][sub]["per_disease_auprc_mean"] - r["modes"][ANCHOR]["per_disease_auprc_mean"] for r in per_seed_results]
        d_pd_auroc = [r["modes"][sub]["per_disease_auroc_mean"] - r["modes"][ANCHOR]["per_disease_auroc_mean"] for r in per_seed_results]
        d_pool_auprc = [r["modes"][sub]["pooled_auprc"] - r["modes"][ANCHOR]["pooled_auprc"] for r in per_seed_results]
        d_pool_auroc = [r["modes"][sub]["pooled_auroc"] - r["modes"][ANCHOR]["pooled_auroc"] for r in per_seed_results]
        t_r30 = paired_t(d_r30)
        t_pd_auprc = paired_t(d_pd_auprc)
        t_pd_auroc = paired_t(d_pd_auroc)
        t_pool_auprc = paired_t(d_pool_auprc)
        t_pool_auroc = paired_t(d_pool_auroc)
        paired_results[sub] = {
            "R@30": {"mean": t_r30["mean"], "p": t_r30["p_two_sided"]},
            "per_disease_AUPRC": {"mean": t_pd_auprc["mean"], "p": t_pd_auprc["p_two_sided"]},
            "per_disease_AUROC": {"mean": t_pd_auroc["mean"], "p": t_pd_auroc["p_two_sided"]},
            "pooled_AUPRC": {"mean": t_pool_auprc["mean"], "p": t_pool_auprc["p_two_sided"]},
            "pooled_AUROC": {"mean": t_pool_auroc["mean"], "p": t_pool_auroc["p_two_sided"]},
        }
        print(
            f"\n  {sub} vs {ANCHOR}:"
            f"\n    R@30           Δ={t_r30['mean']*100:+.4f}pp  p={t_r30['p_two_sided']:.4g}"
            f"\n    per-dis-AUPRC  Δ={t_pd_auprc['mean']:+.5f}  p={t_pd_auprc['p_two_sided']:.4g}"
            f"\n    per-dis-AUROC  Δ={t_pd_auroc['mean']:+.5f}  p={t_pd_auroc['p_two_sided']:.4g}"
            f"\n    pooled-AUPRC   Δ={t_pool_auprc['mean']:+.5f}  p={t_pool_auprc['p_two_sided']:.4g}"
            f"\n    pooled-AUROC   Δ={t_pool_auroc['mean']:+.5f}  p={t_pool_auroc['p_two_sided']:.4g}"
        )

    # Promotion gate
    print("\n" + "=" * 72)
    print("PROMOTION GATE: per-dis AUPRC Δ ≥ +0.0005 with p < 0.05 → new fusion default")
    print("=" * 72)
    promotion: Dict[str, Dict] = {}
    for sub in SUBSETS:
        pr = paired_results[sub]["per_disease_AUPRC"]
        passes = (pr["mean"] >= 0.0005) and (pr["p"] < 0.05)
        promotion[sub] = {
            "delta": pr["mean"],
            "p": pr["p"],
            "passes_gate": bool(passes),
            "decision": "PROMOTE" if passes else "STAY",
        }
        print(f"  {sub}: Δ={pr['mean']:+.5f}  p={pr['p']:.4g}  →  {promotion[sub]['decision']}")

    # Persist JSON
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({
            "hypothesis": "h1264",
            "blend_w": BLEND_W,
            "anchor": ANCHOR,
            "subsets": list(SUBSETS),
            "seeds": seeds,
            "n_diseases_universe": len(all_diseases),
            "aggregate": {m: {k: list(v) if isinstance(v, tuple) else v for k, v in d.items()} for m, d in agg.items()},
            "paired_t_per_seed_vs_anchor": paired_results,
            "promotion": promotion,
            "per_seed_summaries": per_seed_results,
        }, f, indent=2)
    print(f"\nWrote {OUT_JSON.relative_to(PROJECT_ROOT)}")

    # Markdown
    md: List[str] = []
    md.append("# h1264 — Soft-blend subset sweep (does h1259's win generalise?)\n\n")
    md.append(f"**Premise:** h1259 validated soft_blend_w={BLEND_W} ONLY on the n_gt>=51 ")
    md.append("∩ mid-entropy 'flipped' subset (~7-16 diseases per seed): per-dis AUPRC ")
    md.append("Δ=+0.00054 p=0.013. Outside the subset, scores were concat_l2_znorm.\n\n")
    md.append("h1264 tests 4 progressively broader subsets, with concat_l2_RAW outside ")
    md.append("(not znorm), so cross-disease pooled AUPRC remains a valid sanity check.\n\n")

    md.append("## Aggregate (mean ± std across 5 seeds)\n\n")
    md.append("| Mode | R@30 | per-dis-AUPRC | per-dis-AUROC | pooled-AUPRC | pooled-AUROC | in_subset_mean |\n")
    md.append("|---|---|---|---|---|---|---|\n")
    for m in (ANCHOR,) + SUBSETS:
        a = agg[m]
        md.append(
            f"| `{m}` | "
            f"{a['r30'][0]*100:.2f}%±{a['r30'][1]*100:.2f}% | "
            f"{a['per_disease_auprc'][0]:.4f}±{a['per_disease_auprc'][1]:.4f} | "
            f"{a['per_disease_auroc'][0]:.4f}±{a['per_disease_auroc'][1]:.4f} | "
            f"{a['pooled_auprc'][0]:.4f}±{a['pooled_auprc'][1]:.4f} | "
            f"{a['pooled_auroc'][0]:.4f}±{a['pooled_auroc'][1]:.4f} | "
            f"{a['in_subset_mean']:.1f} |\n"
        )

    md.append(f"\n## Per-seed paired-t vs `{ANCHOR}` (n=5)\n\n")
    md.append("| Subset | ΔR@30 (p) | Δper-dis-AUPRC (p) | Δper-dis-AUROC (p) | Δpooled-AUPRC (p) | Δpooled-AUROC (p) |\n")
    md.append("|---|---|---|---|---|---|\n")
    for sub in SUBSETS:
        rt = paired_results[sub]
        md.append(
            f"| `{sub}` | "
            f"{rt['R@30']['mean']*100:+.4f}pp ({rt['R@30']['p']:.3g}) | "
            f"{rt['per_disease_AUPRC']['mean']:+.5f} ({rt['per_disease_AUPRC']['p']:.3g}) | "
            f"{rt['per_disease_AUROC']['mean']:+.5f} ({rt['per_disease_AUROC']['p']:.3g}) | "
            f"{rt['pooled_AUPRC']['mean']:+.5f} ({rt['pooled_AUPRC']['p']:.3g}) | "
            f"{rt['pooled_AUROC']['mean']:+.5f} ({rt['pooled_AUROC']['p']:.3g}) |\n"
        )

    md.append("\n## Promotion gate decisions (Δ per-dis AUPRC ≥ +0.0005, p < 0.05)\n\n")
    md.append("| Subset | Δ per-dis AUPRC | p | Decision |\n|---|---|---|---|\n")
    for sub, info in promotion.items():
        md.append(f"| `{sub}` | {info['delta']:+.5f} | {info['p']:.3g} | **{info['decision']}** |\n")

    with open(OUT_MD, "w") as f:
        f.write("".join(md))
    print(f"Wrote {OUT_MD.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
