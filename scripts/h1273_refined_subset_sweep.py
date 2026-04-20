#!/usr/bin/env python3
"""h1273: Refined GLOBAL recipe — exclude n_gt=1 singletons from soft-blend.

h1272's per-disease audit (5-seed) of the h1269 SUBSET_D_GLOBAL recipe found
that n_gt=1 singleton diseases (n=58/1002 rows = 5.8%) have mean Δ per-disease
AUPRC = -0.0043 with frac_positive = 3.4% — fusion actively HURTS them. The
AUPRC denominator on a single positive amplifies AP fluctuations; on average
the blend adds noise to singletons.

This script tests two refined subsets that exclude low-n_gt diseases from
blending while preserving the h1264 SUBSET_D_GLOBAL framing:

    SUBSET_D_GLOBAL   — (reference) every disease blended
    SUBSET_E_NOSINGLE — n_gt >= 2  (drops 58 singletons)
    SUBSET_F_NGT6     — n_gt >= 6  (drops singletons + 2-5 bucket)

Outside the subset:  score(d) = concat_l2_raw  (no blend)
Inside the subset:   score(d) = 0.5 * z(n2v_score) + 0.5 * z(concat_score)

We report two paired-t comparisons per refined subset:
  (a) vs concat_l2_raw anchor  → does it still gate at p<0.05?
  (b) vs SUBSET_D_GLOBAL       → does it beat the current canonical recipe?

Promotion rule (preregistered in research_roadmap.json h1273):
  if SUBSET_E or SUBSET_F beats GLOBAL on Δ per-dis AUPRC at p<0.05,
  lock as new canonical.

Outputs:
    data/analysis/h1273_refined_subset_sweep.{json,md}
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

OUT_JSON = PROJECT_ROOT / "data/analysis/h1273_refined_subset_sweep.json"
OUT_MD = PROJECT_ROOT / "data/analysis/h1273_refined_subset_sweep.md"

ANCHOR = "concat_l2_raw"
REFERENCE = "SUBSET_D_GLOBAL"
SUBSETS = ("SUBSET_D_GLOBAL", "SUBSET_E_NOSINGLE", "SUBSET_F_NGT6")
BLEND_W = 0.5

# Default: h1269's 20-seed set, for variance comparability.
DEFAULT_SEEDS = "42,123,456,789,2024,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"


def in_subset(name: str, n_gt: int) -> bool:
    if name == "SUBSET_D_GLOBAL":
        return True
    if name == "SUBSET_E_NOSINGLE":
        return n_gt >= 2
    if name == "SUBSET_F_NGT6":
        return n_gt >= 6
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

        label_vec = np.zeros(n_cands, dtype=np.int8)
        for d in gt_drugs:
            idx = cand_index.get(d)
            if idx is not None:
                label_vec[idx] = 1
        n_pos = int(label_vec.sum())
        if n_pos == 0:
            continue

        score_per_mode: Dict[str, np.ndarray] = {ANCHOR: sv_c}
        for sub in SUBSETS:
            if in_subset(sub, n_gt_d):
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
        "modes": out,
    }


def paired_t_vs_ref(
    per_seed_results: List[Dict],
    sub: str,
    ref: str,
) -> Dict[str, Dict]:
    d_r30 = [r["modes"][sub]["per_drug_r30"] - r["modes"][ref]["per_drug_r30"] for r in per_seed_results]
    d_pd_ap = [r["modes"][sub]["per_disease_auprc_mean"] - r["modes"][ref]["per_disease_auprc_mean"] for r in per_seed_results]
    d_pd_auc = [r["modes"][sub]["per_disease_auroc_mean"] - r["modes"][ref]["per_disease_auroc_mean"] for r in per_seed_results]
    d_pool_ap = [r["modes"][sub]["pooled_auprc"] - r["modes"][ref]["pooled_auprc"] for r in per_seed_results]
    d_pool_auc = [r["modes"][sub]["pooled_auroc"] - r["modes"][ref]["pooled_auroc"] for r in per_seed_results]
    return {
        "R@30": {"mean": paired_t(d_r30)["mean"], "p": paired_t(d_r30)["p_two_sided"]},
        "per_disease_AUPRC": {"mean": paired_t(d_pd_ap)["mean"], "p": paired_t(d_pd_ap)["p_two_sided"]},
        "per_disease_AUROC": {"mean": paired_t(d_pd_auc)["mean"], "p": paired_t(d_pd_auc)["p_two_sided"]},
        "pooled_AUPRC": {"mean": paired_t(d_pool_ap)["mean"], "p": paired_t(d_pool_ap)["p_two_sided"]},
        "pooled_AUROC": {"mean": paired_t(d_pool_auc)["mean"], "p": paired_t(d_pool_auc)["p_two_sided"]},
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
    print(f"h1273: Refined GLOBAL recipe — drop n_gt=1 singletons from blend")
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

    intersection_keys = set(lu_a) & set(lu_b)
    all_diseases = [d for d in knn_gt if d in intersection_keys]
    print(f"Universe: {len(all_diseases):,} diseases")

    print("Pre-computing per-disease n_gt...")
    n_gt_per_disease: Dict[str, int] = {d: len(eval_gt.get(d, set())) for d in all_diseases}
    print(f"  setup {time.time()-t0:.1f}s")

    per_seed_results: List[Dict] = []
    for seed in seeds:
        ts = time.time()
        seed_out = evaluate_seed(
            seed=seed, lu_a=lu_a, lu_concat=lu_concat,
            all_diseases=all_diseases, knn_gt=knn_gt, eval_gt=eval_gt,
            n_gt_per_disease=n_gt_per_disease,
            k=args.k,
        )
        per_seed_results.append(seed_out)
        print(f"\n  seed {seed} ({time.time()-ts:.1f}s) n_eval={seed_out['n_eval']}")
        for m in (ANCHOR,) + SUBSETS:
            mm = seed_out["modes"][m]
            print(
                f"    {m:22s} R@30={mm['per_drug_r30']*100:.2f}%  "
                f"per-dis-AUPRC={mm['per_disease_auprc_mean']:.4f}  "
                f"per-dis-AUROC={mm['per_disease_auroc_mean']:.4f}  "
                f"in_subset={mm['in_subset_count']}/{mm['n_test_diseases']}"
            )

    print("\n" + "=" * 72)
    print(f"AGGREGATE (mean ± std across {len(seeds)} seeds)")
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
            f"  {m:22s}  R@30={agg[m]['r30'][0]*100:5.2f}%±{agg[m]['r30'][1]*100:.2f}%  "
            f"per-dis-AUPRC={agg[m]['per_disease_auprc'][0]:.4f}±{agg[m]['per_disease_auprc'][1]:.4f}  "
            f"per-dis-AUROC={agg[m]['per_disease_auroc'][0]:.4f}±{agg[m]['per_disease_auroc'][1]:.4f}  "
            f"| in_subset_mean={agg[m]['in_subset_mean']:.1f}"
        )

    print("\n" + "=" * 72)
    print(f"PAIRED-T per-seed (n={len(seeds)}) vs {ANCHOR}")
    print("=" * 72)
    paired_vs_anchor: Dict[str, Dict] = {}
    for sub in SUBSETS:
        rt = paired_t_vs_ref(per_seed_results, sub, ANCHOR)
        paired_vs_anchor[sub] = rt
        print(
            f"\n  {sub} vs {ANCHOR}:"
            f"\n    R@30           Δ={rt['R@30']['mean']*100:+.4f}pp  p={rt['R@30']['p']:.4g}"
            f"\n    per-dis-AUPRC  Δ={rt['per_disease_AUPRC']['mean']:+.5f}  p={rt['per_disease_AUPRC']['p']:.4g}"
            f"\n    per-dis-AUROC  Δ={rt['per_disease_AUROC']['mean']:+.5f}  p={rt['per_disease_AUROC']['p']:.4g}"
            f"\n    pooled-AUPRC   Δ={rt['pooled_AUPRC']['mean']:+.5f}  p={rt['pooled_AUPRC']['p']:.4g}"
            f"\n    pooled-AUROC   Δ={rt['pooled_AUROC']['mean']:+.5f}  p={rt['pooled_AUROC']['p']:.4g}"
        )

    print("\n" + "=" * 72)
    print(f"PAIRED-T per-seed (n={len(seeds)}) vs {REFERENCE} (current canonical)")
    print("=" * 72)
    paired_vs_ref: Dict[str, Dict] = {}
    for sub in ("SUBSET_E_NOSINGLE", "SUBSET_F_NGT6"):
        rt = paired_t_vs_ref(per_seed_results, sub, REFERENCE)
        paired_vs_ref[sub] = rt
        print(
            f"\n  {sub} vs {REFERENCE}:"
            f"\n    R@30           Δ={rt['R@30']['mean']*100:+.4f}pp  p={rt['R@30']['p']:.4g}"
            f"\n    per-dis-AUPRC  Δ={rt['per_disease_AUPRC']['mean']:+.5f}  p={rt['per_disease_AUPRC']['p']:.4g}"
            f"\n    per-dis-AUROC  Δ={rt['per_disease_AUROC']['mean']:+.5f}  p={rt['per_disease_AUROC']['p']:.4g}"
        )

    # Promotion-rule decision: beat SUBSET_D_GLOBAL on Δ per-dis AUPRC at p<0.05.
    print("\n" + "=" * 72)
    print(f"PROMOTION GATE: refined subset beats {REFERENCE} on Δ per-dis AUPRC at p<0.05")
    print("=" * 72)
    promotion: Dict[str, Dict] = {}
    for sub in ("SUBSET_E_NOSINGLE", "SUBSET_F_NGT6"):
        rt = paired_vs_ref[sub]
        pa = rt["per_disease_AUPRC"]
        passes = (pa["mean"] > 0) and (pa["p"] < 0.05)
        promotion[sub] = {
            "delta_vs_global": pa["mean"],
            "p_vs_global": pa["p"],
            "passes_gate": bool(passes),
            "decision": "PROMOTE over GLOBAL" if passes else "STAY with GLOBAL",
        }
        print(f"  {sub}: Δ={pa['mean']:+.5f}  p={pa['p']:.4g}  →  {promotion[sub]['decision']}")

    # Persist JSON
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({
            "hypothesis": "h1273",
            "blend_w": BLEND_W,
            "anchor": ANCHOR,
            "reference": REFERENCE,
            "subsets": list(SUBSETS),
            "seeds": seeds,
            "n_diseases_universe": len(all_diseases),
            "aggregate": {m: {k: list(v) if isinstance(v, tuple) else v for k, v in d.items()} for m, d in agg.items()},
            "paired_t_vs_anchor": paired_vs_anchor,
            "paired_t_vs_reference": paired_vs_ref,
            "promotion": promotion,
            "per_seed_summaries": per_seed_results,
        }, f, indent=2)
    print(f"\nWrote {OUT_JSON.relative_to(PROJECT_ROOT)}")

    # Markdown
    md: List[str] = []
    md.append("# h1273 — Refined GLOBAL recipe (drop n_gt=1 singletons from soft-blend)\n\n")
    md.append("**Premise:** h1272's per-disease audit found n_gt=1 singletons (n=58/1002 rows) have ")
    md.append("mean Δ per-dis AUPRC = -0.0043 under the GLOBAL soft-blend — fusion HURTS them.\n\n")
    md.append(f"Two refined subsets tested (blend_w={BLEND_W}, n_seeds={len(seeds)}):\n\n")
    md.append("- `SUBSET_D_GLOBAL` — current canonical (every disease blended)\n")
    md.append("- `SUBSET_E_NOSINGLE` — n_gt ≥ 2 (58 singletons dropped)\n")
    md.append("- `SUBSET_F_NGT6` — n_gt ≥ 6 (singletons + 2-5 bucket dropped)\n\n")

    md.append(f"## Aggregate (mean ± std across {len(seeds)} seeds)\n\n")
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

    md.append(f"\n## Paired-t vs `{ANCHOR}` (n={len(seeds)})\n\n")
    md.append("| Subset | ΔR@30 (p) | Δper-dis-AUPRC (p) | Δper-dis-AUROC (p) | Δpooled-AUPRC (p) | Δpooled-AUROC (p) |\n")
    md.append("|---|---|---|---|---|---|\n")
    for sub in SUBSETS:
        rt = paired_vs_anchor[sub]
        md.append(
            f"| `{sub}` | "
            f"{rt['R@30']['mean']*100:+.4f}pp ({rt['R@30']['p']:.3g}) | "
            f"{rt['per_disease_AUPRC']['mean']:+.5f} ({rt['per_disease_AUPRC']['p']:.3g}) | "
            f"{rt['per_disease_AUROC']['mean']:+.5f} ({rt['per_disease_AUROC']['p']:.3g}) | "
            f"{rt['pooled_AUPRC']['mean']:+.5f} ({rt['pooled_AUPRC']['p']:.3g}) | "
            f"{rt['pooled_AUROC']['mean']:+.5f} ({rt['pooled_AUROC']['p']:.3g}) |\n"
        )

    md.append(f"\n## Paired-t vs `{REFERENCE}` (current canonical, n={len(seeds)})\n\n")
    md.append("| Subset | ΔR@30 (p) | Δper-dis-AUPRC (p) | Δper-dis-AUROC (p) |\n")
    md.append("|---|---|---|---|\n")
    for sub in ("SUBSET_E_NOSINGLE", "SUBSET_F_NGT6"):
        rt = paired_vs_ref[sub]
        md.append(
            f"| `{sub}` | "
            f"{rt['R@30']['mean']*100:+.4f}pp ({rt['R@30']['p']:.3g}) | "
            f"{rt['per_disease_AUPRC']['mean']:+.5f} ({rt['per_disease_AUPRC']['p']:.3g}) | "
            f"{rt['per_disease_AUROC']['mean']:+.5f} ({rt['per_disease_AUROC']['p']:.3g}) |\n"
        )

    md.append(f"\n## Promotion gate (beat `{REFERENCE}` on Δ per-dis AUPRC at p<0.05)\n\n")
    md.append("| Subset | Δ vs GLOBAL | p | Decision |\n|---|---|---|---|\n")
    for sub, info in promotion.items():
        md.append(f"| `{sub}` | {info['delta_vs_global']:+.5f} | {info['p_vs_global']:.3g} | **{info['decision']}** |\n")

    with open(OUT_MD, "w") as f:
        f.write("".join(md))
    print(f"Wrote {OUT_MD.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
