#!/usr/bin/env python3
"""h1259: Per-disease AUPRC reframe — confirm that h1255's pooled-AUPRC
collapse is a metric-pooling artifact, not real ranking degradation.

Strategy: re-run h1255's 7-mode benchmark but compute per-disease AUPRC and
per-disease AUROC alongside the existing pooled metrics. Per-disease AUPRC
is rank-equivariant (immune to per-disease score scaling), so:

  - If concat_l2_znorm per-disease AUPRC ≈ concat_l2_raw per-disease AUPRC
    (within 0.001), then the pooled AUPRC -0.0116 collapse confirmed in
    h1255 is purely a cross-disease pooling artifact, not a ranking change.

  - If soft_blend_w0.5 per-disease AUPRC ≥ concat_l2_raw per-disease AUPRC,
    the fusion-routing direction is a NET WIN under the rank-equivariant
    metric — fully reopening h1228 / h1249 as production candidates.

Outputs:
    data/analysis/h1259_per_disease_auprc.{json,md}
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

from atc_features import ATCMapper  # noqa: E402
from clean_embedding_benchmark import (  # noqa: E402
    load_embeddings,
    mean_std,
    split_diseases,
)
from h1215_fusion_benchmark import build_concat_lookup  # noqa: E402
from h1249_entropy_routed_benchmark import (  # noqa: E402
    N_GT_HIGH_DENSITY,
    disease_atc_l3_entropy,
    paired_t,
    score_one_disease,
)
from h1255_soft_blend_fusion import BLEND_WEIGHTS, z_normalise  # noqa: E402

OUT_JSON = PROJECT_ROOT / "data/analysis/h1259_per_disease_auprc.json"
OUT_MD = PROJECT_ROOT / "data/analysis/h1259_per_disease_auprc.md"


def evaluate_seed(
    *,
    holdout_ids: List[str],
    train_ids: List[str],
    lu_node2vec: Dict[str, np.ndarray],
    lu_concat: Dict[str, np.ndarray],
    eval_gt: Dict[str, Set[str]],
    knn_gt: Dict[str, Set[str]],
    candidate_drugs: Set[str],
    k: int,
    n_gt_per_disease: Dict[str, int],
    entropy_per_disease: Dict[str, float],
    train_low_cut: float,
    train_high_cut: float,
    blend_weights: Tuple[float, ...],
) -> Dict:
    """Per-mode aggregate metrics PLUS per-disease AUPRC/AUROC."""
    train_ids_ordered = [d for d in train_ids if d in lu_concat and d in lu_node2vec]
    train_emb_concat = np.stack([lu_concat[d] for d in train_ids_ordered])
    train_emb_n2v = np.stack([lu_node2vec[d] for d in train_ids_ordered])
    train_knn_gt = {d: (knn_gt[d] & set(lu_concat.keys())) for d in train_ids_ordered if d in knn_gt}

    all_holdout_gt_drugs: Set[str] = set()
    for did in holdout_ids:
        all_holdout_gt_drugs |= eval_gt.get(did, set())
    universe = candidate_drugs | all_holdout_gt_drugs
    universe = {d for d in universe if d in lu_concat and d in lu_node2vec}
    cand_list = sorted(universe)
    cand_index = {d: i for i, d in enumerate(cand_list)}

    modes = ["concat_l2_raw", "concat_l2_znorm"] + [f"soft_blend_w{int(w*100):03d}" for w in blend_weights]

    per_drug_r30: Dict[str, List[float]] = {m: [] for m in modes}
    per_disease_auprc: Dict[str, List[float]] = {m: [] for m in modes}
    per_disease_auroc: Dict[str, List[float]] = {m: [] for m in modes}
    pooled_score_buf: Dict[str, List[np.ndarray]] = {m: [] for m in modes}
    pooled_label_buf: Dict[str, List[np.ndarray]] = {m: [] for m in modes}
    n_test_diseases = 0
    flipped_counter = 0

    for disease_id in holdout_ids:
        if disease_id not in lu_concat or disease_id not in lu_node2vec:
            continue
        gt_drugs = eval_gt.get(disease_id, set()) & universe
        if not gt_drugs:
            continue
        n_test_diseases += 1
        n_gt_disease = n_gt_per_disease.get(disease_id, 0)
        ent = entropy_per_disease.get(disease_id, 0.0)
        is_flipped = (
            n_gt_disease >= N_GT_HIGH_DENSITY and (train_low_cut <= ent < train_high_cut)
        )
        if is_flipped:
            flipped_counter += 1

        scores_n2v = score_one_disease(
            lu_node2vec[disease_id], train_emb_n2v, train_ids_ordered, train_knn_gt, k
        )
        scores_concat = score_one_disease(
            lu_concat[disease_id], train_emb_concat, train_ids_ordered, train_knn_gt, k
        )
        vec_n2v = np.zeros(len(cand_list), dtype=np.float32)
        vec_concat = np.zeros(len(cand_list), dtype=np.float32)
        for d, sc in scores_n2v.items():
            if d in cand_index:
                vec_n2v[cand_index[d]] = sc
        for d, sc in scores_concat.items():
            if d in cand_index:
                vec_concat[cand_index[d]] = sc
        z_n2v = z_normalise(vec_n2v)
        z_concat = z_normalise(vec_concat)

        mode_scores: Dict[str, np.ndarray] = {
            "concat_l2_raw": vec_concat,
            "concat_l2_znorm": z_concat,
        }
        for w in blend_weights:
            label = f"soft_blend_w{int(w*100):03d}"
            if is_flipped:
                mode_scores[label] = w * z_n2v + (1.0 - w) * z_concat
            else:
                mode_scores[label] = z_concat

        # Build label vector once
        label_vec = np.zeros(len(cand_list), dtype=np.int8)
        for d in gt_drugs:
            if d in cand_index:
                label_vec[cand_index[d]] = 1

        for mode, score_vec in mode_scores.items():
            order = np.argsort(-score_vec, kind="stable")
            rank_of_drug = {cand_list[idx]: r for r, idx in enumerate(order, start=1)}
            gt_ranks = [rank_of_drug[d] for d in gt_drugs if d in rank_of_drug]
            n_gt_eval = len(gt_drugs)
            hit_30 = sum(1 for r in gt_ranks if r <= 30)
            per_drug_r30[mode].append(hit_30 / n_gt_eval)

            # PER-DISEASE AUPRC + AUROC (the new h1259 metrics)
            if label_vec.sum() > 0 and label_vec.sum() < len(label_vec):
                per_disease_auprc[mode].append(float(average_precision_score(label_vec, score_vec)))
                per_disease_auroc[mode].append(float(roc_auc_score(label_vec, score_vec)))
            elif label_vec.sum() > 0:
                # All candidates are GT — AUROC undefined; treat as 1.0 for AUPRC
                per_disease_auprc[mode].append(1.0)
                # Skip AUROC

            pooled_score_buf[mode].append(score_vec.copy())
            pooled_label_buf[mode].append(label_vec)

    out_metrics: Dict[str, Dict] = {}
    for m in modes:
        scores_flat = np.concatenate(pooled_score_buf[m]) if pooled_score_buf[m] else np.zeros(0)
        labels_flat = np.concatenate(pooled_label_buf[m]) if pooled_label_buf[m] else np.zeros(0)
        pooled_auprc = float(average_precision_score(labels_flat, scores_flat)) if labels_flat.sum() > 0 else 0.0
        pooled_auroc = (
            float(roc_auc_score(labels_flat, scores_flat))
            if 0 < labels_flat.sum() < len(labels_flat) else 0.0
        )
        out_metrics[m] = {
            "n_test_diseases": n_test_diseases,
            "per_drug_r30": float(np.mean(per_drug_r30[m])) if per_drug_r30[m] else 0.0,
            "pooled_auprc": pooled_auprc,
            "pooled_auroc": pooled_auroc,
            "per_disease_auprc_mean": float(np.mean(per_disease_auprc[m])) if per_disease_auprc[m] else 0.0,
            "per_disease_auprc_n": len(per_disease_auprc[m]),
            "per_disease_auroc_mean": float(np.mean(per_disease_auroc[m])) if per_disease_auroc[m] else 0.0,
            "per_disease_auroc_n": len(per_disease_auroc[m]),
        }

    return {
        "modes": out_metrics,
        "n_test_diseases": n_test_diseases,
        "n_flipped": flipped_counter,
        "train_low_cut": train_low_cut,
        "train_high_cut": train_high_cut,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix-a", default="node2vec_256")
    ap.add_argument("--prefix-b", default="fastrp_256")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--seeds", type=str, default="42,123,456,789,2024")
    ap.add_argument("--eval-gt", default="data/reference/expanded_ground_truth.json")
    ap.add_argument("--knn-gt", default="data/cache/ground_truth_cache.json")
    ap.add_argument("--db-lookup", default="data/reference/drugbank_lookup.json")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    print("=" * 72)
    print(f"h1259: Per-disease AUPRC reframe (h1255 with rank-equivariant metric)")
    print("=" * 72)

    t0 = time.time()
    lu_a, _, _ = load_embeddings(args.prefix_a)
    lu_b, _, _ = load_embeddings(args.prefix_b)
    lu_concat = build_concat_lookup(lu_a, lu_b)

    with open(PROJECT_ROOT / args.eval_gt) as f:
        raw_eval_gt = json.load(f)
    eval_gt = {d: set(drugs) for d, drugs in raw_eval_gt.items()}

    with open(PROJECT_ROOT / args.knn_gt) as f:
        raw_knn_gt = json.load(f)
    if isinstance(raw_knn_gt, dict) and "ground_truth" in raw_knn_gt:
        raw_knn_gt = raw_knn_gt["ground_truth"]
    knn_gt = {d: set(drugs) for d, drugs in raw_knn_gt.items()}

    with open(PROJECT_ROOT / args.db_lookup) as f:
        db_lookup = json.load(f)
    mapper = ATCMapper()

    all_diseases = [d for d in knn_gt if d in lu_concat]
    n_gt_per_disease: Dict[str, int] = {}
    entropy_per_disease: Dict[str, float] = {}
    for d in all_diseases:
        gt_drugs = list(eval_gt.get(d, set()))
        n_gt_per_disease[d] = len(gt_drugs)
        ent, _ = disease_atc_l3_entropy(gt_drugs, db_lookup, mapper)
        entropy_per_disease[d] = ent
    print(f"Setup: {len(all_diseases)} diseases ({time.time()-t0:.1f}s)")

    per_seed_results = []
    for si, seed in enumerate(seeds):
        print(f"\n=== SEED {seed} ({si+1}/{len(seeds)}) ===")
        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        train_high_density_entropies = sorted(
            entropy_per_disease[d] for d in train_ids if n_gt_per_disease[d] >= N_GT_HIGH_DENSITY
        )
        n_thd = len(train_high_density_entropies)
        cut_low = train_high_density_entropies[n_thd // 3] if n_thd >= 3 else 3.18
        cut_high = train_high_density_entropies[2 * n_thd // 3] if n_thd >= 3 else 3.95

        candidate_drugs: Set[str] = set()
        for d in train_ids:
            if d in knn_gt:
                candidate_drugs |= (knn_gt[d] & set(lu_concat.keys()))

        tm0 = time.time()
        seed_out = evaluate_seed(
            holdout_ids=holdout_ids, train_ids=train_ids,
            lu_node2vec=lu_a, lu_concat=lu_concat,
            eval_gt=eval_gt, knn_gt=knn_gt,
            candidate_drugs=candidate_drugs, k=args.k,
            n_gt_per_disease=n_gt_per_disease,
            entropy_per_disease=entropy_per_disease,
            train_low_cut=cut_low, train_high_cut=cut_high,
            blend_weights=BLEND_WEIGHTS,
        )
        seed_out["seed"] = seed
        per_seed_results.append(seed_out)

        for m in seed_out["modes"]:
            mm = seed_out["modes"][m]
            print(
                f"  [{m:18s}] R@30={mm['per_drug_r30']*100:.2f}%  "
                f"per-dis-AUPRC={mm['per_disease_auprc_mean']:.4f}  "
                f"per-dis-AUROC={mm['per_disease_auroc_mean']:.4f}  "
                f"|  pooled-AUPRC={mm['pooled_auprc']:.4f}  pooled-AUROC={mm['pooled_auroc']:.4f}"
            )
        print(f"  flipped={seed_out['n_flipped']} ({time.time()-tm0:.1f}s)")

    # Aggregate
    print("\n" + "=" * 72)
    print("AGGREGATE (mean ± std across 5 seeds)")
    print("=" * 72)
    all_modes = list(per_seed_results[0]["modes"].keys())
    agg: Dict[str, Dict] = {}
    for m in all_modes:
        rows = [r["modes"][m] for r in per_seed_results]
        agg[m] = {
            "r30": mean_std([r["per_drug_r30"] for r in rows]),
            "per_disease_auprc": mean_std([r["per_disease_auprc_mean"] for r in rows]),
            "per_disease_auroc": mean_std([r["per_disease_auroc_mean"] for r in rows]),
            "pooled_auprc": mean_std([r["pooled_auprc"] for r in rows]),
            "pooled_auroc": mean_std([r["pooled_auroc"] for r in rows]),
        }
        print(
            f"  {m:18s}  R@30={agg[m]['r30'][0]*100:5.2f}%  "
            f"per-dis-AUPRC={agg[m]['per_disease_auprc'][0]:.4f}±{agg[m]['per_disease_auprc'][1]:.4f}  "
            f"per-dis-AUROC={agg[m]['per_disease_auroc'][0]:.4f}±{agg[m]['per_disease_auroc'][1]:.4f}  "
            f"|  pooled-AUPRC={agg[m]['pooled_auprc'][0]:.4f}  pooled-AUROC={agg[m]['pooled_auroc'][0]:.4f}"
        )

    # Paired-t per-seed vs concat_l2_raw on per-disease AUPRC
    anchor = "concat_l2_raw"
    print("\n" + "=" * 72)
    print(f"PAIRED-T (per-seed, n=5): per-disease AUPRC vs {anchor}")
    print("=" * 72)
    paired_t_per_dis: Dict[str, Dict] = {}
    for m in all_modes:
        if m == anchor:
            continue
        d_pd_auprc = [r["modes"][m]["per_disease_auprc_mean"] - r["modes"][anchor]["per_disease_auprc_mean"] for r in per_seed_results]
        d_pd_auroc = [r["modes"][m]["per_disease_auroc_mean"] - r["modes"][anchor]["per_disease_auroc_mean"] for r in per_seed_results]
        d_pool_auprc = [r["modes"][m]["pooled_auprc"] - r["modes"][anchor]["pooled_auprc"] for r in per_seed_results]
        d_pool_auroc = [r["modes"][m]["pooled_auroc"] - r["modes"][anchor]["pooled_auroc"] for r in per_seed_results]
        d_r30 = [r["modes"][m]["per_drug_r30"] - r["modes"][anchor]["per_drug_r30"] for r in per_seed_results]
        t_pd_auprc = paired_t(d_pd_auprc)
        t_pd_auroc = paired_t(d_pd_auroc)
        t_pool_auprc = paired_t(d_pool_auprc)
        t_pool_auroc = paired_t(d_pool_auroc)
        t_r30 = paired_t(d_r30)
        paired_t_per_dis[m] = {
            "R@30": {"mean": t_r30["mean"], "p": t_r30["p_two_sided"]},
            "per_disease_AUPRC": {"mean": t_pd_auprc["mean"], "p": t_pd_auprc["p_two_sided"]},
            "per_disease_AUROC": {"mean": t_pd_auroc["mean"], "p": t_pd_auroc["p_two_sided"]},
            "pooled_AUPRC": {"mean": t_pool_auprc["mean"], "p": t_pool_auprc["p_two_sided"]},
            "pooled_AUROC": {"mean": t_pool_auroc["mean"], "p": t_pool_auroc["p_two_sided"]},
        }
        print(f"\n  {m} vs {anchor}:")
        print(f"    R@30                Δ={t_r30['mean']*100:+.4f}pp  p={t_r30['p_two_sided']:.3g}")
        print(f"    per-dis-AUPRC       Δ={t_pd_auprc['mean']:+.5f}  p={t_pd_auprc['p_two_sided']:.3g}")
        print(f"    per-dis-AUROC       Δ={t_pd_auroc['mean']:+.5f}  p={t_pd_auroc['p_two_sided']:.3g}")
        print(f"    pooled-AUPRC        Δ={t_pool_auprc['mean']:+.5f}  p={t_pool_auprc['p_two_sided']:.3g}")
        print(f"    pooled-AUROC        Δ={t_pool_auroc['mean']:+.5f}  p={t_pool_auroc['p_two_sided']:.3g}")

    # Save
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({
            "hypothesis": "h1259",
            "blend_weights": list(BLEND_WEIGHTS),
            "anchor": anchor,
            "n_diseases_eligible": len(all_diseases),
            "aggregate": {m: {k: list(v) if isinstance(v, tuple) else v for k, v in d.items()} for m, d in agg.items()},
            "paired_t_per_seed_vs_anchor": paired_t_per_dis,
            "per_seed_summaries": per_seed_results,
        }, f, indent=2)
    print(f"\nWrote {OUT_JSON.relative_to(PROJECT_ROOT)}")

    # Markdown
    md = []
    md.append("# h1259 — Per-disease AUPRC reframe (rank-equivariant metric)\n\n")
    md.append("**Premise:** h1255 found per-disease z-norm collapses pooled AUROC by 0.147 even when the per-disease ranking ")
    md.append("is byte-identical to the raw scores (R@30 unchanged). This script tests whether per-disease AUPRC ")
    md.append("(mean MAP across diseases) — which IS rank-equivariant — confirms that the h1228/h1249/h1255 'AUPRC regression' ")
    md.append("is a pooling artifact rather than a real ranking degradation.\n\n")
    md.append("If the artifact thesis is correct, we expect:\n")
    md.append("- `per-disease AUPRC` for `concat_l2_znorm` ≈ `concat_l2_raw` (z-norm preserves per-disease rank)\n")
    md.append("- `per-disease AUPRC` for `soft_blend_w0.50` ≥ `concat_l2_raw` (R@30 lift translates to AUPRC lift)\n")
    md.append("- `pooled AUPRC` collapses on the z-norm modes (the h1255 finding)\n\n")

    md.append("## Aggregate (mean ± std across 5 seeds)\n\n")
    md.append("| Mode | R@30 | per-dis-AUPRC | per-dis-AUROC | pooled-AUPRC | pooled-AUROC |\n|---|---|---|---|---|---|\n")
    for m in all_modes:
        a = agg[m]
        md.append(
            f"| `{m}` | {a['r30'][0]*100:.2f}%±{a['r30'][1]*100:.2f}% | "
            f"{a['per_disease_auprc'][0]:.4f}±{a['per_disease_auprc'][1]:.4f} | "
            f"{a['per_disease_auroc'][0]:.4f}±{a['per_disease_auroc'][1]:.4f} | "
            f"{a['pooled_auprc'][0]:.4f}±{a['pooled_auprc'][1]:.4f} | "
            f"{a['pooled_auroc'][0]:.4f}±{a['pooled_auroc'][1]:.4f} |\n"
        )

    md.append(f"\n## Per-seed paired-t vs `{anchor}` (n=5)\n\n")
    md.append("| Mode | ΔR@30 (p) | Δper-dis-AUPRC (p) | Δper-dis-AUROC (p) | Δpooled-AUPRC (p) | Δpooled-AUROC (p) |\n")
    md.append("|---|---|---|---|---|---|\n")
    for m in all_modes:
        if m == anchor:
            continue
        rt = paired_t_per_dis[m]
        md.append(
            f"| `{m}` | "
            f"{rt['R@30']['mean']*100:+.4f}pp ({rt['R@30']['p']:.3g}) | "
            f"{rt['per_disease_AUPRC']['mean']:+.5f} ({rt['per_disease_AUPRC']['p']:.3g}) | "
            f"{rt['per_disease_AUROC']['mean']:+.5f} ({rt['per_disease_AUROC']['p']:.3g}) | "
            f"{rt['pooled_AUPRC']['mean']:+.5f} ({rt['pooled_AUPRC']['p']:.3g}) | "
            f"{rt['pooled_AUROC']['mean']:+.5f} ({rt['pooled_AUROC']['p']:.3g}) |\n"
        )

    with open(OUT_MD, "w") as f:
        f.write("".join(md))
    print(f"Wrote {OUT_MD.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
