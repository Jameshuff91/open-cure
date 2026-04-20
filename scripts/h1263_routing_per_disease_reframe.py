#!/usr/bin/env python3
"""h1263: Re-evaluate h1228 (cat_gated) + h1249 (entropy_routed) under
per-disease AUPRC + per-disease AUROC.

Both routing strategies were INVALIDATED on pooled-AUPRC regression in
their original runs. h1259 showed pooled AUPRC is scale-confounded across
embedding spaces (per-disease z-norm collapses pooled AUROC by 0.147 even
when the per-disease ranking is byte-identical to raw scores), so the
"AUPRC regression" was a metric-pooling artifact, not a real ranking
degradation. By the same mechanism, swapping per-disease score *spaces*
(node2vec ↔ concat_l2) within a routing rule should also be neutral on
per-disease AUPRC even though it perturbs pooled AUPRC.

Pipeline (4 modes, 5 seeds, paired-t vs concat_l2_raw anchor):
    1. node2vec        (control)
    2. concat_l2_raw   (anchor)
    3. cat_gated       (h1228 leave-one-seed-out per-category gate)
    4. entropy_routed  (h1249 train-side n_gt × ATC L3 entropy tercile rule)

Metrics per mode per seed:
    - per-drug R@30 (per-disease mean across holdout)
    - per-disease AUPRC (mean MAP)
    - per-disease AUROC (mean disease-level ROC AUC)
    - pooled AUPRC + pooled AUROC (kept for scale-artifact comparison)

Promotion gate:
    Re-open the original hypothesis (h1228 / h1249) if its per-disease AUPRC
    Δ vs concat_l2_raw is ≥ 0 with two-sided p < 0.1 over 5 seeds.

Outputs:
    data/analysis/h1263_routing_per_disease_reframe.json
    data/analysis/h1263_routing_per_disease_reframe.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from atc_features import ATCMapper  # noqa: E402
from clean_embedding_benchmark import (  # noqa: E402
    categorize,
    load_disease_names,
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

OUT_JSON = PROJECT_ROOT / "data/analysis/h1263_routing_per_disease_reframe.json"
OUT_MD = PROJECT_ROOT / "data/analysis/h1263_routing_per_disease_reframe.md"

MODES = ("node2vec", "concat_l2_raw", "cat_gated", "entropy_routed")
ANCHOR = "concat_l2_raw"


def per_seed_per_disease(
    *,
    lu_a: Dict[str, np.ndarray],
    lu_concat: Dict[str, np.ndarray],
    all_diseases: List[str],
    knn_gt: Dict[str, Set[str]],
    eval_gt: Dict[str, Set[str]],
    disease_names: Dict[str, str],
    n_gt_per_disease: Dict[str, int],
    entropy_per_disease: Dict[str, float],
    seed: int,
    k: int,
) -> Dict:
    """Build per-disease score vectors under node2vec and concat_l2 plus
    routing metadata. Train-side entropy tercile cuts are computed locally
    on n_gt>=51 train diseases (no holdout leakage)."""
    train_ids, holdout_ids = split_diseases(all_diseases, seed)
    train_ids_ordered = [d for d in train_ids if d in lu_concat and d in lu_a]
    train_emb_a = np.stack([lu_a[d] for d in train_ids_ordered])
    train_emb_c = np.stack([lu_concat[d] for d in train_ids_ordered])
    train_gt = {
        d: (knn_gt[d] & set(lu_concat.keys()))
        for d in train_ids_ordered
        if d in knn_gt
    }

    # Train-side entropy tercile cuts on n_gt>=51 only
    train_high_density_entropies = sorted(
        entropy_per_disease[d]
        for d in train_ids
        if n_gt_per_disease.get(d, 0) >= N_GT_HIGH_DENSITY
    )
    n_thd = len(train_high_density_entropies)
    if n_thd >= 3:
        cut_low = train_high_density_entropies[n_thd // 3]
        cut_high = train_high_density_entropies[2 * n_thd // 3]
    else:
        cut_low, cut_high = 3.18, 3.95  # h1247 fallback

    # Candidate universe
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

    rows: List[Dict] = []
    for did in holdout_ids:
        if did not in lu_concat or did not in lu_a:
            continue
        gt_drugs = eval_gt.get(did, set()) & universe
        if not gt_drugs:
            continue

        # Score under both embeddings
        ds_a = score_disease_single(lu_a[did], train_emb_a, train_ids_ordered, train_gt, k)
        sv_a = np.zeros(len(cand_list), dtype=np.float32)
        for drug, sc in ds_a.items():
            if drug in cand_index:
                sv_a[cand_index[drug]] = sc

        ds_c = score_disease_single(lu_concat[did], train_emb_c, train_ids_ordered, train_gt, k)
        sv_c = np.zeros(len(cand_list), dtype=np.float32)
        for drug, sc in ds_c.items():
            if drug in cand_index:
                sv_c[cand_index[drug]] = sc

        n_gt_disease = n_gt_per_disease.get(did, 0)
        ent = entropy_per_disease.get(did, 0.0)
        # h1249 routing rule
        if n_gt_disease >= N_GT_HIGH_DENSITY and (cut_low <= ent < cut_high):
            entropy_route_to = "node2vec"  # mid-entropy high-density → n2v
            tercile = "mid"
        elif n_gt_disease >= N_GT_HIGH_DENSITY:
            entropy_route_to = "concat_l2"
            tercile = "low" if ent < cut_low else "high"
        else:
            entropy_route_to = "concat_l2"
            tercile = "n/a"

        rows.append({
            "disease_id": did,
            "category": categorize(disease_names.get(did, did)),
            "gt_drugs": list(gt_drugs),
            "sv_a": sv_a,
            "sv_c": sv_c,
            "n_gt": n_gt_disease,
            "entropy": ent,
            "tercile": tercile,
            "entropy_route_to": entropy_route_to,
        })

    return {
        "seed": seed,
        "n_eval": len(rows),
        "cand_list": cand_list,
        "cand_index": cand_index,
        "rows": rows,
        "cut_low": cut_low,
        "cut_high": cut_high,
        "n_train_high_density": n_thd,
    }


def per_seed_per_category_delta_r30(seed_data: Dict) -> Dict[str, float]:
    """Mean R@30(concat_l2) - R@30(node2vec) per category, used for the
    h1228 leave-one-seed-out gate."""
    cand_index = seed_data["cand_index"]
    cat_a: Dict[str, List[float]] = defaultdict(list)
    cat_c: Dict[str, List[float]] = defaultdict(list)
    for row in seed_data["rows"]:
        n = len(row["gt_drugs"])
        if n == 0:
            continue
        order_a = np.argsort(-row["sv_a"], kind="stable")
        order_c = np.argsort(-row["sv_c"], kind="stable")
        rank_a = {int(idx): r + 1 for r, idx in enumerate(order_a)}
        rank_c = {int(idx): r + 1 for r, idx in enumerate(order_c)}
        hits_a = sum(
            1 for d in row["gt_drugs"]
            if cand_index.get(d) is not None and rank_a.get(cand_index[d], 10**9) <= 30
        )
        hits_c = sum(
            1 for d in row["gt_drugs"]
            if cand_index.get(d) is not None and rank_c.get(cand_index[d], 10**9) <= 30
        )
        cat_a[row["category"]].append(hits_a / n)
        cat_c[row["category"]].append(hits_c / n)
    return {
        cat: (float(np.mean(cat_c[cat])) - float(np.mean(cat_a[cat])))
        for cat in cat_a
    }


def evaluate_seed_with_modes(
    seed_data: Dict,
    cat_gate: Dict[str, bool],
) -> Dict[str, Dict]:
    """Evaluate all four modes on this seed's holdout, returning per-mode
    aggregate metrics (R@30, per-disease AUPRC + AUROC, pooled AUPRC + AUROC)."""
    cand_list = seed_data["cand_list"]
    cand_index = seed_data["cand_index"]

    per_drug_r30: Dict[str, List[float]] = {m: [] for m in MODES}
    per_disease_auprc: Dict[str, List[float]] = {m: [] for m in MODES}
    per_disease_auroc: Dict[str, List[float]] = {m: [] for m in MODES}
    pooled_score_buf: Dict[str, List[np.ndarray]] = {m: [] for m in MODES}
    pooled_label_buf: Dict[str, List[np.ndarray]] = {m: [] for m in MODES}
    routing_count: Dict[str, Counter] = {m: Counter() for m in MODES}

    for row in seed_data["rows"]:
        sv_a = row["sv_a"]
        sv_c = row["sv_c"]
        # cat_gated: route per category — gate True ⇒ concat_l2, else node2vec
        cat = row["category"]
        cat_route_to = "concat_l2" if cat_gate.get(cat, True) else "node2vec"
        # entropy_routed: per-disease rule already pre-computed
        ent_route_to = row["entropy_route_to"]

        score_per_mode = {
            "node2vec": sv_a,
            "concat_l2_raw": sv_c,
            "cat_gated": sv_c if cat_route_to == "concat_l2" else sv_a,
            "entropy_routed": sv_c if ent_route_to == "concat_l2" else sv_a,
        }
        routing_count["cat_gated"][cat_route_to] += 1
        routing_count["entropy_routed"][ent_route_to] += 1

        # Build label vector once
        n_cands = len(cand_list)
        label_vec = np.zeros(n_cands, dtype=np.int8)
        for d in row["gt_drugs"]:
            idx = cand_index.get(d)
            if idx is not None:
                label_vec[idx] = 1
        n_pos = int(label_vec.sum())
        if n_pos == 0:
            continue

        for mode, score_vec in score_per_mode.items():
            # R@30
            order = np.argsort(-score_vec, kind="stable")
            rank_of_idx = {int(idx): r + 1 for r, idx in enumerate(order)}
            n_gt = len(row["gt_drugs"])
            hits_30 = sum(
                1 for d in row["gt_drugs"]
                if cand_index.get(d) is not None and rank_of_idx.get(cand_index[d], 10**9) <= 30
            )
            per_drug_r30[mode].append(hits_30 / n_gt)

            # Per-disease AUPRC + AUROC (rank-equivariant)
            if 0 < n_pos < n_cands:
                per_disease_auprc[mode].append(
                    float(average_precision_score(label_vec, score_vec))
                )
                per_disease_auroc[mode].append(
                    float(roc_auc_score(label_vec, score_vec))
                )
            elif n_pos == n_cands:
                per_disease_auprc[mode].append(1.0)
                # AUROC undefined; skip

            pooled_score_buf[mode].append(score_vec.copy())
            pooled_label_buf[mode].append(label_vec.copy())

    # Aggregate
    out: Dict[str, Dict] = {}
    for m in MODES:
        scores_flat = np.concatenate(pooled_score_buf[m]) if pooled_score_buf[m] else np.zeros(0)
        labels_flat = np.concatenate(pooled_label_buf[m]) if pooled_label_buf[m] else np.zeros(0)
        pooled_auprc = (
            float(average_precision_score(labels_flat, scores_flat))
            if labels_flat.sum() > 0
            else 0.0
        )
        pooled_auroc = (
            float(roc_auc_score(labels_flat, scores_flat))
            if 0 < labels_flat.sum() < len(labels_flat)
            else 0.0
        )
        out[m] = {
            "n_test_diseases": len(per_drug_r30[m]),
            "per_drug_r30": float(np.mean(per_drug_r30[m])) if per_drug_r30[m] else 0.0,
            "per_disease_auprc_mean": (
                float(np.mean(per_disease_auprc[m])) if per_disease_auprc[m] else 0.0
            ),
            "per_disease_auprc_n": len(per_disease_auprc[m]),
            "per_disease_auroc_mean": (
                float(np.mean(per_disease_auroc[m])) if per_disease_auroc[m] else 0.0
            ),
            "per_disease_auroc_n": len(per_disease_auroc[m]),
            "pooled_auprc": pooled_auprc,
            "pooled_auroc": pooled_auroc,
            "routing_count": dict(routing_count[m]),
        }
    return out


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
    print(
        "h1263: Re-evaluate cat_gated (h1228) + entropy_routed (h1249) "
        "under per-disease AUPRC"
    )
    print(f"  prefix_a={args.prefix_a}  prefix_b={args.prefix_b}  seeds={seeds}")
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

    disease_names = load_disease_names()
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
    print(f"  setup {time.time() - t0:.1f}s")

    # Pass 1: build per-seed per-disease score data
    per_seed: Dict[int, Dict] = {}
    for seed in seeds:
        ts = time.time()
        per_seed[seed] = per_seed_per_disease(
            lu_a=lu_a,
            lu_concat=lu_concat,
            all_diseases=all_diseases,
            knn_gt=knn_gt,
            eval_gt=eval_gt,
            disease_names=disease_names,
            n_gt_per_disease=n_gt_per_disease,
            entropy_per_disease=entropy_per_disease,
            seed=seed,
            k=args.k,
        )
        print(
            f"  seed {seed}: n_eval={per_seed[seed]['n_eval']}  "
            f"cuts=({per_seed[seed]['cut_low']:.3f}, "
            f"{per_seed[seed]['cut_high']:.3f}, n_thd={per_seed[seed]['n_train_high_density']})  "
            f"({time.time() - ts:.1f}s)"
        )

    # Pass 2: per-seed per-category Δ R@30, then leave-one-seed-out gate
    per_seed_cat_delta = {seed: per_seed_per_category_delta_r30(per_seed[seed]) for seed in seeds}

    # Pass 3: evaluate all 4 modes per seed
    print("\n" + "=" * 72)
    print("Per-seed evaluation")
    print("=" * 72)
    per_seed_metrics: Dict[int, Dict[str, Dict]] = {}
    per_seed_gate: Dict[int, Dict[str, bool]] = {}
    for seed in seeds:
        other_seeds = [s for s in seeds if s != seed]
        cat_deltas: Dict[str, List[float]] = defaultdict(list)
        for s in other_seeds:
            for cat, d in per_seed_cat_delta[s].items():
                cat_deltas[cat].append(d)
        gate = {cat: (float(np.mean(ds)) > 0) for cat, ds in cat_deltas.items()}
        per_seed_gate[seed] = gate
        sd_metrics = evaluate_seed_with_modes(per_seed[seed], gate)
        per_seed_metrics[seed] = sd_metrics
        for m in MODES:
            mm = sd_metrics[m]
            print(
                f"  seed {seed} [{m:14s}] R@30={mm['per_drug_r30'] * 100:.2f}%  "
                f"per-dis-AUPRC={mm['per_disease_auprc_mean']:.4f}  "
                f"per-dis-AUROC={mm['per_disease_auroc_mean']:.4f}  "
                f"|  pooled-AUPRC={mm['pooled_auprc']:.4f}  "
                f"pooled-AUROC={mm['pooled_auroc']:.4f}"
            )
        gate_gainers = sorted([c for c, v in gate.items() if v])
        print(f"    cat_gate gainers ({len(gate_gainers)}): {gate_gainers}")
        ent_route = sd_metrics["entropy_routed"]["routing_count"]
        print(f"    entropy_routed → {ent_route}")

    # Aggregate
    print("\n" + "=" * 72)
    print("AGGREGATE (mean ± std across 5 seeds)")
    print("=" * 72)
    agg: Dict[str, Dict] = {}
    for m in MODES:
        rows = [per_seed_metrics[s][m] for s in seeds]
        agg[m] = {
            "r30": mean_std([r["per_drug_r30"] for r in rows]),
            "per_disease_auprc": mean_std([r["per_disease_auprc_mean"] for r in rows]),
            "per_disease_auroc": mean_std([r["per_disease_auroc_mean"] for r in rows]),
            "pooled_auprc": mean_std([r["pooled_auprc"] for r in rows]),
            "pooled_auroc": mean_std([r["pooled_auroc"] for r in rows]),
        }
        print(
            f"  {m:14s}  R@30={agg[m]['r30'][0] * 100:5.2f}%±{agg[m]['r30'][1] * 100:.2f}%  "
            f"per-dis-AUPRC={agg[m]['per_disease_auprc'][0]:.4f}±"
            f"{agg[m]['per_disease_auprc'][1]:.4f}  "
            f"per-dis-AUROC={agg[m]['per_disease_auroc'][0]:.4f}±"
            f"{agg[m]['per_disease_auroc'][1]:.4f}  |  "
            f"pooled-AUPRC={agg[m]['pooled_auprc'][0]:.4f}  "
            f"pooled-AUROC={agg[m]['pooled_auroc'][0]:.4f}"
        )

    # Paired-t per-seed vs ANCHOR
    print("\n" + "=" * 72)
    print(f"PAIRED-T (per-seed, n=5): all metrics vs {ANCHOR}")
    print("=" * 72)
    paired_results: Dict[str, Dict] = {}
    for m in MODES:
        if m == ANCHOR:
            continue
        d_r30 = [
            per_seed_metrics[s][m]["per_drug_r30"]
            - per_seed_metrics[s][ANCHOR]["per_drug_r30"]
            for s in seeds
        ]
        d_pd_auprc = [
            per_seed_metrics[s][m]["per_disease_auprc_mean"]
            - per_seed_metrics[s][ANCHOR]["per_disease_auprc_mean"]
            for s in seeds
        ]
        d_pd_auroc = [
            per_seed_metrics[s][m]["per_disease_auroc_mean"]
            - per_seed_metrics[s][ANCHOR]["per_disease_auroc_mean"]
            for s in seeds
        ]
        d_pool_auprc = [
            per_seed_metrics[s][m]["pooled_auprc"]
            - per_seed_metrics[s][ANCHOR]["pooled_auprc"]
            for s in seeds
        ]
        d_pool_auroc = [
            per_seed_metrics[s][m]["pooled_auroc"]
            - per_seed_metrics[s][ANCHOR]["pooled_auroc"]
            for s in seeds
        ]
        t_r30 = paired_t(d_r30)
        t_pd_auprc = paired_t(d_pd_auprc)
        t_pd_auroc = paired_t(d_pd_auroc)
        t_pool_auprc = paired_t(d_pool_auprc)
        t_pool_auroc = paired_t(d_pool_auroc)
        paired_results[m] = {
            "R@30": {"mean": t_r30["mean"], "p": t_r30["p_two_sided"]},
            "per_disease_AUPRC": {
                "mean": t_pd_auprc["mean"],
                "p": t_pd_auprc["p_two_sided"],
            },
            "per_disease_AUROC": {
                "mean": t_pd_auroc["mean"],
                "p": t_pd_auroc["p_two_sided"],
            },
            "pooled_AUPRC": {
                "mean": t_pool_auprc["mean"],
                "p": t_pool_auprc["p_two_sided"],
            },
            "pooled_AUROC": {
                "mean": t_pool_auroc["mean"],
                "p": t_pool_auroc["p_two_sided"],
            },
        }
        print(f"\n  {m} vs {ANCHOR}:")
        print(
            f"    R@30                Δ={t_r30['mean'] * 100:+.4f}pp  "
            f"p={t_r30['p_two_sided']:.4g}"
        )
        print(
            f"    per-dis-AUPRC       Δ={t_pd_auprc['mean']:+.5f}  "
            f"p={t_pd_auprc['p_two_sided']:.4g}"
        )
        print(
            f"    per-dis-AUROC       Δ={t_pd_auroc['mean']:+.5f}  "
            f"p={t_pd_auroc['p_two_sided']:.4g}"
        )
        print(
            f"    pooled-AUPRC        Δ={t_pool_auprc['mean']:+.5f}  "
            f"p={t_pool_auprc['p_two_sided']:.4g}"
        )
        print(
            f"    pooled-AUROC        Δ={t_pool_auroc['mean']:+.5f}  "
            f"p={t_pool_auroc['p_two_sided']:.4g}"
        )

    # Promotion gate evaluation
    print("\n" + "=" * 72)
    print("PROMOTION GATE: per-disease AUPRC Δ ≥ 0 with p < 0.1 → REOPEN")
    print("=" * 72)
    promotion: Dict[str, Dict] = {}
    for m in ("cat_gated", "entropy_routed"):
        pd_auprc = paired_results[m]["per_disease_AUPRC"]
        passes = (pd_auprc["mean"] >= 0) and (pd_auprc["p"] < 0.1)
        promotion[m] = {
            "per_disease_auprc_delta": pd_auprc["mean"],
            "per_disease_auprc_p": pd_auprc["p"],
            "reopen": bool(passes),
            "decision": "REOPEN" if passes else "STAY_INVALIDATED",
        }
        print(
            f"  {m}: Δ per-disease AUPRC = {pd_auprc['mean']:+.5f}  "
            f"p={pd_auprc['p']:.4g}  →  {promotion[m]['decision']}"
        )

    # Persist
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(
            {
                "hypothesis": "h1263",
                "anchor": ANCHOR,
                "seeds": seeds,
                "n_diseases_universe": len(all_diseases),
                "aggregate": {
                    m: {k: list(v) if isinstance(v, tuple) else v for k, v in d.items()}
                    for m, d in agg.items()
                },
                "paired_t_per_seed_vs_anchor": paired_results,
                "promotion": promotion,
                "per_seed_metrics": {
                    str(s): per_seed_metrics[s] for s in seeds
                },
                "per_seed_gate": {str(s): g for s, g in per_seed_gate.items()},
                "per_seed_summary": [
                    {
                        "seed": s,
                        "n_eval": per_seed[s]["n_eval"],
                        "cut_low": per_seed[s]["cut_low"],
                        "cut_high": per_seed[s]["cut_high"],
                        "n_train_high_density": per_seed[s]["n_train_high_density"],
                    }
                    for s in seeds
                ],
            },
            f,
            indent=2,
        )
    print(f"\nWrote {OUT_JSON.relative_to(PROJECT_ROOT)}")

    # Markdown report
    md: List[str] = []
    md.append("# h1263 — Re-evaluate cat_gated + entropy_routed under per-disease AUPRC\n\n")
    md.append("**Premise:** h1259 showed pooled AUPRC/AUROC are scale-confounded across ")
    md.append("embedding spaces. Per-disease AUPRC is rank-equivariant (immune to per-disease ")
    md.append("score scaling). h1228 (category-gated fusion) and h1249 (entropy-routed fusion) ")
    md.append("were both INVALIDATED on pooled-AUPRC regression; this script re-tests them ")
    md.append("under the corrected metric.\n\n")
    md.append(f"**Promotion gate:** per-disease AUPRC Δ ≥ 0 with p < 0.1 over {len(seeds)} seeds.\n\n")

    md.append("## Aggregate (mean ± std across 5 seeds)\n\n")
    md.append("| Mode | R@30 | per-dis-AUPRC | per-dis-AUROC | pooled-AUPRC | pooled-AUROC |\n")
    md.append("|---|---|---|---|---|---|\n")
    for m in MODES:
        a = agg[m]
        md.append(
            f"| `{m}` | "
            f"{a['r30'][0] * 100:.2f}%±{a['r30'][1] * 100:.2f}% | "
            f"{a['per_disease_auprc'][0]:.4f}±{a['per_disease_auprc'][1]:.4f} | "
            f"{a['per_disease_auroc'][0]:.4f}±{a['per_disease_auroc'][1]:.4f} | "
            f"{a['pooled_auprc'][0]:.4f}±{a['pooled_auprc'][1]:.4f} | "
            f"{a['pooled_auroc'][0]:.4f}±{a['pooled_auroc'][1]:.4f} |\n"
        )

    md.append(f"\n## Per-seed paired-t vs `{ANCHOR}` (n=5)\n\n")
    md.append(
        "| Mode | ΔR@30 (p) | Δper-dis-AUPRC (p) | Δper-dis-AUROC (p) "
        "| Δpooled-AUPRC (p) | Δpooled-AUROC (p) |\n"
    )
    md.append("|---|---|---|---|---|---|\n")
    for m in MODES:
        if m == ANCHOR:
            continue
        rt = paired_results[m]
        md.append(
            f"| `{m}` | "
            f"{rt['R@30']['mean'] * 100:+.4f}pp ({rt['R@30']['p']:.3g}) | "
            f"{rt['per_disease_AUPRC']['mean']:+.5f} ({rt['per_disease_AUPRC']['p']:.3g}) | "
            f"{rt['per_disease_AUROC']['mean']:+.5f} ({rt['per_disease_AUROC']['p']:.3g}) | "
            f"{rt['pooled_AUPRC']['mean']:+.5f} ({rt['pooled_AUPRC']['p']:.3g}) | "
            f"{rt['pooled_AUROC']['mean']:+.5f} ({rt['pooled_AUROC']['p']:.3g}) |\n"
        )

    md.append("\n## Promotion gate decisions\n\n")
    md.append("| Mode | Δ per-disease AUPRC | p | Decision |\n|---|---|---|---|\n")
    for m, info in promotion.items():
        md.append(
            f"| `{m}` | {info['per_disease_auprc_delta']:+.5f} | "
            f"{info['per_disease_auprc_p']:.3g} | **{info['decision']}** |\n"
        )

    with open(OUT_MD, "w") as f:
        f.write("".join(md))
    print(f"Wrote {OUT_MD.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
