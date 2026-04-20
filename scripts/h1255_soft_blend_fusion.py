#!/usr/bin/env python3
"""h1255: Score-scale-normalised soft-blend fusion (fixes h1249 AUPRC regression).

h1228 and h1249 both showed: hard-switching from concat_l2 → node2vec on a
targeted subset lifts that subset's R@30 (h1249 +0.76pp p=0.042 on flipped
subset) but regresses global AUPRC/AUROC at p<0.1 because per-disease score
scales differ across embeddings.

Fix candidate: z-normalise per-disease per-embedding, then blend rather than
hard-switch on the (n_gt≥51 + mid-entropy) targeted subset.

Modes evaluated (5 seeds × 1,011 diseases):
  - concat_l2_raw            ← h1249 baseline (anchor)
  - concat_l2_znorm          ← z-norm only, no blend (tests whether z-norm
                                alone shifts AUPRC/AUROC)
  - soft_blend_w*            ← per-disease z(node2vec) blended with
                                z(concat_l2) ONLY on flipped subset; pure
                                z(concat_l2) elsewhere; w ∈ {0.0, 0.25, 0.5,
                                0.75, 1.0}.

w=0.0 ≡ concat_l2_znorm (sanity check); w=1.0 ≡ z-normed entropy_routed
(h1249 with z-norm fix). Sweep tells us the optimal blend ratio.

Outputs:
    data/analysis/h1255_soft_blend_fusion.{json,md}
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
    categorize,
    load_disease_names,
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

OUT_JSON = PROJECT_ROOT / "data/analysis/h1255_soft_blend_fusion.json"
OUT_MD = PROJECT_ROOT / "data/analysis/h1255_soft_blend_fusion.md"

# Sweep range
BLEND_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def z_normalise(scores: np.ndarray) -> np.ndarray:
    """Per-vector z-normalisation. Returns 0s if std is degenerate."""
    mu = float(scores.mean())
    sd = float(scores.std())
    if sd < 1e-9:
        return scores - mu
    return (scores - mu) / sd


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
    disease_names: Dict[str, str],
    n_gt_per_disease: Dict[str, int],
    entropy_per_disease: Dict[str, float],
    train_low_cut: float,
    train_high_cut: float,
    blend_weights: Tuple[float, ...],
) -> Dict:
    """Compute concat_l2_raw, concat_l2_znorm, and soft_blend_w* per seed."""
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
    per_drug_hits: Dict[str, Dict[int, List[float]]] = {
        m: {K: [] for K in (1, 5, 10, 30, 100)} for m in modes
    }
    triple_hits: Dict[str, Dict[int, int]] = {m: {K: 0 for K in (1, 5, 10, 30, 100)} for m in modes}
    triple_rr: Dict[str, List[float]] = {m: [] for m in modes}
    triple_n: Dict[str, int] = {m: 0 for m in modes}
    score_buf: Dict[str, List[np.ndarray]] = {m: [] for m in modes}
    label_buf: Dict[str, List[np.ndarray]] = {m: [] for m in modes}

    per_disease_rows: List[Dict] = []
    flipped_counter = 0
    n_test_diseases = 0

    for disease_id in holdout_ids:
        if disease_id not in lu_concat or disease_id not in lu_node2vec:
            continue
        gt_drugs = eval_gt.get(disease_id, set()) & universe
        if not gt_drugs:
            continue
        n_test_diseases += 1
        cat = categorize(disease_names.get(disease_id, disease_id))
        n_gt_disease = n_gt_per_disease.get(disease_id, 0)
        ent = entropy_per_disease.get(disease_id, 0.0)

        is_flipped = (
            n_gt_disease >= N_GT_HIGH_DENSITY
            and (train_low_cut <= ent < train_high_cut)
        )
        if is_flipped:
            flipped_counter += 1

        # Always compute scores under both single embeddings
        scores_n2v = score_one_disease(
            lu_node2vec[disease_id], train_emb_n2v, train_ids_ordered, train_knn_gt, k
        )
        scores_concat = score_one_disease(
            lu_concat[disease_id], train_emb_concat, train_ids_ordered, train_knn_gt, k
        )

        # Build full-candidate score vectors
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

        # Per-mode score vectors
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

        per_disease_metrics: Dict[str, Dict] = {}
        for mode, score_vec in mode_scores.items():
            order = np.argsort(-score_vec, kind="stable")
            rank_of_drug: Dict[str, int] = {}
            for r, idx in enumerate(order, start=1):
                rank_of_drug[cand_list[idx]] = r

            gt_ranks = [rank_of_drug[d] for d in gt_drugs if d in rank_of_drug]
            n_gt_eval = len(gt_drugs)
            hit_30 = sum(1 for r in gt_ranks if r <= 30)
            r30 = hit_30 / n_gt_eval
            per_drug_r30[mode].append(r30)
            for K in (1, 5, 10, 30, 100):
                hit_k = sum(1 for r in gt_ranks if r <= K)
                per_drug_hits[mode][K].append(hit_k / n_gt_eval)

            for d in gt_drugs:
                r = rank_of_drug.get(d)
                triple_n[mode] += 1
                if r is None:
                    triple_rr[mode].append(0.0)
                    continue
                triple_rr[mode].append(1.0 / r)
                for K in triple_hits[mode]:
                    if r <= K:
                        triple_hits[mode][K] += 1

            label_vec = np.zeros(len(cand_list), dtype=np.int8)
            for d in gt_drugs:
                if d in cand_index:
                    label_vec[cand_index[d]] = 1
            score_buf[mode].append(score_vec.copy())
            label_buf[mode].append(label_vec)

            per_disease_metrics[mode] = {"r30": r30, "hits30": hit_30}

        per_disease_rows.append({
            "disease_id": disease_id,
            "name": disease_names.get(disease_id, disease_id),
            "category": cat,
            "n_gt_eval": len(gt_drugs),
            "n_gt_total_egt": n_gt_disease,
            "atc_l3_entropy": ent,
            "is_flipped": is_flipped,
            "metrics": {m: per_disease_metrics[m] for m in modes},
        })

    out_metrics: Dict[str, Dict] = {}
    for m in modes:
        scores_flat = np.concatenate(score_buf[m]) if score_buf[m] else np.zeros(0)
        labels_flat = np.concatenate(label_buf[m]) if label_buf[m] else np.zeros(0)
        auprc = float(average_precision_score(labels_flat, scores_flat)) if labels_flat.sum() > 0 else 0.0
        auroc = (
            float(roc_auc_score(labels_flat, scores_flat))
            if 0 < labels_flat.sum() < len(labels_flat)
            else 0.0
        )
        out_metrics[m] = {
            "n_test_diseases": n_test_diseases,
            "n_test_triples": triple_n[m],
            "per_drug_r30": float(np.mean(per_drug_r30[m])) if per_drug_r30[m] else 0.0,
            "hits_at_k_drug": {K: float(np.mean(v)) if v else 0.0 for K, v in per_drug_hits[m].items()},
            "hits_at_k_triple": {K: v / triple_n[m] if triple_n[m] else 0.0 for K, v in triple_hits[m].items()},
            "mrr_triple": float(np.mean(triple_rr[m])) if triple_rr[m] else 0.0,
            "auprc": auprc,
            "auroc": auroc,
        }

    return {
        "modes": out_metrics,
        "per_disease_rows": per_disease_rows,
        "n_flipped": flipped_counter,
        "n_test_diseases": n_test_diseases,
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
    print(f"h1255: Soft-blend fusion benchmark — {args.prefix_a} + {args.prefix_b}")
    print(f"       blend_weights={BLEND_WEIGHTS}  k={args.k}  seeds={seeds}")
    print("=" * 72)

    t0 = time.time()
    print("Loading embeddings...")
    lu_a, _, _ = load_embeddings(args.prefix_a)
    lu_b, _, _ = load_embeddings(args.prefix_b)
    print(f"  {len(lu_a):,} (A) + {len(lu_b):,} (B) entities")
    print("Building concat_l2 lookup...")
    lu_concat = build_concat_lookup(lu_a, lu_b)
    print(f"  {len(lu_concat):,} entities in intersection")

    with open(PROJECT_ROOT / args.eval_gt) as f:
        raw_eval_gt = json.load(f)
    eval_gt = {d: set(drugs) for d, drugs in raw_eval_gt.items()}
    print(f"eval_gt: {len(eval_gt):,} diseases, {sum(len(v) for v in eval_gt.values()):,} pairs")

    with open(PROJECT_ROOT / args.knn_gt) as f:
        raw_knn_gt = json.load(f)
    if isinstance(raw_knn_gt, dict) and "ground_truth" in raw_knn_gt:
        raw_knn_gt = raw_knn_gt["ground_truth"]
    knn_gt = {d: set(drugs) for d, drugs in raw_knn_gt.items()}
    print(f"knn_gt: {len(knn_gt):,} diseases")

    with open(PROJECT_ROOT / args.db_lookup) as f:
        db_lookup = json.load(f)
    print(f"db_lookup: {len(db_lookup):,} DrugBank entries")
    mapper = ATCMapper()

    disease_names = load_disease_names()
    all_diseases = [d for d in knn_gt if d in lu_concat]
    print(f"Disease universe (knn_gt ∩ A ∩ B): {len(all_diseases):,}")

    print("Pre-computing per-disease n_gt + ATC L3 entropy...")
    n_gt_per_disease: Dict[str, int] = {}
    entropy_per_disease: Dict[str, float] = {}
    for d in all_diseases:
        gt_drugs = list(eval_gt.get(d, set()))
        n_gt_per_disease[d] = len(gt_drugs)
        ent, _ = disease_atc_l3_entropy(gt_drugs, db_lookup, mapper)
        entropy_per_disease[d] = ent
    print(f"  Setup complete in {time.time()-t0:.1f}s")

    per_seed_results: List[Dict] = []
    for si, seed in enumerate(seeds):
        print(f"\n=== SEED {seed} ({si+1}/{len(seeds)}) ===")
        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        print(f"Train: {len(train_ids)}  Holdout: {len(holdout_ids)}")

        train_high_density_entropies = sorted(
            entropy_per_disease[d] for d in train_ids if n_gt_per_disease[d] >= N_GT_HIGH_DENSITY
        )
        n_thd = len(train_high_density_entropies)
        if n_thd >= 3:
            cut_low = train_high_density_entropies[n_thd // 3]
            cut_high = train_high_density_entropies[2 * n_thd // 3]
        else:
            cut_low, cut_high = 3.18, 3.95
        print(f"  cuts low={cut_low:.3f}, high={cut_high:.3f}")

        candidate_drugs: Set[str] = set()
        for d in train_ids:
            if d in knn_gt:
                candidate_drugs |= (knn_gt[d] & set(lu_concat.keys()))

        tm0 = time.time()
        seed_out = evaluate_seed(
            holdout_ids=holdout_ids,
            train_ids=train_ids,
            lu_node2vec=lu_a,
            lu_concat=lu_concat,
            eval_gt=eval_gt,
            knn_gt=knn_gt,
            candidate_drugs=candidate_drugs,
            k=args.k,
            disease_names=disease_names,
            n_gt_per_disease=n_gt_per_disease,
            entropy_per_disease=entropy_per_disease,
            train_low_cut=cut_low,
            train_high_cut=cut_high,
            blend_weights=BLEND_WEIGHTS,
        )
        seed_out["seed"] = seed
        seed_out["elapsed_s"] = round(time.time() - tm0, 1)
        per_seed_results.append(seed_out)

        for m in seed_out["modes"]:
            mm = seed_out["modes"][m]
            print(
                f"  [{m:18s}] R@30={mm['per_drug_r30']*100:.2f}%  "
                f"H30={mm['hits_at_k_drug'][30]*100:.2f}%  "
                f"MRR={mm['mrr_triple']:.4f}  "
                f"AUPRC={mm['auprc']:.4f}  AUROC={mm['auroc']:.4f}"
            )
        print(f"  flipped={seed_out['n_flipped']} of {seed_out['n_test_diseases']}  ({seed_out['elapsed_s']}s)")

    # Aggregate
    print("\n" + "=" * 72)
    print("AGGREGATE (mean ± std across seeds)")
    print("=" * 72)
    all_modes = list(per_seed_results[0]["modes"].keys())
    agg: Dict[str, Dict] = {}
    for m in all_modes:
        rows = [r["modes"][m] for r in per_seed_results]
        agg[m] = {
            "r30": mean_std([r["per_drug_r30"] for r in rows]),
            "hits30_drug": mean_std([r["hits_at_k_drug"][30] for r in rows]),
            "mrr": mean_std([r["mrr_triple"] for r in rows]),
            "auprc": mean_std([r["auprc"] for r in rows]),
            "auroc": mean_std([r["auroc"] for r in rows]),
        }
        print(
            f"  {m:18s}  R@30={agg[m]['r30'][0]*100:5.2f}%±{agg[m]['r30'][1]*100:.2f}%  "
            f"H30={agg[m]['hits30_drug'][0]*100:5.2f}%  "
            f"MRR={agg[m]['mrr'][0]:.4f}  AUPRC={agg[m]['auprc'][0]:.4f}  "
            f"AUROC={agg[m]['auroc'][0]:.4f}"
        )

    # Per-seed paired-t vs concat_l2_raw anchor (the actual production baseline)
    anchor = "concat_l2_raw"
    print("\n" + "=" * 72)
    print(f"PAIRED-T (per-seed, n=5): each mode vs {anchor}")
    print("=" * 72)
    paired_t_seed: Dict[str, Dict[str, Dict]] = {}
    for m in all_modes:
        if m == anchor:
            continue
        per_metric: Dict[str, Dict] = {}
        for label, key in (
            ("R@30", "per_drug_r30"),
            ("H30drug", "hits_at_k_drug"),
            ("MRR", "mrr_triple"),
            ("AUPRC", "auprc"),
            ("AUROC", "auroc"),
        ):
            if key == "hits_at_k_drug":
                cand = [r["modes"][m]["hits_at_k_drug"][30] for r in per_seed_results]
                base = [r["modes"][anchor]["hits_at_k_drug"][30] for r in per_seed_results]
            else:
                cand = [r["modes"][m][key] for r in per_seed_results]
                base = [r["modes"][anchor][key] for r in per_seed_results]
            deltas = [c - b for c, b in zip(cand, base)]
            t_res = paired_t(deltas)
            per_metric[label] = {"mean": t_res["mean"], "t": t_res["t"], "p": t_res["p_two_sided"]}
        paired_t_seed[m] = per_metric

    for m in all_modes:
        if m == anchor:
            continue
        print(f"\n  {m} (vs {anchor}):")
        for label, rec in paired_t_seed[m].items():
            scale = 100.0 if label in ("R@30", "H30drug") else 1.0
            print(f"    {label:8s} Δ={rec['mean']*scale:+.4f}  t={rec['t']:+.3f}  p={rec['p']:.3g}")

    # Per-disease (1002 rows) paired-t for the best mode by R@30
    best_r30_mode = max((m for m in all_modes if m != anchor), key=lambda m: agg[m]["r30"][0])
    all_rows: List[Dict] = []
    for r in per_seed_results:
        for row in r["per_disease_rows"]:
            all_rows.append({**row, "seed": r["seed"]})

    print(f"\n  Per-disease paired-t (n={len(all_rows)}): {best_r30_mode} vs {anchor}")
    deltas_r30 = [row["metrics"][best_r30_mode]["r30"] - row["metrics"][anchor]["r30"] for row in all_rows]
    deltas_hits = [row["metrics"][best_r30_mode]["hits30"] - row["metrics"][anchor]["hits30"] for row in all_rows]
    t_r30 = paired_t(deltas_r30)
    t_hits = paired_t(deltas_hits)
    print(f"    R@30   n={t_r30['n']}  Δ_mean={t_r30['mean']*100:+.4f}pp  t={t_r30['t']:+.3f}  p={t_r30['p_two_sided']:.3g}")
    print(f"    hits30 n={t_hits['n']}  Δ_mean={t_hits['mean']:+.4f}    t={t_hits['t']:+.3f}  p={t_hits['p_two_sided']:.3g}")

    # Restricted to flipped subset
    flipped_rows = [r for r in all_rows if r["is_flipped"]]
    tr = th = None
    print(f"\n  Flipped-subset paired-t (n={len(flipped_rows)}): {best_r30_mode} vs {anchor}")
    if flipped_rows:
        d_r30 = [row["metrics"][best_r30_mode]["r30"] - row["metrics"][anchor]["r30"] for row in flipped_rows]
        d_hits = [row["metrics"][best_r30_mode]["hits30"] - row["metrics"][anchor]["hits30"] for row in flipped_rows]
        tr = paired_t(d_r30)
        th = paired_t(d_hits)
        print(f"    R@30   n={tr['n']}  Δ_mean={tr['mean']*100:+.4f}pp  t={tr['t']:+.3f}  p={tr['p_two_sided']:.3g}")
        print(f"    hits30 n={th['n']}  Δ_mean={th['mean']:+.4f}    t={th['t']:+.3f}  p={th['p_two_sided']:.3g}")

    # Save report
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "hypothesis": "h1255",
        "blend_weights": list(BLEND_WEIGHTS),
        "anchor": anchor,
        "best_r30_mode": best_r30_mode,
        "n_diseases_eligible": len(all_diseases),
        "aggregate": {m: {k: list(v) if isinstance(v, tuple) else v for k, v in d.items()} for m, d in agg.items()},
        "paired_t_per_seed_vs_anchor": paired_t_seed,
        "per_seed_summaries": [
            {
                "seed": r["seed"],
                "train_low_cut": r["train_low_cut"],
                "train_high_cut": r["train_high_cut"],
                "n_test_diseases": r["n_test_diseases"],
                "n_flipped": r["n_flipped"],
                "modes": r["modes"],
            }
            for r in per_seed_results
        ],
    }
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {OUT_JSON.relative_to(PROJECT_ROOT)}")

    # Markdown
    md = []
    md.append("# h1255 — Score-scale-normalised soft-blend fusion\n\n")
    md.append("**Premise:** h1228 + h1249 both lifted R@30 on a targeted subset but regressed AUPRC/AUROC ")
    md.append("at p<0.1, traceable to per-disease score-scale mismatch when swapping embedding spaces. ")
    md.append("This script tests whether per-disease z-normalisation + soft blend (instead of hard switch) ")
    md.append("recovers the targeted-subset lift WITHOUT the global pooled-AUPRC regression.\n\n")

    md.append("**Modes:**\n\n")
    md.append("- `concat_l2_raw` — production baseline (h1249 anchor)\n")
    md.append("- `concat_l2_znorm` — z-normalise concat_l2 scores per-disease; tests the AUPRC effect of z-norm alone\n")
    md.append("- `soft_blend_w*` — on (n_gt≥51 + mid-entropy) flipped subset only: `score = w·z(node2vec) + (1-w)·z(concat_l2)`. Other diseases use `z(concat_l2)`. Sweep w ∈ {0.0, 0.25, 0.5, 0.75, 1.0}.\n\n")

    md.append("## Aggregate (mean ± std across 5 seeds)\n\n")
    md.append("| Mode | R@30 | H@30 drug | MRR | AUPRC | AUROC |\n|---|---|---|---|---|---|\n")
    for m in all_modes:
        a = agg[m]
        md.append(
            f"| `{m}` | {a['r30'][0]*100:.2f}%±{a['r30'][1]*100:.2f}% | "
            f"{a['hits30_drug'][0]*100:.2f}%±{a['hits30_drug'][1]*100:.2f}% | "
            f"{a['mrr'][0]:.4f}±{a['mrr'][1]:.4f} | "
            f"{a['auprc'][0]:.4f}±{a['auprc'][1]:.4f} | "
            f"{a['auroc'][0]:.4f}±{a['auroc'][1]:.4f} |\n"
        )

    md.append(f"\n## Per-seed paired-t vs `{anchor}` (n=5)\n\n")
    md.append("| Mode | ΔR@30 | t (R@30) | p | ΔAUPRC | t (AUPRC) | p | ΔAUROC | t (AUROC) | p |\n")
    md.append("|---|---|---|---|---|---|---|---|---|---|\n")
    for m in all_modes:
        if m == anchor:
            continue
        rt = paired_t_seed[m]
        md.append(
            f"| `{m}` | "
            f"{rt['R@30']['mean']*100:+.4f}pp | {rt['R@30']['t']:+.2f} | {rt['R@30']['p']:.3g} | "
            f"{rt['AUPRC']['mean']:+.5f} | {rt['AUPRC']['t']:+.2f} | {rt['AUPRC']['p']:.3g} | "
            f"{rt['AUROC']['mean']:+.5f} | {rt['AUROC']['t']:+.2f} | {rt['AUROC']['p']:.3g} |\n"
        )

    md.append(f"\n## Per-disease paired-t (n={len(all_rows)}): best-R@30 mode `{best_r30_mode}` vs `{anchor}`\n\n")
    md.append(f"- R@30: Δ_mean = {t_r30['mean']*100:+.4f}pp, t={t_r30['t']:+.3f}, p={t_r30['p_two_sided']:.3g}\n")
    md.append(f"- hits@30: Δ_mean = {t_hits['mean']:+.4f}, t={t_hits['t']:+.3f}, p={t_hits['p_two_sided']:.3g}\n\n")

    if flipped_rows and tr is not None and th is not None:
        md.append(f"### Restricted to flipped subset (n={len(flipped_rows)} disease-seed rows)\n\n")
        md.append(f"- R@30: Δ_mean = {tr['mean']*100:+.4f}pp, t={tr['t']:+.3f}, p={tr['p_two_sided']:.3g}\n")
        md.append(f"- hits@30: Δ_mean = {th['mean']:+.4f}, t={th['t']:+.3f}, p={th['p_two_sided']:.3g}\n\n")

    md.append("## Per-seed details\n\n")
    md.append("| Seed | n_test | flipped | low_cut | high_cut |\n|---|---|---|---|---|\n")
    for r in per_seed_results:
        md.append(
            f"| {r['seed']} | {r['n_test_diseases']} | {r['n_flipped']} | "
            f"{r['train_low_cut']:.3f} | {r['train_high_cut']:.3f} |\n"
        )

    with open(OUT_MD, "w") as f:
        f.write("".join(md))
    print(f"Wrote {OUT_MD.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
