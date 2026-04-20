#!/usr/bin/env python3
"""h1214: Reconcile 71.2% (compare_honest_embeddings.py) vs 49.5% (h1212 clean benchmark)
treatment-edge leakage retention numbers.

Decomposes the gap by running the SAME kNN pipeline under all 8 combinations of:
  - embedding: full Node2Vec vs no-treatment Node2Vec
  - aggregation: micro (total_hits / total_gt) vs macro (per-drug avg across diseases)
  - GT source: internal indicationList cache vs expanded_ground_truth.json

All 8 cells use:
  - 80/20 disease holdout, 5 seeds, k=20 kNN cosine similarity
  - Apples-to-apples disease universe: diseases covered by BOTH embeddings AND in GT

This resolves CLAUDE.md's 71.2% → 49.5% discrepancy and picks the defensible
external-citation number.

Outputs:
  data/analysis/h1214_leakage_reconcile.json
  data/analysis/h1214_leakage_reconcile.md
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

SEEDS = [42, 123, 456, 789, 2024]
K = 20
TOP_N = 30
EMB_DIR = PROJECT_ROOT / "data" / "embeddings"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"


def load_embeddings(prefix: str) -> Dict[str, np.ndarray]:
    entities = np.load(EMB_DIR / f"{prefix}_entities.npy", allow_pickle=True)
    matrix = np.load(EMB_DIR / f"{prefix}_embeddings.npy").astype(np.float32)
    return {f"drkg:{e}": matrix[i] for i, e in enumerate(entities)}


def load_internal_gt() -> Dict[str, Set[str]]:
    with open(PROJECT_ROOT / "data" / "cache" / "ground_truth_cache.json") as f:
        cache = json.load(f)
    raw = cache["ground_truth"]
    return {d: set(drugs) for d, drugs in raw.items()}


def load_expanded_gt() -> Dict[str, Set[str]]:
    with open(PROJECT_ROOT / "data" / "reference" / "expanded_ground_truth.json") as f:
        raw = json.load(f)
    return {d: set(drugs) for d, drugs in raw.items()}


def disease_split(diseases: List[str], seed: int, ratio: float = 0.8) -> Tuple[List[str], List[str]]:
    rng = np.random.RandomState(seed)
    shuffled = list(diseases)
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * ratio)
    return shuffled[:cut], shuffled[cut:]


def score_kNN(
    disease_emb: np.ndarray,
    train_emb: np.ndarray,
    train_disease_ids: List[str],
    train_gt: Dict[str, Set[str]],
    k: int,
) -> Dict[str, float]:
    sims = cosine_similarity(disease_emb[None, :], train_emb)[0]
    if k >= len(sims):
        top_idx = np.argsort(sims)
    else:
        top_idx = np.argpartition(sims, -k)[-k:]
    scores: Dict[str, float] = defaultdict(float)
    for idx in top_idx:
        neighbour = train_disease_ids[idx]
        for drug in train_gt.get(neighbour, ()):
            scores[drug] += float(sims[idx])
    return dict(scores)


def evaluate_one_seed(
    lookup: Dict[str, np.ndarray],
    train_ids: List[str],
    test_ids: List[str],
    knn_gt: Dict[str, Set[str]],
    eval_gt: Dict[str, Set[str]],
    k: int,
    top_n: int,
) -> Dict[str, float]:
    """Returns {'micro_r30', 'macro_r30', 'n_test_diseases', 'n_micro_gt'}."""
    train_ids = [d for d in train_ids if d in lookup]
    train_emb = np.stack([lookup[d] for d in train_ids])

    # kNN aggregation GT: restrict to entities with embeddings (so a drug can be
    # scored non-zero).
    train_knn_gt = {d: {x for x in knn_gt.get(d, set()) if x in lookup} for d in train_ids}

    per_disease_r30: List[float] = []
    total_hits = 0
    total_gt = 0

    for did in test_ids:
        if did not in lookup:
            continue
        gt_drugs = eval_gt.get(did, set()) & set(lookup.keys())
        if not gt_drugs:
            continue

        scores = score_kNN(lookup[did], train_emb, train_ids, train_knn_gt, k)
        # Top-N drugs by score
        if scores:
            sorted_drugs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top = {d for d, _ in sorted_drugs[:top_n]}
        else:
            top = set()

        hits = len(top & gt_drugs)
        per_disease_r30.append(hits / len(gt_drugs))
        total_hits += hits
        total_gt += len(gt_drugs)

    return {
        "micro_r30": total_hits / total_gt if total_gt > 0 else 0.0,
        "macro_r30": float(np.mean(per_disease_r30)) if per_disease_r30 else 0.0,
        "n_test_diseases": len(per_disease_r30),
        "n_micro_gt": total_gt,
        "n_micro_hits": total_hits,
    }


def run_cell(
    emb_prefix: str,
    diseases: List[str],
    knn_gt: Dict[str, Set[str]],
    eval_gt: Dict[str, Set[str]],
    lookup: Dict[str, np.ndarray],
    label: str,
) -> Dict:
    per_seed_micro: List[float] = []
    per_seed_macro: List[float] = []
    per_seed_meta: List[Dict] = []
    for seed in SEEDS:
        train_ids, test_ids = disease_split(diseases, seed)
        r = evaluate_one_seed(lookup, train_ids, test_ids, knn_gt, eval_gt, K, TOP_N)
        per_seed_micro.append(r["micro_r30"])
        per_seed_macro.append(r["macro_r30"])
        per_seed_meta.append(r)
    return {
        "label": label,
        "embedding": emb_prefix,
        "micro_r30_mean": float(np.mean(per_seed_micro)),
        "micro_r30_std": float(np.std(per_seed_micro)),
        "macro_r30_mean": float(np.mean(per_seed_macro)),
        "macro_r30_std": float(np.std(per_seed_macro)),
        "per_seed_micro": per_seed_micro,
        "per_seed_macro": per_seed_macro,
        "n_eligible_diseases": len(diseases),
    }


def main() -> None:
    t0 = time.time()
    print("=" * 72)
    print("h1214: Reconcile 71.2% vs 49.5% leakage retention")
    print("=" * 72)

    print("\nLoading embeddings...")
    full_lookup = load_embeddings("node2vec_256")
    no_trt_lookup = load_embeddings("node2vec_256_no_treatment")
    print(f"  node2vec_256:             {len(full_lookup):,} entities")
    print(f"  node2vec_256_no_treatment: {len(no_trt_lookup):,} entities")

    print("\nLoading ground truths...")
    internal_gt = load_internal_gt()
    expanded_gt = load_expanded_gt()
    print(f"  internal GT:  {len(internal_gt):,} diseases, "
          f"{sum(len(v) for v in internal_gt.values()):,} pairs")
    print(f"  expanded GT:  {len(expanded_gt):,} diseases, "
          f"{sum(len(v) for v in expanded_gt.values()):,} pairs")

    # Apples-to-apples disease universe: in BOTH embeddings AND internal GT
    # (so same training pool is available for both cells).
    base_diseases = [
        d for d in internal_gt
        if d in full_lookup and d in no_trt_lookup
    ]
    print(f"\nApples-to-apples disease universe (in both embeddings ∩ internal GT):"
          f" {len(base_diseases):,}")

    # Also compute each embedding's individual universe
    full_only_diseases = [d for d in internal_gt if d in full_lookup]
    no_trt_only_diseases = [d for d in internal_gt if d in no_trt_lookup]
    print(f"  full-only eligible:       {len(full_only_diseases):,}")
    print(f"  no_treatment-only:        {len(no_trt_only_diseases):,}")

    # Run the 8 cells
    cells = []

    for emb_label, lookup in [("full", full_lookup), ("no_treatment", no_trt_lookup)]:
        for gt_label, eval_gt in [("internal_gt", internal_gt), ("expanded_gt", expanded_gt)]:
            print(f"\n--- CELL: emb={emb_label}  eval_gt={gt_label}  universe=common ---")
            r = run_cell(
                emb_prefix=emb_label,
                diseases=base_diseases,
                knn_gt=internal_gt,  # knn aggregation GT is ALWAYS internal (production convention)
                eval_gt=eval_gt,
                lookup=lookup,
                label=f"{emb_label}__{gt_label}",
            )
            print(f"  micro R@30: {r['micro_r30_mean']*100:.2f}% ± {r['micro_r30_std']*100:.2f}%")
            print(f"  macro R@30: {r['macro_r30_mean']*100:.2f}% ± {r['macro_r30_std']*100:.2f}%")
            cells.append(r)

    # Also reproduce the compare_honest_embeddings.py methodology: use each
    # embedding's FULL universe (no restriction to the intersection). This
    # captures the effect of embedding-coverage differences.
    print(f"\n--- AUX: full embedding, full-only universe ---")
    aux_full = run_cell(
        emb_prefix="full",
        diseases=full_only_diseases,
        knn_gt=internal_gt,
        eval_gt=internal_gt,
        lookup=full_lookup,
        label="full__internal_gt__full_universe",
    )
    print(f"  micro R@30: {aux_full['micro_r30_mean']*100:.2f}% ± {aux_full['micro_r30_std']*100:.2f}%")
    print(f"  macro R@30: {aux_full['macro_r30_mean']*100:.2f}% ± {aux_full['macro_r30_std']*100:.2f}%")

    print(f"\n--- AUX: no_treatment embedding, no_treatment-only universe ---")
    aux_no_trt = run_cell(
        emb_prefix="no_treatment",
        diseases=no_trt_only_diseases,
        knn_gt=internal_gt,
        eval_gt=internal_gt,
        lookup=no_trt_lookup,
        label="no_treatment__internal_gt__no_treatment_universe",
    )
    print(f"  micro R@30: {aux_no_trt['micro_r30_mean']*100:.2f}% ± {aux_no_trt['micro_r30_std']*100:.2f}%")
    print(f"  macro R@30: {aux_no_trt['macro_r30_mean']*100:.2f}% ± {aux_no_trt['macro_r30_std']*100:.2f}%")

    aux_cells = [aux_full, aux_no_trt]

    # Compute retention for each (aggregation, GT) combo on the common universe
    def _retention(a: float, b: float) -> float:
        return 100.0 * a / b if b > 0 else 0.0

    # Build a lookup from labels
    cell_by = {c["label"]: c for c in cells}
    retentions: List[Dict] = []
    for gt_label in ["internal_gt", "expanded_gt"]:
        full = cell_by[f"full__{gt_label}"]
        nt = cell_by[f"no_treatment__{gt_label}"]
        for agg in ["micro_r30", "macro_r30"]:
            full_val = full[f"{agg}_mean"]
            nt_val = nt[f"{agg}_mean"]
            retentions.append({
                "gt": gt_label,
                "aggregation": agg,
                "universe": "common",
                "full_r30": full_val,
                "no_treatment_r30": nt_val,
                "retention_pct": _retention(nt_val, full_val),
            })

    # Also the "each embedding on its own universe" (mimics compare_honest_embeddings.py)
    retentions.append({
        "gt": "internal_gt",
        "aggregation": "micro_r30",
        "universe": "embedding-native",
        "full_r30": aux_full["micro_r30_mean"],
        "no_treatment_r30": aux_no_trt["micro_r30_mean"],
        "retention_pct": _retention(aux_no_trt["micro_r30_mean"], aux_full["micro_r30_mean"]),
    })
    retentions.append({
        "gt": "internal_gt",
        "aggregation": "macro_r30",
        "universe": "embedding-native",
        "full_r30": aux_full["macro_r30_mean"],
        "no_treatment_r30": aux_no_trt["macro_r30_mean"],
        "retention_pct": _retention(aux_no_trt["macro_r30_mean"], aux_full["macro_r30_mean"]),
    })

    # Emit artifacts
    out_json = ANALYSIS_DIR / "h1214_leakage_reconcile.json"
    out_md = ANALYSIS_DIR / "h1214_leakage_reconcile.md"

    report = {
        "hypothesis": "h1214",
        "description": "Reconcile 71.2% vs 49.5% treatment-edge leakage retention",
        "seeds": SEEDS,
        "k": K,
        "top_n": TOP_N,
        "n_common_universe": len(base_diseases),
        "n_full_universe": len(full_only_diseases),
        "n_no_treatment_universe": len(no_trt_only_diseases),
        "cells_common_universe": cells,
        "cells_embedding_native_universe": aux_cells,
        "retentions": retentions,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out_json}")

    # Markdown table
    lines = [
        "# h1214 — Reconcile Treatment-Edge Leakage Retention",
        "",
        "**Question:** Why does CLAUDE.md cite 71.2% retention after removing treatment edges",
        "while h1212's clean_embedding_benchmark says 49.5%?",
        "",
        f"- Seeds: {SEEDS}",
        f"- k = {K}, top-N = {TOP_N}",
        f"- Common disease universe (both embeddings ∩ internal GT): **{len(base_diseases):,}**",
        f"- Full-only eligible diseases: {len(full_only_diseases):,}",
        f"- No-treatment-only eligible diseases: {len(no_trt_only_diseases):,}",
        "",
        "## 8-cell factorial (common universe)",
        "",
        "| Embedding | GT | micro R@30 | macro R@30 |",
        "|---|---|---|---|",
    ]
    for c in cells:
        emb, gt = c["label"].split("__")
        lines.append(
            f"| {emb} | {gt} | "
            f"{c['micro_r30_mean']*100:.2f}%±{c['micro_r30_std']*100:.2f}% | "
            f"{c['macro_r30_mean']*100:.2f}%±{c['macro_r30_std']*100:.2f}% |"
        )
    lines.append("")
    lines.append("## Retention (no_treatment / full) — by methodology")
    lines.append("")
    lines.append("| Universe | GT | Aggregation | full R@30 | no_treatment R@30 | retention |")
    lines.append("|---|---|---|---|---|---|")
    for r in retentions:
        lines.append(
            f"| {r['universe']} | {r['gt']} | {r['aggregation']} | "
            f"{r['full_r30']*100:.2f}% | {r['no_treatment_r30']*100:.2f}% | "
            f"{r['retention_pct']:.1f}% |"
        )
    lines.append("")
    lines.append("## Embedding-native universe (reproduces compare_honest_embeddings.py)")
    lines.append("")
    lines.append("| Embedding | N eligible | micro R@30 | macro R@30 |")
    lines.append("|---|---|---|---|")
    for c in aux_cells:
        lines.append(
            f"| {c['embedding']} | {c['n_eligible_diseases']} | "
            f"{c['micro_r30_mean']*100:.2f}%±{c['micro_r30_std']*100:.2f}% | "
            f"{c['macro_r30_mean']*100:.2f}%±{c['macro_r30_std']*100:.2f}% |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("The retention number depends on three methodology choices. See the JSON for")
    lines.append("full per-seed detail.")
    lines.append("")

    with open(out_md, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
