#!/usr/bin/env python3
"""
h1199: Clean embedding benchmark — tier-free multi-metric kNN evaluation.

Evaluates raw kNN retrieval quality on disease-level 80/20 holdout splits
across 5 seeds, reporting all five standard metrics:

  1. R@30 per-drug (clinical narrative, our recall metric)
  2. Hits@K (K=1,5,10,30,100) — both per-drug (avg across diseases) and
     per-test-triple (fraction of held-out drugs ranked in top-K)
  3. MRR per-test-triple (KG-completion standard)
  4. AUPRC per-test-triple (TxGNN headline metric)
  5. AUROC per-test-triple (reviewer expectation)

No tier rules, no rescue logic, no literature signals — just
cosine-similarity kNN over an embedding matrix. This gives a clean
embedding-quality ceiling that is comparable across embeddings.

Loads embeddings via OPEN_CURE_EMBEDDINGS_PREFIX (default: node2vec_256).
Uses expanded_ground_truth.json as GT.

Usage:
    python scripts/clean_embedding_benchmark.py [--k 20] [--seeds 42,123,456,789,2024]

Outputs:
    data/analysis/clean_benchmark_<prefix>.json  — full metric tables
    data/analysis/clean_benchmark_<prefix>.md    — markdown summary
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Pull CATEGORY_KEYWORDS + categorize_disease from production_predictor without
# triggering full predictor init (the categorisation tables are free-standing).
from production_predictor import CATEGORY_KEYWORDS  # type: ignore[import-not-found]  # noqa: E402


def categorize(disease_name: str) -> str:
    name_lower = disease_name.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in name_lower:
                return category
    return "other"


def load_embeddings(prefix: str) -> Tuple[Dict[str, np.ndarray], np.ndarray, List[str]]:
    """Load embeddings as dict{entity_id -> vector} keyed by full drkg: id.

    Returns (lookup, matrix, ordered_keys).
    """
    emb_dir = PROJECT_ROOT / "data" / "embeddings"
    entities = np.load(emb_dir / f"{prefix}_entities.npy", allow_pickle=True)
    matrix = np.load(emb_dir / f"{prefix}_embeddings.npy").astype(np.float32)
    keys = [f"drkg:{e}" for e in entities]
    lookup = {k: matrix[i] for i, k in enumerate(keys)}
    return lookup, matrix, keys


def load_disease_names() -> Dict[str, str]:
    """Return {drkg:Disease::...: human_name}.

    Source: data/cache/ground_truth_cache.json (maintained by
    production_predictor — richest coverage).
    """
    path = PROJECT_ROOT / "data" / "cache" / "ground_truth_cache.json"
    if path.exists():
        with open(path) as f:
            cache = json.load(f)
        raw = cache.get("disease_names", {})
        return {k if k.startswith("drkg:") else f"drkg:{k}": v for k, v in raw.items()}
    return {}


def split_diseases(all_diseases: List[str], seed: int, ratio: float = 0.8) -> Tuple[List[str], List[str]]:
    rng = np.random.RandomState(seed)
    shuffled = list(all_diseases)
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * ratio)
    return shuffled[:cut], shuffled[cut:]


def score_disease_knn(
    disease_emb: np.ndarray,
    train_emb: np.ndarray,
    train_disease_ids: List[str],
    train_gt: Dict[str, Set[str]],
    k: int,
) -> Dict[str, float]:
    """Return {drug_id: aggregated_score} from top-k train neighbours."""
    sims = cosine_similarity(disease_emb[None, :], train_emb)[0]
    if k >= len(sims):
        top_idx = np.argsort(sims)
    else:
        top_idx = np.argpartition(sims, -k)[-k:]
    drug_scores: Dict[str, float] = defaultdict(float)
    for idx in top_idx:
        neighbour = train_disease_ids[idx]
        for drug_id in train_gt.get(neighbour, ()):  # noqa: PLE0601
            drug_scores[drug_id] += float(sims[idx])
    return dict(drug_scores)


def compute_metrics_for_seed(
    holdout_ids: List[str],
    train_ids: List[str],
    lookup: Dict[str, np.ndarray],
    gt: Dict[str, Set[str]],
    train_gt: Dict[str, Set[str]],
    candidate_drugs: Set[str],
    k: int,
    disease_names: Dict[str, str],
) -> Dict:
    """Return a metrics dict for a single seed."""
    # Build train embedding matrix in fixed order
    train_ids_ordered = [d for d in train_ids if d in lookup]
    train_emb = np.stack([lookup[d] for d in train_ids_ordered])

    # Per-disease aggregation buckets
    per_drug_r30: List[float] = []
    hits_drug: Dict[int, List[float]] = {K: [] for K in (1, 5, 10, 30, 100)}
    # Ceiling-adjusted per-drug Hits@K: hits_K / min(K, |GT|) ∈ [0,1] (h1240).
    # Captures "fraction of the achievable ceiling recovered" and controls for the
    # recall-denominator artifact flagged by h1211 (large-GT categories can't
    # exceed min(K, |GT|)/|GT| on raw R@K).
    hits_drug_ceiling: Dict[int, List[float]] = {K: [] for K in (1, 5, 10, 30, 100)}
    hits_triple: Dict[int, int] = {K: 0 for K in (1, 5, 10, 30, 100)}
    reciprocal_ranks: List[float] = []

    # For global AUPRC/AUROC — flatten (disease, candidate_drug) scores with binary labels
    all_scores: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    # For per-category breakdown
    per_cat_r30: Dict[str, List[float]] = defaultdict(list)
    per_cat_r30_ceiling: Dict[str, List[float]] = defaultdict(list)  # h1240
    per_cat_ceiling: Dict[str, List[float]] = defaultdict(list)  # h1240
    per_cat_hits_triple_30: Dict[str, List[int]] = defaultdict(list)
    per_cat_hits_total: Dict[str, int] = defaultdict(int)
    per_cat_rr: Dict[str, List[float]] = defaultdict(list)

    n_triples = 0
    n_test_diseases = 0

    # Universe of rankable drugs for AUPRC/AUROC and rank computation.
    # = candidate_drugs (drugs in train-side knn_gt, always scorable)
    # + any held-out GT drug that isn't in candidate_drugs (so it still gets
    # a deterministic rank at the bottom rather than being skipped).
    all_holdout_gt_drugs: Set[str] = set()
    for did in holdout_ids:
        all_holdout_gt_drugs |= gt.get(did, set())
    universe = candidate_drugs | all_holdout_gt_drugs
    # Restrict to drugs with embeddings so AUPRC is computed on embedable
    # candidates (matches the "embedding-based benchmark" framing).
    universe = {d for d in universe if d in lookup}
    cand_list = sorted(universe)
    cand_index = {d: i for i, d in enumerate(cand_list)}

    for disease_id in holdout_ids:
        if disease_id not in lookup:
            continue
        gt_drugs = gt.get(disease_id, set())
        # Restrict GT to drugs with embeddings (un-embeddable drugs are
        # outside the benchmark's scope). Drugs unseen in training neighbours
        # are retained — they still receive a rank (score 0, tie-broken).
        gt_drugs = gt_drugs & universe
        if not gt_drugs:
            continue

        n_test_diseases += 1
        cat = categorize(disease_names.get(disease_id, disease_id))

        disease_emb = lookup[disease_id]
        drug_scores = score_disease_knn(
            disease_emb, train_emb, train_ids_ordered, train_gt, k
        )

        # Build full-candidate score vector; unscored drugs get 0.
        score_vec = np.zeros(len(cand_list), dtype=np.float32)
        for drug, sc in drug_scores.items():
            if drug in cand_index:
                score_vec[cand_index[drug]] = sc

        # Ranking: argsort descending with stable tie-break on drug_id (already
        # lexicographic because cand_list is sorted — stable sort preserves order).
        order = np.argsort(-score_vec, kind="stable")
        rank_of_drug: Dict[str, int] = {}
        for r, idx in enumerate(order, start=1):
            rank_of_drug[cand_list[idx]] = r

        # Per-drug metrics (over full held-out GT)
        gt_ranks = [rank_of_drug[d] for d in gt_drugs if d in rank_of_drug]
        hit_30 = sum(1 for r in gt_ranks if r <= 30)
        n_gt = len(gt_drugs)
        per_drug_r30.append(hit_30 / n_gt)
        per_cat_r30[cat].append(hit_30 / n_gt)
        # Ceiling-adjusted R@30: hits / min(30, |GT|) ∈ [0,1] (h1240).
        ceil_30 = min(30, n_gt) / n_gt
        per_cat_ceiling[cat].append(ceil_30)
        per_cat_r30_ceiling[cat].append(hit_30 / min(30, n_gt))
        for K in hits_drug:
            hit_k = sum(1 for r in gt_ranks if r <= K)
            hits_drug[K].append(hit_k / n_gt)
            hits_drug_ceiling[K].append(hit_k / min(K, n_gt))

        # Per-triple metrics (every held-out GT drug is a test triple, even
        # those that were never scored by kNN).
        for d in gt_drugs:
            r = rank_of_drug.get(d)
            n_triples += 1
            if r is None:
                # Unrankable (shouldn't happen because universe covers it) — treat as bottom.
                reciprocal_ranks.append(0.0)
                per_cat_rr[cat].append(0.0)
                per_cat_hits_triple_30[cat].append(0)
                per_cat_hits_total[cat] += 1
                continue
            reciprocal_ranks.append(1.0 / r)
            per_cat_rr[cat].append(1.0 / r)
            for K in hits_triple:
                if r <= K:
                    hits_triple[K] += 1
            per_cat_hits_triple_30[cat].append(1 if r <= 30 else 0)
            per_cat_hits_total[cat] += 1

        # AUPRC / AUROC contributions
        label_vec = np.zeros(len(cand_list), dtype=np.int8)
        for d in gt_drugs:
            if d in cand_index:
                label_vec[cand_index[d]] = 1
        all_scores.append(score_vec)
        all_labels.append(label_vec)

    # Aggregate
    scores_flat = np.concatenate(all_scores) if all_scores else np.zeros(0)
    labels_flat = np.concatenate(all_labels) if all_labels else np.zeros(0)
    auprc = float(average_precision_score(labels_flat, scores_flat)) if labels_flat.sum() > 0 else 0.0
    auroc = (
        float(roc_auc_score(labels_flat, scores_flat))
        if 0 < labels_flat.sum() < len(labels_flat)
        else 0.0
    )

    metrics = {
        "n_test_diseases": n_test_diseases,
        "n_test_triples": n_triples,
        "per_drug_r30": float(np.mean(per_drug_r30)) if per_drug_r30 else 0.0,
        "hits_at_k_drug": {K: float(np.mean(v)) if v else 0.0 for K, v in hits_drug.items()},
        "hits_at_k_drug_ceiling": {
            K: float(np.mean(v)) if v else 0.0 for K, v in hits_drug_ceiling.items()
        },
        "hits_at_k_triple": {K: v / n_triples if n_triples else 0.0 for K, v in hits_triple.items()},
        "mrr_triple": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "auprc": auprc,
        "auroc": auroc,
        "per_category": {
            cat: {
                "n_diseases": len(per_cat_r30[cat]),
                "r30": float(np.mean(per_cat_r30[cat])) if per_cat_r30[cat] else 0.0,
                "r30_ceiling_mean": (
                    float(np.mean(per_cat_ceiling[cat]))
                    if per_cat_ceiling[cat] else 0.0
                ),
                "r30_over_ceiling": (
                    float(np.mean(per_cat_r30_ceiling[cat]))
                    if per_cat_r30_ceiling[cat] else 0.0
                ),
                "hits30_triple": (
                    float(np.mean(per_cat_hits_triple_30[cat]))
                    if per_cat_hits_triple_30[cat]
                    else 0.0
                ),
                "mrr": float(np.mean(per_cat_rr[cat])) if per_cat_rr[cat] else 0.0,
                "n_triples": per_cat_hits_total[cat],
            }
            for cat in per_cat_r30
        },
    }
    return metrics


def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=20, help="kNN neighbours (default 20)")
    ap.add_argument(
        "--seeds",
        type=str,
        default="42,123,456,789,2024",
        help="comma-separated disease-split seeds",
    )
    ap.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="embedding prefix; falls back to OPEN_CURE_EMBEDDINGS_PREFIX or node2vec_256",
    )
    ap.add_argument(
        "--eval-gt",
        type=str,
        default="data/reference/expanded_ground_truth.json",
        help="GT used for evaluation (per-drug hits). Default: expanded_ground_truth.json",
    )
    ap.add_argument(
        "--knn-gt",
        type=str,
        default="data/cache/ground_truth_cache.json",
        help=(
            "GT used for kNN neighbour aggregation (production convention uses "
            "the internal indicationList GT cache). Accepts either a plain "
            "{disease:[drugs]} JSON or the production cache (extracts "
            "'ground_truth' key)."
        ),
    )
    ap.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="output file prefix (default: data/analysis/clean_benchmark_<prefix>)",
    )
    ap.add_argument(
        "--restrict-to-embedding",
        type=str,
        default=None,
        help=(
            "If set, restrict the disease universe to diseases covered by this "
            "other embedding prefix. Lets us do apples-to-apples comparisons "
            "against an embedding with narrower disease coverage."
        ),
    )
    args = ap.parse_args()

    prefix = args.prefix or os.environ.get("OPEN_CURE_EMBEDDINGS_PREFIX", "node2vec_256")
    seeds = [int(s) for s in args.seeds.split(",")]

    print("=" * 72)
    print(f"h1199: Clean embedding benchmark — prefix={prefix}, k={args.k}, seeds={seeds}")
    print("=" * 72)

    t0 = time.time()
    print("Loading embeddings...")
    lookup, _, _ = load_embeddings(prefix)
    print(f"  {len(lookup):,} entities in {time.time()-t0:.1f}s")

    print("Loading evaluation ground truth...")
    with open(PROJECT_ROOT / args.eval_gt) as f:
        raw_eval_gt = json.load(f)
    eval_gt = {d: set(drugs) for d, drugs in raw_eval_gt.items()}
    print(
        f"  eval_gt: {len(eval_gt):,} diseases, "
        f"{sum(len(v) for v in eval_gt.values()):,} pairs ({args.eval_gt})"
    )

    print("Loading kNN-aggregation ground truth...")
    with open(PROJECT_ROOT / args.knn_gt) as f:
        raw_knn_gt = json.load(f)
    # Support both plain dict and production cache format
    if isinstance(raw_knn_gt, dict) and "ground_truth" in raw_knn_gt:
        raw_knn_gt = raw_knn_gt["ground_truth"]
    knn_gt = {d: set(drugs) for d, drugs in raw_knn_gt.items()}
    print(
        f"  knn_gt: {len(knn_gt):,} diseases, "
        f"{sum(len(v) for v in knn_gt.values()):,} pairs ({args.knn_gt})"
    )

    print("Loading disease names...")
    disease_names = load_disease_names()
    print(f"  {len(disease_names):,} disease names")

    # Disease universe: diseases in knn_gt with embeddings. This matches the
    # production predictor's "train_diseases" set (diseases eligible as kNN
    # neighbours and as test targets).
    all_diseases = [d for d in knn_gt if d in lookup]
    print(f"Disease universe (knn_gt ∩ embeddings): {len(all_diseases):,}")

    if args.restrict_to_embedding:
        other_lookup, _, _ = load_embeddings(args.restrict_to_embedding)
        before = len(all_diseases)
        all_diseases = [d for d in all_diseases if d in other_lookup]
        print(
            f"Restricted to intersection with {args.restrict_to_embedding}: "
            f"{before} → {len(all_diseases)}"
        )

    per_seed_results: List[Dict] = []

    for si, seed in enumerate(seeds):
        print(f"\n--- SEED {seed} ({si+1}/{len(seeds)}) ---")
        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        print(f"Train: {len(train_ids)}  Holdout: {len(holdout_ids)}")

        # kNN aggregation GT (production convention: internal indicationList GT).
        # Holdout diseases are fully hidden from aggregation.
        train_knn_gt = {d: (knn_gt[d] & set(lookup.keys())) for d in train_ids if d in knn_gt}
        # Candidate drug pool: drugs that appear in any training disease's
        # kNN-aggregation GT (these are the only drugs kNN can possibly score
        # non-zero).
        candidate_drugs: Set[str] = set()
        for drugs in train_knn_gt.values():
            candidate_drugs |= drugs
        print(f"Candidate drug pool (train knn_gt): {len(candidate_drugs):,}")

        t1 = time.time()
        metrics = compute_metrics_for_seed(
            holdout_ids=holdout_ids,
            train_ids=train_ids,
            lookup=lookup,
            gt=eval_gt,
            train_gt=train_knn_gt,
            candidate_drugs=candidate_drugs,
            k=args.k,
            disease_names=disease_names,
        )
        metrics["seed"] = seed
        metrics["elapsed_s"] = round(time.time() - t1, 1)
        per_seed_results.append(metrics)

        print(f"  n_test_diseases={metrics['n_test_diseases']}  n_triples={metrics['n_test_triples']}")
        print(f"  R@30 (per-drug):      {metrics['per_drug_r30']*100:.2f}%")
        print(f"  Hits@10 (per-drug):   {metrics['hits_at_k_drug'][10]*100:.2f}%")
        print(f"  Hits@30 (per-triple): {metrics['hits_at_k_triple'][30]*100:.2f}%")
        print(f"  MRR:                  {metrics['mrr_triple']:.4f}")
        print(f"  AUPRC:                {metrics['auprc']:.4f}")
        print(f"  AUROC:                {metrics['auroc']:.4f}")
        print(f"  elapsed: {metrics['elapsed_s']}s")

    # Aggregate across seeds
    def collect(getter) -> List[float]:
        return [getter(m) for m in per_seed_results]

    agg = {
        "per_drug_r30": mean_std(collect(lambda m: m["per_drug_r30"])),
        "mrr_triple": mean_std(collect(lambda m: m["mrr_triple"])),
        "auprc": mean_std(collect(lambda m: m["auprc"])),
        "auroc": mean_std(collect(lambda m: m["auroc"])),
        "hits_at_k_drug": {
            K: mean_std(collect(lambda m, K=K: m["hits_at_k_drug"][K])) for K in (1, 5, 10, 30, 100)
        },
        "hits_at_k_drug_ceiling": {
            K: mean_std(collect(lambda m, K=K: m["hits_at_k_drug_ceiling"][K]))
            for K in (1, 5, 10, 30, 100)
        },
        "hits_at_k_triple": {
            K: mean_std(collect(lambda m, K=K: m["hits_at_k_triple"][K])) for K in (1, 5, 10, 30, 100)
        },
    }

    # Per-category aggregation (mean over seeds)
    cats_seen = set()
    for m in per_seed_results:
        cats_seen.update(m["per_category"].keys())
    per_cat_agg = {}
    for cat in sorted(cats_seen):
        r30s = [m["per_category"].get(cat, {}).get("r30", 0.0) for m in per_seed_results if cat in m["per_category"]]
        mrrs = [m["per_category"].get(cat, {}).get("mrr", 0.0) for m in per_seed_results if cat in m["per_category"]]
        h30s = [m["per_category"].get(cat, {}).get("hits30_triple", 0.0) for m in per_seed_results if cat in m["per_category"]]
        ns = [m["per_category"].get(cat, {}).get("n_diseases", 0) for m in per_seed_results if cat in m["per_category"]]
        ceils = [m["per_category"].get(cat, {}).get("r30_ceiling_mean", 0.0) for m in per_seed_results if cat in m["per_category"]]
        r30_oc = [m["per_category"].get(cat, {}).get("r30_over_ceiling", 0.0) for m in per_seed_results if cat in m["per_category"]]
        per_cat_agg[cat] = {
            "r30_mean": float(np.mean(r30s)) if r30s else 0.0,
            "r30_std": float(np.std(r30s)) if r30s else 0.0,
            "r30_ceiling_mean": float(np.mean(ceils)) if ceils else 0.0,
            "r30_over_ceiling_mean": float(np.mean(r30_oc)) if r30_oc else 0.0,
            "r30_over_ceiling_std": float(np.std(r30_oc)) if r30_oc else 0.0,
            "hits30_triple_mean": float(np.mean(h30s)) if h30s else 0.0,
            "mrr_mean": float(np.mean(mrrs)) if mrrs else 0.0,
            "n_diseases_mean": float(np.mean(ns)) if ns else 0.0,
        }

    # Write JSON
    out_prefix = args.output_prefix or f"data/analysis/clean_benchmark_{prefix}"
    out_json = PROJECT_ROOT / f"{out_prefix}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "hypothesis": "h1199",
        "embedding_prefix": prefix,
        "k": args.k,
        "seeds": seeds,
        "eval_gt_path": args.eval_gt,
        "knn_gt_path": args.knn_gt,
        "n_diseases_eligible": len(all_diseases),
        "per_seed": per_seed_results,
        "aggregate": agg,
        "per_category_aggregate": per_cat_agg,
    }
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out_json}")

    # Write markdown summary
    out_md = PROJECT_ROOT / f"{out_prefix}.md"
    lines = []
    lines.append(f"# Clean Embedding Benchmark — `{prefix}`")
    lines.append("")
    lines.append(f"- Hypothesis: **h1199** (tier-free multi-metric kNN baseline)")
    lines.append(f"- k = {args.k}, seeds = {seeds}")
    lines.append(f"- eval_gt: `{args.eval_gt}`")
    lines.append(f"- knn_gt (aggregation): `{args.knn_gt}`")
    lines.append(f"- Eligible diseases (GT + embeddings): **{len(all_diseases):,}**")
    lines.append("")
    lines.append("## Headline metrics (mean ± std across seeds)")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| R@30 per-drug | {agg['per_drug_r30'][0]*100:.2f}% ± {agg['per_drug_r30'][1]*100:.2f}% |")
    _r30c_mean, _r30c_std = agg["hits_at_k_drug_ceiling"][30]
    lines.append(f"| R@30 per-drug / ceiling (h1240) | {_r30c_mean*100:.2f}% ± {_r30c_std*100:.2f}% |")
    lines.append(f"| MRR (per-triple) | {agg['mrr_triple'][0]:.4f} ± {agg['mrr_triple'][1]:.4f} |")
    lines.append(f"| AUPRC (per-triple) | {agg['auprc'][0]:.4f} ± {agg['auprc'][1]:.4f} |")
    lines.append(f"| AUROC (per-triple) | {agg['auroc'][0]:.4f} ± {agg['auroc'][1]:.4f} |")
    lines.append("")
    lines.append("## Hits@K — per-drug (fraction of GT drugs recovered in top-K, averaged over test diseases)")
    lines.append("")
    lines.append("| K | mean | std | ceiling-adjusted mean (hits/min(K,|GT|)) | ceiling-adj std |")
    lines.append("|---|---|---|---|---|")
    for K in (1, 5, 10, 30, 100):
        m, s = agg["hits_at_k_drug"][K]
        mc, sc = agg["hits_at_k_drug_ceiling"][K]
        lines.append(f"| {K} | {m*100:.2f}% | {s*100:.2f}% | {mc*100:.2f}% | {sc*100:.2f}% |")
    lines.append("")
    lines.append("## Hits@K — per-test-triple (fraction of held-out (disease, drug) pairs ranked ≤ K)")
    lines.append("")
    lines.append("| K | mean | std |")
    lines.append("|---|---|---|")
    for K in (1, 5, 10, 30, 100):
        m, s = agg["hits_at_k_triple"][K]
        lines.append(f"| {K} | {m*100:.2f}% | {s*100:.2f}% |")
    lines.append("")
    lines.append("## Per-category breakdown (averaged over seeds)")
    lines.append("")
    lines.append("| Category | n_dis | R@30 | Ceiling | R@30/Ceiling | Hits@30 triple | MRR |")
    lines.append("|---|---|---|---|---|---|---|")
    for cat, m in sorted(per_cat_agg.items(), key=lambda x: -x[1]["n_diseases_mean"]):
        lines.append(
            f"| {cat} | {m['n_diseases_mean']:.0f} | "
            f"{m['r30_mean']*100:.2f}%±{m['r30_std']*100:.2f}% | "
            f"{m.get('r30_ceiling_mean', 0.0)*100:.2f}% | "
            f"{m.get('r30_over_ceiling_mean', 0.0)*100:.2f}%±{m.get('r30_over_ceiling_std', 0.0)*100:.2f}% | "
            f"{m['hits30_triple_mean']*100:.2f}% | "
            f"{m['mrr_mean']:.4f} |"
        )
    lines.append("")
    lines.append("## Per-seed detail")
    lines.append("")
    lines.append("| Seed | n_dis | n_triples | R@30 | H@10 drug | H@30 triple | MRR | AUPRC | AUROC |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for m in per_seed_results:
        lines.append(
            f"| {m['seed']} | {m['n_test_diseases']} | {m['n_test_triples']} | "
            f"{m['per_drug_r30']*100:.2f}% | {m['hits_at_k_drug'][10]*100:.2f}% | "
            f"{m['hits_at_k_triple'][30]*100:.2f}% | {m['mrr_triple']:.4f} | "
            f"{m['auprc']:.4f} | {m['auroc']:.4f} |"
        )
    lines.append("")
    with open(out_md, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
