#!/usr/bin/env python3
"""h1249: Production routing benchmark — wire (n_gt × ATC entropy) routing.

Implements the two-axis routing rule discovered by h1247 + h1248:

    n_gt < 21              → concat_l2 (default fuse; no signal in h1247)
    n_gt 21-50             → concat_l2 (all terciles positive in h1247)
    n_gt >= 51, low entropy  → concat_l2 (h1248 +0.84 hits, p=0.0055)
    n_gt >= 51, mid entropy  → node2vec (h1248 -0.62 hits, p=0.18; avoid fuse)
    n_gt >= 51, high entropy → concat_l2 (indifferent / slight positive)

Per-seed protocol (leak-bounded):
  - ATC L3 entropy is a per-disease metadata feature derived from
    expanded_ground_truth.json drug ATC distribution. It is observable for a
    holdout disease only if its GT is part of the training-time metadata —
    which is the same assumption h1218/h1247/h1248 used. Document this as
    an *oracle-routing upper bound*; learning a router from disease features
    is the natural h1260 follow-up.
  - Entropy *tercile cuts* are recomputed per seed on n_gt≥51 training-side
    diseases only (no holdout entropies bleed into the cuts). Holdout
    diseases are classified into low/mid/high using those train-side cuts.

Outputs:
    data/analysis/h1249_entropy_routed_benchmark.json
    data/analysis/h1249_entropy_routed_benchmark.md
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Set, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

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

OUT_JSON = PROJECT_ROOT / "data/analysis/h1249_entropy_routed_benchmark.json"
OUT_MD = PROJECT_ROOT / "data/analysis/h1249_entropy_routed_benchmark.md"

# Routing thresholds (from h1247+h1248)
N_GT_HIGH_DENSITY = 51


def shannon_entropy_bits(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counter.values() if c > 0)


def disease_atc_l3_entropy(drugs: List[str], db_lookup: Dict, mapper: ATCMapper) -> Tuple[float, int]:
    """Return (L3 entropy in bits, n_drugs_with_ATC). Mirrors h1247 algorithm."""
    counter: Counter = Counter()
    n_with_atc = 0
    for d in drugs:
        if "Compound::DB" not in d:
            continue
        db_id = d.split("Compound::")[1]
        nm = db_lookup.get(db_id)
        if not nm:
            continue
        codes = mapper.get_atc_codes(nm.lower().strip())
        if not codes:
            continue
        n_with_atc += 1
        primary = codes[0]
        if len(primary) >= 3:
            counter[primary[:3]] += 1
    return shannon_entropy_bits(counter), n_with_atc


def paired_t(deltas: List[float]) -> Dict:
    """Two-sided paired t-test for a list of paired deltas."""
    n = len(deltas)
    if n < 2:
        return {"n": n, "mean": (deltas[0] if deltas else float("nan")),
                "t": float("nan"), "df": 0, "p_two_sided": float("nan")}
    m = mean(deltas)
    s = stdev(deltas)
    if s == 0:
        return {"n": n, "mean": m, "t": float("inf"), "df": n - 1, "p_two_sided": 0.0}
    t = m / (s / math.sqrt(n))
    df = n - 1
    x = df / (df + t * t)
    p = _incomplete_beta(df / 2, 0.5, x)
    return {"n": n, "mean": m, "std": s, "t": t, "df": df, "p_two_sided": p}


def _incomplete_beta(a, b, x):
    if x < 0 or x > 1:
        raise ValueError("x out of range")
    if x == 0:
        return 0.0
    if x == 1:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1 - x) / b


def _betacf(a, b, x, max_iter=200, eps=1e-10):
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m_ in range(1, max_iter + 1):
        m2 = 2 * m_
        aa = m_ * (b - m_) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m_) * (qab + m_) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def score_one_disease(
    disease_emb: np.ndarray,
    train_emb: np.ndarray,
    train_ids_ordered: List[str],
    train_gt: Dict[str, Set[str]],
    k: int,
) -> Dict[str, float]:
    sims = cosine_similarity(disease_emb[None, :], train_emb)[0]
    if k >= len(sims):
        top_idx = np.argsort(sims)
    else:
        top_idx = np.argpartition(sims, -k)[-k:]
    drug_scores: Dict[str, float] = defaultdict(float)
    for idx in top_idx:
        neighbour = train_ids_ordered[idx]
        for drug_id in train_gt.get(neighbour, ()):
            drug_scores[drug_id] += float(sims[idx])
    return dict(drug_scores)


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
) -> Dict:
    """Evaluate three modes in parallel: node2vec, concat_l2, entropy_routed.

    For every holdout disease, compute kNN scores under BOTH node2vec and
    concat_l2, then aggregate metrics per mode. The routed mode picks one of
    the two per-disease score sets according to the routing rule.

    Returns: dict with per-mode aggregate metrics + per-disease deltas.
    """
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

    modes = ("node2vec", "concat_l2", "entropy_routed")
    per_drug_r30: Dict[str, List[float]] = {m: [] for m in modes}
    per_drug_hits: Dict[str, Dict[int, List[float]]] = {
        m: {K: [] for K in (1, 5, 10, 30, 100)} for m in modes
    }
    triple_hits: Dict[str, Dict[int, int]] = {m: {K: 0 for K in (1, 5, 10, 30, 100)} for m in modes}
    triple_rr: Dict[str, List[float]] = {m: [] for m in modes}
    triple_n: Dict[str, int] = {m: 0 for m in modes}
    score_buf: Dict[str, List[np.ndarray]] = {m: [] for m in modes}
    label_buf: Dict[str, List[np.ndarray]] = {m: [] for m in modes}
    per_cat_r30: Dict[str, Dict[str, List[float]]] = {m: defaultdict(list) for m in modes}

    # Per-disease records (for paired-t at disease level)
    per_disease_rows: List[Dict] = []
    routing_counter: Counter = Counter()
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

        # Routing decision based on rule
        if n_gt_disease >= N_GT_HIGH_DENSITY and (train_low_cut <= ent < train_high_cut):
            routed_mode = "node2vec"
            tercile = "mid"
        elif n_gt_disease >= N_GT_HIGH_DENSITY:
            routed_mode = "concat_l2"
            tercile = "low" if ent < train_low_cut else "high"
        else:
            routed_mode = "concat_l2"
            tercile = "n/a"
        routing_counter[(routed_mode, n_gt_disease >= N_GT_HIGH_DENSITY, tercile)] += 1

        # Score under each mode
        scores_n2v = score_one_disease(
            lu_node2vec[disease_id], train_emb_n2v, train_ids_ordered, train_knn_gt, k
        )
        scores_concat = score_one_disease(
            lu_concat[disease_id], train_emb_concat, train_ids_ordered, train_knn_gt, k
        )
        scores_routed = scores_n2v if routed_mode == "node2vec" else scores_concat

        per_disease_metrics: Dict[str, Dict] = {}

        for mode, drug_scores in (
            ("node2vec", scores_n2v),
            ("concat_l2", scores_concat),
            ("entropy_routed", scores_routed),
        ):
            score_vec = np.zeros(len(cand_list), dtype=np.float32)
            for drug, sc in drug_scores.items():
                if drug in cand_index:
                    score_vec[cand_index[drug]] = sc
            order = np.argsort(-score_vec, kind="stable")
            rank_of_drug: Dict[str, int] = {}
            for r, idx in enumerate(order, start=1):
                rank_of_drug[cand_list[idx]] = r

            gt_ranks = [rank_of_drug[d] for d in gt_drugs if d in rank_of_drug]
            n_gt_eval = len(gt_drugs)
            hit_30 = sum(1 for r in gt_ranks if r <= 30)
            r30 = hit_30 / n_gt_eval
            per_drug_r30[mode].append(r30)
            per_cat_r30[mode][cat].append(r30)
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
            score_buf[mode].append(score_vec)
            label_buf[mode].append(label_vec)

            per_disease_metrics[mode] = {"r30": r30, "hits30": hit_30}

        per_disease_rows.append({
            "disease_id": disease_id,
            "name": disease_names.get(disease_id, disease_id),
            "category": cat,
            "n_gt_eval": len(gt_drugs),
            "n_gt_total_egt": n_gt_disease,
            "atc_l3_entropy": ent,
            "tercile_class": tercile,
            "routed_mode": routed_mode,
            "r30_node2vec": per_disease_metrics["node2vec"]["r30"],
            "r30_concat_l2": per_disease_metrics["concat_l2"]["r30"],
            "r30_routed": per_disease_metrics["entropy_routed"]["r30"],
            "hits30_node2vec": per_disease_metrics["node2vec"]["hits30"],
            "hits30_concat_l2": per_disease_metrics["concat_l2"]["hits30"],
            "hits30_routed": per_disease_metrics["entropy_routed"]["hits30"],
        })

    # Aggregate metrics
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
            "per_category_r30": {
                cat: float(np.mean(vals)) if vals else 0.0
                for cat, vals in per_cat_r30[m].items()
            },
        }

    return {
        "modes": out_metrics,
        "per_disease_rows": per_disease_rows,
        "routing_counter": {f"{r}|hd={hd}|t={t}": c for (r, hd, t), c in routing_counter.items()},
        "train_low_cut": train_low_cut,
        "train_high_cut": train_high_cut,
        "n_test_diseases": n_test_diseases,
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
    print(f"h1249: Entropy-routed fusion benchmark — {args.prefix_a} + {args.prefix_b}")
    print(f"       k={args.k} seeds={seeds}")
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

    # Universe: diseases in knn_gt present in BOTH embeddings
    all_diseases = [d for d in knn_gt if d in lu_concat]
    print(f"Disease universe (knn_gt ∩ A ∩ B): {len(all_diseases):,}")

    # Pre-compute per-disease n_gt and ATC L3 entropy from expanded_gt
    print("Pre-computing per-disease n_gt + ATC L3 entropy (using expanded_gt)...")
    n_gt_per_disease: Dict[str, int] = {}
    entropy_per_disease: Dict[str, float] = {}
    coverage_per_disease: Dict[str, int] = {}
    for d in all_diseases:
        gt_drugs = list(eval_gt.get(d, set()))
        n_gt_per_disease[d] = len(gt_drugs)
        ent, n_with_atc = disease_atc_l3_entropy(gt_drugs, db_lookup, mapper)
        entropy_per_disease[d] = ent
        coverage_per_disease[d] = n_with_atc
    n_high_density = sum(1 for d in all_diseases if n_gt_per_disease[d] >= N_GT_HIGH_DENSITY)
    print(f"  n_gt>=51 globally: {n_high_density} of {len(all_diseases)}")
    print(f"  Setup complete in {time.time()-t0:.1f}s")

    per_seed_results: List[Dict] = []
    for si, seed in enumerate(seeds):
        print(f"\n=== SEED {seed} ({si+1}/{len(seeds)}) ===")
        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        print(f"Train: {len(train_ids)}  Holdout: {len(holdout_ids)}")

        # Train-side entropy quantile cuts on n_gt>=51 diseases
        train_high_density_entropies = sorted(
            entropy_per_disease[d] for d in train_ids if n_gt_per_disease[d] >= N_GT_HIGH_DENSITY
        )
        n_thd = len(train_high_density_entropies)
        if n_thd >= 3:
            cut_low = train_high_density_entropies[n_thd // 3]
            cut_high = train_high_density_entropies[2 * n_thd // 3]
        else:
            # Fallback: use globally observed h1247 cuts
            cut_low = 3.18
            cut_high = 3.95
        print(f"  Train-side n_gt>=51 entropy distribution: n={n_thd}, "
              f"tercile cuts low={cut_low:.3f}, high={cut_high:.3f}")

        candidate_drugs: Set[str] = set()
        for d in train_ids:
            if d in knn_gt:
                candidate_drugs |= (knn_gt[d] & set(lu_concat.keys()))
        print(f"  Candidate drug pool: {len(candidate_drugs):,}")

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
        )
        seed_out["seed"] = seed
        seed_out["elapsed_s"] = round(time.time() - tm0, 1)
        per_seed_results.append(seed_out)

        for m in ("node2vec", "concat_l2", "entropy_routed"):
            mm = seed_out["modes"][m]
            print(
                f"  [{m:14s}] R@30={mm['per_drug_r30']*100:.2f}%  "
                f"H30drug={mm['hits_at_k_drug'][30]*100:.2f}%  "
                f"MRR={mm['mrr_triple']:.4f}  "
                f"AUPRC={mm['auprc']:.4f}  AUROC={mm['auroc']:.4f}"
            )
        print(f"  routing: {seed_out['routing_counter']}")
        print(f"  ({seed_out['elapsed_s']}s)")

    # Aggregate
    print("\n" + "=" * 72)
    print("AGGREGATE (mean ± std across seeds)")
    print("=" * 72)
    agg: Dict[str, Dict] = {}
    for m in ("node2vec", "concat_l2", "entropy_routed"):
        rows = [r["modes"][m] for r in per_seed_results]
        agg[m] = {
            "r30": mean_std([r["per_drug_r30"] for r in rows]),
            "hits30_drug": mean_std([r["hits_at_k_drug"][30] for r in rows]),
            "hits30_triple": mean_std([r["hits_at_k_triple"][30] for r in rows]),
            "mrr": mean_std([r["mrr_triple"] for r in rows]),
            "auprc": mean_std([r["auprc"] for r in rows]),
            "auroc": mean_std([r["auroc"] for r in rows]),
        }
        print(
            f"  {m:14s}  R@30={agg[m]['r30'][0]*100:5.2f}%±{agg[m]['r30'][1]*100:.2f}%  "
            f"H30drug={agg[m]['hits30_drug'][0]*100:5.2f}%±{agg[m]['hits30_drug'][1]*100:.2f}%  "
            f"MRR={agg[m]['mrr'][0]:.4f}  AUPRC={agg[m]['auprc'][0]:.4f}  "
            f"AUROC={agg[m]['auroc'][0]:.4f}"
        )

    # Per-seed paired-t (n=5 seeds, df=4) on routed vs concat_l2 anchor
    print("\n" + "=" * 72)
    print("PAIRED-T (per-seed, n=5): entropy_routed vs concat_l2 anchor")
    print("=" * 72)
    paired_seed_t: Dict[str, Dict] = {}
    for label, key in [
        ("R@30", "per_drug_r30"),
        ("hits30_drug", lambda r: r["hits_at_k_drug"][30]),
        ("MRR", "mrr_triple"),
        ("AUPRC", "auprc"),
        ("AUROC", "auroc"),
    ]:
        getter = key if callable(key) else (lambda r, k=key: r[k])
        rt = [getter(r["modes"]["entropy_routed"]) for r in per_seed_results]
        ca = [getter(r["modes"]["concat_l2"]) for r in per_seed_results]
        deltas = [a - b for a, b in zip(rt, ca)]
        t_res = paired_t(deltas)
        paired_seed_t[label] = {
            "deltas": deltas,
            "mean": t_res["mean"],
            "p_two_sided": t_res["p_two_sided"],
            "t": t_res["t"],
        }
        scale = 100.0 if label in ("R@30", "hits30_drug") else 1.0
        print(
            f"  {label:14s} Δ_mean={t_res['mean']*scale:+.4f}  "
            f"t={t_res['t']:+.3f}  p={t_res['p_two_sided']:.3g}"
        )

    # Per-disease paired-t (n=N_diseases × 5_seeds rows) on routed vs concat_l2
    print("\n" + "=" * 72)
    print("PAIRED-T (per-disease rows): entropy_routed vs concat_l2 anchor")
    print("=" * 72)
    per_disease_paired_t: Dict[str, Dict] = {}
    all_rows: List[Dict] = []
    for r in per_seed_results:
        for row in r["per_disease_rows"]:
            row_with_seed = dict(row)
            row_with_seed["seed"] = r["seed"]
            all_rows.append(row_with_seed)

    for label, key_routed, key_concat in [
        ("R@30", "r30_routed", "r30_concat_l2"),
        ("hits30", "hits30_routed", "hits30_concat_l2"),
    ]:
        deltas = [row[key_routed] - row[key_concat] for row in all_rows]
        t_res = paired_t(deltas)
        per_disease_paired_t[label] = {
            "n": t_res["n"],
            "mean": t_res["mean"],
            "p_two_sided": t_res["p_two_sided"],
            "t": t_res["t"],
        }
        scale = 100.0 if label == "R@30" else 1.0
        print(
            f"  {label:14s} n={t_res['n']}  Δ_mean={t_res['mean']*scale:+.4f}  "
            f"t={t_res['t']:+.3f}  p={t_res['p_two_sided']:.3g}"
        )

    # Restricted: only diseases where routing actually flipped to node2vec
    flipped_rows = [r for r in all_rows if r["routed_mode"] == "node2vec"]
    print(f"\n  Diseases routed to node2vec (the only flipped subset): {len(flipped_rows)}")
    if flipped_rows:
        for label, key_routed, key_concat in [
            ("R@30", "r30_routed", "r30_concat_l2"),
            ("hits30", "hits30_routed", "hits30_concat_l2"),
        ]:
            deltas = [row[key_routed] - row[key_concat] for row in flipped_rows]
            t_res = paired_t(deltas)
            scale = 100.0 if label == "R@30" else 1.0
            print(
                f"  flipped {label:8s}  n={t_res['n']}  Δ_mean={t_res['mean']*scale:+.4f}  "
                f"t={t_res['t']:+.3f}  p={t_res['p_two_sided']:.3g}"
            )

    # Save report
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "hypothesis": "h1249",
        "prefix_a": args.prefix_a,
        "prefix_b": args.prefix_b,
        "k": args.k,
        "seeds": seeds,
        "n_diseases_eligible": len(all_diseases),
        "n_high_density_globally": n_high_density,
        "aggregate": {m: {k: list(v) if isinstance(v, tuple) else v for k, v in d.items()} for m, d in agg.items()},
        "per_seed_paired_t": paired_seed_t,
        "per_disease_paired_t": per_disease_paired_t,
        "per_seed_summaries": [
            {
                "seed": r["seed"],
                "train_low_cut": r["train_low_cut"],
                "train_high_cut": r["train_high_cut"],
                "n_test_diseases": r["n_test_diseases"],
                "modes": {m: {k: v for k, v in r["modes"][m].items() if k != "per_category_r30"} for m in r["modes"]},
                "routing_counter": r["routing_counter"],
            }
            for r in per_seed_results
        ],
        "all_disease_rows": all_rows,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {OUT_JSON.relative_to(PROJECT_ROOT)}")

    # Markdown
    md = []
    md.append("# h1249 — Entropy-routed fusion benchmark (production routing rule)\n\n")
    md.append("**Routing rule (from h1247 + h1248):**\n\n")
    md.append("| Stratum | Mode |\n|---|---|\n")
    md.append("| n_gt < 21 | concat_l2 (default fuse) |\n")
    md.append("| n_gt 21-50 | concat_l2 (all terciles) |\n")
    md.append("| n_gt ≥ 51, low entropy | concat_l2 (h1248: +0.84 hits, p=0.0055) |\n")
    md.append("| n_gt ≥ 51, mid entropy | **node2vec** (h1248: -0.62 hits, avoid fuse) |\n")
    md.append("| n_gt ≥ 51, high entropy | concat_l2 (indifferent / +0.41pp) |\n\n")
    md.append("**Caveat:** entropy and n_gt are computed from `expanded_ground_truth.json`. ")
    md.append("For a held-out disease this is technically oracle metadata; production deployment ")
    md.append("would require a learned router on disease features (h1260 follow-up). Tercile cuts ")
    md.append("are leak-bounded — recomputed per-seed on training-side n_gt≥51 diseases only.\n\n")

    md.append("## Aggregate (mean ± std across seeds)\n\n")
    md.append("| Mode | R@30 | hits@30 (drug) | hits@30 (triple) | MRR | AUPRC | AUROC |\n")
    md.append("|---|---|---|---|---|---|---|\n")
    for m in ("node2vec", "concat_l2", "entropy_routed"):
        a = agg[m]
        md.append(
            f"| `{m}` | "
            f"{a['r30'][0]*100:.2f}%±{a['r30'][1]*100:.2f}% | "
            f"{a['hits30_drug'][0]*100:.2f}%±{a['hits30_drug'][1]*100:.2f}% | "
            f"{a['hits30_triple'][0]*100:.2f}%±{a['hits30_triple'][1]*100:.2f}% | "
            f"{a['mrr'][0]:.4f}±{a['mrr'][1]:.4f} | "
            f"{a['auprc'][0]:.4f}±{a['auprc'][1]:.4f} | "
            f"{a['auroc'][0]:.4f}±{a['auroc'][1]:.4f} |\n"
        )

    md.append("\n## Per-seed paired-t — entropy_routed vs concat_l2 (n=5)\n\n")
    md.append("| Metric | Δ (routed − concat_l2) | t | p (two-sided) |\n")
    md.append("|---|---|---|---|\n")
    for label, rec in paired_seed_t.items():
        scale = 100.0 if label in ("R@30", "hits30_drug") else 1.0
        md.append(f"| {label} | {rec['mean']*scale:+.4f} | {rec['t']:+.3f} | {rec['p_two_sided']:.3g} |\n")

    md.append("\n## Per-disease paired-t — entropy_routed vs concat_l2 (all rows)\n\n")
    md.append("| Metric | n | Δ_mean | t | p (two-sided) |\n")
    md.append("|---|---:|---|---|---|\n")
    for label, rec in per_disease_paired_t.items():
        scale = 100.0 if label == "R@30" else 1.0
        md.append(f"| {label} | {rec['n']} | {rec['mean']*scale:+.4f} | {rec['t']:+.3f} | {rec['p_two_sided']:.3g} |\n")

    if flipped_rows:
        md.append(f"\n## Restricted to flipped diseases (routed→node2vec, n={len(flipped_rows)})\n\n")
        md.append("Only the (n_gt≥51 + mid-entropy) subset is routed away from concat_l2; ")
        md.append("everything else is identical to the concat_l2 baseline. The restricted ")
        md.append("paired-t isolates the rule's contribution.\n\n")
        md.append("| Metric | n | Δ_mean (routed − concat_l2) | t | p |\n")
        md.append("|---|---:|---|---|---|\n")
        for label, key_routed, key_concat in [
            ("R@30", "r30_routed", "r30_concat_l2"),
            ("hits30", "hits30_routed", "hits30_concat_l2"),
        ]:
            deltas = [row[key_routed] - row[key_concat] for row in flipped_rows]
            t_res = paired_t(deltas)
            scale = 100.0 if label == "R@30" else 1.0
            md.append(
                f"| {label} | {t_res['n']} | {t_res['mean']*scale:+.4f} | "
                f"{t_res['t']:+.3f} | {t_res['p_two_sided']:.3g} |\n"
            )

    md.append("\n## Per-seed details\n\n")
    md.append("| Seed | n_test | low_cut | high_cut | concat_l2 R@30 | routed R@30 | Δ |\n")
    md.append("|---|---|---|---|---|---|---|\n")
    for r in per_seed_results:
        c_r30 = r["modes"]["concat_l2"]["per_drug_r30"]
        rt_r30 = r["modes"]["entropy_routed"]["per_drug_r30"]
        md.append(
            f"| {r['seed']} | {r['n_test_diseases']} | {r['train_low_cut']:.3f} | "
            f"{r['train_high_cut']:.3f} | {c_r30*100:.2f}% | {rt_r30*100:.2f}% | "
            f"{(rt_r30 - c_r30)*100:+.3f}pp |\n"
        )

    md.append("\n## Per-seed routing histogram\n\n")
    for r in per_seed_results:
        md.append(f"- seed {r['seed']}: {r['routing_counter']}\n")

    with open(OUT_MD, "w") as f:
        f.write("".join(md))
    print(f"Wrote {OUT_MD.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
