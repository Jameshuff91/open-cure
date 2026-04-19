#!/usr/bin/env python3
"""h990: Per-disease kNN-neighbor biologic prior — finer-grained replacement
for h953's category-level prior.

Why h953 failed:
    Category-level priors are too coarse. Dermatological has prior=0.183 but
    bio_r30=72%; demoting biologics in dermatological destroys real hits. The
    same is likely true for any low-prior category that contains rare-but-real
    biologic-treated diseases (e.g. psoriasis under dermatological).

This hypothesis:
    Use the EXACT mechanism kNN uses to score biologics. For each holdout
    disease D:
        prior(D) = (#{top-20 train neighbors of D with ≥1 biologic in
                       expanded GT}) / 20
    If prior(D) < threshold, demote biologics in top-30 and refill from
    next non-biologic ranks. Sweep thresholds {0.00, 0.05, 0.10, 0.20}.

    Strict (0.0) is a sanity check: if zero top-20 neighbors have any
    biologic in GT, then any biologic in top-30 must have arrived from
    a non-kNN path (rescue rule / override).

Decision:
    Ship if any threshold yields p30 lift ≥ +1pp AND bio_r30 drop ≤ 3pp
    (tighter than h953's gate because per-disease should be more precise).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from production_predictor import DrugRepurposingPredictor  # noqa: E402
from h393_holdout_tier_validation import (  # noqa: E402
    split_diseases,
    recompute_gt_structures,
    restore_gt_structures,
)
from h939_biologic_target_overlap_audit import is_biologic  # noqa: E402


SEEDS = [42, 123, 456, 789, 2024]
THRESHOLDS = [0.00, 0.05, 0.10, 0.20]
KNN_K = 20
TOP_N_FETCH = 200
TOP_N_REPORT = 30


def load_expanded_gt(path: Path) -> Dict[str, Set[str]]:
    with open(path) as f:
        raw = json.load(f)
    out: Dict[str, Set[str]] = {}
    for dis_id, drugs in raw.items():
        s: Set[str] = set()
        for d in drugs:
            if isinstance(d, str):
                s.add(d)
            elif isinstance(d, dict):
                did = d.get("drug_id") or d.get("drug")
                if did:
                    s.add(did)
        out[dis_id] = s
    return out


def disease_neighbor_bio_prior(
    predictor: DrugRepurposingPredictor,
    disease_id: str,
    expanded_gt: Dict[str, Set[str]],
    biologic_pool: Set[str],
    k: int = KNN_K,
) -> float | None:
    """Fraction of top-k train neighbors with ≥1 biologic in expanded GT."""
    if disease_id not in predictor.embeddings:
        return None
    test_emb = predictor.embeddings[disease_id].reshape(1, -1)
    sims = cosine_similarity(test_emb, predictor.train_embeddings)[0]
    top_k_idx = np.argsort(sims)[-k:]
    n_with_bio = 0
    for idx in top_k_idx:
        nb = predictor.train_diseases[idx]
        gt = expanded_gt.get(nb, set())
        if gt & biologic_pool:
            n_with_bio += 1
    return n_with_bio / k


def evaluate_seed(
    predictor: DrugRepurposingPredictor,
    expanded_gt: Dict[str, Set[str]],
    biologic_pool: Set[str],
    holdout_ids: List[str],
) -> Dict:
    records: List[Dict] = []
    n_skipped_no_gt = 0
    n_skipped_predict_err = 0
    n_skipped_no_emb = 0

    for dis_id in holdout_ids:
        gt = expanded_gt.get(dis_id, set())
        if not gt:
            n_skipped_no_gt += 1
            continue
        if dis_id not in predictor.embeddings:
            n_skipped_no_emb += 1
            continue

        prior = disease_neighbor_bio_prior(
            predictor, dis_id, expanded_gt, biologic_pool
        )
        if prior is None:
            n_skipped_no_emb += 1
            continue

        try:
            result = predictor.predict(
                dis_id, top_n=TOP_N_FETCH, include_filtered=True
            )
        except Exception:
            n_skipped_predict_err += 1
            continue

        all_drug_ids = [p.drug_id for p in result.predictions]
        if not all_drug_ids:
            continue

        bio_gt = gt & biologic_pool

        baseline_top = all_drug_ids[:TOP_N_REPORT]
        baseline_set = set(baseline_top)
        baseline_bio_in_top = baseline_set & biologic_pool
        baseline_hits = baseline_set & gt
        baseline_bio_hits = baseline_set & bio_gt

        baseline_p30 = len(baseline_hits) / TOP_N_REPORT
        baseline_r30 = len(baseline_hits) / len(gt)
        baseline_bio_r30 = (
            len(baseline_bio_hits) / len(bio_gt) if bio_gt else None
        )

        non_bio_ranking = [d for d in all_drug_ids if d not in biologic_pool]

        per_threshold: Dict[float, Dict] = {}
        for thr in THRESHOLDS:
            # strict=0.0 means "demote if prior <= 0" — i.e. zero neighbors
            # have any biologic GT. Other thresholds use strict <.
            if thr == 0.0:
                demote = prior <= 0.0
            else:
                demote = prior < thr
            top = non_bio_ranking[:TOP_N_REPORT] if demote else baseline_top
            top_set = set(top)
            hits = top_set & gt
            bio_hits = top_set & bio_gt
            bio_in_top = top_set & biologic_pool
            per_threshold[thr] = {
                "demoted": demote,
                "p30": len(hits) / TOP_N_REPORT,
                "r30": len(hits) / len(gt),
                "bio_r30": (
                    len(bio_hits) / len(bio_gt) if bio_gt else None
                ),
                "n_bio_in_top": len(bio_in_top),
            }

        records.append({
            "disease_id": dis_id,
            "category": predictor.categorize_disease(
                predictor.disease_names.get(dis_id, dis_id)
            ),
            "neighbor_bio_prior": prior,
            "gt_size": len(gt),
            "bio_gt_size": len(bio_gt),
            "baseline_n_bio_in_top30": len(baseline_bio_in_top),
            "baseline_p30": baseline_p30,
            "baseline_r30": baseline_r30,
            "baseline_bio_r30": baseline_bio_r30,
            "per_threshold": per_threshold,
        })

    return {
        "records": records,
        "n_skipped_no_gt": n_skipped_no_gt,
        "n_skipped_no_emb": n_skipped_no_emb,
        "n_skipped_predict_err": n_skipped_predict_err,
    }


def aggregate_seed(records: List[Dict]) -> Dict:
    n = len(records)
    base_p = float(np.mean([r["baseline_p30"] for r in records]))
    base_r = float(np.mean([r["baseline_r30"] for r in records]))
    base_br = (
        float(np.mean([r["baseline_bio_r30"] for r in records
                       if r["baseline_bio_r30"] is not None]))
        if any(r["baseline_bio_r30"] is not None for r in records) else 0.0
    )
    out = {
        "n": n,
        "n_with_bio_gt": sum(1 for r in records if r["bio_gt_size"] > 0),
        "baseline": {
            "p30": base_p, "r30": base_r, "bio_r30": base_br,
        },
        "per_threshold": {},
    }
    for thr in THRESHOLDS:
        ps = [r["per_threshold"][thr]["p30"] for r in records]
        rs = [r["per_threshold"][thr]["r30"] for r in records]
        brs = [r["per_threshold"][thr]["bio_r30"] for r in records
               if r["per_threshold"][thr]["bio_r30"] is not None]
        nd = sum(1 for r in records if r["per_threshold"][thr]["demoted"])
        out["per_threshold"][thr] = {
            "p30": float(np.mean(ps)),
            "r30": float(np.mean(rs)),
            "bio_r30": float(np.mean(brs)) if brs else 0.0,
            "n_demoted": nd,
        }
    return out


def main():
    print("=" * 78)
    print("h990: Per-disease kNN-neighbor biologic prior demotion")
    print("=" * 78)
    print("Loading predictor ...")
    predictor = DrugRepurposingPredictor()

    expanded_gt = load_expanded_gt(
        predictor.reference_dir / "expanded_ground_truth.json"
    )
    biologic_pool = {
        d for d in predictor.drug_targets
        if is_biologic(predictor.drug_id_to_name.get(d))
    }
    all_diseases = [
        d for d in predictor.ground_truth if d in predictor.embeddings
    ]
    print(f"Expanded GT: {len(expanded_gt)} dx, "
          f"{sum(len(v) for v in expanded_gt.values())} pairs")
    print(f"Biologic pool: {len(biologic_pool)}")
    print(f"Evaluable: {len(all_diseases)}")
    print(f"Thresholds: {THRESHOLDS}, k={KNN_K}, "
          f"top_n_fetch={TOP_N_FETCH}\n")

    seed_aggs: List[Dict] = []
    seed_records: List[List[Dict]] = []

    for seed in SEEDS:
        print(f"--- Seed {seed} ---")
        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        train_set = set(train_ids)
        originals = recompute_gt_structures(predictor, train_set)
        try:
            ev = evaluate_seed(
                predictor, expanded_gt, biologic_pool, holdout_ids
            )
        finally:
            restore_gt_structures(predictor, originals)
        records = ev["records"]
        agg = aggregate_seed(records)
        seed_aggs.append(agg)
        seed_records.append(records)
        print(f"  evaluated={agg['n']}  with_bio_gt={agg['n_with_bio_gt']}  "
              f"skip_no_gt={ev['n_skipped_no_gt']}  "
              f"skip_no_emb={ev['n_skipped_no_emb']}  "
              f"skip_err={ev['n_skipped_predict_err']}")
        print(f"  baseline   p30={100*agg['baseline']['p30']:5.2f}%  "
              f"r30={100*agg['baseline']['r30']:5.2f}%  "
              f"bio_r30={100*agg['baseline']['bio_r30']:5.2f}%")
        for thr in THRESHOLDS:
            t = agg['per_threshold'][thr]
            dp = 100 * (t['p30'] - agg['baseline']['p30'])
            db = 100 * (t['bio_r30'] - agg['baseline']['bio_r30'])
            print(f"  thr={thr:.2f}   p30={100*t['p30']:5.2f}% (Δ{dp:+.2f}pp)  "
                  f"r30={100*t['r30']:5.2f}%  "
                  f"bio_r30={100*t['bio_r30']:5.2f}% (Δ{db:+.2f}pp)  "
                  f"demoted={t['n_demoted']}/{agg['n']}")
        # Prior distribution diagnostics
        priors = [r["neighbor_bio_prior"] for r in records]
        print(f"  prior   min={min(priors):.3f}  q25={np.percentile(priors,25):.3f}  "
              f"med={np.median(priors):.3f}  q75={np.percentile(priors,75):.3f}  "
              f"max={max(priors):.3f}  zero_count={sum(1 for p in priors if p<=0)}")
        print()

    # Cross-seed
    print("=" * 78)
    print("CROSS-SEED MEAN ± STD (5 seeds)")
    print("=" * 78)
    base_p = [a['baseline']['p30'] for a in seed_aggs]
    base_r = [a['baseline']['r30'] for a in seed_aggs]
    base_br = [a['baseline']['bio_r30'] for a in seed_aggs]
    print(f"  baseline   p30={100*np.mean(base_p):5.2f}% ± {100*np.std(base_p):4.2f}%  "
          f"r30={100*np.mean(base_r):5.2f}% ± {100*np.std(base_r):4.2f}%  "
          f"bio_r30={100*np.mean(base_br):5.2f}% ± {100*np.std(base_br):4.2f}%")

    summary = {
        "hypothesis": "h990",
        "title": "Per-disease kNN-neighbor biologic prior demotion",
        "seeds": SEEDS, "k": KNN_K, "thresholds": THRESHOLDS,
        "biologic_pool_size": len(biologic_pool),
        "evaluable_diseases": len(all_diseases),
        "baseline_5seed": {
            "p30_mean": float(np.mean(base_p)), "p30_std": float(np.std(base_p)),
            "r30_mean": float(np.mean(base_r)), "r30_std": float(np.std(base_r)),
            "bio_r30_mean": float(np.mean(base_br)), "bio_r30_std": float(np.std(base_br)),
        },
        "per_threshold_5seed": {},
        "ship_decision": None,
    }

    best_thr = None
    best_score = -1e9
    for thr in THRESHOLDS:
        ps = [a['per_threshold'][thr]['p30'] for a in seed_aggs]
        rs = [a['per_threshold'][thr]['r30'] for a in seed_aggs]
        brs = [a['per_threshold'][thr]['bio_r30'] for a in seed_aggs]
        nd = [a['per_threshold'][thr]['n_demoted'] for a in seed_aggs]
        dp = 100 * (np.mean(ps) - np.mean(base_p))
        dr = 100 * (np.mean(rs) - np.mean(base_r))
        db = 100 * (np.mean(brs) - np.mean(base_br))
        print(f"  thr={thr:.2f}    p30={100*np.mean(ps):5.2f}% ± {100*np.std(ps):4.2f}% (Δ{dp:+.2f}pp)  "
              f"r30={100*np.mean(rs):5.2f}% (Δ{dr:+.2f}pp)  "
              f"bio_r30={100*np.mean(brs):5.2f}% ± {100*np.std(brs):4.2f}% (Δ{db:+.2f}pp)  "
              f"avg_demoted={np.mean(nd):.1f}")
        summary["per_threshold_5seed"][str(thr)] = {
            "p30_mean": float(np.mean(ps)), "p30_std": float(np.std(ps)),
            "r30_mean": float(np.mean(rs)), "r30_std": float(np.std(rs)),
            "bio_r30_mean": float(np.mean(brs)), "bio_r30_std": float(np.std(brs)),
            "avg_demoted": float(np.mean(nd)),
            "p30_delta_pp": float(dp), "r30_delta_pp": float(dr),
            "bio_r30_delta_pp": float(db),
        }
        # h990 ship gate: p30 ≥ +1pp AND bio_r30 drop ≤ 3pp
        if dp >= 1.0 and db >= -3.0:
            score = dp - max(0.0, -db) * 0.5
            if score > best_score:
                best_score = score
                best_thr = thr

    if best_thr is not None:
        ship = "SHIP — meets gate (p30 lift ≥ +1pp, bio_r30 drop ≤ 3pp)"
    else:
        ship = "DO NOT SHIP — no threshold meets gate"
    summary["ship_decision"] = {"verdict": ship, "best_threshold": best_thr}
    print(f"\nDECISION: {ship}")
    if best_thr is not None:
        print(f"  best_threshold = {best_thr}")

    out_path = ROOT / "data/analysis/h990_neighbor_bio_prior.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
