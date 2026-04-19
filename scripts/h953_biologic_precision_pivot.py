#!/usr/bin/env python3
"""h953: Biologic precision pivot — exploit bio_p30 = 6% with biologic-aware
demotion based on a per-disease (per-category) biologic prior.

Hypothesis:
    For diseases whose category historically has very few biologic GT drugs
    (e.g. infectious, ophthalmic, psychiatric), any biologic that lands in
    top-30 is almost certainly a precision drag. Demote those biologics to
    FILTER and refill top-30 from the next non-biologic predictions. This
    should improve overall precision@30 without hurting bio_r30 much (because
    biologic GT is already empty in those categories).

Method:
    For each of 5 holdout seeds (h393 split):
      1. Recompute GT structures from train-only diseases.
      2. Compute biologic_prior[category] = fraction of train diseases in
         that category that have ≥1 biologic in their expanded GT.
      3. For each holdout disease, run predictor.predict(disease_id,
         top_n=200, include_filtered=True). Record baseline top-30 metrics.
      4. For each demotion threshold τ ∈ {0.05, 0.10, 0.15, 0.20, 0.30}:
            if biologic_prior[cat(disease)] < τ:
                drop biologics from preds; take top-30 of the remainder.
            else:
                top-30 unchanged.
      5. Aggregate per-disease metrics into per-seed means.

Decision:
    Ship if any threshold yields overall_p30 lift ≥ +2pp AND bio_r30 drop
    ≤ 5pp (per the h953 hypothesis success criterion). Otherwise mark
    INVALIDATED but record per-category lift for follow-up.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

import numpy as np

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
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.30]
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


def compute_category_biologic_prior(
    predictor: DrugRepurposingPredictor,
    expanded_gt: Dict[str, Set[str]],
    biologic_pool: Set[str],
    train_disease_ids: Set[str],
) -> Dict[str, float]:
    """Per-category fraction of train diseases that have ≥1 biologic GT."""
    cat_with_bio: Dict[str, int] = defaultdict(int)
    cat_total: Dict[str, int] = defaultdict(int)

    for dis_id in train_disease_ids:
        gt = expanded_gt.get(dis_id, set())
        if not gt:
            continue
        name = predictor.disease_names.get(dis_id, dis_id)
        cat = predictor.categorize_disease(name)
        cat_total[cat] += 1
        if gt & biologic_pool:
            cat_with_bio[cat] += 1

    return {
        c: cat_with_bio[c] / cat_total[c]
        for c in cat_total
    }


def evaluate_seed(
    predictor: DrugRepurposingPredictor,
    expanded_gt: Dict[str, Set[str]],
    biologic_pool: Set[str],
    holdout_ids: List[str],
    bio_prior: Dict[str, float],
) -> Dict:
    """Per-disease eval at baseline + every threshold. Returns raw records."""
    records: List[Dict] = []
    n_skipped_no_gt = 0
    n_skipped_predict_err = 0

    for dis_id in holdout_ids:
        gt = expanded_gt.get(dis_id, set())
        if not gt:
            n_skipped_no_gt += 1
            continue

        try:
            result = predictor.predict(
                dis_id, top_n=TOP_N_FETCH, include_filtered=True
            )
        except Exception:
            n_skipped_predict_err += 1
            continue

        # Order preserved by predict() — already ranked.
        all_drug_ids = [p.drug_id for p in result.predictions]
        if not all_drug_ids:
            continue

        name = predictor.disease_names.get(dis_id, dis_id)
        cat = predictor.categorize_disease(name)
        cat_prior = bio_prior.get(cat, 0.0)

        bio_gt = gt & biologic_pool

        # Baseline top-30
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
        baseline_bio_p30_amongbio = (
            len(baseline_bio_hits) / len(baseline_bio_in_top)
            if baseline_bio_in_top else None
        )

        # Pre-compute the no-biologic ranking once
        non_bio_ranking = [d for d in all_drug_ids if d not in biologic_pool]

        # Apply each threshold
        per_threshold: Dict[float, Dict] = {}
        for thr in THRESHOLDS:
            demote = cat_prior < thr
            if demote:
                top = non_bio_ranking[:TOP_N_REPORT]
            else:
                top = baseline_top
            top_set = set(top)
            hits = top_set & gt
            bio_in_top = top_set & biologic_pool
            bio_hits = top_set & bio_gt
            per_threshold[thr] = {
                "demoted": demote,
                "top_size": len(top),
                "p30": len(hits) / TOP_N_REPORT,
                "r30": len(hits) / len(gt),
                "bio_r30": (
                    len(bio_hits) / len(bio_gt) if bio_gt else None
                ),
                "bio_p30_amongbio": (
                    len(bio_hits) / len(bio_in_top)
                    if bio_in_top else None
                ),
                "n_bio_in_top": len(bio_in_top),
            }

        records.append({
            "disease_id": dis_id,
            "category": cat,
            "cat_prior": cat_prior,
            "gt_size": len(gt),
            "bio_gt_size": len(bio_gt),
            "baseline_n_bio_in_top30": len(baseline_bio_in_top),
            "baseline_p30": baseline_p30,
            "baseline_r30": baseline_r30,
            "baseline_bio_r30": baseline_bio_r30,
            "baseline_bio_p30_amongbio": baseline_bio_p30_amongbio,
            "per_threshold": per_threshold,
        })

    return {
        "records": records,
        "n_skipped_no_gt": n_skipped_no_gt,
        "n_skipped_predict_err": n_skipped_predict_err,
    }


def aggregate(records: List[Dict]) -> Dict:
    """Per-seed aggregation: baseline + per-threshold means."""
    agg = {
        "n": len(records),
        "n_with_bio_gt": sum(1 for r in records if r["bio_gt_size"] > 0),
        "baseline": {
            "p30": float(np.mean([r["baseline_p30"] for r in records])),
            "r30": float(np.mean([r["baseline_r30"] for r in records])),
            "bio_r30": float(np.mean(
                [r["baseline_bio_r30"] for r in records
                 if r["baseline_bio_r30"] is not None]
            )) if any(r["baseline_bio_r30"] is not None for r in records) else 0.0,
            "n_bio_in_top30": float(np.mean(
                [r["baseline_n_bio_in_top30"] for r in records]
            )),
        },
        "per_threshold": {},
    }
    for thr in THRESHOLDS:
        ps = [r["per_threshold"][thr]["p30"] for r in records]
        rs = [r["per_threshold"][thr]["r30"] for r in records]
        brs = [r["per_threshold"][thr]["bio_r30"] for r in records
               if r["per_threshold"][thr]["bio_r30"] is not None]
        n_demoted = sum(1 for r in records if r["per_threshold"][thr]["demoted"])
        agg["per_threshold"][thr] = {
            "p30": float(np.mean(ps)),
            "r30": float(np.mean(rs)),
            "bio_r30": float(np.mean(brs)) if brs else 0.0,
            "n_demoted_diseases": n_demoted,
        }
    return agg


def per_category_lift(records: List[Dict], thr: float) -> Dict[str, Dict]:
    """Per-category baseline vs filtered metrics for one threshold."""
    by_cat: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r)
    out: Dict[str, Dict] = {}
    for cat, recs in by_cat.items():
        ps_b = [r["baseline_p30"] for r in recs]
        ps_f = [r["per_threshold"][thr]["p30"] for r in recs]
        bs_b = [r["baseline_bio_r30"] for r in recs
                if r["baseline_bio_r30"] is not None]
        bs_f = [r["per_threshold"][thr]["bio_r30"] for r in recs
                if r["per_threshold"][thr]["bio_r30"] is not None]
        n_demoted = sum(1 for r in recs if r["per_threshold"][thr]["demoted"])
        out[cat] = {
            "n": len(recs),
            "n_with_bio_gt": len(bs_b),
            "n_demoted": n_demoted,
            "p30_baseline": float(np.mean(ps_b)),
            "p30_filtered": float(np.mean(ps_f)),
            "p30_delta_pp": 100 * (float(np.mean(ps_f)) - float(np.mean(ps_b))),
            "bio_r30_baseline": float(np.mean(bs_b)) if bs_b else None,
            "bio_r30_filtered": float(np.mean(bs_f)) if bs_f else None,
            "bio_r30_delta_pp": (
                100 * (float(np.mean(bs_f)) - float(np.mean(bs_b)))
                if bs_b and bs_f else None
            ),
            "cat_prior": recs[0]["cat_prior"],
        }
    return out


def main():
    print("=" * 78)
    print("h953: Biologic precision pivot — per-category prior demotion")
    print("=" * 78)

    print("Loading predictor ...")
    predictor = DrugRepurposingPredictor()

    expanded_gt = load_expanded_gt(
        predictor.reference_dir / "expanded_ground_truth.json"
    )
    print(f"Expanded GT: {len(expanded_gt)} diseases, "
          f"{sum(len(v) for v in expanded_gt.values())} pairs")

    biologic_pool = {
        d for d in predictor.drug_targets
        if is_biologic(predictor.drug_id_to_name.get(d))
    }
    print(f"Biologic pool: {len(biologic_pool)} of "
          f"{len(predictor.drug_targets)} drugs with targets")

    all_diseases = [
        d for d in predictor.ground_truth if d in predictor.embeddings
    ]
    print(f"Evaluable diseases (GT ∩ embeddings): {len(all_diseases)}")
    print(f"Thresholds: {THRESHOLDS}")
    print(f"top_n_fetch={TOP_N_FETCH}, top_n_report={TOP_N_REPORT}\n")

    seed_aggregates: List[Dict] = []
    seed_records_for_cat: List[List[Dict]] = []
    seed_priors: List[Dict[str, float]] = []

    for seed in SEEDS:
        print(f"--- Seed {seed} ---")
        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        train_set = set(train_ids)
        print(f"Train: {len(train_ids)}, Holdout: {len(holdout_ids)}")

        originals = recompute_gt_structures(predictor, train_set)
        try:
            bio_prior = compute_category_biologic_prior(
                predictor, expanded_gt, biologic_pool, train_set
            )
            ev = evaluate_seed(
                predictor, expanded_gt, biologic_pool, holdout_ids, bio_prior
            )
        finally:
            restore_gt_structures(predictor, originals)

        records = ev["records"]
        agg = aggregate(records)
        seed_aggregates.append(agg)
        seed_records_for_cat.append(records)
        seed_priors.append(bio_prior)

        print(f"  evaluated={agg['n']}  "
              f"with_bio_gt={agg['n_with_bio_gt']}  "
              f"skip_no_gt={ev['n_skipped_no_gt']}  "
              f"skip_err={ev['n_skipped_predict_err']}")
        print(f"  baseline   p30={100*agg['baseline']['p30']:5.2f}%  "
              f"r30={100*agg['baseline']['r30']:5.2f}%  "
              f"bio_r30={100*agg['baseline']['bio_r30']:5.2f}%  "
              f"avg_bio_in_top30={agg['baseline']['n_bio_in_top30']:.2f}")
        for thr in THRESHOLDS:
            t = agg['per_threshold'][thr]
            dp = 100 * (t['p30'] - agg['baseline']['p30'])
            db = 100 * (t['bio_r30'] - agg['baseline']['bio_r30'])
            print(f"  thr={thr:.2f}   p30={100*t['p30']:5.2f}% (Δ{dp:+.2f}pp)  "
                  f"r30={100*t['r30']:5.2f}%  "
                  f"bio_r30={100*t['bio_r30']:5.2f}% (Δ{db:+.2f}pp)  "
                  f"demoted={t['n_demoted_diseases']}/{agg['n']}")
        print()

    # Cross-seed aggregate
    print("=" * 78)
    print("CROSS-SEED MEAN ± STD (5 seeds)")
    print("=" * 78)
    base_p = [a['baseline']['p30'] for a in seed_aggregates]
    base_r = [a['baseline']['r30'] for a in seed_aggregates]
    base_br = [a['baseline']['bio_r30'] for a in seed_aggregates]
    print(f"  baseline   p30={100*np.mean(base_p):5.2f}% ± {100*np.std(base_p):4.2f}%  "
          f"r30={100*np.mean(base_r):5.2f}% ± {100*np.std(base_r):4.2f}%  "
          f"bio_r30={100*np.mean(base_br):5.2f}% ± {100*np.std(base_br):4.2f}%")

    summary: Dict = {
        "hypothesis": "h953",
        "title": "Biologic precision pivot — per-category prior demotion",
        "seeds": SEEDS,
        "thresholds": THRESHOLDS,
        "biologic_pool_size": len(biologic_pool),
        "evaluable_diseases": len(all_diseases),
        "baseline_5seed": {
            "p30_mean": float(np.mean(base_p)),
            "p30_std": float(np.std(base_p)),
            "r30_mean": float(np.mean(base_r)),
            "r30_std": float(np.std(base_r)),
            "bio_r30_mean": float(np.mean(base_br)),
            "bio_r30_std": float(np.std(base_br)),
        },
        "per_threshold_5seed": {},
        "ship_decision": None,
    }

    best_thr = None
    best_score = -1e9
    for thr in THRESHOLDS:
        ps = [a['per_threshold'][thr]['p30'] for a in seed_aggregates]
        rs = [a['per_threshold'][thr]['r30'] for a in seed_aggregates]
        brs = [a['per_threshold'][thr]['bio_r30'] for a in seed_aggregates]
        nd = [a['per_threshold'][thr]['n_demoted_diseases'] for a in seed_aggregates]
        dp = 100 * (np.mean(ps) - np.mean(base_p))
        dr = 100 * (np.mean(rs) - np.mean(base_r))
        db = 100 * (np.mean(brs) - np.mean(base_br))
        print(f"  thr={thr:.2f}    p30={100*np.mean(ps):5.2f}% ± {100*np.std(ps):4.2f}% (Δ{dp:+.2f}pp)  "
              f"r30={100*np.mean(rs):5.2f}% (Δ{dr:+.2f}pp)  "
              f"bio_r30={100*np.mean(brs):5.2f}% ± {100*np.std(brs):4.2f}% (Δ{db:+.2f}pp)  "
              f"avg_demoted={np.mean(nd):.1f}")
        summary["per_threshold_5seed"][str(thr)] = {
            "p30_mean": float(np.mean(ps)),
            "p30_std": float(np.std(ps)),
            "r30_mean": float(np.mean(rs)),
            "r30_std": float(np.std(rs)),
            "bio_r30_mean": float(np.mean(brs)),
            "bio_r30_std": float(np.std(brs)),
            "avg_demoted_diseases": float(np.mean(nd)),
            "p30_delta_pp": float(dp),
            "r30_delta_pp": float(dr),
            "bio_r30_delta_pp": float(db),
        }
        # Ship gate from h953 spec: p30 lift >= +2pp AND bio_r30 drop <= 5pp
        if dp >= 2.0 and db >= -5.0:
            score = dp - max(0.0, -db) * 0.5
            if score > best_score:
                best_score = score
                best_thr = thr

    if best_thr is not None:
        ship = "SHIP — meets gate (p30 lift ≥ +2pp, bio_r30 drop ≤ 5pp)"
    else:
        ship = "DO NOT SHIP — no threshold meets gate"
    summary["ship_decision"] = {"verdict": ship, "best_threshold": best_thr}
    print(f"\nDECISION: {ship}")
    if best_thr is not None:
        print(f"  best_threshold = {best_thr}")

    # Per-category lift at the most-aggressive threshold for diagnostics
    print("\n" + "=" * 78)
    print("PER-CATEGORY (mean across seeds, at thr=0.10)")
    print("=" * 78)
    cat_agg: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    cat_n_demoted: Dict[str, list] = defaultdict(list)
    for records in seed_records_for_cat:
        cl = per_category_lift(records, 0.10)
        for cat, vals in cl.items():
            for k, v in vals.items():
                cat_agg[cat][k].append(v)
            cat_n_demoted[cat].append(vals["n_demoted"])

    print(f"{'category':<18s}  {'prior':>6s}  {'p30_b':>7s}  {'p30_f':>7s}  "
          f"{'Δp30':>7s}  {'br_b':>7s}  {'br_f':>7s}  {'Δbr':>7s}  {'demoted':>7s}")
    cat_summary: Dict[str, Dict] = {}
    for cat in sorted(cat_agg.keys()):
        v = cat_agg[cat]
        prior_m = float(np.mean(v["cat_prior"]))
        pb = float(np.mean(v["p30_baseline"]))
        pf = float(np.mean(v["p30_filtered"]))
        bb = float(np.mean([x for x in v["bio_r30_baseline"] if x is not None])) if any(x is not None for x in v["bio_r30_baseline"]) else None
        bf = float(np.mean([x for x in v["bio_r30_filtered"] if x is not None])) if any(x is not None for x in v["bio_r30_filtered"]) else None
        dp = 100 * (pf - pb)
        db = (100 * (bf - bb)) if (bb is not None and bf is not None) else None
        nd = float(np.mean(cat_n_demoted[cat]))
        print(f"  {cat:<16s}  {prior_m:6.3f}  {100*pb:6.2f}%  {100*pf:6.2f}%  "
              f"{dp:+6.2f}pp  "
              f"{(100*bb if bb else 0):6.2f}%  {(100*bf if bf else 0):6.2f}%  "
              f"{(db if db is not None else 0):+6.2f}pp  {nd:7.1f}")
        cat_summary[cat] = {
            "cat_prior_mean": prior_m,
            "p30_baseline_mean": pb,
            "p30_filtered_mean": pf,
            "p30_delta_pp": dp,
            "bio_r30_baseline_mean": bb,
            "bio_r30_filtered_mean": bf,
            "bio_r30_delta_pp": db,
            "n_demoted_avg": nd,
        }
    summary["category_summary_at_thr_0.10"] = cat_summary

    out_path = ROOT / "data/analysis/h953_biologic_precision_pivot.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
