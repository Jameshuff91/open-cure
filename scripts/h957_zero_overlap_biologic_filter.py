#!/usr/bin/env python3
"""h957: Zero-overlap biologic safety filter — precision pivot for biologics.

Why:
    h951 measured production bio_p30 = 6.37% — biologics in the user-visible
    top-30 are mostly false positives. h955 closed h906/h920/h921/h924
    (biologic recall pivots) because h951 dissolved the recall gap. h949
    proposed the natural inverse: a NEGATIVE filter that drops biologic
    predictions whose drug targets share zero genes with the disease gene
    set. Biologic efficacy almost always requires target engagement
    (mAbs, cytokines, fusion proteins), so a zero-overlap biologic
    prediction is most likely a kNN co-prescription artifact (procedural,
    adjunct, or DRKG noise).

Filter:
    For a top-K candidate (K=200 from include_filtered=True):
      DROP if  is_biologic(drug)
          AND drug_targets[drug]   non-empty
          AND disease_genes[disease] non-empty
          AND |drug_targets[drug] ∩ disease_genes[disease]| == 0

    Skip filter (keep prediction) when:
      - drug is small molecule
      - drug_targets unknown
      - disease_genes unknown
    These three "skip" conditions enforce the principle: only filter when
    we have evidence to support the demotion. Mechanism support
    (== overlap > 0) is the rescue — by construction this filter never
    fires on a prediction with mechanism_support=True.

Eval:
    5-seed h393 holdout, expanded GT.
      bio_r30   = (top30_filtered ∩ bio_GT) / |bio_GT|
      sm_r30    = (top30_filtered ∩ sm_GT)  / |sm_GT|
      overall   = (top30_filtered ∩ GT)     / |GT|
      bio_p30   = (top30_filtered ∩ bio_GT) / |biologics in top30_filtered|

    Compare baseline (no filter, top-30 by rank) vs filtered (top-30 of
    surviving candidates).

Decision (h957):
    ship if bio_p30 +>=3pp AND bio_r30 drop <=2pp.
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

TOP_K_CANDIDATE_POOL = 200  # include_filtered=True pool depth before filter
TOP_N = 30                   # final user-visible top-N


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


def filter_zero_overlap_biologic(
    predictions: List,
    disease_id: str,
    biologic_pool: Set[str],
    drug_targets: Dict[str, Set[str]],
    disease_genes: Dict[str, Set[str]],
):
    """Return (kept_predictions, dropped_predictions) lists.

    Drops biologic preds with non-empty drug targets and non-empty disease
    genes that share zero targets. Disease-genes-empty / drug-targets-empty
    cases are kept (no evidence to demote).
    """
    dis_genes = disease_genes.get(disease_id, set())
    kept, dropped = [], []
    for p in predictions:
        d_id = p.drug_id
        if d_id not in biologic_pool:
            kept.append(p)
            continue
        d_tgt = drug_targets.get(d_id, set())
        if not d_tgt:
            kept.append(p)
            continue
        if not dis_genes:
            kept.append(p)
            continue
        if d_tgt & dis_genes:
            kept.append(p)
        else:
            dropped.append(p)
    return kept, dropped


def evaluate_disease(
    predictor: DrugRepurposingPredictor,
    disease_id: str,
    expanded_gt: Dict[str, Set[str]],
    biologic_pool: Set[str],
):
    """Return baseline + filtered metric dict, or None on skip."""
    gt_drugs = expanded_gt.get(disease_id, set())
    if not gt_drugs:
        return None, "no_gt"
    dis_name = predictor.disease_names.get(disease_id)
    if not dis_name:
        return None, "no_name"
    try:
        result = predictor.predict(
            dis_name, top_n=TOP_K_CANDIDATE_POOL, include_filtered=True
        )
    except Exception as e:
        return None, f"predict_err:{type(e).__name__}"

    preds = result.predictions
    if not preds:
        return None, "empty_preds"

    bio_gt = gt_drugs & biologic_pool
    sm_gt = gt_drugs - biologic_pool

    # Baseline: top-N by rank (no filter)
    base_top = preds[:TOP_N]
    base_top_ids = {p.drug_id for p in base_top}
    base_bio_in_top = base_top_ids & biologic_pool
    base_overall_r30 = len(base_top_ids & gt_drugs) / len(gt_drugs)
    base_bio_r30 = (
        len(base_top_ids & bio_gt) / len(bio_gt) if bio_gt else None
    )
    base_sm_r30 = (
        len(base_top_ids & sm_gt) / len(sm_gt) if sm_gt else None
    )
    base_bio_p30 = (
        len(base_top_ids & bio_gt) / len(base_bio_in_top)
        if base_bio_in_top else None
    )

    # Filter, then take top-N of survivors
    kept, dropped = filter_zero_overlap_biologic(
        preds, disease_id, biologic_pool,
        predictor.drug_targets, predictor.disease_genes,
    )
    filt_top = kept[:TOP_N]
    filt_top_ids = {p.drug_id for p in filt_top}
    filt_bio_in_top = filt_top_ids & biologic_pool
    filt_overall_r30 = len(filt_top_ids & gt_drugs) / len(gt_drugs)
    filt_bio_r30 = (
        len(filt_top_ids & bio_gt) / len(bio_gt) if bio_gt else None
    )
    filt_sm_r30 = (
        len(filt_top_ids & sm_gt) / len(sm_gt) if sm_gt else None
    )
    filt_bio_p30 = (
        len(filt_top_ids & bio_gt) / len(filt_bio_in_top)
        if filt_bio_in_top else None
    )

    # Tier shifts among dropped biologics (which tier did they have?)
    dropped_tiers = defaultdict(int)
    for p in dropped:
        tier = p.confidence_tier
        tier_name = tier.value if hasattr(tier, "value") else str(tier)
        dropped_tiers[tier_name] += 1

    # Of dropped that were in baseline top-30, count GT hits we lost
    base_dropped_gt = 0
    base_dropped_total = 0
    for p in dropped:
        if p.drug_id in base_top_ids:
            base_dropped_total += 1
            if p.drug_id in bio_gt:
                base_dropped_gt += 1

    return {
        "disease_id": disease_id,
        "disease_name": dis_name,
        "category": predictor.categorize_disease(dis_name),
        "gt_size": len(gt_drugs),
        "bio_gt_size": len(bio_gt),
        "sm_gt_size": len(sm_gt),
        "base_bio_in_top30": len(base_bio_in_top),
        "filt_bio_in_top30": len(filt_bio_in_top),
        "base_overall_r30": base_overall_r30,
        "filt_overall_r30": filt_overall_r30,
        "base_bio_r30": base_bio_r30,
        "filt_bio_r30": filt_bio_r30,
        "base_sm_r30": base_sm_r30,
        "filt_sm_r30": filt_sm_r30,
        "base_bio_p30": base_bio_p30,
        "filt_bio_p30": filt_bio_p30,
        "n_dropped_total": len(dropped),
        "n_dropped_was_in_top30": base_dropped_total,
        "n_dropped_was_in_top30_bio_gt": base_dropped_gt,
        "dropped_tier_counts": dict(dropped_tiers),
    }, None


def aggregate_seed(per_disease: List[Dict]) -> Dict:
    def _mean(vals):
        return float(np.mean(vals)) if vals else 0.0

    base_overall = [r["base_overall_r30"] for r in per_disease]
    filt_overall = [r["filt_overall_r30"] for r in per_disease]
    base_bio = [r["base_bio_r30"] for r in per_disease if r["base_bio_r30"] is not None]
    filt_bio = [r["filt_bio_r30"] for r in per_disease if r["filt_bio_r30"] is not None]
    base_sm = [r["base_sm_r30"] for r in per_disease if r["base_sm_r30"] is not None]
    filt_sm = [r["filt_sm_r30"] for r in per_disease if r["filt_sm_r30"] is not None]
    base_bp = [r["base_bio_p30"] for r in per_disease if r["base_bio_p30"] is not None]
    filt_bp = [r["filt_bio_p30"] for r in per_disease if r["filt_bio_p30"] is not None]

    return {
        "n_diseases": len(per_disease),
        "n_with_bio_gt": len(base_bio),
        "n_with_sm_gt": len(base_sm),
        "n_with_bio_in_top30_baseline": len(base_bp),
        "n_with_bio_in_top30_filtered": len(filt_bp),
        "base_overall_r30_mean": _mean(base_overall),
        "filt_overall_r30_mean": _mean(filt_overall),
        "base_bio_r30_mean": _mean(base_bio),
        "filt_bio_r30_mean": _mean(filt_bio),
        "base_sm_r30_mean": _mean(base_sm),
        "filt_sm_r30_mean": _mean(filt_sm),
        "base_bio_p30_mean": _mean(base_bp),
        "filt_bio_p30_mean": _mean(filt_bp),
        "n_dropped_total_sum": sum(r["n_dropped_total"] for r in per_disease),
        "n_dropped_in_top30_sum": sum(r["n_dropped_was_in_top30"] for r in per_disease),
        "n_dropped_in_top30_bio_gt_sum": sum(
            r["n_dropped_was_in_top30_bio_gt"] for r in per_disease
        ),
    }


def main():
    seeds = [42, 123, 456, 789, 2024]
    print("=" * 78)
    print("h957: Zero-overlap biologic safety filter — 5-seed h393 holdout")
    print(f"     candidate pool depth = {TOP_K_CANDIDATE_POOL}, final top-N = {TOP_N}")
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
    print(f"Biologic pool size: {len(biologic_pool)} of "
          f"{len(predictor.drug_targets)} drugs with targets")
    print(f"Diseases with non-empty disease_genes: "
          f"{sum(1 for v in predictor.disease_genes.values() if v)} of "
          f"{len(predictor.disease_genes)}")

    all_diseases = [
        d for d in predictor.ground_truth if d in predictor.embeddings
    ]
    print(f"Evaluable diseases (GT ∩ embeddings): {len(all_diseases)}")

    seed_results = []
    skip_summary = defaultdict(int)
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        train_set = set(train_ids)
        print(f"Train: {len(train_ids)}, Holdout: {len(holdout_ids)}")

        originals = recompute_gt_structures(predictor, train_set)
        per_disease: List[Dict] = []
        try:
            for dis_id in holdout_ids:
                rec, skip_reason = evaluate_disease(
                    predictor, dis_id, expanded_gt, biologic_pool
                )
                if rec is None:
                    skip_summary[skip_reason] += 1
                else:
                    per_disease.append(rec)
        finally:
            restore_gt_structures(predictor, originals)

        agg = aggregate_seed(per_disease)
        d_bio_p30 = 100 * (agg["filt_bio_p30_mean"] - agg["base_bio_p30_mean"])
        d_bio_r30 = 100 * (agg["filt_bio_r30_mean"] - agg["base_bio_r30_mean"])
        d_overall = 100 * (agg["filt_overall_r30_mean"] - agg["base_overall_r30_mean"])

        print(f"  evaluated={agg['n_diseases']}  "
              f"bio_gt_present={agg['n_with_bio_gt']}  "
              f"sm_gt_present={agg['n_with_sm_gt']}")
        print(f"  baseline:  overall={100*agg['base_overall_r30_mean']:5.2f}%  "
              f"bio_r30={100*agg['base_bio_r30_mean']:5.2f}%  "
              f"sm_r30={100*agg['base_sm_r30_mean']:5.2f}%  "
              f"bio_p30={100*agg['base_bio_p30_mean']:5.2f}%")
        print(f"  filtered:  overall={100*agg['filt_overall_r30_mean']:5.2f}%  "
              f"bio_r30={100*agg['filt_bio_r30_mean']:5.2f}%  "
              f"sm_r30={100*agg['filt_sm_r30_mean']:5.2f}%  "
              f"bio_p30={100*agg['filt_bio_p30_mean']:5.2f}%")
        print(f"  delta:     overall={d_overall:+5.2f}pp  "
              f"bio_r30={d_bio_r30:+5.2f}pp  bio_p30={d_bio_p30:+5.2f}pp")
        print(f"  drops_total={agg['n_dropped_total_sum']}  "
              f"in_top30={agg['n_dropped_in_top30_sum']}  "
              f"in_top30_were_bio_gt={agg['n_dropped_in_top30_bio_gt_sum']}")
        seed_results.append({"seed": seed, "agg": agg, "per_disease": per_disease})

    print("\nSkip reasons (summed across seeds):")
    for k, v in sorted(skip_summary.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<20s}  {v}")

    # 5-seed aggregate
    print("\n" + "=" * 78)
    print("5-SEED AGGREGATE (mean ± std)")
    print("=" * 78)
    metric_keys = [
        "base_overall_r30_mean", "filt_overall_r30_mean",
        "base_bio_r30_mean", "filt_bio_r30_mean",
        "base_sm_r30_mean", "filt_sm_r30_mean",
        "base_bio_p30_mean", "filt_bio_p30_mean",
    ]
    aggregate = {}
    for k in metric_keys:
        vals = [r["agg"][k] for r in seed_results]
        aggregate[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        print(f"  {k:<28s} = {100*aggregate[k]['mean']:6.2f}% ± "
              f"{100*aggregate[k]['std']:5.2f}%")

    delta_bio_p30 = 100 * (
        aggregate["filt_bio_p30_mean"]["mean"]
        - aggregate["base_bio_p30_mean"]["mean"]
    )
    delta_bio_r30 = 100 * (
        aggregate["filt_bio_r30_mean"]["mean"]
        - aggregate["base_bio_r30_mean"]["mean"]
    )
    delta_overall = 100 * (
        aggregate["filt_overall_r30_mean"]["mean"]
        - aggregate["base_overall_r30_mean"]["mean"]
    )
    delta_sm = 100 * (
        aggregate["filt_sm_r30_mean"]["mean"]
        - aggregate["base_sm_r30_mean"]["mean"]
    )
    print()
    print(f"  delta bio_p30   = {delta_bio_p30:+5.2f}pp  (target: >=+3pp)")
    print(f"  delta bio_r30   = {delta_bio_r30:+5.2f}pp  (cap: drop <=2pp)")
    print(f"  delta overall   = {delta_overall:+5.2f}pp")
    print(f"  delta sm_r30    = {delta_sm:+5.2f}pp  (should be ~0)")

    # Per-category aggregate (filtered impact on bio_p30)
    print("\n" + "=" * 78)
    print("PER-CATEGORY (5-seed mean of bio_p30 baseline vs filtered)")
    print("=" * 78)
    cat_records: Dict[str, List[Dict]] = defaultdict(list)
    for sr in seed_results:
        for rec in sr["per_disease"]:
            cat_records[rec["category"]].append(rec)
    cat_aggregate = {}
    print(f"{'category':<20s}  {'n_bio_top30':>11s}  "
          f"{'base_p30':>9s}  {'filt_p30':>9s}  {'Δp30':>7s}  "
          f"{'base_r30':>9s}  {'filt_r30':>9s}  {'Δr30':>7s}")
    for cat in sorted(cat_records):
        recs = cat_records[cat]
        base_p = [r["base_bio_p30"] for r in recs if r["base_bio_p30"] is not None]
        filt_p = [r["filt_bio_p30"] for r in recs if r["filt_bio_p30"] is not None]
        base_r = [r["base_bio_r30"] for r in recs if r["base_bio_r30"] is not None]
        filt_r = [r["filt_bio_r30"] for r in recs if r["filt_bio_r30"] is not None]
        if not base_p:
            continue
        bp = float(np.mean(base_p))
        fp = float(np.mean(filt_p))
        br = float(np.mean(base_r)) if base_r else 0.0
        fr = float(np.mean(filt_r)) if filt_r else 0.0
        cat_aggregate[cat] = {
            "n_bio_top30_obs": len(base_p),
            "base_bio_p30": bp,
            "filt_bio_p30": fp,
            "delta_bio_p30": fp - bp,
            "base_bio_r30": br,
            "filt_bio_r30": fr,
            "delta_bio_r30": fr - br,
            "n_bio_r30_obs": len(base_r),
        }
        print(f"  {cat:<18s}  {len(base_p):11d}  "
              f"{100*bp:8.2f}%  {100*fp:8.2f}%  {100*(fp-bp):+6.2f}  "
              f"{100*br:8.2f}%  {100*fr:8.2f}%  {100*(fr-br):+6.2f}")

    # Tier-shift aggregate
    print("\n" + "=" * 78)
    print("DROPPED BIOLOGIC PREDICTIONS BY TIER (5-seed sum)")
    print("=" * 78)
    tier_sum: Dict[str, int] = defaultdict(int)
    for sr in seed_results:
        for rec in sr["per_disease"]:
            for tier, n in rec["dropped_tier_counts"].items():
                tier_sum[tier] += n
    for tier in sorted(tier_sum, key=lambda t: -tier_sum[t]):
        print(f"  {tier:<10s}  {tier_sum[tier]}")

    # Decision
    print("\n" + "=" * 78)
    print("DECISION (h957)")
    print("=" * 78)
    if delta_bio_p30 >= 3.0 and delta_bio_r30 >= -2.0:
        verdict = "SHIP — bio_p30 lift >= +3pp without bio_r30 dropping >2pp"
    elif delta_bio_p30 >= 3.0 and delta_bio_r30 < -2.0:
        verdict = "BORDERLINE — bio_p30 gain offset by bio_r30 drop"
    elif delta_bio_p30 < 1.0:
        verdict = "REJECT — filter does not move bio_p30 meaningfully"
    else:
        verdict = "BELOW THRESHOLD — bio_p30 lift < +3pp"
    print(f"  delta bio_p30 = {delta_bio_p30:+.2f}pp")
    print(f"  delta bio_r30 = {delta_bio_r30:+.2f}pp")
    print(f"  VERDICT: {verdict}")

    out = {
        "hypothesis": "h957",
        "title": "Zero-overlap biologic safety filter (h949 implementation)",
        "seeds": seeds,
        "biologic_pool_size": len(biologic_pool),
        "evaluable_diseases": len(all_diseases),
        "candidate_pool_depth": TOP_K_CANDIDATE_POOL,
        "final_top_n": TOP_N,
        "skip_summary": dict(skip_summary),
        "per_seed_aggregate": [
            {"seed": r["seed"], "agg": r["agg"]} for r in seed_results
        ],
        "five_seed_aggregate": aggregate,
        "delta_bio_p30_pp": delta_bio_p30,
        "delta_bio_r30_pp": delta_bio_r30,
        "delta_overall_r30_pp": delta_overall,
        "delta_sm_r30_pp": delta_sm,
        "cat_aggregate": cat_aggregate,
        "dropped_tier_counts": dict(tier_sum),
        "verdict": verdict,
    }
    out_path = ROOT / "data/analysis/h957_zero_overlap_biologic_filter.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
