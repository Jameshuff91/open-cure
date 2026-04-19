#!/usr/bin/env python3
"""h960: Neurological supplement (h173) + SELECTIVE_BOOST interaction ablation.

Why:
    h952 post-fix seed-42 stratification showed:
        neither_boost_nor_supp (n=125): delta = 0.00pp
        supp_only (GI)          (n=  7): delta = 0.00pp
        boost_only              (n= 17): delta = +1.41pp
        boost+supp (neuro)      (n= 13): delta = -2.85pp
    Pattern: SELECTIVE_BOOST helps when alone; the h173 neurological
    supplement (`_supplement_neurological_predictions`) re-sorts predictions
    by tier priority and can displace kNN top-30 with class-injected
    drugs capped at MEDIUM. When boost is also active, the combination
    over-replaces correct kNN predictions on the n=13 boost+supp
    (= neurological) stratum.

Design:
    Run two 5-seed h393 holdout passes:
      A. Baseline = current production (`predict()` unchanged).
      B. Ablation = monkey-patch `_supplement_neurological_predictions` to
         be a no-op (return existing_predictions). All other production
         logic unchanged including SELECTIVE_BOOST.
    Compare:
      - Neurological-cohort overall_r30, bio_r30, sm_r30 baseline vs ablated.
      - Non-neurological-cohort metrics (control — must be 0pp).
      - Global aggregate (informative, not a ship criterion).

Decision (h960):
    Confirm if delta_neuro_overall_r30 (ablated - baseline) >= +1.0pp.
    If yes → propose disabling supplement entirely OR demoting class_injected
    to LOW annotation (h971). If no → close h960 invalidated.
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

TOP_N = 30


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


def evaluate_disease(predictor, dis_id, expanded_gt, biologic_pool, top_n=TOP_N):
    gt_drugs = expanded_gt.get(dis_id, set())
    if not gt_drugs:
        return None, "no_gt"
    dis_name = predictor.disease_names.get(dis_id)
    if not dis_name:
        return None, "no_name"
    try:
        result = predictor.predict(dis_name, top_n=top_n, include_filtered=True)
    except Exception as e:
        return None, f"predict_err:{type(e).__name__}"
    preds = result.predictions[:top_n]
    if not preds:
        return None, "empty_preds"

    top_ids = {p.drug_id for p in preds}
    bio_gt = gt_drugs & biologic_pool
    sm_gt = gt_drugs - biologic_pool
    overall_r30 = len(top_ids & gt_drugs) / len(gt_drugs)
    bio_r30 = len(top_ids & bio_gt) / len(bio_gt) if bio_gt else None
    sm_r30 = len(top_ids & sm_gt) / len(sm_gt) if sm_gt else None
    category = predictor.categorize_disease(dis_name)
    return {
        "disease_id": dis_id,
        "disease_name": dis_name,
        "category": category,
        "is_neuro": category == "neurological",
        "gt_size": len(gt_drugs),
        "bio_gt_size": len(bio_gt),
        "sm_gt_size": len(sm_gt),
        "overall_r30": overall_r30,
        "bio_r30": bio_r30,
        "sm_r30": sm_r30,
    }, None


def aggregate(records, label):
    def _mean(vs):
        return float(np.mean(vs)) if vs else 0.0
    def _std(vs):
        return float(np.std(vs)) if vs else 0.0
    overall = [r["overall_r30"] for r in records]
    bio = [r["bio_r30"] for r in records if r["bio_r30"] is not None]
    sm = [r["sm_r30"] for r in records if r["sm_r30"] is not None]
    return {
        "label": label,
        "n": len(records),
        "overall_r30": _mean(overall),
        "overall_r30_std": _std(overall),
        "n_bio_gt": len(bio),
        "bio_r30": _mean(bio),
        "bio_r30_std": _std(bio),
        "n_sm_gt": len(sm),
        "sm_r30": _mean(sm),
        "sm_r30_std": _std(sm),
    }


def _disable_neuro_supplement(predictor):
    """Monkey-patch supplement to return existing predictions unchanged."""
    def _noop(self, disease_name, disease_id, disease_tier, category,
              existing_predictions, max_knn_score, top_n, include_filtered):
        return existing_predictions
    # Rebind the method on this instance only
    import types
    predictor._original_supplement_neuro = predictor._supplement_neurological_predictions
    predictor._supplement_neurological_predictions = types.MethodType(
        _noop, predictor
    )


def _restore_neuro_supplement(predictor):
    if hasattr(predictor, "_original_supplement_neuro"):
        predictor._supplement_neurological_predictions = (
            predictor._original_supplement_neuro
        )
        delattr(predictor, "_original_supplement_neuro")


def run_passes(predictor, expanded_gt, biologic_pool, holdout_ids):
    """Returns (baseline_records, ablated_records) for one seed."""
    # Pass A: baseline (supplement on)
    baseline = []
    skip_b: Dict[str, int] = defaultdict(int)
    for dis_id in holdout_ids:
        rec, sk = evaluate_disease(predictor, dis_id, expanded_gt, biologic_pool)
        if rec is None:
            skip_b[sk or "unknown"] += 1
        else:
            baseline.append(rec)

    # Pass B: ablated (supplement disabled)
    _disable_neuro_supplement(predictor)
    try:
        ablated = []
        skip_a: Dict[str, int] = defaultdict(int)
        for dis_id in holdout_ids:
            rec, sk = evaluate_disease(predictor, dis_id, expanded_gt, biologic_pool)
            if rec is None:
                skip_a[sk or "unknown"] += 1
            else:
                ablated.append(rec)
    finally:
        _restore_neuro_supplement(predictor)

    return baseline, ablated, dict(skip_b), dict(skip_a)


def per_disease_delta(baseline, ablated):
    """Pair by disease_id and compute per-disease delta."""
    by_id_b = {r["disease_id"]: r for r in baseline}
    by_id_a = {r["disease_id"]: r for r in ablated}
    common_ids = set(by_id_b) & set(by_id_a)
    deltas = []
    for did in common_ids:
        b = by_id_b[did]
        a = by_id_a[did]
        deltas.append({
            "disease_id": did,
            "disease_name": b["disease_name"],
            "category": b["category"],
            "is_neuro": b["is_neuro"],
            "gt_size": b["gt_size"],
            "delta_overall_r30": a["overall_r30"] - b["overall_r30"],
            "baseline_overall_r30": b["overall_r30"],
            "ablated_overall_r30": a["overall_r30"],
        })
    return deltas


def main():
    seeds = [42, 123, 456, 789, 2024]
    print("=" * 78)
    print("h960: Neurological supplement (h173) ablation — 5-seed h393 holdout")
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
    print(f"Evaluable diseases: {len(all_diseases)}")
    print(f"Biologic pool: {len(biologic_pool)}")

    seed_results = []
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        train_set = set(train_ids)
        originals = recompute_gt_structures(predictor, train_set)
        try:
            baseline, ablated, skip_b, skip_a = run_passes(
                predictor, expanded_gt, biologic_pool, holdout_ids
            )
        finally:
            restore_gt_structures(predictor, originals)

        deltas = per_disease_delta(baseline, ablated)
        neuro_d = [d for d in deltas if d["is_neuro"]]
        nonneuro_d = [d for d in deltas if not d["is_neuro"]]

        agg_b_neuro = aggregate(
            [r for r in baseline if r["is_neuro"]], "baseline_neuro"
        )
        agg_a_neuro = aggregate(
            [r for r in ablated if r["is_neuro"]], "ablated_neuro"
        )
        agg_b_non = aggregate(
            [r for r in baseline if not r["is_neuro"]], "baseline_nonneuro"
        )
        agg_a_non = aggregate(
            [r for r in ablated if not r["is_neuro"]], "ablated_nonneuro"
        )
        agg_b_all = aggregate(baseline, "baseline_all")
        agg_a_all = aggregate(ablated, "ablated_all")

        d_neuro_overall = 100 * (agg_a_neuro["overall_r30"] - agg_b_neuro["overall_r30"])
        d_non_overall = 100 * (agg_a_non["overall_r30"] - agg_b_non["overall_r30"])
        d_all_overall = 100 * (agg_a_all["overall_r30"] - agg_b_all["overall_r30"])

        print(f"  evaluated_baseline={agg_b_all['n']}  evaluated_ablated={agg_a_all['n']}  "
              f"neuro_n_baseline={agg_b_neuro['n']}  neuro_n_ablated={agg_a_neuro['n']}")
        print(f"  NEURO   base_overall={100*agg_b_neuro['overall_r30']:5.2f}%  "
              f"abla_overall={100*agg_a_neuro['overall_r30']:5.2f}%  Δ={d_neuro_overall:+.2f}pp")
        print(f"  NON-N   base_overall={100*agg_b_non['overall_r30']:5.2f}%  "
              f"abla_overall={100*agg_a_non['overall_r30']:5.2f}%  Δ={d_non_overall:+.2f}pp (control: ≈0)")
        print(f"  ALL     base_overall={100*agg_b_all['overall_r30']:5.2f}%  "
              f"abla_overall={100*agg_a_all['overall_r30']:5.2f}%  Δ={d_all_overall:+.2f}pp")
        print(f"  skip_baseline={skip_b}  skip_ablated={skip_a}")

        # Per-neuro-disease delta detail (top movers)
        if neuro_d:
            sorted_neuro = sorted(neuro_d, key=lambda x: x["delta_overall_r30"], reverse=True)
            print(f"  Neuro per-disease deltas (top 5 helpers, top 5 hurt):")
            for d in sorted_neuro[:5]:
                if d["delta_overall_r30"] != 0:
                    print(f"    +{100*d['delta_overall_r30']:5.2f}pp  {d['disease_name'][:40]:40s}  gt={d['gt_size']}")
            for d in sorted_neuro[-5:]:
                if d["delta_overall_r30"] < 0:
                    print(f"    {100*d['delta_overall_r30']:+5.2f}pp  {d['disease_name'][:40]:40s}  gt={d['gt_size']}")

        seed_results.append({
            "seed": seed,
            "agg_baseline_neuro": agg_b_neuro,
            "agg_ablated_neuro": agg_a_neuro,
            "agg_baseline_nonneuro": agg_b_non,
            "agg_ablated_nonneuro": agg_a_non,
            "agg_baseline_all": agg_b_all,
            "agg_ablated_all": agg_a_all,
            "n_neuro_helpers": sum(1 for d in neuro_d if d["delta_overall_r30"] > 0),
            "n_neuro_hurt": sum(1 for d in neuro_d if d["delta_overall_r30"] < 0),
            "n_neuro_unchanged": sum(1 for d in neuro_d if d["delta_overall_r30"] == 0),
            "neuro_delta_records": neuro_d,
        })

    # 5-seed aggregate
    print("\n" + "=" * 78)
    print("5-SEED AGGREGATE (mean ± std)")
    print("=" * 78)

    def five_seed(scope_key):
        bvals = [r[f"agg_baseline_{scope_key}"]["overall_r30"] for r in seed_results]
        avals = [r[f"agg_ablated_{scope_key}"]["overall_r30"] for r in seed_results]
        bmean, bstd = float(np.mean(bvals)), float(np.std(bvals))
        amean, astd = float(np.mean(avals)), float(np.std(avals))
        return {
            "baseline_mean": bmean, "baseline_std": bstd,
            "ablated_mean": amean,  "ablated_std": astd,
            "delta_pp": 100 * (amean - bmean),
        }

    five_neuro = five_seed("neuro")
    five_non = five_seed("nonneuro")
    five_all = five_seed("all")

    print(f"\n[NEUROLOGICAL]")
    print(f"  base = {100*five_neuro['baseline_mean']:5.2f}% ± {100*five_neuro['baseline_std']:5.2f}")
    print(f"  abla = {100*five_neuro['ablated_mean']:5.2f}% ± {100*five_neuro['ablated_std']:5.2f}")
    print(f"  Δ    = {five_neuro['delta_pp']:+5.2f}pp")
    print(f"\n[NON-NEUROLOGICAL] (control)")
    print(f"  base = {100*five_non['baseline_mean']:5.2f}% ± {100*five_non['baseline_std']:5.2f}")
    print(f"  abla = {100*five_non['ablated_mean']:5.2f}% ± {100*five_non['ablated_std']:5.2f}")
    print(f"  Δ    = {five_non['delta_pp']:+5.2f}pp  (must be ≈ 0)")
    print(f"\n[ALL]")
    print(f"  base = {100*five_all['baseline_mean']:5.2f}% ± {100*five_all['baseline_std']:5.2f}")
    print(f"  abla = {100*five_all['ablated_mean']:5.2f}% ± {100*five_all['ablated_std']:5.2f}")
    print(f"  Δ    = {five_all['delta_pp']:+5.2f}pp")

    # Per-disease counts: how many neuro diseases did supplement help vs hurt?
    total_n_helped, total_n_hurt, total_n_unchanged = 0, 0, 0
    for r in seed_results:
        total_n_helped += r["n_neuro_hurt"]   # ablation HELPS where supplement HURTS
        total_n_hurt += r["n_neuro_helpers"]  # ablation HURTS where supplement HELPS
        total_n_unchanged += r["n_neuro_unchanged"]
    print(f"\n[Per-neuro-disease impact summed over 5 seeds]")
    print(f"  ablation helps (supplement hurts): {total_n_helped}")
    print(f"  ablation hurts (supplement helps): {total_n_hurt}")
    print(f"  unchanged:                          {total_n_unchanged}")

    print("\n" + "=" * 78)
    print("DECISION (h960)")
    print("=" * 78)
    if abs(five_non['delta_pp']) > 0.5:
        verdict = (
            f"INVALID CONTROL — non-neuro Δ={five_non['delta_pp']:+.2f}pp suggests "
            "monkey-patch leakage; revisit experiment design."
        )
    elif five_neuro['delta_pp'] >= 1.0:
        verdict = (
            f"VALIDATED — disabling neuro supplement lifts neuro overall_r30 by "
            f"{five_neuro['delta_pp']:+.2f}pp ≥ +1.0pp threshold; "
            "h971 (demote class_injected) recommended."
        )
    elif five_neuro['delta_pp'] <= -1.0:
        verdict = (
            f"INVERSE — supplement HELPS neuro overall_r30 by "
            f"{-five_neuro['delta_pp']:+.2f}pp; h952 'boost+supp' regression is NOT "
            "explained by the supplement alone."
        )
    else:
        verdict = (
            f"INCONCLUSIVE — neuro Δ={five_neuro['delta_pp']:+.2f}pp within ±1pp "
            "noise band; supplement is benign at the cohort level."
        )
    print(f"  neuro Δ overall_r30  = {five_neuro['delta_pp']:+.2f}pp")
    print(f"  non-neuro Δ overall  = {five_non['delta_pp']:+.2f}pp (control)")
    print(f"  VERDICT: {verdict}")

    out = {
        "hypothesis": "h960",
        "title": "Neurological supplement (h173) ablation",
        "seeds": seeds,
        "evaluable_diseases": len(all_diseases),
        "per_seed": [
            {
                "seed": r["seed"],
                "agg_baseline_neuro": r["agg_baseline_neuro"],
                "agg_ablated_neuro": r["agg_ablated_neuro"],
                "agg_baseline_nonneuro": r["agg_baseline_nonneuro"],
                "agg_ablated_nonneuro": r["agg_ablated_nonneuro"],
                "agg_baseline_all": r["agg_baseline_all"],
                "agg_ablated_all": r["agg_ablated_all"],
                "n_neuro_helpers": r["n_neuro_helpers"],
                "n_neuro_hurt": r["n_neuro_hurt"],
                "n_neuro_unchanged": r["n_neuro_unchanged"],
                "neuro_delta_records": r["neuro_delta_records"],
            }
            for r in seed_results
        ],
        "five_seed": {
            "neuro": five_neuro,
            "nonneuro": five_non,
            "all": five_all,
        },
        "per_disease_summed": {
            "ablation_helps_supplement_hurts": total_n_helped,
            "ablation_hurts_supplement_helps": total_n_hurt,
            "unchanged": total_n_unchanged,
        },
        "verdict": verdict,
    }
    out_path = ROOT / "data/analysis/h960_neuro_supplement_ablation.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
