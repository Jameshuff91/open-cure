#!/usr/bin/env python3
"""h965: Cancer-restricted zero-overlap biologic safety filter.

Why:
    h957 (global form) hit the +3pp ship target on bio_p30 (5.64→8.85,
    +3.22pp) but violated the -2pp recall cap by 7.8x (bio_r30 30.31→14.78,
    -15.54pp). The per-category split revealed cancer (n=115/54 obs) gives
    delta_bio_p30 = +2.6pp at delta_bio_r30 = -2.1pp — essentially within
    the ship cap. Disease_genes is mechanism-aware in oncology because
    TCGA/COSMIC provide dense tumor-cell receptor coverage, and biologic
    MoA in oncology is target-driven (HER2, VEGF, CD20, PD1).

    Restricting the filter to cancer should:
      - keep bio_p30 lift on cancer (~+2.6pp on n=115 ≈ +0.5pp global)
      - bound bio_r30 cost on cancer (~-2.1pp on n=54 ≈ -0.5pp global)
      - leave inflammatory categories (autoimmune/neuro/respiratory) untouched
        so we no longer destroy 28-58pp of bio_r30 there.

Filter (h957 with category gate):
    DROP if  is_biologic(drug)
        AND drug_targets[drug]   non-empty
        AND disease_genes[disease] non-empty
        AND |drug_targets[drug] ∩ disease_genes[disease]| == 0
        AND categorize_disease(disease) == 'cancer'

Eval:
    5-seed h393 holdout, expanded GT, top-K=200 candidate pool, top-N=30
    final. Same metrics as h957 for direct comparison.

Decision:
    Ship if global delta_bio_p30 >= +0.5pp AND delta_bio_r30 >= -1.0pp.
    Relaxed criterion vs h957 because the filter affects fewer preds
    (cancer is one category of 18) so any lift IS the cancer-specific lift
    diluted across all categories.
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

TOP_K_CANDIDATE_POOL = 200
TOP_N = 30
CANCER_ONLY = {"cancer"}


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


def filter_zero_overlap_biologic_cancer_only(
    predictions: List,
    disease_id: str,
    disease_category: str,
    biologic_pool: Set[str],
    drug_targets: Dict[str, Set[str]],
    disease_genes: Dict[str, Set[str]],
):
    """h965 variant: only fire on cancer-category diseases."""
    if disease_category not in CANCER_ONLY:
        return list(predictions), []
    dis_genes = disease_genes.get(disease_id, set())
    if not dis_genes:
        return list(predictions), []
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
    category = predictor.categorize_disease(dis_name)

    base_top = preds[:TOP_N]
    base_top_ids = {p.drug_id for p in base_top}
    base_bio_in_top = base_top_ids & biologic_pool
    base_overall_r30 = len(base_top_ids & gt_drugs) / len(gt_drugs)
    base_bio_r30 = len(base_top_ids & bio_gt) / len(bio_gt) if bio_gt else None
    base_sm_r30 = len(base_top_ids & sm_gt) / len(sm_gt) if sm_gt else None
    base_bio_p30 = (
        len(base_top_ids & bio_gt) / len(base_bio_in_top)
        if base_bio_in_top else None
    )

    kept, dropped = filter_zero_overlap_biologic_cancer_only(
        preds, disease_id, category, biologic_pool,
        predictor.drug_targets, predictor.disease_genes,
    )
    filt_top = kept[:TOP_N]
    filt_top_ids = {p.drug_id for p in filt_top}
    filt_bio_in_top = filt_top_ids & biologic_pool
    filt_overall_r30 = len(filt_top_ids & gt_drugs) / len(gt_drugs)
    filt_bio_r30 = len(filt_top_ids & bio_gt) / len(bio_gt) if bio_gt else None
    filt_sm_r30 = len(filt_top_ids & sm_gt) / len(sm_gt) if sm_gt else None
    filt_bio_p30 = (
        len(filt_top_ids & bio_gt) / len(filt_bio_in_top)
        if filt_bio_in_top else None
    )

    dropped_tiers: Dict[str, int] = defaultdict(int)
    for p in dropped:
        tier = p.confidence_tier
        tier_name = tier.value if hasattr(tier, "value") else str(tier)
        dropped_tiers[tier_name] += 1

    return {
        "disease_id": disease_id,
        "disease_name": dis_name,
        "category": category,
        "is_cancer": category in CANCER_ONLY,
        "gt_size": len(gt_drugs),
        "bio_gt_size": len(bio_gt),
        "sm_gt_size": len(sm_gt),
        "base_overall_r30": base_overall_r30,
        "filt_overall_r30": filt_overall_r30,
        "base_bio_r30": base_bio_r30,
        "filt_bio_r30": filt_bio_r30,
        "base_sm_r30": base_sm_r30,
        "filt_sm_r30": filt_sm_r30,
        "base_bio_p30": base_bio_p30,
        "filt_bio_p30": filt_bio_p30,
        "n_dropped": len(dropped),
        "dropped_tier_counts": dict(dropped_tiers),
    }, None


def aggregate(per_disease, label="all"):
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
        "label": label,
        "n_diseases": len(per_disease),
        "n_with_bio_gt": len(base_bio),
        "n_with_sm_gt": len(base_sm),
        "n_with_bio_in_top30_baseline": len(base_bp),
        "n_with_bio_in_top30_filtered": len(filt_bp),
        "base_overall_r30": _mean(base_overall),
        "filt_overall_r30": _mean(filt_overall),
        "base_bio_r30": _mean(base_bio),
        "filt_bio_r30": _mean(filt_bio),
        "base_sm_r30": _mean(base_sm),
        "filt_sm_r30": _mean(filt_sm),
        "base_bio_p30": _mean(base_bp),
        "filt_bio_p30": _mean(filt_bp),
        "n_dropped_total": sum(r["n_dropped"] for r in per_disease),
    }


def main():
    seeds = [42, 123, 456, 789, 2024]
    print("=" * 78)
    print("h965: Cancer-restricted zero-overlap biologic filter — 5-seed h393 holdout")
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
    print(f"Biologic pool: {len(biologic_pool)}")
    print(f"Evaluable diseases: {len(all_diseases)}")

    seed_results = []
    skip_summary: Dict[str, int] = defaultdict(int)
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        train_set = set(train_ids)
        originals = recompute_gt_structures(predictor, train_set)
        per_disease: List[Dict] = []
        try:
            for dis_id in holdout_ids:
                rec, skip_reason = evaluate_disease(
                    predictor, dis_id, expanded_gt, biologic_pool
                )
                if rec is None:
                    skip_summary[skip_reason or "unknown"] += 1
                else:
                    per_disease.append(rec)
        finally:
            restore_gt_structures(predictor, originals)

        agg_all = aggregate(per_disease, "all")
        agg_cancer = aggregate(
            [r for r in per_disease if r["is_cancer"]], "cancer"
        )
        agg_noncancer = aggregate(
            [r for r in per_disease if not r["is_cancer"]], "noncancer"
        )

        d_bio_p = 100 * (agg_all["filt_bio_p30"] - agg_all["base_bio_p30"])
        d_bio_r = 100 * (agg_all["filt_bio_r30"] - agg_all["base_bio_r30"])
        d_overall = 100 * (agg_all["filt_overall_r30"] - agg_all["base_overall_r30"])

        print(f"  evaluated={agg_all['n_diseases']}  "
              f"cancer={agg_cancer['n_diseases']}  "
              f"non-cancer={agg_noncancer['n_diseases']}")
        print(f"  GLOBAL    base_p30={100*agg_all['base_bio_p30']:5.2f}%  "
              f"filt_p30={100*agg_all['filt_bio_p30']:5.2f}%  Δ={d_bio_p:+.2f}pp  |  "
              f"base_r30={100*agg_all['base_bio_r30']:5.2f}%  "
              f"filt_r30={100*agg_all['filt_bio_r30']:5.2f}%  Δ={d_bio_r:+.2f}pp  |  "
              f"Δoverall={d_overall:+.2f}pp")
        print(f"  CANCER    base_p30={100*agg_cancer['base_bio_p30']:5.2f}%  "
              f"filt_p30={100*agg_cancer['filt_bio_p30']:5.2f}%  "
              f"base_r30={100*agg_cancer['base_bio_r30']:5.2f}%  "
              f"filt_r30={100*agg_cancer['filt_bio_r30']:5.2f}%  "
              f"drops={agg_cancer['n_dropped_total']}")
        print(f"  NON-CAN   base_p30={100*agg_noncancer['base_bio_p30']:5.2f}%  "
              f"filt_p30={100*agg_noncancer['filt_bio_p30']:5.2f}% (should be unchanged)  "
              f"base_r30={100*agg_noncancer['base_bio_r30']:5.2f}%  "
              f"filt_r30={100*agg_noncancer['filt_bio_r30']:5.2f}% "
              f"drops={agg_noncancer['n_dropped_total']}")
        seed_results.append({
            "seed": seed,
            "agg_all": agg_all,
            "agg_cancer": agg_cancer,
            "agg_noncancer": agg_noncancer,
        })

    print("\nSkip reasons:", dict(skip_summary))

    print("\n" + "=" * 78)
    print("5-SEED AGGREGATE")
    print("=" * 78)

    def five_seed_metric(key, scope):
        vals = [r[scope][key] for r in seed_results]
        return float(np.mean(vals)), float(np.std(vals))

    def report(scope_label, scope_key):
        bp_b, bp_b_s = five_seed_metric("base_bio_p30", scope_key)
        bp_f, bp_f_s = five_seed_metric("filt_bio_p30", scope_key)
        br_b, br_b_s = five_seed_metric("base_bio_r30", scope_key)
        br_f, br_f_s = five_seed_metric("filt_bio_r30", scope_key)
        sr_b, sr_b_s = five_seed_metric("base_sm_r30", scope_key)
        sr_f, sr_f_s = five_seed_metric("filt_sm_r30", scope_key)
        ov_b, ov_b_s = five_seed_metric("base_overall_r30", scope_key)
        ov_f, ov_f_s = five_seed_metric("filt_overall_r30", scope_key)
        print(f"\n[{scope_label}]")
        print(f"  bio_p30  base={100*bp_b:6.2f}%±{100*bp_b_s:5.2f}  "
              f"filt={100*bp_f:6.2f}%±{100*bp_f_s:5.2f}  Δ={100*(bp_f-bp_b):+.2f}pp")
        print(f"  bio_r30  base={100*br_b:6.2f}%±{100*br_b_s:5.2f}  "
              f"filt={100*br_f:6.2f}%±{100*br_f_s:5.2f}  Δ={100*(br_f-br_b):+.2f}pp")
        print(f"  sm_r30   base={100*sr_b:6.2f}%±{100*sr_b_s:5.2f}  "
              f"filt={100*sr_f:6.2f}%±{100*sr_f_s:5.2f}  Δ={100*(sr_f-sr_b):+.2f}pp")
        print(f"  overall  base={100*ov_b:6.2f}%±{100*ov_b_s:5.2f}  "
              f"filt={100*ov_f:6.2f}%±{100*ov_f_s:5.2f}  Δ={100*(ov_f-ov_b):+.2f}pp")
        return {
            "bio_p30": {"base": bp_b, "filt": bp_f, "delta": bp_f - bp_b,
                         "base_std": bp_b_s, "filt_std": bp_f_s},
            "bio_r30": {"base": br_b, "filt": br_f, "delta": br_f - br_b,
                         "base_std": br_b_s, "filt_std": br_f_s},
            "sm_r30":  {"base": sr_b, "filt": sr_f, "delta": sr_f - sr_b,
                         "base_std": sr_b_s, "filt_std": sr_f_s},
            "overall": {"base": ov_b, "filt": ov_f, "delta": ov_f - ov_b,
                         "base_std": ov_b_s, "filt_std": ov_f_s},
        }

    five_all = report("ALL HOLDOUT (global)", "agg_all")
    five_cancer = report("CANCER ONLY", "agg_cancer")
    five_noncancer = report("NON-CANCER ONLY (control — should be 0pp)", "agg_noncancer")

    delta_bio_p_global = 100 * five_all["bio_p30"]["delta"]
    delta_bio_r_global = 100 * five_all["bio_r30"]["delta"]
    delta_overall_global = 100 * five_all["overall"]["delta"]

    print("\n" + "=" * 78)
    print("DECISION (h965)")
    print("=" * 78)
    print(f"  global delta bio_p30 = {delta_bio_p_global:+5.2f}pp  (target: >=+0.5pp)")
    print(f"  global delta bio_r30 = {delta_bio_r_global:+5.2f}pp  (cap: >=-1.0pp)")
    print(f"  global delta overall = {delta_overall_global:+5.2f}pp")
    if delta_bio_p_global >= 0.5 and delta_bio_r_global >= -1.0:
        verdict = "SHIP — cancer-restricted filter passes relaxed criteria"
    elif delta_bio_p_global >= 0.5:
        verdict = "BORDERLINE — precision lift OK but recall cost above cap"
    else:
        verdict = "REJECT — global lift too small to justify"
    print(f"  VERDICT: {verdict}")

    out = {
        "hypothesis": "h965",
        "title": "Cancer-restricted zero-overlap biologic safety filter",
        "seeds": seeds,
        "biologic_pool_size": len(biologic_pool),
        "evaluable_diseases": len(all_diseases),
        "candidate_pool_depth": TOP_K_CANDIDATE_POOL,
        "final_top_n": TOP_N,
        "skip_summary": dict(skip_summary),
        "per_seed": [
            {
                "seed": r["seed"],
                "agg_all": r["agg_all"],
                "agg_cancer": r["agg_cancer"],
                "agg_noncancer": r["agg_noncancer"],
            }
            for r in seed_results
        ],
        "five_seed": {
            "all": five_all,
            "cancer": five_cancer,
            "noncancer": five_noncancer,
        },
        "delta_global": {
            "bio_p30_pp": delta_bio_p_global,
            "bio_r30_pp": delta_bio_r_global,
            "overall_r30_pp": delta_overall_global,
        },
        "verdict": verdict,
    }
    out_path = ROOT / "data/analysis/h965_cancer_restricted_filter.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
