#!/usr/bin/env python3
"""h951: Biologic R@30 reality-check on the CURRENT production pipeline.

Why:
    h906/h920/h921/h924 all cite a "biologic R@30 = 27.3%" failure baseline
    that traces back to research_spec.md (no methodology) and an early
    snapshot in RESEARCH_ROADMAP.md. h940 already produced a 31.4% biologic
    R@30 with PLAIN kNN on expanded GT (5 seeds, mixed pool). Before
    spending GPU on ESM2 (h921), license effort on DrugBank (h906), or
    dense-embedding ETL on PubMedBERT (h920), measure biologic R@30 on the
    full PRODUCTION pipeline (kNN + selective category boost + tier rules)
    on the same 5-seed holdout framework.

Design:
    Reuse h393's holdout machinery (split + recompute GT-derived structures
    from train-only). For each holdout disease:
      - Call predictor.predict(disease_name, top_n=30, include_filtered=True)
      - top-30 by rank (NOT tier-filtered) — this is the production ranking
        before the user-facing tier filter
      - bio_r30 = (top30 ∩ bio_GT) / |bio_GT|     when |bio_GT| >= 1
      - sm_r30  = (top30 ∩ sm_GT)  / |sm_GT|     when |sm_GT|  >= 1
      - overall_r30 = (top30 ∩ GT) / |GT|

Decision:
    If production bio_r30 (mean over 5 seeds) is within 3pp of overall_r30,
    the biologic-failure premise no longer binds and h906/h921/h924 should
    be marked INCONCLUSIVE-no-longer-motivated. If the gap is >=10pp, the
    motivation still holds.
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

from production_predictor import DrugRepurposingPredictor  # noqa: E402

# h393 machinery (split + recompute_gt_structures + restore_gt_structures)
sys.path.insert(0, str(ROOT / "scripts"))
from h393_holdout_tier_validation import (  # noqa: E402
    split_diseases,
    recompute_gt_structures,
    restore_gt_structures,
)

# h939/h940 biologic proxy
from h939_biologic_target_overlap_audit import is_biologic  # noqa: E402


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


def evaluate_seed(
    predictor: DrugRepurposingPredictor,
    expanded_gt: Dict[str, Set[str]],
    biologic_pool: Set[str],
    holdout_ids: List[str],
    top_n: int = 30,
) -> Dict:
    """Run production predict() on every holdout disease; collect per-disease
    bio_r30, sm_r30, overall_r30, and category."""
    per_disease: List[Dict] = []
    n_skipped_no_id = 0
    n_skipped_no_gt = 0
    n_skipped_predict_err = 0

    for dis_id in holdout_ids:
        gt_drugs = expanded_gt.get(dis_id, set())
        if not gt_drugs:
            n_skipped_no_gt += 1
            continue
        dis_name = predictor.disease_names.get(dis_id)
        if not dis_name:
            n_skipped_no_id += 1
            continue
        try:
            result = predictor.predict(dis_name, top_n=top_n, include_filtered=True)
        except Exception:
            n_skipped_predict_err += 1
            continue

        top_drug_ids = [p.drug_id for p in result.predictions[:top_n]]
        top_set = set(top_drug_ids)
        bio_gt = gt_drugs & biologic_pool
        sm_gt = gt_drugs - biologic_pool

        overall_r30 = len(top_set & gt_drugs) / len(gt_drugs)
        bio_r30 = (
            len(top_set & bio_gt) / len(bio_gt) if bio_gt else None
        )
        sm_r30 = (
            len(top_set & sm_gt) / len(sm_gt) if sm_gt else None
        )

        # biologic precision among biologics in top-30
        bio_in_top = top_set & biologic_pool
        bio_p30 = (
            len(top_set & bio_gt) / len(bio_in_top) if bio_in_top else None
        )

        category = predictor.categorize_disease(dis_name)
        per_disease.append({
            "disease_id": dis_id,
            "disease_name": dis_name,
            "category": category,
            "gt_size": len(gt_drugs),
            "bio_gt_size": len(bio_gt),
            "sm_gt_size": len(sm_gt),
            "bio_in_top30": len(bio_in_top),
            "overall_r30": overall_r30,
            "bio_r30": bio_r30,
            "sm_r30": sm_r30,
            "bio_p30": bio_p30,
        })

    return {
        "per_disease": per_disease,
        "n_skipped_no_id": n_skipped_no_id,
        "n_skipped_no_gt": n_skipped_no_gt,
        "n_skipped_predict_err": n_skipped_predict_err,
    }


def aggregate_seed(per_disease: List[Dict]) -> Dict:
    overall = [r["overall_r30"] for r in per_disease]
    bio = [r["bio_r30"] for r in per_disease if r["bio_r30"] is not None]
    sm = [r["sm_r30"] for r in per_disease if r["sm_r30"] is not None]
    bio_p = [r["bio_p30"] for r in per_disease if r["bio_p30"] is not None]

    return {
        "n_diseases": len(per_disease),
        "n_with_bio_gt": len(bio),
        "n_with_sm_gt": len(sm),
        "overall_r30_mean": float(np.mean(overall)) if overall else 0.0,
        "bio_r30_mean": float(np.mean(bio)) if bio else 0.0,
        "sm_r30_mean": float(np.mean(sm)) if sm else 0.0,
        "bio_p30_mean": float(np.mean(bio_p)) if bio_p else 0.0,
    }


def category_strata(per_disease: List[Dict]) -> Dict[str, Dict]:
    by_cat: Dict[str, List[Dict]] = defaultdict(list)
    for r in per_disease:
        by_cat[r["category"]].append(r)
    out: Dict[str, Dict] = {}
    for cat, recs in by_cat.items():
        bio = [r["bio_r30"] for r in recs if r["bio_r30"] is not None]
        sm = [r["sm_r30"] for r in recs if r["sm_r30"] is not None]
        if not bio:
            continue
        out[cat] = {
            "n_diseases": len(recs),
            "n_with_bio_gt": len(bio),
            "n_with_sm_gt": len(sm),
            "bio_r30_mean": float(np.mean(bio)),
            "sm_r30_mean": float(np.mean(sm)) if sm else 0.0,
        }
    return out


def main():
    seeds = [42, 123, 456, 789, 2024]
    print("=" * 78)
    print("h951: Biologic R@30 reality-check — CURRENT production pipeline (5 seeds)")
    print("=" * 78)

    print("Loading predictor ...")
    predictor = DrugRepurposingPredictor()

    expanded_gt = load_expanded_gt(
        predictor.reference_dir / "expanded_ground_truth.json"
    )
    print(f"Expanded GT: {len(expanded_gt)} diseases, "
          f"{sum(len(v) for v in expanded_gt.values())} pairs")

    # Biologic pool — same proxy used in h939/h940 (266 of 11,656 drugs)
    biologic_pool = {
        d for d in predictor.drug_targets
        if is_biologic(predictor.drug_id_to_name.get(d))
    }
    print(f"Biologic pool size: {len(biologic_pool)} of "
          f"{len(predictor.drug_targets)} drugs with targets")

    # All evaluable diseases
    all_diseases = [
        d for d in predictor.ground_truth if d in predictor.embeddings
    ]
    print(f"Evaluable diseases (GT ∩ embeddings): {len(all_diseases)}")

    seed_results = []
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        train_set = set(train_ids)
        print(f"Train: {len(train_ids)}, Holdout: {len(holdout_ids)}")

        originals = recompute_gt_structures(predictor, train_set)
        try:
            ev = evaluate_seed(predictor, expanded_gt, biologic_pool, holdout_ids)
        finally:
            restore_gt_structures(predictor, originals)

        per_disease = ev["per_disease"]
        agg = aggregate_seed(per_disease)
        cat_strata = category_strata(per_disease)

        print(f"  evaluated={agg['n_diseases']}  "
              f"bio_gt_present={agg['n_with_bio_gt']}  "
              f"sm_gt_present={agg['n_with_sm_gt']}  "
              f"skip_no_id={ev['n_skipped_no_id']}  "
              f"skip_no_gt={ev['n_skipped_no_gt']}  "
              f"skip_err={ev['n_skipped_predict_err']}")
        print(f"  overall_r30={100*agg['overall_r30_mean']:5.2f}%  "
              f"bio_r30={100*agg['bio_r30_mean']:5.2f}%  "
              f"sm_r30={100*agg['sm_r30_mean']:5.2f}%  "
              f"bio_p30={100*agg['bio_p30_mean']:5.2f}%")
        seed_results.append({
            "seed": seed,
            "agg": agg,
            "cat_strata": cat_strata,
        })

    # Aggregate across seeds
    print("\n" + "=" * 78)
    print("AGGREGATE (mean ± std across 5 seeds)")
    print("=" * 78)
    metric_lists: Dict[str, List[float]] = defaultdict(list)
    for r in seed_results:
        for k, v in r["agg"].items():
            if k.endswith("_mean"):
                metric_lists[k].append(v)

    aggregate = {}
    for k, vs in metric_lists.items():
        mean = float(np.mean(vs))
        std = float(np.std(vs))
        aggregate[k] = {"mean": mean, "std": std}
        print(f"  {k:20s} = {100*mean:6.2f}% ± {100*std:5.2f}%")

    # Per-category aggregate
    print("\n" + "=" * 78)
    print("PER-CATEGORY (mean across seeds, n_with_bio_gt averaged)")
    print("=" * 78)
    cat_keys: Set[str] = set()
    for r in seed_results:
        cat_keys.update(r["cat_strata"].keys())

    cat_aggregate = {}
    print(f"{'category':<20s}  {'n_bio_gt':>9s}  {'bio_r30':>10s}  "
          f"{'sm_r30':>10s}  {'ratio':>6s}")
    for cat in sorted(cat_keys):
        bios, sms, n_bios = [], [], []
        for r in seed_results:
            if cat in r["cat_strata"]:
                cs = r["cat_strata"][cat]
                bios.append(cs["bio_r30_mean"])
                sms.append(cs["sm_r30_mean"])
                n_bios.append(cs["n_with_bio_gt"])
        if not bios:
            continue
        bio_m = float(np.mean(bios))
        sm_m = float(np.mean(sms)) if sms else 0.0
        ratio = bio_m / sm_m if sm_m > 0 else float("inf")
        n_bio_avg = float(np.mean(n_bios))
        cat_aggregate[cat] = {
            "n_bio_gt_avg": n_bio_avg,
            "bio_r30_mean": bio_m,
            "sm_r30_mean": sm_m,
            "ratio": ratio,
        }
        print(f"  {cat:<18s}  {n_bio_avg:9.1f}  "
              f"{100*bio_m:9.2f}%  {100*sm_m:9.2f}%  {ratio:6.2f}")

    # Decision
    print("\n" + "=" * 78)
    print("DECISION (h906/h921/h924 motivation)")
    print("=" * 78)
    bio_overall = aggregate["bio_r30_mean"]["mean"]
    overall = aggregate["overall_r30_mean"]["mean"]
    sm_overall = aggregate["sm_r30_mean"]["mean"]
    delta_bio_vs_overall = 100 * (overall - bio_overall)
    delta_bio_vs_sm = 100 * (sm_overall - bio_overall)

    print(f"  production bio_r30 (5 seeds)        = {100*bio_overall:5.2f}%")
    print(f"  production overall_r30 (5 seeds)    = {100*overall:5.2f}%")
    print(f"  production sm_r30 (5 seeds)         = {100*sm_overall:5.2f}%")
    print(f"  delta overall - bio                 = {delta_bio_vs_overall:+5.2f}pp")
    print(f"  delta sm - bio                      = {delta_bio_vs_sm:+5.2f}pp")
    print()
    print(f"  historical baseline (research_spec) = 27.3% biologic R@30")
    print(f"  h940 plain-kNN baseline (5 seeds)   = 31.42% biologic R@30")
    print(f"  delta production - 27.3             = {100*bio_overall - 27.3:+5.2f}pp")
    print(f"  delta production - 31.42            = {100*bio_overall - 31.42:+5.2f}pp")
    print()
    if delta_bio_vs_overall <= 3.0 and delta_bio_vs_sm <= 3.0:
        verdict = "MOTIVATION DISSOLVED — biologics ≈ overall on production pipeline"
    elif delta_bio_vs_overall >= 10.0:
        verdict = "MOTIVATION HOLDS — biologics still substantially below overall"
    else:
        verdict = "MOTIVATION WEAKENED — biologic gap exists but smaller than historical"
    print(f"  VERDICT: {verdict}")

    out = {
        "hypothesis": "h951",
        "title": "Biologic failure-class reality check on production pipeline",
        "seeds": seeds,
        "biologic_pool_size": len(biologic_pool),
        "evaluable_diseases": len(all_diseases),
        "seed_results": seed_results,
        "aggregate": aggregate,
        "cat_aggregate": cat_aggregate,
        "delta_overall_minus_bio_pp": delta_bio_vs_overall,
        "delta_sm_minus_bio_pp": delta_bio_vs_sm,
        "historical_baseline": 27.3,
        "h940_baseline": 31.42,
        "verdict": verdict,
    }
    out_path = ROOT / "data/analysis/h951_biologic_baseline_check.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
