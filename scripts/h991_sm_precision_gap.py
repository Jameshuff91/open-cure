#!/usr/bin/env python3
"""h991: Inverse of h953 — small-molecule precision gap.

h953 baseline: bio_r30 = 30.31% ± 3.57% > overall_r30 = 19.79% ± 1.59% (+10.5pp).
Biologics OVERPERFORM. Therefore any precision lift must come from
SMALL-MOLECULE slots, not biologic slots. We stratify the 14.32% baseline p30
by drug type and compute:

    sm_p30_amongsm  = hits among SM top-30 entries / count of SM in top-30
    bio_p30_amongbio = hits among bio top-30 entries / count of bio in top-30

If sm_p30_amongsm < bio_p30_amongbio by >=5pp in a category with n>=10
diseases, that category is a candidate for a targeted SM demote rule. Also
capture train_frequency and mechanism_support per prediction so the rule can
be specified precisely (e.g. "drop SM picks with freq<3 and no mechanism").

Design (5-seed h393 holdout, mirror h953 plumbing):
  1. Recompute GT structures from train-only diseases (h393-style).
  2. For every holdout disease, fetch top_n=200 with include_filtered=True.
  3. For each of the top-30 predictions, record:
        drug_id, is_bio, hit(=in gt), rank, train_frequency, mechanism_support
  4. Aggregate per category and per (category, is_bio).
  5. Report SM-vs-bio precision gap per category + overall.

Run: python3 scripts/h991_sm_precision_gap.py
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


def evaluate_seed(
    predictor: DrugRepurposingPredictor,
    expanded_gt: Dict[str, Set[str]],
    biologic_pool: Set[str],
    holdout_ids: List[str],
) -> List[Dict]:
    records: List[Dict] = []

    for dis_id in holdout_ids:
        gt = expanded_gt.get(dis_id, set())
        if not gt:
            continue

        try:
            result = predictor.predict(
                dis_id, top_n=TOP_N_FETCH, include_filtered=True
            )
        except Exception:
            continue

        preds = result.predictions
        if not preds:
            continue

        name = predictor.disease_names.get(dis_id, dis_id)
        cat = predictor.categorize_disease(name)
        bio_gt = gt & biologic_pool
        sm_gt = gt - biologic_pool

        # Per-prediction records for top-30
        top30 = preds[:TOP_N_REPORT]
        slot_records = []
        for p in top30:
            is_bio = p.drug_id in biologic_pool
            hit = p.drug_id in gt
            slot_records.append({
                "drug_id": p.drug_id,
                "is_bio": is_bio,
                "hit": hit,
                "rank": p.rank,
                "train_frequency": p.train_frequency,
                "mechanism_support": p.mechanism_support,
            })

        # Scalar stats
        n_bio = sum(1 for s in slot_records if s["is_bio"])
        n_sm = TOP_N_REPORT - n_bio
        bio_hits = sum(1 for s in slot_records if s["is_bio"] and s["hit"])
        sm_hits = sum(1 for s in slot_records if (not s["is_bio"]) and s["hit"])
        total_hits = bio_hits + sm_hits

        records.append({
            "disease_id": dis_id,
            "category": cat,
            "gt_size": len(gt),
            "bio_gt_size": len(bio_gt),
            "sm_gt_size": len(sm_gt),
            "n_bio_in_top30": n_bio,
            "n_sm_in_top30": n_sm,
            "bio_hits_in_top30": bio_hits,
            "sm_hits_in_top30": sm_hits,
            "total_hits_top30": total_hits,
            "p30": total_hits / TOP_N_REPORT,
            "sm_p30_amongsm": (sm_hits / n_sm) if n_sm > 0 else None,
            "bio_p30_amongbio": (bio_hits / n_bio) if n_bio > 0 else None,
            "sm_p30_abs": sm_hits / TOP_N_REPORT,
            "bio_p30_abs": bio_hits / TOP_N_REPORT,
            "slot_records": slot_records,
        })

    return records


def aggregate_per_category(records: List[Dict]) -> Dict[str, Dict]:
    by_cat: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r)

    out: Dict[str, Dict] = {}
    for cat, recs in by_cat.items():
        sm_vals = [r["sm_p30_amongsm"] for r in recs
                   if r["sm_p30_amongsm"] is not None]
        bio_vals = [r["bio_p30_amongbio"] for r in recs
                    if r["bio_p30_amongbio"] is not None]
        out[cat] = {
            "n_diseases": len(recs),
            "n_with_sm": len(sm_vals),
            "n_with_bio": len(bio_vals),
            "p30_mean": float(np.mean([r["p30"] for r in recs])),
            "sm_p30_amongsm_mean": float(np.mean(sm_vals)) if sm_vals else None,
            "bio_p30_amongbio_mean": float(np.mean(bio_vals)) if bio_vals else None,
            "sm_p30_abs_mean": float(np.mean([r["sm_p30_abs"] for r in recs])),
            "bio_p30_abs_mean": float(np.mean([r["bio_p30_abs"] for r in recs])),
            "avg_n_bio_in_top30": float(np.mean([r["n_bio_in_top30"] for r in recs])),
            "avg_n_sm_in_top30": float(np.mean([r["n_sm_in_top30"] for r in recs])),
        }
    return out


def miss_profile(records: List[Dict], is_bio_filter: bool) -> Dict:
    """Distribution of train_frequency & mechanism_support among SM (or bio)
    top-30 slots, stratified by hit/miss."""
    out = {"hit": defaultdict(int), "miss": defaultdict(int)}
    freq_hit: List[int] = []
    freq_miss: List[int] = []
    mech_hit = 0
    mech_miss = 0
    n_hit = 0
    n_miss = 0

    for r in records:
        for s in r["slot_records"]:
            if s["is_bio"] != is_bio_filter:
                continue
            bucket = "hit" if s["hit"] else "miss"
            tf = s["train_frequency"]
            out[bucket][f"freq_{min(tf, 10)}"] += 1
            out[bucket][f"mech_{s['mechanism_support']}"] += 1
            if s["hit"]:
                freq_hit.append(tf)
                if s["mechanism_support"]:
                    mech_hit += 1
                n_hit += 1
            else:
                freq_miss.append(tf)
                if s["mechanism_support"]:
                    mech_miss += 1
                n_miss += 1

    return {
        "n_hit": n_hit,
        "n_miss": n_miss,
        "hit_rate": n_hit / (n_hit + n_miss) if (n_hit + n_miss) else 0.0,
        "freq_hit_mean": float(np.mean(freq_hit)) if freq_hit else 0.0,
        "freq_miss_mean": float(np.mean(freq_miss)) if freq_miss else 0.0,
        "freq_hit_median": float(np.median(freq_hit)) if freq_hit else 0.0,
        "freq_miss_median": float(np.median(freq_miss)) if freq_miss else 0.0,
        "mech_hit_rate": mech_hit / n_hit if n_hit else 0.0,
        "mech_miss_rate": mech_miss / n_miss if n_miss else 0.0,
    }


def main():
    print("=" * 78)
    print("h991: SM vs bio precision gap — diagnostic")
    print("=" * 78)

    predictor = DrugRepurposingPredictor()
    expanded_gt = load_expanded_gt(
        predictor.reference_dir / "expanded_ground_truth.json"
    )
    print(f"Expanded GT: {len(expanded_gt)} diseases")

    biologic_pool = {
        d for d in predictor.drug_targets
        if is_biologic(predictor.drug_id_to_name.get(d))
    }
    print(f"Biologic pool: {len(biologic_pool)} of "
          f"{len(predictor.drug_targets)} drugs with targets")

    all_diseases = [
        d for d in predictor.ground_truth if d in predictor.embeddings
    ]
    print(f"Evaluable diseases: {len(all_diseases)}\n")

    all_records: List[Dict] = []
    per_seed_agg: List[Dict] = []

    for seed in SEEDS:
        print(f"--- Seed {seed} ---")
        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        train_set = set(train_ids)
        originals = recompute_gt_structures(predictor, train_set)
        try:
            recs = evaluate_seed(
                predictor, expanded_gt, biologic_pool, holdout_ids
            )
        finally:
            restore_gt_structures(predictor, originals)

        all_records.extend({**r, "seed": seed} for r in recs)

        # Per-seed scalars
        p30 = np.mean([r["p30"] for r in recs])
        sm_vals = [r["sm_p30_amongsm"] for r in recs if r["sm_p30_amongsm"] is not None]
        bio_vals = [r["bio_p30_amongbio"] for r in recs if r["bio_p30_amongbio"] is not None]
        per_seed_agg.append({
            "seed": seed,
            "n_diseases": len(recs),
            "p30_mean": float(p30),
            "sm_p30_amongsm_mean": float(np.mean(sm_vals)) if sm_vals else 0.0,
            "bio_p30_amongbio_mean": float(np.mean(bio_vals)) if bio_vals else 0.0,
            "n_with_sm_top30": len(sm_vals),
            "n_with_bio_top30": len(bio_vals),
            "avg_n_bio_in_top30": float(np.mean([r["n_bio_in_top30"] for r in recs])),
        })
        a = per_seed_agg[-1]
        print(f"  n={a['n_diseases']} p30={100*a['p30_mean']:.2f}%  "
              f"sm_p30|sm={100*a['sm_p30_amongsm_mean']:.2f}%  "
              f"bio_p30|bio={100*a['bio_p30_amongbio_mean']:.2f}%  "
              f"avg_bio_slots={a['avg_n_bio_in_top30']:.2f}  "
              f"n_dx_w/bio_slot={a['n_with_bio_top30']}")

    # Cross-seed
    print("\n" + "=" * 78)
    print("CROSS-SEED MEAN (5 seeds)")
    print("=" * 78)
    p30_m = np.mean([a["p30_mean"] for a in per_seed_agg])
    sm_m = np.mean([a["sm_p30_amongsm_mean"] for a in per_seed_agg])
    bio_m = np.mean([a["bio_p30_amongbio_mean"] for a in per_seed_agg])
    bio_slots_m = np.mean([a["avg_n_bio_in_top30"] for a in per_seed_agg])
    print(f"  p30         = {100*p30_m:.2f}%")
    print(f"  sm_p30|sm   = {100*sm_m:.2f}%")
    print(f"  bio_p30|bio = {100*bio_m:.2f}%")
    print(f"  gap (bio-sm)= {100*(bio_m-sm_m):+.2f}pp")
    print(f"  avg bio slots in top-30 = {bio_slots_m:.2f}")

    # Per-category, pooled across seeds
    print("\n" + "=" * 78)
    print("PER CATEGORY (pooled 5 seeds)")
    print("=" * 78)
    cat_agg = aggregate_per_category(all_records)

    print(f"{'category':<18s} {'n':>4s} {'n_sm':>5s} {'n_bio':>5s} "
          f"{'p30':>7s} {'sm|sm':>7s} {'bio|bio':>8s} {'gap':>7s} "
          f"{'sm_abs':>7s} {'bio_abs':>8s} {'avg_bio':>8s}")

    cat_flagged: List[str] = []
    for cat in sorted(cat_agg, key=lambda c: cat_agg[c]["n_diseases"], reverse=True):
        v = cat_agg[cat]
        n = v["n_diseases"]
        sm_v = v["sm_p30_amongsm_mean"]
        bio_v = v["bio_p30_amongbio_mean"]
        gap_pp = (100 * (bio_v - sm_v)) if (sm_v is not None and bio_v is not None) else None
        sm_s = f"{100*sm_v:.2f}%" if sm_v is not None else "  n/a "
        bio_s = f"{100*bio_v:.2f}%" if bio_v is not None else "   n/a  "
        gap_s = f"{gap_pp:+.2f}pp" if gap_pp is not None else "   n/a "
        print(f"{cat:<18s} {n:>4d} {v['n_with_sm']:>5d} {v['n_with_bio']:>5d} "
              f"{100*v['p30_mean']:>6.2f}% {sm_s:>7s} {bio_s:>8s} {gap_s:>7s} "
              f"{100*v['sm_p30_abs_mean']:>6.2f}% {100*v['bio_p30_abs_mean']:>7.2f}% "
              f"{v['avg_n_bio_in_top30']:>8.2f}")
        # Flag if n >= 50 pooled (i.e. ≥10/seed avg) and gap >= 5pp
        if (n >= 50 and gap_pp is not None and gap_pp >= 5.0
                and v["n_with_bio"] >= 10):
            cat_flagged.append(cat)

    print(f"\nFLAGGED (n>=50 pooled, gap>=5pp, n_with_bio>=10): {cat_flagged}")

    # Miss-profile diagnostics for SM top-30
    print("\n" + "=" * 78)
    print("SM MISS-PROFILE (top-30 SM slots pooled across seeds)")
    print("=" * 78)
    sm_profile = miss_profile(all_records, is_bio_filter=False)
    bio_profile = miss_profile(all_records, is_bio_filter=True)
    for label, p in [("sm", sm_profile), ("bio", bio_profile)]:
        print(f"  {label}: n_hit={p['n_hit']} n_miss={p['n_miss']} "
              f"hit_rate={100*p['hit_rate']:.2f}%")
        print(f"    freq: hit_mean={p['freq_hit_mean']:.2f} "
              f"miss_mean={p['freq_miss_mean']:.2f}  "
              f"hit_median={p['freq_hit_median']:.1f} "
              f"miss_median={p['freq_miss_median']:.1f}")
        print(f"    mech: hit_rate={100*p['mech_hit_rate']:.2f}%  "
              f"miss_rate={100*p['mech_miss_rate']:.2f}%")

    # Frequency/mechanism cross-tab for SM slots ONLY
    print("\n" + "=" * 78)
    print("SM TOP-30 SLOTS: hit_rate by (freq_bucket, mech_support)")
    print("=" * 78)
    cells: Dict = defaultdict(lambda: {"n": 0, "hits": 0})
    for r in all_records:
        for s in r["slot_records"]:
            if s["is_bio"]:
                continue
            tf = s["train_frequency"]
            if tf <= 1:
                bucket = "f=1"
            elif tf == 2:
                bucket = "f=2"
            elif tf <= 4:
                bucket = "f=3-4"
            elif tf <= 9:
                bucket = "f=5-9"
            else:
                bucket = "f>=10"
            key = (bucket, bool(s["mechanism_support"]))
            cells[key]["n"] += 1
            if s["hit"]:
                cells[key]["hits"] += 1

    print(f"{'freq':<8s} {'mech':<6s} {'n_slots':>8s} {'hits':>6s} {'hit_rate':>9s}")
    for key in sorted(cells.keys()):
        c = cells[key]
        hr = c["hits"] / c["n"] if c["n"] else 0.0
        print(f"{key[0]:<8s} {str(key[1]):<6s} {c['n']:>8d} {c['hits']:>6d} "
              f"{100*hr:>8.2f}%")

    # Save
    summary = {
        "hypothesis": "h991",
        "title": "SM vs bio precision gap — diagnostic",
        "seeds": SEEDS,
        "top_n_report": TOP_N_REPORT,
        "biologic_pool_size": len(biologic_pool),
        "evaluable_diseases": len(all_diseases),
        "per_seed": per_seed_agg,
        "cross_seed": {
            "p30_mean": float(p30_m),
            "sm_p30_amongsm_mean": float(sm_m),
            "bio_p30_amongbio_mean": float(bio_m),
            "gap_pp_bio_minus_sm": float(100 * (bio_m - sm_m)),
            "avg_bio_slots_in_top30": float(bio_slots_m),
        },
        "per_category": cat_agg,
        "categories_flagged": cat_flagged,
        "sm_miss_profile": sm_profile,
        "bio_miss_profile": bio_profile,
        "sm_freq_mech_crosstab": {
            f"{k[0]}|mech={k[1]}": {
                "n": cells[k]["n"],
                "hits": cells[k]["hits"],
                "hit_rate": cells[k]["hits"] / cells[k]["n"] if cells[k]["n"] else 0.0,
            }
            for k in cells
        },
    }
    out_path = ROOT / "data/analysis/h991_sm_precision_gap.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
