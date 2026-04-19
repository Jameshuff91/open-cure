#!/usr/bin/env python3
"""h995b: De-biased re-analysis of h995 slot records.

h995 showed ratio=inf (hr_miss=0%) for target_match across every category.
That is the signature of a tautology: if candidate ∈ bio_gt, its targets
are trivially ⊂ gt_target_union (self-inclusion), so target_match is
guaranteed True for any in-GT biologic with annotated targets.

This script re-computes target_match with a leave-one-out reference set:
    gt_target_union_LOO = ⋃ drug_targets[g] for g ∈ bio_gt \\ {candidate}

If the de-biased hit-rate ratio collapses (match ≈ miss), target_match is
self-inclusion only — NO cross-drug signal — and the filter is illusory.
If it stays ≥3x, the family signal is genuine.

Uses the per-slot records written by h995 so we don't rerun the predictor.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from production_predictor import DrugRepurposingPredictor  # noqa: E402
from h939_biologic_target_overlap_audit import is_biologic  # noqa: E402
from h995_autoimmune_biologic_family_audit import (  # noqa: E402
    load_expanded_gt,
    usan_suffix,
    usan_substem,
)


def main():
    print("=" * 78)
    print("h995b: De-biased target/suffix/substem match (leave-one-out)")
    print("=" * 78)

    records_path = ROOT / "data/analysis/h995_slot_records.json"
    with open(records_path) as f:
        slot_records = json.load(f)
    print(f"Loaded {len(slot_records)} slot records from h995")

    # Re-establish predictor-side lookups.
    predictor = DrugRepurposingPredictor()
    expanded_gt = load_expanded_gt(
        predictor.reference_dir / "expanded_ground_truth.json"
    )
    biologic_pool = {
        d for d in predictor.drug_id_to_name
        if is_biologic(predictor.drug_id_to_name.get(d))
    }

    # Per disease, cache bio_gt + per-drug reference sets
    # key: (seed, disease_id) -> {bio_gt_ids}
    # (seed is in slot_records; bio_gt does not depend on seed since expanded_gt is fixed)
    # But we compute leave-one-out per candidate, so we only need bio_gt per disease.
    bio_gt_by_disease = {}
    for r in slot_records:
        dis = r["disease_id"]
        if dis not in bio_gt_by_disease:
            gt = expanded_gt.get(dis, set())
            bio_gt_by_disease[dis] = gt & biologic_pool

    # Compute de-biased matches per slot record
    results = []
    for r in slot_records:
        dis = r["disease_id"]
        bio_gt = bio_gt_by_disease[dis]
        cand = r["drug_id"]

        # Leave-one-out reference sets (exclude candidate itself)
        ref_drugs = bio_gt - {cand}
        ref_suffixes = set()
        ref_substems = set()
        ref_target_union = set()
        for g in ref_drugs:
            nm = predictor.drug_id_to_name.get(g)
            suf = usan_suffix(nm)
            if suf:
                ref_suffixes.add(suf)
            sub = usan_substem(nm)
            if sub:
                ref_substems.add(sub)
            tgts = predictor.drug_targets.get(g)
            if tgts:
                ref_target_union |= tgts

        cand_suf = r["suffix"]
        cand_sub = r["substem"]
        cand_tgts = predictor.drug_targets.get(cand, set())

        suffix_match_loo = bool(cand_suf and cand_suf in ref_suffixes)
        substem_match_loo = bool(cand_sub and cand_sub in ref_substems)
        target_match_loo = bool(
            cand_tgts and ref_target_union and
            (cand_tgts & ref_target_union)
        )

        results.append({
            **r,
            "suffix_match_loo": suffix_match_loo,
            "substem_match_loo": substem_match_loo,
            "target_match_loo": target_match_loo,
            "bio_gt_size_eff": len(bio_gt),
            "ref_drugs_n": len(ref_drugs),
        })

    # Summarize: for each rule, hit rate among match vs miss, both orig and loo
    def summarize(recs, key):
        match = [r for r in recs if r[key]]
        miss = [r for r in recs if not r[key]]
        n_m = len(match)
        n_mi = len(miss)
        h_m = sum(1 for r in match if r["hit"])
        h_mi = sum(1 for r in miss if r["hit"])
        hr_m = h_m / n_m if n_m else 0.0
        hr_mi = h_mi / n_mi if n_mi else 0.0
        ratio = (hr_m / hr_mi) if hr_mi > 0 else None
        return n_m, h_m, hr_m, n_mi, h_mi, hr_mi, ratio

    print("\n" + "=" * 78)
    print("AUTOIMMUNE — de-biased vs original (hit rate by rule)")
    print("=" * 78)
    ai = [r for r in results if r["category"] == "autoimmune"]
    print(f"Autoimmune slots: {len(ai)} pooled 5 seeds, "
          f"hits={sum(1 for r in ai if r['hit'])}")
    for key in ("suffix_match", "substem_match", "target_match"):
        for variant in ("", "_loo"):
            k = key + variant
            n_m, h_m, hr_m, n_mi, h_mi, hr_mi, ratio = summarize(ai, k)
            ratio_s = f"{ratio:.2f}x" if ratio else "inf"
            label = "ORIG" if variant == "" else "LOO "
            print(f"  {label} [{key:<13s}]: match n={n_m:>3d} hits={h_m:>2d} "
                  f"({100*hr_m:5.2f}%)  "
                  f"miss n={n_mi:>3d} hits={h_mi:>2d} ({100*hr_mi:5.2f}%) "
                  f"ratio={ratio_s}")

    # Global per-category table (de-biased target_match)
    print("\n" + "=" * 78)
    print("PER CATEGORY — target_match LOO (pooled 5 seeds)")
    print("=" * 78)
    cats = defaultdict(list)
    for r in results:
        cats[r["category"]].append(r)
    print(f"{'category':<18s} {'slots':>6s} {'match_n':>8s} "
          f"{'hr_match':>9s} {'miss_n':>7s} {'hr_miss':>8s} {'ratio':>7s}")
    for cat in sorted(cats, key=lambda c: len(cats[c]), reverse=True):
        recs = cats[cat]
        n_m, h_m, hr_m, n_mi, h_mi, hr_mi, ratio = summarize(
            recs, "target_match_loo"
        )
        ratio_s = f"{ratio:.2f}x" if ratio else "inf"
        print(f"{cat:<18s} {len(recs):>6d} {n_m:>8d} "
              f"{100*hr_m:>7.2f}% {n_mi:>7d} {100*hr_mi:>6.2f}% {ratio_s:>7s}")

    # Tautology fraction: among orig target_match=True hits, how many
    # flip to target_match_loo=False? That fraction is the self-inclusion load.
    flip = 0
    orig_hits_match = 0
    for r in results:
        if r["hit"] and r["target_match"]:
            orig_hits_match += 1
            if not r["target_match_loo"]:
                flip += 1
    print("\n" + "=" * 78)
    print("TAUTOLOGY DIAGNOSTIC")
    print("=" * 78)
    print(f"Hits with target_match=True (orig): {orig_hits_match}")
    print(f"Of which flip to target_match_loo=False: {flip} "
          f"({100*flip/max(orig_hits_match,1):.1f}%)")
    print("  (Higher flip% ⇒ more of target_match signal was "
          "self-inclusion tautology)")

    # Save
    out_path = ROOT / "data/analysis/h995b_debias_results.json"
    summary = {
        "hypothesis": "h995b",
        "title": "De-biased target/suffix/substem match",
        "n_slots": len(results),
        "autoimmune_n": len(ai),
        "autoimmune_hits": sum(1 for r in ai if r["hit"]),
        "autoimmune_rules": {
            key + variant: dict(zip(
                ("n_match", "hits_match", "hr_match",
                 "n_miss", "hits_miss", "hr_miss", "ratio"),
                summarize(ai, key + variant),
            ))
            for key in ("suffix_match", "substem_match", "target_match")
            for variant in ("", "_loo")
        },
        "per_cat_target_match_loo": {
            cat: dict(zip(
                ("n_match", "hits_match", "hr_match",
                 "n_miss", "hits_miss", "hr_miss", "ratio"),
                summarize(cats[cat], "target_match_loo"),
            ))
            for cat in cats
        },
        "tautology_flip_frac": flip / max(orig_hits_match, 1),
        "tautology_n_flipped": flip,
        "tautology_n_orig_hits_match": orig_hits_match,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
