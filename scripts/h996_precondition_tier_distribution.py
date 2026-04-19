#!/usr/bin/env python3
"""h996 precondition: what tier do SM f<=2 no-mech top-30 slots currently
occupy? If they are already tier-capped at LOW/FILTER, the proposed new rule
is a no-op and h996 should be invalidated before implementation.

Method: 5-seed h393 holdout. For each holdout disease, fetch top-30 with
include_filtered=True. For each slot, record (is_bio, train_frequency,
mechanism_support, confidence_tier, hit). Cross-tab tier x (f<=2, no_mech)
for SM slots.

Run: python3 scripts/h996_precondition_tier_distribution.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set

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


def main():
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
    print(f"Biologic pool: {len(biologic_pool)}\n")

    # Cross-tab: tier x (is_target_rule, hit)
    # target_rule = SM AND train_frequency<=2 AND NOT mechanism_support
    by_tier_target: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"n": 0, "hits": 0}
    )
    by_tier_nontarget: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"n": 0, "hits": 0}
    )
    by_tier_all: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"n": 0, "hits": 0}
    )

    for seed in SEEDS:
        print(f"--- Seed {seed} ---")
        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        train_set = set(train_ids)
        originals = recompute_gt_structures(predictor, train_set)
        try:
            n_dx = 0
            for dis_id in holdout_ids:
                gt = expanded_gt.get(dis_id, set())
                if not gt:
                    continue
                try:
                    result = predictor.predict(
                        dis_id, top_n=TOP_N, include_filtered=True
                    )
                except Exception:
                    continue
                if not result.predictions:
                    continue
                n_dx += 1
                for p in result.predictions[:TOP_N]:
                    is_bio = p.drug_id in biologic_pool
                    hit = p.drug_id in gt
                    tier = p.confidence_tier.value if hasattr(
                        p.confidence_tier, "value"
                    ) else str(p.confidence_tier)
                    is_target = (
                        (not is_bio)
                        and p.train_frequency <= 2
                        and (not p.mechanism_support)
                    )
                    bucket = by_tier_target if is_target else by_tier_nontarget
                    bucket[tier]["n"] += 1
                    if hit:
                        bucket[tier]["hits"] += 1
                    by_tier_all[tier]["n"] += 1
                    if hit:
                        by_tier_all[tier]["hits"] += 1
        finally:
            restore_gt_structures(predictor, originals)
        print(f"  {n_dx} diseases evaluated")

    # Print results
    print("\n" + "=" * 78)
    print("TIER DISTRIBUTION: SM AND f<=2 AND no_mech  (pooled 5 seeds)")
    print("=" * 78)
    tiers = ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]
    print(f"{'tier':<8s} {'target_n':>10s} {'target_hits':>12s} {'target_hr':>10s} "
          f"{'other_n':>10s} {'other_hr':>10s} {'all_n':>10s} {'all_hr':>10s}")
    for t in tiers:
        tn = by_tier_target.get(t, {"n": 0, "hits": 0})
        on = by_tier_nontarget.get(t, {"n": 0, "hits": 0})
        an = by_tier_all.get(t, {"n": 0, "hits": 0})
        thr = tn["hits"] / tn["n"] if tn["n"] else 0
        ohr = on["hits"] / on["n"] if on["n"] else 0
        ahr = an["hits"] / an["n"] if an["n"] else 0
        print(f"{t:<8s} {tn['n']:>10d} {tn['hits']:>12d} {100*thr:>8.2f}% "
              f"{on['n']:>10d} {100*ohr:>8.2f}% "
              f"{an['n']:>10d} {100*ahr:>8.2f}%")

    tot_target = sum(v["n"] for v in by_tier_target.values())
    tot_target_above_low = sum(
        by_tier_target.get(t, {"n": 0})["n"]
        for t in ("GOLDEN", "HIGH", "MEDIUM")
    )
    print(f"\nTotal SM f<=2 no-mech slots: {tot_target}")
    print(f"  Already at LOW or FILTER: "
          f"{tot_target - tot_target_above_low} "
          f"({100*(tot_target - tot_target_above_low)/tot_target:.1f}%)")
    print(f"  At MEDIUM/HIGH/GOLDEN (rule would touch): "
          f"{tot_target_above_low} "
          f"({100*tot_target_above_low/tot_target:.1f}%)")

    # Decision
    if tot_target_above_low < 500:
        verdict = "LIKELY NO-OP \u2014 few slots would be demoted"
    elif tot_target_above_low / tot_target < 0.10:
        verdict = "LIKELY NO-OP \u2014 <10% of target slots at MEDIUM+"
    else:
        verdict = "PROCEED \u2014 meaningful volume would be demoted"
    print(f"\nPRECONDITION VERDICT: {verdict}")

    out = {
        "hypothesis": "h996_precondition",
        "by_tier_target": dict(by_tier_target),
        "by_tier_nontarget": dict(by_tier_nontarget),
        "by_tier_all": dict(by_tier_all),
        "total_target": tot_target,
        "target_at_medium_plus": tot_target_above_low,
        "verdict": verdict,
    }
    out_path = ROOT / "data/analysis/h996_precondition_tier_distribution.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
