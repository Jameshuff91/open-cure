#!/usr/bin/env python3
"""
h402: Comprehensive rule-by-rule precision audit with holdout validation.

Goal: Identify which rules are genuinely useful and which can be pruned.
For each rule, measure:
1. Full-data precision (n, GT hits)
2. Holdout precision (5-seed mean ± std)
3. Volume (how many predictions it affects)
4. Whether removing it would change tier ordering
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from production_predictor import (
    DrugRepurposingPredictor,
    ConfidenceTier,
    extract_cancer_types,
    DISEASE_HIERARCHY_GROUPS,
)


def split_diseases(all_diseases, seed, train_ratio=0.8):
    rng = np.random.RandomState(seed)
    shuffled = list(all_diseases)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def recompute_gt_structures(predictor, train_disease_ids):
    originals = {
        "drug_train_freq": dict(predictor.drug_train_freq),
        "drug_to_diseases": dict(predictor.drug_to_diseases),
        "drug_cancer_types": dict(predictor.drug_cancer_types),
        "drug_disease_groups": dict(predictor.drug_disease_groups),
        "train_diseases": list(predictor.train_diseases),
        "train_embeddings": predictor.train_embeddings.copy(),
        "train_disease_categories": dict(predictor.train_disease_categories),
    }

    new_freq = defaultdict(int)
    new_d2d = defaultdict(set)
    new_cancer = defaultdict(set)
    new_groups = defaultdict(set)

    for disease_id in train_disease_ids:
        if disease_id not in predictor.ground_truth:
            continue
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        category = predictor.categorize_disease(disease_name)
        cancer_types = extract_cancer_types(disease_name)

        for drug_id in predictor.ground_truth[disease_id]:
            new_freq[drug_id] += 1
            new_d2d[drug_id].add(disease_name.lower())
            if cancer_types:
                new_cancer[drug_id].update(cancer_types)
            if category in DISEASE_HIERARCHY_GROUPS:
                for group_name, keywords in DISEASE_HIERARCHY_GROUPS[category].items():
                    if any(kw in disease_name.lower() for kw in keywords):
                        new_groups[drug_id].add((category, group_name))

    predictor.drug_train_freq = dict(new_freq)
    predictor.drug_to_diseases = dict(new_d2d)
    predictor.drug_cancer_types = dict(new_cancer)
    predictor.drug_disease_groups = dict(new_groups)

    predictor.train_diseases = [d for d in train_disease_ids if d in predictor.embeddings]
    predictor.train_embeddings = np.array(
        [predictor.embeddings[d] for d in predictor.train_diseases], dtype=np.float32
    )
    predictor.train_disease_categories = {}
    for d in predictor.train_diseases:
        name = predictor.disease_names.get(d, d)
        predictor.train_disease_categories[d] = predictor.categorize_disease(name)

    return originals


def restore_gt_structures(predictor, originals):
    for key, val in originals.items():
        setattr(predictor, key, val)


def evaluate_rules(predictor, disease_ids, gt_data):
    """Get per-rule precision and tier-level precision."""
    rule_counts = defaultdict(lambda: {"total": 0, "gt": 0, "tier": None})
    tier_counts = defaultdict(lambda: {"total": 0, "gt": 0})

    for disease_id in disease_ids:
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        gt_drugs = set()
        if disease_id in gt_data:
            for drug in gt_data[disease_id]:
                if isinstance(drug, str):
                    gt_drugs.add(drug)
                elif isinstance(drug, dict):
                    gt_drugs.add(drug.get('drug_id') or drug.get('drug'))

        if not gt_drugs:
            continue

        try:
            result = predictor.predict(disease_name, top_n=30, include_filtered=True)
        except Exception:
            continue

        for pred in result.predictions:
            tier = pred.confidence_tier.name
            rule = pred.category_specific_tier or 'standard'
            is_gt = pred.drug_id in gt_drugs

            rule_counts[rule]["total"] += 1
            rule_counts[rule]["gt"] += int(is_gt)
            rule_counts[rule]["tier"] = tier  # Last seen tier (should be consistent)

            tier_counts[tier]["total"] += 1
            tier_counts[tier]["gt"] += int(is_gt)

    return dict(rule_counts), dict(tier_counts)


def main():
    print("Loading predictor...")
    predictor = DrugRepurposingPredictor()

    gt_path = predictor.reference_dir / "expanded_ground_truth.json"
    with open(gt_path) as f:
        gt_data = json.load(f)

    all_diseases = [
        d for d in predictor.disease_names.keys()
        if d in predictor.ground_truth and d in predictor.embeddings
    ]
    print(f"Total diseases: {len(all_diseases)}")

    # === Phase 1: Full-data rule precision ===
    print("\n=== PHASE 1: Full-data rule precision ===")
    rule_counts, tier_counts = evaluate_rules(predictor, all_diseases, gt_data)

    # Organize rules by tier
    rules_by_tier = defaultdict(list)
    for rule, counts in rule_counts.items():
        tier = counts["tier"]
        prec = counts["gt"] / counts["total"] * 100 if counts["total"] > 0 else 0
        rules_by_tier[tier].append({
            "rule": rule,
            "precision": prec,
            "gt": counts["gt"],
            "total": counts["total"],
        })

    tier_order = ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]
    print(f"\nTier precision (full data):")
    for tier in tier_order:
        tc = tier_counts.get(tier, {"total": 0, "gt": 0})
        prec = tc["gt"] / tc["total"] * 100 if tc["total"] > 0 else 0
        print(f"  {tier}: {tc['gt']}/{tc['total']} = {prec:.1f}%")

    print(f"\nRules by tier (sorted by precision, full data):")
    for tier in tier_order:
        rules = sorted(rules_by_tier.get(tier, []), key=lambda x: -x["precision"])
        print(f"\n  --- {tier} ---")
        for r in rules:
            flag = " <<<" if (
                (tier == "GOLDEN" and r["precision"] < 40) or
                (tier == "HIGH" and r["precision"] < 30) or
                (tier == "MEDIUM" and r["precision"] < 15) or
                (tier == "LOW" and r["precision"] < 5) or
                (tier == "FILTER" and r["precision"] > 20)
            ) else ""
            print(f"    {r['rule']:50s} {r['gt']:4d}/{r['total']:4d} = {r['precision']:5.1f}%{flag}")

    # === Phase 2: Holdout validation (5-seed) ===
    print(f"\n{'='*80}")
    print("=== PHASE 2: Holdout Validation (5-seed) ===")
    print(f"{'='*80}")

    seeds = [42, 123, 456, 789, 2024]
    holdout_rule_precisions = defaultdict(list)  # rule -> list of precision values
    holdout_tier_precisions = defaultdict(list)

    for seed in seeds:
        print(f"\n  Processing seed {seed}...")
        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        originals = recompute_gt_structures(predictor, set(train_ids))

        rule_counts_h, tier_counts_h = evaluate_rules(predictor, holdout_ids, gt_data)

        for rule, counts in rule_counts_h.items():
            prec = counts["gt"] / counts["total"] * 100 if counts["total"] > 0 else 0
            holdout_rule_precisions[rule].append((prec, counts["total"]))

        for tier in tier_order:
            tc = tier_counts_h.get(tier, {"total": 0, "gt": 0})
            prec = tc["gt"] / tc["total"] * 100 if tc["total"] > 0 else 0
            holdout_tier_precisions[tier].append(prec)

        restore_gt_structures(predictor, originals)

    # === Summary ===
    print(f"\n{'='*80}")
    print("=== HOLDOUT SUMMARY ===")
    print(f"{'='*80}")

    print(f"\nTier precision (holdout, 5-seed):")
    for tier in tier_order:
        vals = holdout_tier_precisions[tier]
        print(f"  {tier}: {np.mean(vals):.1f}% ± {np.std(vals):.1f}%")

    # Build comprehensive rule report
    rule_report = []
    for rule, counts in rule_counts.items():
        tier = counts["tier"]
        full_prec = counts["gt"] / counts["total"] * 100 if counts["total"] > 0 else 0

        h_data = holdout_rule_precisions.get(rule, [])
        if h_data:
            h_precs = [p for p, n in h_data]
            h_ns = [n for p, n in h_data]
            h_mean = np.mean(h_precs)
            h_std = np.std(h_precs)
            h_n_mean = np.mean(h_ns)
        else:
            h_mean = 0
            h_std = 0
            h_n_mean = 0

        rule_report.append({
            "rule": rule,
            "tier": tier,
            "full_precision": round(full_prec, 1),
            "full_n": counts["total"],
            "full_gt": counts["gt"],
            "holdout_precision": round(h_mean, 1),
            "holdout_std": round(h_std, 1),
            "holdout_n_mean": round(h_n_mean, 1),
            "delta": round(h_mean - full_prec, 1),
            "seeds_seen": len(h_data),
        })

    # Sort by tier then precision
    tier_rank = {"GOLDEN": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "FILTER": 4}
    rule_report.sort(key=lambda x: (tier_rank.get(x["tier"], 5), -x["full_precision"]))

    print(f"\n{'='*120}")
    print(f"{'Rule':50s} {'Tier':8s} {'Full':>8s} {'n':>6s} {'Holdout':>10s} {'±':>5s} {'Δ':>6s} {'n_ho':>6s} {'Status':>12s}")
    print(f"{'='*120}")

    # Determine status for each rule
    tier_thresholds = {
        "GOLDEN": {"keep": 40, "concern": 25},
        "HIGH": {"keep": 30, "concern": 20},
        "MEDIUM": {"keep": 15, "concern": 8},
        "LOW": {"keep": 5, "concern": 0},
        "FILTER": {"keep": 0, "concern": 15},  # FILTER should be LOW precision
    }

    prune_candidates = []
    keep_rules = []

    for r in rule_report:
        tier = r["tier"]
        thresholds = tier_thresholds.get(tier, {"keep": 0, "concern": 0})

        if tier == "FILTER":
            # FILTER rules: higher precision means the FILTER is wrong
            if r["holdout_precision"] > thresholds["concern"]:
                status = "OVER-FILTER"
            else:
                status = "OK"
        else:
            if r["holdout_precision"] >= thresholds["keep"]:
                status = "KEEP"
            elif r["holdout_precision"] >= thresholds["concern"]:
                status = "MARGINAL"
            elif r["holdout_n_mean"] < 3:
                status = "TOO_SMALL"
            else:
                status = "PRUNE"

        r["status"] = status
        if status == "PRUNE":
            prune_candidates.append(r)
        elif status == "KEEP":
            keep_rules.append(r)

        flag = " <<<" if status in ("PRUNE", "OVER-FILTER") else ""
        print(f"  {r['rule']:48s} {r['tier']:8s} {r['full_precision']:7.1f}% {r['full_n']:5d} {r['holdout_precision']:9.1f}% {r['holdout_std']:5.1f} {r['delta']:+5.1f} {r['holdout_n_mean']:5.1f} {status:>12s}{flag}")

    print(f"\n{'='*80}")
    print(f"SUMMARY:")
    print(f"  Total rules: {len(rule_report)}")
    print(f"  KEEP: {sum(1 for r in rule_report if r['status'] == 'KEEP')}")
    print(f"  MARGINAL: {sum(1 for r in rule_report if r['status'] == 'MARGINAL')}")
    print(f"  PRUNE: {sum(1 for r in rule_report if r['status'] == 'PRUNE')}")
    print(f"  TOO_SMALL: {sum(1 for r in rule_report if r['status'] == 'TOO_SMALL')}")
    print(f"  OVER-FILTER: {sum(1 for r in rule_report if r['status'] == 'OVER-FILTER')}")
    print(f"  OK (FILTER): {sum(1 for r in rule_report if r['status'] == 'OK')}")

    if prune_candidates:
        print(f"\nPRUNE CANDIDATES (rules performing below tier threshold on holdout):")
        total_prune_preds = 0
        for r in prune_candidates:
            print(f"  {r['rule']:48s} {r['tier']:8s} full={r['full_precision']:.1f}% holdout={r['holdout_precision']:.1f}% n={r['full_n']}")
            total_prune_preds += r["full_n"]
        print(f"  Total predictions affected by pruning: {total_prune_preds}")

    # Save results
    output = {
        "rule_report": rule_report,
        "tier_holdout": {
            tier: {
                "mean": round(np.mean(holdout_tier_precisions[tier]), 1),
                "std": round(np.std(holdout_tier_precisions[tier]), 1),
            }
            for tier in tier_order
        },
        "prune_candidates": [r["rule"] for r in prune_candidates],
    }
    out_path = Path(__file__).parent.parent / "data" / "analysis" / "h402_rule_audit.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
