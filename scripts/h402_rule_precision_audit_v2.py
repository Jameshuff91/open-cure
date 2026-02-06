#!/usr/bin/env python3
"""
h402 v2: Fixed rule-by-rule precision audit using (rule, tier) pairs.

The v1 audit had a bug: one rule name (e.g., 'cancer') can map to multiple tiers
(GOLDEN, HIGH, MEDIUM, LOW) depending on drug class matches. v2 uses (rule, tier)
as the key.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

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
    """Get per-(rule,tier) precision."""
    # Key is (rule, tier)
    rule_tier_counts = defaultdict(lambda: {"total": 0, "gt": 0})
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

            key = (rule, tier)
            rule_tier_counts[key]["total"] += 1
            rule_tier_counts[key]["gt"] += int(is_gt)

            tier_counts[tier]["total"] += 1
            tier_counts[tier]["gt"] += int(is_gt)

    return dict(rule_tier_counts), dict(tier_counts)


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

    # === Phase 1: Full-data ===
    print("\n=== PHASE 1: Full-data (after h402 demotions) ===")
    rule_tier_counts, tier_counts = evaluate_rules(predictor, all_diseases, gt_data)

    tier_order = ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]
    print(f"\nTier precision:")
    for tier in tier_order:
        tc = tier_counts.get(tier, {"total": 0, "gt": 0})
        prec = tc["gt"] / tc["total"] * 100 if tc["total"] > 0 else 0
        print(f"  {tier}: {tc['gt']}/{tc['total']} = {prec:.1f}%")

    print(f"\nRules by (rule, tier):")
    for (rule, tier), counts in sorted(rule_tier_counts.items(), key=lambda x: (
        {"GOLDEN": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "FILTER": 4}.get(x[0][1], 5),
        -x[1]["gt"] / max(x[1]["total"], 1)
    )):
        prec = counts["gt"] / counts["total"] * 100 if counts["total"] > 0 else 0
        print(f"  ({rule}, {tier}): {counts['gt']}/{counts['total']} = {prec:.1f}%")

    # === Phase 2: Holdout ===
    print(f"\n{'='*80}")
    print("=== PHASE 2: Holdout Validation (5-seed) ===")
    seeds = [42, 123, 456, 789, 2024]

    holdout_rule_precs = defaultdict(list)
    holdout_tier_precs = defaultdict(list)

    for seed in seeds:
        print(f"  Seed {seed}...")
        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        originals = recompute_gt_structures(predictor, set(train_ids))

        rt_counts_h, tier_counts_h = evaluate_rules(predictor, holdout_ids, gt_data)

        for (rule, tier), counts in rt_counts_h.items():
            prec = counts["gt"] / counts["total"] * 100 if counts["total"] > 0 else 0
            holdout_rule_precs[(rule, tier)].append((prec, counts["total"]))

        for tier in tier_order:
            tc = tier_counts_h.get(tier, {"total": 0, "gt": 0})
            prec = tc["gt"] / tc["total"] * 100 if tc["total"] > 0 else 0
            holdout_tier_precs[tier].append(prec)

        restore_gt_structures(predictor, originals)

    # === Summary ===
    print(f"\n{'='*80}")
    print("=== HOLDOUT SUMMARY ===")
    for tier in tier_order:
        vals = holdout_tier_precs[tier]
        print(f"  {tier}: {np.mean(vals):.1f}% ± {np.std(vals):.1f}%")

    # Build report
    rule_report = []
    for (rule, tier), counts in rule_tier_counts.items():
        full_prec = counts["gt"] / counts["total"] * 100 if counts["total"] > 0 else 0

        h_data = holdout_rule_precs.get((rule, tier), [])
        if h_data:
            h_precs = [p for p, n in h_data]
            h_mean = np.mean(h_precs)
            h_std = np.std(h_precs)
            h_n_mean = np.mean([n for p, n in h_data])
        else:
            h_mean = h_std = h_n_mean = 0.0

        rule_report.append({
            "rule": rule,
            "tier": tier,
            "full_precision": round(full_prec, 1),
            "full_n": counts["total"],
            "full_gt": counts["gt"],
            "holdout_mean": round(h_mean, 1),
            "holdout_std": round(h_std, 1),
            "holdout_n_mean": round(h_n_mean, 1),
            "delta": round(h_mean - full_prec, 1),
        })

    tier_rank = {"GOLDEN": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "FILTER": 4}
    rule_report.sort(key=lambda x: (tier_rank.get(x["tier"], 5), -x["full_precision"]))

    print(f"\n{'='*120}")
    print(f"{'Rule':50s} {'Tier':8s} {'Full':>8s} {'n':>6s} {'Holdout':>10s} {'±':>5s} {'Δ':>6s}")
    print(f"{'='*120}")
    for r in rule_report:
        print(f"  {r['rule']:48s} {r['tier']:8s} {r['full_precision']:7.1f}% {r['full_n']:5d} {r['holdout_mean']:9.1f}% {r['holdout_std']:5.1f} {r['delta']:+5.1f}")

    # Save
    output = {
        "rule_report": rule_report,
        "tier_holdout": {
            tier: {"mean": round(np.mean(holdout_tier_precs[tier]), 1),
                   "std": round(np.std(holdout_tier_precs[tier]), 1)}
            for tier in tier_order
        },
    }
    out_path = Path(__file__).parent.parent / "data" / "analysis" / "h402_rule_audit_v2.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
