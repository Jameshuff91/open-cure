#!/usr/bin/env python3
"""
h615: Expanded GT-Based Tier Recalibration

The tier rules were developed by looking at full-data precision with internal GT
(~3,070 pairs). But expanded GT (59,584 pairs) gives much higher precision across
all tiers. This script:

1. Computes full-data precision for every rule using BOTH internal and expanded GT
2. Identifies rules where expanded GT precision crosses tier boundaries
3. Identifies rules currently placed too low that could be promoted
4. Runs 5-seed holdout with expanded GT to confirm any promotions

Key question: Are any rules miscategorized because they were evaluated with incomplete GT?
"""

import json
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from production_predictor import (
    ConfidenceTier,
    DrugRepurposingPredictor,
    extract_cancer_types,
    DISEASE_HIERARCHY_GROUPS,
    HIERARCHY_EXCLUSIONS,
)


# Tier boundaries based on holdout precision
# GOLDEN > 55%, HIGH > 40%, MEDIUM > 25%, LOW > 12%
TIER_BOUNDARIES = {
    "GOLDEN": 55.0,  # Above HIGH
    "HIGH": 40.0,    # Above MEDIUM
    "MEDIUM": 25.0,  # Above LOW
    "LOW": 12.0,     # Above FILTER
    "FILTER": 0.0,
}


def evaluate_full_data(
    predictor: DrugRepurposingPredictor,
    gt_set: Set[Tuple[str, str]],
    label: str,
) -> Dict:
    """Evaluate all diseases using given GT set. Returns per-rule and per-tier precision."""
    all_diseases = [d for d in predictor.ground_truth if d in predictor.embeddings]

    rule_stats = defaultdict(lambda: {"tier": None, "hits": 0, "misses": 0})
    tier_stats = defaultdict(lambda: {"hits": 0, "misses": 0})

    n_evaluated = 0
    for disease_id in all_diseases:
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        try:
            result = predictor.predict(disease_name, top_n=30, include_filtered=True)
        except Exception:
            continue

        n_evaluated += 1
        for pred in result.predictions:
            drug_id = pred.drug_id
            is_hit = (disease_id, drug_id) in gt_set

            rule = pred.category_specific_tier or "default"
            tier = pred.confidence_tier.name

            rule_stats[rule]["tier"] = tier
            if is_hit:
                rule_stats[rule]["hits"] += 1
            else:
                rule_stats[rule]["misses"] += 1

            if is_hit:
                tier_stats[tier]["hits"] += 1
            else:
                tier_stats[tier]["misses"] += 1

    # Compute precisions
    rule_results = {}
    for rule, stats in rule_stats.items():
        total = stats["hits"] + stats["misses"]
        precision = stats["hits"] / total if total > 0 else 0
        rule_results[rule] = {
            "tier": stats["tier"],
            "hits": stats["hits"],
            "total": total,
            "precision": round(precision * 100, 1),
        }

    tier_results = {}
    for tier, stats in tier_stats.items():
        total = stats["hits"] + stats["misses"]
        precision = stats["hits"] / total if total > 0 else 0
        tier_results[tier] = {
            "hits": stats["hits"],
            "total": total,
            "precision": round(precision * 100, 1),
        }

    return {
        "label": label,
        "n_diseases": n_evaluated,
        "tier_precision": tier_results,
        "rule_precision": rule_results,
    }


def build_gt_set(gt_data: Dict) -> Set[Tuple[str, str]]:
    """Build (disease_id, drug_id) set from GT data."""
    gt_set = set()
    for disease_id, drugs in gt_data.items():
        for drug in drugs:
            if isinstance(drug, str):
                gt_set.add((disease_id, drug))
            elif isinstance(drug, dict):
                gt_set.add((disease_id, drug.get("drug_id") or drug.get("drug")))
    return gt_set


def split_diseases(all_diseases, seed, train_ratio=0.8):
    rng = np.random.RandomState(seed)
    shuffled = list(all_diseases)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def recompute_gt_structures(predictor, train_disease_ids):
    """Recompute all GT-derived data structures from training diseases only."""
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
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            for drug_id in predictor.ground_truth[disease_id]:
                new_freq[drug_id] += 1
    predictor.drug_train_freq = dict(new_freq)

    new_d2d = defaultdict(set)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            for drug_id in predictor.ground_truth[disease_id]:
                new_d2d[drug_id].add(disease_name.lower())
    predictor.drug_to_diseases = dict(new_d2d)

    new_cancer = defaultdict(set)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            cancer_types = extract_cancer_types(disease_name)
            if cancer_types:
                for drug_id in predictor.ground_truth[disease_id]:
                    new_cancer[drug_id].update(cancer_types)
    predictor.drug_cancer_types = dict(new_cancer)

    new_groups = defaultdict(set)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            disease_lower = disease_name.lower()
            for category, groups in DISEASE_HIERARCHY_GROUPS.items():
                for group_name, keywords in groups.items():
                    exclusions = HIERARCHY_EXCLUSIONS.get((category, group_name), [])
                    if any(excl in disease_lower for excl in exclusions):
                        continue
                    if any(kw in disease_lower or disease_lower in kw for kw in keywords):
                        for drug_id in predictor.ground_truth[disease_id]:
                            new_groups[drug_id].add((category, group_name))
    predictor.drug_disease_groups = dict(new_groups)

    predictor.train_diseases = [
        d for d in train_disease_ids if d in predictor.embeddings
    ]
    predictor.train_embeddings = np.array(
        [predictor.embeddings[d] for d in predictor.train_diseases], dtype=np.float32
    )
    predictor.train_disease_categories = {}
    for d in predictor.train_diseases:
        name = predictor.disease_names.get(d, d)
        predictor.train_disease_categories[d] = predictor.categorize_disease(name)

    return originals


def restore_gt_structures(predictor, originals):
    predictor.drug_train_freq = originals["drug_train_freq"]
    predictor.drug_to_diseases = originals["drug_to_diseases"]
    predictor.drug_cancer_types = originals["drug_cancer_types"]
    predictor.drug_disease_groups = originals["drug_disease_groups"]
    predictor.train_diseases = originals["train_diseases"]
    predictor.train_embeddings = originals["train_embeddings"]
    predictor.train_disease_categories = originals["train_disease_categories"]


def evaluate_holdout_per_rule(predictor, holdout_ids, gt_set):
    """Evaluate holdout diseases, return per-rule stats."""
    rule_stats = defaultdict(lambda: {"tier": None, "hits": 0, "misses": 0})
    tier_stats = defaultdict(lambda: {"hits": 0, "misses": 0})

    for disease_id in holdout_ids:
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        try:
            result = predictor.predict(disease_name, top_n=30, include_filtered=True)
        except Exception:
            continue

        for pred in result.predictions:
            drug_id = pred.drug_id
            is_hit = (disease_id, drug_id) in gt_set
            rule = pred.category_specific_tier or "default"
            tier = pred.confidence_tier.name

            rule_stats[rule]["tier"] = tier
            if is_hit:
                rule_stats[rule]["hits"] += 1
            else:
                rule_stats[rule]["misses"] += 1

            if is_hit:
                tier_stats[tier]["hits"] += 1
            else:
                tier_stats[tier]["misses"] += 1

    rule_results = {}
    for rule, stats in rule_stats.items():
        total = stats["hits"] + stats["misses"]
        precision = stats["hits"] / total if total > 0 else 0
        rule_results[rule] = {
            "tier": stats["tier"],
            "hits": stats["hits"],
            "total": total,
            "precision": round(precision * 100, 1),
        }

    tier_results = {}
    for tier, stats in tier_stats.items():
        total = stats["hits"] + stats["misses"]
        precision = stats["hits"] / total if total > 0 else 0
        tier_results[tier] = {
            "hits": stats["hits"],
            "total": total,
            "precision": round(precision * 100, 1),
        }

    return {"rule_precision": rule_results, "tier_precision": tier_results}


def main():
    print("=" * 80)
    print("h615: Expanded GT-Based Tier Recalibration")
    print("=" * 80)

    print("\nLoading predictor...")
    predictor = DrugRepurposingPredictor()

    # Load both GT sources
    # Internal GT (what predictor uses for training frequency etc.)
    internal_gt_set = set()
    for disease_id, drugs in predictor.ground_truth.items():
        for drug_id in drugs:
            internal_gt_set.add((disease_id, drug_id))
    print(f"Internal GT: {len(internal_gt_set)} pairs")

    # Expanded GT (for proper evaluation)
    gt_path = predictor.reference_dir / "expanded_ground_truth.json"
    with open(gt_path) as f:
        expanded_gt_data = json.load(f)
    expanded_gt_set = build_gt_set(expanded_gt_data)
    print(f"Expanded GT: {len(expanded_gt_set)} pairs")
    print(f"Ratio: {len(expanded_gt_set)/len(internal_gt_set):.1f}x")

    # Step 1: Full-data evaluation with BOTH GT sources
    print("\n" + "=" * 80)
    print("STEP 1: Full-data precision with internal vs expanded GT")
    print("=" * 80)

    internal_result = evaluate_full_data(predictor, internal_gt_set, "internal")
    expanded_result = evaluate_full_data(predictor, expanded_gt_set, "expanded")

    # Print tier comparison
    print(f"\n{'Tier':<10s} {'Internal%':>10s} {'Expanded%':>10s} {'Δ':>8s} {'Ratio':>8s}")
    print("-" * 50)
    for tier in ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]:
        int_prec = internal_result["tier_precision"].get(tier, {}).get("precision", 0)
        exp_prec = expanded_result["tier_precision"].get(tier, {}).get("precision", 0)
        delta = exp_prec - int_prec
        ratio = exp_prec / int_prec if int_prec > 0 else float("inf")
        print(f"{tier:<10s} {int_prec:9.1f}% {exp_prec:9.1f}% {delta:+7.1f}pp {ratio:7.2f}x")

    # Step 2: Per-rule comparison — find boundary crossings
    print(f"\n{'='*80}")
    print("STEP 2: Per-rule precision with internal vs expanded GT")
    print(f"{'='*80}")

    # Determine "natural" tier from expanded precision
    def natural_tier(precision):
        if precision >= TIER_BOUNDARIES["GOLDEN"]:
            return "GOLDEN"
        elif precision >= TIER_BOUNDARIES["HIGH"]:
            return "HIGH"
        elif precision >= TIER_BOUNDARIES["MEDIUM"]:
            return "MEDIUM"
        elif precision >= TIER_BOUNDARIES["LOW"]:
            return "LOW"
        else:
            return "FILTER"

    tier_order = {"GOLDEN": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "FILTER": 0}

    crossings = []  # Rules where expanded GT suggests different tier

    print(f"\n{'Rule':<42s} {'Tier':<8s} {'Int%':>6s} {'Exp%':>6s} {'Δ':>7s} {'n':>6s} {'Natural':>8s} {'Cross?':>6s}")
    print("-" * 95)

    all_rules = set(internal_result["rule_precision"].keys()) | set(expanded_result["rule_precision"].keys())

    # Sort by tier then rule name
    def sort_key(rule):
        tier = internal_result["rule_precision"].get(rule, {}).get("tier", "FILTER")
        return (tier_order.get(tier, 0), rule)

    for rule in sorted(all_rules, key=sort_key, reverse=True):
        int_stats = internal_result["rule_precision"].get(rule, {"tier": "?", "precision": 0, "total": 0})
        exp_stats = expanded_result["rule_precision"].get(rule, {"tier": "?", "precision": 0, "total": 0})

        current_tier = int_stats.get("tier") or exp_stats.get("tier") or "?"
        int_prec = int_stats.get("precision", 0)
        exp_prec = exp_stats.get("precision", 0)
        delta = exp_prec - int_prec
        n = exp_stats.get("total", int_stats.get("total", 0))

        nat_tier = natural_tier(exp_prec)
        is_crossing = (
            current_tier in tier_order
            and nat_tier in tier_order
            and tier_order[nat_tier] != tier_order[current_tier]
            and n >= 20  # Only flag rules with sufficient n
        )

        cross_str = "YES" if is_crossing else ""
        if is_crossing:
            direction = "UP" if tier_order[nat_tier] > tier_order[current_tier] else "DOWN"
            cross_str = direction
            crossings.append({
                "rule": rule,
                "current_tier": current_tier,
                "natural_tier": nat_tier,
                "direction": direction,
                "internal_precision": int_prec,
                "expanded_precision": exp_prec,
                "n": n,
                "delta": delta,
            })

        print(f"  {rule:<40s} {current_tier:<8s} {int_prec:5.1f}% {exp_prec:5.1f}% {delta:+6.1f}pp {n:5d} {nat_tier:>8s} {cross_str:>6s}")

    # Step 3: Analyze crossings
    print(f"\n{'='*80}")
    print("STEP 3: Tier Boundary Crossings (expanded GT suggests different tier)")
    print(f"{'='*80}")

    if not crossings:
        print("No tier boundary crossings found with n >= 20.")
    else:
        print(f"\nFound {len(crossings)} rules with tier boundary crossings:")
        promotions = [c for c in crossings if c["direction"] == "UP"]
        demotions = [c for c in crossings if c["direction"] == "DOWN"]

        if promotions:
            print(f"\n--- POTENTIAL PROMOTIONS ({len(promotions)}) ---")
            for c in sorted(promotions, key=lambda x: x["expanded_precision"], reverse=True):
                print(f"  {c['rule']}: {c['current_tier']} → {c['natural_tier']} "
                      f"(internal={c['internal_precision']:.1f}%, expanded={c['expanded_precision']:.1f}%, "
                      f"Δ=+{c['delta']:.1f}pp, n={c['n']})")

        if demotions:
            print(f"\n--- POTENTIAL DEMOTIONS ({len(demotions)}) ---")
            for c in sorted(demotions, key=lambda x: x["expanded_precision"]):
                print(f"  {c['rule']}: {c['current_tier']} → {c['natural_tier']} "
                      f"(internal={c['internal_precision']:.1f}%, expanded={c['expanded_precision']:.1f}%, "
                      f"Δ={c['delta']:.1f}pp, n={c['n']})")

    # Step 4: Holdout validation for any crossing rules with n >= 20
    if crossings:
        print(f"\n{'='*80}")
        print("STEP 4: 5-Seed Holdout Validation of Crossing Rules")
        print(f"{'='*80}")

        seeds = [42, 123, 456, 789, 2024]
        all_diseases = [d for d in predictor.ground_truth if d in predictor.embeddings]

        crossing_rules = {c["rule"] for c in crossings}
        holdout_rule_precs = defaultdict(list)
        holdout_rule_ns = defaultdict(list)
        holdout_tier_precs = defaultdict(list)

        for seed_idx, seed in enumerate(seeds):
            print(f"\n  Seed {seed} ({seed_idx+1}/{len(seeds)})...")
            train_ids, holdout_ids = split_diseases(all_diseases, seed)
            train_set = set(train_ids)

            originals = recompute_gt_structures(predictor, train_set)
            ho_result = evaluate_holdout_per_rule(predictor, holdout_ids, expanded_gt_set)

            for rule, stats in ho_result["rule_precision"].items():
                if rule in crossing_rules:
                    holdout_rule_precs[rule].append(stats["precision"])
                    holdout_rule_ns[rule].append(stats["total"])

            for tier, stats in ho_result["tier_precision"].items():
                holdout_tier_precs[tier].append(stats["precision"])

            restore_gt_structures(predictor, originals)

        print(f"\n--- Holdout Results for Crossing Rules ---")
        print(f"{'Rule':<42s} {'Tier':<8s} {'Exp-Full%':>9s} {'Holdout%':>10s} {'±std':>6s} {'n/seed':>7s} {'Verdict':<12s}")
        print("-" * 100)

        final_crossings = []
        for c in crossings:
            rule = c["rule"]
            if rule in holdout_rule_precs and len(holdout_rule_precs[rule]) > 0:
                ho_mean = np.mean(holdout_rule_precs[rule])
                ho_std = np.std(holdout_rule_precs[rule])
                ho_n = np.mean(holdout_rule_ns[rule])
                ho_nat = natural_tier(ho_mean)

                # Verdict: does holdout confirm the crossing?
                if ho_nat == c["natural_tier"] and ho_n >= 5:
                    verdict = "CONFIRMED"
                elif tier_order.get(ho_nat, 0) > tier_order.get(c["current_tier"], 0):
                    verdict = "PARTIAL"  # Holdout also says higher, but different level
                else:
                    verdict = "NOT CONFIRMED"

                print(f"  {rule:<40s} {c['current_tier']:<8s} {c['expanded_precision']:8.1f}% "
                      f"{ho_mean:9.1f}% ±{ho_std:4.1f} {ho_n:6.1f} {verdict}")

                final_crossings.append({
                    **c,
                    "holdout_mean": round(ho_mean, 1),
                    "holdout_std": round(ho_std, 1),
                    "holdout_n": round(ho_n, 1),
                    "holdout_natural_tier": ho_nat,
                    "verdict": verdict,
                })
            else:
                print(f"  {rule:<40s} {c['current_tier']:<8s} {c['expanded_precision']:8.1f}% "
                      f"{'N/A':>10s} {'':>6s} {'':>7s} {'NO DATA'}")
                final_crossings.append({
                    **c,
                    "holdout_mean": None,
                    "holdout_std": None,
                    "holdout_n": None,
                    "holdout_natural_tier": None,
                    "verdict": "NO DATA",
                })

        # Also show overall tier holdout
        print(f"\n--- Overall Tier Holdout (expanded GT) ---")
        for tier in ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]:
            if tier in holdout_tier_precs:
                m = np.mean(holdout_tier_precs[tier])
                s = np.std(holdout_tier_precs[tier])
                print(f"  {tier}: {m:.1f}% ± {s:.1f}%")
    else:
        final_crossings = []

    # Save results
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    confirmed = [c for c in final_crossings if c.get("verdict") == "CONFIRMED"]
    partial = [c for c in final_crossings if c.get("verdict") == "PARTIAL"]

    print(f"Total crossing rules: {len(crossings)}")
    print(f"Holdout confirmed: {len(confirmed)}")
    print(f"Holdout partial: {len(partial)}")

    if confirmed:
        print(f"\n--- ACTIONABLE TIER CHANGES ---")
        for c in confirmed:
            print(f"  {c['rule']}: {c['current_tier']} → {c['natural_tier']} "
                  f"(holdout={c['holdout_mean']:.1f}% ± {c['holdout_std']:.1f}%, "
                  f"n={c['holdout_n']:.0f}/seed)")

    # Convert numpy for JSON
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    output = {
        "internal_gt_pairs": len(internal_gt_set),
        "expanded_gt_pairs": len(expanded_gt_set),
        "tier_boundaries_used": TIER_BOUNDARIES,
        "crossings": final_crossings if final_crossings else crossings,
        "confirmed_changes": confirmed,
        "partial_changes": partial,
    }

    output_path = Path(__file__).parent.parent / "data" / "analysis" / "h615_expanded_gt_recalibration.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=convert)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
