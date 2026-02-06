#!/usr/bin/env python3
"""
h393: Holdout Validation of All Tier Rules (Overfitting Audit)

All 40+ hand-crafted tier rules were derived by analyzing the SAME ground truth
used for evaluation. This is training-on-test-set. This script validates whether
the tier system is real or overfitted by:

1. Splitting diseases 80/20 (5 seeds)
2. Recomputing ALL GT-derived data structures from training diseases only
3. Running predictions on holdout diseases
4. Measuring per-rule and per-tier precision on holdout vs full data

Success criteria:
- GOLDEN holdout precision > 30% (below this, tier system is broken)
- Individual rules: holdout precision within 15pp of full-data precision
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


def split_diseases(
    all_diseases: List[str], seed: int, train_ratio: float = 0.8
) -> Tuple[List[str], List[str]]:
    """Split diseases into train/holdout sets."""
    rng = np.random.RandomState(seed)
    shuffled = list(all_diseases)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def recompute_gt_structures(
    predictor: DrugRepurposingPredictor,
    train_disease_ids: Set[str],
) -> Dict:
    """Recompute all GT-derived data structures from training diseases only.

    Returns dict of original values so they can be restored.
    """
    originals = {
        "drug_train_freq": dict(predictor.drug_train_freq),
        "drug_to_diseases": dict(predictor.drug_to_diseases),
        "drug_cancer_types": dict(predictor.drug_cancer_types),
        "drug_disease_groups": dict(predictor.drug_disease_groups),
        "train_diseases": list(predictor.train_diseases),
        "train_embeddings": predictor.train_embeddings.copy(),
        "train_disease_categories": dict(predictor.train_disease_categories),
    }

    # 1. Recompute drug_train_freq from training diseases only
    new_freq: Dict[str, int] = defaultdict(int)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            for drug_id in predictor.ground_truth[disease_id]:
                new_freq[drug_id] += 1
    predictor.drug_train_freq = dict(new_freq)

    # 2. Recompute drug_to_diseases from training diseases only
    new_d2d: Dict[str, Set[str]] = defaultdict(set)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            for drug_id in predictor.ground_truth[disease_id]:
                new_d2d[drug_id].add(disease_name.lower())
    predictor.drug_to_diseases = dict(new_d2d)

    # 3. Recompute drug_cancer_types from training diseases only
    new_cancer: Dict[str, Set[str]] = defaultdict(set)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            cancer_types = extract_cancer_types(disease_name)
            if cancer_types:
                for drug_id in predictor.ground_truth[disease_id]:
                    new_cancer[drug_id].update(cancer_types)
    predictor.drug_cancer_types = dict(new_cancer)

    # 4. Recompute drug_disease_groups from training diseases only
    # h469: Use HIERARCHY_EXCLUSIONS to prevent false matches (was missing before)
    new_groups: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
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

    # 5. Rebuild kNN index from training diseases only
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


def restore_gt_structures(
    predictor: DrugRepurposingPredictor, originals: Dict
) -> None:
    """Restore original GT-derived data structures."""
    predictor.drug_train_freq = originals["drug_train_freq"]
    predictor.drug_to_diseases = originals["drug_to_diseases"]
    predictor.drug_cancer_types = originals["drug_cancer_types"]
    predictor.drug_disease_groups = originals["drug_disease_groups"]
    predictor.train_diseases = originals["train_diseases"]
    predictor.train_embeddings = originals["train_embeddings"]
    predictor.train_disease_categories = originals["train_disease_categories"]


def evaluate_on_diseases(
    predictor: DrugRepurposingPredictor,
    disease_ids: List[str],
    gt_data: Dict[str, List[str]],
) -> Dict:
    """Evaluate tier precision on a set of diseases.

    Returns per-rule and per-tier precision stats.
    """
    # Build GT set for quick lookup
    gt_set = set()
    for disease_id, drugs in gt_data.items():
        for drug in drugs:
            if isinstance(drug, str):
                gt_set.add((disease_id, drug))
            elif isinstance(drug, dict):
                gt_set.add((disease_id, drug.get("drug_id") or drug.get("drug")))

    rule_stats: Dict[str, Dict] = defaultdict(
        lambda: {"tier": None, "hits": 0, "misses": 0}
    )
    tier_stats: Dict[str, Dict] = defaultdict(lambda: {"hits": 0, "misses": 0})

    n_evaluated = 0
    n_preds = 0

    for disease_id in disease_ids:
        disease_name = predictor.disease_names.get(disease_id, disease_id)

        try:
            result = predictor.predict(
                disease_name, top_n=30, include_filtered=True
            )
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

            n_preds += 1

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
        "n_diseases": n_evaluated,
        "n_predictions": n_preds,
        "tier_precision": tier_results,
        "rule_precision": rule_results,
    }


def main() -> None:
    seeds = [42, 123, 456, 789, 2024]
    print("=" * 70)
    print("h393: Holdout Validation of All Tier Rules (Overfitting Audit)")
    print("=" * 70)

    print("\nLoading predictor...")
    predictor = DrugRepurposingPredictor()

    # Load expanded ground truth for evaluation
    gt_path = predictor.reference_dir / "expanded_ground_truth.json"
    with open(gt_path) as f:
        gt_data = json.load(f)
    print(f"Loaded {len(gt_data)} GT diseases, {sum(len(v) for v in gt_data.values())} pairs")

    # Get all diseases that have both GT and embeddings
    all_diseases = [
        d for d in predictor.ground_truth if d in predictor.embeddings
    ]
    print(f"Diseases with GT + embeddings: {len(all_diseases)}")

    # First: compute full-data baseline (what we've been measuring)
    print("\n--- FULL-DATA BASELINE (no holdout) ---")
    full_result = evaluate_on_diseases(predictor, all_diseases, gt_data)
    print(f"Evaluated {full_result['n_diseases']} diseases, {full_result['n_predictions']} predictions")
    print("\nFull-data tier precision:")
    for tier in ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]:
        if tier in full_result["tier_precision"]:
            t = full_result["tier_precision"][tier]
            print(f"  {tier}: {t['precision']}% ({t['hits']}/{t['total']})")

    # Holdout evaluation across seeds
    all_holdout_tier = defaultdict(list)  # tier -> [precision per seed]
    all_holdout_rule = defaultdict(list)  # rule -> [precision per seed]
    all_holdout_rule_tier = {}  # rule -> tier (for reporting)
    all_holdout_rule_n = defaultdict(list)  # rule -> [n per seed]

    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'='*70}")
        print(f"SEED {seed} ({seed_idx+1}/{len(seeds)})")
        print(f"{'='*70}")

        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        train_set = set(train_ids)
        print(f"Train: {len(train_ids)}, Holdout: {len(holdout_ids)}")

        # Recompute GT structures from training diseases only
        originals = recompute_gt_structures(predictor, train_set)

        # Evaluate on holdout diseases
        holdout_result = evaluate_on_diseases(predictor, holdout_ids, gt_data)

        print(f"Holdout: {holdout_result['n_diseases']} diseases, {holdout_result['n_predictions']} predictions")
        print("\nHoldout tier precision:")
        for tier in ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]:
            if tier in holdout_result["tier_precision"]:
                t = holdout_result["tier_precision"][tier]
                print(f"  {tier}: {t['precision']}% ({t['hits']}/{t['total']})")
                all_holdout_tier[tier].append(t["precision"])

        # Collect per-rule holdout precision
        for rule, stats in holdout_result["rule_precision"].items():
            all_holdout_rule[rule].append(stats["precision"])
            all_holdout_rule_tier[rule] = stats["tier"]
            all_holdout_rule_n[rule].append(stats["total"])

        # Restore original structures
        restore_gt_structures(predictor, originals)

    # Aggregate results
    print("\n" + "=" * 70)
    print("AGGREGATE HOLDOUT RESULTS (mean ± std across 5 seeds)")
    print("=" * 70)

    print("\n--- TIER PRECISION: Full-Data vs Holdout ---")
    tier_comparison = {}
    for tier in ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]:
        full_prec = full_result["tier_precision"].get(tier, {}).get("precision", 0)
        if tier in all_holdout_tier and len(all_holdout_tier[tier]) > 0:
            holdout_mean = np.mean(all_holdout_tier[tier])
            holdout_std = np.std(all_holdout_tier[tier])
            delta = holdout_mean - full_prec
            print(
                f"  {tier:8s}: Full={full_prec:5.1f}% | "
                f"Holdout={holdout_mean:5.1f}% ± {holdout_std:4.1f}% | "
                f"Delta={delta:+5.1f}pp"
            )
            tier_comparison[tier] = {
                "full_precision": full_prec,
                "holdout_mean": round(holdout_mean, 1),
                "holdout_std": round(holdout_std, 1),
                "delta": round(delta, 1),
                "seed_values": all_holdout_tier[tier],
            }

    # Check success criteria
    golden_holdout = tier_comparison.get("GOLDEN", {}).get("holdout_mean", 0)
    print(f"\n--- SUCCESS CRITERIA ---")
    print(f"GOLDEN holdout precision > 30%: {'PASS' if golden_holdout > 30 else 'FAIL'} ({golden_holdout:.1f}%)")

    # Check tier ordering
    tier_order = ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]
    tier_holdout_values = [
        tier_comparison.get(t, {}).get("holdout_mean", 0) for t in tier_order
    ]
    is_monotonic = all(
        tier_holdout_values[i] >= tier_holdout_values[i + 1]
        for i in range(len(tier_holdout_values) - 1)
    )
    print(f"Tier ordering (GOLDEN > HIGH > MEDIUM > LOW > FILTER): {'PASS' if is_monotonic else 'FAIL'}")
    print(f"  Values: {' > '.join(f'{t}={v:.1f}%' for t, v in zip(tier_order, tier_holdout_values))}")

    # Per-rule analysis: identify overfitted rules
    print(f"\n--- PER-RULE HOLDOUT ANALYSIS ---")
    print(f"{'Rule':<40s} {'Tier':<8s} {'Full%':>6s} {'Hold%':>6s} {'±std':>5s} {'Δ':>6s} {'n':>5s} {'Status':<12s}")
    print("-" * 100)

    rule_comparison = {}
    overfitted_rules = []

    for rule in sorted(
        full_result["rule_precision"].keys(),
        key=lambda r: full_result["rule_precision"][r].get("tier", ""),
    ):
        full_stats = full_result["rule_precision"][rule]
        full_prec = full_stats["precision"]
        full_n = full_stats["total"]

        if rule in all_holdout_rule and len(all_holdout_rule[rule]) > 0:
            holdout_mean = np.mean(all_holdout_rule[rule])
            holdout_std = np.std(all_holdout_rule[rule])
            mean_n = np.mean(all_holdout_rule_n[rule])
            delta = holdout_mean - full_prec
        else:
            holdout_mean = 0
            holdout_std = 0
            mean_n = 0
            delta = -full_prec

        # Flag as overfitted if holdout drops >15pp from full
        status = ""
        if full_n >= 10 and delta < -15:
            status = "OVERFITTED?"
            overfitted_rules.append(rule)
        elif full_n >= 10 and delta >= -5:
            status = "GENUINE"
        elif full_n < 10:
            status = "small-n"
        else:
            status = "degraded"

        tier = full_stats.get("tier", "?")
        print(
            f"  {rule:<38s} {tier:<8s} {full_prec:5.1f}% {holdout_mean:5.1f}% "
            f"±{holdout_std:4.1f} {delta:+5.1f}  {mean_n:5.0f} {status}"
        )

        rule_comparison[rule] = {
            "tier": tier,
            "full_precision": full_prec,
            "full_n": full_n,
            "holdout_mean": round(holdout_mean, 1),
            "holdout_std": round(holdout_std, 1),
            "delta": round(delta, 1),
            "mean_n": round(mean_n, 1),
            "status": status,
        }

    if overfitted_rules:
        print(f"\n--- OVERFITTED RULES (holdout drops >15pp, n>=10) ---")
        for rule in overfitted_rules:
            r = rule_comparison[rule]
            print(
                f"  {rule}: Full={r['full_precision']}% → Holdout={r['holdout_mean']}% "
                f"(Δ={r['delta']}pp, n={r['full_n']})"
            )
    else:
        print(f"\n--- No rules flagged as overfitted (all within 15pp of full-data) ---")

    # Summary statistics
    n_genuine = sum(1 for r in rule_comparison.values() if r["status"] == "GENUINE")
    n_overfitted = sum(1 for r in rule_comparison.values() if r["status"] == "OVERFITTED?")
    n_degraded = sum(1 for r in rule_comparison.values() if r["status"] == "degraded")
    n_small = sum(1 for r in rule_comparison.values() if r["status"] == "small-n")

    print(f"\n--- SUMMARY ---")
    print(f"Rules tested: {len(rule_comparison)}")
    print(f"  GENUINE (Δ > -5pp, n>=10): {n_genuine}")
    print(f"  Degraded (-15pp < Δ < -5pp, n>=10): {n_degraded}")
    print(f"  OVERFITTED? (Δ < -15pp, n>=10): {n_overfitted}")
    print(f"  Small sample (n<10): {n_small}")

    # Save results
    output = {
        "tier_comparison": tier_comparison,
        "rule_comparison": rule_comparison,
        "overfitted_rules": overfitted_rules,
        "summary": {
            "golden_holdout_precision": golden_holdout,
            "golden_pass": golden_holdout > 30,
            "tier_ordering_correct": is_monotonic,
            "n_rules_tested": len(rule_comparison),
            "n_genuine": n_genuine,
            "n_overfitted": n_overfitted,
            "n_degraded": n_degraded,
            "n_small_sample": n_small,
        },
    }

    output_path = Path(__file__).parent.parent / "data" / "analysis" / "h393_holdout_validation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert_for_json(i) for i in obj]
        if isinstance(obj, bool):
            return obj
        return obj

    with open(output_path, "w") as f:
        json.dump(convert_for_json(output), f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
