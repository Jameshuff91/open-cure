#!/usr/bin/env python3
"""
h462: Category-Specific MEDIUM→HIGH Promotion / LOW Demotion

Test whether specific categories within MEDIUM tier should be promoted (renal,
musculoskeletal) or demoted (GI, cancer) based on 5-seed holdout validation.

Key prior results to verify:
- Renal MEDIUM: 55.6% holdout (only 3 seeds) → promote to HIGH?
- Musculoskeletal MEDIUM: 51.5% holdout (high variance) → promote to HIGH?
- GI MEDIUM: 0.0% holdout → demote to LOW (already implemented as h463)
- Cancer MEDIUM: 8.7% holdout → demote to LOW?
- Dermatological MEDIUM: 36.6% holdout → leave as MEDIUM

HIGH threshold: 50.8% holdout
MEDIUM avg: 21.2% holdout
LOW avg: 12.2% holdout
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from production_predictor import (
    ConfidenceTier,
    DrugRepurposingPredictor,
    extract_cancer_types,
    DISEASE_HIERARCHY_GROUPS,
)


def split_diseases(
    all_diseases: List[str], seed: int, train_ratio: float = 0.8
) -> Tuple[List[str], List[str]]:
    rng = np.random.RandomState(seed)
    shuffled = list(all_diseases)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def recompute_gt_structures(
    predictor: DrugRepurposingPredictor,
    train_disease_ids: Set[str],
) -> Dict:
    originals = {
        "drug_train_freq": dict(predictor.drug_train_freq),
        "drug_to_diseases": dict(predictor.drug_to_diseases),
        "drug_cancer_types": dict(predictor.drug_cancer_types),
        "drug_disease_groups": dict(predictor.drug_disease_groups),
        "train_diseases": list(predictor.train_diseases),
        "train_embeddings": predictor.train_embeddings.copy(),
        "train_disease_categories": dict(predictor.train_disease_categories),
    }

    new_freq: Dict[str, int] = defaultdict(int)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            for drug_id in predictor.ground_truth[disease_id]:
                new_freq[drug_id] += 1
    predictor.drug_train_freq = dict(new_freq)

    new_d2d: Dict[str, Set[str]] = defaultdict(set)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            for drug_id in predictor.ground_truth[disease_id]:
                new_d2d[drug_id].add(disease_name.lower())
    predictor.drug_to_diseases = dict(new_d2d)

    new_cancer: Dict[str, Set[str]] = defaultdict(set)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            cancer_types = extract_cancer_types(disease_name)
            if cancer_types:
                for drug_id in predictor.ground_truth[disease_id]:
                    new_cancer[drug_id].update(cancer_types)
    predictor.drug_cancer_types = dict(new_cancer)

    new_groups: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            category = predictor.categorize_disease(disease_name)
            if category in DISEASE_HIERARCHY_GROUPS:
                for group_name, keywords in DISEASE_HIERARCHY_GROUPS[category].items():
                    if any(kw in disease_name.lower() for kw in keywords):
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


def restore_gt_structures(
    predictor: DrugRepurposingPredictor, originals: Dict
) -> None:
    predictor.drug_train_freq = originals["drug_train_freq"]
    predictor.drug_to_diseases = originals["drug_to_diseases"]
    predictor.drug_cancer_types = originals["drug_cancer_types"]
    predictor.drug_disease_groups = originals["drug_disease_groups"]
    predictor.train_diseases = originals["train_diseases"]
    predictor.train_embeddings = originals["train_embeddings"]
    predictor.train_disease_categories = originals["train_disease_categories"]


def evaluate_medium_by_category(
    predictor: DrugRepurposingPredictor,
    disease_ids: List[str],
    gt_data: Dict[str, List[str]],
) -> Dict[str, Dict]:
    """Evaluate MEDIUM tier precision broken down by disease category."""
    gt_set = set()
    for disease_id, drugs in gt_data.items():
        for drug in drugs:
            if isinstance(drug, str):
                gt_set.add((disease_id, drug))
            elif isinstance(drug, dict):
                gt_set.add((disease_id, drug.get("drug_id") or drug.get("drug")))

    # Track per-category stats for MEDIUM tier
    cat_stats: Dict[str, Dict] = defaultdict(lambda: {"hits": 0, "misses": 0, "diseases": set()})
    # Also track overall tier stats
    tier_stats: Dict[str, Dict] = defaultdict(lambda: {"hits": 0, "misses": 0})
    # Track per-category per-tier stats (for all tiers)
    cat_tier_stats: Dict[str, Dict[str, Dict]] = defaultdict(
        lambda: defaultdict(lambda: {"hits": 0, "misses": 0})
    )

    for disease_id in disease_ids:
        disease_name = predictor.disease_names.get(disease_id, disease_id)

        try:
            result = predictor.predict(disease_name, top_n=30, include_filtered=True)
        except Exception:
            continue

        for pred in result.predictions:
            drug_id = pred.drug_id
            is_hit = (disease_id, drug_id) in gt_set
            tier = pred.confidence_tier.name
            category = result.category

            # Overall tier
            if is_hit:
                tier_stats[tier]["hits"] += 1
            else:
                tier_stats[tier]["misses"] += 1

            # Per-category per-tier
            if is_hit:
                cat_tier_stats[category][tier]["hits"] += 1
            else:
                cat_tier_stats[category][tier]["misses"] += 1

            # MEDIUM-specific tracking
            if tier == "MEDIUM":
                cat_stats[category]["diseases"].add(disease_id)
                if is_hit:
                    cat_stats[category]["hits"] += 1
                else:
                    cat_stats[category]["misses"] += 1

    # Compute precisions
    results = {}
    for cat, stats in sorted(cat_stats.items()):
        total = stats["hits"] + stats["misses"]
        precision = stats["hits"] / total * 100 if total > 0 else 0
        results[cat] = {
            "hits": stats["hits"],
            "total": total,
            "precision": round(precision, 1),
            "n_diseases": len(stats["diseases"]),
        }

    tier_results = {}
    for tier, stats in tier_stats.items():
        total = stats["hits"] + stats["misses"]
        precision = stats["hits"] / total * 100 if total > 0 else 0
        tier_results[tier] = {
            "hits": stats["hits"],
            "total": total,
            "precision": round(precision, 1),
        }

    # Per-category per-tier
    cat_tier_results = {}
    for cat in sorted(cat_tier_stats.keys()):
        cat_tier_results[cat] = {}
        for tier in ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]:
            if tier in cat_tier_stats[cat]:
                stats = cat_tier_stats[cat][tier]
                total = stats["hits"] + stats["misses"]
                precision = stats["hits"] / total * 100 if total > 0 else 0
                cat_tier_results[cat][tier] = {
                    "hits": stats["hits"],
                    "total": total,
                    "precision": round(precision, 1),
                }

    return {
        "medium_by_category": results,
        "tier_precision": tier_results,
        "category_tier_precision": cat_tier_results,
    }


def main() -> None:
    seeds = [42, 123, 456, 789, 2024]
    print("=" * 70)
    print("h462: Category-Specific MEDIUM Tier Holdout Validation")
    print("=" * 70)

    print("\nLoading predictor...")
    predictor = DrugRepurposingPredictor()

    gt_path = predictor.reference_dir / "expanded_ground_truth.json"
    with open(gt_path) as f:
        gt_data = json.load(f)

    all_diseases = [d for d in predictor.ground_truth if d in predictor.embeddings]
    print(f"Diseases with GT + embeddings: {len(all_diseases)}")

    # --- FULL-DATA BASELINE ---
    print("\n--- FULL-DATA BASELINE ---")
    full_result = evaluate_medium_by_category(predictor, all_diseases, gt_data)

    print("\nMEDIUM tier precision by category (full-data):")
    print(f"  {'Category':<25s} {'Prec%':>6s} {'Hits':>5s} {'Total':>6s} {'Diseases':>8s}")
    print(f"  {'-'*55}")
    for cat, stats in sorted(full_result["medium_by_category"].items(), key=lambda x: -x[1]["precision"]):
        print(f"  {cat:<25s} {stats['precision']:5.1f}% {stats['hits']:5d} {stats['total']:6d} {stats['n_diseases']:8d}")

    # --- HOLDOUT VALIDATION ---
    all_holdout_medium_cat: Dict[str, List[float]] = defaultdict(list)
    all_holdout_medium_cat_n: Dict[str, List[int]] = defaultdict(list)
    all_holdout_medium_cat_hits: Dict[str, List[int]] = defaultdict(list)
    all_holdout_tier: Dict[str, List[float]] = defaultdict(list)

    for seed_idx, seed in enumerate(seeds):
        print(f"\n--- SEED {seed} ({seed_idx+1}/{len(seeds)}) ---")

        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        train_set = set(train_ids)

        originals = recompute_gt_structures(predictor, train_set)

        holdout_result = evaluate_medium_by_category(predictor, holdout_ids, gt_data)

        # Print seed results
        for cat, stats in sorted(holdout_result["medium_by_category"].items(), key=lambda x: -x[1]["precision"]):
            print(f"  {cat:<25s} {stats['precision']:5.1f}% ({stats['hits']}/{stats['total']}, {stats['n_diseases']} diseases)")
            all_holdout_medium_cat[cat].append(stats["precision"])
            all_holdout_medium_cat_n[cat].append(stats["total"])
            all_holdout_medium_cat_hits[cat].append(stats["hits"])

        # Track tier-level
        for tier, stats in holdout_result["tier_precision"].items():
            all_holdout_tier[tier].append(stats["precision"])

        restore_gt_structures(predictor, originals)

    # --- AGGREGATE ---
    print("\n" + "=" * 70)
    print("AGGREGATE HOLDOUT RESULTS (mean ± std across 5 seeds)")
    print("=" * 70)

    print("\n--- Overall Tier Precision ---")
    for tier in ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]:
        if tier in all_holdout_tier:
            mean_p = np.mean(all_holdout_tier[tier])
            std_p = np.std(all_holdout_tier[tier])
            print(f"  {tier:8s}: {mean_p:5.1f}% ± {std_p:4.1f}%")

    print("\n--- MEDIUM Tier by Category: Full-Data vs Holdout ---")
    print(f"  {'Category':<25s} {'Full%':>6s} {'Hold%':>7s} {'±std':>5s} {'Δ':>7s} {'FullN':>6s} {'HoldN':>6s} {'Decision':>10s}")
    print(f"  {'-'*80}")

    # Thresholds
    HIGH_THRESHOLD = 50.8  # holdout average for HIGH tier
    LOW_THRESHOLD = 12.2   # holdout average for LOW tier

    category_decisions = {}

    all_cats = set(full_result["medium_by_category"].keys()) | set(all_holdout_medium_cat.keys())
    cat_list = []
    for cat in sorted(all_cats):
        full_prec = full_result["medium_by_category"].get(cat, {}).get("precision", 0)
        full_n = full_result["medium_by_category"].get(cat, {}).get("total", 0)
        if cat in all_holdout_medium_cat and len(all_holdout_medium_cat[cat]) > 0:
            holdout_mean = np.mean(all_holdout_medium_cat[cat])
            holdout_std = np.std(all_holdout_medium_cat[cat])
            holdout_n = np.mean(all_holdout_medium_cat_n[cat])
            delta = holdout_mean - full_prec
        else:
            holdout_mean = 0
            holdout_std = 0
            holdout_n = 0
            delta = -full_prec

        # Decision logic
        if holdout_mean >= HIGH_THRESHOLD and holdout_n >= 5:
            decision = "→HIGH"
        elif holdout_mean < LOW_THRESHOLD and holdout_n >= 5:
            decision = "→LOW"
        elif holdout_n < 5:
            decision = "TOO_SMALL"
        else:
            decision = "KEEP"

        cat_list.append((cat, full_prec, holdout_mean, holdout_std, delta, full_n, holdout_n, decision))
        category_decisions[cat] = {
            "full_precision": full_prec,
            "holdout_mean": round(float(holdout_mean), 1),
            "holdout_std": round(float(holdout_std), 1),
            "delta": round(float(delta), 1),
            "full_n": int(full_n),
            "mean_holdout_n": round(float(holdout_n), 1),
            "decision": decision,
            "seed_values": [float(v) for v in all_holdout_medium_cat.get(cat, [])],
            "seed_hits": [int(v) for v in all_holdout_medium_cat_hits.get(cat, [])],
            "seed_n": [int(v) for v in all_holdout_medium_cat_n.get(cat, [])],
        }

    # Sort by holdout precision descending
    cat_list.sort(key=lambda x: -x[2])
    for cat, full_prec, holdout_mean, holdout_std, delta, full_n, holdout_n, decision in cat_list:
        print(f"  {cat:<25s} {full_prec:5.1f}% {holdout_mean:6.1f}% ±{holdout_std:4.1f} {delta:+6.1f}pp {full_n:6.0f} {holdout_n:6.0f} {decision:>10s}")

    # --- SPECIFIC PROMOTION/DEMOTION ANALYSIS ---
    print("\n" + "=" * 70)
    print("PROMOTION/DEMOTION RECOMMENDATIONS")
    print("=" * 70)

    promote_cats = [c for c, d in category_decisions.items() if d["decision"] == "→HIGH"]
    demote_cats = [c for c, d in category_decisions.items() if d["decision"] == "→LOW"]
    keep_cats = [c for c, d in category_decisions.items() if d["decision"] == "KEEP"]
    small_cats = [c for c, d in category_decisions.items() if d["decision"] == "TOO_SMALL"]

    if promote_cats:
        print("\nPROMOTE to HIGH:")
        for cat in promote_cats:
            d = category_decisions[cat]
            print(f"  {cat}: {d['holdout_mean']}% ± {d['holdout_std']}% (n={d['mean_holdout_n']:.0f})")
            print(f"    Per-seed: {d['seed_values']}")

    if demote_cats:
        print("\nDEMOTE to LOW:")
        for cat in demote_cats:
            d = category_decisions[cat]
            print(f"  {cat}: {d['holdout_mean']}% ± {d['holdout_std']}% (n={d['mean_holdout_n']:.0f})")
            print(f"    Per-seed: {d['seed_values']}")

    if keep_cats:
        print("\nKEEP as MEDIUM:")
        for cat in keep_cats:
            d = category_decisions[cat]
            print(f"  {cat}: {d['holdout_mean']}% ± {d['holdout_std']}%")

    if small_cats:
        print("\nTOO SMALL to evaluate (n<5 per seed):")
        for cat in small_cats:
            d = category_decisions[cat]
            print(f"  {cat}: {d['holdout_mean']}% (n={d['mean_holdout_n']:.0f})")

    # --- IMPACT ANALYSIS ---
    print("\n--- Impact of Category-Specific Rules ---")
    # Count how many predictions are affected
    total_medium = sum(d["full_n"] for d in category_decisions.values())
    for cat in promote_cats + demote_cats:
        d = category_decisions[cat]
        pct = d["full_n"] / total_medium * 100 if total_medium > 0 else 0
        print(f"  {cat} ({d['decision']}): {d['full_n']} predictions ({pct:.1f}% of MEDIUM)")

    # Save results
    output = {
        "category_decisions": category_decisions,
        "overall_tier_holdout": {
            tier: {
                "mean": round(float(np.mean(vals)), 1),
                "std": round(float(np.std(vals)), 1),
                "seeds": [float(v) for v in vals],
            }
            for tier, vals in all_holdout_tier.items()
        },
        "thresholds": {
            "HIGH_threshold": HIGH_THRESHOLD,
            "LOW_threshold": LOW_THRESHOLD,
        },
        "recommendations": {
            "promote_to_HIGH": promote_cats,
            "demote_to_LOW": demote_cats,
            "keep_MEDIUM": keep_cats,
            "too_small": small_cats,
        },
    }

    output_path = Path(__file__).parent.parent / "data" / "analysis" / "h462_category_medium_holdout.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
