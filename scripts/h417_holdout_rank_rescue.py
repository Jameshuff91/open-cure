#!/usr/bin/env python3
"""
h417: Holdout Validation of Rank 21-30 Rescue Rules

Tests whether proposed rank 21-30 rescue rules generalize on holdout:
1. target_overlap>=3 at rank 21-30 → MEDIUM (35% full-data precision)
2. freq>=10 + mechanism at rank 21-30 → MEDIUM (27.3% full-data precision)

Uses same 80/20 split methodology as h393 with 5 seeds.
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

    # Recompute from training diseases only
    new_freq: Dict[str, int] = defaultdict(int)
    new_d2d: Dict[str, Set[str]] = defaultdict(set)
    new_cancer: Dict[str, Set[str]] = defaultdict(set)
    new_groups: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)

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


def restore_gt_structures(predictor: DrugRepurposingPredictor, originals: Dict) -> None:
    """Restore original GT-derived data structures."""
    for key, val in originals.items():
        setattr(predictor, key, val)


def evaluate_rank_rescue(
    predictor: DrugRepurposingPredictor,
    disease_ids: List[str],
    gt_data: Dict[str, List[str]],
) -> Dict:
    """Evaluate precision of rank 21-30 rescue candidates.

    Returns precision stats for:
    - Current tiers (baseline)
    - Proposed rescue rules at rank 21-30
    """
    gt_set = set()
    for disease_id, drugs in gt_data.items():
        for drug in drugs:
            if isinstance(drug, str):
                gt_set.add((disease_id, drug))
            elif isinstance(drug, dict):
                gt_set.add((disease_id, drug.get("drug_id") or drug.get("drug")))

    # Track current tier stats
    tier_stats: Dict[str, Dict] = defaultdict(lambda: {"hits": 0, "misses": 0})

    # Track rescue rule stats
    rescue_stats: Dict[str, Dict] = defaultdict(lambda: {"hits": 0, "misses": 0})

    # Track what tier rescued predictions WOULD go to
    rescue_tier_stats: Dict[str, Dict] = defaultdict(lambda: {"hits": 0, "misses": 0})

    for disease_id in disease_ids:
        disease_name = predictor.disease_names.get(disease_id, disease_id)

        try:
            result = predictor.predict(disease_name, top_n=30, include_filtered=True)
        except Exception:
            continue

        for pred in result.predictions:
            is_hit = (disease_id, pred.drug_id) in gt_set
            tier = pred.confidence_tier.name

            # Record current tier
            if is_hit:
                tier_stats[tier]["hits"] += 1
            else:
                tier_stats[tier]["misses"] += 1

            # Only analyze rank 21-30 (currently all FILTER)
            if pred.rank < 21 or pred.rank > 30:
                continue

            # Compute rescue features
            target_overlap = predictor._get_target_overlap_count(
                pred.drug_id, disease_id
            ) if pred.drug_id else 0

            # Rule 1: target_overlap >= 3
            if target_overlap >= 3:
                key = "overlap>=3"
                if is_hit:
                    rescue_stats[key]["hits"] += 1
                else:
                    rescue_stats[key]["misses"] += 1

            # Rule 2: freq>=10 + mechanism
            if pred.train_frequency >= 10 and pred.mechanism_support:
                key = "freq10_mech"
                if is_hit:
                    rescue_stats[key]["hits"] += 1
                else:
                    rescue_stats[key]["misses"] += 1

            # Rule 3: overlap>=3 OR (freq>=10 + mech)
            if target_overlap >= 3 or (pred.train_frequency >= 10 and pred.mechanism_support):
                key = "overlap3_OR_freq10mech"
                if is_hit:
                    rescue_stats[key]["hits"] += 1
                else:
                    rescue_stats[key]["misses"] += 1

            # Rule 4: overlap>=3 AND mechanism
            if target_overlap >= 3 and pred.mechanism_support:
                key = "overlap3_AND_mech"
                if is_hit:
                    rescue_stats[key]["hits"] += 1
                else:
                    rescue_stats[key]["misses"] += 1

            # Rule 5: overlap>=1 AND mechanism AND freq>=5
            if target_overlap >= 1 and pred.mechanism_support and pred.train_frequency >= 5:
                key = "overlap1_mech_freq5"
                if is_hit:
                    rescue_stats[key]["hits"] += 1
                else:
                    rescue_stats[key]["misses"] += 1

            # Simulated tier assignment if rescued to MEDIUM
            # Track impact on MEDIUM tier precision
            any_rescue = (
                target_overlap >= 3
                or (pred.train_frequency >= 10 and pred.mechanism_support)
            )
            if any_rescue:
                new_tier = "MEDIUM"
            else:
                new_tier = tier  # stays FILTER

            if is_hit:
                rescue_tier_stats[new_tier]["hits"] += 1
            else:
                rescue_tier_stats[new_tier]["misses"] += 1

    # Compute results
    result = {}

    result["current_tiers"] = {}
    for tier in ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]:
        stats = tier_stats.get(tier, {"hits": 0, "misses": 0})
        total = stats["hits"] + stats["misses"]
        prec = stats["hits"] / total * 100 if total else 0
        result["current_tiers"][tier] = {
            "hits": stats["hits"],
            "total": total,
            "precision": round(prec, 1),
        }

    result["rescue_rules"] = {}
    for rule, stats in rescue_stats.items():
        total = stats["hits"] + stats["misses"]
        prec = stats["hits"] / total * 100 if total else 0
        result["rescue_rules"][rule] = {
            "hits": stats["hits"],
            "total": total,
            "precision": round(prec, 1),
        }

    result["simulated_tiers"] = {}
    for tier in ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]:
        # Merge current (non-rank-21-30) and rescue tier stats
        current = tier_stats.get(tier, {"hits": 0, "misses": 0})
        rescue = rescue_tier_stats.get(tier, {"hits": 0, "misses": 0})

        # For non-rank-21-30, keep current stats
        # For rank-21-30, use rescue stats
        # The tier_stats already includes rank 21-30 as FILTER
        # So simulated = current_tiers - (FILTER rank21-30) + rescue_tiers
        # This is a bit tricky, so we just report the rescue_tier_stats for rank 21-30

    # Simpler: just show what MEDIUM and FILTER would look like
    result["rescue_impact"] = {}
    # Current MEDIUM
    med_cur = tier_stats.get("MEDIUM", {"hits": 0, "misses": 0})
    # Rescued predictions that would go to MEDIUM
    med_add = rescue_tier_stats.get("MEDIUM", {"hits": 0, "misses": 0})
    # Current FILTER minus rescued
    filt_cur = tier_stats.get("FILTER", {"hits": 0, "misses": 0})
    filt_sub = rescue_tier_stats.get("FILTER", {"hits": 0, "misses": 0})

    # After rescue: MEDIUM = current MEDIUM + rescued to MEDIUM
    med_new_total = med_cur["hits"] + med_cur["misses"] + med_add["hits"] + med_add["misses"]
    med_new_hits = med_cur["hits"] + med_add["hits"]
    # After rescue: FILTER = current FILTER - rescued from FILTER
    filt_new_total = (filt_cur["hits"] + filt_cur["misses"]) - (med_add["hits"] + med_add["misses"])
    filt_new_hits = filt_cur["hits"] - med_add["hits"]

    result["rescue_impact"]["MEDIUM_before"] = {
        "hits": med_cur["hits"],
        "total": med_cur["hits"] + med_cur["misses"],
        "precision": round(med_cur["hits"] / (med_cur["hits"] + med_cur["misses"]) * 100, 1) if (med_cur["hits"] + med_cur["misses"]) else 0,
    }
    result["rescue_impact"]["MEDIUM_after"] = {
        "hits": med_new_hits,
        "total": med_new_total,
        "precision": round(med_new_hits / med_new_total * 100, 1) if med_new_total else 0,
    }
    result["rescue_impact"]["FILTER_before"] = {
        "hits": filt_cur["hits"],
        "total": filt_cur["hits"] + filt_cur["misses"],
        "precision": round(filt_cur["hits"] / (filt_cur["hits"] + filt_cur["misses"]) * 100, 1) if (filt_cur["hits"] + filt_cur["misses"]) else 0,
    }
    result["rescue_impact"]["FILTER_after"] = {
        "hits": filt_new_hits,
        "total": filt_new_total,
        "precision": round(filt_new_hits / filt_new_total * 100, 1) if filt_new_total else 0,
    }

    return result


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
    print(f"Total diseases with GT + embeddings: {len(all_diseases)}")

    # === Full-data evaluation ===
    print("\n=== FULL-DATA EVALUATION ===")
    full_results = evaluate_rank_rescue(predictor, all_diseases, gt_data)

    print("\nCurrent tier precision:")
    for tier in ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]:
        stats = full_results["current_tiers"].get(tier, {})
        print(f"  {tier:8s}: hits={stats.get('hits', 0):4d}, total={stats.get('total', 0):5d}, precision={stats.get('precision', 0):5.1f}%")

    print("\nRescue rule precision (rank 21-30 only):")
    for rule, stats in full_results["rescue_rules"].items():
        print(f"  {rule:30s}: hits={stats['hits']:4d}, total={stats['total']:5d}, precision={stats['precision']:5.1f}%")

    print("\nRescue impact on MEDIUM/FILTER:")
    for key in ["MEDIUM_before", "MEDIUM_after", "FILTER_before", "FILTER_after"]:
        stats = full_results["rescue_impact"].get(key, {})
        print(f"  {key:20s}: hits={stats.get('hits', 0):4d}, total={stats.get('total', 0):5d}, precision={stats.get('precision', 0):5.1f}%")

    # === 5-seed holdout evaluation ===
    seeds = [42, 123, 456, 789, 2024]
    holdout_results = []

    for seed in seeds:
        print(f"\n=== SEED {seed} ===")
        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        print(f"  Train: {len(train_ids)}, Holdout: {len(holdout_ids)}")

        # Recompute GT structures from training only
        originals = recompute_gt_structures(predictor, set(train_ids))

        # Evaluate on holdout
        holdout_result = evaluate_rank_rescue(predictor, holdout_ids, gt_data)
        holdout_results.append(holdout_result)

        print(f"  Holdout tier precision:")
        for tier in ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]:
            stats = holdout_result["current_tiers"].get(tier, {})
            print(f"    {tier:8s}: {stats.get('precision', 0):5.1f}% (n={stats.get('total', 0)})")

        print(f"  Holdout rescue rule precision:")
        for rule, stats in holdout_result["rescue_rules"].items():
            print(f"    {rule:30s}: {stats['precision']:5.1f}% (n={stats['total']})")

        print(f"  Holdout rescue impact:")
        for key in ["MEDIUM_before", "MEDIUM_after", "FILTER_before", "FILTER_after"]:
            stats = holdout_result["rescue_impact"].get(key, {})
            print(f"    {key:20s}: {stats.get('precision', 0):5.1f}% (n={stats.get('total', 0)})")

        # Restore
        restore_gt_structures(predictor, originals)

    # === SUMMARY: Average across seeds ===
    print("\n" + "=" * 70)
    print("=== HOLDOUT SUMMARY (5-seed average) ===")
    print("=" * 70)

    # Average rescue rule precision across seeds
    all_rules = set()
    for hr in holdout_results:
        all_rules.update(hr["rescue_rules"].keys())

    print("\nRescue rule precision (holdout avg ± std):")
    for rule in sorted(all_rules):
        precisions = []
        counts = []
        for hr in holdout_results:
            stats = hr["rescue_rules"].get(rule, {"precision": 0, "total": 0})
            precisions.append(stats["precision"])
            counts.append(stats["total"])
        mean_p = np.mean(precisions)
        std_p = np.std(precisions)
        mean_n = np.mean(counts)
        full_p = full_results["rescue_rules"].get(rule, {}).get("precision", 0)
        delta = mean_p - full_p
        print(f"  {rule:30s}: {mean_p:5.1f}% ± {std_p:4.1f}% (n≈{mean_n:.0f})  [full={full_p:.1f}%, delta={delta:+.1f}pp]")

    # Average tier impact
    print("\nTier impact (holdout avg ± std):")
    for key in ["MEDIUM_before", "MEDIUM_after", "FILTER_before", "FILTER_after"]:
        precisions = []
        for hr in holdout_results:
            stats = hr["rescue_impact"].get(key, {"precision": 0})
            precisions.append(stats["precision"])
        mean_p = np.mean(precisions)
        std_p = np.std(precisions)
        full_p = full_results["rescue_impact"].get(key, {}).get("precision", 0)
        delta = mean_p - full_p
        print(f"  {key:20s}: {mean_p:5.1f}% ± {std_p:4.1f}%  [full={full_p:.1f}%, delta={delta:+.1f}pp]")

    # Average tier precision
    print("\nCurrent tier precision (holdout avg ± std):")
    for tier in ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]:
        precisions = []
        for hr in holdout_results:
            stats = hr["current_tiers"].get(tier, {"precision": 0})
            precisions.append(stats["precision"])
        mean_p = np.mean(precisions)
        std_p = np.std(precisions)
        full_p = full_results["current_tiers"].get(tier, {}).get("precision", 0)
        print(f"  {tier:8s}: {mean_p:5.1f}% ± {std_p:4.1f}%  [full={full_p:.1f}%]")

    # Save results
    output = {
        "full_data": full_results,
        "holdout_seeds": seeds,
        "holdout_results": holdout_results,
    }
    out_path = Path(__file__).parent.parent / "data" / "analysis" / "h417_holdout_rank_rescue.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
