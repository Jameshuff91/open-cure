#!/usr/bin/env python3
"""
h423: Category-Specific Rank Cutoffs Instead of Global rank>20

h417 showed massive precision variation by category at rank 21-30:
- psychiatric 28%, immunological 22% vs neurological 1.8%, musculoskeletal 1.3%

This script analyzes whether category-specific rank cutoffs could improve
tier precision compared to the global rank>20 FILTER.

Analysis:
1. Precision by rank bucket (1-5, 6-10, 11-15, 16-20, 21-25, 26-30) per category
2. Identify optimal rank cutoff per category
3. Compute expected tier precision with category-specific cutoffs
4. Holdout validation
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


def collect_predictions(
    predictor: DrugRepurposingPredictor,
    disease_ids: List[str],
    gt_set: Set[Tuple[str, str]],
) -> List[Dict]:
    """Collect all predictions with rank, category, and GT hit info."""
    preds = []
    for disease_id in disease_ids:
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        try:
            result = predictor.predict(disease_name, top_n=30, include_filtered=True)
        except Exception:
            continue

        for pred in result.predictions:
            is_hit = (disease_id, pred.drug_id) in gt_set
            preds.append({
                'rank': pred.rank,
                'category': pred.category,
                'is_hit': is_hit,
                'tier': pred.confidence_tier.name,
                'rule': pred.category_specific_tier,
            })
    return preds


def analyze_category_rank_buckets(preds: List[Dict]) -> Dict:
    """Analyze precision by rank bucket per category."""
    buckets = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 30)]

    # Category -> bucket -> {hits, total}
    stats: Dict[str, Dict[str, Dict]] = defaultdict(
        lambda: {f"{lo}-{hi}": {"hits": 0, "total": 0} for lo, hi in buckets}
    )
    # Also track overall
    overall: Dict[str, Dict] = {f"{lo}-{hi}": {"hits": 0, "total": 0} for lo, hi in buckets}

    for p in preds:
        cat = p['category']
        rank = p['rank']
        for lo, hi in buckets:
            if lo <= rank <= hi:
                bucket_key = f"{lo}-{hi}"
                stats[cat][bucket_key]["total"] += 1
                if p['is_hit']:
                    stats[cat][bucket_key]["hits"] += 1
                overall[bucket_key]["total"] += 1
                if p['is_hit']:
                    overall[bucket_key]["hits"] += 1
                break

    return {"by_category": dict(stats), "overall": overall}


def find_optimal_cutoffs(bucket_stats: Dict, filter_threshold: float = 12.0) -> Dict[str, int]:
    """Find the optimal rank cutoff per category.

    The cutoff is the highest rank where precision is still >= filter_threshold.
    If a category's precision at rank 16-20 is below threshold, cutoff is 15.
    If even rank 1-5 is below threshold, cutoff is 5 (minimum).

    filter_threshold: precision below which we should FILTER
    Using 12% as threshold (approx LOW tier precision)
    """
    cutoffs = {}
    buckets_ordered = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 30)]

    for cat, cat_stats in bucket_stats["by_category"].items():
        best_cutoff = 5  # minimum
        for lo, hi in buckets_ordered:
            bucket_key = f"{lo}-{hi}"
            stats = cat_stats.get(bucket_key, {"hits": 0, "total": 0})
            if stats["total"] < 10:
                continue  # Too few to judge
            precision = stats["hits"] / stats["total"] * 100
            if precision >= filter_threshold:
                best_cutoff = hi
            else:
                break  # Stop extending once precision drops below threshold

        cutoffs[cat] = best_cutoff

    return cutoffs


def simulate_cutoffs(
    preds: List[Dict],
    cutoffs: Dict[str, int],
    default_cutoff: int = 20,
) -> Dict:
    """Simulate tier precision with category-specific cutoffs.

    Returns precision for FILTER and would-be-rescued predictions.
    """
    # Current: everything rank>20 is FILTER
    # Proposed: rank > cutoffs[category] is FILTER

    current_filter = {"hits": 0, "total": 0}
    proposed_filter = {"hits": 0, "total": 0}
    rescued_from_filter = {"hits": 0, "total": 0}
    newly_filtered = {"hits": 0, "total": 0}

    for p in preds:
        cat = p['category']
        rank = p['rank']
        cutoff = cutoffs.get(cat, default_cutoff)

        currently_filtered = rank > 20
        would_be_filtered = rank > cutoff

        if currently_filtered:
            current_filter["total"] += 1
            if p['is_hit']:
                current_filter["hits"] += 1

        if would_be_filtered:
            proposed_filter["total"] += 1
            if p['is_hit']:
                proposed_filter["hits"] += 1

        # Rescued: currently FILTER, proposed NOT FILTER
        if currently_filtered and not would_be_filtered:
            rescued_from_filter["total"] += 1
            if p['is_hit']:
                rescued_from_filter["hits"] += 1

        # Newly filtered: currently NOT FILTER, proposed FILTER
        if not currently_filtered and would_be_filtered:
            newly_filtered["total"] += 1
            if p['is_hit']:
                newly_filtered["hits"] += 1

    return {
        "current_filter": current_filter,
        "proposed_filter": proposed_filter,
        "rescued_from_filter": rescued_from_filter,
        "newly_filtered": newly_filtered,
    }


def main():
    print("Loading predictor...")
    predictor = DrugRepurposingPredictor()

    gt_path = predictor.reference_dir / "expanded_ground_truth.json"
    with open(gt_path) as f:
        gt_data = json.load(f)

    gt_set = set()
    for disease_id, drugs in gt_data.items():
        for drug in drugs:
            if isinstance(drug, str):
                gt_set.add((disease_id, drug))
            elif isinstance(drug, dict):
                gt_set.add((disease_id, drug.get("drug_id") or drug.get("drug")))

    all_diseases = [
        d for d in predictor.disease_names.keys()
        if d in predictor.ground_truth and d in predictor.embeddings
    ]
    print(f"Total diseases: {len(all_diseases)}")

    # === FULL DATA ANALYSIS ===
    print("\n=== FULL-DATA: Collecting predictions ===")
    preds = collect_predictions(predictor, all_diseases, gt_set)
    print(f"Total predictions: {len(preds)}")

    bucket_stats = analyze_category_rank_buckets(preds)

    # Print results
    print("\n=== PRECISION BY RANK BUCKET PER CATEGORY ===")
    print(f"{'Category':25s} | {'1-5':>10s} | {'6-10':>10s} | {'11-15':>10s} | {'16-20':>10s} | {'21-25':>10s} | {'26-30':>10s} |")
    print("-" * 100)

    # Sort by precision at rank 21-30
    cats_sorted = []
    for cat, stats in bucket_stats["by_category"].items():
        r21_25 = stats.get("21-25", {"hits": 0, "total": 0})
        r26_30 = stats.get("26-30", {"hits": 0, "total": 0})
        total_21_30 = r21_25["total"] + r26_30["total"]
        hits_21_30 = r21_25["hits"] + r26_30["hits"]
        prec_21_30 = hits_21_30 / total_21_30 * 100 if total_21_30 else 0
        cats_sorted.append((cat, prec_21_30))
    cats_sorted.sort(key=lambda x: -x[1])

    for cat, _ in cats_sorted:
        stats = bucket_stats["by_category"][cat]
        row = f"{cat:25s} |"
        for bucket in ["1-5", "6-10", "11-15", "16-20", "21-25", "26-30"]:
            s = stats.get(bucket, {"hits": 0, "total": 0})
            if s["total"] > 0:
                prec = s["hits"] / s["total"] * 100
                row += f" {prec:5.1f}%({s['total']:3d}) |"
            else:
                row += f"      n/a  |"
        print(row)

    # Overall
    print("-" * 100)
    row = f"{'OVERALL':25s} |"
    for bucket in ["1-5", "6-10", "11-15", "16-20", "21-25", "26-30"]:
        s = bucket_stats["overall"].get(bucket, {"hits": 0, "total": 0})
        if s["total"] > 0:
            prec = s["hits"] / s["total"] * 100
            row += f" {prec:5.1f}%({s['total']:3d}) |"
        else:
            row += f"      n/a  |"
    print(row)

    # Find optimal cutoffs
    # Using LOW tier precision (13%) as threshold
    cutoffs = find_optimal_cutoffs(bucket_stats, filter_threshold=13.0)

    print(f"\n=== PROPOSED CATEGORY-SPECIFIC CUTOFFS (threshold: 13% ≈ LOW tier) ===")
    for cat, cutoff in sorted(cutoffs.items(), key=lambda x: -x[1]):
        current = 20
        change = cutoff - current
        marker = ""
        if change > 0:
            marker = f" (EXTEND +{change} ranks)"
        elif change < 0:
            marker = f" (TIGHTEN {change} ranks)"
        print(f"  {cat:25s}: rank <= {cutoff:2d}{marker}")

    # Simulate impact of category-specific cutoffs
    print(f"\n=== IMPACT SIMULATION ===")
    sim = simulate_cutoffs(preds, cutoffs)

    for key, stats in sim.items():
        if stats["total"] > 0:
            prec = stats["hits"] / stats["total"] * 100
            print(f"  {key:25s}: hits={stats['hits']:4d}, total={stats['total']:5d}, precision={prec:5.1f}%")
        else:
            print(f"  {key:25s}: n=0")

    # === HOLDOUT VALIDATION ===
    print("\n\n=== HOLDOUT VALIDATION (5-seed) ===")
    seeds = [42, 123, 456, 789, 2024]

    holdout_rescued_precs = []
    holdout_new_filter_precs = []
    holdout_filter_before = []
    holdout_filter_after = []

    for seed in seeds:
        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        originals = recompute_gt_structures(predictor, set(train_ids))

        holdout_preds = collect_predictions(predictor, holdout_ids, gt_set)
        holdout_buckets = analyze_category_rank_buckets(holdout_preds)

        # Derive cutoffs from TRAINING data
        train_preds = collect_predictions(predictor, train_ids, gt_set)
        train_buckets = analyze_category_rank_buckets(train_preds)
        train_cutoffs = find_optimal_cutoffs(train_buckets, filter_threshold=13.0)

        # Apply training-derived cutoffs to holdout data
        holdout_sim = simulate_cutoffs(holdout_preds, train_cutoffs)

        rescued = holdout_sim["rescued_from_filter"]
        new_filt = holdout_sim["newly_filtered"]
        cur_filt = holdout_sim["current_filter"]
        prop_filt = holdout_sim["proposed_filter"]

        rescued_prec = rescued["hits"] / rescued["total"] * 100 if rescued["total"] else 0
        new_filt_prec = new_filt["hits"] / new_filt["total"] * 100 if new_filt["total"] else 0
        cur_filt_prec = cur_filt["hits"] / cur_filt["total"] * 100 if cur_filt["total"] else 0
        prop_filt_prec = prop_filt["hits"] / prop_filt["total"] * 100 if prop_filt["total"] else 0

        holdout_rescued_precs.append(rescued_prec)
        holdout_new_filter_precs.append(new_filt_prec)
        holdout_filter_before.append(cur_filt_prec)
        holdout_filter_after.append(prop_filt_prec)

        print(f"\n  Seed {seed}:")
        print(f"    Train cutoffs: {dict(sorted(train_cutoffs.items(), key=lambda x: -x[1]))}")
        print(f"    Rescued from FILTER: n={rescued['total']}, hits={rescued['hits']}, prec={rescued_prec:.1f}%")
        print(f"    Newly filtered:      n={new_filt['total']}, hits={new_filt['hits']}, prec={new_filt_prec:.1f}%")
        print(f"    FILTER before:       n={cur_filt['total']}, prec={cur_filt_prec:.1f}%")
        print(f"    FILTER after:        n={prop_filt['total']}, prec={prop_filt_prec:.1f}%")

        restore_gt_structures(predictor, originals)

    # Summary
    print(f"\n\n{'=' * 70}")
    print("=== HOLDOUT SUMMARY ===")
    print(f"{'=' * 70}")
    print(f"  Rescued predictions precision: {np.mean(holdout_rescued_precs):.1f}% ± {np.std(holdout_rescued_precs):.1f}%")
    print(f"  Newly filtered precision:      {np.mean(holdout_new_filter_precs):.1f}% ± {np.std(holdout_new_filter_precs):.1f}%")
    print(f"  FILTER before (rank>20):       {np.mean(holdout_filter_before):.1f}% ± {np.std(holdout_filter_before):.1f}%")
    print(f"  FILTER after (cat-specific):   {np.mean(holdout_filter_after):.1f}% ± {np.std(holdout_filter_after):.1f}%")

    rescued_above_filter = np.mean(holdout_rescued_precs) > np.mean(holdout_filter_before)
    newly_filtered_below = np.mean(holdout_new_filter_precs) < np.mean(holdout_filter_before)

    print(f"\n  Decision criteria:")
    print(f"    Rescued precision > old FILTER: {rescued_above_filter} ({np.mean(holdout_rescued_precs):.1f}% vs {np.mean(holdout_filter_before):.1f}%)")
    print(f"    Newly filtered < old FILTER:    {newly_filtered_below} ({np.mean(holdout_new_filter_precs):.1f}% vs {np.mean(holdout_filter_before):.1f}%)")

    if rescued_above_filter and newly_filtered_below:
        print(f"\n  RECOMMENDATION: IMPLEMENT category-specific cutoffs")
        print(f"  Both criteria met: rescued predictions ARE better, newly filtered ARE worse")
    elif rescued_above_filter:
        print(f"\n  RECOMMENDATION: PARTIAL - rescued predictions are good but newly filtered ones aren't clearly bad")
    else:
        print(f"\n  RECOMMENDATION: DO NOT IMPLEMENT - rescued predictions aren't better than current FILTER")

    # Save
    output = {
        "bucket_stats": bucket_stats,
        "cutoffs": cutoffs,
        "simulation": sim,
        "holdout_rescued_precs": holdout_rescued_precs,
        "holdout_new_filter_precs": holdout_new_filter_precs,
    }
    out_path = Path(__file__).parent.parent / "data" / "analysis" / "h423_category_rank_cutoffs.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
