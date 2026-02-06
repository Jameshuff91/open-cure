#!/usr/bin/env python3
"""
h533: FILTER Tier Precision Audit - Are We Over-Filtering Good Predictions?

Stratifies ~7,300 FILTER predictions by their filter reason and computes
holdout precision for each sub-rule. Goals:
1. Identify which FILTER reasons dominate
2. Find sub-populations with >15% holdout precision (could be LOW/MEDIUM)
3. Test TransE consilience within FILTER tier
4. Quantify rescue opportunity (how many predictions could be upgraded)
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
    HIERARCHY_EXCLUSIONS,
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

    # Recompute drug_train_freq
    new_freq: Dict[str, int] = defaultdict(int)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            for drug_id in predictor.ground_truth[disease_id]:
                new_freq[drug_id] += 1
    predictor.drug_train_freq = dict(new_freq)

    # Recompute drug_to_diseases
    new_d2d: Dict[str, Set[str]] = defaultdict(set)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            for drug_id in predictor.ground_truth[disease_id]:
                new_d2d[drug_id].add(disease_name.lower())
    predictor.drug_to_diseases = dict(new_d2d)

    # Recompute drug_cancer_types
    new_cancer: Dict[str, Set[str]] = defaultdict(set)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            cancer_types = extract_cancer_types(disease_name)
            if cancer_types:
                for drug_id in predictor.ground_truth[disease_id]:
                    new_cancer[drug_id].update(cancer_types)
    predictor.drug_cancer_types = dict(new_cancer)

    # Recompute drug_disease_groups
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

    # Rebuild kNN index
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


def evaluate_filter_detailed(
    predictor: DrugRepurposingPredictor,
    disease_ids: List[str],
    gt_data: Dict[str, List[str]],
    transe_top30: Dict[str, Set[str]] = None,
) -> Dict:
    """Evaluate FILTER tier with detailed sub-rule stratification.

    Returns per-filter-reason stats and TransE consilience analysis.
    """
    gt_set = set()
    for disease_id, drugs in gt_data.items():
        for drug in drugs:
            if isinstance(drug, str):
                gt_set.add((disease_id, drug))
            elif isinstance(drug, dict):
                gt_set.add((disease_id, drug.get("drug_id") or drug.get("drug")))

    # Track stats per filter reason
    filter_reason_stats = defaultdict(lambda: {"hits": 0, "misses": 0, "examples": []})
    # Track FILTER + TransE consilience
    filter_transe_stats = {"hits": 0, "misses": 0}
    filter_no_transe_stats = {"hits": 0, "misses": 0}
    # Track by category within FILTER
    filter_category_stats = defaultdict(lambda: {"hits": 0, "misses": 0})
    # Track all tiers for context
    tier_stats = defaultdict(lambda: {"hits": 0, "misses": 0})
    # Track rank distribution within FILTER
    filter_rank_stats = defaultdict(lambda: {"hits": 0, "misses": 0})

    for disease_id in disease_ids:
        disease_name = predictor.disease_names.get(disease_id, disease_id)

        try:
            result = predictor.predict(
                disease_name, top_n=30, include_filtered=True
            )
        except Exception:
            continue

        for pred in result.predictions:
            drug_id = pred.drug_id
            is_hit = (disease_id, drug_id) in gt_set
            tier = pred.confidence_tier.name

            if is_hit:
                tier_stats[tier]["hits"] += 1
            else:
                tier_stats[tier]["misses"] += 1

            if tier != "FILTER":
                continue

            # Determine the filter reason
            reason = pred.category_specific_tier or "standard_filter"
            # Map None to more descriptive reasons based on code logic
            if reason is None:
                reason = "standard_filter"

            if is_hit:
                filter_reason_stats[reason]["hits"] += 1
            else:
                filter_reason_stats[reason]["misses"] += 1

            # Track examples (first 3 per reason)
            if len(filter_reason_stats[reason]["examples"]) < 3:
                drug_name = pred.drug_name if hasattr(pred, 'drug_name') else drug_id
                filter_reason_stats[reason]["examples"].append({
                    "drug": drug_name,
                    "disease": disease_name,
                    "hit": is_hit,
                })

            # Category within FILTER
            cat = predictor.categorize_disease(disease_name)
            if is_hit:
                filter_category_stats[cat]["hits"] += 1
            else:
                filter_category_stats[cat]["misses"] += 1

            # Rank bucket
            rank = pred.knn_rank if hasattr(pred, 'knn_rank') else 0
            if rank <= 5:
                bucket = "1-5"
            elif rank <= 10:
                bucket = "6-10"
            elif rank <= 15:
                bucket = "11-15"
            elif rank <= 20:
                bucket = "16-20"
            else:
                bucket = "21-30"
            if is_hit:
                filter_rank_stats[bucket]["hits"] += 1
            else:
                filter_rank_stats[bucket]["misses"] += 1

            # TransE consilience
            if transe_top30 and disease_id in transe_top30:
                has_transe = drug_id in transe_top30[disease_id]
                if has_transe:
                    if is_hit:
                        filter_transe_stats["hits"] += 1
                    else:
                        filter_transe_stats["misses"] += 1
                else:
                    if is_hit:
                        filter_no_transe_stats["hits"] += 1
                    else:
                        filter_no_transe_stats["misses"] += 1

    return {
        "tier_stats": dict(tier_stats),
        "filter_reason_stats": dict(filter_reason_stats),
        "filter_transe_stats": dict(filter_transe_stats),
        "filter_no_transe_stats": dict(filter_no_transe_stats),
        "filter_category_stats": dict(filter_category_stats),
        "filter_rank_stats": dict(filter_rank_stats),
    }


def compute_precision(stats):
    total = stats["hits"] + stats["misses"]
    if total == 0:
        return 0.0, 0
    return stats["hits"] / total * 100, total


def main():
    seeds = [42, 123, 456, 789, 2024]

    print("=" * 80)
    print("h533: FILTER Tier Precision Audit - Are We Over-Filtering?")
    print("=" * 80)

    print("\nLoading predictor...")
    predictor = DrugRepurposingPredictor()

    gt_path = predictor.reference_dir / "expanded_ground_truth.json"
    with open(gt_path) as f:
        gt_data = json.load(f)
    print(f"Loaded {len(gt_data)} GT diseases, {sum(len(v) for v in gt_data.values())} pairs")

    all_diseases = [d for d in predictor.ground_truth if d in predictor.embeddings]
    print(f"Diseases with GT + embeddings: {len(all_diseases)}")

    # Load TransE top-30 if available
    transe_top30 = None
    try:
        transe_path = Path("models/transe.pt")
        if transe_path.exists() and hasattr(predictor, '_load_transe_model'):
            print("Loading TransE model for consilience analysis...")
            predictor._load_transe_model()
            if hasattr(predictor, 'transe_model') and predictor.transe_model is not None:
                # Build TransE top-30 per disease
                transe_top30 = {}
                for disease_id in all_diseases:
                    disease_name = predictor.disease_names.get(disease_id, disease_id)
                    top_drugs = predictor._get_transe_top_n(disease_name, n=30)
                    if top_drugs:
                        transe_top30[disease_id] = set(top_drugs)
                print(f"TransE top-30 loaded for {len(transe_top30)} diseases")
            else:
                print("TransE model not loadable, skipping consilience analysis")
        else:
            print("TransE model not available, skipping consilience analysis")
    except Exception as e:
        print(f"TransE loading failed: {e}, skipping consilience analysis")

    # === FULL-DATA ANALYSIS ===
    print("\n" + "=" * 80)
    print("FULL-DATA FILTER ANALYSIS (all diseases)")
    print("=" * 80)

    full_result = evaluate_filter_detailed(predictor, all_diseases, gt_data, transe_top30)

    print("\n--- Tier Overview ---")
    for tier in ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]:
        if tier in full_result["tier_stats"]:
            prec, n = compute_precision(full_result["tier_stats"][tier])
            print(f"  {tier:8s}: {prec:5.1f}% ({n:5d} predictions)")

    print("\n--- FILTER by Reason (Full-Data) ---")
    print(f"{'Reason':<35s} {'Prec':>6s} {'Hits':>5s} {'Total':>6s}")
    print("-" * 60)
    sorted_reasons = sorted(
        full_result["filter_reason_stats"].items(),
        key=lambda x: x[1]["hits"] + x[1]["misses"],
        reverse=True
    )
    for reason, stats in sorted_reasons:
        prec, n = compute_precision(stats)
        examples = stats.get("examples", [])
        print(f"  {reason:<33s} {prec:5.1f}% {stats['hits']:>5d} {n:>6d}")
        for ex in examples[:1]:
            print(f"    e.g. {ex['drug'][:30]} → {ex['disease'][:30]} ({'HIT' if ex['hit'] else 'miss'})")

    print("\n--- FILTER by Category (Full-Data) ---")
    print(f"{'Category':<25s} {'Prec':>6s} {'Hits':>5s} {'Total':>6s}")
    print("-" * 50)
    sorted_cats = sorted(
        full_result["filter_category_stats"].items(),
        key=lambda x: x[1]["hits"] + x[1]["misses"],
        reverse=True
    )
    for cat, stats in sorted_cats:
        prec, n = compute_precision(stats)
        print(f"  {cat:<23s} {prec:5.1f}% {stats['hits']:>5d} {n:>6d}")

    print("\n--- FILTER by Rank Bucket (Full-Data) ---")
    for bucket in ["1-5", "6-10", "11-15", "16-20", "21-30"]:
        if bucket in full_result["filter_rank_stats"]:
            prec, n = compute_precision(full_result["filter_rank_stats"][bucket])
            print(f"  Rank {bucket:>5s}: {prec:5.1f}% ({n:>5d} predictions)")

    if transe_top30:
        print("\n--- FILTER + TransE Consilience (Full-Data) ---")
        prec_with, n_with = compute_precision(full_result["filter_transe_stats"])
        prec_without, n_without = compute_precision(full_result["filter_no_transe_stats"])
        print(f"  FILTER + TransE top-30:  {prec_with:5.1f}% ({n_with} predictions)")
        print(f"  FILTER - TransE top-30:  {prec_without:5.1f}% ({n_without} predictions)")
        if n_with > 0 and n_without > 0:
            print(f"  Delta: {prec_with - prec_without:+.1f}pp")

    # === HOLDOUT ANALYSIS ===
    print("\n" + "=" * 80)
    print("HOLDOUT FILTER ANALYSIS (5 seeds)")
    print("=" * 80)

    # Aggregate holdout stats per reason
    holdout_reason_hits = defaultdict(list)
    holdout_reason_n = defaultdict(list)
    holdout_transe_hits = []
    holdout_transe_n = []
    holdout_no_transe_hits = []
    holdout_no_transe_n = []
    holdout_category_hits = defaultdict(list)
    holdout_category_n = defaultdict(list)
    holdout_rank_hits = defaultdict(list)
    holdout_rank_n = defaultdict(list)
    holdout_tier_prec = defaultdict(list)

    for seed_idx, seed in enumerate(seeds):
        print(f"\n--- Seed {seed} ({seed_idx+1}/{len(seeds)}) ---")

        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        train_set = set(train_ids)

        originals = recompute_gt_structures(predictor, train_set)

        result = evaluate_filter_detailed(predictor, holdout_ids, gt_data, transe_top30)

        # Tier precision
        for tier in ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]:
            if tier in result["tier_stats"]:
                prec, n = compute_precision(result["tier_stats"][tier])
                holdout_tier_prec[tier].append(prec)
                if tier == "FILTER":
                    print(f"  FILTER: {prec:.1f}% ({n} predictions)")

        # Per-reason holdout
        for reason, stats in result["filter_reason_stats"].items():
            prec, n = compute_precision(stats)
            holdout_reason_hits[reason].append(stats["hits"])
            holdout_reason_n[reason].append(n)

        # TransE
        if transe_top30:
            h, n = result["filter_transe_stats"]["hits"], result["filter_transe_stats"]["hits"] + result["filter_transe_stats"]["misses"]
            holdout_transe_hits.append(result["filter_transe_stats"]["hits"])
            holdout_transe_n.append(n)
            h2, n2 = result["filter_no_transe_stats"]["hits"], result["filter_no_transe_stats"]["hits"] + result["filter_no_transe_stats"]["misses"]
            holdout_no_transe_hits.append(result["filter_no_transe_stats"]["hits"])
            holdout_no_transe_n.append(n2)

        # Per-category holdout
        for cat, stats in result["filter_category_stats"].items():
            holdout_category_hits[cat].append(stats["hits"])
            holdout_category_n[cat].append(stats["hits"] + stats["misses"])

        # Per-rank holdout
        for bucket, stats in result["filter_rank_stats"].items():
            holdout_rank_hits[bucket].append(stats["hits"])
            holdout_rank_n[bucket].append(stats["hits"] + stats["misses"])

        restore_gt_structures(predictor, originals)

    # === AGGREGATE HOLDOUT RESULTS ===
    print("\n" + "=" * 80)
    print("AGGREGATE HOLDOUT RESULTS (mean ± std)")
    print("=" * 80)

    print("\n--- Tier Precision (holdout) ---")
    for tier in ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]:
        if tier in holdout_tier_prec:
            vals = holdout_tier_prec[tier]
            print(f"  {tier:8s}: {np.mean(vals):5.1f}% ± {np.std(vals):4.1f}%")

    print("\n--- FILTER by Reason (Holdout) ---")
    print(f"{'Reason':<35s} {'Holdout':>8s} {'±std':>6s} {'Full':>6s} {'n/seed':>7s} {'Rescue?':<10s}")
    print("-" * 80)

    # Sort by holdout precision
    reason_holdout_results = []
    for reason in holdout_reason_hits:
        hits = holdout_reason_hits[reason]
        ns = holdout_reason_n[reason]
        if sum(ns) == 0:
            continue
        # Per-seed precision
        per_seed_prec = [h/n*100 if n > 0 else 0 for h, n in zip(hits, ns)]
        mean_prec = np.mean(per_seed_prec)
        std_prec = np.std(per_seed_prec)
        avg_n = np.mean(ns)
        # Full data precision
        full_stats = full_result["filter_reason_stats"].get(reason, {"hits": 0, "misses": 0})
        full_prec, full_n = compute_precision(full_stats)

        rescue = "YES" if mean_prec > 15 and avg_n >= 10 else "maybe" if mean_prec > 10 else "no"

        reason_holdout_results.append((reason, mean_prec, std_prec, full_prec, avg_n, rescue))

    reason_holdout_results.sort(key=lambda x: x[1], reverse=True)
    for reason, mean_prec, std_prec, full_prec, avg_n, rescue in reason_holdout_results:
        print(f"  {reason:<33s} {mean_prec:5.1f}%  {std_prec:5.1f}% {full_prec:5.1f}% {avg_n:6.1f}  {rescue}")

    print("\n--- FILTER by Category (Holdout) ---")
    print(f"{'Category':<25s} {'Holdout':>8s} {'±std':>6s} {'Full':>6s} {'n/seed':>7s}")
    print("-" * 60)
    cat_results = []
    for cat in holdout_category_hits:
        hits = holdout_category_hits[cat]
        ns = holdout_category_n[cat]
        per_seed_prec = [h/n*100 if n > 0 else 0 for h, n in zip(hits, ns)]
        mean_prec = np.mean(per_seed_prec)
        std_prec = np.std(per_seed_prec)
        avg_n = np.mean(ns)
        full_stats = full_result["filter_category_stats"].get(cat, {"hits": 0, "misses": 0})
        full_prec, _ = compute_precision(full_stats)
        cat_results.append((cat, mean_prec, std_prec, full_prec, avg_n))

    cat_results.sort(key=lambda x: x[1], reverse=True)
    for cat, mean_prec, std_prec, full_prec, avg_n in cat_results:
        print(f"  {cat:<23s} {mean_prec:5.1f}%  {std_prec:5.1f}% {full_prec:5.1f}% {avg_n:6.1f}")

    print("\n--- FILTER by Rank Bucket (Holdout) ---")
    for bucket in ["1-5", "6-10", "11-15", "16-20", "21-30"]:
        if bucket in holdout_rank_hits:
            hits = holdout_rank_hits[bucket]
            ns = holdout_rank_n[bucket]
            per_seed_prec = [h/n*100 if n > 0 else 0 for h, n in zip(hits, ns)]
            mean_prec = np.mean(per_seed_prec)
            std_prec = np.std(per_seed_prec)
            avg_n = np.mean(ns)
            print(f"  Rank {bucket:>5s}: {mean_prec:5.1f}% ± {std_prec:4.1f}% (n={avg_n:.0f}/seed)")

    if transe_top30 and holdout_transe_n:
        print("\n--- FILTER + TransE Consilience (Holdout) ---")
        with_prec = [h/n*100 if n > 0 else 0 for h, n in zip(holdout_transe_hits, holdout_transe_n)]
        without_prec = [h/n*100 if n > 0 else 0 for h, n in zip(holdout_no_transe_hits, holdout_no_transe_n)]
        print(f"  FILTER + TransE:  {np.mean(with_prec):5.1f}% ± {np.std(with_prec):4.1f}% (n={np.mean(holdout_transe_n):.0f}/seed)")
        print(f"  FILTER - TransE:  {np.mean(without_prec):5.1f}% ± {np.std(without_prec):4.1f}% (n={np.mean(holdout_no_transe_n):.0f}/seed)")
        delta = np.mean(with_prec) - np.mean(without_prec)
        print(f"  Delta: {delta:+.1f}pp")
        if delta > 5 and np.mean(with_prec) > 15:
            print("  ** RESCUE CANDIDATE: FILTER + TransE could be promoted to LOW **")

    # === SUMMARY ===
    print("\n" + "=" * 80)
    print("SUMMARY: RESCUE CANDIDATES")
    print("=" * 80)

    rescue_candidates = [r for r in reason_holdout_results if r[5] == "YES"]
    if rescue_candidates:
        total_rescuable = 0
        for reason, mean_prec, std_prec, full_prec, avg_n, rescue in rescue_candidates:
            full_n = full_result["filter_reason_stats"].get(reason, {"hits": 0, "misses": 0})
            full_total = full_n["hits"] + full_n["misses"]
            total_rescuable += full_total
            print(f"  {reason}: {mean_prec:.1f}% holdout, {full_total} predictions → could be LOW")
        print(f"\n  Total potentially rescuable: {total_rescuable} predictions")
    else:
        print("  No FILTER sub-populations with >15% holdout precision (n>=10)")
        print("  FILTER tier appears appropriately calibrated")

    maybe_candidates = [r for r in reason_holdout_results if r[5] == "maybe"]
    if maybe_candidates:
        print("\n  Marginal candidates (10-15% holdout, comparable to LOW):")
        for reason, mean_prec, std_prec, full_prec, avg_n, rescue in maybe_candidates:
            full_n = full_result["filter_reason_stats"].get(reason, {"hits": 0, "misses": 0})
            full_total = full_n["hits"] + full_n["misses"]
            print(f"    {reason}: {mean_prec:.1f}% holdout, {full_total} predictions")

    # Save results
    output = {
        "full_data": {
            "filter_by_reason": {r: {"precision": p, "n": int(n)}
                                for r, stats in full_result["filter_reason_stats"].items()
                                for p, n in [compute_precision(stats)]},
        },
        "holdout": {
            "filter_by_reason": {r[0]: {"holdout_prec": r[1], "std": r[2], "full_prec": r[3], "avg_n": r[4], "rescue": r[5]}
                                for r in reason_holdout_results},
        },
        "rescue_candidates": [r[0] for r in rescue_candidates],
    }

    output_path = Path("data/analysis/h533_filter_precision_audit.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
