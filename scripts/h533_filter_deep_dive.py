#!/usr/bin/env python3
"""
h533 Deep Dive: Cross-tabulate FILTER reason × category and standard_filter sub-reasons.

The initial audit showed strong category signals within FILTER:
- respiratory: 27.6% holdout, autoimmune: 15.2%, cardiovascular: 20.3%, endocrine: 17.2%
These exceed LOW tier average (15.6%).

This script digs deeper:
1. Cross-tabulate filter-reason × category
2. Break down "standard_filter" into sub-reasons (rank>20, low_freq, no_targets)
3. Test category-specific rescue within standard_filter
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
    HIERARCHY_EXCLUSIONS,
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
    predictor.drug_train_freq = originals["drug_train_freq"]
    predictor.drug_to_diseases = originals["drug_to_diseases"]
    predictor.drug_cancer_types = originals["drug_cancer_types"]
    predictor.drug_disease_groups = originals["drug_disease_groups"]
    predictor.train_diseases = originals["train_diseases"]
    predictor.train_embeddings = originals["train_embeddings"]
    predictor.train_disease_categories = originals["train_disease_categories"]


def evaluate_filter_cross(predictor, disease_ids, gt_data):
    """Detailed FILTER analysis with reason × category cross-tabulation.

    Also breaks down "standard_filter" into sub-reasons:
    - rank_gt20: rank > 20
    - no_targets: drug has no known targets
    - low_freq_no_mech: freq <= 2 and no mechanism support
    """
    gt_set = set()
    for disease_id, drugs in gt_data.items():
        for drug in drugs:
            if isinstance(drug, str):
                gt_set.add((disease_id, drug))
            elif isinstance(drug, dict):
                gt_set.add((disease_id, drug.get("drug_id") or drug.get("drug")))

    # reason × category
    cross_stats = defaultdict(lambda: defaultdict(lambda: {"hits": 0, "misses": 0}))
    # standard_filter sub-reasons × category
    std_sub_stats = defaultdict(lambda: defaultdict(lambda: {"hits": 0, "misses": 0}))
    # ATC coherence within standard_filter × category
    atc_in_filter = defaultdict(lambda: defaultdict(lambda: {"hits": 0, "misses": 0}))
    # SOC drug class within standard_filter
    soc_in_filter = defaultdict(lambda: {"hits": 0, "misses": 0})
    # Frequency distribution within standard_filter
    freq_bucket_stats = defaultdict(lambda: defaultdict(lambda: {"hits": 0, "misses": 0}))

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

            if tier != "FILTER":
                continue

            reason = pred.category_specific_tier or "standard_filter"
            cat = predictor.categorize_disease(disease_name)
            drug_name = pred.drug_name if hasattr(pred, 'drug_name') else ""

            key = "hits" if is_hit else "misses"
            cross_stats[reason][cat][key] += 1

            # For standard_filter, determine sub-reason
            if reason == "standard_filter" or reason is None:
                rank = pred.knn_rank if hasattr(pred, 'knn_rank') else 0
                freq = predictor.drug_train_freq.get(drug_id, 0)
                has_targets = bool(predictor.drug_targets.get(drug_id))
                has_mechanism = pred.mechanism_support if hasattr(pred, 'mechanism_support') else False

                if rank > 20:
                    sub_reason = "rank_gt20"
                elif not has_targets:
                    sub_reason = "no_targets"
                elif freq <= 2 and not has_mechanism:
                    sub_reason = "low_freq_no_mech"
                else:
                    sub_reason = "other_standard"

                std_sub_stats[sub_reason][cat][key] += 1

                # Check ATC coherence
                is_coherent = predictor._is_atc_coherent(drug_name, cat) if drug_name else False
                atc_key = "coherent" if is_coherent else "incoherent"
                atc_in_filter[atc_key][cat][key] += 1

                # Frequency bucket
                if freq <= 2:
                    fbucket = "0-2"
                elif freq <= 5:
                    fbucket = "3-5"
                elif freq <= 10:
                    fbucket = "6-10"
                else:
                    fbucket = "11+"
                freq_bucket_stats[fbucket][cat][key] += 1

                # SOC drug class check
                soc_class = getattr(pred, 'soc_drug_class', None)
                if soc_class:
                    soc_in_filter[soc_class][key] += 1

    return {
        "cross_stats": {r: dict(cats) for r, cats in cross_stats.items()},
        "std_sub_stats": {r: dict(cats) for r, cats in std_sub_stats.items()},
        "atc_in_filter": {k: dict(cats) for k, cats in atc_in_filter.items()},
        "soc_in_filter": dict(soc_in_filter),
        "freq_bucket_stats": {k: dict(cats) for k, cats in freq_bucket_stats.items()},
    }


def prec(stats):
    total = stats["hits"] + stats["misses"]
    if total == 0:
        return 0.0, 0
    return stats["hits"] / total * 100, total


def main():
    seeds = [42, 123, 456, 789, 2024]

    print("=" * 80)
    print("h533 Deep Dive: FILTER Reason × Category Cross-Analysis")
    print("=" * 80)

    print("\nLoading predictor...")
    predictor = DrugRepurposingPredictor()

    gt_path = predictor.reference_dir / "expanded_ground_truth.json"
    with open(gt_path) as f:
        gt_data = json.load(f)

    all_diseases = [d for d in predictor.ground_truth if d in predictor.embeddings]
    print(f"Diseases: {len(all_diseases)}")

    # === FULL-DATA ===
    print("\n" + "=" * 80)
    print("FULL-DATA: Standard Filter Sub-Reasons × Category")
    print("=" * 80)

    full = evaluate_filter_cross(predictor, all_diseases, gt_data)

    print("\n--- Standard Filter Sub-Reasons ---")
    for sub_reason in ["rank_gt20", "no_targets", "low_freq_no_mech", "other_standard"]:
        if sub_reason not in full["std_sub_stats"]:
            continue
        total_hits = sum(s["hits"] for s in full["std_sub_stats"][sub_reason].values())
        total_n = sum(s["hits"] + s["misses"] for s in full["std_sub_stats"][sub_reason].values())
        p = total_hits / total_n * 100 if total_n > 0 else 0
        print(f"\n  {sub_reason}: {p:.1f}% ({total_hits}/{total_n})")
        for cat in sorted(full["std_sub_stats"][sub_reason].keys()):
            stats = full["std_sub_stats"][sub_reason][cat]
            p2, n2 = prec(stats)
            if n2 >= 10:
                print(f"    {cat:<23s}: {p2:5.1f}% ({n2:4d})")

    print("\n--- ATC Coherence within Standard Filter ---")
    for atc_key in ["coherent", "incoherent"]:
        if atc_key not in full["atc_in_filter"]:
            continue
        total_hits = sum(s["hits"] for s in full["atc_in_filter"][atc_key].values())
        total_n = sum(s["hits"] + s["misses"] for s in full["atc_in_filter"][atc_key].values())
        p = total_hits / total_n * 100 if total_n > 0 else 0
        print(f"\n  {atc_key}: {p:.1f}% ({total_hits}/{total_n})")
        for cat in sorted(full["atc_in_filter"][atc_key].keys()):
            stats = full["atc_in_filter"][atc_key][cat]
            p2, n2 = prec(stats)
            if n2 >= 10:
                print(f"    {cat:<23s}: {p2:5.1f}% ({n2:4d})")

    print("\n--- Frequency Bucket within Standard Filter ---")
    for fbucket in ["0-2", "3-5", "6-10", "11+"]:
        if fbucket not in full["freq_bucket_stats"]:
            continue
        total_hits = sum(s["hits"] for s in full["freq_bucket_stats"][fbucket].values())
        total_n = sum(s["hits"] + s["misses"] for s in full["freq_bucket_stats"][fbucket].values())
        p = total_hits / total_n * 100 if total_n > 0 else 0
        print(f"\n  freq {fbucket}: {p:.1f}% ({total_hits}/{total_n})")

    # === HOLDOUT ===
    print("\n" + "=" * 80)
    print("HOLDOUT: Standard Filter Sub-Reasons × Category (5 seeds)")
    print("=" * 80)

    # Aggregate holdout per sub-reason × category
    holdout_sub_cat = defaultdict(lambda: defaultdict(list))  # sub_reason -> cat -> [prec per seed]
    holdout_sub_cat_n = defaultdict(lambda: defaultdict(list))
    holdout_atc_cat = defaultdict(lambda: defaultdict(list))
    holdout_atc_cat_n = defaultdict(lambda: defaultdict(list))

    for seed_idx, seed in enumerate(seeds):
        print(f"\n  Seed {seed} ({seed_idx+1}/5)")
        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        originals = recompute_gt_structures(predictor, set(train_ids))

        result = evaluate_filter_cross(predictor, holdout_ids, gt_data)

        for sub_reason, cats in result["std_sub_stats"].items():
            for cat, stats in cats.items():
                p, n = prec(stats)
                holdout_sub_cat[sub_reason][cat].append(p)
                holdout_sub_cat_n[sub_reason][cat].append(n)

        for atc_key, cats in result["atc_in_filter"].items():
            for cat, stats in cats.items():
                p, n = prec(stats)
                holdout_atc_cat[atc_key][cat].append(p)
                holdout_atc_cat_n[atc_key][cat].append(n)

        restore_gt_structures(predictor, originals)

    print("\n" + "=" * 80)
    print("AGGREGATE HOLDOUT: Sub-Reason × Category")
    print("=" * 80)

    # Focus on rank>20 (the biggest bucket)
    print("\n--- rank_gt20 × Category (Holdout) ---")
    print(f"{'Category':<25s} {'Hold':>6s} {'±std':>6s} {'Full':>6s} {'n/seed':>7s} {'Rescue?'}")
    print("-" * 65)

    rescue_candidates = []
    for sub_reason in ["rank_gt20", "no_targets", "low_freq_no_mech"]:
        if sub_reason not in holdout_sub_cat:
            continue
        print(f"\n  === {sub_reason} ===")
        items = []
        for cat in holdout_sub_cat[sub_reason]:
            vals = holdout_sub_cat[sub_reason][cat]
            ns = holdout_sub_cat_n[sub_reason][cat]
            if len(vals) < 3:
                continue
            mean_p = np.mean(vals)
            std_p = np.std(vals)
            avg_n = np.mean(ns)
            # Full data
            full_stats = full["std_sub_stats"].get(sub_reason, {}).get(cat, {"hits": 0, "misses": 0})
            full_p, full_n = prec(full_stats)
            rescue = "YES" if mean_p > 15 and avg_n >= 10 else "maybe" if mean_p > 10 else ""
            items.append((cat, mean_p, std_p, full_p, avg_n, rescue, full_n))
            if rescue == "YES":
                rescue_candidates.append((sub_reason, cat, mean_p, std_p, avg_n, full_n))

        items.sort(key=lambda x: x[1], reverse=True)
        for cat, mean_p, std_p, full_p, avg_n, rescue, full_n in items:
            if avg_n >= 5:
                print(f"  {cat:<23s} {mean_p:5.1f}% {std_p:5.1f}% {full_p:5.1f}% {avg_n:6.1f}  {rescue}")

    print("\n--- ATC Coherence × Category within Standard Filter (Holdout) ---")
    for atc_key in ["coherent", "incoherent"]:
        if atc_key not in holdout_atc_cat:
            continue
        print(f"\n  === {atc_key} ===")
        items = []
        for cat in holdout_atc_cat[atc_key]:
            vals = holdout_atc_cat[atc_key][cat]
            ns = holdout_atc_cat_n[atc_key][cat]
            if len(vals) < 3:
                continue
            mean_p = np.mean(vals)
            std_p = np.std(vals)
            avg_n = np.mean(ns)
            items.append((cat, mean_p, std_p, avg_n))
        items.sort(key=lambda x: x[1], reverse=True)
        for cat, mean_p, std_p, avg_n in items:
            if avg_n >= 5:
                print(f"  {cat:<23s} {mean_p:5.1f}% {std_p:5.1f}% {avg_n:6.1f}")

    # === SUMMARY ===
    print("\n" + "=" * 80)
    print("RESCUE CANDIDATE SUMMARY")
    print("=" * 80)
    if rescue_candidates:
        print(f"\nSub-populations with >15% holdout precision AND n>=10/seed:")
        for sub_reason, cat, mean_p, std_p, avg_n, full_n in rescue_candidates:
            print(f"  {sub_reason} × {cat}: {mean_p:.1f}% ± {std_p:.1f}% holdout ({avg_n:.0f}/seed, {full_n} full)")
    else:
        print("\nNo rescue candidates found.")

    # Save results
    output_path = Path("data/analysis/h533_deep_dive.json")
    with open(output_path, "w") as f:
        json.dump({
            "rescue_candidates": [(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rescue_candidates],
            "n_seeds": len(seeds),
        }, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
