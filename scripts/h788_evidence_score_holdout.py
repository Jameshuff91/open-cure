#!/usr/bin/env python3
"""
h788: Literature Evidence Score as Continuous Deliverable Feature

Tests whether the continuous evidence_score from literature mining correlates
with holdout precision within tiers. If gradient exists within tiers,
the evidence_score adds value beyond the categorical evidence_level.

Approach:
1. Generate predictions for all diseases
2. Match predictions to literature mining cache via drug_name|disease_name key
3. Run 5-seed holdout evaluation, stratified by evidence_score quartiles WITHIN each tier
4. Report gradient and statistical significance
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


def main() -> None:
    seeds = [42, 123, 456, 789, 2024]
    print("=" * 70)
    print("h788: Evidence Score as Continuous Holdout Predictor")
    print("=" * 70)

    # Load literature mining cache
    cache_path = Path("data/validation/literature_mining_cache.json")
    with open(cache_path) as f:
        lit_cache = json.load(f)
    print(f"Literature cache: {len(lit_cache)} entries")

    # Load predictor
    print("Loading predictor...")
    predictor = DrugRepurposingPredictor()

    # Load expanded GT
    gt_path = predictor.reference_dir / "expanded_ground_truth.json"
    with open(gt_path) as f:
        gt_data = json.load(f)
    print(f"GT: {len(gt_data)} diseases, {sum(len(v) for v in gt_data.values())} pairs")

    # Get evaluable diseases
    all_diseases = [d for d in predictor.ground_truth if d in predictor.embeddings]
    print(f"Diseases with GT + embeddings: {len(all_diseases)}")

    # Build GT set
    gt_set: Set[Tuple[str, str]] = set()
    for disease_id, drugs in gt_data.items():
        for drug in drugs:
            if isinstance(drug, str):
                gt_set.add((disease_id, drug))
            elif isinstance(drug, dict):
                gt_set.add((disease_id, drug.get("drug_id") or drug.get("drug")))

    # ---- STEP 1: Full-data analysis (match rate, score distribution) ----
    print("\n--- STEP 1: Full-data evidence score matching ---")

    # Generate all predictions and match to cache
    matched_count = 0
    unmatched_count = 0
    tier_score_full: Dict[str, List[float]] = defaultdict(list)

    for disease_id in all_diseases[:50]:  # Quick sample first
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        try:
            result = predictor.predict(disease_name, top_n=30, include_filtered=True)
        except Exception:
            continue
        for pred in result.predictions:
            drug_name = pred.drug_name.lower()
            disease_lower = disease_name.lower()
            cache_key = f"{drug_name}|{disease_lower}"
            if cache_key in lit_cache:
                matched_count += 1
                score = lit_cache[cache_key].get("evidence_score", 0)
                tier_score_full[pred.confidence_tier.name].append(score)
            else:
                unmatched_count += 1

    total = matched_count + unmatched_count
    print(f"Sample (50 diseases): {matched_count}/{total} matched ({100*matched_count/total:.1f}%)")

    if matched_count < 50:
        # Try alternative key formats
        print("Low match rate - checking key format...")
        sample_keys = list(lit_cache.keys())[:5]
        print(f"Cache keys sample: {sample_keys}")

    # ---- STEP 2: Full holdout evaluation with evidence score stratification ----
    print("\n--- STEP 2: 5-seed holdout evaluation by evidence score ---")

    # Collect per-seed results: tier -> score_bucket -> [hits, total]
    # Score buckets: [0, 2), [2, 4), [4, 6), [6, 9], no_cache
    SCORE_BINS = [(0, 2, "0-2 (none/weak)"), (2, 4, "2-4 (moderate)"),
                  (4, 6, "4-6 (strong-low)"), (6, 9.01, "6-9 (strong-high)")]

    # Also track by evidence_level categories
    EVIDENCE_LEVELS = ["NO_EVIDENCE", "WEAK_EVIDENCE", "MODERATE_EVIDENCE", "STRONG_EVIDENCE"]

    # Per-seed accumulators
    all_tier_bin_hits: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    all_tier_bin_total: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    all_tier_level_hits: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    all_tier_level_total: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))

    # No-cache accumulator
    all_tier_nocache_hits: Dict[str, List[int]] = defaultdict(list)
    all_tier_nocache_total: Dict[str, List[int]] = defaultdict(list)

    for seed_idx, seed in enumerate(seeds):
        print(f"\n  Seed {seed} ({seed_idx+1}/{len(seeds)})...")

        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        train_set = set(train_ids)

        originals = recompute_gt_structures(predictor, train_set)

        # Per-seed accumulators
        tier_bin_hits: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        tier_bin_total: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        tier_level_hits: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        tier_level_total: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        tier_nocache_hits: Dict[str, int] = defaultdict(int)
        tier_nocache_total: Dict[str, int] = defaultdict(int)

        for disease_id in holdout_ids:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            try:
                result = predictor.predict(disease_name, top_n=30, include_filtered=True)
            except Exception:
                continue

            for pred in result.predictions:
                drug_name = pred.drug_name.lower()
                disease_lower = disease_name.lower()
                cache_key = f"{drug_name}|{disease_lower}"
                tier = pred.confidence_tier.name
                is_hit = (disease_id, pred.drug_id) in gt_set

                if cache_key in lit_cache:
                    entry = lit_cache[cache_key]
                    score = entry.get("evidence_score", 0)
                    level = entry.get("evidence_level", "NO_EVIDENCE")

                    # Score bin
                    for lo, hi, label in SCORE_BINS:
                        if lo <= score < hi:
                            tier_bin_total[tier][label] += 1
                            if is_hit:
                                tier_bin_hits[tier][label] += 1
                            break

                    # Evidence level
                    tier_level_total[tier][level] += 1
                    if is_hit:
                        tier_level_hits[tier][level] += 1
                else:
                    tier_nocache_total[tier] += 1
                    if is_hit:
                        tier_nocache_hits[tier] += 1

        # Store per-seed
        for tier in ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]:
            for _, _, label in SCORE_BINS:
                all_tier_bin_hits[tier][label].append(tier_bin_hits[tier][label])
                all_tier_bin_total[tier][label].append(tier_bin_total[tier][label])
            for level in EVIDENCE_LEVELS:
                all_tier_level_hits[tier][level].append(tier_level_hits[tier][level])
                all_tier_level_total[tier][level].append(tier_level_total[tier][level])
            all_tier_nocache_hits[tier].append(tier_nocache_hits[tier])
            all_tier_nocache_total[tier].append(tier_nocache_total[tier])

        restore_gt_structures(predictor, originals)

    # ---- STEP 3: Report results ----
    print("\n" + "=" * 70)
    print("RESULTS: Evidence Score Bins Within Tiers (5-seed holdout)")
    print("=" * 70)

    for tier in ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]:
        print(f"\n  {tier}:")
        print(f"  {'Score Bin':<22s} {'Holdout%':>8s} {'±std':>6s} {'n/seed':>7s}")
        print(f"  {'-'*50}")

        for _, _, label in SCORE_BINS:
            hits_list = all_tier_bin_hits[tier][label]
            total_list = all_tier_bin_total[tier][label]
            precs = []
            for h, t in zip(hits_list, total_list):
                if t > 0:
                    precs.append(100 * h / t)
            mean_n = np.mean(total_list) if total_list else 0

            if precs:
                mean_p = np.mean(precs)
                std_p = np.std(precs)
                print(f"  {label:<22s} {mean_p:7.1f}% {std_p:5.1f}% {mean_n:6.1f}")
            else:
                print(f"  {label:<22s} {'N/A':>8s} {'':>6s} {mean_n:6.1f}")

        # No-cache
        nc_hits = all_tier_nocache_hits[tier]
        nc_total = all_tier_nocache_total[tier]
        nc_precs = []
        for h, t in zip(nc_hits, nc_total):
            if t > 0:
                nc_precs.append(100 * h / t)
        mean_nc_n = np.mean(nc_total) if nc_total else 0
        if nc_precs:
            print(f"  {'no_cache':<22s} {np.mean(nc_precs):7.1f}% {np.std(nc_precs):5.1f}% {mean_nc_n:6.1f}")

    # Evidence level breakdown
    print("\n" + "=" * 70)
    print("RESULTS: Evidence Level Within Tiers (5-seed holdout)")
    print("=" * 70)

    for tier in ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]:
        print(f"\n  {tier}:")
        print(f"  {'Evidence Level':<22s} {'Holdout%':>8s} {'±std':>6s} {'n/seed':>7s}")
        print(f"  {'-'*50}")

        for level in EVIDENCE_LEVELS:
            hits_list = all_tier_level_hits[tier][level]
            total_list = all_tier_level_total[tier][level]
            precs = []
            for h, t in zip(hits_list, total_list):
                if t > 0:
                    precs.append(100 * h / t)
            mean_n = np.mean(total_list) if total_list else 0

            if precs:
                mean_p = np.mean(precs)
                std_p = np.std(precs)
                print(f"  {level:<22s} {mean_p:7.1f}% {std_p:5.1f}% {mean_n:6.1f}")
            else:
                print(f"  {level:<22s} {'N/A':>8s} {'':>6s} {mean_n:6.1f}")

    # ---- STEP 4: Within-MEDIUM detailed analysis ----
    print("\n" + "=" * 70)
    print("FOCUS: MEDIUM tier evidence score gradient")
    print("=" * 70)

    med_bins = all_tier_bin_hits["MEDIUM"]
    for _, _, label in SCORE_BINS:
        hits_list = all_tier_bin_hits["MEDIUM"][label]
        total_list = all_tier_bin_total["MEDIUM"][label]
        precs = []
        for h, t in zip(hits_list, total_list):
            if t > 0:
                precs.append(100 * h / t)
        mean_n = np.mean(total_list) if total_list else 0
        if precs:
            print(f"  {label:<22s}: {np.mean(precs):.1f}% ± {np.std(precs):.1f}% (n={mean_n:.0f}/seed)")

    # Check: does MEDIUM + high evidence score exceed HIGH tier?
    print("\n--- Comparison: MEDIUM+STRONG vs HIGH tier ---")
    # MEDIUM score 6-9
    med_hi = all_tier_bin_hits["MEDIUM"]["6-9 (strong-high)"]
    med_hi_t = all_tier_bin_total["MEDIUM"]["6-9 (strong-high)"]
    med_hi_precs = [100*h/t for h, t in zip(med_hi, med_hi_t) if t > 0]

    # Overall HIGH
    high_hits = [sum(all_tier_bin_hits["HIGH"][l][i] for _, _, l in SCORE_BINS) + all_tier_nocache_hits["HIGH"][i]
                 for i in range(len(seeds))]
    high_total = [sum(all_tier_bin_total["HIGH"][l][i] for _, _, l in SCORE_BINS) + all_tier_nocache_total["HIGH"][i]
                  for i in range(len(seeds))]
    high_precs = [100*h/t for h, t in zip(high_hits, high_total) if t > 0]

    if med_hi_precs:
        print(f"  MEDIUM + score 6-9: {np.mean(med_hi_precs):.1f}% ± {np.std(med_hi_precs):.1f}% (n={np.mean(med_hi_t):.0f}/seed)")
    if high_precs:
        print(f"  HIGH overall:       {np.mean(high_precs):.1f}% ± {np.std(high_precs):.1f}%")

    # Save detailed results
    results = {
        "description": "h788: Evidence score holdout validation within tiers",
        "seeds": seeds,
        "score_bins": {},
        "evidence_levels": {},
        "cache_coverage": {}
    }

    for tier in ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]:
        results["score_bins"][tier] = {}
        for _, _, label in SCORE_BINS:
            hits = all_tier_bin_hits[tier][label]
            total = all_tier_bin_total[tier][label]
            precs = [100*h/t for h, t in zip(hits, total) if t > 0]
            results["score_bins"][tier][label] = {
                "holdout_mean": round(np.mean(precs), 1) if precs else None,
                "holdout_std": round(np.std(precs), 1) if precs else None,
                "mean_n": round(float(np.mean(total)), 1),
                "seed_hits": hits,
                "seed_totals": total
            }

        results["evidence_levels"][tier] = {}
        for level in EVIDENCE_LEVELS:
            hits = all_tier_level_hits[tier][level]
            total = all_tier_level_total[tier][level]
            precs = [100*h/t for h, t in zip(hits, total) if t > 0]
            results["evidence_levels"][tier][level] = {
                "holdout_mean": round(np.mean(precs), 1) if precs else None,
                "holdout_std": round(np.std(precs), 1) if precs else None,
                "mean_n": round(float(np.mean(total)), 1)
            }

        # Cache coverage
        cached = sum(np.mean(all_tier_bin_total[tier][l]) for _, _, l in SCORE_BINS)
        not_cached = np.mean(all_tier_nocache_total[tier])
        results["cache_coverage"][tier] = {
            "cached_per_seed": round(cached, 1),
            "not_cached_per_seed": round(float(not_cached), 1),
            "coverage_pct": round(100 * cached / (cached + not_cached), 1) if (cached + not_cached) > 0 else 0
        }

    output_path = Path("data/analysis/h788_evidence_score_holdout.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
