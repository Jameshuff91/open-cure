#!/usr/bin/env python3
"""
h789: Literature Tier Override — Promote LOW/FILTER with STRONG Evidence

Tests: If we promote LOW/FILTER predictions with STRONG_EVIDENCE to HIGH,
what happens to tier precisions? Exclude inverse_indication and other safety
filters from promotion.

This is implemented as a post-prediction override, not a core tier rule change.
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

# Sub-reasons that should NEVER be promoted (safety filters)
NEVER_PROMOTE = {
    "inverse_indication",
    "withdrawn_drug",
    "non_therapeutic",
    "cancer_no_gt",
    "antimicrobial_mismatch",
}


def split_diseases(
    all_diseases: List[str], seed: int, train_ratio: float = 0.8
) -> Tuple[List[str], List[str]]:
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


def restore_gt_structures(predictor, originals):
    predictor.drug_train_freq = originals["drug_train_freq"]
    predictor.drug_to_diseases = originals["drug_to_diseases"]
    predictor.drug_cancer_types = originals["drug_cancer_types"]
    predictor.drug_disease_groups = originals["drug_disease_groups"]
    predictor.train_diseases = originals["train_diseases"]
    predictor.train_embeddings = originals["train_embeddings"]
    predictor.train_disease_categories = originals["train_disease_categories"]


def main():
    seeds = [42, 123, 456, 789, 2024]
    print("=" * 70)
    print("h789: Literature Tier Override — STRONG Evidence Promotion")
    print("=" * 70)

    # Load literature cache
    with open("data/validation/literature_mining_cache.json") as f:
        lit_cache = json.load(f)
    print(f"Literature cache: {len(lit_cache)} entries")

    predictor = DrugRepurposingPredictor()

    with open(predictor.reference_dir / "expanded_ground_truth.json") as f:
        gt_data = json.load(f)

    gt_set: Set[Tuple[str, str]] = set()
    for disease_id, drugs in gt_data.items():
        for drug in drugs:
            if isinstance(drug, str):
                gt_set.add((disease_id, drug))
            elif isinstance(drug, dict):
                did = drug.get("drug_id") or drug.get("drug")
                if did:
                    gt_set.add((disease_id, did))

    all_diseases = [d for d in predictor.ground_truth if d in predictor.embeddings]
    print(f"Diseases: {len(all_diseases)}")

    # Test multiple promotion thresholds
    THRESHOLDS = [
        ("STRONG (any score)", "STRONG_EVIDENCE", 0),
        ("Score >= 4", None, 4),
        ("Score >= 6", None, 6),
    ]

    for thresh_name, level_filter, score_filter in THRESHOLDS:
        print(f"\n{'='*70}")
        print(f"THRESHOLD: {thresh_name}")
        print(f"{'='*70}")

        # Accumulators: before and after promotion
        before_tier: Dict[str, List[float]] = defaultdict(list)
        after_tier: Dict[str, List[float]] = defaultdict(list)
        promoted_counts: List[int] = []
        promoted_gt: List[int] = []

        for seed_idx, seed in enumerate(seeds):
            print(f"  Seed {seed} ({seed_idx+1}/{len(seeds)})...")

            train_ids, holdout_ids = split_diseases(all_diseases, seed)
            train_set = set(train_ids)
            originals = recompute_gt_structures(predictor, train_set)

            # Collect predictions with tiers BEFORE and AFTER promotion
            tier_hits_before: Dict[str, int] = defaultdict(int)
            tier_total_before: Dict[str, int] = defaultdict(int)
            tier_hits_after: Dict[str, int] = defaultdict(int)
            tier_total_after: Dict[str, int] = defaultdict(int)
            n_promoted = 0
            n_promoted_gt = 0

            for disease_id in holdout_ids:
                disease_name = predictor.disease_names.get(disease_id, disease_id)
                try:
                    result = predictor.predict(disease_name, top_n=30, include_filtered=True)
                except Exception:
                    continue

                for pred in result.predictions:
                    tier = pred.confidence_tier.name
                    sub_reason = pred.category_specific_tier or "default"
                    is_hit = (disease_id, pred.drug_id) in gt_set

                    # BEFORE
                    tier_total_before[tier] += 1
                    if is_hit:
                        tier_hits_before[tier] += 1

                    # Check for promotion
                    new_tier = tier
                    if tier in ("LOW", "FILTER") and sub_reason not in NEVER_PROMOTE:
                        cache_key = f"{pred.drug_name.lower()}|{disease_name.lower()}"
                        if cache_key in lit_cache:
                            entry = lit_cache[cache_key]
                            score = entry.get("evidence_score", 0)
                            level = entry.get("evidence_level", "")

                            should_promote = False
                            if level_filter and level == level_filter:
                                should_promote = True
                            elif score_filter and score >= score_filter:
                                should_promote = True

                            if should_promote:
                                # Promote LOW→MEDIUM, FILTER→MEDIUM (conservative)
                                new_tier = "MEDIUM"
                                n_promoted += 1
                                if is_hit:
                                    n_promoted_gt += 1

                    # AFTER
                    tier_total_after[new_tier] += 1
                    if is_hit:
                        tier_hits_after[new_tier] += 1

            # Compute per-seed precision
            for tier_name in ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]:
                if tier_total_before[tier_name] > 0:
                    before_tier[tier_name].append(
                        100 * tier_hits_before[tier_name] / tier_total_before[tier_name]
                    )
                if tier_total_after[tier_name] > 0:
                    after_tier[tier_name].append(
                        100 * tier_hits_after[tier_name] / tier_total_after[tier_name]
                    )

            promoted_counts.append(n_promoted)
            promoted_gt.append(n_promoted_gt)

            restore_gt_structures(predictor, originals)

        # Report
        print(f"\n  Promoted per seed: {np.mean(promoted_counts):.0f} "
              f"({np.mean(promoted_gt):.0f} GT hits, "
              f"{100*np.mean(promoted_gt)/np.mean(promoted_counts):.1f}% precision)")

        print(f"\n  {'Tier':<10s} {'Before':>10s} {'After':>10s} {'Delta':>8s}")
        print(f"  {'-'*40}")
        for tier_name in ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]:
            b = np.mean(before_tier[tier_name]) if before_tier[tier_name] else 0
            a = np.mean(after_tier[tier_name]) if after_tier[tier_name] else 0
            b_std = np.std(before_tier[tier_name]) if before_tier[tier_name] else 0
            a_std = np.std(after_tier[tier_name]) if after_tier[tier_name] else 0
            delta = a - b
            print(f"  {tier_name:<10s} {b:5.1f}±{b_std:4.1f}% {a:5.1f}±{a_std:4.1f}% {delta:+5.1f}pp")

    # Also test: promote to HIGH instead of MEDIUM
    print(f"\n{'='*70}")
    print(f"ALTERNATIVE: Promote STRONG to HIGH (not MEDIUM)")
    print(f"{'='*70}")

    before_tier2: Dict[str, List[float]] = defaultdict(list)
    after_tier2: Dict[str, List[float]] = defaultdict(list)

    for seed_idx, seed in enumerate(seeds):
        print(f"  Seed {seed} ({seed_idx+1}/{len(seeds)})...")

        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        train_set = set(train_ids)
        originals = recompute_gt_structures(predictor, train_set)

        tier_hits_before: Dict[str, int] = defaultdict(int)
        tier_total_before: Dict[str, int] = defaultdict(int)
        tier_hits_after: Dict[str, int] = defaultdict(int)
        tier_total_after: Dict[str, int] = defaultdict(int)

        for disease_id in holdout_ids:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            try:
                result = predictor.predict(disease_name, top_n=30, include_filtered=True)
            except Exception:
                continue

            for pred in result.predictions:
                tier = pred.confidence_tier.name
                sub_reason = pred.category_specific_tier or "default"
                is_hit = (disease_id, pred.drug_id) in gt_set

                tier_total_before[tier] += 1
                if is_hit:
                    tier_hits_before[tier] += 1

                new_tier = tier
                if tier in ("LOW", "FILTER") and sub_reason not in NEVER_PROMOTE:
                    cache_key = f"{pred.drug_name.lower()}|{disease_name.lower()}"
                    if cache_key in lit_cache:
                        entry = lit_cache[cache_key]
                        if entry.get("evidence_level") == "STRONG_EVIDENCE":
                            new_tier = "HIGH"

                tier_total_after[new_tier] += 1
                if is_hit:
                    tier_hits_after[new_tier] += 1

        for tier_name in ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]:
            if tier_total_before[tier_name] > 0:
                before_tier2[tier_name].append(
                    100 * tier_hits_before[tier_name] / tier_total_before[tier_name]
                )
            if tier_total_after[tier_name] > 0:
                after_tier2[tier_name].append(
                    100 * tier_hits_after[tier_name] / tier_total_after[tier_name]
                )

        restore_gt_structures(predictor, originals)

    print(f"\n  {'Tier':<10s} {'Before':>10s} {'After':>10s} {'Delta':>8s}")
    print(f"  {'-'*40}")
    for tier_name in ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]:
        b = np.mean(before_tier2[tier_name]) if before_tier2[tier_name] else 0
        a = np.mean(after_tier2[tier_name]) if after_tier2[tier_name] else 0
        b_std = np.std(before_tier2[tier_name]) if before_tier2[tier_name] else 0
        a_std = np.std(after_tier2[tier_name]) if after_tier2[tier_name] else 0
        delta = a - b
        print(f"  {tier_name:<10s} {b:5.1f}±{b_std:4.1f}% {a:5.1f}±{a_std:4.1f}% {delta:+5.1f}pp")

    print("\nDone.")


if __name__ == "__main__":
    main()
