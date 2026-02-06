#!/usr/bin/env python3
"""
h430: Holdout validation of narrowed diabetes hierarchy (T2D → GOLDEN, rest → HIGH).
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from production_predictor import (
    DrugRepurposingPredictor,
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


def evaluate_tier_precision(predictor, disease_ids, gt_data):
    tier_counts = defaultdict(lambda: {"total": 0, "gt": 0})
    rule_counts = defaultdict(lambda: {"total": 0, "gt": 0})

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

            tier_counts[tier]["total"] += 1
            tier_counts[tier]["gt"] += int(is_gt)
            rule_counts[(rule, tier)]["total"] += 1
            rule_counts[(rule, tier)]["gt"] += int(is_gt)

    return tier_counts, rule_counts


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
    print("\n=== PHASE 1: Full-data ===")
    tier_counts, rule_counts = evaluate_tier_precision(predictor, all_diseases, gt_data)

    tier_order = ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]
    print(f"\nTier precision:")
    for tier in tier_order:
        tc = tier_counts.get(tier, {"total": 0, "gt": 0})
        prec = tc["gt"] / tc["total"] * 100 if tc["total"] > 0 else 0
        print(f"  {tier}: {tc['gt']}/{tc['total']} = {prec:.1f}%")

    # Diabetes-specific rules
    print(f"\nDiabetes-related rules:")
    for (rule, tier), counts in sorted(rule_counts.items()):
        if 'diabetes' in rule:
            prec = counts['gt'] / counts['total'] * 100 if counts['total'] > 0 else 0
            print(f"  ({rule}, {tier}): {counts['gt']}/{counts['total']} = {prec:.1f}%")

    # === Phase 2: Holdout ===
    print(f"\n{'='*80}")
    print("=== PHASE 2: Holdout (5-seed) ===")
    seeds = [42, 123, 456, 789, 2024]
    holdout_tiers = defaultdict(list)
    holdout_diabetes = defaultdict(list)

    for seed in seeds:
        print(f"  Seed {seed}...")
        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        originals = recompute_gt_structures(predictor, set(train_ids))

        tier_counts_h, rule_counts_h = evaluate_tier_precision(predictor, holdout_ids, gt_data)

        for tier in tier_order:
            tc = tier_counts_h.get(tier, {"total": 0, "gt": 0})
            prec = tc["gt"] / tc["total"] * 100 if tc["total"] > 0 else 0
            holdout_tiers[tier].append(prec)

        for (rule, tier), counts in rule_counts_h.items():
            if 'diabetes' in rule:
                prec = counts['gt'] / counts['total'] * 100 if counts['total'] > 0 else 0
                holdout_diabetes[(rule, tier)].append(prec)

        restore_gt_structures(predictor, originals)

    # === Summary ===
    print(f"\n{'='*80}")
    print("=== HOLDOUT SUMMARY ===")
    for tier in tier_order:
        vals = holdout_tiers[tier]
        print(f"  {tier}: {np.mean(vals):.1f}% ± {np.std(vals):.1f}%")

    print(f"\nDiabetes-related holdout:")
    for (rule, tier), vals in sorted(holdout_diabetes.items()):
        print(f"  ({rule}, {tier}): {np.mean(vals):.1f}% ± {np.std(vals):.1f}% (n_seeds={len(vals)})")

    # h402 baseline for comparison
    print(f"\n--- Comparison with h402 baseline ---")
    h402_baseline = {
        "GOLDEN": (52.9, 6.0),
        "HIGH": (50.6, 10.4),
        "MEDIUM": (21.2, 1.9),
        "LOW": (12.2, 1.9),
        "FILTER": (7.0, 1.5),
    }
    for tier in tier_order:
        vals = holdout_tiers[tier]
        base_mean, base_std = h402_baseline[tier]
        delta = np.mean(vals) - base_mean
        print(f"  {tier}: h402={base_mean:.1f}% → h430={np.mean(vals):.1f}% (Δ={delta:+.1f}pp)")

    # Save
    output = {
        "holdout_tiers": {tier: holdout_tiers[tier] for tier in tier_order},
        "holdout_diabetes": {
            f"{rule}_{tier}": vals for (rule, tier), vals in holdout_diabetes.items()
        },
    }
    out_path = Path(__file__).parent.parent / "data" / "analysis" / "h430_diabetes_narrowing.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
