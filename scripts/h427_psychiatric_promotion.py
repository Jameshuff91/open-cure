#!/usr/bin/env python3
"""
h427: Holdout validation of psychiatric target_overlap_promotion → HIGH/GOLDEN.

h390 found psychiatric MEDIUM has 53.3% precision (100/105 from target_overlap_promotion).
Need to verify this holds on 80/20 holdout before promoting.

Also test h428 (category-specific incoherent demotion) and h429 (infectious low-rank rescue).
"""

import json
import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import production_predictor as pp
from production_predictor import (
    DrugRepurposingPredictor,
    ConfidenceTier,
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
    """Get per-tier and per-category-tier precision."""
    tier_counts = defaultdict(lambda: {"total": 0, "gt": 0})
    cat_tier_counts = defaultdict(lambda: defaultdict(lambda: {"total": 0, "gt": 0}))
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

        category = predictor.categorize_disease(disease_name)

        for pred in result.predictions:
            tier = pred.confidence_tier.name
            rule = pred.category_specific_tier or 'standard'
            is_gt = pred.drug_id in gt_drugs

            tier_counts[tier]["total"] += 1
            tier_counts[tier]["gt"] += int(is_gt)
            cat_tier_counts[category][tier]["total"] += 1
            cat_tier_counts[category][tier]["gt"] += int(is_gt)
            rule_counts[rule]["total"] += 1
            rule_counts[rule]["gt"] += int(is_gt)

    return tier_counts, cat_tier_counts, rule_counts


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

    # === Phase 1: Full-data baseline ===
    print("\n=== PHASE 1: Full-data baseline ===")
    tier_counts, cat_tier_counts, rule_counts = evaluate_tier_precision(
        predictor, all_diseases, gt_data
    )

    tier_order = ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]
    print(f"\nBaseline tier precision:")
    for tier in tier_order:
        tc = tier_counts.get(tier, {"total": 0, "gt": 0})
        prec = tc["gt"] / tc["total"] * 100 if tc["total"] > 0 else 0
        print(f"  {tier}: {tc['gt']}/{tc['total']} = {prec:.1f}%")

    # Psychiatric MEDIUM detail
    psych_med = cat_tier_counts.get('psychiatric', {}).get('MEDIUM', {"total": 0, "gt": 0})
    psych_prec = psych_med['gt'] / psych_med['total'] * 100 if psych_med['total'] > 0 else 0
    print(f"\nPsychiatric MEDIUM: {psych_med['gt']}/{psych_med['total']} = {psych_prec:.1f}%")

    # Relevant rules
    for rule in ['target_overlap_promotion', 'incoherent_demotion']:
        rc = rule_counts.get(rule, {"total": 0, "gt": 0})
        prec = rc['gt'] / rc['total'] * 100 if rc['total'] > 0 else 0
        print(f"Rule '{rule}': {rc['gt']}/{rc['total']} = {prec:.1f}%")

    # === Phase 2: Holdout validation (5-seed) ===
    print("\n=== PHASE 2: Holdout Validation (5-seed) ===")
    seeds = [42, 123, 456, 789, 2024]

    holdout_tiers = defaultdict(list)  # tier -> list of precision values
    holdout_psych_med = []  # psychiatric MEDIUM precision per seed
    holdout_inf_low_rank = []  # infectious LOW rank 1-5 per seed

    for seed in seeds:
        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        originals = recompute_gt_structures(predictor, set(train_ids))

        tier_counts_h, cat_tier_h, rule_counts_h = evaluate_tier_precision(
            predictor, holdout_ids, gt_data
        )

        # Tier precision
        for tier in tier_order:
            tc = tier_counts_h.get(tier, {"total": 0, "gt": 0})
            prec = tc["gt"] / tc["total"] * 100 if tc["total"] > 0 else 0
            holdout_tiers[tier].append(prec)

        # Psychiatric MEDIUM
        psych_med_h = cat_tier_h.get('psychiatric', {}).get('MEDIUM', {"total": 0, "gt": 0})
        psych_prec_h = psych_med_h['gt'] / psych_med_h['total'] * 100 if psych_med_h['total'] > 0 else 0
        holdout_psych_med.append(psych_prec_h)

        # Target overlap promotion precision
        top_h = rule_counts_h.get('target_overlap_promotion', {"total": 0, "gt": 0})
        top_prec = top_h['gt'] / top_h['total'] * 100 if top_h['total'] > 0 else 0

        # Incoherent demotion
        id_h = rule_counts_h.get('incoherent_demotion', {"total": 0, "gt": 0})
        id_prec = id_h['gt'] / id_h['total'] * 100 if id_h['total'] > 0 else 0

        print(f"  Seed {seed}: tiers=[" +
              ", ".join(f"{tier}={holdout_tiers[tier][-1]:.1f}%" for tier in tier_order) +
              f"] psych_med={psych_prec_h:.1f}% target_overlap={top_prec:.1f}% incoh={id_prec:.1f}%")

        restore_gt_structures(predictor, originals)

    # === Summary ===
    print(f"\n{'='*70}")
    print("=== HOLDOUT SUMMARY ===")
    print(f"{'='*70}")

    print(f"\nTier precision (holdout, 5-seed):")
    for tier in tier_order:
        vals = holdout_tiers[tier]
        print(f"  {tier}: {np.mean(vals):.1f}% ± {np.std(vals):.1f}%")

    psych_mean = np.mean(holdout_psych_med)
    psych_std = np.std(holdout_psych_med)
    print(f"\nPsychiatric MEDIUM holdout: {psych_mean:.1f}% ± {psych_std:.1f}%")
    print(f"  Full-data: {psych_prec:.1f}%")
    print(f"  Delta: {psych_mean - psych_prec:+.1f}pp")

    if psych_mean > 50:
        print(f"  → GOLDEN level (>{50}%): PROMOTE to GOLDEN")
    elif psych_mean > 35:
        print(f"  → HIGH level (>{35}%): PROMOTE to HIGH")
    elif psych_mean > 20:
        print(f"  → MEDIUM level: KEEP at MEDIUM")
    else:
        print(f"  → Below MEDIUM: CHECK for overfitting")

    # Save results
    output = {
        "holdout_tiers": {tier: holdout_tiers[tier] for tier in tier_order},
        "psychiatric_medium_holdout": holdout_psych_med,
        "psychiatric_medium_full": psych_prec,
    }
    out_path = Path(__file__).parent.parent / "data" / "analysis" / "h427_psychiatric_promotion.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
