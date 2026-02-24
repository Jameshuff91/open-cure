#!/usr/bin/env python3
"""h749: Targeted analysis of hierarchy rule changes.

Based on h393 holdout data, implement and validate:
1. epilepsy GOLDEN→HIGH (20.0% holdout)
2. gout GOLDEN→MEDIUM (14.3% holdout)
3. GI HIGH→GOLDEN promotion (90.4% holdout)
4. UTI HIGH→GOLDEN promotion (80.0% holdout)

Also check: stroke, alzheimers, obesity (small-n GOLDEN rules)
And: neurological HIGH (8.6%), respiratory HIGH (21.8%), diabetes HIGH (21.1%)
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from production_predictor import (
    ConfidenceTier,
    DrugRepurposingPredictor,
)

t0 = time.time()

# Load predictor
print("Loading predictor...")
predictor = DrugRepurposingPredictor()

# Load expanded GT
with open(predictor.data_dir / "data" / "reference" / "expanded_ground_truth.json") as f:
    expanded_gt = json.load(f)
print(f"Expanded GT: {len(expanded_gt)} diseases")

# Get all diseases with GT
all_diseases = list(predictor.ground_truth.keys())
gt_diseases_with_embeddings = [d for d in all_diseases if d in predictor.embeddings]
print(f"GT diseases with embeddings: {len(gt_diseases_with_embeddings)}")

# ============================================================
# PART 1: Full-data analysis of hierarchy predictions
# ============================================================
print("\n" + "=" * 60)
print("PART 1: Full-data hierarchy rule analysis")
print("=" * 60)

rule_preds = defaultdict(lambda: {'hits': 0, 'total': 0, 'drugs': set(), 'diseases': set()})

for disease_id in gt_diseases_with_embeddings:
    disease_name = predictor.disease_names.get(disease_id, "")
    result = predictor.predict(disease_name, top_n=30, include_filtered=True)
    exp_gt_drugs = set(expanded_gt.get(disease_id, []))

    for pred in result.predictions:
        reason = pred.category_specific_tier or "default"
        drug_id = pred.drug_id
        is_hit = drug_id in exp_gt_drugs
        tier = pred.confidence_tier

        # Track hierarchy rules
        if 'hierarchy' in reason or reason in ('gastrointestinal', 'neurological', 'respiratory', 'reproductive'):
            key = f"{reason}|{tier.name}"
            rule_preds[key]['total'] += 1
            rule_preds[key]['hits'] += int(is_hit)
            rule_preds[key]['drugs'].add(pred.drug_name)
            rule_preds[key]['diseases'].add(disease_name)

print(f"\nFull-data analysis ({time.time()-t0:.0f}s):")
for key in sorted(rule_preds.keys()):
    d = rule_preds[key]
    prec = 100.0 * d['hits'] / d['total'] if d['total'] > 0 else 0
    print(f"  {key}: {prec:.1f}% ({d['hits']}/{d['total']}) | {len(d['drugs'])} drugs, {len(d['diseases'])} diseases")

# ============================================================
# PART 2: 5-seed holdout — focused on target rules
# ============================================================
print("\n" + "=" * 60)
print("PART 2: 5-seed holdout evaluation")
print("=" * 60)

SEEDS = [42, 123, 456, 789, 2024]

# Rules we want to test changes for
target_rules = [
    # Current GOLDEN that might need demotion
    'neurological_hierarchy_epilepsy',
    'metabolic_hierarchy_gout',
    'neurological_hierarchy_stroke',
    'neurological_hierarchy_alzheimers',
    'metabolic_hierarchy_obesity',
    # Current HIGH that might need demotion
    'metabolic_hierarchy_diabetes',
    'neurological',
    'respiratory',
    'respiratory_hierarchy_copd',
    'respiratory_hierarchy_asthma',
    'autoimmune_hierarchy_lupus',
    # Current HIGH that might deserve promotion
    'gastrointestinal',
    'infectious_hierarchy_uti',
]

rule_seed_results = {r: {'precisions': [], 'ns': []} for r in target_rules}

# Track overall tier impact
tier_changes = {
    'GOLDEN_before': {'hits': [], 'total': []},
    'GOLDEN_after': {'hits': [], 'total': []},
    'HIGH_before': {'hits': [], 'total': []},
    'HIGH_after': {'hits': [], 'total': []},
    'MEDIUM_before': {'hits': [], 'total': []},
    'MEDIUM_after': {'hits': [], 'total': []},
}

for seed in SEEDS:
    print(f"\n=== Seed {seed} ===")
    rng = np.random.RandomState(seed)
    shuffled = list(gt_diseases_with_embeddings)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * 0.8)
    train_diseases = set(shuffled[:split_idx])
    holdout_diseases = set(shuffled[split_idx:])

    # Recompute training frequency
    orig_drug_train_freq = dict(predictor.drug_train_freq)
    orig_drug_to_diseases = dict(predictor.drug_to_diseases)

    new_freq = defaultdict(int)
    new_d2d = defaultdict(set)
    for disease_id in train_diseases:
        for drug_id in predictor.ground_truth.get(disease_id, set()):
            new_freq[drug_id] += 1
            new_d2d[drug_id].add(disease_id)
    predictor.drug_train_freq = dict(new_freq)
    predictor.drug_to_diseases = dict(new_d2d)

    # Per-rule counters
    rule_hits = defaultdict(int)
    rule_total = defaultdict(int)

    # Per-tier counters
    tier_hits_before = defaultdict(int)
    tier_total_before = defaultdict(int)
    tier_hits_after = defaultdict(int)
    tier_total_after = defaultdict(int)

    for disease_id in holdout_diseases:
        disease_name = predictor.disease_names.get(disease_id, "")
        result = predictor.predict(disease_name, top_n=30, include_filtered=True)
        exp_gt_drugs = set(expanded_gt.get(disease_id, []))

        for pred in result.predictions:
            drug_id = pred.drug_id
            is_hit = drug_id in exp_gt_drugs
            tier = pred.confidence_tier
            reason = pred.category_specific_tier or "default"

            # Track rule-level stats
            if reason in target_rules:
                rule_hits[reason] += int(is_hit)
                rule_total[reason] += 1

            # Track current tier assignment
            if tier in (ConfidenceTier.GOLDEN, ConfidenceTier.HIGH, ConfidenceTier.MEDIUM):
                key = f'{tier.name}_before'
                tier_hits_before[tier.name] += int(is_hit)
                tier_total_before[tier.name] += 1

            # Compute proposed tier after changes
            proposed_tier = tier
            if reason == 'neurological_hierarchy_epilepsy' and tier == ConfidenceTier.GOLDEN:
                proposed_tier = ConfidenceTier.HIGH  # GOLDEN→HIGH
            elif reason == 'metabolic_hierarchy_gout' and tier == ConfidenceTier.GOLDEN:
                proposed_tier = ConfidenceTier.MEDIUM  # GOLDEN→MEDIUM
            elif reason == 'metabolic_hierarchy_obesity' and tier == ConfidenceTier.GOLDEN:
                proposed_tier = ConfidenceTier.HIGH  # GOLDEN→HIGH (tiny n, uncertain)
            elif reason == 'neurological_hierarchy_alzheimers' and tier == ConfidenceTier.GOLDEN:
                proposed_tier = ConfidenceTier.HIGH  # GOLDEN→HIGH (tiny n)
            elif reason == 'gastrointestinal' and tier == ConfidenceTier.HIGH:
                proposed_tier = ConfidenceTier.GOLDEN  # HIGH→GOLDEN
            elif reason == 'infectious_hierarchy_uti' and tier == ConfidenceTier.HIGH:
                proposed_tier = ConfidenceTier.GOLDEN  # HIGH→GOLDEN

            if proposed_tier in (ConfidenceTier.GOLDEN, ConfidenceTier.HIGH, ConfidenceTier.MEDIUM):
                tier_hits_after[proposed_tier.name] += int(is_hit)
                tier_total_after[proposed_tier.name] += 1

    # Record per-rule results
    for rule in target_rules:
        h, t = rule_hits[rule], rule_total[rule]
        prec = 100.0 * h / t if t > 0 else 0
        rule_seed_results[rule]['precisions'].append(prec)
        rule_seed_results[rule]['ns'].append(t)
        if t > 0:
            print(f"  {rule}: {prec:.1f}% ({h}/{t})")

    # Record tier results
    for tier_name in ('GOLDEN', 'HIGH', 'MEDIUM'):
        tier_changes[f'{tier_name}_before']['hits'].append(tier_hits_before[tier_name])
        tier_changes[f'{tier_name}_before']['total'].append(tier_total_before[tier_name])
        tier_changes[f'{tier_name}_after']['hits'].append(tier_hits_after[tier_name])
        tier_changes[f'{tier_name}_after']['total'].append(tier_total_after[tier_name])

    # Restore
    predictor.drug_train_freq = orig_drug_train_freq
    predictor.drug_to_diseases = orig_drug_to_diseases

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("RULE-LEVEL SUMMARY")
print("=" * 60)

output = {}
for rule in target_rules:
    rd = rule_seed_results[rule]
    precs = rd['precisions']
    ns = rd['ns']
    mean_p = np.mean(precs) if precs else 0
    std_p = np.std(precs) if precs else 0
    mean_n = np.mean(ns) if ns else 0
    total_n_any = sum(1 for n in ns if n > 0)

    if mean_n > 0:
        print(f"\n{rule}:")
        print(f"  Holdout: {mean_p:.1f}% ± {std_p:.1f}% (n={mean_n:.1f}/seed, appeared in {total_n_any}/5 seeds)")
        print(f"  Seeds: {[f'{p:.1f}' for p in precs]}")
        print(f"  N per seed: {ns}")

    output[rule] = {
        'mean_precision': round(float(mean_p), 1),
        'std_precision': round(float(std_p), 1),
        'mean_n': round(float(mean_n), 1),
        'seeds_with_data': total_n_any,
        'seed_precisions': [round(p, 1) for p in precs],
        'seed_n': [int(n) for n in ns],
    }

print("\n" + "=" * 60)
print("TIER IMPACT SUMMARY")
print("=" * 60)

for tier_name in ('GOLDEN', 'HIGH', 'MEDIUM'):
    b_key = f'{tier_name}_before'
    a_key = f'{tier_name}_after'

    b_precs = [100.0 * h / t if t > 0 else 0
               for h, t in zip(tier_changes[b_key]['hits'], tier_changes[b_key]['total'])]
    a_precs = [100.0 * h / t if t > 0 else 0
               for h, t in zip(tier_changes[a_key]['hits'], tier_changes[a_key]['total'])]

    b_mean = np.mean(b_precs)
    a_mean = np.mean(a_precs)
    b_std = np.std(b_precs)
    a_std = np.std(a_precs)
    b_n = np.mean(tier_changes[b_key]['total'])
    a_n = np.mean(tier_changes[a_key]['total'])

    delta = a_mean - b_mean
    print(f"\n{tier_name}:")
    print(f"  Before: {b_mean:.1f}% ± {b_std:.1f}% (n={b_n:.0f}/seed)")
    print(f"  After:  {a_mean:.1f}% ± {a_std:.1f}% (n={a_n:.0f}/seed)")
    print(f"  Delta:  {delta:+.1f}pp")

    output[f'{tier_name}_before'] = {'mean': round(b_mean, 1), 'std': round(b_std, 1), 'mean_n': round(b_n, 1)}
    output[f'{tier_name}_after'] = {'mean': round(a_mean, 1), 'std': round(a_std, 1), 'mean_n': round(a_n, 1), 'delta': round(delta, 1)}

# Save
output_path = Path("data/analysis/h749_targeted_analysis.json")
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nSaved to {output_path}")
print(f"Total time: {time.time()-t0:.0f}s")
