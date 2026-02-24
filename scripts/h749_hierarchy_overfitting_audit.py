#!/usr/bin/env python3
"""h749: Audit overfitted hierarchy rules and implement demotions.

h393 holdout validation reveals several hierarchy rules with dramatic full-to-holdout gaps:
1. neurological_hierarchy_epilepsy (GOLDEN): 58.2% full → 20.0% holdout (67 preds)
2. metabolic_hierarchy_gout (GOLDEN): 37.5% full → 14.3% holdout (8 preds)
3. infectious_hierarchy_skin_infection (HIGH): 25.0% full → 25.0% holdout (32 preds)

This script:
1. Examines the specific drugs/diseases in epilepsy hierarchy predictions
2. Runs independent 5-seed holdout for epilepsy to confirm 20% number
3. Checks whether demoting improves GOLDEN and doesn't hurt MEDIUM
4. Tests gout demotion impact
"""

import json
import sys
import time
from collections import defaultdict, Counter
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
# PART 1: Analyze epilepsy predictions in full-data
# ============================================================
print("\n" + "=" * 60)
print("PART 1: Full-data epilepsy hierarchy analysis")
print("=" * 60)

epilepsy_preds = []
gout_preds = []
all_predictions_by_rule = defaultdict(list)

for disease_id in gt_diseases_with_embeddings:
    disease_name = predictor.disease_names.get(disease_id, "")
    result = predictor.predict(disease_name, top_n=30, include_filtered=True)

    exp_gt_drugs = set(expanded_gt.get(disease_id, []))

    for pred in result.predictions:
        tier = pred.confidence_tier
        reason = pred.category_specific_tier or "default"
        drug_id = pred.drug_id
        is_hit = drug_id in exp_gt_drugs

        if 'hierarchy_epilepsy' in (reason or ''):
            epilepsy_preds.append({
                'disease': disease_name,
                'disease_id': disease_id,
                'drug': pred.drug_name,
                'drug_id': drug_id,
                'tier': str(tier),
                'reason': reason,
                'hit': is_hit,
                'score': getattr(pred, 'knn_score', 0),
                'rank': getattr(pred, 'rank', 0),
            })
        if 'hierarchy_gout' in (reason or ''):
            gout_preds.append({
                'disease': disease_name,
                'drug': pred.drug_name,
                'tier': str(tier),
                'hit': is_hit,
            })

print(f"\nEpilepsy hierarchy predictions: {len(epilepsy_preds)}")
if epilepsy_preds:
    hits = sum(1 for p in epilepsy_preds if p['hit'])
    print(f"  Hits: {hits}/{len(epilepsy_preds)} = {100*hits/len(epilepsy_preds):.1f}%")

    # Show unique diseases
    diseases = Counter(p['disease'] for p in epilepsy_preds)
    print(f"  Unique diseases: {len(diseases)}")
    for d, count in diseases.most_common(10):
        d_hits = sum(1 for p in epilepsy_preds if p['disease'] == d and p['hit'])
        print(f"    {d}: {count} preds, {d_hits} hits ({100*d_hits/count:.0f}%)")

    # Show unique drugs
    drugs = Counter(p['drug'] for p in epilepsy_preds)
    print(f"  Unique drugs: {len(drugs)}")
    for dr, count in drugs.most_common(10):
        dr_hits = sum(1 for p in epilepsy_preds if p['drug'] == dr and p['hit'])
        print(f"    {dr}: {count} preds, {dr_hits} hits ({100*dr_hits/count:.0f}%)")

print(f"\nGout hierarchy predictions: {len(gout_preds)}")
if gout_preds:
    hits = sum(1 for p in gout_preds if p['hit'])
    print(f"  Hits: {hits}/{len(gout_preds)} = {100*hits/len(gout_preds):.1f}%")

# ============================================================
# PART 2: 5-seed holdout for epilepsy specifically
# ============================================================
print("\n" + "=" * 60)
print("PART 2: 5-seed holdout evaluation")
print("=" * 60)

SEEDS = [42, 123, 456, 789, 2024]

# Track results for each rule we want to test
rules_to_test = {
    'neurological_hierarchy_epilepsy': {'tier': 'GOLDEN'},
    'neurological_hierarchy_stroke': {'tier': 'GOLDEN'},
    'metabolic_hierarchy_gout': {'tier': 'GOLDEN'},
    'infectious_hierarchy_skin_infection': {'tier': 'HIGH'},
    # Also check promotable rules
    'gastrointestinal': {'tier': 'HIGH'},
    'infectious_hierarchy_uti': {'tier': 'HIGH'},
}

rule_results = {k: {'hits': [], 'total': [], 'precision': []} for k in rules_to_test}

# Track tier-level impact of proposed changes
tier_results = {
    'GOLDEN_current': {'hits': [], 'total': []},
    'GOLDEN_after': {'hits': [], 'total': []},
    'HIGH_current': {'hits': [], 'total': []},
    'HIGH_after': {'hits': [], 'total': []},
    'MEDIUM_current': {'hits': [], 'total': []},
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

    # Reset counters
    seed_rule_hits = {k: 0 for k in rules_to_test}
    seed_rule_total = {k: 0 for k in rules_to_test}

    seed_tier_hits = {k: 0 for k in tier_results}
    seed_tier_total = {k: 0 for k in tier_results}

    for disease_id in holdout_diseases:
        disease_name = predictor.disease_names.get(disease_id, "")
        result = predictor.predict(disease_name, top_n=30, include_filtered=True)
        exp_gt_drugs = set(expanded_gt.get(disease_id, []))

        for pred in result.predictions:
            drug_id = pred.drug_id
            is_hit = drug_id in exp_gt_drugs
            tier = pred.confidence_tier
            reason = pred.category_specific_tier or "default"

            # Track each rule
            for rule_name in rules_to_test:
                if reason == rule_name:
                    seed_rule_hits[rule_name] += int(is_hit)
                    seed_rule_total[rule_name] += 1

            # Track current tiers
            if tier == ConfidenceTier.GOLDEN:
                seed_tier_hits['GOLDEN_current'] += int(is_hit)
                seed_tier_total['GOLDEN_current'] += 1
            elif tier == ConfidenceTier.HIGH:
                seed_tier_hits['HIGH_current'] += int(is_hit)
                seed_tier_total['HIGH_current'] += 1
            elif tier == ConfidenceTier.MEDIUM:
                seed_tier_hits['MEDIUM_current'] += int(is_hit)
                seed_tier_total['MEDIUM_current'] += 1

            # Track after proposed changes:
            # 1. epilepsy GOLDEN→MEDIUM
            # 2. gout GOLDEN→MEDIUM
            # 3. GI HIGH→GOLDEN
            # 4. UTI HIGH→GOLDEN
            proposed_tier = tier
            if reason == 'neurological_hierarchy_epilepsy' and tier == ConfidenceTier.GOLDEN:
                proposed_tier = ConfidenceTier.MEDIUM
            elif reason == 'metabolic_hierarchy_gout' and tier == ConfidenceTier.GOLDEN:
                proposed_tier = ConfidenceTier.MEDIUM
            elif reason == 'neurological_hierarchy_stroke' and tier == ConfidenceTier.GOLDEN:
                proposed_tier = ConfidenceTier.HIGH  # Test demoting stroke too
            elif reason == 'gastrointestinal' and tier == ConfidenceTier.HIGH:
                proposed_tier = ConfidenceTier.GOLDEN
            elif reason == 'infectious_hierarchy_uti' and tier == ConfidenceTier.HIGH:
                proposed_tier = ConfidenceTier.GOLDEN

            if proposed_tier == ConfidenceTier.GOLDEN:
                seed_tier_hits['GOLDEN_after'] += int(is_hit)
                seed_tier_total['GOLDEN_after'] += 1
            elif proposed_tier == ConfidenceTier.HIGH:
                seed_tier_hits['HIGH_after'] += int(is_hit)
                seed_tier_total['HIGH_after'] += 1
            elif proposed_tier == ConfidenceTier.MEDIUM:
                seed_tier_hits['MEDIUM_after'] += int(is_hit)
                seed_tier_total['MEDIUM_after'] += 1

    # Record results
    for rule_name in rules_to_test:
        h = seed_rule_hits[rule_name]
        t = seed_rule_total[rule_name]
        prec = 100.0 * h / t if t > 0 else 0
        rule_results[rule_name]['hits'].append(h)
        rule_results[rule_name]['total'].append(t)
        rule_results[rule_name]['precision'].append(prec)
        print(f"  {rule_name}: {prec:.1f}% ({h}/{t})")

    for tier_name in tier_results:
        tier_results[tier_name]['hits'].append(seed_tier_hits[tier_name])
        tier_results[tier_name]['total'].append(seed_tier_total[tier_name])

    # Restore
    predictor.drug_train_freq = orig_drug_train_freq
    predictor.drug_to_diseases = orig_drug_to_diseases

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

output = {}
for rule_name, rd in rule_results.items():
    precs = rd['precision']
    tots = rd['total']
    mean_p = np.mean(precs)
    std_p = np.std(precs)
    mean_n = np.mean(tots)
    print(f"\n{rule_name} ({rules_to_test[rule_name]['tier']}):")
    print(f"  Holdout: {mean_p:.1f}% ± {std_p:.1f}% (n={mean_n:.1f}/seed)")
    print(f"  Seeds: {[f'{p:.1f}' for p in precs]}")
    print(f"  N per seed: {tots}")
    output[rule_name] = {
        'mean_precision': round(float(mean_p), 1),
        'std_precision': round(float(std_p), 1),
        'mean_n': round(float(mean_n), 1),
        'seed_precisions': [round(p, 1) for p in precs],
        'seed_n': [int(n) for n in tots],
    }

print("\n\n=== TIER IMPACT ===")
for tier_label in ['GOLDEN', 'HIGH', 'MEDIUM']:
    curr_key = f'{tier_label}_current'
    after_key = f'{tier_label}_after'

    curr_precs = [100.0 * h / t if t > 0 else 0
                  for h, t in zip(tier_results[curr_key]['hits'], tier_results[curr_key]['total'])]
    after_precs = [100.0 * h / t if t > 0 else 0
                   for h, t in zip(tier_results[after_key]['hits'], tier_results[after_key]['total'])]

    curr_mean = np.mean(curr_precs)
    after_mean = np.mean(after_precs)
    curr_std = np.std(curr_precs)
    after_std = np.std(after_precs)
    curr_n = np.mean(tier_results[curr_key]['total'])
    after_n = np.mean(tier_results[after_key]['total'])

    delta = after_mean - curr_mean
    print(f"{tier_label}: {curr_mean:.1f}% ± {curr_std:.1f}% (n={curr_n:.0f}) → {after_mean:.1f}% ± {after_std:.1f}% (n={after_n:.0f}) [Δ={delta:+.1f}pp]")

    output[f'{tier_label}_current'] = {
        'mean': round(float(curr_mean), 1),
        'std': round(float(curr_std), 1),
        'mean_n': round(float(curr_n), 1),
    }
    output[f'{tier_label}_after'] = {
        'mean': round(float(after_mean), 1),
        'std': round(float(after_std), 1),
        'mean_n': round(float(after_n), 1),
        'delta': round(float(delta), 1),
    }

# Save
output_path = Path("data/analysis/h749_hierarchy_overfitting_audit.json")
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nSaved to {output_path}")
print(f"Time: {time.time()-t0:.0f}s")
