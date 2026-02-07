#!/usr/bin/env python3
"""
h617: HIGH Tier Stabilization — Investigate Seed-42 Outlier

After h615 promoted 139 HIGH→GOLDEN, HIGH holdout = 52.8% ± 13.5%.
Seed 42 is the outlier (30.0%, n=40). Investigate:
1. Which holdout diseases cause the low n on seed 42?
2. Which HIGH rules have most variance?
3. Is this structural (disease split) or addressable?
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from production_predictor import (
    DrugRepurposingPredictor,
    DISEASE_HIERARCHY_GROUPS,
    extract_cancer_types,
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
        "drug_to_diseases": {k: set(v) for k, v in predictor.drug_to_diseases.items()},
        "drug_cancer_types": {k: set(v) for k, v in predictor.drug_cancer_types.items()},
        "drug_disease_groups": {k: dict(v) for k, v in predictor.drug_disease_groups.items()},
    }

    new_freq = defaultdict(int)
    new_d2d = defaultdict(set)
    new_cancer_types = defaultdict(set)
    new_disease_groups = defaultdict(lambda: defaultdict(set))

    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            for drug_id in predictor.ground_truth[disease_id]:
                new_freq[drug_id] += 1
                new_d2d[drug_id].add(disease_name)
                for ct in extract_cancer_types(disease_name):
                    new_cancer_types[drug_id].add(ct)
                for group_name, group_info in DISEASE_HIERARCHY_GROUPS.items():
                    parent_kws = group_info.get("parent_keywords", [])
                    if any(kw.lower() in disease_name.lower() for kw in parent_kws):
                        new_disease_groups[drug_id][group_name].add(disease_name)

    predictor.drug_train_freq = dict(new_freq)
    predictor.drug_to_diseases = dict(new_d2d)
    predictor.drug_cancer_types = dict(new_cancer_types)
    predictor.drug_disease_groups = dict(new_disease_groups)
    return originals


def restore_gt_structures(predictor, originals):
    for k, v in originals.items():
        setattr(predictor, k, v)


def main():
    print("=" * 70)
    print("h617: HIGH Tier Stabilization — Seed-42 Outlier Analysis")
    print("=" * 70)

    predictor = DrugRepurposingPredictor()
    gt_path = predictor.reference_dir / "expanded_ground_truth.json"
    with open(gt_path) as f:
        gt_data = json.load(f)
    print(f"Expanded GT: {len(gt_data)} diseases, {sum(len(v) for v in gt_data.values())} pairs")

    all_diseases = sorted(
        d for d in predictor.disease_names
        if d in predictor.embeddings and d in gt_data
    )
    print(f"Diseases: {len(all_diseases)}")

    seeds = [42, 123, 456, 789, 2024]

    # Per-seed analysis of HIGH tier
    for seed in seeds:
        print(f"\n{'=' * 70}")
        print(f"SEED {seed}")
        print(f"{'=' * 70}")

        train_diseases, holdout_diseases = split_diseases(all_diseases, seed)
        train_set = set(train_diseases)
        originals = recompute_gt_structures(predictor, train_set)

        # Count HIGH predictions per holdout disease
        high_per_disease = {}
        high_rule_counts = defaultdict(list)  # rule -> [is_hit...]
        high_category_counts = defaultdict(list)

        for disease_id in holdout_diseases:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            try:
                result = predictor.predict(disease_name, top_n=30, include_filtered=True)
            except Exception:
                continue

            gt_drugs = set(gt_data.get(disease_id, []))
            disease_high = []

            for pred in result.predictions:
                if pred.confidence_tier.value == 'HIGH':
                    is_hit = pred.drug_id in gt_drugs
                    rule = pred.category_specific_tier or 'standard'
                    disease_high.append({
                        'drug': pred.drug_name,
                        'rule': rule,
                        'category': pred.category,
                        'hit': is_hit,
                    })
                    high_rule_counts[rule].append(is_hit)
                    high_category_counts[pred.category].append(is_hit)

            if disease_high:
                high_per_disease[disease_name] = disease_high

        restore_gt_structures(predictor, originals)

        total_high = sum(len(v) for v in high_per_disease.values())
        total_hits = sum(p['hit'] for v in high_per_disease.values() for p in v)
        precision = 100 * total_hits / total_high if total_high > 0 else 0

        print(f"  Total HIGH: {total_high} preds, {total_hits} hits, {precision:.1f}%")
        print(f"  Diseases with HIGH preds: {len(high_per_disease)}")

        # Per-rule breakdown
        print(f"\n  Per-rule:")
        print(f"  {'Rule':<45s} {'N':>4s} {'Hits':>4s} {'Prec%':>6s}")
        print(f"  {'-'*65}")
        for rule, results in sorted(high_rule_counts.items(), key=lambda x: -len(x[1])):
            n = len(results)
            hits = sum(results)
            prec = 100 * hits / n if n > 0 else 0
            print(f"  {rule:<45s} {n:4d} {hits:4d} {prec:5.1f}%")

        # Per-category breakdown
        print(f"\n  Per-category:")
        for cat, results in sorted(high_category_counts.items(), key=lambda x: -len(x[1])):
            n = len(results)
            hits = sum(results)
            prec = 100 * hits / n if n > 0 else 0
            print(f"  {cat:<30s} {n:4d} {hits:4d} {prec:5.1f}%")

        # Show diseases with most HIGH predictions
        print(f"\n  Top diseases by HIGH prediction count:")
        for disease, preds in sorted(high_per_disease.items(), key=lambda x: -len(x[1]))[:10]:
            n = len(preds)
            hits = sum(p['hit'] for p in preds)
            rules = set(p['rule'] for p in preds)
            print(f"    {disease[:45]:<46s}: {hits}/{n} ({100*hits/n:.0f}%) rules={rules}")

    # Summary: compare seed variability sources
    print("\n" + "=" * 70)
    print("CROSS-SEED COMPARISON")
    print("=" * 70)

    # Re-run to get per-rule per-seed data
    rule_seed_data = defaultdict(lambda: defaultdict(list))
    cat_seed_data = defaultdict(lambda: defaultdict(list))
    total_seed_data = defaultdict(list)

    for seed in seeds:
        train_diseases, holdout_diseases = split_diseases(all_diseases, seed)
        train_set = set(train_diseases)
        originals = recompute_gt_structures(predictor, train_set)

        for disease_id in holdout_diseases:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            try:
                result = predictor.predict(disease_name, top_n=30, include_filtered=True)
            except Exception:
                continue

            gt_drugs = set(gt_data.get(disease_id, []))
            for pred in result.predictions:
                if pred.confidence_tier.value == 'HIGH':
                    is_hit = pred.drug_id in gt_drugs
                    rule = pred.category_specific_tier or 'standard'
                    rule_seed_data[rule][seed].append(is_hit)
                    cat_seed_data[pred.category][seed].append(is_hit)
                    total_seed_data[seed].append(is_hit)

        restore_gt_structures(predictor, originals)

    print(f"\n{'Rule':<45s} ", end='')
    for seed in seeds:
        print(f"  S{seed:<5d}", end='')
    print(f"  {'Mean':>6s} {'Std':>5s}")
    print("-" * 100)

    for rule in sorted(rule_seed_data.keys()):
        precs = []
        line = f"  {rule:<43s} "
        for seed in seeds:
            r = rule_seed_data[rule][seed]
            if r:
                p = 100 * sum(r) / len(r)
                precs.append(p)
                line += f"  {p:5.1f}%"
            else:
                precs.append(0)
                line += f"    N/A"
        line += f"  {np.mean(precs):5.1f}% {np.std(precs):4.1f}%"
        print(line)

    print("-" * 100)
    precs = []
    line = f"  {'OVERALL HIGH':<43s} "
    for seed in seeds:
        r = total_seed_data[seed]
        p = 100 * sum(r) / len(r) if r else 0
        precs.append(p)
        line += f"  {p:5.1f}%"
    line += f"  {np.mean(precs):5.1f}% {np.std(precs):4.1f}%"
    print(line)

    # Identify the biggest variance drivers
    print("\n\nVARIANCE DRIVERS (rules with highest seed-to-seed std):")
    rule_var = []
    for rule in rule_seed_data:
        precs = []
        ns = []
        for seed in seeds:
            r = rule_seed_data[rule][seed]
            if r:
                precs.append(100 * sum(r) / len(r))
                ns.append(len(r))
        if len(precs) >= 3 and np.mean(ns) >= 3:
            rule_var.append((rule, np.std(precs), np.mean(precs), np.mean(ns)))

    for rule, std, mean, n in sorted(rule_var, key=lambda x: -x[1])[:10]:
        print(f"  {rule:<45s}: {mean:.1f}% ± {std:.1f}% (n={n:.1f}/seed)")


if __name__ == "__main__":
    main()
