#!/usr/bin/env python3
"""
h490: Cardiovascular ATC Coherent Full-to-Holdout Gap Investigation

Investigate the cardiovascular category's performance across all MEDIUM sub-rules.
Key questions:
1. Which MEDIUM sub-rules exist for cardiovascular predictions?
2. What is the full-data vs holdout precision for each?
3. Which diseases/drugs drive any gaps?
4. Are there sub-rules that should be demoted?

Note: 'atc_coherent_cardiovascular' is a LOW→MEDIUM boost that may be rare
if most CV drugs already reach MEDIUM through standard rules.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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
        "drug_disease_groups": {k: set(v) for k, v in predictor.drug_disease_groups.items()},
        "train_diseases": list(predictor.train_diseases),
        "train_embeddings": predictor.train_embeddings.copy(),
        "train_disease_categories": dict(predictor.train_disease_categories),
    }

    # 1. Recompute drug_train_freq
    new_freq: Dict[str, int] = defaultdict(int)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            for drug_id in predictor.ground_truth[disease_id]:
                new_freq[drug_id] += 1
    predictor.drug_train_freq = dict(new_freq)

    # 2. Recompute drug_to_diseases
    new_d2d: Dict[str, Set[str]] = defaultdict(set)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            for drug_id in predictor.ground_truth[disease_id]:
                new_d2d[drug_id].add(disease_name.lower())
    predictor.drug_to_diseases = dict(new_d2d)

    # 3. Recompute drug_cancer_types
    new_cancer: Dict[str, Set[str]] = defaultdict(set)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            ctypes = extract_cancer_types(disease_name)
            if ctypes:
                for drug_id in predictor.ground_truth[disease_id]:
                    new_cancer[drug_id].update(ctypes)
    predictor.drug_cancer_types = dict(new_cancer)

    # 4. Recompute drug_disease_groups (stores Set[Tuple[category, group_name]])
    new_groups: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            disease_lower = disease_name.lower()
            for category, groups in DISEASE_HIERARCHY_GROUPS.items():
                for group_name, variants in groups.items():
                    exclusions = HIERARCHY_EXCLUSIONS.get((category, group_name), [])
                    if any(excl in disease_lower for excl in exclusions):
                        continue
                    if any(variant in disease_lower or disease_lower in variant
                           for variant in variants):
                        for drug_id in predictor.ground_truth[disease_id]:
                            new_groups[drug_id].add((category, group_name))
    predictor.drug_disease_groups = dict(new_groups)

    # 5. Recompute train_diseases and train_embeddings
    new_train = [d for d in originals["train_diseases"] if d in train_disease_ids]
    predictor.train_diseases = new_train
    predictor.train_embeddings = np.array(
        [predictor.embeddings[d] for d in new_train]
    )

    # 6. Recompute train_disease_categories
    new_cats: Dict[str, str] = {}
    for d_id in new_train:
        d_name = predictor.disease_names.get(d_id, d_id)
        new_cats[d_id] = predictor.categorize_disease(d_name)
    predictor.train_disease_categories = new_cats

    return originals


def restore_originals(predictor: DrugRepurposingPredictor, originals: Dict) -> None:
    """Restore predictor to original state."""
    for attr, val in originals.items():
        setattr(predictor, attr, val)


def collect_predictions(predictor: DrugRepurposingPredictor,
                        disease_list: List[Tuple[str, str]],
                        target_category: str = 'cardiovascular') -> List[Dict]:
    """Collect all predictions for given diseases, focused on a category."""
    results = []
    for d_id, d_name in disease_list:
        cat = predictor.categorize_disease(d_name)
        if cat != target_category:
            continue
        result = predictor.predict(d_name, include_filtered=True)
        if result is None:
            continue
        gt_drugs = predictor.ground_truth.get(d_id, {})
        for pred in result.predictions:
            drug_gt = pred.drug_id in gt_drugs if pred.drug_id else False
            results.append({
                'disease': d_name,
                'disease_id': d_id,
                'drug': pred.drug_name,
                'drug_id': pred.drug_id,
                'rank': pred.rank,
                'score': pred.knn_score,
                'train_freq': pred.train_frequency,
                'mechanism': pred.mechanism_support,
                'in_gt': drug_gt,
                'tier': pred.confidence_tier.value,
                'rule': pred.category_specific_tier or 'standard',
            })
    return results


def run_analysis():
    print("=" * 80)
    print("h490: Cardiovascular Full-to-Holdout Gap Investigation")
    print("=" * 80)

    # Initialize predictor
    print("\nLoading predictor...")
    predictor = DrugRepurposingPredictor()

    # Get all CV diseases
    all_disease_ids = list(predictor.ground_truth.keys())
    all_diseases = [(d_id, predictor.disease_names.get(d_id, d_id)) for d_id in all_disease_ids]
    cv_diseases = [(d_id, d_name) for d_id, d_name in all_diseases
                   if predictor.categorize_disease(d_name) == 'cardiovascular']

    print(f"Total diseases with GT: {len(all_disease_ids)}")
    print(f"Cardiovascular diseases: {len(cv_diseases)}")
    print(f"CV diseases: {[d[1] for d in cv_diseases]}")

    # ======================================================================
    # Part 1: Full-data analysis - ALL tiers for CV
    # ======================================================================
    print("\n" + "=" * 60)
    print("PART 1: Full-Data Analysis - All CV Predictions")
    print("=" * 60)

    full_data_preds = collect_predictions(predictor, cv_diseases)

    # Tier breakdown
    tier_stats = defaultdict(lambda: {'n': 0, 'gt': 0})
    for p in full_data_preds:
        tier_stats[p['tier']]['n'] += 1
        if p['in_gt']:
            tier_stats[p['tier']]['gt'] += 1

    print("\n--- Tier Breakdown (CV, full-data) ---")
    for tier in ['GOLDEN', 'HIGH', 'MEDIUM', 'LOW', 'FILTER']:
        s = tier_stats[tier]
        prec = s['gt'] / s['n'] * 100 if s['n'] > 0 else 0
        print(f"  {tier}: {s['gt']}/{s['n']} = {prec:.1f}%")

    # Rule breakdown for MEDIUM
    rule_stats = defaultdict(lambda: {'n': 0, 'gt': 0})
    for p in full_data_preds:
        if p['tier'] == 'MEDIUM':
            rule_stats[p['rule']]['n'] += 1
            if p['in_gt']:
                rule_stats[p['rule']]['gt'] += 1

    print("\n--- MEDIUM Rule Breakdown (CV, full-data) ---")
    for rule, s in sorted(rule_stats.items(), key=lambda x: -x[1]['n']):
        prec = s['gt'] / s['n'] * 100 if s['n'] > 0 else 0
        print(f"  {rule}: {s['gt']}/{s['n']} = {prec:.1f}%")

    # Rule breakdown for HIGH
    rule_stats_high = defaultdict(lambda: {'n': 0, 'gt': 0})
    for p in full_data_preds:
        if p['tier'] == 'HIGH':
            rule_stats_high[p['rule']]['n'] += 1
            if p['in_gt']:
                rule_stats_high[p['rule']]['gt'] += 1

    print("\n--- HIGH Rule Breakdown (CV, full-data) ---")
    for rule, s in sorted(rule_stats_high.items(), key=lambda x: -x[1]['n']):
        prec = s['gt'] / s['n'] * 100 if s['n'] > 0 else 0
        print(f"  {rule}: {s['gt']}/{s['n']} = {prec:.1f}%")

    # Rule breakdown for GOLDEN
    rule_stats_golden = defaultdict(lambda: {'n': 0, 'gt': 0})
    for p in full_data_preds:
        if p['tier'] == 'GOLDEN':
            rule_stats_golden[p['rule']]['n'] += 1
            if p['in_gt']:
                rule_stats_golden[p['rule']]['gt'] += 1

    print("\n--- GOLDEN Rule Breakdown (CV, full-data) ---")
    for rule, s in sorted(rule_stats_golden.items(), key=lambda x: -x[1]['n']):
        prec = s['gt'] / s['n'] * 100 if s['n'] > 0 else 0
        print(f"  {rule}: {s['gt']}/{s['n']} = {prec:.1f}%")

    # Per-disease breakdown
    disease_tier_stats = defaultdict(lambda: defaultdict(lambda: {'n': 0, 'gt': 0}))
    for p in full_data_preds:
        disease_tier_stats[p['disease']][p['tier']]['n'] += 1
        if p['in_gt']:
            disease_tier_stats[p['disease']][p['tier']]['gt'] += 1

    print("\n--- Per-Disease Tier Breakdown (CV, full-data) ---")
    for disease in sorted(disease_tier_stats.keys()):
        tiers = disease_tier_stats[disease]
        total_n = sum(s['n'] for s in tiers.values())
        total_gt = sum(s['gt'] for s in tiers.values())
        total_prec = total_gt / total_n * 100 if total_n > 0 else 0
        parts = []
        for tier in ['GOLDEN', 'HIGH', 'MEDIUM', 'LOW', 'FILTER']:
            s = tiers[tier]
            if s['n'] > 0:
                p = s['gt'] / s['n'] * 100
                parts.append(f"{tier}:{s['gt']}/{s['n']}({p:.0f}%)")
        print(f"  {disease}: {total_gt}/{total_n} ({total_prec:.0f}%) | {' '.join(parts)}")

    # ======================================================================
    # Part 2: Holdout analysis - ALL tiers for CV
    # ======================================================================
    print("\n" + "=" * 60)
    print("PART 2: Holdout Analysis (5 seeds)")
    print("=" * 60)

    seeds = [42, 123, 456, 789, 2024]
    all_seed_results = []

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")

        train_ids, holdout_ids = split_diseases(all_disease_ids, seed)
        train_set = set(train_ids)
        holdout_set = set(holdout_ids)

        cv_holdout = [(d_id, d_name) for d_id, d_name in cv_diseases if d_id in holdout_set]
        print(f"  CV holdout diseases: {len(cv_holdout)} - {[d[1] for d in cv_holdout]}")

        # Recompute GT structures
        originals = recompute_gt_structures(predictor, train_set)

        holdout_preds = collect_predictions(predictor, cv_holdout)

        # Restore
        restore_originals(predictor, originals)

        # Tier breakdown
        tier_stats_ho = defaultdict(lambda: {'n': 0, 'gt': 0})
        for p in holdout_preds:
            tier_stats_ho[p['tier']]['n'] += 1
            if p['in_gt']:
                tier_stats_ho[p['tier']]['gt'] += 1

        for tier in ['GOLDEN', 'HIGH', 'MEDIUM', 'LOW', 'FILTER']:
            s = tier_stats_ho[tier]
            prec = s['gt'] / s['n'] * 100 if s['n'] > 0 else 0
            if s['n'] > 0:
                print(f"  {tier}: {s['gt']}/{s['n']} = {prec:.1f}%")

        # Rule breakdown for MEDIUM
        rule_stats_ho = defaultdict(lambda: {'n': 0, 'gt': 0})
        for p in holdout_preds:
            if p['tier'] == 'MEDIUM':
                rule_stats_ho[p['rule']]['n'] += 1
                if p['in_gt']:
                    rule_stats_ho[p['rule']]['gt'] += 1

        if rule_stats_ho:
            print(f"  MEDIUM rules:")
            for rule, s in sorted(rule_stats_ho.items(), key=lambda x: -x[1]['n']):
                prec = s['gt'] / s['n'] * 100 if s['n'] > 0 else 0
                print(f"    {rule}: {s['gt']}/{s['n']} = {prec:.1f}%")

        # Per-disease
        disease_ho = defaultdict(lambda: {'n': 0, 'gt': 0})
        for p in holdout_preds:
            disease_ho[p['disease']]['n'] += 1
            if p['in_gt']:
                disease_ho[p['disease']]['gt'] += 1

        for disease, s in sorted(disease_ho.items(), key=lambda x: -x[1]['n']):
            prec = s['gt'] / s['n'] * 100 if s['n'] > 0 else 0
            print(f"    {disease}: {s['gt']}/{s['n']} = {prec:.1f}%")

        all_seed_results.append({
            'seed': seed,
            'holdout_preds': holdout_preds,
            'tier_stats': {t: dict(s) for t, s in tier_stats_ho.items()},
        })

    # ======================================================================
    # Part 3: Aggregate Holdout Results
    # ======================================================================
    print("\n" + "=" * 60)
    print("PART 3: Aggregate Results")
    print("=" * 60)

    # Per-tier aggregation
    for tier in ['GOLDEN', 'HIGH', 'MEDIUM', 'LOW', 'FILTER']:
        precs = []
        ns = []
        for r in all_seed_results:
            s = r['tier_stats'].get(tier, {'n': 0, 'gt': 0})
            if s['n'] > 0:
                precs.append(s['gt'] / s['n'] * 100)
                ns.append(s['n'])
        if precs:
            mean_p = np.mean(precs)
            std_p = np.std(precs)
            mean_n = np.mean(ns)
            # Get full-data for comparison
            fd = tier_stats[tier]
            fd_prec = fd['gt'] / fd['n'] * 100 if fd['n'] > 0 else 0
            gap = fd_prec - mean_p
            print(f"  {tier}: holdout={mean_p:.1f}% ± {std_p:.1f}% (n={mean_n:.0f}/seed), "
                  f"full={fd_prec:.1f}% (n={fd['n']}), gap={gap:+.1f}pp")

    # Per-rule aggregation for MEDIUM
    print("\n--- MEDIUM Rule Aggregation ---")
    rule_agg = defaultdict(lambda: {'precs': [], 'ns': []})
    for r in all_seed_results:
        rule_ho = defaultdict(lambda: {'n': 0, 'gt': 0})
        for p in r['holdout_preds']:
            if p['tier'] == 'MEDIUM':
                rule_ho[p['rule']]['n'] += 1
                if p['in_gt']:
                    rule_ho[p['rule']]['gt'] += 1
        for rule, s in rule_ho.items():
            if s['n'] > 0:
                rule_agg[rule]['precs'].append(s['gt'] / s['n'] * 100)
                rule_agg[rule]['ns'].append(s['n'])

    for rule, agg in sorted(rule_agg.items(), key=lambda x: -np.mean(x[1]['ns'])):
        mean_p = np.mean(agg['precs'])
        std_p = np.std(agg['precs'])
        mean_n = np.mean(agg['ns'])
        fd = rule_stats.get(rule, {'n': 0, 'gt': 0})
        fd_prec = fd['gt'] / fd['n'] * 100 if fd['n'] > 0 else 0
        gap = fd_prec - mean_p
        print(f"  {rule}: holdout={mean_p:.1f}% ± {std_p:.1f}% (n={mean_n:.0f}/seed), "
              f"full={fd_prec:.1f}% (n={fd['n']}), gap={gap:+.1f}pp")

    # ======================================================================
    # Part 4: Self-referential check
    # ======================================================================
    print("\n" + "=" * 60)
    print("PART 4: Self-Referential Pattern Check")
    print("=" * 60)

    from sklearn.metrics.pairwise import cosine_similarity

    for d_id, d_name in cv_diseases:
        if d_id not in predictor.embeddings:
            continue
        test_emb = predictor.embeddings[d_id].reshape(1, -1)
        sims = cosine_similarity(test_emb, predictor.train_embeddings)[0]
        top_k_idx = np.argsort(sims)[::-1][:20]
        neighbors = [predictor.train_diseases[i] for i in top_k_idx]

        self_in_neighbors = d_id in neighbors
        self_rank = neighbors.index(d_id) + 1 if self_in_neighbors else -1

        # What fraction of GT comes from self?
        gt_drugs = set(predictor.ground_truth.get(d_id, {}).keys())
        if not gt_drugs:
            continue

        # Count how many GT drugs appear in neighbors' GT
        gt_from_self = 0
        gt_from_others = 0
        for drug_id in gt_drugs:
            found_in_self = False
            found_in_other = False
            for n_id in neighbors:
                if n_id == d_id:
                    if drug_id in predictor.ground_truth.get(n_id, {}):
                        found_in_self = True
                else:
                    if drug_id in predictor.ground_truth.get(n_id, {}):
                        found_in_other = True
            if found_in_self and not found_in_other:
                gt_from_self += 1
            elif found_in_other:
                gt_from_others += 1

        total = gt_from_self + gt_from_others
        self_pct = gt_from_self / total * 100 if total > 0 else 0
        marker = " ** SELF-REF" if self_pct > 50 else ""
        print(f"  {d_name}: self={'rank'+str(self_rank) if self_in_neighbors else 'no'}, "
              f"GT self-only={gt_from_self}/{total} ({self_pct:.0f}%){marker}")

    # ======================================================================
    # Part 5: Drug frequency analysis across seeds
    # ======================================================================
    print("\n" + "=" * 60)
    print("PART 5: Drug Frequency Drop Analysis")
    print("=" * 60)

    # For MEDIUM CV predictions, check train_freq on full-data vs holdout
    # Focus on drugs that appear in both
    full_drug_freqs = {}
    for p in full_data_preds:
        if p['tier'] == 'MEDIUM':
            full_drug_freqs[p['drug']] = p['train_freq']

    print(f"  Unique CV MEDIUM drugs (full-data): {len(full_drug_freqs)}")

    # Aggregate holdout frequencies for same drugs
    holdout_drug_freqs = defaultdict(list)
    for r in all_seed_results:
        for p in r['holdout_preds']:
            if p['tier'] == 'MEDIUM':
                holdout_drug_freqs[p['drug']].append(p['train_freq'])

    # Compare
    freq_drops = []
    for drug in full_drug_freqs:
        if drug in holdout_drug_freqs:
            full_freq = full_drug_freqs[drug]
            mean_ho_freq = np.mean(holdout_drug_freqs[drug])
            drop = full_freq - mean_ho_freq
            freq_drops.append((drug, full_freq, mean_ho_freq, drop))

    freq_drops.sort(key=lambda x: -x[3])
    print(f"\n  Top 15 drugs by frequency drop (full→holdout):")
    for drug, ff, hf, drop in freq_drops[:15]:
        print(f"    {drug}: {ff} → {hf:.1f} (drop={drop:.1f})")

    if freq_drops:
        mean_drop = np.mean([d[3] for d in freq_drops])
        mean_pct = np.mean([d[3]/d[1]*100 for d in freq_drops if d[1] > 0])
        print(f"\n  Mean absolute drop: {mean_drop:.1f}")
        print(f"  Mean relative drop: {mean_pct:.1f}%")

    # ======================================================================
    # Summary
    # ======================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("""
This analysis examines cardiovascular predictions across ALL tiers and rules
to understand the full-to-holdout precision gap for the cardiovascular category.

Key findings will help determine:
1. Whether CV ATC coherent boost is effective
2. Which CV diseases are self-referential
3. Whether frequency inflation drives the gap
4. What specific actions to take
""")

    # Save results
    output = {
        'full_data_tier_stats': {t: dict(s) for t, s in tier_stats.items()},
        'full_data_medium_rules': {r: dict(s) for r, s in rule_stats.items()},
        'cv_diseases': [{'id': d[0], 'name': d[1]} for d in cv_diseases],
    }

    with open('data/analysis/h490_cv_atc_coherent_gap.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print("\nSaved to data/analysis/h490_cv_atc_coherent_gap.json")


if __name__ == '__main__':
    run_analysis()
