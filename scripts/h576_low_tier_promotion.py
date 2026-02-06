#!/usr/bin/env python3
"""
h576: LOW Tier Promotion Analysis

Uses exact same methodology as h393_holdout_tier_validation.py.
"""

import json
import sys
from collections import Counter, defaultdict
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


def split_diseases(all_diseases, seed, train_ratio=0.8):
    rng = np.random.RandomState(seed)
    shuffled = list(all_diseases)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def recompute_gt_structures(predictor, train_disease_ids):
    """Exact copy from h393."""
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


def analyze_subgroup(filter_fn, all_preds, min_seeds=3, min_n=2):
    """Compute 5-seed holdout for a subgroup."""
    results = []
    for seed_preds in all_preds:
        subset = [p for p in seed_preds if filter_fn(p)]
        if subset:
            prec = 100 * sum(p['is_hit'] for p in subset) / len(subset)
            results.append((prec, len(subset)))
    if len(results) >= min_seeds:
        precs = [x[0] for x in results]
        ns = [x[1] for x in results]
        mean_n = np.mean(ns)
        if mean_n >= min_n:
            return np.mean(precs), np.std(precs), mean_n
    return None, None, None


def run_holdout(seeds=[42, 123, 456, 789, 2024]):
    predictor = DrugRepurposingPredictor()
    
    # Load GT same as h393
    gt_path = predictor.reference_dir / "expanded_ground_truth.json"
    with open(gt_path) as f:
        gt_data = json.load(f)
    
    # Build GT set for lookup (same as h393)
    gt_set = set()
    for disease_id, drugs in gt_data.items():
        for drug in drugs:
            if isinstance(drug, str):
                gt_set.add((disease_id, drug))
            elif isinstance(drug, dict):
                gt_set.add((disease_id, drug.get("drug_id") or drug.get("drug")))
    
    # Use ONLY diseases with GT + embeddings (same as h393)
    all_diseases = [d for d in predictor.ground_truth if d in predictor.embeddings]
    print(f"Diseases with GT + embeddings: {len(all_diseases)}")
    
    all_low = []
    all_medium = []
    all_tiers = defaultdict(list)
    
    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        train_diseases, holdout_diseases = split_diseases(all_diseases, seed)
        train_set = set(train_diseases)
        
        originals = recompute_gt_structures(predictor, train_set)
        
        low_preds = []
        medium_preds = []
        tier_counts = defaultdict(lambda: {'hits': 0, 'total': 0})
        
        for disease_id in holdout_diseases:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            try:
                result = predictor.predict(disease_name, top_n=30, include_filtered=True)
            except Exception:
                continue
            
            for p in result.predictions:
                is_hit = (disease_id, p.drug_id) in gt_set
                tier_str = p.confidence_tier.name
                
                tier_counts[tier_str]['total'] += 1
                if is_hit:
                    tier_counts[tier_str]['hits'] += 1
                
                pred_info = {
                    'disease_id': disease_id,
                    'disease_name': disease_name,
                    'drug_id': p.drug_id,
                    'drug_name': p.drug_name,
                    'tier': tier_str,
                    'reason': p.category_specific_tier or 'default',
                    'rank': p.rank,
                    'knn_score': float(p.knn_score),
                    'norm_score': float(p.norm_score),
                    'mechanism': p.mechanism_support,
                    'transe': p.transe_consilience,
                    'category': p.category,
                    'frequency': p.train_frequency,
                    'is_hit': is_hit,
                    'has_targets': p.has_targets,
                }
                
                if tier_str == 'LOW':
                    low_preds.append(pred_info)
                elif tier_str == 'MEDIUM':
                    medium_preds.append(pred_info)
        
        restore_gt_structures(predictor, originals)
        
        all_low.append(low_preds)
        all_medium.append(medium_preds)
        
        for tier, counts in tier_counts.items():
            prec = 100 * counts['hits'] / max(1, counts['total'])
            all_tiers[tier].append((prec, counts['total']))
        
        n_low = len(low_preds)
        n_low_hits = sum(p['is_hit'] for p in low_preds)
        n_med = len(medium_preds)
        n_med_hits = sum(p['is_hit'] for p in medium_preds)
        print(f"  LOW: {n_low} preds, {n_low_hits} hits ({100*n_low_hits/max(1,n_low):.1f}%)")
        print(f"  MEDIUM: {n_med} preds, {n_med_hits} hits ({100*n_med_hits/max(1,n_med):.1f}%)")
    
    # === ANALYSIS ===
    print("\n" + "="*80)
    print("ANALYSIS: LOW Tier Sub-Population Holdout Precision")
    print("="*80)
    
    # Tier overview
    print("\n--- Tier Overview (5-seed mean ± std) ---")
    for tier in ['GOLDEN', 'HIGH', 'MEDIUM', 'LOW', 'FILTER']:
        if all_tiers[tier]:
            precs = [x[0] for x in all_tiers[tier]]
            ns = [x[1] for x in all_tiers[tier]]
            print(f"  {tier}: {np.mean(precs):.1f}% ± {np.std(precs):.1f}% (n={np.mean(ns):.0f}/seed)")
    
    # 1. LOW by tier_rule
    print("\n--- 1. LOW by tier_rule ---")
    all_reasons = set()
    for preds in all_low:
        for p in preds:
            all_reasons.add(p['reason'])
    
    rule_data = {}
    for reason in all_reasons:
        mean, std, n = analyze_subgroup(lambda p, r=reason: p['reason'] == r, all_low)
        if mean is not None:
            rule_data[reason] = (mean, std, n)
    
    print(f"  {'Rule':<45} {'Holdout%':>8} {'±std':>6} {'n/seed':>7}")
    print("  " + "-" * 68)
    for reason in sorted(rule_data.keys(), key=lambda r: -rule_data[r][0]):
        mean, std, n = rule_data[reason]
        marker = " ***" if mean > 25 and n >= 5 else ""
        print(f"  {reason:<45} {mean:>7.1f}% {std:>5.1f} {n:>7.1f}{marker}")
    
    # 2. TransE within LOW
    print("\n--- 2. TransE consilience within LOW ---")
    for flag in [True, False]:
        mean, std, n = analyze_subgroup(lambda p, f=flag: p['transe'] == f, all_low)
        if mean is not None:
            print(f"  TransE={flag}: {mean:.1f}% ± {std:.1f}% (n={n:.0f}/seed)")
    
    # 3. TransE × tier_rule
    print("\n--- 3. TransE=True × tier_rule ---")
    for reason in sorted(all_reasons):
        mean, std, n = analyze_subgroup(
            lambda p, r=reason: p['transe'] and p['reason'] == r, all_low, min_n=1)
        if mean is not None:
            marker = " ***" if mean > 30 else ""
            print(f"  {reason:<45} {mean:>7.1f}% {std:>5.1f} {n:>7.1f}{marker}")
    
    # 4. Mechanism within LOW
    print("\n--- 4. Mechanism support within LOW ---")
    for flag in [True, False]:
        mean, std, n = analyze_subgroup(lambda p, f=flag: p['mechanism'] == f, all_low)
        if mean is not None:
            print(f"  Mechanism={flag}: {mean:.1f}% ± {std:.1f}% (n={n:.0f}/seed)")
    
    # 5. Category within LOW
    print("\n--- 5. Category within LOW ---")
    all_cats = set()
    for preds in all_low:
        for p in preds:
            all_cats.add(p['category'])
    
    cat_data = {}
    for cat in sorted(all_cats):
        mean, std, n = analyze_subgroup(lambda p, c=cat: p['category'] == c, all_low, min_n=3)
        if mean is not None:
            cat_data[cat] = (mean, std, n)
    
    for cat in sorted(cat_data.keys(), key=lambda c: -cat_data[c][0]):
        mean, std, n = cat_data[cat]
        print(f"  {cat:<23} {mean:>7.1f}% {std:>5.1f} {n:>7.1f}")
    
    # 6. Rank within LOW
    print("\n--- 6. Rank bucket within LOW ---")
    for lo, hi, label in [(1,5,'rank_1-5'), (6,10,'rank_6-10'), (11,20,'rank_11-20')]:
        mean, std, n = analyze_subgroup(lambda p, l=lo, h=hi: l <= p['rank'] <= h, all_low)
        if mean is not None:
            print(f"  {label}: {mean:.1f}% ± {std:.1f}% (n={n:.0f}/seed)")
    
    # 7. Compound signals
    print("\n--- 7. Compound signal analysis ---")
    groups = [
        ('TransE only', lambda p: p['transe'] and not p['mechanism']),
        ('Mech only', lambda p: p['mechanism'] and not p['transe']),
        ('TransE+Mech', lambda p: p['transe'] and p['mechanism']),
        ('TransE+Rank<=10', lambda p: p['transe'] and p['rank'] <= 10),
        ('Mech+Rank<=10', lambda p: p['mechanism'] and p['rank'] <= 10),
        ('TransE+Mech+Rank<=10', lambda p: p['transe'] and p['mechanism'] and p['rank'] <= 10),
        ('Rank<=5+Mech', lambda p: p['rank'] <= 5 and p['mechanism']),
        ('Rank<=5+TransE', lambda p: p['rank'] <= 5 and p['transe']),
        ('Freq>=10+Rank<=5', lambda p: p['frequency'] >= 10 and p['rank'] <= 5),
        ('Freq>=10+Mech', lambda p: p['frequency'] >= 10 and p['mechanism']),
        ('Freq>=15+Rank<=10', lambda p: p['frequency'] >= 15 and p['rank'] <= 10),
        ('Freq>=15+Mech', lambda p: p['frequency'] >= 15 and p['mechanism']),
        ('Freq>=10+Rank<=10+Mech', lambda p: p['frequency'] >= 10 and p['rank'] <= 10 and p['mechanism']),
    ]
    
    print(f"  {'Signal combination':<35} {'Holdout%':>8} {'±std':>6} {'n/seed':>7}")
    print("  " + "-" * 58)
    for name, fn in groups:
        mean, std, n = analyze_subgroup(fn, all_low, min_n=2)
        if mean is not None:
            marker = " ***" if mean > 30 and n >= 5 else ""
            print(f"  {name:<35} {mean:>7.1f}% {std:>5.1f} {n:>7.1f}{marker}")
    
    # 8. MEDIUM baseline
    print("\n--- 8. MEDIUM holdout baseline ---")
    mean, std, n = analyze_subgroup(lambda p: True, all_medium)
    if mean is not None:
        print(f"  MEDIUM overall: {mean:.1f}% ± {std:.1f}% (n={n:.0f}/seed)")
    
    # 9. Promotion candidates
    print("\n--- 9. PROMOTION CANDIDATES (holdout > 25%, n >= 5) ---")
    found = False
    for reason in sorted(rule_data.keys(), key=lambda r: -rule_data[r][0]):
        mean, std, n = rule_data[reason]
        if mean > 25 and n >= 5:
            found = True
            print(f"\n  *** {reason}: {mean:.1f}% ± {std:.1f}% (n={n:.0f}/seed)")
            for cat in sorted(all_cats):
                cat_mean, cat_std, cat_n = analyze_subgroup(
                    lambda p, r=reason, c=cat: p['reason'] == r and p['category'] == c,
                    all_low, min_n=2)
                if cat_mean is not None:
                    print(f"      {cat}: {cat_mean:.1f}% ± {cat_std:.1f}% (n={cat_n:.0f})")
    if not found:
        print("  None found. Checking 20% threshold...")
        for reason in sorted(rule_data.keys(), key=lambda r: -rule_data[r][0]):
            mean, std, n = rule_data[reason]
            if mean > 20 and n >= 5:
                print(f"  ** {reason}: {mean:.1f}% ± {std:.1f}% (n={n:.0f}/seed)")
    
    # 10. TransE LOW vs MEDIUM
    print("\n--- 10. TransE LOW vs MEDIUM ---")
    m1, s1, n1 = analyze_subgroup(lambda p: p['transe'], all_low)
    m2, s2, n2 = analyze_subgroup(lambda p: p['transe'], all_medium)
    m3, _, n3 = analyze_subgroup(lambda p: not p['transe'], all_medium)
    if m1: print(f"  TransE LOW:       {m1:.1f}% ± {s1:.1f}% (n={n1:.0f})")
    if m2: print(f"  TransE MEDIUM:    {m2:.1f}% ± {s2:.1f}% (n={n2:.0f})")
    if m3: print(f"  No-TransE MEDIUM: {m3:.1f}% (n={n3:.0f})")
    
    # 11. Demoted rules
    print("\n--- 11. Demoted LOW rules ---")
    demotion_rules = [r for r in rule_data.keys() 
                     if 'demotion' in r or 'mismatch' in r or 'isolated' in r]
    for reason in sorted(demotion_rules, key=lambda r: -rule_data[r][0]):
        mean, std, n = rule_data[reason]
        print(f"  {reason:<45} {mean:>7.1f}% {std:>5.1f} {n:>7.1f}")
    
    print("\nDone.")


if __name__ == "__main__":
    run_holdout()

