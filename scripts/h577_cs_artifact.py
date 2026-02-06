#!/usr/bin/env python3
"""
h577: Quantify corticosteroid holdout artifact across all tiers.

High-frequency CS drugs (dexamethasone freq=42, prednisone freq=33, etc.) have
such broad GT coverage that they inflate holdout precision in any subgroup.
Measure the artifact size by computing CS vs non-CS precision per tier.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from production_predictor import (
    DrugRepurposingPredictor, extract_cancer_types,
    DISEASE_HIERARCHY_GROUPS, HIERARCHY_EXCLUSIONS,
)

# All corticosteroid drug names (lowercase)
CORTICOSTEROIDS = {
    'hydrocortisone', 'cortisone', 'prednisone', 'prednisolone',
    'methylprednisolone', 'triamcinolone', 'dexamethasone',
    'betamethasone', 'fluticasone', 'budesonide', 'mometasone',
    'beclomethasone', 'fluocinolone', 'clobetasol', 'halobetasol',
    'corticotropin', 'fludrocortisone', 'cortisone acetate',
    'prednisolone acetate', 'fluocinonide', 'desoximetasone',
    'alclometasone', 'desonide', 'diflorasone', 'flurandrenolide',
    'amcinonide', 'halcinonide', 'ciclesonide',
}


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
    for key, value in originals.items():
        setattr(predictor, key, value)


def run_analysis(seeds=[42, 123, 456, 789, 2024]):
    predictor = DrugRepurposingPredictor()
    
    gt_path = predictor.reference_dir / "expanded_ground_truth.json"
    with open(gt_path) as f:
        gt_data = json.load(f)
    
    gt_set = set()
    for disease_id, drugs in gt_data.items():
        for drug in drugs:
            if isinstance(drug, str):
                gt_set.add((disease_id, drug))
            elif isinstance(drug, dict):
                gt_set.add((disease_id, drug.get("drug_id") or drug.get("drug")))
    
    all_diseases = [d for d in predictor.ground_truth if d in predictor.embeddings]
    
    # Track per-tier, CS vs non-CS
    tier_cs = defaultdict(list)      # tier -> [(prec, n)]
    tier_noncs = defaultdict(list)   # tier -> [(prec, n)]
    tier_all = defaultdict(list)     # tier -> [(prec, n)]
    
    # Also by category × CS
    cat_cs = defaultdict(list)
    cat_noncs = defaultdict(list)
    
    for seed in seeds:
        print(f"Seed {seed}...")
        train_diseases, holdout_diseases = split_diseases(all_diseases, seed)
        train_set = set(train_diseases)
        
        originals = recompute_gt_structures(predictor, train_set)
        
        cs_counts = defaultdict(lambda: defaultdict(lambda: {'hits': 0, 'total': 0}))
        cat_counts = defaultdict(lambda: defaultdict(lambda: {'hits': 0, 'total': 0}))
        
        for disease_id in holdout_diseases:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            try:
                result = predictor.predict(disease_name, top_n=30, include_filtered=True)
            except Exception:
                continue
            
            for p in result.predictions:
                is_hit = (disease_id, p.drug_id) in gt_set
                tier = p.confidence_tier.name
                is_cs = p.drug_name.lower() in CORTICOSTEROIDS
                drug_type = 'CS' if is_cs else 'non-CS'
                
                cs_counts[tier][drug_type]['total'] += 1
                if is_hit:
                    cs_counts[tier][drug_type]['hits'] += 1
                
                cs_counts[tier]['all']['total'] += 1
                if is_hit:
                    cs_counts[tier]['all']['hits'] += 1
                
                cat_counts[p.category][drug_type]['total'] += 1
                if is_hit:
                    cat_counts[p.category][drug_type]['hits'] += 1
        
        restore_gt_structures(predictor, originals)
        
        for tier in cs_counts:
            for dtype in ['CS', 'non-CS', 'all']:
                c = cs_counts[tier][dtype]
                if c['total'] > 0:
                    prec = 100 * c['hits'] / c['total']
                    if dtype == 'CS':
                        tier_cs[tier].append((prec, c['total']))
                    elif dtype == 'non-CS':
                        tier_noncs[tier].append((prec, c['total']))
                    else:
                        tier_all[tier].append((prec, c['total']))
        
        for cat in cat_counts:
            for dtype in ['CS', 'non-CS']:
                c = cat_counts[cat][dtype]
                if c['total'] > 0:
                    prec = 100 * c['hits'] / c['total']
                    if dtype == 'CS':
                        cat_cs[cat].append((prec, c['total']))
                    else:
                        cat_noncs[cat].append((prec, c['total']))
    
    # Results
    print("\n" + "="*80)
    print("h577: Corticosteroid Holdout Artifact")
    print("="*80)
    
    print(f"\n{'Tier':<10} {'All':>18} {'CS':>18} {'Non-CS':>18} {'CS Inflation':>14}")
    print("-" * 80)
    
    for tier in ['GOLDEN', 'HIGH', 'MEDIUM', 'LOW', 'FILTER']:
        all_results = tier_all.get(tier, [])
        cs_results = tier_cs.get(tier, [])
        noncs_results = tier_noncs.get(tier, [])
        
        all_str = ""
        cs_str = ""
        noncs_str = ""
        inflation = ""
        
        if all_results and len(all_results) >= 3:
            all_precs = [x[0] for x in all_results]
            all_ns = [x[1] for x in all_results]
            all_str = f"{np.mean(all_precs):5.1f}% (n={np.mean(all_ns):.0f})"
        
        if cs_results and len(cs_results) >= 3:
            cs_precs = [x[0] for x in cs_results]
            cs_ns = [x[1] for x in cs_results]
            cs_str = f"{np.mean(cs_precs):5.1f}% (n={np.mean(cs_ns):.0f})"
        
        if noncs_results and len(noncs_results) >= 3:
            noncs_precs = [x[0] for x in noncs_results]
            noncs_ns = [x[1] for x in noncs_results]
            noncs_str = f"{np.mean(noncs_precs):5.1f}% (n={np.mean(noncs_ns):.0f})"
        
        if cs_results and noncs_results and len(cs_results) >= 3 and len(noncs_results) >= 3:
            diff = np.mean([x[0] for x in cs_results]) - np.mean([x[0] for x in noncs_results])
            inflation = f"{diff:+.1f}pp"
        
        print(f"  {tier:<8} {all_str:>18} {cs_str:>18} {noncs_str:>18} {inflation:>14}")
    
    # CS as % of each tier
    print(f"\n{'Tier':<10} {'CS count':>12} {'Total':>12} {'CS %':>8}")
    print("-" * 45)
    for tier in ['GOLDEN', 'HIGH', 'MEDIUM', 'LOW', 'FILTER']:
        cs_n = np.mean([x[1] for x in tier_cs.get(tier, [(0,0)])]) if tier_cs.get(tier) else 0
        all_n = np.mean([x[1] for x in tier_all.get(tier, [(0,0)])]) if tier_all.get(tier) else 0
        pct = 100 * cs_n / all_n if all_n > 0 else 0
        print(f"  {tier:<8} {cs_n:>11.0f} {all_n:>11.0f} {pct:>7.0f}%")
    
    # Category × CS
    print(f"\n=== CS vs non-CS by category (all tiers combined) ===")
    print(f"{'Category':<20} {'CS holdout':>12} {'non-CS':>12} {'inflation':>10} {'CS n':>8} {'nonCS n':>8}")
    print("-" * 72)
    
    all_cats = sorted(set(list(cat_cs.keys()) + list(cat_noncs.keys())))
    for cat in all_cats:
        cs_r = cat_cs.get(cat, [])
        noncs_r = cat_noncs.get(cat, [])
        
        if cs_r and noncs_r and len(cs_r) >= 3 and len(noncs_r) >= 3:
            cs_mean = np.mean([x[0] for x in cs_r])
            noncs_mean = np.mean([x[0] for x in noncs_r])
            cs_n = np.mean([x[1] for x in cs_r])
            noncs_n = np.mean([x[1] for x in noncs_r])
            diff = cs_mean - noncs_mean
            print(f"  {cat:<18} {cs_mean:>11.1f}% {noncs_mean:>11.1f}% {diff:>+9.1f}pp {cs_n:>7.0f} {noncs_n:>7.0f}")
    
    print("\nDone.")


if __name__ == "__main__":
    run_analysis()

