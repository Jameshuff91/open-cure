#!/usr/bin/env python3
"""
h579: MEDIUM Novel-Only Precision Analysis

For each tier, compute holdout precision split into:
- Known indications (drug-disease pair in training GT)
- Novel predictions (not in training GT)

This reveals the true discovery potential of each tier.
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
    
    # Track per-tier, per-novelty results
    # novelty_type: "known" = drug known to treat THIS disease in training GT
    #               "novel_drug" = drug treats other diseases but not this one
    #               "novel_overall" = drug not in training GT at all
    tier_novel = defaultdict(lambda: defaultdict(list))  # tier -> novelty -> [(prec, n)]
    tier_reason_novel = defaultdict(lambda: defaultdict(list))
    
    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        train_diseases, holdout_diseases = split_diseases(all_diseases, seed)
        train_set = set(train_diseases)
        
        # Build training GT set: which (drug, disease) pairs are in training?
        train_gt_pairs = set()
        train_gt_drugs = set()
        for d_id in train_set:
            if d_id in predictor.ground_truth:
                for drug_id in predictor.ground_truth[d_id]:
                    train_gt_pairs.add((d_id, drug_id))
                    train_gt_drugs.add(drug_id)
        
        originals = recompute_gt_structures(predictor, train_set)
        
        # Per-seed accumulators
        tier_counts = defaultdict(lambda: defaultdict(lambda: {'hits': 0, 'total': 0}))
        reason_counts = defaultdict(lambda: defaultdict(lambda: {'hits': 0, 'total': 0}))
        
        for disease_id in holdout_diseases:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            try:
                result = predictor.predict(disease_name, top_n=30, include_filtered=True)
            except Exception:
                continue
            
            for p in result.predictions:
                is_hit = (disease_id, p.drug_id) in gt_set
                tier = p.confidence_tier.name
                reason = p.category_specific_tier or 'default'
                
                # Determine novelty type
                # "known_to_disease" = drug known for THIS holdout disease
                #   (not possible in holdout - but drug may treat a RELATED disease that shares GT)
                # Actually, in holdout context:
                # - The drug may have been known to treat OTHER training diseases
                # - The prediction to THIS holdout disease is always "novel" in a sense
                # But what we really want is: does the drug treat ANY disease in training?
                drug_in_training = p.drug_id in train_gt_drugs
                
                if drug_in_training:
                    novelty = 'repurposing'  # Drug known but being repurposed to new disease
                else:
                    novelty = 'novel_drug'  # Drug not in training at all
                
                tier_counts[tier][novelty]['total'] += 1
                if is_hit:
                    tier_counts[tier][novelty]['hits'] += 1
                
                reason_counts[reason][novelty]['total'] += 1
                if is_hit:
                    reason_counts[reason][novelty]['hits'] += 1
                
                # Also track ALL
                tier_counts[tier]['all']['total'] += 1
                if is_hit:
                    tier_counts[tier]['all']['hits'] += 1
        
        restore_gt_structures(predictor, originals)
        
        # Accumulate per-seed
        for tier in tier_counts:
            for novelty in tier_counts[tier]:
                c = tier_counts[tier][novelty]
                if c['total'] > 0:
                    prec = 100 * c['hits'] / c['total']
                    tier_novel[tier][novelty].append((prec, c['total']))
        
        for reason in reason_counts:
            for novelty in reason_counts[reason]:
                c = reason_counts[reason][novelty]
                if c['total'] > 0:
                    prec = 100 * c['hits'] / c['total']
                    tier_reason_novel[reason][novelty].append((prec, c['total']))
    
    # === Print results ===
    print("\n" + "="*80)
    print("h579: Novel vs Known Holdout Precision by Tier")
    print("="*80)
    
    print(f"\n{'Tier':<10} {'Type':<15} {'Holdout%':>8} {'±std':>6} {'n/seed':>7} {'% of tier':>10}")
    print("-" * 60)
    
    for tier in ['GOLDEN', 'HIGH', 'MEDIUM', 'LOW', 'FILTER']:
        for novelty in ['all', 'repurposing', 'novel_drug']:
            results = tier_novel[tier][novelty]
            if results and len(results) >= 3:
                precs = [x[0] for x in results]
                ns = [x[1] for x in results]
                mean_n = np.mean(ns)
                
                # Calculate % of tier
                all_n = np.mean([x[1] for x in tier_novel[tier]['all']]) if tier_novel[tier]['all'] else 0
                pct = 100 * mean_n / all_n if all_n > 0 else 0
                
                label = novelty if novelty != 'all' else 'TOTAL'
                print(f"  {tier:<8} {label:<15} {np.mean(precs):>7.1f}% {np.std(precs):>5.1f} {mean_n:>7.0f} {pct:>9.0f}%")
        print()
    
    # MEDIUM breakdown by reason
    print("\n=== MEDIUM by reason × novelty ===")
    print(f"{'Reason':<40} {'Type':<15} {'Holdout%':>8} {'±std':>6} {'n/seed':>7}")
    print("-" * 78)
    
    # Only show MEDIUM-relevant reasons
    medium_reasons = set()
    for seed_preds in tier_novel['MEDIUM']:
        pass
    # Actually, tier_reason_novel tracks ALL reasons. Let me filter.
    for reason in sorted(tier_reason_novel.keys()):
        for novelty in ['repurposing', 'novel_drug']:
            results = tier_reason_novel[reason][novelty]
            if results and len(results) >= 3:
                precs = [x[0] for x in results]
                ns = [x[1] for x in results]
                mean_n = np.mean(ns)
                if mean_n >= 3:
                    print(f"  {reason:<38} {novelty:<15} {np.mean(precs):>7.1f}% {np.std(precs):>5.1f} {mean_n:>7.0f}")
    
    # Key question: what % of MEDIUM hits are from "novel_drug" (true discoveries)?
    print("\n=== Discovery potential ===")
    for tier in ['GOLDEN', 'HIGH', 'MEDIUM', 'LOW', 'FILTER']:
        rep_results = tier_novel[tier].get('repurposing', [])
        novel_results = tier_novel[tier].get('novel_drug', [])
        
        if rep_results and novel_results and len(rep_results) >= 3 and len(novel_results) >= 3:
            rep_hits = [x[0] * x[1] / 100 for x in rep_results]
            novel_hits = [x[0] * x[1] / 100 for x in novel_results]
            total_hits = [r + n for r, n in zip(rep_hits, novel_hits)]
            novel_pct = [100 * n / max(1, t) for n, t in zip(novel_hits, total_hits)]
            
            rep_precs = [x[0] for x in rep_results]
            novel_precs = [x[0] for x in novel_results]
            
            print(f"  {tier}: repurposing={np.mean(rep_precs):.1f}%, novel_drug={np.mean(novel_precs):.1f}%, "
                  f"novel % of hits={np.mean(novel_pct):.0f}%")
    
    print("\nDone.")


if __name__ == "__main__":
    run_analysis()

