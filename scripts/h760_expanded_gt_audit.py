#!/usr/bin/env python3
"""h760 Part 2: Check expanded GT coverage for hierarchy rules.

The internal GT has only ~3000 pairs. Expanded GT has ~57K pairs.
Many hierarchy rules may become evaluable with expanded GT.
"""

import sys
sys.path.insert(0, '.')

import json
from collections import defaultdict

from src.production_predictor import (
    DISEASE_HIERARCHY_GROUPS,
    HIERARCHY_EXCLUSIONS,
    DrugRepurposingPredictor,
)


def main():
    print("Loading predictor...")
    predictor = DrugRepurposingPredictor()
    
    # Load expanded GT
    with open('data/reference/expanded_ground_truth.json') as f:
        expanded_gt = json.load(f)
    
    print(f"Internal GT: {sum(len(v) for v in predictor.ground_truth.values())} pairs across {len(predictor.ground_truth)} diseases")
    print(f"Expanded GT: {sum(len(v) for v in expanded_gt.items())} entries")
    
    # For expanded GT, map disease IDs to names
    # expanded GT is keyed by disease_id, values are lists of drug_ids
    
    # Match expanded GT diseases to hierarchy groups
    expanded_per_group = defaultdict(set)
    expanded_drugs_per_group = defaultdict(set)
    
    for disease_id, drug_ids in expanded_gt.items():
        # Get disease name from predictor
        disease_name = predictor.disease_names.get(disease_id, '')
        if not disease_name:
            # Try DRKG names
            disease_name = predictor.drkg_disease_names.get(disease_id, '') if hasattr(predictor, 'drkg_disease_names') else ''
        if not disease_name:
            continue
            
        d_lower = disease_name.lower()
        
        for category, groups in DISEASE_HIERARCHY_GROUPS.items():
            for group_name, variants in groups.items():
                excl = HIERARCHY_EXCLUSIONS.get((category, group_name), [])
                if any(e in d_lower for e in excl):
                    continue
                if any(v in d_lower or d_lower in v for v in variants):
                    expanded_per_group[(category, group_name)].add(disease_id)
                    if isinstance(drug_ids, list):
                        for d in drug_ids:
                            expanded_drugs_per_group[(category, group_name)].add(d)
                    break
    
    # Also get internal GT for comparison
    internal_per_group = defaultdict(set)
    for disease_id, gt_drugs in predictor.ground_truth.items():
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        d_lower = disease_name.lower()
        
        for category, groups in DISEASE_HIERARCHY_GROUPS.items():
            for group_name, variants in groups.items():
                excl = HIERARCHY_EXCLUSIONS.get((category, group_name), [])
                if any(e in d_lower for e in excl):
                    continue
                if any(v in d_lower or d_lower in v for v in variants):
                    internal_per_group[(category, group_name)].add(disease_id)
                    break
    
    # Get tier assignments
    HIERARCHY_PROMOTE_TO_GOLDEN = {'coronary', 'arrhythmia', 'rheumatoid_arthritis', 'colitis', 'uti'}
    HIERARCHY_DEMOTE_TO_HIGH = {'thyroid'}
    HIERARCHY_DEMOTE_TO_MEDIUM = {'parkinsons', 'migraine', 'diabetes', 'skin_infection'}
    HIERARCHY_DEMOTE_TO_LOW = {'pneumonia', 'epilepsy', 'gout'}
    HIERARCHY_GOLDEN_CATEGORIES = {'metabolic', 'neurological'}
    
    def get_tier(category, group_name):
        if group_name in HIERARCHY_DEMOTE_TO_LOW:
            return 'LOW'
        if group_name in HIERARCHY_DEMOTE_TO_MEDIUM:
            return 'MEDIUM'
        if group_name in HIERARCHY_DEMOTE_TO_HIGH:
            return 'HIGH'
        if group_name in HIERARCHY_PROMOTE_TO_GOLDEN:
            return 'GOLDEN'
        if category in HIERARCHY_GOLDEN_CATEGORIES:
            return 'GOLDEN'
        return 'HIGH'
    
    print("\n" + "="*110)
    print(f"{'Rule':<50} {'Tier':<8} {'Internal GT':<12} {'Expanded GT':<12} {'Lift':<8} {'Status'}")
    print("="*110)
    
    for category, groups in sorted(DISEASE_HIERARCHY_GROUPS.items()):
        for group_name in sorted(groups.keys()):
            key = (category, group_name)
            tier = get_tier(category, group_name)
            n_internal = len(internal_per_group.get(key, set()))
            n_expanded = len(expanded_per_group.get(key, set()))
            rule_name = f"{category}_hierarchy_{group_name}"
            
            # Evaluability with expanded GT
            if n_expanded == 0:
                status = "DEAD"
            elif n_expanded == 1:
                status = "LIKELY INVISIBLE"
            elif n_expanded <= 3:
                status = "OFTEN INVISIBLE"
            elif n_expanded <= 5:
                status = "MARGINAL"
            else:
                status = "EVALUABLE"
            
            lift = n_expanded - n_internal
            print(f"{rule_name:<50} {tier:<8} {n_internal:<12} {n_expanded:<12} {'+' + str(lift) if lift > 0 else str(lift):<8} {status}")
    
    # Key finding: does expanded GT help evaluability?
    print("\n\nEVALUABILITY COMPARISON:")
    for threshold_name, threshold in [("EVALUABLE (6+)", 6), ("MARGINAL+ (4+)", 4)]:
        internal_eval = sum(1 for cat, groups in DISEASE_HIERARCHY_GROUPS.items()
                          for g in groups if len(internal_per_group.get((cat, g), set())) >= threshold)
        expanded_eval = sum(1 for cat, groups in DISEASE_HIERARCHY_GROUPS.items()
                          for g in groups if len(expanded_per_group.get((cat, g), set())) >= threshold)
        print(f"  {threshold_name}: Internal GT = {internal_eval}/32, Expanded GT = {expanded_eval}/32")
    
    # Now the real question: what are the h757 numbers for these rules?
    # Let me load h757 data if available
    print("\n\nCROSS-REFERENCE WITH h757 HOLDOUT DATA:")
    print("(From h757 sub-reason analysis)")
    
    # h757 numbers from the progress notes/commit
    h757_data = {
        # From h757 analysis - holdout precision per hierarchy rule
        'metabolic_hierarchy_diabetes': {'holdout': 21.1, 'n_per_seed': 5, 'full': 73.9},
        'metabolic_hierarchy_thyroid': {'holdout': 0.0, 'n_per_seed': 0, 'full': 0.0},  # actually it was not 0
        'infectious_hierarchy_skin_infection': {'holdout': 25.0, 'n_per_seed': 8, 'full': 25.8},
        'neurological_hierarchy_parkinsons': {'holdout': 0.0, 'n_per_seed': 0, 'full': 18.2},
        'neurological_hierarchy_migraine': {'holdout': 0.0, 'n_per_seed': 0, 'full': 62.5},
        'infectious_hierarchy_pneumonia': {'holdout': 16.7, 'n_per_seed': 6, 'full': 66.7},
        'neurological_hierarchy_epilepsy': {'holdout': 20.0, 'n_per_seed': 10, 'full': 71.6},
        'metabolic_hierarchy_gout': {'holdout': 0.0, 'n_per_seed': 4, 'full': 50.0},
        'infectious_hierarchy_uti': {'holdout': 80.0, 'n_per_seed': 10, 'full': 75.0},
    }
    
    print(f"\n{'Rule':<50} {'Tier':<8} {'GT Dis':<8} {'Full%':<8} {'Hold%':<8} {'n/seed':<8} {'Gap'}")
    print("-"*100)
    for rule_name, data in sorted(h757_data.items()):
        for cat, groups in DISEASE_HIERARCHY_GROUPS.items():
            for g in groups:
                if f"{cat}_hierarchy_{g}" == rule_name:
                    key = (cat, g)
                    tier = get_tier(cat, g)
                    n_dis = len(internal_per_group.get(key, set()))
                    gap = data['full'] - data['holdout']
                    print(f"{rule_name:<50} {tier:<8} {n_dis:<8} {data['full']:<8.1f} {data['holdout']:<8.1f} {data['n_per_seed']:<8} {'+' if gap > 0 else ''}{gap:.1f}")
    
    print("\nNOTE: Rules with n/seed=0 are holdout-invisible. High full-data but 0% holdout")
    print("indicates the rule ONLY fires when the GT disease is in training (self-referential).")


if __name__ == '__main__':
    main()
