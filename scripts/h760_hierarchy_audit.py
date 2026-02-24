#!/usr/bin/env python3
"""h760: Audit hierarchy rules for holdout visibility and evaluability.

Strategy: Use the predictor to generate predictions for all diseases,
then count predictions by hierarchy sub-reason and cross-reference with
GT to understand evaluability.
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
    
    # Step 1: Find GT diseases per hierarchy group
    # GT diseases are in predictor.disease_names (disease_id -> EC name)
    print("\nStep 1: Matching GT diseases to hierarchy groups...")
    
    gt_per_group = defaultdict(set)  # (category, group) -> set of disease_ids
    drugs_per_group = defaultdict(set)  # (category, group) -> set of drug_ids
    
    for disease_id, gt_drugs in predictor.ground_truth.items():
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        d_lower = disease_name.lower()
        
        for category, groups in DISEASE_HIERARCHY_GROUPS.items():
            for group_name, variants in groups.items():
                excl = HIERARCHY_EXCLUSIONS.get((category, group_name), [])
                if any(e in d_lower for e in excl):
                    continue
                if any(v in d_lower or d_lower in v for v in variants):
                    gt_per_group[(category, group_name)].add(disease_id)
                    for drug_id in gt_drugs:
                        drugs_per_group[(category, group_name)].add(drug_id)
                    break
    
    # Step 2: Find DRKG diseases per hierarchy group
    print("Step 2: Matching DRKG diseases to hierarchy groups...")
    
    drkg_per_group = defaultdict(set)
    all_drkg_diseases = set()
    for disease_id, disease_name in predictor.disease_names.items():
        all_drkg_diseases.add(disease_id)
    
    # Actually we need DRKG diseases from the embedding/prediction space
    # The disease_names on predictor are EC GT names. DRKG diseases are different.
    # Let me use drug_disease_groups which is already built from GT
    
    # Step 3: Count predictions by sub-reason (use cached h757 data if available)
    # Instead of running full predictions, let me run the holdout evaluator
    print("Step 3: Running holdout evaluation for hierarchy rules...")
    
    # Use the h393 evaluator approach
    
    # Actually, let me just run per-subreason analysis
    # First: count drug_disease_groups
    print("\nDrug coverage per hierarchy group:")
    print(f"{'Category':<15} {'Group':<25} {'Tier':<8} {'GT Diseases':<12} {'GT Drugs (unique)':<18} {'In drug_disease_groups'}")
    print("="*100)
    
    for category, groups in sorted(DISEASE_HIERARCHY_GROUPS.items()):
        for group_name in sorted(groups.keys()):
            key = (category, group_name)
            tier = get_tier(category, group_name)
            n_diseases = len(gt_per_group.get(key, set()))
            n_drugs = len(drugs_per_group.get(key, set()))
            
            # How many drugs have this group in drug_disease_groups?
            drugs_with_group = sum(
                1 for drug_id, groups_set in predictor.drug_disease_groups.items()
                if (category, group_name) in groups_set
            )
            
            print(f"{category:<15} {group_name:<25} {tier:<8} {n_diseases:<12} {n_drugs:<18} {drugs_with_group}")
    
    # Step 4: For evaluability, what matters is:
    # How many HOLDOUT diseases can trigger this rule?
    # The holdout splits diseases randomly. If a rule's GT diseases are few,
    # then often ALL are in training, making the rule untestable.
    
    print("\n\nHOLDOUT EVALUABILITY:")
    print(f"{'Rule':<50} {'Tier':<8} {'GT Dis':<8} {'P(invisible)':<15} {'Status'}")
    print("="*100)
    
    results = []
    for category, groups in sorted(DISEASE_HIERARCHY_GROUPS.items()):
        for group_name in sorted(groups.keys()):
            key = (category, group_name)
            tier = get_tier(category, group_name)
            n_diseases = len(gt_per_group.get(key, set()))
            n_drugs = len(drugs_per_group.get(key, set()))
            
            # Probability that all GT diseases end up in training (50% split)
            if n_diseases == 0:
                p_invis = 1.0
                status = "DEAD"
            else:
                # P(all in train) = 0.5^n
                p_invis = 0.5 ** n_diseases
                if p_invis >= 0.5:
                    status = "LIKELY INVISIBLE"
                elif p_invis >= 0.125:
                    status = "OFTEN INVISIBLE"
                elif p_invis >= 0.03:
                    status = "MARGINAL"
                else:
                    status = "EVALUABLE"
            
            rule_name = f"{category}_hierarchy_{group_name}"
            print(f"{rule_name:<50} {tier:<8} {n_diseases:<8} {p_invis:<15.4f} {status}")
            
            results.append({
                'rule': rule_name,
                'category': category,
                'group': group_name,
                'tier': tier,
                'gt_diseases': n_diseases,
                'gt_drugs': n_drugs,
                'p_invisible': p_invis,
                'status': status,
                'gt_disease_names': [predictor.disease_names.get(d, d) for d in gt_per_group.get(key, set())]
            })
    
    # Step 5: Detailed analysis of problematic rules
    print("\n\nDETAILED ANALYSIS OF PROBLEMATIC RULES:")
    print("="*100)
    
    for r in sorted(results, key=lambda x: x['gt_diseases']):
        if r['gt_diseases'] <= 3:
            print(f"\n{r['rule']} ({r['tier']}, {r['gt_diseases']} GT diseases):")
            if r['gt_disease_names']:
                for d in sorted(r['gt_disease_names']):
                    print(f"  - {d}")
            else:
                print("  (no GT diseases match this hierarchy group)")
    
    # Step 6: Which hierarchy rules are assigned to GOLDEN/HIGH but can't be validated?
    print("\n\nRISK ASSESSMENT: Unvalidatable high-tier rules:")
    print("="*80)
    
    high_risk = [r for r in results if r['tier'] in ('GOLDEN', 'HIGH') and r['gt_diseases'] <= 3]
    for r in high_risk:
        print(f"  {r['rule']}: {r['tier']} tier with only {r['gt_diseases']} GT diseases")
    
    low_risk = [r for r in results if r['tier'] in ('LOW', 'MEDIUM', 'FILTER') and r['gt_diseases'] <= 3]
    if low_risk:
        print(f"\n  (Also {len(low_risk)} low-tier rules with <=3 GT diseases - less concerning)")
    
    # Step 7: What fraction of all hierarchy predictions are from evaluable rules?
    # Need to count predictions per rule. Let me use the drug_disease_groups more carefully.
    
    # Count how many drug-disease pairs would trigger each hierarchy rule
    # This is an approximation - we count drugs in drug_disease_groups for each group
    pred_estimate = {}
    for category, groups in DISEASE_HIERARCHY_GROUPS.items():
        for group_name in groups.keys():
            key = (category, group_name)
            # Count drugs that have GT in this group
            drugs_in_group = sum(
                1 for drug_id, groups_set in predictor.drug_disease_groups.items()
                if (category, group_name) in groups_set
            )
            pred_estimate[f"{category}_hierarchy_{group_name}"] = drugs_in_group
    
    # Summary
    total_evaluable_drugs = sum(pred_estimate[r['rule']] for r in results if r['status'] == 'EVALUABLE')
    total_problem_drugs = sum(pred_estimate[r['rule']] for r in results if r['status'] != 'EVALUABLE')
    total_all = sum(pred_estimate.values())
    
    print(f"\n\nSUMMARY:")
    print(f"  Total hierarchy groups: {len(results)}")
    print(f"  DEAD (0 GT diseases): {sum(1 for r in results if r['status'] == 'DEAD')}")
    print(f"  LIKELY INVISIBLE (1 GT): {sum(1 for r in results if r['status'] == 'LIKELY INVISIBLE')}")
    print(f"  OFTEN INVISIBLE (2-3 GT): {sum(1 for r in results if r['status'] == 'OFTEN INVISIBLE')}")
    print(f"  MARGINAL (4 GT): {sum(1 for r in results if r['status'] == 'MARGINAL')}")
    print(f"  EVALUABLE (5+ GT): {sum(1 for r in results if r['status'] == 'EVALUABLE')}")
    print(f"\n  Estimated drug coverage: {total_evaluable_drugs}/{total_all} ({100*total_evaluable_drugs/max(total_all,1):.1f}%) from evaluable rules")
    
    # Save
    with open('data/analysis/h760_hierarchy_audit.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("\nResults saved to data/analysis/h760_hierarchy_audit.json")


if __name__ == '__main__':
    main()
