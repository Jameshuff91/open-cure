#!/usr/bin/env python3
"""h760 Part 3: Determine which invisible rules are potentially harmful.

For invisible rules at HIGH/GOLDEN:
- If full-data precision is high AND the medical logic is sound → likely fine
- If full-data precision is suspect → consider demotion

Also check: can we MERGE small hierarchy groups into parent categories?
"""

import json

# Load h757 data
with open('data/analysis/h757_holdout_validation.json') as f:
    data = json.load(f)

# Categorize all hierarchy rules
invisible_high = []
invisible_golden = []
all_rules = {}

for rule_name, rd in data['rule_comparison'].items():
    if 'hierarchy' not in rule_name:
        continue
    all_rules[rule_name] = rd
    if rd['mean_n'] == 0:  # holdout invisible
        if rd['tier'] == 'GOLDEN':
            invisible_golden.append((rule_name, rd))
        elif rd['tier'] == 'HIGH':
            invisible_high.append((rule_name, rd))

print("="*100)
print("HOLDOUT-INVISIBLE GOLDEN RULES — Risk Assessment")
print("="*100)
for rule, rd in sorted(invisible_golden, key=lambda x: -x[1]['full_precision']):
    print(f"\n{rule}:")
    print(f"  Full-data: {rd['full_precision']:.1f}% ({rd['full_n']} predictions)")
    
    # Medical assessment
    group = rule.split('_hierarchy_')[1]
    if group == 'obesity':
        print(f"  Medical logic: STRONG — anti-obesity drugs are a well-defined class")
        print(f"  Risk: LOW — 100% full-data, 9 predictions. Likely genuine.")
        print(f"  Action: KEEP GOLDEN (2 GT diseases, but drugs are specific)")
    elif group == 'alzheimers':
        print(f"  Medical logic: MODERATE — Alzheimer's/dementia drugs limited, well-known")
        print(f"  Risk: MODERATE — 67% full-data suggests some false positives")
        print(f"  Action: Consider DEMOTION to HIGH (can't validate)")
    elif group == 'neuropathy':
        print(f"  Medical logic: MODERATE — gabapentinoids + TCAs, but neuropathy is broad")
        print(f"  Risk: MODERATE — 50% full-data, only 2 predictions")
        print(f"  Action: KEEP GOLDEN (tiny n, not impactful)")

print("\n" + "="*100)
print("HOLDOUT-INVISIBLE HIGH RULES — Risk Assessment")
print("="*100)
for rule, rd in sorted(invisible_high, key=lambda x: -x[1]['full_precision']):
    print(f"\n{rule}:")
    print(f"  Full-data: {rd['full_precision']:.1f}% ({rd['full_n']} predictions)")
    
    group = rule.split('_hierarchy_')[1]
    category = rule.split('_hierarchy_')[0]
    
    if group == 'multiple_sclerosis':
        print(f"  Medical logic: STRONG — MS drugs are well-defined (interferons, natalizumab, etc)")
        print(f"  Risk: LOW — 100% full-data, 17 predictions. Strong drug class specificity.")
        print(f"  But: Only 1 GT disease. Full-data precision is ALL self-referential.")
        print(f"  Action: KEEP HIGH — medical logic supports it, but CANNOT validate")
    elif group == 'asthma':
        print(f"  Medical logic: STRONG — asthma drugs very specific (ICS, LABA, LTRA)")
        print(f"  Risk: LOW — 94% full-data, 18 predictions. Strong class specificity.")
        print(f"  But: Only 2 GT diseases (bronchial asthma, bronchospasm)")
        print(f"  Action: KEEP HIGH — medical logic supports it")
    elif group == 'scleroderma':
        print(f"  Medical logic: MODERATE — scleroderma treatments less standardized")
        print(f"  Risk: HIGH — 100% full-data but only 3 predictions. Too few to judge.")
        print(f"  Action: KEEP HIGH — tiny impact (3 predictions)")
    elif group == 'respiratory_infection':
        print(f"  Medical logic: MODERATE — broad category, antibiotics overlap")
        print(f"  Risk: MODERATE — 100% full-data but only 6 predictions.")
        print(f"  Action: KEEP HIGH — small impact")
    elif group == 'sepsis':
        print(f"  Medical logic: MODERATE — sepsis treatment = broad-spectrum antibiotics + supportive")
        print(f"  Risk: MODERATE — 77% full-data, 13 predictions. Can't validate.")
        print(f"  Action: KEEP HIGH — reasonable full-data precision")
    elif group == 'atherosclerosis':
        print(f"  Medical logic: MODERATE — statins work but 1 GT disease is suspicious")
        print(f"  Risk: MODERATE — 75% but only 4 predictions")
        print(f"  Action: KEEP HIGH — tiny impact")

# Now check overfitted rules that HAVE holdout data
print("\n" + "="*100)
print("OVERFITTED RULES WITH HOLDOUT — Current Tier Appropriateness")
print("="*100)
overfitted = [(k, v) for k, v in all_rules.items() if v['mean_n'] > 0 and v['status'] == 'OVERFITTED?']
for rule, rd in sorted(overfitted, key=lambda x: x[1]['holdout_mean']):
    tier = rd['tier']
    holdout = rd['holdout_mean']
    
    # Check if holdout supports current tier
    tier_thresholds = {'GOLDEN': 70, 'HIGH': 50, 'MEDIUM': 25, 'LOW': 15}
    threshold = tier_thresholds.get(tier, 0)
    appropriate = holdout >= threshold * 0.8  # Allow 20% margin
    
    status = "OK" if appropriate else f"WRONG TIER (holdout {holdout:.1f}% < {threshold}% threshold)"
    print(f"  {rule:<50} {tier:<8} holdout={holdout:.1f}%±{rd['holdout_std']:.1f} n={rd['mean_n']:.0f}/seed → {status}")

# Summary of actionable findings
print("\n" + "="*100)
print("ACTIONABLE FINDINGS")
print("="*100)

print("""
1. HOLDOUT-INVISIBLE GOLDEN (3 rules, ~17 predictions):
   - metabolic_hierarchy_obesity: 9 preds, KEEP (medical logic strong)
   - neurological_hierarchy_alzheimers: 6 preds, CONSIDER demoting to HIGH
   - neurological_hierarchy_neuropathy: 2 preds, KEEP (too small to matter)

2. HOLDOUT-INVISIBLE HIGH (6 rules, ~61 predictions):
   - All medically reasonable, small prediction counts
   - No actionable demotions needed

3. OVERFITTED RULES requiring attention:
""")

for rule, rd in sorted(overfitted, key=lambda x: x[1]['holdout_mean']):
    tier = rd['tier']
    holdout = rd['holdout_mean']
    tier_thresholds = {'GOLDEN': 70, 'HIGH': 50, 'MEDIUM': 25, 'LOW': 15}
    threshold = tier_thresholds.get(tier, 0)
    if holdout < threshold * 0.8:
        print(f"   - {rule}: {tier} tier but {holdout:.1f}% holdout (below {threshold}% threshold)")
        # What tier should it be?
        if holdout >= 50:
            suggested = "HIGH"
        elif holdout >= 25:
            suggested = "MEDIUM"
        elif holdout >= 15:
            suggested = "LOW (borderline MEDIUM)"
        else:
            suggested = "LOW"
        print(f"     → Suggested: {suggested}")

print("""
4. DEAD RULE:
   - autoimmune_hierarchy_spondylitis: 0 GT diseases, generates 0 predictions
     → No harm, but could be removed for code hygiene

5. OVERALL ASSESSMENT:
   - 10/32 rules are holdout-invisible (KNOWN limitation, documented in h653)
   - Most invisible rules are medically reasonable with decent full-data precision
   - The h757 demotions already caught the worst offenders
   - Remaining invisible rules affect ~78 predictions total (very small impact)
   - Key finding: THESE RULES ARE EVALUATION BLIND SPOTS but not harmful
""")

if __name__ == '__main__':
    pass
