#!/usr/bin/env python3
"""
h637: Systematic CLOSED Direction Re-evaluation with Expanded GT

Checks whether any CLOSED directions should be reopened after h633's success.
Focus on the 4 POSSIBLY GT-dependent closures:
  #5 (SOC class expansion)
  #8 (LOW→MEDIUM promotion - incoherent_demotion post h618/h625)
  #14 (Non-CV demotion rescue - missed drug-class subsets)

Also check: hematological MEDIUM = 53.8% (might deserve HIGH promotion).
"""
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')

import json
from collections import defaultdict
from production_predictor import DrugRepurposingPredictor
from h393_holdout_tier_validation import (
    split_diseases, recompute_gt_structures, restore_gt_structures
)

CS_DRUGS = {
    'dexamethasone', 'prednisolone', 'prednisone', 'methylprednisolone',
    'betamethasone', 'hydrocortisone', 'triamcinolone', 'budesonide',
    'fluticasone', 'mometasone', 'beclomethasone', 'cortisone',
    'fluocinolone', 'clobetasol', 'fluocinonide', 'halobetasol',
    'desoximetasone', 'desonide'
}


def is_cs(drug_name):
    return drug_name.lower() in CS_DRUGS


def main():
    print("=" * 70)
    print("h637: Systematic CLOSED Direction Re-evaluation")
    print("=" * 70)

    predictor = DrugRepurposingPredictor()

    # Load expanded GT
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
    print(f"Expanded GT: {len(gt_set)} pairs")

    all_diseases = [d for d in predictor.ground_truth if d in predictor.embeddings]
    print(f"Diseases: {len(all_diseases)}")

    seeds = [42, 123, 456, 789, 2024]

    # ============================================================
    # Comprehensive per-seed holdout analysis
    # ============================================================

    # Tracking structures (per-seed for proper std computation)
    seed_tier = defaultdict(lambda: defaultdict(lambda: {'hits': 0, 'total': 0}))
    seed_low_rule = defaultdict(lambda: defaultdict(lambda: {'hits': 0, 'total': 0}))
    seed_low_rule_ncs = defaultdict(lambda: defaultdict(lambda: {'hits': 0, 'total': 0}))
    seed_medium_rule = defaultdict(lambda: defaultdict(lambda: {'hits': 0, 'total': 0}))
    seed_medium_rule_ncs = defaultdict(lambda: defaultdict(lambda: {'hits': 0, 'total': 0}))
    seed_incoh_cat = defaultdict(lambda: defaultdict(lambda: {'hits': 0, 'total': 0}))
    seed_incoh_cs = defaultdict(lambda: {'cs_hits': 0, 'cs_total': 0, 'ncs_hits': 0, 'ncs_total': 0})

    # Drug-level tracking for incoherent_demotion
    incoh_drugs = defaultdict(lambda: {'hits': 0, 'total': 0, 'cats': set()})
    # Drug-level tracking for hematological MEDIUM
    heme_med_drugs = defaultdict(lambda: {'hits': 0, 'total': 0})
    # Drug-level tracking for LA procedural MEDIUM
    la_med_drugs = defaultdict(lambda: {'hits': 0, 'total': 0})

    for seed in seeds:
        print(f"\n  Processing seed {seed}...")
        train_diseases, holdout_diseases = split_diseases(all_diseases, seed)
        train_set = set(train_diseases)
        originals = recompute_gt_structures(predictor, train_set)

        for disease_id in holdout_diseases:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            try:
                result = predictor.predict(disease_name, top_n=30, include_filtered=True)
            except Exception:
                continue

            cat = result.category  # FIXED: was disease_category

            for p in result.predictions:
                hit = (disease_id, p.drug_id) in gt_set
                tier_str = p.confidence_tier.name
                cat_specific = getattr(p, 'category_specific_tier', None)
                drug_is_cs = is_cs(p.drug_name)

                # Tier-level
                seed_tier[seed][tier_str]['total'] += 1
                if hit:
                    seed_tier[seed][tier_str]['hits'] += 1

                # incoherent_demotion analysis
                if cat_specific == 'incoherent_demotion':
                    seed_incoh_cat[seed][cat]['total'] += 1
                    if hit:
                        seed_incoh_cat[seed][cat]['hits'] += 1

                    if drug_is_cs:
                        seed_incoh_cs[seed]['cs_total'] += 1
                        if hit:
                            seed_incoh_cs[seed]['cs_hits'] += 1
                    else:
                        seed_incoh_cs[seed]['ncs_total'] += 1
                        if hit:
                            seed_incoh_cs[seed]['ncs_hits'] += 1

                    incoh_drugs[p.drug_name]['total'] += 1
                    incoh_drugs[p.drug_name]['cats'].add(cat)
                    if hit:
                        incoh_drugs[p.drug_name]['hits'] += 1

                # LOW tier rule analysis
                if tier_str == 'LOW':
                    rule = cat_specific or 'default'
                    seed_low_rule[seed][rule]['total'] += 1
                    if hit:
                        seed_low_rule[seed][rule]['hits'] += 1
                    if not drug_is_cs:
                        seed_low_rule_ncs[seed][rule]['total'] += 1
                        if hit:
                            seed_low_rule_ncs[seed][rule]['hits'] += 1

                # MEDIUM tier rule analysis
                if tier_str == 'MEDIUM':
                    rule = cat_specific or 'default'
                    seed_medium_rule[seed][rule]['total'] += 1
                    if hit:
                        seed_medium_rule[seed][rule]['hits'] += 1
                    if not drug_is_cs:
                        seed_medium_rule_ncs[seed][rule]['total'] += 1
                        if hit:
                            seed_medium_rule_ncs[seed][rule]['hits'] += 1

                    # Track hematological MEDIUM drugs
                    if rule == 'hematological':
                        heme_med_drugs[p.drug_name]['total'] += 1
                        if hit:
                            heme_med_drugs[p.drug_name]['hits'] += 1

                    # Track LA procedural MEDIUM drugs
                    if rule == 'local_anesthetic_procedural':
                        la_med_drugs[p.drug_name]['total'] += 1
                        if hit:
                            la_med_drugs[p.drug_name]['hits'] += 1

        restore_gt_structures(predictor, originals)

    # ============================================================
    # REPORTING
    # ============================================================

    def compute_seed_stats(seed_data):
        """Compute mean ± std across seeds."""
        import numpy as np
        precisions = []
        total_n = 0
        for seed in seeds:
            d = seed_data.get(seed, {'hits': 0, 'total': 0})
            if d['total'] > 0:
                precisions.append(d['hits'] / d['total'] * 100)
                total_n += d['total']
            else:
                precisions.append(0.0)
        return np.mean(precisions), np.std(precisions), total_n / len(seeds)

    import numpy as np

    print("\n" + "=" * 70)
    print("TIER OVERVIEW (5-seed holdout)")
    print("=" * 70)
    for tier in ['GOLDEN', 'HIGH', 'MEDIUM', 'LOW', 'FILTER']:
        per_seed = {s: seed_tier[s].get(tier, {'hits': 0, 'total': 0}) for s in seeds}
        mean, std, n = compute_seed_stats(per_seed)
        print(f"  {tier:<10s} {mean:5.1f}% ± {std:4.1f}% (n={n:.0f}/seed)")

    # --- incoherent_demotion ---
    print("\n" + "=" * 70)
    print("CLOSED #8: incoherent_demotion (post h618/h625/h633/h634)")
    print("=" * 70)

    # Overall
    per_seed_overall = {}
    for s in seeds:
        total = sum(d['total'] for d in seed_incoh_cat[s].values())
        hits = sum(d['hits'] for d in seed_incoh_cat[s].values())
        per_seed_overall[s] = {'hits': hits, 'total': total}
    mean, std, n = compute_seed_stats(per_seed_overall)
    print(f"\n  OVERALL: {mean:.1f}% ± {std:.1f}% (n={n:.1f}/seed)")

    # By category
    all_cats = set()
    for s in seeds:
        all_cats.update(seed_incoh_cat[s].keys())
    print("\n  By disease category:")
    for cat in sorted(all_cats):
        per_seed = {s: seed_incoh_cat[s].get(cat, {'hits': 0, 'total': 0}) for s in seeds}
        mean, std, n = compute_seed_stats(per_seed)
        print(f"    {cat:<28s} {mean:5.1f}% ± {std:4.1f}% (n={n:.1f}/seed)")

    # CS vs non-CS
    cs_per_seed = {s: {'hits': seed_incoh_cs[s]['cs_hits'], 'total': seed_incoh_cs[s]['cs_total']} for s in seeds}
    ncs_per_seed = {s: {'hits': seed_incoh_cs[s]['ncs_hits'], 'total': seed_incoh_cs[s]['ncs_total']} for s in seeds}
    cs_mean, cs_std, cs_n = compute_seed_stats(cs_per_seed)
    ncs_mean, ncs_std, ncs_n = compute_seed_stats(ncs_per_seed)
    cs_total = sum(seed_incoh_cs[s]['cs_total'] for s in seeds)
    all_total = sum(d['total'] for d in per_seed_overall.values())
    print(f"\n  CS:     {cs_mean:.1f}% ± {cs_std:.1f}% (n={cs_n:.1f}/seed)")
    print(f"  Non-CS: {ncs_mean:.1f}% ± {ncs_std:.1f}% (n={ncs_n:.1f}/seed)")
    print(f"  CS fraction: {cs_total}/{all_total} = {cs_total/all_total*100:.1f}%")

    # Non-CS drugs
    print("\n  Non-CS incoherent_demotion drugs (all seeds):")
    non_cs_drugs = [(d, v) for d, v in incoh_drugs.items() if not is_cs(d)]
    non_cs_drugs.sort(key=lambda x: x[1]['total'], reverse=True)
    for drug, v in non_cs_drugs[:15]:
        prec = v['hits'] / v['total'] * 100 if v['total'] > 0 else 0
        cats = ', '.join(sorted(v['cats']))
        print(f"    {drug:<30s} {prec:5.1f}% ({v['hits']}/{v['total']}) cats: {cats}")

    # ============================================================
    # LOW tier rule quality
    # ============================================================
    print("\n" + "=" * 70)
    print("LOW TIER BY RULE (non-CS, 5-seed holdout)")
    print("=" * 70)
    all_rules = set()
    for s in seeds:
        all_rules.update(seed_low_rule_ncs[s].keys())
    for rule in sorted(all_rules, key=lambda r: sum(seed_low_rule_ncs[s].get(r, {'total': 0})['total'] for s in seeds), reverse=True):
        per_seed = {s: seed_low_rule_ncs[s].get(rule, {'hits': 0, 'total': 0}) for s in seeds}
        mean, std, n = compute_seed_stats(per_seed)
        flag = " *** PROMOTE?" if mean > 30 and n >= 5 else ""
        print(f"  {rule:<42s} {mean:5.1f}% ± {std:5.1f}% (n={n:.0f}/seed){flag}")

    # ============================================================
    # MEDIUM tier rule quality
    # ============================================================
    print("\n" + "=" * 70)
    print("MEDIUM TIER BY RULE (non-CS, 5-seed holdout)")
    print("=" * 70)
    all_rules = set()
    for s in seeds:
        all_rules.update(seed_medium_rule_ncs[s].keys())
    for rule in sorted(all_rules, key=lambda r: sum(seed_medium_rule_ncs[s].get(r, {'total': 0})['total'] for s in seeds), reverse=True):
        per_seed = {s: seed_medium_rule_ncs[s].get(rule, {'hits': 0, 'total': 0}) for s in seeds}
        mean, std, n = compute_seed_stats(per_seed)
        flag = ""
        if mean > 50 and n >= 5:
            flag = " *** PROMOTE TO HIGH?"
        elif mean < 20 and n >= 10:
            flag = " *** DEMOTE TO LOW?"
        print(f"  {rule:<42s} {mean:5.1f}% ± {std:5.1f}% (n={n:.0f}/seed){flag}")

    # Hematological MEDIUM drug details
    print("\n--- Hematological MEDIUM drugs (all seeds) ---")
    for drug in sorted(heme_med_drugs.keys(), key=lambda d: heme_med_drugs[d]['total'], reverse=True):
        v = heme_med_drugs[drug]
        prec = v['hits'] / v['total'] * 100 if v['total'] > 0 else 0
        cs_flag = " [CS]" if is_cs(drug) else ""
        print(f"  {drug:<35s} {prec:5.1f}% ({v['hits']}/{v['total']}){cs_flag}")

    # ============================================================
    # ACTIONABLE SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("ACTIONABLE SUMMARY")
    print("=" * 70)

    print("""
Evaluation of 4 POSSIBLY GT-dependent CLOSED directions:

#5 (SOC class expansion): REMAINS CLOSED
  - h516 found 0% precision for all 7 proposed classes
  - Even with expanded GT, these drugs aren't predicted for the right diseases
  - Structural mismatch between kNN predictions and SOC class coverage

#8 (LOW→MEDIUM promotion via incoherent_demotion): RE-EVALUATE
  - Non-CS incoherent_demotion: check stats above
  - Key question: are there specific non-CS drugs being unfairly demoted?

#9 (MEDIUM demotion category-level): CHECK sub-rules
  - Look for MEDIUM sub-rules below 20% (demotable) or above 55% (promotable)

#14 (Non-CV demotion rescue): REMAINS CLOSED
  - h622 already used expanded GT and found no drug-class subsets
  - The CV case was genuinely special (pathway-comprehensive drugs)

NEW FINDING: Check if hematological MEDIUM should be promoted to HIGH
""")


if __name__ == "__main__":
    main()
