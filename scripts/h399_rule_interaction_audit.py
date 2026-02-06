#!/usr/bin/env python3
"""
h399: Rule Interaction Audit - Test Priority/Overlap Edge Cases

Systematically evaluates every drug-disease prediction through ALL rules
to identify cases where rule shadowing leads to suboptimal tier assignment.
"""

import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.production_predictor import (
    DrugRepurposingPredictor, ConfidenceTier,
    CORTICOSTEROID_DRUGS, STATIN_NAMES, DISEASE_HIERARCHY_GROUPS,
    SELECTIVE_BOOST_CATEGORIES, SELECTIVE_BOOST_ALPHA,
)
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def check_all_rules(predictor, rank, train_frequency, mechanism_support,
                    has_targets, disease_tier, category, drug_name,
                    disease_name, drug_id):
    """
    Check which rules a drug-disease pair would match if each rule were
    evaluated independently (ignoring early returns).
    Returns list of (rule_name, tier) for ALL matching rules.
    """
    matching_rules = []
    drug_lower = drug_name.lower()
    disease_lower = disease_name.lower()

    # === RULE 1: Cancer same-type/no-GT ===
    if category == 'cancer' and drug_id:
        has_cancer_gt, same_type_match, _ = predictor._check_cancer_type_match(drug_id, disease_name)
        if same_type_match:
            matching_rules.append(('cancer_same_type', ConfidenceTier.MEDIUM))
        elif not has_cancer_gt:
            matching_rules.append(('cancer_no_gt', ConfidenceTier.FILTER))

    # === RULE 2: FILTER basic checks ===
    if rank > 20:
        matching_rules.append(('rank_too_low', ConfidenceTier.FILTER))
    if not has_targets:
        matching_rules.append(('no_targets', ConfidenceTier.FILTER))
    if train_frequency <= 2 and not mechanism_support:
        matching_rules.append(('low_freq_no_mech', ConfidenceTier.FILTER))

    # === RULE 3: Corticosteroids for metabolic ===
    if category == 'metabolic':
        if any(steroid in drug_lower for steroid in CORTICOSTEROID_DRUGS):
            matching_rules.append(('corticosteroid_metabolic', ConfidenceTier.FILTER))

    # === RULE 4: Cross-domain isolated ===
    if drug_id and predictor._is_cross_domain_isolated(drug_id, category):
        matching_rules.append(('cross_domain_isolated', ConfidenceTier.FILTER))

    # === RULE 5: Base->complication ===
    if drug_id and predictor._is_base_to_complication(drug_id, disease_name):
        matching_rules.append(('base_to_complication', ConfidenceTier.FILTER))

    # === RULE 6: Comp->base ===
    if drug_id:
        is_comp_to_base, transferability, is_statin_cv = predictor._is_comp_to_base(drug_id, disease_name)
        if is_comp_to_base:
            if is_statin_cv:
                matching_rules.append(('statin_cv_event', ConfidenceTier.GOLDEN))
            else:
                drug_name_for_check = predictor._get_drug_name(drug_id).lower()
                is_statin = any(s in drug_name_for_check for s in STATIN_NAMES)
                is_cv_pred = 'athero' in disease_lower
                if is_cv_pred and not is_statin:
                    pass  # No boost
                elif transferability >= 50:
                    matching_rules.append((f'comp_to_base_high_{transferability:.0f}', ConfidenceTier.HIGH))
                elif transferability >= 20:
                    matching_rules.append((f'comp_to_base_med_{transferability:.0f}', ConfidenceTier.MEDIUM))

    # === RULE 7: Mechanism-specific disease ===
    if predictor._is_mechanism_specific_disease(disease_name):
        matching_rules.append(('mechanism_specific', ConfidenceTier.LOW))

    # === RULE 8: Cancer-only drug non-cancer ===
    if predictor._is_cancer_only_drug_non_cancer(drug_name, disease_name):
        matching_rules.append(('cancer_only_non_cancer', ConfidenceTier.FILTER))

    # === RULE 9: Complication non-validated class ===
    if predictor._is_complication_non_validated_class(drug_name, disease_name):
        matching_rules.append(('complication_non_validated', ConfidenceTier.FILTER))

    # === RULE 10: CV pathway-comprehensive ===
    if predictor._is_cv_complication(disease_name):
        if predictor._is_cv_pathway_comprehensive(drug_name):
            matching_rules.append(('cv_pathway_comprehensive', ConfidenceTier.HIGH))

    # === RULE 11: Disease hierarchy match ===
    HIERARCHY_GOLDEN_CATEGORIES = {'metabolic', 'neurological'}
    HIERARCHY_DEMOTE_TO_HIGH = {'thyroid'}
    HIERARCHY_DEMOTE_TO_MEDIUM = {'parkinsons', 'migraine'}

    if category in DISEASE_HIERARCHY_GROUPS and drug_id:
        has_category_gt, same_group_match, matching_group = predictor._check_disease_hierarchy_match(
            drug_id, disease_name, category
        )
        if same_group_match:
            if matching_group in HIERARCHY_DEMOTE_TO_MEDIUM:
                matching_rules.append((f'hierarchy_{matching_group}', ConfidenceTier.MEDIUM))
            elif matching_group in HIERARCHY_DEMOTE_TO_HIGH:
                matching_rules.append((f'hierarchy_{matching_group}', ConfidenceTier.HIGH))
            elif category in HIERARCHY_GOLDEN_CATEGORIES:
                matching_rules.append((f'hierarchy_{matching_group}', ConfidenceTier.GOLDEN))
            else:
                matching_rules.append((f'hierarchy_{matching_group}', ConfidenceTier.HIGH))

    # === RULE 12: Category-specific rescue ===
    if disease_tier > 1:
        rescued_tier = predictor._apply_category_rescue(
            rank, train_frequency, mechanism_support, category, drug_name, disease_name, drug_id
        )
        if rescued_tier is not None:
            matching_rules.append((f'rescue_{category}', rescued_tier))

    # === RULE 13: Zero-precision ATC mismatch ===
    if drug_name and category:
        _, is_zero_prec_mismatch, _ = predictor._check_atc_mismatch_rules(drug_name, category)
        if is_zero_prec_mismatch:
            matching_rules.append(('zero_precision_mismatch', ConfidenceTier.FILTER))

    # === Standard rules (14-18) ===
    is_coherent = drug_name and category and predictor._is_atc_coherent(drug_name, category)

    # GOLDEN
    if disease_tier == 1 and train_frequency >= 10 and mechanism_support:
        if not is_coherent:
            matching_rules.append(('std_golden_incoherent', ConfidenceTier.HIGH))
        else:
            matching_rules.append(('std_golden', ConfidenceTier.GOLDEN))

    # HIGH
    if train_frequency >= 15 and mechanism_support:
        if not is_coherent:
            matching_rules.append(('std_high_f15_incoherent', ConfidenceTier.MEDIUM))
        else:
            matching_rules.append(('std_high_f15', ConfidenceTier.HIGH))
    elif rank <= 5 and train_frequency >= 10 and mechanism_support:
        if not is_coherent:
            matching_rules.append(('std_high_r5_incoherent', ConfidenceTier.MEDIUM))
        else:
            matching_rules.append(('std_high_r5', ConfidenceTier.HIGH))

    # MEDIUM
    if train_frequency >= 5 and mechanism_support:
        matching_rules.append(('std_medium_f5_mech', ConfidenceTier.MEDIUM))
    elif train_frequency >= 10:
        matching_rules.append(('std_medium_f10', ConfidenceTier.MEDIUM))

    # Highly repurposable
    if predictor._is_highly_repurposable_disease(disease_name):
        if mechanism_support or train_frequency >= 5:
            matching_rules.append(('highly_repurposable', ConfidenceTier.MEDIUM))

    # ATC coherent boost
    ATC_COHERENT_EXCLUDED = {'metabolic', 'neurological'}
    if drug_name and category and category not in ATC_COHERENT_EXCLUDED:
        if predictor._is_atc_coherent(drug_name, category):
            if rank <= 10 and (mechanism_support or train_frequency >= 3):
                matching_rules.append((f'atc_coherent_{category}', ConfidenceTier.MEDIUM))

    # Default
    matching_rules.append(('default_low', ConfidenceTier.LOW))

    return matching_rules


TIER_ORDER = {
    ConfidenceTier.GOLDEN: 4,
    ConfidenceTier.HIGH: 3,
    ConfidenceTier.MEDIUM: 2,
    ConfidenceTier.LOW: 1,
    ConfidenceTier.FILTER: 0,
}


def main():
    print("=" * 80)
    print("h399: Rule Interaction Audit")
    print("=" * 80)

    predictor = DrugRepurposingPredictor()
    gt_data = predictor.ground_truth

    # Get GT diseases with embeddings
    gt_diseases = [d for d in gt_data if len(gt_data[d]) > 0 and d in predictor.embeddings]
    print(f"GT diseases with embeddings: {len(gt_diseases)}")

    # Stats
    total_preds = 0
    multi_match = 0

    # Per-rule firing stats (actual assigned rule) -> {total, gt_hits}
    firing_stats = defaultdict(lambda: [0, 0])

    # Interaction tracking: (firing_rule, shadowed_rule) -> {count, gt_hits, firing_tier, shadow_tier}
    interactions = defaultdict(lambda: {'count': 0, 'gt_when_fire': 0, 'gt_when_shadow': 0,
                                        'fire_tier': None, 'shadow_tier': None})

    # Cases where shadowed rule would give HIGHER or LOWER tier
    shadow_upgrade_gt = 0  # GT hit where shadow tier > fire tier
    shadow_downgrade_gt = 0

    # Per-rule precision when matched (regardless of firing)
    rule_match_stats = defaultdict(lambda: [0, 0])  # [total_matches, gt_hits]

    print("\nRunning predictions for all diseases...")

    for idx, disease_id in enumerate(gt_diseases):
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        gt_drugs = gt_data[disease_id]

        try:
            result = predictor.predict(disease_name, top_n=30, include_filtered=True)
        except Exception:
            continue

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx+1}/{len(gt_diseases)} diseases...")

        for pred in result.predictions:
            total_preds += 1
            is_gt = pred.drug_id in gt_drugs

            actual_rule = pred.category_specific_tier or 'standard'
            actual_tier = pred.confidence_tier

            firing_stats[actual_rule][0] += 1
            if is_gt:
                firing_stats[actual_rule][1] += 1

            # Check ALL rules
            all_rules = check_all_rules(
                predictor, pred.rank, pred.train_frequency,
                pred.mechanism_support, pred.has_targets,
                result.disease_tier, result.category,
                pred.drug_name, disease_name, pred.drug_id
            )

            # Record per-rule match stats
            for rule_name, rule_tier in all_rules:
                rule_match_stats[rule_name][0] += 1
                if is_gt:
                    rule_match_stats[rule_name][1] += 1

            if len(all_rules) <= 1:
                continue

            multi_match += 1

            # First rule = firing rule
            fire_name, fire_tier = all_rules[0]
            fire_val = TIER_ORDER[fire_tier]

            for shadow_name, shadow_tier in all_rules[1:]:
                shadow_val = TIER_ORDER[shadow_tier]

                key = (fire_name, shadow_name)
                interactions[key]['count'] += 1
                interactions[key]['fire_tier'] = fire_tier
                interactions[key]['shadow_tier'] = shadow_tier
                if is_gt:
                    interactions[key]['gt_when_fire'] += 1

                if shadow_val > fire_val and is_gt:
                    shadow_upgrade_gt += 1
                elif shadow_val < fire_val and is_gt:
                    shadow_downgrade_gt += 1

    # Print results
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Total predictions: {total_preds}")
    print(f"Multi-match: {multi_match} ({100*multi_match/max(1,total_preds):.1f}%)")
    print(f"GT hits where shadow would UPGRADE: {shadow_upgrade_gt}")
    print(f"GT hits where shadow would DOWNGRADE: {shadow_downgrade_gt}")

    # Firing rule precision
    print(f"\n{'='*80}")
    print("FIRING RULE PRECISION (what actually gets assigned)")
    print(f"{'='*80}")
    sorted_fire = sorted(firing_stats.items(), key=lambda x: x[1][0], reverse=True)
    print(f"{'Rule':<45} {'Total':>7} {'GT':>5} {'Prec':>7}")
    print("-" * 68)
    for rule, (total, gt) in sorted_fire[:35]:
        prec = 100 * gt / total if total > 0 else 0
        print(f"{rule:<45} {total:>7} {gt:>5} {prec:>6.1f}%")

    # Interactions where shadow tier > fire tier (potential improvements)
    print(f"\n{'='*80}")
    print("UPGRADE INTERACTIONS (shadow rule would assign HIGHER tier)")
    print("  These represent potential precision loss from current ordering")
    print(f"{'='*80}")

    upgrade_ints = {k: v for k, v in interactions.items()
                    if TIER_ORDER[v['shadow_tier']] > TIER_ORDER[v['fire_tier']] and v['count'] >= 5}
    sorted_up = sorted(upgrade_ints.items(), key=lambda x: x[1]['count'], reverse=True)

    print(f"{'Fire Rule':<30} {'Shadow Rule':<25} {'N':>5} {'GT':>4} {'Fire':>7} {'Shadow':>7}")
    print("-" * 85)
    for (fire, shadow), stats in sorted_up[:30]:
        print(f"{fire:<30} {shadow:<25} {stats['count']:>5} {stats['gt_when_fire']:>4} "
              f"{stats['fire_tier'].value:>7} {stats['shadow_tier'].value:>7}")

    # Interactions where shadow tier < fire tier (current ordering is CORRECT)
    print(f"\n{'='*80}")
    print("DOWNGRADE INTERACTIONS (shadow rule would assign LOWER tier)")
    print("  These represent cases where current ordering CORRECTLY overrides worse rule")
    print(f"{'='*80}")

    downgrade_ints = {k: v for k, v in interactions.items()
                      if TIER_ORDER[v['shadow_tier']] < TIER_ORDER[v['fire_tier']] and v['count'] >= 5}
    sorted_down = sorted(downgrade_ints.items(), key=lambda x: x[1]['count'], reverse=True)

    print(f"{'Fire Rule':<30} {'Shadow Rule':<25} {'N':>5} {'GT':>4} {'Fire':>7} {'Shadow':>7}")
    print("-" * 85)
    for (fire, shadow), stats in sorted_down[:30]:
        print(f"{fire:<30} {shadow:<25} {stats['count']:>5} {stats['gt_when_fire']:>4} "
              f"{stats['fire_tier'].value:>7} {stats['shadow_tier'].value:>7}")

    # KEY ANALYSIS: For upgrade interactions, compute precision of BOTH rules
    print(f"\n{'='*80}")
    print("PRECISION COMPARISON FOR UPGRADE INTERACTIONS")
    print("  fire_prec = precision when fire rule is the one that applies")
    print("  shadow_prec = precision when shadow rule matches (including shadowed cases)")
    print(f"{'='*80}")

    for (fire, shadow), stats in sorted_up[:20]:
        fire_total, fire_gt = rule_match_stats.get(fire, [0, 0])
        shadow_total, shadow_gt = rule_match_stats.get(shadow, [0, 0])
        fire_prec = 100 * fire_gt / fire_total if fire_total > 0 else 0
        shadow_prec = 100 * shadow_gt / shadow_total if shadow_total > 0 else 0

        print(f"\n  {fire} (fires) -> {shadow} (shadowed)")
        print(f"    Count: {stats['count']} | GT in overlap: {stats['gt_when_fire']}")
        print(f"    Fire rule:   {fire_prec:.1f}% (n={fire_total})")
        print(f"    Shadow rule:  {shadow_prec:.1f}% (n={shadow_total})")
        print(f"    Tier: {stats['fire_tier'].value} → {stats['shadow_tier'].value}")

        # Is the shadow rule genuinely better for THIS specific overlap?
        # We can't know exactly, but if shadow rule has higher overall precision
        # AND would give higher tier, this is a strong signal
        if shadow_prec > fire_prec + 5:  # >5pp better
            print(f"    *** CANDIDATE FOR REORDER: shadow precision {shadow_prec:.1f}% >> fire precision {fire_prec:.1f}% ***")

    # Same-tier interactions (rules compete but assign same tier)
    print(f"\n{'='*80}")
    print("SAME-TIER INTERACTIONS (both rules assign same tier, n>=10)")
    print(f"{'='*80}")

    same_ints = {k: v for k, v in interactions.items()
                 if TIER_ORDER[v['shadow_tier']] == TIER_ORDER[v['fire_tier']] and v['count'] >= 10}
    sorted_same = sorted(same_ints.items(), key=lambda x: x[1]['count'], reverse=True)

    print(f"{'Fire Rule':<30} {'Shadow Rule':<25} {'N':>5} {'GT':>4} {'Tier':>7}")
    print("-" * 75)
    for (fire, shadow), stats in sorted_same[:20]:
        print(f"{fire:<30} {shadow:<25} {stats['count']:>5} {stats['gt_when_fire']:>4} "
              f"{stats['fire_tier'].value:>7}")

    print(f"\n{'='*80}")
    print("DONE")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
