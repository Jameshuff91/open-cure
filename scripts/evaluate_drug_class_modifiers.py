#!/usr/bin/env python3
"""
h265: Evaluate Drug Class-Based Tier Modifiers

Based on h101 findings:
- Steroid + autoimmune = 76.4% precision (BOOST)
- Statin + metabolic = 68.0% precision (BOOST) - already implemented
- mAb overall = 1.9% precision (WARN/FILTER)
- Kinase inhibitor overall = 4.1% precision (WARN/FILTER)

This script evaluates the impact of adding drug class tier modifiers.
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Drug class patterns (suffix-based from h101)
DRUG_CLASS_PATTERNS = {
    'steroid': [
        r'.*sone$', r'.*solone$', r'.*olone$',  # prednisone, prednisolone, triamcinolone
        r'dexamethasone', r'hydrocortisone', r'cortisone',
        r'budesonide', r'fluticasone', r'beclomethasone',
        r'betamethasone', r'clobetasol', r'mometasone',
    ],
    'mab': [r'.*mab$'],  # Monoclonal antibodies
    'kinase_inhibitor': [r'.*tinib$', r'.*nib$'],  # imatinib, gefitinib, etc.
    'statin': [r'.*statin$'],  # atorvastatin, simvastatin, etc.
    'beta_blocker': [r'.*olol$'],  # metoprolol, atenolol, etc.
    'ace_inhibitor': [r'.*pril$'],  # lisinopril, enalapril, etc.
    'arb': [r'.*sartan$'],  # losartan, valsartan, etc.
    'fusion_protein': [r'.*cept$'],  # etanercept, aflibercept, etc.
}

# High-precision drug class × category combinations from h101
# Format: (drug_class, category) -> expected_precision
HIGH_PRECISION_COMBOS = {
    ('steroid', 'autoimmune'): 76.4,
    ('statin', 'metabolic'): 68.0,
    ('steroid', 'dermatological'): 60.0,
    ('steroid', 'respiratory'): 49.2,
    ('steroid', 'renal'): 48.6,
    ('beta_blocker', 'cardiovascular'): 40.0,  # Overall, some combos are 33%
    ('ace_inhibitor', 'cardiovascular'): 50.0,  # General CV drugs
}

# Low-precision drug class × category combinations from h101
# These should be WARNED/FILTERED (not actually used due to sparse GT)
LOW_PRECISION_COMBOS = {
    ('mab', 'other'): 1.9,
    ('mab', 'cancer'): 1.6,  # Counterintuitive!
    ('kinase_inhibitor', 'cancer'): 4.6,
    ('kinase_inhibitor', 'other'): 1.3,
    ('fusion_protein', 'other'): 3.2,
}


def classify_drug_class(drug_name: str) -> Optional[str]:
    """Classify drug into a class based on suffix patterns."""
    drug_lower = drug_name.lower()
    for drug_class, patterns in DRUG_CLASS_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, drug_lower):
                return drug_class
    return None


def load_data():
    """Load prediction data and ground truth."""
    data_dir = Path(__file__).parent.parent / 'data'

    # Load kNN predictions with tiers
    predictions_path = data_dir / 'deliverables' / 'drug_repurposing_predictions_with_confidence.xlsx'
    if predictions_path.exists():
        df = pd.read_excel(predictions_path)
        return df

    # Fallback: try JSON format
    predictions_json = data_dir / 'analysis' / 'knn_predictions_with_tiers.json'
    if predictions_json.exists():
        with open(predictions_json) as f:
            return json.load(f)

    return None


def analyze_drug_class_precision(df: pd.DataFrame) -> Dict:
    """Analyze precision by drug class and category."""
    results = defaultdict(lambda: {'total': 0, 'hits': 0})

    for _, row in df.iterrows():
        drug_name = row.get('drug', row.get('drug_name', ''))
        category = row.get('category', 'other')
        is_gt = row.get('is_ground_truth', row.get('is_gt', False))

        drug_class = classify_drug_class(drug_name)
        if drug_class:
            key = (drug_class, category)
            results[key]['total'] += 1
            if is_gt:
                results[key]['hits'] += 1

    # Calculate precision
    precision_results = {}
    for key, counts in results.items():
        if counts['total'] >= 10:  # Minimum sample size
            precision = counts['hits'] / counts['total'] * 100
            precision_results[key] = {
                'precision': precision,
                'total': counts['total'],
                'hits': counts['hits'],
            }

    return precision_results


def simulate_tier_modifiers(df: pd.DataFrame) -> Dict:
    """Simulate adding tier modifiers and measure impact."""

    # Track current vs modified tiers
    tier_changes = {'upgraded': 0, 'downgraded': 0, 'unchanged': 0}
    precision_before = {'GOLDEN': {'total': 0, 'hits': 0},
                       'HIGH': {'total': 0, 'hits': 0},
                       'MEDIUM': {'total': 0, 'hits': 0},
                       'LOW': {'total': 0, 'hits': 0}}
    precision_after = {'GOLDEN': {'total': 0, 'hits': 0},
                      'HIGH': {'total': 0, 'hits': 0},
                      'MEDIUM': {'total': 0, 'hits': 0},
                      'LOW': {'total': 0, 'hits': 0}}

    warnings = []  # Low-precision combos that should be warned

    for _, row in df.iterrows():
        drug_name = row.get('drug', row.get('drug_name', ''))
        category = row.get('category', 'other')
        current_tier = row.get('confidence_tier', row.get('tier', 'MEDIUM'))
        is_gt = row.get('is_ground_truth', row.get('is_gt', False))

        drug_class = classify_drug_class(drug_name)

        # Track before precision
        if current_tier in precision_before:
            precision_before[current_tier]['total'] += 1
            if is_gt:
                precision_before[current_tier]['hits'] += 1

        # Apply modifiers
        new_tier = current_tier
        modifier_applied = None

        if drug_class:
            combo = (drug_class, category)

            # Check for high-precision combos that should be BOOSTED
            if combo in HIGH_PRECISION_COMBOS:
                expected_prec = HIGH_PRECISION_COMBOS[combo]
                if expected_prec >= 50:  # Boost to GOLDEN
                    if current_tier in ['HIGH', 'MEDIUM', 'LOW']:
                        new_tier = 'GOLDEN'
                        modifier_applied = f'+GOLDEN ({expected_prec:.1f}%)'
                elif expected_prec >= 30:  # Boost to HIGH
                    if current_tier in ['MEDIUM', 'LOW']:
                        new_tier = 'HIGH'
                        modifier_applied = f'+HIGH ({expected_prec:.1f}%)'

            # Check for low-precision combos that should be WARNED/DOWNGRADED
            elif combo in LOW_PRECISION_COMBOS:
                expected_prec = LOW_PRECISION_COMBOS[combo]
                if expected_prec < 5:  # Demote if currently HIGH/MEDIUM
                    if current_tier in ['GOLDEN', 'HIGH', 'MEDIUM']:
                        new_tier = 'LOW'
                        modifier_applied = f'-LOW (only {expected_prec:.1f}%)'
                        warnings.append({
                            'drug': drug_name,
                            'category': category,
                            'drug_class': drug_class,
                            'expected_prec': expected_prec,
                            'original_tier': current_tier,
                            'is_gt': is_gt,
                        })

        # Track tier changes
        if new_tier != current_tier:
            tier_order = ['FILTER', 'LOW', 'MEDIUM', 'HIGH', 'GOLDEN']
            if tier_order.index(new_tier) > tier_order.index(current_tier):
                tier_changes['upgraded'] += 1
            else:
                tier_changes['downgraded'] += 1
        else:
            tier_changes['unchanged'] += 1

        # Track after precision
        if new_tier in precision_after:
            precision_after[new_tier]['total'] += 1
            if is_gt:
                precision_after[new_tier]['hits'] += 1

    return {
        'tier_changes': tier_changes,
        'precision_before': precision_before,
        'precision_after': precision_after,
        'warnings': warnings[:50],  # Limit to first 50
        'total_warnings': len(warnings),
    }


def main():
    """Main evaluation script."""
    print("=" * 70)
    print("h265: Drug Class-Based Tier Modifier Evaluation")
    print("=" * 70)

    # Load data
    df = load_data()
    if df is None:
        print("\nNo prediction data found. Generating analysis from scratch...")
        # We'll need to generate predictions first
        print("Run: python -m src.production_predictor --batch-analyze")
        return

    print(f"\nLoaded {len(df)} predictions")

    # Step 1: Analyze current drug class × category precision
    print("\n" + "=" * 70)
    print("STEP 1: Drug Class × Category Precision Analysis")
    print("=" * 70)

    precision_results = analyze_drug_class_precision(df)

    print("\nHigh-precision combinations (>30%):")
    high_prec = [(k, v) for k, v in precision_results.items() if v['precision'] >= 30]
    high_prec.sort(key=lambda x: -x[1]['precision'])
    for (drug_class, category), stats in high_prec:
        print(f"  {drug_class:20} + {category:15} = {stats['precision']:5.1f}% "
              f"({stats['hits']}/{stats['total']})")

    print("\nLow-precision combinations (<10%):")
    low_prec = [(k, v) for k, v in precision_results.items() if v['precision'] < 10]
    low_prec.sort(key=lambda x: x[1]['precision'])
    for (drug_class, category), stats in low_prec:
        print(f"  {drug_class:20} + {category:15} = {stats['precision']:5.1f}% "
              f"({stats['hits']}/{stats['total']})")

    # Step 2: Simulate tier modifier impact
    print("\n" + "=" * 70)
    print("STEP 2: Tier Modifier Impact Simulation")
    print("=" * 70)

    impact = simulate_tier_modifiers(df)

    print(f"\nTier changes:")
    print(f"  Upgraded:   {impact['tier_changes']['upgraded']}")
    print(f"  Downgraded: {impact['tier_changes']['downgraded']}")
    print(f"  Unchanged:  {impact['tier_changes']['unchanged']}")

    print("\nPrecision BEFORE modifiers:")
    for tier, stats in impact['precision_before'].items():
        if stats['total'] > 0:
            prec = stats['hits'] / stats['total'] * 100
            print(f"  {tier:8}: {prec:5.1f}% ({stats['hits']}/{stats['total']})")

    print("\nPrecision AFTER modifiers:")
    for tier, stats in impact['precision_after'].items():
        if stats['total'] > 0:
            prec = stats['hits'] / stats['total'] * 100
            print(f"  {tier:8}: {prec:5.1f}% ({stats['hits']}/{stats['total']})")

    # Step 3: Show warnings for low-precision biologics
    print("\n" + "=" * 70)
    print("STEP 3: Low-Precision Biologic Warnings")
    print("=" * 70)
    print(f"\nTotal predictions flagged for warning: {impact['total_warnings']}")

    if impact['warnings']:
        print("\nSample warnings (first 20):")
        for w in impact['warnings'][:20]:
            gt_marker = " ✓GT" if w['is_gt'] else ""
            print(f"  {w['drug']:30} | {w['category']:15} | {w['drug_class']:15} | "
                  f"was {w['original_tier']:8} → LOW ({w['expected_prec']:.1f}%){gt_marker}")

    # Step 4: Summary and recommendations
    print("\n" + "=" * 70)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 70)

    print("\n1. HIGH-CONFIDENCE RULES TO ADD:")
    for (drug_class, category), prec in HIGH_PRECISION_COMBOS.items():
        if prec >= 50:
            print(f"   - {drug_class} + {category} → GOLDEN ({prec:.1f}%)")
        elif prec >= 30:
            print(f"   - {drug_class} + {category} → HIGH ({prec:.1f}%)")

    print("\n2. LOW-CONFIDENCE WARNINGS TO ADD:")
    for (drug_class, category), prec in LOW_PRECISION_COMBOS.items():
        print(f"   - {drug_class} + {category} → WARN/FILTER ({prec:.1f}%)")


if __name__ == '__main__':
    main()
