#!/usr/bin/env python3
"""
Hypothesis h205: Lymphoma Mechanism-Based Production Rules (CD30+/CD20+).

PURPOSE:
    Implement production rules that match lymphoma subtypes to appropriate targeted therapies.

MECHANISM GROUPS:
    - CD30+: Hodgkin, ALCL, PTCL, CTCL → Adcetris (brentuximab vedotin)
    - CD20+: B-cell NHL, DLBCL, FL, Burkitt → Rituximab

APPROACH:
    1. Identify CD30+ and CD20+ lymphoma diseases in predictions
    2. Check if the corresponding drug is being predicted
    3. Evaluate whether adding mechanism-based boosting improves precision
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"
DELIVERABLE_PATH = PROJECT_ROOT / "data" / "deliverables" / "drug_repurposing_predictions_with_confidence.xlsx"

# CD30+ lymphoma keywords (Adcetris targets)
CD30_KEYWORDS = [
    'hodgkin',
    'anaplastic large cell',
    'alcl',
    't-cell lymphoma',
    't cell lymphoma',
    'ptcl',  # peripheral T-cell lymphoma
    'ctcl',  # cutaneous T-cell lymphoma
    'cutaneous t cell',
    'cutaneous t-cell',
    'angioimmunoblastic',
    'mycosis fungoides',
]

# CD20+ lymphoma keywords (Rituximab targets)
CD20_KEYWORDS = [
    'b-cell',
    'b cell',
    'follicular',
    'dlbcl',
    'diffuse large',
    'burkitt',
    'nhl',
    'non-hodgkin',
    'nonhodgkin',
    'mantle cell',
    'marginal zone',
    'lymphoplasmacytic',
    'waldenstrom',
    'small lymphocytic',
    'indolent',
]

# Target drugs
ADCETRIS_NAMES = ['adcetris', 'brentuximab']
RITUXIMAB_NAMES = ['rituximab']


def classify_lymphoma_type(disease_name: str) -> str:
    """Classify lymphoma as CD30+, CD20+, both, or neither."""
    disease_lower = disease_name.lower()

    # Check if it's a lymphoma
    if 'lymphoma' not in disease_lower:
        return None

    is_cd30 = any(kw in disease_lower for kw in CD30_KEYWORDS)
    is_cd20 = any(kw in disease_lower for kw in CD20_KEYWORDS)

    if is_cd30 and is_cd20:
        return 'both'  # Some diseases could have both markers
    elif is_cd30:
        return 'CD30+'
    elif is_cd20:
        return 'CD20+'
    else:
        return 'unclassified'


def drug_matches_target(drug_name: str, target_list: list) -> bool:
    """Check if drug name matches target drug patterns."""
    drug_lower = drug_name.lower()
    return any(t in drug_lower for t in target_list)


def main():
    print("=" * 70)
    print("h205: Lymphoma Mechanism-Based Production Rules Analysis")
    print("=" * 70)
    print()

    # Load predictions
    pred_df = pd.read_excel(DELIVERABLE_PATH)
    print(f"Loaded {len(pred_df)} predictions")

    # Load ground truth
    gt_df = pd.read_excel(REFERENCE_DIR / "everycure" / "indicationList.xlsx")

    # Build GT lookup by disease name (approximate matching)
    gt_by_disease = defaultdict(set)
    for _, row in gt_df.iterrows():
        disease = str(row.get('final normalized disease label', '')).lower()
        drug = str(row.get('final normalized drug label', '')).lower()
        if disease and drug:
            gt_by_disease[disease].add(drug)

    # Check drug pool for Adcetris and Rituximab
    all_drugs = pred_df['drug_name'].unique()
    adcetris_in_pool = [d for d in all_drugs if drug_matches_target(d, ADCETRIS_NAMES)]
    rituximab_in_pool = [d for d in all_drugs if drug_matches_target(d, RITUXIMAB_NAMES)]

    print("=== DRUG POOL CHECK ===")
    print(f"Adcetris/Brentuximab in pool: {adcetris_in_pool if adcetris_in_pool else 'NOT FOUND'}")
    print(f"Rituximab in pool: {rituximab_in_pool if rituximab_in_pool else 'NOT FOUND'}")
    print()

    # Analyze lymphoma predictions
    lymphoma_pred = pred_df[pred_df['disease_name'].str.lower().str.contains('lymphoma', na=False)]
    print(f"Total lymphoma predictions: {len(lymphoma_pred)}")
    print(f"Unique lymphoma diseases: {lymphoma_pred['disease_name'].nunique()}")
    print()

    # Classify lymphomas
    results = {
        'CD30+': {'diseases': [], 'total_preds': 0, 'has_target_drug': 0, 'known_hits': 0},
        'CD20+': {'diseases': [], 'total_preds': 0, 'has_target_drug': 0, 'known_hits': 0},
        'both': {'diseases': [], 'total_preds': 0, 'has_target_drug': 0, 'known_hits': 0},
        'unclassified': {'diseases': [], 'total_preds': 0, 'has_target_drug': 0, 'known_hits': 0},
    }

    disease_analysis = []

    for disease in lymphoma_pred['disease_name'].unique():
        lymphoma_type = classify_lymphoma_type(disease)
        if lymphoma_type is None:
            continue

        disease_preds = lymphoma_pred[lymphoma_pred['disease_name'] == disease]
        n_preds = len(disease_preds)
        known_hits = disease_preds['is_known_indication'].sum()
        precision = known_hits / n_preds * 100 if n_preds > 0 else 0

        # Check for target drug
        has_rituximab = any(drug_matches_target(d, RITUXIMAB_NAMES) for d in disease_preds['drug_name'])
        has_adcetris = any(drug_matches_target(d, ADCETRIS_NAMES) for d in disease_preds['drug_name'])

        results[lymphoma_type]['diseases'].append(disease)
        results[lymphoma_type]['total_preds'] += n_preds
        results[lymphoma_type]['known_hits'] += known_hits

        if lymphoma_type == 'CD30+' and has_adcetris:
            results[lymphoma_type]['has_target_drug'] += 1
        elif lymphoma_type == 'CD20+' and has_rituximab:
            results[lymphoma_type]['has_target_drug'] += 1
        elif lymphoma_type == 'both' and (has_rituximab or has_adcetris):
            results[lymphoma_type]['has_target_drug'] += 1

        disease_analysis.append({
            'disease': disease,
            'type': lymphoma_type,
            'n_predictions': int(n_preds),
            'known_hits': int(known_hits),
            'precision': float(precision),
            'has_rituximab': bool(has_rituximab),
            'has_adcetris': bool(has_adcetris),
            'current_tier': disease_preds['confidence_tier'].iloc[0] if len(disease_preds) > 0 else 'N/A',
        })

    print("=== LYMPHOMA CLASSIFICATION RESULTS ===")
    for ltype, data in results.items():
        n_diseases = len(data['diseases'])
        if n_diseases == 0:
            continue
        precision = data['known_hits'] / data['total_preds'] * 100 if data['total_preds'] > 0 else 0
        has_target_pct = data['has_target_drug'] / n_diseases * 100

        print(f"\n{ltype}:")
        print(f"  Diseases: {n_diseases}")
        print(f"  Predictions: {data['total_preds']}")
        print(f"  Known hits: {data['known_hits']} ({precision:.1f}% precision)")
        print(f"  Has target drug predicted: {data['has_target_drug']} ({has_target_pct:.1f}%)")
        print(f"  Diseases: {data['diseases'][:5]}{'...' if len(data['diseases']) > 5 else ''}")

    # Check ground truth for targeted therapy
    print("\n=== GROUND TRUTH CHECK: Should Rituximab be predicted? ===")
    for disease in results['CD20+']['diseases']:
        disease_lower = disease.lower()
        # Check GT for any disease containing this name
        ritux_in_gt = False
        for gt_disease, gt_drugs in gt_by_disease.items():
            # Fuzzy match disease names
            if any(word in gt_disease for word in disease_lower.split()[:3] if len(word) > 3):
                if any('rituximab' in d for d in gt_drugs):
                    ritux_in_gt = True
                    break

        # Check predictions
        disease_preds = lymphoma_pred[lymphoma_pred['disease_name'] == disease]
        has_ritux_pred = any(drug_matches_target(d, RITUXIMAB_NAMES) for d in disease_preds['drug_name'])

        status = "✓ PREDICTED" if has_ritux_pred else "✗ NOT PREDICTED"
        gt_status = "IN GT" if ritux_in_gt else "NOT IN GT"
        print(f"  {disease}: {status} | {gt_status}")

    print("\n=== GROUND TRUTH CHECK: Should Adcetris be predicted? ===")
    for disease in results['CD30+']['diseases']:
        disease_lower = disease.lower()
        adcetris_in_gt = False
        for gt_disease, gt_drugs in gt_by_disease.items():
            if any(word in gt_disease for word in disease_lower.split()[:3] if len(word) > 3):
                if any('adcetris' in d or 'brentuximab' in d for d in gt_drugs):
                    adcetris_in_gt = True
                    break

        disease_preds = lymphoma_pred[lymphoma_pred['disease_name'] == disease]
        has_adcetris_pred = any(drug_matches_target(d, ADCETRIS_NAMES) for d in disease_preds['drug_name'])

        status = "✓ PREDICTED" if has_adcetris_pred else "✗ NOT PREDICTED"
        gt_status = "IN GT" if adcetris_in_gt else "NOT IN GT"
        print(f"  {disease}: {status} | {gt_status}")

    # Calculate potential improvement
    print("\n=== POTENTIAL IMPROVEMENT ANALYSIS ===")

    # For CD20+ diseases where Rituximab IS in pool but NOT predicted
    cd20_diseases_missing_rituximab = []
    for analysis in disease_analysis:
        if analysis['type'] == 'CD20+' and not analysis['has_rituximab']:
            # Check if Rituximab would be correct (in GT)
            disease_lower = analysis['disease'].lower()
            ritux_would_be_correct = False
            for gt_disease, gt_drugs in gt_by_disease.items():
                if any(word in gt_disease for word in disease_lower.split()[:3] if len(word) > 3):
                    if any('rituximab' in d for d in gt_drugs):
                        ritux_would_be_correct = True
                        break

            cd20_diseases_missing_rituximab.append({
                'disease': analysis['disease'],
                'current_precision': analysis['precision'],
                'rituximab_would_be_correct': ritux_would_be_correct,
            })

    print(f"\nCD20+ diseases missing Rituximab prediction: {len(cd20_diseases_missing_rituximab)}")
    for d in cd20_diseases_missing_rituximab:
        correct_str = "✓ Would be correct!" if d['rituximab_would_be_correct'] else "? Unknown"
        print(f"  {d['disease']}: current precision={d['current_precision']:.1f}% | {correct_str}")

    # Summary findings
    findings = {
        'hypothesis': 'h205',
        'title': 'Lymphoma Mechanism-Based Production Rules',
        'key_findings': {
            'adcetris_in_pool': len(adcetris_in_pool) > 0,
            'rituximab_in_pool': len(rituximab_in_pool) > 0,
            'cd30_diseases': len(results['CD30+']['diseases']),
            'cd20_diseases': len(results['CD20+']['diseases']),
            'cd30_has_target_predicted': results['CD30+']['has_target_drug'],
            'cd20_has_target_predicted': results['CD20+']['has_target_drug'],
            'cd30_precision': float(results['CD30+']['known_hits'] / results['CD30+']['total_preds'] * 100) if results['CD30+']['total_preds'] > 0 else 0.0,
            'cd20_precision': float(results['CD20+']['known_hits'] / results['CD20+']['total_preds'] * 100) if results['CD20+']['total_preds'] > 0 else 0.0,
        },
        'root_cause': {
            'cd30_failure': 'Adcetris NOT IN DRKG DRUG POOL - cannot be predicted',
            'cd20_partial': 'Rituximab in pool but not being recommended for all CD20+ diseases',
        },
        'disease_analysis': disease_analysis,
    }

    if len(adcetris_in_pool) == 0:
        print("\n" + "=" * 70)
        print("CRITICAL FINDING: Adcetris NOT in drug pool!")
        print("The model CANNOT predict Adcetris because it doesn't have embeddings.")
        print("This is a DRKG coverage gap, not a model failure.")
        print("=" * 70)
        findings['actionable_insight'] = 'Adcetris missing from DRKG - manual rules or external data needed'

    # Save findings
    output_path = ANALYSIS_DIR / "h205_lymphoma_mechanism_rules.json"
    with open(output_path, 'w') as f:
        json.dump(findings, f, indent=2)

    print(f"\nSaved findings to: {output_path}")

    return findings


if __name__ == "__main__":
    main()
