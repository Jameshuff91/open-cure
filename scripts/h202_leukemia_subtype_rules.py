#!/usr/bin/env python3
"""
Hypothesis h202: Subtype-Specific Leukemia Production Rules.

PURPOSE:
    h199 found leukemia has 49 diseases with 45% having only 1 drug.
    h201 showed CML+imatinib works.

    This script:
    1. Identifies key drugs for each leukemia subtype from GT
    2. Defines keyword matchers for AML, CML, ALL, CLL
    3. Evaluates whether subtype-drug matching improves precision

LEUKEMIA SUBTYPES:
    - AML: cytarabine, azacitidine, daunorubicin, idarubicin, venetoclax
    - CML: imatinib, nilotinib, dasatinib, bosutinib, ponatinib
    - ALL: asparaginase, vincristine, methotrexate, cytarabine
    - CLL: ibrutinib, venetoclax, acalabrutinib, obinutuzumab
"""

import json
from pathlib import Path
from collections import defaultdict

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"
DELIVERABLE_PATH = PROJECT_ROOT / "data" / "deliverables" / "drug_repurposing_predictions_with_confidence.xlsx"

# Leukemia subtype keywords
LEUKEMIA_SUBTYPES = {
    'AML': {
        'keywords': ['acute myeloid', 'aml', 'acute myelogenous', 'acute myelocytic', 'acute nonlymphocytic'],
        'expected_drugs': ['cytarabine', 'azacitidine', 'daunorubicin', 'idarubicin', 'venetoclax', 'midostaurin', 'gilteritinib', 'glasdegib'],
    },
    'CML': {
        'keywords': ['chronic myeloid', 'cml', 'chronic myelogenous', 'chronic myelocytic'],
        'expected_drugs': ['imatinib', 'nilotinib', 'dasatinib', 'bosutinib', 'ponatinib'],
    },
    'ALL': {
        'keywords': ['acute lymphoblastic', 'all', 'acute lymphocytic', 'acute lymphoid'],
        'expected_drugs': ['asparaginase', 'vincristine', 'methotrexate', 'daunorubicin', 'prednisone', 'blinatumomab', 'inotuzumab'],
    },
    'CLL': {
        'keywords': ['chronic lymphocytic', 'cll', 'chronic lymphoid'],
        'expected_drugs': ['ibrutinib', 'venetoclax', 'acalabrutinib', 'obinutuzumab', 'rituximab', 'idelalisib', 'duvelisib'],
    },
}


def classify_leukemia_subtype(disease_name: str) -> str:
    """Classify disease as AML, CML, ALL, CLL, or other leukemia."""
    disease_lower = disease_name.lower()

    # Check if it's leukemia
    if 'leukemia' not in disease_lower and 'leukaemia' not in disease_lower:
        return None

    for subtype, data in LEUKEMIA_SUBTYPES.items():
        if any(kw in disease_lower for kw in data['keywords']):
            return subtype

    return 'other_leukemia'


def main():
    print("=" * 70)
    print("h202: Subtype-Specific Leukemia Production Rules")
    print("=" * 70)
    print()

    # Load GT
    gt_df = pd.read_excel(REFERENCE_DIR / "everycure" / "indicationList.xlsx")
    print(f"GT entries: {len(gt_df)}")

    # Find leukemia diseases
    leukemia_mask = gt_df['final normalized disease label'].str.lower().str.contains('leukemia|leukaemia', na=False)
    leukemia_gt = gt_df[leukemia_mask]
    print(f"Leukemia GT entries: {len(leukemia_gt)}")
    print(f"Unique leukemia diseases: {leukemia_gt['final normalized disease label'].nunique()}")

    # Classify leukemias by subtype
    subtype_stats = defaultdict(lambda: {'diseases': [], 'drugs': set(), 'gt_pairs': []})

    for disease in leukemia_gt['final normalized disease label'].unique():
        subtype = classify_leukemia_subtype(disease)
        if subtype:
            subtype_stats[subtype]['diseases'].append(disease)
            disease_drugs = leukemia_gt[leukemia_gt['final normalized disease label'] == disease]['final normalized drug label'].tolist()
            subtype_stats[subtype]['drugs'].update([d.lower() for d in disease_drugs])
            for drug in disease_drugs:
                subtype_stats[subtype]['gt_pairs'].append((disease, drug))

    print("\n=== LEUKEMIA SUBTYPE ANALYSIS ===")
    for subtype, data in subtype_stats.items():
        print(f"\n{subtype}:")
        print(f"  Diseases: {len(data['diseases'])}")
        print(f"  Unique drugs: {len(data['drugs'])}")
        print(f"  GT pairs: {len(data['gt_pairs'])}")
        print(f"  Diseases: {data['diseases'][:5]}{'...' if len(data['diseases']) > 5 else ''}")

        # Check expected drugs
        if subtype in LEUKEMIA_SUBTYPES:
            expected = set(d.lower() for d in LEUKEMIA_SUBTYPES[subtype]['expected_drugs'])
            actual = data['drugs']
            found = expected & actual
            missing = expected - actual
            print(f"  Expected drugs found: {len(found)}/{len(expected)}")
            if found:
                print(f"    Found: {list(found)[:5]}")
            if missing:
                print(f"    Missing: {list(missing)[:5]}")

    # Load predictions
    print("\n" + "=" * 70)
    print("PREDICTION ANALYSIS")
    print("=" * 70)

    pred_df = pd.read_excel(DELIVERABLE_PATH)

    # Find leukemia predictions
    leukemia_pred = pred_df[pred_df['disease_name'].str.lower().str.contains('leukemia|leukaemia', na=False)]
    print(f"\nLeukemia predictions: {len(leukemia_pred)}")
    print(f"Unique leukemia diseases in predictions: {leukemia_pred['disease_name'].nunique()}")

    # Analyze by subtype
    results = {}
    for disease in leukemia_pred['disease_name'].unique():
        subtype = classify_leukemia_subtype(disease)
        if subtype not in results:
            results[subtype] = {
                'diseases': [],
                'total_preds': 0,
                'known_hits': 0,
                'expected_drug_hits': 0,
            }

        disease_preds = leukemia_pred[leukemia_pred['disease_name'] == disease]
        results[subtype]['diseases'].append(disease)
        results[subtype]['total_preds'] += len(disease_preds)
        results[subtype]['known_hits'] += disease_preds['is_known_indication'].sum()

        # Check if expected drugs are in predictions
        if subtype in LEUKEMIA_SUBTYPES:
            expected_drugs = [d.lower() for d in LEUKEMIA_SUBTYPES[subtype]['expected_drugs']]
            for _, row in disease_preds.iterrows():
                if any(exp in row['drug_name'].lower() for exp in expected_drugs):
                    results[subtype]['expected_drug_hits'] += 1

    print("\n=== PREDICTION PERFORMANCE BY SUBTYPE ===")
    print("| Subtype | Diseases | Preds | Known Hits | Precision | Expected Drug Hits |")
    print("|---------|----------|-------|------------|-----------|-------------------|")

    for subtype in ['AML', 'CML', 'ALL', 'CLL', 'other_leukemia', None]:
        if subtype not in results:
            continue
        data = results[subtype]
        n_diseases = len(data['diseases'])
        total = data['total_preds']
        known = data['known_hits']
        precision = known / total * 100 if total > 0 else 0
        exp_hits = data.get('expected_drug_hits', 'N/A')
        subtype_name = subtype if subtype else 'None'
        print(f"| {subtype_name:7} | {n_diseases:8} | {total:5} | {known:10} | {precision:8.1f}% | {exp_hits:17} |")

    # Check for subtype-drug matches
    print("\n=== SUBTYPE-DRUG MATCH ANALYSIS ===")

    for subtype, spec in LEUKEMIA_SUBTYPES.items():
        print(f"\n{subtype}:")
        expected_drugs = [d.lower() for d in spec['expected_drugs']]

        for disease in results.get(subtype, {}).get('diseases', []):
            disease_preds = leukemia_pred[leukemia_pred['disease_name'] == disease]

            matched_drugs = []
            for _, row in disease_preds.iterrows():
                drug_lower = row['drug_name'].lower()
                for exp in expected_drugs:
                    if exp in drug_lower:
                        matched_drugs.append({
                            'drug': row['drug_name'],
                            'score': row['knn_score'],
                            'known': row['is_known_indication'],
                            'tier': row['confidence_tier'],
                        })
                        break

            if matched_drugs:
                print(f"  {disease}:")
                for m in matched_drugs[:5]:
                    known_str = "[KNOWN]" if m['known'] else ""
                    print(f"    ✓ {m['drug']}: score={m['score']:.3f}, tier={m['tier']} {known_str}")
            else:
                print(f"  {disease}: NO expected drugs in top 30")

    # Summary findings
    findings = {
        'hypothesis': 'h202',
        'title': 'Subtype-Specific Leukemia Production Rules',
        'subtype_analysis': {
            subtype: {
                'n_diseases': len(subtype_stats[subtype]['diseases']),
                'n_drugs': len(subtype_stats[subtype]['drugs']),
                'n_gt_pairs': len(subtype_stats[subtype]['gt_pairs']),
            }
            for subtype in subtype_stats
        },
        'prediction_results': {
            subtype: {
                'n_diseases': len(data['diseases']),
                'total_preds': int(data['total_preds']),
                'known_hits': int(data['known_hits']),
                'precision': float(data['known_hits'] / data['total_preds'] * 100) if data['total_preds'] > 0 else 0,
            }
            for subtype, data in results.items()
            if subtype
        },
    }

    # Save findings
    output_path = ANALYSIS_DIR / "h202_leukemia_subtype_rules.json"
    with open(output_path, 'w') as f:
        json.dump(findings, f, indent=2)

    print(f"\nSaved findings to: {output_path}")

    return findings


if __name__ == "__main__":
    main()
