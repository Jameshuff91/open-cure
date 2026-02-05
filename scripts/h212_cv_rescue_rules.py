#!/usr/bin/env python3
"""
h212: Cardiovascular Disease-Specific Rescue Rules

h209 found that CV diseases (heart failure, hypertension) have the most blocked predictions.
This analysis:
1. Identifies all CV drug classes by ATC codes
2. For CV diseases + CV drug class match, tests rescue precision
3. Evaluates if rescue rules can recover blocked predictions with acceptable precision
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# ATC Level 2 codes for cardiovascular drugs
# C01 - Cardiac therapy
# C02 - Antihypertensives
# C03 - Diuretics
# C04 - Peripheral vasodilators
# C05 - Vasoprotectives
# C07 - Beta blocking agents
# C08 - Calcium channel blockers
# C09 - Agents acting on the renin-angiotensin system
# C10 - Lipid modifying agents

CV_ATC_LEVEL2 = {
    'C01': 'cardiac_therapy',
    'C02': 'antihypertensives',
    'C03': 'diuretics',
    'C04': 'peripheral_vasodilators',
    'C05': 'vasoprotectives',
    'C07': 'beta_blockers',
    'C08': 'calcium_channel_blockers',
    'C09': 'raas_agents',  # ACE inhibitors, ARBs
    'C10': 'lipid_modifying',
}

# More specific CV drug subclasses (ATC Level 4)
CV_SUBCLASSES = {
    # ACE Inhibitors
    'C09AA': 'ace_inhibitors',
    # ARBs
    'C09CA': 'arbs',
    # Beta Blockers
    'C07AB': 'beta_blockers_selective',
    'C07AA': 'beta_blockers_nonselective',
    # Calcium Channel Blockers
    'C08CA': 'dihydropyridines',  # amlodipine, nifedipine
    'C08DA': 'phenylalkylamines',  # verapamil
    'C08DB': 'benzothiazepines',  # diltiazem
    # Diuretics
    'C03AA': 'thiazides',
    'C03CA': 'loop_diuretics',  # furosemide
    'C03DA': 'aldosterone_antagonists',  # spironolactone
    # Cardiac glycosides
    'C01AA': 'cardiac_glycosides',  # digoxin
}

# CV disease patterns
CV_DISEASE_PATTERNS = [
    'heart failure', 'cardiac failure', 'cardiomyopathy',
    'hypertension', 'hypertensive',
    'angina', 'coronary', 'ischemic heart',
    'atrial fibrillation', 'arrhythmia', 'tachycardia',
    'myocardial infarction', 'heart attack',
    'stroke', 'cerebrovascular',
    'peripheral arterial', 'atherosclerosis',
    'pulmonary hypertension',
]


def load_data():
    """Load required data for analysis."""
    from production_predictor import DrugRepurposingPredictor
    from atc_features import ATCMapper

    predictor = DrugRepurposingPredictor(project_root)
    atc_mapper = ATCMapper()

    return predictor, atc_mapper


def is_cv_disease(disease_name: str) -> bool:
    """Check if disease is cardiovascular."""
    disease_lower = disease_name.lower()
    return any(pattern in disease_lower for pattern in CV_DISEASE_PATTERNS)


def get_cv_drug_class(drug_name: str, atc_mapper) -> Optional[str]:
    """Get CV drug class from ATC codes."""
    atc_codes = atc_mapper.get_atc_codes(drug_name)

    # Check level 4 (more specific)
    for code in atc_codes:
        if len(code) >= 5:
            l4 = code[:5]
            if l4 in CV_SUBCLASSES:
                return CV_SUBCLASSES[l4]

    # Check level 2 (broader)
    for code in atc_codes:
        if len(code) >= 3:
            l2 = code[:3]
            if l2 in CV_ATC_LEVEL2:
                return CV_ATC_LEVEL2[l2]

    return None


def analyze_cv_rescue(predictor, atc_mapper) -> Dict:
    """Analyze CV drug rescue potential."""

    # Load h209 blocked pairs
    with open(project_root / "data" / "analysis" / "h209_gt_coverage_analysis.json") as f:
        h209_data = json.load(f)

    blocked_pairs = h209_data['sample_blocked_pairs']

    # Focus on CV diseases
    cv_blocked = []
    non_cv_blocked = []

    for pair in blocked_pairs:
        disease_name = pair['disease_name']
        drug_name = pair['drug_name']

        if is_cv_disease(disease_name):
            drug_class = get_cv_drug_class(drug_name, atc_mapper)
            cv_blocked.append({
                **pair,
                'is_cv_drug': drug_class is not None,
                'cv_drug_class': drug_class,
            })
        else:
            non_cv_blocked.append(pair)

    # Now evaluate rescue potential more comprehensively
    # Load GT for precision calculation
    ground_truth = predictor.ground_truth
    disease_names = predictor.disease_names

    # Identify all CV diseases
    cv_diseases = []
    for disease_id in ground_truth:
        name = disease_names.get(disease_id, '')
        if is_cv_disease(name):
            cv_diseases.append((disease_id, name))

    print(f"Found {len(cv_diseases)} CV diseases in GT")

    # For each CV disease, identify CV drugs
    rescue_candidates = []
    rescue_hits = []
    rescue_misses = []

    for disease_id, disease_name in cv_diseases:
        gt_drugs = ground_truth.get(disease_id, set())

        # Get all drugs with ATC code 'C'
        cv_drugs_in_drkg = []
        for drug_id, drug_name in predictor.drug_id_to_name.items():
            if drug_id not in predictor.embeddings:
                continue
            atc_l1 = atc_mapper.get_atc_level1(drug_name)
            if 'C' in atc_l1:
                drug_class = get_cv_drug_class(drug_name, atc_mapper)
                cv_drugs_in_drkg.append((drug_id, drug_name, drug_class))

        # How many CV drugs are in GT for this disease?
        cv_gt_count = 0
        for drug_id in gt_drugs:
            if drug_id in predictor.drug_id_to_name:
                atc_l1 = atc_mapper.get_atc_level1(predictor.drug_id_to_name[drug_id])
                if 'C' in atc_l1:
                    cv_gt_count += 1

        if cv_gt_count > 0:
            rescue_candidates.append({
                'disease_id': disease_id,
                'disease_name': disease_name,
                'total_cv_drugs': len(cv_drugs_in_drkg),
                'cv_gt_drugs': cv_gt_count,
            })

    # Calculate precision at different rescue thresholds
    # For CV disease + CV drug (ATC = C), what's the precision?

    total_cv_drug_predictions = 0
    cv_drug_hits = 0

    for disease_id, disease_name in cv_diseases[:30]:  # Sample
        gt_drugs = ground_truth.get(disease_id, set())

        for drug_id, drug_name in predictor.drug_id_to_name.items():
            if drug_id not in predictor.embeddings:
                continue
            atc_l1 = atc_mapper.get_atc_level1(drug_name)
            if 'C' in atc_l1:
                total_cv_drug_predictions += 1
                if drug_id in gt_drugs:
                    cv_drug_hits += 1

    overall_precision = cv_drug_hits / total_cv_drug_predictions if total_cv_drug_predictions > 0 else 0

    # Now analyze by subclass for more targeted rescue
    subclass_stats = defaultdict(lambda: {'total': 0, 'hits': 0, 'diseases': set()})

    for disease_id, disease_name in cv_diseases:
        gt_drugs = ground_truth.get(disease_id, set())

        for drug_id, drug_name in predictor.drug_id_to_name.items():
            if drug_id not in predictor.embeddings:
                continue

            drug_class = get_cv_drug_class(drug_name, atc_mapper)
            if drug_class:
                subclass_stats[drug_class]['total'] += 1
                subclass_stats[drug_class]['diseases'].add(disease_id)
                if drug_id in gt_drugs:
                    subclass_stats[drug_class]['hits'] += 1

    # Calculate precision per subclass
    subclass_precision = {}
    for subclass, stats in subclass_stats.items():
        precision = stats['hits'] / stats['total'] if stats['total'] > 0 else 0
        subclass_precision[subclass] = {
            'total': stats['total'],
            'hits': stats['hits'],
            'precision': round(precision * 100, 2),
            'diseases_covered': len(stats['diseases']),
        }

    # Sort by precision
    sorted_subclasses = sorted(subclass_precision.items(), key=lambda x: x[1]['precision'], reverse=True)

    # Now test specific rescue rule: CV disease + specific drug classes
    # Test: heart failure + diuretics/ACE inhibitors
    hf_diseases = [(d_id, name) for d_id, name in cv_diseases if 'heart failure' in name.lower() or 'cardiac failure' in name.lower()]

    hf_rescue_stats = defaultdict(lambda: {'total': 0, 'hits': 0})

    for disease_id, disease_name in hf_diseases:
        gt_drugs = ground_truth.get(disease_id, set())

        for drug_id, drug_name in predictor.drug_id_to_name.items():
            if drug_id not in predictor.embeddings:
                continue

            drug_class = get_cv_drug_class(drug_name, atc_mapper)
            if drug_class in ['loop_diuretics', 'ace_inhibitors', 'arbs', 'beta_blockers_selective', 'aldosterone_antagonists']:
                hf_rescue_stats[drug_class]['total'] += 1
                if drug_id in gt_drugs:
                    hf_rescue_stats[drug_class]['hits'] += 1

    hf_rescue_precision = {
        cls: {
            'total': stats['total'],
            'hits': stats['hits'],
            'precision': round(stats['hits'] / stats['total'] * 100, 2) if stats['total'] > 0 else 0,
        }
        for cls, stats in hf_rescue_stats.items()
    }

    # Test: hypertension + antihypertensives
    htn_diseases = [(d_id, name) for d_id, name in cv_diseases if 'hypertension' in name.lower()]

    htn_rescue_stats = defaultdict(lambda: {'total': 0, 'hits': 0})

    for disease_id, disease_name in htn_diseases:
        gt_drugs = ground_truth.get(disease_id, set())

        for drug_id, drug_name in predictor.drug_id_to_name.items():
            if drug_id not in predictor.embeddings:
                continue

            drug_class = get_cv_drug_class(drug_name, atc_mapper)
            if drug_class in ['ace_inhibitors', 'arbs', 'beta_blockers_selective', 'beta_blockers_nonselective',
                              'dihydropyridines', 'thiazides', 'loop_diuretics']:
                htn_rescue_stats[drug_class]['total'] += 1
                if drug_id in gt_drugs:
                    htn_rescue_stats[drug_class]['hits'] += 1

    htn_rescue_precision = {
        cls: {
            'total': stats['total'],
            'hits': stats['hits'],
            'precision': round(stats['hits'] / stats['total'] * 100, 2) if stats['total'] > 0 else 0,
        }
        for cls, stats in htn_rescue_stats.items()
    }

    results = {
        'summary': {
            'cv_diseases_in_gt': len(cv_diseases),
            'cv_blocked_in_h209_sample': len(cv_blocked),
            'cv_blocked_with_cv_drug': sum(1 for p in cv_blocked if p['is_cv_drug']),
            'overall_cv_drug_precision': round(overall_precision * 100, 2),
        },
        'blocked_pairs_analysis': cv_blocked[:20],  # Sample
        'subclass_precision': dict(sorted_subclasses),
        'heart_failure_rescue': hf_rescue_precision,
        'hypertension_rescue': htn_rescue_precision,
    }

    return results


def main():
    print("h212: Cardiovascular Disease-Specific Rescue Rules")
    print("=" * 60)

    # Load data
    print("\n1. Loading data...")
    predictor, atc_mapper = load_data()

    # Run analysis
    print("\n2. Analyzing CV rescue potential...")
    results = analyze_cv_rescue(predictor, atc_mapper)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    s = results['summary']
    print(f"CV diseases in GT: {s['cv_diseases_in_gt']}")
    print(f"CV blocked pairs in h209 sample: {s['cv_blocked_in_h209_sample']}")
    print(f"  - with CV drug (ATC C): {s['cv_blocked_with_cv_drug']}")
    print(f"Overall CV drug precision (ATC C for CV disease): {s['overall_cv_drug_precision']}%")

    print("\n" + "-" * 60)
    print("PRECISION BY CV DRUG SUBCLASS (all CV diseases)")
    print("-" * 60)
    print(f"{'Subclass':<30} {'Hits':>8} {'Total':>8} {'Precision':>10}")
    for subclass, stats in results['subclass_precision'].items():
        print(f"{subclass:<30} {stats['hits']:>8} {stats['total']:>8} {stats['precision']:>9.1f}%")

    print("\n" + "-" * 60)
    print("HEART FAILURE SPECIFIC RESCUE")
    print("-" * 60)
    print(f"{'Drug Class':<30} {'Hits':>8} {'Total':>8} {'Precision':>10}")
    for cls, stats in sorted(results['heart_failure_rescue'].items(), key=lambda x: x[1]['precision'], reverse=True):
        print(f"{cls:<30} {stats['hits']:>8} {stats['total']:>8} {stats['precision']:>9.1f}%")

    print("\n" + "-" * 60)
    print("HYPERTENSION SPECIFIC RESCUE")
    print("-" * 60)
    print(f"{'Drug Class':<30} {'Hits':>8} {'Total':>8} {'Precision':>10}")
    for cls, stats in sorted(results['hypertension_rescue'].items(), key=lambda x: x[1]['precision'], reverse=True):
        print(f"{cls:<30} {stats['hits']:>8} {stats['total']:>8} {stats['precision']:>9.1f}%")

    # Save results
    output_path = project_root / "data" / "analysis" / "h212_cv_rescue_rules.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    results = main()
