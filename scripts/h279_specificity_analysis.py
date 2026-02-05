#!/usr/bin/env python3
"""
h279: Disease Specificity Scoring Analysis

Test if disease specificity (generic vs specific subtypes) correlates with precision.

Hypothesis: When drug GT is generic (e.g., "psoriasis") and prediction is specific
(e.g., "plaque psoriasis"), precision is higher than vice versa.

Specificity levels:
- Level 1: Root term (psoriasis, diabetes, heart failure)
- Level 2: Common subtypes (plaque psoriasis, type 2 diabetes)
- Level 3: Rare/complex variants (chronic plaque psoriasis, diabetic nephropathy)
"""

import json
from collections import defaultdict
from pathlib import Path

# Define specificity hierarchy with levels
# Format: {category: {group: [(term, level), ...]}}
DISEASE_SPECIFICITY = {
    'autoimmune': {
        'psoriasis': [
            ('psoriasis', 1),
            ('psoriasis vulgaris', 2),
            ('plaque psoriasis', 2),
            ('chronic plaque psoriasis', 3),
            ('scalp psoriasis', 3),
            ('erythrodermic psoriasis', 3),
            ('pustular psoriasis', 3),
            ('guttate psoriasis', 3),
        ],
        'rheumatoid_arthritis': [
            ('arthritis', 1),
            ('rheumatoid arthritis', 2),
            ('osteoarthritis', 2),
            ('psoriatic arthritis', 2),
            ('juvenile arthritis', 2),
            ('juvenile rheumatoid arthritis', 3),
            ('juvenile idiopathic arthritis', 3),
            ('polyarticular juvenile idiopathic arthritis', 3),
            ('systemic juvenile idiopathic arthritis', 3),
        ],
        'multiple_sclerosis': [
            ('multiple sclerosis', 1),
            ('relapsing multiple sclerosis', 2),
            ('progressive multiple sclerosis', 2),
            ('relapsing-remitting multiple sclerosis', 3),
            ('primary progressive multiple sclerosis', 3),
            ('secondary progressive multiple sclerosis', 3),
        ],
        'lupus': [
            ('lupus', 1),
            ('sle', 1),
            ('systemic lupus erythematosus', 2),
            ('cutaneous lupus', 2),
            ('discoid lupus', 3),
            ('lupus nephritis', 3),
            ('membranous lupus nephritis', 3),
        ],
        'colitis': [
            ('inflammatory bowel disease', 1),
            ('colitis', 1),
            ('ulcerative colitis', 2),
            ('crohns disease', 2),
            ('crohn disease', 2),
            ('chronic ulcerative colitis', 3),
            ('pediatric ulcerative colitis', 3),
            ('crohn colitis', 3),
        ],
    },
    'metabolic': {
        'diabetes': [
            ('diabetes', 1),
            ('diabetes mellitus', 1),
            ('type 2 diabetes', 2),
            ('type 1 diabetes', 2),
            ('diabetic', 2),
            ('hyperglycemia', 2),
            ('diabetic nephropathy', 3),
            ('diabetic neuropathy', 3),
            ('diabetic retinopathy', 3),
            ('diabetes insipidus', 3),
        ],
        'lipid': [
            ('dyslipidemia', 1),
            ('hyperlipidemia', 2),
            ('hypercholesterolemia', 2),
            ('hypertriglyceridemia', 3),
            ('familial hypercholesterolemia', 3),
        ],
        'thyroid': [
            ('thyroid', 1),
            ('thyroid disorder', 1),
            ('hypothyroidism', 2),
            ('hyperthyroidism', 2),
            ('goiter', 2),
            ('thyroiditis', 3),
        ],
    },
    'neurological': {
        'epilepsy': [
            ('seizure', 1),
            ('seizure disorder', 1),
            ('epilepsy', 2),
            ('partial seizure', 2),
            ('focal seizure', 2),
            ('generalized seizure', 2),
            ('absence seizure', 3),
            ('tonic-clonic seizure', 3),
            ('status epilepticus', 3),
        ],
        'parkinsons': [
            ('parkinsonism', 1),
            ('tremor', 1),
            ('parkinson', 2),
            ("parkinson's disease", 2),
            ('parkinsons disease', 2),
        ],
        'alzheimers': [
            ('cognitive impairment', 1),
            ('dementia', 1),
            ('memory loss', 1),
            ('alzheimer', 2),
            ("alzheimer's disease", 2),
        ],
        'migraine': [
            ('headache', 1),
            ('migraine', 2),
            ('chronic migraine', 3),
            ('episodic migraine', 3),
            ('cluster headache', 3),
            ('tension headache', 3),
        ],
    },
    'cardiovascular': {
        'heart_failure': [
            ('heart failure', 1),
            ('cardiomyopathy', 1),
            ('congestive heart failure', 2),
            ('chf', 2),
            ('dilated cardiomyopathy', 2),
            ('left ventricular failure', 3),
            ('right heart failure', 3),
        ],
        'hypertension': [
            ('hypertension', 1),
            ('high blood pressure', 1),
            ('essential hypertension', 2),
            ('pulmonary hypertension', 2),
            ('resistant hypertension', 3),
            ('renovascular hypertension', 3),
        ],
        'arrhythmia': [
            ('arrhythmia', 1),
            ('atrial fibrillation', 2),
            ('afib', 2),
            ('tachycardia', 2),
            ('bradycardia', 2),
            ('ventricular arrhythmia', 3),
            ('supraventricular tachycardia', 3),
            ('ventricular tachycardia', 3),
        ],
        'coronary': [
            ('coronary', 1),
            ('ischemic heart disease', 1),
            ('coronary artery disease', 2),
            ('angina', 2),
            ('acute coronary syndrome', 3),
            ('myocardial infarction', 3),
            ('heart attack', 3),
        ],
    },
    'respiratory': {
        'asthma': [
            ('asthma', 1),
            ('asthmatic', 1),
            ('bronchospasm', 2),
            ('reactive airway', 2),
            ('exercise-induced asthma', 3),
        ],
        'copd': [
            ('chronic obstructive', 1),
            ('copd', 2),
            ('emphysema', 2),
        ],
    },
    'infectious': {
        'pneumonia': [
            ('pneumonia', 1),
            ('bronchopneumonia', 2),
            ('bacterial pneumonia', 2),
            ('community-acquired pneumonia', 3),
            ('hospital-acquired pneumonia', 3),
            ('streptococcal pneumonia', 3),
            ('pneumococcal pneumonia', 3),
            ('aspiration pneumonia', 3),
        ],
        'hepatitis': [
            ('hepatitis', 1),
            ('viral hepatitis', 1),
            ('chronic hepatitis', 2),
            ('hepatitis b', 2),
            ('hepatitis c', 2),
            ('hepatitis c genotype 1', 3),
            ('hepatitis c genotype 2', 3),
            ('hepatitis c genotype 3', 3),
        ],
        'uti': [
            ('urinary tract infection', 1),
            ('uti', 1),
            ('cystitis', 2),
            ('pyelonephritis', 2),
            ('complicated urinary tract infection', 3),
            ('uncomplicated uti', 3),
            ('recurrent uti', 3),
        ],
    },
}


def get_disease_specificity(disease_name: str) -> tuple:
    """
    Get specificity level and group for a disease name.

    Returns: (category, group, level) or (None, None, None) if not found
    """
    disease_lower = disease_name.lower()

    for category, groups in DISEASE_SPECIFICITY.items():
        for group_name, terms in groups.items():
            for term, level in terms:
                if term in disease_lower or disease_lower in term:
                    return (category, group_name, level)

    return (None, None, None)


def analyze_specificity_vs_precision():
    """
    Main analysis: correlate specificity match direction with precision.
    """
    print("=" * 70)
    print("h279: Disease Specificity Scoring Analysis")
    print("=" * 70)

    # Load ground truth and disease name mapping from cache
    with open('data/cache/ground_truth_cache.json') as f:
        cache_data = json.load(f)

    ground_truth = cache_data['ground_truth']
    disease_names = cache_data['disease_names']

    # Convert GT to set format
    ground_truth = {k: set(v) for k, v in ground_truth.items()}

    # Build drug -> disease GT mapping with specificity
    drug_gt_specificity = defaultdict(list)

    for disease_id, drug_ids in ground_truth.items():
        disease_name = disease_names.get(disease_id, disease_id)
        category, group, level = get_disease_specificity(disease_name)

        if category is not None:
            for drug_id in drug_ids:
                drug_gt_specificity[drug_id].append({
                    'disease_id': disease_id,
                    'disease_name': disease_name,
                    'category': category,
                    'group': group,
                    'level': level
                })

    print(f"\nDrugs with GT specificity info: {len(drug_gt_specificity)}")

    # Load predictions
    with open('data/deliverables/drug_repurposing_predictions_with_confidence.json') as f:
        predictions = json.load(f)

    print(f"Total predictions: {len(predictions)}")

    # Analyze specificity relationships
    results = {
        'exact_disease': {'total': 0, 'correct': 0, 'examples': []},  # Same disease (trivial)
        'generic_to_specific': {'total': 0, 'correct': 0, 'examples': []},  # Level 1 GT -> Level 2/3 pred
        'specific_to_generic': {'total': 0, 'correct': 0, 'examples': []},  # Level 2/3 GT -> Level 1 pred
        'same_level_diff': {'total': 0, 'correct': 0, 'examples': []},  # Same level, different disease
        'within_generic_diff': {'total': 0, 'correct': 0, 'examples': []},  # Both level 1, different
        'no_match': {'total': 0}  # No group match
    }

    for pred in predictions:
        drug_id = pred['drug_id']
        pred_disease_name = pred['disease_name']
        is_correct = pred.get('is_known_indication', False)

        # Get prediction disease specificity
        pred_category, pred_group, pred_level = get_disease_specificity(pred_disease_name)

        if pred_category is None:
            continue  # Skip if we can't classify

        # Check if drug has GT in the same group
        drug_gts = drug_gt_specificity.get(drug_id, [])

        matching_gt = None
        for gt in drug_gts:
            if gt['category'] == pred_category and gt['group'] == pred_group:
                matching_gt = gt
                break

        if matching_gt is None:
            results['no_match']['total'] += 1
            continue

        gt_level = matching_gt['level']

        # Categorize the relationship
        example = {
            'drug_name': pred['drug_name'],
            'gt_disease': matching_gt['disease_name'],
            'gt_level': gt_level,
            'pred_disease': pred_disease_name,
            'pred_level': pred_level,
            'is_correct': is_correct,
            'category': pred_category,
            'group': pred_group
        }

        # Check if GT disease and prediction disease are the SAME
        gt_disease_lower = matching_gt['disease_name'].lower()
        pred_disease_lower = pred_disease_name.lower()
        is_exact_match = (gt_disease_lower == pred_disease_lower or
                          gt_disease_lower in pred_disease_lower or
                          pred_disease_lower in gt_disease_lower)

        if is_exact_match:
            # Same disease (exact or fuzzy match)
            key = 'exact_disease'
        elif gt_level < pred_level:
            # Generic GT -> Specific prediction (e.g., psoriasis -> plaque psoriasis)
            key = 'generic_to_specific'
        elif gt_level > pred_level:
            # Specific GT -> Generic prediction (e.g., plaque psoriasis -> psoriasis)
            key = 'specific_to_generic'
        elif gt_level == 1:
            # Both generic, different diseases in same group
            key = 'within_generic_diff'
        else:
            # Both specific (same level 2 or 3), different diseases
            key = 'same_level_diff'

        results[key]['total'] += 1
        if is_correct:
            results[key]['correct'] += 1

        # Store some examples
        if len(results[key]['examples']) < 5:
            results[key]['examples'].append(example)

    # Print results
    print("\n" + "=" * 70)
    print("SPECIFICITY DIRECTION VS PRECISION")
    print("=" * 70)

    for category, data in results.items():
        if category == 'no_match':
            print(f"\n{category}: {data['total']} predictions (no group match)")
            continue

        total = data['total']
        correct = data['correct']
        precision = (correct / total * 100) if total > 0 else 0

        print(f"\n{category.upper()}")
        print(f"  Total: {total}, Correct: {correct}, Precision: {precision:.1f}%")

        if data['examples']:
            print("  Examples:")
            for ex in data['examples'][:3]:
                gt_str = f"L{ex['gt_level']}: {ex['gt_disease']}"
                pred_str = f"L{ex['pred_level']}: {ex['pred_disease']}"
                status = "✓" if ex['is_correct'] else "✗"
                print(f"    {status} {ex['drug_name']}: {gt_str} -> {pred_str}")

    # Calculate correlation
    print("\n" + "=" * 70)
    print("PRECISION BY SPECIFICITY RELATIONSHIP")
    print("=" * 70)

    precision_data = []
    for category in ['exact_disease', 'generic_to_specific', 'specific_to_generic', 'same_level_diff', 'within_generic_diff']:
        data = results[category]
        total = data['total']
        if total >= 10:  # Only include if enough samples
            precision = data['correct'] / total * 100
            precision_data.append({
                'category': category,
                'total': total,
                'precision': precision
            })
            print(f"{category:25s}: {precision:5.1f}% (n={total})")

    # Test the hypothesis
    print("\n" + "=" * 70)
    print("HYPOTHESIS TEST")
    print("=" * 70)

    g2s = results['generic_to_specific']
    s2g = results['specific_to_generic']

    if g2s['total'] > 0 and s2g['total'] > 0:
        g2s_prec = g2s['correct'] / g2s['total'] * 100
        s2g_prec = s2g['correct'] / s2g['total'] * 100

        print(f"\nGeneric->Specific precision: {g2s_prec:.1f}% (n={g2s['total']})")
        print(f"Specific->Generic precision: {s2g_prec:.1f}% (n={s2g['total']})")
        print(f"Difference: {g2s_prec - s2g_prec:+.1f} pp")

        if g2s_prec > s2g_prec + 5:  # At least 5pp difference
            print("\n✓ HYPOTHESIS SUPPORTED: Generic->Specific predictions have higher precision")
        elif s2g_prec > g2s_prec + 5:
            print("\n✗ HYPOTHESIS REVERSED: Specific->Generic predictions have higher precision")
        else:
            print("\n~ INCONCLUSIVE: No significant precision difference")

    # Save detailed results
    output = {
        'hypothesis': 'h279',
        'summary': {
            cat: {
                'total': data['total'],
                'correct': data.get('correct', 0),
                'precision': (data.get('correct', 0) / data['total'] * 100) if data['total'] > 0 else 0
            }
            for cat, data in results.items()
        },
        'examples': {
            cat: data.get('examples', [])[:10]
            for cat, data in results.items()
            if cat != 'no_match'
        }
    }

    with open('data/analysis/h279_specificity_analysis.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to data/analysis/h279_specificity_analysis.json")

    return results


if __name__ == '__main__':
    analyze_specificity_vs_precision()
