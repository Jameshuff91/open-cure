#!/usr/bin/env python3
"""
h282: Disease Hierarchy Depth as Confidence Signal

Test if the magnitude of hierarchy depth delta correlates with precision.
Delta = pred_level - gt_level
- Positive delta = prediction is MORE specific than GT
- Negative delta = prediction is LESS specific than GT
- Zero delta = same specificity level
"""

import json
from collections import defaultdict

# Reuse DISEASE_SPECIFICITY from h279
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
        ],
        'multiple_sclerosis': [
            ('multiple sclerosis', 1),
            ('relapsing multiple sclerosis', 2),
            ('progressive multiple sclerosis', 2),
            ('relapsing-remitting multiple sclerosis', 3),
            ('primary progressive multiple sclerosis', 3),
        ],
        'lupus': [
            ('lupus', 1),
            ('sle', 1),
            ('systemic lupus erythematosus', 2),
            ('cutaneous lupus', 2),
            ('discoid lupus', 3),
            ('lupus nephritis', 3),
        ],
        'colitis': [
            ('inflammatory bowel disease', 1),
            ('colitis', 1),
            ('ulcerative colitis', 2),
            ('crohns disease', 2),
            ('crohn disease', 2),
            ('chronic ulcerative colitis', 3),
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
        ],
        'lipid': [
            ('dyslipidemia', 1),
            ('hyperlipidemia', 2),
            ('hypercholesterolemia', 2),
            ('hypertriglyceridemia', 3),
            ('familial hypercholesterolemia', 3),
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
        ],
        'migraine': [
            ('headache', 1),
            ('migraine', 2),
            ('chronic migraine', 3),
            ('episodic migraine', 3),
            ('cluster headache', 3),
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
        ],
        'hypertension': [
            ('hypertension', 1),
            ('high blood pressure', 1),
            ('essential hypertension', 2),
            ('pulmonary hypertension', 2),
            ('resistant hypertension', 3),
        ],
        'coronary': [
            ('coronary', 1),
            ('ischemic heart disease', 1),
            ('coronary artery disease', 2),
            ('angina', 2),
            ('acute coronary syndrome', 3),
            ('myocardial infarction', 3),
        ],
    },
    'respiratory': {
        'asthma': [
            ('asthma', 1),
            ('bronchospasm', 2),
            ('exercise-induced asthma', 3),
        ],
        'copd': [
            ('chronic obstructive', 1),
            ('copd', 2),
            ('emphysema', 2),
        ],
    },
    'infectious': {
        'hepatitis': [
            ('hepatitis', 1),
            ('viral hepatitis', 1),
            ('chronic hepatitis', 2),
            ('hepatitis b', 2),
            ('hepatitis c', 2),
            ('hepatitis c genotype 1', 3),
        ],
        'uti': [
            ('urinary tract infection', 1),
            ('uti', 1),
            ('cystitis', 2),
            ('pyelonephritis', 2),
            ('complicated urinary tract infection', 3),
        ],
    },
}


def get_disease_specificity(disease_name: str) -> tuple:
    """Get (category, group, level) for a disease name."""
    disease_lower = disease_name.lower()
    for category, groups in DISEASE_SPECIFICITY.items():
        for group_name, terms in groups.items():
            for term, level in terms:
                if term in disease_lower or disease_lower in term:
                    return (category, group_name, level)
    return (None, None, None)


def analyze_depth_delta():
    """Main analysis."""
    print("=" * 70)
    print("h282: Disease Hierarchy Depth as Confidence Signal")
    print("=" * 70)

    # Load data
    with open('data/cache/ground_truth_cache.json') as f:
        cache = json.load(f)
    gt = {k: set(v) for k, v in cache['ground_truth'].items()}
    disease_names = cache['disease_names']

    with open('data/deliverables/drug_repurposing_predictions_with_confidence.json') as f:
        predictions = json.load(f)

    # Build drug -> GT specificity mapping
    drug_gt_specificity = defaultdict(list)
    for disease_id, drug_ids in gt.items():
        disease_name = disease_names.get(disease_id, disease_id)
        cat, group, level = get_disease_specificity(disease_name)
        if cat is not None:
            for drug_id in drug_ids:
                drug_gt_specificity[drug_id].append({
                    'disease_name': disease_name,
                    'category': cat,
                    'group': group,
                    'level': level
                })

    # Analyze depth deltas
    delta_results = defaultdict(lambda: {'total': 0, 'correct': 0, 'examples': []})

    for pred in predictions:
        drug_id = pred['drug_id']
        pred_disease = pred['disease_name']
        is_correct = pred.get('is_known_indication', False)

        pred_cat, pred_group, pred_level = get_disease_specificity(pred_disease)
        if pred_cat is None:
            continue

        # Find matching GT in same group
        drug_gts = drug_gt_specificity.get(drug_id, [])
        matching_gt = None
        for gt_entry in drug_gts:
            if gt_entry['category'] == pred_cat and gt_entry['group'] == pred_group:
                matching_gt = gt_entry
                break

        if matching_gt is None:
            continue  # No same-group GT

        gt_level = matching_gt['level']
        delta = pred_level - gt_level

        delta_results[delta]['total'] += 1
        if is_correct:
            delta_results[delta]['correct'] += 1

        if len(delta_results[delta]['examples']) < 5:
            delta_results[delta]['examples'].append({
                'drug': pred['drug_name'],
                'gt_disease': matching_gt['disease_name'],
                'gt_level': gt_level,
                'pred_disease': pred_disease,
                'pred_level': pred_level,
                'delta': delta,
                'is_correct': is_correct
            })

    # Print results
    print("\n" + "=" * 70)
    print("PRECISION BY HIERARCHY DELTA")
    print("(delta = pred_level - gt_level; positive = more specific)")
    print("=" * 70)

    for delta in sorted(delta_results.keys()):
        d = delta_results[delta]
        prec = d['correct'] / d['total'] * 100 if d['total'] > 0 else 0

        if delta > 0:
            direction = f"MORE specific by {delta}"
        elif delta < 0:
            direction = f"LESS specific by {abs(delta)}"
        else:
            direction = "SAME level"

        print(f"\nDelta {delta:+d} ({direction})")
        print(f"  Total: {d['total']}, Correct: {d['correct']}, Precision: {prec:.1f}%")

        if d['examples']:
            print("  Examples:")
            for ex in d['examples'][:3]:
                status = '✓' if ex['is_correct'] else '✗'
                print(f"    {status} {ex['drug']}: L{ex['gt_level']} '{ex['gt_disease']}' → L{ex['pred_level']} '{ex['pred_disease']}'")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_analyzed = sum(d['total'] for d in delta_results.values())
    print(f"\nTotal predictions analyzed: {total_analyzed}")
    print("\nPrecision by delta magnitude:")

    # Group by absolute delta
    abs_delta_prec = defaultdict(lambda: {'total': 0, 'correct': 0})
    for delta, d in delta_results.items():
        abs_d = abs(delta)
        abs_delta_prec[abs_d]['total'] += d['total']
        abs_delta_prec[abs_d]['correct'] += d['correct']

    for abs_d in sorted(abs_delta_prec.keys()):
        d = abs_delta_prec[abs_d]
        prec = d['correct'] / d['total'] * 100 if d['total'] > 0 else 0
        print(f"  |delta| = {abs_d}: {prec:.1f}% (n={d['total']})")

    # Test hypothesis: does |delta| predict precision?
    print("\n" + "=" * 70)
    print("HYPOTHESIS TEST")
    print("=" * 70)

    if 0 in abs_delta_prec and 1 in abs_delta_prec:
        same_prec = abs_delta_prec[0]['correct'] / abs_delta_prec[0]['total'] * 100
        off_by_1_prec = abs_delta_prec[1]['correct'] / abs_delta_prec[1]['total'] * 100

        print(f"\nSame level (|delta|=0): {same_prec:.1f}%")
        print(f"Off by 1 (|delta|=1): {off_by_1_prec:.1f}%")
        print(f"Difference: {same_prec - off_by_1_prec:.1f} pp")

        if same_prec > off_by_1_prec + 5:
            print("\n✓ HYPOTHESIS SUPPORTED: Same-level predictions have higher precision")
        elif off_by_1_prec > same_prec + 5:
            print("\n✗ HYPOTHESIS REVERSED: Cross-level predictions have higher precision")
        else:
            print("\n~ INCONCLUSIVE: No significant precision difference by delta")

    # Save results
    output = {
        'hypothesis': 'h282',
        'results_by_delta': {
            str(delta): {
                'total': d['total'],
                'correct': d['correct'],
                'precision': d['correct'] / d['total'] * 100 if d['total'] > 0 else 0
            }
            for delta, d in delta_results.items()
        },
        'results_by_abs_delta': {
            str(abs_d): {
                'total': d['total'],
                'correct': d['correct'],
                'precision': d['correct'] / d['total'] * 100 if d['total'] > 0 else 0
            }
            for abs_d, d in abs_delta_prec.items()
        }
    }

    with open('data/analysis/h282_depth_delta_analysis.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to data/analysis/h282_depth_delta_analysis.json")

    return delta_results


if __name__ == '__main__':
    analyze_depth_delta()
