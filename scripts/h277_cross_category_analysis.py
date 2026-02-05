#!/usr/bin/env python3
"""
h277: Cross-Category Hierarchy Matching Analysis

Test if drugs that treat conditions in one category also have precision
for related conditions in other categories.

Examples:
- Diabetes drugs → diabetic cardiovascular disease (metabolic → cardiovascular)
- Diabetes drugs → diabetic nephropathy (metabolic → renal)
- Cardiovascular drugs → diabetic heart disease (cardiovascular → metabolic)
"""

import json
from collections import defaultdict

# Define cross-category disease relationships
# Format: {(category1, category2): [('condition_keywords', 'base_category')]}
CROSS_CATEGORY_DISEASES = {
    # Diabetic complications span metabolic + other categories
    ('metabolic', 'cardiovascular'): [
        ('diabetic cardiomyopathy', 'metabolic'),
        ('diabetic cardiovascular', 'metabolic'),
        ('diabetes heart', 'metabolic'),
    ],
    ('metabolic', 'renal'): [
        ('diabetic nephropathy', 'metabolic'),
        ('diabetic kidney', 'metabolic'),
        ('diabetes renal', 'metabolic'),
    ],
    ('metabolic', 'neurological'): [
        ('diabetic neuropathy', 'metabolic'),
        ('diabetic nerve', 'metabolic'),
    ],
    ('metabolic', 'ophthalmological'): [
        ('diabetic retinopathy', 'metabolic'),
        ('diabetic macular', 'metabolic'),
    ],
    # Hypertensive complications
    ('cardiovascular', 'renal'): [
        ('hypertensive nephropathy', 'cardiovascular'),
        ('hypertensive kidney', 'cardiovascular'),
        ('renal hypertension', 'cardiovascular'),
    ],
    ('cardiovascular', 'neurological'): [
        ('hypertensive encephalopathy', 'cardiovascular'),
    ],
    # Autoimmune with organ involvement
    ('autoimmune', 'renal'): [
        ('lupus nephritis', 'autoimmune'),
        ('scleroderma renal', 'autoimmune'),
    ],
    ('autoimmune', 'neurological'): [
        ('neurological lupus', 'autoimmune'),
        ('cns lupus', 'autoimmune'),
    ],
}

# Category keywords for disease categorization
CATEGORY_KEYWORDS = {
    'metabolic': ['diabetes', 'diabetic', 'metabolic', 'obesity', 'lipid', 'cholesterol',
                  'thyroid', 'gout', 'hyperglycemia'],
    'cardiovascular': ['heart', 'cardiac', 'cardiovascular', 'hypertension', 'arrhythmia',
                       'coronary', 'atherosclerosis', 'stroke', 'vascular', 'angina',
                       'myocardial', 'infarction'],
    'neurological': ['neuro', 'brain', 'alzheimer', 'parkinson', 'epilepsy', 'seizure',
                     'migraine', 'dementia', 'neuropathy'],
    'autoimmune': ['autoimmune', 'lupus', 'arthritis', 'psoriasis', 'multiple sclerosis',
                   'systemic sclerosis', 'scleroderma', 'colitis', 'crohn'],  # Fixed: removed 'sclerosis' to avoid atherosclerosis match
    'infectious': ['infection', 'pneumonia', 'hepatitis', 'hiv', 'tuberculosis',
                   'sepsis', 'bacterial', 'viral', 'fungal', 'antibiotic'],
    'cancer': ['cancer', 'tumor', 'carcinoma', 'lymphoma', 'leukemia', 'melanoma',
               'neoplasm', 'oncology', 'malignant'],
    'renal': ['kidney', 'renal', 'nephro', 'glomerulo', 'urinary'],
    'respiratory': ['lung', 'pulmonary', 'asthma', 'copd', 'respiratory', 'bronch'],
    'gastrointestinal': ['gastro', 'intestinal', 'liver', 'hepatic', 'pancrea',
                         'stomach', 'colon', 'bowel', 'esophag'],
}


def categorize_disease(disease_name: str) -> str:
    """Categorize a disease by keywords."""
    name_lower = disease_name.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in name_lower:
                return category
    return 'other'


def get_disease_categories(disease_name: str) -> set:
    """Get ALL categories that apply to a disease (multi-label)."""
    name_lower = disease_name.lower()
    categories = set()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in name_lower:
                categories.add(category)
    return categories if categories else {'other'}


def is_cross_category_disease(disease_name: str) -> tuple:
    """
    Check if disease spans multiple categories.
    Returns (is_cross, primary_category, secondary_categories)
    """
    cats = get_disease_categories(disease_name)
    if len(cats) > 1:
        # Determine primary category (first match)
        primary = categorize_disease(disease_name)
        secondary = cats - {primary}
        return True, primary, secondary
    return False, list(cats)[0] if cats else 'other', set()


def analyze_cross_category():
    """Main analysis."""
    print("=" * 70)
    print("h277: Cross-Category Hierarchy Matching Analysis")
    print("=" * 70)

    # Load data
    with open('data/cache/ground_truth_cache.json') as f:
        cache = json.load(f)
    gt = {k: set(v) for k, v in cache['ground_truth'].items()}
    disease_names = cache['disease_names']

    with open('data/deliverables/drug_repurposing_predictions_with_confidence.json') as f:
        predictions = json.load(f)

    # Build drug -> GT categories mapping
    drug_gt_categories = defaultdict(set)
    for disease_id, drug_ids in gt.items():
        disease_name = disease_names.get(disease_id, disease_id)
        categories = get_disease_categories(disease_name)
        for drug_id in drug_ids:
            drug_gt_categories[drug_id].update(categories)

    print(f"\nDrugs in GT: {len(drug_gt_categories)}")

    # Analyze cross-category diseases in GT
    cross_cat_gt = 0
    for disease_id, drug_ids in gt.items():
        disease_name = disease_names.get(disease_id, disease_id)
        is_cross, _, _ = is_cross_category_disease(disease_name)
        if is_cross:
            cross_cat_gt += 1

    print(f"Cross-category diseases in GT: {cross_cat_gt}")

    # Analyze predictions
    results = {
        'within_category': {'total': 0, 'correct': 0, 'examples': []},
        'cross_category_match': {'total': 0, 'correct': 0, 'examples': []},
        'cross_category_no_match': {'total': 0, 'correct': 0, 'examples': []},
    }

    for pred in predictions:
        drug_id = pred['drug_id']
        pred_disease = pred['disease_name']
        is_correct = pred.get('is_known_indication', False)

        # Get prediction disease categories
        pred_cats = get_disease_categories(pred_disease)
        pred_primary = categorize_disease(pred_disease)

        # Get drug's GT categories
        drug_cats = drug_gt_categories.get(drug_id, set())

        if not drug_cats:
            continue  # Drug not in GT

        # Determine relationship
        overlap = pred_cats & drug_cats
        is_cross_pred, _, secondary = is_cross_category_disease(pred_disease)

        example = {
            'drug_name': pred['drug_name'],
            'disease': pred_disease,
            'pred_cats': list(pred_cats),
            'drug_gt_cats': list(drug_cats),
            'is_correct': is_correct,
            'tier': pred['confidence_tier']
        }

        if overlap:
            # At least one category matches
            if is_cross_pred and secondary - drug_cats:
                # Cross-category disease where drug has GT in primary but not secondary
                results['cross_category_match']['total'] += 1
                if is_correct:
                    results['cross_category_match']['correct'] += 1
                if len(results['cross_category_match']['examples']) < 10:
                    results['cross_category_match']['examples'].append(example)
            else:
                # Within-category match
                results['within_category']['total'] += 1
                if is_correct:
                    results['within_category']['correct'] += 1
                if len(results['within_category']['examples']) < 5:
                    results['within_category']['examples'].append(example)
        else:
            # No category match at all
            results['cross_category_no_match']['total'] += 1
            if is_correct:
                results['cross_category_no_match']['correct'] += 1
            if len(results['cross_category_no_match']['examples']) < 10:
                results['cross_category_no_match']['examples'].append(example)

    # Print results
    print("\n" + "=" * 70)
    print("PRECISION BY CATEGORY MATCH TYPE")
    print("=" * 70)

    for cat_type, data in results.items():
        total = data['total']
        correct = data['correct']
        prec = correct / total * 100 if total > 0 else 0

        print(f"\n{cat_type.upper()}")
        print(f"  Total: {total}, Correct: {correct}, Precision: {prec:.1f}%")

        if data['examples']:
            print("  Examples:")
            for ex in data['examples'][:5]:
                status = '✓' if ex['is_correct'] else '✗'
                cats = '+'.join(ex['pred_cats'])
                gt_cats = '+'.join(ex['drug_gt_cats'])
                print(f"    {status} {ex['drug_name']}: {ex['disease']}")
                print(f"       Pred cats: [{cats}], Drug GT cats: [{gt_cats}]")

    # Save results
    output = {
        'hypothesis': 'h277',
        'summary': {
            k: {
                'total': v['total'],
                'correct': v['correct'],
                'precision': v['correct'] / v['total'] * 100 if v['total'] > 0 else 0
            }
            for k, v in results.items()
        },
        'examples': {k: v['examples'] for k, v in results.items()}
    }

    with open('data/analysis/h277_cross_category_analysis.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to data/analysis/h277_cross_category_analysis.json")

    return results


if __name__ == '__main__':
    analyze_cross_category()
