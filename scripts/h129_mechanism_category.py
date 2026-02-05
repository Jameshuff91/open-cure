#!/usr/bin/env python3
"""h129: Mechanism-Category Interaction Deep Dive

Identify which disease categories benefit most from mechanism evidence.
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"

# Category tiers
TIER_1_CATEGORIES = {'autoimmune', 'dermatological', 'psychiatric', 'ophthalmic'}
TIER_2_CATEGORIES = {'cardiovascular', 'other', 'cancer'}
TIER_3_CATEGORIES = {'metabolic', 'respiratory', 'gastrointestinal', 'hematological',
                     'infectious', 'neurological', 'renal', 'musculoskeletal', 'endocrine'}

# Disease category keywords
CATEGORY_KEYWORDS = {
    'autoimmune': ['autoimmune', 'lupus', 'rheumatoid', 'arthritis', 'scleroderma', 'myasthenia',
                   'multiple sclerosis', 'crohn', 'colitis', 'psoriasis', 'sjögren', 'sjogren'],
    'infectious': ['infection', 'bacterial', 'viral', 'fungal', 'hiv', 'aids', 'hepatitis',
                   'tuberculosis', 'malaria', 'pneumonia', 'sepsis', 'meningitis', 'influenza'],
    'cancer': ['cancer', 'carcinoma', 'tumor', 'leukemia', 'lymphoma', 'melanoma',
               'neoplasm', 'oncology', 'sarcoma', 'myeloma'],
    'cardiovascular': ['cardiac', 'heart', 'coronary', 'hypertension', 'arrhythmia',
                       'atherosclerosis', 'stroke', 'vascular', 'myocardial', 'angina'],
    'neurological': ['neurological', 'alzheimer', 'parkinson', 'epilepsy', 'neuropathy',
                     'dementia', 'huntington', 'brain', 'ataxia', 'dystonia'],
    'metabolic': ['diabetes', 'metabolic', 'obesity', 'thyroid', 'hyperlipidemia',
                  'hypercholesterolemia', 'gout', 'porphyria', 'amyloid'],
    'psychiatric': ['depression', 'anxiety', 'bipolar', 'schizophrenia', 'psychiatric',
                    'ptsd', 'ocd', 'adhd', 'psychosis', 'mood disorder'],
    'respiratory': ['respiratory', 'asthma', 'copd', 'pulmonary', 'lung', 'bronchitis',
                    'pneumonitis', 'fibrosis'],
    'gastrointestinal': ['gastrointestinal', 'gastric', 'intestinal', 'bowel', 'liver',
                         'hepatic', 'cirrhosis', 'pancreatitis', 'celiac', 'ibs'],
    'dermatological': ['skin', 'dermatitis', 'eczema', 'psoriasis', 'dermatological',
                       'acne', 'urticaria', 'vitiligo', 'pemphigus'],
    'ophthalmic': ['eye', 'retinal', 'glaucoma', 'macular', 'ophthalmic', 'uveitis',
                   'conjunctivitis', 'keratitis', 'retinopathy'],
    'hematological': ['anemia', 'leukemia', 'lymphoma', 'hemophilia', 'thrombocytopenia',
                      'neutropenia', 'hematological', 'myelodysplastic', 'thalassemia'],
}


def classify_disease_category(disease_name):
    """Classify disease by name pattern."""
    name_lower = disease_name.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in name_lower for kw in keywords):
            return category
    return 'other'


def get_category_tier(category):
    """Get tier for a category (1=best, 3=worst)."""
    if category in TIER_1_CATEGORIES:
        return 1
    elif category in TIER_2_CATEGORIES:
        return 2
    return 3


def main():
    print("=" * 70)
    print("h129: MECHANISM-CATEGORY INTERACTION DEEP DIVE")
    print("=" * 70)

    # Load h135 tiered confidence data which has per-prediction mechanism support
    h135_path = ANALYSIS_DIR / "h135_tiered_confidence.json"
    if not h135_path.exists():
        print("ERROR: h135 data not found")
        return

    with open(h135_path) as f:
        h135_data = json.load(f)

    # Load h136 category rescue data which has per-category mechanism stats
    h136_path = ANALYSIS_DIR / "h136_tier23_category_rescue.json"
    with open(h136_path) as f:
        h136_data = json.load(f)

    # Analyze h136 data for mechanism effect by category
    print("\nMECHANISM PRECISION BY CATEGORY (from h136)")
    print("=" * 70)
    print(f"{'Category':20} {'Tier':>5} {'Base':>8} {'w/Mech':>8} {'Ratio':>8}")
    print("-" * 55)

    category_mech_effect = []
    for cat, results in h136_data['category_results'].items():
        if not isinstance(results, dict):
            continue
        base_prec = results.get('base_precision', 0)
        mech_prec = results.get('filters', {}).get('mech', {}).get('precision', 0)
        tier = get_category_tier(cat)

        if base_prec > 0 and mech_prec > 0:
            ratio = mech_prec / base_prec
            category_mech_effect.append({
                'category': cat,
                'tier': tier,
                'base_precision': base_prec,
                'mech_precision': mech_prec,
                'ratio': ratio,
            })
            print(f"{cat:20} {tier:5} {base_prec:7.1f}% {mech_prec:7.1f}% {ratio:7.2f}x")

    # Find categories where mechanism has >2x effect
    print("\n" + "=" * 70)
    print("CATEGORIES WHERE MECHANISM HAS >2x PREDICTIVE VALUE")
    print("=" * 70)

    high_mech = [c for c in category_mech_effect if c['ratio'] >= 2.0]
    if high_mech:
        for c in sorted(high_mech, key=lambda x: -x['ratio']):
            print(f"  {c['category']}: {c['ratio']:.2f}x (base {c['base_precision']:.1f}% -> mech {c['mech_precision']:.1f}%)")
    else:
        print("  None found with >2x ratio")

    # Analyze by tier
    print("\n" + "=" * 70)
    print("MECHANISM EFFECT BY TIER")
    print("=" * 70)

    for tier in [1, 2, 3]:
        tier_cats = [c for c in category_mech_effect if c['tier'] == tier]
        if tier_cats:
            avg_ratio = np.mean([c['ratio'] for c in tier_cats])
            print(f"Tier {tier}: avg mechanism ratio = {avg_ratio:.2f}x")
            for c in tier_cats:
                print(f"    {c['category']}: {c['ratio']:.2f}x")

    # Check if mechanism helps more in lower tiers
    print("\n" + "=" * 70)
    print("DOES MECHANISM HELP MORE IN LOWER TIERS?")
    print("=" * 70)

    tier1_ratios = [c['ratio'] for c in category_mech_effect if c['tier'] == 1]
    tier2_ratios = [c['ratio'] for c in category_mech_effect if c['tier'] == 2]
    tier3_ratios = [c['ratio'] for c in category_mech_effect if c['tier'] == 3]

    if tier1_ratios:
        print(f"Tier 1 mean ratio: {np.mean(tier1_ratios):.2f}x (n={len(tier1_ratios)})")
    if tier2_ratios:
        print(f"Tier 2 mean ratio: {np.mean(tier2_ratios):.2f}x (n={len(tier2_ratios)})")
    if tier3_ratios:
        print(f"Tier 3 mean ratio: {np.mean(tier3_ratios):.2f}x (n={len(tier3_ratios)})")

    # Check specific category interactions
    print("\n" + "=" * 70)
    print("CATEGORY-SPECIFIC INSIGHTS")
    print("=" * 70)

    # Infectious diseases - surprising success in h136
    infectious = next((c for c in category_mech_effect if c['category'] == 'infectious'), None)
    if infectious:
        print(f"\nInfectious diseases:")
        print(f"  Base: {infectious['base_precision']:.1f}%")
        print(f"  With mechanism: {infectious['mech_precision']:.1f}%")
        print(f"  Ratio: {infectious['ratio']:.2f}x")
        print(f"  Implication: Mechanism evidence is highly predictive for infectious diseases")

    # Cancer - typically hard category
    cancer = next((c for c in category_mech_effect if c['category'] == 'cancer'), None)
    if cancer:
        print(f"\nCancer:")
        print(f"  Base: {cancer['base_precision']:.1f}%")
        print(f"  With mechanism: {cancer['mech_precision']:.1f}%")
        print(f"  Ratio: {cancer['ratio']:.2f}x")
        print(f"  Implication: Mechanism helps modestly for cancer")

    # Save results
    results = {
        'category_effects': category_mech_effect,
        'high_mechanism_categories': [c['category'] for c in high_mech],
        'tier_mean_ratios': {
            'tier1': np.mean(tier1_ratios) if tier1_ratios else None,
            'tier2': np.mean(tier2_ratios) if tier2_ratios else None,
            'tier3': np.mean(tier3_ratios) if tier3_ratios else None,
        },
        'success_criteria_met': len(high_mech) >= 1,
    }

    output_path = ANALYSIS_DIR / "h129_mechanism_category.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Categories analyzed: {len(category_mech_effect)}")
    print(f"Categories with >2x mechanism effect: {len(high_mech)}")
    if high_mech:
        print(f"  Best: {max(high_mech, key=lambda x: x['ratio'])['category']} ({max(c['ratio'] for c in high_mech):.2f}x)")
    print(f"Success criteria (>2x in at least one category): {'MET' if len(high_mech) >= 1 else 'NOT MET'}")

    return results


if __name__ == "__main__":
    main()
