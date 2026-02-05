#!/usr/bin/env python3
"""h139: Hybrid Confidence - Category + Per-Disease Features

Test whether combining category tier expectations with per-disease confidence
features (prob_h52, prob_h65) improves calibration over either alone.
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"

# Category tiers from h71/h135
TIER_1_CATEGORIES = {'autoimmune', 'dermatological', 'psychiatric', 'ophthalmic'}
TIER_2_CATEGORIES = {'cardiovascular', 'other', 'cancer'}
TIER_3_CATEGORIES = {'metabolic', 'respiratory', 'gastrointestinal', 'hematological',
                     'infectious', 'neurological', 'renal', 'musculoskeletal', 'endocrine'}

# Expected hit rates by category (from h136 + Tier 1 assumption)
CATEGORY_EXPECTED = {
    'autoimmune': 0.25, 'dermatological': 0.25, 'psychiatric': 0.25, 'ophthalmic': 0.25,
    'cardiovascular': 0.11, 'other': 0.08, 'cancer': 0.05,
    'metabolic': 0.08, 'respiratory': 0.12, 'gastrointestinal': 0.0,
    'hematological': 0.03, 'infectious': 0.10, 'neurological': 0.02,
    'renal': 0.05, 'musculoskeletal': 0.05, 'endocrine': 0.05
}


def load_per_disease_data():
    """Load per-disease data from h79."""
    h79_path = ANALYSIS_DIR / "h79_per_disease_confidence.json"
    with open(h79_path) as f:
        return json.load(f)


def aggregate_disease_stats(h79_data):
    """Aggregate per-disease statistics across seeds."""
    disease_stats = defaultdict(lambda: {
        'hits': 0, 'total': 0, 'category': None,
        'prob_h52': [], 'prob_h65': [], 'prob_category': [], 'combined_avg': []
    })

    for seed_result in h79_data['seed_results']:
        for d in seed_result['diseases']:
            name = d['disease_name']
            disease_stats[name]['hits'] += d['hit']
            disease_stats[name]['total'] += 1
            disease_stats[name]['category'] = d['category']
            disease_stats[name]['prob_h52'].append(d.get('prob_h52', 0))
            disease_stats[name]['prob_h65'].append(d.get('prob_h65', 0))
            disease_stats[name]['prob_category'].append(d.get('prob_category', 0))
            disease_stats[name]['combined_avg'].append(d.get('combined_avg', 0))

    # Compute averages
    for name, stats in disease_stats.items():
        stats['hit_rate'] = stats['hits'] / stats['total'] if stats['total'] > 0 else 0
        stats['avg_prob_h52'] = np.mean(stats['prob_h52'])
        stats['avg_prob_h65'] = np.mean(stats['prob_h65'])
        stats['avg_prob_category'] = np.mean(stats['prob_category'])
        stats['avg_combined'] = np.mean(stats['combined_avg'])

    return disease_stats


def compute_calibration_error(disease_stats, prediction_fn):
    """Compute mean absolute calibration error given a prediction function."""
    errors = []
    for name, stats in disease_stats.items():
        actual = stats['hit_rate']
        predicted = prediction_fn(stats)
        errors.append(abs(actual - predicted))
    return np.mean(errors), np.std(errors)


def main():
    print("=" * 70)
    print("h139: HYBRID CONFIDENCE - Category + Per-Disease Features")
    print("=" * 70)

    # Load data
    h79_data = load_per_disease_data()
    disease_stats = aggregate_disease_stats(h79_data)
    n_diseases = len(disease_stats)

    print(f"\nLoaded {n_diseases} diseases")

    # Define prediction functions
    def pred_category_only(stats):
        """Pure category-based prediction."""
        return CATEGORY_EXPECTED.get(stats['category'], 0.05)

    def pred_prob_h52_only(stats):
        """Pure prob_h52 prediction."""
        return stats['avg_prob_h52']

    def pred_prob_h65_only(stats):
        """Pure prob_h65 prediction."""
        return stats['avg_prob_h65']

    def pred_combined_avg_only(stats):
        """Pure combined average prediction."""
        return stats['avg_combined']

    def pred_hybrid_equal(stats):
        """Equal weight: 0.5 * category + 0.5 * prob_h52."""
        cat = CATEGORY_EXPECTED.get(stats['category'], 0.05)
        per_disease = stats['avg_prob_h52']
        return 0.5 * cat + 0.5 * per_disease

    def pred_hybrid_weighted_h52(stats):
        """0.3 * category + 0.7 * prob_h52."""
        cat = CATEGORY_EXPECTED.get(stats['category'], 0.05)
        per_disease = stats['avg_prob_h52']
        return 0.3 * cat + 0.7 * per_disease

    def pred_hybrid_weighted_combined(stats):
        """0.3 * category + 0.7 * combined_avg."""
        cat = CATEGORY_EXPECTED.get(stats['category'], 0.05)
        per_disease = stats['avg_combined']
        return 0.3 * cat + 0.7 * per_disease

    def pred_max_category_h52(stats):
        """Max of category and prob_h52."""
        cat = CATEGORY_EXPECTED.get(stats['category'], 0.05)
        return max(cat, stats['avg_prob_h52'])

    def pred_category_adjusted_by_h52(stats):
        """Category expectation adjusted by prob_h52 deviation from mean."""
        cat = CATEGORY_EXPECTED.get(stats['category'], 0.05)
        h52_mean = 0.55  # Approximate mean of prob_h52
        h52_dev = stats['avg_prob_h52'] - h52_mean
        return np.clip(cat + 0.5 * h52_dev, 0, 1)

    def pred_category_if_other_else_h52(stats):
        """Use prob_h52 for 'other', category for rest."""
        if stats['category'] == 'other':
            return stats['avg_prob_h52']
        return CATEGORY_EXPECTED.get(stats['category'], 0.05)

    # Evaluate all approaches
    approaches = [
        ("Category only", pred_category_only),
        ("prob_h52 only", pred_prob_h52_only),
        ("prob_h65 only", pred_prob_h65_only),
        ("combined_avg only", pred_combined_avg_only),
        ("Hybrid 0.5/0.5 (cat+h52)", pred_hybrid_equal),
        ("Hybrid 0.3/0.7 (cat+h52)", pred_hybrid_weighted_h52),
        ("Hybrid 0.3/0.7 (cat+combined)", pred_hybrid_weighted_combined),
        ("Max(cat, h52)", pred_max_category_h52),
        ("Category + h52 deviation", pred_category_adjusted_by_h52),
        ("h52 for 'other', cat for rest", pred_category_if_other_else_h52),
    ]

    print(f"\n{'='*70}")
    print("CALIBRATION ERROR BY APPROACH")
    print(f"{'='*70}")
    print(f"\n{'Approach':35} {'Mean Error':>12} {'Std':>10}")
    print("-" * 60)

    results = []
    for name, fn in approaches:
        mean_err, std_err = compute_calibration_error(disease_stats, fn)
        results.append((name, mean_err, std_err))
        print(f"{name:35} {mean_err*100:10.1f} pp {std_err*100:8.1f} pp")

    # Find best approach
    best = min(results, key=lambda x: x[1])
    print(f"\nBest approach: {best[0]} ({best[1]*100:.1f} pp)")

    # Compare to baselines
    cat_only = next(r for r in results if r[0] == "Category only")
    h52_only = next(r for r in results if r[0] == "prob_h52 only")

    print(f"\n{'='*70}")
    print("COMPARISON TO BASELINES")
    print(f"{'='*70}")
    print(f"Category only: {cat_only[1]*100:.1f} pp")
    print(f"prob_h52 only: {h52_only[1]*100:.1f} pp")
    print(f"Best hybrid:   {best[1]*100:.1f} pp")

    improvement_vs_cat = (cat_only[1] - best[1]) / cat_only[1] * 100
    improvement_vs_h52 = (h52_only[1] - best[1]) / h52_only[1] * 100

    print(f"\nImprovement vs category: {improvement_vs_cat:.1f}%")
    print(f"Improvement vs prob_h52: {improvement_vs_h52:.1f}%")

    # Analyze by category
    print(f"\n{'='*70}")
    print("PER-CATEGORY ANALYSIS")
    print(f"{'='*70}")

    def analyze_category(cat_name, pred_fn):
        cat_diseases = [(n, s) for n, s in disease_stats.items() if s['category'] == cat_name]
        if not cat_diseases:
            return None
        errors = [abs(s['hit_rate'] - pred_fn(s)) for n, s in cat_diseases]
        return np.mean(errors), len(cat_diseases)

    categories = ['other', 'cancer', 'autoimmune', 'infectious', 'cardiovascular',
                  'respiratory', 'metabolic', 'dermatological']

    print(f"\n{'Category':20} {'N':>5} {'Cat Only':>10} {'h52 Only':>10} {'Best Hybrid':>12}")
    print("-" * 60)

    for cat in categories:
        cat_result = analyze_category(cat, pred_category_only)
        h52_result = analyze_category(cat, pred_prob_h52_only)
        best_fn = next(fn for name, fn in approaches if name == best[0])
        best_result = analyze_category(cat, best_fn)

        if cat_result is not None:
            print(f"{cat:20} {cat_result[1]:5} {cat_result[0]*100:9.1f}% {h52_result[0]*100:9.1f}% {best_result[0]*100:11.1f}%")

    # Check if hybrid beats both baselines for "other" specifically
    other_diseases = [(n, s) for n, s in disease_stats.items() if s['category'] == 'other']
    if other_diseases:
        print(f"\n{'='*70}")
        print("'OTHER' CATEGORY DETAILED ANALYSIS")
        print(f"{'='*70}")
        print(f"N = {len(other_diseases)} diseases")

        # Split by hit rate
        high_hit = [s for n, s in other_diseases if s['hit_rate'] > 0.5]
        low_hit = [s for n, s in other_diseases if s['hit_rate'] <= 0.5]

        print(f"  High hit (>50%): {len(high_hit)}, mean h52={np.mean([s['avg_prob_h52'] for s in high_hit]):.2f}")
        print(f"  Low hit (<=50%): {len(low_hit)}, mean h52={np.mean([s['avg_prob_h52'] for s in low_hit]):.2f}")

        # Test targeted approach for 'other'
        print("\nCalibration error for 'other' by approach:")
        for name, fn in approaches:
            err = np.mean([abs(s['hit_rate'] - fn(s)) for n, s in other_diseases])
            print(f"  {name:35} {err*100:.1f} pp")

    # Save results
    save_results = {
        'n_diseases': n_diseases,
        'calibration_results': {name: {'mean': mean, 'std': std} for name, mean, std in results},
        'best_approach': best[0],
        'best_error': best[1],
        'improvement_vs_category': improvement_vs_cat,
        'improvement_vs_h52': improvement_vs_h52,
    }

    output_path = ANALYSIS_DIR / "h139_hybrid_confidence.json"
    with open(output_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Final verdict
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")

    if best[1] < cat_only[1] and best[1] < h52_only[1]:
        print("VALIDATED: Hybrid approach beats both category and per-disease alone")
    elif best[1] < cat_only[1]:
        print("PARTIALLY VALIDATED: Hybrid beats category but not per-disease")
    else:
        print("INVALIDATED: Hybrid does not improve calibration")

    return save_results


if __name__ == "__main__":
    main()
