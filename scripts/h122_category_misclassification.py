#!/usr/bin/env python3
"""h122: Category Misclassification Analysis

Find diseases where actual hit rate deviates from expected tier/category performance.
This could indicate miscategorization or anomalous diseases worth investigating.
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"

# Category tiers from h71
TIER_1_CATEGORIES = {'autoimmune', 'dermatological', 'psychiatric', 'ophthalmic'}
TIER_2_CATEGORIES = {'cardiovascular', 'other', 'cancer'}
TIER_3_CATEGORIES = {'metabolic', 'respiratory', 'gastrointestinal', 'hematological', 'infectious', 'neurological'}

# Disease category keywords (from h106)
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

# Expected precision by tier (from h135/h136 findings)
TIER_EXPECTED_PRECISION = {
    1: 0.25,  # Tier 1 expected ~25% base (up to 58% with filters)
    2: 0.08,  # Tier 2 expected ~8%
    3: 0.05,  # Tier 3 expected ~5%
}


def classify_disease_category(disease_name: str) -> str:
    """Classify disease by name pattern."""
    name_lower = disease_name.lower()

    # Check each category's keywords
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in name_lower for kw in keywords):
            return category

    return 'other'


def get_category_tier(category: str) -> int:
    """Get tier for a category (1=best, 3=worst)."""
    if category in TIER_1_CATEGORIES:
        return 1
    elif category in TIER_2_CATEGORIES:
        return 2
    else:
        return 3


def load_per_disease_results():
    """Load per-disease hit data from h79 confidence analysis."""
    h79_path = ANALYSIS_DIR / "h79_per_disease_confidence.json"
    if h79_path.exists():
        with open(h79_path) as f:
            return json.load(f)
    return None


def main():
    print("=" * 70)
    print("h122: CATEGORY MISCLASSIFICATION ANALYSIS")
    print("=" * 70)

    # Load h79 per-disease confidence data (contains per-disease hit rates)
    h79_data = load_per_disease_results()

    if h79_data is None:
        print("ERROR: h79 data not found")
        return

    # Aggregate across all seeds
    disease_stats = defaultdict(lambda: {'hits': 0, 'total': 0, 'categories': set()})

    for seed_result in h79_data['seed_results']:
        for disease in seed_result['diseases']:
            name = disease['disease_name']
            disease_stats[name]['hits'] += disease['hit']
            disease_stats[name]['total'] += 1
            disease_stats[name]['categories'].add(disease['category'])

    print(f"Loaded {len(disease_stats)} diseases across {len(h79_data['seed_results'])} seeds")

    # Load h136 category results for expected precision by category
    with open(ANALYSIS_DIR / "h136_tier23_category_rescue.json") as f:
        h136_data = json.load(f)

    # Get expected precision per category (from h136)
    category_base_precision = {}
    for cat, results in h136_data['category_results'].items():
        if isinstance(results, dict):
            category_base_precision[cat] = results.get('base_precision', 0) / 100  # Convert to decimal

    # Also add categories not in h136 (Tier 1 categories from h135)
    # These typically have higher precision
    tier1_avg = 0.25  # Tier 1 average from h135
    for cat in TIER_1_CATEGORIES:
        if cat not in category_base_precision:
            category_base_precision[cat] = tier1_avg

    print("\nExpected precision by category:")
    for cat, prec in sorted(category_base_precision.items(), key=lambda x: -x[1]):
        tier = get_category_tier(cat)
        print(f"  {cat}: {prec*100:.1f}% (Tier {tier})")

    # Analyze each disease for misclassification
    misclassifications = []
    all_diseases = []

    for disease_name, stats in disease_stats.items():
        hit_rate = stats['hits'] / stats['total'] if stats['total'] > 0 else 0

        # Get the assigned category (use most common if multiple)
        assigned_cats = list(stats['categories'])
        assigned_cat = assigned_cats[0] if assigned_cats else 'other'

        # Also compute what the keyword classifier would say
        keyword_cat = classify_disease_category(disease_name)

        tier = get_category_tier(assigned_cat)
        expected = category_base_precision.get(assigned_cat, 0.05)

        # Calculate deviation from expected
        deviation = hit_rate - expected

        all_diseases.append({
            'disease': disease_name,
            'assigned_category': assigned_cat,
            'keyword_category': keyword_cat,
            'tier': tier,
            'hit_rate': hit_rate,
            'expected': expected,
            'deviation': deviation,
            'n_seeds': stats['total'],
            'n_hits': stats['hits'],
            'category_mismatch': assigned_cat != keyword_cat,
        })

        # Flag significant deviations
        if abs(deviation) > 0.20:  # >20 pp deviation
            misclassifications.append(all_diseases[-1])

    # Sort by absolute deviation
    misclassifications.sort(key=lambda x: abs(x['deviation']), reverse=True)
    all_diseases.sort(key=lambda x: x['hit_rate'], reverse=True)

    # Print analysis
    print(f"\n{'='*70}")
    print("DISEASES WITH HIT RATE >> EXPECTED")
    print("(Over-performing their category - potential recategorization)")
    print(f"{'='*70}")

    over_performers = [m for m in misclassifications if m['deviation'] > 0.20]
    for m in over_performers[:25]:
        mismatch = " [MISMATCH]" if m['category_mismatch'] else ""
        print(f"\n{m['disease']}:{mismatch}")
        print(f"  Assigned: {m['assigned_category']} (Tier {m['tier']}) | Keyword: {m['keyword_category']}")
        print(f"  Hit rate: {m['hit_rate']*100:.0f}% ({m['n_hits']}/{m['n_seeds']}) vs expected {m['expected']*100:.0f}%")
        print(f"  Deviation: +{m['deviation']*100:.0f} pp")

    print(f"\n{'='*70}")
    print("DISEASES WITH HIT RATE << EXPECTED")
    print("(Under-performing their category - may need filtering)")
    print(f"{'='*70}")

    under_performers = [m for m in misclassifications if m['deviation'] < -0.20]
    for m in under_performers[:15]:
        mismatch = " [MISMATCH]" if m['category_mismatch'] else ""
        print(f"\n{m['disease']}:{mismatch}")
        print(f"  Assigned: {m['assigned_category']} (Tier {m['tier']}) | Keyword: {m['keyword_category']}")
        print(f"  Hit rate: {m['hit_rate']*100:.0f}% ({m['n_hits']}/{m['n_seeds']}) vs expected {m['expected']*100:.0f}%")
        print(f"  Deviation: {m['deviation']*100:.0f} pp")

    # Category mismatch analysis
    print(f"\n{'='*70}")
    print("CATEGORY ASSIGNMENT MISMATCHES")
    print("(Keyword classifier disagrees with h79 category)")
    print(f"{'='*70}")

    mismatches = [d for d in all_diseases if d['category_mismatch']]
    print(f"\n{len(mismatches)} diseases have category mismatches")

    # Group by assigned -> keyword category
    mismatch_patterns = defaultdict(list)
    for m in mismatches:
        key = f"{m['assigned_category']} -> {m['keyword_category']}"
        mismatch_patterns[key].append(m)

    print("\nMismatch patterns (assigned -> keyword):")
    for pattern, diseases in sorted(mismatch_patterns.items(), key=lambda x: -len(x[1])):
        avg_hit = np.mean([d['hit_rate'] for d in diseases])
        print(f"  {pattern}: {len(diseases)} diseases, avg hit rate {avg_hit*100:.0f}%")
        for d in diseases[:3]:
            print(f"    - {d['disease']}: {d['hit_rate']*100:.0f}%")

    # Pattern analysis for over-performers
    print(f"\n{'='*70}")
    print("PATTERN ANALYSIS: OVER-PERFORMERS")
    print(f"{'='*70}")

    over_categories = defaultdict(list)
    for m in over_performers:
        over_categories[m['assigned_category']].append(m)

    print("\nBy assigned category:")
    for cat, diseases in sorted(over_categories.items(), key=lambda x: -len(x[1])):
        tier = get_category_tier(cat)
        avg_hit = np.mean([d['hit_rate'] for d in diseases])
        print(f"  {cat} (Tier {tier}): {len(diseases)} diseases, avg hit {avg_hit*100:.0f}%")

    # Identify diseases that should be recategorized
    print(f"\n{'='*70}")
    print("RECATEGORIZATION RECOMMENDATIONS")
    print(f"{'='*70}")

    recategorize = []
    for d in all_diseases:
        assigned = d['assigned_category']
        assigned_tier = get_category_tier(assigned)
        hit_rate = d['hit_rate']

        # If hit rate suggests a different tier
        if hit_rate >= 0.60 and assigned_tier > 1:  # Should be Tier 1
            recategorize.append({**d, 'recommended_tier': 1, 'reason': 'Very high hit rate (>60%)'})
        elif hit_rate >= 0.40 and assigned_tier == 3:  # Should be Tier 2
            recategorize.append({**d, 'recommended_tier': 2, 'reason': 'High hit rate (>40%) for Tier 3'})
        elif hit_rate <= 0.05 and assigned_tier == 1:  # Should be downgraded
            recategorize.append({**d, 'recommended_tier': 3, 'reason': 'Very low hit rate for Tier 1'})

    print(f"\n{len(recategorize)} diseases recommended for recategorization:")
    for r in recategorize[:20]:
        print(f"\n{r['disease']}:")
        print(f"  Current: {r['assigned_category']} (Tier {r['tier']})")
        print(f"  Recommended: Tier {r['recommended_tier']}")
        print(f"  Reason: {r['reason']} (hit rate: {r['hit_rate']*100:.0f}%)")

    # Save results
    results = {
        'n_diseases_analyzed': len(all_diseases),
        'n_significant_deviations': len(misclassifications),
        'over_performers': over_performers,
        'under_performers': under_performers,
        'category_mismatches': mismatches,
        'recategorization_recommendations': recategorize,
        'category_base_precision': category_base_precision,
    }

    output_path = ANALYSIS_DIR / "h122_misclassification_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total diseases analyzed: {len(all_diseases)}")
    print(f"Significant deviations (>20 pp): {len(misclassifications)}")
    print(f"  Over-performers: {len(over_performers)}")
    print(f"  Under-performers: {len(under_performers)}")
    print(f"Category mismatches: {len(mismatches)}")
    print(f"Recategorization recommendations: {len(recategorize)}")

    # Calculate potential improvement
    if recategorize:
        current_avg = np.mean([r['hit_rate'] for r in recategorize])
        print(f"\nPotential impact:")
        print(f"  Avg hit rate of recategorization candidates: {current_avg*100:.0f}%")

    return results


def run_knn_evaluation():
    """Run kNN evaluation to get per-disease hit rates."""
    import pickle
    import torch

    MODELS_DIR = PROJECT_ROOT / "models"
    EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"

    # Load resources
    print("Loading Node2Vec embeddings...")
    checkpoint = torch.load(MODELS_DIR / "transe.pt", map_location="cpu", weights_only=False)
    embeddings = None
    if "entity_embeddings" in checkpoint:
        embeddings = checkpoint["entity_embeddings"].numpy()
    elif "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
        for key in ["entity_embeddings.weight", "ent_embeddings.weight"]:
            if key in state:
                embeddings = state[key].numpy()
                break
    entity2id = checkpoint.get("entity2id", {})

    # Load Node2Vec embeddings
    node2vec_path = EMBEDDINGS_DIR / "node2vec_embeddings.npy"
    if node2vec_path.exists():
        node2vec = np.load(node2vec_path)
        print(f"Loaded Node2Vec embeddings: {node2vec.shape}")
    else:
        print("Node2Vec embeddings not found")
        return

    # Load ground truth
    import pandas as pd
    gt_df = pd.read_excel(REFERENCE_DIR / "everycure" / "indicationList.xlsx")
    print(f"Ground truth: {len(gt_df)} drug-disease pairs")

    # Load disease mapping
    with open(REFERENCE_DIR / "disease_ontology_mapping.json") as f:
        disease_mapping = json.load(f)

    # Get disease IDs
    disease_ids = [eid for eid in entity2id if eid.startswith("Disease::")]
    drug_ids = [eid for eid in entity2id if eid.startswith("Compound::")]

    print(f"Diseases: {len(disease_ids)}, Drugs: {len(drug_ids)}")

    # Build GT dictionary
    gt_pairs = set()
    for _, row in gt_df.iterrows():
        drug = row.get('drugName', '')
        disease = row.get('diseaseName', '')
        if drug and disease:
            gt_pairs.add((str(drug).lower(), str(disease).lower()))

    print(f"GT pairs: {len(gt_pairs)}")

    # This is getting complex - let's use existing evaluation results instead
    # Check if we have per-disease results from a recent evaluation

    # Look for recent kNN evaluation outputs
    eval_files = list(ANALYSIS_DIR.glob("*knn*.json"))
    print(f"Found {len(eval_files)} kNN-related files")

    for f in eval_files[:5]:
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
