#!/usr/bin/env python3
"""
Prepare a validation batch of novel predictions.
Excludes already-validated, standard-of-care, and false positive patterns.
"""

import json
import re
from pathlib import Path

# Known FDA-approved drugs (recognizable names, not chemical codes)
# Using a heuristic: capitalize first letter, reasonable length, no long systematic names
def is_likely_fda_drug(name: str) -> bool:
    # Exclude patterns
    if name.startswith('CHEMBL'): return False
    if name.startswith('MESH:'): return False
    if name.startswith('CHEBI:'): return False
    if name.startswith('DB0') or name.startswith('DB1'): return False  # DrugBank codes
    if re.match(r'^[A-Z0-9\-]+$', name) and len(name) > 10: return False
    if len(name) > 60: return False  # Long chemical names
    if 'AMINO' in name.upper() and 'ACID' in name.upper(): return False
    if name.count('-') > 3 and len(name) > 30: return False
    if 'YL)' in name or 'INE-' in name.upper() and len(name) > 40: return False
    if name.startswith('('): return False
    if name.isupper() and len(name) > 15: return False  # Long all-caps codes
    if re.match(r'^\d', name): return False  # Starts with number
    if 'INHIBITOR' in name.upper(): return False  # Research compound
    if 'BIIF' in name: return False  # Research codes
    return True

# Already validated drugs (from previous sessions)
ALREADY_VALIDATED = {
    'Pitavastatin', 'Estradiol', 'Treprostinil', 'Paclitaxel',
    'Idarubicin', 'Betamethasone', 'Dantrolene', 'Empagliflozin',
    'Lecanemab', 'Tezepelumab', 'Rivastigmine', 'Atezolizumab',
    'Lidocaine', 'Formoterol', 'Thiamine', 'Corticotropin',
}

# Known false positive drug types for specific diseases
FALSE_POSITIVE_RULES = [
    ('antibiotics', 'diabetes'),
    ('antibiotics', 'metabolic'),
    ('sympathomimetics', 'diabetes'),
    ('alpha_blocker', 'heart failure'),
    ('tca', 'hypertension'),
    ('ppi', 'hypertension'),
    ('chemo', 'diabetes'),
    ('diagnostic', 'any'),
]

def main() -> None:
    data_dir = Path(__file__).parent.parent / "data" / "analysis"

    # Load filtered predictions
    with open(data_dir / "filtered_predictions.json") as f:
        filtered = json.load(f)

    # Load novel predictions for additional data
    with open(data_dir / "novel_predictions.json") as f:
        novel = json.load(f)

    # Create lookup for novel predictions by drug+disease
    novel_lookup = {}
    for pred in novel.get('top_100', []) + novel.get('high_confidence_predictions', []):
        key = (pred['drug_name'], pred['disease_name'])
        novel_lookup[key] = pred

    # Filter clean predictions
    candidates = []
    for pred in filtered['clean_predictions']:
        drug = pred['drug']
        disease = pred['disease']

        # Skip if already validated
        if drug in ALREADY_VALIDATED:
            continue

        # Skip if not a recognizable FDA drug
        if not is_likely_fda_drug(drug):
            continue

        # Get additional data from novel predictions if available
        extra = novel_lookup.get((drug, disease), {})

        candidates.append({
            'drug': drug,
            'disease': disease,
            'score': pred['score'],
            'confidence': pred['confidence'],
            'drug_type': pred.get('drug_type', 'unknown'),
            'target_overlap': extra.get('target_overlap', 0),
            'rank_for_disease': extra.get('rank_for_disease', 0),
        })

    # Sort by score
    candidates.sort(key=lambda x: -x['score'])

    print("=" * 70)
    print("VALIDATION CANDIDATES (filtered, not yet validated)")
    print("=" * 70)
    print(f"\nTotal candidates: {len(candidates)}")

    # Categorize by confidence
    high = [c for c in candidates if c['confidence'] == 'high']
    medium = [c for c in candidates if c['confidence'] == 'medium']

    print(f"High confidence: {len(high)}")
    print(f"Medium confidence: {len(medium)}")

    # Show high confidence candidates
    print("\n" + "=" * 70)
    print("HIGH CONFIDENCE - READY FOR VALIDATION")
    print("=" * 70)

    for i, pred in enumerate(high[:20], 1):
        print(f"\n{i}. {pred['drug']} → {pred['disease']}")
        print(f"   Score: {pred['score']:.3f} | Type: {pred['drug_type']}")
        if pred['target_overlap']:
            print(f"   Target overlap: {pred['target_overlap']}")

    # Show top medium confidence candidates
    print("\n" + "=" * 70)
    print("MEDIUM CONFIDENCE - TOP 30 BY SCORE")
    print("=" * 70)

    for i, pred in enumerate(medium[:30], 1):
        print(f"\n{i}. {pred['drug']} → {pred['disease']}")
        print(f"   Score: {pred['score']:.3f} | Type: {pred['drug_type']}")

    # Group by disease for interesting analysis
    by_disease: dict = {}
    for pred in candidates[:100]:
        d = pred['disease']
        if d not in by_disease:
            by_disease[d] = []
        by_disease[d].append(pred)

    print("\n" + "=" * 70)
    print("DISEASES WITH MULTIPLE CANDIDATES")
    print("=" * 70)

    for disease, preds in sorted(by_disease.items(), key=lambda x: -len(x[1])):
        if len(preds) >= 3:
            print(f"\n{disease} ({len(preds)} candidates):")
            for pred in preds[:5]:
                print(f"  - {pred['drug']} (score: {pred['score']:.3f})")

    # Save validation batch
    validation_batch = {
        'date': '2026-01-24',
        'methodology': 'Filtered novel predictions for validation',
        'filters_applied': [
            'Excluded already-validated drugs',
            'Excluded chemical compound codes',
            'Excluded false positive patterns',
            'Confidence filter applied',
        ],
        'high_confidence_candidates': high[:20],
        'medium_confidence_top30': medium[:30],
        'all_candidates': candidates,
        'summary': {
            'total_candidates': len(candidates),
            'high_confidence': len(high),
            'medium_confidence': len(medium),
        }
    }

    output_path = data_dir / "validation_batch_next.json"
    with open(output_path, 'w') as f:
        json.dump(validation_batch, f, indent=2)

    print(f"\n\nSaved to: {output_path}")

    # Priority list for immediate validation
    print("\n" + "=" * 70)
    print("PRIORITY VALIDATION LIST (Top 15)")
    print("=" * 70)
    print("\nThese are high-potential, FDA-drug predictions ready for literature validation:")

    priority = high[:10] + medium[:5]
    for i, pred in enumerate(priority, 1):
        print(f"\n{i}. {pred['drug']} → {pred['disease']}")
        print(f"   Score: {pred['score']:.3f} | Confidence: {pred['confidence']}")


if __name__ == "__main__":
    main()
