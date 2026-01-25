#!/usr/bin/env python3
"""
Prioritize novel predictions for Every Cure review.

Enhances existing novel predictions with ATC scores and creates
a tiered priority list.
"""

import json
from pathlib import Path
from typing import Dict, List

# Add parent dir to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.atc_features import ATCMapper


def load_existing_predictions() -> List[Dict]:
    """Load existing novel predictions."""
    with open('data/analysis/novel_predictions.json') as f:
        data = json.load(f)
    return data.get('top_100', [])


def get_atc_score(drug_name: str, disease_name: str, atc_mapper: ATCMapper) -> float:
    """Get ATC mechanism relevance score."""
    return atc_mapper.get_mechanism_score(drug_name, disease_name)


def load_validation_results() -> Dict[str, str]:
    """Load existing validation results."""
    validated = {}

    validation_files = [
        'data/analysis/validated_novel_predictions.json',
        'data/analysis/validation_session_20260124_complete.json',
    ]

    for vf in validation_files:
        if Path(vf).exists():
            with open(vf) as f:
                data = json.load(f)
                for pred in data.get('predictions', data if isinstance(data, list) else []):
                    if isinstance(pred, dict):
                        key = f"{pred.get('drug_name', '')}|{pred.get('disease_name', '')}"
                        status = pred.get('validation_status', pred.get('status', 'unknown'))
                        validated[key] = status

    return validated


def main():
    print("=" * 70)
    print("NOVEL PREDICTION PRIORITIZATION")
    print("=" * 70)

    # Load ATC mapper
    print("\nLoading ATC mapper...")
    atc_mapper = ATCMapper()

    # Load existing predictions
    print("Loading existing novel predictions...")
    predictions = load_existing_predictions()
    print(f"  Loaded {len(predictions)} predictions")

    # Load validation results
    print("Loading prior validation results...")
    validated = load_validation_results()
    print(f"  Found {len(validated)} prior validations")

    # Load FDA approved pairs for filtering
    fda_pairs_file = Path('data/reference/fda_approved_pairs.json')
    fda_pairs = set()
    if fda_pairs_file.exists():
        with open(fda_pairs_file) as f:
            fda_data = json.load(f)
            pairs = fda_data.get('pairs', fda_data)
            if isinstance(pairs, list):
                for pair in pairs:
                    drug = pair.get('drug_name', '').lower()
                    disease = pair.get('disease_name', '').lower()
                    if drug and disease:
                        fda_pairs.add(f"{drug}|{disease}")
            elif isinstance(pairs, dict):
                for drug, diseases in pairs.items():
                    for disease in diseases:
                        if isinstance(disease, str):
                            fda_pairs.add(f"{drug.lower()}|{disease.lower()}")
    print(f"  Loaded {len(fda_pairs)} FDA-approved pairs for filtering")

    # Enhance with ATC scores
    print("\nEnhancing with ATC scores...")
    enhanced = []

    for pred in predictions:
        drug_name = pred.get('drug_name', '')
        disease_name = pred.get('disease_name', '')

        # Get ATC score
        atc_score = get_atc_score(drug_name, disease_name, atc_mapper)
        pred['atc_score'] = atc_score

        # Get ATC codes for reference
        atc_codes = atc_mapper.get_atc_level1(drug_name)
        pred['atc_level1'] = atc_codes

        # Check if previously validated
        key = f"{drug_name}|{disease_name}"
        pred['prior_validation'] = validated.get(key, None)

        # Check if FDA approved (should be filtered as ground truth gap, not novel)
        is_fda = key.lower() in fda_pairs
        pred['is_fda_approved'] = is_fda

        # Calculate enhanced composite score
        base_score = pred.get('score', 0)
        target_overlap = pred.get('target_overlap', 0)

        # New formula: base + target contribution + atc contribution
        enhanced_score = (
            base_score * (1 + 0.01 * min(target_overlap, 10) + 0.05 * atc_score)
        )
        pred['enhanced_score'] = enhanced_score

        # Priority composite (for ranking)
        pred['priority_score'] = (
            enhanced_score +
            0.05 * min(target_overlap, 20) +
            0.1 * atc_score
        )

        enhanced.append(pred)

    # Filter out FDA approved (ground truth gaps)
    truly_novel = [p for p in enhanced if not p.get('is_fda_approved', False)]
    print(f"  Truly novel (not FDA-approved): {len(truly_novel)}")

    # Sort by priority score
    truly_novel.sort(key=lambda x: x.get('priority_score', 0), reverse=True)

    # Create tiered output
    tiers = {
        'tier1_highest': [],  # High score + target overlap + ATC match
        'tier2_high': [],     # High score + one of (target or ATC)
        'tier3_moderate': [], # High score only
    }

    for p in truly_novel:
        target_overlap = p.get('target_overlap', 0)
        atc_score = p.get('atc_score', 0)

        if target_overlap >= 10 and atc_score >= 0.5:
            tiers['tier1_highest'].append(p)
        elif target_overlap >= 5 or atc_score >= 0.5:
            tiers['tier2_high'].append(p)
        else:
            tiers['tier3_moderate'].append(p)

    # Create output
    output = {
        'generated': '2026-01-25',
        'model': 'GB + Target + ATC Boost (39.76% R@30)',
        'methodology': {
            'score_formula': 'base × (1 + 0.01×target_overlap + 0.05×atc_score)',
            'priority_formula': 'enhanced_score + 0.05×target_overlap + 0.1×atc_score',
            'tiers': {
                'tier1': 'target_overlap >= 10 AND atc_score >= 0.5',
                'tier2': 'target_overlap >= 5 OR atc_score >= 0.5',
                'tier3': 'all others with score > 0.9',
            }
        },
        'summary': {
            'total_predictions': len(enhanced),
            'truly_novel': len(truly_novel),
            'tier1_count': len(tiers['tier1_highest']),
            'tier2_count': len(tiers['tier2_high']),
            'tier3_count': len(tiers['tier3_moderate']),
        },
        'tiers': tiers,
    }

    # Save output
    output_path = 'data/analysis/prioritized_novel_predictions.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("PRIORITY SUMMARY")
    print("=" * 70)
    print(f"\nTier 1 (Highest - strong biological signal): {len(tiers['tier1_highest'])}")
    print(f"Tier 2 (High - good signal): {len(tiers['tier2_high'])}")
    print(f"Tier 3 (Moderate - worth investigating): {len(tiers['tier3_moderate'])}")

    # Print top predictions from each tier
    print("\n" + "=" * 70)
    print("TOP 5 FROM EACH TIER")
    print("=" * 70)

    for tier_name, tier_preds in tiers.items():
        print(f"\n--- {tier_name.upper()} ---")
        for i, p in enumerate(tier_preds[:5], 1):
            validated_str = ""
            if p.get('prior_validation'):
                validated_str = f" [VALIDATED: {p['prior_validation']}]"

            print(f"\n{i}. {p['drug_name']} → {p['disease_name']}{validated_str}")
            print(f"   Enhanced score: {p.get('enhanced_score', 0):.3f}")
            print(f"   Target overlap: {p.get('target_overlap', 0)}")
            print(f"   ATC score: {p.get('atc_score', 0):.1f} ({p.get('atc_level1', [])})")
            print(f"   Priority score: {p.get('priority_score', 0):.3f}")

    # Print actionable recommendations
    print("\n" + "=" * 70)
    print("ACTIONABLE RECOMMENDATIONS FOR EVERY CURE")
    print("=" * 70)

    # Get tier 1 predictions that haven't been validated
    unvalidated_tier1 = [p for p in tiers['tier1_highest'] if not p.get('prior_validation')]

    print(f"\n{len(unvalidated_tier1)} Tier 1 predictions need validation:")
    for p in unvalidated_tier1[:10]:
        print(f"  - {p['drug_name']} for {p['disease_name']}")
        print(f"    Score: {p.get('enhanced_score', 0):.3f}, Target overlap: {p.get('target_overlap', 0)}")


if __name__ == '__main__':
    main()
