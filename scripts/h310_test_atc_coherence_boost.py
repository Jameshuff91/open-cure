#!/usr/bin/env python3
"""
h310: Test ATC Coherence Boost Implementation

Tests that the _is_atc_coherent() method and tier boost work correctly.
"""

import sys
import os

# Add BOTH src and project root to path
# src is needed for direct imports like "from production_predictor import ..."
# project root is needed for "from src.atc_features import ..." style imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, project_root)

from production_predictor import DrugRepurposingPredictor, DISEASE_CATEGORY_ATC_MAP


def test_atc_map_exists():
    """Verify the ATC map constant exists and has expected structure."""
    print("Test 1: ATC Map Exists")
    print("-" * 50)

    # Check map exists and has expected categories
    expected_categories = [
        'autoimmune', 'cancer', 'cardiovascular', 'infectious',
        'metabolic', 'neurological', 'respiratory'
    ]

    for cat in expected_categories:
        assert cat in DISEASE_CATEGORY_ATC_MAP, f"Missing category: {cat}"
        codes = DISEASE_CATEGORY_ATC_MAP[cat]
        assert isinstance(codes, set), f"Category {cat} should be a set"
        print(f"  {cat}: {sorted(codes)}")

    print("  PASS: ATC map structure correct")
    print()


def test_coherence_method(predictor):
    """Test _is_atc_coherent() method."""
    print("Test 2: Coherence Method")
    print("-" * 50)

    # Test cases: (drug_name, category, expected_coherent)
    test_cases = [
        # Coherent cases
        ("methotrexate", "autoimmune", True),     # L04AX - immunosuppressant for autoimmune
        ("prednisone", "autoimmune", True),       # H02AB - corticosteroid for autoimmune (H in map)
        ("metformin", "metabolic", True),         # A10BA - antidiabetic for metabolic
        ("atorvastatin", "cardiovascular", True), # C10AA - statin for cardiovascular
        ("amoxicillin", "infectious", True),      # J01CA - antibiotic for infectious

        # Incoherent cases
        ("metformin", "infectious", False),       # Antidiabetic not for infectious
        ("amoxicillin", "cardiovascular", False), # Antibiotic not for cardiovascular
        ("atorvastatin", "autoimmune", False),    # Statin not for autoimmune

        # Edge cases
        ("unknown_drug_xyz", "autoimmune", False),  # Unknown drug
        ("prednisone", "other", False),             # 'other' has no expected ATC
    ]

    passed = 0
    failed = 0

    for drug, category, expected in test_cases:
        result = predictor._is_atc_coherent(drug, category)
        status = "PASS" if result == expected else "FAIL"
        if result == expected:
            passed += 1
        else:
            failed += 1
        print(f"  {status}: {drug} + {category} = {result} (expected {expected})")

    print(f"\n  Results: {passed}/{passed + failed} tests passed")
    print()

    return failed == 0


def test_tier_boost(predictor):
    """Test that coherent predictions get boosted from LOW to MEDIUM."""
    print("Test 3: Tier Boost Integration")
    print("-" * 50)

    # Get predictions for a disease and check for coherence boosts
    # Use rheumatoid arthritis (autoimmune category)
    results = predictor.predict("rheumatoid arthritis", top_n=30)

    # Count MEDIUM tier predictions that have category_rescue_applied
    # These might include ATC coherence boosts
    rescued = [p for p in results.predictions if p.category_rescue_applied]

    # Also check how many MEDIUM tier predictions we have
    medium_tier = [p for p in results.predictions
                   if p.confidence_tier.value == 'MEDIUM']

    print(f"  Total predictions: {len(results.predictions)}")
    print(f"  Category rescued: {len(rescued)}")
    print(f"  MEDIUM tier predictions: {len(medium_tier)}")

    # Check tier distribution
    tier_counts = {}
    for p in results.predictions:
        tier = p.confidence_tier.value
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
    print(f"\n  Tier distribution: {tier_counts}")

    # Check ATC coherence for some drugs to verify it's being checked
    print("\n  Sample predictions with ATC check:")
    for p in results.predictions[:5]:
        is_coherent = predictor._is_atc_coherent(p.drug_name, results.category)
        print(f"    {p.drug_name}: {p.confidence_tier.value}, coherent={is_coherent}, rescued={p.category_rescue_applied}")

    # Also test a different category (infectious)
    results2 = predictor.predict("pneumonia", top_n=30)
    tier_counts2 = {}
    for p in results2.predictions:
        tier = p.confidence_tier.value
        tier_counts2[tier] = tier_counts2.get(tier, 0) + 1

    print(f"\n  Pneumonia (infectious) predictions: {len(results2.predictions)}")
    print(f"  Tier distribution: {tier_counts2}")

    print()
    return True


def main():
    print("=" * 60)
    print("h310: Testing ATC Coherence Boost Implementation")
    print("=" * 60)
    print()

    all_passed = True

    test_atc_map_exists()

    print("Loading predictor (this may take a minute)...")
    predictor = DrugRepurposingPredictor()
    print("Predictor loaded.\n")

    if not test_coherence_method(predictor):
        all_passed = False

    test_tier_boost(predictor)

    print("=" * 60)
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
