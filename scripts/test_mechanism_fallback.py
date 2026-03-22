"""Test h900: mechanism-only fallback for rare diseases with zero kNN coverage."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

from src.production_predictor import DrugRepurposingPredictor

def main() -> None:
    print("Loading predictor...")
    predictor = DrugRepurposingPredictor()
    print(f"Loaded. {len(predictor.drug_targets)} drugs with targets, "
          f"{len(predictor.disease_genes)} diseases with genes.")

    # Find diseases that have disease_genes but might lack kNN coverage
    # Test with a few known rare diseases
    test_diseases = [
        "Niemann-Pick disease",
        "Gaucher disease",
        "Fabry disease",
        "Huntington disease",
        "amyotrophic lateral sclerosis",
        "Wilson disease",
        "sickle cell disease",
        "cystic fibrosis",
    ]

    for disease in test_diseases:
        print(f"\n{'='*60}")
        print(f"Disease: {disease}")
        result = predictor.predict(disease, top_n=10, include_filtered=True)

        if result.coverage_warning:
            print(f"  WARNING: {result.coverage_warning}")

        if not result.predictions:
            print("  No predictions generated.")
            continue

        print(f"  {len(result.predictions)} predictions:")
        for pred in result.predictions[:5]:
            print(f"    #{pred.rank}: {pred.drug_name} | "
                  f"tier={pred.confidence_tier.value} | "
                  f"mechanism={pred.mechanism_support} | "
                  f"norm_score={pred.norm_score:.3f} | "
                  f"sub_reason={pred.category_specific_tier} | "
                  f"lit={pred.literature_evidence_level}")

        # Count by tier
        tier_counts: dict[str, int] = {}
        for pred in result.predictions:
            t = pred.confidence_tier.value
            tier_counts[t] = tier_counts.get(t, 0) + 1
        print(f"  Tier distribution: {tier_counts}")

        # Check if fallback was used
        fallback_count = sum(
            1 for p in result.predictions
            if p.category_specific_tier and 'mechanism' in p.category_specific_tier
        )
        if fallback_count > 0:
            print(f"  >>> MECHANISM FALLBACK ACTIVE: {fallback_count} predictions")


if __name__ == '__main__':
    main()
