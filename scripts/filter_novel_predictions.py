#!/usr/bin/env python3
"""
Filter novel predictions using both ML confidence and rule-based exclusions.

Combines:
1. ML confidence calibrator (predicts probability of being in top-30)
2. Rule-based filter (excludes known harmful patterns)
"""

import json
import sys
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from confidence_filter import (
    filter_prediction,
    ConfidenceLevel,
)

PROJECT_ROOT = Path(__file__).parent.parent
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"


def main() -> None:
    print("=" * 70)
    print("FILTERING NOVEL PREDICTIONS")
    print("=" * 70)

    # Load novel predictions with ML confidence
    input_path = ANALYSIS_DIR / "novel_predictions_with_confidence.json"
    with open(input_path) as f:
        predictions = json.load(f)

    print(f"\nLoaded {len(predictions)} predictions with ML confidence ≥ 0.7")

    # Apply rule-based filter
    print("\nApplying rule-based filter...")

    filtered = []
    excluded = []
    exclusion_reasons: Dict[str, int] = {}

    for pred in predictions:
        result = filter_prediction(
            drug=pred['drug'],
            disease=pred['disease'],
            score=pred['boosted_score'],
        )

        if result.confidence == ConfidenceLevel.EXCLUDED:
            excluded.append({
                **pred,
                'exclusion_reason': result.reason,
            })
            exclusion_reasons[result.reason] = exclusion_reasons.get(result.reason, 0) + 1
        else:
            # Combine ML confidence with rule-based adjustment
            combined_score = pred['confidence'] * (result.adjusted_score / result.original_score if result.original_score > 0 else 1.0)
            filtered.append({
                **pred,
                'filter_confidence': result.confidence.value,
                'filter_reason': result.reason,
                'combined_confidence': combined_score,
                'drug_type_filter': result.drug_type,
            })

    print(f"\nFiltered: {len(filtered)} predictions passed")
    print(f"Excluded: {len(excluded)} predictions removed")

    # Print exclusion reasons
    print("\n" + "=" * 70)
    print("EXCLUSION REASONS")
    print("=" * 70)
    for reason, count in sorted(exclusion_reasons.items(), key=lambda x: -x[1]):
        print(f"  {count:5d} - {reason}")

    # Sort filtered by combined confidence
    filtered.sort(key=lambda x: x['combined_confidence'], reverse=True)

    # Print top filtered predictions
    print("\n" + "=" * 70)
    print("TOP 30 FILTERED PREDICTIONS (rule-based filter applied)")
    print("=" * 70)

    print(f"\n{'Drug':<25} {'Disease':<25} {'ML Conf':>8} {'Combined':>10} {'Type':<15}")
    print("-" * 90)

    for pred in filtered[:30]:
        print(f"{pred['drug'][:23]:<25} {pred['disease'][:23]:<25} {pred['confidence']:>7.2f} {pred['combined_confidence']:>9.2f} {pred['drug_type_filter']:<15}")

    # Stats by filter confidence
    print("\n" + "=" * 70)
    print("FILTER CONFIDENCE DISTRIBUTION")
    print("=" * 70)

    conf_counts = {}
    for pred in filtered:
        fc = pred['filter_confidence']
        conf_counts[fc] = conf_counts.get(fc, 0) + 1

    for fc, count in sorted(conf_counts.items()):
        print(f"  {fc}: {count}")

    # High-confidence biologic predictions (best candidates)
    print("\n" + "=" * 70)
    print("HIGH-CONFIDENCE BIOLOGIC PREDICTIONS (best candidates)")
    print("=" * 70)

    biologics = [p for p in filtered if p['drug_type_filter'] == 'biologic' and p['confidence'] >= 0.8]
    biologics.sort(key=lambda x: x['combined_confidence'], reverse=True)

    print(f"\nFound {len(biologics)} high-confidence biologic predictions")
    print(f"\n{'Drug':<30} {'Disease':<30} {'Conf':>6} {'Overlap':>8}")
    print("-" * 80)

    for pred in biologics[:20]:
        print(f"{pred['drug'][:28]:<30} {pred['disease'][:28]:<30} {pred['confidence']:>5.2f} {pred['target_overlap']:>8}")

    # High-confidence small molecules for cancer
    print("\n" + "=" * 70)
    print("HIGH-CONFIDENCE SMALL MOLECULES FOR CANCER")
    print("=" * 70)

    cancer_preds = [
        p for p in filtered
        if p['disease_category'] == 'cancer'
        and p['filter_confidence'] == 'high'
        and p['confidence'] >= 0.85
    ]
    cancer_preds.sort(key=lambda x: x['combined_confidence'], reverse=True)

    print(f"\nFound {len(cancer_preds)} high-confidence cancer predictions")
    print(f"\n{'Drug':<30} {'Disease':<30} {'Conf':>6} {'Overlap':>8}")
    print("-" * 80)

    for pred in cancer_preds[:20]:
        print(f"{pred['drug'][:28]:<30} {pred['disease'][:28]:<30} {pred['confidence']:>5.2f} {pred['target_overlap']:>8}")

    # Save filtered predictions
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    # Save all filtered
    output_path = ANALYSIS_DIR / "filtered_novel_predictions.json"
    with open(output_path, 'w') as f:
        json.dump(filtered, f, indent=2)
    print(f"\nAll filtered: {output_path}")

    # Save excluded for review
    excluded_path = ANALYSIS_DIR / "excluded_predictions.json"
    with open(excluded_path, 'w') as f:
        json.dump(excluded, f, indent=2)
    print(f"Excluded: {excluded_path}")

    # Save top candidates for validation
    top_candidates = [
        p for p in filtered
        if p['combined_confidence'] >= 0.7 and p['filter_confidence'] in ['high', 'medium']
    ][:100]

    top_path = ANALYSIS_DIR / "top_candidates_for_validation.json"
    with open(top_path, 'w') as f:
        json.dump(top_candidates, f, indent=2)
    print(f"Top 100 candidates: {top_path}")

    # Summary stats
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nTotal input predictions:     {len(predictions)}")
    print(f"Excluded (harmful patterns): {len(excluded)}")
    print(f"Passed filter:               {len(filtered)}")
    print(f"High-confidence biologics:   {len(biologics)}")
    print(f"Top validation candidates:   {len(top_candidates)}")


if __name__ == "__main__":
    main()
