#!/usr/bin/env python3
"""
Filter the 400 high-confidence novel predictions using the confidence filter.
Outputs statistics and a filtered list for validation.
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from confidence_filter import (
    filter_predictions,
    filter_prediction,
    print_filter_report,
    ConfidenceLevel,
)


def main() -> None:
    data_dir = Path(__file__).parent.parent / "data" / "analysis"

    # Load novel predictions
    with open(data_dir / "novel_predictions.json") as f:
        data = json.load(f)

    high_confidence = data.get("high_confidence_predictions", [])
    print(f"Loaded {len(high_confidence)} high-confidence predictions\n")

    # Convert to filter format
    predictions = []
    for pred in high_confidence:
        predictions.append({
            "drug": pred["drug_name"],
            "disease": pred["disease_name"],
            "score": pred["score"],
            "target_overlap": pred.get("target_overlap", 0),
            "drugbank_id": pred.get("drugbank_id", ""),
            "mesh_id": pred.get("mesh_id", ""),
            "base_score": pred.get("base_score", 0),
            "rank_for_disease": pred.get("rank_for_disease", 0),
        })

    # Run filter
    filtered, stats = filter_predictions(predictions)
    print_filter_report(stats)

    # Show excluded predictions with details
    print("\n" + "=" * 70)
    print("EXCLUDED PREDICTIONS (False Positives)")
    print("=" * 70)

    excluded = []
    for pred in predictions:
        result = filter_prediction(pred["drug"], pred["disease"], pred["score"])
        if result.confidence == ConfidenceLevel.EXCLUDED:
            excluded.append({
                "drug": pred["drug"],
                "disease": pred["disease"],
                "score": pred["score"],
                "target_overlap": pred.get("target_overlap", 0),
                "drug_type": result.drug_type,
                "reason": result.reason,
            })
            print(f"\n  ❌ {pred['drug']} → {pred['disease']}")
            print(f"     Score: {pred['score']:.3f}, Overlap: {pred.get('target_overlap', 0)}")
            print(f"     Drug type: {result.drug_type}")
            print(f"     Reason: {result.reason}")

    # Show high-confidence predictions for validation
    print("\n" + "=" * 70)
    print("HIGH CONFIDENCE PREDICTIONS (For Validation)")
    print("=" * 70)

    high_conf_for_validation = []
    for result in filtered:
        if result.confidence == ConfidenceLevel.HIGH:
            orig = next((p for p in predictions if p["drug"] == result.drug and p["disease"] == result.disease), {})
            high_conf_for_validation.append({
                "drug": result.drug,
                "disease": result.disease,
                "score": result.original_score,
                "target_overlap": orig.get("target_overlap", 0),
                "drug_type": result.drug_type,
                "drugbank_id": orig.get("drugbank_id", ""),
                "mesh_id": orig.get("mesh_id", ""),
                "rank_for_disease": orig.get("rank_for_disease", 0),
            })
            print(f"\n  ✓ {result.drug} → {result.disease}")
            print(f"     Score: {result.original_score:.3f}, Overlap: {orig.get('target_overlap', 0)}")
            print(f"     Drug type: {result.drug_type}")

    # Show medium-confidence predictions (sorted by score)
    print("\n" + "=" * 70)
    print("MEDIUM CONFIDENCE PREDICTIONS (Top 50 by Score)")
    print("=" * 70)

    medium_conf = []
    for result in filtered:
        if result.confidence == ConfidenceLevel.MEDIUM:
            orig = next((p for p in predictions if p["drug"] == result.drug and p["disease"] == result.disease), {})
            medium_conf.append({
                "drug": result.drug,
                "disease": result.disease,
                "score": result.original_score,
                "target_overlap": orig.get("target_overlap", 0),
                "drug_type": result.drug_type,
                "drugbank_id": orig.get("drugbank_id", ""),
                "mesh_id": orig.get("mesh_id", ""),
                "rank_for_disease": orig.get("rank_for_disease", 0),
            })

    # Sort by score and show top 50
    medium_conf.sort(key=lambda x: -x["score"])
    for pred in medium_conf[:50]:
        print(f"\n  • {pred['drug']} → {pred['disease']}")
        print(f"     Score: {pred['score']:.3f}, Overlap: {pred['target_overlap']}, Type: {pred['drug_type']}")

    # Save filtered results
    output = {
        "summary": {
            "total_input": len(predictions),
            "excluded": stats["excluded"],
            "high_confidence": stats["high_confidence"],
            "medium_confidence": stats["medium_confidence"],
            "low_confidence": stats["low_confidence"],
            "exclusion_reasons": stats["exclusion_reasons"],
        },
        "excluded_predictions": excluded,
        "high_confidence_for_validation": high_conf_for_validation,
        "medium_confidence_top50": medium_conf[:50],
        "all_filtered_predictions": [
            {
                "drug": r.drug,
                "disease": r.disease,
                "score": r.original_score,
                "confidence": r.confidence.value,
                "drug_type": r.drug_type,
                "reason": r.reason,
            }
            for r in filtered
        ],
    }

    output_path = data_dir / "filtered_high_confidence.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n\nResults saved to: {output_path}")

    # Summary for next steps
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print(f"\n1. Excluded {stats['excluded']} obvious false positives")
    print(f"2. {stats['high_confidence']} high-confidence predictions ready for validation")
    print(f"3. {stats['medium_confidence']} medium-confidence predictions available")
    print(f"\nRecommend validating the {stats['high_confidence']} high-confidence predictions first,")
    print("then sampling from top medium-confidence predictions.")


if __name__ == "__main__":
    main()
