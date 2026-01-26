#!/usr/bin/env python3
"""
Run extended validation on 1000+ predictions.

Uses the integrated confounding detection and caches results.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from external_validation import ExternalValidator, VALIDATION_DIR
from confounding_detector import detect_confounding, ConfoundingType

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def main():
    print("=" * 70)
    print("EXTENDED VALIDATION PIPELINE")
    print("=" * 70)

    # Load predictions
    pred_file = DATA_DIR / "analysis" / "novel_predictions_with_confidence.json"
    print(f"\n1. Loading predictions from {pred_file}")

    with open(pred_file) as f:
        predictions = json.load(f)

    print(f"   Total predictions available: {len(predictions)}")

    # Convert format for validator
    formatted_preds = []
    for p in predictions:
        formatted_preds.append({
            "drug_name": p["drug"],
            "disease_name": p["disease"],
            "drugbank_id": "",  # Not available in this format
            "mesh_id": "",
            "score": p.get("boosted_score", p.get("base_score", 0)),
        })

    # Initialize validator
    print("\n2. Initializing validator...")
    validator = ExternalValidator()
    print(f"   Cached entries: {len(validator.cache)}")

    # Calculate how many new validations needed
    cached_keys = set(validator.cache.keys())
    new_preds = []
    for p in formatted_preds:
        key = f"{p['drug_name'].lower()}|{p['disease_name'].lower()}"
        if key not in cached_keys:
            new_preds.append(p)

    print(f"   New validations needed: {len(new_preds)}")

    # Target: validate up to 1000 total (adding ~430 more)
    target_new = 500  # Validate 500 new predictions
    print(f"   Will validate: {min(target_new, len(new_preds))} new predictions")

    # Validate new predictions
    print("\n3. Validating new predictions...")
    print("   (This will take a while due to API rate limits)")

    results = validator.validate_predictions(
        new_preds[:target_new],
        save_every=25
    )

    # Now load ALL cached validations and generate comprehensive report
    print("\n4. Generating comprehensive report...")

    # Reload all predictions with validation data
    all_results = []
    for p in formatted_preds[:2000]:  # Top 2000 predictions
        result = validator.validate_prediction(
            drug_name=p["drug_name"],
            disease_name=p["disease_name"],
            drugbank_id=p.get("drugbank_id", ""),
            mesh_id=p.get("mesh_id", ""),
            model_score=p.get("score", 0),
            use_cache=True
        )
        all_results.append(result)

    # Generate report
    report = validator.generate_report(all_results)

    # Save extended report
    report_file = VALIDATION_DIR / "extended_validation_report_1k.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"   Saved to {report_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("EXTENDED VALIDATION SUMMARY")
    print("=" * 70)

    summary = report["summary"]
    print(f"\nTotal validated: {summary['total_validated']}")
    print(f"With clinical trials: {summary['with_clinical_trials']} ({100*summary['with_clinical_trials']/summary['total_validated']:.1f}%)")
    print(f"With publications: {summary['with_publications']} ({100*summary['with_publications']/summary['total_validated']:.1f}%)")

    print(f"\nEvidence categories:")
    print(f"  Strong (≥0.5):      {summary['strong_evidence']} ({100*summary['strong_evidence']/summary['total_validated']:.1f}%)")
    print(f"  Moderate (0.2-0.5): {summary['moderate_evidence']} ({100*summary['moderate_evidence']/summary['total_validated']:.1f}%)")
    print(f"  Weak (<0.2):        {summary['weak_evidence']} ({100*summary['weak_evidence']/summary['total_validated']:.1f}%)")
    print(f"  None (0):           {summary['no_evidence']} ({100*summary['no_evidence']/summary['total_validated']:.1f}%)")

    print(f"\nConfounding detection:")
    print(f"  Confounded predictions: {summary.get('confounded_predictions', 0)}")
    print(f"  High-confidence confounded: {summary.get('high_confidence_confounded', 0)}")

    # Show confounded predictions
    if report.get("confounded_predictions"):
        print("\n" + "-" * 70)
        print("CONFOUNDED PREDICTIONS (flagged as potential false positives)")
        print("-" * 70)
        for cp in report["confounded_predictions"][:10]:
            print(f"  ❌ {cp['drug']} → {cp['disease']}")
            print(f"     Type: {cp['confounding_type']}, Confidence: {cp['confidence']:.0%}")
            print(f"     Reason: {cp['reason']}")

    print("\n" + "=" * 70)
    print("Done!")


if __name__ == "__main__":
    main()
