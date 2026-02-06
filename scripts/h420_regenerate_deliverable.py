#!/usr/bin/env python3
"""
h420: Regenerate production deliverable with current tier rules.

The existing deliverable has 58% stale categories (h349).
After h395, h396, h398, h399, h415 modified tier rules,
the deliverable needs regeneration.
"""

import json
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.production_predictor import DrugRepurposingPredictor, ConfidenceTier, classify_literature_status

try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False
    print("WARNING: openpyxl not installed, will save as CSV instead")


def load_self_referential_data() -> Dict[str, Dict]:
    """Load h504 self-referential analysis data.

    Returns dict of disease_id -> {self_only_pct, n_gt, therapeutic_island}.
    """
    sr_path = Path(__file__).parent.parent / "data" / "analysis" / "h504_self_referential.json"
    if not sr_path.exists():
        print(f"WARNING: {sr_path} not found, self-referential annotations will be empty")
        return {}

    with open(sr_path) as f:
        sr_data = json.load(f)

    result: Dict[str, Dict] = {}
    for entry in sr_data:
        disease_id = entry["disease_id"]
        n_gt = entry["n_gt"]
        self_only_pct = entry["self_only_pct"]
        # h517: Therapeutic island = GT>5 and 100% self-referential
        therapeutic_island = n_gt > 5 and self_only_pct == 100.0
        result[disease_id] = {
            "self_referential_pct": self_only_pct,
            "therapeutic_island": therapeutic_island,
        }

    n_islands = sum(1 for v in result.values() if v["therapeutic_island"])
    print(f"Loaded self-referential data: {len(result)} diseases, {n_islands} therapeutic islands")
    return result


def main():
    start = time.time()
    print("=" * 70)
    print("h420: Regenerate Production Deliverable")
    print("=" * 70)

    predictor = DrugRepurposingPredictor()
    gt_data = predictor.ground_truth

    # h517: Load self-referential annotations
    self_ref_data = load_self_referential_data()

    # Get all diseases with embeddings
    all_diseases = [d for d in predictor.embeddings if d in predictor.disease_names]
    print(f"Diseases with embeddings: {len(all_diseases)}")

    # Generate predictions for all diseases
    all_predictions = []
    tier_counts = defaultdict(int)
    category_counts = defaultdict(int)

    for idx, disease_id in enumerate(all_diseases):
        disease_name = predictor.disease_names.get(disease_id, disease_id)

        try:
            result = predictor.predict(disease_name, top_n=30, include_filtered=True)
        except Exception as e:
            continue

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - start
            print(f"  Processed {idx+1}/{len(all_diseases)} diseases... ({elapsed:.0f}s)")

        for pred in result.predictions:
            is_gt = disease_id in gt_data and pred.drug_id in gt_data[disease_id]

            # h481: Classify literature status
            lit_status, soc_class = classify_literature_status(
                pred.drug_name, disease_name, result.category, is_gt
            )

            # h517: Self-referential annotations
            sr_info = self_ref_data.get(disease_id, {})
            self_ref_pct = sr_info.get("self_referential_pct", "")
            therapeutic_island = sr_info.get("therapeutic_island", False)

            all_predictions.append({
                'disease_name': disease_name,
                'disease_id': disease_id,
                'drug_name': pred.drug_name,
                'drug_id': pred.drug_id,
                'rank': pred.rank,
                'knn_score': round(pred.knn_score, 6),
                'normalized_score': round(pred.norm_score, 4),
                'confidence_tier': pred.confidence_tier.value,
                'tier_rule': pred.category_specific_tier or 'standard',
                'category': result.category,
                'disease_tier': result.disease_tier,
                'train_frequency': pred.train_frequency,
                'mechanism_support': pred.mechanism_support,
                'has_targets': pred.has_targets,
                'is_known_indication': is_gt,
                'rescue_applied': pred.category_rescue_applied,
                'transe_consilience': pred.transe_consilience,
                'rank_bucket_precision': pred.rank_bucket_precision,
                'category_holdout_precision': pred.category_holdout_precision,
                'literature_status': lit_status,
                'soc_drug_class': soc_class or '',
                'self_referential_pct': self_ref_pct,
                'therapeutic_island': therapeutic_island,
            })

            tier_counts[pred.confidence_tier.value] += 1
            category_counts[result.category] += 1

    elapsed = time.time() - start
    print(f"\nGenerated {len(all_predictions)} predictions in {elapsed:.0f}s")

    # Print tier distribution
    print(f"\n{'Tier':<10} {'Count':>8} {'Pct':>7}")
    print("-" * 30)
    for tier in ['GOLDEN', 'HIGH', 'MEDIUM', 'LOW', 'FILTER']:
        count = tier_counts[tier]
        pct = 100 * count / len(all_predictions) if all_predictions else 0
        print(f"{tier:<10} {count:>8} {pct:>6.1f}%")

    print(f"\nTop categories:")
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        pct = 100 * count / len(all_predictions)
        print(f"  {cat}: {count} ({pct:.1f}%)")

    # Save to file
    output_dir = Path(__file__).parent.parent / "data" / "deliverables"
    output_dir.mkdir(parents=True, exist_ok=True)

    if HAS_OPENPYXL:
        output_path = output_dir / "drug_repurposing_predictions_with_confidence.xlsx"
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Predictions"

        # Header
        headers = list(all_predictions[0].keys())
        ws.append(headers)

        # Data
        for pred in all_predictions:
            ws.append([pred[h] for h in headers])

        # Format
        for col in ws.columns:
            max_length = max(len(str(cell.value or "")) for cell in col)
            ws.column_dimensions[col[0].column_letter].width = min(max_length + 2, 40)

        wb.save(str(output_path))
        print(f"\nSaved to {output_path}")

        # Also save JSON for programmatic use
        import json

        def _json_safe(obj: object) -> object:
            """Convert numpy types to Python native for JSON serialization."""
            import numpy as np
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            return obj

        json_preds = [
            {k: _json_safe(v) for k, v in p.items()} for p in all_predictions
        ]
        json_path = output_dir / "drug_repurposing_predictions_with_confidence.json"
        with open(json_path, 'w') as jf:
            json.dump(json_preds, jf, indent=2)
        print(f"Saved JSON to {json_path}")
    else:
        output_path = output_dir / "drug_repurposing_predictions_with_confidence.csv"
        import csv
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_predictions[0].keys())
            writer.writeheader()
            writer.writerows(all_predictions)
        print(f"\nSaved to {output_path}")

    # Compare with old deliverable if it exists
    old_path = output_dir / "drug_repurposing_predictions_with_confidence.xlsx"
    if HAS_OPENPYXL and old_path.exists() and old_path != output_path:
        print("\nNote: Old deliverable exists but comparison skipped (different format)")

    total_time = time.time() - start
    print(f"\nTotal time: {total_time:.0f}s")
    print(f"Total predictions: {len(all_predictions)}")
    print(f"Unique diseases: {len(set(p['disease_id'] for p in all_predictions))}")
    print(f"Unique drugs: {len(set(p['drug_id'] for p in all_predictions))}")

    # Summary statistics
    gt_preds = [p for p in all_predictions if p['is_known_indication']]
    print(f"\nKnown indications in predictions: {len(gt_preds)}")
    for tier in ['GOLDEN', 'HIGH', 'MEDIUM', 'LOW', 'FILTER']:
        tier_preds = [p for p in all_predictions if p['confidence_tier'] == tier]
        tier_gt = [p for p in tier_preds if p['is_known_indication']]
        prec = 100 * len(tier_gt) / len(tier_preds) if tier_preds else 0
        print(f"  {tier}: {len(tier_gt)}/{len(tier_preds)} = {prec:.1f}%")


if __name__ == '__main__':
    main()
