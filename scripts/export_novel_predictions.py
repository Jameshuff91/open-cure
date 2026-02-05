#!/usr/bin/env python3
"""
Export Novel Drug Repurposing Predictions (h145)

Generates a comprehensive deliverable of truly novel predictions:
1. Batch processes all evaluable diseases with kNN
2. Applies production tiered confidence (h135)
3. Filters out FDA-approved pairs using ground truth + confidence_filter
4. Applies confidence filter exclusions (withdrawn drugs, etc.)
5. Exports to data/deliverables with GOLDEN/HIGH/MEDIUM tiers

Output: data/deliverables/novel_predictions_YYYYMMDD.json/.xlsx
"""

import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from production_predictor import (
    ConfidenceTier,
    DrugPrediction,
    DrugRepurposingPredictor,
    PredictionResult,
)
from confidence_filter import (
    filter_prediction,
    ConfidenceLevel,
    classify_drug_type,
    WITHDRAWN_DRUG_PATTERNS,
    DISCONTINUED_DRUG_PATTERNS,
)
import re


@dataclass
class NovelPrediction:
    """A novel (non-FDA-approved) drug prediction."""
    disease_name: str
    disease_id: str
    disease_category: str
    disease_tier: int
    drug_name: str
    drug_id: str
    drug_type: str
    rank: int
    knn_score: float
    norm_score: float
    confidence_tier: str  # GOLDEN, HIGH, MEDIUM
    train_frequency: int
    mechanism_support: bool
    has_targets: bool
    category_rescue_applied: bool
    is_fda_approved: bool  # From ground truth
    filter_exclusion: Optional[str]  # If excluded by confidence_filter

    def to_dict(self) -> Dict:
        return asdict(self)


class NovelPredictionExporter:
    """Export novel predictions for all evaluable diseases."""

    def __init__(self, data_dir: Optional[Path] = None):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent

        self.data_dir = data_dir
        self.reference_dir = data_dir / "data" / "reference"
        self.deliverables_dir = data_dir / "data" / "deliverables"
        self.deliverables_dir.mkdir(exist_ok=True)

        print("Loading production predictor...")
        self.predictor = DrugRepurposingPredictor(data_dir)

        # Load FDA-approved pairs for more comprehensive filtering
        self._load_fda_approved_pairs()

    def _load_fda_approved_pairs(self) -> None:
        """Load FDA-approved pairs from multiple sources."""
        self.fda_approved: Set[Tuple[str, str]] = set()

        # 1. From fda_approved_pairs.json
        fda_path = self.reference_dir / "fda_approved_pairs.json"
        if fda_path.exists():
            with open(fda_path) as f:
                data = json.load(f)
            for pair in data.get("pairs", []):
                drug = pair["drug"].lower()
                disease = pair["disease"].lower()
                self.fda_approved.add((drug, disease))

        # 2. From ground truth (Every Cure GT = FDA approved indications)
        # The ground truth IS the set of known approved indications
        for disease_id, drug_ids in self.predictor.ground_truth.items():
            disease_name = self.predictor.disease_names.get(disease_id, "").lower()
            for drug_id in drug_ids:
                drug_name = self.predictor.drug_id_to_name.get(drug_id, "").lower()
                if drug_name and disease_name:
                    self.fda_approved.add((drug_name, disease_name))

        print(f"Loaded {len(self.fda_approved)} FDA-approved pairs")

    def is_fda_approved(self, drug_name: str, disease_name: str) -> bool:
        """Check if a drug-disease pair is FDA approved."""
        drug_lower = drug_name.lower()
        disease_lower = disease_name.lower()

        # Exact match
        if (drug_lower, disease_lower) in self.fda_approved:
            return True

        # Fuzzy match: check if drug is in any disease name variant
        for approved_drug, approved_disease in self.fda_approved:
            if approved_drug == drug_lower:
                # Check if disease names are similar
                if (approved_disease in disease_lower or
                    disease_lower in approved_disease):
                    return True

        return False

    def check_filter_exclusion(self, drug_name: str, disease_name: str, score: float) -> Optional[str]:
        """Check if prediction should be excluded by confidence_filter rules."""
        result = filter_prediction(drug_name, disease_name, score)
        if result.confidence == ConfidenceLevel.EXCLUDED:
            return result.reason
        return None

    def is_withdrawn_or_discontinued(self, drug_name: str) -> bool:
        """Check if drug is withdrawn or discontinued."""
        drug_lower = drug_name.lower()
        for pattern in WITHDRAWN_DRUG_PATTERNS + DISCONTINUED_DRUG_PATTERNS:
            if re.search(pattern, drug_lower):
                return True
        return False

    def get_evaluable_diseases(self) -> List[str]:
        """Get all diseases that can be evaluated (have embeddings)."""
        evaluable = []
        for disease_id in self.predictor.ground_truth:
            if disease_id in self.predictor.embeddings:
                disease_name = self.predictor.disease_names.get(disease_id)
                if disease_name:
                    evaluable.append(disease_name)
        return evaluable

    def export_predictions(
        self,
        min_tier: str = "MEDIUM",  # GOLDEN, HIGH, MEDIUM
        top_n_per_disease: int = 30,
        exclude_fda_approved: bool = True,
    ) -> Tuple[List[NovelPrediction], Dict]:
        """
        Export novel predictions for all evaluable diseases.

        Args:
            min_tier: Minimum confidence tier to include
            top_n_per_disease: Max predictions per disease
            exclude_fda_approved: If True, exclude known FDA-approved pairs

        Returns:
            (list of predictions, statistics dict)
        """
        tier_order = {
            "GOLDEN": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "FILTER": 4
        }
        min_tier_rank = tier_order[min_tier]

        diseases = self.get_evaluable_diseases()
        print(f"Processing {len(diseases)} evaluable diseases...")

        all_predictions: List[NovelPrediction] = []
        stats = {
            "total_diseases": len(diseases),
            "diseases_with_predictions": 0,
            "total_predictions": 0,
            "novel_predictions": 0,
            "fda_approved_filtered": 0,
            "excluded_by_filter": 0,
            "withdrawn_filtered": 0,
            "by_tier": defaultdict(int),
            "by_category": defaultdict(int),
            "exclusion_reasons": defaultdict(int),
        }

        for i, disease_name in enumerate(diseases):
            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(diseases)}...")

            # Get predictions
            result = self.predictor.predict(
                disease_name,
                k=20,
                top_n=top_n_per_disease,
                include_filtered=False,  # Already excludes FILTER tier
            )

            if not result.predictions:
                continue

            disease_has_predictions = False

            for pred in result.predictions:
                # Check tier threshold
                tier_rank = tier_order.get(pred.confidence_tier.value, 4)
                if tier_rank > min_tier_rank:
                    continue

                stats["total_predictions"] += 1

                # Check if withdrawn/discontinued
                if self.is_withdrawn_or_discontinued(pred.drug_name):
                    stats["withdrawn_filtered"] += 1
                    continue

                # Check FDA approval
                is_approved = self.is_fda_approved(pred.drug_name, disease_name)
                if exclude_fda_approved and is_approved:
                    stats["fda_approved_filtered"] += 1
                    continue

                # Check confidence_filter exclusions
                filter_exclusion = self.check_filter_exclusion(
                    pred.drug_name, disease_name, pred.norm_score
                )
                if filter_exclusion:
                    stats["excluded_by_filter"] += 1
                    stats["exclusion_reasons"][filter_exclusion] += 1
                    continue

                # This is a valid novel prediction
                drug_type = classify_drug_type(pred.drug_name)

                novel_pred = NovelPrediction(
                    disease_name=disease_name,
                    disease_id=result.disease_id or "",
                    disease_category=result.category,
                    disease_tier=result.disease_tier,
                    drug_name=pred.drug_name,
                    drug_id=pred.drug_id,
                    drug_type=drug_type,
                    rank=pred.rank,
                    knn_score=float(pred.knn_score),
                    norm_score=float(pred.norm_score),
                    confidence_tier=pred.confidence_tier.value,
                    train_frequency=pred.train_frequency,
                    mechanism_support=pred.mechanism_support,
                    has_targets=pred.has_targets,
                    category_rescue_applied=pred.category_rescue_applied,
                    is_fda_approved=is_approved,
                    filter_exclusion=None,
                )

                all_predictions.append(novel_pred)
                disease_has_predictions = True
                stats["novel_predictions"] += 1
                stats["by_tier"][pred.confidence_tier.value] += 1
                stats["by_category"][result.category] += 1

            if disease_has_predictions:
                stats["diseases_with_predictions"] += 1

        # Convert defaultdicts to regular dicts
        stats["by_tier"] = dict(stats["by_tier"])
        stats["by_category"] = dict(stats["by_category"])
        stats["exclusion_reasons"] = dict(stats["exclusion_reasons"])

        return all_predictions, stats

    def save_deliverable(
        self,
        predictions: List[NovelPrediction],
        stats: Dict,
        output_prefix: str = "novel_predictions",
    ) -> Tuple[Path, Path]:
        """Save predictions to JSON and Excel files."""
        timestamp = datetime.now().strftime("%Y%m%d")

        # Sort by tier then score
        tier_order = {"GOLDEN": 0, "HIGH": 1, "MEDIUM": 2}
        predictions_sorted = sorted(
            predictions,
            key=lambda p: (tier_order.get(p.confidence_tier, 3), -p.norm_score)
        )

        # JSON output
        json_path = self.deliverables_dir / f"{output_prefix}_{timestamp}.json"
        output_data = {
            "metadata": {
                "generated": datetime.now().isoformat(),
                "description": "Novel drug repurposing predictions (not FDA-approved)",
                "methodology": "kNN collaborative filtering with tiered confidence (h135/h136)",
                "statistics": stats,
            },
            "predictions": [p.to_dict() for p in predictions_sorted],
        }
        with open(json_path, "w") as f:
            json.dump(output_data, f, indent=2)

        # Excel output
        xlsx_path = self.deliverables_dir / f"{output_prefix}_{timestamp}.xlsx"
        df = pd.DataFrame([p.to_dict() for p in predictions_sorted])

        # Reorder columns for readability
        col_order = [
            "confidence_tier", "disease_name", "drug_name", "rank", "norm_score",
            "train_frequency", "mechanism_support", "drug_type", "disease_category",
            "disease_tier", "category_rescue_applied", "knn_score", "has_targets",
            "disease_id", "drug_id", "is_fda_approved", "filter_exclusion",
        ]
        df = df[[c for c in col_order if c in df.columns]]

        # Save with multiple sheets
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            # All predictions
            df.to_excel(writer, sheet_name="All Predictions", index=False)

            # By tier
            for tier in ["GOLDEN", "HIGH", "MEDIUM"]:
                tier_df = df[df["confidence_tier"] == tier]
                if len(tier_df) > 0:
                    tier_df.to_excel(writer, sheet_name=f"{tier} Tier", index=False)

            # Summary stats
            summary_data = [
                {"Metric": "Total Diseases Evaluated", "Value": stats["total_diseases"]},
                {"Metric": "Diseases with Predictions", "Value": stats["diseases_with_predictions"]},
                {"Metric": "Total Novel Predictions", "Value": stats["novel_predictions"]},
                {"Metric": "FDA-Approved Filtered", "Value": stats["fda_approved_filtered"]},
                {"Metric": "Excluded by Safety Filter", "Value": stats["excluded_by_filter"]},
                {"Metric": "Withdrawn Drugs Filtered", "Value": stats["withdrawn_filtered"]},
            ]
            for tier, count in stats["by_tier"].items():
                summary_data.append({"Metric": f"  {tier} Tier Count", "Value": count})

            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

        return json_path, xlsx_path


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Export novel drug repurposing predictions"
    )
    parser.add_argument(
        "--min-tier",
        choices=["GOLDEN", "HIGH", "MEDIUM"],
        default="MEDIUM",
        help="Minimum confidence tier to include (default: MEDIUM)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Max predictions per disease (default: 30)"
    )
    parser.add_argument(
        "--include-fda-approved",
        action="store_true",
        help="Include FDA-approved pairs in output"
    )

    args = parser.parse_args()

    exporter = NovelPredictionExporter()

    print(f"\nExporting predictions (min tier: {args.min_tier})...")
    predictions, stats = exporter.export_predictions(
        min_tier=args.min_tier,
        top_n_per_disease=args.top_n,
        exclude_fda_approved=not args.include_fda_approved,
    )

    print(f"\n{'='*60}")
    print("EXPORT STATISTICS")
    print(f"{'='*60}")
    print(f"Total diseases evaluated: {stats['total_diseases']}")
    print(f"Diseases with predictions: {stats['diseases_with_predictions']}")
    print(f"Total predictions before filtering: {stats['total_predictions']}")
    print(f"Novel predictions exported: {stats['novel_predictions']}")
    print(f"\nFiltered out:")
    print(f"  - FDA-approved pairs: {stats['fda_approved_filtered']}")
    print(f"  - Withdrawn drugs: {stats['withdrawn_filtered']}")
    print(f"  - Safety filter exclusions: {stats['excluded_by_filter']}")

    print(f"\nBy confidence tier:")
    for tier, count in sorted(stats["by_tier"].items()):
        print(f"  - {tier}: {count}")

    print(f"\nBy disease category:")
    for cat, count in sorted(stats["by_category"].items(), key=lambda x: -x[1]):
        print(f"  - {cat}: {count}")

    if stats["exclusion_reasons"]:
        print(f"\nSafety filter exclusion reasons:")
        for reason, count in sorted(stats["exclusion_reasons"].items(), key=lambda x: -x[1])[:10]:
            print(f"  - {reason}: {count}")

    # Save deliverables
    print(f"\nSaving deliverables...")
    json_path, xlsx_path = exporter.save_deliverable(predictions, stats)
    print(f"  - JSON: {json_path}")
    print(f"  - Excel: {xlsx_path}")

    # Show top GOLDEN predictions
    golden = [p for p in predictions if p.confidence_tier == "GOLDEN"]
    if golden:
        print(f"\n{'='*60}")
        print(f"TOP 10 GOLDEN TIER PREDICTIONS (highest confidence)")
        print(f"{'='*60}")
        for p in sorted(golden, key=lambda x: -x.norm_score)[:10]:
            mech = "+" if p.mechanism_support else "-"
            rescued = " [rescued]" if p.category_rescue_applied else ""
            print(f"  {p.drug_name} -> {p.disease_name}")
            print(f"     Score: {p.norm_score:.3f} | Freq: {p.train_frequency} | Mech: {mech}{rescued}")


if __name__ == "__main__":
    main()
