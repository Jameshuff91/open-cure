#!/usr/bin/env python3
"""h1100 Step 1: trace sub_reason for FILTER-tier positive-control misfires.

For each positive-control case that landed in FILTER in the baseline run,
re-run the predictor, pull the matching DrugPrediction, then call
_assign_confidence_tier directly with the prediction's own inputs so we can
see the sub_reason that drove the FILTER decision.

Also reports whether each drug-disease pair is in expanded_ground_truth.json
so we can tell genuine FDA indications from noise.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from production_predictor import DrugRepurposingPredictor  # type: ignore[import-not-found]

ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"

# Pairs to trace. Mix of (1) h1100 FILTER-misfire targets from
# positive_controls_baseline.json and (2) h1101 dantrolene cases.
TARGETS: List[Dict] = [
    # h1100: FILTER at rank <= 30 for FDA-approved indications
    {"drug": "metformin", "disease": "diabetes mellitus, type 2",
     "expected_mesh": "D003922"},
    {"drug": "tetrabenazine", "disease": "huntington disease",
     "expected_mesh": "D006816"},
    {"drug": "trastuzumab", "disease": "breast neoplasms",
     "expected_mesh": "D001943"},
    {"drug": "sildenafil", "disease": "hypertension, pulmonary",
     "expected_mesh": "D006973"},
    {"drug": "riluzole", "disease": "amyotrophic lateral sclerosis",
     "expected_mesh": "D000690"},
    # h1101: dantrolene landed FILTER at top ranks on VT-family diseases
    {"drug": "dantrolene", "disease": "ventricular tachycardia",
     "expected_mesh": "D017180"},
    {"drug": "dantrolene", "disease": "ventricular fibrillation",
     "expected_mesh": "D014693"},
    {"drug": "dantrolene", "disease": "tachycardia",
     "expected_mesh": "D013610"},
    {"drug": "dantrolene", "disease": "arrhythmia",
     "expected_mesh": "D001145"},
    {"drug": "dantrolene", "disease": "heart failure",
     "expected_mesh": "D006333"},
]


def build_name_to_id(predictor) -> Dict[str, str]:
    idx: Dict[str, str] = {}
    for did, name in predictor.drug_id_to_name.items():
        if isinstance(name, str):
            idx.setdefault(name.lower(), did)
    return idx


def load_expanded_gt() -> Dict[str, set]:
    with open(PROJECT_ROOT / "data" / "reference" / "expanded_ground_truth.json") as f:
        raw = json.load(f)
    return {k: set(v) for k, v in raw.items()}


def trace_pair(
    predictor,
    drug_name: str,
    disease_query: str,
    drug_name_to_id: Dict[str, str],
    expanded_gt: Dict[str, set],
) -> Dict:
    out: Dict = {"drug": drug_name, "disease_query": disease_query}

    result = predictor.predict(disease_query, top_n=100, include_filtered=True)
    if not result or not getattr(result, "predictions", None):
        out["status"] = "no_predictions"
        return out

    out["disease_id"] = result.disease_id
    out["disease_name"] = result.disease_name
    out["category"] = getattr(result, "category", "?")

    drug_id_hint = drug_name_to_id.get(drug_name.lower())

    target_pred = None
    for p in result.predictions:
        if p.drug_name.lower() == drug_name.lower():
            target_pred = p
            break
        if drug_id_hint and p.drug_id == drug_id_hint:
            target_pred = p
            break
    if target_pred is None:
        out["status"] = "not_in_top_100"
        return out

    # Expanded GT presence
    gt_set = expanded_gt.get(result.disease_id, set())
    out["in_expanded_gt"] = target_pred.drug_id in gt_set
    out["gt_size"] = len(gt_set)

    out["status"] = "found"
    out["rank"] = int(target_pred.rank)
    out["tier"] = target_pred.confidence_tier.value
    out["knn_score"] = round(float(target_pred.knn_score), 4)
    out["norm_score"] = round(float(target_pred.norm_score), 4)
    out["train_frequency"] = int(target_pred.train_frequency)
    out["mechanism_support"] = bool(target_pred.mechanism_support)
    out["has_targets"] = bool(target_pred.has_targets)
    out["disease_tier"] = int(target_pred.disease_tier)
    out["category"] = target_pred.category
    out["drug_id"] = target_pred.drug_id

    # Re-run tier assignment in isolation so we see sub_reason
    tier, rescue, sub_reason = predictor._assign_confidence_tier(
        rank=target_pred.rank,
        train_frequency=target_pred.train_frequency,
        mechanism_support=target_pred.mechanism_support,
        has_targets=target_pred.has_targets,
        disease_tier=target_pred.disease_tier,
        category=target_pred.category,
        drug_name=target_pred.drug_name,
        disease_name=result.disease_name,
        drug_id=target_pred.drug_id,
        knn_score=target_pred.knn_score,
        disease_id=result.disease_id,
    )
    out["assign_tier"] = tier.value
    out["assign_rescue"] = bool(rescue)
    out["assign_sub_reason"] = sub_reason
    out["tier_from_assign_matches_final"] = tier.value == target_pred.confidence_tier.value

    return out


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("Loading predictor ...")
    predictor = DrugRepurposingPredictor()
    print(f"  Loaded in {time.time()-t0:.0f}s")

    drug_name_to_id = build_name_to_id(predictor)
    expanded_gt = load_expanded_gt()

    rows = []
    for t in TARGETS:
        row = trace_pair(predictor, t["drug"], t["disease"], drug_name_to_id, expanded_gt)
        rows.append(row)

    # Pretty print
    print("\n" + "=" * 120)
    print(f"{'drug':<16} {'disease':<30} {'rank':>4} {'tier':>6} {'assign':>6} {'sub_reason':<36} {'in_gt':>5} {'match':>5}")
    print("-" * 120)
    for r in rows:
        if r.get("status") != "found":
            print(f"{r['drug']:<16} {r['disease_query'][:30]:<30} {r.get('status','?')}")
            continue
        print(
            f"{r['drug']:<16} "
            f"{r['disease_query'][:30]:<30} "
            f"{r['rank']:>4} "
            f"{r['tier']:>6} "
            f"{r['assign_tier']:>6} "
            f"{str(r['assign_sub_reason'])[:36]:<36} "
            f"{str(r['in_expanded_gt']):>5} "
            f"{str(r['tier_from_assign_matches_final']):>5}"
        )

    out_path = ANALYSIS_DIR / "h1100_filter_misfire_diagnostic.json"
    with open(out_path, "w") as f:
        json.dump({"targets": rows}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
