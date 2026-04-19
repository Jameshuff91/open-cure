#!/usr/bin/env python3
"""
Positive control suite for drug repurposing predictions.

Implements the 'Run Positive Controls' rule from CLAUDE.md's Scientific
Reasoning Protocol. Given a fixed benchmark of known repurposing hits, we
check their rank, tier, and confidence in the current production pipeline.

Usage:
    python3 scripts/positive_controls.py [--baseline | --label TAG]
    OPEN_CURE_EMBEDDINGS_PREFIX=graphsage_256 python3 scripts/positive_controls.py --label graphsage

Outputs:
    data/analysis/positive_controls_<tag>.json
    stdout: summary table + per-pair verdict

The set is deliberately fixed: never edit this to match model output, only
edit when a case genuinely changes (new evidence, renamed entity, etc.).
Changes should be explained in git log.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from production_predictor import DrugRepurposingPredictor  # type: ignore[import-not-found]

ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"


# Fixed benchmark of known repurposing hits.
# Format: (drug_name, disease_query, source, category)
# source: RCT, guideline, cohort, FDA-approved, WHO-essential, well-established
# category: grouping for per-category breakdown
CONTROLS: List[dict] = [
    # RCT-validated repurposing hits from the paper
    {"drug": "dantrolene", "disease": "heart failure",
     "source": "RCT P=0.034, 66% VT reduction",
     "category": "rct_validated"},
    {"drug": "lovastatin", "disease": "multiple myeloma",
     "source": "RCT improved OS/PFS",
     "category": "rct_validated"},
    {"drug": "rituximab", "disease": "multiple sclerosis",
     "source": "WHO Essential Medicines 2023",
     "category": "rct_validated"},
    {"drug": "pitavastatin", "disease": "rheumatoid arthritis",
     "source": "Clinical: superior to MTX alone",
     "category": "rct_validated"},
    {"drug": "empagliflozin", "disease": "parkinson disease",
     "source": "Cohort HR 0.80 (0.68-0.92)",
     "category": "rct_validated"},

    # Classic well-established indications (positive controls in the
    # strict sense — model MUST surface these; failure = broken pipeline)
    {"drug": "metformin", "disease": "diabetes mellitus, type 2",
     "source": "First-line T2D treatment",
     "category": "well_established"},
    {"drug": "aspirin", "disease": "myocardial infarction",
     "source": "ACS/MI secondary prevention standard",
     "category": "well_established"},
    {"drug": "sildenafil", "disease": "hypertension, pulmonary",
     "source": "FDA-approved PAH",
     "category": "well_established"},
    {"drug": "thalidomide", "disease": "multiple myeloma",
     "source": "FDA-approved MM",
     "category": "well_established"},
    {"drug": "baricitinib", "disease": "covid-19",
     "source": "FDA EUA severe COVID-19",
     "category": "well_established"},

    # Classic repurposing successes — discovered repurposing, not
    # primary indication
    {"drug": "minoxidil", "disease": "alopecia",
     "source": "Rogaine; antihypertensive repurposed for hair loss",
     "category": "classic_repurpose"},
    {"drug": "bupropion", "disease": "tobacco use disorder",
     "source": "Zyban; antidepressant repurposed for smoking cessation",
     "category": "classic_repurpose"},
    {"drug": "finasteride", "disease": "alopecia",
     "source": "Propecia; BPH drug repurposed for hair loss",
     "category": "classic_repurpose"},
    {"drug": "raloxifene", "disease": "osteoporosis",
     "source": "SERM repurposed from breast cancer prevention",
     "category": "classic_repurpose"},

    # Biologics — the failure class h906/h921 was meant to address.
    # Model currently underperforms on these; useful stress test.
    {"drug": "adalimumab", "disease": "rheumatoid arthritis",
     "source": "FDA-approved RA",
     "category": "biologic"},
    {"drug": "trastuzumab", "disease": "breast neoplasms",
     "source": "FDA-approved HER2+ breast cancer",
     "category": "biologic"},
    {"drug": "bevacizumab", "disease": "colorectal neoplasms",
     "source": "FDA-approved metastatic CRC",
     "category": "biologic"},
    {"drug": "infliximab", "disease": "crohn disease",
     "source": "FDA-approved Crohn's",
     "category": "biologic"},

    # Rare diseases — should work per h977 (UTI rescue) if name resolves.
    # Known weakness: small GT → low kNN signal.
    {"drug": "riluzole", "disease": "amyotrophic lateral sclerosis",
     "source": "FDA-approved ALS",
     "category": "rare_disease"},
    {"drug": "tetrabenazine", "disease": "huntington disease",
     "source": "FDA-approved HD chorea",
     "category": "rare_disease"},
]


def resolve_drug_id(predictor: DrugRepurposingPredictor, drug_name: str) -> str:
    """Best-effort drug name -> DRKG compound ID lookup."""
    # predictor has id_to_drug_name somewhere. Build reverse index once.
    if not hasattr(predictor, "_drug_name_to_id"):
        idx: Dict[str, str] = {}
        for did, name in getattr(predictor, "id_to_drug_name", {}).items():
            if isinstance(name, str):
                idx.setdefault(name.lower(), did)
        # Also try drugbank_lookup pattern
        if not idx:
            for attr in ("drugbank_id_to_name", "drug_id_to_name"):
                src = getattr(predictor, attr, None)
                if isinstance(src, dict):
                    for did, name in src.items():
                        if isinstance(name, str):
                            idx.setdefault(name.lower(), did)
        predictor._drug_name_to_id = idx  # type: ignore[attr-defined]
    return predictor._drug_name_to_id.get(drug_name.lower(), "")  # type: ignore[attr-defined]


def evaluate_control(
    predictor: DrugRepurposingPredictor, ctrl: dict
) -> dict:
    disease = ctrl["disease"]
    drug = ctrl["drug"]
    out = {
        "drug": drug,
        "disease": disease,
        "source": ctrl["source"],
        "category": ctrl["category"],
    }
    try:
        result = predictor.predict(disease, top_n=100, include_filtered=True)
    except Exception as e:  # noqa: BLE001
        out["status"] = "error"
        out["error"] = str(e)[:200]
        return out

    if not result or not getattr(result, "predictions", None):
        out["status"] = "no_predictions"
        return out

    out["disease_resolved_to"] = getattr(result, "disease_id", "?")
    out["disease_category"] = getattr(result, "category", "?")

    for p in result.predictions:
        if p.drug_name.lower() == drug.lower():
            out["status"] = "found"
            out["rank"] = int(p.rank)
            out["tier"] = p.confidence_tier.value
            out["knn_score"] = round(float(p.knn_score), 4)
            out["normalized_score"] = round(float(p.norm_score), 4)
            out["is_known_indication"] = bool(
                disease in (predictor.disease_names.get(result.disease_id, "") or "")
                and drug in getattr(predictor, "id_to_drug_name", {}).values()
            )
            out["in_top_30"] = p.rank <= 30
            return out

    out["status"] = "not_in_top_100"
    return out


def summarize(results: List[dict]) -> dict:
    from collections import Counter
    tier_counts = Counter(r.get("tier", "NOT_FOUND") for r in results)
    status_counts = Counter(r.get("status", "?") for r in results)
    by_category: Dict[str, Dict[str, int]] = {}
    in_top30 = sum(1 for r in results if r.get("in_top_30"))
    in_top100 = sum(1 for r in results if r.get("rank", 999) <= 100)
    for r in results:
        c = r.get("category", "?")
        d = by_category.setdefault(c, {"n": 0, "in_top_30": 0, "in_top_100": 0,
                                        "golden_or_high": 0, "not_found": 0})
        d["n"] += 1
        if r.get("in_top_30"):
            d["in_top_30"] += 1
        if r.get("rank", 999) <= 100:
            d["in_top_100"] += 1
        if r.get("tier") in ("GOLDEN", "HIGH"):
            d["golden_or_high"] += 1
        if r.get("status") != "found":
            d["not_found"] += 1
    return {
        "total": len(results),
        "tier_counts": dict(tier_counts),
        "status_counts": dict(status_counts),
        "in_top_30": in_top30,
        "in_top_100": in_top100,
        "by_category": by_category,
    }


def print_report(results: List[dict], summary: dict) -> None:
    print("\n" + "=" * 78)
    print("  Positive Control Suite")
    print("=" * 78)
    print(f"  {'drug':<20} {'disease':<30} {'rank':>5} {'tier':>7}  {'status':<12}")
    print(f"  {'-'*20} {'-'*30} {'-'*5} {'-'*7}  {'-'*12}")
    for r in results:
        drug = r.get("drug", "")[:20]
        disease = r.get("disease", "")[:30]
        rank = r.get("rank", "-")
        tier = r.get("tier", "-")
        status = r.get("status", "?")
        marker = ""
        if r.get("in_top_30") and tier in ("GOLDEN", "HIGH"):
            marker = " PASS"
        elif status == "found" and rank <= 100:
            marker = " ok"
        elif status == "not_in_top_100":
            marker = " MISS"
        elif status in ("no_predictions", "error"):
            marker = " FAIL"
        print(f"  {drug:<20} {disease:<30} {str(rank):>5} {tier:>7}  {status:<12}{marker}")

    print()
    print(f"  In top 30:  {summary['in_top_30']}/{summary['total']}")
    print(f"  In top 100: {summary['in_top_100']}/{summary['total']}")
    print("  Tier distribution:")
    for t, n in summary["tier_counts"].items():
        print(f"    {t:<10} {n}")
    print("  By category (in_top_30 / n):")
    for c, d in summary["by_category"].items():
        print(f"    {c:<20} {d['in_top_30']:>2}/{d['n']:<2}  GOLDEN+HIGH: {d['golden_or_high']}  NOT FOUND: {d['not_found']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="baseline",
                    help="Output tag; e.g. baseline, graphsage")
    args = ap.parse_args()

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    embedding_prefix = os.environ.get("OPEN_CURE_EMBEDDINGS_PREFIX", "node2vec_256")
    print(f"Loading predictor (embeddings: {embedding_prefix}) ...")
    t0 = time.time()
    predictor = DrugRepurposingPredictor()
    print(f"  Loaded in {time.time()-t0:.0f}s")

    print(f"Running {len(CONTROLS)} positive controls ...")
    results = [evaluate_control(predictor, c) for c in CONTROLS]
    summary = summarize(results)
    print_report(results, summary)

    out_path = ANALYSIS_DIR / f"positive_controls_{args.label}.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "label": args.label,
                "embedding_prefix": embedding_prefix,
                "summary": summary,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
