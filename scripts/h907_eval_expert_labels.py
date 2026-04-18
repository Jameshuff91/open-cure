#!/usr/bin/env python3
"""
h907: Parallel tier precision split — DRKG-GT vs expert (Ryland) labels.

This is the evaluation-side counterpart to scripts/import_ryland_review.py.
It runs the deliverable through two independent precision columns:

    drkg_gt_precision  — hit iff (disease_id, drug_id) in expanded_ground_truth
    expert_precision   — hit iff Ryland marked verdict in {plausible, known}
                         with reviewer_confidence >= min_confidence

The two columns are reported SEPARATELY per tier. Expert-label denominator
is the count of predictions that Ryland actually reviewed — predictions he
did not see are excluded from that column entirely.

Leakage guarantee: expert_labels are loaded via src/expert_labels.py and are
NEVER merged into predictor.ground_truth. They are used only for hit/miss
bookkeeping at eval time.

Usage:
    python scripts/h907_eval_expert_labels.py
    python scripts/h907_eval_expert_labels.py --tier GOLDEN --tier HIGH
    python scripts/h907_eval_expert_labels.py --min-reviewer-confidence 4
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from expert_labels import load_expert_labels, ExpertLabels  # type: ignore

DELIVERABLE_XLSX = REPO_ROOT / "data" / "deliverables" / "drug_repurposing_predictions_with_confidence.xlsx"
EXPANDED_GT_PATH = REPO_ROOT / "data" / "reference" / "expanded_ground_truth.json"
DEFAULT_OUT = REPO_ROOT / "data" / "analysis" / "h907_expert_label_precision.json"


def _build_drkg_gt_set() -> Set[Tuple[str, str]]:
    with open(EXPANDED_GT_PATH) as f:
        gt = json.load(f)
    out: Set[Tuple[str, str]] = set()
    for disease_id, drugs in gt.items():
        for d in drugs:
            if isinstance(d, str):
                out.add((disease_id, d))
            elif isinstance(d, dict):
                did = d.get("drug_id") or d.get("drug")
                if did:
                    out.add((disease_id, did))
    return out


def evaluate(
    tiers: Optional[List[str]],
    min_reviewer_confidence: int,
) -> Dict[str, Any]:
    import pandas as pd

    df = pd.read_excel(DELIVERABLE_XLSX)
    if tiers:
        df = df[df["confidence_tier"].isin(tiers)]
    df = df.copy()

    drkg_gt = _build_drkg_gt_set()
    labels: ExpertLabels = load_expert_labels()

    per_tier = defaultdict(lambda: {
        "n_preds": 0,
        "drkg_hits": 0, "drkg_total": 0,
        "expert_hits": 0, "expert_total": 0,
        "expert_misses": 0,
        "expert_reviewed_overlap_drkg_hit": 0,
        "expert_reviewed_overlap_drkg_miss": 0,
    })

    for _, row in df.iterrows():
        tier = row["confidence_tier"]
        disease_id = row["disease_id"]
        drug_id = row["drug_id"]
        bucket = per_tier[tier]
        bucket["n_preds"] += 1

        is_drkg_hit = (disease_id, drug_id) in drkg_gt
        bucket["drkg_total"] += 1
        if is_drkg_hit:
            bucket["drkg_hits"] += 1

        expert_verdict = labels.is_hit(disease_id, drug_id, min_confidence=min_reviewer_confidence)
        if expert_verdict is not None:
            bucket["expert_total"] += 1
            if expert_verdict:
                bucket["expert_hits"] += 1
            else:
                bucket["expert_misses"] += 1
            if is_drkg_hit:
                bucket["expert_reviewed_overlap_drkg_hit"] += 1
            else:
                bucket["expert_reviewed_overlap_drkg_miss"] += 1

    results: Dict[str, Dict] = {}
    for tier, b in per_tier.items():
        drkg_prec = (100 * b["drkg_hits"] / b["drkg_total"]) if b["drkg_total"] else None
        expert_prec = (100 * b["expert_hits"] / b["expert_total"]) if b["expert_total"] else None
        coverage = (100 * b["expert_total"] / b["n_preds"]) if b["n_preds"] else 0
        results[tier] = {
            "n_preds": b["n_preds"],
            "drkg_precision_pct": round(drkg_prec, 1) if drkg_prec is not None else None,
            "drkg_hits": b["drkg_hits"],
            "drkg_total": b["drkg_total"],
            "expert_precision_pct": round(expert_prec, 1) if expert_prec is not None else None,
            "expert_hits": b["expert_hits"],
            "expert_misses": b["expert_misses"],
            "expert_total": b["expert_total"],
            "expert_review_coverage_pct": round(coverage, 1),
            "expert_reviewed_drkg_hits": b["expert_reviewed_overlap_drkg_hit"],
            "expert_reviewed_drkg_misses": b["expert_reviewed_overlap_drkg_miss"],
        }

    return {
        "min_reviewer_confidence": min_reviewer_confidence,
        "n_expert_labels_loaded": len(labels),
        "expert_labels_source": labels.source,
        "tiers": dict(results),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tier", action="append", default=None, help="Filter to these tiers (repeatable)")
    p.add_argument("--min-reviewer-confidence", type=int, default=3)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    result = evaluate(args.tier, args.min_reviewer_confidence)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps({
        "min_reviewer_confidence": result["min_reviewer_confidence"],
        "n_expert_labels_loaded": result["n_expert_labels_loaded"],
        "expert_labels_source": result["expert_labels_source"],
        "tiers": result["tiers"],
    }, indent=2))
    print(f"\n=> wrote {args.out}")
    if result["n_expert_labels_loaded"] == 0:
        print("NOTE: no expert labels loaded; expert_precision columns will be null "
              "until scripts/import_ryland_review.py has been run against Ryland's review.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
