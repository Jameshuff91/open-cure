#!/usr/bin/env python3
"""
h907: Import Ryland Mortlock's blinded expert review of GOLDEN-tier predictions.

Ryland (Yale, dermatology) is blinded-reviewing ~855 GOLDEN-tier derm predictions.
Once returned, the labels are expert verdicts independent of DRKG — exactly what
the tier system needs to calibrate beyond the DRKG ceiling.

This importer:
    1. Accepts the raw review file (CSV, XLSX, or JSON).
    2. Validates every row against data/reference/ryland_review_schema.json.
    3. Resolves missing drug_id / disease_id from the current predictor aliasing
       layer so downstream code can join on entity IDs.
    4. Writes the validated records to data/reference/expert_labels_ryland.json
       keyed by prediction_id (== '<disease_id>||<drug_id>').
    5. Emits a concise validation report (n accepted, n rejected, reasons).

Leakage-safe guarantees:
    * The importer never touches expanded_ground_truth.json.
    * Every record is stamped with provenance='expert_ryland'.
    * The downstream evaluator (h907: scripts/h907_eval_expert_labels.py)
      keeps provenance tags separate from predictor.ground_truth and only uses
      expert labels to compute a parallel precision column.

Usage:
    python scripts/import_ryland_review.py path/to/review.csv
    python scripts/import_ryland_review.py path/to/review.xlsx --out data/reference/expert_labels_ryland.json
    python scripts/import_ryland_review.py --dry-run path/to/review.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
REF_DIR = REPO_ROOT / "data" / "reference"
SCHEMA_PATH = REF_DIR / "ryland_review_schema.json"
DEFAULT_OUT = REF_DIR / "expert_labels_ryland.json"

VALID_VERDICTS = {"plausible", "implausible", "known", "adverse", "unsure"}
PROVENANCE_TAG = "expert_ryland"


def _load_schema() -> Dict[str, Any]:
    with open(SCHEMA_PATH) as f:
        return json.load(f)


def _resolve_ids(
    record: Dict[str, Any],
    name_to_drug_id: Dict[str, str],
    disease_name_to_id: Dict[str, str],
) -> Tuple[Optional[str], Optional[str], List[str]]:
    """Return (drug_id, disease_id, warnings) — best-effort name -> ID resolution."""
    warnings: List[str] = []

    drug_id = record.get("drug_id") or ""
    if not drug_id:
        name_lc = (record.get("drug") or "").strip().lower()
        drug_id = name_to_drug_id.get(name_lc, "")
        if not drug_id:
            warnings.append(f"unresolved drug name: {record.get('drug')!r}")

    disease_id = record.get("disease_id") or ""
    if not disease_id:
        dname_lc = (record.get("disease") or "").strip().lower()
        disease_id = disease_name_to_id.get(dname_lc, "")
        if not disease_id:
            warnings.append(f"unresolved disease name: {record.get('disease')!r}")

    return (drug_id or None, disease_id or None, warnings)


def _load_raw(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = list(data.values())
        return [dict(r) for r in data]
    if suffix == ".csv":
        import csv
        with open(path, newline="") as f:
            return [dict(r) for r in csv.DictReader(f)]
    if suffix in (".xlsx", ".xls"):
        import pandas as pd
        df = pd.read_excel(path)
        return df.to_dict(orient="records")
    raise ValueError(f"Unsupported review file extension: {suffix}")


def _validate_record(
    record: Dict[str, Any],
    schema: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    # prediction_id is nominally required by schema but the importer can
    # synthesize it from resolved drug_id + disease_id, so defer that check.
    for field in schema["required"]:
        if field in ("provenance", "prediction_id"):
            continue
        if record.get(field) in (None, "", float("nan")):
            errors.append(f"missing required field: {field}")

    verdict = record.get("verdict")
    if verdict is not None and verdict not in VALID_VERDICTS:
        errors.append(f"invalid verdict: {verdict!r}")

    rc = record.get("reviewer_confidence")
    if rc not in (None, ""):
        try:
            rc_int = int(rc)
            if not 1 <= rc_int <= 5:
                errors.append(f"reviewer_confidence out of range [1,5]: {rc_int}")
        except (TypeError, ValueError):
            errors.append(f"reviewer_confidence not an integer: {rc!r}")

    if verdict in ("implausible", "adverse") and not record.get("reasoning"):
        errors.append(f"verdict={verdict} requires non-empty reasoning")

    return (len(errors) == 0, errors)


def _build_lookups() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return (drug_name_to_id, disease_name_to_id) using existing infra."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from production_predictor import DrugRepurposingPredictor  # type: ignore

    predictor = DrugRepurposingPredictor()
    disease_name_to_id: Dict[str, str] = {}
    for disease_id, name in predictor.disease_names.items():
        if name:
            disease_name_to_id[name.strip().lower()] = disease_id
    return (dict(predictor.name_to_drug_id), disease_name_to_id)


def import_review(
    source: Path,
    out: Path,
    dry_run: bool = False,
) -> Dict[str, Any]:
    schema = _load_schema()
    raw_records = _load_raw(source)

    drug_name_to_id, disease_name_to_id = _build_lookups()

    accepted: Dict[str, Dict[str, Any]] = {}
    rejected: List[Dict[str, Any]] = []
    warning_counter: Counter[str] = Counter()
    verdict_counter: Counter[str] = Counter()

    for raw in raw_records:
        ok, errors = _validate_record(raw, schema)

        drug_id, disease_id, warnings = _resolve_ids(
            raw, drug_name_to_id, disease_name_to_id
        )
        for w in warnings:
            warning_counter[w.split(":")[0]] += 1

        pred_id_raw = raw.get("prediction_id")
        pred_id: Optional[str] = str(pred_id_raw) if pred_id_raw else None
        if not pred_id and drug_id and disease_id:
            pred_id = f"{disease_id}||{drug_id}"

        if not pred_id:
            errors.append("missing prediction_id and could not synthesize one")
            ok = False

        if not ok or pred_id is None:
            rejected.append({"record": raw, "errors": errors})
            continue

        if pred_id in accepted:
            rejected.append({
                "record": raw,
                "errors": [f"duplicate prediction_id: {pred_id}"],
            })
            continue

        record: Dict[str, Any] = {
            "prediction_id": pred_id,
            "drug": raw["drug"],
            "disease": raw["disease"],
            "verdict": raw["verdict"],
            "provenance": PROVENANCE_TAG,
        }
        if drug_id:
            record["drug_id"] = drug_id
        if disease_id:
            record["disease_id"] = disease_id
        for field in (
            "reasoning",
            "reviewer_confidence",
            "reviewer",
            "review_date",
            "evidence_citations",
        ):
            if raw.get(field) not in (None, "", float("nan")):
                record[field] = raw[field]
        record.setdefault("reviewer", "ryland_mortlock")

        accepted[pred_id] = record
        verdict_counter[record["verdict"]] += 1

    report = {
        "source": str(source),
        "n_raw": len(raw_records),
        "n_accepted": len(accepted),
        "n_rejected": len(rejected),
        "verdict_distribution": dict(verdict_counter),
        "warning_summary": dict(warning_counter),
        "rejected_sample": rejected[:5],
    }

    if not dry_run:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(accepted, f, indent=2, sort_keys=True)
        report["out_path"] = str(out)

    return report


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", type=Path, help="Path to review CSV/XLSX/JSON")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"Output path (default: {DEFAULT_OUT})")
    p.add_argument("--dry-run", action="store_true", help="Validate only; do not write output")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.source.exists():
        print(f"ERROR: source file not found: {args.source}", file=sys.stderr)
        return 2

    report = import_review(args.source, args.out, dry_run=args.dry_run)
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["n_rejected"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
