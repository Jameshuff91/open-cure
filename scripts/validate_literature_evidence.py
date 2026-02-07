#!/usr/bin/env python3
"""
Validate literature evidence levels against holdout precision.

Splits predictions by evidence level and computes holdout precision
for each level to determine if literature evidence independently
justifies tier changes.

Usage:
    python scripts/validate_literature_evidence.py
    python scripts/validate_literature_evidence.py --tier MEDIUM
    python scripts/validate_literature_evidence.py --seeds 10
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def load_expanded_gt() -> dict[str, set[str]]:
    """Load expanded ground truth."""
    gt_path = DATA_DIR / "reference" / "expanded_ground_truth.json"
    if not gt_path.exists():
        print(f"ERROR: {gt_path} not found")
        sys.exit(1)
    with open(gt_path) as f:
        raw = json.load(f)
    return {did: set(drugs) if isinstance(drugs, list) else set()
            for did, drugs in raw.items()}


def load_literature_cache() -> dict[str, dict]:
    """Load literature mining cache."""
    cache_path = DATA_DIR / "validation" / "literature_mining_cache.json"
    if not cache_path.exists():
        print(f"ERROR: {cache_path} not found")
        print("Run scripts/run_literature_mining.py first.")
        sys.exit(1)
    with open(cache_path) as f:
        return json.load(f)


def load_deliverable() -> list[dict]:
    """Load current deliverable predictions."""
    json_path = DATA_DIR / "deliverables" / "drug_repurposing_predictions_with_confidence.json"
    if not json_path.exists():
        print(f"ERROR: {json_path} not found")
        sys.exit(1)
    with open(json_path) as f:
        return json.load(f)


def holdout_evaluation(preds: list[dict], expanded_gt: dict[str, set[str]],
                       n_seeds: int = 5) -> dict:
    """Run holdout evaluation: split diseases, compute precision per group.

    Uses disease-level 50/50 split (same as h393 evaluator).
    """
    # Get unique diseases with their predictions
    disease_preds: dict[str, list[dict]] = defaultdict(list)
    for p in preds:
        disease_preds[p["disease_id"]].append(p)

    all_diseases = list(disease_preds.keys())
    seed_results = []

    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        rng.shuffle(all_diseases)
        holdout = set(all_diseases[len(all_diseases) // 2:])

        hits = 0
        total = 0
        for did in holdout:
            gt_drugs = expanded_gt.get(did, set())
            for p in disease_preds[did]:
                total += 1
                if p["drug_id"] in gt_drugs:
                    hits += 1

        precision = hits / total if total > 0 else 0
        seed_results.append({
            "precision": precision,
            "hits": hits,
            "total": total,
        })

    precisions = [r["precision"] for r in seed_results]
    return {
        "mean_precision": round(np.mean(precisions) * 100, 1),
        "std_precision": round(np.std(precisions) * 100, 1),
        "mean_hits": round(np.mean([r["hits"] for r in seed_results]), 1),
        "mean_total": round(np.mean([r["total"] for r in seed_results]), 1),
        "n_seeds": n_seeds,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate literature evidence levels against holdout")
    parser.add_argument("--tier", type=str, default=None,
                        help="Focus on specific tier (GOLDEN,HIGH,MEDIUM,LOW)")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of holdout seeds (default: 5)")
    args = parser.parse_args()

    print("=" * 70)
    print("LITERATURE EVIDENCE HOLDOUT VALIDATION")
    print("=" * 70)

    # Load data
    expanded_gt = load_expanded_gt()
    lit_cache = load_literature_cache()
    all_preds = load_deliverable()

    print(f"Expanded GT: {len(expanded_gt)} diseases")
    print(f"Literature cache: {len(lit_cache)} entries")
    print(f"Predictions: {len(all_preds)}")

    # Annotate predictions with literature evidence
    for p in all_preds:
        key = f"{p['drug_name'].lower()}|{p['disease_name'].lower()}"
        entry = lit_cache.get(key, {})
        p["_lit_level"] = entry.get("evidence_level", "NOT_ASSESSED")
        p["_lit_score"] = entry.get("evidence_score", 0.0)

    # Filter by tier if specified
    if args.tier:
        tiers = [t.strip().upper() for t in args.tier.split(",")]
        all_preds = [p for p in all_preds if p.get("confidence_tier") in tiers]
        print(f"Filtered to tier(s) {args.tier}: {len(all_preds)} predictions")

    # Group by evidence level
    by_level: dict[str, list[dict]] = defaultdict(list)
    for p in all_preds:
        by_level[p["_lit_level"]].append(p)

    print(f"\nEvidence level distribution:")
    for level in ["STRONG_EVIDENCE", "MODERATE_EVIDENCE", "WEAK_EVIDENCE",
                  "ADVERSE_EFFECT", "NO_EVIDENCE", "NOT_ASSESSED"]:
        count = len(by_level.get(level, []))
        pct = 100 * count / len(all_preds) if all_preds else 0
        print(f"  {level:<20} {count:>6} ({pct:.1f}%)")

    # Holdout evaluation per evidence level
    print(f"\n{'Level':<20} {'N':>6} {'Holdout%':>10} {'±Std':>8} {'Hits/seed':>10}")
    print("-" * 60)

    for level in ["STRONG_EVIDENCE", "MODERATE_EVIDENCE", "WEAK_EVIDENCE",
                  "ADVERSE_EFFECT", "NO_EVIDENCE", "NOT_ASSESSED"]:
        preds_subset = by_level.get(level, [])
        if len(preds_subset) < 10:
            print(f"  {level:<20} {len(preds_subset):>6}   (too few)")
            continue

        result = holdout_evaluation(preds_subset, expanded_gt, n_seeds=args.seeds)
        print(f"  {level:<20} {len(preds_subset):>6} {result['mean_precision']:>9.1f}% "
              f"±{result['std_precision']:>5.1f}% {result['mean_hits']:>9.1f}")

    # Cross-tabulation: tier × evidence level
    if not args.tier:
        print("\n" + "=" * 70)
        print("CROSS-TABULATION: Tier × Evidence Level")
        print("=" * 70)

        for tier in ["GOLDEN", "HIGH", "MEDIUM", "LOW"]:
            tier_preds = [p for p in all_preds if p.get("confidence_tier") == tier]
            if not tier_preds:
                continue

            print(f"\n{tier} tier ({len(tier_preds)} predictions):")
            tier_by_level: dict[str, list[dict]] = defaultdict(list)
            for p in tier_preds:
                tier_by_level[p["_lit_level"]].append(p)

            for level in ["STRONG_EVIDENCE", "MODERATE_EVIDENCE", "WEAK_EVIDENCE",
                          "NO_EVIDENCE", "NOT_ASSESSED"]:
                subset = tier_by_level.get(level, [])
                if len(subset) < 5:
                    continue
                result = holdout_evaluation(subset, expanded_gt, n_seeds=args.seeds)
                print(f"  {level:<20} n={len(subset):>5}  "
                      f"{result['mean_precision']:>5.1f}% ± {result['std_precision']:.1f}%")

    # Adverse effect analysis
    adverse_preds = by_level.get("ADVERSE_EFFECT", [])
    if adverse_preds:
        print("\n" + "=" * 70)
        print(f"ADVERSE EFFECTS DETECTED: {len(adverse_preds)}")
        print("=" * 70)
        for p in adverse_preds[:20]:
            key = f"{p['drug_name'].lower()}|{p['disease_name'].lower()}"
            entry = lit_cache.get(key, {})
            summary = entry.get("llm_summary", "")
            print(f"  {p['drug_name']} → {p['disease_name']} "
                  f"[{p.get('confidence_tier', '?')}]: {summary}")

    # Score-based analysis (continuous)
    print("\n" + "=" * 70)
    print("EVIDENCE SCORE QUARTILE ANALYSIS")
    print("=" * 70)

    assessed = [p for p in all_preds if p["_lit_level"] != "NOT_ASSESSED"]
    if assessed:
        scores = sorted([p["_lit_score"] for p in assessed])
        q25 = np.percentile(scores, 25)
        q50 = np.percentile(scores, 50)
        q75 = np.percentile(scores, 75)

        quartiles = {
            f"Q1 (>{q75:.1f})": [p for p in assessed if p["_lit_score"] > q75],
            f"Q2 ({q50:.1f}-{q75:.1f})": [p for p in assessed if q50 < p["_lit_score"] <= q75],
            f"Q3 ({q25:.1f}-{q50:.1f})": [p for p in assessed if q25 < p["_lit_score"] <= q50],
            f"Q4 (≤{q25:.1f})": [p for p in assessed if p["_lit_score"] <= q25],
        }

        for label, subset in quartiles.items():
            if len(subset) < 10:
                continue
            result = holdout_evaluation(subset, expanded_gt, n_seeds=args.seeds)
            print(f"  {label:<25} n={len(subset):>5}  "
                  f"{result['mean_precision']:>5.1f}% ± {result['std_precision']:.1f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
