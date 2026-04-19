#!/usr/bin/env python3
"""h1103: Audit the 543 known-indication FILTER@rank<=30 cases whose
sub_reason is 'rank_over_20'. Cluster by drug class, disease category,
and rank bucket to identify structural patterns that should inform
the h1200 supervised-GNN loss weighting.

Extends h1100's scan: scans top_n=30, captures ALL (no 5-example limit)
rank_over_20 known-indication misfires with full metadata for clustering.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from production_predictor import DrugRepurposingPredictor, ConfidenceTier  # type: ignore[import-not-found]

ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"


def drug_class(drug_name: str) -> str:
    """Suffix-based drug class for clustering — good enough for rank>20 audit."""
    name_lower = drug_name.lower()
    for suffix, cls in [
        ("mab", "biologic_mab"),
        ("cept", "biologic_fusion"),
        ("tinib", "kinase_inhibitor"),
        ("parib", "parp_inhibitor"),
        ("statin", "statin"),
        ("prazole", "ppi"),
        ("sartan", "arb"),
        ("pril", "ace_inhibitor"),
        ("olol", "beta_blocker"),
        ("dipine", "ccb"),
        ("flozin", "sglt2"),
        ("gliptin", "dpp4"),
        ("glutide", "glp1"),
        ("caine", "local_anesthetic"),
        ("cillin", "beta_lactam_abx"),
        ("mycin", "macrolide_or_other_abx"),
        ("oxacin", "quinolone_abx"),
        ("cycline", "tetracycline_abx"),
        ("azole", "azole_antifungal"),
        ("vir", "antiviral"),
        ("oxetine", "ssri"),
        ("pram", "ssri"),
        ("triptan", "triptan"),
        ("sone", "corticosteroid"),
        ("olone", "corticosteroid"),
    ]:
        if name_lower.endswith(suffix):
            return cls
    return "other_sm"


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("Loading predictor ...")
    predictor = DrugRepurposingPredictor()
    print(f"  Loaded in {time.time()-t0:.0f}s")

    eval_diseases = [d for d in predictor.ground_truth if d in predictor.embeddings]
    print(f"Scanning {len(eval_diseases)} diseases (top_n=30, rank_over_20 only) ...")

    rank_over_20_cases: List[Dict] = []

    for i, disease_id in enumerate(eval_diseases, 1):
        disease_name = predictor.disease_names.get(disease_id, "")
        try:
            result = predictor.predict(disease_id, top_n=30, include_filtered=True)
        except Exception:
            continue
        if not result or not result.predictions:
            continue

        gt_drugs = predictor.ground_truth.get(disease_id, set())
        for p in result.predictions:
            if p.rank > 30:
                continue
            if p.confidence_tier != ConfidenceTier.FILTER:
                continue
            if p.drug_id not in gt_drugs:
                continue
            try:
                assign = predictor._assign_confidence_tier(
                    rank=p.rank,
                    train_frequency=p.train_frequency,
                    mechanism_support=p.mechanism_support,
                    has_targets=p.has_targets,
                    disease_tier=p.disease_tier,
                    category=p.category,
                    drug_name=p.drug_name,
                    disease_name=result.disease_name,
                    drug_id=p.drug_id,
                    knn_score=p.knn_score,
                    disease_id=disease_id,
                )
                sub_reason = assign[2]
            except Exception:
                sub_reason = "<error>"
            if sub_reason != "rank_over_20":
                continue

            rank_over_20_cases.append({
                "drug": p.drug_name,
                "drug_id": p.drug_id,
                "drug_class": drug_class(p.drug_name),
                "disease": disease_name,
                "disease_id": disease_id,
                "disease_category": p.category,
                "rank": int(p.rank),
                "freq": int(p.train_frequency),
                "mech": bool(p.mechanism_support),
                "has_targets": bool(p.has_targets),
                "knn_score": round(float(p.knn_score), 4),
            })

        if i % 200 == 0:
            elapsed = time.time() - t0
            print(f"  [{i}/{len(eval_diseases)}] elapsed={elapsed:.0f}s, "
                  f"cases={len(rank_over_20_cases)}")

    print(f"\nScan complete in {time.time()-t0:.0f}s, total={len(rank_over_20_cases)}")

    # Cluster analysis
    by_class = Counter(c["drug_class"] for c in rank_over_20_cases)
    by_category = Counter(c["disease_category"] for c in rank_over_20_cases)
    by_rank_bucket = Counter()
    for c in rank_over_20_cases:
        if c["rank"] <= 25:
            by_rank_bucket["21-25"] += 1
        else:
            by_rank_bucket["26-30"] += 1

    by_class_x_category = Counter()
    for c in rank_over_20_cases:
        by_class_x_category[(c["drug_class"], c["disease_category"])] += 1

    mech_counts = Counter(("mech=True" if c["mech"] else "mech=False") for c in rank_over_20_cases)
    freq_buckets = Counter()
    for c in rank_over_20_cases:
        if c["freq"] <= 2:
            freq_buckets["f<=2"] += 1
        elif c["freq"] <= 9:
            freq_buckets["f=3-9"] += 1
        else:
            freq_buckets["f>=10"] += 1

    print("\n" + "=" * 78)
    print("  Drug class distribution")
    print("=" * 78)
    for cls, n in by_class.most_common():
        print(f"  {cls:<30} {n:>4}")

    print("\n" + "=" * 78)
    print("  Disease category distribution")
    print("=" * 78)
    for cat, n in by_category.most_common():
        print(f"  {cat:<30} {n:>4}")

    print("\n" + "=" * 78)
    print("  Rank bucket distribution")
    print("=" * 78)
    for bucket, n in by_rank_bucket.most_common():
        print(f"  {bucket:<10} {n:>4}")

    print("\n" + "=" * 78)
    print("  Mechanism support")
    print("=" * 78)
    for k, n in mech_counts.most_common():
        print(f"  {k:<12} {n:>4}")

    print("\n" + "=" * 78)
    print("  Train frequency buckets")
    print("=" * 78)
    for k, n in freq_buckets.most_common():
        print(f"  {k:<12} {n:>4}")

    print("\n" + "=" * 78)
    print("  Top 15 (drug_class x disease_category) buckets")
    print("=" * 78)
    for (cls, cat), n in by_class_x_category.most_common(15):
        print(f"  {cls:<25} x {cat:<18} {n:>4}")

    out = {
        "summary": {
            "total_cases": len(rank_over_20_cases),
            "by_drug_class": dict(by_class.most_common()),
            "by_disease_category": dict(by_category.most_common()),
            "by_rank_bucket": dict(by_rank_bucket.most_common()),
            "by_mechanism": dict(mech_counts.most_common()),
            "by_freq_bucket": dict(freq_buckets.most_common()),
            "top_class_x_category": [
                {"drug_class": cls, "disease_category": cat, "n": n}
                for (cls, cat), n in by_class_x_category.most_common(30)
            ],
        },
        "cases": rank_over_20_cases,
    }
    out_path = ANALYSIS_DIR / "h1103_rank_over_20_audit.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
