#!/usr/bin/env python3
"""h1100 Step 2: Cluster rank<=30 FILTER predictions for known indications.

Iterates every disease in the predictor's internal GT, runs predict(), and
for each top-30 prediction that (a) lands in FILTER AND (b) is a known
indication (drug_id in internal GT for the disease), records the sub_reason.

Output: counts per sub_reason + examples, written to
data/analysis/h1100_filter_known_scan.json.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from production_predictor import DrugRepurposingPredictor, ConfidenceTier  # type: ignore[import-not-found]

ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("Loading predictor ...")
    predictor = DrugRepurposingPredictor()
    print(f"  Loaded in {time.time()-t0:.0f}s")

    # Eval pool: diseases with both GT AND embedding (these are the ones
    # the deliverable covers).
    eval_diseases = [d for d in predictor.ground_truth if d in predictor.embeddings]
    print(f"Scanning {len(eval_diseases)} diseases ...")

    counter: Counter = Counter()
    examples: Dict[str, List[Dict]] = defaultdict(list)
    total_filter_in_top30 = 0
    total_known_in_top30 = 0
    total_known_filter_in_top30 = 0
    per_disease_fired = 0

    for i, disease_id in enumerate(eval_diseases, 1):
        disease_name = predictor.disease_names.get(disease_id, "")
        try:
            result = predictor.predict(disease_id, top_n=30, include_filtered=True)
        except Exception as e:  # noqa: BLE001
            continue
        if not result or not result.predictions:
            continue

        gt_drugs = predictor.ground_truth.get(disease_id, set())
        fired_here = False
        for p in result.predictions:
            if p.rank > 30:
                continue
            is_filter = p.confidence_tier == ConfidenceTier.FILTER
            is_known = p.drug_id in gt_drugs
            if is_filter:
                total_filter_in_top30 += 1
            if is_known:
                total_known_in_top30 += 1
            if is_filter and is_known:
                total_known_filter_in_top30 += 1
                fired_here = True
                # Retrieve sub_reason via direct tier-assign call
                try:
                    _tier, _, sub_reason = predictor._assign_confidence_tier(
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
                except Exception:
                    sub_reason = "<trace_error>"
                key = sub_reason or "<raw_filter>"
                counter[key] += 1
                if len(examples[key]) < 5:
                    examples[key].append({
                        "drug": p.drug_name,
                        "disease": disease_name,
                        "disease_id": disease_id,
                        "rank": int(p.rank),
                        "freq": int(p.train_frequency),
                        "mech": bool(p.mechanism_support),
                        "has_targets": bool(p.has_targets),
                        "category": p.category,
                    })
        if fired_here:
            per_disease_fired += 1

        if i % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{i}/{len(eval_diseases)}] elapsed={elapsed:.0f}s, "
                  f"hits_so_far={total_known_filter_in_top30}")

    elapsed = time.time() - t0
    print(f"\nScan complete in {elapsed:.0f}s")

    summary = {
        "eval_diseases": len(eval_diseases),
        "total_filter_in_top30": total_filter_in_top30,
        "total_known_in_top30": total_known_in_top30,
        "total_known_filter_in_top30": total_known_filter_in_top30,
        "diseases_with_known_filter_misfire": per_disease_fired,
        "sub_reason_counts": dict(counter.most_common()),
    }

    print("\n" + "=" * 78)
    print("  Known-indication FILTER misfires (rank<=30)")
    print("=" * 78)
    print(f"  Total FILTER@rank<=30: {total_filter_in_top30}")
    print(f"  Total known@rank<=30: {total_known_in_top30}")
    print(f"  Total known+FILTER@rank<=30: {total_known_filter_in_top30}")
    print(f"  Diseases with >=1 misfire: {per_disease_fired}")
    print()
    print(f"  {'sub_reason':<44} {'count':>6}")
    print(f"  {'-'*44} {'-'*6}")
    for sub_reason, n in counter.most_common():
        print(f"  {sub_reason:<44} {n:>6}")

    out = {"summary": summary, "examples": {k: examples[k] for k in counter}}
    out_path = ANALYSIS_DIR / "h1100_filter_known_scan.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
