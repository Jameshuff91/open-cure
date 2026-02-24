#!/usr/bin/env python3
"""
h769: Mine literature evidence for non-hierarchy GOLDEN/HIGH GT gaps.

Batch mines PubMed + ClinicalTrials.gov for GOLDEN/HIGH predictions
not in expanded_ground_truth.json.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.external_validation import PubMedAPI, ClinicalTrialsAPI

PROJECT_ROOT = Path(__file__).parent.parent
GAPS_FILE = PROJECT_ROOT / "data" / "analysis" / "h769_non_hierarchy_gt_gaps.json"
VERIFIED_FILE = PROJECT_ROOT / "data" / "analysis" / "h769_literature_verification.json"

pubmed = PubMedAPI()
ctgov = ClinicalTrialsAPI()


def mine_pair(drug: str, disease: str) -> dict:
    """Mine literature evidence for a single drug-disease pair."""
    # PubMed search - returns PubMedEvidence
    try:
        pm_ev = pubmed.search(drug, disease)
        pm_count = pm_ev.publication_count
        pm_recent = pm_ev.recent_count
    except Exception:
        pm_count = 0
        pm_recent = 0

    # ClinicalTrials.gov - returns ClinicalTrialEvidence
    try:
        ct_ev = ctgov.search(drug, disease)
        trial_count = ct_ev.trial_count
        has_phase3 = any(
            p in phase for phase in ct_ev.phases
            for p in ["Phase 3", "Phase 4", "PHASE3", "PHASE4"]
        ) if ct_ev.phases else False
    except Exception:
        trial_count = 0
        has_phase3 = False

    # Determine evidence level
    if has_phase3 or (pm_count >= 20 and pm_recent >= 5):
        level = "STRONG"
    elif trial_count > 0 or (pm_count >= 5 and pm_recent >= 2):
        level = "MODERATE"
    elif pm_count >= 1:
        level = "WEAK"
    else:
        level = "NONE"

    return {
        "pubmed": pm_count,
        "recent": pm_recent,
        "trials": trial_count,
        "phase3": has_phase3,
        "level": level,
    }


def main():
    with open(GAPS_FILE) as f:
        data = json.load(f)
    gaps = data["gaps"]

    # Load already verified
    if VERIFIED_FILE.exists():
        with open(VERIFIED_FILE) as f:
            verified = json.load(f)
    else:
        verified = []

    verified_pairs = set((v["drug_id"], v["disease_id"]) for v in verified)
    remaining = [g for g in gaps if (g["drug_id"], g["disease_id"]) not in verified_pairs]

    # Prioritize: GOLDEN first, then HIGH by rank
    golden = [g for g in remaining if g["tier"] == "GOLDEN"]
    high = sorted([g for g in remaining if g["tier"] == "HIGH"], key=lambda x: x["rank"])
    batch = golden + high

    print(f"Total gaps: {len(gaps)}, Already verified: {len(verified_pairs)}, Remaining: {len(batch)}")
    print(f"  GOLDEN: {len(golden)}, HIGH: {len(high)}")

    # Mine in batches with progress tracking
    new_results = []
    try:
        for i, gap in enumerate(batch):
            drug = gap["drug"]
            disease = gap["disease"]
            print(f"[{i+1}/{len(batch)}] {drug} → {disease} ...", end=" ", flush=True)

            result = mine_pair(drug, disease)
            result["drug"] = drug
            result["disease"] = disease
            result["drug_id"] = gap["drug_id"]
            result["disease_id"] = gap["disease_id"]
            result["tier"] = gap["tier"]
            result["sub_reason"] = gap.get("sub_reason", "")
            result["rank"] = gap["rank"]
            result["category"] = gap.get("category", "")

            print(f"{result['level']} (PM:{result['pubmed']}, T:{result['trials']})")
            new_results.append(result)
            verified.append(result)

            # Save every 25
            if (i + 1) % 25 == 0:
                with open(VERIFIED_FILE, "w") as f:
                    json.dump(verified, f, indent=2)
                print(f"  [Saved {len(verified)} results]")

    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving progress...")
    finally:
        with open(VERIFIED_FILE, "w") as f:
            json.dump(verified, f, indent=2)
        print(f"\nSaved {len(verified)} total results ({len(new_results)} new)")

    # Summary
    levels = {}
    for r in new_results:
        levels[r["level"]] = levels.get(r["level"], 0) + 1
    print(f"\nNew results by level: {levels}")

    strong = [r for r in verified if r["level"] == "STRONG"]
    print(f"\nAll STRONG evidence ({len(strong)}):")
    for r in strong:
        print(f"  {r['drug']} → {r['disease']} (PM:{r['pubmed']}, T:{r['trials']}, tier={r['tier']})")


if __name__ == "__main__":
    main()
