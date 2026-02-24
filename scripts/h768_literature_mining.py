#!/usr/bin/env python3
"""
h768: Literature mining for MEDIUM sub-reason validation.

Runs PubMed + ClinicalTrials.gov literature mining on 277 MEDIUM predictions
that belong to sub-reasons with small n or high variance.
"""

import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.literature_miner import LiteratureMiner

PROJECT_ROOT = Path(__file__).parent.parent
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "analysis" / "h768_medium_predictions_to_mine.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "analysis" / "h768_literature_results.json"


def main():
    # Load predictions
    with open(PREDICTIONS_PATH) as f:
        preds = json.load(f)
    print(f"Loaded {len(preds)} MEDIUM predictions to mine")

    # Count by sub-reason
    reasons = Counter(p['sub_reason'] for p in preds)
    for r, c in reasons.most_common():
        print(f"  {r}: {c}")

    # Convert to mine_batch format
    pairs = [{"drug_name": p["drug"], "disease_name": p["disease"]} for p in preds]

    # Initialize miner and run batch
    miner = LiteratureMiner(use_llm=False)
    evidence_list = miner.mine_batch(pairs, save_every=25)

    # Build results with sub-reason metadata
    results = []
    for pred, evidence in zip(preds, evidence_list):
        results.append({
            'drug': pred['drug'],
            'disease': pred['disease'],
            'sub_reason': pred['sub_reason'],
            'rank': pred.get('rank'),
            'mechanism': pred.get('mechanism'),
            'frequency': pred.get('frequency'),
            'evidence_level': evidence.evidence_level,
            'evidence_score': evidence.evidence_score,
            'pubmed_total': evidence.pubmed_total,
            'pubmed_clinical_trial': evidence.pubmed_clinical_trial,
            'trial_count': evidence.trial_count,
            'has_phase3_plus': evidence.has_phase3_plus,
        })

    # Save results
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Evidence Level Distribution ===")
    levels = Counter(r['evidence_level'] for r in results)
    for level, count in levels.most_common():
        print(f"  {level}: {count}")

    # Summary by sub-reason x evidence level
    print(f"\n=== Sub-Reason x Evidence Level ===")
    reason_evidence: dict[str, Counter] = {}
    for r in results:
        sr = r['sub_reason']
        el = r['evidence_level']
        if sr not in reason_evidence:
            reason_evidence[sr] = Counter()
        reason_evidence[sr][el] += 1

    for sr in sorted(reason_evidence.keys()):
        ev_counts = reason_evidence[sr]
        total = sum(ev_counts.values())
        strong = ev_counts.get('STRONG_EVIDENCE', 0)
        moderate = ev_counts.get('MODERATE_EVIDENCE', 0)
        weak = ev_counts.get('WEAK_EVIDENCE', 0)
        none_ev = ev_counts.get('NO_EVIDENCE', 0)
        print(f"  {sr} (n={total}): STRONG={strong} MODERATE={moderate} WEAK={weak} NO={none_ev}")
        if total > 0:
            strong_pct = 100 * strong / total
            has_evidence_pct = 100 * (strong + moderate) / total
            print(f"    Strong%: {strong_pct:.1f}%, HasEvidence%: {has_evidence_pct:.1f}%")


if __name__ == "__main__":
    main()
