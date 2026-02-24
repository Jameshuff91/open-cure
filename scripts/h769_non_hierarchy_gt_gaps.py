#!/usr/bin/env python3
"""
h769: Find non-hierarchy GOLDEN/HIGH predictions missing from expanded GT.
Literature-mine top candidates and add confirmed pairs.
"""

import json
import sys
import time
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.production_predictor import DrugRepurposingPredictor, ConfidenceTier


def main():
    # Load expanded GT
    egt_path = Path(__file__).parent.parent / "data" / "reference" / "expanded_ground_truth.json"
    with open(egt_path) as f:
        egt = json.load(f)

    egt_set = set()
    for disease_id, drug_ids in egt.items():
        for drug_id in drug_ids:
            egt_set.add((drug_id, disease_id))
    print(f"Expanded GT: {len(egt_set)} pairs across {len(egt)} diseases")

    # Initialize predictor
    predictor = DrugRepurposingPredictor()

    # Get all diseases with embeddings
    all_diseases = [d for d in predictor.embeddings if d in predictor.disease_names]
    print(f"Diseases with embeddings: {len(all_diseases)}")

    # Collect GOLDEN/HIGH predictions not in expanded GT
    non_hierarchy_gaps = []
    hierarchy_gaps = []
    total_golden_high = 0
    total_in_gt = 0
    tier_counts = Counter()

    start = time.time()
    for idx, disease_id in enumerate(all_diseases):
        disease_name = predictor.disease_names.get(disease_id, disease_id)

        try:
            result = predictor.predict(disease_name, top_n=30, include_filtered=True)
        except Exception as e:
            continue

        if (idx + 1) % 200 == 0:
            elapsed = time.time() - start
            print(f"  Processed {idx+1}/{len(all_diseases)} ({elapsed:.0f}s) - gaps so far: {len(non_hierarchy_gaps)} non-hier, {len(hierarchy_gaps)} hier")

        for pred in result.predictions:
            tier_val = pred.confidence_tier.value if hasattr(pred.confidence_tier, 'value') else pred.confidence_tier
            if tier_val not in ('GOLDEN', 'HIGH'):
                continue

            total_golden_high += 1
            tier_counts[tier_val] += 1
            pair = (pred.drug_id, disease_id)

            if pair in egt_set:
                total_in_gt += 1
                continue

            sub = getattr(pred, 'confidence_sub_reason', '') or ''
            is_hierarchy = 'hierarchy' in sub.lower()

            entry = {
                'drug': pred.drug_name,
                'disease': disease_name,
                'drug_id': pred.drug_id,
                'disease_id': disease_id,
                'tier': tier_val,
                'sub_reason': sub,
                'rank': getattr(pred, 'rank', None),
                'knn_score': round(getattr(pred, 'knn_score', 0), 6),
            }

            if is_hierarchy:
                hierarchy_gaps.append(entry)
            else:
                non_hierarchy_gaps.append(entry)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.0f}s")
    print(f"\nTotal GOLDEN/HIGH predictions: {total_golden_high}")
    print(f"  GOLDEN: {tier_counts['GOLDEN']}, HIGH: {tier_counts['HIGH']}")
    print(f"  In expanded GT: {total_in_gt} ({100*total_in_gt/max(total_golden_high,1):.1f}%)")
    print(f"  NOT in GT (hierarchy): {len(hierarchy_gaps)}")
    print(f"  NOT in GT (non-hierarchy): {len(non_hierarchy_gaps)}")

    # Break down by sub_reason
    print(f"\nNon-hierarchy GT gaps by sub_reason:")
    sr_counts = Counter(g['sub_reason'] for g in non_hierarchy_gaps)
    for sr, count in sr_counts.most_common(30):
        tier_dist = Counter(g['tier'] for g in non_hierarchy_gaps if g['sub_reason'] == sr)
        print(f"  {sr}: {count} ({dict(tier_dist)})")

    print(f"\nHierarchy GT gaps by sub_reason:")
    sr_counts_h = Counter(g['sub_reason'] for g in hierarchy_gaps)
    for sr, count in sr_counts_h.most_common(20):
        print(f"  {sr}: {count}")

    # Show top candidates for literature mining (non-hierarchy, highest rank)
    non_hierarchy_gaps.sort(key=lambda x: (0 if x['tier'] == 'GOLDEN' else 1, x.get('rank') or 99))

    print(f"\nTop non-hierarchy GOLDEN gaps (for literature mining):")
    golden_gaps = [g for g in non_hierarchy_gaps if g['tier'] == 'GOLDEN']
    for g in golden_gaps[:30]:
        print(f"  {g['drug']} → {g['disease']} (rank {g.get('rank')}, {g['sub_reason']})")

    print(f"\nTop non-hierarchy HIGH gaps (for literature mining):")
    high_gaps = [g for g in non_hierarchy_gaps if g['tier'] == 'HIGH']
    for g in high_gaps[:30]:
        print(f"  {g['drug']} → {g['disease']} (rank {g.get('rank')}, {g['sub_reason']})")

    # Save full results
    output_path = Path(__file__).parent.parent / "data" / "analysis" / "h769_non_hierarchy_gt_gaps.json"
    with open(output_path, 'w') as f:
        json.dump({
            'total_golden_high': total_golden_high,
            'total_in_gt': total_in_gt,
            'non_hierarchy_gaps': non_hierarchy_gaps,
            'hierarchy_gaps': hierarchy_gaps,
        }, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == '__main__':
    main()
