#!/usr/bin/env python3
"""Quick debug: why do some FILTER predictions have None reason but high freq?"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from production_predictor import DrugRepurposingPredictor

predictor = DrugRepurposingPredictor()
all_diseases = [d for d in predictor.ground_truth if d in predictor.embeddings]

reason_detail = Counter()

for disease_id in all_diseases[:50]:
    disease_name = predictor.disease_names.get(disease_id, disease_id)
    try:
        result = predictor.predict(disease_name, top_n=30, include_filtered=True)
    except Exception:
        continue

    for pred in result.predictions:
        if pred.confidence_tier.name != "FILTER":
            continue
        if pred.category_specific_tier is not None:
            continue  # Only care about None-reason

        rank = pred.rank
        freq = pred.train_frequency
        has_targets = pred.has_targets
        has_mech = pred.mechanism_support
        cat = pred.category

        if rank > 20:
            bucket = "rank>20"
        elif not has_targets:
            bucket = "no_targets"
        elif freq <= 2 and not has_mech:
            bucket = "low_freq_no_mech"
        elif cat == 'metabolic':
            # Check if it's a corticosteroid
            bucket = f"metabolic_corticosteroid?"
        else:
            bucket = f"UNKNOWN(rank={rank},freq={freq},targets={has_targets},mech={has_mech},cat={cat})"

        reason_detail[bucket] += 1

print("=== None-reason FILTER predictions breakdown ===")
for bucket, count in reason_detail.most_common(20):
    print(f"  {bucket}: {count}")
