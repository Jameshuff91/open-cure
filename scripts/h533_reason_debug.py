#!/usr/bin/env python3
"""Quick debug: what category_specific_tier values exist for FILTER predictions?"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from production_predictor import DrugRepurposingPredictor

predictor = DrugRepurposingPredictor()
all_diseases = [d for d in predictor.ground_truth if d in predictor.embeddings]

reason_counter = Counter()
# Also check the actual reason for "standard_filter" / None predictions
reason_with_rank = Counter()

for disease_id in all_diseases[:50]:  # Sample 50 diseases
    disease_name = predictor.disease_names.get(disease_id, disease_id)
    try:
        result = predictor.predict(disease_name, top_n=30, include_filtered=True)
    except Exception:
        continue

    for pred in result.predictions:
        if pred.confidence_tier.name != "FILTER":
            continue
        reason = pred.category_specific_tier
        reason_counter[reason] += 1

        # For None reasons, check what ACTUALLY triggered FILTER
        if reason is None:
            rank = pred.knn_rank if hasattr(pred, 'knn_rank') else 0
            freq = predictor.drug_train_freq.get(pred.drug_id, 0)
            has_targets = bool(predictor.drug_targets.get(pred.drug_id))
            if rank > 20:
                reason_with_rank[f"None->rank>20 (rank={rank})"] += 1
            elif not has_targets:
                reason_with_rank[f"None->no_targets"] += 1
            elif freq <= 2:
                reason_with_rank[f"None->low_freq (freq={freq})"] += 1
            else:
                reason_with_rank[f"None->OTHER (rank={rank}, freq={freq}, targets={'Y' if has_targets else 'N'})"] += 1

print("=== FILTER category_specific_tier values ===")
for reason, count in reason_counter.most_common():
    print(f"  {reason!r}: {count}")

print("\n=== Debug for None-reason FILTER predictions ===")
for reason, count in reason_with_rank.most_common():
    print(f"  {reason}: {count}")
