#!/usr/bin/env python3
"""
h573: kNN Score Gap / Absolute Score as Prediction Confidence Signal

The kNN score for a drug = weighted frequency across k=20 neighbors.
Rank captures ordinal position but loses information about margin.

Test whether:
1. Absolute kNN score predicts holdout hit probability
2. Score gap (drug score - next drug score) predicts precision
3. Normalized score (drug / max score for disease) adds signal beyond rank

Key: this must add information BEYOND what rank already captures.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from production_predictor import (
    ConfidenceTier,
    DrugRepurposingPredictor,
    DISEASE_HIERARCHY_GROUPS,
)


def split_diseases(all_diseases, seed, train_ratio=0.8):
    rng = np.random.RandomState(seed)
    shuffled = list(all_diseases)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def recompute_gt_structures(predictor, train_disease_ids):
    """Minimal recompute for holdout evaluation."""
    from production_predictor import extract_cancer_types

    originals = {
        "drug_train_freq": dict(predictor.drug_train_freq),
        "drug_to_diseases": dict(predictor.drug_to_diseases),
        "drug_cancer_types": dict(predictor.drug_cancer_types),
        "drug_disease_groups": dict(predictor.drug_disease_groups),
        "train_diseases": list(predictor.train_diseases),
        "train_embeddings": predictor.train_embeddings.copy(),
        "train_disease_categories": dict(predictor.train_disease_categories),
    }

    new_freq = defaultdict(int)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            for drug_id in predictor.ground_truth[disease_id]:
                new_freq[drug_id] += 1
    predictor.drug_train_freq = dict(new_freq)

    new_d2d = defaultdict(set)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            for drug_id in predictor.ground_truth[disease_id]:
                new_d2d[drug_id].add(disease_name)
    predictor.drug_to_diseases = dict(new_d2d)

    new_cancer_types = {}
    for drug_id, diseases in new_d2d.items():
        cancer_types = set()
        for d in diseases:
            cancer_types.update(extract_cancer_types(d))
        if cancer_types:
            new_cancer_types[drug_id] = cancer_types
    predictor.drug_cancer_types = new_cancer_types

    new_groups = defaultdict(set)
    for drug_id, diseases in new_d2d.items():
        for d in diseases:
            for group_name, group_data in DISEASE_HIERARCHY_GROUPS.items():
                for disease_pattern in group_data.get("diseases", []):
                    if disease_pattern.lower() in d.lower():
                        new_groups[drug_id].add(group_name)
    predictor.drug_disease_groups = dict(new_groups)

    train_disease_list = [d for d in predictor.train_diseases if d in train_disease_ids]
    predictor.train_diseases = train_disease_list
    indices = [i for i, d in enumerate(originals["train_diseases"]) if d in train_disease_ids]
    predictor.train_embeddings = originals["train_embeddings"][indices]
    predictor.train_disease_categories = {
        d: originals["train_disease_categories"][d]
        for d in train_disease_list
        if d in originals["train_disease_categories"]
    }

    return originals


def restore_gt_structures(predictor, originals):
    for key, val in originals.items():
        setattr(predictor, key, val)


def main():
    print("=" * 70)
    print("h573: kNN Score Gap / Absolute Score as Confidence Signal")
    print("=" * 70)

    predictor = DrugRepurposingPredictor()
    gt_diseases = [d for d in predictor.ground_truth if d in predictor.embeddings]
    print(f"\nGT diseases with embeddings: {len(gt_diseases)}")

    # Step 1: Full-data analysis — understand score distribution
    print("\n--- Step 1: Score Distribution Analysis ---")

    all_scores = []
    all_norm_scores = []
    rank_to_scores = defaultdict(list)

    for disease_id in gt_diseases[:50]:  # Sample for speed
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        result = predictor.predict(disease_name, k=20, top_n=30, include_filtered=True)

        max_score = max(p.knn_score for p in result.predictions) if result.predictions else 1
        for pred in result.predictions:
            all_scores.append(pred.knn_score)
            norm = pred.knn_score / max_score if max_score > 0 else 0
            all_norm_scores.append(norm)
            rank_to_scores[pred.rank].append(pred.knn_score)

    print(f"Score range: [{min(all_scores):.4f}, {max(all_scores):.4f}]")
    print(f"Score distribution: mean={np.mean(all_scores):.4f}, median={np.median(all_scores):.4f}")
    print(f"\nNorm score by rank (sample of 50 diseases):")
    for rank in [1, 2, 3, 5, 10, 15, 20, 25, 30]:
        if rank in rank_to_scores:
            scores = rank_to_scores[rank]
            print(f"  Rank {rank:2d}: mean={np.mean(scores):.4f}, std={np.std(scores):.4f}, n={len(scores)}")

    # Step 2: Holdout analysis — does score predict precision within rank?
    print("\n--- Step 2: Holdout Precision by Score Quantile (Within Rank) ---")

    seeds = [42, 123, 456, 789, 1337]

    # Collect all holdout predictions with their scores and hit status
    holdout_data = []  # list of (rank, score, norm_score, is_hit, tier)

    for seed in seeds:
        train, holdout = split_diseases(gt_diseases, seed)
        train_set = set(train)
        originals = recompute_gt_structures(predictor, train_set)

        for disease_id in holdout:
            if disease_id not in predictor.embeddings:
                continue
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            result = predictor.predict(disease_name, k=20, top_n=30, include_filtered=True)
            gt_drugs = set(predictor.ground_truth.get(disease_id, []))
            if not gt_drugs:
                continue

            max_score = max(p.knn_score for p in result.predictions) if result.predictions else 1

            for pred in result.predictions:
                is_hit = pred.drug_id in gt_drugs
                norm = pred.knn_score / max_score if max_score > 0 else 0
                holdout_data.append({
                    'rank': pred.rank,
                    'score': pred.knn_score,
                    'norm_score': norm,
                    'hit': is_hit,
                    'tier': pred.confidence_tier.value,
                    'seed': seed,
                })

        restore_gt_structures(predictor, originals)

    print(f"Total holdout predictions: {len(holdout_data)}")

    # Analysis 1: Does norm_score predict hit probability within rank?
    print("\n--- Analysis 1: Norm Score vs Hit Rate by Rank ---")

    for rank_range, rank_min, rank_max in [("1-5", 1, 5), ("6-10", 6, 10), ("11-20", 11, 20), ("21-30", 21, 30)]:
        preds_in_range = [d for d in holdout_data if rank_min <= d['rank'] <= rank_max]
        if not preds_in_range:
            continue

        # Split by norm_score median within this rank range
        norm_scores = [d['norm_score'] for d in preds_in_range]
        median_norm = np.median(norm_scores)

        high_score = [d for d in preds_in_range if d['norm_score'] >= median_norm]
        low_score = [d for d in preds_in_range if d['norm_score'] < median_norm]

        h_hit_rate = sum(d['hit'] for d in high_score) / len(high_score) * 100 if high_score else 0
        l_hit_rate = sum(d['hit'] for d in low_score) / len(low_score) * 100 if low_score else 0

        print(f"  Rank {rank_range}: high-score={h_hit_rate:.1f}% (n={len(high_score)}), "
              f"low-score={l_hit_rate:.1f}% (n={len(low_score)}), delta={h_hit_rate - l_hit_rate:+.1f}pp")

    # Analysis 2: Norm score quartiles across all predictions
    print("\n--- Analysis 2: Hit Rate by Norm Score Quartile ---")
    all_norms = np.array([d['norm_score'] for d in holdout_data])
    quartiles = np.percentile(all_norms, [25, 50, 75])

    for label, lo, hi in [("Q1 (lowest)", 0, quartiles[0]),
                           ("Q2", quartiles[0], quartiles[1]),
                           ("Q3", quartiles[1], quartiles[2]),
                           ("Q4 (highest)", quartiles[2], 1.01)]:
        q_preds = [d for d in holdout_data if lo <= d['norm_score'] < hi]
        if q_preds:
            hit_rate = sum(d['hit'] for d in q_preds) / len(q_preds) * 100
            mean_rank = np.mean([d['rank'] for d in q_preds])
            print(f"  {label}: {hit_rate:.1f}% (n={len(q_preds)}, mean_rank={mean_rank:.1f})")

    # Analysis 3: Within MEDIUM tier, does score predict precision?
    print("\n--- Analysis 3: MEDIUM Tier — Score vs Hit Rate ---")
    medium_preds = [d for d in holdout_data if d['tier'] == 'MEDIUM']
    if medium_preds:
        med_norms = np.array([d['norm_score'] for d in medium_preds])
        med_quartiles = np.percentile(med_norms, [25, 50, 75])

        for label, lo, hi in [("Q1 (lowest)", 0, med_quartiles[0]),
                               ("Q2", med_quartiles[0], med_quartiles[1]),
                               ("Q3", med_quartiles[1], med_quartiles[2]),
                               ("Q4 (highest)", med_quartiles[2], 1.01)]:
            q_preds = [d for d in medium_preds if lo <= d['norm_score'] < hi]
            if q_preds:
                hit_rate = sum(d['hit'] for d in q_preds) / len(q_preds) * 100
                mean_rank = np.mean([d['rank'] for d in q_preds])
                print(f"  {label}: {hit_rate:.1f}% (n={len(q_preds)}, mean_rank={mean_rank:.1f})")

    # Analysis 4: Score gap (drug score - next drug score)
    print("\n--- Analysis 4: Score Gap Analysis ---")
    # Recompute with score gaps
    gap_data = []
    for seed in seeds:
        train, holdout = split_diseases(gt_diseases, seed)
        train_set = set(train)
        originals = recompute_gt_structures(predictor, train_set)

        for disease_id in holdout:
            if disease_id not in predictor.embeddings:
                continue
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            result = predictor.predict(disease_name, k=20, top_n=30, include_filtered=True)
            gt_drugs = set(predictor.ground_truth.get(disease_id, []))
            if not gt_drugs:
                continue

            # Sort by knn_score descending
            sorted_preds = sorted(result.predictions, key=lambda p: p.knn_score, reverse=True)

            for i, pred in enumerate(sorted_preds):
                next_score = sorted_preds[i + 1].knn_score if i + 1 < len(sorted_preds) else 0
                gap = pred.knn_score - next_score
                is_hit = pred.drug_id in gt_drugs
                gap_data.append({
                    'rank': pred.rank,
                    'gap': gap,
                    'hit': is_hit,
                    'tier': pred.confidence_tier.value,
                })

        restore_gt_structures(predictor, originals)

    # Split by gap size
    all_gaps = np.array([d['gap'] for d in gap_data])
    gap_quartiles = np.percentile(all_gaps, [25, 50, 75])

    print(f"Score gap distribution: mean={np.mean(all_gaps):.4f}, median={np.median(all_gaps):.4f}")

    for label, lo, hi in [("Q1 (smallest gap)", -0.01, gap_quartiles[0]),
                           ("Q2", gap_quartiles[0], gap_quartiles[1]),
                           ("Q3", gap_quartiles[1], gap_quartiles[2]),
                           ("Q4 (largest gap)", gap_quartiles[2], 999)]:
        q_preds = [d for d in gap_data if lo <= d['gap'] < hi]
        if q_preds:
            hit_rate = sum(d['hit'] for d in q_preds) / len(q_preds) * 100
            mean_rank = np.mean([d['rank'] for d in q_preds])
            print(f"  {label}: {hit_rate:.1f}% (n={len(q_preds)}, mean_rank={mean_rank:.1f})")

    # Within MEDIUM
    medium_gaps = [d for d in gap_data if d['tier'] == 'MEDIUM']
    if medium_gaps:
        med_all_gaps = np.array([d['gap'] for d in medium_gaps])
        med_gap_quartiles = np.percentile(med_all_gaps, [25, 50, 75])

        print(f"\nMEDIUM tier score gap:")
        for label, lo, hi in [("Q1 (smallest)", -0.01, med_gap_quartiles[0]),
                               ("Q2", med_gap_quartiles[0], med_gap_quartiles[1]),
                               ("Q3", med_gap_quartiles[1], med_gap_quartiles[2]),
                               ("Q4 (largest)", med_gap_quartiles[2], 999)]:
            q_preds = [d for d in medium_gaps if lo <= d['gap'] < hi]
            if q_preds:
                hit_rate = sum(d['hit'] for d in q_preds) / len(q_preds) * 100
                mean_rank = np.mean([d['rank'] for d in q_preds])
                print(f"  {label}: {hit_rate:.1f}% (n={len(q_preds)}, mean_rank={mean_rank:.1f})")

    # Correlation analysis
    print("\n--- Analysis 5: Correlation Summary ---")
    scores_arr = np.array([d['norm_score'] for d in holdout_data])
    ranks_arr = np.array([d['rank'] for d in holdout_data], dtype=float)
    hits_arr = np.array([d['hit'] for d in holdout_data], dtype=float)

    r_score_hit = np.corrcoef(scores_arr, hits_arr)[0, 1]
    r_rank_hit = np.corrcoef(ranks_arr, hits_arr)[0, 1]
    r_score_rank = np.corrcoef(scores_arr, ranks_arr)[0, 1]

    print(f"r(norm_score, hit): {r_score_hit:.4f}")
    print(f"r(rank, hit): {r_rank_hit:.4f}")
    print(f"r(norm_score, rank): {r_score_rank:.4f}")

    if medium_preds:
        med_scores = np.array([d['norm_score'] for d in medium_preds])
        med_ranks = np.array([d['rank'] for d in medium_preds], dtype=float)
        med_hits = np.array([d['hit'] for d in medium_preds], dtype=float)

        r_med_score = np.corrcoef(med_scores, med_hits)[0, 1]
        r_med_rank = np.corrcoef(med_ranks, med_hits)[0, 1]
        print(f"\nWithin MEDIUM:")
        print(f"r(norm_score, hit): {r_med_score:.4f}")
        print(f"r(rank, hit): {r_med_rank:.4f}")

    # Save
    results = {
        "hypothesis": "h573",
        "total_holdout_predictions": len(holdout_data),
        "r_score_hit": round(float(r_score_hit), 4),
        "r_rank_hit": round(float(r_rank_hit), 4),
    }
    with open("data/analysis/h573_output.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to data/analysis/h573_output.json")


if __name__ == "__main__":
    main()
