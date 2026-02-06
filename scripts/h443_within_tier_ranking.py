#!/usr/bin/env python3
"""
h443: Within-Tier Ranking Using TransE + kNN Score Composite

h405/h439 showed TransE consilience is a strong signal (+13.6pp for MEDIUM tier).
Currently TransE is a boolean flag (in top-30 or not).

Hypothesis: A composite score (kNN * alpha + TransE * (1-alpha)) for within-tier
ranking will create a >10pp precision gap between top-half and bottom-half of
each tier, enabling better clinical prioritization.

Method:
1. For each prediction, compute both kNN score and TransE score
2. Create composite: alpha * kNN_norm + (1-alpha) * TransE_norm
3. Within each tier, rank by composite
4. Compare top-half vs bottom-half precision by tier
5. Test multiple alpha values (0.3, 0.5, 0.7, 0.9)

Success criteria:
- Top-half vs bottom-half precision gap > 10pp for at least one tier
- This should hold on holdout validation
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from production_predictor import (
    ConfidenceTier,
    DrugRepurposingPredictor,
    DISEASE_HIERARCHY_GROUPS,
    extract_cancer_types,
)


def get_transe_score(predictor: DrugRepurposingPredictor, drug_id: str, disease_id: str) -> Optional[float]:
    """Get raw TransE score for a drug-disease pair."""
    if (
        predictor.transe_entity_emb is None
        or predictor.transe_entity2id is None
        or predictor.transe_treat_vec is None
    ):
        return None

    if drug_id not in predictor.transe_entity2id or disease_id not in predictor.transe_entity2id:
        return None

    drug_emb = predictor.transe_entity_emb[predictor.transe_entity2id[drug_id]]
    disease_emb = predictor.transe_entity_emb[predictor.transe_entity2id[disease_id]]
    score = -float(np.linalg.norm(drug_emb + predictor.transe_treat_vec - disease_emb))
    return score


def split_diseases(all_diseases: List[str], seed: int, train_ratio: float = 0.8) -> Tuple[List[str], List[str]]:
    """Split diseases into train/holdout sets."""
    rng = np.random.RandomState(seed)
    shuffled = list(all_diseases)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def recompute_gt_structures(
    predictor: DrugRepurposingPredictor,
    train_disease_ids: Set[str],
) -> Dict:
    """Recompute all GT-derived data structures from training diseases only."""
    originals = {
        "drug_train_freq": dict(predictor.drug_train_freq),
        "drug_to_diseases": dict(predictor.drug_to_diseases),
        "drug_cancer_types": dict(predictor.drug_cancer_types),
        "drug_disease_groups": dict(predictor.drug_disease_groups),
        "train_diseases": list(predictor.train_diseases),
        "train_embeddings": predictor.train_embeddings.copy(),
        "train_disease_categories": dict(predictor.train_disease_categories),
    }

    # 1. Recompute drug_train_freq
    new_freq: Dict[str, int] = defaultdict(int)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            for drug_id in predictor.ground_truth[disease_id]:
                new_freq[drug_id] += 1
    predictor.drug_train_freq = dict(new_freq)

    # 2. Recompute drug_to_diseases
    new_d2d: Dict[str, Set[str]] = defaultdict(set)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            for drug_id in predictor.ground_truth[disease_id]:
                new_d2d[drug_id].add(disease_name.lower())
    predictor.drug_to_diseases = dict(new_d2d)

    # 3. Recompute drug_cancer_types
    new_cancer: Dict[str, Set[str]] = defaultdict(set)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            cancer_types = extract_cancer_types(disease_name)
            if cancer_types:
                for drug_id in predictor.ground_truth[disease_id]:
                    new_cancer[drug_id].update(cancer_types)
    predictor.drug_cancer_types = dict(new_cancer)

    # 4. Recompute drug_disease_groups
    new_groups: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            category = predictor.categorize_disease(disease_name)
            if category in DISEASE_HIERARCHY_GROUPS:
                for group_name, keywords in DISEASE_HIERARCHY_GROUPS[category].items():
                    if any(kw in disease_name.lower() for kw in keywords):
                        for drug_id in predictor.ground_truth[disease_id]:
                            new_groups[drug_id].add((category, group_name))
    predictor.drug_disease_groups = dict(new_groups)

    # 5. Rebuild kNN index
    predictor.train_diseases = [d for d in train_disease_ids if d in predictor.embeddings]
    predictor.train_embeddings = np.array(
        [predictor.embeddings[d] for d in predictor.train_diseases], dtype=np.float32
    )
    predictor.train_disease_categories = {}
    for d in predictor.train_diseases:
        name = predictor.disease_names.get(d, d)
        predictor.train_disease_categories[d] = predictor.categorize_disease(name)

    return originals


def restore_originals(predictor: DrugRepurposingPredictor, originals: Dict) -> None:
    """Restore original state."""
    predictor.drug_train_freq = originals["drug_train_freq"]
    predictor.drug_to_diseases = originals["drug_to_diseases"]
    predictor.drug_cancer_types = originals["drug_cancer_types"]
    predictor.drug_disease_groups = originals["drug_disease_groups"]
    predictor.train_diseases = originals["train_diseases"]
    predictor.train_embeddings = originals["train_embeddings"]
    predictor.train_disease_categories = originals["train_disease_categories"]


def evaluate_within_tier_ranking(
    predictor: DrugRepurposingPredictor,
    diseases: List[str],
    alphas: List[float],
    is_holdout: bool = False,
) -> Dict:
    """Evaluate within-tier ranking for a set of diseases.

    Returns per-tier, per-alpha results with top-half vs bottom-half precision.
    """
    # Collect all predictions with TransE scores
    all_preds = []  # (disease_id, drug_id, rank, tier, knn_score, transe_score, is_gt)

    for disease_id in diseases:
        if disease_id not in predictor.embeddings:
            continue

        gt_drugs = predictor.ground_truth.get(disease_id, set())
        disease_name = predictor.disease_names.get(disease_id, disease_id)

        # Get predictions
        result = predictor.predict(disease_name, k=20, top_n=30, include_filtered=True)

        for pred in result.predictions:
            transe_score = get_transe_score(predictor, pred.drug_id, disease_id)
            all_preds.append({
                "disease_id": disease_id,
                "drug_id": pred.drug_id,
                "rank": pred.rank,
                "tier": pred.confidence_tier.value,
                "knn_score": pred.knn_score,
                "transe_score": transe_score,
                "is_gt": pred.drug_id in gt_drugs,
            })

    # Group by tier
    tier_preds: Dict[str, list] = defaultdict(list)
    for p in all_preds:
        tier_preds[p["tier"]].append(p)

    results = {}

    for tier_name, preds in tier_preds.items():
        n = len(preds)
        gt_count = sum(1 for p in preds if p["is_gt"])
        overall_precision = gt_count / n if n > 0 else 0

        # Count predictions with TransE scores
        with_transe = [p for p in preds if p["transe_score"] is not None]
        transe_coverage = len(with_transe) / n if n > 0 else 0

        tier_result = {
            "n": n,
            "gt_count": gt_count,
            "overall_precision": round(overall_precision * 100, 2),
            "transe_coverage": round(transe_coverage * 100, 1),
            "alphas": {},
        }

        for alpha in alphas:
            # Compute composite scores for this alpha
            # For predictions without TransE, use kNN-only
            scored_preds = []
            for p in preds:
                knn_norm = p["knn_score"]  # Already has relative magnitude
                if p["transe_score"] is not None:
                    transe_norm = p["transe_score"]
                    composite = alpha * knn_norm + (1 - alpha) * transe_norm
                else:
                    composite = knn_norm  # No TransE, use kNN only
                scored_preds.append({**p, "composite": composite})

            # Group by disease (within-tier ranking is per-disease)
            disease_groups: Dict[str, list] = defaultdict(list)
            for p in scored_preds:
                disease_groups[p["disease_id"]].append(p)

            # For each disease, sort within-tier by composite, split into halves
            top_half_gt = 0
            top_half_n = 0
            bottom_half_gt = 0
            bottom_half_n = 0

            for disease_id, disease_preds in disease_groups.items():
                if len(disease_preds) < 2:
                    # Can't split into halves
                    top_half_n += len(disease_preds)
                    top_half_gt += sum(1 for p in disease_preds if p["is_gt"])
                    continue

                # Sort by composite score (higher = better)
                disease_preds.sort(key=lambda x: x["composite"], reverse=True)
                mid = len(disease_preds) // 2

                top_half = disease_preds[:mid]
                bottom_half = disease_preds[mid:]

                top_half_n += len(top_half)
                top_half_gt += sum(1 for p in top_half if p["is_gt"])
                bottom_half_n += len(bottom_half)
                bottom_half_gt += sum(1 for p in bottom_half if p["is_gt"])

            top_prec = top_half_gt / top_half_n if top_half_n > 0 else 0
            bot_prec = bottom_half_gt / bottom_half_n if bottom_half_n > 0 else 0

            tier_result["alphas"][str(alpha)] = {
                "top_half_precision": round(top_prec * 100, 2),
                "top_half_n": top_half_n,
                "top_half_gt": top_half_gt,
                "bottom_half_precision": round(bot_prec * 100, 2),
                "bottom_half_n": bottom_half_n,
                "bottom_half_gt": bottom_half_gt,
                "gap_pp": round((top_prec - bot_prec) * 100, 2),
            }

        # Also compute: kNN-only ranking (alpha=1.0) for baseline
        # and TransE-only ranking (alpha=0.0)
        results[tier_name] = tier_result

    return results


def main():
    print("=" * 70)
    print("h443: Within-Tier Ranking Using TransE + kNN Composite")
    print("=" * 70)

    # Load predictor
    print("\nLoading predictor...")
    t0 = time.time()
    predictor = DrugRepurposingPredictor()
    print(f"Loaded in {time.time() - t0:.1f}s")

    if predictor.transe_entity_emb is None:
        print("ERROR: TransE model not loaded. Cannot proceed.")
        return

    # Get all diseases with GT and embeddings
    diseases_with_gt = [
        d for d in predictor.ground_truth
        if d in predictor.embeddings
    ]
    print(f"Diseases with GT and embeddings: {len(diseases_with_gt)}")

    alphas = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]

    # ===== PART 1: Full-data evaluation =====
    print("\n" + "=" * 70)
    print("PART 1: Full-Data Within-Tier Ranking")
    print("=" * 70)
    print(f"Alpha values: {alphas}")
    print("(alpha=1.0 = kNN only, alpha=0.0 = TransE only)")

    t0 = time.time()
    full_results = evaluate_within_tier_ranking(
        predictor, diseases_with_gt, alphas
    )
    print(f"Completed in {time.time() - t0:.1f}s")

    # Print results
    tier_order = ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]
    for tier_name in tier_order:
        if tier_name not in full_results:
            continue
        tr = full_results[tier_name]
        print(f"\n--- {tier_name} (n={tr['n']}, GT={tr['gt_count']}, prec={tr['overall_precision']:.1f}%, TransE coverage={tr['transe_coverage']:.0f}%) ---")
        print(f"  {'Alpha':<8} {'Top-half prec':>14} {'Bot-half prec':>14} {'Gap':>8}")
        for alpha in alphas:
            a = tr["alphas"][str(alpha)]
            print(f"  {alpha:<8.1f} {a['top_half_precision']:>12.1f}% {a['bottom_half_precision']:>12.1f}% {a['gap_pp']:>+7.1f}pp")

    # ===== PART 2: Alternative approach - global ranking within tier =====
    print("\n" + "=" * 70)
    print("PART 2: Global (Cross-Disease) Within-Tier Ranking")
    print("=" * 70)
    print("(Instead of per-disease halves, rank ALL predictions in tier globally)")

    # Collect all predictions with TransE scores
    all_preds = []
    for disease_id in diseases_with_gt:
        if disease_id not in predictor.embeddings:
            continue
        gt_drugs = predictor.ground_truth.get(disease_id, set())
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        result = predictor.predict(disease_name, k=20, top_n=30, include_filtered=True)

        for pred in result.predictions:
            transe_score = get_transe_score(predictor, pred.drug_id, disease_id)
            all_preds.append({
                "disease_id": disease_id,
                "drug_id": pred.drug_id,
                "rank": pred.rank,
                "tier": pred.confidence_tier.value,
                "knn_score": pred.knn_score,
                "norm_score": pred.norm_score,
                "transe_score": transe_score,
                "is_gt": pred.drug_id in gt_drugs,
                "transe_bool": pred.transe_consilience,
            })

    tier_preds: Dict[str, list] = defaultdict(list)
    for p in all_preds:
        tier_preds[p["tier"]].append(p)

    for tier_name in tier_order:
        if tier_name not in tier_preds:
            continue
        preds = tier_preds[tier_name]
        n = len(preds)
        gt_count = sum(1 for p in preds if p["is_gt"])

        # Method 1: TransE boolean (existing flag)
        with_transe = [p for p in preds if p["transe_bool"]]
        without_transe = [p for p in preds if not p["transe_bool"]]
        transe_prec = sum(1 for p in with_transe if p["is_gt"]) / len(with_transe) * 100 if with_transe else 0
        no_transe_prec = sum(1 for p in without_transe if p["is_gt"]) / len(without_transe) * 100 if without_transe else 0

        # Method 2: rank-based split (top-half by kNN rank vs bottom-half)
        # Sort by original rank within tier
        preds_by_rank = sorted(preds, key=lambda x: x["rank"])
        mid = len(preds_by_rank) // 2
        rank_top_prec = sum(1 for p in preds_by_rank[:mid] if p["is_gt"]) / mid * 100 if mid > 0 else 0
        rank_bot_prec = sum(1 for p in preds_by_rank[mid:] if p["is_gt"]) / len(preds_by_rank[mid:]) * 100 if len(preds_by_rank) > mid else 0

        # Method 3: Combined kNN rank + TransE boolean
        # Score: 1 point for TransE top-30, then sort by kNN rank
        combo_sorted = sorted(preds, key=lambda x: (-int(x["transe_bool"]), x["rank"]))
        mid = len(combo_sorted) // 2
        combo_top_prec = sum(1 for p in combo_sorted[:mid] if p["is_gt"]) / mid * 100 if mid > 0 else 0
        combo_bot_prec = sum(1 for p in combo_sorted[mid:] if p["is_gt"]) / len(combo_sorted[mid:]) * 100 if len(combo_sorted) > mid else 0

        print(f"\n--- {tier_name} (n={n}, GT={gt_count}, prec={gt_count/n*100:.1f}%) ---")
        print(f"  TransE=True:  {transe_prec:.1f}% (n={len(with_transe)})")
        print(f"  TransE=False: {no_transe_prec:.1f}% (n={len(without_transe)})")
        print(f"  TransE gap:   {transe_prec - no_transe_prec:+.1f}pp")
        print(f"  kNN rank top-half:    {rank_top_prec:.1f}%")
        print(f"  kNN rank bottom-half: {rank_bot_prec:.1f}%")
        print(f"  kNN rank gap:         {rank_top_prec - rank_bot_prec:+.1f}pp")
        print(f"  Combo (TransE>kNN) top-half:    {combo_top_prec:.1f}%")
        print(f"  Combo (TransE>kNN) bottom-half: {combo_bot_prec:.1f}%")
        print(f"  Combo gap:                      {combo_top_prec - combo_bot_prec:+.1f}pp")

    # ===== PART 3: Holdout validation =====
    print("\n" + "=" * 70)
    print("PART 3: Holdout Validation (5-seed)")
    print("=" * 70)

    seeds = [42, 123, 456, 789, 2024]
    holdout_gaps: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    # holdout_gaps[tier][method] = [gap_seed1, gap_seed2, ...]

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        train_diseases, holdout_diseases = split_diseases(diseases_with_gt, seed)
        train_set = set(train_diseases)

        originals = recompute_gt_structures(predictor, train_set)

        # Collect holdout predictions
        holdout_preds = []
        for disease_id in holdout_diseases:
            if disease_id not in predictor.embeddings:
                continue
            # Use ORIGINAL ground truth (not training GT) for evaluation
            gt_drugs = originals["drug_train_freq"]  # This is wrong - need actual GT
            # Actually we need the original ground truth, which doesn't change
            gt_drugs_for_eval = predictor.ground_truth.get(disease_id, set())

            disease_name = predictor.disease_names.get(disease_id, disease_id)
            result = predictor.predict(disease_name, k=20, top_n=30, include_filtered=True)

            for pred in result.predictions:
                transe_score = get_transe_score(predictor, pred.drug_id, disease_id)
                holdout_preds.append({
                    "disease_id": disease_id,
                    "drug_id": pred.drug_id,
                    "rank": pred.rank,
                    "tier": pred.confidence_tier.value,
                    "knn_score": pred.knn_score,
                    "transe_score": transe_score,
                    "is_gt": pred.drug_id in gt_drugs_for_eval,
                    "transe_bool": pred.transe_consilience,
                })

        restore_originals(predictor, originals)

        # Analyze per tier
        tier_holdout: Dict[str, list] = defaultdict(list)
        for p in holdout_preds:
            tier_holdout[p["tier"]].append(p)

        for tier_name in tier_order:
            if tier_name not in tier_holdout:
                continue
            preds = tier_holdout[tier_name]
            n = len(preds)
            if n < 4:
                continue

            gt_count = sum(1 for p in preds if p["is_gt"])
            overall_prec = gt_count / n * 100 if n > 0 else 0

            # Method 1: TransE boolean
            with_transe = [p for p in preds if p["transe_bool"]]
            without_transe = [p for p in preds if not p["transe_bool"]]
            transe_prec = sum(1 for p in with_transe if p["is_gt"]) / len(with_transe) * 100 if with_transe else 0
            no_transe_prec = sum(1 for p in without_transe if p["is_gt"]) / len(without_transe) * 100 if without_transe else 0
            holdout_gaps[tier_name]["transe_bool"].append(transe_prec - no_transe_prec)

            # Method 2: kNN rank
            preds_by_rank = sorted(preds, key=lambda x: x["rank"])
            mid = len(preds_by_rank) // 2
            rank_top = sum(1 for p in preds_by_rank[:mid] if p["is_gt"]) / mid * 100 if mid > 0 else 0
            rank_bot = sum(1 for p in preds_by_rank[mid:] if p["is_gt"]) / len(preds_by_rank[mid:]) * 100 if len(preds_by_rank) > mid else 0
            holdout_gaps[tier_name]["knn_rank"].append(rank_top - rank_bot)

            # Method 3: Combo (TransE priority, then kNN rank)
            combo_sorted = sorted(preds, key=lambda x: (-int(x["transe_bool"]), x["rank"]))
            mid = len(combo_sorted) // 2
            combo_top = sum(1 for p in combo_sorted[:mid] if p["is_gt"]) / mid * 100 if mid > 0 else 0
            combo_bot = sum(1 for p in combo_sorted[mid:] if p["is_gt"]) / len(combo_sorted[mid:]) * 100 if len(combo_sorted) > mid else 0
            holdout_gaps[tier_name]["combo"].append(combo_top - combo_bot)

            print(f"  {tier_name}: prec={overall_prec:.1f}%, TransE gap={transe_prec - no_transe_prec:+.1f}pp, kNN gap={rank_top - rank_bot:+.1f}pp, Combo gap={combo_top - combo_bot:+.1f}pp")

    # Holdout summary
    print(f"\n--- Holdout Summary (5-seed mean ± std) ---")
    print(f"  {'Tier':<10} {'TransE Gap':>14} {'kNN Rank Gap':>14} {'Combo Gap':>14}")
    for tier_name in tier_order:
        if tier_name not in holdout_gaps:
            continue
        t_gaps = holdout_gaps[tier_name]["transe_bool"]
        k_gaps = holdout_gaps[tier_name]["knn_rank"]
        c_gaps = holdout_gaps[tier_name]["combo"]
        if t_gaps:
            print(f"  {tier_name:<10} {np.mean(t_gaps):>+7.1f}±{np.std(t_gaps):>4.1f}pp {np.mean(k_gaps):>+7.1f}±{np.std(k_gaps):>4.1f}pp {np.mean(c_gaps):>+7.1f}±{np.std(c_gaps):>4.1f}pp")

    # Save results
    results = {
        "hypothesis": "h443",
        "title": "Within-Tier Ranking Using TransE + kNN Composite",
        "full_data": full_results,
        "holdout_gaps": {
            tier: {
                method: {
                    "mean": round(float(np.mean(gaps)), 2),
                    "std": round(float(np.std(gaps)), 2),
                    "values": [round(float(g), 2) for g in gaps],
                }
                for method, gaps in methods.items()
            }
            for tier, methods in holdout_gaps.items()
        },
    }

    output_path = Path(__file__).parent.parent / "data" / "analysis" / "h443_within_tier_ranking.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
