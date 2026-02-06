#!/usr/bin/env python3
"""
h533 Follow-up: Validate respiratory low_freq_no_mech rescue candidate.

The deep dive found: low_freq_no_mech × respiratory = 23.3% ± 9.2% holdout (19/seed, 45 full).
23.3% would place them in MEDIUM range (28.8%).

Questions:
1. What drugs/diseases are these?
2. Would promoting them to LOW actually improve metrics?
3. How does this interact with existing respiratory rules?
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from production_predictor import (
    DrugRepurposingPredictor,
    extract_cancer_types,
    DISEASE_HIERARCHY_GROUPS,
    HIERARCHY_EXCLUSIONS,
    CORTICOSTEROID_DRUGS,
)


def split_diseases(all_diseases, seed, train_ratio=0.8):
    rng = np.random.RandomState(seed)
    shuffled = list(all_diseases)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def recompute_gt_structures(predictor, train_disease_ids):
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
                new_d2d[drug_id].add(disease_name.lower())
    predictor.drug_to_diseases = dict(new_d2d)

    new_cancer = defaultdict(set)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            cancer_types = extract_cancer_types(disease_name)
            if cancer_types:
                for drug_id in predictor.ground_truth[disease_id]:
                    new_cancer[drug_id].update(cancer_types)
    predictor.drug_cancer_types = dict(new_cancer)

    new_groups = defaultdict(set)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            disease_lower = disease_name.lower()
            for category, groups in DISEASE_HIERARCHY_GROUPS.items():
                for group_name, keywords in groups.items():
                    exclusions = HIERARCHY_EXCLUSIONS.get((category, group_name), [])
                    if any(excl in disease_lower for excl in exclusions):
                        continue
                    if any(kw in disease_lower or disease_lower in kw for kw in keywords):
                        for drug_id in predictor.ground_truth[disease_id]:
                            new_groups[drug_id].add((category, group_name))
    predictor.drug_disease_groups = dict(new_groups)

    predictor.train_diseases = [d for d in train_disease_ids if d in predictor.embeddings]
    predictor.train_embeddings = np.array(
        [predictor.embeddings[d] for d in predictor.train_diseases], dtype=np.float32
    )
    predictor.train_disease_categories = {}
    for d in predictor.train_diseases:
        name = predictor.disease_names.get(d, d)
        predictor.train_disease_categories[d] = predictor.categorize_disease(name)

    return originals


def restore_gt_structures(predictor, originals):
    predictor.drug_train_freq = originals["drug_train_freq"]
    predictor.drug_to_diseases = originals["drug_to_diseases"]
    predictor.drug_cancer_types = originals["drug_cancer_types"]
    predictor.drug_disease_groups = originals["drug_disease_groups"]
    predictor.train_diseases = originals["train_diseases"]
    predictor.train_embeddings = originals["train_embeddings"]
    predictor.train_disease_categories = originals["train_disease_categories"]


def main():
    seeds = [42, 123, 456, 789, 2024]

    print("=" * 80)
    print("h533 Respiratory Rescue: Characterize & Validate")
    print("=" * 80)

    predictor = DrugRepurposingPredictor()
    gt_path = predictor.reference_dir / "expanded_ground_truth.json"
    with open(gt_path) as f:
        gt_data = json.load(f)

    all_diseases = [d for d in predictor.ground_truth if d in predictor.embeddings]

    # Build GT set
    gt_set = set()
    for disease_id, drugs in gt_data.items():
        for drug in drugs:
            if isinstance(drug, str):
                gt_set.add((disease_id, drug))
            elif isinstance(drug, dict):
                gt_set.add((disease_id, drug.get("drug_id") or drug.get("drug")))

    # Find all respiratory FILTER low_freq_no_mech predictions
    print("\n--- Respiratory FILTER (low_freq_no_mech) Predictions ---")
    resp_filter_preds = []

    for disease_id in all_diseases:
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        cat = predictor.categorize_disease(disease_name)
        if cat != 'respiratory':
            continue

        try:
            result = predictor.predict(disease_name, top_n=30, include_filtered=True)
        except Exception:
            continue

        for pred in result.predictions:
            if pred.confidence_tier.name != "FILTER":
                continue
            if pred.category_specific_tier is not None:
                continue  # Only standard filter
            if pred.rank > 20:
                continue  # Skip rank>20
            if pred.has_targets and (pred.train_frequency > 2 or pred.mechanism_support):
                continue  # Not low_freq_no_mech

            is_hit = (disease_id, pred.drug_id) in gt_set
            drug_name = predictor.drug_id_to_name.get(pred.drug_id, pred.drug_id)
            resp_filter_preds.append({
                "drug_id": pred.drug_id,
                "drug_name": drug_name,
                "disease_id": disease_id,
                "disease_name": disease_name,
                "rank": pred.rank,
                "freq": pred.train_frequency,
                "mech": pred.mechanism_support,
                "has_targets": pred.has_targets,
                "is_hit": is_hit,
                "is_corticosteroid": any(s in drug_name.lower() for s in CORTICOSTEROID_DRUGS),
            })

    print(f"Total: {len(resp_filter_preds)} predictions")
    hits = sum(1 for p in resp_filter_preds if p["is_hit"])
    prec = hits / len(resp_filter_preds) * 100 if resp_filter_preds else 0
    print(f"GT hits: {hits} ({prec:.1f}%)")
    print(f"Corticosteroids: {sum(1 for p in resp_filter_preds if p['is_corticosteroid'])}")

    print(f"\n{'Drug':<30s} {'Disease':<35s} {'Rank':>4s} {'Freq':>4s} {'Mech':>4s} {'Tgt':>3s} {'Hit':>3s}")
    print("-" * 90)
    for p in sorted(resp_filter_preds, key=lambda x: (-x["is_hit"], x["rank"])):
        print(f"  {p['drug_name'][:28]:<28s} {p['disease_name'][:33]:<33s} {p['rank']:>4d} {p['freq']:>4d} {'Y' if p['mech'] else 'N':>4s} {'Y' if p['has_targets'] else 'N':>3s} {'HIT' if p['is_hit'] else '':>3s}")

    # Holdout validation: if we promoted these to LOW, what happens?
    print("\n--- Holdout Impact Simulation ---")
    print("If respiratory low_freq_no_mech FILTER → LOW:")

    for seed_idx, seed in enumerate(seeds):
        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        originals = recompute_gt_structures(predictor, set(train_ids))

        filter_hits = 0
        filter_n = 0
        low_hits = 0
        low_n = 0

        for disease_id in holdout_ids:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            try:
                result = predictor.predict(disease_name, top_n=30, include_filtered=True)
            except Exception:
                continue

            for pred in result.predictions:
                is_hit = (disease_id, pred.drug_id) in gt_set
                tier = pred.confidence_tier.name
                cat = predictor.categorize_disease(disease_name)

                if tier == "LOW":
                    low_hits += int(is_hit)
                    low_n += 1

                if tier == "FILTER" and pred.category_specific_tier is None:
                    if pred.rank <= 20 and cat == 'respiratory':
                        if not pred.has_targets or (pred.train_frequency <= 2 and not pred.mechanism_support):
                            # This would be promoted to LOW
                            filter_hits += int(is_hit)
                            filter_n += 1

                if tier == "FILTER":
                    # Also track overall FILTER
                    pass

        low_prec_before = low_hits / low_n * 100 if low_n > 0 else 0
        low_prec_after = (low_hits + filter_hits) / (low_n + filter_n) * 100 if (low_n + filter_n) > 0 else 0

        print(f"  Seed {seed}: rescued {filter_n} preds ({filter_hits} hits = {filter_hits/filter_n*100:.0f}% if filter_n > 0 else 0%)")
        print(f"    LOW before: {low_prec_before:.1f}% ({low_n}), after: {low_prec_after:.1f}% ({low_n + filter_n})")

        restore_gt_structures(predictor, originals)


if __name__ == "__main__":
    main()
