#!/usr/bin/env python3
"""
h447: Cancer Subtype Leakage - Which drug classes transfer across cancer subtypes?

h416 showed cancer_same_type has 45.5pp full-to-holdout gap overall.
But maybe some drug CLASSES (e.g., checkpoint inhibitors) genuinely transfer.

Quick holdout validation of top drug classes.
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from production_predictor import (
    DrugRepurposingPredictor,
    DISEASE_HIERARCHY_GROUPS,
    extract_cancer_types,
)


SEEDS = [42, 123, 456, 789, 2024]


def classify_cancer_drug(drug_name: str) -> str:
    drug = drug_name.lower()
    if any(kw in drug for kw in ['pembrolizumab', 'nivolumab', 'atezolizumab', 'durvalumab', 'ipilimumab', 'avelumab', 'cemiplimab']):
        return 'checkpoint_inhibitor'
    if any(kw in drug for kw in ['paclitaxel', 'docetaxel', 'vinblastine', 'vincristine', 'vinorelbine']):
        return 'taxane_vinca'
    if any(kw in drug for kw in ['doxorubicin', 'epirubicin', 'daunorubicin', 'idarubicin']):
        return 'anthracycline'
    if any(kw in drug for kw in ['cisplatin', 'carboplatin', 'oxaliplatin']):
        return 'platinum'
    if any(kw in drug for kw in ['fluorouracil', 'capecitabine', 'gemcitabine', 'cytarabine', 'methotrexate', 'pemetrexed']):
        return 'antimetabolite'
    if any(kw in drug for kw in ['cyclophosphamide', 'ifosfamide', 'melphalan', 'bendamustine', 'temozolomide', 'thiotepa']):
        return 'alkylating'
    if any(kw in drug for kw in ['bevacizumab', 'ramucirumab']):
        return 'anti_vegf'
    return 'other'


def split_diseases(all_diseases: List[str], seed: int, train_ratio: float = 0.8) -> Tuple[List[str], List[str]]:
    rng = np.random.RandomState(seed)
    shuffled = list(all_diseases)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def recompute_gt_structures(predictor: DrugRepurposingPredictor, train_disease_ids: Set[str]) -> Dict:
    originals = {
        "drug_train_freq": dict(predictor.drug_train_freq),
        "drug_to_diseases": dict(predictor.drug_to_diseases),
        "drug_cancer_types": dict(predictor.drug_cancer_types),
        "drug_disease_groups": dict(predictor.drug_disease_groups),
        "train_diseases": list(predictor.train_diseases),
        "train_embeddings": predictor.train_embeddings.copy(),
        "train_disease_categories": dict(predictor.train_disease_categories),
    }

    new_freq: Dict[str, int] = defaultdict(int)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            for drug_id in predictor.ground_truth[disease_id]:
                new_freq[drug_id] += 1
    predictor.drug_train_freq = dict(new_freq)

    new_d2d: Dict[str, Set[str]] = defaultdict(set)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            for drug_id in predictor.ground_truth[disease_id]:
                new_d2d[drug_id].add(disease_name.lower())
    predictor.drug_to_diseases = dict(new_d2d)

    new_cancer: Dict[str, Set[str]] = defaultdict(set)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            cancer_types = extract_cancer_types(disease_name)
            if cancer_types:
                for drug_id in predictor.ground_truth[disease_id]:
                    new_cancer[drug_id].update(cancer_types)
    predictor.drug_cancer_types = dict(new_cancer)

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
    predictor.drug_train_freq = originals["drug_train_freq"]
    predictor.drug_to_diseases = originals["drug_to_diseases"]
    predictor.drug_cancer_types = originals["drug_cancer_types"]
    predictor.drug_disease_groups = originals["drug_disease_groups"]
    predictor.train_diseases = originals["train_diseases"]
    predictor.train_embeddings = originals["train_embeddings"]
    predictor.train_disease_categories = originals["train_disease_categories"]


def collect_cancer_same_type(predictor, diseases):
    preds = []
    for disease_id in diseases:
        if disease_id not in predictor.embeddings:
            continue
        gt_drugs = predictor.ground_truth.get(disease_id, set())
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        result = predictor.predict(disease_name, k=20, top_n=30, include_filtered=True)
        for pred in result.predictions:
            if pred.category_specific_tier == "cancer_same_type":
                drug_class = classify_cancer_drug(pred.drug_name)
                preds.append({
                    "drug_name": pred.drug_name,
                    "drug_class": drug_class,
                    "is_gt": pred.drug_id in gt_drugs,
                    "rank": pred.rank,
                })
    return preds


def main():
    print("=" * 70)
    print("h447: Cancer Drug Class Holdout Validation")
    print("=" * 70)

    predictor = DrugRepurposingPredictor()
    diseases_with_gt = [d for d in predictor.ground_truth if d in predictor.embeddings]

    # Full-data baseline
    full_preds = collect_cancer_same_type(predictor, diseases_with_gt)
    print(f"\nFull-data cancer_same_type: {len(full_preds)} predictions")

    classes = defaultdict(lambda: {"n": 0, "gt": 0})
    for p in full_preds:
        classes[p["drug_class"]]["n"] += 1
        classes[p["drug_class"]]["gt"] += int(p["is_gt"])

    print(f"\n  {'Class':<25} {'Full-data Prec':>15} {'n':>5}")
    for cls, stats in sorted(classes.items(), key=lambda x: -x[1]["n"]):
        prec = stats["gt"] / stats["n"] * 100 if stats["n"] > 0 else 0
        print(f"  {cls:<25} {prec:>13.1f}% {stats['n']:>5}")

    # Holdout validation
    print(f"\n{'='*70}")
    print("Holdout Validation (5-seed)")
    print(f"{'='*70}")

    holdout_class_precs: Dict[str, list] = defaultdict(list)
    holdout_overall: list = []

    for seed in SEEDS:
        train_diseases, holdout_diseases = split_diseases(diseases_with_gt, seed)
        train_set = set(train_diseases)
        originals = recompute_gt_structures(predictor, train_set)

        holdout_preds = collect_cancer_same_type(predictor, holdout_diseases)
        restore_originals(predictor, originals)

        overall_prec = sum(1 for p in holdout_preds if p["is_gt"]) / len(holdout_preds) * 100 if holdout_preds else 0
        holdout_overall.append(overall_prec)

        seed_classes = defaultdict(lambda: {"n": 0, "gt": 0})
        for p in holdout_preds:
            seed_classes[p["drug_class"]]["n"] += 1
            seed_classes[p["drug_class"]]["gt"] += int(p["is_gt"])

        print(f"\n  Seed {seed}: {len(holdout_preds)} preds, {overall_prec:.1f}% overall")
        for cls, stats in sorted(seed_classes.items(), key=lambda x: -x[1]["n"]):
            prec = stats["gt"] / stats["n"] * 100 if stats["n"] > 0 else 0
            holdout_class_precs[cls].append(prec)
            if stats["n"] >= 5:
                print(f"    {cls}: {prec:.1f}% (n={stats['n']})")

    # Summary
    print(f"\n{'='*70}")
    print("Summary (5-seed mean ± std)")
    print(f"{'='*70}")
    print(f"\n  Overall cancer_same_type holdout: {np.mean(holdout_overall):.1f}% ± {np.std(holdout_overall):.1f}")
    print(f"\n  {'Class':<25} {'Full-data':>10} {'Holdout':>12} {'Gap':>8}")
    for cls in sorted(classes.keys(), key=lambda x: -classes[x]["n"]):
        full_prec = classes[cls]["gt"] / classes[cls]["n"] * 100 if classes[cls]["n"] > 0 else 0
        if cls in holdout_class_precs and len(holdout_class_precs[cls]) >= 3:
            h_mean = np.mean(holdout_class_precs[cls])
            h_std = np.std(holdout_class_precs[cls])
            gap = h_mean - full_prec
            print(f"  {cls:<25} {full_prec:>9.1f}% {h_mean:>7.1f}±{h_std:>3.1f}% {gap:>+7.1f}")

    # Save
    output = {
        "hypothesis": "h447",
        "overall_holdout": {"mean": round(float(np.mean(holdout_overall)), 2), "std": round(float(np.std(holdout_overall)), 2)},
    }
    output_path = Path(__file__).parent.parent / "data" / "analysis" / "h447_cancer_leakage.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
