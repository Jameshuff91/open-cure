#!/usr/bin/env python3
"""
h416: Cancer Same-Type + HIGH Criteria → Conditional HIGH Promotion

h399 found cancer_same_type predictions meeting standard HIGH criteria have 48.1% precision.
Currently cancer_same_type is MEDIUM (demoted from GOLDEN in h396).

Test whether selective cancer_same_type predictions can be promoted to HIGH:
1. Multiple threshold combinations (freq, rank, overlap)
2. Holdout validation of the best threshold
3. Only implement if holdout precision >= HIGH avg (50.6%)
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


def collect_cancer_same_type(predictor: DrugRepurposingPredictor, diseases: List[str]) -> List[Dict]:
    """Collect cancer_same_type MEDIUM predictions with detailed features."""
    preds = []
    for disease_id in diseases:
        if disease_id not in predictor.embeddings:
            continue
        gt_drugs = predictor.ground_truth.get(disease_id, set())
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        result = predictor.predict(disease_name, k=20, top_n=30, include_filtered=True)

        for pred in result.predictions:
            if pred.category_specific_tier != "cancer_same_type":
                continue
            overlap = predictor._get_target_overlap_count(pred.drug_id, disease_id)
            preds.append({
                "disease_id": disease_id,
                "drug_id": pred.drug_id,
                "drug_name": pred.drug_name,
                "rank": pred.rank,
                "tier": pred.confidence_tier.value,
                "freq": pred.train_frequency,
                "mech": pred.mechanism_support,
                "overlap": overlap,
                "is_gt": pred.drug_id in gt_drugs,
                "transe": pred.transe_consilience,
            })
    return preds


def main():
    print("=" * 70)
    print("h416: Cancer Same-Type → Conditional HIGH Promotion")
    print("=" * 70)

    print("\nLoading predictor...")
    t0 = time.time()
    predictor = DrugRepurposingPredictor()
    print(f"Loaded in {time.time() - t0:.1f}s")

    diseases_with_gt = [d for d in predictor.ground_truth if d in predictor.embeddings]

    # ===== PART 1: Full-data threshold analysis =====
    print("\n" + "=" * 70)
    print("PART 1: Full-Data Threshold Analysis")
    print("=" * 70)

    t0 = time.time()
    cst_preds = collect_cancer_same_type(predictor, diseases_with_gt)
    print(f"Cancer same-type MEDIUM predictions: {len(cst_preds)}")
    print(f"Collected in {time.time() - t0:.1f}s")

    overall_prec = sum(1 for p in cst_preds if p["is_gt"]) / len(cst_preds) * 100 if cst_preds else 0
    print(f"Overall precision: {overall_prec:.1f}%")

    # Test various thresholds
    thresholds = [
        ("freq>=10", lambda p: p["freq"] >= 10),
        ("freq>=15", lambda p: p["freq"] >= 15),
        ("freq>=20", lambda p: p["freq"] >= 20),
        ("rank<=5", lambda p: p["rank"] <= 5),
        ("rank<=3", lambda p: p["rank"] <= 3),
        ("rank<=10", lambda p: p["rank"] <= 10),
        ("mech", lambda p: p["mech"]),
        ("overlap>=1", lambda p: p["overlap"] >= 1),
        ("overlap>=3", lambda p: p["overlap"] >= 3),
        ("transe", lambda p: p["transe"]),
        ("freq>=10 + mech", lambda p: p["freq"] >= 10 and p["mech"]),
        ("freq>=15 + mech", lambda p: p["freq"] >= 15 and p["mech"]),
        ("rank<=5 + freq>=10", lambda p: p["rank"] <= 5 and p["freq"] >= 10),
        ("rank<=5 + mech", lambda p: p["rank"] <= 5 and p["mech"]),
        ("rank<=5 + overlap>=1", lambda p: p["rank"] <= 5 and p["overlap"] >= 1),
        ("freq>=10 + overlap>=1", lambda p: p["freq"] >= 10 and p["overlap"] >= 1),
        ("freq>=10 + mech + overlap>=1", lambda p: p["freq"] >= 10 and p["mech"] and p["overlap"] >= 1),
        ("rank<=5 + freq>=10 + mech", lambda p: p["rank"] <= 5 and p["freq"] >= 10 and p["mech"]),
    ]

    print(f"\n  {'Threshold':<35} {'n':>5} {'GT':>4} {'Precision':>10} {'Above HIGH?':>12}")
    results = []
    for name, fn in thresholds:
        matching = [p for p in cst_preds if fn(p)]
        if not matching:
            continue
        prec = sum(1 for p in matching if p["is_gt"]) / len(matching) * 100
        above_high = "YES" if prec >= 50.8 else "no"
        print(f"  {name:<35} {len(matching):>5} {sum(1 for p in matching if p['is_gt']):>4} {prec:>9.1f}% {above_high:>12}")
        results.append({"name": name, "n": len(matching), "precision": prec, "fn": fn})

    # ===== PART 2: Holdout validation of promising thresholds =====
    print("\n" + "=" * 70)
    print("PART 2: Holdout Validation (5-seed)")
    print("=" * 70)

    # Only validate thresholds with full-data precision >= 45% and n >= 20
    promising = [r for r in results if r["precision"] >= 45 and r["n"] >= 20]
    if not promising:
        promising = [r for r in results if r["precision"] >= 40 and r["n"] >= 15]

    print(f"Validating {len(promising)} promising thresholds")

    holdout_results: Dict[str, list] = defaultdict(list)

    for seed in SEEDS:
        train_diseases, holdout_diseases = split_diseases(diseases_with_gt, seed)
        train_set = set(train_diseases)
        originals = recompute_gt_structures(predictor, train_set)

        holdout_cst = collect_cancer_same_type(predictor, holdout_diseases)

        restore_originals(predictor, originals)

        print(f"\n  Seed {seed}: {len(holdout_cst)} cancer_same_type predictions")
        for r in promising:
            matching = [p for p in holdout_cst if r["fn"](p)]
            if matching:
                prec = sum(1 for p in matching if p["is_gt"]) / len(matching) * 100
            else:
                prec = 0
            holdout_results[r["name"]].append(prec)
            print(f"    {r['name']:<35} n={len(matching):>3} prec={prec:.1f}%")

    # Summary
    print("\n" + "=" * 70)
    print("PART 3: Holdout Summary")
    print("=" * 70)

    print(f"\n  {'Threshold':<35} {'Full-Data':>10} {'Holdout Mean':>13} {'Std':>5} {'>=HIGH?':>8}")
    for r in promising:
        vals = holdout_results[r["name"]]
        if vals:
            mean_v = np.mean(vals)
            std_v = np.std(vals)
            above = "YES" if mean_v >= 50.6 else "no"
            print(f"  {r['name']:<35} {r['precision']:>9.1f}% {mean_v:>10.1f}% ± {std_v:>3.1f} {above:>8}")

    # Save
    output = {
        "hypothesis": "h416",
        "n_cancer_same_type": len(cst_preds),
        "overall_precision": round(overall_prec, 2),
        "holdout_results": {
            name: {
                "mean": round(float(np.mean(vals)), 2),
                "std": round(float(np.std(vals)), 2),
                "values": [round(float(v), 2) for v in vals],
            }
            for name, vals in holdout_results.items()
        },
    }

    output_path = Path(__file__).parent.parent / "data" / "analysis" / "h416_cancer_promotion.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
