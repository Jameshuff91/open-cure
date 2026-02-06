#!/usr/bin/env python3
"""
h464: Psychiatric MEDIUM→HIGH Promotion with Additional Evidence

Test whether top-ranked psychiatric MEDIUM predictions (rank<=5 or rank<=10,
with mechanism support) achieve >50% holdout precision and can be promoted to HIGH.

Psychiatric MEDIUM: 45.7% ± 5.4% holdout overall (highest MEDIUM category).
Full-data: 55.9%. HIGH threshold: 50.8%.

Key questions:
1. Can a rank+mechanism filter push precision above 50% on holdout?
2. Is the sample size large enough (n>=10 per seed) for reliable estimates?
3. Which diseases contribute most to the precision?
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from production_predictor import (
    ConfidenceTier,
    DrugRepurposingPredictor,
    extract_cancer_types,
    DISEASE_HIERARCHY_GROUPS,
)


def split_diseases(
    all_diseases: List[str], seed: int, train_ratio: float = 0.8
) -> Tuple[List[str], List[str]]:
    rng = np.random.RandomState(seed)
    shuffled = list(all_diseases)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def recompute_gt_structures(
    predictor: DrugRepurposingPredictor,
    train_disease_ids: Set[str],
) -> Dict:
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

    predictor.train_diseases = [
        d for d in train_disease_ids if d in predictor.embeddings
    ]
    predictor.train_embeddings = np.array(
        [predictor.embeddings[d] for d in predictor.train_diseases], dtype=np.float32
    )
    predictor.train_disease_categories = {}
    for d in predictor.train_diseases:
        name = predictor.disease_names.get(d, d)
        predictor.train_disease_categories[d] = predictor.categorize_disease(name)

    return originals


def restore_gt_structures(
    predictor: DrugRepurposingPredictor, originals: Dict
) -> None:
    predictor.drug_train_freq = originals["drug_train_freq"]
    predictor.drug_to_diseases = originals["drug_to_diseases"]
    predictor.drug_cancer_types = originals["drug_cancer_types"]
    predictor.drug_disease_groups = originals["drug_disease_groups"]
    predictor.train_diseases = originals["train_diseases"]
    predictor.train_embeddings = originals["train_embeddings"]
    predictor.train_disease_categories = originals["train_disease_categories"]


def evaluate_psychiatric_subsets(
    predictor: DrugRepurposingPredictor,
    disease_ids: List[str],
    gt_data: Dict[str, List[str]],
) -> Dict:
    """Evaluate precision for various psychiatric MEDIUM subsets."""
    gt_set = set()
    for disease_id, drugs in gt_data.items():
        for drug in drugs:
            if isinstance(drug, str):
                gt_set.add((disease_id, drug))
            elif isinstance(drug, dict):
                gt_set.add((disease_id, drug.get("drug_id") or drug.get("drug")))

    # Define subsets to test
    subsets = {
        'all_medium': lambda p, r: True,
        'rank_1_5': lambda p, r: p.rank <= 5,
        'rank_1_10': lambda p, r: p.rank <= 10,
        'mechanism': lambda p, r: p.mechanism_support,
        'rank_1_5_mech': lambda p, r: p.rank <= 5 and p.mechanism_support,
        'rank_1_10_mech': lambda p, r: p.rank <= 10 and p.mechanism_support,
        'freq_ge_10': lambda p, r: p.train_frequency >= 10,
        'rank_1_5_freq_ge_10': lambda p, r: p.rank <= 5 and p.train_frequency >= 10,
    }

    results = {name: {'hits': 0, 'total': 0, 'diseases': set()} for name in subsets}

    for disease_id in disease_ids:
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        try:
            result = predictor.predict(disease_name, top_n=30, include_filtered=True)
        except Exception:
            continue

        if result.category != 'psychiatric':
            continue

        for pred in result.predictions:
            if pred.confidence_tier.name != 'MEDIUM':
                continue

            is_hit = (disease_id, pred.drug_id) in gt_set

            for name, filter_fn in subsets.items():
                if filter_fn(pred, result):
                    results[name]['total'] += 1
                    results[name]['diseases'].add(disease_id)
                    if is_hit:
                        results[name]['hits'] += 1

    # Compute precisions
    for name in results:
        total = results[name]['total']
        hits = results[name]['hits']
        results[name]['precision'] = hits / total * 100 if total > 0 else 0
        results[name]['n_diseases'] = len(results[name]['diseases'])
        del results[name]['diseases']  # Not JSON serializable

    return results


def main() -> None:
    seeds = [42, 123, 456, 789, 2024]
    print("=" * 70)
    print("h464: Psychiatric MEDIUM→HIGH Promotion Analysis")
    print("=" * 70)

    print("\nLoading predictor...")
    predictor = DrugRepurposingPredictor()

    gt_path = predictor.reference_dir / "expanded_ground_truth.json"
    with open(gt_path) as f:
        gt_data = json.load(f)

    all_diseases = [d for d in predictor.ground_truth if d in predictor.embeddings]
    print(f"Diseases with GT + embeddings: {len(all_diseases)}")

    # Full-data baseline
    print("\n--- FULL-DATA BASELINE ---")
    full_result = evaluate_psychiatric_subsets(predictor, all_diseases, gt_data)
    print(f"  {'Subset':<25s} {'Prec%':>6s} {'Hits':>5s} {'Total':>6s} {'Diseases':>8s}")
    print(f"  {'-'*55}")
    for name, stats in full_result.items():
        print(f"  {name:<25s} {stats['precision']:5.1f}% {stats['hits']:5d} {stats['total']:6d} {stats['n_diseases']:8d}")

    # Holdout validation
    all_holdout: Dict[str, List[float]] = defaultdict(list)
    all_holdout_n: Dict[str, List[int]] = defaultdict(list)
    all_holdout_hits: Dict[str, List[int]] = defaultdict(list)

    for seed_idx, seed in enumerate(seeds):
        print(f"\n--- SEED {seed} ({seed_idx+1}/{len(seeds)}) ---")

        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        train_set = set(train_ids)

        originals = recompute_gt_structures(predictor, train_set)
        holdout_result = evaluate_psychiatric_subsets(predictor, holdout_ids, gt_data)

        for name, stats in holdout_result.items():
            print(f"  {name:<25s} {stats['precision']:5.1f}% ({stats['hits']}/{stats['total']}, {stats['n_diseases']} diseases)")
            all_holdout[name].append(stats['precision'])
            all_holdout_n[name].append(stats['total'])
            all_holdout_hits[name].append(stats['hits'])

        restore_gt_structures(predictor, originals)

    # Aggregate
    print("\n" + "=" * 70)
    print("AGGREGATE HOLDOUT RESULTS (mean ± std across 5 seeds)")
    print("=" * 70)

    HIGH_THRESHOLD = 50.8

    print(f"\n  {'Subset':<25s} {'Full%':>6s} {'Hold%':>7s} {'±std':>5s} {'Δ':>7s} {'MeanN':>6s} {'Decision':>10s}")
    print(f"  {'-'*70}")

    output = {}
    for name in full_result:
        full_prec = full_result[name]['precision']
        if all_holdout[name]:
            holdout_mean = np.mean(all_holdout[name])
            holdout_std = np.std(all_holdout[name])
            mean_n = np.mean(all_holdout_n[name])
            delta = holdout_mean - full_prec
        else:
            holdout_mean = 0
            holdout_std = 0
            mean_n = 0
            delta = -full_prec

        decision = "PROMOTE" if holdout_mean >= HIGH_THRESHOLD and mean_n >= 5 else "KEEP"

        print(f"  {name:<25s} {full_prec:5.1f}% {holdout_mean:6.1f}% ±{holdout_std:4.1f} {delta:+6.1f}pp {mean_n:6.0f} {decision:>10s}")

        output[name] = {
            'full_precision': round(full_prec, 1),
            'holdout_mean': round(float(holdout_mean), 1),
            'holdout_std': round(float(holdout_std), 1),
            'delta': round(float(delta), 1),
            'mean_n': round(float(mean_n), 1),
            'decision': decision,
            'seed_values': [float(v) for v in all_holdout[name]],
            'seed_n': [int(v) for v in all_holdout_n[name]],
            'seed_hits': [int(v) for v in all_holdout_hits[name]],
        }

    # Save
    output_path = Path(__file__).parent.parent / "data" / "analysis" / "h464_psychiatric_medium_holdout.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
