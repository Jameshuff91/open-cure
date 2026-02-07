#!/usr/bin/env python3
"""
h634: Remaining Cancer Same-Type MEDIUM Quality: No-Mechanism Demotion

After h633 promoted mech+rank<=10 to HIGH (62.4%), remaining cancer_same_type
without mechanism support has 18.3% ± 3.9% holdout — below MEDIUM (36.8%).

This script:
1. Confirms the no-mechanism residual precision
2. Checks drug class composition of demoted predictions
3. Simulates tier impact of demotion
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Set

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from production_predictor import (
    DrugRepurposingPredictor,
    CORTICOSTEROID_DRUGS,
)


def load_expanded_gt() -> Dict[str, Set[str]]:
    gt_path = Path("data/reference/expanded_ground_truth.json")
    with open(gt_path) as f:
        raw = json.load(f)
    return {k: set(v) for k, v in raw.items()}


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
                new_d2d[drug_id].add(disease_name)
    predictor.drug_to_diseases = dict(new_d2d)

    from production_predictor import extract_cancer_types
    new_cancer_types = {}
    for drug_id, diseases in new_d2d.items():
        types = set()
        for d in diseases:
            types.update(extract_cancer_types(d))
        if types:
            new_cancer_types[drug_id] = types
    predictor.drug_cancer_types = new_cancer_types

    from production_predictor import DISEASE_HIERARCHY_GROUPS
    new_groups = defaultdict(lambda: defaultdict(set))
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            disease_name = predictor.disease_names.get(disease_id, "").lower()
            for group_name, group_info in DISEASE_HIERARCHY_GROUPS.items():
                for subtype, keywords in group_info.get("subtypes", {}).items():
                    if any(kw in disease_name for kw in keywords):
                        for drug_id in predictor.ground_truth[disease_id]:
                            new_groups[drug_id][group_name].add(subtype)
    predictor.drug_disease_groups = dict(new_groups)

    train_idx = [
        i for i, d in enumerate(predictor.train_diseases) if d in train_disease_ids
    ]
    predictor.train_diseases = [predictor.train_diseases[i] for i in train_idx]
    predictor.train_embeddings = predictor.train_embeddings[train_idx]
    predictor.train_disease_categories = {
        d: predictor.train_disease_categories[d]
        for d in predictor.train_diseases
        if d in predictor.train_disease_categories
    }

    return originals


def restore_originals(predictor, originals):
    for key, val in originals.items():
        setattr(predictor, key, val)


def main():
    print("=" * 70)
    print("h634: Cancer Same-Type No-Mechanism Demotion Analysis")
    print("=" * 70)

    predictor = DrugRepurposingPredictor()
    expanded_gt = load_expanded_gt()
    all_diseases = list(predictor.ground_truth.keys())

    # Phase 1: Production prediction analysis
    print("\n=== Phase 1: Production Predictions ===")

    cst_remaining = []
    for disease_id in all_diseases:
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        result = predictor.predict(disease_name, top_n=30, k=20)
        for p in result.predictions:
            if p.category_specific_tier == 'cancer_same_type':
                cst_remaining.append(p)

    print(f"Remaining cancer_same_type MEDIUM: {len(cst_remaining)}")

    # Split by mechanism
    mech_yes = [p for p in cst_remaining if p.mechanism_support]
    mech_no = [p for p in cst_remaining if not p.mechanism_support]
    print(f"  With mechanism: {len(mech_yes)}")
    print(f"  Without mechanism: {len(mech_no)}")

    # Rank distribution for no-mechanism
    print(f"\n  No-mechanism rank distribution:")
    for r_min, r_max in [(1, 5), (6, 10), (11, 15), (16, 20)]:
        n = len([p for p in mech_no if r_min <= p.rank <= r_max])
        print(f"    Rank {r_min}-{r_max}: {n}")

    # Top drugs for no-mechanism
    print(f"\n  No-mechanism top drugs:")
    drug_counts = Counter(p.drug_name for p in mech_no)
    for drug, count in drug_counts.most_common(10):
        print(f"    {drug}: {count}")

    # Phase 2: 5-seed holdout confirmation
    print("\n=== Phase 2: Holdout Confirmation ===")
    seeds = [42, 123, 456, 789, 2024]  # Match h393 evaluator seeds

    groups = {
        'cancer_same_type_mech': {'total': [], 'hits': []},
        'cancer_same_type_nomech': {'total': [], 'hits': []},
        'cancer_same_type_nomech_rank11_20': {'total': [], 'hits': []},
        'cancer_same_type_nomech_rank1_10': {'total': [], 'hits': []},
    }

    for seed in seeds:
        train_diseases, holdout_diseases = split_diseases(all_diseases, seed)
        train_set = set(train_diseases)
        originals = recompute_gt_structures(predictor, train_set)

        seed_counts = {k: {'total': 0, 'hits': 0} for k in groups}

        for disease_id in holdout_diseases:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            try:
                result = predictor.predict(disease_name, top_n=30, k=20)
            except Exception:
                continue

            for p in result.predictions:
                if p.category_specific_tier != 'cancer_same_type':
                    continue

                is_hit = p.drug_id in expanded_gt.get(disease_id, set())

                if p.mechanism_support:
                    seed_counts['cancer_same_type_mech']['total'] += 1
                    if is_hit:
                        seed_counts['cancer_same_type_mech']['hits'] += 1
                else:
                    seed_counts['cancer_same_type_nomech']['total'] += 1
                    if is_hit:
                        seed_counts['cancer_same_type_nomech']['hits'] += 1

                    if p.rank > 10:
                        seed_counts['cancer_same_type_nomech_rank11_20']['total'] += 1
                        if is_hit:
                            seed_counts['cancer_same_type_nomech_rank11_20']['hits'] += 1
                    else:
                        seed_counts['cancer_same_type_nomech_rank1_10']['total'] += 1
                        if is_hit:
                            seed_counts['cancer_same_type_nomech_rank1_10']['hits'] += 1

        for key in groups:
            groups[key]['total'].append(seed_counts[key]['total'])
            groups[key]['hits'].append(seed_counts[key]['hits'])

        restore_originals(predictor, originals)

    # Print results
    print(f"\n{'Group':<45} {'Holdout':>8} {'±std':>8} {'N/seed':>8}")
    print("-" * 73)
    for key, data in groups.items():
        precisions = []
        for h, t in zip(data['hits'], data['total']):
            if t > 0:
                precisions.append(h / t)
        if precisions:
            mean_p = np.mean(precisions) * 100
            std_p = np.std(precisions) * 100
            mean_n = np.mean(data['total'])
            marker = ""
            if mean_p < 20:
                marker = " *** BELOW MEDIUM ***"
            print(f"{key:<45} {mean_p:>7.1f}% {std_p:>7.1f}% {mean_n:>7.1f}{marker}")

    # Phase 3: Tier impact simulation
    print("\n=== Phase 3: Tier Impact Simulation ===")
    nomech_data = groups['cancer_same_type_nomech']
    nomech_precisions = []
    for h, t in zip(nomech_data['hits'], nomech_data['total']):
        if t > 0:
            nomech_precisions.append(h / t)
    nomech_mean = np.mean(nomech_precisions) * 100

    print(f"No-mechanism holdout: {nomech_mean:.1f}%")
    print(f"Current MEDIUM: 36.8% (1972 preds)")
    print(f"Current LOW: 14.2% (3622 preds)")

    n_demoted = len(mech_no)
    new_medium_n = 1972 - n_demoted
    # Approximate new MEDIUM precision
    # (1972 × 0.368 - n_demoted × nomech_mean/100) / new_medium_n
    new_medium_pct = (1972 * 0.368 - n_demoted * nomech_mean / 100) / new_medium_n * 100
    new_low_n = 3622 + n_demoted
    new_low_pct = (3622 * 0.142 + n_demoted * nomech_mean / 100) / new_low_n * 100

    print(f"\nIf demotion applied ({n_demoted} preds):")
    print(f"  MEDIUM: 36.8% → ~{new_medium_pct:.1f}% ({new_medium_n} preds)")
    print(f"  LOW: 14.2% → ~{new_low_pct:.1f}% ({new_low_n} preds)")

    # Save results
    output = {
        'n_remaining_cancer_same_type': len(cst_remaining),
        'n_mechanism': len(mech_yes),
        'n_no_mechanism': len(mech_no),
        'holdout_results': {
            key: {
                'mean_precision': round(np.mean([h/t if t > 0 else 0 for h, t in zip(d['hits'], d['total'])]) * 100, 1),
                'std_precision': round(np.std([h/t if t > 0 else 0 for h, t in zip(d['hits'], d['total'])]) * 100, 1),
                'mean_n': round(np.mean(d['total']), 1),
            }
            for key, d in groups.items()
        },
    }
    output_path = Path("data/analysis/h634_cancer_nomech_demotion.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
