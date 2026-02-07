#!/usr/bin/env python3
"""
h623: MEDIUM Precision Recovery — Tighten CV Rescue by Rank

h618 CV rescue diluted MEDIUM from 41.3% to 38.9%. Test whether restricting
to rank<=10 recovers MEDIUM precision while keeping the best rescues.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from production_predictor import (
    DrugRepurposingPredictor,
    DISEASE_HIERARCHY_GROUPS,
    extract_cancer_types,
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
        "drug_to_diseases": {k: set(v) for k, v in predictor.drug_to_diseases.items()},
        "drug_cancer_types": {k: set(v) for k, v in predictor.drug_cancer_types.items()},
        "drug_disease_groups": {k: dict(v) for k, v in predictor.drug_disease_groups.items()},
    }

    new_freq = defaultdict(int)
    new_d2d = defaultdict(set)
    new_cancer_types = defaultdict(set)
    new_disease_groups = defaultdict(lambda: defaultdict(set))

    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            for drug_id in predictor.ground_truth[disease_id]:
                new_freq[drug_id] += 1
                new_d2d[drug_id].add(disease_name)
                for ct in extract_cancer_types(disease_name):
                    new_cancer_types[drug_id].add(ct)
                for group_name, group_info in DISEASE_HIERARCHY_GROUPS.items():
                    parent_kws = group_info.get("parent_keywords", [])
                    if any(kw.lower() in disease_name.lower() for kw in parent_kws):
                        new_disease_groups[drug_id][group_name].add(disease_name)

    predictor.drug_train_freq = dict(new_freq)
    predictor.drug_to_diseases = dict(new_d2d)
    predictor.drug_cancer_types = dict(new_cancer_types)
    predictor.drug_disease_groups = dict(new_disease_groups)
    return originals


def restore_gt_structures(predictor, originals):
    for k, v in originals.items():
        setattr(predictor, k, v)


def main():
    print("=" * 70)
    print("h623: CV Rescue Tightening — Rank Threshold Analysis")
    print("=" * 70)

    predictor = DrugRepurposingPredictor()
    gt_path = predictor.reference_dir / "expanded_ground_truth.json"
    with open(gt_path) as f:
        gt_data = json.load(f)

    all_diseases = sorted(
        d for d in predictor.disease_names
        if d in predictor.embeddings and d in gt_data
    )

    seeds = [42, 123, 456, 789, 2024]

    # Test different rank thresholds for cv_established_drug_rescue
    for max_rank in [5, 10, 15, 20]:
        print(f"\n{'=' * 50}")
        print(f"RANK <= {max_rank} THRESHOLD")
        print(f"{'=' * 50}")

        rescue_results = defaultdict(list)
        remaining_results = defaultdict(list)

        for seed in seeds:
            train_diseases, holdout_diseases = split_diseases(all_diseases, seed)
            train_set = set(train_diseases)
            originals = recompute_gt_structures(predictor, train_set)

            for disease_id in holdout_diseases:
                disease_name = predictor.disease_names.get(disease_id, disease_id)
                try:
                    result = predictor.predict(disease_name, top_n=30, include_filtered=True)
                except Exception:
                    continue

                gt_drugs = set(gt_data.get(disease_id, []))
                for pred in result.predictions:
                    if pred.category_specific_tier == 'cv_established_drug_rescue':
                        is_hit = pred.drug_id in gt_drugs
                        if pred.rank <= max_rank:
                            rescue_results[seed].append(is_hit)
                        else:
                            remaining_results[seed].append(is_hit)

            restore_gt_structures(predictor, originals)

        # Report
        rescue_precs = []
        remaining_precs = []
        for seed in seeds:
            r = rescue_results[seed]
            if r:
                rescue_precs.append(100 * sum(r) / len(r))
            rem = remaining_results[seed]
            if rem:
                remaining_precs.append(100 * sum(rem) / len(rem))

        rescue_n = np.mean([len(rescue_results[s]) for s in seeds])
        remaining_n = np.mean([len(remaining_results[s]) for s in seeds])

        if rescue_precs:
            print(f"  Rescued (rank<=  {max_rank}): {np.mean(rescue_precs):.1f}% ± {np.std(rescue_precs):.1f}% (n={rescue_n:.1f}/seed)")
        if remaining_precs:
            print(f"  Dropped (rank>{max_rank:2d}): {np.mean(remaining_precs):.1f}% ± {np.std(remaining_precs):.1f}% (n={remaining_n:.1f}/seed)")

    # Also test mechanism_support interaction
    print(f"\n{'=' * 50}")
    print(f"MECHANISM SUPPORT INTERACTION")
    print(f"{'=' * 50}")

    for mech_filter in [True, False, None]:  # None = no filter
        label = "+mech" if mech_filter is True else "-mech" if mech_filter is False else "all"

        seed_results = defaultdict(list)
        for seed in seeds:
            train_diseases, holdout_diseases = split_diseases(all_diseases, seed)
            train_set = set(train_diseases)
            originals = recompute_gt_structures(predictor, train_set)

            for disease_id in holdout_diseases:
                disease_name = predictor.disease_names.get(disease_id, disease_id)
                try:
                    result = predictor.predict(disease_name, top_n=30, include_filtered=True)
                except Exception:
                    continue

                gt_drugs = set(gt_data.get(disease_id, []))
                for pred in result.predictions:
                    if pred.category_specific_tier == 'cv_established_drug_rescue':
                        if mech_filter is None or pred.mechanism_support == mech_filter:
                            is_hit = pred.drug_id in gt_drugs
                            seed_results[seed].append(is_hit)

            restore_gt_structures(predictor, originals)

        precs = [100 * sum(seed_results[s]) / len(seed_results[s]) for s in seeds if seed_results[s]]
        mean_n = np.mean([len(seed_results[s]) for s in seeds])
        if precs:
            print(f"  {label:5s}: {np.mean(precs):.1f}% ± {np.std(precs):.1f}% (n={mean_n:.1f}/seed)")


if __name__ == "__main__":
    main()
