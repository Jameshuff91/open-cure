#!/usr/bin/env python3
"""
h614: MEDIUM Sub-Pathway Quality Map v2 — Correct GT Evaluation

Break down MEDIUM predictions by (tier_rule × category) to identify
sub-pathways with below-MEDIUM holdout. Uses expanded GT (h611 lesson).

Focus on the "standard" MEDIUM predictions (category_specific_tier=None)
broken by disease category, and the named MEDIUM rules by category.
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
    print("h614: MEDIUM Sub-Pathway Quality Map v2")
    print("=" * 70)

    predictor = DrugRepurposingPredictor()
    gt_path = predictor.reference_dir / "expanded_ground_truth.json"
    with open(gt_path) as f:
        gt_data = json.load(f)
    print(f"Expanded GT: {len(gt_data)} diseases, {sum(len(v) for v in gt_data.values())} pairs")

    all_diseases = sorted(
        d for d in predictor.disease_names
        if d in predictor.embeddings and d in gt_data
    )
    print(f"Diseases: {len(all_diseases)}")

    # Full-data: map all MEDIUM predictions by (tier_rule × category)
    print("\n--- Full-data MEDIUM prediction map ---")
    subpath_preds = defaultdict(list)

    for disease_id in all_diseases:
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        try:
            result = predictor.predict(disease_name, top_n=30, include_filtered=True)
        except Exception:
            continue

        for pred in result.predictions:
            if pred.confidence_tier.value == 'MEDIUM':
                rule = pred.category_specific_tier or 'standard'
                cat = pred.category
                key = f"{rule}|{cat}"
                subpath_preds[key].append({
                    'drug_name': pred.drug_name,
                    'drug_id': pred.drug_id,
                    'disease_id': disease_id,
                    'rank': pred.rank,
                    'mechanism_support': pred.mechanism_support,
                    'train_frequency': pred.train_frequency,
                })

    print(f"\nMEDIUM sub-pathways (rule × category):")
    print(f"{'Rule|Category':<55s} {'Preds':>5s} {'Drugs':>5s}")
    print("-" * 70)
    for key in sorted(subpath_preds.keys(), key=lambda x: -len(subpath_preds[x])):
        preds = subpath_preds[key]
        n_drugs = len(set(p['drug_name'] for p in preds))
        print(f"  {key:<53s} {len(preds):5d} {n_drugs:5d}")

    # 5-seed holdout by sub-pathway
    print("\n" + "=" * 70)
    print("5-SEED HOLDOUT BY SUB-PATHWAY (expanded GT)")
    print("=" * 70)

    seeds = [42, 123, 456, 789, 2024]
    subpath_seed_results = defaultdict(lambda: defaultdict(list))

    for seed_idx, seed in enumerate(seeds):
        print(f"\n  Seed {seed} ({seed_idx+1}/5)...")
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
                if pred.confidence_tier.value == 'MEDIUM':
                    rule = pred.category_specific_tier or 'standard'
                    cat = pred.category
                    key = f"{rule}|{cat}"
                    is_hit = pred.drug_id in gt_drugs
                    subpath_seed_results[key][seed].append(is_hit)

        restore_gt_structures(predictor, originals)

    # Summarize
    print(f"\n{'Sub-Pathway':<55s} {'Hold%':>6s} {'±Std':>6s} {'N/seed':>7s} {'Preds':>5s} {'Action':>10s}")
    print("-" * 95)

    results = []
    for key in sorted(subpath_seed_results.keys()):
        seed_precs = []
        seed_ns = []
        for seed in seeds:
            r = subpath_seed_results[key][seed]
            if r:
                seed_precs.append(100 * sum(r) / len(r))
                seed_ns.append(len(r))
            else:
                seed_precs.append(0)
                seed_ns.append(0)

        mean_prec = np.mean(seed_precs)
        std_prec = np.std(seed_precs)
        mean_n = np.mean(seed_ns)
        n_preds = len(subpath_preds.get(key, []))

        # Determine action
        if mean_n < 3:
            action = "small-n"
        elif mean_prec >= 40:
            action = "GOOD"
        elif mean_prec >= 25:
            action = "OK"
        elif mean_prec >= 15:
            action = "MARGINAL"
        else:
            action = "DEMOTE?"

        print(f"  {key:<53s} {mean_prec:5.1f}% {std_prec:5.1f}% {mean_n:6.1f} {n_preds:5d} {action:>10s}")
        results.append({
            'subpath': key,
            'holdout_mean': mean_prec,
            'holdout_std': std_prec,
            'n_per_seed': mean_n,
            'n_preds': n_preds,
            'action': action,
        })

    # Overall MEDIUM
    overall_precs = []
    for seed in seeds:
        all_results = []
        for key in subpath_seed_results:
            all_results.extend(subpath_seed_results[key][seed])
        if all_results:
            overall_precs.append(100 * sum(all_results) / len(all_results))
    print("-" * 95)
    if overall_precs:
        print(f"  {'OVERALL MEDIUM':<53s} {np.mean(overall_precs):5.1f}% {np.std(overall_precs):5.1f}%")

    # Actionable findings
    print("\n" + "=" * 70)
    print("ACTIONABLE FINDINGS")
    print("=" * 70)
    demotable = [r for r in results if r['action'] == 'DEMOTE?' and r['n_per_seed'] >= 3]
    marginal = [r for r in results if r['action'] == 'MARGINAL' and r['n_per_seed'] >= 3]

    if demotable:
        print("\nDemotable (<15% holdout, n>=3/seed):")
        for r in demotable:
            print(f"  {r['subpath']:<53s}: {r['holdout_mean']:.1f}% ± {r['holdout_std']:.1f}% (n={r['n_per_seed']:.1f}/seed, {r['n_preds']} preds)")
    else:
        print("\nNo demotable sub-pathways found.")

    if marginal:
        print("\nMarginal (15-25% holdout, n>=3/seed):")
        for r in marginal:
            print(f"  {r['subpath']:<53s}: {r['holdout_mean']:.1f}% ± {r['holdout_std']:.1f}% (n={r['n_per_seed']:.1f}/seed, {r['n_preds']} preds)")

    # Save results
    output_path = Path(__file__).parent.parent / "data" / "analysis" / "h614_medium_subpathway_v2.json"
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
