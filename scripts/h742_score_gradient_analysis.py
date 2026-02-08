#!/usr/bin/env python3
"""
h742: kNN Score as Continuous Quality Signal — Within-MEDIUM Score Gradient Analysis

Holdout evaluation of kNN score bins within each MEDIUM sub-rule.
h740 demoted score < 3.0 for nomech predictions (+3.2pp MEDIUM).
Question: do other sub-rules have similar low-score LOW-quality pockets?

Uses absolute score thresholds to bin predictions:
  <2.0, 2.0-3.0, 3.0-5.0, 5.0-8.0, >=8.0
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
    extract_cancer_types,
    DISEASE_HIERARCHY_GROUPS,
    HIERARCHY_EXCLUSIONS,
)


def split_diseases(all_diseases, seed, train_ratio=0.8):
    rng = np.random.RandomState(seed)
    shuffled = list(all_diseases)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def recompute_gt_structures(predictor, train_disease_ids):
    """Recompute GT-derived structures from training diseases only. Matches h393."""
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
            ctypes = extract_cancer_types(disease_name)
            if ctypes:
                for drug_id in predictor.ground_truth[disease_id]:
                    new_cancer[drug_id].update(ctypes)
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


def restore_gt_structures(predictor, originals):
    predictor.drug_train_freq = originals["drug_train_freq"]
    predictor.drug_to_diseases = originals["drug_to_diseases"]
    predictor.drug_cancer_types = originals["drug_cancer_types"]
    predictor.drug_disease_groups = originals["drug_disease_groups"]
    predictor.train_diseases = originals["train_diseases"]
    predictor.train_embeddings = originals["train_embeddings"]
    predictor.train_disease_categories = originals["train_disease_categories"]


def get_score_bin(score):
    """Assign score to a bin."""
    if score < 1.0:
        return '<1.0'
    elif score < 2.0:
        return '1.0-2.0'
    elif score < 3.0:
        return '2.0-3.0'
    elif score < 5.0:
        return '3.0-5.0'
    elif score < 8.0:
        return '5.0-8.0'
    else:
        return '>=8.0'


SCORE_BIN_ORDER = ['<1.0', '1.0-2.0', '2.0-3.0', '3.0-5.0', '5.0-8.0', '>=8.0']


def main():
    seeds = [42, 123, 456, 789, 2024]

    print("=" * 70)
    print("h742: kNN Score Gradient Analysis Across ALL MEDIUM Sub-Rules")
    print("=" * 70)

    print("\nLoading predictor...")
    predictor = DrugRepurposingPredictor()

    gt_path = predictor.reference_dir / "expanded_ground_truth.json"
    with open(gt_path) as f:
        gt_data = json.load(f)
    print(f"Loaded {len(gt_data)} GT diseases, {sum(len(v) for v in gt_data.values())} pairs")

    all_diseases = [d for d in predictor.ground_truth if d in predictor.embeddings]
    print(f"Diseases with GT + embeddings: {len(all_diseases)}")

    gt_set = set()
    for did, drugs in gt_data.items():
        for drug_id in drugs:
            gt_set.add((did, drug_id))

    # Holdout evaluation: sub_rule × score_bin → {seed: (hits, total)}
    # Structure: results[sub_rule][score_bin] = [[hits_s1, total_s1], [hits_s2, total_s2], ...]
    results = defaultdict(lambda: defaultdict(lambda: []))
    # Also track ALL MEDIUM combined
    all_medium = defaultdict(lambda: [])
    # Also track ALL predictions (all tiers) by score bin for context
    all_tiers_by_score = defaultdict(lambda: defaultdict(lambda: []))

    for seed_idx, seed in enumerate(seeds):
        print(f"\nSeed {seed} ({seed_idx+1}/{len(seeds)})...")

        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        train_set = set(train_ids)

        originals = recompute_gt_structures(predictor, train_set)

        n_diseases = 0
        for i, disease_id in enumerate(holdout_ids):
            if i % 50 == 0:
                print(f"  Disease {i}/{len(holdout_ids)}...", flush=True)

            disease_name = predictor.disease_names.get(disease_id, disease_id)
            try:
                result = predictor.predict(disease_name, top_n=30, include_filtered=True)
            except Exception:
                continue

            n_diseases += 1

            for pred in result.predictions:
                is_hit = (disease_id, pred.drug_id) in gt_set
                score_bin = get_score_bin(pred.knn_score)
                tier = pred.confidence_tier.name

                # Track all tiers by score
                while len(all_tiers_by_score[tier][score_bin]) <= seed_idx:
                    all_tiers_by_score[tier][score_bin].append([0, 0])
                all_tiers_by_score[tier][score_bin][seed_idx][1] += 1
                if is_hit:
                    all_tiers_by_score[tier][score_bin][seed_idx][0] += 1

                if pred.confidence_tier == ConfidenceTier.MEDIUM:
                    sub_rule = pred.category_specific_tier or "default"

                    # Per sub-rule
                    while len(results[sub_rule][score_bin]) <= seed_idx:
                        results[sub_rule][score_bin].append([0, 0])
                    results[sub_rule][score_bin][seed_idx][1] += 1
                    if is_hit:
                        results[sub_rule][score_bin][seed_idx][0] += 1

                    # All MEDIUM combined
                    while len(all_medium[score_bin]) <= seed_idx:
                        all_medium[score_bin].append([0, 0])
                    all_medium[score_bin][seed_idx][1] += 1
                    if is_hit:
                        all_medium[score_bin][seed_idx][0] += 1

        print(f"  Evaluated {n_diseases} diseases")
        restore_gt_structures(predictor, originals)

    # ============================================
    # Report Results
    # ============================================
    print("\n" + "=" * 70)
    print("ALL MEDIUM COMBINED — Score Gradient (Holdout)")
    print("=" * 70)

    print(f"\n{'Score Bin':>12s} {'Holdout%':>8s} {'±std':>6s} {'n/seed':>7s}")
    print(f"{'-'*40}")

    combined_results = {}
    for sbin in SCORE_BIN_ORDER:
        data = all_medium.get(sbin, [])
        if not data:
            continue
        # Pad to 5 seeds
        while len(data) < len(seeds):
            data.append([0, 0])

        precisions = []
        totals = []
        for s in range(len(seeds)):
            totals.append(data[s][1])
            if data[s][1] > 0:
                precisions.append(data[s][0] / data[s][1] * 100)

        if precisions:
            mean_p = np.mean(precisions)
            std_p = np.std(precisions)
            mean_n = np.mean(totals)
            combined_results[sbin] = {
                'holdout_mean': round(mean_p, 1),
                'holdout_std': round(std_p, 1),
                'n_per_seed': round(mean_n, 1),
            }
            flag = " *** LOW ***" if mean_p < 20 and mean_n >= 5 else ""
            print(f"{sbin:>12s} {mean_p:>7.1f}% {std_p:>5.1f}% {mean_n:>7.1f}{flag}")

    # Per sub-rule
    print("\n" + "=" * 70)
    print("PER SUB-RULE — Score Gradient (Holdout)")
    print("=" * 70)

    sub_rule_results = {}
    for rule in sorted(results.keys()):
        bins = results[rule]
        total_n = sum(
            np.mean([d[1] for d in bins.get(sbin, [[0,0]])]) for sbin in SCORE_BIN_ORDER
        )
        if total_n < 5:
            continue

        print(f"\n  {rule} (total n/seed≈{total_n:.0f}):")
        print(f"    {'Score Bin':>12s} {'Holdout%':>8s} {'±std':>6s} {'n/seed':>7s}")

        rule_result = {}
        for sbin in SCORE_BIN_ORDER:
            data = bins.get(sbin, [])
            if not data:
                continue
            while len(data) < len(seeds):
                data.append([0, 0])

            precisions = []
            totals = []
            for s in range(len(seeds)):
                totals.append(data[s][1])
                if data[s][1] > 0:
                    precisions.append(data[s][0] / data[s][1] * 100)

            if precisions:
                mean_p = np.mean(precisions)
                std_p = np.std(precisions)
                mean_n = np.mean(totals)
                rule_result[sbin] = {
                    'holdout_mean': round(mean_p, 1),
                    'holdout_std': round(std_p, 1),
                    'n_per_seed': round(mean_n, 1),
                }
                flag = " *** LOW ***" if mean_p < 20 and mean_n >= 5 else ""
                print(f"    {sbin:>12s} {mean_p:>7.1f}% {std_p:>5.1f}% {mean_n:>7.1f}{flag}")

        sub_rule_results[rule] = rule_result

    # Demotion targets
    print("\n" + "=" * 70)
    print("POTENTIAL DEMOTION TARGETS (holdout < 25%, n >= 5/seed)")
    print("=" * 70)

    demotion_targets = []
    for rule, bins in sub_rule_results.items():
        for sbin, stats in bins.items():
            if stats['holdout_mean'] < 25.0 and stats['n_per_seed'] >= 5:
                demotion_targets.append({
                    'rule': rule,
                    'score_bin': sbin,
                    'holdout': stats['holdout_mean'],
                    'std': stats['holdout_std'],
                    'n_per_seed': stats['n_per_seed'],
                })

    if demotion_targets:
        demotion_targets.sort(key=lambda x: x['holdout'])
        for t in demotion_targets:
            print(f"  {t['rule']:<40s} score={t['score_bin']:>8s}: {t['holdout']:.1f}% ± {t['std']:.1f}% (n={t['n_per_seed']:.1f}/seed)")
    else:
        print("  None found")

    # Score percentile analysis for annotation
    print("\n" + "=" * 70)
    print("SCORE PERCENTILE ANNOTATION VALUE")
    print("=" * 70)
    print("Score gradient across ALL MEDIUM:")
    if combined_results:
        scores = [(sbin, r) for sbin, r in combined_results.items()]
        lo = min(r['holdout_mean'] for _, r in scores if r['n_per_seed'] >= 5)
        hi = max(r['holdout_mean'] for _, r in scores if r['n_per_seed'] >= 5)
        print(f"  Range: {lo:.1f}% to {hi:.1f}% (spread: {hi-lo:.1f}pp)")
        print(f"  {'YES — score is a strong continuous quality signal' if hi - lo > 15 else 'Modest signal'}")

    # Save results
    output = {
        'all_medium_score_gradient': combined_results,
        'per_subrule_score_gradient': sub_rule_results,
        'demotion_targets': demotion_targets,
    }

    output_path = Path(__file__).parent.parent / "data" / "analysis" / "h742_score_gradient.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
