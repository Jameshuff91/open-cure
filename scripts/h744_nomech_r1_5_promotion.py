#!/usr/bin/env python3
"""
h744: Analyze nomech R1-5 score>=5.0 for potential promotion to HIGH.

After h742 demotion of R1-5 score<5.0, the remaining default_freq10_nomech_r1_5
(score>=5.0) was reported at 68.2% ± 6.7% holdout. This script verifies:
1. Seed stability across all 5 seeds
2. CS composition (corticosteroid artifact check)
3. Score gradient within the group
4. Whether promotion to HIGH is safe
5. R6-10 high-score check too

Uses the h393 evaluator pattern for correctness.
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
    HIERARCHY_EXCLUSIONS,
)

# Corticosteroid drug names
CS_DRUGS = {
    'prednisone', 'prednisolone', 'methylprednisolone', 'dexamethasone',
    'hydrocortisone', 'betamethasone', 'triamcinolone', 'budesonide',
    'fluticasone', 'mometasone', 'beclomethasone', 'cortisone',
    'fludrocortisone', 'clobetasol', 'fluocinolone', 'fluocinonide',
    'halobetasol', 'desoximetasone', 'desonide', 'ciclesonide',
}


def split_diseases(all_diseases, seed, train_ratio=0.8):
    rng = np.random.RandomState(seed)
    shuffled = list(all_diseases)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def recompute_gt_structures(predictor, train_disease_ids):
    """Recompute GT structures from h393 pattern."""
    originals = {
        "drug_train_freq": dict(predictor.drug_train_freq),
        "drug_to_diseases": dict(predictor.drug_to_diseases),
        "drug_cancer_types": dict(predictor.drug_cancer_types),
        "drug_disease_groups": dict(predictor.drug_disease_groups),
        "train_diseases": list(predictor.train_diseases),
        "train_embeddings": predictor.train_embeddings.copy(),
        "train_disease_categories": dict(predictor.train_disease_categories),
    }

    # Recompute drug_train_freq
    new_freq = defaultdict(int)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            for drug_id in predictor.ground_truth[disease_id]:
                new_freq[drug_id] += 1
    predictor.drug_train_freq = dict(new_freq)

    # Recompute drug_to_diseases
    new_d2d = defaultdict(set)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            for drug_id in predictor.ground_truth[disease_id]:
                new_d2d[drug_id].add(disease_name.lower())
    predictor.drug_to_diseases = dict(new_d2d)

    # Recompute drug_cancer_types
    new_cancer = defaultdict(set)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            cancer_types = extract_cancer_types(disease_name)
            if cancer_types:
                for drug_id in predictor.ground_truth[disease_id]:
                    new_cancer[drug_id].update(cancer_types)
    predictor.drug_cancer_types = dict(new_cancer)

    # Recompute drug_disease_groups
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

    # Rebuild kNN index from training diseases only
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
    for k, v in originals.items():
        setattr(predictor, k, v)


def main():
    seeds = [42, 123, 456, 789, 2024]
    print("=" * 70)
    print("h744: Nomech R1-5 Score>=5 Promotion Analysis")
    print("=" * 70)

    print("\nLoading predictor...")
    predictor = DrugRepurposingPredictor()

    # Load expanded GT
    gt_path = predictor.reference_dir / "expanded_ground_truth.json"
    with open(gt_path) as f:
        gt_data = json.load(f)
    print(f"Loaded {len(gt_data)} GT diseases, {sum(len(v) for v in gt_data.values())} pairs")

    # Build GT set for quick lookup
    gt_set = set()
    for disease_id, drugs in gt_data.items():
        for drug in drugs:
            if isinstance(drug, str):
                gt_set.add((disease_id, drug))
            elif isinstance(drug, dict):
                gt_set.add((disease_id, drug.get("drug_id") or drug.get("drug")))

    # Get all diseases with GT + embeddings
    all_diseases = [
        d for d in predictor.ground_truth if d in predictor.embeddings
    ]
    print(f"Diseases with GT + embeddings: {len(all_diseases)}")

    # PART 1: Full-data analysis
    print("\n" + "=" * 70)
    print("PART 1: Full-Data Analysis")
    print("=" * 70)

    sub_reason_data = defaultdict(list)  # reason -> list of (is_gt, is_cs, score, drug, disease)

    for i, disease_id in enumerate(all_diseases):
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        try:
            result = predictor.predict(disease_name, top_n=30, include_filtered=True)
        except Exception as e:
            continue

        for pred in result.predictions:
            reason = pred.category_specific_tier or 'default'
            is_gt = (disease_id, pred.drug_id) in gt_set
            is_cs = pred.drug_name.lower() in CS_DRUGS
            score = pred.knn_score

            sub_reason_data[reason].append({
                'is_gt': is_gt,
                'is_cs': is_cs,
                'score': score,
                'drug': pred.drug_name,
                'disease': disease_name,
                'tier': pred.confidence_tier.name,
            })

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(all_diseases)} diseases...")

    # Print full-data summary
    for reason in ['default_freq10_nomech_r1_5', 'default_freq10_nomech_r6_10',
                    'default_freq10_mechanism', 'default_nomech_low_score',
                    'default_nomech_r1_5_low_score']:
        preds = sub_reason_data.get(reason, [])
        if not preds:
            print(f"\n{reason}: 0 predictions")
            continue
        total = len(preds)
        gt_hits = sum(1 for p in preds if p['is_gt'])
        cs_count = sum(1 for p in preds if p['is_cs'])
        non_cs = [p for p in preds if not p['is_cs']]
        non_cs_gt = sum(1 for p in non_cs if p['is_gt'])
        scores = [p['score'] for p in preds]

        print(f"\n{reason}: {total} preds")
        print(f"  Full-data precision: {100*gt_hits/total:.1f}% ({gt_hits}/{total})")
        print(f"  CS: {cs_count}/{total} ({100*cs_count/total:.1f}%)")
        if non_cs:
            print(f"  Non-CS precision: {100*non_cs_gt/len(non_cs):.1f}% ({non_cs_gt}/{len(non_cs)})")
        print(f"  Score: {min(scores):.2f} - {max(scores):.2f}, mean={np.mean(scores):.2f}")

        # Score buckets for R1-5
        if reason == 'default_freq10_nomech_r1_5':
            for lo, hi in [(5.0, 6.0), (6.0, 8.0), (8.0, 10.0), (10.0, 999.0)]:
                bucket = [p for p in preds if lo <= p['score'] < hi]
                if bucket:
                    bgt = sum(1 for p in bucket if p['is_gt'])
                    bcs = sum(1 for p in bucket if p['is_cs'])
                    label = f"{lo:.0f}-{hi:.0f}" if hi < 999 else f"{lo:.0f}+"
                    print(f"  Score {label}: {len(bucket)} preds, {100*bgt/len(bucket):.1f}% prec, {bcs} CS")

            # Top drugs
            drug_counts = defaultdict(int)
            for p in preds:
                drug_counts[p['drug']] += 1
            print("\n  Top 15 drugs:")
            for drug, count in sorted(drug_counts.items(), key=lambda x: -x[1])[:15]:
                drug_preds = [p for p in preds if p['drug'] == drug]
                drug_gt = sum(1 for p in drug_preds if p['is_gt'])
                cs_flag = " [CS]" if drug.lower() in CS_DRUGS else ""
                print(f"    {drug}{cs_flag}: {count} preds, {100*drug_gt/count:.0f}% full-data prec")

    # PART 2: Holdout evaluation
    print("\n" + "=" * 70)
    print("PART 2: Holdout Evaluation (5 seeds)")
    print("=" * 70)

    # Track per-seed precision for each sub-reason
    seed_data = defaultdict(lambda: defaultdict(list))
    # seed_data[reason][seed] = list of (is_gt, is_cs, score)

    for seed_idx, seed in enumerate(seeds):
        print(f"\n--- Seed {seed} ({seed_idx+1}/5) ---")
        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        train_set = set(train_ids)

        originals = recompute_gt_structures(predictor, train_set)

        for disease_id in holdout_ids:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            try:
                result = predictor.predict(disease_name, top_n=30, include_filtered=True)
            except Exception:
                continue

            for pred in result.predictions:
                reason = pred.category_specific_tier or 'default'
                is_gt = (disease_id, pred.drug_id) in gt_set
                is_cs = pred.drug_name.lower() in CS_DRUGS
                score = pred.knn_score
                tier = pred.confidence_tier.name

                seed_data[reason][seed].append({'is_gt': is_gt, 'is_cs': is_cs, 'score': score})
                seed_data[f'tier_{tier}'][seed].append({'is_gt': is_gt, 'is_cs': is_cs, 'score': score})

                # Score sub-buckets for R1-5
                if reason == 'default_freq10_nomech_r1_5':
                    if score >= 8.0:
                        seed_data['nomech_r1_5_s8plus'][seed].append({'is_gt': is_gt, 'is_cs': is_cs, 'score': score})
                    else:
                        seed_data['nomech_r1_5_s5_8'][seed].append({'is_gt': is_gt, 'is_cs': is_cs, 'score': score})

                # Score sub-buckets for R6-10
                if reason == 'default_freq10_nomech_r6_10':
                    if score >= 5.0:
                        seed_data['nomech_r6_10_s5plus'][seed].append({'is_gt': is_gt, 'is_cs': is_cs, 'score': score})
                    else:
                        seed_data['nomech_r6_10_s3_5'][seed].append({'is_gt': is_gt, 'is_cs': is_cs, 'score': score})

        restore_gt_structures(predictor, originals)
        print(f"  Done. nomech_r1_5: {len(seed_data['default_freq10_nomech_r1_5'].get(seed, []))} preds")

    # PART 3: Aggregate results
    print("\n" + "=" * 70)
    print("PART 3: Aggregate Holdout Results (mean ± std, 5 seeds)")
    print("=" * 70)

    results_summary = {}

    for group_name in [
        'tier_GOLDEN', 'tier_HIGH', 'tier_MEDIUM', 'tier_LOW', 'tier_FILTER',
        '---',
        'default_freq10_nomech_r1_5',
        'nomech_r1_5_s5_8', 'nomech_r1_5_s8plus',
        '---',
        'default_freq10_nomech_r6_10',
        'nomech_r6_10_s3_5', 'nomech_r6_10_s5plus',
        '---',
        'default_freq10_mechanism',
        'default_nomech_low_score',
        'default_nomech_r1_5_low_score',
    ]:
        if group_name == '---':
            print()
            continue

        seed_precisions = []
        seed_ns = []
        seed_cs_fracs = []
        seed_noncs_precs = []

        for seed in seeds:
            preds = seed_data[group_name].get(seed, [])
            n = len(preds)
            if n == 0:
                continue
            hits = sum(1 for p in preds if p['is_gt'])
            cs = sum(1 for p in preds if p['is_cs'])
            non_cs = [p for p in preds if not p['is_cs']]
            non_cs_hits = sum(1 for p in non_cs if p['is_gt'])

            seed_precisions.append(100 * hits / n)
            seed_ns.append(n)
            seed_cs_fracs.append(100 * cs / n if n > 0 else 0)
            if non_cs:
                seed_noncs_precs.append(100 * non_cs_hits / len(non_cs))

        if not seed_precisions:
            continue

        mean_p = np.mean(seed_precisions)
        std_p = np.std(seed_precisions)
        mean_n = np.mean(seed_ns)
        mean_cs = np.mean(seed_cs_fracs)
        mean_noncs_p = np.mean(seed_noncs_precs) if seed_noncs_precs else 0
        std_noncs_p = np.std(seed_noncs_precs) if seed_noncs_precs else 0

        label = group_name.replace('tier_', '').replace('default_freq10_', '').replace('default_', '')
        print(f"  {label:40s}: {mean_p:5.1f}% ± {std_p:4.1f}% (n={mean_n:.1f}/seed) | CS={mean_cs:.0f}% | nonCS={mean_noncs_p:.1f}% ± {std_noncs_p:.1f}%")

        results_summary[group_name] = {
            'mean_precision': round(mean_p, 1),
            'std_precision': round(std_p, 1),
            'mean_n': round(mean_n, 1),
            'cs_frac': round(mean_cs, 1),
            'noncs_precision': round(mean_noncs_p, 1),
            'noncs_std': round(std_noncs_p, 1),
            'seed_values': [round(v, 1) for v in seed_precisions],
        }

        # Per-seed detail for key groups
        if group_name in ('default_freq10_nomech_r1_5', 'nomech_r1_5_s5_8', 'nomech_r1_5_s8plus',
                          'nomech_r6_10_s5plus'):
            per_seed_str = [f"{p:.1f}%" for p in seed_precisions]
            per_n_str = [f"n={n}" for n in seed_ns]
            print(f"    Per seed: {', '.join(per_seed_str)}")
            print(f"    Per seed n: {', '.join(per_n_str)}")

    # PART 4: Decision
    print("\n" + "=" * 70)
    print("PART 4: Decision Analysis")
    print("=" * 70)

    r1_5 = results_summary.get('default_freq10_nomech_r1_5', {})
    high_tier = results_summary.get('tier_HIGH', {})
    medium_tier = results_summary.get('tier_MEDIUM', {})

    r1_5_mean = r1_5.get('mean_precision', 0)
    r1_5_std = r1_5.get('std_precision', 0)
    r1_5_n = r1_5.get('mean_n', 0)
    r1_5_cs = r1_5.get('cs_frac', 0)
    r1_5_noncs = r1_5.get('noncs_precision', 0)

    high_mean = high_tier.get('mean_precision', 0)
    medium_mean = medium_tier.get('mean_precision', 0)

    print(f"\n  Nomech R1-5 (score>=5): {r1_5_mean:.1f}% ± {r1_5_std:.1f}% (n={r1_5_n:.1f})")
    print(f"  CS fraction: {r1_5_cs:.1f}%")
    print(f"  Non-CS precision: {r1_5_noncs:.1f}%")
    print(f"  Current HIGH tier: {high_mean:.1f}%")
    print(f"  Current MEDIUM tier: {medium_mean:.1f}%")

    if r1_5_noncs >= 55.0:
        print(f"\n  VERDICT: PROMOTE to HIGH (non-CS {r1_5_noncs:.1f}% >= 55% threshold)")
    elif r1_5_mean >= high_mean:
        print(f"\n  VERDICT: PROMOTE to HIGH (all: {r1_5_mean:.1f}% >= HIGH {high_mean:.1f}%)")
        if r1_5_cs > 30:
            print(f"  WARNING: CS fraction is {r1_5_cs:.0f}% — check if non-CS still qualifies")
    elif r1_5_mean >= 55.0:
        print(f"\n  VERDICT: PROMOTE to HIGH (above 55% absolute threshold)")
    else:
        print(f"\n  VERDICT: KEEP at MEDIUM ({r1_5_mean:.1f}% below thresholds)")

    # Save results
    out_path = Path('data/analysis/h744_nomech_r1_5_promotion.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
