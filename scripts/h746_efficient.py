#!/usr/bin/env python3
"""
h746: Default Freq5 Mechanism Low-Score Demotion — Efficient Analysis

h742 found default_freq5_mechanism at score<1.0 has 31.7% ± 6.4% (n=18.6/seed).
Current MEDIUM is ~47.7%. z=-2.5 below MEDIUM avg — clear demotion candidate.

This script does a single pass per seed to collect all freq5_mech predictions,
then analyzes score thresholds post-hoc.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from production_predictor import (
    ConfidenceTier,
    DrugRepurposingPredictor,
)
from h393_holdout_tier_validation import (
    split_diseases,
    recompute_gt_structures,
    restore_gt_structures,
)

SEEDS = [42, 123, 456, 789, 2024]

CS_DRUGS = {
    'prednisone', 'prednisolone', 'dexamethasone', 'methylprednisolone',
    'hydrocortisone', 'cortisone', 'betamethasone', 'triamcinolone',
    'budesonide', 'fluticasone', 'mometasone', 'beclomethasone',
    'fluocinolone', 'clobetasol', 'desonide', 'halobetasol',
    'fludrocortisone',
}

def is_cs(drug_name: str) -> bool:
    return any(cs in drug_name.lower() for cs in CS_DRUGS)


def main():
    print("=" * 70)
    print("h746: Default Freq5 Mechanism Low-Score Demotion — Efficient")
    print("=" * 70)

    # Load predictor
    print("\nLoading predictor...")
    predictor = DrugRepurposingPredictor()

    # Load expanded GT
    gt_path = predictor.reference_dir / "expanded_ground_truth.json"
    with open(gt_path) as f:
        expanded_gt = json.load(f)
    gt_pair_count = sum(len(v) for v in expanded_gt.values())
    print(f"Expanded GT: {gt_pair_count} pairs")

    # Build GT set
    gt_set = set()
    for disease_id, drugs in expanded_gt.items():
        for drug in drugs:
            if isinstance(drug, str):
                gt_set.add((disease_id, drug))

    # All evaluable diseases
    all_diseases = [d for d in predictor.ground_truth if d in predictor.embeddings]
    print(f"Evaluable diseases: {len(all_diseases)}")

    # ================================================================
    # SECTION 1: Full-data composition
    # ================================================================
    print("\n" + "=" * 70)
    print("SECTION 1: Full-data composition of default_freq5_mechanism")
    print("=" * 70)

    all_freq5_mech = []
    all_medium = []

    for i, disease_id in enumerate(all_diseases):
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        try:
            result = predictor.predict(disease_name, top_n=30, include_filtered=True)
        except Exception:
            continue

        for pred in result.predictions:
            sr = pred.category_specific_tier
            if sr == 'default_freq5_mechanism':
                in_gt = (disease_id, pred.drug_id) in gt_set
                all_freq5_mech.append({
                    'disease_id': disease_id,
                    'drug_name': pred.drug_name,
                    'drug_id': pred.drug_id,
                    'score': pred.knn_score,
                    'rank': pred.rank,
                    'in_gt': in_gt,
                    'is_cs': is_cs(pred.drug_name),
                    'transe': getattr(pred, 'transe_consilience', False),
                    'lit_evidence': getattr(pred, 'literature_evidence_level', None),
                    'category': getattr(pred, 'disease_category', 'unknown'),
                })
            if pred.confidence_tier == ConfidenceTier.MEDIUM:
                all_medium.append({
                    'disease_id': disease_id,
                    'drug_id': pred.drug_id,
                    'score': pred.knn_score,
                    'in_gt': (disease_id, pred.drug_id) in gt_set,
                    'sub_reason': sr,
                })

        if (i + 1) % 100 == 0:
            print(f"  Full-data pass: {i+1}/{len(all_diseases)}...")

    print(f"\nTotal default_freq5_mechanism: {len(all_freq5_mech)}")
    print(f"Total MEDIUM: {len(all_medium)}")

    # Score brackets
    brackets = [
        ("[0.0, 0.5)", 0.0, 0.5),
        ("[0.5, 1.0)", 0.5, 1.0),
        ("[1.0, 1.5)", 1.0, 1.5),
        ("[1.5, 2.0)", 1.5, 2.0),
        ("[2.0, 3.0)", 2.0, 3.0),
        ("[3.0, 5.0)", 3.0, 5.0),
        ("[5.0, inf)", 5.0, 100.0),
    ]

    for name, lo, hi in brackets:
        preds = [p for p in all_freq5_mech if lo <= p['score'] < hi]
        if not preds:
            continue
        n = len(preds)
        hits = sum(1 for p in preds if p['in_gt'])
        cs_n = sum(1 for p in preds if p['is_cs'])
        transe_n = sum(1 for p in preds if p['transe'])
        ranks = [p['rank'] for p in preds]
        print(f"\n{name}: {n} preds, full-data={100*hits/n:.1f}% ({hits}/{n}), "
              f"CS={100*cs_n/n:.1f}%, TransE={100*transe_n/n:.1f}%, mean_rank={np.mean(ranks):.1f}")
        drug_counts = Counter(p['drug_name'] for p in preds)
        top5 = drug_counts.most_common(5)
        print(f"  Top drugs: {', '.join(f'{d}({c})' for d, c in top5)}")
        cat_counts = Counter(p['category'] for p in preds)
        top3_cats = cat_counts.most_common(3)
        print(f"  Top categories: {', '.join(f'{c}({n})' for c, n in top3_cats)}")

    # Detailed analysis of score < 1.0
    low_score = [p for p in all_freq5_mech if p['score'] < 1.0]
    print(f"\n--- Score < 1.0: {len(low_score)} predictions ---")
    if low_score:
        drug_counts = Counter(p['drug_name'] for p in low_score)
        print("Top 15 drugs:")
        for drug, cnt in drug_counts.most_common(15):
            hits_d = sum(1 for p in low_score if p['drug_name'] == drug and p['in_gt'])
            print(f"  {drug}: {cnt} preds, {hits_d} GT hits ({100*hits_d/cnt:.0f}%)")

        cat_counts = Counter(p['category'] for p in low_score)
        print("Category breakdown:")
        for cat, cnt in cat_counts.most_common():
            hits_c = sum(1 for p in low_score if p['category'] == cat and p['in_gt'])
            print(f"  {cat}: {cnt} ({100*hits_c/cnt:.0f}% full-data)")

    # ================================================================
    # SECTION 2: 5-seed holdout evaluation (single pass per seed)
    # ================================================================
    print("\n" + "=" * 70)
    print("SECTION 2: 5-seed holdout evaluation")
    print("=" * 70)

    # Collect per-seed data for all freq5_mech preds
    thresholds = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    seed_data = {t: {'all': [], 'noncs': []} for t in thresholds}
    seed_above_data = {t: {'all': []} for t in thresholds}
    seed_medium_data = []  # Track overall MEDIUM per seed

    for seed in SEEDS:
        print(f"\nSeed {seed}...")
        train_diseases, holdout_diseases = split_diseases(all_diseases, seed)
        train_ids = set(train_diseases)
        holdout_ids = set(holdout_diseases)

        originals = recompute_gt_structures(predictor, train_ids)

        # Collect all freq5_mech predictions for holdout diseases
        freq5_below = {t: {'hits': 0, 'total': 0, 'noncs_hits': 0, 'noncs_total': 0} for t in thresholds}
        freq5_above = {t: {'hits': 0, 'total': 0} for t in thresholds}
        medium_hits = 0
        medium_total = 0

        for disease_id in holdout_ids:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            try:
                result = predictor.predict(disease_name, top_n=30, include_filtered=True)
            except Exception:
                continue

            for pred in result.predictions:
                sr = pred.category_specific_tier

                # Track overall MEDIUM
                if pred.confidence_tier == ConfidenceTier.MEDIUM:
                    medium_total += 1
                    if (disease_id, pred.drug_id) in gt_set:
                        medium_hits += 1

                if sr == 'default_freq5_mechanism':
                    in_gt = (disease_id, pred.drug_id) in gt_set
                    cs = is_cs(pred.drug_name)
                    score = pred.knn_score

                    for t in thresholds:
                        if score < t:
                            freq5_below[t]['total'] += 1
                            if in_gt:
                                freq5_below[t]['hits'] += 1
                            if not cs:
                                freq5_below[t]['noncs_total'] += 1
                                if in_gt:
                                    freq5_below[t]['noncs_hits'] += 1
                        else:
                            freq5_above[t]['total'] += 1
                            if in_gt:
                                freq5_above[t]['hits'] += 1

        restore_gt_structures(predictor, originals)

        # Record seed results
        if medium_total > 0:
            seed_medium_data.append(100 * medium_hits / medium_total)

        for t in thresholds:
            d = freq5_below[t]
            if d['total'] > 0:
                seed_data[t]['all'].append({
                    'precision': 100 * d['hits'] / d['total'],
                    'n': d['total'],
                    'hits': d['hits']
                })
            if d['noncs_total'] > 0:
                seed_data[t]['noncs'].append({
                    'precision': 100 * d['noncs_hits'] / d['noncs_total'],
                    'n': d['noncs_total']
                })
            d2 = freq5_above[t]
            if d2['total'] > 0:
                seed_above_data[t]['all'].append({
                    'precision': 100 * d2['hits'] / d2['total'],
                    'n': d2['total']
                })

    # Print results
    print("\n" + "-" * 50)
    print(f"Current MEDIUM holdout: {np.mean(seed_medium_data):.1f}% ± {np.std(seed_medium_data):.1f}%")
    print(f"  Seeds: {[f'{v:.1f}' for v in seed_medium_data]}")

    print("\n--- Below threshold (candidates for demotion) ---")
    for t in thresholds:
        if seed_data[t]['all']:
            precs = [s['precision'] for s in seed_data[t]['all']]
            ns = [s['n'] for s in seed_data[t]['all']]
            print(f"\nScore < {t}:")
            print(f"  All:   {np.mean(precs):.1f}% ± {np.std(precs):.1f}% (n={np.mean(ns):.1f}/seed)")
            print(f"  Seeds: {[f'{v:.1f}' for v in precs]}")
            if seed_data[t]['noncs']:
                nc_precs = [s['precision'] for s in seed_data[t]['noncs']]
                nc_ns = [s['n'] for s in seed_data[t]['noncs']]
                print(f"  NonCS: {np.mean(nc_precs):.1f}% ± {np.std(nc_precs):.1f}% (n={np.mean(nc_ns):.1f}/seed)")

    print("\n--- Above threshold (remaining MEDIUM) ---")
    for t in thresholds:
        if seed_above_data[t]['all']:
            precs = [s['precision'] for s in seed_above_data[t]['all']]
            ns = [s['n'] for s in seed_above_data[t]['all']]
            print(f"\nScore >= {t}:")
            print(f"  All:   {np.mean(precs):.1f}% ± {np.std(precs):.1f}% (n={np.mean(ns):.1f}/seed)")
            print(f"  Seeds: {[f'{v:.1f}' for v in precs]}")

    # ================================================================
    # SECTION 3: Estimated MEDIUM impact
    # ================================================================
    print("\n" + "=" * 70)
    print("SECTION 3: Estimated MEDIUM tier impact")
    print("=" * 70)

    for t in [0.75, 1.0, 1.25]:
        below_n = len([p for p in all_freq5_mech if p['score'] < t])
        above_n = len([p for p in all_freq5_mech if p['score'] >= t])
        print(f"\nThreshold {t}: {below_n} demoted, {above_n} remain in freq5_mech")

    # Save results
    results = {
        'total_freq5_mech': len(all_freq5_mech),
        'total_medium': len(all_medium),
        'score_below': {},
        'score_above': {},
        'medium_holdout': {
            'mean': float(np.mean(seed_medium_data)),
            'std': float(np.std(seed_medium_data)),
            'seeds': seed_medium_data,
        },
    }
    for t in thresholds:
        if seed_data[t]['all']:
            precs = [s['precision'] for s in seed_data[t]['all']]
            ns = [s['n'] for s in seed_data[t]['all']]
            results['score_below'][str(t)] = {
                'mean_precision': float(np.mean(precs)),
                'std_precision': float(np.std(precs)),
                'mean_n': float(np.mean(ns)),
                'seeds': [float(v) for v in precs],
            }
        if seed_above_data[t]['all']:
            precs = [s['precision'] for s in seed_above_data[t]['all']]
            ns = [s['n'] for s in seed_above_data[t]['all']]
            results['score_above'][str(t)] = {
                'mean_precision': float(np.mean(precs)),
                'std_precision': float(np.std(precs)),
                'mean_n': float(np.mean(ns)),
                'seeds': [float(v) for v in precs],
            }

    with open('data/analysis/h746_freq5_mech_low_score.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to data/analysis/h746_freq5_mech_low_score.json")
    print("\nDone.")


if __name__ == "__main__":
    main()
