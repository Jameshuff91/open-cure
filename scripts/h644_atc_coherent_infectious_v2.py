#!/usr/bin/env python3
"""
h644: ATC Coherent Infectious Quality Investigation (v2)

Simpler analysis focusing on:
1. All atc_coherent MEDIUM by category + mechanism
2. Detailed infectious NoMech drug-level analysis
3. Holdout evaluation with 5 seeds
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from production_predictor import (
    ConfidenceTier,
    DrugRepurposingPredictor,
)

sys.path.insert(0, str(Path(__file__).parent))
from h393_holdout_tier_validation import (
    split_diseases,
    recompute_gt_structures,
    restore_gt_structures,
)


def collect_atc_coherent(predictor, disease_ids, gt_set):
    """Collect all atc_coherent MEDIUM predictions with metadata."""
    preds_list = []
    for disease_id in disease_ids:
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        try:
            result = predictor.predict(disease_name, top_n=30, include_filtered=True)
        except Exception:
            continue
        for pred in result.predictions:
            if pred.confidence_tier != ConfidenceTier.MEDIUM:
                continue
            cs = pred.category_specific_tier or ''
            if not cs.startswith('atc_coherent'):
                continue
            is_hit = (disease_id, pred.drug_id) in gt_set
            preds_list.append({
                'disease_id': disease_id,
                'disease': disease_name,
                'drug': pred.drug_name,
                'drug_id': pred.drug_id,
                'rank': pred.rank,
                'hit': is_hit,
                'mech': pred.mechanism_support,
                'transe': pred.transe_consilience,
                'category': pred.category,
                'cat_specific': cs,
            })
    return preds_list


def stratify_and_report(preds_list, label=""):
    """Stratify predictions and compute precision."""
    strats = defaultdict(lambda: {'hits': 0, 'total': 0})

    for p in preds_list:
        cat = p['category']
        mech = 'M' if p['mech'] else 'NM'
        transe = 'T' if p['transe'] else 'NT'
        cs = p['cat_specific']

        for key in [
            'ALL',
            f'cat={cat}',
            f'mech={mech}',
            f'rule={cs}',
            f'cat={cat}+mech={mech}',
            f'cat={cat}+transe={transe}',
            f'rule={cs}+mech={mech}',
        ]:
            strats[key]['total'] += 1
            if p['hit']:
                strats[key]['hits'] += 1

    results = {}
    for key, s in strats.items():
        total = s['total']
        prec = s['hits'] / total * 100 if total > 0 else 0
        results[key] = {'hits': s['hits'], 'total': total, 'precision': round(prec, 1)}

    return results


def main():
    seeds = [42, 123, 456, 789, 2024]

    print("=" * 70)
    print("h644: ATC Coherent Infectious Quality Investigation")
    print("=" * 70)

    predictor = DrugRepurposingPredictor()

    gt_path = predictor.reference_dir / "expanded_ground_truth.json"
    with open(gt_path) as f:
        gt_data = json.load(f)

    gt_set = set()
    for disease_id, drugs in gt_data.items():
        for drug in drugs:
            if isinstance(drug, str):
                gt_set.add((disease_id, drug))
            elif isinstance(drug, dict):
                gt_set.add((disease_id, drug.get("drug_id") or drug.get("drug")))

    all_diseases = [d for d in predictor.ground_truth if d in predictor.embeddings]
    print(f"Diseases: {len(all_diseases)}, GT pairs: {len(gt_set)}")

    # Full-data baseline
    print("\n--- FULL-DATA BASELINE ---")
    full_preds = collect_atc_coherent(predictor, all_diseases, gt_set)
    print(f"Total atc_coherent MEDIUM: {len(full_preds)}")

    full_strats = stratify_and_report(full_preds)
    print(f"\n{'Key':<50} {'Prec%':>8} {'Hits/Total':>12}")
    print("-" * 75)
    for key in sorted(full_strats.keys()):
        s = full_strats[key]
        if s['total'] >= 3:
            print(f"  {key:<50} {s['precision']:>7.1f}% {s['hits']:>4}/{s['total']:<5}")

    # Infectious NoMech drug breakdown
    inf_nomech = [p for p in full_preds if p['category'] == 'infectious' and not p['mech']]
    print(f"\n--- INFECTIOUS NoMech DRUGS (full data, n={len(inf_nomech)}) ---")
    drug_counts = Counter(p['drug'] for p in inf_nomech)
    drug_hits = Counter(p['drug'] for p in inf_nomech if p['hit'])
    print(f"\n{'Drug':<30} {'Hits/Total':>12} {'Prec%':>8}")
    print("-" * 55)
    for drug, total in drug_counts.most_common():
        hits = drug_hits.get(drug, 0)
        prec = hits / total * 100
        print(f"  {drug:<30} {hits:>4}/{total:<5} {prec:>7.1f}%")

    # Disease breakdown
    print(f"\n--- INFECTIOUS NoMech DISEASES (full data) ---")
    by_disease = defaultdict(list)
    for p in inf_nomech:
        by_disease[p['disease']].append(p)
    for disease, preds in sorted(by_disease.items()):
        hits = sum(1 for p in preds if p['hit'])
        print(f"\n  {disease}: {hits}/{len(preds)}")
        for p in sorted(preds, key=lambda x: x['rank']):
            marker = 'HIT' if p['hit'] else 'miss'
            extra = ' [TransE]' if p['transe'] else ''
            print(f"    R{p['rank']:>2}: {p['drug']:<25} [{marker}]{extra}")

    # Holdout evaluation
    print("\n\n--- HOLDOUT EVALUATION (5 seeds) ---")
    all_seed_strats = defaultdict(list)
    all_seed_n = defaultdict(list)

    for seed_idx, seed in enumerate(seeds):
        print(f"Seed {seed} ({seed_idx+1}/{len(seeds)})...", end=" ", flush=True)
        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        originals = recompute_gt_structures(predictor, set(train_ids))
        holdout_preds = collect_atc_coherent(predictor, holdout_ids, gt_set)
        restore_gt_structures(predictor, originals)

        strats = stratify_and_report(holdout_preds)
        for key, s in strats.items():
            all_seed_strats[key].append(s['precision'])
            all_seed_n[key].append(s['total'])
        print(f"n={len(holdout_preds)}")

    # Aggregate
    print(f"\n{'Key':<50} {'Holdout%':>10} {'±std':>8} {'N/seed':>8} {'Full%':>8}")
    print("-" * 90)

    results_list = []
    for key in all_seed_strats:
        mean_p = np.mean(all_seed_strats[key])
        std_p = np.std(all_seed_strats[key])
        mean_n = np.mean(all_seed_n[key])
        full_p = full_strats.get(key, {}).get('precision', 0)
        results_list.append((key, mean_p, std_p, mean_n, full_p))

    results_list.sort(key=lambda x: -x[1])
    for key, mp, sp, mn, fp in results_list:
        if mn >= 3:
            print(f"  {key:<50} {mp:>8.1f}% {sp:>7.1f}% {mn:>7.1f} {fp:>7.1f}%")


if __name__ == "__main__":
    main()
