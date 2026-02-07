#!/usr/bin/env python3
"""
h642: MEDIUM Default Sub-Stratification for HIGH Promotion

After h630 promoted TransE+mech/rank≤5 to HIGH, what remains in MEDIUM?
Can we find additional promotable subsets?

Focus on mechanism + rank≤10 (without TransE) as the next best signal.
"""
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')

import json
import numpy as np
from collections import defaultdict
from production_predictor import DrugRepurposingPredictor
from h393_holdout_tier_validation import (
    split_diseases, recompute_gt_structures, restore_gt_structures
)

CS_DRUGS = {
    'dexamethasone', 'prednisolone', 'prednisone', 'methylprednisolone',
    'betamethasone', 'hydrocortisone', 'triamcinolone', 'budesonide',
    'fluticasone', 'mometasone', 'beclomethasone', 'cortisone',
    'fluocinolone', 'clobetasol', 'fluocinonide', 'halobetasol',
    'desoximetasone', 'desonide'
}


def is_cs(name):
    return name.lower() in CS_DRUGS


def main():
    print("=" * 70)
    print("h642: MEDIUM Default Sub-Stratification")
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
    seeds = [42, 123, 456, 789, 2024]

    print(f"Expanded GT: {len(gt_set)} pairs")
    print(f"Diseases: {len(all_diseases)}")

    # Per-seed tracking for MEDIUM predictions
    # Signal buckets: TransE, mechanism, rank, category
    seed_buckets = defaultdict(lambda: defaultdict(lambda: {'hits': 0, 'total': 0}))
    seed_tiers = defaultdict(lambda: defaultdict(lambda: {'hits': 0, 'total': 0}))

    # Drug-level detail for promotable subsets
    drug_detail = defaultdict(lambda: {'hits': 0, 'total': 0, 'cats': set()})

    for seed in seeds:
        print(f"\n  Processing seed {seed}...")
        train_diseases, holdout_diseases = split_diseases(all_diseases, seed)
        train_set = set(train_diseases)
        originals = recompute_gt_structures(predictor, train_set)

        for disease_id in holdout_diseases:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            try:
                result = predictor.predict(disease_name, top_n=30, include_filtered=True)
            except Exception:
                continue

            cat = result.category

            for p in result.predictions:
                hit = (disease_id, p.drug_id) in gt_set
                tier_str = p.confidence_tier.name
                cat_specific = getattr(p, 'category_specific_tier', None)
                drug_is_cs = is_cs(p.drug_name)

                seed_tiers[seed][tier_str]['total'] += 1
                if hit:
                    seed_tiers[seed][tier_str]['hits'] += 1

                if tier_str != 'MEDIUM':
                    continue

                # Get signals
                has_transe = getattr(p, 'transe_consilience', False)
                has_mech = getattr(p, 'mechanism_support', False) or getattr(p, 'has_mechanism', False)
                rank = getattr(p, 'knn_rank', 99)

                # Classify by signal combination (what REMAINS in MEDIUM after h630)
                # h630 already promoted: TransE + (mech OR rank≤5) → HIGH
                # So remaining TransE predictions in MEDIUM must be TransE without mech and rank>5

                # Non-CS only for clean analysis
                if drug_is_cs:
                    bucket = 'CS'
                elif has_transe:
                    # These are TransE but NOT promoted by h630 — must be TransE without mech and rank>5
                    bucket = 'TransE_no_mech_rank>5'
                elif has_mech and rank <= 5:
                    bucket = 'Mech_Rank<=5'
                elif has_mech and rank <= 10:
                    bucket = 'Mech_Rank6-10'
                elif has_mech:
                    bucket = 'Mech_Rank>10'
                elif rank <= 5:
                    bucket = 'Rank<=5_NoMech'
                elif rank <= 10:
                    bucket = 'Rank6-10_NoMech'
                else:
                    bucket = 'NoSignal_Rank>10'

                seed_buckets[seed][bucket]['total'] += 1
                if hit:
                    seed_buckets[seed][bucket]['hits'] += 1

                # Track detail for promotable buckets
                if bucket in ('Mech_Rank<=5', 'Mech_Rank6-10'):
                    drug_detail[f"{bucket}:{p.drug_name}"]['total'] += 1
                    drug_detail[f"{bucket}:{p.drug_name}"]['cats'].add(cat)
                    if hit:
                        drug_detail[f"{bucket}:{p.drug_name}"]['hits'] += 1

        restore_gt_structures(predictor, originals)

    # Report
    def compute_stats(per_seed_data):
        precisions = []
        total_n = 0
        for seed in seeds:
            d = per_seed_data.get(seed, {'hits': 0, 'total': 0})
            if d['total'] > 0:
                precisions.append(d['hits'] / d['total'] * 100)
                total_n += d['total']
            else:
                precisions.append(0.0)
        return np.mean(precisions), np.std(precisions), total_n / len(seeds)

    print("\n" + "=" * 70)
    print("TIER OVERVIEW")
    print("=" * 70)
    for tier in ['GOLDEN', 'HIGH', 'MEDIUM', 'LOW', 'FILTER']:
        per_seed = {s: seed_tiers[s].get(tier, {'hits': 0, 'total': 0}) for s in seeds}
        mean, std, n = compute_stats(per_seed)
        print(f"  {tier:<10s} {mean:5.1f}% ± {std:4.1f}% (n={n:.0f}/seed)")

    print("\n" + "=" * 70)
    print("MEDIUM SIGNAL BUCKETS (what remains after h630)")
    print("=" * 70)

    # Collect all bucket names
    all_buckets = set()
    for s in seeds:
        all_buckets.update(seed_buckets[s].keys())

    for bucket in sorted(all_buckets, key=lambda b: sum(seed_buckets[s].get(b, {'total': 0})['total'] for s in seeds), reverse=True):
        per_seed = {s: seed_buckets[s].get(bucket, {'hits': 0, 'total': 0}) for s in seeds}
        mean, std, n = compute_stats(per_seed)
        total = sum(seed_buckets[s].get(bucket, {'total': 0})['total'] for s in seeds)
        flag = ""
        if mean > 55 and n >= 5:
            flag = " *** PROMOTE TO HIGH?"
        elif mean > 45 and n >= 10:
            flag = " ** NEAR-HIGH"
        elif mean < 20 and n >= 10:
            flag = " ** BELOW LOW"
        print(f"  {bucket:<30s} {mean:5.1f}% ± {std:5.1f}% (n={n:.0f}/seed, total={total}){flag}")

    # Composite groups
    print("\n--- Composite signal groups ---")

    # Mech + rank<=10 combined
    mech_r10 = {}
    for s in seeds:
        h = seed_buckets[s].get('Mech_Rank<=5', {'hits': 0, 'total': 0})['hits'] + \
            seed_buckets[s].get('Mech_Rank6-10', {'hits': 0, 'total': 0})['hits']
        t = seed_buckets[s].get('Mech_Rank<=5', {'hits': 0, 'total': 0})['total'] + \
            seed_buckets[s].get('Mech_Rank6-10', {'hits': 0, 'total': 0})['total']
        mech_r10[s] = {'hits': h, 'total': t}
    mean, std, n = compute_stats(mech_r10)
    print(f"  Mech+Rank<=10 (non-CS,non-TransE) {mean:5.1f}% ± {std:5.1f}% (n={n:.0f}/seed)")

    # All mechanism (non-CS, non-TransE)
    all_mech = {}
    for s in seeds:
        h = seed_buckets[s].get('Mech_Rank<=5', {'hits': 0, 'total': 0})['hits'] + \
            seed_buckets[s].get('Mech_Rank6-10', {'hits': 0, 'total': 0})['hits'] + \
            seed_buckets[s].get('Mech_Rank>10', {'hits': 0, 'total': 0})['hits']
        t = seed_buckets[s].get('Mech_Rank<=5', {'hits': 0, 'total': 0})['total'] + \
            seed_buckets[s].get('Mech_Rank6-10', {'hits': 0, 'total': 0})['total'] + \
            seed_buckets[s].get('Mech_Rank>10', {'hits': 0, 'total': 0})['total']
        all_mech[s] = {'hits': h, 'total': t}
    mean, std, n = compute_stats(all_mech)
    print(f"  All Mech (non-CS, non-TransE) {mean:5.1f}% ± {std:5.1f}% (n={n:.0f}/seed)")

    # All no-signal
    no_signal = {}
    for s in seeds:
        h = seed_buckets[s].get('Rank<=5_NoMech', {'hits': 0, 'total': 0})['hits'] + \
            seed_buckets[s].get('Rank6-10_NoMech', {'hits': 0, 'total': 0})['hits'] + \
            seed_buckets[s].get('NoSignal_Rank>10', {'hits': 0, 'total': 0})['hits']
        t = seed_buckets[s].get('Rank<=5_NoMech', {'hits': 0, 'total': 0})['total'] + \
            seed_buckets[s].get('Rank6-10_NoMech', {'hits': 0, 'total': 0})['total'] + \
            seed_buckets[s].get('NoSignal_Rank>10', {'hits': 0, 'total': 0})['total']
        no_signal[s] = {'hits': h, 'total': t}
    mean, std, n = compute_stats(no_signal)
    print(f"  No Mech, No TransE (all ranks) {mean:5.1f}% ± {std:5.1f}% (n={n:.0f}/seed)")

    # Drug detail for Mech+Rank<=5 and Mech+Rank6-10
    print("\n" + "=" * 70)
    print("TOP DRUGS in Mech+Rank<=5/6-10 MEDIUM (non-CS, non-TransE)")
    print("=" * 70)
    mech_rank_drugs = [(k, v) for k, v in drug_detail.items() if v['total'] >= 3]
    mech_rank_drugs.sort(key=lambda x: x[1]['total'], reverse=True)
    for key, v in mech_rank_drugs[:20]:
        bucket, drug = key.split(':', 1)
        prec = v['hits'] / v['total'] * 100 if v['total'] > 0 else 0
        cats = ', '.join(sorted(v['cats']))
        print(f"  [{bucket}] {drug:<25s} {prec:5.1f}% ({v['hits']}/{v['total']}) cats: {cats}")


if __name__ == "__main__":
    main()
