#!/usr/bin/env python3
"""h758 v2: Deeper analysis of target_overlap_promotion.

Previous analysis found:
- Overall: 36.3% ± 5.0% (n=55.8/seed) — right at MEDIUM avg
- autoimmune: 18.0% (n=3.4/seed) — too small to demote
- "other" dominates (n=36/seed) at 35.4%

This script:
1. Breaks down "other" category by specific disease and drug
2. Checks kNN score distribution for target_overlap (is h740 score threshold applied?)
3. Checks if overlap count > 1 subgroup exists and performs better
4. Looks for drug-class patterns (biologics, hormones, etc.)
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from production_predictor import (
    DrugRepurposingPredictor,
    extract_cancer_types,
    DISEASE_HIERARCHY_GROUPS,
    HIERARCHY_EXCLUSIONS,
)

SEEDS = [42, 123, 456, 789, 2024]


def split_diseases(all_diseases, seed, train_ratio=0.8):
    rng = np.random.RandomState(seed)
    shuffled = list(all_diseases)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def recompute_gt_structures(predictor, train_set):
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
    for disease_id in train_set:
        if disease_id in predictor.ground_truth:
            for drug_id in predictor.ground_truth[disease_id]:
                new_freq[drug_id] += 1
    predictor.drug_train_freq = dict(new_freq)

    new_d2d = defaultdict(set)
    for disease_id in train_set:
        if disease_id in predictor.ground_truth:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            for drug_id in predictor.ground_truth[disease_id]:
                new_d2d[drug_id].add(disease_name)
    predictor.drug_to_diseases = dict(new_d2d)

    new_cancer = defaultdict(set)
    for disease_id in train_set:
        if disease_id in predictor.ground_truth:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            ctypes = extract_cancer_types(disease_name)
            if ctypes:
                for drug_id in predictor.ground_truth[disease_id]:
                    new_cancer[drug_id].update(ctypes)
    predictor.drug_cancer_types = dict(new_cancer)

    new_groups = defaultdict(set)
    for disease_id in train_set:
        if disease_id in predictor.ground_truth:
            disease_name = predictor.disease_names.get(disease_id, disease_id).lower()
            for group_name, data in DISEASE_HIERARCHY_GROUPS.items():
                if group_name in HIERARCHY_EXCLUSIONS:
                    exc_patterns = HIERARCHY_EXCLUSIONS[group_name]
                    if any(p in disease_name for p in exc_patterns):
                        continue
                parent = data.get("parent", "")
                subtypes = data.get("subtypes", [])
                all_patterns = [parent.lower()] + [s.lower() for s in subtypes]
                if any(p in disease_name for p in all_patterns if p):
                    for drug_id in predictor.ground_truth[disease_id]:
                        new_groups[drug_id].add(group_name)
    predictor.drug_disease_groups = dict(new_groups)

    predictor.train_diseases = [d for d in train_set if d in predictor.embeddings]
    if len(predictor.train_diseases) > 0:
        predictor.train_embeddings = np.array(
            [predictor.embeddings[d] for d in predictor.train_diseases]
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


def main():
    print("=" * 70)
    print("h758 v2: target_overlap_promotion Deep Analysis")
    print("=" * 70)

    print("\nLoading predictor...")
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
    print(f"Diseases with GT + embeddings: {len(all_diseases)}")

    # Collect per-prediction details across seeds
    all_preds = []

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n--- SEED {seed} ({seed_idx+1}/{len(SEEDS)}) ---")
        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        train_set = set(train_ids)
        originals = recompute_gt_structures(predictor, train_set)

        n_to = 0
        for disease_id in holdout_ids:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            category = predictor.categorize_disease(disease_name)
            try:
                result = predictor.predict(disease_name, top_n=30, include_filtered=True)
            except Exception:
                continue
            for pred in result.predictions:
                if (pred.category_specific_tier or "default") != 'target_overlap_promotion':
                    continue
                n_to += 1
                is_hit = (disease_id, pred.drug_id) in gt_set
                overlap_count = getattr(pred, 'target_overlap_count', 0)
                score = getattr(pred, 'score', 0)
                knn_rank = getattr(pred, 'knn_rank', 99)

                all_preds.append({
                    'seed': seed,
                    'drug_name': pred.drug_name,
                    'drug_id': pred.drug_id,
                    'disease_name': disease_name,
                    'disease_id': disease_id,
                    'category': category,
                    'is_hit': is_hit,
                    'overlap_count': overlap_count,
                    'score': score,
                    'knn_rank': knn_rank,
                })

        hits = sum(1 for p in all_preds if p['seed'] == seed and p['is_hit'])
        print(f"  {n_to} target_overlap preds, {hits} hits ({hits/n_to*100:.1f}%)" if n_to > 0 else "  0 preds")

        restore_gt_structures(predictor, originals)

    # === OVERALL ===
    total = len(all_preds)
    total_hits = sum(1 for p in all_preds if p['is_hit'])
    print(f"\n{'='*70}")
    print(f"TOTAL: {total} pred-instances across {len(SEEDS)} seeds")
    print(f"Overall: {total_hits/total*100:.1f}% ({total_hits}/{total})")

    # === kNN SCORE DISTRIBUTION ===
    print(f"\n{'='*70}")
    print("kNN SCORE DISTRIBUTION")
    print("="*70)
    scores = [p['score'] for p in all_preds]
    print(f"  Score range: {min(scores):.4f} - {max(scores):.4f}")
    print(f"  Mean: {np.mean(scores):.4f}, Median: {np.median(scores):.4f}")

    # Score bins
    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
    print(f"\n  Score Threshold Analysis (demote if score < threshold):")
    print(f"  {'Threshold':<12} {'Below%':>8} {'Below_n':>8} {'Above%':>8} {'Above_n':>8} {'Removed':>8}")
    for t in thresholds:
        below = [p for p in all_preds if p['score'] < t]
        above = [p for p in all_preds if p['score'] >= t]
        b_prec = sum(1 for p in below if p['is_hit']) / len(below) * 100 if below else 0
        a_prec = sum(1 for p in above if p['is_hit']) / len(above) * 100 if above else 0
        print(f"  score<{t:<5.3f} {b_prec:>7.1f}% {len(below):>8} {a_prec:>7.1f}% {len(above):>8} {len(below)/5:>7.1f}/seed")

    # === OVERLAP COUNT DISTRIBUTION ===
    print(f"\n{'='*70}")
    print("OVERLAP COUNT DISTRIBUTION")
    print("="*70)
    ov_counts = [p['overlap_count'] for p in all_preds]
    from collections import Counter
    ov_dist = Counter(ov_counts)
    for ov in sorted(ov_dist.keys()):
        preds = [p for p in all_preds if p['overlap_count'] == ov]
        hits = sum(1 for p in preds if p['is_hit'])
        prec = hits / len(preds) * 100 if preds else 0
        print(f"  overlap={ov}: {prec:.1f}% ({hits}/{len(preds)})")

    # === RANK DISTRIBUTION ===
    print(f"\n{'='*70}")
    print("RANK DISTRIBUTION")
    print("="*70)
    ranks = [p['knn_rank'] for p in all_preds]
    rank_dist = Counter(ranks)
    print(f"  Rank range: {min(ranks)} - {max(ranks)}")
    # Group into bins
    rank_bins = [(21, 25), (26, 30)]
    for lo, hi in rank_bins:
        preds = [p for p in all_preds if lo <= p['knn_rank'] <= hi]
        if preds:
            hits = sum(1 for p in preds if p['is_hit'])
            prec = hits / len(preds) * 100
            print(f"  R{lo}-{hi}: {prec:.1f}% ({hits}/{len(preds)})")

    # === PER-DRUG ANALYSIS (HOLDOUT) ===
    print(f"\n{'='*70}")
    print("PER-DRUG HOLDOUT ANALYSIS")
    print("="*70)

    drug_stats = defaultdict(lambda: {'hits': 0, 'total': 0, 'categories': set(), 'diseases': set(), 'scores': []})
    for p in all_preds:
        d = drug_stats[p['drug_name']]
        d['hits'] += int(p['is_hit'])
        d['total'] += 1
        d['categories'].add(p['category'])
        d['diseases'].add(p['disease_name'])
        d['scores'].append(p['score'])

    print(f"\n  {'Drug':<30} {'Prec':>6} {'H':>3}/{'N':>4} {'n/seed':>7} {'Cats':<20} {'AvgScore':>9}")
    print(f"  {'-'*30} {'-'*6} {'-'*3} {'-'*4} {'-'*7} {'-'*20} {'-'*9}")
    for drug, st in sorted(drug_stats.items(), key=lambda x: -x[1]['total']):
        prec = st['hits'] / st['total'] * 100
        n_seed = st['total'] / len(SEEDS)
        cats = ','.join(sorted(st['categories']))[:20]
        avg_score = np.mean(st['scores'])
        print(f"  {drug[:30]:<30} {prec:5.1f}% {st['hits']:>3}/{st['total']:>4} {n_seed:>6.1f} {cats:<20} {avg_score:>8.4f}")

    # === PER-DISEASE ANALYSIS (HOLDOUT) ===
    print(f"\n{'='*70}")
    print("PER-DISEASE HOLDOUT ANALYSIS (top 25 by n)")
    print("="*70)

    disease_stats = defaultdict(lambda: {'hits': 0, 'total': 0, 'category': '', 'drugs': set()})
    for p in all_preds:
        d = disease_stats[p['disease_name']]
        d['hits'] += int(p['is_hit'])
        d['total'] += 1
        d['category'] = p['category']
        d['drugs'].add(p['drug_name'])

    print(f"\n  {'Disease':<45} {'Prec':>6} {'H':>3}/{'N':>4} {'Cat':<15} {'Drugs'}")
    print(f"  {'-'*45} {'-'*6} {'-'*3} {'-'*4} {'-'*15} {'-'*30}")
    for disease, st in sorted(disease_stats.items(), key=lambda x: -x[1]['total'])[:25]:
        prec = st['hits'] / st['total'] * 100
        drugs = ', '.join(sorted(st['drugs'])[:3])
        if len(st['drugs']) > 3:
            drugs += f"... (+{len(st['drugs'])-3})"
        print(f"  {disease[:45]:<45} {prec:5.1f}% {st['hits']:>3}/{st['total']:>4} {st['category']:<15} {drugs}")

    # === ZERO-HIT DRUGS (potential demotions) ===
    print(f"\n{'='*70}")
    print("ZERO-HIT DRUGS (0% holdout, n >= 3/seed)")
    print("="*70)
    zero_drugs = [(drug, st) for drug, st in drug_stats.items()
                  if st['hits'] == 0 and st['total'] / len(SEEDS) >= 3]
    zero_drugs.sort(key=lambda x: -x[1]['total'])
    zero_preds_total = sum(st['total'] for _, st in zero_drugs)
    print(f"\n  {len(zero_drugs)} drugs with 0% holdout and n>=3/seed ({zero_preds_total} pred-instances)")
    for drug, st in zero_drugs:
        n_seed = st['total'] / len(SEEDS)
        cats = ','.join(sorted(st['categories']))
        diseases = sorted(st['diseases'])
        print(f"  {drug:<30} n={n_seed:.1f}/seed  cats={cats}")
        for d in diseases[:5]:
            print(f"    → {d}")

    # === SIMULATED DEMOTION IMPACT ===
    print(f"\n{'='*70}")
    print("SIMULATED DEMOTION IMPACT")
    print("="*70)

    # What if we remove zero-hit drugs?
    zero_drug_names = {d for d, _ in zero_drugs}
    remaining = [p for p in all_preds if p['drug_name'] not in zero_drug_names]
    if remaining:
        rem_hits = sum(1 for p in remaining if p['is_hit'])
        rem_total = len(remaining)
        rem_prec = rem_hits / rem_total * 100
        print(f"\n  Remove zero-hit drugs (n>=3/seed):")
        print(f"    Remaining: {rem_prec:.1f}% ({rem_hits}/{rem_total})")
        print(f"    Removed: {zero_preds_total} pred-instances ({zero_preds_total/5:.0f}/seed)")
        print(f"    NOTE: Drug-specific demotions need n≈30 for reliable signal")

    # What if we combine autoimmune + respiratory demotion?
    weak_cats = {'autoimmune', 'respiratory'}
    remaining2 = [p for p in all_preds if p['category'] not in weak_cats]
    if remaining2:
        rem_hits2 = sum(1 for p in remaining2 if p['is_hit'])
        rem_total2 = len(remaining2)
        rem_prec2 = rem_hits2 / rem_total2 * 100
        demoted2 = [p for p in all_preds if p['category'] in weak_cats]
        dem_hits2 = sum(1 for p in demoted2 if p['is_hit'])
        dem_total2 = len(demoted2)
        print(f"\n  Demote autoimmune + respiratory:")
        print(f"    Remaining: {rem_prec2:.1f}% ({rem_hits2}/{rem_total2})")
        print(f"    Demoted: {dem_hits2}/{dem_total2} = {dem_hits2/dem_total2*100:.1f}%")
        print(f"    Moved: {dem_total2/5:.1f}/seed")

    # What about score < 0.03 demotion?
    low_score = [p for p in all_preds if p['score'] < 0.03]
    high_score = [p for p in all_preds if p['score'] >= 0.03]
    if low_score and high_score:
        ls_hits = sum(1 for p in low_score if p['is_hit'])
        hs_hits = sum(1 for p in high_score if p['is_hit'])
        print(f"\n  Demote score < 0.03:")
        print(f"    Below: {ls_hits/len(low_score)*100:.1f}% ({ls_hits}/{len(low_score)})")
        print(f"    Above: {hs_hits/len(high_score)*100:.1f}% ({hs_hits}/{len(high_score)})")
        print(f"    Moved: {len(low_score)/5:.1f}/seed")


if __name__ == "__main__":
    main()
