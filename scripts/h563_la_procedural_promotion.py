#!/usr/bin/env python3
"""
h563: Local Anesthetic Procedural — Promotion or Block Analysis

h561 showed local_anesthetic_procedural within MEDIUM has ~44-50% holdout.
These are LA drugs demoted to LOW but rescued back to MEDIUM by target_overlap.

Questions:
1. What's the holdout precision for these predictions?
2. Should they be promoted to HIGH (if high precision)?
3. Or should they be blocked from target_overlap rescue (if leakage)?
4. What specific drugs/diseases are involved?
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
    DISEASE_HIERARCHY_GROUPS,
)


def split_diseases(all_diseases, seed, train_ratio=0.8):
    rng = np.random.RandomState(seed)
    shuffled = list(all_diseases)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def recompute_gt_structures(predictor, train_disease_ids):
    """Recompute GT-derived structures from training diseases only."""
    from production_predictor import extract_cancer_types

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

    new_cancer_types = {}
    for drug_id, diseases in new_d2d.items():
        cancer_types = set()
        for d in diseases:
            cancer_types.update(extract_cancer_types(d))
        if cancer_types:
            new_cancer_types[drug_id] = cancer_types
    predictor.drug_cancer_types = new_cancer_types

    new_groups = defaultdict(set)
    for drug_id, diseases in new_d2d.items():
        for d in diseases:
            for group_name, group_data in DISEASE_HIERARCHY_GROUPS.items():
                for disease_pattern in group_data.get("diseases", []):
                    if disease_pattern.lower() in d.lower():
                        new_groups[drug_id].add(group_name)
    predictor.drug_disease_groups = dict(new_groups)

    train_disease_list = [d for d in predictor.train_diseases if d in train_disease_ids]
    predictor.train_diseases = train_disease_list
    indices = [i for i, d in enumerate(originals["train_diseases"]) if d in train_disease_ids]
    predictor.train_embeddings = originals["train_embeddings"][indices]
    predictor.train_disease_categories = {
        d: originals["train_disease_categories"][d]
        for d in train_disease_list
        if d in originals["train_disease_categories"]
    }

    return originals


def restore_gt_structures(predictor, originals):
    for key, val in originals.items():
        setattr(predictor, key, val)


def main():
    print("=" * 70)
    print("h563: Local Anesthetic Procedural — Promotion or Block Analysis")
    print("=" * 70)

    predictor = DrugRepurposingPredictor()
    gt_diseases = list(predictor.ground_truth.keys())
    print(f"\nGT diseases: {len(gt_diseases)}")

    # First: full-data analysis of LA procedural predictions by tier
    print("\n--- Full-Data LA Procedural Analysis ---")
    la_by_tier = defaultdict(list)
    for disease_id in gt_diseases:
        if disease_id not in predictor.embeddings:
            continue
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        result = predictor.predict(disease_name, k=20, top_n=30, include_filtered=True)
        gt_drugs = set(predictor.ground_truth.get(disease_id, []))

        for pred in result.predictions:
            if pred.category_specific_tier and 'local_anesthetic_procedural' in pred.category_specific_tier:
                is_hit = pred.drug_id in gt_drugs
                la_by_tier[pred.confidence_tier.value].append({
                    'drug': pred.drug_name,
                    'disease': disease_name,
                    'category': pred.category,
                    'hit': is_hit,
                    'rank': pred.rank,
                })

    print(f"\nFull-data LA procedural predictions by tier:")
    for tier in ['GOLDEN', 'HIGH', 'MEDIUM', 'LOW', 'FILTER']:
        preds = la_by_tier.get(tier, [])
        hits = sum(1 for p in preds if p['hit'])
        pct = hits / len(preds) * 100 if preds else 0
        print(f"  {tier}: {len(preds)} predictions, {pct:.1f}% full-data precision ({hits}/{len(preds)})")
        if tier == 'MEDIUM':
            # Show details
            drugs = Counter(p['drug'] for p in preds)
            cats = Counter(p['category'] for p in preds)
            print(f"    Drugs: {dict(drugs)}")
            print(f"    Categories: {dict(cats)}")
            for p in preds:
                gt_str = "HIT" if p['hit'] else "miss"
                print(f"    {p['drug']} → {p['disease']} [{p['category']}] rank={p['rank']} {gt_str}")

    # Now holdout analysis
    print("\n\n--- Holdout Analysis (5 seeds) ---")
    seeds = [42, 123, 456, 789, 1337]

    # Track per-seed stats
    la_medium_results = []
    la_low_results = []
    overall_results = {tier: {'hits': [], 'total': []} for tier in ['GOLDEN', 'HIGH', 'MEDIUM', 'LOW', 'FILTER']}

    for seed in seeds:
        train, holdout = split_diseases(gt_diseases, seed)
        train_set = set(train)

        originals = recompute_gt_structures(predictor, train_set)

        la_med_hits, la_med_total = 0, 0
        la_low_hits, la_low_total = 0, 0
        tier_hits = defaultdict(int)
        tier_total = defaultdict(int)

        for disease_id in holdout:
            if disease_id not in predictor.embeddings:
                continue
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            result = predictor.predict(disease_name, k=20, top_n=30, include_filtered=True)
            gt_drugs = set(predictor.ground_truth.get(disease_id, []))
            if not gt_drugs:
                continue

            for pred in result.predictions:
                is_hit = pred.drug_id in gt_drugs
                tier_val = pred.confidence_tier.value
                tier_hits[tier_val] += int(is_hit)
                tier_total[tier_val] += 1

                if pred.category_specific_tier and 'local_anesthetic_procedural' in pred.category_specific_tier:
                    if pred.confidence_tier == ConfidenceTier.MEDIUM:
                        la_med_hits += int(is_hit)
                        la_med_total += 1
                    elif pred.confidence_tier == ConfidenceTier.LOW:
                        la_low_hits += int(is_hit)
                        la_low_total += 1

        la_medium_results.append((la_med_hits, la_med_total))
        la_low_results.append((la_low_hits, la_low_total))

        for tier in overall_results:
            overall_results[tier]['hits'].append(tier_hits[tier])
            overall_results[tier]['total'].append(tier_total[tier])

        la_m_pct = la_med_hits / la_med_total * 100 if la_med_total > 0 else 0
        la_l_pct = la_low_hits / la_low_total * 100 if la_low_total > 0 else 0
        print(f"  Seed {seed}: LA MEDIUM = {la_m_pct:.1f}% ({la_med_hits}/{la_med_total}), LA LOW = {la_l_pct:.1f}% ({la_low_hits}/{la_low_total})")

        restore_gt_structures(predictor, originals)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # LA MEDIUM
    la_m_precs = [h / t * 100 if t > 0 else 0 for h, t in la_medium_results]
    la_m_ns = [t for _, t in la_medium_results]
    print(f"\nLA Procedural MEDIUM (target_overlap rescued):")
    print(f"  Holdout: {np.mean(la_m_precs):.1f}% ± {np.std(la_m_precs):.1f}%")
    print(f"  n/seed: {la_m_ns} (mean={np.mean(la_m_ns):.1f})")

    # LA LOW
    la_l_precs = [h / t * 100 if t > 0 else 0 for h, t in la_low_results]
    la_l_ns = [t for _, t in la_low_results]
    print(f"\nLA Procedural LOW (remained demoted):")
    print(f"  Holdout: {np.mean(la_l_precs):.1f}% ± {np.std(la_l_precs):.1f}%")
    print(f"  n/seed: {la_l_ns} (mean={np.mean(la_l_ns):.1f})")

    # Tier baselines
    print(f"\nTier Baselines:")
    for tier in ['GOLDEN', 'HIGH', 'MEDIUM', 'LOW', 'FILTER']:
        hits = overall_results[tier]['hits']
        total = overall_results[tier]['total']
        precs = [h / t * 100 if t > 0 else 0 for h, t in zip(hits, total)]
        print(f"  {tier}: {np.mean(precs):.1f}% ± {np.std(precs):.1f}% (n/seed={np.mean(total):.0f})")

    # Impact projection: what if we promote LA MEDIUM → HIGH?
    print(f"\n--- Projection: LA Procedural MEDIUM → HIGH ---")
    curr_h_precs = [h / t * 100 if t > 0 else 0 for h, t in zip(overall_results['HIGH']['hits'], overall_results['HIGH']['total'])]
    curr_m_precs = [h / t * 100 if t > 0 else 0 for h, t in zip(overall_results['MEDIUM']['hits'], overall_results['MEDIUM']['total'])]

    new_h_precs = []
    new_m_precs = []
    for i in range(len(seeds)):
        nh_h = overall_results['HIGH']['hits'][i] + la_medium_results[i][0]
        nh_t = overall_results['HIGH']['total'][i] + la_medium_results[i][1]
        new_h_precs.append(nh_h / nh_t * 100 if nh_t > 0 else 0)

        nm_h = overall_results['MEDIUM']['hits'][i] - la_medium_results[i][0]
        nm_t = overall_results['MEDIUM']['total'][i] - la_medium_results[i][1]
        new_m_precs.append(nm_h / nm_t * 100 if nm_t > 0 else 0)

    print(f"  HIGH: {np.mean(curr_h_precs):.1f}% → {np.mean(new_h_precs):.1f}% ({np.mean(new_h_precs) - np.mean(curr_h_precs):+.1f}pp)")
    print(f"  MEDIUM: {np.mean(curr_m_precs):.1f}% → {np.mean(new_m_precs):.1f}% ({np.mean(new_m_precs) - np.mean(curr_m_precs):+.1f}pp)")

    # Impact projection: what if we block LA from target_overlap rescue?
    print(f"\n--- Projection: Block LA from target_overlap rescue → stays LOW ---")
    new_m2_precs = []
    new_l2_precs = []
    for i in range(len(seeds)):
        # Remove from MEDIUM, add to LOW
        nm_h = overall_results['MEDIUM']['hits'][i] - la_medium_results[i][0]
        nm_t = overall_results['MEDIUM']['total'][i] - la_medium_results[i][1]
        new_m2_precs.append(nm_h / nm_t * 100 if nm_t > 0 else 0)

        nl_h = overall_results['LOW']['hits'][i] + la_medium_results[i][0]
        nl_t = overall_results['LOW']['total'][i] + la_medium_results[i][1]
        new_l2_precs.append(nl_h / nl_t * 100 if nl_t > 0 else 0)

    curr_l_precs = [h / t * 100 if t > 0 else 0 for h, t in zip(overall_results['LOW']['hits'], overall_results['LOW']['total'])]
    print(f"  MEDIUM: {np.mean(curr_m_precs):.1f}% → {np.mean(new_m2_precs):.1f}% ({np.mean(new_m2_precs) - np.mean(curr_m_precs):+.1f}pp)")
    print(f"  LOW: {np.mean(curr_l_precs):.1f}% → {np.mean(new_l2_precs):.1f}% ({np.mean(new_l2_precs) - np.mean(curr_l_precs):+.1f}pp)")

    # Decision
    print(f"\n" + "=" * 70)
    print("DECISION")
    print("=" * 70)

    mean_la_m = np.mean(la_m_precs)
    mean_la_n = np.mean(la_m_ns)
    mean_high = np.mean(curr_h_precs)
    mean_medium = np.mean(curr_m_precs)

    if mean_la_n < 5:
        print(f"INSUFFICIENT: Too few LA MEDIUM predictions (n/seed={mean_la_n:.1f} < 5)")
        print("Cannot reliably evaluate. Consider keeping as-is.")
    elif mean_la_m >= mean_high:
        print(f"PROMOTE TO HIGH: LA MEDIUM ({mean_la_m:.1f}%) >= HIGH ({mean_high:.1f}%)")
    elif mean_la_m >= mean_medium:
        print(f"KEEP AS MEDIUM: LA MEDIUM ({mean_la_m:.1f}%) is at MEDIUM level ({mean_medium:.1f}%), no change needed")
    else:
        print(f"BLOCK RESCUE: LA MEDIUM ({mean_la_m:.1f}%) < MEDIUM avg ({mean_medium:.1f}%), should block from target_overlap")

    # Save
    results = {
        "hypothesis": "h563",
        "la_medium_holdout": round(np.mean(la_m_precs), 1),
        "la_medium_std": round(np.std(la_m_precs), 1),
        "la_medium_n_per_seed": la_m_ns,
        "la_low_holdout": round(np.mean(la_l_precs), 1),
        "la_low_n_per_seed": la_l_ns,
        "current_high": round(np.mean(curr_h_precs), 1),
        "current_medium": round(np.mean(curr_m_precs), 1),
        "projected_promote_high": round(np.mean(new_h_precs), 1),
        "projected_promote_medium": round(np.mean(new_m_precs), 1),
        "projected_block_medium": round(np.mean(new_m2_precs), 1),
        "projected_block_low": round(np.mean(new_l2_precs), 1),
    }

    with open("data/analysis/h563_output.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to data/analysis/h563_output.json")


if __name__ == "__main__":
    main()
