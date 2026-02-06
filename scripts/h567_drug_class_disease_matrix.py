#!/usr/bin/env python3
"""
h567: Comprehensive Drug Class × Disease Category Mismatch Matrix

Build a cross-tabulation of SOC drug class × disease category holdout precision
for MEDIUM predictions. Identify zero/near-zero combinations that could be demoted.

Extends the successful antimicrobial mismatch (h560, +0.9pp) approach to all categories.
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
    DRUG_CLASS_SOC_MAPPINGS,
    _DRUG_TO_SOC,
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


def classify_drug(drug_name):
    """Classify a drug into SOC drug class(es)."""
    drug_lower = drug_name.lower()
    soc_matches = _DRUG_TO_SOC.get(drug_lower, [])
    if soc_matches:
        return [s['class'] for s in soc_matches]
    return ['unclassified']


def main():
    print("=" * 70)
    print("h567: Drug Class × Disease Category Mismatch Matrix")
    print("=" * 70)

    predictor = DrugRepurposingPredictor()
    gt_diseases = [d for d in predictor.ground_truth if d in predictor.embeddings]
    print(f"\nGT diseases with embeddings: {len(gt_diseases)}")

    # Step 1: Full-data analysis of drug class × category for MEDIUM predictions
    print("\n--- Step 1: Full-Data Drug Class × Category Matrix (MEDIUM only) ---")

    # Collect MEDIUM predictions with drug class classification
    medium_matrix = defaultdict(lambda: {'hits': 0, 'total': 0, 'examples': []})

    for disease_id in gt_diseases:
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        result = predictor.predict(disease_name, k=20, top_n=30, include_filtered=True)
        gt_drugs = set(predictor.ground_truth.get(disease_id, []))

        for pred in result.predictions:
            if pred.confidence_tier != ConfidenceTier.MEDIUM:
                continue

            drug_classes = classify_drug(pred.drug_name)
            is_hit = pred.drug_id in gt_drugs
            category = pred.category

            for dc in drug_classes:
                key = (dc, category)
                medium_matrix[key]['total'] += 1
                medium_matrix[key]['hits'] += int(is_hit)
                if len(medium_matrix[key]['examples']) < 3:
                    medium_matrix[key]['examples'].append(
                        f"{pred.drug_name}→{disease_name}"
                    )

    # Print full-data matrix
    all_classes = sorted(set(k[0] for k in medium_matrix.keys()))
    all_cats = sorted(set(k[1] for k in medium_matrix.keys()))

    print(f"\nDrug classes found: {len(all_classes)}")
    print(f"Disease categories: {len(all_cats)}")

    print(f"\n{'Drug Class':<25} | {'Category':<15} | {'Prec':<7} | {'n':<5} | {'Expected?':<9} | Examples")
    print("-" * 110)

    mismatch_candidates = []
    for dc in all_classes:
        # Get expected categories for this drug class
        expected_cats = set()
        if dc in DRUG_CLASS_SOC_MAPPINGS:
            expected_cats = DRUG_CLASS_SOC_MAPPINGS[dc]['categories']

        for cat in all_cats:
            key = (dc, cat)
            if key not in medium_matrix:
                continue
            data = medium_matrix[key]
            if data['total'] < 3:  # Skip tiny cells
                continue
            prec = data['hits'] / data['total'] * 100
            expected = "YES" if cat in expected_cats else ("n/a" if dc == 'unclassified' else "NO")
            examples = "; ".join(data['examples'][:2])

            # Flag mismatches with low precision
            flag = ""
            if expected == "NO" and prec < 20:
                flag = " *** LOW"
                mismatch_candidates.append({
                    'drug_class': dc,
                    'category': cat,
                    'precision': prec,
                    'n': data['total'],
                    'hits': data['hits'],
                    'expected': expected,
                })
            elif expected == "YES" and prec >= 40:
                flag = " (strong)"

            print(f"{dc:<25} | {cat:<15} | {prec:5.1f}% | {data['total']:<5} | {expected:<9} | {examples}{flag}")

    print(f"\n=== Mismatch Candidates (NOT expected category, <20% full-data precision) ===")
    mismatch_candidates.sort(key=lambda x: x['n'], reverse=True)
    for mc in mismatch_candidates:
        print(f"  {mc['drug_class']}→{mc['category']}: {mc['precision']:.1f}% ({mc['hits']}/{mc['n']})")

    # Step 2: Holdout validation of top mismatch candidates
    print(f"\n\n--- Step 2: Holdout Validation of Mismatches (5 seeds) ---")

    # Only validate candidates with n >= 10 (need enough predictions for holdout)
    valid_candidates = [mc for mc in mismatch_candidates if mc['n'] >= 10]
    print(f"Candidates with n>=10: {len(valid_candidates)}")

    if not valid_candidates:
        print("No candidates with sufficient sample size. Analysis complete.")
        return

    seeds = [42, 123, 456, 789, 1337]

    # For each candidate, track per-seed holdout hits/total
    candidate_holdout = {
        (mc['drug_class'], mc['category']): {'hits': [], 'total': []}
        for mc in valid_candidates
    }
    # Also track overall MEDIUM baseline
    medium_baseline = {'hits': [], 'total': []}
    # And matched SOC (expected=YES) baseline
    matched_holdout = {'hits': [], 'total': []}

    for seed in seeds:
        train, holdout = split_diseases(gt_diseases, seed)
        train_set = set(train)
        originals = recompute_gt_structures(predictor, train_set)

        seed_med_hits = 0
        seed_med_total = 0
        seed_matched_hits = 0
        seed_matched_total = 0
        seed_candidate = {k: [0, 0] for k in candidate_holdout}

        for disease_id in holdout:
            if disease_id not in predictor.embeddings:
                continue
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            result = predictor.predict(disease_name, k=20, top_n=30, include_filtered=True)
            gt_drugs = set(predictor.ground_truth.get(disease_id, []))
            if not gt_drugs:
                continue

            for pred in result.predictions:
                if pred.confidence_tier != ConfidenceTier.MEDIUM:
                    continue

                is_hit = pred.drug_id in gt_drugs
                seed_med_hits += int(is_hit)
                seed_med_total += 1

                drug_classes = classify_drug(pred.drug_name)
                category = pred.category

                for dc in drug_classes:
                    key = (dc, category)
                    if key in seed_candidate:
                        seed_candidate[key][0] += int(is_hit)
                        seed_candidate[key][1] += 1

                    # Check if this is a matched (expected) combo
                    if dc in DRUG_CLASS_SOC_MAPPINGS:
                        expected_cats = DRUG_CLASS_SOC_MAPPINGS[dc]['categories']
                        if category in expected_cats:
                            seed_matched_hits += int(is_hit)
                            seed_matched_total += 1

        medium_baseline['hits'].append(seed_med_hits)
        medium_baseline['total'].append(seed_med_total)
        matched_holdout['hits'].append(seed_matched_hits)
        matched_holdout['total'].append(seed_matched_total)

        for k, v in seed_candidate.items():
            candidate_holdout[k]['hits'].append(v[0])
            candidate_holdout[k]['total'].append(v[1])

        restore_gt_structures(predictor, originals)

    # Print holdout results
    med_precs = [h / t * 100 if t > 0 else 0 for h, t in zip(medium_baseline['hits'], medium_baseline['total'])]
    matched_precs = [h / t * 100 if t > 0 else 0 for h, t in zip(matched_holdout['hits'], matched_holdout['total'])]

    print(f"\nMEDIUM baseline holdout: {np.mean(med_precs):.1f}% ± {np.std(med_precs):.1f}% (n/seed={np.mean(medium_baseline['total']):.0f})")
    print(f"Matched SOC holdout: {np.mean(matched_precs):.1f}% ± {np.std(matched_precs):.1f}% (n/seed={np.mean(matched_holdout['total']):.0f})")

    print(f"\n{'Drug Class':<25} | {'Category':<15} | {'Holdout':<12} | {'n/seed':<8} | {'Full-data':<10} | Action")
    print("-" * 100)

    actionable = []
    for mc in valid_candidates:
        key = (mc['drug_class'], mc['category'])
        data = candidate_holdout[key]
        precs = [h / t * 100 if t > 0 else 0 for h, t in zip(data['hits'], data['total'])]
        mean_n = np.mean(data['total'])
        mean_prec = np.mean(precs)
        std_prec = np.std(precs)

        # Decision logic
        if mean_n < 3:
            action = "SKIP (tiny n)"
        elif mean_prec < 10 and mean_n >= 5:
            action = "DEMOTE → LOW"
            actionable.append({
                'drug_class': mc['drug_class'],
                'category': mc['category'],
                'holdout': mean_prec,
                'holdout_std': std_prec,
                'n_per_seed': mean_n,
                'full_data': mc['precision'],
                'full_n': mc['n'],
            })
        elif mean_prec < 20:
            action = "MARGINAL (monitor)"
        else:
            action = "KEEP (decent holdout)"

        print(f"{mc['drug_class']:<25} | {mc['category']:<15} | {mean_prec:5.1f}%±{std_prec:4.1f}% | {mean_n:6.1f} | {mc['precision']:5.1f}% ({mc['n']:>3}) | {action}")

    # Summary of actionable demotions
    print(f"\n" + "=" * 70)
    print(f"ACTIONABLE DEMOTIONS")
    print("=" * 70)

    if actionable:
        total_demoted = sum(a['full_n'] for a in actionable)
        print(f"Found {len(actionable)} mismatch rules to implement:")
        for a in actionable:
            print(f"  {a['drug_class']}→{a['category']}: {a['holdout']:.1f}% holdout (n/seed={a['n_per_seed']:.1f}), {a['full_n']} predictions")
        print(f"\nTotal predictions that would be demoted: {total_demoted}")

        # Estimate MEDIUM impact
        total_mismatch_hits = sum(round(a['holdout'] / 100 * a['n_per_seed']) for a in actionable)
        total_mismatch_preds = sum(a['n_per_seed'] for a in actionable)
        mean_medium_n = np.mean(medium_baseline['total'])
        mean_medium_hits = np.mean(medium_baseline['hits'])

        new_medium_hits = mean_medium_hits - total_mismatch_hits
        new_medium_total = mean_medium_n - total_mismatch_preds
        if new_medium_total > 0:
            new_medium_prec = new_medium_hits / new_medium_total * 100
            print(f"\nEstimated MEDIUM impact: {np.mean(med_precs):.1f}% → {new_medium_prec:.1f}% ({new_medium_prec - np.mean(med_precs):+.1f}pp)")
    else:
        print("No actionable demotions found. Existing filters are comprehensive.")

    # Save results
    results = {
        "hypothesis": "h567",
        "medium_baseline_holdout": round(np.mean(med_precs), 1),
        "matched_soc_holdout": round(np.mean(matched_precs), 1),
        "mismatch_candidates": len(valid_candidates),
        "actionable_demotions": len(actionable),
        "actionable_details": actionable,
    }

    with open("data/analysis/h567_output.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to data/analysis/h567_output.json")


if __name__ == "__main__":
    main()
