#!/usr/bin/env python3
"""
h606: ATC Coherent Respiratory/Endocrine Validation

h603 found that respiratory (22.3%) and endocrine (24.5%) demotion holdout was
inflated because MEDIUM_DEMOTED_CATEGORIES intercepts ATC coherent predictions.
This script checks if atc_coherent_respiratory and atc_coherent_endocrine should
be added to ATC_COHERENT_EXCLUDED (like metabolic already is).
"""

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
    for d in train_disease_ids:
        if d in predictor.ground_truth:
            for drug in predictor.ground_truth[d]:
                new_freq[drug] += 1
    predictor.drug_train_freq = dict(new_freq)

    new_d2d = defaultdict(set)
    for d in train_disease_ids:
        if d in predictor.ground_truth:
            dn = predictor.disease_names.get(d, d)
            for drug in predictor.ground_truth[d]:
                new_d2d[drug].add(dn.lower())
    predictor.drug_to_diseases = dict(new_d2d)

    new_cancer = defaultdict(set)
    for d in train_disease_ids:
        if d in predictor.ground_truth:
            dn = predictor.disease_names.get(d, d)
            ct = extract_cancer_types(dn)
            if ct:
                for drug in predictor.ground_truth[d]:
                    new_cancer[drug].update(ct)
    predictor.drug_cancer_types = dict(new_cancer)

    new_groups = defaultdict(set)
    for d in train_disease_ids:
        if d in predictor.ground_truth:
            dn = predictor.disease_names.get(d, d).lower()
            for cat, groups in DISEASE_HIERARCHY_GROUPS.items():
                for gname, keywords in groups.items():
                    excl = HIERARCHY_EXCLUSIONS.get((cat, gname), [])
                    if any(e in dn for e in excl):
                        continue
                    if any(k in dn or dn in k for k in keywords):
                        for drug in predictor.ground_truth[d]:
                            new_groups[drug].add((cat, gname))
    predictor.drug_disease_groups = dict(new_groups)

    predictor.train_diseases = [d for d in train_disease_ids if d in predictor.embeddings]
    predictor.train_embeddings = np.array(
        [predictor.embeddings[d] for d in predictor.train_diseases], dtype=np.float32
    )
    predictor.train_disease_categories = {}
    for d in predictor.train_diseases:
        name = predictor.disease_names.get(d, d)
        predictor.train_disease_categories[d] = predictor.categorize_disease(name)

    return originals


def restore(predictor, originals):
    for k, v in originals.items():
        setattr(predictor, k, v)


def main():
    print("=" * 80)
    print("h606: ATC Coherent Respiratory/Endocrine Validation")
    print("=" * 80)

    predictor = DrugRepurposingPredictor()
    all_disease_ids = list(predictor.ground_truth.keys())

    # ===== STEP 1: Full-data analysis of all ATC coherent MEDIUM =====
    print("\n--- Step 1: Full-data ATC coherent MEDIUM by category ---")

    atc_coherent_preds = defaultdict(list)
    all_medium = []

    for disease_id in all_disease_ids:
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        try:
            result = predictor.predict(disease_name, top_n=30, include_filtered=True)
        except Exception:
            continue

        gt_drugs = set(predictor.ground_truth[disease_id])
        for pred in result.predictions:
            if pred.confidence_tier != ConfidenceTier.MEDIUM:
                continue
            cst = pred.category_specific_tier
            if cst and cst.startswith('atc_coherent_'):
                cat = cst.replace('atc_coherent_', '')
                is_hit = pred.drug_id in gt_drugs
                atc_coherent_preds[cat].append({
                    'drug_name': pred.drug_name,
                    'disease_name': disease_name,
                    'is_hit': is_hit,
                })

    print(f"\nATC coherent MEDIUM by category:")
    for cat in sorted(atc_coherent_preds.keys(), key=lambda c: -len(atc_coherent_preds[c])):
        preds = atc_coherent_preds[cat]
        hits = sum(1 for p in preds if p['is_hit'])
        prec = hits / len(preds) * 100 if preds else 0
        print(f"  atc_coherent_{cat:<20} {len(preds):>5} preds, {prec:>5.1f}% ({hits} hits)")

    # ===== STEP 2: Holdout validation =====
    print("\n\n" + "=" * 80)
    print("Step 2: 5-seed holdout validation for ATC coherent by category")
    print("=" * 80)

    seeds = [42, 123, 456, 789, 2024]
    cat_holdout = defaultdict(list)

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        train_d, holdout_d = split_diseases(all_disease_ids, seed)
        train_set = set(train_d)
        originals = recompute_gt_structures(predictor, train_set)

        cat_hits = defaultdict(int)
        cat_total = defaultdict(int)

        for disease_id in holdout_d:
            if disease_id not in predictor.ground_truth:
                continue
            gt_drugs = set(predictor.ground_truth[disease_id])
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            try:
                result = predictor.predict(disease_name, top_n=30, include_filtered=True)
            except Exception:
                continue

            for pred in result.predictions:
                if pred.confidence_tier != ConfidenceTier.MEDIUM:
                    continue
                cst = pred.category_specific_tier
                if cst and cst.startswith('atc_coherent_'):
                    cat = cst.replace('atc_coherent_', '')
                    is_hit = pred.drug_id in gt_drugs
                    cat_hits[cat] += int(is_hit)
                    cat_total[cat] += 1

        for cat in set(cat_total.keys()):
            cat_holdout[cat].append((cat_hits[cat], cat_total[cat]))

        for cat in sorted(cat_total.keys()):
            h, t = cat_hits[cat], cat_total[cat]
            prec = h / t * 100 if t > 0 else 0
            print(f"  atc_coherent_{cat:<15} {h:>3}/{t:<4} = {prec:>5.1f}%")

        restore(predictor, originals)

    # ===== STEP 3: Summary =====
    print("\n\n" + "=" * 80)
    print("Step 3: Holdout Summary")
    print("=" * 80)

    def summarize(results_list):
        precisions = []
        total_n = 0
        for hits, total in results_list:
            precisions.append(hits / total * 100 if total > 0 else 0)
            total_n += total
        mean = np.mean(precisions)
        std = np.std(precisions)
        mean_n = total_n / len(results_list)
        return mean, std, mean_n

    print(f"\n{'Category':<30} {'Holdout %':>10} {'±':>3} {'Std':>6} {'n/seed':>8} {'Full%':>8}")
    print("-" * 70)

    for cat in sorted(cat_holdout.keys(), key=lambda c: -summarize(cat_holdout[c])[2]):
        m, s, n = summarize(cat_holdout[cat])
        # Get full-data
        fd_preds = atc_coherent_preds.get(cat, [])
        fd_hits = sum(1 for p in fd_preds if p['is_hit'])
        fd_prec = fd_hits / len(fd_preds) * 100 if fd_preds else 0
        marker = " ***" if m < 15 and n >= 3 else ""
        print(f"  atc_coherent_{cat:<14} {m:>10.1f} {'±':>3} {s:>6.1f} {n:>8.1f} {fd_prec:>7.1f}%{marker}")

    # Recommendation
    print("\n\nRecommendation:")
    for cat in ['respiratory', 'endocrine']:
        if cat in cat_holdout:
            m, s, n = summarize(cat_holdout[cat])
            if m < 15:
                print(f"  {cat}: {m:.1f}% holdout → ADD to ATC_COHERENT_EXCLUDED")
            elif m < 25:
                print(f"  {cat}: {m:.1f}% holdout → BORDERLINE, consider excluding")
            else:
                print(f"  {cat}: {m:.1f}% holdout → KEEP (adequate precision)")
        else:
            print(f"  {cat}: No ATC coherent predictions found")


if __name__ == "__main__":
    main()
