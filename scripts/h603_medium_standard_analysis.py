#!/usr/bin/env python3
"""
h603: MEDIUM Standard Rule Category Refinement

Analyze the 'standard' MEDIUM rule (train_freq>=5+mech OR train_freq>=10+rank<=10)
by disease category. Hypothesis: metabolic, respiratory, and endocrine categories
within standard MEDIUM have below-average precision and could be demoted to LOW.
"""

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


def split_diseases(
    all_diseases: List[str], seed: int, train_ratio: float = 0.8
) -> Tuple[List[str], List[str]]:
    rng = np.random.RandomState(seed)
    shuffled = list(all_diseases)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def recompute_gt_structures(
    predictor: DrugRepurposingPredictor,
    train_disease_ids: Set[str],
) -> Dict:
    """Recompute GT structures from training diseases only. Matches h393 logic."""
    originals = {
        "drug_train_freq": dict(predictor.drug_train_freq),
        "drug_to_diseases": dict(predictor.drug_to_diseases),
        "drug_cancer_types": dict(predictor.drug_cancer_types),
        "drug_disease_groups": dict(predictor.drug_disease_groups),
        "train_diseases": list(predictor.train_diseases),
        "train_embeddings": predictor.train_embeddings.copy(),
        "train_disease_categories": dict(predictor.train_disease_categories),
    }

    # 1. drug_train_freq
    new_freq: Dict[str, int] = defaultdict(int)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            for drug_id in predictor.ground_truth[disease_id]:
                new_freq[drug_id] += 1
    predictor.drug_train_freq = dict(new_freq)

    # 2. drug_to_diseases
    new_d2d: Dict[str, Set[str]] = defaultdict(set)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            for drug_id in predictor.ground_truth[disease_id]:
                new_d2d[drug_id].add(disease_name.lower())
    predictor.drug_to_diseases = dict(new_d2d)

    # 3. drug_cancer_types
    new_cancer: Dict[str, Set[str]] = defaultdict(set)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            cancer_types = extract_cancer_types(disease_name)
            if cancer_types:
                for drug_id in predictor.ground_truth[disease_id]:
                    new_cancer[drug_id].update(cancer_types)
    predictor.drug_cancer_types = dict(new_cancer)

    # 4. drug_disease_groups (h469: HIERARCHY_EXCLUSIONS)
    new_groups: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
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

    # 5. Rebuild kNN index
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


def restore_gt_structures(predictor: DrugRepurposingPredictor, originals: Dict):
    predictor.drug_train_freq = originals["drug_train_freq"]
    predictor.drug_to_diseases = originals["drug_to_diseases"]
    predictor.drug_cancer_types = originals["drug_cancer_types"]
    predictor.drug_disease_groups = originals["drug_disease_groups"]
    predictor.train_diseases = originals["train_diseases"]
    predictor.train_embeddings = originals["train_embeddings"]
    predictor.train_disease_categories = originals["train_disease_categories"]


def is_standard_medium(pred) -> bool:
    """Standard MEDIUM = category_specific_tier is None."""
    return pred.confidence_tier == ConfidenceTier.MEDIUM and pred.category_specific_tier is None


def main():
    print("=" * 80)
    print("h603: MEDIUM Standard Rule Category Analysis")
    print("=" * 80)

    predictor = DrugRepurposingPredictor()

    # ===== STEP 1: Full-data analysis =====
    print("\n--- Step 1: Full-data MEDIUM analysis ---")
    all_disease_ids = list(predictor.ground_truth.keys())
    print(f"Total diseases with GT: {len(all_disease_ids)}")

    medium_by_reason = defaultdict(list)
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
            cst = pred.category_specific_tier or 'standard'
            is_hit = pred.drug_id in gt_drugs
            cat = pred.category

            medium_by_reason[cst].append({
                'drug_id': pred.drug_id,
                'disease_id': disease_id,
                'category': cat,
                'is_hit': is_hit,
                'drug_name': pred.drug_name,
                'disease_name': disease_name,
            })
            all_medium.append((cst, cat, is_hit))

    print(f"\nTotal MEDIUM predictions: {len(all_medium)}")
    print("\nMEDIUM by sub_reason:")
    for reason in sorted(medium_by_reason.keys(), key=lambda r: -len(medium_by_reason[r])):
        preds = medium_by_reason[reason]
        hits = sum(1 for p in preds if p['is_hit'])
        total = len(preds)
        prec = hits / total * 100 if total > 0 else 0
        print(f"  {reason:<40} {total:>5} preds, {prec:>5.1f}% ({hits} hits)")

    standard = medium_by_reason.get('standard', [])
    print(f"\n\nStandard MEDIUM predictions: {len(standard)}")

    cat_groups = defaultdict(list)
    for p in standard:
        cat_groups[p['category']].append(p)

    print("\nStandard MEDIUM by disease category (full-data precision):")
    for cat in sorted(cat_groups.keys(), key=lambda c: -len(cat_groups[c])):
        preds = cat_groups[cat]
        hits = sum(1 for p in preds if p['is_hit'])
        total = len(preds)
        prec = hits / total * 100 if total > 0 else 0
        print(f"  {cat:<30} {total:>5} preds, {prec:>5.1f}% ({hits} hits)")

    # ===== STEP 2: Holdout validation =====
    print("\n\n" + "=" * 80)
    print("Step 2: 5-seed holdout validation for standard MEDIUM by category")
    print("=" * 80)

    seeds = [42, 123, 456, 789, 2024]
    cat_holdout_results = defaultdict(list)
    overall_holdout_results = []
    pool_holdout_results = defaultdict(list)

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        train_diseases, holdout_diseases = split_diseases(all_disease_ids, seed)
        train_set = set(train_diseases)

        originals = recompute_gt_structures(predictor, train_set)

        holdout_cat_hits = defaultdict(int)
        holdout_cat_total = defaultdict(int)
        overall_hits = 0
        overall_total = 0

        for disease_id in holdout_diseases:
            if disease_id not in predictor.ground_truth:
                continue

            gt_drugs = set(predictor.ground_truth[disease_id])
            disease_name = predictor.disease_names.get(disease_id, disease_id)

            try:
                result = predictor.predict(disease_name, top_n=30, include_filtered=True)
            except Exception:
                continue

            for pred in result.predictions:
                if not is_standard_medium(pred):
                    continue
                cat = pred.category
                is_hit = pred.drug_id in gt_drugs
                holdout_cat_hits[cat] += int(is_hit)
                holdout_cat_total[cat] += 1
                overall_hits += int(is_hit)
                overall_total += 1

        overall_holdout_results.append((overall_hits, overall_total))

        all_cats = set(holdout_cat_total.keys())
        for cat in all_cats:
            cat_holdout_results[cat].append(
                (holdout_cat_hits[cat], holdout_cat_total[cat])
            )

        # Pools
        pool_cats = {'metabolic', 'respiratory', 'endocrine'}
        pool_hits = sum(holdout_cat_hits[c] for c in pool_cats)
        pool_total = sum(holdout_cat_total[c] for c in pool_cats)
        pool_holdout_results['metabolic+respiratory+endocrine'].append((pool_hits, pool_total))

        other_cats = all_cats - pool_cats
        other_hits = sum(holdout_cat_hits[c] for c in other_cats)
        other_total = sum(holdout_cat_total[c] for c in other_cats)
        pool_holdout_results['all_other_standard'].append((other_hits, other_total))

        for cat in sorted(holdout_cat_total.keys()):
            h, t = holdout_cat_hits[cat], holdout_cat_total[cat]
            prec = h / t * 100 if t > 0 else 0
            print(f"  {cat:<25} {h:>3}/{t:<4} = {prec:>5.1f}%")

        if overall_total > 0:
            print(f"  {'OVERALL':<25} {overall_hits:>3}/{overall_total:<4} = {overall_hits/overall_total*100:>5.1f}%")

        restore_gt_structures(predictor, originals)

    # ===== STEP 3: Summary =====
    print("\n\n" + "=" * 80)
    print("Step 3: 5-Seed Holdout Summary")
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

    print(f"\n{'Category':<35} {'Holdout %':>10} {'±':>3} {'Std':>6} {'n/seed':>8}")
    print("-" * 65)

    mean, std, mean_n = summarize(overall_holdout_results)
    print(f"{'ALL STANDARD MEDIUM':<35} {mean:>10.1f} {'±':>3} {std:>6.1f} {mean_n:>8.1f}")

    print()
    cat_summaries = {}
    for cat in sorted(cat_holdout_results.keys()):
        results = cat_holdout_results[cat]
        mean, std, mean_n = summarize(results)
        cat_summaries[cat] = (mean, std, mean_n)

    for cat in sorted(cat_summaries.keys(), key=lambda c: -cat_summaries[c][2]):
        mean, std, mean_n = cat_summaries[cat]
        marker = " ***" if mean < 25 and mean_n >= 6 else ""
        print(f"  {cat:<33} {mean:>10.1f} {'±':>3} {std:>6.1f} {mean_n:>8.1f}{marker}")

    print()
    print("Pooled groups:")
    for gn in ['metabolic+respiratory+endocrine', 'all_other_standard']:
        results = pool_holdout_results[gn]
        mean, std, mean_n = summarize(results)
        marker = " ***" if mean < 25 and mean_n >= 10 else ""
        print(f"  {gn:<33} {mean:>10.1f} {'±':>3} {std:>6.1f} {mean_n:>8.1f}{marker}")

    # ===== STEP 4: Demotion analysis =====
    print("\n\n" + "=" * 80)
    print("Step 4: Demotion Analysis")
    print("=" * 80)

    pool_mean, pool_std, pool_n = summarize(pool_holdout_results['metabolic+respiratory+endocrine'])
    other_mean, other_std, other_n = summarize(pool_holdout_results['all_other_standard'])

    print(f"\nPooled met+resp+endo: {pool_mean:.1f}% ± {pool_std:.1f}% (n={pool_n:.1f}/seed)")
    print(f"All other standard:   {other_mean:.1f}% ± {other_std:.1f}% (n={other_n:.1f}/seed)")
    print(f"Gap:                  {other_mean - pool_mean:.1f}pp")

    pool_count = sum(len(cat_groups.get(c, [])) for c in ['metabolic', 'respiratory', 'endocrine'])
    print(f"\nPredictions that would move MEDIUM→LOW: {pool_count}")

    total_medium = len(all_medium)
    total_medium_hits = sum(1 for _, _, hit in all_medium if hit)
    pool_hits_fd = sum(1 for p in standard if p['category'] in ['metabolic', 'respiratory', 'endocrine'] and p['is_hit'])

    if total_medium > pool_count:
        new_prec = (total_medium_hits - pool_hits_fd) / (total_medium - pool_count) * 100
        old_prec = total_medium_hits / total_medium * 100
        print(f"\nFull-data MEDIUM: {old_prec:.1f}% → {new_prec:.1f}% (+{new_prec - old_prec:.1f}pp)")

    if pool_mean < 25 and pool_n >= 15:
        print("\n>>> RECOMMENDATION: DEMOTE metabolic+respiratory+endocrine standard MEDIUM → LOW")
    elif pool_mean < 25 and pool_n >= 6:
        print(f"\n>>> MARGINAL: Pool precision ({pool_mean:.1f}%) < 25% but n={pool_n:.1f}/seed")
    else:
        print(f"\n>>> NO ACTION: Pool precision ({pool_mean:.1f}%) adequate or n too small")

    print("\nIndividual categories:")
    for cat in ['metabolic', 'respiratory', 'endocrine', 'autoimmune', 'dermatological',
                'infectious', 'musculoskeletal', 'ophthalmic', 'psychiatric', 'renal', 'other']:
        if cat in cat_summaries:
            mean, std, mean_n = cat_summaries[cat]
            status = "DEMOTE" if mean < 25 and mean_n >= 6 else ("LOW-N" if mean_n < 6 else "OK")
            print(f"  {cat:<25} {mean:>6.1f}% ± {std:>5.1f}% (n={mean_n:>5.1f}/seed) → {status}")


if __name__ == "__main__":
    main()
