#!/usr/bin/env python3
"""
h520 deep dive: Understand corticosteroid MEDIUM predictions.

Questions:
1. What disease categories do corticosteroid MEDIUM predictions cover?
2. Which specific diseases are involved?
3. What is the category-level holdout precision?
4. Would promotion to HIGH be safe (not double-counting)?
5. How many predictions would be promoted?
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
    classify_literature_status,
    extract_cancer_types,
    DISEASE_HIERARCHY_GROUPS,
    HIERARCHY_EXCLUSIONS,
    CORTICOSTEROID_DRUGS,
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
    originals = {
        "drug_train_freq": dict(predictor.drug_train_freq),
        "drug_to_diseases": dict(predictor.drug_to_diseases),
        "drug_cancer_types": dict(predictor.drug_cancer_types),
        "drug_disease_groups": dict(predictor.drug_disease_groups),
        "train_diseases": list(predictor.train_diseases),
        "train_embeddings": predictor.train_embeddings.copy(),
        "train_disease_categories": dict(predictor.train_disease_categories),
    }

    new_freq: Dict[str, int] = defaultdict(int)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            for drug_id in predictor.ground_truth[disease_id]:
                new_freq[drug_id] += 1
    predictor.drug_train_freq = dict(new_freq)

    new_d2d: Dict[str, Set[str]] = defaultdict(set)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            for drug_id in predictor.ground_truth[disease_id]:
                new_d2d[drug_id].add(disease_name.lower())
    predictor.drug_to_diseases = dict(new_d2d)

    new_cancer: Dict[str, Set[str]] = defaultdict(set)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            cancer_types = extract_cancer_types(disease_name)
            if cancer_types:
                for drug_id in predictor.ground_truth[disease_id]:
                    new_cancer[drug_id].update(cancer_types)
    predictor.drug_cancer_types = dict(new_cancer)

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


def restore_gt_structures(
    predictor: DrugRepurposingPredictor, originals: Dict
) -> None:
    predictor.drug_train_freq = originals["drug_train_freq"]
    predictor.drug_to_diseases = originals["drug_to_diseases"]
    predictor.drug_cancer_types = originals["drug_cancer_types"]
    predictor.drug_disease_groups = originals["drug_disease_groups"]
    predictor.train_diseases = originals["train_diseases"]
    predictor.train_embeddings = originals["train_embeddings"]
    predictor.train_disease_categories = originals["train_disease_categories"]


def main() -> None:
    seeds = [42, 123, 456, 789, 1024]
    print("=" * 70)
    print("h520 Deep Dive: Corticosteroid MEDIUM Predictions")
    print("=" * 70)

    print("\nLoading predictor...")
    predictor = DrugRepurposingPredictor()

    gt_path = predictor.reference_dir / "expanded_ground_truth.json"
    with open(gt_path) as f:
        gt_data = json.load(f)

    all_diseases = [d for d in predictor.ground_truth if d in predictor.embeddings]
    corticosteroid_lower = {d.lower() for d in CORTICOSTEROID_DRUGS}

    # --- 1. Full-data: What do corticosteroid MEDIUM predictions look like? ---
    print("\n--- FULL DATA: Corticosteroid MEDIUM Predictions ---\n")

    gt_set = set()
    for disease_id, drugs in gt_data.items():
        for drug in drugs:
            if isinstance(drug, str):
                gt_set.add((disease_id, drug))
            elif isinstance(drug, dict):
                gt_set.add((disease_id, drug.get("drug_id") or drug.get("drug")))

    # Collect corticosteroid MEDIUM predictions
    category_stats: Dict[str, Dict] = defaultdict(lambda: {"hits": 0, "total": 0, "diseases": set()})
    tier_rule_stats: Dict[str, Dict] = defaultdict(lambda: {"hits": 0, "total": 0})
    disease_detail: Dict[str, List] = defaultdict(list)  # disease -> [(drug, hit?)]

    for disease_id in all_diseases:
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        try:
            result = predictor.predict(disease_name, top_n=30, include_filtered=True)
        except Exception:
            continue

        category = result.category

        for pred in result.predictions:
            if pred.confidence_tier.name != "MEDIUM":
                continue
            if pred.drug_name.lower() not in corticosteroid_lower:
                continue

            # This is a corticosteroid MEDIUM prediction
            lit_status, soc_class = classify_literature_status(
                pred.drug_name, disease_name, category, False
            )
            if lit_status != 'LIKELY_GT_GAP' or soc_class != 'corticosteroids':
                continue  # Not classified as SOC corticosteroid

            is_hit = (disease_id, pred.drug_id) in gt_set
            category_stats[category]["total"] += 1
            category_stats[category]["diseases"].add(disease_name)
            if is_hit:
                category_stats[category]["hits"] += 1

            rule = pred.category_specific_tier or "default"
            tier_rule_stats[rule]["total"] += 1
            if is_hit:
                tier_rule_stats[rule]["hits"] += 1

            disease_detail[disease_name].append((pred.drug_name, is_hit))

    print(f"{'Category':<25} {'Hits':<6} {'Total':<8} {'Precision':<10} {'# Diseases'}")
    print("-" * 65)
    total_hits = 0
    total_total = 0
    for cat, stats in sorted(category_stats.items(), key=lambda x: -x[1]["total"]):
        prec = stats["hits"] / stats["total"] * 100 if stats["total"] > 0 else 0
        n_diseases = len(stats["diseases"])
        print(f"{cat:<25} {stats['hits']:<6} {stats['total']:<8} {prec:.1f}%      {n_diseases}")
        total_hits += stats["hits"]
        total_total += stats["total"]

    overall_prec = total_hits / total_total * 100 if total_total > 0 else 0
    print(f"\n{'TOTAL':<25} {total_hits:<6} {total_total:<8} {overall_prec:.1f}%")

    print(f"\n\n--- Tier Rules for Corticosteroid MEDIUM ---\n")
    print(f"{'Rule':<40} {'Hits':<6} {'Total':<8} {'Precision'}")
    print("-" * 60)
    for rule, stats in sorted(tier_rule_stats.items(), key=lambda x: -x[1]["total"]):
        prec = stats["hits"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"{rule:<40} {stats['hits']:<6} {stats['total']:<8} {prec:.1f}%")

    # --- 2. Holdout: category-stratified corticosteroid precision ---
    print("\n\n--- HOLDOUT: Per-Category Corticosteroid MEDIUM Precision ---\n")

    cat_holdout: Dict[str, List[Dict]] = defaultdict(list)  # cat -> [{hits, total}, ...]
    overall_holdout: List[Dict] = []

    for seed_idx, seed in enumerate(seeds):
        print(f"Seed {seed}...", end=" ", flush=True)
        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        train_set = set(train_ids)

        originals = recompute_gt_structures(predictor, train_set)

        cat_seed: Dict[str, Dict] = defaultdict(lambda: {"hits": 0, "total": 0})
        seed_overall = {"hits": 0, "total": 0}

        for disease_id in holdout_ids:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            try:
                result = predictor.predict(disease_name, top_n=30, include_filtered=True)
            except Exception:
                continue

            category = result.category

            for pred in result.predictions:
                if pred.confidence_tier.name != "MEDIUM":
                    continue
                if pred.drug_name.lower() not in corticosteroid_lower:
                    continue

                lit_status, soc_class = classify_literature_status(
                    pred.drug_name, disease_name, category, False
                )
                if lit_status != 'LIKELY_GT_GAP' or soc_class != 'corticosteroids':
                    continue

                is_hit = (disease_id, pred.drug_id) in gt_set
                cat_seed[category]["total"] += 1
                seed_overall["total"] += 1
                if is_hit:
                    cat_seed[category]["hits"] += 1
                    seed_overall["hits"] += 1

        for cat, stats in cat_seed.items():
            cat_holdout[cat].append(stats)
        overall_holdout.append(seed_overall)

        restore_gt_structures(predictor, originals)
        print(f"done ({seed_overall['total']} preds)")

    # Aggregate holdout results per category
    print(f"\n{'Category':<25} {'Mean Prec':<12} {'SE':<8} {'Mean N':<8}")
    print("-" * 55)

    for cat in sorted(cat_holdout.keys()):
        precisions = []
        ns = []
        for s in cat_holdout[cat]:
            prec = s["hits"] / s["total"] * 100 if s["total"] > 0 else 0
            precisions.append(prec)
            ns.append(s["total"])
        # Pad to 5 seeds (some categories may not appear in all seeds)
        while len(precisions) < 5:
            precisions.append(0)
            ns.append(0)

        mean_prec = np.mean(precisions)
        se = np.std(precisions, ddof=1) / np.sqrt(len(precisions)) if len(precisions) > 1 else 0
        mean_n = np.mean(ns)
        print(f"{cat:<25} {mean_prec:>7.1f}%    {se:>5.1f}pp {mean_n:>7.0f}")

    # Overall corticosteroid MEDIUM holdout
    overall_precs = [s["hits"] / s["total"] * 100 if s["total"] > 0 else 0 for s in overall_holdout]
    overall_ns = [s["total"] for s in overall_holdout]
    mean_overall = np.mean(overall_precs)
    se_overall = np.std(overall_precs, ddof=1) / np.sqrt(len(overall_precs))
    mean_n_overall = np.mean(overall_ns)
    print(f"\n{'TOTAL':<25} {mean_overall:>7.1f}%    {se_overall:>5.1f}pp {mean_n_overall:>7.0f}")

    # --- 3. How many predictions would be promoted? ---
    print(f"\n\n--- PROMOTION IMPACT ---")
    print(f"\nTotal corticosteroid MEDIUM SOC predictions (full data): {total_total}")
    print(f"Full-data precision: {overall_prec:.1f}%")
    print(f"Holdout precision: {mean_overall:.1f}% ± {se_overall:.1f}%")
    print(f"\nHIGH tier current holdout: 51.5% ± 5.3% ({507} predictions)")
    print(f"Corticosteroid MEDIUM holdout: {mean_overall:.1f}% ± {se_overall:.1f}%")

    if mean_overall >= 40:
        print(f"\n** ACTIONABLE: Corticosteroid MEDIUM ({mean_overall:.1f}%) approaches HIGH ({51.5}%)")
        print(f"   Promoting {total_total} predictions from MEDIUM to HIGH")
        print(f"   New HIGH total would be: ~{507 + total_total} predictions")
        # Estimate new HIGH precision
        # Current HIGH: 51.5% of 507 = ~261 hits
        new_high_hits = 51.5 / 100 * 507 + mean_overall / 100 * mean_n_overall * 5
        new_high_total = 507 + total_total
        new_high_prec = new_high_hits / new_high_total * 100 if new_high_total > 0 else 0
        print(f"   Estimated new HIGH holdout: ~{new_high_prec:.1f}%")
    else:
        print(f"\n   NOT actionable: {mean_overall:.1f}% < 40% threshold")

    # --- 4. Specific disease examples ---
    print(f"\n\n--- SAMPLE CORTICOSTEROID MEDIUM PREDICTIONS ---")
    print(f"(Top 15 diseases by # predictions)\n")
    for disease, preds in sorted(disease_detail.items(), key=lambda x: -len(x[1]))[:15]:
        hits = sum(1 for _, h in preds if h)
        total = len(preds)
        drug_list = ", ".join(f"{'✓' if h else '✗'}{d}" for d, h in preds[:4])
        extra = f"... +{total-4} more" if total > 4 else ""
        print(f"  {disease[:50]:<50} {hits}/{total} ({hits/total*100:.0f}%) {drug_list}{extra}")


if __name__ == "__main__":
    main()
