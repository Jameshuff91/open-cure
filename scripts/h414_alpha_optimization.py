#!/usr/bin/env python3
"""
h414: Optimize selective boosting alpha values.

h170 uses alpha=0.5 for all boosted categories. This script tests:
1. Global alpha optimization: is 0.5 optimal?
2. Category-specific alpha: does each category benefit from different alpha?

Uses 5-seed holdout evaluation for robustness.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import production_predictor as pp
from production_predictor import (
    DrugRepurposingPredictor,
    extract_cancer_types,
    DISEASE_HIERARCHY_GROUPS,
    SELECTIVE_BOOST_CATEGORIES,
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
    new_d2d: Dict[str, Set[str]] = defaultdict(set)
    new_cancer: Dict[str, Set[str]] = defaultdict(set)
    new_groups: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)

    for disease_id in train_disease_ids:
        if disease_id not in predictor.ground_truth:
            continue
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        category = predictor.categorize_disease(disease_name)
        cancer_types = extract_cancer_types(disease_name)

        for drug_id in predictor.ground_truth[disease_id]:
            new_freq[drug_id] += 1
            new_d2d[drug_id].add(disease_name.lower())
            if cancer_types:
                new_cancer[drug_id].update(cancer_types)
            if category in DISEASE_HIERARCHY_GROUPS:
                for group_name, keywords in DISEASE_HIERARCHY_GROUPS[category].items():
                    if any(kw in disease_name.lower() for kw in keywords):
                        new_groups[drug_id].add((category, group_name))

    predictor.drug_train_freq = dict(new_freq)
    predictor.drug_to_diseases = dict(new_d2d)
    predictor.drug_cancer_types = dict(new_cancer)
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


def restore_gt_structures(predictor: DrugRepurposingPredictor, originals: Dict) -> None:
    for key, val in originals.items():
        setattr(predictor, key, val)


def evaluate_recall_at_30(
    predictor: DrugRepurposingPredictor,
    disease_ids: List[str],
    gt_data: Dict[str, List],
) -> Tuple[float, Dict[str, float]]:
    """Evaluate R@30 overall and by category.

    Returns: (overall_r30, {category: r30})
    """
    per_disease_recalls = []
    cat_recalls: Dict[str, List[float]] = defaultdict(list)

    for disease_id in disease_ids:
        if disease_id not in gt_data:
            continue

        disease_name = predictor.disease_names.get(disease_id, disease_id)
        gt_drugs = set()
        for drug in gt_data[disease_id]:
            if isinstance(drug, str):
                gt_drugs.add(drug)
            elif isinstance(drug, dict):
                gt_drugs.add(drug.get('drug_id') or drug.get('drug'))

        if not gt_drugs:
            continue

        try:
            result = predictor.predict(disease_name, top_n=30, include_filtered=True)
        except Exception:
            continue

        pred_drugs = {p.drug_id for p in result.predictions[:30]}
        hits = len(pred_drugs & gt_drugs)
        recall = hits / len(gt_drugs)
        per_disease_recalls.append(recall)

        category = predictor.categorize_disease(disease_name)
        cat_recalls[category].append(recall)

    overall = np.mean(per_disease_recalls) * 100 if per_disease_recalls else 0
    cat_means = {cat: np.mean(vals) * 100 for cat, vals in cat_recalls.items()}

    return overall, cat_means


def main():
    print("Loading predictor...")
    predictor = DrugRepurposingPredictor()

    gt_path = predictor.reference_dir / "expanded_ground_truth.json"
    with open(gt_path) as f:
        gt_data = json.load(f)

    all_diseases = [
        d for d in predictor.disease_names.keys()
        if d in predictor.ground_truth and d in predictor.embeddings
    ]
    print(f"Total diseases: {len(all_diseases)}")

    # === Phase 1: Global alpha sweep on full data ===
    print("\n=== PHASE 1: Global Alpha Sweep (Full Data) ===")
    alphas = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    original_alpha = pp.SELECTIVE_BOOST_ALPHA

    for alpha in alphas:
        pp.SELECTIVE_BOOST_ALPHA = alpha
        overall, cat_means = evaluate_recall_at_30(predictor, all_diseases, gt_data)

        # Show boosted category means
        boost_cats = sorted(SELECTIVE_BOOST_CATEGORIES)
        boost_str = ", ".join(f"{c}={cat_means.get(c, 0):.1f}%" for c in boost_cats)
        print(f"  alpha={alpha:4.1f}: R@30={overall:5.1f}%  [{boost_str}]")

    pp.SELECTIVE_BOOST_ALPHA = original_alpha

    # === Phase 2: Holdout validation of best alpha candidates ===
    print("\n=== PHASE 2: Holdout Validation (5-seed) ===")
    seeds = [42, 123, 456, 789, 2024]
    test_alphas = [0.0, 0.3, 0.5, 0.7, 1.0]

    holdout_results: Dict[float, List[float]] = {a: [] for a in test_alphas}
    holdout_cat_results: Dict[float, Dict[str, List[float]]] = {
        a: defaultdict(list) for a in test_alphas
    }

    for seed in seeds:
        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        originals = recompute_gt_structures(predictor, set(train_ids))

        for alpha in test_alphas:
            pp.SELECTIVE_BOOST_ALPHA = alpha
            overall, cat_means = evaluate_recall_at_30(predictor, holdout_ids, gt_data)
            holdout_results[alpha].append(overall)
            for cat, val in cat_means.items():
                holdout_cat_results[alpha][cat].append(val)

        restore_gt_structures(predictor, originals)
        pp.SELECTIVE_BOOST_ALPHA = original_alpha
        print(f"  Seed {seed} complete")

    # Summary
    print(f"\n{'=' * 70}")
    print("=== HOLDOUT SUMMARY ===")
    print(f"{'=' * 70}")
    print(f"\n{'Alpha':>6s} | {'R@30 (holdout)':>18s} | {'Improvement':>12s}")
    print("-" * 45)
    baseline_holdout = np.mean(holdout_results[0.0])  # alpha=0 is no boost

    for alpha in test_alphas:
        mean_r30 = np.mean(holdout_results[alpha])
        std_r30 = np.std(holdout_results[alpha])
        delta = mean_r30 - baseline_holdout
        print(f"  {alpha:4.1f} | {mean_r30:5.1f}% ± {std_r30:4.1f}% | {delta:+5.1f}pp")

    # Category-specific results for boosted categories
    print(f"\n=== CATEGORY-SPECIFIC R@30 (holdout avg) ===")
    print(f"{'Category':15s} | " + " | ".join(f"α={a:.1f}" for a in test_alphas) + " |")
    print("-" * (20 + 10 * len(test_alphas)))

    for cat in sorted(SELECTIVE_BOOST_CATEGORIES):
        row = f"{cat:15s} |"
        for alpha in test_alphas:
            vals = holdout_cat_results[alpha].get(cat, [])
            if vals:
                mean_v = np.mean(vals)
                row += f" {mean_v:5.1f}%  |"
            else:
                row += "   n/a  |"
        print(row)

    # Non-boosted categories (should be unchanged)
    print(f"\n{'---Non-boosted---':15s}")
    all_cats = set()
    for alpha_cats in holdout_cat_results.values():
        all_cats.update(alpha_cats.keys())
    non_boost = sorted(all_cats - SELECTIVE_BOOST_CATEGORIES)

    for cat in non_boost:
        row = f"{cat:15s} |"
        for alpha in test_alphas:
            vals = holdout_cat_results[alpha].get(cat, [])
            if vals:
                mean_v = np.mean(vals)
                row += f" {mean_v:5.1f}%  |"
            else:
                row += "   n/a  |"
        print(row)

    # Find optimal
    best_alpha = max(test_alphas, key=lambda a: np.mean(holdout_results[a]))
    best_r30 = np.mean(holdout_results[best_alpha])
    current_r30 = np.mean(holdout_results[0.5])
    improvement = best_r30 - current_r30
    print(f"\nBest alpha: {best_alpha:.1f} (R@30={best_r30:.1f}%)")
    print(f"Current alpha=0.5: R@30={current_r30:.1f}%")
    print(f"Improvement: {improvement:+.1f}pp")

    if abs(improvement) < 1.0:
        print("\nConclusion: Current alpha=0.5 is near-optimal. No change needed.")
    else:
        print(f"\nConclusion: alpha={best_alpha} improves R@30 by {improvement:.1f}pp. Consider deploying.")

    # Save
    output = {
        "holdout_results": {str(a): holdout_results[a] for a in test_alphas},
        "best_alpha": best_alpha,
        "best_r30": best_r30,
        "current_r30": current_r30,
        "improvement": improvement,
    }
    out_path = Path(__file__).parent.parent / "data" / "analysis" / "h414_alpha_optimization.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
