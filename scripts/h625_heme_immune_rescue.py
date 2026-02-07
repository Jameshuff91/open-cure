#!/usr/bin/env python3
"""
h625: Hematological Immune-Mediated Corticosteroid Rescue

Hypothesis: Corticosteroid→immune-mediated hematological diseases should be MEDIUM,
not LOW. The hematological_corticosteroid_demotion is too broad — it catches both:
- Immune-mediated cytopenias (CS genuinely treats) → 47.5% full-data precision
- Genetic/structural disorders (CS doesn't help) → 4.7% full-data precision

Test with 5-seed holdout validation.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from production_predictor import (
    DrugRepurposingPredictor,
    DISEASE_HIERARCHY_GROUPS,
    extract_cancer_types,
)


# Immune-mediated hematological diseases where corticosteroids are genuine treatment
IMMUNE_MEDIATED_HEME_KEYWORDS = [
    'autoimmune hemolytic', 'warm autoimmune', 'cold autoimmune',
    'immune thrombocytopeni', 'idiopathic thrombocytopeni',
    'pure red cell aplasia', 'aplastic anemia',
    'evans syndrome',
    'heparin-induced thrombocytopeni', 'heparininduced thrombocytopeni',
    'thrombotic thrombocytopenic purpura',
    'hemolytic uremic',
    'hypereosinophilic',
    'hemolytic anemia',
    'graft versus host', 'gvhd',
    'acquired hemophilia',
]


def is_immune_mediated_heme(disease_name: str) -> bool:
    """Check if a hematological disease is immune-mediated."""
    dl = disease_name.lower()
    for kw in IMMUNE_MEDIATED_HEME_KEYWORDS:
        if kw in dl:
            return True
    # "anemia" without genetic qualifiers
    if 'anemia' in dl and 'sickle' not in dl and 'thalassemia' not in dl:
        return True
    return False


def split_diseases(all_diseases, seed, train_ratio=0.8):
    rng = np.random.RandomState(seed)
    shuffled = list(all_diseases)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def recompute_gt_structures(predictor, train_disease_ids):
    originals = {
        "drug_train_freq": dict(predictor.drug_train_freq),
        "drug_to_diseases": {k: set(v) for k, v in predictor.drug_to_diseases.items()},
        "drug_cancer_types": {k: set(v) for k, v in predictor.drug_cancer_types.items()},
        "drug_disease_groups": {k: dict(v) for k, v in predictor.drug_disease_groups.items()},
    }

    new_freq = defaultdict(int)
    new_d2d = defaultdict(set)
    new_cancer_types = defaultdict(set)
    new_disease_groups = defaultdict(lambda: defaultdict(set))

    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            for drug_id in predictor.ground_truth[disease_id]:
                new_freq[drug_id] += 1
                new_d2d[drug_id].add(disease_name)
                for ct in extract_cancer_types(disease_name):
                    new_cancer_types[drug_id].add(ct)
                for group_name, group_info in DISEASE_HIERARCHY_GROUPS.items():
                    parent_kws = group_info.get("parent_keywords", [])
                    if any(kw.lower() in disease_name.lower() for kw in parent_kws):
                        new_disease_groups[drug_id][group_name].add(disease_name)

    predictor.drug_train_freq = dict(new_freq)
    predictor.drug_to_diseases = dict(new_d2d)
    predictor.drug_cancer_types = dict(new_cancer_types)
    predictor.drug_disease_groups = dict(new_disease_groups)
    return originals


def restore_gt_structures(predictor, originals):
    for k, v in originals.items():
        setattr(predictor, k, v)


def main():
    print("=" * 70)
    print("h625: Hematological Immune-Mediated Corticosteroid Rescue")
    print("=" * 70)

    predictor = DrugRepurposingPredictor()
    gt_path = predictor.reference_dir / "expanded_ground_truth.json"
    with open(gt_path) as f:
        gt_data = json.load(f)

    all_diseases = sorted(
        d for d in predictor.disease_names
        if d in predictor.embeddings and d in gt_data
    )

    seeds = [42, 123, 456, 789, 2024]

    # Track results for the hematological_corticosteroid_demotion rule
    # Split by immune-mediated vs non-immune
    immune_results = defaultdict(list)
    non_immune_results = defaultdict(list)

    # Also track hematological_medium_demotion for immune-mediated diseases
    immune_med_results = defaultdict(list)
    non_immune_med_results = defaultdict(list)

    # Track all current tier results for MEDIUM impact calculation
    tier_results = defaultdict(lambda: defaultdict(list))

    for seed in seeds:
        print(f"\nSeed {seed}...")
        train_diseases, holdout_diseases = split_diseases(all_diseases, seed)
        train_set = set(train_diseases)
        originals = recompute_gt_structures(predictor, train_set)

        for disease_id in holdout_diseases:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            try:
                result = predictor.predict(disease_name, top_n=30, include_filtered=True)
            except Exception:
                continue

            gt_drugs = set(gt_data.get(disease_id, []))
            for pred in result.predictions:
                is_hit = pred.drug_id in gt_drugs
                tier = pred.confidence_tier.name
                tier_results[tier][seed].append(is_hit)

                if pred.category_specific_tier == 'hematological_corticosteroid_demotion':
                    if is_immune_mediated_heme(disease_name):
                        immune_results[seed].append(is_hit)
                    else:
                        non_immune_results[seed].append(is_hit)

                if pred.category_specific_tier == 'hematological_medium_demotion':
                    if is_immune_mediated_heme(disease_name):
                        immune_med_results[seed].append(is_hit)
                    else:
                        non_immune_med_results[seed].append(is_hit)

        restore_gt_structures(predictor, originals)

    # Report
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\n--- hematological_corticosteroid_demotion ---")
    for label, results in [("immune_mediated", immune_results), ("non_immune", non_immune_results)]:
        precs = [100 * sum(results[s]) / len(results[s]) for s in seeds if results[s]]
        mean_n = np.mean([len(results[s]) for s in seeds])
        if precs:
            print(f"  {label}: {np.mean(precs):.1f}% ± {np.std(precs):.1f}% (n={mean_n:.1f}/seed, total~{int(mean_n*5)})")
        else:
            print(f"  {label}: no predictions in holdout")

    print("\n--- hematological_medium_demotion ---")
    for label, results in [("immune_mediated", immune_med_results), ("non_immune", non_immune_med_results)]:
        precs = [100 * sum(results[s]) / len(results[s]) for s in seeds if results[s]]
        mean_n = np.mean([len(results[s]) for s in seeds])
        if precs:
            print(f"  {label}: {np.mean(precs):.1f}% ± {np.std(precs):.1f}% (n={mean_n:.1f}/seed, total~{int(mean_n*5)})")
        else:
            print(f"  {label}: no predictions in holdout")

    print("\n--- Current tier precision ---")
    for tier in ['GOLDEN', 'HIGH', 'MEDIUM', 'LOW', 'FILTER']:
        precs = [100 * sum(tier_results[tier][s]) / len(tier_results[tier][s])
                for s in seeds if tier_results[tier][s]]
        mean_n = np.mean([len(tier_results[tier][s]) for s in seeds])
        if precs:
            print(f"  {tier}: {np.mean(precs):.1f}% ± {np.std(precs):.1f}% (n={mean_n:.1f}/seed)")

    # Impact analysis: if immune_mediated CS predictions are rescued to MEDIUM
    print("\n--- Impact Analysis ---")
    immune_cs_n = np.mean([len(immune_results[s]) for s in seeds])
    immune_cs_prec = np.mean([100 * sum(immune_results[s]) / len(immune_results[s])
                              for s in seeds if immune_results[s]]) if any(immune_results[s] for s in seeds) else 0

    medium_n = np.mean([len(tier_results['MEDIUM'][s]) for s in seeds])
    medium_prec = np.mean([100 * sum(tier_results['MEDIUM'][s]) / len(tier_results['MEDIUM'][s])
                          for s in seeds if tier_results['MEDIUM'][s]])

    new_medium_n = medium_n + immune_cs_n
    # Weighted average
    if new_medium_n > 0:
        new_medium_prec = (medium_prec * medium_n + immune_cs_prec * immune_cs_n) / new_medium_n
        print(f"  Current MEDIUM: {medium_prec:.1f}% (n={medium_n:.0f}/seed)")
        print(f"  Immune CS rescue: {immune_cs_prec:.1f}% (n={immune_cs_n:.1f}/seed)")
        print(f"  New MEDIUM: {new_medium_prec:.1f}% (n={new_medium_n:.0f}/seed)")
        print(f"  Delta: {new_medium_prec - medium_prec:+.1f}pp")


if __name__ == "__main__":
    main()
