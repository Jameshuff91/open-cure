#!/usr/bin/env python3
"""
h629: MEDIUM Precision Stratification by Multiple Signals

Can we identify a MEDIUM subset with >50% holdout precision by combining
existing signals (TransE, gene_overlap, mechanism, rank, tier_rule)?

This would identify the best MEDIUM predictions for experimental testing.
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
    """Recompute GT-derived structures for training diseases only."""
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
                new_d2d[drug_id].add(disease_name.lower())
    predictor.drug_to_diseases = dict(new_d2d)

    new_cancer = defaultdict(set)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            cancer_types = extract_cancer_types(disease_name)
            if cancer_types:
                for drug_id in predictor.ground_truth[disease_id]:
                    new_cancer[drug_id].update(cancer_types)
    predictor.drug_cancer_types = dict(new_cancer)

    new_groups = defaultdict(set)
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
    print("h629: MEDIUM Precision Stratification by Multiple Signals")
    print("=" * 70)

    # Load predictor
    print("\nLoading predictor...")
    predictor = DrugRepurposingPredictor()

    # Load expanded GT for evaluation
    with open("data/reference/expanded_ground_truth.json") as f:
        expanded_gt = json.load(f)

    # Use predictor's own disease list (455 diseases with internal GT + embeddings)
    all_diseases = [d for d in predictor.ground_truth.keys() if d in predictor.embeddings]
    print(f"Diseases with GT + embeddings: {len(all_diseases)}")

    seeds = [42, 123, 456, 789, 1024]

    # Collect per-prediction data across seeds
    # Signal groups to test
    group_results = defaultdict(lambda: defaultdict(lambda: {"hits": [], "totals": []}))

    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"SEED {seed} ({seed_idx+1}/{len(seeds)})")
        print(f"{'='*60}")

        train, holdout = split_diseases(all_diseases, seed)
        train_set = set(train)
        holdout_set = set(holdout)

        originals = recompute_gt_structures(predictor, train_set)

        # Use EXPANDED GT for evaluation (h611: always use expanded GT)
        gt_set = set()
        for disease_id, drugs in expanded_gt.items():
            for drug_id in drugs:
                gt_set.add((disease_id, drug_id))

        # Evaluate holdout
        medium_preds = []
        for disease_id in holdout:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            try:
                result = predictor.predict(
                    disease_name, top_n=30, include_filtered=True
                )
            except Exception:
                continue

            for pred in result.predictions:
                if pred.confidence_tier.name != "MEDIUM":
                    continue

                drug_id = pred.drug_id
                is_hit = (disease_id, drug_id) in gt_set

                medium_preds.append({
                    "disease_id": disease_id,
                    "drug_id": drug_id,
                    "is_hit": is_hit,
                    "tier_rule": pred.category_specific_tier or "default",
                    "rank": pred.rank,
                    "transe": getattr(pred, "transe_consilience", False),
                    "mechanism": getattr(pred, "mechanism_support", False),
                    "gene_overlap": getattr(pred, "gene_overlap_count", 0),
                    "category": pred.category,
                })

        print(f"  MEDIUM predictions: {len(medium_preds)}")
        print(f"  MEDIUM hits: {sum(p['is_hit'] for p in medium_preds)}")
        print(f"  MEDIUM precision: {sum(p['is_hit'] for p in medium_preds) / max(len(medium_preds), 1) * 100:.1f}%")

        # Stratify by various signal combinations
        signal_groups = {
            # Single signals
            "all_medium": lambda p: True,
            "transe_yes": lambda p: p["transe"],
            "transe_no": lambda p: not p["transe"],
            "mechanism_yes": lambda p: p["mechanism"],
            "mechanism_no": lambda p: not p["mechanism"],
            "rank_1_5": lambda p: p["rank"] <= 5,
            "rank_6_10": lambda p: 6 <= p["rank"] <= 10,
            "rank_11_20": lambda p: p["rank"] > 10,
            "gene_overlap_yes": lambda p: p["gene_overlap"] > 0,
            "gene_overlap_no": lambda p: p["gene_overlap"] == 0,

            # Double combinations
            "transe_AND_mechanism": lambda p: p["transe"] and p["mechanism"],
            "transe_AND_rank_1_5": lambda p: p["transe"] and p["rank"] <= 5,
            "transe_AND_rank_1_10": lambda p: p["transe"] and p["rank"] <= 10,
            "mechanism_AND_rank_1_5": lambda p: p["mechanism"] and p["rank"] <= 5,
            "mechanism_AND_rank_1_10": lambda p: p["mechanism"] and p["rank"] <= 10,
            "transe_AND_gene_overlap": lambda p: p["transe"] and p["gene_overlap"] > 0,
            "mechanism_AND_gene_overlap": lambda p: p["mechanism"] and p["gene_overlap"] > 0,

            # Triple combinations
            "transe_AND_mechanism_AND_rank_1_5": lambda p: p["transe"] and p["mechanism"] and p["rank"] <= 5,
            "transe_AND_mechanism_AND_rank_1_10": lambda p: p["transe"] and p["mechanism"] and p["rank"] <= 10,
            "all_signals": lambda p: p["transe"] and p["mechanism"] and p["gene_overlap"] > 0,
            "all_signals_AND_rank_1_5": lambda p: p["transe"] and p["mechanism"] and p["gene_overlap"] > 0 and p["rank"] <= 5,

            # Tier rule specific
            "cancer_same_type": lambda p: p["tier_rule"] == "cancer_same_type",
            "target_overlap": lambda p: p["tier_rule"] == "target_overlap_promotion",
            "cv_pathway": lambda p: p["tier_rule"] == "cv_pathway_comprehensive",
            "standard": lambda p: p["tier_rule"] == "standard",
            "atc_coherent": lambda p: "atc_coherent" in p["tier_rule"],

            # Tier rule + signal combos
            "cancer_same_type_AND_transe": lambda p: p["tier_rule"] == "cancer_same_type" and p["transe"],
            "cancer_same_type_AND_rank_1_5": lambda p: p["tier_rule"] == "cancer_same_type" and p["rank"] <= 5,
            "target_overlap_AND_transe": lambda p: p["tier_rule"] == "target_overlap_promotion" and p["transe"],
            "standard_AND_transe": lambda p: p["tier_rule"] == "standard" and p["transe"],
            "standard_AND_mechanism": lambda p: p["tier_rule"] == "standard" and p["mechanism"],
            "standard_AND_rank_1_5": lambda p: p["tier_rule"] == "standard" and p["rank"] <= 5,
            "standard_AND_transe_AND_mechanism": lambda p: p["tier_rule"] == "standard" and p["transe"] and p["mechanism"],
        }

        for group_name, filter_fn in signal_groups.items():
            matching = [p for p in medium_preds if filter_fn(p)]
            hits = sum(p["is_hit"] for p in matching)
            total = len(matching)
            group_results[group_name][seed] = {"hits": hits, "total": total}

        restore_gt_structures(predictor, originals)

    # Aggregate results
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS (5-seed holdout)")
    print("=" * 70)

    summary = []
    for group_name, seed_data in group_results.items():
        precisions = []
        ns = []
        for seed in seeds:
            data = seed_data.get(seed, {"hits": 0, "total": 0})
            if data["total"] > 0:
                precisions.append(data["hits"] / data["total"] * 100)
            else:
                precisions.append(0)
            ns.append(data["total"])

        mean_p = np.mean(precisions)
        std_p = np.std(precisions)
        mean_n = np.mean(ns)
        summary.append((group_name, mean_p, std_p, mean_n))

    summary.sort(key=lambda x: -x[1])  # Sort by precision

    print(f"\n{'Signal Group':<50} {'Holdout':>8} {'±std':>6} {'N/seed':>8} {'Total':>6}")
    print("-" * 82)
    for name, prec, std, n in summary:
        if n >= 3:  # Skip very small groups
            marker = " ***" if prec >= 50 and n >= 5 else " **" if prec >= 40 and n >= 5 else ""
            print(f"  {name:<48} {prec:>6.1f}% {std:>5.1f}% {n:>8.1f} {n*5:>6.0f}{marker}")

    # Highlight: groups with >50% AND n>=5
    print("\n" + "=" * 70)
    print("HIGH-PRECISION MEDIUM SUBSETS (>50% holdout, n>=5/seed)")
    print("=" * 70)
    high_prec = [(n, p, s, nn) for n, p, s, nn in summary if p >= 50 and nn >= 5]
    if high_prec:
        for name, prec, std, n in high_prec:
            print(f"  {name}: {prec:.1f}% ± {std:.1f}% (n={n:.0f}/seed, ~{n*5:.0f} total)")
    else:
        print("  None found with >50% holdout and n>=5/seed")

    # Groups between 40-50% with good n
    print("\n" + "=" * 70)
    print("PROMISING MEDIUM SUBSETS (40-50% holdout, n>=5/seed)")
    print("=" * 70)
    promising = [(n, p, s, nn) for n, p, s, nn in summary if 40 <= p < 50 and nn >= 5]
    if promising:
        for name, prec, std, n in promising:
            print(f"  {name}: {prec:.1f}% ± {std:.1f}% (n={n:.0f}/seed, ~{n*5:.0f} total)")
    else:
        print("  None found")

    # Save results
    output = {
        "summary": [
            {"group": name, "holdout_mean": round(prec, 1), "holdout_std": round(std, 1), "mean_n": round(n, 1)}
            for name, prec, std, n in summary
        ]
    }

    with open("data/analysis/h629_medium_stratification.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to data/analysis/h629_medium_stratification.json")


if __name__ == "__main__":
    main()
