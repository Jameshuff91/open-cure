#!/usr/bin/env python3
"""
h638: Target Overlap MEDIUM → HIGH Promotion Analysis

Stratify target_overlap_promotion MEDIUM predictions by:
- TransE consilience (yes/no)
- Mechanism support (yes/no)
- kNN rank bucket (1-5, 6-10, 11-15, 16-20)
- Disease category
- Corticosteroid status

Goal: Find a high-signal subset (>55% holdout, n>=10/seed) for HIGH promotion.

Uses h393-style holdout evaluation with expanded GT.
"""

import json
import sys
from collections import defaultdict
from copy import deepcopy
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

# Import h393 helpers
sys.path.insert(0, str(Path(__file__).parent))
from h393_holdout_tier_validation import (
    split_diseases,
    recompute_gt_structures,
    restore_gt_structures,
)

# Known corticosteroid drug names (lowercase)
CORTICOSTEROIDS = {
    'dexamethasone', 'prednisolone', 'prednisone', 'methylprednisolone',
    'hydrocortisone', 'cortisone', 'betamethasone', 'triamcinolone',
    'budesonide', 'fluticasone', 'beclomethasone', 'mometasone',
    'fluocinolone', 'clobetasol', 'halobetasol', 'desoximetasone',
    'desonide', 'fluocinonide', 'halcinonide', 'amcinonide',
    'diflorasone', 'ciclesonide', 'fludrocortisone',
}


def is_corticosteroid(drug_name: str) -> bool:
    """Check if drug is a corticosteroid."""
    return drug_name.lower().strip() in CORTICOSTEROIDS


def stratify_target_overlap(
    predictor: DrugRepurposingPredictor,
    disease_ids: List[str],
    gt_data: Dict[str, List[str]],
) -> Dict:
    """Stratify target_overlap_promotion MEDIUM predictions by multiple signals."""

    # Build GT set
    gt_set = set()
    for disease_id, drugs in gt_data.items():
        for drug in drugs:
            if isinstance(drug, str):
                gt_set.add((disease_id, drug))
            elif isinstance(drug, dict):
                gt_set.add((disease_id, drug.get("drug_id") or drug.get("drug")))

    # Collect stratified counts
    strats = defaultdict(lambda: {"hits": 0, "total": 0, "drugs": set(), "diseases": set()})

    n_evaluated = 0
    for disease_id in disease_ids:
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        try:
            result = predictor.predict(disease_name, top_n=30, include_filtered=True)
        except Exception:
            continue

        n_evaluated += 1

        for pred in result.predictions:
            # Only interested in target_overlap_promotion MEDIUM
            if pred.confidence_tier != ConfidenceTier.MEDIUM:
                continue
            if pred.category_specific_tier != 'target_overlap_promotion':
                continue

            drug_id = pred.drug_id
            is_hit = (disease_id, drug_id) in gt_set

            # Get signals
            transe = pred.transe_consilience
            mech = pred.mechanism_support
            rank = pred.rank
            category = pred.category
            cs = is_corticosteroid(pred.drug_name)

            # Rank bucket
            if rank <= 5:
                rank_bucket = "R1-5"
            elif rank <= 10:
                rank_bucket = "R6-10"
            elif rank <= 15:
                rank_bucket = "R11-15"
            else:
                rank_bucket = "R16-20"

            # Collect into strats
            for key_name, key_val in [
                ("ALL", "all"),
                ("TransE", "Y" if transe else "N"),
                ("Mech", "Y" if mech else "N"),
                ("Rank", rank_bucket),
                ("Category", category),
                ("CS", "Y" if cs else "N"),
                # Combinations
                ("TransE+Mech", f"{'T' if transe else 'NT'}+{'M' if mech else 'NM'}"),
                ("TransE+Rank", f"{'T' if transe else 'NT'}+{rank_bucket}"),
                ("Mech+Rank", f"{'M' if mech else 'NM'}+{rank_bucket}"),
                # Triple
                ("TransE+Mech+Rank", f"{'T' if transe else 'NT'}+{'M' if mech else 'NM'}+{rank_bucket}"),
                # Non-CS
                ("NonCS", "Y" if not cs else "N"),
                ("NonCS+TransE", f"{'NC' if not cs else 'CS'}+{'T' if transe else 'NT'}"),
                ("NonCS+Mech", f"{'NC' if not cs else 'CS'}+{'M' if mech else 'NM'}"),
                ("NonCS+Rank", f"{'NC' if not cs else 'CS'}+{rank_bucket}"),
                ("NonCS+TransE+Mech", f"{'NC' if not cs else 'CS'}+{'T' if transe else 'NT'}+{'M' if mech else 'NM'}"),
            ]:
                key = f"{key_name}={key_val}"
                strats[key]["total"] += 1
                if is_hit:
                    strats[key]["hits"] += 1
                strats[key]["drugs"].add(pred.drug_name)
                strats[key]["diseases"].add(disease_id)

    # Convert sets to counts
    results = {}
    for key, stats in strats.items():
        total = stats["total"]
        precision = stats["hits"] / total * 100 if total > 0 else 0
        results[key] = {
            "hits": stats["hits"],
            "total": total,
            "precision": round(precision, 1),
            "n_drugs": len(stats["drugs"]),
            "n_diseases": len(stats["diseases"]),
        }

    return {"n_diseases_evaluated": n_evaluated, "strats": results}


def main():
    seeds = [42, 123, 456, 789, 2024]

    print("=" * 70)
    print("h638: Target Overlap MEDIUM Stratification Analysis")
    print("=" * 70)

    print("\nLoading predictor...")
    predictor = DrugRepurposingPredictor()

    # Load expanded GT
    gt_path = predictor.reference_dir / "expanded_ground_truth.json"
    with open(gt_path) as f:
        gt_data = json.load(f)
    print(f"Loaded {len(gt_data)} GT diseases, {sum(len(v) for v in gt_data.values())} pairs")

    all_diseases = [d for d in predictor.ground_truth if d in predictor.embeddings]
    print(f"Diseases with GT + embeddings: {len(all_diseases)}")

    # Full-data baseline
    print("\n--- FULL-DATA BASELINE ---")
    full_result = stratify_target_overlap(predictor, all_diseases, gt_data)
    print(f"Evaluated {full_result['n_diseases_evaluated']} diseases")
    print("\nFull-data stratification:")
    for key in sorted(full_result['strats'].keys()):
        s = full_result['strats'][key]
        if s['total'] >= 5:
            print(f"  {key}: {s['precision']}% ({s['hits']}/{s['total']}) [{s['n_drugs']} drugs, {s['n_diseases']} diseases]")

    # Holdout evaluation
    print("\n--- HOLDOUT EVALUATION (5 seeds) ---")
    all_seed_strats = defaultdict(list)  # key -> [precision per seed]
    all_seed_n = defaultdict(list)       # key -> [n per seed]

    for seed_idx, seed in enumerate(seeds):
        print(f"\nSeed {seed} ({seed_idx+1}/{len(seeds)})...")

        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        train_set = set(train_ids)

        originals = recompute_gt_structures(predictor, train_set)
        holdout_result = stratify_target_overlap(predictor, holdout_ids, gt_data)
        restore_gt_structures(predictor, originals)

        for key, stats in holdout_result['strats'].items():
            total = stats['total']
            precision = stats['precision']
            all_seed_strats[key].append(precision)
            all_seed_n[key].append(total)

    # Aggregate holdout results
    print("\n" + "=" * 70)
    print("HOLDOUT RESULTS (5-seed average)")
    print("=" * 70)

    print(f"\n{'Stratification':<50} {'Holdout%':>10} {'±std':>8} {'N/seed':>8} {'Full%':>8}")
    print("-" * 90)

    # Sort by holdout precision descending
    results_list = []
    for key in all_seed_strats:
        mean_prec = np.mean(all_seed_strats[key])
        std_prec = np.std(all_seed_strats[key])
        mean_n = np.mean(all_seed_n[key])
        full_prec = full_result['strats'].get(key, {}).get('precision', 0)
        results_list.append((key, mean_prec, std_prec, mean_n, full_prec))

    results_list.sort(key=lambda x: -x[1])

    for key, mean_prec, std_prec, mean_n, full_prec in results_list:
        if mean_n >= 3:  # Only show if reasonable sample size
            marker = ""
            if mean_prec >= 55 and mean_n >= 10:
                marker = " *** PROMOTABLE"
            elif mean_prec >= 50 and mean_n >= 10:
                marker = " ** NEAR-PROMOTABLE"
            elif mean_prec >= 45 and mean_n >= 10:
                marker = " * HIGH-QUALITY"
            print(f"  {key:<50} {mean_prec:>8.1f}% {std_prec:>7.1f}% {mean_n:>7.1f} {full_prec:>7.1f}%{marker}")

    # Specific analysis: Top drugs
    print("\n" + "=" * 70)
    print("TOP DRUGS IN target_overlap_promotion MEDIUM (full data)")
    print("=" * 70)

    drug_stats = defaultdict(lambda: {"hits": 0, "total": 0, "categories": set()})

    for disease_id in all_diseases:
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        try:
            result = predictor.predict(disease_name, top_n=30, include_filtered=True)
        except Exception:
            continue

        for pred in result.predictions:
            if pred.confidence_tier != ConfidenceTier.MEDIUM:
                continue
            if pred.category_specific_tier != 'target_overlap_promotion':
                continue

            is_hit = (disease_id, pred.drug_id) in set()  # We'll compute this separately
            drug_stats[pred.drug_name]["total"] += 1
            drug_stats[pred.drug_name]["categories"].add(pred.category)

    print(f"\n{'Drug':<35} {'Preds':>6} {'Categories'}")
    print("-" * 70)
    for drug, stats in sorted(drug_stats.items(), key=lambda x: -x[1]['total'])[:30]:
        cats = ", ".join(sorted(stats['categories']))
        print(f"  {drug:<35} {stats['total']:>6} {cats}")

    # Save results
    output = {
        "full_data": full_result['strats'],
        "holdout": {
            key: {
                "mean_precision": round(np.mean(all_seed_strats[key]), 1),
                "std_precision": round(np.std(all_seed_strats[key]), 1),
                "mean_n_per_seed": round(np.mean(all_seed_n[key]), 1),
            }
            for key in all_seed_strats
        }
    }

    output_path = Path(__file__).parent.parent / "data" / "analysis" / "h638_target_overlap_stratify.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
