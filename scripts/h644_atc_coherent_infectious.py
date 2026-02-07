#!/usr/bin/env python3
"""
h644: ATC Coherent Infectious Quality Investigation

h642 found atc_coherent_infectious NoMech = 42.4% (n=34/seed) — above MEDIUM average
despite no mechanism support. This is surprising because infectious disease drugs
typically need mechanism for precision.

Questions:
1. What drugs are driving this high precision?
2. Are they genuinely good predictions or self-referential artifacts?
3. Are they dominated by broad-spectrum antibiotics that happen to hit GT?
4. Should this sub-rule be annotated differently?

Also investigate the overall atc_coherent breakdown by category and mechanism.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from production_predictor import (
    ConfidenceTier,
    DrugRepurposingPredictor,
)

sys.path.insert(0, str(Path(__file__).parent))
from h393_holdout_tier_validation import (
    split_diseases,
    recompute_gt_structures,
    restore_gt_structures,
)


def analyze_atc_coherent(
    predictor: DrugRepurposingPredictor,
    disease_ids: List[str],
    gt_data: Dict[str, List[str]],
    label: str = "full",
) -> Dict:
    """Analyze atc_coherent MEDIUM predictions by category and mechanism."""

    gt_set = set()
    for disease_id, drugs in gt_data.items():
        for drug in drugs:
            if isinstance(drug, str):
                gt_set.add((disease_id, drug))
            elif isinstance(drug, dict):
                gt_set.add((disease_id, drug.get("drug_id") or drug.get("drug")))

    # Collect all atc_coherent MEDIUM predictions
    strats: Dict[str, Dict] = defaultdict(lambda: {"hits": 0, "total": 0, "drugs": set(), "diseases": set(), "predictions": []})

    for disease_id in disease_ids:
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        try:
            result = predictor.predict(disease_name, top_n=30, include_filtered=True)
        except Exception:
            continue

        for pred in result.predictions:
            if pred.confidence_tier != ConfidenceTier.MEDIUM:
                continue
            cat_specific = pred.category_specific_tier or ""
            if not cat_specific.startswith("atc_coherent"):
                continue

            drug_id = pred.drug_id
            is_hit = (disease_id, drug_id) in gt_set
            mech = pred.mechanism_support
            rank = pred.rank
            category = pred.category
            transe = pred.transe_consilience

            # Various stratifications
            for key_name, key_val in [
                ("ALL", "all"),
                ("cat_specific", cat_specific),
                ("category", category),
                ("mech", "Y" if mech else "N"),
                ("transe", "Y" if transe else "N"),
                # Combinations
                (f"cat={category}", f"mech={'Y' if mech else 'N'}"),
                (f"cat={category}", f"transe={'Y' if transe else 'N'}"),
                (f"rule={cat_specific}", f"mech={'Y' if mech else 'N'}"),
                (f"rule={cat_specific}", f"transe={'Y' if transe else 'N'}"),
                # Category + mechanism combo
                (f"cat+mech", f"{category}+{'M' if mech else 'NM'}"),
            ]:
                key = f"{key_name}={key_val}"
                strats[key]["hits"] += 1 if is_hit else 0
                strats[key]["total"] += 1
                strats[key]["drugs"].add(pred.drug_name)
                strats[key]["diseases"].add(disease_id)

            # For infectious NoMech, collect detailed predictions
            if category == "infectious" and not mech:
                strats["infectious_nomech_detail"]["predictions"].append({
                    "disease": disease_name,
                    "drug": pred.drug_name,
                    "rank": rank,
                    "hit": is_hit,
                    "transe": transe,
                    "cat_specific": cat_specific,
                })
                strats["infectious_nomech_detail"]["hits"] += 1 if is_hit else 0
                strats["infectious_nomech_detail"]["total"] += 1
                strats["infectious_nomech_detail"]["drugs"].add(pred.drug_name)
                strats["infectious_nomech_detail"]["diseases"].add(disease_id)

    results = {}
    for key, stats in strats.items():
        total = stats["total"]
        precision = stats["hits"] / total * 100 if total > 0 else 0
        entry = {
            "hits": stats["hits"],
            "total": total,
            "precision": round(precision, 1),
            "n_drugs": len(stats["drugs"]),
            "n_diseases": len(stats["diseases"]),
        }
        if stats.get("predictions"):
            entry["predictions"] = stats["predictions"]
        results[key] = entry

    return results


def main():
    seeds = [42, 123, 456, 789, 2024]

    print("=" * 70)
    print("h644: ATC Coherent Infectious Quality Investigation")
    print("=" * 70)

    print("\nLoading predictor...")
    predictor = DrugRepurposingPredictor()

    gt_path = predictor.reference_dir / "expanded_ground_truth.json"
    with open(gt_path) as f:
        gt_data = json.load(f)
    print(f"Loaded {len(gt_data)} GT diseases, {sum(len(v) for v in gt_data.values())} pairs")

    all_diseases = [d for d in predictor.ground_truth if d in predictor.embeddings]
    print(f"Diseases with GT + embeddings: {len(all_diseases)}")

    # Full-data baseline
    print("\n--- FULL-DATA BASELINE ---")
    full_result = analyze_atc_coherent(predictor, all_diseases, gt_data)

    print(f"\n{'Stratification':<55} {'Prec%':>8} {'Hits/Total':>12} {'Drugs':>6} {'Diseases':>8}")
    print("-" * 95)
    for key in sorted(full_result.keys()):
        s = full_result[key]
        if s['total'] >= 3 and key != 'infectious_nomech_detail':
            print(f"  {key:<55} {s['precision']:>7.1f}% {s['hits']}/{s['total']:>5} {s['n_drugs']:>6} {s['n_diseases']:>8}")

    # Print infectious NoMech detail
    if 'infectious_nomech_detail' in full_result:
        print("\n--- INFECTIOUS NoMech DETAIL (full data) ---")
        detail = full_result['infectious_nomech_detail']
        print(f"Total: {detail['hits']}/{detail['total']} ({detail['precision']}%)")

        # Group by disease
        by_disease: Dict[str, list] = defaultdict(list)
        for p in detail.get('predictions', []):
            by_disease[p['disease']].append(p)

        for disease, preds in sorted(by_disease.items()):
            hits = sum(1 for p in preds if p['hit'])
            print(f"\n  {disease}: {hits}/{len(preds)} hits")
            for p in sorted(preds, key=lambda x: x['rank']):
                marker = 'HIT' if p['hit'] else 'miss'
                transe_marker = ' [TransE]' if p['transe'] else ''
                print(f"    R{p['rank']:>2}: {p['drug']:<30} [{marker}]{transe_marker}")

        # Drug frequency
        from collections import Counter
        drug_counts = Counter(p['drug'] for p in detail.get('predictions', []))
        drug_hits = Counter(p['drug'] for p in detail.get('predictions', []) if p['hit'])
        print("\n  Top drugs:")
        for drug, count in drug_counts.most_common(20):
            h = drug_hits.get(drug, 0)
            print(f"    {drug:<30} {h}/{count} ({h/count*100:.0f}%)")

    # Holdout evaluation
    print("\n--- HOLDOUT EVALUATION (5 seeds) ---")
    all_seed_strats: Dict[str, list] = defaultdict(list)
    all_seed_n: Dict[str, list] = defaultdict(list)

    for seed_idx, seed in enumerate(seeds):
        print(f"\nSeed {seed} ({seed_idx+1}/{len(seeds)})...")
        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        train_set = set(train_ids)
        originals = recompute_gt_structures(predictor, train_set)
        holdout_result = analyze_atc_coherent(predictor, holdout_ids, gt_data)
        restore_gt_structures(predictor, originals)

        for key, stats in holdout_result.items():
            if key == 'infectious_nomech_detail':
                continue
            all_seed_strats[key].append(stats['precision'])
            all_seed_n[key].append(stats['total'])

    # Aggregate holdout
    print("\n" + "=" * 70)
    print("HOLDOUT RESULTS (5-seed average)")
    print("=" * 70)

    print(f"\n{'Stratification':<55} {'Holdout%':>10} {'±std':>8} {'N/seed':>8} {'Full%':>8}")
    print("-" * 95)

    results_list = []
    for key in all_seed_strats:
        mean_prec = np.mean(all_seed_strats[key])
        std_prec = np.std(all_seed_strats[key])
        mean_n = np.mean(all_seed_n[key])
        full_prec = full_result.get(key, {}).get('precision', 0)
        results_list.append((key, mean_prec, std_prec, mean_n, full_prec))

    results_list.sort(key=lambda x: -x[1])

    for key, mean_prec, std_prec, mean_n, full_prec in results_list:
        if mean_n >= 3:
            print(f"  {key:<55} {mean_prec:>8.1f}% {std_prec:>7.1f}% {mean_n:>7.1f} {full_prec:>7.1f}%")


if __name__ == "__main__":
    main()
