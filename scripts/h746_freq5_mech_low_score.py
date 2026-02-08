#!/usr/bin/env python3
"""
h746: Default Freq5 Mechanism Low-Score Demotion Analysis

h742 analysis found default_freq5_mechanism at score<1.0 has 31.7% ± 6.4% (n=18.6/seed)
— borderline LOW vs MEDIUM. z-score vs MEDIUM avg (47.7%): z=-2.5 below MEDIUM.

This script investigates:
1. What drugs/diseases are in the freq5_mech score<1.0 group
2. Whether the mechanism signal compensates for low kNN score
3. Score gradient within freq5_mech
4. CS artifact analysis
5. Whether demotion to LOW is warranted
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from production_predictor import (
    ConfidenceTier,
    DrugRepurposingPredictor,
)
from h393_holdout_tier_validation import (
    split_diseases,
    recompute_gt_structures,
    restore_gt_structures,
)

SEEDS = [42, 123, 456, 789, 2024]

CS_DRUGS = {
    'prednisone', 'prednisolone', 'dexamethasone', 'methylprednisolone',
    'hydrocortisone', 'cortisone', 'betamethasone', 'triamcinolone',
    'budesonide', 'fluticasone', 'mometasone', 'beclomethasone',
    'fluocinolone', 'clobetasol', 'desonide', 'halobetasol',
    'fludrocortisone',
}


def is_cs(drug_name: str) -> bool:
    return any(cs in drug_name.lower() for cs in CS_DRUGS)


def main():
    print("=" * 70)
    print("h746: Default Freq5 Mechanism Low-Score Demotion Analysis")
    print("=" * 70)

    # Load predictor
    print("\nLoading predictor...")
    predictor = DrugRepurposingPredictor()

    # Load expanded GT
    gt_path = predictor.reference_dir / "expanded_ground_truth.json"
    with open(gt_path) as f:
        expanded_gt = json.load(f)
    print(f"Expanded GT: {sum(len(v) for v in expanded_gt.values())} pairs")

    # Build GT set for matching
    gt_set = set()
    for disease_id, drugs in expanded_gt.items():
        for drug in drugs:
            if isinstance(drug, str):
                gt_set.add((disease_id, drug))

    # Get all evaluable diseases
    all_diseases = [d for d in predictor.ground_truth if d in predictor.embeddings]
    print(f"Evaluable diseases: {len(all_diseases)}")

    # ================================================================
    # SECTION 1: Full-data composition of freq5_mech by score
    # ================================================================
    print("\n" + "=" * 70)
    print("SECTION 1: Full-data composition by score bracket")
    print("=" * 70)

    score_brackets = {
        "[0.0, 0.5)": (0.0, 0.5),
        "[0.5, 1.0)": (0.5, 1.0),
        "[1.0, 1.5)": (1.0, 1.5),
        "[1.5, 2.0)": (1.5, 2.0),
        "[2.0, 3.0)": (2.0, 3.0),
        "[3.0, 5.0)": (3.0, 5.0),
        "[5.0, inf)": (5.0, 100.0),
    }

    bracket_preds = {k: [] for k in score_brackets}
    all_freq5_mech = []

    for disease_id in all_diseases:
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        try:
            result = predictor.predict(disease_name, top_n=30, include_filtered=True)
        except Exception:
            continue

        for pred in result.predictions:
            sr = pred.category_specific_tier
            if sr == 'default_freq5_mechanism':
                all_freq5_mech.append((disease_id, pred))
                for bracket_name, (lo, hi) in score_brackets.items():
                    if lo <= pred.knn_score < hi:
                        bracket_preds[bracket_name].append((disease_id, pred))
                        break

    print(f"\nTotal default_freq5_mechanism: {len(all_freq5_mech)}")

    for bracket_name, preds in bracket_preds.items():
        if not preds:
            continue
        n = len(preds)
        hits = sum(1 for did, p in preds if (did, p.drug_id) in gt_set)
        cs_n = sum(1 for _, p in preds if is_cs(p.drug_name))
        ranks = [p.rank for _, p in preds]
        print(f"\n{bracket_name}: {n} preds, full-data={100*hits/n:.1f}% ({hits}/{n}), "
              f"CS={100*cs_n/n:.1f}%, mean_rank={np.mean(ranks):.1f}")

        # Top drugs in this bracket
        drug_counts = Counter(p.drug_name for _, p in preds)
        top5 = drug_counts.most_common(5)
        print(f"  Top drugs: {', '.join(f'{d}({c})' for d, c in top5)}")

    # ================================================================
    # SECTION 2: Detailed analysis of score < 1.0
    # ================================================================
    print("\n" + "=" * 70)
    print("SECTION 2: Detailed analysis of score < 1.0")
    print("=" * 70)

    low_score_preds = []
    for k in ["[0.0, 0.5)", "[0.5, 1.0)"]:
        low_score_preds.extend(bracket_preds[k])

    print(f"\nTotal score < 1.0: {len(low_score_preds)}")

    if low_score_preds:
        # Drug distribution
        drug_counts = Counter(p.drug_name for _, p in low_score_preds)
        print(f"\nTop 20 drugs:")
        for drug, cnt in drug_counts.most_common(20):
            print(f"  {drug}: {cnt}")

        # Disease category distribution
        cat_counts = Counter(getattr(p, 'disease_category', 'unknown') for _, p in low_score_preds)
        print(f"\nDisease categories:")
        for cat, cnt in cat_counts.most_common():
            print(f"  {cat}: {cnt}")

        # TransE consilience
        transe_preds = [p for _, p in low_score_preds if getattr(p, 'transe_consilience', False)]
        print(f"\nTransE consilience: {len(transe_preds)}/{len(low_score_preds)} "
              f"({100*len(transe_preds)/len(low_score_preds):.1f}%)")

        # Literature evidence
        for level in ['STRONG_EVIDENCE', 'MODERATE_EVIDENCE', 'WEAK_EVIDENCE', 'NO_EVIDENCE']:
            lit = [p for _, p in low_score_preds
                   if getattr(p, 'literature_evidence_level', None) == level]
            if lit:
                print(f"  Literature {level}: {len(lit)}")

        # CS analysis
        cs_n = sum(1 for _, p in low_score_preds if is_cs(p.drug_name))
        noncs = [(did, p) for did, p in low_score_preds if not is_cs(p.drug_name)]
        print(f"\nCS: {cs_n}/{len(low_score_preds)} ({100*cs_n/len(low_score_preds):.1f}%)")

        # Sample predictions
        print(f"\nSample predictions (sorted by score):")
        sorted_preds = sorted(low_score_preds, key=lambda x: x[1].knn_score)
        for did, p in sorted_preds[:15]:
            dname = predictor.disease_names.get(did, did)[:40]
            in_gt = "GT" if (did, p.drug_id) in gt_set else "novel"
            print(f"  {p.drug_name:25s} → {dname:40s} score={p.knn_score:.3f} rank={p.rank:2d} [{in_gt}]")

    # ================================================================
    # SECTION 3: 5-seed holdout evaluation by score bracket
    # ================================================================
    print("\n" + "=" * 70)
    print("SECTION 3: 5-seed holdout evaluation")
    print("=" * 70)

    # Define score thresholds to test
    thresholds = [0.5, 0.75, 1.0, 1.25, 1.5]

    for threshold in thresholds:
        seed_precisions = []
        seed_ns = []
        seed_noncs_precisions = []
        seed_noncs_ns = []

        for seed in SEEDS:
            train_diseases, holdout_diseases = split_diseases(all_diseases, seed)
            train_ids = set(train_diseases)
            holdout_ids = set(holdout_diseases)

            originals = recompute_gt_structures(predictor, train_ids)

            hits = 0
            total = 0
            noncs_hits = 0
            noncs_total = 0

            for disease_id in holdout_ids:
                disease_name = predictor.disease_names.get(disease_id, disease_id)
                try:
                    result = predictor.predict(disease_name, top_n=30, include_filtered=True)
                except Exception:
                    continue

                for pred in result.predictions:
                    sr = pred.category_specific_tier
                    if sr == 'default_freq5_mechanism' and pred.knn_score < threshold:
                        total += 1
                        if (disease_id, pred.drug_id) in gt_set:
                            hits += 1
                        if not is_cs(pred.drug_name):
                            noncs_total += 1
                            if (disease_id, pred.drug_id) in gt_set:
                                noncs_hits += 1

            restore_gt_structures(predictor, originals)

            if total > 0:
                seed_precisions.append(100 * hits / total)
                seed_ns.append(total)
            if noncs_total > 0:
                seed_noncs_precisions.append(100 * noncs_hits / noncs_total)
                seed_noncs_ns.append(noncs_total)

        if seed_precisions:
            mean_p = np.mean(seed_precisions)
            std_p = np.std(seed_precisions)
            mean_n = np.mean(seed_ns)
            print(f"\nScore < {threshold}:")
            print(f"  All:    {mean_p:.1f}% ± {std_p:.1f}% (n={mean_n:.1f}/seed)")
            print(f"  Seeds:  {[f'{v:.1f}' for v in seed_precisions]}")
            if seed_noncs_precisions:
                mean_nc = np.mean(seed_noncs_precisions)
                std_nc = np.std(seed_noncs_precisions)
                mean_nc_n = np.mean(seed_noncs_ns)
                print(f"  NonCS:  {mean_nc:.1f}% ± {std_nc:.1f}% (n={mean_nc_n:.1f}/seed)")

    # ================================================================
    # SECTION 4: Also test score >= threshold for remaining MEDIUM
    # ================================================================
    print("\n" + "=" * 70)
    print("SECTION 4: Remaining MEDIUM (score >= threshold)")
    print("=" * 70)

    for threshold in [0.75, 1.0, 1.25]:
        seed_precisions = []
        seed_ns = []

        for seed in SEEDS:
            train_diseases, holdout_diseases = split_diseases(all_diseases, seed)
            train_ids = set(train_diseases)
            holdout_ids = set(holdout_diseases)

            originals = recompute_gt_structures(predictor, train_ids)

            hits = 0
            total = 0

            for disease_id in holdout_ids:
                disease_name = predictor.disease_names.get(disease_id, disease_id)
                try:
                    result = predictor.predict(disease_name, top_n=30, include_filtered=True)
                except Exception:
                    continue

                for pred in result.predictions:
                    sr = pred.category_specific_tier
                    if sr == 'default_freq5_mechanism' and pred.knn_score >= threshold:
                        total += 1
                        if (disease_id, pred.drug_id) in gt_set:
                            hits += 1

            restore_gt_structures(predictor, originals)

            if total > 0:
                seed_precisions.append(100 * hits / total)
                seed_ns.append(total)

        if seed_precisions:
            mean_p = np.mean(seed_precisions)
            std_p = np.std(seed_precisions)
            mean_n = np.mean(seed_ns)
            print(f"\nScore >= {threshold}:")
            print(f"  Holdout: {mean_p:.1f}% ± {std_p:.1f}% (n={mean_n:.1f}/seed)")
            print(f"  Seeds:   {[f'{v:.1f}' for v in seed_precisions]}")

    # ================================================================
    # SECTION 5: Impact on MEDIUM tier
    # ================================================================
    print("\n" + "=" * 70)
    print("SECTION 5: Estimated MEDIUM tier impact")
    print("=" * 70)

    # Current MEDIUM stats from h744 analysis: 47.7% ± 2.4% (n=249.2)
    current_medium_mean = 47.7
    current_medium_n = 249.2

    for threshold in [0.75, 1.0, 1.25]:
        below = [p for _, p in all_freq5_mech if p.knn_score < threshold]
        n_demoted = len(below)
        remaining = len(all_freq5_mech) - n_demoted
        print(f"\nDemote score < {threshold}:")
        print(f"  {n_demoted} predictions demoted to LOW")
        print(f"  {remaining} remain in MEDIUM")
        # Very rough estimate: if demoted group was ~31.7% and total MEDIUM was 47.7%
        # then removing low-precision preds should lift MEDIUM precision

    print("\nDone.")


if __name__ == "__main__":
    main()
