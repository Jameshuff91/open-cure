#!/usr/bin/env python3
"""
h745: Cancer Same-Type High-Rank Rescue Analysis

cancer_same_type_high_rank (mech + rank>=21) was demoted LOW by h648 at 25.5% holdout.
Current cached holdout shows 40.7% ± 8.7% — at MEDIUM level.
This script investigates whether promotion back to MEDIUM is justified.

Checks:
1. Fresh 5-seed holdout precision with per-seed breakdown
2. CS artifact analysis
3. Known vs novel split
4. Score gradient within sub-reason
5. Comparison to MEDIUM avg
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
    extract_cancer_types,
    DISEASE_HIERARCHY_GROUPS,
    HIERARCHY_EXCLUSIONS,
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
    print("h745: Cancer Same-Type High-Rank Rescue Analysis")
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
    # SECTION 1: Full-data composition
    # ================================================================
    print("\n" + "=" * 70)
    print("SECTION 1: Full-data composition")
    print("=" * 70)

    target_preds = []
    drug_counter = Counter()
    disease_counter = Counter()
    scores_list = []
    ranks_list = []
    cs_count = 0

    for disease_id in all_diseases:
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        try:
            result = predictor.predict(disease_name, top_n=30, include_filtered=True)
        except Exception:
            continue

        for pred in result.predictions:
            sr = pred.category_specific_tier
            if sr == 'cancer_same_type_high_rank':
                target_preds.append((disease_id, pred))
                drug_counter[pred.drug_name] += 1
                disease_counter[disease_name] += 1
                scores_list.append(pred.knn_score)
                ranks_list.append(pred.rank)
                if is_cs(pred.drug_name):
                    cs_count += 1

    n = len(target_preds)
    print(f"\ncancer_same_type_high_rank: {n} predictions")
    print(f"CS predictions: {cs_count}/{n} ({100*cs_count/n:.1f}%)")
    print(f"Score: mean={np.mean(scores_list):.2f}, median={np.median(scores_list):.2f}, "
          f"range=[{np.min(scores_list):.2f}, {np.max(scores_list):.2f}]")
    print(f"Rank: mean={np.mean(ranks_list):.1f}, range=[{min(ranks_list)}, {max(ranks_list)}]")

    # Full-data precision
    hits_full = sum(1 for did, p in target_preds if (did, p.drug_id) in gt_set)
    print(f"Full-data precision: {100*hits_full/n:.1f}% ({hits_full}/{n})")

    print(f"\nTop drugs:")
    for drug, cnt in drug_counter.most_common(15):
        print(f"  {drug}: {cnt}")

    print(f"\nTop diseases:")
    for disease, cnt in disease_counter.most_common(10):
        print(f"  {disease}: {cnt}")

    # ================================================================
    # SECTION 2: 5-seed holdout with detailed breakdown
    # ================================================================
    print("\n" + "=" * 70)
    print("SECTION 2: 5-seed holdout evaluation")
    print("=" * 70)

    all_prec = []
    all_n = []
    all_prec_non_cs = []
    all_n_non_cs = []
    all_known_prec = []
    all_novel_prec = []
    all_known_n = []
    all_novel_n = []
    score_bucket_results = defaultdict(list)  # (lo,hi) -> [prec per seed]
    score_bucket_n = defaultdict(list)

    score_buckets = [(0, 1.5), (1.5, 3.0), (3.0, 5.0), (5.0, 100.0)]

    for seed in SEEDS:
        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        train_set = set(train_ids)

        originals = recompute_gt_structures(predictor, train_set)

        hits = 0
        total = 0
        hits_non_cs = 0
        total_non_cs = 0
        known_hits = 0
        known_total = 0
        novel_hits = 0
        novel_total = 0
        bucket_hits = defaultdict(int)
        bucket_total = defaultdict(int)

        for disease_id in holdout_ids:
            if disease_id not in predictor.ground_truth:
                continue
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            try:
                result = predictor.predict(disease_name, top_n=30, include_filtered=True)
            except Exception:
                continue

            for pred in result.predictions:
                sr = pred.category_specific_tier
                if sr != 'cancer_same_type_high_rank':
                    continue

                total += 1
                hit = (disease_id, pred.drug_id) in gt_set
                if hit:
                    hits += 1

                # CS
                if not is_cs(pred.drug_name):
                    total_non_cs += 1
                    if hit:
                        hits_non_cs += 1

                # Known vs novel
                is_known = pred.drug_id in predictor.ground_truth.get(disease_id, set())
                if is_known:
                    known_total += 1
                    if hit:
                        known_hits += 1
                else:
                    novel_total += 1
                    if hit:
                        novel_hits += 1

                # Score bucket
                for lo, hi in score_buckets:
                    if lo <= pred.knn_score < hi:
                        bucket_total[(lo, hi)] += 1
                        if hit:
                            bucket_hits[(lo, hi)] += 1
                        break

        prec = 100 * hits / total if total > 0 else 0
        prec_nc = 100 * hits_non_cs / total_non_cs if total_non_cs > 0 else 0
        known_p = 100 * known_hits / known_total if known_total > 0 else 0
        novel_p = 100 * novel_hits / novel_total if novel_total > 0 else 0

        all_prec.append(prec)
        all_n.append(total)
        all_prec_non_cs.append(prec_nc)
        all_n_non_cs.append(total_non_cs)
        all_known_prec.append(known_p)
        all_novel_prec.append(novel_p)
        all_known_n.append(known_total)
        all_novel_n.append(novel_total)

        for b in score_buckets:
            bp = 100 * bucket_hits[b] / bucket_total[b] if bucket_total[b] > 0 else 0
            score_bucket_results[b].append(bp)
            score_bucket_n[b].append(bucket_total[b])

        print(f"  Seed {seed}: {prec:.1f}% (n={total}), non-CS: {prec_nc:.1f}% (n={total_non_cs}), "
              f"known: {known_p:.1f}% (n={known_total}), novel: {novel_p:.1f}% (n={novel_total})")

        restore_gt_structures(predictor, originals)

    mean_prec = np.mean(all_prec)
    std_prec = np.std(all_prec)
    mean_n = np.mean(all_n)
    mean_prec_nc = np.mean(all_prec_non_cs)
    std_prec_nc = np.std(all_prec_non_cs)
    mean_n_nc = np.mean(all_n_non_cs)
    known_mean = np.mean(all_known_prec)
    novel_mean = np.mean(all_novel_prec)
    known_n_mean = np.mean(all_known_n)
    novel_n_mean = np.mean(all_novel_n)

    print(f"\n  ALL:     {mean_prec:.1f}% ± {std_prec:.1f}% (n={mean_n:.1f}/seed)")
    print(f"  Non-CS:  {mean_prec_nc:.1f}% ± {std_prec_nc:.1f}% (n={mean_n_nc:.1f}/seed)")
    print(f"  Known:   {known_mean:.1f}% (n={known_n_mean:.1f}/seed)")
    print(f"  Novel:   {novel_mean:.1f}% (n={novel_n_mean:.1f}/seed)")

    # ================================================================
    # SECTION 3: Score gradient
    # ================================================================
    print("\n" + "=" * 70)
    print("SECTION 3: Score gradient within cancer_same_type_high_rank")
    print("=" * 70)

    for b in score_buckets:
        mn = np.mean(score_bucket_results[b])
        sd = np.std(score_bucket_results[b])
        n_avg = np.mean(score_bucket_n[b])
        print(f"  Score [{b[0]:.1f}, {b[1]:.1f}): {mn:.1f}% ± {sd:.1f}% (n={n_avg:.1f}/seed)")

    # ================================================================
    # SECTION 4: Context — compare with other cancer sub-reasons and MEDIUM avg
    # ================================================================
    print("\n" + "=" * 70)
    print("SECTION 4: Context and Recommendation")
    print("=" * 70)

    # Current MEDIUM holdout (from h741)
    medium_avg = 47.7
    # Current MEDIUM novel holdout (from h738)
    medium_novel = 22.8

    z = (mean_prec - medium_avg) / (std_prec / np.sqrt(len(SEEDS))) if std_prec > 0 else 0
    z_novel = (novel_mean - medium_novel) / 5.0  # rough estimate of novel std

    print(f"\n  Current assignment: LOW")
    print(f"  Holdout:      {mean_prec:.1f}% ± {std_prec:.1f}%")
    print(f"  Non-CS:       {mean_prec_nc:.1f}% ± {std_prec_nc:.1f}%")
    print(f"  Novel:        {novel_mean:.1f}%")
    print(f"  MEDIUM avg:   {medium_avg:.1f}%")
    print(f"  MEDIUM novel: {medium_novel:.1f}%")
    print(f"  z vs MEDIUM:  {z:.2f}")
    print(f"  z novel:      {z_novel:.2f}")
    print(f"  CS fraction:  {100*cs_count/n:.1f}%")
    print(f"  Predictions:  {n}")

    # Decision logic
    if mean_prec >= 40 and novel_mean >= 15:
        print(f"\n  >>> RECOMMENDATION: PROMOTE to MEDIUM")
        print(f"      Holdout {mean_prec:.1f}% >= 40% (MEDIUM threshold)")
        print(f"      Novel {novel_mean:.1f}% >= 15% (not just known-indication hits)")
        print(f"      {n} predictions would move LOW → MEDIUM")
    elif mean_prec >= 35:
        print(f"\n  >>> RECOMMENDATION: BORDERLINE — consider promotion with score gating")
        print(f"      Holdout {mean_prec:.1f}% is 35-40% range")
    else:
        print(f"\n  >>> RECOMMENDATION: KEEP as LOW")
        print(f"      Holdout {mean_prec:.1f}% < 35% (below MEDIUM threshold)")

    # Save results
    results = {
        "hypothesis": "h745",
        "sub_reason": "cancer_same_type_high_rank",
        "current_tier": "LOW",
        "n_predictions": n,
        "holdout_mean": round(mean_prec, 1),
        "holdout_std": round(std_prec, 1),
        "holdout_seeds": [round(x, 1) for x in all_prec],
        "mean_n_per_seed": round(mean_n, 1),
        "non_cs_holdout_mean": round(mean_prec_nc, 1),
        "non_cs_holdout_std": round(std_prec_nc, 1),
        "non_cs_n_per_seed": round(mean_n_nc, 1),
        "known_holdout_mean": round(known_mean, 1),
        "novel_holdout_mean": round(novel_mean, 1),
        "known_n_per_seed": round(known_n_mean, 1),
        "novel_n_per_seed": round(novel_n_mean, 1),
        "cs_fraction": round(cs_count / n, 3),
        "z_vs_medium": round(z, 2),
        "score_gradient": {
            f"[{b[0]:.1f}, {b[1]:.1f})": {
                "holdout_mean": round(np.mean(score_bucket_results[b]), 1),
                "holdout_std": round(np.std(score_bucket_results[b]), 1),
                "n_per_seed": round(np.mean(score_bucket_n[b]), 1),
            }
            for b in score_buckets
        },
    }

    out_path = "data/analysis/h745_cancer_high_rank_rescue.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
