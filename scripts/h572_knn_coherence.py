#!/usr/bin/env python3
"""
h572: kNN Neighborhood Category Coherence as Prediction Quality Signal

For each disease, compute the fraction of k=20 nearest neighbors that share
the same disease category. This is a non-circular signal (uses embeddings only,
no GT) that may predict per-disease prediction quality.

h569 showed GT size strongly predicts precision (r=0.732). This tests whether
embedding neighborhood coherence is a usable proxy.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from production_predictor import (
    ConfidenceTier,
    DrugRepurposingPredictor,
    DISEASE_HIERARCHY_GROUPS,
)


def split_diseases(all_diseases, seed, train_ratio=0.8):
    rng = np.random.RandomState(seed)
    shuffled = list(all_diseases)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def recompute_gt_structures(predictor, train_disease_ids):
    """Minimal recompute for holdout evaluation."""
    from production_predictor import extract_cancer_types

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
                new_d2d[drug_id].add(disease_name)
    predictor.drug_to_diseases = dict(new_d2d)

    new_cancer_types = {}
    for drug_id, diseases in new_d2d.items():
        cancer_types = set()
        for d in diseases:
            cancer_types.update(extract_cancer_types(d))
        if cancer_types:
            new_cancer_types[drug_id] = cancer_types
    predictor.drug_cancer_types = new_cancer_types

    new_groups = defaultdict(set)
    for drug_id, diseases in new_d2d.items():
        for d in diseases:
            for group_name, group_data in DISEASE_HIERARCHY_GROUPS.items():
                for disease_pattern in group_data.get("diseases", []):
                    if disease_pattern.lower() in d.lower():
                        new_groups[drug_id].add(group_name)
    predictor.drug_disease_groups = dict(new_groups)

    train_disease_list = [d for d in predictor.train_diseases if d in train_disease_ids]
    predictor.train_diseases = train_disease_list
    indices = [i for i, d in enumerate(originals["train_diseases"]) if d in train_disease_ids]
    predictor.train_embeddings = originals["train_embeddings"][indices]
    predictor.train_disease_categories = {
        d: originals["train_disease_categories"][d]
        for d in train_disease_list
        if d in originals["train_disease_categories"]
    }

    return originals


def restore_gt_structures(predictor, originals):
    for key, val in originals.items():
        setattr(predictor, key, val)


def compute_category_coherence(predictor, disease_id, k=20):
    """Compute fraction of k nearest neighbors that share same category.

    Uses ALL disease embeddings (not just training diseases) since this
    is a structural property of the embedding space, not GT-dependent.
    """
    if disease_id not in predictor.embeddings:
        return None

    disease_name = predictor.disease_names.get(disease_id, disease_id)
    disease_cat = predictor.categorize_disease(disease_name)

    disease_emb = predictor.embeddings[disease_id].reshape(1, -1)

    # Get all disease embeddings
    all_disease_ids = list(predictor.embeddings.keys())
    all_embeddings = np.array([predictor.embeddings[d] for d in all_disease_ids], dtype=np.float32)

    # Compute similarities
    sims = cosine_similarity(disease_emb, all_embeddings)[0]

    # Get top-k+1 (excluding self)
    top_indices = np.argsort(-sims)[:k + 1]

    same_cat_count = 0
    neighbor_count = 0
    for idx in top_indices:
        neighbor_id = all_disease_ids[idx]
        if neighbor_id == disease_id:
            continue
        if neighbor_count >= k:
            break
        neighbor_name = predictor.disease_names.get(neighbor_id, neighbor_id)
        neighbor_cat = predictor.categorize_disease(neighbor_name)
        if neighbor_cat == disease_cat:
            same_cat_count += 1
        neighbor_count += 1

    return same_cat_count / k if k > 0 else 0


def main():
    print("=" * 70)
    print("h572: kNN Neighborhood Category Coherence as Quality Signal")
    print("=" * 70)

    predictor = DrugRepurposingPredictor()
    gt_diseases = [d for d in predictor.ground_truth if d in predictor.embeddings]
    print(f"\nGT diseases with embeddings: {len(gt_diseases)}")

    # Step 1: Compute category coherence for all GT diseases
    print("\n--- Step 1: Compute Category Coherence ---")
    coherence_scores = {}
    for disease_id in gt_diseases:
        score = compute_category_coherence(predictor, disease_id)
        if score is not None:
            coherence_scores[disease_id] = score

    print(f"Computed coherence for {len(coherence_scores)} diseases")

    # Distribution
    scores = list(coherence_scores.values())
    print(f"Coherence distribution: mean={np.mean(scores):.3f}, median={np.median(scores):.3f}, "
          f"std={np.std(scores):.3f}, min={np.min(scores):.3f}, max={np.max(scores):.3f}")

    # By category
    cat_scores = defaultdict(list)
    for disease_id, score in coherence_scores.items():
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        cat = predictor.categorize_disease(disease_name)
        cat_scores[cat].append(score)

    print("\nBy category:")
    for cat in sorted(cat_scores.keys()):
        scores_cat = cat_scores[cat]
        print(f"  {cat}: mean={np.mean(scores_cat):.3f} (n={len(scores_cat)})")

    # Step 2: Load per-disease holdout precision from h569
    print("\n--- Step 2: Correlate with Per-Disease Holdout Precision ---")
    try:
        with open('data/reference/disease_holdout_precision.json') as f:
            disease_precision = json.load(f)
        print(f"Loaded {len(disease_precision)} disease precision values")
    except FileNotFoundError:
        print("disease_holdout_precision.json not found. Computing from scratch...")
        disease_precision = None

    if disease_precision:
        # Correlate coherence with precision
        both = []
        for disease_id in coherence_scores:
            entry = disease_precision.get(disease_id)
            if entry is not None and isinstance(entry, dict):
                prec = entry.get('holdout_precision')
                gt_size = entry.get('gt_size', 0)
                if prec is not None:
                    both.append((coherence_scores[disease_id], float(prec), gt_size))

        if both:
            coherence_vals = np.array([b[0] for b in both], dtype=float)
            prec_vals = np.array([b[1] for b in both], dtype=float)
            gt_vals = np.array([b[2] for b in both], dtype=float)
            r_prec = np.corrcoef(coherence_vals, prec_vals)[0, 1]
            r_gt = np.corrcoef(coherence_vals, gt_vals)[0, 1]
            print(f"Correlation (coherence vs holdout precision): r={r_prec:.3f} (n={len(both)})")
            print(f"Correlation (coherence vs GT size): r={r_gt:.3f}")
            print(f"For comparison: GT size vs holdout precision: r={np.corrcoef(gt_vals, prec_vals)[0, 1]:.3f}")

    # Step 3: Direct holdout test - bin diseases by coherence and measure precision
    print("\n--- Step 3: Holdout Precision by Coherence Bin ---")

    # Bin diseases by coherence
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    seeds = [42, 123, 456, 789, 1337]

    bin_results = {b: {'hits': [], 'total': []} for b in bins}
    overall_results = {'hits': [], 'total': []}

    for seed in seeds:
        train, holdout = split_diseases(gt_diseases, seed)
        train_set = set(train)
        originals = recompute_gt_structures(predictor, train_set)

        seed_bin_hits = {b: 0 for b in bins}
        seed_bin_total = {b: 0 for b in bins}
        seed_all_hits = 0
        seed_all_total = 0

        for disease_id in holdout:
            if disease_id not in predictor.embeddings:
                continue
            if disease_id not in coherence_scores:
                continue

            coherence = coherence_scores[disease_id]
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            result = predictor.predict(disease_name, k=20, top_n=30, include_filtered=True)
            gt_drugs = set(predictor.ground_truth.get(disease_id, []))
            if not gt_drugs:
                continue

            for pred in result.predictions:
                # Count all tiers (not just MEDIUM) for disease-level quality
                is_hit = pred.drug_id in gt_drugs
                seed_all_hits += int(is_hit)
                seed_all_total += 1

                for (lo, hi) in bins:
                    if lo <= coherence < hi:
                        seed_bin_hits[(lo, hi)] += int(is_hit)
                        seed_bin_total[(lo, hi)] += 1
                        break

        for b in bins:
            bin_results[b]['hits'].append(seed_bin_hits[b])
            bin_results[b]['total'].append(seed_bin_total[b])
        overall_results['hits'].append(seed_all_hits)
        overall_results['total'].append(seed_all_total)

        restore_gt_structures(predictor, originals)

    # Print results
    overall_prec = [h / t * 100 if t > 0 else 0 for h, t in zip(overall_results['hits'], overall_results['total'])]
    print(f"\nOverall holdout precision: {np.mean(overall_prec):.1f}% ± {np.std(overall_prec):.1f}%")

    print(f"\n{'Coherence Bin':<18} | {'Holdout':<12} | {'n/seed':<8} | {'Diseases':<10} | {'vs Overall':<12}")
    print("-" * 70)

    for (lo, hi) in bins:
        data = bin_results[(lo, hi)]
        precs = [h / t * 100 if t > 0 else 0 for h, t in zip(data['hits'], data['total'])]
        mean_n = np.mean(data['total'])
        # Count diseases in this bin (holdout only)
        n_diseases = sum(1 for d in gt_diseases if d in coherence_scores and lo <= coherence_scores[d] < hi)
        delta = np.mean(precs) - np.mean(overall_prec)
        print(f"  [{lo:.1f}, {hi:.1f})" + " " * (10 - len(f"[{lo:.1f}, {hi:.1f})")) +
              f"| {np.mean(precs):5.1f}%±{np.std(precs):4.1f}% | {mean_n:6.1f} | {n_diseases:<10} | {delta:+5.1f}pp")

    # Step 4: Per-tier analysis
    print("\n--- Step 4: Per-Tier Coherence Analysis ---")

    # For each tier, check if coherence bins matter
    for target_tier in ['GOLDEN', 'HIGH', 'MEDIUM', 'LOW']:
        tier_enum = getattr(ConfidenceTier, target_tier)
        tier_bin_results = {b: {'hits': [], 'total': []} for b in bins}

        for seed in seeds:
            train, holdout = split_diseases(gt_diseases, seed)
            train_set = set(train)
            originals = recompute_gt_structures(predictor, train_set)

            for b in bins:
                tier_bin_results[b]['hits'].append(0)
                tier_bin_results[b]['total'].append(0)

            for disease_id in holdout:
                if disease_id not in predictor.embeddings:
                    continue
                if disease_id not in coherence_scores:
                    continue

                coherence = coherence_scores[disease_id]
                disease_name = predictor.disease_names.get(disease_id, disease_id)
                result = predictor.predict(disease_name, k=20, top_n=30, include_filtered=True)
                gt_drugs = set(predictor.ground_truth.get(disease_id, []))
                if not gt_drugs:
                    continue

                for pred in result.predictions:
                    if pred.confidence_tier != tier_enum:
                        continue
                    is_hit = pred.drug_id in gt_drugs

                    for (lo, hi) in bins:
                        if lo <= coherence < hi:
                            tier_bin_results[(lo, hi)]['hits'][-1] += int(is_hit)
                            tier_bin_results[(lo, hi)]['total'][-1] += 1
                            break

            restore_gt_structures(predictor, originals)

        print(f"\n{target_tier}:")
        for (lo, hi) in bins:
            data = tier_bin_results[(lo, hi)]
            precs = [h / t * 100 if t > 0 else 0 for h, t in zip(data['hits'], data['total'])]
            mean_n = np.mean(data['total'])
            if mean_n >= 1:
                print(f"  [{lo:.1f}, {hi:.1f}): {np.mean(precs):5.1f}%±{np.std(precs):4.1f}% (n/seed={mean_n:.1f})")

    # Step 5: Correlation with GT size (to check if coherence is just a proxy)
    print("\n--- Step 5: Coherence vs GT Size ---")
    gt_sizes = {}
    for disease_id in gt_diseases:
        gt_sizes[disease_id] = len(predictor.ground_truth.get(disease_id, []))

    coh_gt_both = [(coherence_scores[d], gt_sizes[d]) for d in gt_diseases if d in coherence_scores]
    if coh_gt_both:
        coh_vals = np.array([b[0] for b in coh_gt_both], dtype=float)
        gt_vals = np.array([b[1] for b in coh_gt_both], dtype=float)
        r = np.corrcoef(coh_vals, gt_vals)[0, 1]
        print(f"Correlation (coherence vs GT size): r={r:.3f}")
        print(f"If high, coherence is just a proxy for GT size (circular)")
        print(f"If low, coherence captures something independent")

    # Save
    results_summary = {
        "hypothesis": "h572",
        "n_diseases": len(coherence_scores),
        "coherence_mean": round(np.mean(list(coherence_scores.values())), 3),
        "coherence_std": round(np.std(list(coherence_scores.values())), 3),
    }
    with open("data/analysis/h572_output.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSaved to data/analysis/h572_output.json")


if __name__ == "__main__":
    main()
