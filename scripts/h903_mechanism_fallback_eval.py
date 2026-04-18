#!/usr/bin/env python3
"""h903: Holdout evaluation of h900 mechanism-only fallback.

Question: When kNN yields no drugs in the neighborhood, h900 ranks drugs by
target-gene overlap instead. Does this fallback produce usable predictions?

Design:
    - 5-seed 80/20 disease split (matches h393 methodology).
    - Recompute train-side GT structures from training diseases only.
    - For each holdout disease, run predict() and look for predictions tagged
      with mechanism_* category_specific_tier.
    - Measure hit rate vs expanded_ground_truth.json, bucketed by top-k.
    - Compare fallback precision to LOW tier baseline and a size-matched
      kNN-tier diseases sample (same per-disease GT pair count).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from production_predictor import DrugRepurposingPredictor  # noqa: E402
from h393_holdout_tier_validation import (  # noqa: E402
    recompute_gt_structures,
    restore_gt_structures,
    split_diseases,
)


FALLBACK_TAGS = {
    "mechanism_only_fallback",
    "mechanism_fallback_literature_strong",
    "mechanism_fallback_literature_moderate",
}


def eval_seed(
    predictor: DrugRepurposingPredictor,
    seed: int,
    gt_data: Dict[str, List],
    all_diseases: List[str],
) -> Dict:
    """Run one seed: split, recompute, evaluate fallback-triggered diseases."""
    train_ids, holdout_ids = split_diseases(all_diseases, seed)
    train_set = set(train_ids)

    originals = recompute_gt_structures(predictor, train_set)

    # Build expanded GT lookup for holdout diseases.
    gt_pairs: Set[Tuple[str, str]] = set()
    for disease_id, drugs in gt_data.items():
        for drug in drugs:
            if isinstance(drug, str):
                gt_pairs.add((disease_id, drug))
            elif isinstance(drug, dict):
                gt_pairs.add((disease_id, drug.get("drug_id") or drug.get("drug")))

    fallback_diseases = []
    fallback_hits_per_k = {5: 0, 10: 0, 30: 0}
    fallback_total_per_k = {5: 0, 10: 0, 30: 0}
    fallback_recall_per_disease = []  # for per-drug R@30
    fallback_gt_counts = []

    # Counts of predictions by tier for fallback diseases vs kNN diseases
    low_tier_hits = low_tier_total = 0  # non-fallback LOW preds (baseline)

    for disease_id in holdout_ids:
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        try:
            result = predictor.predict(disease_name, top_n=30, include_filtered=True)
        except Exception:
            continue

        preds = result.predictions
        fallback_preds = [
            p for p in preds
            if (p.category_specific_tier or "") in FALLBACK_TAGS
        ]
        is_fallback_disease = len(fallback_preds) > 0

        # Known GT drugs for this disease
        disease_gt = {d for (dis, d) in gt_pairs if dis == disease_id}

        if is_fallback_disease:
            fallback_diseases.append(disease_id)
            fallback_gt_counts.append(len(disease_gt))
            # Sort by rank
            fallback_preds.sort(key=lambda p: p.rank)
            # Per-drug R@30: fraction of GT drugs in top-30 preds
            pred_drugs_30 = {p.drug_id for p in fallback_preds[:30]}
            if disease_gt:
                recall = len(disease_gt & pred_drugs_30) / len(disease_gt)
            else:
                recall = None
            if recall is not None:
                fallback_recall_per_disease.append(recall)

            for k in (5, 10, 30):
                top_k = fallback_preds[:k]
                for p in top_k:
                    fallback_total_per_k[k] += 1
                    if (disease_id, p.drug_id) in gt_pairs:
                        fallback_hits_per_k[k] += 1
        else:
            # Track LOW tier from kNN diseases as baseline
            for p in preds:
                if p.confidence_tier.name == "LOW" and p.rank <= 30:
                    low_tier_total += 1
                    if (disease_id, p.drug_id) in gt_pairs:
                        low_tier_hits += 1

    restore_gt_structures(predictor, originals)

    fallback_prec = {
        k: (
            fallback_hits_per_k[k] / fallback_total_per_k[k]
            if fallback_total_per_k[k] > 0 else 0.0
        )
        for k in (5, 10, 30)
    }

    low_prec = low_tier_hits / low_tier_total if low_tier_total > 0 else 0.0

    mean_recall = (
        float(np.mean(fallback_recall_per_disease))
        if fallback_recall_per_disease else 0.0
    )

    return {
        "seed": seed,
        "n_holdout_diseases": len(holdout_ids),
        "n_fallback_diseases": len(fallback_diseases),
        "fallback_disease_ids": fallback_diseases,
        "fallback_gt_counts": fallback_gt_counts,
        "fallback_prec_at_k": fallback_prec,
        "fallback_total_at_k": fallback_total_per_k,
        "fallback_hits_at_k": fallback_hits_per_k,
        "low_tier_prec": low_prec,
        "low_tier_total": low_tier_total,
        "low_tier_hits": low_tier_hits,
        "mean_fallback_recall_at_30": mean_recall,
        "n_diseases_with_gt": len(fallback_recall_per_disease),
    }


def main():
    print("=" * 70)
    print("h903: Mechanism-Only Fallback Holdout Evaluation")
    print("=" * 70)

    predictor = DrugRepurposingPredictor()
    gt_path = predictor.reference_dir / "expanded_ground_truth.json"
    with open(gt_path) as f:
        gt_data = json.load(f)
    print(f"Loaded expanded GT: {len(gt_data)} diseases")

    all_diseases = [d for d in predictor.ground_truth if d in predictor.embeddings]
    print(f"All diseases with GT + embeddings: {len(all_diseases)}")

    seeds = [42, 123, 456, 789, 2024]
    results = []
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        r = eval_seed(predictor, seed, gt_data, all_diseases)
        print(
            f"  n_fallback_diseases={r['n_fallback_diseases']}/{r['n_holdout_diseases']}"
            f"  prec@30={r['fallback_prec_at_k'][30]*100:.1f}%"
            f"  prec@5={r['fallback_prec_at_k'][5]*100:.1f}%"
            f"  LOW_tier_prec={r['low_tier_prec']*100:.1f}% (n={r['low_tier_total']})"
            f"  mean_R@30={r['mean_fallback_recall_at_30']*100:.1f}%"
        )
        results.append(r)

    # Aggregate
    prec30 = [r["fallback_prec_at_k"][30] for r in results]
    prec5 = [r["fallback_prec_at_k"][5] for r in results]
    n_fbk = [r["n_fallback_diseases"] for r in results]
    low_prec = [r["low_tier_prec"] for r in results]
    recalls = [r["mean_fallback_recall_at_30"] for r in results]

    print("\n" + "=" * 70)
    print("AGGREGATE (5 seeds)")
    print("=" * 70)
    print(
        f"  Fallback diseases per seed: {np.mean(n_fbk):.1f} ± {np.std(n_fbk):.1f}"
        f" (range {min(n_fbk)}–{max(n_fbk)})"
    )
    print(
        f"  Fallback precision@30: {np.mean(prec30)*100:.1f}% ± {np.std(prec30)*100:.1f}%"
    )
    print(
        f"  Fallback precision@5:  {np.mean(prec5)*100:.1f}% ± {np.std(prec5)*100:.1f}%"
    )
    print(
        f"  LOW tier baseline (non-fallback): {np.mean(low_prec)*100:.1f}% ± "
        f"{np.std(low_prec)*100:.1f}%"
    )
    print(
        f"  Mean per-disease R@30: {np.mean(recalls)*100:.1f}% ± "
        f"{np.std(recalls)*100:.1f}%"
    )

    out = {
        "seeds": seeds,
        "per_seed": results,
        "aggregate": {
            "n_fallback_mean": float(np.mean(n_fbk)),
            "n_fallback_std": float(np.std(n_fbk)),
            "prec_at_30_mean": float(np.mean(prec30)),
            "prec_at_30_std": float(np.std(prec30)),
            "prec_at_5_mean": float(np.mean(prec5)),
            "prec_at_5_std": float(np.std(prec5)),
            "low_tier_prec_mean": float(np.mean(low_prec)),
            "low_tier_prec_std": float(np.std(low_prec)),
            "recall_at_30_mean": float(np.mean(recalls)),
            "recall_at_30_std": float(np.std(recalls)),
        },
    }

    # Make JSON-serializable
    for r in out["per_seed"]:
        r["fallback_prec_at_k"] = {str(k): v for k, v in r["fallback_prec_at_k"].items()}
        r["fallback_total_at_k"] = {str(k): v for k, v in r["fallback_total_at_k"].items()}
        r["fallback_hits_at_k"] = {str(k): v for k, v in r["fallback_hits_at_k"].items()}

    out_path = ROOT / "data/analysis/h903_fallback_holdout.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
