#!/usr/bin/env python3
"""
h520: SOC Drug Class Precision Heterogeneity Analysis

h518 showed MEDIUM SOC has +6pp over NOVEL on holdout. But this may be driven
by specific drug classes. This script stratifies holdout precision by soc_drug_class
to identify which classes drive the signal and whether any achieve HIGH-level precision.

Approach:
1. Run 5-seed holdout evaluation (same as h393)
2. For each prediction, classify literature_status and capture soc_drug_class
3. Compute per-class, per-tier holdout precision
4. Test statistical significance of per-class lift over NOVEL
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from production_predictor import (
    ConfidenceTier,
    DrugRepurposingPredictor,
    classify_literature_status,
    extract_cancer_types,
    DISEASE_HIERARCHY_GROUPS,
    HIERARCHY_EXCLUSIONS,
)


def split_diseases(
    all_diseases: List[str], seed: int, train_ratio: float = 0.8
) -> Tuple[List[str], List[str]]:
    """Split diseases into train/holdout sets."""
    rng = np.random.RandomState(seed)
    shuffled = list(all_diseases)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def recompute_gt_structures(
    predictor: DrugRepurposingPredictor,
    train_disease_ids: Set[str],
) -> Dict:
    """Recompute all GT-derived data structures from training diseases only."""
    originals = {
        "drug_train_freq": dict(predictor.drug_train_freq),
        "drug_to_diseases": dict(predictor.drug_to_diseases),
        "drug_cancer_types": dict(predictor.drug_cancer_types),
        "drug_disease_groups": dict(predictor.drug_disease_groups),
        "train_diseases": list(predictor.train_diseases),
        "train_embeddings": predictor.train_embeddings.copy(),
        "train_disease_categories": dict(predictor.train_disease_categories),
    }

    # 1. Recompute drug_train_freq
    new_freq: Dict[str, int] = defaultdict(int)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            for drug_id in predictor.ground_truth[disease_id]:
                new_freq[drug_id] += 1
    predictor.drug_train_freq = dict(new_freq)

    # 2. Recompute drug_to_diseases
    new_d2d: Dict[str, Set[str]] = defaultdict(set)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            for drug_id in predictor.ground_truth[disease_id]:
                new_d2d[drug_id].add(disease_name.lower())
    predictor.drug_to_diseases = dict(new_d2d)

    # 3. Recompute drug_cancer_types
    new_cancer: Dict[str, Set[str]] = defaultdict(set)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            cancer_types = extract_cancer_types(disease_name)
            if cancer_types:
                for drug_id in predictor.ground_truth[disease_id]:
                    new_cancer[drug_id].update(cancer_types)
    predictor.drug_cancer_types = dict(new_cancer)

    # 4. Recompute drug_disease_groups
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

    # 5. Rebuild kNN index
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
    """Restore original GT-derived data structures."""
    predictor.drug_train_freq = originals["drug_train_freq"]
    predictor.drug_to_diseases = originals["drug_to_diseases"]
    predictor.drug_cancer_types = originals["drug_cancer_types"]
    predictor.drug_disease_groups = originals["drug_disease_groups"]
    predictor.train_diseases = originals["train_diseases"]
    predictor.train_embeddings = originals["train_embeddings"]
    predictor.train_disease_categories = originals["train_disease_categories"]


def evaluate_soc_stratified(
    predictor: DrugRepurposingPredictor,
    disease_ids: List[str],
    gt_data: Dict[str, List[str]],
) -> Dict:
    """Evaluate holdout with per-SOC-class stratification.

    Returns per-tier, per-class {hits, misses} counts.
    """
    # Build GT set
    gt_set = set()
    for disease_id, drugs in gt_data.items():
        for drug in drugs:
            if isinstance(drug, str):
                gt_set.add((disease_id, drug))
            elif isinstance(drug, dict):
                gt_set.add((disease_id, drug.get("drug_id") or drug.get("drug")))

    # Key: (tier, soc_class_or_NOVEL) -> {hits, total}
    class_stats: Dict[Tuple[str, str], Dict] = defaultdict(lambda: {"hits": 0, "total": 0})

    n_evaluated = 0

    for disease_id in disease_ids:
        disease_name = predictor.disease_names.get(disease_id, disease_id)

        try:
            result = predictor.predict(
                disease_name, top_n=30, include_filtered=True
            )
        except Exception:
            continue

        n_evaluated += 1
        category = result.category

        for pred in result.predictions:
            drug_id = pred.drug_id
            drug_name = pred.drug_name
            is_hit = (disease_id, drug_id) in gt_set
            tier = pred.confidence_tier.name

            # Classify literature status
            is_known = (disease_id, drug_id) in gt_set  # Not quite right; use GT check
            lit_status, soc_class = classify_literature_status(
                drug_name, disease_name, category, False  # Don't count as known
            )

            # Use soc_class if LIKELY_GT_GAP, else "NOVEL"
            group = soc_class if lit_status == 'LIKELY_GT_GAP' else 'NOVEL'

            key = (tier, group)
            class_stats[key]["total"] += 1
            if is_hit:
                class_stats[key]["hits"] += 1

    return {"n_diseases": n_evaluated, "class_stats": dict(class_stats)}


def main() -> None:
    seeds = [42, 123, 456, 789, 1024]
    print("=" * 70)
    print("h520: SOC Drug Class Precision Heterogeneity")
    print("=" * 70)

    print("\nLoading predictor...")
    predictor = DrugRepurposingPredictor()

    # Load expanded ground truth
    gt_path = predictor.reference_dir / "expanded_ground_truth.json"
    with open(gt_path) as f:
        gt_data = json.load(f)
    print(f"Loaded {len(gt_data)} GT diseases, {sum(len(v) for v in gt_data.values())} pairs")

    # Get all diseases with GT + embeddings
    all_diseases = [d for d in predictor.ground_truth if d in predictor.embeddings]
    print(f"Diseases with GT + embeddings: {len(all_diseases)}")

    # --- FULL-DATA baseline per SOC class ---
    print("\n" + "=" * 70)
    print("FULL-DATA BASELINE (per SOC class)")
    print("=" * 70)
    full_result = evaluate_soc_stratified(predictor, all_diseases, gt_data)
    print(f"Evaluated {full_result['n_diseases']} diseases")

    # Print full-data results
    tiers = ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]
    all_classes = sorted(set(k[1] for k in full_result["class_stats"].keys()))
    soc_classes = [c for c in all_classes if c != "NOVEL"]

    print(f"\nSOC classes found: {soc_classes}")
    print(f"\n{'Tier':<10} {'Class':<22} {'Hits':<6} {'Total':<8} {'Precision':<10}")
    print("-" * 60)
    for tier in tiers:
        for cls in ["NOVEL"] + soc_classes:
            key = (tier, cls)
            if key in full_result["class_stats"]:
                s = full_result["class_stats"][key]
                prec = s["hits"] / s["total"] * 100 if s["total"] > 0 else 0
                print(f"{tier:<10} {cls:<22} {s['hits']:<6} {s['total']:<8} {prec:.1f}%")

    # --- HOLDOUT evaluation per SOC class ---
    print("\n" + "=" * 70)
    print("HOLDOUT EVALUATION (5-seed)")
    print("=" * 70)

    # Accumulate per-seed results: (tier, class) -> [{hits, total}, ...]
    seed_results: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)

    for seed_idx, seed in enumerate(seeds):
        print(f"\nSeed {seed} ({seed_idx + 1}/{len(seeds)})...", end=" ", flush=True)

        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        train_set = set(train_ids)

        originals = recompute_gt_structures(predictor, train_set)
        holdout_result = evaluate_soc_stratified(predictor, holdout_ids, gt_data)
        restore_gt_structures(predictor, originals)

        print(f"{holdout_result['n_diseases']} diseases evaluated")

        for key, stats in holdout_result["class_stats"].items():
            seed_results[key].append(stats)

    # --- Analyze results ---
    print("\n" + "=" * 70)
    print("PER-CLASS HOLDOUT PRECISION (5-seed mean)")
    print("=" * 70)

    # Collect all classes seen across holdout
    holdout_classes = sorted(set(k[1] for k in seed_results.keys()))
    holdout_soc = [c for c in holdout_classes if c != "NOVEL"]

    output_data: Dict = {
        "seeds": seeds,
        "per_class_holdout": {},
        "per_class_fulldata": {},
    }

    print(f"\n{'Tier':<10} {'Class':<22} {'Mean Prec':<12} {'SE':<8} {'Mean N':<8} {'vs NOVEL':<12} {'p-value':<10}")
    print("-" * 82)

    for tier in tiers:
        # Get NOVEL baseline for this tier
        novel_key = (tier, "NOVEL")
        novel_precisions = []
        if novel_key in seed_results:
            for s in seed_results[novel_key]:
                prec = s["hits"] / s["total"] * 100 if s["total"] > 0 else 0
                novel_precisions.append(prec)

        novel_mean = np.mean(novel_precisions) if novel_precisions else 0
        novel_se = np.std(novel_precisions, ddof=1) / np.sqrt(len(novel_precisions)) if len(novel_precisions) > 1 else 0
        novel_n = np.mean([s["total"] for s in seed_results.get(novel_key, [])]) if novel_key in seed_results else 0

        print(f"{tier:<10} {'NOVEL':<22} {novel_mean:>7.1f}%    {novel_se:>5.1f}pp {novel_n:>7.0f}  {'(baseline)':<12}")

        output_data["per_class_holdout"][(tier, "NOVEL")] = {
            "mean_precision": round(novel_mean, 2),
            "se": round(novel_se, 2),
            "mean_n": round(novel_n, 1),
        }

        for cls in holdout_soc:
            key = (tier, cls)
            if key not in seed_results:
                continue

            class_precisions = []
            class_ns = []
            for s in seed_results[key]:
                prec = s["hits"] / s["total"] * 100 if s["total"] > 0 else 0
                class_precisions.append(prec)
                class_ns.append(s["total"])

            mean_prec = np.mean(class_precisions)
            se = np.std(class_precisions, ddof=1) / np.sqrt(len(class_precisions)) if len(class_precisions) > 1 else 0
            mean_n = np.mean(class_ns)
            delta = mean_prec - novel_mean

            # Paired t-test if both have 5 seeds
            p_val = "N/A"
            if len(class_precisions) == 5 and len(novel_precisions) == 5:
                try:
                    _, p = scipy_stats.ttest_rel(class_precisions, novel_precisions)
                    p_val = f"{p:.4f}"
                except Exception:
                    p_val = "error"

            delta_str = f"{delta:+.1f}pp"
            if mean_n < 5:
                delta_str += " (tiny-n!)"

            print(f"{tier:<10} {cls:<22} {mean_prec:>7.1f}%    {se:>5.1f}pp {mean_n:>7.0f}  {delta_str:<12} {p_val}")

            output_data["per_class_holdout"][(tier, cls)] = {
                "mean_precision": round(mean_prec, 2),
                "se": round(se, 2),
                "mean_n": round(mean_n, 1),
                "vs_novel_delta": round(delta, 2),
                "p_value": p_val,
            }

        print()  # Blank line between tiers

    # Save full-data results too
    for key, stats in full_result["class_stats"].items():
        prec = stats["hits"] / stats["total"] * 100 if stats["total"] > 0 else 0
        output_data["per_class_fulldata"][f"{key[0]}_{key[1]}"] = {
            "hits": stats["hits"],
            "total": stats["total"],
            "precision": round(prec, 2),
        }

    # Convert tuple keys to strings for JSON
    json_data = {
        "seeds": seeds,
        "per_class_holdout": {f"{k[0]}_{k[1]}": v for k, v in output_data["per_class_holdout"].items()},
        "per_class_fulldata": output_data["per_class_fulldata"],
    }

    output_path = Path("data/analysis/h520_soc_class_holdout.json")
    with open(output_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # --- Summary: Which classes are actionable? ---
    print("\n" + "=" * 70)
    print("ACTIONABLE CLASSES (MEDIUM tier, holdout > 25%, n >= 10)")
    print("=" * 70)

    actionable = []
    for cls in holdout_soc:
        key = (("MEDIUM", cls))
        if key in output_data["per_class_holdout"]:
            d = output_data["per_class_holdout"][key]
            if d["mean_precision"] > 25 and d["mean_n"] >= 10:
                actionable.append((cls, d["mean_precision"], d["mean_n"], d.get("p_value", "N/A")))

    if actionable:
        for cls, prec, n, p in sorted(actionable, key=lambda x: -x[1]):
            print(f"  {cls:<22} {prec:.1f}% holdout (n={n:.0f}, p={p})")
    else:
        print("  None found. SOC signal may be diffuse across all classes.")

    # Check if any MEDIUM SOC class reaches HIGH-level precision (>= 40%)
    print("\n" + "=" * 70)
    print("HIGH-LEVEL CANDIDATES (MEDIUM tier, holdout >= 40%)")
    print("=" * 70)

    high_candidates = []
    for cls in holdout_soc:
        key = ("MEDIUM", cls)
        if key in output_data["per_class_holdout"]:
            d = output_data["per_class_holdout"][key]
            if d["mean_precision"] >= 40 and d["mean_n"] >= 5:
                high_candidates.append((cls, d["mean_precision"], d["mean_n"]))

    if high_candidates:
        for cls, prec, n in sorted(high_candidates, key=lambda x: -x[1]):
            print(f"  {cls:<22} {prec:.1f}% holdout (n={n:.0f}) → PROMOTE to HIGH?")
    else:
        print("  None found. No single SOC class reaches HIGH-level precision in MEDIUM tier.")


if __name__ == "__main__":
    main()
