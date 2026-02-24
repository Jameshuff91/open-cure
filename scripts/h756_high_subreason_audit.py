#!/usr/bin/env python3
"""
h756: HIGH Tier Drug-Class Sub-Reason Audit

Evaluate each HIGH sub-reason (category_specific_tier) via holdout to find
which ones exceed GOLDEN threshold (>78% holdout, n>10/seed).
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from production_predictor import (
    DrugRepurposingPredictor,
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
    from collections import defaultdict as dd
    from production_predictor import extract_cancer_types, DISEASE_HIERARCHY_GROUPS, HIERARCHY_EXCLUSIONS

    originals = {
        "drug_train_freq": dict(predictor.drug_train_freq),
        "drug_to_diseases": dict(predictor.drug_to_diseases),
        "drug_cancer_types": dict(predictor.drug_cancer_types),
        "drug_disease_groups": dict(predictor.drug_disease_groups),
        "train_diseases": list(predictor.train_diseases),
        "train_embeddings": predictor.train_embeddings.copy(),
        "train_disease_categories": dict(predictor.train_disease_categories),
    }

    new_freq: Dict[str, int] = dd(int)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            for drug_id in predictor.ground_truth[disease_id]:
                new_freq[drug_id] += 1
    predictor.drug_train_freq = dict(new_freq)

    new_d2d: Dict[str, Set[str]] = dd(set)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            for drug_id in predictor.ground_truth[disease_id]:
                new_d2d[drug_id].add(disease_name)
    predictor.drug_to_diseases = dict(new_d2d)

    new_cancer: Dict[str, Set[str]] = dd(set)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            ctypes = extract_cancer_types(disease_name)
            if ctypes:
                for drug_id in predictor.ground_truth[disease_id]:
                    new_cancer[drug_id].update(ctypes)
    predictor.drug_cancer_types = dict(new_cancer)

    new_groups: Dict[str, Set[str]] = dd(set)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            disease_name = predictor.disease_names.get(disease_id, disease_id).lower()
            for group_name, data in DISEASE_HIERARCHY_GROUPS.items():
                if group_name in HIERARCHY_EXCLUSIONS:
                    exc_patterns = HIERARCHY_EXCLUSIONS[group_name]
                    if any(p in disease_name for p in exc_patterns):
                        continue
                parent = data.get("parent", "")
                subtypes = data.get("subtypes", [])
                all_patterns = [parent.lower()] + [s.lower() for s in subtypes]
                if any(p in disease_name for p in all_patterns if p):
                    for drug_id in predictor.ground_truth[disease_id]:
                        new_groups[drug_id].add(group_name)
    predictor.drug_disease_groups = dict(new_groups)

    predictor.train_diseases = [d for d in train_disease_ids if d in predictor.embeddings]
    if len(predictor.train_diseases) > 0:
        predictor.train_embeddings = np.array(
            [predictor.embeddings[d] for d in predictor.train_diseases]
        )

    predictor.train_disease_categories = {}
    for d in predictor.train_diseases:
        name = predictor.disease_names.get(d, d)
        predictor.train_disease_categories[d] = predictor.categorize_disease(name)

    return originals


def restore_gt_structures(predictor: DrugRepurposingPredictor, originals: Dict) -> None:
    predictor.drug_train_freq = originals["drug_train_freq"]
    predictor.drug_to_diseases = originals["drug_to_diseases"]
    predictor.drug_cancer_types = originals["drug_cancer_types"]
    predictor.drug_disease_groups = originals["drug_disease_groups"]
    predictor.train_diseases = originals["train_diseases"]
    predictor.train_embeddings = originals["train_embeddings"]
    predictor.train_disease_categories = originals["train_disease_categories"]


def main() -> None:
    seeds = [42, 123, 456, 789, 2024]

    print("=" * 70)
    print("h756: HIGH Tier Sub-Reason Audit — Finding GOLDEN-Worthy Rules")
    print("=" * 70)

    print("\nLoading predictor...")
    predictor = DrugRepurposingPredictor()

    gt_path = predictor.reference_dir / "expanded_ground_truth.json"
    with open(gt_path) as f:
        gt_data = json.load(f)
    print(f"Loaded {len(gt_data)} GT diseases, {sum(len(v) for v in gt_data.values())} pairs")

    gt_set = set()
    for disease_id, drugs in gt_data.items():
        for drug in drugs:
            if isinstance(drug, str):
                gt_set.add((disease_id, drug))
            elif isinstance(drug, dict):
                gt_set.add((disease_id, drug.get("drug_id") or drug.get("drug")))

    all_diseases = [d for d in predictor.ground_truth if d in predictor.embeddings]
    print(f"Diseases with GT + embeddings: {len(all_diseases)}")

    # First: full-data baseline for HIGH sub-reasons
    print("\n--- FULL-DATA BASELINE ---")
    full_sr: Dict[str, Dict] = defaultdict(lambda: {"hits": 0, "misses": 0})
    full_tier: Dict[str, Dict] = defaultdict(lambda: {"hits": 0, "misses": 0})

    for disease_id in all_diseases:
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        try:
            result = predictor.predict(disease_name, top_n=30, include_filtered=True)
        except Exception:
            continue
        for pred in result.predictions:
            is_hit = (disease_id, pred.drug_id) in gt_set
            tier = pred.confidence_tier.name
            rule = pred.category_specific_tier or "default"
            key = "hits" if is_hit else "misses"
            full_tier[tier][key] += 1
            if tier == "HIGH":
                full_sr[rule][key] += 1

    print("\nTier precision (full-data):")
    for tier in ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]:
        s = full_tier[tier]
        t = s["hits"] + s["misses"]
        p = s["hits"] / t * 100 if t > 0 else 0
        print(f"  {tier}: {p:.1f}% ({s['hits']}/{t})")

    print("\nHIGH sub-reasons (full-data):")
    for rule in sorted(full_sr, key=lambda r: -(full_sr[r]["hits"] + full_sr[r]["misses"])):
        s = full_sr[rule]
        t = s["hits"] + s["misses"]
        p = s["hits"] / t * 100 if t > 0 else 0
        print(f"  {rule:<50s} {p:6.1f}% ({s['hits']:>3}/{t:>4})")

    # Holdout evaluation
    all_holdout_sr: Dict[str, List[float]] = defaultdict(list)
    all_holdout_sr_n: Dict[str, List[int]] = defaultdict(list)
    all_holdout_tier: Dict[str, List[float]] = defaultdict(list)

    for seed_idx, seed in enumerate(seeds):
        print(f"\n--- SEED {seed} ({seed_idx+1}/{len(seeds)}) ---")
        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        train_set = set(train_ids)

        originals = recompute_gt_structures(predictor, train_set)

        seed_sr: Dict[str, Dict] = defaultdict(lambda: {"hits": 0, "misses": 0})
        seed_tier: Dict[str, Dict] = defaultdict(lambda: {"hits": 0, "misses": 0})
        n_diseases = 0

        for disease_id in holdout_ids:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            try:
                result = predictor.predict(disease_name, top_n=30, include_filtered=True)
            except Exception:
                continue
            n_diseases += 1
            for pred in result.predictions:
                is_hit = (disease_id, pred.drug_id) in gt_set
                tier = pred.confidence_tier.name
                rule = pred.category_specific_tier or "default"
                key = "hits" if is_hit else "misses"
                seed_tier[tier][key] += 1
                if tier == "HIGH":
                    seed_sr[rule][key] += 1

        print(f"  Evaluated {n_diseases} holdout diseases")

        for tier in ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]:
            s = seed_tier[tier]
            t = s["hits"] + s["misses"]
            p = s["hits"] / t * 100 if t > 0 else 0
            all_holdout_tier[tier].append(p)

        for rule in set(list(full_sr.keys()) + list(seed_sr.keys())):
            s = seed_sr[rule]
            t = s["hits"] + s["misses"]
            p = s["hits"] / t * 100 if t > 0 else 0
            all_holdout_sr[rule].append(p)
            all_holdout_sr_n[rule].append(t)

        restore_gt_structures(predictor, originals)

    # Aggregate results
    print("\n" + "=" * 70)
    print("AGGREGATE HOLDOUT RESULTS (mean ± std across 5 seeds)")
    print("=" * 70)

    print("\n--- TIER PRECISION ---")
    for tier in ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]:
        vals = all_holdout_tier[tier]
        fs = full_tier[tier]
        ft = fs["hits"] + fs["misses"]
        fp = fs["hits"] / ft * 100 if ft > 0 else 0
        hm = np.mean(vals)
        hs = np.std(vals)
        print(f"  {tier:8s}: Full={fp:5.1f}% | Holdout={hm:5.1f}% ± {hs:4.1f}%")

    print("\n--- HIGH SUB-REASONS: FULL-DATA vs HOLDOUT ---")
    print(f"{'Sub-reason':<50s} {'Full%':>6s} {'Hold%':>6s} {'±std':>5s} {'n/seed':>7s} {'GOLDEN?':>8s}")
    print("-" * 90)

    results = []
    for rule in sorted(full_sr, key=lambda r: -(full_sr[r]["hits"] + full_sr[r]["misses"])):
        fs = full_sr[rule]
        ft = fs["hits"] + fs["misses"]
        fp = fs["hits"] / ft * 100 if ft > 0 else 0

        if rule in all_holdout_sr:
            hm = np.mean(all_holdout_sr[rule])
            hs = np.std(all_holdout_sr[rule])
            mn = np.mean(all_holdout_sr_n[rule])
        else:
            hm, hs, mn = 0, 0, 0

        golden = "YES" if hm >= 78.0 and mn >= 10 else ("maybe" if hm >= 70.0 and mn >= 10 else "no")
        print(f"  {rule:<48s} {fp:6.1f}% {hm:6.1f}% {hs:5.1f}% {mn:7.1f} {golden:>8s}")
        results.append({
            "rule": rule,
            "full_precision": round(fp, 1),
            "holdout_mean": round(hm, 1),
            "holdout_std": round(hs, 1),
            "mean_n_per_seed": round(mn, 1),
            "golden_candidate": golden,
        })

    # Also check ALL sub-reasons across all tiers for completeness
    print("\n--- ALL TIERS: SUB-REASONS EXCEEDING GOLDEN THRESHOLD ---")
    # We need to re-run to get all tiers - but for now just report HIGH

    # Save results
    output = {
        "hypothesis": "h756",
        "description": "HIGH tier sub-reason audit for GOLDEN promotion candidates",
        "high_sub_reasons": results,
        "tier_holdout": {
            tier: {
                "holdout_mean": round(np.mean(all_holdout_tier[tier]), 1),
                "holdout_std": round(np.std(all_holdout_tier[tier]), 1),
            }
            for tier in all_holdout_tier
        },
    }

    output_path = Path("data/analysis/h756_high_subreason_audit.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary
    golden_candidates = [r for r in results if r["golden_candidate"] == "YES"]
    maybe_candidates = [r for r in results if r["golden_candidate"] == "maybe"]
    print(f"\n--- SUMMARY ---")
    print(f"GOLDEN candidates (>=78% holdout, n>=10/seed): {len(golden_candidates)}")
    for r in golden_candidates:
        print(f"  {r['rule']}: {r['holdout_mean']}% ± {r['holdout_std']}% (n={r['mean_n_per_seed']})")
    print(f"Maybe candidates (>=70% holdout, n>=10/seed): {len(maybe_candidates)}")
    for r in maybe_candidates:
        print(f"  {r['rule']}: {r['holdout_mean']}% ± {r['holdout_std']}% (n={r['mean_n_per_seed']})")


if __name__ == "__main__":
    main()
