#!/usr/bin/env python3
"""
h489: Mechanism-Required ATC Coherent Rules for Psychiatric and Respiratory

h487 found:
- Psychiatric ATC coherent: 35.1% holdout WITH mechanism, 0% WITHOUT
- Respiratory ATC coherent: 38.9% with mechanism, 6.7% without

BUT these splits have small n (1.3 and 3.4/seed for nomech).
Memory note #26: n<5/seed = UNRELIABLE.

This script:
1. Runs full 5-seed holdout with CURRENT code (baseline)
2. Modifies to require mechanism for psychiatric/respiratory ATC coherent
3. Re-runs holdout to measure actual impact
4. Checks GT loss (how many valid predictions lose MEDIUM status)
"""

import json
import sys
import time
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
    originals = {
        "drug_train_freq": dict(predictor.drug_train_freq),
        "drug_to_diseases": dict(predictor.drug_to_diseases),
        "drug_cancer_types": dict(predictor.drug_cancer_types),
        "drug_disease_groups": dict(predictor.drug_disease_groups),
        "train_diseases": list(predictor.train_diseases),
        "train_embeddings": predictor.train_embeddings.copy(),
        "train_disease_categories": dict(predictor.train_disease_categories),
    }

    new_freq: Dict[str, int] = defaultdict(int)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            for drug_id in predictor.ground_truth[disease_id]:
                new_freq[drug_id] += 1
    predictor.drug_train_freq = dict(new_freq)

    new_d2d: Dict[str, Set[str]] = defaultdict(set)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            for drug_id in predictor.ground_truth[disease_id]:
                new_d2d[drug_id].add(disease_name.lower())
    predictor.drug_to_diseases = dict(new_d2d)

    new_cancer: Dict[str, Set[str]] = defaultdict(set)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            cancer_types = extract_cancer_types(disease_name)
            if cancer_types:
                for drug_id in predictor.ground_truth[disease_id]:
                    new_cancer[drug_id].update(cancer_types)
    predictor.drug_cancer_types = dict(new_cancer)

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
    predictor.drug_train_freq = originals["drug_train_freq"]
    predictor.drug_to_diseases = originals["drug_to_diseases"]
    predictor.drug_cancer_types = originals["drug_cancer_types"]
    predictor.drug_disease_groups = originals["drug_disease_groups"]
    predictor.train_diseases = originals["train_diseases"]
    predictor.train_embeddings = originals["train_embeddings"]
    predictor.train_disease_categories = originals["train_disease_categories"]


def evaluate_with_atc_detail(
    predictor: DrugRepurposingPredictor,
    disease_ids: List[str],
    gt_data: Dict[str, List[str]],
) -> Dict:
    """Evaluate with detailed ATC coherent breakdown by category and mechanism."""
    gt_set = set()
    for disease_id, drugs in gt_data.items():
        for drug in drugs:
            if isinstance(drug, str):
                gt_set.add((disease_id, drug))
            elif isinstance(drug, dict):
                gt_set.add((disease_id, drug.get("drug_id") or drug.get("drug")))

    tier_stats: Dict[str, Dict] = defaultdict(lambda: {"hits": 0, "misses": 0})
    # Detailed tracking for atc_coherent rules
    atc_detail: Dict[str, Dict] = defaultdict(lambda: {"hits": 0, "misses": 0})

    for disease_id in disease_ids:
        disease_name = predictor.disease_names.get(disease_id, disease_id)

        try:
            result = predictor.predict(
                disease_name, top_n=30, include_filtered=True
            )
        except Exception:
            continue

        for pred in result.predictions:
            drug_id = pred.drug_id
            is_hit = (disease_id, drug_id) in gt_set
            tier = pred.confidence_tier.name
            rule = pred.category_specific_tier or "default"

            if is_hit:
                tier_stats[tier]["hits"] += 1
            else:
                tier_stats[tier]["misses"] += 1

            # Track ATC coherent detail
            if rule and 'atc_coherent' in rule:
                mech_suffix = "_mech" if pred.mechanism_support else "_nomech"
                detail_key = f"{rule}{mech_suffix}"
                if is_hit:
                    atc_detail[detail_key]["hits"] += 1
                else:
                    atc_detail[detail_key]["misses"] += 1

    # Compute precisions
    tier_results = {}
    for tier, stats in tier_stats.items():
        total = stats["hits"] + stats["misses"]
        precision = stats["hits"] / total if total > 0 else 0
        tier_results[tier] = {
            "hits": stats["hits"],
            "total": total,
            "precision": round(precision * 100, 1),
        }

    atc_results = {}
    for key, stats in atc_detail.items():
        total = stats["hits"] + stats["misses"]
        precision = stats["hits"] / total if total > 0 else 0
        atc_results[key] = {
            "hits": stats["hits"],
            "total": total,
            "precision": round(precision * 100, 1),
        }

    return {"tier_precision": tier_results, "atc_detail": atc_results}


def main() -> None:
    seeds = [42, 123, 456, 789, 2024]
    t0 = time.time()
    output_lines = []

    def log(msg: str = "") -> None:
        print(msg)
        output_lines.append(msg)

    log("=" * 70)
    log("h489: Mechanism-Required ATC Coherent for Psychiatric/Respiratory")
    log("=" * 70)

    log("\nLoading predictor...")
    predictor = DrugRepurposingPredictor()

    gt_path = predictor.reference_dir / "expanded_ground_truth.json"
    with open(gt_path) as f:
        gt_data = json.load(f)

    all_diseases = [
        d for d in predictor.ground_truth if d in predictor.embeddings
    ]
    log(f"Diseases: {len(all_diseases)}, GT pairs: {sum(len(v) for v in gt_data.values())}")

    # ========== PART 1: Baseline holdout (current code) ==========
    log("\n" + "=" * 70)
    log("PART 1: BASELINE HOLDOUT (current code)")
    log("=" * 70)

    baseline_tier = defaultdict(list)
    baseline_atc = defaultdict(list)
    baseline_atc_n = defaultdict(list)

    for seed_idx, seed in enumerate(seeds):
        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        train_set = set(train_ids)
        originals = recompute_gt_structures(predictor, train_set)
        result = evaluate_with_atc_detail(predictor, holdout_ids, gt_data)
        restore_gt_structures(predictor, originals)

        for tier, stats in result["tier_precision"].items():
            baseline_tier[tier].append(stats["precision"])

        for key, stats in result["atc_detail"].items():
            baseline_atc[key].append(stats["precision"])
            baseline_atc_n[key].append(stats["total"])

        log(f"Seed {seed} done ({time.time()-t0:.0f}s)")

    log("\nBaseline tier precision (5-seed holdout):")
    for tier in ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]:
        if tier in baseline_tier:
            vals = baseline_tier[tier]
            log(f"  {tier}: {np.mean(vals):.1f}% ± {np.std(vals):.1f}%")

    log("\nBaseline ATC coherent detail:")
    log(f"{'Rule':<45} {'n/seed':>8} {'Holdout':>10} {'Std':>8}")
    log("-" * 75)
    for key in sorted(baseline_atc.keys()):
        vals = baseline_atc[key]
        ns = baseline_atc_n[key]
        log(f"{key:<45} {np.mean(ns):>8.1f} {np.mean(vals):>9.1f}% {np.std(vals):>7.1f}%")

    # ========== PART 2: Full-data check of affected predictions ==========
    log("\n" + "=" * 70)
    log("PART 2: FULL-DATA AFFECTED PREDICTIONS")
    log("=" * 70)

    # Count predictions that would move from MEDIUM to LOW
    affected_psych = []
    affected_resp = []
    all_preds_count = 0

    for disease_id in all_diseases:
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        try:
            result = predictor.predict(disease_name, top_n=30, include_filtered=True)
        except Exception:
            continue

        for pred in result.predictions:
            all_preds_count += 1
            rule = pred.category_specific_tier or ""
            if 'atc_coherent_psychiatric' in rule and not pred.mechanism_support:
                affected_psych.append((pred.drug_name, disease_name, pred.drug_id, disease_id))
            elif 'atc_coherent_respiratory' in rule and not pred.mechanism_support:
                affected_resp.append((pred.drug_name, disease_name, pred.drug_id, disease_id))

    log(f"\nTotal predictions: {all_preds_count}")
    log(f"Psychiatric nomech to demote: {len(affected_psych)}")
    log(f"Respiratory nomech to demote: {len(affected_resp)}")
    log(f"Total affected: {len(affected_psych) + len(affected_resp)}")

    # Check GT status of affected predictions
    gt_set_full = set()
    for disease_id, drugs in gt_data.items():
        for drug in drugs:
            if isinstance(drug, str):
                gt_set_full.add((disease_id, drug))
            elif isinstance(drug, dict):
                gt_set_full.add((disease_id, drug.get("drug_id") or drug.get("drug")))

    psych_gt = sum(1 for _, _, drug_id, disease_id in affected_psych
                   if (disease_id, drug_id) in gt_set_full)
    resp_gt = sum(1 for _, _, drug_id, disease_id in affected_resp
                  if (disease_id, drug_id) in gt_set_full)

    log(f"\nGT hits among affected:")
    log(f"  Psychiatric: {psych_gt}/{len(affected_psych)} = {psych_gt/len(affected_psych)*100:.1f}%" if affected_psych else "  Psychiatric: 0/0")
    log(f"  Respiratory: {resp_gt}/{len(affected_resp)} = {resp_gt/len(affected_resp)*100:.1f}%" if affected_resp else "  Respiratory: 0/0")

    log(f"\nAffected psychiatric predictions:")
    for drug, disease, drug_id, disease_id in affected_psych:
        is_gt = "GT" if (disease_id, drug_id) in gt_set_full else "  "
        log(f"  [{is_gt}] {drug:25s} → {disease}")

    log(f"\nAffected respiratory predictions:")
    for drug, disease, drug_id, disease_id in affected_resp:
        is_gt = "GT" if (disease_id, drug_id) in gt_set_full else "  "
        log(f"  [{is_gt}] {drug:25s} → {disease}")

    # ========== PART 3: Simulate mechanism requirement ==========
    log("\n" + "=" * 70)
    log("PART 3: SIMULATED MECHANISM REQUIREMENT (what-if analysis)")
    log("=" * 70)

    # Instead of modifying production code, simulate:
    # predictions that were atc_coherent_psychiatric/respiratory without mechanism
    # would become LOW instead of MEDIUM

    # Run holdout again, this time counting the affected predictions as LOW
    sim_tier = defaultdict(list)

    for seed_idx, seed in enumerate(seeds):
        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        train_set = set(train_ids)
        originals = recompute_gt_structures(predictor, train_set)

        # Track with simulated demotion
        gt_set = set()
        for disease_id, drugs in gt_data.items():
            for drug in drugs:
                if isinstance(drug, str):
                    gt_set.add((disease_id, drug))
                elif isinstance(drug, dict):
                    gt_set.add((disease_id, drug.get("drug_id") or drug.get("drug")))

        sim_stats: Dict[str, Dict] = defaultdict(lambda: {"hits": 0, "misses": 0})

        for disease_id in holdout_ids:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            try:
                result = predictor.predict(disease_name, top_n=30, include_filtered=True)
            except Exception:
                continue

            for pred in result.predictions:
                drug_id = pred.drug_id
                is_hit = (disease_id, drug_id) in gt_set
                tier = pred.confidence_tier.name
                rule = pred.category_specific_tier or ""

                # Simulate: demote psychiatric/respiratory nomech to LOW
                if ('atc_coherent_psychiatric' in rule or 'atc_coherent_respiratory' in rule) and not pred.mechanism_support:
                    tier = "LOW"

                if is_hit:
                    sim_stats[tier]["hits"] += 1
                else:
                    sim_stats[tier]["misses"] += 1

        restore_gt_structures(predictor, originals)

        for tier, stats in sim_stats.items():
            total = stats["hits"] + stats["misses"]
            precision = stats["hits"] / total * 100 if total > 0 else 0
            sim_tier[tier].append(precision)

        log(f"Seed {seed} done ({time.time()-t0:.0f}s)")

    log("\nSimulated tier precision (mechanism required for psych/resp):")
    log(f"{'Tier':<10} {'Baseline':>12} {'Simulated':>12} {'Δ':>8}")
    log("-" * 50)
    for tier in ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]:
        base_val = np.mean(baseline_tier.get(tier, [0]))
        sim_val = np.mean(sim_tier.get(tier, [0]))
        delta = sim_val - base_val
        log(f"{tier:<10} {base_val:>11.1f}% {sim_val:>11.1f}% {delta:>+7.1f}pp")

    # ========== SUMMARY ==========
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    total_affected = len(affected_psych) + len(affected_resp)
    total_gt_loss = psych_gt + resp_gt
    log(f"Predictions affected: {total_affected} ({len(affected_psych)} psych + {len(affected_resp)} resp)")
    log(f"GT losses from MEDIUM: {total_gt_loss} ({psych_gt} psych + {resp_gt} resp)")
    log(f"Non-GT demoted: {total_affected - total_gt_loss}")
    log(f"\nFull-data precision of affected predictions: {total_gt_loss}/{total_affected} = {total_gt_loss/total_affected*100:.1f}%" if total_affected > 0 else "")

    log(f"\nKey question: Is the holdout reliable with n={np.mean(baseline_atc_n.get('atc_coherent_psychiatric_nomech', [0])):.1f} psych + {np.mean(baseline_atc_n.get('atc_coherent_respiratory_nomech', [0])):.1f} resp per seed?")
    log(f"Memory note #26: n<5/seed = UNRELIABLE holdout")

    elapsed = time.time() - t0
    log(f"\nDone in {elapsed:.0f}s")

    # Save output
    output_path = Path("data/analysis/h489_mechanism_required_atc.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(output_lines) + "\n")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
