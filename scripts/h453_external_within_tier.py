#!/usr/bin/env python3
"""
h453: External-Signal-Only Within-Tier Ranking

h441 proved kNN-derived metrics are circular for within-tier ranking.
h445 proved TransE is negative within tiers.
h443 proved kNN rank IS the within-tier signal (but is it circular?).

This hypothesis tests whether EXTERNAL-ONLY signals (target overlap, mechanism,
ATC coherence, train_frequency) provide a holdout-validated within-tier ranking
that adds value beyond raw kNN rank.

Key question: Does kNN rank's within-tier gap survive holdout, or does it
collapse like other kNN-derived metrics (h441)?

External signals tested:
1. target_overlap_count (genes shared between drug targets and disease genes)
2. mechanism_support (boolean: overlap > 0)
3. ATC coherence (drug ATC L1 matches expected ATC for disease category)
4. train_frequency (# GT diseases drug treats - NOTE: partially circular)
5. Composite of above

Method:
- For each tier, split predictions into quintiles by each signal
- Compare Q1 (best) vs Q5 (worst) precision
- Run 5-seed holdout validation to check if signals hold
- If composite AUC > 0.6 on holdout, use for within-tier ranking
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from production_predictor import (
    ConfidenceTier,
    DrugRepurposingPredictor,
    DISEASE_HIERARCHY_GROUPS,
    DISEASE_CATEGORY_ATC_MAP,
    extract_cancer_types,
)


def get_atc_l1(predictor: DrugRepurposingPredictor, drug_id: str) -> Optional[str]:
    """Get ATC L1 code for a drug."""
    try:
        mapper = None
        from src.atc_features import ATCMapper
        mapper = ATCMapper()
        codes = mapper.get_atc_codes(drug_id)
        if codes:
            return list(codes)[0][0]  # First character = L1
    except Exception:
        pass
    return None


def is_atc_coherent(atc_l1: Optional[str], category: str) -> Optional[bool]:
    """Check if drug's ATC L1 matches expected ATC for disease category."""
    if atc_l1 is None:
        return None
    expected = DISEASE_CATEGORY_ATC_MAP.get(category.lower(), set())
    if not expected:
        return None
    return atc_l1 in expected


def split_diseases(all_diseases: List[str], seed: int, train_ratio: float = 0.8) -> Tuple[List[str], List[str]]:
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
            category = predictor.categorize_disease(disease_name)
            if category in DISEASE_HIERARCHY_GROUPS:
                for group_name, keywords in DISEASE_HIERARCHY_GROUPS[category].items():
                    if any(kw in disease_name.lower() for kw in keywords):
                        for drug_id in predictor.ground_truth[disease_id]:
                            new_groups[drug_id].add((category, group_name))
    predictor.drug_disease_groups = dict(new_groups)

    # 5. Rebuild kNN index
    predictor.train_diseases = [d for d in train_disease_ids if d in predictor.embeddings]
    predictor.train_embeddings = np.array(
        [predictor.embeddings[d] for d in predictor.train_diseases], dtype=np.float32
    )
    predictor.train_disease_categories = {}
    for d in predictor.train_diseases:
        name = predictor.disease_names.get(d, d)
        predictor.train_disease_categories[d] = predictor.categorize_disease(name)

    return originals


def restore_originals(predictor: DrugRepurposingPredictor, originals: Dict) -> None:
    """Restore original state."""
    predictor.drug_train_freq = originals["drug_train_freq"]
    predictor.drug_to_diseases = originals["drug_to_diseases"]
    predictor.drug_cancer_types = originals["drug_cancer_types"]
    predictor.drug_disease_groups = originals["drug_disease_groups"]
    predictor.train_diseases = originals["train_diseases"]
    predictor.train_embeddings = originals["train_embeddings"]
    predictor.train_disease_categories = originals["train_disease_categories"]


def collect_predictions_with_signals(
    predictor: DrugRepurposingPredictor,
    diseases: List[str],
    atc_cache: Dict[str, Optional[str]],
) -> List[Dict]:
    """Collect all predictions with external signal features."""
    all_preds = []

    for disease_id in diseases:
        if disease_id not in predictor.embeddings:
            continue

        gt_drugs = predictor.ground_truth.get(disease_id, set())
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        category = predictor.categorize_disease(disease_name)

        result = predictor.predict(disease_name, k=20, top_n=30, include_filtered=True)

        for pred in result.predictions:
            # External signal 1: target overlap count
            target_overlap = predictor._get_target_overlap_count(pred.drug_id, disease_id)

            # External signal 2: ATC coherence
            if pred.drug_id not in atc_cache:
                atc_cache[pred.drug_id] = get_atc_l1(predictor, pred.drug_id)
            atc_l1 = atc_cache[pred.drug_id]
            atc_coherent = is_atc_coherent(atc_l1, category)

            all_preds.append({
                "disease_id": disease_id,
                "drug_id": pred.drug_id,
                "drug_name": pred.drug_name,
                "rank": pred.rank,
                "tier": pred.confidence_tier.value,
                "knn_score": pred.knn_score,
                "norm_score": pred.norm_score,
                "is_gt": pred.drug_id in gt_drugs,
                "category": category,
                # External signals
                "target_overlap": target_overlap,
                "mechanism_support": pred.mechanism_support,
                "has_targets": pred.has_targets,
                "train_frequency": pred.train_frequency,
                "atc_coherent": atc_coherent,
                "transe_consilience": pred.transe_consilience,
            })

    return all_preds


def analyze_signal(preds: List[Dict], signal_name: str, continuous: bool = True) -> Dict:
    """Analyze a single signal's within-tier predictive power.

    For continuous signals: split into tertiles (top/mid/bottom third).
    For boolean signals: split into True/False groups.
    """
    n = len(preds)
    if n < 10:
        return {"n": n, "too_small": True}

    gt_count = sum(1 for p in preds if p["is_gt"])
    overall_prec = gt_count / n * 100

    result = {
        "n": n,
        "gt_count": gt_count,
        "overall_precision": round(overall_prec, 2),
    }

    if continuous:
        # Get signal values
        values = [p.get(signal_name, 0) for p in preds]
        if all(v == values[0] for v in values):
            result["no_variance"] = True
            return result

        # Sort by signal (higher = better assumed)
        sorted_preds = sorted(preds, key=lambda x: x.get(signal_name, 0), reverse=True)
        third = n // 3

        top_third = sorted_preds[:third]
        mid_third = sorted_preds[third:2*third]
        bot_third = sorted_preds[2*third:]

        top_prec = sum(1 for p in top_third if p["is_gt"]) / len(top_third) * 100 if top_third else 0
        mid_prec = sum(1 for p in mid_third if p["is_gt"]) / len(mid_third) * 100 if mid_third else 0
        bot_prec = sum(1 for p in bot_third if p["is_gt"]) / len(bot_third) * 100 if bot_third else 0

        result["top_third_prec"] = round(top_prec, 2)
        result["top_third_n"] = len(top_third)
        result["mid_third_prec"] = round(mid_prec, 2)
        result["mid_third_n"] = len(mid_third)
        result["bot_third_prec"] = round(bot_prec, 2)
        result["bot_third_n"] = len(bot_third)
        result["top_vs_bot_gap"] = round(top_prec - bot_prec, 2)
        result["monotonic"] = top_prec >= mid_prec >= bot_prec
    else:
        # Boolean signal
        true_preds = [p for p in preds if p.get(signal_name)]
        false_preds = [p for p in preds if not p.get(signal_name)]
        none_preds = [p for p in preds if p.get(signal_name) is None]

        true_prec = sum(1 for p in true_preds if p["is_gt"]) / len(true_preds) * 100 if true_preds else 0
        false_prec = sum(1 for p in false_preds if p["is_gt"]) / len(false_preds) * 100 if false_preds else 0

        result["true_prec"] = round(true_prec, 2)
        result["true_n"] = len(true_preds)
        result["false_prec"] = round(false_prec, 2)
        result["false_n"] = len(false_preds)
        result["none_n"] = len(none_preds)
        result["gap"] = round(true_prec - false_prec, 2)

    return result


def compute_composite_score(pred: Dict, weights: Dict[str, float]) -> float:
    """Compute composite within-tier score from external signals."""
    score = 0.0

    # target_overlap (continuous, higher = better)
    if "target_overlap" in weights:
        score += weights["target_overlap"] * min(pred.get("target_overlap", 0), 5)  # cap at 5

    # mechanism_support (boolean)
    if "mechanism" in weights:
        score += weights["mechanism"] * (1.0 if pred.get("mechanism_support") else 0.0)

    # ATC coherence (boolean, None = neutral)
    if "atc" in weights:
        atc_val = pred.get("atc_coherent")
        if atc_val is True:
            score += weights["atc"] * 1.0
        elif atc_val is False:
            score += weights["atc"] * -0.5  # penalize incoherence

    # train_frequency (continuous, higher = better)
    if "freq" in weights:
        score += weights["freq"] * min(pred.get("train_frequency", 0), 20) / 20.0

    # kNN rank (lower = better, invert for composite)
    if "rank" in weights:
        score += weights["rank"] * (1.0 - pred.get("rank", 20) / 30.0)

    return score


def main():
    print("=" * 70)
    print("h453: External-Signal-Only Within-Tier Ranking")
    print("=" * 70)

    # Load predictor
    print("\nLoading predictor...")
    t0 = time.time()
    predictor = DrugRepurposingPredictor()
    print(f"Loaded in {time.time() - t0:.1f}s")

    # Get all diseases with GT and embeddings
    diseases_with_gt = [
        d for d in predictor.ground_truth
        if d in predictor.embeddings
    ]
    print(f"Diseases with GT and embeddings: {len(diseases_with_gt)}")

    # ATC cache
    atc_cache: Dict[str, Optional[str]] = {}

    # ===== PART 1: Full-data signal analysis =====
    print("\n" + "=" * 70)
    print("PART 1: Full-Data External Signal Analysis (Within-Tier)")
    print("=" * 70)

    t0 = time.time()
    all_preds = collect_predictions_with_signals(predictor, diseases_with_gt, atc_cache)
    print(f"Collected {len(all_preds)} predictions in {time.time() - t0:.1f}s")

    # Group by tier
    tier_preds: Dict[str, List[Dict]] = defaultdict(list)
    for p in all_preds:
        tier_preds[p["tier"]].append(p)

    tier_order = ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]

    continuous_signals = ["target_overlap", "train_frequency", "rank"]
    boolean_signals = ["mechanism_support", "atc_coherent", "transe_consilience"]

    full_data_results = {}

    for tier_name in tier_order:
        if tier_name not in tier_preds:
            continue
        preds = tier_preds[tier_name]
        n = len(preds)
        gt_count = sum(1 for p in preds if p["is_gt"])

        print(f"\n--- {tier_name} (n={n}, GT={gt_count}, prec={gt_count/n*100:.1f}%) ---")

        tier_results = {}

        # Continuous signals
        for sig in continuous_signals:
            result = analyze_signal(preds, sig, continuous=True)
            tier_results[sig] = result
            if "top_vs_bot_gap" in result:
                mono = "✓" if result.get("monotonic") else "✗"
                print(f"  {sig:>20s}: top={result['top_third_prec']:>5.1f}% mid={result['mid_third_prec']:>5.1f}% bot={result['bot_third_prec']:>5.1f}% gap={result['top_vs_bot_gap']:>+6.1f}pp mono={mono}")
            elif result.get("no_variance"):
                print(f"  {sig:>20s}: NO VARIANCE")

        # Boolean signals
        for sig in boolean_signals:
            result = analyze_signal(preds, sig, continuous=False)
            tier_results[sig] = result
            if "gap" in result:
                print(f"  {sig:>20s}: True={result['true_prec']:>5.1f}% (n={result['true_n']}) False={result['false_prec']:>5.1f}% (n={result['false_n']}) gap={result['gap']:>+6.1f}pp")

        # Composite tests (external only, no kNN rank)
        weight_configs = {
            "external_only": {"target_overlap": 2.0, "mechanism": 1.0, "atc": 0.5, "freq": 1.0},
            "target_heavy": {"target_overlap": 3.0, "mechanism": 1.0},
            "freq_heavy": {"freq": 3.0, "mechanism": 1.0},
            "rank_only": {"rank": 1.0},
            "rank_plus_external": {"rank": 1.0, "target_overlap": 1.0, "mechanism": 0.5, "atc": 0.3},
        }

        print(f"  --- Composite Scores ---")
        for config_name, weights in weight_configs.items():
            scored = [(compute_composite_score(p, weights), p) for p in preds]
            scored.sort(key=lambda x: x[0], reverse=True)

            third = n // 3
            top = scored[:third]
            bot = scored[2*third:]

            top_prec = sum(1 for _, p in top if p["is_gt"]) / len(top) * 100 if top else 0
            bot_prec = sum(1 for _, p in bot if p["is_gt"]) / len(bot) * 100 if bot else 0
            gap = top_prec - bot_prec

            print(f"  {config_name:>24s}: top={top_prec:>5.1f}% bot={bot_prec:>5.1f}% gap={gap:>+6.1f}pp")
            tier_results[f"composite_{config_name}"] = {
                "top_third_prec": round(top_prec, 2),
                "bot_third_prec": round(bot_prec, 2),
                "gap": round(gap, 2),
            }

        full_data_results[tier_name] = tier_results

    # ===== PART 2: Holdout Validation =====
    print("\n" + "=" * 70)
    print("PART 2: Holdout Validation (5-seed)")
    print("=" * 70)

    seeds = [42, 123, 456, 789, 2024]

    # Track per-tier, per-signal holdout gaps
    holdout_results: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    # holdout_results[tier][signal] = [gap_seed1, gap_seed2, ...]

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        train_diseases, holdout_diseases = split_diseases(diseases_with_gt, seed)
        train_set = set(train_diseases)

        originals = recompute_gt_structures(predictor, train_set)

        # Collect holdout predictions with signals
        holdout_preds = collect_predictions_with_signals(predictor, holdout_diseases, atc_cache)

        restore_originals(predictor, originals)

        # Analyze per tier
        tier_holdout: Dict[str, List[Dict]] = defaultdict(list)
        for p in holdout_preds:
            tier_holdout[p["tier"]].append(p)

        for tier_name in tier_order:
            if tier_name not in tier_holdout:
                continue
            preds = tier_holdout[tier_name]
            n = len(preds)
            if n < 20:
                continue

            gt_count = sum(1 for p in preds if p["is_gt"])
            overall_prec = gt_count / n * 100

            # Continuous signals - tertile gap
            for sig in continuous_signals:
                values = [p.get(sig, 0) for p in preds]
                if all(v == values[0] for v in values):
                    continue
                sorted_preds = sorted(preds, key=lambda x: x.get(sig, 0), reverse=True)
                third = n // 3
                top_prec = sum(1 for p in sorted_preds[:third] if p["is_gt"]) / third * 100 if third > 0 else 0
                bot_prec = sum(1 for p in sorted_preds[2*third:] if p["is_gt"]) / len(sorted_preds[2*third:]) * 100 if len(sorted_preds) > 2*third else 0
                holdout_results[tier_name][sig].append(top_prec - bot_prec)

            # Boolean signals - True vs False gap
            for sig in boolean_signals:
                true_preds = [p for p in preds if p.get(sig) is True]
                false_preds = [p for p in preds if p.get(sig) is False]
                if len(true_preds) >= 5 and len(false_preds) >= 5:
                    true_prec = sum(1 for p in true_preds if p["is_gt"]) / len(true_preds) * 100
                    false_prec = sum(1 for p in false_preds if p["is_gt"]) / len(false_preds) * 100
                    holdout_results[tier_name][sig].append(true_prec - false_prec)

            # Composite scores
            weight_configs = {
                "external_only": {"target_overlap": 2.0, "mechanism": 1.0, "atc": 0.5, "freq": 1.0},
                "target_heavy": {"target_overlap": 3.0, "mechanism": 1.0},
                "rank_only": {"rank": 1.0},
                "rank_plus_external": {"rank": 1.0, "target_overlap": 1.0, "mechanism": 0.5, "atc": 0.3},
            }

            for config_name, weights in weight_configs.items():
                scored = [(compute_composite_score(p, weights), p) for p in preds]
                scored.sort(key=lambda x: x[0], reverse=True)
                third = n // 3
                top_prec = sum(1 for _, p in scored[:third] if p["is_gt"]) / third * 100 if third > 0 else 0
                bot_prec = sum(1 for _, p in scored[2*third:] if p["is_gt"]) / len(scored[2*third:]) * 100 if len(scored) > 2*third else 0
                holdout_results[tier_name][f"composite_{config_name}"].append(top_prec - bot_prec)

            # Print seed summary
            signals_str = []
            for sig in continuous_signals + boolean_signals:
                if holdout_results[tier_name].get(sig):
                    signals_str.append(f"{sig}={holdout_results[tier_name][sig][-1]:+.1f}")
            print(f"  {tier_name}: prec={overall_prec:.1f}%, " + ", ".join(signals_str))

    # ===== HOLDOUT SUMMARY =====
    print("\n" + "=" * 70)
    print("HOLDOUT SUMMARY (5-seed mean ± std)")
    print("=" * 70)

    all_signals = continuous_signals + boolean_signals + [
        f"composite_{c}" for c in ["external_only", "target_heavy", "rank_only", "rank_plus_external"]
    ]

    summary_data = {}

    for tier_name in tier_order:
        if tier_name not in holdout_results:
            continue
        print(f"\n--- {tier_name} ---")
        tier_summary = {}
        for sig in all_signals:
            gaps = holdout_results[tier_name].get(sig, [])
            if len(gaps) >= 3:
                mean_gap = np.mean(gaps)
                std_gap = np.std(gaps)
                # Significance: gap > 2*std = probably real
                significant = abs(mean_gap) > 2 * std_gap and abs(mean_gap) > 3.0
                sig_marker = " ***" if significant else ""
                print(f"  {sig:>28s}: {mean_gap:>+7.1f} ± {std_gap:>4.1f}pp{sig_marker}")
                tier_summary[sig] = {
                    "mean": round(float(mean_gap), 2),
                    "std": round(float(std_gap), 2),
                    "n_seeds": len(gaps),
                    "significant": significant,
                }
        summary_data[tier_name] = tier_summary

    # ===== KEY FINDINGS =====
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # For each tier, find best external-only signal
    for tier_name in tier_order:
        if tier_name not in summary_data:
            continue
        ts = summary_data[tier_name]

        # External-only signals (exclude rank_only and composites with rank)
        external_only = {k: v for k, v in ts.items()
                        if k not in ["rank", "composite_rank_only", "composite_rank_plus_external"]}

        if not external_only:
            continue

        best_ext = max(external_only.items(), key=lambda x: x[1]["mean"])
        rank_gap = ts.get("rank", {}).get("mean", 0)
        rank_plus = ts.get("composite_rank_plus_external", {}).get("mean", 0)

        print(f"\n{tier_name}:")
        print(f"  Best external-only signal: {best_ext[0]} ({best_ext[1]['mean']:+.1f} ± {best_ext[1]['std']:.1f}pp)")
        print(f"  kNN rank alone:            {rank_gap:+.1f}pp")
        print(f"  rank + external composite: {rank_plus:+.1f}pp")

        additive = rank_plus - rank_gap
        print(f"  Additive value of external: {additive:+.1f}pp")

        if best_ext[1].get("significant"):
            print(f"  ==> {best_ext[0]} is SIGNIFICANT on holdout!")
        else:
            print(f"  ==> No external signal is significant on holdout")

    # Save results
    results = {
        "hypothesis": "h453",
        "title": "External-Signal-Only Within-Tier Ranking",
        "full_data": {tier: {sig: analyze_signal(tier_preds[tier], sig, sig in continuous_signals)
                             for sig in continuous_signals + boolean_signals}
                      for tier in tier_order if tier in tier_preds},
        "holdout_summary": summary_data,
    }

    output_path = Path(__file__).parent.parent / "data" / "analysis" / "h453_external_within_tier.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
