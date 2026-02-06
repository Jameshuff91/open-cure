#!/usr/bin/env python3
"""
h497: GOLDEN Standard Rule Holdout Validation

Tests whether non-hierarchy GOLDEN predictions ("standard" rule: freq>=10 + mechanism + tier1)
have significantly lower holdout precision than hierarchy-based GOLDEN predictions.

h479 found 34% of GOLDEN predictions come from the 'standard' rule, with higher false
positive risk (4/6 uncertain novels were standard-rule GOLDEN).
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from production_predictor import (
    ConfidenceTier,
    DrugRepurposingPredictor,
    DISEASE_HIERARCHY_GROUPS,
    HIERARCHY_EXCLUSIONS,
    extract_cancer_types,
)


def split_diseases(all_diseases: List[str], seed: int, train_ratio: float = 0.8):
    rng = np.random.RandomState(seed)
    shuffled = list(all_diseases)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def recompute_gt_structures(predictor: DrugRepurposingPredictor, train_disease_ids: Set[str]) -> Dict:
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

    predictor.train_diseases = [d for d in train_disease_ids if d in predictor.embeddings]
    predictor.train_embeddings = np.array(
        [predictor.embeddings[d] for d in predictor.train_diseases], dtype=np.float32
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


def classify_golden_rule(cat_specific: Optional[str]) -> str:
    """Classify a GOLDEN prediction's category_specific_tier into rule types."""
    if cat_specific is None:
        return 'standard'
    s = str(cat_specific)
    if 'hierarchy' in s:
        return 'hierarchy'
    if 'statin_cv' in s:
        return 'statin_cv'
    if 'comp_to_base' in s:
        return 'comp_to_base'
    if 'overlap' in s:
        return 'overlap_promotion'
    return f'category_rescue_{s}'


def main() -> None:
    seeds = [42, 123, 456, 789, 2024]
    print("=" * 70)
    print("h497: GOLDEN Standard vs Hierarchy Holdout Validation")
    print("=" * 70)
    print(f"DEBUG: classify_golden_rule(None) = {classify_golden_rule(None)}")
    print(f"DEBUG: script version = 2")

    print("\nLoading predictor...")
    predictor = DrugRepurposingPredictor()

    gt_path = predictor.reference_dir / "expanded_ground_truth.json"
    with open(gt_path) as f:
        gt_data = json.load(f)
    print(f"Loaded {len(gt_data)} GT diseases, {sum(len(v) for v in gt_data.values())} pairs")

    all_diseases = [d for d in predictor.ground_truth if d in predictor.embeddings]
    print(f"Diseases with GT + embeddings: {len(all_diseases)}")

    # Build full GT set
    gt_set_full: Set[Tuple[str, str]] = set()
    for did, drugs in gt_data.items():
        for drug in drugs:
            if isinstance(drug, str):
                gt_set_full.add((did, drug))
            elif isinstance(drug, dict):
                gt_set_full.add((did, drug.get("drug_id") or drug.get("drug")))

    # ---- FULL-DATA BASELINE ----
    print("\n--- FULL-DATA BASELINE ---")
    full_stats: Dict[str, Dict] = defaultdict(lambda: {"hits": 0, "total": 0})
    full_examples: Dict[str, list] = defaultdict(list)

    debug_done = False
    for disease_id in all_diseases:
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        try:
            result = predictor.predict(disease_name, top_n=30, include_filtered=True)
        except Exception:
            continue

        if 'atopic' in disease_name.lower() and not debug_done:
            debug_done = True
            print(f"\nDEBUG: atopic dermatitis predictions:")
            for p in result.predictions[:5]:
                print(f"  tier={p.confidence_tier.value} cat_spec={repr(p.category_specific_tier)} drug={p.drug_name}")
            print()

        for pred in result.predictions:
            if pred.confidence_tier != ConfidenceTier.GOLDEN:
                continue
            rule_type = classify_golden_rule(pred.category_specific_tier)
            is_hit = (disease_id, pred.drug_id) in gt_set_full
            full_stats[rule_type]["hits"] += int(is_hit)
            full_stats[rule_type]["total"] += 1
            if len(full_examples[rule_type]) < 3:
                full_examples[rule_type].append({
                    'drug': pred.drug_name, 'disease': disease_name,
                    'is_hit': is_hit, 'freq': pred.train_frequency,
                    'rank': pred.rank, 'category': pred.category,
                })

    total_full = sum(s['total'] for s in full_stats.values())
    # Debug: check if standard found
    print(f"\nDEBUG: full_stats keys = {list(full_stats.keys())}")
    print(f"DEBUG: 'standard' in full_stats = {'standard' in full_stats}")
    if 'standard' in full_stats:
        print(f"DEBUG: standard = {full_stats['standard']}")
    print(f"Total GOLDEN (full data): {total_full}")
    for rt in sorted(full_stats.keys(), key=lambda x: -full_stats[x]['total']):
        s = full_stats[rt]
        prec = s['hits'] / s['total'] * 100 if s['total'] > 0 else 0
        pct = s['total'] / total_full * 100
        print(f"  {rt}: {prec:.1f}% ({s['hits']}/{s['total']}, {pct:.0f}% of GOLDEN)")
        for ex in full_examples[rt]:
            hit_str = "HIT" if ex['is_hit'] else "MISS"
            print(f"    [{hit_str}] {ex['drug']} → {ex['disease']} (freq={ex['freq']}, rank={ex['rank']}, cat={ex['category']})")

    # ---- HOLDOUT EVALUATION ----
    print("\n--- HOLDOUT EVALUATION (5 seeds) ---")

    holdout_by_rule: Dict[str, List[float]] = defaultdict(list)
    holdout_n_by_rule: Dict[str, List[int]] = defaultdict(list)
    holdout_golden_total: List[float] = []

    # Standard GOLDEN breakdown
    std_by_rank: Dict[str, List[float]] = defaultdict(list)
    std_n_by_rank: Dict[str, List[int]] = defaultdict(list)
    std_by_cat: Dict[str, List[float]] = defaultdict(list)
    std_n_by_cat: Dict[str, List[int]] = defaultdict(list)

    # Track misses
    all_misses: Dict[str, int] = defaultdict(int)
    miss_info: Dict[str, dict] = {}

    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"SEED {seed} ({seed_idx+1}/{len(seeds)})")
        print(f"{'='*60}")

        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        train_set = set(train_ids)
        print(f"Train: {len(train_ids)}, Holdout: {len(holdout_ids)}")

        originals = recompute_gt_structures(predictor, train_set)

        # Build holdout GT set
        holdout_gt: Set[Tuple[str, str]] = set()
        for did in holdout_ids:
            if did in gt_data:
                for drug in gt_data[did]:
                    if isinstance(drug, str):
                        holdout_gt.add((did, drug))
                    elif isinstance(drug, dict):
                        holdout_gt.add((did, drug.get("drug_id") or drug.get("drug")))

        # Count GOLDEN by rule type
        seed_stats: Dict[str, Dict] = defaultdict(lambda: {"hits": 0, "total": 0})
        seed_std_rank: Dict[str, Dict] = defaultdict(lambda: {"hits": 0, "total": 0})
        seed_std_cat: Dict[str, Dict] = defaultdict(lambda: {"hits": 0, "total": 0})
        seed_golden_hits = 0
        seed_golden_total = 0

        for disease_id in holdout_ids:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            try:
                result = predictor.predict(disease_name, top_n=30, include_filtered=True)
            except Exception:
                continue

            for pred in result.predictions:
                if pred.confidence_tier != ConfidenceTier.GOLDEN:
                    continue

                rule_type = classify_golden_rule(pred.category_specific_tier)
                is_hit = (disease_id, pred.drug_id) in holdout_gt

                seed_stats[rule_type]["hits"] += int(is_hit)
                seed_stats[rule_type]["total"] += 1
                seed_golden_hits += int(is_hit)
                seed_golden_total += 1

                if rule_type == 'standard':
                    if pred.rank <= 5:
                        bucket = 'R1-5'
                    elif pred.rank <= 10:
                        bucket = 'R6-10'
                    elif pred.rank <= 15:
                        bucket = 'R11-15'
                    else:
                        bucket = 'R16-20'
                    seed_std_rank[bucket]["hits"] += int(is_hit)
                    seed_std_rank[bucket]["total"] += 1
                    seed_std_cat[pred.category]["hits"] += int(is_hit)
                    seed_std_cat[pred.category]["total"] += 1

                    if not is_hit:
                        key = f"{pred.drug_name} → {disease_name}"
                        all_misses[key] += 1
                        miss_info[key] = {
                            'drug': pred.drug_name, 'disease': disease_name,
                            'category': pred.category, 'freq': pred.train_frequency,
                            'rank': pred.rank,
                        }

        golden_prec = seed_golden_hits / seed_golden_total * 100 if seed_golden_total > 0 else 0
        holdout_golden_total.append(golden_prec)

        print(f"\nOverall GOLDEN: {golden_prec:.1f}% ({seed_golden_hits}/{seed_golden_total})")

        for rule_type in sorted(seed_stats.keys(), key=lambda x: -seed_stats[x]['total']):
            s = seed_stats[rule_type]
            prec = s['hits'] / s['total'] * 100 if s['total'] > 0 else 0
            holdout_by_rule[rule_type].append(prec)
            holdout_n_by_rule[rule_type].append(s['total'])
            print(f"  {rule_type}: {prec:.1f}% ({s['hits']}/{s['total']})")

        # Fill in 0 for rules that didn't appear this seed
        all_rule_types = set(full_stats.keys()) | set(seed_stats.keys())
        for rule_type in all_rule_types:
            if rule_type not in seed_stats:
                holdout_by_rule[rule_type].append(0.0)
                holdout_n_by_rule[rule_type].append(0)

        # Standard GOLDEN by rank
        for bucket in ['R1-5', 'R6-10', 'R11-15', 'R16-20']:
            if bucket in seed_std_rank:
                s = seed_std_rank[bucket]
                prec = s['hits'] / s['total'] * 100 if s['total'] > 0 else 0
                std_by_rank[bucket].append(prec)
                std_n_by_rank[bucket].append(s['total'])

        # Standard GOLDEN by category
        for cat in seed_std_cat:
            s = seed_std_cat[cat]
            prec = s['hits'] / s['total'] * 100 if s['total'] > 0 else 0
            std_by_cat[cat].append(prec)
            std_n_by_cat[cat].append(s['total'])

        if seed_std_rank:
            print(f"\n  Standard GOLDEN by rank:")
            for bucket in ['R1-5', 'R6-10', 'R11-15', 'R16-20']:
                if bucket in seed_std_rank:
                    s = seed_std_rank[bucket]
                    prec = s['hits'] / s['total'] * 100 if s['total'] > 0 else 0
                    print(f"    {bucket}: {prec:.1f}% ({s['hits']}/{s['total']})")

        if seed_std_cat:
            print(f"\n  Standard GOLDEN by category:")
            for cat in sorted(seed_std_cat.keys(), key=lambda x: -seed_std_cat[x]['total']):
                s = seed_std_cat[cat]
                prec = s['hits'] / s['total'] * 100 if s['total'] > 0 else 0
                print(f"    {cat}: {prec:.1f}% ({s['hits']}/{s['total']})")

        restore_gt_structures(predictor, originals)

    # ---- AGGREGATE RESULTS ----
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS (5-seed mean ± std)")
    print("=" * 70)

    print(f"\nOverall GOLDEN holdout: {np.mean(holdout_golden_total):.1f}% ± {np.std(holdout_golden_total):.1f}%")

    print(f"\n{'Rule':<30} {'Full-Data':>10} {'Holdout':>12} {'±std':>6} {'n/seed':>8} {'Gap':>8}")
    print("-" * 80)

    all_rules = set(full_stats.keys()) | set(holdout_by_rule.keys())
    for rt in sorted(all_rules, key=lambda x: -full_stats.get(x, {}).get('total', 0)):
        full_prec = full_stats[rt]['hits'] / full_stats[rt]['total'] * 100 if rt in full_stats and full_stats[rt]['total'] > 0 else 0.0
        full_n = full_stats.get(rt, {}).get('total', 0)

        if rt in holdout_by_rule and holdout_by_rule[rt]:
            vals = holdout_by_rule[rt]
            ns = holdout_n_by_rule[rt]
            h_mean = np.mean(vals)
            h_std = np.std(vals)
            h_n = np.mean(ns)
            gap = full_prec - h_mean
            print(f"  {rt:<28} {full_prec:>8.1f}% {h_mean:>10.1f}% {h_std:>5.1f} {h_n:>7.1f} {gap:>+7.1f}pp")
        else:
            print(f"  {rt:<28} {full_prec:>8.1f}%      N/A           {full_n:>7}      N/A")

    # Standard GOLDEN rank analysis
    if std_by_rank:
        print("\nStandard GOLDEN by rank bucket (holdout):")
        print(f"  {'Bucket':<10} {'Holdout':>10} {'±std':>6} {'n/seed':>8}")
        print("  " + "-" * 40)
        for bucket in ['R1-5', 'R6-10', 'R11-15', 'R16-20']:
            if bucket in std_by_rank:
                vals = std_by_rank[bucket]
                ns = std_n_by_rank[bucket]
                print(f"  {bucket:<10} {np.mean(vals):>8.1f}% {np.std(vals):>5.1f} {np.mean(ns):>7.1f}")

    # Standard GOLDEN category analysis
    if std_by_cat:
        print("\nStandard GOLDEN by category (holdout):")
        print(f"  {'Category':<20} {'Holdout':>10} {'±std':>6} {'n/seed':>8}")
        print("  " + "-" * 50)
        for cat in sorted(std_by_cat.keys(), key=lambda x: -np.mean(std_by_cat[x])):
            vals = std_by_cat[cat]
            ns = std_n_by_cat[cat]
            n_seeds = len(vals)
            print(f"  {cat:<20} {np.mean(vals):>8.1f}% {np.std(vals):>5.1f} {np.mean(ns):>7.1f} ({n_seeds} seeds)")

    # Common misses
    if all_misses:
        print("\nMost common standard GOLDEN misses (across seeds):")
        for key, count in sorted(all_misses.items(), key=lambda x: -x[1])[:15]:
            d = miss_info[key]
            print(f"  [{count}/5] {key} (cat={d['category']}, freq={d['freq']}, rank={d['rank']})")

    # Statistical test: standard vs hierarchy
    print(f"\n--- STATISTICAL COMPARISON ---")
    if 'standard' in holdout_by_rule and 'hierarchy' in holdout_by_rule:
        std_vals = holdout_by_rule['standard']
        hier_vals = holdout_by_rule['hierarchy']

        # Align by seed (both should have 5 entries)
        n_common = min(len(std_vals), len(hier_vals))
        if n_common >= 3:
            std_v = std_vals[:n_common]
            hier_v = hier_vals[:n_common]
            diff = [h - s for h, s in zip(hier_v, std_v)]
            mean_diff = np.mean(diff)
            std_diff = np.std(diff, ddof=1)
            t_stat = mean_diff / (std_diff / np.sqrt(len(diff))) if std_diff > 0 else 0

            print(f"Hierarchy GOLDEN holdout: {np.mean(hier_v):.1f}% ± {np.std(hier_v):.1f}% (n/seed={np.mean(holdout_n_by_rule['hierarchy']):.1f})")
            print(f"Standard GOLDEN holdout:  {np.mean(std_v):.1f}% ± {np.std(std_v):.1f}% (n/seed={np.mean(holdout_n_by_rule['standard']):.1f})")
            print(f"Difference (H-S):         {mean_diff:+.1f}pp ± {std_diff:.1f}pp")
            print(f"Paired t-stat:            {t_stat:.2f}")
            print(f"Significant (|t|>2.78)?   {'YES' if abs(t_stat) > 2.78 else 'NO'} (df={n_common-1}, p<0.05)")
        else:
            print(f"Not enough common seeds for comparison")
            print(f"Standard seeds: {len(std_vals)}, Hierarchy seeds: {len(hier_vals)}")
    else:
        print(f"Rules present: {list(holdout_by_rule.keys())}")

    # Decision framework
    print(f"\n--- DECISION ---")
    if 'standard' in holdout_by_rule and holdout_by_rule['standard']:
        std_holdout = np.mean(holdout_by_rule['standard'])
        if std_holdout < 40:
            print(f"Standard GOLDEN holdout ({std_holdout:.1f}%) is below 40% — consider demotion to HIGH")
            print(f"  HIGH holdout avg: 60.8% ± 7.2% (h478)")
            print(f"  Standard GOLDEN would {'IMPROVE' if std_holdout > 60.8 else 'DRAG DOWN'} HIGH tier")
        elif std_holdout < 50:
            print(f"Standard GOLDEN holdout ({std_holdout:.1f}%) is 40-50% — borderline GOLDEN/HIGH")
        else:
            print(f"Standard GOLDEN holdout ({std_holdout:.1f}%) is ≥50% — justified as GOLDEN")
    else:
        print("No standard GOLDEN predictions found on holdout")


if __name__ == "__main__":
    main()
