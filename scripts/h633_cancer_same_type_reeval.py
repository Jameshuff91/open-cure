#!/usr/bin/env python3
"""
h633: Re-evaluate Cancer Same-Type with Expanded GT

CLOSED direction #4 (cancer_same_type→HIGH) was closed based on internal GT
showing 10.7% holdout. h611/h629 showed expanded GT lifts cancer_same_type to 37.7%.

This script investigates whether SUBSETS of cancer_same_type reach HIGH (>50%)
with expanded GT, specifically:
1. TransE + cancer_same_type
2. Rank stratification within cancer_same_type
3. Drug class stratification (cytotoxic vs targeted vs hormonal)
4. CS artifact check
5. Formal holdout evaluation of candidate subsets
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from production_predictor import (
    ConfidenceTier,
    DrugRepurposingPredictor,
    CANCER_TARGETED_THERAPY,
    CORTICOSTEROID_DRUGS,
)

# Load expanded GT
def load_expanded_gt() -> Dict[str, Set[str]]:
    gt_path = Path("data/reference/expanded_ground_truth.json")
    with open(gt_path) as f:
        raw = json.load(f)
    return {k: set(v) for k, v in raw.items()}


def split_diseases(all_diseases: List[str], seed: int, train_ratio: float = 0.8):
    rng = np.random.RandomState(seed)
    shuffled = list(all_diseases)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def recompute_gt_structures(predictor, train_disease_ids: Set[str]):
    """Recompute GT-derived structures from training diseases only."""
    originals = {
        "drug_train_freq": dict(predictor.drug_train_freq),
        "drug_to_diseases": dict(predictor.drug_to_diseases),
        "drug_cancer_types": dict(predictor.drug_cancer_types),
        "drug_disease_groups": dict(predictor.drug_disease_groups),
        "train_diseases": list(predictor.train_diseases),
        "train_embeddings": predictor.train_embeddings.copy(),
        "train_disease_categories": dict(predictor.train_disease_categories),
    }

    # Recompute from training only
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

    from production_predictor import extract_cancer_types
    new_cancer_types = {}
    for drug_id, diseases in new_d2d.items():
        types = set()
        for d in diseases:
            types.update(extract_cancer_types(d))
        if types:
            new_cancer_types[drug_id] = types
    predictor.drug_cancer_types = new_cancer_types

    from production_predictor import DISEASE_HIERARCHY_GROUPS
    new_groups = defaultdict(lambda: defaultdict(set))
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            disease_name = predictor.disease_names.get(disease_id, "").lower()
            for group_name, group_info in DISEASE_HIERARCHY_GROUPS.items():
                for subtype, keywords in group_info.get("subtypes", {}).items():
                    if any(kw in disease_name for kw in keywords):
                        for drug_id in predictor.ground_truth[disease_id]:
                            new_groups[drug_id][group_name].add(subtype)
    predictor.drug_disease_groups = dict(new_groups)

    train_idx = [
        i for i, d in enumerate(predictor.train_diseases) if d in train_disease_ids
    ]
    predictor.train_diseases = [predictor.train_diseases[i] for i in train_idx]
    predictor.train_embeddings = predictor.train_embeddings[train_idx]
    predictor.train_disease_categories = {
        d: predictor.train_disease_categories[d]
        for d in predictor.train_diseases
        if d in predictor.train_disease_categories
    }

    return originals


def restore_originals(predictor, originals):
    for key, val in originals.items():
        setattr(predictor, key, val)


def main():
    print("=" * 80)
    print("h633: Cancer Same-Type Re-evaluation with Expanded GT")
    print("=" * 80)

    predictor = DrugRepurposingPredictor()
    expanded_gt = load_expanded_gt()

    # Get all diseases
    all_diseases = list(predictor.ground_truth.keys())
    print(f"\nTotal diseases: {len(all_diseases)}")

    # Phase 1: Full-data analysis of cancer_same_type predictions
    print("\n" + "=" * 60)
    print("PHASE 1: Full-Data Signal Analysis")
    print("=" * 60)

    all_predictions = []
    for disease_id in all_diseases:
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        result = predictor.predict(disease_name, top_n=30, k=20)
        for p in result.predictions:
            if p.category_specific_tier == "cancer_same_type":
                all_predictions.append((disease_id, p))

    print(f"\nTotal cancer_same_type predictions: {len(all_predictions)}")

    # Signal breakdown
    combos = Counter()
    for disease_id, p in all_predictions:
        te = getattr(p, "transe_consilience", False)
        me = p.mechanism_support
        r5 = p.rank <= 5
        r10 = p.rank <= 10
        is_cs = any(
            cs in p.drug_name.lower() for cs in ["dexamethasone", "predniso", "methylprednisolone", "hydrocortisone", "betamethasone", "triamcinolone", "budesonide", "fluticasone", "beclomethasone", "cortisone"]
        )
        combo = []
        if te:
            combo.append("TransE")
        if me:
            combo.append("Mech")
        if r5:
            combo.append("R<=5")
        elif r10:
            combo.append("R<=10")
        key = "+".join(combo) if combo else "None"
        combos[key] += 1

    print("\nSignal combinations:")
    for key, count in combos.most_common(15):
        print(f"  {key}: {count}")

    # Check CS prevalence
    cs_count = sum(
        1
        for _, p in all_predictions
        if any(
            cs in p.drug_name.lower()
            for cs in CORTICOSTEROID_DRUGS
        )
    )
    print(f"\nCorticosteroid predictions: {cs_count} / {len(all_predictions)} ({100*cs_count/len(all_predictions):.1f}%)")

    # Phase 2: 5-seed holdout evaluation
    print("\n" + "=" * 60)
    print("PHASE 2: 5-Seed Holdout Evaluation by Signal Combination")
    print("=" * 60)

    seeds = [42, 123, 456, 789, 1024]

    # Track results per signal combination
    signal_results = defaultdict(lambda: {"hits": [], "totals": [], "precisions": []})

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        train_diseases, holdout_diseases = split_diseases(all_diseases, seed)
        holdout_set = set(holdout_diseases)
        train_set = set(train_diseases)

        originals = recompute_gt_structures(predictor, train_set)

        # Run predictions on holdout diseases
        for disease_id in holdout_diseases:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            try:
                result = predictor.predict(disease_name, top_n=30, k=20)
            except Exception:
                continue

            for p in result.predictions:
                if p.category_specific_tier != "cancer_same_type":
                    continue

                # Check if hit in expanded GT
                drug_id = p.drug_id
                is_hit = drug_id in expanded_gt.get(disease_id, set())

                # Classify signals
                te = getattr(p, "transe_consilience", False)
                me = p.mechanism_support
                r5 = p.rank <= 5
                r10 = p.rank <= 10
                is_cs = any(
                    cs_drug.lower() in p.drug_name.lower()
                    for cs_drug in CORTICOSTEROID_DRUGS
                )

                # Signal keys for different analyses
                signal_keys = []

                # Overall
                signal_keys.append("ALL")

                # TransE split
                if te:
                    signal_keys.append("TransE_yes")
                else:
                    signal_keys.append("TransE_no")

                # Rank buckets
                if r5:
                    signal_keys.append("rank_1_5")
                elif r10:
                    signal_keys.append("rank_6_10")
                else:
                    signal_keys.append("rank_11_20")

                # Mechanism
                if me:
                    signal_keys.append("mech_yes")
                else:
                    signal_keys.append("mech_no")

                # CS split
                if is_cs:
                    signal_keys.append("CS")
                else:
                    signal_keys.append("non_CS")

                # Key combinations for promotion candidates
                if te and r5:
                    signal_keys.append("TransE+R<=5")
                if te and me:
                    signal_keys.append("TransE+Mech")
                if te and (me or r5):
                    signal_keys.append("TransE+(Mech_or_R<=5)")
                if te and r10:
                    signal_keys.append("TransE+R<=10")
                if me and r5:
                    signal_keys.append("Mech+R<=5")
                if me and r10:
                    signal_keys.append("Mech+R<=10")

                # Non-CS versions
                if not is_cs:
                    signal_keys.append("ALL_non_CS")
                    if te:
                        signal_keys.append("TransE_yes_non_CS")
                    if te and (me or r5):
                        signal_keys.append("TransE+(Mech_or_R<=5)_non_CS")
                    if te and r10:
                        signal_keys.append("TransE+R<=10_non_CS")

                for key in signal_keys:
                    if is_hit:
                        signal_results[key]["hits"].append(1)
                    else:
                        signal_results[key]["hits"].append(0)

        # Compute per-seed precision
        # (We need to track per-seed, so let's restructure)
        restore_originals(predictor, originals)

    # Actually, let me restructure to track per-seed properly
    print("\n" + "=" * 60)
    print("RE-RUNNING with per-seed tracking...")
    print("=" * 60)

    # Clear and restart with proper per-seed tracking
    seed_results = defaultdict(lambda: defaultdict(lambda: {"hits": 0, "total": 0}))

    for seed in seeds:
        train_diseases, holdout_diseases = split_diseases(all_diseases, seed)
        holdout_set = set(holdout_diseases)
        train_set = set(train_diseases)

        originals = recompute_gt_structures(predictor, train_set)

        for disease_id in holdout_diseases:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            try:
                result = predictor.predict(disease_name, top_n=30, k=20)
            except Exception:
                continue

            for p in result.predictions:
                if p.category_specific_tier != "cancer_same_type":
                    continue

                drug_id = p.drug_id
                is_hit = drug_id in expanded_gt.get(disease_id, set())

                te = getattr(p, "transe_consilience", False)
                me = p.mechanism_support
                r5 = p.rank <= 5
                r10 = p.rank <= 10
                is_cs = any(
                    cs_drug.lower() in p.drug_name.lower()
                    for cs_drug in CORTICOSTEROID_DRUGS
                )

                signal_keys = ["ALL"]
                if te:
                    signal_keys.append("TransE_yes")
                else:
                    signal_keys.append("TransE_no")
                if r5:
                    signal_keys.append("rank_1_5")
                elif r10:
                    signal_keys.append("rank_6_10")
                else:
                    signal_keys.append("rank_11_20")
                if me:
                    signal_keys.append("mech_yes")
                else:
                    signal_keys.append("mech_no")
                if is_cs:
                    signal_keys.append("CS")
                else:
                    signal_keys.append("non_CS")

                if te and r5:
                    signal_keys.append("TransE+R<=5")
                if te and me:
                    signal_keys.append("TransE+Mech")
                if te and (me or r5):
                    signal_keys.append("TransE+(Mech_or_R<=5)")
                if te and r10:
                    signal_keys.append("TransE+R<=10")
                if me and r5:
                    signal_keys.append("Mech+R<=5")
                if me and r10:
                    signal_keys.append("Mech+R<=10")

                if not is_cs:
                    signal_keys.append("ALL_non_CS")
                    if te:
                        signal_keys.append("TransE_yes_non_CS")
                    if te and (me or r5):
                        signal_keys.append("TransE+(Mech_or_R<=5)_non_CS")
                    if te and r10:
                        signal_keys.append("TransE+R<=10_non_CS")
                    if me and r10:
                        signal_keys.append("Mech+R<=10_non_CS")

                for key in signal_keys:
                    seed_results[seed][key]["total"] += 1
                    if is_hit:
                        seed_results[seed][key]["hits"] += 1

        restore_originals(predictor, originals)

    # Aggregate across seeds
    print("\n" + "=" * 60)
    print("RESULTS: Cancer Same-Type Holdout by Signal (Expanded GT)")
    print("=" * 60)

    all_keys = set()
    for seed in seeds:
        all_keys.update(seed_results[seed].keys())

    results_table = []
    for key in sorted(all_keys):
        precisions = []
        totals = []
        for seed in seeds:
            data = seed_results[seed][key]
            if data["total"] > 0:
                precisions.append(data["hits"] / data["total"])
                totals.append(data["total"])
        if len(precisions) >= 3:  # At least 3 seeds
            mean_p = np.mean(precisions) * 100
            std_p = np.std(precisions) * 100
            mean_n = np.mean(totals)
            results_table.append((key, mean_p, std_p, mean_n, len(precisions)))

    # Sort by precision
    results_table.sort(key=lambda x: -x[1])

    print(f"\n{'Signal':<40} {'Holdout':>8} {'±std':>8} {'N/seed':>8} {'Seeds':>6}")
    print("-" * 74)
    for key, mean_p, std_p, mean_n, n_seeds in results_table:
        marker = ""
        if mean_p >= 50:
            marker = " *** HIGH ***"
        elif mean_p >= 40:
            marker = " ** MEDIUM+ **"
        print(f"{key:<40} {mean_p:>7.1f}% {std_p:>7.1f}% {mean_n:>7.1f} {n_seeds:>5}{marker}")

    # Phase 3: Identify promotion candidates
    print("\n" + "=" * 60)
    print("PHASE 3: Promotion Candidates (holdout >= 50%, n >= 5/seed)")
    print("=" * 60)

    candidates = [
        (key, mean_p, std_p, mean_n)
        for key, mean_p, std_p, mean_n, n_seeds in results_table
        if mean_p >= 50 and mean_n >= 5
    ]

    if candidates:
        for key, mean_p, std_p, mean_n in candidates:
            print(f"  {key}: {mean_p:.1f}% ± {std_p:.1f}% (n={mean_n:.1f}/seed)")
    else:
        print("  No subsets reach HIGH threshold (50%) with sufficient n.")
        print("\n  Closest candidates:")
        close = [
            (key, mean_p, std_p, mean_n)
            for key, mean_p, std_p, mean_n, n_seeds in results_table
            if mean_p >= 35 and mean_n >= 3
        ][:10]
        for key, mean_p, std_p, mean_n in close:
            print(f"  {key}: {mean_p:.1f}% ± {std_p:.1f}% (n={mean_n:.1f}/seed)")

    # Phase 4: Check drug class breakdown
    print("\n" + "=" * 60)
    print("PHASE 4: Drug Class Breakdown within Cancer Same-Type")
    print("=" * 60)

    drug_class_results = defaultdict(lambda: defaultdict(lambda: {"hits": 0, "total": 0}))

    for seed in seeds:
        train_diseases, holdout_diseases = split_diseases(all_diseases, seed)
        train_set = set(train_diseases)

        originals = recompute_gt_structures(predictor, train_set)

        for disease_id in holdout_diseases:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            try:
                result = predictor.predict(disease_name, top_n=30, k=20)
            except Exception:
                continue

            for p in result.predictions:
                if p.category_specific_tier != "cancer_same_type":
                    continue

                drug_id = p.drug_id
                is_hit = drug_id in expanded_gt.get(disease_id, set())
                drug_lower = p.drug_name.lower()

                # Classify drug
                is_targeted = drug_id in CANCER_TARGETED_THERAPY
                is_cs = any(
                    cs.lower() in drug_lower for cs in CORTICOSTEROID_DRUGS
                )

                # Common cancer drug classes
                if is_cs:
                    drug_class = "corticosteroid"
                elif is_targeted:
                    drug_class = "targeted_therapy"
                elif any(x in drug_lower for x in ["platinum", "cisplatin", "carboplatin", "oxaliplatin"]):
                    drug_class = "platinum"
                elif any(x in drug_lower for x in ["doxorubicin", "epirubicin", "daunorubicin", "idarubicin"]):
                    drug_class = "anthracycline"
                elif any(x in drug_lower for x in ["paclitaxel", "docetaxel", "cabazitaxel"]):
                    drug_class = "taxane"
                elif any(x in drug_lower for x in ["fluorouracil", "capecitabine", "gemcitabine", "cytarabine", "methotrexate", "pemetrexed"]):
                    drug_class = "antimetabolite"
                elif any(x in drug_lower for x in ["cyclophosphamide", "ifosfamide", "melphalan", "chlorambucil", "busulfan", "temozolomide", "dacarbazine"]):
                    drug_class = "alkylating"
                elif any(x in drug_lower for x in ["vincristine", "vinblastine", "vinorelbine"]):
                    drug_class = "vinca_alkaloid"
                elif any(x in drug_lower for x in ["etoposide", "topotecan", "irinotecan"]):
                    drug_class = "topo_inhibitor"
                elif any(x in drug_lower for x in ["bleomycin", "mitomycin", "actinomycin"]):
                    drug_class = "antitumor_antibiotic"
                elif any(x in drug_lower for x in ["tamoxifen", "letrozole", "anastrozole", "exemestane", "fulvestrant"]):
                    drug_class = "hormonal"
                else:
                    drug_class = "other_cytotoxic"

                drug_class_results[seed][drug_class]["total"] += 1
                if is_hit:
                    drug_class_results[seed][drug_class]["hits"] += 1

        restore_originals(predictor, originals)

    # Aggregate drug class results
    all_classes = set()
    for seed in seeds:
        all_classes.update(drug_class_results[seed].keys())

    print(f"\n{'Drug Class':<25} {'Holdout':>8} {'±std':>8} {'N/seed':>8}")
    print("-" * 53)
    class_table = []
    for dc in sorted(all_classes):
        precisions = []
        totals = []
        for seed in seeds:
            data = drug_class_results[seed][dc]
            if data["total"] > 0:
                precisions.append(data["hits"] / data["total"])
                totals.append(data["total"])
        if precisions:
            mean_p = np.mean(precisions) * 100
            std_p = np.std(precisions) * 100
            mean_n = np.mean(totals)
            class_table.append((dc, mean_p, std_p, mean_n))

    class_table.sort(key=lambda x: -x[1])
    for dc, mean_p, std_p, mean_n in class_table:
        marker = ""
        if mean_p >= 50:
            marker = " ***"
        print(f"{dc:<25} {mean_p:>7.1f}% {std_p:>7.1f}% {mean_n:>7.1f}{marker}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Find best subset
    promotable = [
        (key, mean_p, std_p, mean_n)
        for key, mean_p, std_p, mean_n, n_seeds in results_table
        if mean_p >= 50 and mean_n >= 5
    ]

    if promotable:
        print("\nPROMOTABLE subsets found:")
        for key, mean_p, std_p, mean_n in promotable:
            print(f"  {key}: {mean_p:.1f}% ± {std_p:.1f}% (n={mean_n:.1f})")
        print("\nRecommendation: Implement promotion for above subsets → HIGH")
    else:
        print("\nNo cancer_same_type subsets reach HIGH threshold (50%).")
        print("Cancer same-type→HIGH remains CLOSED even with expanded GT.")
        # Check if the overall level is at least adequate for MEDIUM
        all_result = next((r for r in results_table if r[0] == "ALL"), None)
        if all_result:
            print(f"\nOverall cancer_same_type holdout: {all_result[1]:.1f}% ± {all_result[2]:.1f}% (MEDIUM adequate)")

    # Save results
    output = {
        "signal_results": [
            {"signal": key, "holdout_pct": round(mean_p, 1), "std_pct": round(std_p, 1), "n_per_seed": round(mean_n, 1)}
            for key, mean_p, std_p, mean_n, _ in results_table
        ],
        "drug_class_results": [
            {"class": dc, "holdout_pct": round(mean_p, 1), "std_pct": round(std_p, 1), "n_per_seed": round(mean_n, 1)}
            for dc, mean_p, std_p, mean_n in class_table
        ],
    }
    output_path = Path("data/analysis/h633_cancer_same_type_reeval.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
