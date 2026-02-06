#!/usr/bin/env python3
"""
h533 TransE Consilience: Does TransE agreement rescue FILTER predictions?

Previous work (h405/h439/h440) showed TransE consilience provides +13.6pp MEDIUM holdout.
Question: Does TransE agreement also help within FILTER tier?
If FILTER + TransE top-30 has >15% holdout, those could be promoted to LOW.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from production_predictor import (
    DrugRepurposingPredictor,
    extract_cancer_types,
    DISEASE_HIERARCHY_GROUPS,
    HIERARCHY_EXCLUSIONS,
)

try:
    import torch
except ImportError:
    print("torch not available, cannot test TransE")
    sys.exit(1)


def split_diseases(all_diseases, seed, train_ratio=0.8):
    rng = np.random.RandomState(seed)
    shuffled = list(all_diseases)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def recompute_gt_structures(predictor, train_disease_ids):
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
                new_d2d[drug_id].add(disease_name.lower())
    predictor.drug_to_diseases = dict(new_d2d)

    new_cancer = defaultdict(set)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            cancer_types = extract_cancer_types(disease_name)
            if cancer_types:
                for drug_id in predictor.ground_truth[disease_id]:
                    new_cancer[drug_id].update(cancer_types)
    predictor.drug_cancer_types = dict(new_cancer)

    new_groups = defaultdict(set)
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


def restore_gt_structures(predictor, originals):
    predictor.drug_train_freq = originals["drug_train_freq"]
    predictor.drug_to_diseases = originals["drug_to_diseases"]
    predictor.drug_cancer_types = originals["drug_cancer_types"]
    predictor.drug_disease_groups = originals["drug_disease_groups"]
    predictor.train_diseases = originals["train_diseases"]
    predictor.train_embeddings = originals["train_embeddings"]
    predictor.train_disease_categories = originals["train_disease_categories"]


def build_transe_top_n(predictor, disease_ids, n=30):
    """Build TransE top-N drugs per disease.

    Uses all drugs that have TransE embeddings as candidates.
    """
    # Build candidate drug set from all drugs with embeddings
    all_drug_ids = set()
    for disease_id, drugs in predictor.ground_truth.items():
        for drug_id in drugs:
            if drug_id in (predictor.transe_entity2id or {}):
                all_drug_ids.add(drug_id)
    # Also add drugs from drug_targets
    for drug_id in predictor.drug_targets:
        if drug_id in (predictor.transe_entity2id or {}):
            all_drug_ids.add(drug_id)
    print(f"  TransE candidate drugs: {len(all_drug_ids)}")

    transe_top = {}
    for disease_id in disease_ids:
        top_drugs = predictor._get_transe_top_n(disease_id, all_drug_ids, n=n)
        if top_drugs:
            transe_top[disease_id] = set(top_drugs)
    return transe_top


def main():
    seeds = [42, 123, 456, 789, 2024]

    print("=" * 80)
    print("h533 TransE Consilience in FILTER Tier")
    print("=" * 80)

    predictor = DrugRepurposingPredictor()
    gt_path = predictor.reference_dir / "expanded_ground_truth.json"
    with open(gt_path) as f:
        gt_data = json.load(f)

    all_diseases = [d for d in predictor.ground_truth if d in predictor.embeddings]

    gt_set = set()
    for disease_id, drugs in gt_data.items():
        for drug in drugs:
            if isinstance(drug, str):
                gt_set.add((disease_id, drug))
            elif isinstance(drug, dict):
                gt_set.add((disease_id, drug.get("drug_id") or drug.get("drug")))

    # Load TransE model
    print("Loading TransE model...")
    predictor._load_transe_model()
    if predictor.transe_entity_emb is None:
        print("ERROR: TransE model failed to load")
        return

    # Build TransE top-30 for all diseases
    print("Building TransE top-30 per disease...")
    transe_top30 = build_transe_top_n(predictor, all_diseases, n=30)
    print(f"TransE top-30 available for {len(transe_top30)}/{len(all_diseases)} diseases")

    # === FULL-DATA ANALYSIS ===
    print("\n" + "=" * 80)
    print("FULL-DATA: TransE Consilience in FILTER Tier")
    print("=" * 80)

    # Track FILTER predictions with/without TransE consilience
    filter_with_transe = {"hits": 0, "misses": 0}
    filter_without_transe = {"hits": 0, "misses": 0}
    # Also by category
    filter_transe_by_cat = defaultdict(lambda: {"hits": 0, "misses": 0})
    filter_no_transe_by_cat = defaultdict(lambda: {"hits": 0, "misses": 0})
    # By filter reason
    filter_transe_by_reason = defaultdict(lambda: {"hits": 0, "misses": 0})
    filter_no_transe_by_reason = defaultdict(lambda: {"hits": 0, "misses": 0})

    for disease_id in all_diseases:
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        transe_drugs = transe_top30.get(disease_id, set())

        try:
            result = predictor.predict(disease_name, top_n=30, include_filtered=True)
        except Exception:
            continue

        for pred in result.predictions:
            if pred.confidence_tier.name != "FILTER":
                continue

            is_hit = (disease_id, pred.drug_id) in gt_set
            cat = predictor.categorize_disease(disease_name)
            reason = pred.category_specific_tier or "standard_filter"
            has_transe = pred.drug_id in transe_drugs

            key = "hits" if is_hit else "misses"
            if has_transe:
                filter_with_transe[key] += 1
                filter_transe_by_cat[cat][key] += 1
                filter_transe_by_reason[reason][key] += 1
            else:
                filter_without_transe[key] += 1
                filter_no_transe_by_cat[cat][key] += 1
                filter_no_transe_by_reason[reason][key] += 1

    def prec(stats):
        total = stats["hits"] + stats["misses"]
        return (stats["hits"] / total * 100 if total > 0 else 0), total

    p_with, n_with = prec(filter_with_transe)
    p_without, n_without = prec(filter_without_transe)
    print(f"\n  FILTER + TransE top-30: {p_with:.1f}% ({n_with} predictions)")
    print(f"  FILTER - TransE top-30: {p_without:.1f}% ({n_without} predictions)")
    print(f"  Delta: {p_with - p_without:+.1f}pp")

    print(f"\n--- FILTER + TransE by Category (Full-Data) ---")
    print(f"{'Category':<23s} {'With':>6s} {'nWith':>6s} {'W/o':>6s} {'nW/o':>6s} {'Delta':>6s}")
    for cat in sorted(filter_transe_by_cat.keys()):
        pw, nw = prec(filter_transe_by_cat[cat])
        pwo, nwo = prec(filter_no_transe_by_cat.get(cat, {"hits": 0, "misses": 0}))
        if nw >= 5:
            print(f"  {cat:<21s} {pw:5.1f}% {nw:>5d} {pwo:5.1f}% {nwo:>5d} {pw-pwo:+5.1f}")

    print(f"\n--- FILTER + TransE by Reason (Full-Data) ---")
    for reason in sorted(filter_transe_by_reason.keys()):
        pw, nw = prec(filter_transe_by_reason[reason])
        pwo, nwo = prec(filter_no_transe_by_reason.get(reason, {"hits": 0, "misses": 0}))
        if nw >= 5:
            print(f"  {reason:<33s} {pw:5.1f}% ({nw:>4d}) vs {pwo:5.1f}% ({nwo:>4d}) delta={pw-pwo:+.1f}")

    # === HOLDOUT ===
    print("\n" + "=" * 80)
    print("HOLDOUT: TransE Consilience in FILTER Tier (5 seeds)")
    print("=" * 80)

    holdout_with_prec = []
    holdout_without_prec = []
    holdout_with_n = []
    holdout_without_n = []
    holdout_with_by_cat = defaultdict(list)
    holdout_without_by_cat = defaultdict(list)
    holdout_with_by_cat_n = defaultdict(list)

    for seed_idx, seed in enumerate(seeds):
        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        originals = recompute_gt_structures(predictor, set(train_ids))

        with_stats = {"hits": 0, "misses": 0}
        without_stats = {"hits": 0, "misses": 0}
        cat_with = defaultdict(lambda: {"hits": 0, "misses": 0})
        cat_without = defaultdict(lambda: {"hits": 0, "misses": 0})

        for disease_id in holdout_ids:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            transe_drugs = transe_top30.get(disease_id, set())

            try:
                result = predictor.predict(disease_name, top_n=30, include_filtered=True)
            except Exception:
                continue

            for pred in result.predictions:
                if pred.confidence_tier.name != "FILTER":
                    continue

                is_hit = (disease_id, pred.drug_id) in gt_set
                cat = predictor.categorize_disease(disease_name)
                has_transe = pred.drug_id in transe_drugs
                key = "hits" if is_hit else "misses"

                if has_transe:
                    with_stats[key] += 1
                    cat_with[cat][key] += 1
                else:
                    without_stats[key] += 1
                    cat_without[cat][key] += 1

        p_w, n_w = prec(with_stats)
        p_wo, n_wo = prec(without_stats)
        holdout_with_prec.append(p_w)
        holdout_without_prec.append(p_wo)
        holdout_with_n.append(n_w)
        holdout_without_n.append(n_wo)

        print(f"  Seed {seed}: FILTER+TransE={p_w:.1f}% ({n_w}), FILTER-TransE={p_wo:.1f}% ({n_wo})")

        for cat in cat_with:
            pw, nw = prec(cat_with[cat])
            pwo, _ = prec(cat_without.get(cat, {"hits": 0, "misses": 0}))
            holdout_with_by_cat[cat].append(pw)
            holdout_without_by_cat[cat].append(pwo)
            holdout_with_by_cat_n[cat].append(nw)

        restore_gt_structures(predictor, originals)

    print(f"\n--- Aggregate Holdout ---")
    mean_with = np.mean(holdout_with_prec)
    std_with = np.std(holdout_with_prec)
    mean_without = np.mean(holdout_without_prec)
    std_without = np.std(holdout_without_prec)
    print(f"  FILTER + TransE:  {mean_with:.1f}% ± {std_with:.1f}% (avg {np.mean(holdout_with_n):.0f}/seed)")
    print(f"  FILTER - TransE:  {mean_without:.1f}% ± {std_without:.1f}% (avg {np.mean(holdout_without_n):.0f}/seed)")
    delta = mean_with - mean_without
    print(f"  Delta: {delta:+.1f}pp")

    print(f"\n--- FILTER + TransE by Category (Holdout) ---")
    print(f"{'Category':<23s} {'With':>6s} {'±':>5s} {'W/o':>6s} {'nWith':>6s} {'Delta':>6s} {'Rescue?'}")
    print("-" * 70)

    rescue_candidates = []
    for cat in sorted(holdout_with_by_cat.keys()):
        if len(holdout_with_by_cat[cat]) < 3:
            continue
        mw = np.mean(holdout_with_by_cat[cat])
        sw = np.std(holdout_with_by_cat[cat])
        mwo = np.mean(holdout_without_by_cat.get(cat, [0]))
        avg_n = np.mean(holdout_with_by_cat_n[cat])
        rescue = "YES" if mw > 15 and avg_n >= 5 else "maybe" if mw > 10 else ""
        if rescue == "YES":
            rescue_candidates.append((cat, mw, sw, avg_n))
        if avg_n >= 3:
            print(f"  {cat:<21s} {mw:5.1f}% {sw:4.1f}% {mwo:5.1f}% {avg_n:5.1f} {mw-mwo:+5.1f}  {rescue}")

    print(f"\n{'='*80}")
    print("CONCLUSION")
    print(f"{'='*80}")
    if delta > 5 and mean_with > 15:
        print(f"\n  ** FILTER + TransE = {mean_with:.1f}% holdout → LOW-level promotion candidate **")
        print(f"  Would rescue {np.mean(holdout_with_n):.0f} predictions per seed from FILTER to LOW")
    elif delta > 3:
        print(f"\n  TransE provides +{delta:.1f}pp in FILTER (marginal, not sufficient for promotion)")
    else:
        print(f"\n  TransE provides only +{delta:.1f}pp in FILTER (no rescue opportunity)")

    if rescue_candidates:
        print(f"\n  Category-specific rescue candidates (>15% holdout + TransE):")
        for cat, mw, sw, avg_n in rescue_candidates:
            print(f"    {cat}: {mw:.1f}% ± {sw:.1f}% holdout ({avg_n:.0f}/seed)")

    # Save results
    output = {
        "holdout_with_transe": {"mean": round(mean_with, 1), "std": round(std_with, 1), "avg_n": round(np.mean(holdout_with_n), 0)},
        "holdout_without_transe": {"mean": round(mean_without, 1), "std": round(std_without, 1)},
        "delta_pp": round(delta, 1),
        "rescue_candidates": [(c[0], round(c[1], 1), round(c[2], 1)) for c in rescue_candidates],
    }
    output_path = Path("data/analysis/h533_transe_in_filter.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
