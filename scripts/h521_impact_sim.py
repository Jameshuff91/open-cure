#!/usr/bin/env python3
"""
h521 Impact Simulation: What happens if we promote cytotoxic cancer drug MEDIUM → HIGH?

Candidates: anthracyclines (73.1%), taxanes (71.2%), antimetabolites (47.7%), platinum (35.9%)
Test: promote cytotoxic cancer_same_type predictions from MEDIUM → HIGH
Measure: holdout impact on MEDIUM and HIGH tiers
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

# Cytotoxic drug classes with holdout > 40%
CYTOTOXIC_PROMOTE = {
    'antimetabolites': ['methotrexate', 'fluorouracil', '5-fu', 'capecitabine', 'gemcitabine',
                       'cytarabine', 'pemetrexed', 'cladribine', 'fludarabine', 'mercaptopurine',
                       'azacitidine', 'decitabine'],
    'anthracyclines': ['doxorubicin', 'daunorubicin', 'epirubicin', 'idarubicin'],
    'taxanes': ['paclitaxel', 'docetaxel', 'cabazitaxel'],
    'platinum': ['cisplatin', 'carboplatin', 'oxaliplatin'],
    'vinca_alkaloids': ['vincristine', 'vinblastine', 'vinorelbine'],
}

# Build flat set of cytotoxic drug names
CYTOTOXIC_NAMES = set()
for drugs in CYTOTOXIC_PROMOTE.values():
    for d in drugs:
        CYTOTOXIC_NAMES.add(d)


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


def is_cytotoxic(drug_name):
    lower = drug_name.lower()
    return any(d in lower for d in CYTOTOXIC_NAMES)


def main():
    seeds = [42, 123, 456, 789, 2024]

    print("=" * 80)
    print("h521 Impact Simulation: Cytotoxic Cancer Drug MEDIUM → HIGH")
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

    # Count how many would be promoted (full data)
    promoted_full = 0
    promoted_hits = 0
    for disease_id in all_diseases:
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        cat = predictor.categorize_disease(disease_name)
        if cat != 'cancer':
            continue

        try:
            result = predictor.predict(disease_name, top_n=30, include_filtered=True)
        except Exception:
            continue

        for pred in result.predictions:
            if pred.confidence_tier.name != "MEDIUM":
                continue
            if pred.category_specific_tier != 'cancer_same_type':
                continue

            drug_name = predictor.drug_id_to_name.get(pred.drug_id, pred.drug_id)
            if is_cytotoxic(drug_name):
                promoted_full += 1
                if (disease_id, pred.drug_id) in gt_set:
                    promoted_hits += 1

    prec_full = promoted_hits / promoted_full * 100 if promoted_full > 0 else 0
    print(f"\nFull-data: {promoted_full} cytotoxic cancer_same_type MEDIUM predictions")
    print(f"  Precision: {promoted_hits}/{promoted_full} = {prec_full:.1f}%")

    # Holdout impact simulation
    print(f"\n{'='*80}")
    print("HOLDOUT IMPACT SIMULATION (5 seeds)")
    print(f"{'='*80}")

    holdout_high_before = []
    holdout_high_after = []
    holdout_med_before = []
    holdout_med_after = []
    holdout_promoted_n = []
    holdout_promoted_prec = []

    for seed in seeds:
        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        originals = recompute_gt_structures(predictor, set(train_ids))

        high_hits = 0
        high_n = 0
        med_hits = 0
        med_n = 0
        promo_hits = 0
        promo_n = 0

        for disease_id in holdout_ids:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            try:
                result = predictor.predict(disease_name, top_n=30, include_filtered=True)
            except Exception:
                continue

            for pred in result.predictions:
                is_hit = (disease_id, pred.drug_id) in gt_set
                tier = pred.confidence_tier.name

                if tier == "HIGH":
                    high_hits += int(is_hit)
                    high_n += 1

                if tier == "MEDIUM":
                    med_hits += int(is_hit)
                    med_n += 1

                    # Would this be promoted?
                    cat = predictor.categorize_disease(disease_name)
                    if cat == 'cancer' and pred.category_specific_tier == 'cancer_same_type':
                        drug_name = predictor.drug_id_to_name.get(pred.drug_id, pred.drug_id)
                        if is_cytotoxic(drug_name):
                            promo_hits += int(is_hit)
                            promo_n += 1

        # Compute metrics
        high_prec_before = high_hits / high_n * 100 if high_n > 0 else 0
        high_prec_after = (high_hits + promo_hits) / (high_n + promo_n) * 100 if (high_n + promo_n) > 0 else 0
        med_prec_before = med_hits / med_n * 100 if med_n > 0 else 0
        med_prec_after = (med_hits - promo_hits) / (med_n - promo_n) * 100 if (med_n - promo_n) > 0 else 0
        promo_prec = promo_hits / promo_n * 100 if promo_n > 0 else 0

        holdout_high_before.append(high_prec_before)
        holdout_high_after.append(high_prec_after)
        holdout_med_before.append(med_prec_before)
        holdout_med_after.append(med_prec_after)
        holdout_promoted_n.append(promo_n)
        holdout_promoted_prec.append(promo_prec)

        print(f"  Seed {seed}: promoted {promo_n} ({promo_hits} hits = {promo_prec:.0f}%)")
        print(f"    HIGH: {high_prec_before:.1f}% → {high_prec_after:.1f}% ({high_n}→{high_n+promo_n})")
        print(f"    MEDIUM: {med_prec_before:.1f}% → {med_prec_after:.1f}% ({med_n}→{med_n-promo_n})")

        restore_gt_structures(predictor, originals)

    print(f"\n--- Aggregate ---")
    print(f"  Promoted: {np.mean(holdout_promoted_n):.0f}/seed, prec: {np.mean(holdout_promoted_prec):.1f}% ± {np.std(holdout_promoted_prec):.1f}%")
    print(f"  HIGH before: {np.mean(holdout_high_before):.1f}% ± {np.std(holdout_high_before):.1f}%")
    print(f"  HIGH after:  {np.mean(holdout_high_after):.1f}% ± {np.std(holdout_high_after):.1f}%")
    delta_high = np.mean(holdout_high_after) - np.mean(holdout_high_before)
    print(f"  HIGH delta:  {delta_high:+.1f}pp")
    print(f"  MEDIUM before: {np.mean(holdout_med_before):.1f}% ± {np.std(holdout_med_before):.1f}%")
    print(f"  MEDIUM after:  {np.mean(holdout_med_after):.1f}% ± {np.std(holdout_med_after):.1f}%")
    delta_med = np.mean(holdout_med_after) - np.mean(holdout_med_before)
    print(f"  MEDIUM delta: {delta_med:+.1f}pp")

    print(f"\n{'='*80}")
    print("VERDICT")
    print(f"{'='*80}")
    if delta_high >= 0 and delta_med >= 0:
        print(f"  ** BOTH tiers improve: HIGH {delta_high:+.1f}pp, MEDIUM {delta_med:+.1f}pp → IMPLEMENT **")
    elif delta_high >= -1 and delta_med >= 0:
        print(f"  MEDIUM improves {delta_med:+.1f}pp, HIGH minor drop {delta_high:+.1f}pp → CONSIDER")
    else:
        print(f"  HIGH {delta_high:+.1f}pp, MEDIUM {delta_med:+.1f}pp → NOT RECOMMENDED")


if __name__ == "__main__":
    main()
