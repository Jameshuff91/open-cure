#!/usr/bin/env python3
"""
h538: Cancer Targeted Therapy Demotion - Kinase Inhibitors + Immunotherapy MEDIUM → LOW

h521 found:
- Kinase inhibitors: 9.6% ± 6.2% holdout in MEDIUM (11/seed)
- Immunotherapy: 18.3% ± 11.7% holdout in MEDIUM (13/seed)
- vs cytotoxic: 53% holdout

These targeted therapies are mutation-specific and don't transfer across cancer subtypes.
Test: demote kinase inhibitor + immunotherapy cancer_same_type predictions MEDIUM → LOW
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


KINASE_INHIBITORS = {
    'imatinib', 'dasatinib', 'nilotinib', 'sunitinib', 'sorafenib',
    'erlotinib', 'gefitinib', 'lapatinib', 'crizotinib', 'ruxolitinib',
    'ibrutinib', 'palbociclib', 'ribociclib', 'lenvatinib', 'regorafenib',
    'axitinib', 'pazopanib', 'vemurafenib', 'dabrafenib', 'trametinib',
    'cobimetinib', 'osimertinib', 'afatinib', 'cabozantinib', 'ponatinib',
    'bosutinib', 'vandetanib', 'tofacitinib', 'baricitinib', 'fedratinib',
    'gilteritinib', 'midostaurin', 'entrectinib', 'larotrectinib',
    'capmatinib', 'tepotinib', 'tucatinib', 'neratinib', 'lorlatinib',
    'alectinib', 'brigatinib', 'ceritinib', 'encorafenib', 'binimetinib',
    'futibatinib', 'infigratinib', 'pemigatinib', 'erdafitinib',
}

IMMUNOTHERAPY = {
    'nivolumab', 'pembrolizumab', 'atezolizumab', 'ipilimumab',
    'durvalumab', 'avelumab', 'tremelimumab', 'cemiplimab',
}

TARGETED_THERAPY = KINASE_INHIBITORS | IMMUNOTHERAPY


def is_targeted_therapy(drug_name):
    lower = drug_name.lower()
    return any(d in lower for d in TARGETED_THERAPY)


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


def main():
    seeds = [42, 123, 456, 789, 2024]

    print("=" * 80)
    print("h538: Targeted Therapy Demotion Impact Simulation")
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

    # Full data count
    demoted_full = 0
    demoted_hits = 0
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
            if is_targeted_therapy(drug_name):
                demoted_full += 1
                if (disease_id, pred.drug_id) in gt_set:
                    demoted_hits += 1

    prec_full = demoted_hits / demoted_full * 100 if demoted_full > 0 else 0
    print(f"\nFull-data: {demoted_full} targeted therapy cancer_same_type MEDIUM predictions")
    print(f"  Precision: {demoted_hits}/{demoted_full} = {prec_full:.1f}%")

    # Holdout impact
    print(f"\n{'='*80}")
    print("HOLDOUT IMPACT (5 seeds)")
    print(f"{'='*80}")

    holdout_med_before = []
    holdout_med_after = []
    holdout_low_before = []
    holdout_low_after = []
    holdout_demoted_n = []
    holdout_demoted_prec = []

    for seed in seeds:
        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        originals = recompute_gt_structures(predictor, set(train_ids))

        med_hits = 0
        med_n = 0
        low_hits = 0
        low_n = 0
        demo_hits = 0
        demo_n = 0

        for disease_id in holdout_ids:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            try:
                result = predictor.predict(disease_name, top_n=30, include_filtered=True)
            except Exception:
                continue

            for pred in result.predictions:
                is_hit = (disease_id, pred.drug_id) in gt_set
                tier = pred.confidence_tier.name

                if tier == "MEDIUM":
                    med_hits += int(is_hit)
                    med_n += 1

                    cat = predictor.categorize_disease(disease_name)
                    if cat == 'cancer' and pred.category_specific_tier == 'cancer_same_type':
                        drug_name = predictor.drug_id_to_name.get(pred.drug_id, pred.drug_id)
                        if is_targeted_therapy(drug_name):
                            demo_hits += int(is_hit)
                            demo_n += 1

                if tier == "LOW":
                    low_hits += int(is_hit)
                    low_n += 1

        med_prec_before = med_hits / med_n * 100 if med_n > 0 else 0
        med_prec_after = (med_hits - demo_hits) / (med_n - demo_n) * 100 if (med_n - demo_n) > 0 else 0
        low_prec_before = low_hits / low_n * 100 if low_n > 0 else 0
        low_prec_after = (low_hits + demo_hits) / (low_n + demo_n) * 100 if (low_n + demo_n) > 0 else 0
        demo_prec = demo_hits / demo_n * 100 if demo_n > 0 else 0

        holdout_med_before.append(med_prec_before)
        holdout_med_after.append(med_prec_after)
        holdout_low_before.append(low_prec_before)
        holdout_low_after.append(low_prec_after)
        holdout_demoted_n.append(demo_n)
        holdout_demoted_prec.append(demo_prec)

        print(f"  Seed {seed}: demoted {demo_n} ({demo_hits} hits = {demo_prec:.0f}%)")
        print(f"    MEDIUM: {med_prec_before:.1f}% → {med_prec_after:.1f}% ({med_n}→{med_n-demo_n})")
        print(f"    LOW: {low_prec_before:.1f}% → {low_prec_after:.1f}% ({low_n}→{low_n+demo_n})")

        restore_gt_structures(predictor, originals)

    print(f"\n--- Aggregate ---")
    print(f"  Demoted: {np.mean(holdout_demoted_n):.0f}/seed, prec: {np.mean(holdout_demoted_prec):.1f}% ± {np.std(holdout_demoted_prec):.1f}%")
    delta_med = np.mean(holdout_med_after) - np.mean(holdout_med_before)
    delta_low = np.mean(holdout_low_after) - np.mean(holdout_low_before)
    print(f"  MEDIUM: {np.mean(holdout_med_before):.1f}% → {np.mean(holdout_med_after):.1f}% ({delta_med:+.1f}pp)")
    print(f"  LOW: {np.mean(holdout_low_before):.1f}% → {np.mean(holdout_low_after):.1f}% ({delta_low:+.1f}pp)")

    print(f"\n{'='*80}")
    print("VERDICT")
    print(f"{'='*80}")
    if delta_med > 0 and delta_low >= -1:
        print(f"  MEDIUM improved {delta_med:+.1f}pp. LOW {delta_low:+.1f}pp → IMPLEMENT")
    elif delta_med > 0:
        print(f"  MEDIUM improved {delta_med:+.1f}pp but LOW dropped {delta_low:+.1f}pp → CONSIDER")
    else:
        print(f"  MEDIUM {delta_med:+.1f}pp, LOW {delta_low:+.1f}pp → NOT RECOMMENDED")


if __name__ == "__main__":
    main()
