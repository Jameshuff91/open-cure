#!/usr/bin/env python3
"""
h521: Cancer Drug Same-Category SOC Promotion

h520 showed cancer_drugs MEDIUM has 34.0% holdout (n=63/seed, p=0.25).
This script stratifies cancer_drugs MEDIUM by:
1. Cancer subtype (solid vs hematologic vs CNS)
2. Drug mechanism class (antimetabolites, kinase inhibitors, etc.)
3. Whether cancer_same_type interaction applies
4. Holdout validation of any promising subgroups
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
    extract_cancer_types,
    DISEASE_HIERARCHY_GROUPS,
    HIERARCHY_EXCLUSIONS,
)


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


# Cancer subtypes for stratification
HEMATOLOGIC_KEYWORDS = [
    'leukemia', 'lymphoma', 'myeloma', 'myelodysplastic', 'myeloproliferative',
    'hodgkin', 'myeloid', 'lymphocytic', 'lymphoblastic', 'polycythemia',
    'thrombocythemia', 'myelofibrosis', 'waldenstrom', 'hairy cell',
]

CNS_CANCER_KEYWORDS = [
    'glioma', 'glioblastoma', 'astrocytoma', 'meningioma', 'medulloblastoma',
    'neuroblastoma', 'brain tumor', 'brain cancer',
]

# Cancer drug classes
CANCER_DRUG_CLASSES = {
    'antimetabolites': ['methotrexate', 'fluorouracil', '5-fu', 'capecitabine', 'gemcitabine',
                       'cytarabine', 'pemetrexed', 'cladribine', 'fludarabine', 'mercaptopurine',
                       'azacitidine', 'decitabine'],
    'alkylating': ['cyclophosphamide', 'temozolomide', 'carmustine', 'lomustine', 'busulfan',
                   'chlorambucil', 'melphalan', 'bendamustine', 'ifosfamide'],
    'anthracyclines': ['doxorubicin', 'daunorubicin', 'epirubicin', 'idarubicin'],
    'platinum': ['cisplatin', 'carboplatin', 'oxaliplatin'],
    'vinca_alkaloids': ['vincristine', 'vinblastine', 'vinorelbine'],
    'taxanes': ['paclitaxel', 'docetaxel', 'cabazitaxel'],
    'kinase_inhibitors': ['imatinib', 'dasatinib', 'nilotinib', 'sunitinib', 'sorafenib',
                         'erlotinib', 'gefitinib', 'lapatinib', 'crizotinib', 'ruxolitinib',
                         'ibrutinib', 'palbociclib', 'ribociclib', 'lenvatinib', 'regorafenib',
                         'axitinib', 'pazopanib', 'vemurafenib', 'dabrafenib', 'trametinib',
                         'cobimetinib', 'osimertinib', 'afatinib', 'cabozantinib'],
    'hormonal': ['tamoxifen', 'letrozole', 'anastrozole', 'exemestane', 'bicalutamide',
                'enzalutamide', 'abiraterone', 'leuprolide', 'goserelin', 'fulvestrant'],
    'immunotherapy': ['nivolumab', 'pembrolizumab', 'atezolizumab', 'ipilimumab',
                     'durvalumab', 'avelumab', 'tremelimumab'],
    'proteasome_inhibitors': ['bortezomib', 'carfilzomib', 'ixazomib'],
    'imids': ['thalidomide', 'lenalidomide', 'pomalidomide'],
    'misc_targeted': ['rituximab', 'trastuzumab', 'bevacizumab', 'cetuximab',
                     'obinutuzumab', 'brentuximab', 'venetoclax', 'olaparib',
                     'rucaparib', 'niraparib'],
}


def classify_cancer_subtype(disease_name: str) -> str:
    lower = disease_name.lower()
    if any(kw in lower for kw in HEMATOLOGIC_KEYWORDS):
        return 'hematologic'
    if any(kw in lower for kw in CNS_CANCER_KEYWORDS):
        return 'cns'
    return 'solid'


def classify_cancer_drug(drug_name: str) -> str:
    lower = drug_name.lower()
    for drug_class, drugs in CANCER_DRUG_CLASSES.items():
        if any(d in lower for d in drugs):
            return drug_class
    return 'other'


def main():
    seeds = [42, 123, 456, 789, 2024]

    print("=" * 80)
    print("h521: Cancer Drug SOC Stratification in MEDIUM Tier")
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

    # Identify which drugs are "cancer_drugs" SOC class
    from production_predictor import DRUG_CLASS_SOC_MAPPINGS
    cancer_soc_info = DRUG_CLASS_SOC_MAPPINGS.get('cancer_drugs', {})
    cancer_drug_names = {d.lower() for d in cancer_soc_info.get('drugs', set())}
    print(f"Cancer drugs in SOC: {len(cancer_drug_names)}")

    # === FULL-DATA ANALYSIS ===
    print("\n" + "=" * 80)
    print("FULL-DATA: Cancer Drug MEDIUM Predictions")
    print("=" * 80)

    # Collect all cancer-category MEDIUM predictions that have cancer_drugs SOC
    cancer_medium_preds = []

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

            drug_name = predictor.drug_id_to_name.get(pred.drug_id, pred.drug_id)
            is_cancer_soc = drug_name.lower() in cancer_drug_names
            is_hit = (disease_id, pred.drug_id) in gt_set

            # Check cancer_same_type
            has_cancer_gt, same_type, _ = predictor._check_cancer_type_match(pred.drug_id, disease_name)

            cancer_subtype = classify_cancer_subtype(disease_name)
            drug_class = classify_cancer_drug(drug_name)

            cancer_medium_preds.append({
                'drug_id': pred.drug_id,
                'drug_name': drug_name,
                'disease_id': disease_id,
                'disease_name': disease_name,
                'is_hit': is_hit,
                'is_cancer_soc': is_cancer_soc,
                'cancer_subtype': cancer_subtype,
                'drug_class': drug_class,
                'same_type': same_type,
                'has_cancer_gt': has_cancer_gt,
                'rank': pred.rank,
                'freq': pred.train_frequency,
                'category_specific': pred.category_specific_tier,
            })

    total = len(cancer_medium_preds)
    hits = sum(1 for p in cancer_medium_preds if p['is_hit'])
    prec_full = hits / total * 100 if total > 0 else 0
    print(f"\nTotal cancer MEDIUM predictions: {total}")
    print(f"GT hits: {hits} ({prec_full:.1f}%)")

    # Stratify by SOC class
    soc_preds = [p for p in cancer_medium_preds if p['is_cancer_soc']]
    non_soc = [p for p in cancer_medium_preds if not p['is_cancer_soc']]
    print(f"\nSOC cancer_drugs: {len(soc_preds)} preds, {sum(1 for p in soc_preds if p['is_hit'])} hits ({sum(1 for p in soc_preds if p['is_hit'])/len(soc_preds)*100:.1f}% if soc_preds else 0)")
    print(f"Non-SOC: {len(non_soc)} preds, {sum(1 for p in non_soc if p['is_hit'])} hits ({sum(1 for p in non_soc if p['is_hit'])/len(non_soc)*100:.1f}% if non_soc else 0)")

    # Stratify by cancer subtype
    print(f"\n--- By Cancer Subtype ---")
    for subtype in ['solid', 'hematologic', 'cns']:
        sub_preds = [p for p in cancer_medium_preds if p['cancer_subtype'] == subtype]
        if not sub_preds:
            continue
        sub_hits = sum(1 for p in sub_preds if p['is_hit'])
        print(f"  {subtype}: {sub_hits}/{len(sub_preds)} = {sub_hits/len(sub_preds)*100:.1f}%")

    # Stratify by drug class
    print(f"\n--- By Drug Class ---")
    class_stats = defaultdict(lambda: {'hits': 0, 'total': 0})
    for p in cancer_medium_preds:
        class_stats[p['drug_class']]['total'] += 1
        if p['is_hit']:
            class_stats[p['drug_class']]['hits'] += 1

    for cls, stats in sorted(class_stats.items(), key=lambda x: -x[1]['total']):
        prec_cls = stats['hits'] / stats['total'] * 100 if stats['total'] > 0 else 0
        if stats['total'] >= 5:
            print(f"  {cls:<25s}: {stats['hits']}/{stats['total']} = {prec_cls:.1f}%")

    # Stratify by category_specific_tier (cancer_same_type vs other)
    print(f"\n--- By Category-Specific Tier Rule ---")
    rule_stats = defaultdict(lambda: {'hits': 0, 'total': 0})
    for p in cancer_medium_preds:
        rule = p['category_specific'] or 'default_medium'
        rule_stats[rule]['total'] += 1
        if p['is_hit']:
            rule_stats[rule]['hits'] += 1

    for rule, stats in sorted(rule_stats.items(), key=lambda x: -x[1]['total']):
        prec_r = stats['hits'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  {rule:<30s}: {stats['hits']}/{stats['total']} = {prec_r:.1f}%")

    # === HOLDOUT ===
    print("\n" + "=" * 80)
    print("HOLDOUT: Cancer Drug MEDIUM Stratification (5 seeds)")
    print("=" * 80)

    holdout_subtype = defaultdict(list)
    holdout_subtype_n = defaultdict(list)
    holdout_drug_class = defaultdict(list)
    holdout_drug_class_n = defaultdict(list)
    holdout_soc = defaultdict(list)
    holdout_soc_n = defaultdict(list)

    for seed_idx, seed in enumerate(seeds):
        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        originals = recompute_gt_structures(predictor, set(train_ids))

        subtype_stats = defaultdict(lambda: {'hits': 0, 'total': 0})
        dclass_stats = defaultdict(lambda: {'hits': 0, 'total': 0})
        soc_stats = defaultdict(lambda: {'hits': 0, 'total': 0})

        for disease_id in holdout_ids:
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

                drug_name = predictor.drug_id_to_name.get(pred.drug_id, pred.drug_id)
                is_hit = (disease_id, pred.drug_id) in gt_set
                cancer_subtype = classify_cancer_subtype(disease_name)
                drug_class = classify_cancer_drug(drug_name)
                is_soc = drug_name.lower() in cancer_drug_names

                subtype_stats[cancer_subtype]['total'] += 1
                if is_hit:
                    subtype_stats[cancer_subtype]['hits'] += 1

                dclass_stats[drug_class]['total'] += 1
                if is_hit:
                    dclass_stats[drug_class]['hits'] += 1

                soc_key = 'cancer_soc' if is_soc else 'non_soc'
                soc_stats[soc_key]['total'] += 1
                if is_hit:
                    soc_stats[soc_key]['hits'] += 1

        for subtype, stats in subtype_stats.items():
            prec_h = stats['hits'] / stats['total'] * 100 if stats['total'] > 0 else 0
            holdout_subtype[subtype].append(prec_h)
            holdout_subtype_n[subtype].append(stats['total'])

        for dclass, stats in dclass_stats.items():
            prec_h = stats['hits'] / stats['total'] * 100 if stats['total'] > 0 else 0
            holdout_drug_class[dclass].append(prec_h)
            holdout_drug_class_n[dclass].append(stats['total'])

        for soc_key, stats in soc_stats.items():
            prec_h = stats['hits'] / stats['total'] * 100 if stats['total'] > 0 else 0
            holdout_soc[soc_key].append(prec_h)
            holdout_soc_n[soc_key].append(stats['total'])

        restore_gt_structures(predictor, originals)

    print("\n--- By Cancer Subtype (Holdout) ---")
    for subtype in ['solid', 'hematologic', 'cns']:
        if subtype in holdout_subtype:
            vals = holdout_subtype[subtype]
            ns = holdout_subtype_n[subtype]
            print(f"  {subtype:<15s}: {np.mean(vals):5.1f}% ± {np.std(vals):4.1f}% (n={np.mean(ns):.0f}/seed)")

    print("\n--- By Drug Class (Holdout) ---")
    for dclass in sorted(holdout_drug_class.keys()):
        vals = holdout_drug_class[dclass]
        ns = holdout_drug_class_n[dclass]
        if np.mean(ns) >= 3:
            print(f"  {dclass:<25s}: {np.mean(vals):5.1f}% ± {np.std(vals):4.1f}% (n={np.mean(ns):.0f}/seed)")

    print("\n--- SOC Cancer Drugs vs Non-SOC (Holdout) ---")
    for soc_key in ['cancer_soc', 'non_soc']:
        if soc_key in holdout_soc:
            vals = holdout_soc[soc_key]
            ns = holdout_soc_n[soc_key]
            print(f"  {soc_key:<15s}: {np.mean(vals):5.1f}% ± {np.std(vals):4.1f}% (n={np.mean(ns):.0f}/seed)")

    # === CONCLUSION ===
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    # Check if any subtype reaches HIGH-level (>50% holdout)
    any_promotion = False
    for subtype, vals in holdout_subtype.items():
        if np.mean(vals) > 40:
            print(f"  ** {subtype} cancer MEDIUM: {np.mean(vals):.1f}% → HIGH candidate **")
            any_promotion = True

    for dclass, vals in holdout_drug_class.items():
        if np.mean(vals) > 40 and np.mean(holdout_drug_class_n[dclass]) >= 5:
            print(f"  ** {dclass} cancer MEDIUM: {np.mean(vals):.1f}% → HIGH candidate **")
            any_promotion = True

    if not any_promotion:
        print("  No cancer MEDIUM subgroup reaches HIGH-level precision (>40%)")
        print("  Cancer drug SOC promotion NOT recommended")

    # Save results
    output_path = Path("data/analysis/h521_cancer_soc_output.json")
    with open(output_path, "w") as f:
        json.dump({
            "full_data_total": total,
            "full_data_precision": prec_full,
            "holdout_subtype": {k: {"mean": np.mean(v), "std": np.std(v), "avg_n": np.mean(holdout_subtype_n[k])}
                               for k, v in holdout_subtype.items()},
            "any_promotion_candidate": any_promotion,
        }, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
