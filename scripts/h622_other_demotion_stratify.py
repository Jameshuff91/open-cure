#!/usr/bin/env python3
"""
h622: Expanded GT Recalibration of Other Demoted Categories

Test whether drug-class stratification (like h618 for CV) can identify
genuinely MEDIUM-quality drug subsets within other MEDIUM_DEMOTED_CATEGORIES.

Focus on neurological and hematological (most predictions).
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from production_predictor import (
    DrugRepurposingPredictor,
    DISEASE_HIERARCHY_GROUPS,
    extract_cancer_types,
)


# Known drug classes for neurological
NEURO_DRUGS = {
    'anticonvulsant': ['carbamazepine', 'phenobarbital', 'diazepam', 'cannabidiol',
                       'valproic', 'lamotrigine', 'levetiracetam', 'phenytoin',
                       'topiramate', 'gabapentin', 'pregabalin', 'clonazepam',
                       'lorazepam', 'oxcarbazepine', 'lacosamide', 'zonisamide'],
    'antiparkinsonian': ['bromocriptine', 'amantadine', 'levodopa', 'pramipexole',
                         'ropinirole', 'selegiline', 'rasagiline', 'entacapone',
                         'droxidopa', 'rotigotine'],
    'muscle_relaxant': ['baclofen', 'dantrolene', 'tizanidine', 'cyclobenzaprine'],
    'neuro_other': ['everolimus', 'modafinil', 'riluzole', 'edaravone',
                    'memantine', 'donepezil', 'galantamine', 'rivastigmine'],
}

# Known drug classes for hematological
HEMO_DRUGS = {
    'enzyme_replacement': ['imiglucerase', 'velaglucerase', 'taliglucerase'],
    'complement_inhibitor': ['eculizumab', 'ravulizumab'],
    'thrombopoietin_agonist': ['eltrombopag', 'romiplostim'],
    'growth_factor': ['filgrastim', 'pegfilgrastim', 'erythropoietin', 'epoetin'],
    'antineoplastic_heme': ['fludarabine', 'hydroxyurea', 'imatinib', 'rituximab'],
    'supplement': ['cyanocobalamin', 'phylloquinone', 'folic acid', 'deferoxamine'],
}


def classify_neuro(drug_name):
    dl = drug_name.lower()
    for cls, drugs in NEURO_DRUGS.items():
        if any(d in dl for d in drugs):
            return cls
    return 'non_neuro'


def classify_hemo(drug_name):
    dl = drug_name.lower()
    for cls, drugs in HEMO_DRUGS.items():
        if any(d in dl for d in drugs):
            return cls
    return 'non_hemo'


def split_diseases(all_diseases, seed, train_ratio=0.8):
    rng = np.random.RandomState(seed)
    shuffled = list(all_diseases)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def recompute_gt_structures(predictor, train_disease_ids):
    originals = {
        "drug_train_freq": dict(predictor.drug_train_freq),
        "drug_to_diseases": {k: set(v) for k, v in predictor.drug_to_diseases.items()},
        "drug_cancer_types": {k: set(v) for k, v in predictor.drug_cancer_types.items()},
        "drug_disease_groups": {k: dict(v) for k, v in predictor.drug_disease_groups.items()},
    }

    new_freq = defaultdict(int)
    new_d2d = defaultdict(set)
    new_cancer_types = defaultdict(set)
    new_disease_groups = defaultdict(lambda: defaultdict(set))

    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            for drug_id in predictor.ground_truth[disease_id]:
                new_freq[drug_id] += 1
                new_d2d[drug_id].add(disease_name)
                for ct in extract_cancer_types(disease_name):
                    new_cancer_types[drug_id].add(ct)
                for group_name, group_info in DISEASE_HIERARCHY_GROUPS.items():
                    parent_kws = group_info.get("parent_keywords", [])
                    if any(kw.lower() in disease_name.lower() for kw in parent_kws):
                        new_disease_groups[drug_id][group_name].add(disease_name)

    predictor.drug_train_freq = dict(new_freq)
    predictor.drug_to_diseases = dict(new_d2d)
    predictor.drug_cancer_types = dict(new_cancer_types)
    predictor.drug_disease_groups = dict(new_disease_groups)
    return originals


def restore_gt_structures(predictor, originals):
    for k, v in originals.items():
        setattr(predictor, k, v)


def run_category_analysis(predictor, gt_data, all_diseases, category, classify_fn, label):
    print(f"\n{'=' * 70}")
    print(f"{label.upper()} MEDIUM DEMOTION: DRUG-CLASS STRATIFICATION")
    print(f"{'=' * 70}")

    # Full-data: identify demotion predictions
    demotion_preds = []
    reason = f'{category}_medium_demotion'

    for disease_id in all_diseases:
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        cat = predictor.categorize_disease(disease_name)
        if cat != category:
            continue
        try:
            result = predictor.predict(disease_name, top_n=30, include_filtered=True)
        except Exception:
            continue
        for pred in result.predictions:
            if pred.category_specific_tier == reason:
                drug_class = classify_fn(pred.drug_name)
                demotion_preds.append({
                    'drug_name': pred.drug_name,
                    'drug_id': pred.drug_id,
                    'disease_name': disease_name,
                    'disease_id': disease_id,
                    'drug_class': drug_class,
                    'rank': pred.rank,
                    'mechanism_support': pred.mechanism_support,
                    'train_frequency': pred.train_frequency,
                })

    print(f"Total {reason}: {len(demotion_preds)}")

    class_counts = defaultdict(list)
    for p in demotion_preds:
        class_counts[p['drug_class']].append(p)

    print(f"\nDrug class distribution:")
    for cls, preds in sorted(class_counts.items(), key=lambda x: -len(x[1])):
        drugs = set(p['drug_name'] for p in preds)
        print(f"  {cls:25s}: {len(preds):3d} preds, {len(drugs):2d} drugs")

    # 5-seed holdout
    seeds = [42, 123, 456, 789, 2024]
    class_seed_results = defaultdict(lambda: defaultdict(list))
    overall_seed_results = defaultdict(list)

    for seed_idx, seed in enumerate(seeds):
        train_diseases, holdout_diseases = split_diseases(all_diseases, seed)
        train_set = set(train_diseases)
        originals = recompute_gt_structures(predictor, train_set)

        for disease_id in holdout_diseases:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            cat = predictor.categorize_disease(disease_name)
            if cat != category:
                continue
            try:
                result = predictor.predict(disease_name, top_n=30, include_filtered=True)
            except Exception:
                continue

            gt_drugs = set(gt_data.get(disease_id, []))
            for pred in result.predictions:
                if pred.category_specific_tier == reason:
                    drug_class = classify_fn(pred.drug_name)
                    is_hit = pred.drug_id in gt_drugs
                    class_seed_results[drug_class][seed].append(is_hit)
                    overall_seed_results[seed].append(is_hit)

        restore_gt_structures(predictor, originals)

    # Summarize
    print(f"\n{'Drug Class':<28s} {'Holdout%':>8s} {'±Std':>6s} {'N/seed':>7s} {'Preds':>5s}")
    print("-" * 60)

    for cls in sorted(class_seed_results.keys()):
        seed_precs = []
        seed_ns = []
        for seed in seeds:
            r = class_seed_results[cls][seed]
            if r:
                seed_precs.append(100 * sum(r) / len(r))
                seed_ns.append(len(r))
            else:
                seed_precs.append(0)
                seed_ns.append(0)

        mean_prec = np.mean(seed_precs)
        std_prec = np.std(seed_precs)
        mean_n = np.mean(seed_ns)
        n_preds = len(class_counts.get(cls, []))
        print(f"  {cls:<26s} {mean_prec:7.1f}% {std_prec:5.1f}% {mean_n:6.1f} {n_preds:5d}")

    overall_precs = [100 * sum(overall_seed_results[s]) / len(overall_seed_results[s])
                     for s in seeds if overall_seed_results[s]]
    print("-" * 60)
    if overall_precs:
        print(f"  {'OVERALL':<26s} {np.mean(overall_precs):7.1f}% {np.std(overall_precs):5.1f}%")


def main():
    print("=" * 70)
    print("h622: Expanded GT Recalibration of Other Demoted Categories")
    print("=" * 70)

    predictor = DrugRepurposingPredictor()
    gt_path = predictor.reference_dir / "expanded_ground_truth.json"
    with open(gt_path) as f:
        gt_data = json.load(f)
    print(f"Expanded GT: {len(gt_data)} diseases, {sum(len(v) for v in gt_data.values())} pairs")

    all_diseases = sorted(
        d for d in predictor.disease_names
        if d in predictor.embeddings and d in gt_data
    )
    print(f"Diseases: {len(all_diseases)}")

    # Test neurological
    run_category_analysis(predictor, gt_data, all_diseases,
                          'neurological', classify_neuro, 'Neurological')

    # Test hematological
    run_category_analysis(predictor, gt_data, all_diseases,
                          'hematological', classify_hemo, 'Hematological')


if __name__ == "__main__":
    main()
