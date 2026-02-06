#!/usr/bin/env python3
"""
h604: Standard MEDIUM Infectious Drug-Class Stratification

Standard MEDIUM infectious is the largest sub-category (314 preds, 22.7% holdout).
Can we find specific drug classes that drag down the average?

Approach:
1. Classify all standard MEDIUM infectious drugs by class
2. Run per-drug holdout analysis
3. Identify drugs/classes with <15% holdout (LOW-quality within MEDIUM)
"""

import sys
from collections import defaultdict
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


def split_diseases(all_diseases, seed, train_ratio=0.8):
    rng = np.random.RandomState(seed)
    shuffled = list(all_diseases)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def recompute_gt_structures(predictor, train_disease_ids):
    """Match h393 logic exactly."""
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
    for d in train_disease_ids:
        if d in predictor.ground_truth:
            for drug in predictor.ground_truth[d]:
                new_freq[drug] += 1
    predictor.drug_train_freq = dict(new_freq)

    new_d2d = defaultdict(set)
    for d in train_disease_ids:
        if d in predictor.ground_truth:
            dn = predictor.disease_names.get(d, d)
            for drug in predictor.ground_truth[d]:
                new_d2d[drug].add(dn.lower())
    predictor.drug_to_diseases = dict(new_d2d)

    new_cancer = defaultdict(set)
    for d in train_disease_ids:
        if d in predictor.ground_truth:
            dn = predictor.disease_names.get(d, d)
            ct = extract_cancer_types(dn)
            if ct:
                for drug in predictor.ground_truth[d]:
                    new_cancer[drug].update(ct)
    predictor.drug_cancer_types = dict(new_cancer)

    new_groups = defaultdict(set)
    for d in train_disease_ids:
        if d in predictor.ground_truth:
            dn = predictor.disease_names.get(d, d).lower()
            for cat, groups in DISEASE_HIERARCHY_GROUPS.items():
                for gname, keywords in groups.items():
                    excl = HIERARCHY_EXCLUSIONS.get((cat, gname), [])
                    if any(e in dn for e in excl):
                        continue
                    if any(k in dn or dn in k for k in keywords):
                        for drug in predictor.ground_truth[d]:
                            new_groups[drug].add((cat, gname))
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


def restore(predictor, originals):
    for k, v in originals.items():
        setattr(predictor, k, v)


# Drug classification for infectious context
DRUG_CLASSES = {
    'corticosteroid': [
        'dexamethasone', 'prednisone', 'prednisolone', 'methylprednisolone',
        'hydrocortisone', 'betamethasone', 'triamcinolone', 'budesonide',
        'fluticasone', 'beclomethasone', 'mometasone',
    ],
    'macrolide_antibiotic': [
        'erythromycin', 'azithromycin', 'clarithromycin',
    ],
    'fluoroquinolone': [
        'ciprofloxacin', 'levofloxacin', 'moxifloxacin', 'ofloxacin', 'norfloxacin',
        'gemifloxacin', 'gatifloxacin',
    ],
    'beta_lactam': [
        'amoxicillin', 'ampicillin', 'penicillin', 'piperacillin',
        'cephalexin', 'ceftriaxone', 'cefazolin', 'cefepime', 'ceftazidime',
        'cefuroxime', 'cefaclor', 'cefadroxil', 'cefdinir', 'cefditoren',
        'cefotaxime', 'ceftizoxime', 'meropenem', 'imipenem', 'ertapenem',
    ],
    'aminoglycoside': [
        'gentamicin', 'tobramycin', 'amikacin', 'streptomycin',
    ],
    'tetracycline': [
        'doxycycline', 'tetracycline', 'minocycline', 'demeclocycline', 'oxytetracycline',
    ],
    'antifungal': [
        'fluconazole', 'itraconazole', 'ketoconazole', 'voriconazole',
        'posaconazole', 'caspofungin', 'micafungin', 'anidulafungin',
        'clotrimazole', 'miconazole', 'terbinafine', 'nystatin', 'griseofulvin',
        'amphotericin b', 'flucytosine',
    ],
    'antiviral': [
        'acyclovir', 'valacyclovir', 'ganciclovir', 'valganciclovir',
        'oseltamivir', 'zanamivir', 'ribavirin', 'sofosbuvir', 'tenofovir',
        'lamivudine', 'entecavir', 'adefovir', 'remdesivir',
        'lopinavir', 'ritonavir', 'darunavir', 'atazanavir',
        'efavirenz', 'nevirapine', 'dolutegravir', 'raltegravir',
        'emtricitabine', 'abacavir', 'zidovudine', 'didanosine',
    ],
    'antiparasitic': [
        'chloroquine', 'hydroxychloroquine', 'mefloquine', 'primaquine',
        'ivermectin', 'albendazole', 'mebendazole', 'praziquantel',
        'pyrimethamine', 'atovaquone', 'pentamidine', 'metronidazole',
        'quinine', 'artesunate', 'artemether',
    ],
    'sulfonamide': [
        'sulfamethoxazole', 'trimethoprim', 'sulfadiazine',
    ],
    'other_antibiotic': [
        'vancomycin', 'clindamycin', 'nitrofurantoin', 'linezolid', 'daptomycin',
        'colistin', 'polymyxin', 'fosfomycin', 'chloramphenicol',
        'cycloserine', 'bedaquiline', 'tigecycline', 'rifampin', 'rifampicin',
        'isoniazid', 'pyrazinamide', 'ethambutol',
    ],
}


def classify_drug(drug_name):
    """Classify a drug into a class for infectious analysis."""
    dl = drug_name.lower()
    for cls, drugs in DRUG_CLASSES.items():
        if any(d in dl for d in drugs):
            return cls
    return 'other'


def is_standard_medium_infectious(pred):
    return (pred.confidence_tier == ConfidenceTier.MEDIUM
            and pred.category_specific_tier is None
            and pred.category == 'infectious')


def main():
    print("=" * 80)
    print("h604: Standard MEDIUM Infectious Drug-Class Stratification")
    print("=" * 80)

    predictor = DrugRepurposingPredictor()
    all_disease_ids = list(predictor.ground_truth.keys())

    # ===== STEP 1: Full-data analysis by drug class =====
    print("\n--- Step 1: Full-data standard MEDIUM infectious by drug class ---")

    drug_class_preds = defaultdict(list)
    per_drug_preds = defaultdict(list)

    for disease_id in all_disease_ids:
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        try:
            result = predictor.predict(disease_name, top_n=30, include_filtered=True)
        except Exception:
            continue

        gt_drugs = set(predictor.ground_truth[disease_id])
        for pred in result.predictions:
            if not is_standard_medium_infectious(pred):
                continue
            is_hit = pred.drug_id in gt_drugs
            drug_cls = classify_drug(pred.drug_name)
            drug_class_preds[drug_cls].append({
                'drug_name': pred.drug_name,
                'drug_id': pred.drug_id,
                'disease_name': disease_name,
                'disease_id': disease_id,
                'is_hit': is_hit,
                'rank': pred.rank,
            })
            per_drug_preds[pred.drug_name.lower()].append({
                'disease_name': disease_name,
                'is_hit': is_hit,
                'rank': pred.rank,
            })

    total = sum(len(v) for v in drug_class_preds.values())
    print(f"\nTotal standard MEDIUM infectious: {total}")

    print(f"\n{'Drug Class':<25} {'Preds':>6} {'Hits':>5} {'Precision':>10}")
    print("-" * 50)
    for cls in sorted(drug_class_preds.keys(), key=lambda c: -len(drug_class_preds[c])):
        preds = drug_class_preds[cls]
        hits = sum(1 for p in preds if p['is_hit'])
        prec = hits / len(preds) * 100 if preds else 0
        print(f"  {cls:<23} {len(preds):>6} {hits:>5} {prec:>9.1f}%")

    # ===== STEP 2: Per-drug analysis (top drugs by prediction count) =====
    print("\n\n--- Step 2: Per-drug analysis (drugs with >=5 predictions) ---")
    print(f"\n{'Drug':<30} {'Preds':>6} {'Hits':>5} {'Precision':>10} {'Class':<20}")
    print("-" * 75)
    for drug in sorted(per_drug_preds.keys(), key=lambda d: -len(per_drug_preds[d])):
        preds = per_drug_preds[drug]
        if len(preds) < 3:
            continue
        hits = sum(1 for p in preds if p['is_hit'])
        prec = hits / len(preds) * 100 if preds else 0
        cls = classify_drug(drug)
        marker = " ***" if prec < 15 and len(preds) >= 5 else ""
        print(f"  {drug:<28} {len(preds):>6} {hits:>5} {prec:>9.1f}% {cls:<20}{marker}")

    # ===== STEP 3: Holdout validation by drug class =====
    print("\n\n" + "=" * 80)
    print("Step 3: 5-seed holdout validation by drug class")
    print("=" * 80)

    seeds = [42, 123, 456, 789, 2024]
    class_holdout = defaultdict(list)  # cls -> list of (hits, total)
    drug_holdout = defaultdict(list)  # drug -> list of (hits, total)
    overall_holdout = []

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        train_d, holdout_d = split_diseases(all_disease_ids, seed)
        train_set = set(train_d)
        originals = recompute_gt_structures(predictor, train_set)

        cls_hits = defaultdict(int)
        cls_total = defaultdict(int)
        drug_hits_s = defaultdict(int)
        drug_total_s = defaultdict(int)
        ov_hits = 0
        ov_total = 0

        for disease_id in holdout_d:
            if disease_id not in predictor.ground_truth:
                continue
            gt_drugs = set(predictor.ground_truth[disease_id])
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            try:
                result = predictor.predict(disease_name, top_n=30, include_filtered=True)
            except Exception:
                continue

            for pred in result.predictions:
                if not is_standard_medium_infectious(pred):
                    continue
                is_hit = pred.drug_id in gt_drugs
                cls = classify_drug(pred.drug_name)
                cls_hits[cls] += int(is_hit)
                cls_total[cls] += 1
                dn = pred.drug_name.lower()
                drug_hits_s[dn] += int(is_hit)
                drug_total_s[dn] += 1
                ov_hits += int(is_hit)
                ov_total += 1

        overall_holdout.append((ov_hits, ov_total))
        for cls in set(cls_total.keys()):
            class_holdout[cls].append((cls_hits[cls], cls_total[cls]))
        for dn in set(drug_total_s.keys()):
            drug_holdout[dn].append((drug_hits_s[dn], drug_total_s[dn]))

        for cls in sorted(cls_total.keys()):
            h, t = cls_hits[cls], cls_total[cls]
            prec = h / t * 100 if t > 0 else 0
            print(f"  {cls:<25} {h:>3}/{t:<4} = {prec:>5.1f}%")

        if ov_total > 0:
            print(f"  {'OVERALL':<25} {ov_hits:>3}/{ov_total:<4} = {ov_hits/ov_total*100:>5.1f}%")

        restore(predictor, originals)

    # ===== STEP 4: Summary =====
    print("\n\n" + "=" * 80)
    print("Step 4: Holdout Summary by Drug Class")
    print("=" * 80)

    def summarize(results_list):
        precisions = []
        total_n = 0
        for hits, total in results_list:
            precisions.append(hits / total * 100 if total > 0 else 0)
            total_n += total
        mean = np.mean(precisions)
        std = np.std(precisions)
        mean_n = total_n / len(results_list)
        return mean, std, mean_n

    print(f"\n{'Drug Class':<25} {'Holdout %':>10} {'±':>3} {'Std':>6} {'n/seed':>8} {'Full%':>8}")
    print("-" * 65)

    mean, std, mn = summarize(overall_holdout)
    print(f"{'ALL INFECTIOUS STD':<25} {mean:>10.1f} {'±':>3} {std:>6.1f} {mn:>8.1f}")
    print()

    cls_summaries = {}
    for cls in sorted(class_holdout.keys()):
        results = class_holdout[cls]
        m, s, n = summarize(results)
        # Get full-data precision
        fd_preds = drug_class_preds.get(cls, [])
        fd_hits = sum(1 for p in fd_preds if p['is_hit'])
        fd_prec = fd_hits / len(fd_preds) * 100 if fd_preds else 0
        cls_summaries[cls] = (m, s, n, fd_prec, len(fd_preds))

    for cls in sorted(cls_summaries.keys(), key=lambda c: -cls_summaries[c][2]):
        m, s, n, fd, fd_n = cls_summaries[cls]
        marker = " ***" if m < 15 and n >= 5 else ""
        print(f"  {cls:<23} {m:>10.1f} {'±':>3} {s:>6.1f} {n:>8.1f} {fd:>7.1f}%{marker}")

    # ===== STEP 5: Per-drug holdout (top drugs) =====
    print("\n\n" + "=" * 80)
    print("Step 5: Per-Drug Holdout (drugs with avg n>=2/seed)")
    print("=" * 80)

    print(f"\n{'Drug':<30} {'Holdout %':>10} {'±':>3} {'Std':>6} {'n/seed':>8} {'Class':<15}")
    print("-" * 75)

    drug_summaries = {}
    for dn in sorted(drug_holdout.keys()):
        results = drug_holdout[dn]
        m, s, n = summarize(results)
        if n >= 2:
            cls = classify_drug(dn)
            drug_summaries[dn] = (m, s, n, cls)

    for dn in sorted(drug_summaries.keys(), key=lambda d: -drug_summaries[d][2]):
        m, s, n, cls = drug_summaries[dn]
        marker = " ***" if m < 10 and n >= 3 else ""
        print(f"  {dn:<28} {m:>10.1f} {'±':>3} {s:>6.1f} {n:>8.1f} {cls:<15}{marker}")


if __name__ == "__main__":
    main()
