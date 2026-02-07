#!/usr/bin/env python3
"""
h618: CV Medium Demotion Reversal — Drug-Class Stratification

h462 demoted ALL cardiovascular MEDIUM→LOW. h615 found cardiovascular_medium_demotion
holdout = 25.1% ± 19.4% (n=69/seed) — above MEDIUM boundary but very high variance.

This script stratifies the 319 demoted CV predictions by drug class to identify:
- Drug classes with holdout > 30% (genuine MEDIUM, promotable)
- Drug classes with holdout < 15% (genuine LOW, keep demoted)
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from production_predictor import (
    ConfidenceTier,
    DrugRepurposingPredictor,
    DISEASE_HIERARCHY_GROUPS,
    extract_cancer_types,
)


def classify_cv_drug(drug_name: str) -> str:
    """Classify a CV drug into pharmacological class based on name patterns."""
    drug_lower = drug_name.lower()

    # ACE inhibitors
    if any(d in drug_lower for d in ['enalapril', 'lisinopril', 'ramipril', 'captopril',
            'perindopril', 'quinapril', 'benazepril', 'fosinopril', 'trandolapril', 'moexipril']):
        return 'ACE_inhibitor'

    # ARBs
    if any(d in drug_lower for d in ['losartan', 'valsartan', 'irbesartan', 'candesartan',
            'telmisartan', 'olmesartan', 'eprosartan', 'azilsartan']):
        return 'ARB'

    # Beta-blockers
    if any(d in drug_lower for d in ['metoprolol', 'atenolol', 'propranolol', 'carvedilol',
            'bisoprolol', 'nebivolol', 'labetalol', 'nadolol', 'sotalol', 'timolol',
            'acebutolol', 'pindolol', 'esmolol']):
        return 'beta_blocker'

    # CCBs
    if any(d in drug_lower for d in ['amlodipine', 'nifedipine', 'diltiazem', 'verapamil',
            'felodipine', 'nicardipine', 'nimodipine', 'isradipine', 'clevidipine']):
        return 'CCB'

    # Diuretics
    if any(d in drug_lower for d in ['hydrochlorothiazide', 'furosemide', 'spironolactone',
            'chlorthalidone', 'bumetanide', 'torsemide', 'amiloride', 'triamterene',
            'indapamide', 'metolazone', 'eplerenone']):
        return 'diuretic'

    # Statins
    if any(d in drug_lower for d in ['atorvastatin', 'simvastatin', 'rosuvastatin',
            'pravastatin', 'lovastatin', 'fluvastatin', 'pitavastatin']):
        return 'statin'

    # Antiarrhythmics
    if any(d in drug_lower for d in ['amiodarone', 'flecainide', 'propafenone', 'dronedarone',
            'dofetilide', 'ibutilide', 'mexiletine', 'procainamide', 'quinidine', 'disopyramide']):
        return 'antiarrhythmic'

    # Anticoagulants/antiplatelets
    if any(d in drug_lower for d in ['warfarin', 'heparin', 'enoxaparin', 'rivaroxaban',
            'apixaban', 'dabigatran', 'edoxaban', 'clopidogrel', 'ticagrelor', 'prasugrel',
            'aspirin', 'dipyridamole', 'cilostazol', 'fondaparinux']):
        return 'anticoagulant_antiplatelet'

    # Nitrates/vasodilators
    if any(d in drug_lower for d in ['nitroglycerin', 'isosorbide', 'hydralazine',
            'minoxidil', 'nitroprusside']):
        return 'nitrate_vasodilator'

    # SGLT2 inhibitors
    if any(d in drug_lower for d in ['empagliflozin', 'dapagliflozin', 'canagliflozin']):
        return 'SGLT2_inhibitor'

    # Fibrates
    if any(d in drug_lower for d in ['fenofibrate', 'gemfibrozil', 'bezafibrate']):
        return 'fibrate'

    # Alpha blockers
    if any(d in drug_lower for d in ['doxazosin', 'prazosin', 'terazosin']):
        return 'alpha_blocker'

    # Centrally acting
    if any(d in drug_lower for d in ['clonidine', 'methyldopa', 'guanfacine']):
        return 'centrally_acting'

    # Corticosteroids
    if any(d in drug_lower for d in ['dexamethasone', 'prednisone', 'prednisolone',
            'methylprednisolone', 'hydrocortisone', 'betamethasone', 'budesonide',
            'triamcinolone', 'cortisone']):
        return 'corticosteroid'

    # Cardiac glycosides
    if 'digoxin' in drug_lower or 'digitoxin' in drug_lower:
        return 'cardiac_glycoside'

    # PDE inhibitors
    if any(d in drug_lower for d in ['milrinone', 'sildenafil', 'tadalafil']):
        return 'PDE_inhibitor'

    # ERA
    if any(d in drug_lower for d in ['bosentan', 'ambrisentan', 'macitentan']):
        return 'ERA'

    # Antianginal
    if 'ranolazine' in drug_lower or 'ivabradine' in drug_lower:
        return 'antianginal'

    # NSAIDs (sometimes in CV demotion)
    if any(d in drug_lower for d in ['aspirin', 'ibuprofen', 'naproxen', 'celecoxib',
            'indomethacin']):
        return 'NSAID'

    return 'other_CV'


def split_diseases(all_diseases, seed, train_ratio=0.8):
    rng = np.random.RandomState(seed)
    shuffled = list(all_diseases)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def recompute_gt_structures(predictor, train_disease_ids):
    """Recompute GT-derived structures from training diseases only."""
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

                cancer_types_found = extract_cancer_types(disease_name)
                for ct in cancer_types_found:
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
    predictor.drug_train_freq = originals["drug_train_freq"]
    predictor.drug_to_diseases = originals["drug_to_diseases"]
    predictor.drug_cancer_types = originals["drug_cancer_types"]
    predictor.drug_disease_groups = originals["drug_disease_groups"]


def main():
    print("=" * 70)
    print("h618: CV Medium Demotion Reversal — Drug-Class Stratification")
    print("=" * 70)

    print("\nLoading predictor...")
    predictor = DrugRepurposingPredictor()

    # Load expanded GT
    gt_path = predictor.reference_dir / "expanded_ground_truth.json"
    with open(gt_path) as f:
        gt_data = json.load(f)
    print(f"Loaded expanded GT: {len(gt_data)} diseases, {sum(len(v) for v in gt_data.values())} pairs")

    # Get all diseases with both GT and embeddings
    all_diseases = sorted(
        d for d in predictor.disease_names
        if d in predictor.embeddings and d in gt_data
    )
    print(f"Diseases with GT + embeddings: {len(all_diseases)}")

    # First pass: identify all cardiovascular_medium_demotion predictions (full data)
    print("\n--- Full-data pass: Identify CV medium demotion predictions ---")
    cv_demotion_preds = []

    for disease_id in all_diseases:
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        category = predictor.categorize_disease(disease_name)
        if category != 'cardiovascular':
            continue

        try:
            result = predictor.predict(disease_name, top_n=30, include_filtered=True)
        except Exception as e:
            continue

        for pred in result.predictions:
            if pred.category_specific_tier == 'cardiovascular_medium_demotion':
                drug_class = classify_cv_drug(pred.drug_name)
                cv_demotion_preds.append({
                    'drug_name': pred.drug_name,
                    'drug_id': pred.drug_id,
                    'disease_name': disease_name,
                    'disease_id': disease_id,
                    'drug_class': drug_class,
                    'knn_rank': pred.rank,
                    'knn_score': pred.knn_score,
                    'mechanism_support': pred.mechanism_support,
                    'train_frequency': pred.train_frequency,
                })

    print(f"Total CV medium demotion predictions: {len(cv_demotion_preds)}")

    # Group by drug class
    class_counts = defaultdict(list)
    for p in cv_demotion_preds:
        class_counts[p['drug_class']].append(p)

    print(f"\nDrug class distribution:")
    for cls, preds in sorted(class_counts.items(), key=lambda x: -len(x[1])):
        drugs = set(p['drug_name'] for p in preds)
        avg_rank = np.mean([p['knn_rank'] for p in preds])
        mech_pct = 100 * sum(1 for p in preds if p['mechanism_support']) / len(preds)
        print(f"  {cls:30s}: {len(preds):3d} preds, {len(drugs):2d} drugs, "
              f"avg_rank={avg_rank:.1f}, mech={mech_pct:.0f}%")
        for drug in sorted(drugs):
            drug_preds = [p for p in preds if p['drug_name'] == drug]
            print(f"    {drug:40s}: {len(drug_preds):2d} preds, "
                  f"rank={np.mean([p['knn_rank'] for p in drug_preds]):.1f}, "
                  f"freq={drug_preds[0]['train_frequency']}")

    # 5-seed holdout evaluation per drug class
    print("\n" + "=" * 70)
    print("5-SEED HOLDOUT EVALUATION BY DRUG CLASS")
    print("=" * 70)

    seeds = [42, 123, 456, 789, 2024]
    class_seed_results = defaultdict(lambda: defaultdict(list))
    drug_seed_results = defaultdict(lambda: defaultdict(list))
    overall_seed_results = defaultdict(list)

    for seed_idx, seed in enumerate(seeds):
        print(f"\n--- Seed {seed} ({seed_idx + 1}/5) ---")

        train_diseases, holdout_diseases = split_diseases(all_diseases, seed)
        holdout_set = set(holdout_diseases)
        train_set = set(train_diseases)

        # Recompute GT structures from training diseases only
        originals = recompute_gt_structures(predictor, train_set)

        # Get predictions for holdout diseases (CV only)
        for disease_id in holdout_diseases:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            category = predictor.categorize_disease(disease_name)
            if category != 'cardiovascular':
                continue

            try:
                result = predictor.predict(disease_name, top_n=30, include_filtered=True)
            except Exception:
                continue

            # GT drugs for this holdout disease
            gt_drugs = set()
            if disease_id in gt_data:
                gt_drugs = set(gt_data[disease_id])

            for pred in result.predictions:
                if pred.category_specific_tier == 'cardiovascular_medium_demotion':
                    drug_class = classify_cv_drug(pred.drug_name)
                    is_hit = pred.drug_id in gt_drugs

                    class_seed_results[drug_class][seed].append(is_hit)
                    drug_seed_results[pred.drug_name][seed].append(is_hit)
                    overall_seed_results[seed].append(is_hit)

        # Restore GT
        restore_gt_structures(predictor, originals)

        total_n = len(overall_seed_results[seed])
        total_hits = sum(overall_seed_results[seed])
        if total_n > 0:
            print(f"  Overall: {total_hits}/{total_n} = {100*total_hits/total_n:.1f}% (n={total_n})")
        else:
            print(f"  Overall: no CV demotion predictions in holdout")

    # Summarize per drug class
    print("\n" + "=" * 70)
    print("PER-DRUG-CLASS HOLDOUT RESULTS (5-SEED)")
    print("=" * 70)
    print(f"{'Drug Class':<32s} {'Holdout%':>8s} {'±Std':>6s} {'N/seed':>7s} {'Drugs':>5s} {'Preds':>5s}")
    print("-" * 70)

    class_summary = []
    for cls in sorted(class_seed_results.keys()):
        seed_precisions = []
        seed_ns = []
        for seed in seeds:
            results = class_seed_results[cls][seed]
            if results:
                seed_precisions.append(100 * sum(results) / len(results))
                seed_ns.append(len(results))
            else:
                seed_precisions.append(0)
                seed_ns.append(0)

        mean_prec = np.mean(seed_precisions)
        std_prec = np.std(seed_precisions)
        mean_n = np.mean(seed_ns)
        n_drugs = len(set(p['drug_name'] for p in class_counts.get(cls, [])))
        n_preds = len(class_counts.get(cls, []))

        print(f"  {cls:<30s} {mean_prec:7.1f}% {std_prec:5.1f}% {mean_n:6.1f} {n_drugs:5d} {n_preds:5d}")
        class_summary.append({
            'class': cls,
            'holdout_mean': mean_prec,
            'holdout_std': std_prec,
            'n_per_seed': mean_n,
            'n_drugs': n_drugs,
            'n_preds': n_preds,
            'seed_precisions': seed_precisions,
        })

    # Overall
    overall_precs = []
    for seed in seeds:
        results = overall_seed_results[seed]
        if results:
            overall_precs.append(100 * sum(results) / len(results))
    print("-" * 70)
    if overall_precs:
        print(f"  {'OVERALL':<30s} {np.mean(overall_precs):7.1f}% {np.std(overall_precs):5.1f}%")

    # Per-drug results (>= 2 preds/seed for any visibility)
    print("\n" + "=" * 70)
    print("PER-DRUG HOLDOUT RESULTS (5-SEED, drugs with >=2 preds/seed)")
    print("=" * 70)
    print(f"{'Drug':<42s} {'Class':<22s} {'Holdout%':>8s} {'±Std':>6s} {'N/seed':>7s}")
    print("-" * 87)

    drug_summary = []
    for drug in sorted(drug_seed_results.keys()):
        seed_precisions = []
        seed_ns = []
        for seed in seeds:
            results = drug_seed_results[drug][seed]
            if results:
                seed_precisions.append(100 * sum(results) / len(results))
                seed_ns.append(len(results))
            else:
                seed_precisions.append(0)
                seed_ns.append(0)

        mean_n = np.mean(seed_ns)
        if mean_n < 2:
            continue

        mean_prec = np.mean(seed_precisions)
        std_prec = np.std(seed_precisions)
        drug_class = classify_cv_drug(drug)

        print(f"  {drug:<40s} {drug_class:<22s} {mean_prec:7.1f}% {std_prec:5.1f}% {mean_n:6.1f}")
        drug_summary.append({
            'drug': drug,
            'class': drug_class,
            'holdout_mean': mean_prec,
            'holdout_std': std_prec,
            'n_per_seed': mean_n,
        })

    # Analysis: identify promotable classes
    print("\n" + "=" * 70)
    print("ANALYSIS: PROMOTABLE VS DEMOTABLE DRUG CLASSES")
    print("=" * 70)

    promotable = [c for c in class_summary if c['holdout_mean'] >= 30 and c['n_per_seed'] >= 3]
    borderline = [c for c in class_summary if 15 <= c['holdout_mean'] < 30 and c['n_per_seed'] >= 3]
    low_quality = [c for c in class_summary if c['holdout_mean'] < 15 and c['n_per_seed'] >= 3]
    small_n = [c for c in class_summary if c['n_per_seed'] < 3]

    print(f"\nPromotable (>=30% holdout, n>=3/seed):")
    for c in sorted(promotable, key=lambda x: -x['holdout_mean']):
        print(f"  {c['class']:<30s}: {c['holdout_mean']:.1f}% ± {c['holdout_std']:.1f}% "
              f"(n={c['n_per_seed']:.1f}/seed, {c['n_preds']} preds)")

    print(f"\nBorderline (15-30% holdout, n>=3/seed):")
    for c in sorted(borderline, key=lambda x: -x['holdout_mean']):
        print(f"  {c['class']:<30s}: {c['holdout_mean']:.1f}% ± {c['holdout_std']:.1f}% "
              f"(n={c['n_per_seed']:.1f}/seed, {c['n_preds']} preds)")

    print(f"\nLow quality (<15% holdout, n>=3/seed):")
    for c in sorted(low_quality, key=lambda x: -x['holdout_mean']):
        print(f"  {c['class']:<30s}: {c['holdout_mean']:.1f}% ± {c['holdout_std']:.1f}% "
              f"(n={c['n_per_seed']:.1f}/seed, {c['n_preds']} preds)")

    print(f"\nSmall-n (n<3/seed, unreliable):")
    for c in sorted(small_n, key=lambda x: -x['holdout_mean']):
        print(f"  {c['class']:<30s}: {c['holdout_mean']:.1f}% ± {c['holdout_std']:.1f}% "
              f"(n={c['n_per_seed']:.1f}/seed, {c['n_preds']} preds)")

    # Impact estimate
    if promotable:
        total_promoted = sum(c['n_preds'] for c in promotable)
        avg_promoted_prec = np.mean([c['holdout_mean'] for c in promotable])
        print(f"\nImpact if promotable classes → MEDIUM:")
        print(f"  {total_promoted} predictions rescued (LOW → MEDIUM)")
        print(f"  Average holdout precision: {avg_promoted_prec:.1f}%")

    # Also check: what about the combo of drug class + mechanism support?
    print("\n" + "=" * 70)
    print("DRUG CLASS × MECHANISM SUPPORT HOLDOUT")
    print("=" * 70)

    for cls in sorted(class_seed_results.keys()):
        # Get full-data preds for this class to check mechanism
        cls_preds = class_counts.get(cls, [])
        mech_drugs = set(p['drug_name'] for p in cls_preds if p['mechanism_support'])
        no_mech_drugs = set(p['drug_name'] for p in cls_preds if not p['mechanism_support'])

        # For holdout: split results by mechanism
        mech_holdout = defaultdict(list)
        nomech_holdout = defaultdict(list)

        for seed in seeds:
            for drug, results_map in drug_seed_results.items():
                if classify_cv_drug(drug) != cls:
                    continue
                for is_hit in results_map.get(seed, []):
                    if drug in mech_drugs:
                        mech_holdout[seed].append(is_hit)
                    else:
                        nomech_holdout[seed].append(is_hit)

        mech_precs = []
        mech_ns = []
        for seed in seeds:
            r = mech_holdout[seed]
            if r:
                mech_precs.append(100 * sum(r) / len(r))
                mech_ns.append(len(r))
        nomech_precs = []
        nomech_ns = []
        for seed in seeds:
            r = nomech_holdout[seed]
            if r:
                nomech_precs.append(100 * sum(r) / len(r))
                nomech_ns.append(len(r))

        n_total = len(cls_preds)
        if n_total < 5:
            continue

        mech_str = f"{np.mean(mech_precs):.1f}% (n={np.mean(mech_ns):.1f})" if mech_precs else "N/A"
        nomech_str = f"{np.mean(nomech_precs):.1f}% (n={np.mean(nomech_ns):.1f})" if nomech_precs else "N/A"
        print(f"  {cls:<30s}: +mech={mech_str:<25s} -mech={nomech_str}")

    # Save results
    results = {
        'per_class': class_summary,
        'per_drug': drug_summary,
        'overall_holdout': {
            'mean': float(np.mean(overall_precs)) if overall_precs else 0,
            'std': float(np.std(overall_precs)) if overall_precs else 0,
        },
        'promotable_classes': [c['class'] for c in promotable],
        'promotable_preds': sum(c['n_preds'] for c in promotable) if promotable else 0,
    }

    output_path = Path(__file__).parent.parent / "data" / "analysis" / "h618_cv_demotion_stratify.json"
    output_path.parent.mkdir(exist_ok=True, parents=True)

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=convert)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
