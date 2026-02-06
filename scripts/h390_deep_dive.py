#!/usr/bin/env python3
"""
h390 deep dive: Investigate specific rule anomalies found in coverage analysis.

1. metabolic_hierarchy_diabetes at GOLDEN (32.4%) - should it be demoted?
2. psychiatric MEDIUM at 55.4% - what rules are these hitting? Can we promote?
3. incoherent_demotion at HIGH (33.9%) - which categories are dragging it down?
4. infectious LOW at 20.5% - any rescue signal?
5. cardiovascular LOW at 19.4% - any rescue signal?
"""

import json
import sys
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from production_predictor import (
    DrugRepurposingPredictor,
    ConfidenceTier,
)


def main():
    print("Loading predictor...")
    predictor = DrugRepurposingPredictor()

    gt_path = predictor.reference_dir / "expanded_ground_truth.json"
    with open(gt_path) as f:
        gt_data = json.load(f)

    all_diseases = [
        d for d in predictor.disease_names.keys()
        if d in predictor.ground_truth and d in predictor.embeddings
    ]

    # Collect detailed predictions
    preds_by_rule = defaultdict(list)
    preds_by_cat_tier = defaultdict(list)

    for disease_id in all_diseases:
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        gt_drugs = set()
        if disease_id in gt_data:
            for drug in gt_data[disease_id]:
                if isinstance(drug, str):
                    gt_drugs.add(drug)
                elif isinstance(drug, dict):
                    gt_drugs.add(drug.get('drug_id') or drug.get('drug'))

        try:
            result = predictor.predict(disease_name, top_n=30, include_filtered=True)
        except Exception:
            continue

        category = predictor.categorize_disease(disease_name)
        for pred in result.predictions:
            tier_name = pred.confidence_tier.name
            rule = pred.category_specific_tier or 'standard'
            is_gt = pred.drug_id in gt_drugs

            info = {
                "drug": pred.drug_name,
                "drug_id": pred.drug_id,
                "disease": disease_name,
                "category": category,
                "rank": pred.rank,
                "freq": pred.train_frequency,
                "mech": pred.mechanism_support,
                "tier": tier_name,
                "rule": rule,
                "is_gt": is_gt,
            }
            preds_by_rule[rule].append(info)
            preds_by_cat_tier[(category, tier_name)].append(info)

    # === 1. metabolic_hierarchy_diabetes at GOLDEN ===
    print("\n" + "="*70)
    print("=== 1. metabolic_hierarchy_diabetes (GOLDEN, 32.4%) ===")
    print("="*70)
    diabetes_preds = preds_by_rule.get('metabolic_hierarchy_diabetes', [])
    if diabetes_preds:
        gt_hits = [p for p in diabetes_preds if p['is_gt']]
        non_gt = [p for p in diabetes_preds if not p['is_gt']]
        print(f"Total: {len(diabetes_preds)}, GT hits: {len(gt_hits)}, Precision: {len(gt_hits)/len(diabetes_preds)*100:.1f}%")

        # Group by disease
        by_disease = defaultdict(list)
        for p in diabetes_preds:
            by_disease[p['disease']].append(p)

        print(f"\nBy disease ({len(by_disease)} diseases):")
        for disease in sorted(by_disease.keys()):
            dp = by_disease[disease]
            hits = sum(1 for p in dp if p['is_gt'])
            print(f"  {disease}: {hits}/{len(dp)} GT ({hits/len(dp)*100:.0f}%)")
            if hits > 0:
                for p in dp:
                    if p['is_gt']:
                        print(f"    ✓ {p['drug']} (freq={p['freq']}, rank={p['rank']})")

        # Show freq/rank distribution of GT vs non-GT
        gt_freqs = [p['freq'] for p in gt_hits]
        non_gt_freqs = [p['freq'] for p in non_gt]
        print(f"\nGT hit freq: {np.mean(gt_freqs):.1f} ± {np.std(gt_freqs):.1f}")
        print(f"Non-GT freq: {np.mean(non_gt_freqs):.1f} ± {np.std(non_gt_freqs):.1f}")

    # === 2. psychiatric MEDIUM (55.4%) ===
    print("\n" + "="*70)
    print("=== 2. psychiatric MEDIUM (55.4% precision) ===")
    print("="*70)
    psych_med = preds_by_cat_tier.get(('psychiatric', 'MEDIUM'), [])
    if psych_med:
        gt_hits = [p for p in psych_med if p['is_gt']]
        print(f"Total: {len(psych_med)}, GT hits: {len(gt_hits)}, Precision: {len(gt_hits)/len(psych_med)*100:.1f}%")

        # What rules assign MEDIUM?
        rule_counts = Counter(p['rule'] for p in psych_med)
        print(f"\nRules assigning MEDIUM:")
        for rule, count in rule_counts.most_common():
            rule_preds = [p for p in psych_med if p['rule'] == rule]
            rule_gt = sum(1 for p in rule_preds if p['is_gt'])
            print(f"  {rule}: {rule_gt}/{count} ({rule_gt/count*100:.1f}%)")

        # Drug patterns
        drug_counts = Counter(p['drug'] for p in gt_hits)
        print(f"\nTop drugs (GT hits):")
        for drug, count in drug_counts.most_common(10):
            print(f"  {drug}: {count} hits")

        # Disease patterns
        by_disease = defaultdict(list)
        for p in psych_med:
            by_disease[p['disease']].append(p)
        print(f"\nBy disease ({len(by_disease)} diseases):")
        for disease in sorted(by_disease.keys(), key=lambda d: -sum(1 for p in by_disease[d] if p['is_gt'])):
            dp = by_disease[disease]
            hits = sum(1 for p in dp if p['is_gt'])
            print(f"  {disease}: {hits}/{len(dp)} ({hits/len(dp)*100:.0f}%)")

    # === 3. incoherent_demotion at HIGH (33.9%) ===
    print("\n" + "="*70)
    print("=== 3. incoherent_demotion (HIGH, 33.9%) ===")
    print("="*70)
    incoh = preds_by_rule.get('incoherent_demotion', [])
    if incoh:
        gt_hits = [p for p in incoh if p['is_gt']]
        print(f"Total: {len(incoh)}, GT hits: {len(gt_hits)}, Precision: {len(gt_hits)/len(incoh)*100:.1f}%")

        # By category
        cat_counts = defaultdict(lambda: {"total": 0, "gt": 0})
        for p in incoh:
            cat_counts[p['category']]["total"] += 1
            if p['is_gt']:
                cat_counts[p['category']]["gt"] += 1

        print(f"\nBy category:")
        for cat in sorted(cat_counts.keys(), key=lambda c: -cat_counts[c]['total']):
            cc = cat_counts[cat]
            prec = cc['gt'] / cc['total'] * 100 if cc['total'] > 0 else 0
            print(f"  {cat}: {cc['gt']}/{cc['total']} ({prec:.1f}%)")

    # === 4. infectious LOW (20.5%) ===
    print("\n" + "="*70)
    print("=== 4. infectious LOW (20.5%, n=667) ===")
    print("="*70)
    inf_low = preds_by_cat_tier.get(('infectious', 'LOW'), [])
    if inf_low:
        gt_hits = [p for p in inf_low if p['is_gt']]
        print(f"Total: {len(inf_low)}, GT hits: {len(gt_hits)}")

        # Rule breakdown
        rule_counts = Counter(p['rule'] for p in inf_low)
        print(f"\nRules:")
        for rule, count in rule_counts.most_common():
            rule_gt = sum(1 for p in inf_low if p['rule'] == rule and p['is_gt'])
            print(f"  {rule}: {rule_gt}/{count} ({rule_gt/count*100:.1f}%)")

        # freq distribution
        gt_freqs = [p['freq'] for p in gt_hits]
        non_gt_freqs = [p['freq'] for p in inf_low if not p['is_gt']]
        print(f"\nGT freq: {np.mean(gt_freqs):.1f} ± {np.std(gt_freqs):.1f} (range: {min(gt_freqs)}-{max(gt_freqs)})")
        print(f"Non-GT freq: {np.mean(non_gt_freqs):.1f} ± {np.std(non_gt_freqs):.1f}")

        # By rank
        for r_start in [1, 6, 11, 16]:
            r_end = r_start + 4
            rank_preds = [p for p in inf_low if r_start <= p['rank'] <= r_end]
            rank_gt = sum(1 for p in rank_preds if p['is_gt'])
            if rank_preds:
                print(f"  Rank {r_start}-{r_end}: {rank_gt}/{len(rank_preds)} ({rank_gt/len(rank_preds)*100:.1f}%)")

    # === 5. cardiovascular LOW (19.4%) ===
    print("\n" + "="*70)
    print("=== 5. cardiovascular LOW (19.4%, n=216) ===")
    print("="*70)
    cv_low = preds_by_cat_tier.get(('cardiovascular', 'LOW'), [])
    if cv_low:
        gt_hits = [p for p in cv_low if p['is_gt']]
        print(f"Total: {len(cv_low)}, GT hits: {len(gt_hits)}")

        # Rule breakdown
        rule_counts = Counter(p['rule'] for p in cv_low)
        print(f"\nRules:")
        for rule, count in rule_counts.most_common():
            rule_gt = sum(1 for p in cv_low if p['rule'] == rule and p['is_gt'])
            print(f"  {rule}: {rule_gt}/{count} ({rule_gt/count*100:.1f}%)")

        # By disease
        by_disease = defaultdict(lambda: {"total": 0, "gt": 0})
        for p in cv_low:
            by_disease[p['disease']]["total"] += 1
            if p['is_gt']:
                by_disease[p['disease']]["gt"] += 1

        print(f"\nBy disease (top 10 by GT):")
        sorted_diseases = sorted(by_disease.keys(), key=lambda d: -by_disease[d]['gt'])
        for disease in sorted_diseases[:10]:
            dd = by_disease[disease]
            prec = dd['gt'] / dd['total'] * 100
            print(f"  {disease}: {dd['gt']}/{dd['total']} ({prec:.1f}%)")

    # === 6. Metabolic tier structure issues ===
    print("\n" + "="*70)
    print("=== 6. metabolic HIGH (17.3%, n=52) — below MEDIUM avg ===")
    print("="*70)
    met_high = preds_by_cat_tier.get(('metabolic', 'HIGH'), [])
    if met_high:
        gt_hits = [p for p in met_high if p['is_gt']]
        print(f"Total: {len(met_high)}, GT hits: {len(gt_hits)}")

        # Rule breakdown
        rule_counts = Counter(p['rule'] for p in met_high)
        print(f"\nRules:")
        for rule, count in rule_counts.most_common():
            rule_gt = sum(1 for p in met_high if p['rule'] == rule and p['is_gt'])
            print(f"  {rule}: {rule_gt}/{count} ({rule_gt/count*100:.1f}%)")


if __name__ == "__main__":
    main()
