#!/usr/bin/env python3
"""
h417: Rank 21-30 Rule Coverage Gap Analysis

h399 showed hierarchy matches at rank 21-30 have very high precision (~60%).
But hierarchy only covers specific disease groups. What other signals exist
at rank 21-30 that could rescue predictions from FILTER?

NOTE: h418 showed hierarchy-before-rank FAILED holdout (-6.2pp for HIGH).
So we need to find signals that are more generalizable than hierarchy.

Analyze:
1. All predictions at rank 21-30 with their features
2. Precision by: freq>=10, mechanism_support, ATC coherence, category
3. Feature combinations with >25% precision
4. Identify which ones are NOT just hierarchy matches
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from production_predictor import DrugRepurposingPredictor, ConfidenceTier, DISEASE_HIERARCHY_GROUPS


def analyze_rank_21_30():
    """Analyze precision at rank 21-30 by various features."""
    print("Loading predictor...")
    predictor = DrugRepurposingPredictor()

    # Load expanded ground truth
    gt_path = predictor.reference_dir / "expanded_ground_truth.json"
    with open(gt_path) as f:
        gt_data = json.load(f)

    gt_set = set()
    for disease_id, drugs in gt_data.items():
        for drug in drugs:
            if isinstance(drug, str):
                gt_set.add((disease_id, drug))
            elif isinstance(drug, dict):
                gt_set.add((disease_id, drug.get('drug_id') or drug.get('drug')))

    print(f"Loaded {len(gt_set)} GT pairs")

    # Collect ALL rank 21-30 predictions with features
    diseases = list(predictor.disease_names.keys())
    print(f"Evaluating {len(diseases)} diseases...")

    rank_21_30_preds = []

    for i, disease_id in enumerate(diseases):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(diseases)} diseases...")

        disease_name = predictor.disease_names.get(disease_id, disease_id)

        try:
            result = predictor.predict(disease_name, top_n=30, include_filtered=True)
        except Exception as e:
            continue

        for pred in result.predictions:
            if pred.rank < 21 or pred.rank > 30:
                continue

            # Check GT
            is_hit = (disease_id, pred.drug_id) in gt_set

            # Get additional features
            drug_name = pred.drug_name
            category = pred.category

            # Check ATC coherence
            is_coherent = predictor._is_atc_coherent(drug_name, category) if drug_name and category else False

            # Check hierarchy match
            has_hierarchy_match = False
            matching_group = None
            if category in DISEASE_HIERARCHY_GROUPS and pred.drug_id:
                try:
                    has_gt, same_group, group = predictor._check_disease_hierarchy_match(
                        pred.drug_id, disease_name, category
                    )
                    has_hierarchy_match = same_group
                    matching_group = group
                except Exception:
                    pass

            # Check CV pathway comprehensive
            is_cv_pathway = False
            if predictor._is_cv_complication(disease_name):
                is_cv_pathway = predictor._is_cv_pathway_comprehensive(drug_name)

            # Check target overlap
            target_overlap = predictor._get_target_overlap_count(pred.drug_id, disease_id) if pred.drug_id else 0

            # Check cancer type match
            is_cancer_same_type = False
            if category == 'cancer' and pred.drug_id:
                try:
                    _, same_type, _ = predictor._check_cancer_type_match(pred.drug_id, disease_name)
                    is_cancer_same_type = same_type
                except Exception:
                    pass

            # Check comp→base
            is_comp_to_base = False
            if pred.drug_id:
                try:
                    ctb, _, _ = predictor._is_comp_to_base(pred.drug_id, disease_name)
                    is_comp_to_base = ctb
                except Exception:
                    pass

            # Check mechanism-specific disease
            is_mech_specific = predictor._is_mechanism_specific_disease(disease_name)

            rank_21_30_preds.append({
                'disease_id': disease_id,
                'disease_name': disease_name,
                'drug_id': pred.drug_id,
                'drug_name': drug_name,
                'rank': pred.rank,
                'category': category,
                'is_hit': is_hit,
                'train_freq': pred.train_frequency,
                'mechanism_support': pred.mechanism_support,
                'has_targets': pred.has_targets,
                'is_atc_coherent': is_coherent,
                'has_hierarchy_match': has_hierarchy_match,
                'matching_group': matching_group,
                'is_cv_pathway': is_cv_pathway,
                'target_overlap': target_overlap,
                'is_cancer_same_type': is_cancer_same_type,
                'is_comp_to_base': is_comp_to_base,
                'is_mech_specific': is_mech_specific,
                'current_tier': pred.confidence_tier.name,
                'current_rule': pred.category_specific_tier,
            })

    print(f"\nTotal rank 21-30 predictions: {len(rank_21_30_preds)}")

    # === ANALYSIS 1: Overall precision at rank 21-30 ===
    total = len(rank_21_30_preds)
    hits = sum(1 for p in rank_21_30_preds if p['is_hit'])
    print(f"\n=== OVERALL RANK 21-30 ===")
    print(f"Total: {total}, Hits: {hits}, Precision: {hits/total*100:.1f}%")

    # === ANALYSIS 2: By individual feature ===
    print(f"\n=== PRECISION BY INDIVIDUAL FEATURE ===")
    features = [
        ('freq>=10', lambda p: p['train_freq'] >= 10),
        ('freq>=15', lambda p: p['train_freq'] >= 15),
        ('freq>=5', lambda p: p['train_freq'] >= 5),
        ('mechanism_support', lambda p: p['mechanism_support']),
        ('has_targets', lambda p: p['has_targets']),
        ('atc_coherent', lambda p: p['is_atc_coherent']),
        ('hierarchy_match', lambda p: p['has_hierarchy_match']),
        ('cv_pathway', lambda p: p['is_cv_pathway']),
        ('target_overlap>=1', lambda p: p['target_overlap'] >= 1),
        ('target_overlap>=3', lambda p: p['target_overlap'] >= 3),
        ('cancer_same_type', lambda p: p['is_cancer_same_type']),
        ('comp_to_base', lambda p: p['is_comp_to_base']),
        ('NOT mech_specific', lambda p: not p['is_mech_specific']),
        ('mech_specific', lambda p: p['is_mech_specific']),
    ]

    for name, fn in features:
        subset = [p for p in rank_21_30_preds if fn(p)]
        if not subset:
            print(f"  {name:25s}: n=0")
            continue
        h = sum(1 for p in subset if p['is_hit'])
        n = len(subset)
        prec = h / n * 100
        print(f"  {name:25s}: n={n:5d}, hits={h:4d}, precision={prec:5.1f}%")

    # === ANALYSIS 3: By category ===
    print(f"\n=== PRECISION BY CATEGORY ===")
    cats = defaultdict(lambda: {'hits': 0, 'total': 0})
    for p in rank_21_30_preds:
        cats[p['category']]['total'] += 1
        if p['is_hit']:
            cats[p['category']]['hits'] += 1

    for cat, stats in sorted(cats.items(), key=lambda x: -x[1]['hits'] / max(x[1]['total'], 1)):
        prec = stats['hits'] / stats['total'] * 100 if stats['total'] else 0
        print(f"  {cat:25s}: n={stats['total']:5d}, hits={stats['hits']:4d}, precision={prec:5.1f}%")

    # === ANALYSIS 4: Feature combinations (exclude hierarchy since h418 showed it fails holdout) ===
    print(f"\n=== FEATURE COMBINATIONS (EXCLUDING hierarchy) ===")
    # Focus on non-hierarchy signals that might generalize
    non_hierarchy = [p for p in rank_21_30_preds if not p['has_hierarchy_match']]
    h_preds = [p for p in rank_21_30_preds if p['has_hierarchy_match']]

    nh_total = len(non_hierarchy)
    nh_hits = sum(1 for p in non_hierarchy if p['is_hit'])
    h_total = len(h_preds)
    h_hits = sum(1 for p in h_preds if p['is_hit'])
    print(f"  Non-hierarchy: n={nh_total}, hits={nh_hits}, precision={nh_hits/nh_total*100:.1f}%")
    print(f"  Hierarchy:     n={h_total}, hits={h_hits}, precision={h_hits/h_total*100:.1f}%")

    # Now analyze non-hierarchy predictions by feature combos
    combos = [
        ('freq>=10 + mechanism', lambda p: p['train_freq'] >= 10 and p['mechanism_support']),
        ('freq>=15 + mechanism', lambda p: p['train_freq'] >= 15 and p['mechanism_support']),
        ('freq>=10 + atc_coherent', lambda p: p['train_freq'] >= 10 and p['is_atc_coherent']),
        ('freq>=15 + atc_coherent', lambda p: p['train_freq'] >= 15 and p['is_atc_coherent']),
        ('freq>=10 + mechanism + atc', lambda p: p['train_freq'] >= 10 and p['mechanism_support'] and p['is_atc_coherent']),
        ('freq>=5 + mechanism + atc', lambda p: p['train_freq'] >= 5 and p['mechanism_support'] and p['is_atc_coherent']),
        ('target_overlap>=1 + mechanism', lambda p: p['target_overlap'] >= 1 and p['mechanism_support']),
        ('target_overlap>=1 + freq>=10', lambda p: p['target_overlap'] >= 1 and p['train_freq'] >= 10),
        ('target_overlap>=3', lambda p: p['target_overlap'] >= 3),
        ('cancer_same_type', lambda p: p['is_cancer_same_type']),
        ('comp_to_base', lambda p: p['is_comp_to_base']),
        ('cv_pathway', lambda p: p['is_cv_pathway']),
        ('freq>=10 + mechanism + NOT mech_specific', lambda p: p['train_freq'] >= 10 and p['mechanism_support'] and not p['is_mech_specific']),
        ('freq>=15 + mechanism + NOT mech_specific', lambda p: p['train_freq'] >= 15 and p['mechanism_support'] and not p['is_mech_specific']),
        ('freq>=15 + mechanism + atc + NOT mech_specific', lambda p: p['train_freq'] >= 15 and p['mechanism_support'] and p['is_atc_coherent'] and not p['is_mech_specific']),
    ]

    print(f"\n  Feature combinations on NON-HIERARCHY predictions:")
    for name, fn in combos:
        subset = [p for p in non_hierarchy if fn(p)]
        if not subset:
            print(f"    {name:45s}: n=0")
            continue
        h = sum(1 for p in subset if p['is_hit'])
        n = len(subset)
        prec = h / n * 100
        marker = " ***" if prec >= 25 and n >= 10 else ""
        print(f"    {name:45s}: n={n:5d}, hits={h:4d}, precision={prec:5.1f}%{marker}")

    # === ANALYSIS 5: Category-specific non-hierarchy combos ===
    print(f"\n=== CATEGORY x FEATURE COMBOS (n>=5, precision>=20%) ===")
    for cat in sorted(cats.keys()):
        cat_preds = [p for p in non_hierarchy if p['category'] == cat]
        if len(cat_preds) < 5:
            continue

        cat_h = sum(1 for p in cat_preds if p['is_hit'])
        cat_n = len(cat_preds)
        cat_prec = cat_h / cat_n * 100

        subsets = [
            ('freq>=10 + mech', lambda p: p['train_freq'] >= 10 and p['mechanism_support']),
            ('freq>=10 + atc', lambda p: p['train_freq'] >= 10 and p['is_atc_coherent']),
            ('freq>=10 + mech + atc', lambda p: p['train_freq'] >= 10 and p['mechanism_support'] and p['is_atc_coherent']),
            ('target_overlap>=1', lambda p: p['target_overlap'] >= 1),
            ('freq>=15 + mech', lambda p: p['train_freq'] >= 15 and p['mechanism_support']),
        ]

        any_printed = False
        for sub_name, sub_fn in subsets:
            sub = [p for p in cat_preds if sub_fn(p)]
            if len(sub) < 5:
                continue
            sh = sum(1 for p in sub if p['is_hit'])
            sn = len(sub)
            sp = sh / sn * 100
            if sp >= 20:
                if not any_printed:
                    print(f"\n  {cat} (overall: n={cat_n}, hits={cat_h}, precision={cat_prec:.1f}%):")
                    any_printed = True
                marker = " ***" if sp >= 25 and sn >= 10 else ""
                print(f"    {sub_name:35s}: n={sn:4d}, hits={sh:3d}, precision={sp:5.1f}%{marker}")

    # === ANALYSIS 6: How many would we rescue and at what precision? ===
    print(f"\n=== RESCUE CANDIDATES ===")
    print(f"Looking for feature combos that: precision >= 25% AND n >= 20")
    print(f"These could be promoted from FILTER to MEDIUM or HIGH")

    rescue_rules = [
        ('freq>=10 + mechanism (all)', lambda p: p['train_freq'] >= 10 and p['mechanism_support']),
        ('freq>=15 + mechanism (all)', lambda p: p['train_freq'] >= 15 and p['mechanism_support']),
        ('freq>=10 + mechanism + atc_coherent', lambda p: p['train_freq'] >= 10 and p['mechanism_support'] and p['is_atc_coherent']),
        ('freq>=15 + mechanism + NOT mech_specific', lambda p: p['train_freq'] >= 15 and p['mechanism_support'] and not p['is_mech_specific']),
        ('cancer_same_type (rank 21-30)', lambda p: p['is_cancer_same_type']),
        ('target_overlap>=3 (rank 21-30)', lambda p: p['target_overlap'] >= 3),
    ]

    for name, fn in rescue_rules:
        # Apply on all rank 21-30 (they're all currently FILTER)
        subset = [p for p in rank_21_30_preds if fn(p)]
        if not subset:
            print(f"  {name:50s}: n=0")
            continue
        h = sum(1 for p in subset if p['is_hit'])
        n = len(subset)
        prec = h / n * 100
        rescued = "RESCUE" if prec >= 25 and n >= 20 else ""
        print(f"  {name:50s}: n={n:5d}, hits={h:4d}, precision={prec:5.1f}% {rescued}")

    # === ANALYSIS 7: What are the GT hits at rank 21-30? (examples) ===
    print(f"\n=== SAMPLE GT HITS AT RANK 21-30 ===")
    hits_list = [p for p in rank_21_30_preds if p['is_hit']]
    hits_list.sort(key=lambda p: (-p['train_freq'], -p['target_overlap']))
    for p in hits_list[:30]:
        features_str = []
        if p['train_freq'] >= 10:
            features_str.append(f"freq={p['train_freq']}")
        if p['mechanism_support']:
            features_str.append("mech")
        if p['is_atc_coherent']:
            features_str.append("atc")
        if p['has_hierarchy_match']:
            features_str.append(f"hier={p['matching_group']}")
        if p['target_overlap'] >= 1:
            features_str.append(f"overlap={p['target_overlap']}")
        if p['is_cv_pathway']:
            features_str.append("cv_path")
        feat = ", ".join(features_str) if features_str else "none"
        print(f"  rank={p['rank']} {p['drug_name'][:25]:25s} → {p['disease_name'][:35]:35s} [{p['category']}] {feat}")

    # Save results
    output = {
        'total_rank_21_30': total,
        'total_hits': hits,
        'overall_precision': hits / total * 100 if total else 0,
        'hierarchy_count': h_total,
        'hierarchy_hits': h_hits,
        'non_hierarchy_count': nh_total,
        'non_hierarchy_hits': nh_hits,
    }

    out_path = Path(__file__).parent.parent / "data" / "analysis" / "h417_rank21_30_gap.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    analyze_rank_21_30()
