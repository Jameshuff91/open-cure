#!/usr/bin/env python3
"""h791: HIGH No-Evidence Demotion to MEDIUM

h788 found HIGH + NO/WEAK_EVIDENCE = 28-34% holdout (below MEDIUM ~36-40%).
This script tests demoting these predictions to MEDIUM.
"""

import json
import sys
from collections import defaultdict
from copy import deepcopy
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

DATA_DIR = Path(__file__).parent.parent
SEEDS = [42, 123, 456, 789, 2024]


def split_diseases(all_diseases: List[str], seed: int) -> Tuple[List[str], List[str]]:
    rng = np.random.RandomState(seed)
    shuffled = list(all_diseases)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * 0.8)
    return shuffled[:split_idx], shuffled[split_idx:]


def recompute_gt_structures(
    predictor: DrugRepurposingPredictor,
    train_disease_ids: Set[str],
) -> Dict:
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

    new_freq: Dict[str, int] = defaultdict(int)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            for drug_id in predictor.ground_truth[disease_id]:
                new_freq[drug_id] += 1
    predictor.drug_train_freq = dict(new_freq)

    new_d2d: Dict[str, Set[str]] = defaultdict(set)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            for drug_id in predictor.ground_truth[disease_id]:
                new_d2d[drug_id].add(disease_name.lower())
    predictor.drug_to_diseases = dict(new_d2d)

    new_cancer: Dict[str, Set[str]] = defaultdict(set)
    for disease_id in train_disease_ids:
        if disease_id in predictor.ground_truth:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            cancer_types = extract_cancer_types(disease_name)
            if cancer_types:
                for drug_id in predictor.ground_truth[disease_id]:
                    new_cancer[drug_id].update(cancer_types)
    predictor.drug_cancer_types = dict(new_cancer)

    new_groups: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
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

    predictor.train_diseases = [
        d for d in train_disease_ids if d in predictor.embeddings
    ]
    predictor.train_embeddings = np.array(
        [predictor.embeddings[d] for d in predictor.train_diseases], dtype=np.float32
    )
    predictor.train_disease_categories = {}
    for d in predictor.train_diseases:
        name = predictor.disease_names.get(d, d)
        predictor.train_disease_categories[d] = predictor.categorize_disease(name)

    return originals


def restore_gt_structures(predictor: DrugRepurposingPredictor, originals: Dict) -> None:
    predictor.drug_train_freq = originals["drug_train_freq"]
    predictor.drug_to_diseases = originals["drug_to_diseases"]
    predictor.drug_cancer_types = originals["drug_cancer_types"]
    predictor.drug_disease_groups = originals["drug_disease_groups"]
    predictor.train_diseases = originals["train_diseases"]
    predictor.train_embeddings = originals["train_embeddings"]
    predictor.train_disease_categories = originals["train_disease_categories"]


def main() -> None:
    print("=" * 70)
    print("h791: HIGH No-Evidence Demotion to MEDIUM")
    print("=" * 70)
    sys.stdout.flush()

    predictor = DrugRepurposingPredictor()

    # Load expanded GT for evaluation
    gt_path = predictor.reference_dir / "expanded_ground_truth.json"
    with open(gt_path) as f:
        gt_data = json.load(f)
    print(f"GT: {len(gt_data)} diseases, {sum(len(v) for v in gt_data.values())} pairs")

    # Build GT set (disease_id, drug_id)
    gt_set: Set[Tuple[str, str]] = set()
    for disease_id, drugs in gt_data.items():
        for drug in drugs:
            if isinstance(drug, str):
                gt_set.add((disease_id, drug))
            elif isinstance(drug, dict):
                gt_set.add((disease_id, drug.get("drug_id") or drug.get("drug")))

    all_diseases = [d for d in predictor.ground_truth if d in predictor.embeddings]
    print(f"Diseases with GT + embeddings: {len(all_diseases)}")
    sys.stdout.flush()

    # Track results across seeds
    tier_before = defaultdict(lambda: {'hits': [], 'totals': []})
    tier_after = defaultdict(lambda: {'hits': [], 'totals': []})
    demoted_stats = {'hits': [], 'totals': []}
    demoted_by_ev = defaultdict(lambda: {'hits': [], 'totals': []})
    demoted_by_sub = defaultdict(lambda: {'hits': [], 'totals': []})

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n  Seed {seed} ({seed_idx+1}/{len(SEEDS)})...", end="", flush=True)

        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        train_set = set(train_ids)

        # Recompute GT structures from training diseases only
        originals = recompute_gt_structures(predictor, train_set)

        s_tier_before: Dict[str, Dict[str, int]] = defaultdict(lambda: {'hits': 0, 'total': 0})
        s_tier_after: Dict[str, Dict[str, int]] = defaultdict(lambda: {'hits': 0, 'total': 0})
        s_demoted_hits = 0
        s_demoted_total = 0
        s_demoted_ev: Dict[str, Dict[str, int]] = defaultdict(lambda: {'hits': 0, 'total': 0})
        s_demoted_sub: Dict[str, Dict[str, int]] = defaultdict(lambda: {'hits': 0, 'total': 0})

        for disease_id in holdout_ids:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            try:
                result = predictor.predict(disease_name, top_n=30, include_filtered=True)
            except Exception:
                continue

            for p in result.predictions:
                tier_name = p.confidence_tier.value
                is_hit = (disease_id, p.drug_id) in gt_set

                # Before demotion
                s_tier_before[tier_name]['hits'] += (1 if is_hit else 0)
                s_tier_before[tier_name]['total'] += 1

                # Check if HIGH + NO/WEAK evidence → should demote
                demoted = False
                if p.confidence_tier == ConfidenceTier.HIGH:
                    ev = p.literature_evidence_level
                    if ev in ('NO_EVIDENCE', 'WEAK_EVIDENCE'):
                        demoted = True
                        s_demoted_hits += (1 if is_hit else 0)
                        s_demoted_total += 1
                        s_demoted_ev[ev]['hits'] += (1 if is_hit else 0)
                        s_demoted_ev[ev]['total'] += 1
                        sub = p.category_specific_tier or 'default'
                        s_demoted_sub[sub]['hits'] += (1 if is_hit else 0)
                        s_demoted_sub[sub]['total'] += 1

                # After demotion
                after_tier = 'MEDIUM' if demoted else tier_name
                s_tier_after[after_tier]['hits'] += (1 if is_hit else 0)
                s_tier_after[after_tier]['total'] += 1

        # Aggregate seed results
        demoted_stats['hits'].append(s_demoted_hits)
        demoted_stats['totals'].append(s_demoted_total)

        for ev in ['NO_EVIDENCE', 'WEAK_EVIDENCE']:
            demoted_by_ev[ev]['hits'].append(s_demoted_ev[ev]['hits'])
            demoted_by_ev[ev]['totals'].append(s_demoted_ev[ev]['total'])

        for sub, data in s_demoted_sub.items():
            demoted_by_sub[sub]['hits'].append(data['hits'])
            demoted_by_sub[sub]['totals'].append(data['total'])

        for tier in ['GOLDEN', 'HIGH', 'MEDIUM', 'LOW', 'FILTER']:
            tier_before[tier]['hits'].append(s_tier_before[tier]['hits'])
            tier_before[tier]['totals'].append(s_tier_before[tier]['total'])
            tier_after[tier]['hits'].append(s_tier_after[tier]['hits'])
            tier_after[tier]['totals'].append(s_tier_after[tier]['total'])

        # Restore
        restore_gt_structures(predictor, originals)

        print(f" demoted {s_demoted_total} ({s_demoted_hits} hits)", flush=True)

    # ===== RESULTS =====
    print("\n" + "=" * 70)
    print("RESULTS: h791 HIGH No-Evidence Demotion")
    print("=" * 70)

    def calc_precs(hits_list, totals_list):
        precs = []
        for h, t in zip(hits_list, totals_list):
            if t > 0:
                precs.append(100.0 * h / t)
        return precs

    # Demoted group
    dem_precs = calc_precs(demoted_stats['hits'], demoted_stats['totals'])
    mean_n = np.mean(demoted_stats['totals'])
    print(f"\nDemoted group (NO+WEAK evidence):")
    if dem_precs:
        print(f"  Holdout: {np.mean(dem_precs):.1f}% ± {np.std(dem_precs):.1f}% (n={mean_n:.1f}/seed)")
    else:
        print(f"  No demoted predictions found!")

    for ev in ['NO_EVIDENCE', 'WEAK_EVIDENCE']:
        ev_precs = calc_precs(demoted_by_ev[ev]['hits'], demoted_by_ev[ev]['totals'])
        ev_n = np.mean(demoted_by_ev[ev]['totals'])
        if ev_precs:
            print(f"  {ev}: {np.mean(ev_precs):.1f}% ± {np.std(ev_precs):.1f}% (n={ev_n:.1f}/seed)")

    # Sub-reason breakdown
    if demoted_by_sub:
        print(f"\n  Demoted by sub-reason (mean across seeds):")
        for sub in sorted(demoted_by_sub.keys(), key=lambda k: -np.mean(demoted_by_sub[k]['totals'])):
            sub_precs = calc_precs(demoted_by_sub[sub]['hits'], demoted_by_sub[sub]['totals'])
            sub_n = np.mean(demoted_by_sub[sub]['totals'])
            if sub_precs and sub_n >= 1:
                print(f"    {sub:<40} {np.mean(sub_precs):5.1f}% ± {np.std(sub_precs):4.1f}% (n={sub_n:.1f})")

    # Tier comparison
    print(f"\n{'Tier':<10} {'Before':<22} {'After':<22} {'Delta':>8}  n(before→after)")
    print("-" * 80)
    tier_comparison = {}
    for tier in ['GOLDEN', 'HIGH', 'MEDIUM', 'LOW', 'FILTER']:
        b_precs = calc_precs(tier_before[tier]['hits'], tier_before[tier]['totals'])
        a_precs = calc_precs(tier_after[tier]['hits'], tier_after[tier]['totals'])

        b_str = f"{np.mean(b_precs):.1f}% ± {np.std(b_precs):.1f}%" if b_precs else "N/A"
        a_str = f"{np.mean(a_precs):.1f}% ± {np.std(a_precs):.1f}%" if a_precs else "N/A"
        delta = ""
        if b_precs and a_precs:
            d = np.mean(a_precs) - np.mean(b_precs)
            delta = f"{d:+.1f}pp"
        b_n = np.mean(tier_before[tier]['totals'])
        a_n = np.mean(tier_after[tier]['totals'])
        print(f"  {tier:<8} {b_str:<22} {a_str:<22} {delta:>8}  ({b_n:.0f}→{a_n:.0f})")

        if b_precs and a_precs:
            tier_comparison[tier] = {
                'before': round(float(np.mean(b_precs)), 1),
                'after': round(float(np.mean(a_precs)), 1),
                'delta': round(float(np.mean(a_precs) - np.mean(b_precs)), 1),
            }

    # Decision
    print("\n--- DECISION ---")
    if not dem_precs or mean_n < 1:
        print("  No HIGH predictions with NO/WEAK evidence found in holdout.")
        print("  This likely means evidence is only cached for a subset of predictions.")
        print("  → INCONCLUSIVE: Need to expand literature cache coverage first (h790).")
    else:
        dem_mean = float(np.mean(dem_precs))
        med_before = float(np.mean(calc_precs(tier_before['MEDIUM']['hits'], tier_before['MEDIUM']['totals'])))
        high_before = float(np.mean(calc_precs(tier_before['HIGH']['hits'], tier_before['HIGH']['totals'])))
        high_after = float(np.mean(calc_precs(tier_after['HIGH']['hits'], tier_after['HIGH']['totals'])))

        if dem_mean < med_before * 0.85:
            print(f"  Demoted group ({dem_mean:.1f}%) << MEDIUM ({med_before:.1f}%)")
            print(f"  → Consider demoting to LOW instead of MEDIUM")
        elif dem_mean < 50:
            print(f"  Demoted group ({dem_mean:.1f}%) is below HIGH threshold")
            print(f"  HIGH improvement: {high_before:.1f}% → {high_after:.1f}% ({high_after - high_before:+.1f}pp)")
            print(f"  → VALIDATED: Demotion to MEDIUM is appropriate")
        else:
            print(f"  Demoted group ({dem_mean:.1f}%) is not clearly below HIGH")
            print(f"  → INVALIDATED: No demotion needed")

    # Save
    results = {
        'description': 'h791: HIGH no-evidence demotion to MEDIUM',
        'demoted_holdout_mean': round(float(np.mean(dem_precs)), 1) if dem_precs else None,
        'demoted_holdout_std': round(float(np.std(dem_precs)), 1) if dem_precs else None,
        'demoted_n_per_seed': round(float(mean_n), 1),
        'per_seed_demoted': dem_precs,
        'tier_comparison': tier_comparison,
    }
    out_path = DATA_DIR / "data" / "analysis" / "h791_high_no_evidence_demotion.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
