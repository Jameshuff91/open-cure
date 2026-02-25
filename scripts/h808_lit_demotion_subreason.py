#!/usr/bin/env python3
"""
h808: Literature High Demotion Sub-Reason Audit

Which original HIGH sub-reasons are being demoted by literature_high_demotion?
Approach: Run holdout with literature disabled to capture original sub-reasons,
then cross-reference with literature levels from the normal run.
"""

import json
import sys
import os
from collections import defaultdict
from typing import Set, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.production_predictor import DrugRepurposingPredictor
from scripts.h393_holdout_tier_validation import (
    split_diseases,
    recompute_gt_structures,
    restore_gt_structures,
)


def main() -> None:
    seeds = [42, 123, 456, 789, 2024]
    print("=" * 70)
    print("h808: Literature High Demotion Sub-Reason Audit")
    print("=" * 70)

    predictor = DrugRepurposingPredictor()

    gt_path = predictor.reference_dir / "expanded_ground_truth.json"
    with open(gt_path) as f:
        gt_data = json.load(f)
    gt_set: Set[Tuple[str, str]] = set()
    for disease_id, drugs in gt_data.items():
        for drug in drugs:
            if isinstance(drug, str):
                gt_set.add((disease_id, drug))

    all_diseases = [d for d in predictor.ground_truth if d in predictor.embeddings]
    print(f"Diseases: {len(all_diseases)}, GT pairs: {len(gt_set)}")

    # Store original method
    original_lit_method = predictor._get_literature_evidence

    # Results: (original_sub_reason, lit_level) -> list of (hits, total) per seed
    cross_tab = defaultdict(lambda: defaultdict(lambda: {'hits': [], 'total': []}))
    # original_sub_reason -> list of (hits, total) per seed (regardless of lit)
    orig_all = defaultdict(lambda: {'hits': [], 'total': []})

    for seed_idx, seed in enumerate(seeds):
        print(f"\nSeed {seed} ({seed_idx+1}/{len(seeds)})...")
        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        train_set = set(train_ids)
        originals = recompute_gt_structures(predictor, train_set)

        # Pass 1: Normal run — get tier and lit_level for each prediction
        normal_info = {}  # (disease_id, drug_name) -> {tier, lit_level}
        for disease_id in holdout_ids:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            try:
                result = predictor.predict(disease_name, top_n=30, include_filtered=True)
            except Exception:
                continue
            for pred in result.predictions:
                key = (disease_id, pred.drug_name)
                normal_info[key] = {
                    'tier': pred.confidence_tier.name,
                    'lit_level': pred.literature_evidence_level,
                    'drug_id': pred.drug_id,
                }

        # Pass 2: Literature-disabled run — get original tier and sub-reason
        predictor._get_literature_evidence = lambda drug_name, disease_name: ('NOT_ASSESSED', 0.0)

        seed_cross = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # [hits, total]
        seed_orig = defaultdict(lambda: [0, 0])

        for disease_id in holdout_ids:
            disease_name = predictor.disease_names.get(disease_id, disease_id)
            try:
                result = predictor.predict(disease_name, top_n=30, include_filtered=True)
            except Exception:
                continue
            for pred in result.predictions:
                key = (disease_id, pred.drug_name)
                if key not in normal_info:
                    continue

                ninfo = normal_info[key]
                orig_tier = pred.confidence_tier.name
                orig_sub = pred.category_specific_tier or 'default'

                # We care about predictions that were HIGH without literature
                # and got demoted to MEDIUM by literature_high_demotion
                if orig_tier == 'HIGH' and ninfo['tier'] == 'MEDIUM' and \
                   ninfo['lit_level'] in ('NO_EVIDENCE', 'WEAK_EVIDENCE'):
                    is_hit = (disease_id, ninfo['drug_id']) in gt_set
                    h = 1 if is_hit else 0
                    lit = ninfo['lit_level']

                    seed_cross[orig_sub][lit][0] += h
                    seed_cross[orig_sub][lit][1] += 1
                    seed_cross[orig_sub]['ALL'][0] += h
                    seed_cross[orig_sub]['ALL'][1] += 1
                    seed_orig[orig_sub][0] += h
                    seed_orig[orig_sub][1] += 1

        predictor._get_literature_evidence = original_lit_method
        restore_gt_structures(predictor, originals)

        # Record per-seed results
        for sr in set(list(seed_cross.keys()) + list(seed_orig.keys())):
            for lit in seed_cross[sr]:
                h, t = seed_cross[sr][lit]
                cross_tab[sr][lit]['hits'].append(h)
                cross_tab[sr][lit]['total'].append(t)
            h, t = seed_orig[sr]
            orig_all[sr]['hits'].append(h)
            orig_all[sr]['total'].append(t)

        # Print seed summary
        total_demoted = sum(v[1] for v in seed_orig.values())
        total_hits = sum(v[0] for v in seed_orig.values())
        print(f"  Demoted: {total_hits}/{total_demoted} = {total_hits/total_demoted*100:.1f}%" if total_demoted > 0 else "  No demotions")

    # Print results
    print("\n" + "=" * 70)
    print("ORIGINAL SUB-REASON BREAKDOWN (predictions demoted HIGH→MEDIUM by literature)")
    print("=" * 70)

    all_srs = sorted(orig_all.keys(), key=lambda sr: -sum(orig_all[sr]['total']))
    for sr in all_srs:
        total_h = sum(orig_all[sr]['hits'])
        total_t = sum(orig_all[sr]['total'])
        if total_t == 0:
            continue
        per_seed_precs = []
        per_seed_ns = []
        for h, t in zip(orig_all[sr]['hits'], orig_all[sr]['total']):
            per_seed_ns.append(t)
            if t > 0:
                per_seed_precs.append(h / t * 100)
        avg_n = np.mean(per_seed_ns)
        if per_seed_precs:
            mean_p = np.mean(per_seed_precs)
            std_p = np.std(per_seed_precs)
            print(f"\n  {sr}: {mean_p:.1f}% ± {std_p:.1f}% (n={avg_n:.0f}/seed, total={total_t})")
            # Compare to HIGH threshold (55%)
            if mean_p >= 55 and avg_n >= 5:
                print(f"    → PROTECTABLE: {mean_p:.1f}% >= 55% HIGH threshold")
            elif mean_p >= 35:
                print(f"    → MEDIUM quality: correctly demoted")
            else:
                print(f"    → LOW quality: could demote further")

            # Show lit_level split
            for lit in ['NO_EVIDENCE', 'WEAK_EVIDENCE']:
                if lit in cross_tab[sr]:
                    lt_h = sum(cross_tab[sr][lit]['hits'])
                    lt_t = sum(cross_tab[sr][lit]['total'])
                    if lt_t > 0:
                        lt_precs = []
                        for h, t in zip(cross_tab[sr][lit]['hits'], cross_tab[sr][lit]['total']):
                            if t > 0:
                                lt_precs.append(h / t * 100)
                        if lt_precs:
                            lt_mean = np.mean(lt_precs)
                            print(f"      {lit}: {lt_h}/{lt_t} = {lt_mean:.1f}% ({lt_t//5}/seed)")

    # Summary decision
    print("\n" + "=" * 70)
    print("DECISION SUMMARY")
    print("=" * 70)
    protectable = [sr for sr in all_srs
                   if sum(orig_all[sr]['total']) >= 25  # >= 5/seed
                   and np.mean([h/t*100 for h, t in zip(orig_all[sr]['hits'], orig_all[sr]['total']) if t > 0]) >= 55]
    if protectable:
        print(f"Sub-reasons to PROTECT from literature demotion: {protectable}")
    else:
        print("No sub-reasons meet protection criteria (>=55% holdout, n>=5/seed)")
        print("Literature demotion is correctly applied to all HIGH sub-reasons.")


if __name__ == '__main__':
    main()
