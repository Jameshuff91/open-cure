#!/usr/bin/env python3
"""
h390: Production Tier Rule Coverage Analysis.

For every prediction in the deliverable, analyze:
1. Which rule assigned its tier?
2. What is the precision of each rule?
3. Where are gaps - predictions at LOW that could be rescued?
4. What fraction of GT hits are in each tier/rule?

Goal: Identify actionable rule opportunities (rules that could promote LOW→MEDIUM or MEDIUM→HIGH
with good precision).
"""

import json
import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from production_predictor import (
    DrugRepurposingPredictor,
    ConfidenceTier,
)


def analyze_rule_coverage(predictor: DrugRepurposingPredictor) -> Dict:
    """Run predictions for all diseases and collect rule assignments."""

    gt_path = predictor.reference_dir / "expanded_ground_truth.json"
    with open(gt_path) as f:
        gt_data = json.load(f)

    all_diseases = [
        d for d in predictor.disease_names.keys()
        if d in predictor.ground_truth and d in predictor.embeddings
    ]

    # Collect per-rule stats
    rule_stats: Dict[str, Dict] = defaultdict(lambda: {
        "tier": None,
        "total": 0,
        "gt_hits": 0,
        "categories": Counter(),
        "example_drugs": [],
        "example_diseases": [],
    })

    # Per-tier stats
    tier_stats: Dict[str, Dict] = defaultdict(lambda: {
        "total": 0,
        "gt_hits": 0,
        "rules": Counter(),
    })

    # LOW tier predictions that are GT hits (rescue candidates)
    low_gt_hits = []

    # Track by category
    cat_tier_stats: Dict[str, Dict[str, Dict]] = defaultdict(lambda: defaultdict(lambda: {
        "total": 0, "gt_hits": 0
    }))

    diseases_processed = 0
    for disease_id in all_diseases:
        disease_name = predictor.disease_names.get(disease_id, disease_id)

        # Get GT drugs for this disease
        gt_drugs = set()
        if disease_id in gt_data:
            for drug in gt_data[disease_id]:
                if isinstance(drug, str):
                    gt_drugs.add(drug)
                elif isinstance(drug, dict):
                    gt_drugs.add(drug.get('drug_id') or drug.get('drug'))

        try:
            result = predictor.predict(disease_name, top_n=30, include_filtered=True)
        except Exception as e:
            continue

        category = predictor.categorize_disease(disease_name)

        for pred in result.predictions:
            tier_name = pred.confidence_tier.name if hasattr(pred.confidence_tier, 'name') else str(pred.confidence_tier)
            rule = pred.category_specific_tier or 'standard'
            is_gt = pred.drug_id in gt_drugs

            # Rule stats
            rs = rule_stats[rule]
            rs["tier"] = tier_name
            rs["total"] += 1
            rs["gt_hits"] += int(is_gt)
            rs["categories"][category] += 1
            if len(rs["example_drugs"]) < 3 and is_gt:
                rs["example_drugs"].append(pred.drug_name)
                rs["example_diseases"].append(disease_name)

            # Tier stats
            ts = tier_stats[tier_name]
            ts["total"] += 1
            ts["gt_hits"] += int(is_gt)
            ts["rules"][rule] += 1

            # Category × tier
            cts = cat_tier_stats[category][tier_name]
            cts["total"] += 1
            cts["gt_hits"] += int(is_gt)

            # LOW tier GT hits = rescue candidates
            if tier_name == "LOW" and is_gt:
                low_gt_hits.append({
                    "drug": pred.drug_name,
                    "drug_id": pred.drug_id,
                    "disease": disease_name,
                    "category": category,
                    "rank": pred.rank,
                    "train_freq": pred.train_frequency,
                    "mechanism": pred.mechanism_support,
                    "score": pred.knn_score,
                })

        diseases_processed += 1
        if diseases_processed % 50 == 0:
            print(f"  Processed {diseases_processed}/{len(all_diseases)} diseases...")

    return {
        "rule_stats": rule_stats,
        "tier_stats": tier_stats,
        "cat_tier_stats": cat_tier_stats,
        "low_gt_hits": low_gt_hits,
        "n_diseases": len(all_diseases),
    }


def main():
    print("Loading predictor...")
    predictor = DrugRepurposingPredictor()

    print("Analyzing rule coverage...")
    results = analyze_rule_coverage(predictor)

    # === Report ===
    print(f"\n{'='*70}")
    print("=== h390: PRODUCTION TIER RULE COVERAGE ANALYSIS ===")
    print(f"{'='*70}")
    print(f"Diseases analyzed: {results['n_diseases']}")

    # --- Tier Summary ---
    print(f"\n--- TIER SUMMARY ---")
    print(f"{'Tier':<10} {'Total':>8} {'GT Hits':>10} {'Precision':>10} {'# Rules':>8}")
    print("-" * 50)
    tier_order = ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]
    for tier in tier_order:
        ts = results["tier_stats"].get(tier, {"total": 0, "gt_hits": 0, "rules": Counter()})
        prec = ts["gt_hits"] / ts["total"] * 100 if ts["total"] > 0 else 0
        n_rules = len(ts["rules"])
        print(f"{tier:<10} {ts['total']:>8} {ts['gt_hits']:>10} {prec:>9.1f}% {n_rules:>8}")

    # --- Rule Breakdown ---
    print(f"\n--- TOP RULES BY VOLUME (n>=20) ---")
    rule_list = []
    for rule_name, rs in results["rule_stats"].items():
        if rs["total"] >= 20:
            prec = rs["gt_hits"] / rs["total"] * 100 if rs["total"] > 0 else 0
            rule_list.append((rule_name, rs["tier"], rs["total"], rs["gt_hits"], prec))
    rule_list.sort(key=lambda x: -x[2])

    print(f"{'Rule':<40} {'Tier':<8} {'N':>6} {'GT':>6} {'Prec':>7}")
    print("-" * 70)
    for name, tier, total, gt, prec in rule_list:
        print(f"{name:<40} {tier:<8} {total:>6} {gt:>6} {prec:>6.1f}%")

    # --- HIGH-precision rules at each tier ---
    print(f"\n--- RULES SORTED BY PRECISION (n>=10) ---")
    prec_list = []
    for rule_name, rs in results["rule_stats"].items():
        if rs["total"] >= 10:
            prec = rs["gt_hits"] / rs["total"] * 100 if rs["total"] > 0 else 0
            prec_list.append((rule_name, rs["tier"], rs["total"], rs["gt_hits"], prec))
    prec_list.sort(key=lambda x: -x[4])

    print(f"{'Rule':<40} {'Tier':<8} {'N':>6} {'GT':>6} {'Prec':>7}")
    print("-" * 70)
    for name, tier, total, gt, prec in prec_list:
        marker = " ← POTENTIAL UPGRADE" if (tier in ("MEDIUM", "LOW") and prec > 30) else ""
        marker = " ← POTENTIAL DOWNGRADE" if (tier in ("GOLDEN", "HIGH") and prec < 15) else marker
        print(f"{name:<40} {tier:<8} {total:>6} {gt:>6} {prec:>6.1f}%{marker}")

    # --- LOW tier GT hits analysis ---
    print(f"\n--- LOW TIER GT HITS (rescue candidates) ---")
    print(f"Total LOW tier GT hits: {len(results['low_gt_hits'])}")

    if results['low_gt_hits']:
        # Group by category
        cat_low = defaultdict(list)
        for h in results['low_gt_hits']:
            cat_low[h['category']].append(h)

        print(f"\nBy category:")
        for cat in sorted(cat_low.keys(), key=lambda c: -len(cat_low[c])):
            hits = cat_low[cat]
            print(f"  {cat}: {len(hits)} GT hits in LOW tier")
            # Show features of these hits
            freqs = [h['train_freq'] for h in hits]
            mechs = [h['mechanism'] for h in hits]
            ranks = [h['rank'] for h in hits]
            print(f"    freq: {np.mean(freqs):.1f} ± {np.std(freqs):.1f}, "
                  f"mechanism: {sum(mechs)}/{len(mechs)} ({sum(mechs)/len(mechs)*100:.0f}%), "
                  f"rank: {np.mean(ranks):.1f}")

            # Show some examples
            for h in hits[:3]:
                print(f"    - {h['drug']} → {h['disease']} (freq={h['train_freq']}, mech={h['mechanism']}, rank={h['rank']})")

    # --- Category × Tier heatmap ---
    print(f"\n--- CATEGORY × TIER PRECISION ---")
    categories = sorted(results['cat_tier_stats'].keys())
    print(f"{'Category':<20}", end="")
    for tier in tier_order:
        print(f" {tier:>10}", end="")
    print()
    print("-" * (20 + 11 * len(tier_order)))

    for cat in categories:
        print(f"{cat:<20}", end="")
        for tier in tier_order:
            cts = results['cat_tier_stats'][cat].get(tier, {"total": 0, "gt_hits": 0})
            if cts["total"] >= 5:
                prec = cts["gt_hits"] / cts["total"] * 100
                print(f" {prec:>8.1f}%({cts['total']:>3})", end="")
            elif cts["total"] > 0:
                print(f"    n={cts['total']:<3}", end="")
            else:
                print(f"          -", end="")
        print()

    # Save results
    output = {
        "n_diseases": results['n_diseases'],
        "tier_summary": {},
        "rule_precision": {},
        "low_gt_hits_count": len(results['low_gt_hits']),
        "low_gt_by_category": {},
    }

    for tier in tier_order:
        ts = results["tier_stats"].get(tier, {"total": 0, "gt_hits": 0})
        output["tier_summary"][tier] = {
            "total": ts["total"],
            "gt_hits": ts["gt_hits"],
            "precision": ts["gt_hits"] / ts["total"] * 100 if ts["total"] > 0 else 0,
        }

    for rule_name, rs in results["rule_stats"].items():
        if rs["total"] >= 5:
            output["rule_precision"][rule_name] = {
                "tier": rs["tier"],
                "total": rs["total"],
                "gt_hits": rs["gt_hits"],
                "precision": rs["gt_hits"] / rs["total"] * 100 if rs["total"] > 0 else 0,
            }

    cat_low = defaultdict(int)
    for h in results['low_gt_hits']:
        cat_low[h['category']] += 1
    output["low_gt_by_category"] = dict(cat_low)

    out_path = Path(__file__).parent.parent / "data" / "analysis" / "h390_rule_coverage.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
