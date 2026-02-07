#!/usr/bin/env python3
"""
h620: Expanded GT Safety Filter Audit

h615 showed several FILTER rules with surprisingly high expanded GT precision:
- non_therapeutic_compound: 33.3% (should be 0% for diagnostic agents)
- base_to_complication: 37.5% (should be 0% for indirect associations)
- inverse_indication: 16.9% (drugs that CAUSE the disease)
- corticosteroid_iatrogenic: 10.5%

This script audits the specific (disease, drug) pairs where FILTER predictions
are hits in expanded GT, to check if the GT entries are valid therapeutic
associations or contamination (diagnostic, causal, indirect).
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from production_predictor import DrugRepurposingPredictor, ConfidenceTier


def main():
    print("=" * 80)
    print("h620: Expanded GT Safety Filter Audit")
    print("=" * 80)

    print("\nLoading predictor...")
    predictor = DrugRepurposingPredictor()

    # Load expanded GT
    gt_path = predictor.reference_dir / "expanded_ground_truth.json"
    with open(gt_path) as f:
        expanded_gt = json.load(f)
    exp_set = set()
    for d, drugs in expanded_gt.items():
        for drug in drugs:
            if isinstance(drug, str):
                exp_set.add((d, drug))
            elif isinstance(drug, dict):
                exp_set.add((d, drug.get("drug_id") or drug.get("drug")))

    # Internal GT
    int_set = set()
    for d, drugs in predictor.ground_truth.items():
        for drug_id in drugs:
            int_set.add((d, drug_id))

    print(f"Internal GT: {len(int_set)} pairs")
    print(f"Expanded GT: {len(exp_set)} pairs")
    print(f"Expanded-only pairs: {len(exp_set - int_set)}")

    # Run predictions and collect FILTER hits
    all_diseases = [d for d in predictor.ground_truth if d in predictor.embeddings]

    filter_rules_of_interest = [
        "non_therapeutic_compound",
        "inverse_indication",
        "base_to_complication",
        "corticosteroid_iatrogenic",
        "cross_domain_isolated",
        "cancer_only_non_cancer",
        "cancer_no_gt",
        "complication_non_validated",
    ]

    # Collect hits per rule
    rule_hits = defaultdict(list)  # rule -> list of (disease_name, drug_name, in_internal, in_expanded)
    rule_totals = defaultdict(int)

    for disease_id in all_diseases:
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        try:
            result = predictor.predict(disease_name, top_n=30, include_filtered=True)
        except Exception:
            continue

        for pred in result.predictions:
            if pred.confidence_tier != ConfidenceTier.FILTER:
                continue

            rule = pred.category_specific_tier or "default"
            rule_totals[rule] += 1

            drug_id = pred.drug_id
            in_internal = (disease_id, drug_id) in int_set
            in_expanded = (disease_id, drug_id) in exp_set

            if in_expanded:
                drug_name = predictor.drug_id_to_name.get(drug_id, drug_id)
                category = pred.category
                rule_hits[rule].append({
                    "disease_id": disease_id,
                    "disease_name": disease_name,
                    "drug_id": drug_id,
                    "drug_name": drug_name,
                    "category": category,
                    "in_internal_gt": in_internal,
                    "in_expanded_only": in_expanded and not in_internal,
                })

    # Report
    print(f"\n{'='*80}")
    print("FILTER Rule GT Hit Analysis")
    print(f"{'='*80}")

    print(f"\n{'Rule':<40s} {'Hits':>5s} {'Total':>6s} {'Prec%':>6s} {'IntGT':>6s} {'ExpOnly':>7s}")
    print("-" * 76)

    for rule in filter_rules_of_interest:
        hits = rule_hits.get(rule, [])
        total = rule_totals.get(rule, 0)
        prec = len(hits) / total * 100 if total > 0 else 0
        n_internal = sum(1 for h in hits if h["in_internal_gt"])
        n_expanded_only = sum(1 for h in hits if h["in_expanded_only"])
        print(f"  {rule:<38s} {len(hits):>5d} {total:>6d} {prec:>5.1f}% {n_internal:>6d} {n_expanded_only:>7d}")

    # Also show "default" FILTER
    for rule in ["default"]:
        hits = rule_hits.get(rule, [])
        total = rule_totals.get(rule, 0)
        prec = len(hits) / total * 100 if total > 0 else 0
        n_internal = sum(1 for h in hits if h["in_internal_gt"])
        n_expanded_only = sum(1 for h in hits if h["in_expanded_only"])
        print(f"  {rule:<38s} {len(hits):>5d} {total:>6d} {prec:>5.1f}% {n_internal:>6d} {n_expanded_only:>7d}")

    # Detailed audit for each rule of interest
    for rule in filter_rules_of_interest:
        hits = rule_hits.get(rule, [])
        if not hits:
            continue

        print(f"\n{'='*80}")
        print(f"DETAILED AUDIT: {rule} ({len(hits)} GT hits)")
        print(f"{'='*80}")

        for h in sorted(hits, key=lambda x: (x["category"], x["drug_name"])):
            gt_source = "INTERNAL+EXPANDED" if h["in_internal_gt"] else "EXPANDED-ONLY"
            print(f"  {h['drug_name']:<30s} → {h['disease_name']:<40s} [{h['category']}] {gt_source}")

    # Summary analysis
    print(f"\n{'='*80}")
    print("SUMMARY: GT Source Breakdown for FILTER Hits")
    print(f"{'='*80}")

    total_filter_hits = sum(len(v) for v in rule_hits.values())
    total_internal = sum(sum(1 for h in v if h["in_internal_gt"]) for v in rule_hits.values())
    total_expanded_only = sum(sum(1 for h in v if h["in_expanded_only"]) for v in rule_hits.values())

    print(f"Total FILTER GT hits: {total_filter_hits}")
    print(f"  From internal GT: {total_internal} ({total_internal/total_filter_hits*100:.1f}%)")
    print(f"  From expanded GT only: {total_expanded_only} ({total_expanded_only/total_filter_hits*100:.1f}%)")
    print(f"\nIf expanded-only hits are NOT valid therapeutic associations,")
    print(f"all expanded-GT-based precision numbers are inflated by these entries.")

    # Check inverse_indication specifically
    inv_hits = rule_hits.get("inverse_indication", [])
    if inv_hits:
        print(f"\n--- INVERSE INDICATION HITS ({len(inv_hits)}) ---")
        print(f"These are drugs that CAUSE the disease but appear in expanded GT as treatments:")
        for h in inv_hits:
            gt_source = "INTERNAL" if h["in_internal_gt"] else "EXP-ONLY"
            print(f"  {h['drug_name']} → {h['disease_name']} [{gt_source}]")

    # Check non_therapeutic_compound
    nt_hits = rule_hits.get("non_therapeutic_compound", [])
    if nt_hits:
        print(f"\n--- NON-THERAPEUTIC COMPOUND HITS ({len(nt_hits)}) ---")
        print(f"These are diagnostic agents (FDG PET, ICG dye) with GT entries:")
        for h in nt_hits:
            gt_source = "INTERNAL" if h["in_internal_gt"] else "EXP-ONLY"
            print(f"  {h['drug_name']} → {h['disease_name']} [{gt_source}]")

    # Save results
    output = {
        "rule_hits": {rule: hits for rule, hits in rule_hits.items() if rule in filter_rules_of_interest},
        "rule_totals": {rule: rule_totals[rule] for rule in filter_rules_of_interest if rule in rule_totals},
        "total_filter_hits": total_filter_hits,
        "internal_gt_hits": total_internal,
        "expanded_only_hits": total_expanded_only,
    }

    output_path = Path(__file__).parent.parent / "data" / "analysis" / "h620_filter_gt_audit.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
