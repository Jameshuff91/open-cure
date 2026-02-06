#!/usr/bin/env python3
"""
h591: LOW-Tier Success Pattern Analysis

LOW tier has 15.5% holdout precision — ~580 out of 3733 predictions are correct.
What makes a LOW prediction succeed? Understanding this could inform:
1. Deliverable prioritization (which LOW predictions to review first)
2. Potential rescue rules for specific patterns
3. Calibration of existing demotion rules

We use full-data GT hits (not holdout) to identify successful predictions,
then characterize the SUCCESS population vs FAILURE population.
Note: full-data is inflated but pattern analysis is valid for characterization.
"""

import json
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from scipy import stats
from pathlib import Path

DATA_DIR = Path("data")
REF_DIR = DATA_DIR / "reference"
OUT_DIR = DATA_DIR / "analysis"

# ----- Step 1: Load deliverable and GT -----

print("=" * 60)
print("STEP 1: Load deliverable and ground truth")
print("=" * 60)

with open(DATA_DIR / "deliverables" / "drug_repurposing_predictions_with_confidence.json") as f:
    preds = json.load(f)

with open(REF_DIR / "expanded_ground_truth.json") as f:
    gt_raw = json.load(f)

# Build GT lookup: (disease_id, drug_id) -> True
gt_set = set()
for disease_id, drug_list in gt_raw.items():
    for drug_id in drug_list:
        gt_set.add((disease_id, drug_id))

print(f"Total predictions: {len(preds)}")
print(f"GT pairs: {len(gt_set)}")

# ----- Step 2: Filter LOW predictions and mark GT hits -----

print("\n" + "=" * 60)
print("STEP 2: Identify LOW predictions and GT hits")
print("=" * 60)

low_preds = [p for p in preds if p["confidence_tier"] == "LOW"]
print(f"LOW predictions: {len(low_preds)}")

# Mark GT hits
for p in low_preds:
    p["gt_hit"] = (p["disease_id"], p["drug_id"]) in gt_set

hits = [p for p in low_preds if p["gt_hit"]]
misses = [p for p in low_preds if not p["gt_hit"]]
print(f"GT hits: {len(hits)} ({len(hits)/len(low_preds):.1%})")
print(f"GT misses: {len(misses)} ({len(misses)/len(low_preds):.1%})")

# ----- Step 3: Characterize by tier_rule -----

print("\n" + "=" * 60)
print("STEP 3: GT hit rate by tier_rule")
print("=" * 60)

rule_stats = defaultdict(lambda: {"hits": 0, "total": 0})
for p in low_preds:
    rule = p["tier_rule"]
    rule_stats[rule]["total"] += 1
    if p["gt_hit"]:
        rule_stats[rule]["hits"] += 1

print(f"\n{'Tier Rule':<45s} {'Hits':>5s} {'Total':>6s} {'Hit%':>6s}")
print("-" * 65)
for rule, s in sorted(rule_stats.items(), key=lambda x: x[1]["hits"]/max(x[1]["total"],1), reverse=True):
    pct = s["hits"] / s["total"] * 100 if s["total"] > 0 else 0
    print(f"{rule:<45s} {s['hits']:>5d} {s['total']:>6d} {pct:>5.1f}%")

# ----- Step 4: Characterize by category -----

print("\n" + "=" * 60)
print("STEP 4: GT hit rate by disease category")
print("=" * 60)

cat_stats = defaultdict(lambda: {"hits": 0, "total": 0})
for p in low_preds:
    cat = p.get("category", "unknown")
    cat_stats[cat]["total"] += 1
    if p["gt_hit"]:
        cat_stats[cat]["hits"] += 1

print(f"\n{'Category':<25s} {'Hits':>5s} {'Total':>6s} {'Hit%':>6s}")
print("-" * 45)
for cat, s in sorted(cat_stats.items(), key=lambda x: x[1]["hits"]/max(x[1]["total"],1), reverse=True):
    pct = s["hits"] / s["total"] * 100 if s["total"] > 0 else 0
    print(f"{cat:<25s} {s['hits']:>5d} {s['total']:>6d} {pct:>5.1f}%")

# ----- Step 5: Characterize by kNN rank -----

print("\n" + "=" * 60)
print("STEP 5: GT hit rate by kNN rank bucket")
print("=" * 60)

for rank_min, rank_max in [(1, 5), (6, 10), (11, 15), (16, 20), (21, 30)]:
    bucket = [p for p in low_preds if rank_min <= p["rank"] <= rank_max]
    bucket_hits = sum(1 for p in bucket if p["gt_hit"])
    if bucket:
        print(f"  Rank {rank_min}-{rank_max}: {bucket_hits}/{len(bucket)} = {bucket_hits/len(bucket):.1%}")

# ----- Step 6: Characterize by mechanism support -----

print("\n" + "=" * 60)
print("STEP 6: GT hit rate by mechanism support")
print("=" * 60)

for mech in [True, False]:
    subset = [p for p in low_preds if p.get("mechanism_support") == mech]
    sub_hits = sum(1 for p in subset if p["gt_hit"])
    if subset:
        print(f"  mechanism_support={mech}: {sub_hits}/{len(subset)} = {sub_hits/len(subset):.1%}")

# TransE consilience
for transe in [True, False]:
    subset = [p for p in low_preds if p.get("transe_consilience") == transe]
    sub_hits = sum(1 for p in subset if p["gt_hit"])
    if subset:
        print(f"  transe_consilience={transe}: {sub_hits}/{len(subset)} = {sub_hits/len(subset):.1%}")

# ----- Step 7: Compound signals -----

print("\n" + "=" * 60)
print("STEP 7: Compound signal analysis")
print("=" * 60)

# TransE + mechanism + high rank
for combo_name, condition in [
    ("TransE + Mech", lambda p: p.get("transe_consilience") and p.get("mechanism_support")),
    ("TransE + Rank<=10", lambda p: p.get("transe_consilience") and p["rank"] <= 10),
    ("Mech + Rank<=5", lambda p: p.get("mechanism_support") and p["rank"] <= 5),
    ("TransE + Mech + Rank<=10", lambda p: p.get("transe_consilience") and p.get("mechanism_support") and p["rank"] <= 10),
    ("Gene overlap > 0", lambda p: p.get("gene_overlap_count", 0) and p.get("gene_overlap_count", 0) > 0),
    ("TransE + Gene>0", lambda p: p.get("transe_consilience") and p.get("gene_overlap_count", 0) and p.get("gene_overlap_count", 0) > 0),
]:
    subset = [p for p in low_preds if condition(p)]
    sub_hits = sum(1 for p in subset if p["gt_hit"])
    if subset:
        print(f"  {combo_name:<35s}: {sub_hits}/{len(subset)} = {sub_hits/len(subset):.1%}")

# ----- Step 8: Category × rule cross-tabulation -----

print("\n" + "=" * 60)
print("STEP 8: Category × tier_rule success rates (top 10 by count)")
print("=" * 60)

cross = defaultdict(lambda: {"hits": 0, "total": 0})
for p in low_preds:
    key = (p.get("category", "unknown"), p["tier_rule"])
    cross[key]["total"] += 1
    if p["gt_hit"]:
        cross[key]["hits"] += 1

# Sort by hit rate, filter n>=20
combos = [(k, v) for k, v in cross.items() if v["total"] >= 20]
combos.sort(key=lambda x: x[1]["hits"]/max(x[1]["total"], 1), reverse=True)

print(f"\n{'Category × Rule':<55s} {'Hits':>5s} {'Total':>6s} {'Hit%':>6s}")
print("-" * 75)
for (cat, rule), s in combos[:20]:
    pct = s["hits"] / s["total"] * 100
    print(f"{cat+' × '+rule:<55s} {s['hits']:>5d} {s['total']:>6d} {pct:>5.1f}%")

# ----- Step 9: Top individual successful LOW predictions -----

print("\n" + "=" * 60)
print("STEP 9: Top 20 successful LOW predictions (by kNN score)")
print("=" * 60)

hits_sorted = sorted(hits, key=lambda p: p["knn_score"], reverse=True)
print(f"\n{'Disease':<35s} {'Drug':<25s} {'Score':>6s} {'Rule':<30s} {'Cat':<15s}")
print("-" * 115)
for p in hits_sorted[:20]:
    dname = str(p["disease_name"])[:34]
    drname = str(p["drug_name"])[:24]
    print(f"{dname:<35s} {drname:<25s} {p['knn_score']:>6.2f} {p['tier_rule']:<30s} {p.get('category',''):<15s}")

# ----- Step 10: Known indication check -----

print("\n" + "=" * 60)
print("STEP 10: Known indication rate in LOW hits")
print("=" * 60)

known = sum(1 for p in hits if p.get("is_known_indication"))
print(f"Known indications in LOW GT hits: {known}/{len(hits)} ({known/len(hits):.1%})")
print(f"Known indications in LOW misses: {sum(1 for p in misses if p.get('is_known_indication'))}/{len(misses)}")

# ----- Step 11: Self-referentiality -----

print("\n" + "=" * 60)
print("STEP 11: Self-referentiality in LOW hits")
print("=" * 60)

sr_values = []
for p in hits:
    v = p.get("self_referential_pct")
    if v is not None and v != "":
        try:
            sr_values.append(float(v))
        except (ValueError, TypeError):
            pass

sr_miss = []
for p in misses:
    v = p.get("self_referential_pct")
    if v is not None and v != "":
        try:
            sr_miss.append(float(v))
        except (ValueError, TypeError):
            pass

if sr_values:
    print(f"Self-ref % in GT hits: mean={np.mean(sr_values):.1f}%, median={np.median(sr_values):.1f}%")
if sr_miss:
    print(f"Self-ref % in GT misses: mean={np.mean(sr_miss):.1f}%, median={np.median(sr_miss):.1f}%")

# High self-ref (>80%) breakdown
high_sr_hits = sum(1 for v in sr_values if v > 80)
all_sr = []
for p in low_preds:
    v = p.get("self_referential_pct")
    if v is not None and v != "":
        try:
            all_sr.append(float(v))
        except (ValueError, TypeError):
            pass
high_sr_total = sum(1 for v in all_sr if v > 80)
print(f"High self-ref (>80%): {high_sr_hits} hits out of {high_sr_total} predictions")

# ----- Step 12: Summary -----

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"\nLOW predictions: {len(low_preds)}")
print(f"Full-data GT hits: {len(hits)} ({len(hits)/len(low_preds):.1%})")
print(f"Known indications: {known}/{len(hits)} ({known/len(hits):.1%})")
print()
print("Top success patterns:")
for (cat, rule), s in combos[:5]:
    pct = s["hits"] / s["total"] * 100
    print(f"  {cat} × {rule}: {pct:.1f}% ({s['hits']}/{s['total']})")
