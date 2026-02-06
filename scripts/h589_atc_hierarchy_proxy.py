#!/usr/bin/env python3
"""
h589: ATC Hierarchy as GT-Free Treatment Paradigm Proxy

h586 showed gene overlap cannot proxy drug Jaccard (r=0.086).
h588 showed HPO phenotype does better (r=0.390) but sparse.

ATC codes directly encode therapeutic use. For kNN neighbors, ATC
overlap at L2/L3 level might capture "same therapeutic area" even
when drugs don't overlap.

CIRCULARITY ANALYSIS: ATC codes come from GT drugs, so direct ATC
overlap IS circular. But the signal may be LESS circular than drug
Jaccard because:
1. ATC L2 groups many drugs → diseases can share ATC class without sharing drugs
2. ATC captures THERAPEUTIC INTENT, not just drug identity

We test: does ATC overlap predict holdout precision BEYOND drug Jaccard?
If so, ATC encodes information that drug identity alone doesn't.
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import stats
from pathlib import Path

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw" / "drkg"
REF_DIR = DATA_DIR / "reference"
EMBED_DIR = DATA_DIR / "embeddings"
OUT_DIR = DATA_DIR / "analysis"

# ----- Step 1: Build drug -> ATC mapping -----

print("=" * 60)
print("STEP 1: Build drug -> ATC mapping from DRKG")
print("=" * 60)

df = pd.read_csv(RAW_DIR / "drkg.tsv", sep="\t", header=None, names=["head", "relation", "tail"])
atc_edges = df[df["relation"] == "DRUGBANK::x-atc::Compound:Atc"]

# Build compound -> set of ATC codes
drug_atc = defaultdict(set)
for _, row in atc_edges.iterrows():
    compound = row["head"]  # Compound::DB00001
    atc_code = row["tail"].replace("Atc::", "")  # B01AE02
    drug_atc[compound].add(atc_code)

print(f"Drugs with ATC codes: {len(drug_atc)}")
print(f"Total ATC codes: {sum(len(v) for v in drug_atc.values())}")

# Build ATC at different levels
def atc_level(code, level):
    """Extract ATC code at given level (1-5)."""
    lengths = {1: 1, 2: 3, 3: 4, 4: 5, 5: 7}
    return code[:lengths.get(level, 7)]

def drug_atc_at_level(drug, level):
    """Get set of ATC codes at given level for a drug."""
    codes = drug_atc.get(drug, set())
    return {atc_level(c, level) for c in codes}

# ----- Step 2: Load GT, embeddings, holdout -----

print("\n" + "=" * 60)
print("STEP 2: Load GT, embeddings, holdout precision")
print("=" * 60)

with open(REF_DIR / "expanded_ground_truth.json") as f:
    gt_raw = json.load(f)

disease_drugs_gt = defaultdict(set)
for disease_id, drug_list in gt_raw.items():
    d = disease_id.replace("drkg:", "")
    disease_drugs_gt[d] = {drug.replace("drkg:", "") for drug in drug_list}

with open(REF_DIR / "disease_holdout_precision.json") as f:
    holdout_raw = json.load(f)

holdout_prec = {}
for k, v in holdout_raw.items():
    disease_key = k.replace("drkg:", "")
    if isinstance(v, dict):
        hp_val = v.get("holdout_precision")
        if hp_val is not None:
            holdout_prec[disease_key] = float(hp_val) / 100.0

embed_df = pd.read_csv(EMBED_DIR / "node2vec_256_named.csv")
disease_embeds = embed_df[embed_df["entity"].str.startswith("Disease::")]
embed_cols = [c for c in disease_embeds.columns if c != "entity"]
embed_lookup = {}
for _, row in disease_embeds.iterrows():
    embed_lookup[row["entity"]] = np.array([row[c] for c in embed_cols])

print(f"GT diseases: {len(disease_drugs_gt)}")
print(f"Holdout diseases: {len(holdout_prec)}")
print(f"Embeddings: {len(embed_lookup)}")

# ----- Step 3: Build disease ATC profiles -----

print("\n" + "=" * 60)
print("STEP 3: Build disease ATC profiles")
print("=" * 60)

# For each disease, get ATC codes of its GT drugs at each level
disease_atc = {}
for d, drugs in disease_drugs_gt.items():
    atc_profiles = {}
    for level in [1, 2, 3, 4, 5]:
        atc_set = set()
        for drug in drugs:
            atc_set.update(drug_atc_at_level(drug, level))
        atc_profiles[level] = atc_set
    disease_atc[d] = atc_profiles

# ATC coverage stats
has_atc = sum(1 for d, p in disease_atc.items() if len(p.get(2, set())) > 0)
print(f"Diseases with ATC L2 codes: {has_atc}/{len(disease_atc)}")

# Distribution of ATC L2 codes per disease
l2_counts = [len(p.get(2, set())) for p in disease_atc.values()]
print(f"ATC L2 per disease: mean={np.mean(l2_counts):.1f}, median={np.median(l2_counts):.1f}")

# ----- Step 4: Compute kNN and ATC signals -----

print("\n" + "=" * 60)
print("STEP 4: Compute kNN neighborhoods and ATC signals")
print("=" * 60)

from sklearn.metrics.pairwise import cosine_similarity

# Build kNN for all diseases with embeddings + GT
all_gt_diseases = sorted(set(embed_lookup.keys()) & set(disease_drugs_gt.keys()))
embed_matrix = np.array([embed_lookup[d] for d in all_gt_diseases])
sim_matrix = cosine_similarity(embed_matrix)
np.fill_diagonal(sim_matrix, 0)

k = 20
knn_neighbors = {}
for i, d in enumerate(all_gt_diseases):
    sims = sim_matrix[i]
    top_k_idx = np.argsort(sims)[-k:][::-1]
    knn_neighbors[d] = [(all_gt_diseases[j], sims[j]) for j in top_k_idx]

# Diseases with holdout, embeddings, and GT
common = sorted(set(embed_lookup.keys()) & set(holdout_prec.keys()) & set(disease_drugs_gt.keys()))
print(f"Common diseases: {len(common)}")

def jaccard(s1, s2):
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)

results = []

for d in common:
    neighbors = knn_neighbors.get(d, [])
    if not neighbors:
        continue

    d_drugs = disease_drugs_gt.get(d, set())
    d_atc = disease_atc.get(d, {})
    gt_size = len(d_drugs)
    hp = holdout_prec[d]

    embed_sims = [s for _, s in neighbors]
    mean_embed_sim = np.mean(embed_sims)

    # Drug Jaccard (oracle)
    drug_jaccards = []
    for nb, _ in neighbors:
        nb_drugs = disease_drugs_gt.get(nb, set())
        drug_jaccards.append(jaccard(d_drugs, nb_drugs))
    mean_drug_jaccard = np.mean(drug_jaccards)

    # ATC Jaccard at each level
    atc_jaccards = {}
    for level in [1, 2, 3, 4, 5]:
        d_atc_set = d_atc.get(level, set())
        jaccards = []
        for nb, _ in neighbors:
            nb_atc = disease_atc.get(nb, {}).get(level, set())
            jaccards.append(jaccard(d_atc_set, nb_atc))
        atc_jaccards[level] = np.mean(jaccards)

    # ATC coverage (fraction of neighbors with ATC codes)
    atc_coverage = sum(1 for nb, _ in neighbors if len(disease_atc.get(nb, {}).get(2, set())) > 0) / k

    results.append({
        "disease": d,
        "holdout_precision": hp,
        "gt_size": gt_size,
        "mean_embed_sim": mean_embed_sim,
        "mean_drug_jaccard": mean_drug_jaccard,
        "atc_l1_jaccard": atc_jaccards[1],
        "atc_l2_jaccard": atc_jaccards[2],
        "atc_l3_jaccard": atc_jaccards[3],
        "atc_l4_jaccard": atc_jaccards[4],
        "atc_l5_jaccard": atc_jaccards[5],
        "atc_coverage": atc_coverage,
    })

rdf = pd.DataFrame(results)
print(f"Results computed for {len(rdf)} diseases")

# ----- Step 5: Correlations -----

print("\n" + "=" * 60)
print("STEP 5: Correlations with holdout precision")
print("=" * 60)

prec = rdf["holdout_precision"].values
gt = rdf["gt_size"].values

def partial_corr(x, y, z):
    slope_xz = np.polyfit(z, x, 1)
    x_resid = x - np.polyval(slope_xz, z)
    slope_yz = np.polyfit(z, y, 1)
    y_resid = y - np.polyval(slope_yz, z)
    return stats.pearsonr(x_resid, y_resid)

print("\n--- Raw correlations ---")
for col in ["mean_drug_jaccard", "atc_l1_jaccard", "atc_l2_jaccard",
            "atc_l3_jaccard", "atc_l4_jaccard", "atc_l5_jaccard",
            "mean_embed_sim", "gt_size"]:
    vals = rdf[col].values
    r, p = stats.pearsonr(vals, prec)
    print(f"  {col:25s}: r={r:+.3f} (p={p:.4f})")

print("\n--- Partial correlations (controlling for GT size) ---")
for col in ["mean_drug_jaccard", "atc_l1_jaccard", "atc_l2_jaccard",
            "atc_l3_jaccard", "atc_l4_jaccard", "atc_l5_jaccard",
            "mean_embed_sim"]:
    vals = rdf[col].values
    pr, pp = partial_corr(vals, prec, gt)
    print(f"  {col:25s}: partial_r={pr:+.3f} (p={pp:.4f})")

# ----- Step 6: ATC as drug Jaccard proxy -----

print("\n" + "=" * 60)
print("STEP 6: ATC overlap vs Drug Jaccard")
print("=" * 60)

dj = rdf["mean_drug_jaccard"].values
for col in ["atc_l1_jaccard", "atc_l2_jaccard", "atc_l3_jaccard",
            "atc_l4_jaccard", "atc_l5_jaccard"]:
    vals = rdf[col].values
    r, p = stats.pearsonr(vals, dj)
    pr, pp = partial_corr(vals, dj, gt)
    print(f"  {col:25s}: r={r:+.3f} (partial={pr:+.3f}) with drug_jaccard")

# Compare with h586's gene overlap
print(f"\n  (h586 gene_jaccard: r=+0.086 with drug_jaccard)")
print(f"  (h588 HPO_sim: r=+0.390 with drug_jaccard)")

# ----- Step 7: ATC BEYOND drug Jaccard -----

print("\n" + "=" * 60)
print("STEP 7: Does ATC add signal beyond drug Jaccard?")
print("=" * 60)

# Partial correlation of ATC with holdout, controlling for drug Jaccard
print("\n--- ATC signal controlling for drug Jaccard ---")
for col in ["atc_l2_jaccard", "atc_l3_jaccard"]:
    vals = rdf[col].values
    pr, pp = partial_corr(vals, prec, dj)
    print(f"  {col:25s}: partial_r={pr:+.3f} (p={pp:.4f}) controlling drug_jaccard")

# Multiple regression
from sklearn.linear_model import LinearRegression

X_base = np.column_stack([gt, rdf["mean_embed_sim"].values])
r2_base = LinearRegression().fit(X_base, prec).score(X_base, prec)

X_dj = np.column_stack([gt, rdf["mean_embed_sim"].values, dj])
r2_dj = LinearRegression().fit(X_dj, prec).score(X_dj, prec)

X_atc2 = np.column_stack([gt, rdf["mean_embed_sim"].values, rdf["atc_l2_jaccard"].values])
r2_atc2 = LinearRegression().fit(X_atc2, prec).score(X_atc2, prec)

X_atc3 = np.column_stack([gt, rdf["mean_embed_sim"].values, rdf["atc_l3_jaccard"].values])
r2_atc3 = LinearRegression().fit(X_atc3, prec).score(X_atc3, prec)

X_both = np.column_stack([gt, rdf["mean_embed_sim"].values, dj, rdf["atc_l2_jaccard"].values])
r2_both = LinearRegression().fit(X_both, prec).score(X_both, prec)

print(f"\n--- Multiple regression R² ---")
print(f"  Base (GT + embed):          {r2_base:.4f}")
print(f"  + Drug Jaccard:             {r2_dj:.4f} (+{(r2_dj-r2_base)*100:.1f}%)")
print(f"  + ATC L2:                   {r2_atc2:.4f} (+{(r2_atc2-r2_base)*100:.1f}%)")
print(f"  + ATC L3:                   {r2_atc3:.4f} (+{(r2_atc3-r2_base)*100:.1f}%)")
print(f"  + Drug Jaccard + ATC L2:    {r2_both:.4f} (+{(r2_both-r2_base)*100:.1f}%)")
print(f"  ATC L2 incremental over DJ: {(r2_both-r2_dj)*100:.2f}%")

# ----- Step 8: Circularity analysis -----

print("\n" + "=" * 60)
print("STEP 8: Circularity analysis")
print("=" * 60)

# ATC overlap uses GT drugs → circular with drug Jaccard
# But ATC L2 aggregates many drugs into the same class
# So two diseases can have high ATC L2 overlap with zero drug overlap

# Let's test: among pairs with ZERO drug Jaccard, does ATC L2 still predict?
# This would show ATC captures something drug identity doesn't

# For each disease, compute drug Jaccard and ATC L2 Jaccard with each neighbor
zero_dj_atc = []
nonzero_dj_atc = []

for _, row in rdf.iterrows():
    d = row["disease"]
    d_drugs = disease_drugs_gt.get(d, set())
    d_atc_l2 = disease_atc.get(d, {}).get(2, set())

    for nb, esim in knn_neighbors.get(d, []):
        nb_drugs = disease_drugs_gt.get(nb, set())
        nb_atc_l2 = disease_atc.get(nb, {}).get(2, set())

        dj_val = jaccard(d_drugs, nb_drugs)
        atc_val = jaccard(d_atc_l2, nb_atc_l2)

        if dj_val == 0:
            zero_dj_atc.append(atc_val)
        else:
            nonzero_dj_atc.append(atc_val)

print(f"Neighbor pairs with drug_jaccard=0: {len(zero_dj_atc)}")
print(f"Neighbor pairs with drug_jaccard>0: {len(nonzero_dj_atc)}")
print(f"Mean ATC L2 Jaccard (drug_J=0): {np.mean(zero_dj_atc):.4f}")
print(f"Mean ATC L2 Jaccard (drug_J>0): {np.mean(nonzero_dj_atc):.4f}")
print(f"ATC L2 > 0 when drug_J=0: {sum(1 for x in zero_dj_atc if x > 0)}/{len(zero_dj_atc)} ({sum(1 for x in zero_dj_atc if x > 0)/len(zero_dj_atc):.1%})")

# ----- Step 9: Quartile analysis -----

print("\n" + "=" * 60)
print("STEP 9: Quartile analysis")
print("=" * 60)

for signal_name in ["atc_l2_jaccard", "atc_l3_jaccard", "mean_drug_jaccard"]:
    print(f"\n--- {signal_name} quartiles ---")
    vals = rdf[signal_name]
    try:
        quartiles = pd.qcut(vals, 4, labels=False, duplicates='drop')
        q_labels = ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]
        n_bins = int(quartiles.max()) + 1
        for qi in range(n_bins):
            mask = quartiles == qi
            q_prec = rdf.loc[mask, "holdout_precision"]
            q_gt = rdf.loc[mask, "gt_size"]
            q_val = rdf.loc[mask, signal_name]
            n = mask.sum()
            label = q_labels[qi] if qi < len(q_labels) else f"Q{qi+1}"
            print(f"  {label}: holdout={q_prec.mean():.1%} ± {q_prec.std():.1%}, "
                  f"n={n}, mean_gt={q_gt.mean():.1f}, mean_signal={q_val.mean():.4f}")
    except Exception as e:
        print(f"  Quartile analysis failed: {e}")

# ----- Step 10: Summary -----

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

# Save
rdf.to_csv(OUT_DIR / "h589_atc_hierarchy.csv", index=False)
print(f"\nResults saved to {OUT_DIR / 'h589_atc_hierarchy.csv'}")
