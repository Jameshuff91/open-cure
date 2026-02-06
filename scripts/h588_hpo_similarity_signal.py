#!/usr/bin/env python3
"""
h588: HPO Symptom Phenotype Similarity as Quality Signal

h586 found that Hetionet symptom Jaccard has partial_r=+0.296 (p=0.07)
with holdout precision, but only 39 diseases have symptom edges.
HPO similarity matrix covers 799 diseases — can it extend the signal?

Approach:
1. Load HPO similarity matrix (799 diseases)
2. For each disease with holdout precision, compute mean HPO similarity to kNN neighbors
3. Test correlation with holdout precision
4. Compute HPO-based "paradigm mismatch" (embed_sim - hpo_sim)
5. Compare with drug Jaccard (oracle) signal
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import stats
from pathlib import Path

DATA_DIR = Path("data")
REF_DIR = DATA_DIR / "reference"
EMBED_DIR = DATA_DIR / "embeddings"
OUT_DIR = DATA_DIR / "analysis"

# ----- Step 1: Load HPO similarity matrix -----

print("=" * 60)
print("STEP 1: Load HPO similarity matrix")
print("=" * 60)

hpo_data = np.load(REF_DIR / "hpo_similarity_matrix.npz", allow_pickle=True)
hpo_sim = hpo_data["similarities"]  # 799x799
hpo_diseases = hpo_data["diseases"]  # 799 disease IDs

# Strip 'drkg:' prefix
hpo_diseases_clean = [d.replace("drkg:", "") for d in hpo_diseases]
hpo_idx = {d: i for i, d in enumerate(hpo_diseases_clean)}

print(f"HPO similarity matrix: {hpo_sim.shape}")
print(f"HPO diseases: {len(hpo_diseases_clean)}")
print(f"HPO sim range: [{hpo_sim.min():.3f}, {hpo_sim.max():.3f}]")
print(f"HPO sim mean (off-diagonal): {hpo_sim[np.triu_indices_from(hpo_sim, k=1)].mean():.4f}")

# ----- Step 2: Load Node2Vec embeddings -----

print("\n" + "=" * 60)
print("STEP 2: Load Node2Vec embeddings")
print("=" * 60)

embed_df = pd.read_csv(EMBED_DIR / "node2vec_256_named.csv")
disease_embeds = embed_df[embed_df["entity"].str.startswith("Disease::")]
embed_cols = [c for c in disease_embeds.columns if c != "entity"]

embed_lookup = {}
for _, row in disease_embeds.iterrows():
    embed_lookup[row["entity"]] = np.array([row[c] for c in embed_cols])

print(f"Disease embeddings: {len(embed_lookup)}")

# ----- Step 3: Load GT and holdout precision -----

print("\n" + "=" * 60)
print("STEP 3: Load GT and holdout precision")
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

print(f"GT diseases: {len(disease_drugs_gt)}")
print(f"Holdout diseases: {len(holdout_prec)}")

# ----- Step 4: Compute kNN and signals -----

print("\n" + "=" * 60)
print("STEP 4: Compute kNN neighborhoods and HPO signals")
print("=" * 60)

from sklearn.metrics.pairwise import cosine_similarity

# Find diseases with embeddings, GT, AND holdout precision
common = sorted(set(embed_lookup.keys()) & set(holdout_prec.keys()) & set(disease_drugs_gt.keys()))
print(f"Diseases with embeddings + holdout + GT: {len(common)}")

# How many also have HPO?
hpo_common = [d for d in common if d in hpo_idx]
print(f"Of those, with HPO coverage: {len(hpo_common)} ({len(hpo_common)/len(common):.1%})")

# Build embedding matrix for ALL diseases with embeddings + GT (for kNN)
all_gt_diseases = sorted(set(embed_lookup.keys()) & set(disease_drugs_gt.keys()))
embed_matrix = np.array([embed_lookup[d] for d in all_gt_diseases])
disease_list_idx = {d: i for i, d in enumerate(all_gt_diseases)}

# Compute pairwise cosine similarity
sim_matrix = cosine_similarity(embed_matrix)
np.fill_diagonal(sim_matrix, 0)

# Get k=20 nearest neighbors
k = 20
knn_neighbors = {}
for i, d in enumerate(all_gt_diseases):
    sims = sim_matrix[i]
    top_k_idx = np.argsort(sims)[-k:][::-1]
    knn_neighbors[d] = [(all_gt_diseases[j], sims[j]) for j in top_k_idx]

# ----- Step 5: Compute HPO-based signals for holdout diseases -----

print("\n" + "=" * 60)
print("STEP 5: Compute HPO-based signals")
print("=" * 60)

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
    gt_size = len(d_drugs)
    hp = holdout_prec[d]

    # Embedding similarity stats
    embed_sims = [s for _, s in neighbors]
    mean_embed_sim = np.mean(embed_sims)

    # Drug Jaccard (oracle)
    drug_jaccards = []
    for nb, _ in neighbors:
        nb_drugs = disease_drugs_gt.get(nb, set())
        drug_jaccards.append(jaccard(d_drugs, nb_drugs))
    mean_drug_jaccard = np.mean(drug_jaccards)

    # HPO similarity with neighbors
    hpo_sims = []
    hpo_coverage = 0
    if d in hpo_idx:
        d_hpo_i = hpo_idx[d]
        for nb, _ in neighbors:
            if nb in hpo_idx:
                nb_hpo_i = hpo_idx[nb]
                hpo_sims.append(hpo_sim[d_hpo_i, nb_hpo_i])
                hpo_coverage += 1

    mean_hpo_sim = np.mean(hpo_sims) if hpo_sims else np.nan
    hpo_neighbor_coverage = hpo_coverage / k if k > 0 else 0

    # HPO-based mismatch: embed_sim - hpo_sim
    hpo_mismatch = mean_embed_sim - mean_hpo_sim if not np.isnan(mean_hpo_sim) else np.nan

    # GT mismatch (oracle)
    gt_mismatch = mean_embed_sim - mean_drug_jaccard

    results.append({
        "disease": d,
        "holdout_precision": hp,
        "gt_size": gt_size,
        "mean_embed_sim": mean_embed_sim,
        "mean_drug_jaccard": mean_drug_jaccard,
        "mean_hpo_sim": mean_hpo_sim,
        "hpo_neighbor_coverage": hpo_neighbor_coverage,
        "hpo_mismatch": hpo_mismatch,
        "gt_mismatch": gt_mismatch,
        "has_hpo": d in hpo_idx,
    })

rdf = pd.DataFrame(results)
print(f"Total diseases analyzed: {len(rdf)}")
print(f"With HPO coverage: {rdf['has_hpo'].sum()} ({rdf['has_hpo'].mean():.1%})")
print(f"With HPO sim computed: {rdf['mean_hpo_sim'].notna().sum()}")

# ----- Step 6: Correlations -----

print("\n" + "=" * 60)
print("STEP 6: Correlations with holdout precision")
print("=" * 60)

# All diseases
prec = rdf["holdout_precision"].values

print("\n--- All diseases ---")
for col in ["mean_embed_sim", "mean_drug_jaccard", "gt_mismatch", "gt_size"]:
    vals = rdf[col].values
    r, p = stats.pearsonr(vals, prec)
    print(f"  {col:30s}: r={r:+.3f} (p={p:.4f}), n={len(vals)}")

# HPO-covered diseases only
hpo_df = rdf.dropna(subset=["mean_hpo_sim"])
print(f"\n--- HPO-covered diseases (n={len(hpo_df)}) ---")

for col in ["mean_hpo_sim", "hpo_mismatch", "hpo_neighbor_coverage",
            "mean_embed_sim", "mean_drug_jaccard", "gt_mismatch", "gt_size"]:
    vals = hpo_df[col].values
    hp_vals = hpo_df["holdout_precision"].values
    r, p = stats.pearsonr(vals, hp_vals)
    print(f"  {col:30s}: r={r:+.3f} (p={p:.4f})")

# Partial correlations controlling for GT size
print(f"\n--- Partial correlations (controlling for GT size), n={len(hpo_df)} ---")

def partial_corr(x, y, z):
    slope_xz = np.polyfit(z, x, 1)
    x_resid = x - np.polyval(slope_xz, z)
    slope_yz = np.polyfit(z, y, 1)
    y_resid = y - np.polyval(slope_yz, z)
    return stats.pearsonr(x_resid, y_resid)

hp_prec = hpo_df["holdout_precision"].values
hp_gt = hpo_df["gt_size"].values

for col in ["mean_hpo_sim", "hpo_mismatch", "hpo_neighbor_coverage",
            "mean_drug_jaccard", "gt_mismatch", "mean_embed_sim"]:
    vals = hpo_df[col].values
    pr, pp = partial_corr(vals, hp_prec, hp_gt)
    print(f"  {col:30s}: partial_r={pr:+.3f} (p={pp:.4f})")

# ----- Step 7: HPO sim vs Drug Jaccard -----

print("\n" + "=" * 60)
print("STEP 7: Can HPO similarity proxy Drug Jaccard?")
print("=" * 60)

r, p = stats.pearsonr(hpo_df["mean_hpo_sim"].values, hpo_df["mean_drug_jaccard"].values)
print(f"HPO sim vs Drug Jaccard: r={r:+.3f} (p={p:.4f})")

pr, pp = partial_corr(hpo_df["mean_hpo_sim"].values, hpo_df["mean_drug_jaccard"].values, hp_gt)
print(f"Partial (ctrl GT): r={pr:+.3f} (p={pp:.4f})")

# ----- Step 8: Quartile analysis -----

print("\n" + "=" * 60)
print("STEP 8: Quartile analysis")
print("=" * 60)

for signal_name in ["mean_hpo_sim", "hpo_mismatch", "gt_mismatch"]:
    print(f"\n--- {signal_name} quartiles ---")
    vals = hpo_df[signal_name]
    try:
        quartiles = pd.qcut(vals, 4, labels=False, duplicates='drop')
        q_labels = ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]
        n_bins = int(quartiles.max()) + 1
        for qi in range(n_bins):
            mask = quartiles == qi
            q_prec = hpo_df.loc[mask, "holdout_precision"]
            q_gt = hpo_df.loc[mask, "gt_size"]
            q_val = hpo_df.loc[mask, signal_name]
            n = mask.sum()
            label = q_labels[qi] if qi < len(q_labels) else f"Q{qi+1}"
            print(f"  {label}: holdout={q_prec.mean():.1%} ± {q_prec.std():.1%}, "
                  f"n={n}, mean_gt={q_gt.mean():.1f}, mean_signal={q_val.mean():.4f}")
    except Exception as e:
        print(f"  Quartile analysis failed: {e}")

# ----- Step 9: Independence from self-referentiality -----

print("\n" + "=" * 60)
print("STEP 9: Independence from self-referentiality")
print("=" * 60)

# Compute self-ref for HPO diseases
self_ref_pcts = {}
for d in hpo_df["disease"].values:
    neighbors = knn_neighbors.get(d, [])
    d_drugs = disease_drugs_gt.get(d, set())
    if not d_drugs or not neighbors:
        continue
    drug_freq = defaultdict(float)
    for nb, esim in neighbors:
        nb_drugs = disease_drugs_gt.get(nb, set())
        for drug in nb_drugs:
            drug_freq[drug] += esim
    top_drugs = sorted(drug_freq.keys(), key=lambda x: drug_freq[x], reverse=True)[:20]
    if top_drugs:
        self_ref = sum(1 for d2 in top_drugs if d2 in d_drugs) / len(top_drugs)
        self_ref_pcts[d] = self_ref

hpo_df_sr = hpo_df.copy()
hpo_df_sr["self_ref_pct"] = hpo_df_sr["disease"].map(self_ref_pcts)
hpo_df_sr = hpo_df_sr.dropna(subset=["self_ref_pct"])

print(f"HPO diseases with self-ref: {len(hpo_df_sr)}")

for col in ["mean_hpo_sim", "hpo_mismatch"]:
    r_sr, p_sr = stats.pearsonr(hpo_df_sr[col].values, hpo_df_sr["self_ref_pct"].values)
    print(f"  {col} vs self_ref: r={r_sr:+.3f} (p={p_sr:.4f})")

# Partial correlation controlling for self-ref
sr_vals = hpo_df_sr["self_ref_pct"].values
hp_vals2 = hpo_df_sr["holdout_precision"].values

print("\n--- Partial correlations (controlling for self_ref) ---")
for col in ["mean_hpo_sim", "hpo_mismatch"]:
    vals = hpo_df_sr[col].values
    pr, pp = partial_corr(vals, hp_vals2, sr_vals)
    print(f"  {col:30s}: partial_r={pr:+.3f} (p={pp:.4f})")

# Double control: GT size AND self-ref
print("\n--- Partial correlations (controlling for GT size AND self_ref) ---")
gt_vals = hpo_df_sr["gt_size"].values
from numpy.linalg import lstsq
for col in ["mean_hpo_sim", "hpo_mismatch"]:
    vals = hpo_df_sr[col].values
    Z = np.column_stack([gt_vals, sr_vals, np.ones_like(gt_vals)])
    bx, _, _, _ = lstsq(Z, vals, rcond=None)
    x_resid = vals - Z @ bx
    by, _, _, _ = lstsq(Z, hp_vals2, rcond=None)
    y_resid = hp_vals2 - Z @ by
    pr, pp = stats.pearsonr(x_resid, y_resid)
    print(f"  {col:30s}: partial_r={pr:+.3f} (p={pp:.4f})")

# ----- Step 10: Summary -----

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"\nHPO coverage: {len(hpo_df)} of {len(rdf)} holdout diseases ({len(hpo_df)/len(rdf):.1%})")
print(f"vs Hetionet symptom coverage: 39 of 312 (12.5%)")
print()

# Save results
rdf.to_csv(OUT_DIR / "h588_hpo_similarity.csv", index=False)
print(f"Results saved to {OUT_DIR / 'h588_hpo_similarity.csv'}")
