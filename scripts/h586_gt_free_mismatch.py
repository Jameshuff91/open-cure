#!/usr/bin/env python3
"""
h586: GT-Free Paradigm Mismatch Analysis

Goal: Can we approximate Drug Jaccard (which requires GT) using DRKG edge-based
disease-disease similarity? If yes, we get a non-circular quality signal for predictions.

Approach:
1. Compute disease-disease similarity using shared genes, anatomy, symptoms from DRKG
2. Compare with drug Jaccard (GT-based) from h583
3. Compute GT-free "paradigm mismatch" = embedding_sim - drkg_edge_sim
4. Test if GT-free mismatch predicts holdout precision
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

# ----- Step 1: Load DRKG edges and build disease neighbor sets -----

print("=" * 60)
print("STEP 1: Load DRKG edges and build disease neighbor sets")
print("=" * 60)

df = pd.read_csv(RAW_DIR / "drkg.tsv", sep="\t", header=None, names=["head", "relation", "tail"])
print(f"Total DRKG edges: {len(df):,}")

# Classify disease edges
DISEASE_GENE_RELS = [
    "GNBR::L::Gene:Disease", "GNBR::J::Gene:Disease", "GNBR::U::Gene:Disease",
    "GNBR::G::Gene:Disease", "GNBR::X::Gene:Disease", "GNBR::Y::Gene:Disease",
    "GNBR::Md::Gene:Disease", "GNBR::Ud::Gene:Disease", "GNBR::Te::Gene:Disease",
    "GNBR::D::Gene:Disease",
    "Hetionet::DaG::Disease:Gene", "Hetionet::DuG::Disease:Gene",
    "Hetionet::DdG::Disease:Gene",
    "bioarx::Covid2_acc_host_gene::Disease:Gene",
    "bioarx::Coronavirus_ass_host_gene::Disease:Gene",
]
DISEASE_ANATOMY_RELS = ["Hetionet::DlA::Disease:Anatomy"]
DISEASE_SYMPTOM_RELS = ["Hetionet::DpS::Disease:Symptom"]
DISEASE_DISEASE_RELS = ["Hetionet::DrD::Disease:Disease"]

# Treatment/compound edges (EXCLUDE from similarity)
TREATMENT_RELS = [
    "GNBR::T::Compound:Disease", "DRUGBANK::treats::Compound:Disease",
    "Hetionet::CtD::Compound:Disease", "GNBR::Pa::Compound:Disease",
    "GNBR::C::Compound:Disease", "GNBR::Pr::Compound:Disease",
    "GNBR::Sa::Compound:Disease", "GNBR::Mp::Compound:Disease",
    "GNBR::J::Compound:Disease", "Hetionet::CpD::Compound:Disease",
]

# Build disease -> {genes}, {anatomy}, {symptoms}, {similar_diseases}
disease_genes = defaultdict(set)
disease_anatomy = defaultdict(set)
disease_symptoms = defaultdict(set)
disease_resembles = defaultdict(set)

for _, row in df.iterrows():
    h, r, t = row["head"], row["relation"], row["tail"]

    if r in DISEASE_GENE_RELS:
        # Gene:Disease -> head=Gene, tail=Disease  OR Disease:Gene -> head=Disease, tail=Gene
        if h.startswith("Disease::"):
            disease_genes[h].add(t)
        elif t.startswith("Disease::"):
            disease_genes[t].add(h)
    elif r in DISEASE_ANATOMY_RELS:
        if h.startswith("Disease::"):
            disease_anatomy[h].add(t)
        elif t.startswith("Disease::"):
            disease_anatomy[t].add(h)
    elif r in DISEASE_SYMPTOM_RELS:
        if h.startswith("Disease::"):
            disease_symptoms[h].add(t)
        elif t.startswith("Disease::"):
            disease_symptoms[t].add(h)
    elif r in DISEASE_DISEASE_RELS:
        if h.startswith("Disease::") and t.startswith("Disease::"):
            disease_resembles[h].add(t)
            disease_resembles[t].add(h)

print(f"\nDiseases with gene edges: {len(disease_genes)}")
print(f"Diseases with anatomy edges: {len(disease_anatomy)}")
print(f"Diseases with symptom edges: {len(disease_symptoms)}")
print(f"Diseases with resembles edges: {len(disease_resembles)}")

# Gene edge stats
gene_counts = [len(v) for v in disease_genes.values()]
print(f"\nGene edges per disease: mean={np.mean(gene_counts):.1f}, median={np.median(gene_counts):.1f}, max={max(gene_counts)}")

# ----- Step 2: Load Node2Vec embeddings and GT -----

print("\n" + "=" * 60)
print("STEP 2: Load Node2Vec embeddings and ground truth")
print("=" * 60)

# Load embeddings
embed_df = pd.read_csv(EMBED_DIR / "node2vec_256_named.csv")
disease_embeds = embed_df[embed_df["entity"].str.startswith("Disease::")]
print(f"Disease embeddings: {len(disease_embeds)}")

# Build embedding lookup
embed_cols = [c for c in disease_embeds.columns if c != "entity"]
embed_lookup = {}
for _, row in disease_embeds.iterrows():
    embed_lookup[row["entity"]] = np.array([row[c] for c in embed_cols])

# Load GT - format: dict of "drkg:Disease::..." -> ["drkg:Compound::...", ...]
with open(REF_DIR / "expanded_ground_truth.json") as f:
    gt = json.load(f)

# Build disease -> {drugs} from GT, stripping 'drkg:' prefix
disease_drugs_gt = defaultdict(set)
for disease_id, drug_list in gt.items():
    d = disease_id.replace("drkg:", "")  # "Disease::MESH:D003550"
    disease_drugs_gt[d] = {drug.replace("drkg:", "") for drug in drug_list}

print(f"Diseases with GT drugs: {len(disease_drugs_gt)}")

# ----- Step 3: Load kNN neighbors (from production predictor) -----

print("\n" + "=" * 60)
print("STEP 3: Compute kNN neighborhoods")
print("=" * 60)

# Compute embedding-based kNN (k=20)
from sklearn.metrics.pairwise import cosine_similarity

# Get all diseases that have both embeddings and GT
common_diseases = sorted(set(embed_lookup.keys()) & set(disease_drugs_gt.keys()))
print(f"Diseases with both embeddings and GT: {len(common_diseases)}")

# Build embedding matrix for common diseases
embed_matrix = np.array([embed_lookup[d] for d in common_diseases])
disease_idx = {d: i for i, d in enumerate(common_diseases)}

# Compute pairwise cosine similarity
sim_matrix = cosine_similarity(embed_matrix)
np.fill_diagonal(sim_matrix, 0)  # Zero out self-similarity

# Get k=20 nearest neighbors for each disease
k = 20
knn_neighbors = {}
for i, d in enumerate(common_diseases):
    sims = sim_matrix[i]
    top_k_idx = np.argsort(sims)[-k:][::-1]
    knn_neighbors[d] = [(common_diseases[j], sims[j]) for j in top_k_idx]

# ----- Step 4: Compute disease-disease similarity metrics -----

print("\n" + "=" * 60)
print("STEP 4: Compute disease-disease similarity metrics")
print("=" * 60)

def jaccard(s1, s2):
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)

# For each disease, compute metrics with its kNN neighbors
results = []

for d in common_diseases:
    neighbors = knn_neighbors[d]

    d_genes = disease_genes.get(d, set())
    d_anat = disease_anatomy.get(d, set())
    d_symp = disease_symptoms.get(d, set())
    d_drugs = disease_drugs_gt.get(d, set())

    gene_jaccards = []
    anatomy_jaccards = []
    symptom_jaccards = []
    drug_jaccards = []
    embed_sims = []
    resembles_count = 0

    # Combined DRKG feature set for each neighbor
    combined_jaccards = []

    for nb, esim in neighbors:
        nb_genes = disease_genes.get(nb, set())
        nb_anat = disease_anatomy.get(nb, set())
        nb_symp = disease_symptoms.get(nb, set())
        nb_drugs = disease_drugs_gt.get(nb, set())

        gj = jaccard(d_genes, nb_genes)
        aj = jaccard(d_anat, nb_anat)
        sj = jaccard(d_symp, nb_symp)
        dj = jaccard(d_drugs, nb_drugs)

        gene_jaccards.append(gj)
        anatomy_jaccards.append(aj)
        symptom_jaccards.append(sj)
        drug_jaccards.append(dj)
        embed_sims.append(esim)

        # Combined: union of all non-treatment features
        d_combined = d_genes | d_anat | d_symp
        nb_combined = nb_genes | nb_anat | nb_symp
        cj = jaccard(d_combined, nb_combined)
        combined_jaccards.append(cj)

        if nb in disease_resembles.get(d, set()):
            resembles_count += 1

    gt_size = len(d_drugs)

    results.append({
        "disease": d,
        "gt_size": gt_size,
        "n_genes": len(d_genes),
        "n_anatomy": len(d_anat),
        "n_symptoms": len(d_symp),
        "mean_gene_jaccard": np.mean(gene_jaccards),
        "mean_anatomy_jaccard": np.mean(anatomy_jaccards),
        "mean_symptom_jaccard": np.mean(symptom_jaccards),
        "mean_drug_jaccard": np.mean(drug_jaccards),  # GT-based (oracle)
        "mean_combined_jaccard": np.mean(combined_jaccards),  # GT-free!
        "mean_embed_sim": np.mean(embed_sims),
        "resembles_count": resembles_count,
        # Mismatch scores
        "gt_mismatch": np.mean(embed_sims) - np.mean(drug_jaccards),  # h583 oracle
        "gtfree_mismatch": np.mean(embed_sims) - np.mean(combined_jaccards),  # GT-free!
        "gene_mismatch": np.mean(embed_sims) - np.mean(gene_jaccards),  # Gene-only
    })

results_df = pd.DataFrame(results)
print(f"Computed metrics for {len(results_df)} diseases")

# ----- Step 5: Load holdout precision -----

print("\n" + "=" * 60)
print("STEP 5: Load holdout precision and correlate")
print("=" * 60)

with open(REF_DIR / "disease_holdout_precision.json") as f:
    holdout_prec_raw = json.load(f)

# Holdout keys have 'drkg:' prefix; values are dicts with 'holdout_precision' key
holdout_prec = {}
for k, v in holdout_prec_raw.items():
    disease_key = k.replace("drkg:", "")
    if isinstance(v, dict):
        hp_val = v.get("holdout_precision")
        if hp_val is not None:
            holdout_prec[disease_key] = float(hp_val) / 100.0  # Convert % to fraction
    else:
        holdout_prec[disease_key] = float(v)

# Merge with results
results_df["holdout_precision"] = results_df["disease"].map(holdout_prec)
has_holdout = results_df.dropna(subset=["holdout_precision"]).copy()
print(f"Diseases with holdout precision: {len(has_holdout)}")

# Compute correlations
print("\n--- Correlations with holdout precision ---")
signals = [
    "mean_gene_jaccard", "mean_anatomy_jaccard", "mean_symptom_jaccard",
    "mean_drug_jaccard", "mean_combined_jaccard", "mean_embed_sim",
    "resembles_count",
    "gt_mismatch", "gtfree_mismatch", "gene_mismatch",
    "gt_size", "n_genes", "n_anatomy", "n_symptoms",
]

for sig in signals:
    vals = has_holdout[sig].values
    prec = has_holdout["holdout_precision"].values
    if np.std(vals) > 0:
        r, p = stats.pearsonr(vals, prec)
        print(f"  {sig:30s}: r={r:+.3f} (p={p:.4f})")
    else:
        print(f"  {sig:30s}: constant (no variance)")

# ----- Step 6: Partial correlations controlling for GT size -----

print("\n--- Partial correlations (controlling for gt_size) ---")

def partial_corr(x, y, z):
    """Partial correlation of x and y controlling for z."""
    # Residualize x on z
    slope_xz = np.polyfit(z, x, 1)
    x_resid = x - np.polyval(slope_xz, z)
    # Residualize y on z
    slope_yz = np.polyfit(z, y, 1)
    y_resid = y - np.polyval(slope_yz, z)
    return stats.pearsonr(x_resid, y_resid)

prec = has_holdout["holdout_precision"].values
gt = has_holdout["gt_size"].values

key_signals = [
    "mean_drug_jaccard", "mean_combined_jaccard", "mean_gene_jaccard",
    "gt_mismatch", "gtfree_mismatch", "gene_mismatch",
    "mean_embed_sim", "resembles_count",
]

for sig in key_signals:
    vals = has_holdout[sig].values
    if np.std(vals) > 0:
        pr, pp = partial_corr(vals, prec, gt)
        print(f"  {sig:30s}: partial_r={pr:+.3f} (p={pp:.4f})")

# ----- Step 7: Correlation between GT-free signals and drug Jaccard -----

print("\n" + "=" * 60)
print("STEP 6: Correlation between GT-free signals and drug Jaccard (oracle)")
print("=" * 60)

print("\n--- Can GT-free signals approximate drug Jaccard? ---")
dj = has_holdout["mean_drug_jaccard"].values
for sig in ["mean_combined_jaccard", "mean_gene_jaccard", "mean_anatomy_jaccard",
            "mean_symptom_jaccard", "resembles_count", "n_genes"]:
    vals = has_holdout[sig].values
    if np.std(vals) > 0:
        r, p = stats.pearsonr(vals, dj)
        print(f"  {sig:30s}: r={r:+.3f} (p={p:.4f}) with drug_jaccard")

# Partial correlation controlling for GT size
print("\n--- Partial correlation with drug_jaccard (ctrl GT size) ---")
for sig in ["mean_combined_jaccard", "mean_gene_jaccard"]:
    vals = has_holdout[sig].values
    if np.std(vals) > 0:
        pr, pp = partial_corr(vals, dj, gt)
        print(f"  {sig:30s}: partial_r={pr:+.3f} (p={pp:.4f})")

# ----- Step 8: Quartile analysis -----

print("\n" + "=" * 60)
print("STEP 7: Quartile analysis of GT-free mismatch vs holdout")
print("=" * 60)

for signal_name in ["gtfree_mismatch", "gene_mismatch", "mean_combined_jaccard", "gt_mismatch"]:
    print(f"\n--- {signal_name} quartiles ---")
    vals = has_holdout[signal_name]
    try:
        quartiles = pd.qcut(vals, 4, labels=False, duplicates='drop')
        q_labels = ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]
        n_bins = int(quartiles.max()) + 1
        for qi in range(n_bins):
            mask = quartiles == qi
            q_prec = has_holdout.loc[mask, "holdout_precision"]
            q_gt = has_holdout.loc[mask, "gt_size"]
            q_val = has_holdout.loc[mask, signal_name]
            n = mask.sum()
            label = q_labels[qi] if qi < len(q_labels) else f"Q{qi+1}"
            print(f"  {label}: holdout={q_prec.mean():.1%} ± {q_prec.std():.1%}, "
                  f"n={n}, mean_gt={q_gt.mean():.1f}, mean_signal={q_val.mean():.4f}")
    except Exception as e:
        print(f"  Quartile analysis failed: {e}")

# ----- Step 9: Compare GT-free mismatch with self-referentiality -----

print("\n" + "=" * 60)
print("STEP 8: GT-free mismatch vs self-referentiality independence")
print("=" * 60)

# Load self-referentiality if available
try:
    # Compute self-referentiality from GT
    # Self-ref = fraction of predictions that are already known
    # We'll approximate: diseases with small GT tend to be self-referential
    # But more precisely, let's check the production predictor data

    # Actually, let's compute it properly: for each disease, what fraction of its
    # kNN-predicted drugs are already in its GT?
    self_ref_pcts = {}
    for d in common_diseases:
        neighbors = knn_neighbors[d]
        d_drugs = disease_drugs_gt.get(d, set())
        if not d_drugs:
            continue

        # Get kNN predictions: drugs recommended by neighbors
        drug_freq = defaultdict(float)
        for nb, esim in neighbors:
            nb_drugs = disease_drugs_gt.get(nb, set())
            for drug in nb_drugs:
                drug_freq[drug] += esim

        # Top-20 predictions
        top_drugs = sorted(drug_freq.keys(), key=lambda x: drug_freq[x], reverse=True)[:20]

        # Self-ref = fraction of top-20 that are in GT
        if top_drugs:
            self_ref = sum(1 for d2 in top_drugs if d2 in d_drugs) / len(top_drugs)
            self_ref_pcts[d] = self_ref

    results_df["self_ref_pct"] = results_df["disease"].map(self_ref_pcts)
    has_sr = has_holdout.copy()
    has_sr["self_ref_pct"] = has_sr["disease"].map(self_ref_pcts)
    has_sr = has_sr.dropna(subset=["holdout_precision", "self_ref_pct"])

    # Correlation between GT-free mismatch and self-ref
    for sig in ["gtfree_mismatch", "gene_mismatch", "mean_combined_jaccard"]:
        r, p = stats.pearsonr(has_sr[sig].values, has_sr["self_ref_pct"].values)
        print(f"  {sig:30s} vs self_ref: r={r:+.3f} (p={p:.4f})")

    # Partial correlation with holdout, controlling for self-ref
    print("\n--- Partial correlations (controlling for self_ref) ---")
    sr = has_sr["self_ref_pct"].values
    prec2 = has_sr["holdout_precision"].values
    for sig in ["gtfree_mismatch", "gene_mismatch", "mean_combined_jaccard"]:
        vals = has_sr[sig].values
        pr, pp = partial_corr(vals, prec2, sr)
        print(f"  {sig:30s}: partial_r={pr:+.3f} (p={pp:.4f})")

    # Double control: GT size AND self-ref
    print("\n--- Partial correlations (controlling for gt_size AND self_ref) ---")
    gt2 = has_sr["gt_size"].values
    for sig in ["gtfree_mismatch", "gene_mismatch", "mean_combined_jaccard"]:
        vals = has_sr[sig].values
        # Residualize on both
        from numpy.linalg import lstsq
        Z = np.column_stack([gt2, sr, np.ones_like(gt2)])
        bx, _, _, _ = lstsq(Z, vals, rcond=None)
        x_resid = vals - Z @ bx
        by, _, _, _ = lstsq(Z, prec2, rcond=None)
        y_resid = prec2 - Z @ by
        pr, pp = stats.pearsonr(x_resid, y_resid)
        print(f"  {sig:30s}: partial_r={pr:+.3f} (p={pp:.4f})")

except Exception as e:
    print(f"Self-referentiality analysis failed: {e}")
    import traceback
    traceback.print_exc()

# ----- Step 10: Test specific prediction-level signal -----

print("\n" + "=" * 60)
print("STEP 9: Disease-level DRKG similarity as deliverable annotation")
print("=" * 60)

# Distribution of combined Jaccard
cj_vals = results_df["mean_combined_jaccard"]
print(f"mean_combined_jaccard: mean={cj_vals.mean():.4f}, median={cj_vals.median():.4f}, "
      f"max={cj_vals.max():.4f}")
print(f"  >0: {(cj_vals > 0).sum()} diseases ({(cj_vals > 0).mean():.1%})")
print(f"  =0: {(cj_vals == 0).sum()} diseases ({(cj_vals == 0).mean():.1%})")

gj_vals = results_df["mean_gene_jaccard"]
print(f"\nmean_gene_jaccard: mean={gj_vals.mean():.4f}, median={gj_vals.median():.4f}, "
      f"max={gj_vals.max():.4f}")
print(f"  >0: {(gj_vals > 0).sum()} diseases ({(gj_vals > 0).mean():.1%})")
print(f"  =0: {(gj_vals == 0).sum()} diseases ({(gj_vals == 0).mean():.1%})")

# ----- Step 11: Summary -----

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print("\nKey correlations with holdout precision:")
key_results = []
for sig in ["mean_drug_jaccard", "mean_combined_jaccard", "mean_gene_jaccard",
            "gt_mismatch", "gtfree_mismatch", "gt_size"]:
    vals = has_holdout[sig].values
    prec = has_holdout["holdout_precision"].values
    r, p = stats.pearsonr(vals, prec)
    pr, pp = partial_corr(vals, prec, has_holdout["gt_size"].values) if sig != "gt_size" else (r, p)
    key_results.append({"signal": sig, "r": r, "p": p, "partial_r": pr, "partial_p": pp})

for kr in key_results:
    print(f"  {kr['signal']:30s}: r={kr['r']:+.3f}, partial_r(ctrl GT)={kr['partial_r']:+.3f}")

# Save detailed results
results_df.to_csv(OUT_DIR / "h586_gt_free_mismatch.csv", index=False)
print(f"\nDetailed results saved to {OUT_DIR / 'h586_gt_free_mismatch.csv'}")
