#!/usr/bin/env python3
"""
h590: Hetionet Disease-Resembles as Augmented kNN Signal

h586 found Hetionet DrD (disease resembles disease) edges have
partial_r=+0.228 (significant) with holdout, controlling for GT.
Only 216 diseases participate, but this is curated medical knowledge.

Can we use "resembles" edges to augment kNN neighborhoods?
If a disease has Hetionet-confirmed similar diseases, do those
neighbors provide better drug recommendations than the kNN-only neighbors?
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

# ----- Step 1: Load DRKG disease-disease resembles edges -----

print("=" * 60)
print("STEP 1: Load Hetionet disease-resembles edges")
print("=" * 60)

df = pd.read_csv(RAW_DIR / "drkg.tsv", sep="\t", header=None, names=["head", "relation", "tail"])
resem = df[df["relation"] == "Hetionet::DrD::Disease:Disease"]

# Build bidirectional resembles graph
resembles = defaultdict(set)
for _, row in resem.iterrows():
    h, t = row["head"], row["tail"]
    if h.startswith("Disease::") and t.startswith("Disease::"):
        resembles[h].add(t)
        resembles[t].add(h)

print(f"Resembles edges: {len(resem)} (raw)")
print(f"Diseases with resembles: {len(resembles)}")
resem_counts = [len(v) for v in resembles.values()]
print(f"Resembles per disease: mean={np.mean(resem_counts):.1f}, median={np.median(resem_counts):.1f}, max={max(resem_counts)}")

# ----- Step 2: Load GT, embeddings, holdout -----

print("\n" + "=" * 60)
print("STEP 2: Load GT, embeddings, holdout")
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

# ----- Step 3: Build kNN neighborhoods -----

print("\n" + "=" * 60)
print("STEP 3: Build kNN neighborhoods")
print("=" * 60)

from sklearn.metrics.pairwise import cosine_similarity

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

# ----- Step 4: Overlap between kNN and resembles -----

print("\n" + "=" * 60)
print("STEP 4: Overlap between kNN and resembles neighbors")
print("=" * 60)

# How many resembles neighbors are in kNN top-20?
common = sorted(set(embed_lookup.keys()) & set(holdout_prec.keys()) & set(disease_drugs_gt.keys()))
print(f"Holdout diseases with embeds + GT: {len(common)}")

# Focus on diseases that HAVE resembles edges
resem_holdout = [d for d in common if d in resembles]
print(f"Of those, with resembles edges: {len(resem_holdout)}")

overlap_stats = []
for d in resem_holdout:
    knn_set = {nb for nb, _ in knn_neighbors.get(d, [])}
    resem_set = resembles.get(d, set())
    # Only count resembles with GT (so they can contribute drugs)
    resem_with_gt = resem_set & set(disease_drugs_gt.keys())

    in_knn = knn_set & resem_with_gt
    not_in_knn = resem_with_gt - knn_set

    overlap_stats.append({
        "disease": d,
        "n_resembles": len(resem_set),
        "n_resembles_with_gt": len(resem_with_gt),
        "n_in_knn": len(in_knn),
        "n_not_in_knn": len(not_in_knn),
        "holdout": holdout_prec[d],
        "gt_size": len(disease_drugs_gt[d]),
    })

osdf = pd.DataFrame(overlap_stats)
print(f"\nOverlap statistics:")
print(f"  Mean resembles with GT: {osdf.n_resembles_with_gt.mean():.1f}")
print(f"  Mean in kNN: {osdf.n_in_knn.mean():.1f}")
print(f"  Mean NOT in kNN: {osdf.n_not_in_knn.mean():.1f}")
print(f"  Overlap rate: {osdf.n_in_knn.sum()/(osdf.n_resembles_with_gt.sum()):.1%}")

# ----- Step 5: Drug overlap for resembles vs kNN neighbors -----

print("\n" + "=" * 60)
print("STEP 5: Drug overlap: resembles vs kNN neighbors")
print("=" * 60)

def jaccard(s1, s2):
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)

results = []

for d in resem_holdout:
    d_drugs = disease_drugs_gt.get(d, set())
    knn_nbs = knn_neighbors.get(d, [])
    resem_nbs = resembles.get(d, set()) & set(disease_drugs_gt.keys())

    # Drug Jaccard with kNN neighbors
    knn_drug_j = [jaccard(d_drugs, disease_drugs_gt.get(nb, set())) for nb, _ in knn_nbs]
    mean_knn_dj = np.mean(knn_drug_j) if knn_drug_j else 0

    # Drug Jaccard with resembles neighbors
    resem_drug_j = [jaccard(d_drugs, disease_drugs_gt.get(nb, set())) for nb in resem_nbs]
    mean_resem_dj = np.mean(resem_drug_j) if resem_drug_j else 0

    # Embedding similarity with resembles
    resem_esims = []
    for nb in resem_nbs:
        if nb in embed_lookup and d in embed_lookup:
            sim = np.dot(embed_lookup[d], embed_lookup[nb]) / (
                np.linalg.norm(embed_lookup[d]) * np.linalg.norm(embed_lookup[nb])
            )
            resem_esims.append(sim)
    mean_resem_esim = np.mean(resem_esims) if resem_esims else 0

    # kNN embedding similarity
    knn_esims = [s for _, s in knn_nbs]
    mean_knn_esim = np.mean(knn_esims) if knn_esims else 0

    # Drugs from kNN vs from resembles
    knn_drugs = set()
    for nb, _ in knn_nbs:
        knn_drugs.update(disease_drugs_gt.get(nb, set()))
    resem_drugs = set()
    for nb in resem_nbs:
        resem_drugs.update(disease_drugs_gt.get(nb, set()))

    # How many GT drugs are found by each method?
    knn_gt_hits = len(knn_drugs & d_drugs)
    resem_gt_hits = len(resem_drugs & d_drugs)
    both_gt_hits = len((knn_drugs | resem_drugs) & d_drugs)
    resem_only_hits = len((resem_drugs - knn_drugs) & d_drugs)

    results.append({
        "disease": d,
        "holdout": holdout_prec[d],
        "gt_size": len(d_drugs),
        "n_resembles": len(resem_nbs),
        "mean_knn_dj": mean_knn_dj,
        "mean_resem_dj": mean_resem_dj,
        "dj_delta": mean_resem_dj - mean_knn_dj,
        "mean_knn_esim": mean_knn_esim,
        "mean_resem_esim": mean_resem_esim,
        "knn_gt_hits": knn_gt_hits,
        "resem_gt_hits": resem_gt_hits,
        "both_gt_hits": both_gt_hits,
        "resem_only_hits": resem_only_hits,
        "knn_drugs": len(knn_drugs),
        "resem_drugs": len(resem_drugs),
    })

rdf = pd.DataFrame(results)

print(f"\nDrug overlap comparison (n={len(rdf)} diseases):")
print(f"  Mean kNN drug Jaccard: {rdf.mean_knn_dj.mean():.4f}")
print(f"  Mean resembles drug Jaccard: {rdf.mean_resem_dj.mean():.4f}")
print(f"  Delta: {rdf.dj_delta.mean():+.4f}")
print()
print(f"  kNN GT hits: {rdf.knn_gt_hits.mean():.1f} per disease")
print(f"  Resembles GT hits: {rdf.resem_gt_hits.mean():.1f} per disease")
print(f"  Combined GT hits: {rdf.both_gt_hits.mean():.1f} per disease")
print(f"  Resembles-ONLY GT hits: {rdf.resem_only_hits.mean():.1f} per disease")
print()
print(f"  Diseases where resembles finds NEW GT drugs: {(rdf.resem_only_hits > 0).sum()}/{len(rdf)}")

# ----- Step 6: Could resembles augment kNN? -----

print("\n" + "=" * 60)
print("STEP 6: Potential for resembles-augmented kNN")
print("=" * 60)

# For diseases where resembles provide unique GT hits, how many new predictions?
has_unique = rdf[rdf.resem_only_hits > 0]
no_unique = rdf[rdf.resem_only_hits == 0]

print(f"\nDiseases WITH unique resembles GT hits: n={len(has_unique)}")
if len(has_unique) > 0:
    print(f"  Mean holdout: {has_unique.holdout.mean():.1%}")
    print(f"  Mean GT size: {has_unique.gt_size.mean():.1f}")
    print(f"  Mean unique hits: {has_unique.resem_only_hits.mean():.1f}")
    print(f"  Mean kNN GT hits: {has_unique.knn_gt_hits.mean():.1f}")

print(f"\nDiseases WITHOUT unique resembles GT hits: n={len(no_unique)}")
if len(no_unique) > 0:
    print(f"  Mean holdout: {no_unique.holdout.mean():.1%}")
    print(f"  Mean GT size: {no_unique.gt_size.mean():.1f}")

# Embedding similarity comparison
print(f"\n--- Embedding similarity ---")
print(f"  Mean kNN embed sim: {rdf.mean_knn_esim.mean():.3f}")
print(f"  Mean resembles embed sim: {rdf.mean_resem_esim.mean():.3f}")
print(f"  Resembles are {'closer' if rdf.mean_resem_esim.mean() > rdf.mean_knn_esim.mean() else 'farther'} in embedding space")

# ----- Step 7: Case studies -----

print("\n" + "=" * 60)
print("STEP 7: Case studies of diseases with unique resembles hits")
print("=" * 60)

# Load disease names
with open(REF_DIR / "disease_ontology_mapping.json") as f:
    disease_names = json.load(f)
name_lookup = {}
for entry in disease_names:
    if isinstance(entry, dict) and "drkg_id" in entry:
        name_lookup[entry["drkg_id"].replace("drkg:", "")] = entry.get("disease_name", "unknown")

top_unique = rdf.nlargest(10, "resem_only_hits")
for _, row in top_unique.iterrows():
    d = row["disease"]
    name = name_lookup.get(d, d)
    print(f"\n  {name} ({d})")
    print(f"    GT: {int(row['gt_size'])}, Holdout: {row['holdout']:.1%}")
    print(f"    kNN hits: {int(row['knn_gt_hits'])}, Resembles hits: {int(row['resem_gt_hits'])}, Unique: {int(row['resem_only_hits'])}")
    print(f"    kNN drug J: {row['mean_knn_dj']:.4f}, Resembles drug J: {row['mean_resem_dj']:.4f}")

    # Show resembles neighbors
    resem_nbs = resembles.get(d, set()) & set(disease_drugs_gt.keys())
    for nb in sorted(resem_nbs)[:5]:
        nb_name = name_lookup.get(nb, nb)
        dj = jaccard(disease_drugs_gt[d], disease_drugs_gt.get(nb, set()))
        print(f"      → {nb_name}: drug J={dj:.3f}")

# ----- Step 8: Verdict -----

print("\n" + "=" * 60)
print("VERDICT")
print("=" * 60)

print(f"\n1. Resembles coverage: {len(resem_holdout)}/{len(common)} holdout diseases ({len(resem_holdout)/len(common):.1%})")
print(f"2. Drug overlap: resembles = {rdf.mean_resem_dj.mean():.4f} vs kNN = {rdf.mean_knn_dj.mean():.4f}")
print(f"3. Unique GT hits from resembles: {(rdf.resem_only_hits > 0).sum()} diseases")
print(f"4. Mean unique hits: {rdf.resem_only_hits.mean():.1f} per disease")

# Save
rdf.to_csv(OUT_DIR / "h590_hetionet_resembles.csv", index=False)
print(f"\nResults saved to {OUT_DIR / 'h590_hetionet_resembles.csv'}")
