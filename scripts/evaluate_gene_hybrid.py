#!/usr/bin/env python3
"""
h35: Node2Vec + Gene-Disease Feature Hybrid

Tests whether adding gene-based features (drug targets, disease genes, shared genes)
to Node2Vec embeddings improves generalization on held-out diseases.

Features:
- n_shared: number of shared drug target / disease genes
- jaccard: Jaccard similarity of gene sets
- dice: Dice coefficient
- overlap_coeff: Szymkiewicz-Simpson overlap coefficient
- n_drug_targets: total drug target count
- n_disease_genes: total disease gene count
- has_drug_targets/has_disease_genes/has_both: binary flags
- log versions: log1p of counts

Configurations tested:
A. Node2Vec only (baseline, same as h29: expected ~28.73%)
B. Gene features only (inductive baseline)
C. Node2Vec + Gene features (HYBRID)
"""

import json
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from disease_name_matcher import DiseaseMatcher, load_mesh_mappings

REFERENCE_DIR = Path(__file__).parent.parent / "data" / "reference"
MODELS_DIR = Path(__file__).parent.parent / "models"
EMBEDDINGS_DIR = Path(__file__).parent.parent / "data" / "embeddings"
ANALYSIS_DIR = Path(__file__).parent.parent / "data" / "analysis"

start_time = time.time()

print("=" * 70)
print("h35: NODE2VEC + GENE-DISEASE FEATURE HYBRID")
print("=" * 70)
print("Goal: Improve beyond 28.73% R@30 by adding inductive gene features")
print()

# --- Load Node2Vec embeddings ---
print("1. Loading Node2Vec embeddings...")
df_emb = pd.read_csv(EMBEDDINGS_DIR / "node2vec_256_named.csv")
dim_cols = [c for c in df_emb.columns if c.startswith("dim_")]
n2v_emb: Dict[str, np.ndarray] = {}
for _, row in df_emb.iterrows():
    entity = f"drkg:{row['entity']}"
    n2v_emb[entity] = row[dim_cols].values.astype(np.float32)
print(f"  {len(n2v_emb)} entities, {len(dim_cols)} dims")

# --- Load gene data ---
print("\n2. Loading gene data...")
with open(REFERENCE_DIR / "drug_targets.json") as f:
    drug_targets_raw = json.load(f)
with open(REFERENCE_DIR / "disease_genes.json") as f:
    disease_genes_raw = json.load(f)

drug_targets: Dict[str, Set[str]] = {
    f"drkg:Compound::{k}": set(v) for k, v in drug_targets_raw.items()
}
disease_genes: Dict[str, Set[str]] = {
    f"drkg:Disease::{k}": set(v) for k, v in disease_genes_raw.items()
}

print(f"  Drug targets: {len(drug_targets)} drugs")
print(f"  Disease genes: {len(disease_genes)} diseases")

# --- Load ground truth ---
print("\n3. Loading ground truth...")
with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
    id_to_name = json.load(f)
name_to_drug_id = {name.lower(): f"drkg:Compound::{db_id}" for db_id, name in id_to_name.items()}

mesh_path = REFERENCE_DIR / "mesh_mappings_from_agents.json"
mesh_mappings: Dict[str, str] = {}
if mesh_path.exists():
    with open(mesh_path) as f:
        mesh_data = json.load(f)
    for batch_data in mesh_data.values():
        if isinstance(batch_data, dict):
            for disease_name, mesh_id in batch_data.items():
                if mesh_id:
                    mesh_str = str(mesh_id)
                    if mesh_str.startswith("D") or mesh_str.startswith("C"):
                        mesh_mappings[disease_name.lower()] = f"drkg:Disease::MESH:{mesh_str}"

df_gt = pd.read_excel(REFERENCE_DIR / "everycure" / "indicationList.xlsx")
fuzzy_mappings = load_mesh_mappings()
matcher = DiseaseMatcher(fuzzy_mappings)

gt_pairs: Dict[str, Set[str]] = defaultdict(set)
for _, row in df_gt.iterrows():
    disease = str(row.get("disease name", "")).strip()
    drug = str(row.get("final normalized drug label", "")).strip()
    if not disease or not drug:
        continue
    disease_id = matcher.get_mesh_id(disease)
    if not disease_id:
        disease_id = mesh_mappings.get(disease.lower())
    if not disease_id:
        continue
    drug_id = name_to_drug_id.get(drug.lower())
    if drug_id:
        gt_pairs[disease_id].add(drug_id)
gt_pairs_dict = dict(gt_pairs)
print(f"  GT: {len(gt_pairs_dict)} diseases, {sum(len(v) for v in gt_pairs_dict.values())} pairs")

# --- Disease-level split (same as h29, seed=42) ---
print("\n4. Disease-level split (seed=42)...")
valid_diseases = [d for d in gt_pairs_dict if d in n2v_emb]
rng = np.random.RandomState(42)
rng.shuffle(valid_diseases)
n_test = max(1, int(len(valid_diseases) * 0.2))
test_diseases = set(valid_diseases[:n_test])
train_diseases = set(valid_diseases[n_test:])

train_gt = {d: gt_pairs_dict[d] for d in train_diseases}
test_gt = {d: gt_pairs_dict[d] for d in test_diseases}
print(f"  Train: {len(train_gt)} diseases, Test: {len(test_gt)} diseases")

# --- Gene feature computation ---
N_GENE_FEATURES = 12


def compute_gene_features(drug_id: str, disease_id: str) -> np.ndarray:
    """Compute gene-based features for a drug-disease pair."""
    drug_genes = drug_targets.get(drug_id, set())
    dis_genes = disease_genes.get(disease_id, set())

    n_drug = len(drug_genes)
    n_disease = len(dis_genes)

    shared = drug_genes & dis_genes
    n_shared = len(shared)

    union = drug_genes | dis_genes
    jaccard = n_shared / len(union) if union else 0.0
    dice = 2 * n_shared / (n_drug + n_disease) if (n_drug + n_disease) > 0 else 0.0
    overlap_coeff = n_shared / min(n_drug, n_disease) if min(n_drug, n_disease) > 0 else 0.0

    return np.array([
        n_shared, jaccard, dice, overlap_coeff,
        n_drug, n_disease,
        1.0 if n_drug > 0 else 0.0,
        1.0 if n_disease > 0 else 0.0,
        1.0 if (n_drug > 0 and n_disease > 0) else 0.0,
        np.log1p(n_drug),
        np.log1p(n_disease),
        np.log1p(n_shared),
    ], dtype=np.float32)


# --- Build training data ---
print("\n5. Building training data...")

all_drugs = [e for e in n2v_emb if "Compound::" in e]
drug_embs_array = np.array([n2v_emb[d] for d in all_drugs], dtype=np.float32)


def build_data(use_n2v: bool, use_genes: bool, seed: int = 42):
    """Build training data."""
    rng_b = np.random.RandomState(seed)
    X_list, y_list = [], []

    for disease_id, drug_ids in train_gt.items():
        if use_n2v and disease_id not in n2v_emb:
            continue
        valid_drugs = [d for d in drug_ids if (not use_n2v or d in n2v_emb)]
        if not valid_drugs:
            continue

        for drug_id in valid_drugs:
            parts = []
            if use_n2v:
                parts.append(np.concatenate([n2v_emb[drug_id], n2v_emb[disease_id]]))
            if use_genes:
                parts.append(compute_gene_features(drug_id, disease_id))
            X_list.append(np.concatenate(parts))
            y_list.append(1)

        drugs_set = set(drug_ids)
        neg_pool = [d for d in all_drugs if d not in drugs_set]
        n_neg = min(len(valid_drugs) * 5, len(neg_pool))
        neg_samples = rng_b.choice(neg_pool, n_neg, replace=False)
        for neg_drug in neg_samples:
            parts = []
            if use_n2v:
                parts.append(np.concatenate([n2v_emb[neg_drug], n2v_emb[disease_id]]))
            if use_genes:
                parts.append(compute_gene_features(neg_drug, disease_id))
            X_list.append(np.concatenate(parts))
            y_list.append(0)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)


configs = {
    "A_n2v_only": (True, False, "Node2Vec only (baseline)"),
    "B_gene_only": (False, True, "Gene features only"),
    "C_n2v_gene_hybrid": (True, True, "Node2Vec + Gene features (HYBRID)"),
}

training_data = {}
for name, (use_n2v, use_genes, desc) in configs.items():
    print(f"  {name}: {desc}...")
    X, y = build_data(use_n2v, use_genes)
    training_data[name] = (X, y)
    print(f"    Shape: {X.shape}, Pos: {(y==1).sum()}, Neg: {(y==0).sum()}")


# --- Train and evaluate ---
print("\n6. Training and evaluating...")


def evaluate_model(model, use_n2v: bool, use_genes: bool, k: int = 30):
    """Evaluate R@K on held-out test diseases."""
    total_hits = 0
    total_gt_drugs = 0
    per_disease = []

    for disease_id in tqdm(list(test_gt.keys()), desc="Eval"):
        if use_n2v and disease_id not in n2v_emb:
            continue
        gt_set = {d for d in test_gt[disease_id] if (not use_n2v or d in n2v_emb)}
        if not gt_set:
            continue

        batch_parts = []
        if use_n2v:
            disease_emb = n2v_emb[disease_id]
            disease_tiled = np.tile(disease_emb, (len(all_drugs), 1))
            batch_parts.append(np.hstack([drug_embs_array, disease_tiled]))
        if use_genes:
            gene_feats = np.array(
                [compute_gene_features(d, disease_id) for d in all_drugs],
                dtype=np.float32,
            )
            batch_parts.append(gene_feats)

        X_batch = np.hstack(batch_parts).astype(np.float32)
        scores = model.predict_proba(X_batch)[:, 1]

        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_set = {all_drugs[i] for i in top_k_idx}

        hits = len(top_k_set & gt_set)
        total_hits += hits
        total_gt_drugs += len(gt_set)

        per_disease.append({
            "disease_id": disease_id,
            "gt_drugs": len(gt_set),
            "hits": hits,
            "recall": hits / len(gt_set),
        })

    recall = total_hits / total_gt_drugs if total_gt_drugs > 0 else 0
    return recall, per_disease


results = {}
for name, (use_n2v, use_genes, desc) in configs.items():
    print(f"\n  --- {name}: {desc} ---")
    X, y = training_data[name]

    model = XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=42, n_jobs=-1, eval_metric="auc", verbosity=0,
    )
    model.fit(X, y)

    recall, per_disease = evaluate_model(model, use_n2v, use_genes)
    print(f"  R@30: {recall*100:.2f}%")

    importance = None
    if name == "C_n2v_gene_hybrid":
        imp = model.feature_importances_
        n2v_dim = 512  # concat
        n2v_imp = float(imp[:n2v_dim].sum())
        gene_imp = float(imp[n2v_dim:].sum())
        total_imp = float(imp.sum())
        print(f"  Feature importance: Node2Vec {n2v_imp/total_imp*100:.1f}%, Genes {gene_imp/total_imp*100:.1f}%")

        gene_names = [
            "n_shared", "jaccard", "dice", "overlap_coeff",
            "n_drug_targets", "n_disease_genes",
            "has_drug_targets", "has_disease_genes", "has_both",
            "log_drug_targets", "log_disease_genes", "log_shared",
        ]
        gene_importances = imp[n2v_dim:]
        for gn, gi in sorted(zip(gene_names, gene_importances), key=lambda x: -x[1]):
            print(f"    {gn:<20}: {gi:.4f}")

        importance = {
            "n2v_pct": n2v_imp / total_imp * 100,
            "gene_pct": gene_imp / total_imp * 100,
            "gene_breakdown": {gn: float(gi) for gn, gi in zip(gene_names, gene_importances)},
        }

    results[name] = {
        "description": desc,
        "recall_at_30": float(recall),
        "diseases_evaluated": len(per_disease),
        "total_hits": sum(d["hits"] for d in per_disease),
        "total_gt_drugs": sum(d["gt_drugs"] for d in per_disease),
        "importance": importance,
        "per_disease": per_disease,
    }

# --- Summary ---
elapsed = time.time() - start_time
print("\n" + "=" * 70)
print("SUMMARY: h35 NODE2VEC + GENE-DISEASE FEATURE HYBRID")
print("=" * 70)

print(f"\n{'Config':<45} {'R@30':>8} {'Hits':>12}")
print("-" * 70)
for name in ["A_n2v_only", "B_gene_only", "C_n2v_gene_hybrid"]:
    r = results[name]
    print(f"  {r['description']:<43} {r['recall_at_30']*100:>7.2f}% {r['total_hits']}/{r['total_gt_drugs']}")

hybrid_r = results["C_n2v_gene_hybrid"]["recall_at_30"]
baseline_r = results["A_n2v_only"]["recall_at_30"]
delta = hybrid_r - baseline_r
gene_only_r = results["B_gene_only"]["recall_at_30"]

print(f"\n  Delta (hybrid vs n2v-only): {delta*100:+.2f}%")
print(f"  Gene-only baseline: {gene_only_r*100:.2f}%")

if delta > 0.04:
    print(f"\n  SUCCESS: Hybrid improves by {delta*100:.1f} pp!")
elif delta > 0:
    print(f"\n  PARTIAL: Small improvement ({delta*100:.2f} pp)")
else:
    print(f"\n  NEGATIVE: Gene features did not improve generalization")

print(f"\n  Elapsed: {elapsed:.0f}s")

# Save results
output = {
    "hypothesis": "h35",
    "title": "Node2Vec + Gene-Disease Feature Hybrid",
    "date": "2026-01-27",
    "results": {},
    "elapsed_seconds": elapsed,
    "gene_data_stats": {
        "drugs_with_targets": len(drug_targets),
        "diseases_with_genes": len(disease_genes),
    },
}

for name, r in results.items():
    output["results"][name] = {k: v for k, v in r.items() if k != "per_disease"}
    output["results"][name]["diseases_evaluated"] = len(r.get("per_disease", []))

output_path = ANALYSIS_DIR / "h35_gene_hybrid_results.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    pass  # Already ran
