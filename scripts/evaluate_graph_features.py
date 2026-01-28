#!/usr/bin/env python3
"""
h34: Node2Vec + Graph Topological Features Hybrid

Tests whether adding graph structural features to Node2Vec embeddings
improves generalization on held-out diseases.

Graph features (per drug-disease pair):
1. drug_degree: Number of neighbors of the drug in DRKG
2. disease_degree: Number of neighbors of the disease in DRKG
3. shared_neighbors: Number of common neighbors between drug and disease
4. shared_gene_neighbors: Shared Gene-type neighbors
5. n_drug_compounds_2hop: Number of compounds 2-hop from drug (not used, expensive)
6. direct_connection: Whether drug and disease are directly connected
7. jaccard_neighbors: Jaccard similarity of neighbor sets
8. adamic_adar: Sum of 1/log(degree(shared_neighbors)) — link prediction feature
9. common_gene_count: Drug-Gene and Disease-Gene shared neighbors count
10. drug_gene_degree: Number of Gene neighbors of drug
11. disease_gene_degree: Number of Gene neighbors of disease
"""

import json
import csv
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from disease_name_matcher import DiseaseMatcher, load_mesh_mappings

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
MODELS_DIR = PROJECT_ROOT / "models"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"

start_time = time.time()

print("=" * 70)
print("h34: NODE2VEC + GRAPH TOPOLOGICAL FEATURES HYBRID")
print("=" * 70)
print()

# --- Load DRKG graph ---
print("1. Loading DRKG graph...")
graph: Dict[str, Set[str]] = defaultdict(set)
with open(PROJECT_ROOT / "data" / "raw" / "drkg" / "drkg.tsv") as f:
    for row in csv.reader(f, delimiter="\t"):
        if len(row) >= 3:
            graph[row[0]].add(row[2])
            graph[row[2]].add(row[0])
print(f"  {len(graph)} nodes, edges loaded in {time.time()-start_time:.1f}s")

# Pre-compute gene neighbors for each node
print("  Pre-computing gene neighbor sets...")
gene_neighbors: Dict[str, Set[str]] = {}
for node, neighbors in graph.items():
    gene_neighbors[node] = {n for n in neighbors if n.startswith("Gene::")}

# --- Load Node2Vec embeddings ---
print("\n2. Loading Node2Vec embeddings...")
df_emb = pd.read_csv(EMBEDDINGS_DIR / "node2vec_256_named.csv")
dim_cols = [c for c in df_emb.columns if c.startswith("dim_")]
n2v_emb: Dict[str, np.ndarray] = {}
for _, row in df_emb.iterrows():
    entity = f"drkg:{row['entity']}"
    n2v_emb[entity] = row[dim_cols].values.astype(np.float32)
print(f"  {len(n2v_emb)} entities, {len(dim_cols)} dims")

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

# --- Disease-level split ---
print("\n4. Disease-level split (seed=42)...")
valid_diseases = [d for d in gt_pairs_dict if d in n2v_emb]
rng = np.random.RandomState(42)
rng.shuffle(valid_diseases)
n_test = max(1, int(len(valid_diseases) * 0.2))
test_diseases = set(valid_diseases[:n_test])
train_diseases = set(valid_diseases[n_test:])
train_gt = {d: gt_pairs_dict[d] for d in train_diseases}
test_gt = {d: gt_pairs_dict[d] for d in test_diseases}
print(f"  Train: {len(train_gt)}, Test: {len(test_gt)}")

# --- Graph feature computation ---
N_GRAPH_FEATURES = 11


def compute_graph_features(drug_id: str, disease_id: str) -> np.ndarray:
    """Compute graph topological features for a drug-disease pair."""
    # Remove drkg: prefix for graph lookup
    drug_graph = drug_id.replace("drkg:", "")
    disease_graph = disease_id.replace("drkg:", "")

    drug_nbrs = graph.get(drug_graph, set())
    disease_nbrs = graph.get(disease_graph, set())

    drug_deg = len(drug_nbrs)
    disease_deg = len(disease_nbrs)

    shared = drug_nbrs & disease_nbrs
    n_shared = len(shared)

    # Shared gene neighbors
    drug_genes = gene_neighbors.get(drug_graph, set())
    disease_genes = gene_neighbors.get(disease_graph, set())
    shared_genes = drug_genes & disease_genes
    n_shared_genes = len(shared_genes)

    # Direct connection
    direct = 1.0 if disease_graph in drug_nbrs else 0.0

    # Jaccard similarity of neighbor sets
    union = drug_nbrs | disease_nbrs
    jaccard = n_shared / len(union) if union else 0.0

    # Adamic-Adar index (link prediction score)
    adamic_adar = 0.0
    for shared_node in shared:
        deg = len(graph.get(shared_node, set()))
        if deg > 1:
            adamic_adar += 1.0 / np.log(deg)

    return np.array([
        drug_deg, disease_deg, n_shared,
        n_shared_genes,
        direct, jaccard, adamic_adar,
        len(drug_genes), len(disease_genes),
        np.log1p(drug_deg), np.log1p(disease_deg),
    ], dtype=np.float32)


# --- Build training data ---
print("\n5. Building training data...")

all_drugs = [e for e in n2v_emb if "Compound::" in e]
drug_embs_array = np.array([n2v_emb[d] for d in all_drugs], dtype=np.float32)


def build_data(use_n2v: bool, use_graph: bool, seed: int = 42):
    """Build training data."""
    rng_b = np.random.RandomState(seed)
    X_list, y_list = [], []

    for disease_id, drug_ids in tqdm(train_gt.items(), desc="Building data"):
        if use_n2v and disease_id not in n2v_emb:
            continue
        valid_drugs = [d for d in drug_ids if (not use_n2v or d in n2v_emb)]
        if not valid_drugs:
            continue

        for drug_id in valid_drugs:
            parts = []
            if use_n2v:
                parts.append(np.concatenate([n2v_emb[drug_id], n2v_emb[disease_id]]))
            if use_graph:
                parts.append(compute_graph_features(drug_id, disease_id))
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
            if use_graph:
                parts.append(compute_graph_features(neg_drug, disease_id))
            X_list.append(np.concatenate(parts))
            y_list.append(0)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)


configs = {
    "A_n2v_only": (True, False, "Node2Vec only (baseline)"),
    "B_graph_only": (False, True, "Graph features only"),
    "C_n2v_graph_hybrid": (True, True, "Node2Vec + Graph features (HYBRID)"),
}

training_data = {}
for name, (use_n2v, use_graph, desc) in configs.items():
    print(f"\n  {name}: {desc}...")
    X, y = build_data(use_n2v, use_graph)
    training_data[name] = (X, y)
    print(f"    Shape: {X.shape}, Pos: {(y==1).sum()}, Neg: {(y==0).sum()}")


# --- Train and evaluate ---
print("\n6. Training and evaluating...")


def evaluate_model(model, use_n2v: bool, use_graph: bool, k: int = 30):
    """Evaluate R@K on held-out test diseases."""
    total_hits = 0
    total_gt = 0
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
        if use_graph:
            graph_feats = np.array(
                [compute_graph_features(d, disease_id) for d in all_drugs],
                dtype=np.float32,
            )
            batch_parts.append(graph_feats)

        X_batch = np.hstack(batch_parts).astype(np.float32)
        scores = model.predict_proba(X_batch)[:, 1]
        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_set = {all_drugs[i] for i in top_k_idx}

        hits = len(top_k_set & gt_set)
        total_hits += hits
        total_gt += len(gt_set)
        per_disease.append({
            "disease_id": disease_id,
            "gt_drugs": len(gt_set),
            "hits": hits,
            "recall": hits / len(gt_set),
        })

    recall = total_hits / total_gt if total_gt > 0 else 0
    return recall, per_disease


results = {}
for name, (use_n2v, use_graph, desc) in configs.items():
    print(f"\n  --- {name}: {desc} ---")
    X, y = training_data[name]

    model = XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=42, n_jobs=-1, eval_metric="auc", verbosity=0,
    )
    model.fit(X, y)

    recall, per_disease = evaluate_model(model, use_n2v, use_graph)
    print(f"  R@30: {recall*100:.2f}%")

    importance = None
    if name == "C_n2v_graph_hybrid":
        imp = model.feature_importances_
        n2v_dim = 512
        n2v_imp = float(imp[:n2v_dim].sum())
        graph_imp = float(imp[n2v_dim:].sum())
        total_imp = float(imp.sum())
        print(f"  Feature importance: Node2Vec {n2v_imp/total_imp*100:.1f}%, Graph {graph_imp/total_imp*100:.1f}%")

        graph_names = [
            "drug_degree", "disease_degree", "shared_neighbors",
            "shared_gene_neighbors", "direct_connection", "jaccard",
            "adamic_adar", "drug_gene_degree", "disease_gene_degree",
            "log_drug_degree", "log_disease_degree",
        ]
        graph_importances = imp[n2v_dim:]
        for gn, gi in sorted(zip(graph_names, graph_importances), key=lambda x: -x[1]):
            print(f"    {gn:<25}: {gi:.4f}")

        importance = {
            "n2v_pct": n2v_imp / total_imp * 100,
            "graph_pct": graph_imp / total_imp * 100,
            "graph_breakdown": {gn: float(gi) for gn, gi in zip(graph_names, graph_importances)},
        }

    results[name] = {
        "description": desc,
        "recall_at_30": float(recall),
        "diseases_evaluated": len(per_disease),
        "total_hits": sum(d["hits"] for d in per_disease),
        "total_gt_drugs": sum(d["gt_drugs"] for d in per_disease),
        "importance": importance,
    }

# --- Summary ---
elapsed = time.time() - start_time
print("\n" + "=" * 70)
print("SUMMARY: h34 NODE2VEC + GRAPH TOPOLOGICAL FEATURES")
print("=" * 70)

print(f"\n{'Config':<45} {'R@30':>8} {'Hits':>12}")
print("-" * 70)
for name in ["A_n2v_only", "B_graph_only", "C_n2v_graph_hybrid"]:
    r = results[name]
    print(f"  {r['description']:<43} {r['recall_at_30']*100:>7.2f}% {r['total_hits']}/{r['total_gt_drugs']}")

hybrid_r = results["C_n2v_graph_hybrid"]["recall_at_30"]
baseline_r = results["A_n2v_only"]["recall_at_30"]
graph_only_r = results["B_graph_only"]["recall_at_30"]
delta = hybrid_r - baseline_r

print(f"\n  Delta (hybrid vs n2v-only): {delta*100:+.2f}%")
print(f"  Graph-only baseline: {graph_only_r*100:.2f}%")

if delta > 0.04:
    print(f"\n  SUCCESS: Hybrid improves by {delta*100:.1f} pp!")
elif delta > 0:
    print(f"\n  PARTIAL: Small improvement ({delta*100:.2f} pp)")
else:
    print(f"\n  NEGATIVE: Graph features did not improve generalization")

print(f"\n  Elapsed: {elapsed:.0f}s")

# Save results
output = {
    "hypothesis": "h34",
    "title": "Node2Vec + Graph Topological Features Hybrid",
    "date": "2026-01-27",
    "results": {name: {k: v for k, v in r.items()} for name, r in results.items()},
    "elapsed_seconds": elapsed,
}

with open(ANALYSIS_DIR / "h34_graph_features_results.json", "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nResults saved to {ANALYSIS_DIR / 'h34_graph_features_results.json'}")
