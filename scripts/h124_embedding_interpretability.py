#!/usr/bin/env python3
"""
h124: Disease Embedding Interpretability - What Makes Diseases Similar?

PURPOSE:
    Node2Vec embeddings work well but are black boxes. Understanding what makes
    two diseases similar in embedding space could guide feature engineering and
    provide interpretable explanations for predictions.

APPROACH:
    1. For top kNN disease pairs, compute multiple similarity measures
    2. Correlate each measure with Node2Vec cosine similarity
    3. Identify which biological features best explain embedding similarity
    4. Use insights to create interpretable similarity measure

SUCCESS CRITERIA:
    Identify biological basis for >50% of embedding similarity.
"""

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from disease_name_matcher import DiseaseMatcher, load_mesh_mappings

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
GRAPHS_DIR = PROJECT_ROOT / "data" / "graphs"


def load_node2vec_embeddings() -> Dict[str, np.ndarray]:
    """Load Node2Vec embeddings."""
    embeddings_path = EMBEDDINGS_DIR / "node2vec_256_named.csv"
    df = pd.read_csv(embeddings_path)
    dim_cols = [c for c in df.columns if c.startswith("dim_")]
    embeddings: Dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        entity = f"drkg:{row['entity']}"
        embeddings[entity] = row[dim_cols].values.astype(np.float32)
    return embeddings


def load_drugbank_lookup() -> Dict[str, str]:
    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)
    name_to_id = {name.lower(): f"drkg:Compound::{db_id}" for db_id, name in id_to_name.items()}
    return name_to_id


def load_ground_truth(name_to_drug_id: Dict[str, str]) -> Dict[str, Set[str]]:
    """Load ground truth."""
    mesh_mappings = load_mesh_mappings()
    matcher = DiseaseMatcher(mesh_mappings)
    df = pd.read_excel(REFERENCE_DIR / "everycure" / "indicationList.xlsx")

    gt: Dict[str, Set[str]] = defaultdict(set)
    for _, row in df.iterrows():
        disease = str(row.get("disease name", "")).strip()
        drug = str(row.get("final normalized drug label", "")).strip()
        if not disease or not drug:
            continue
        disease_id = matcher.get_mesh_id(disease)
        if not disease_id:
            continue
        drug_id = name_to_drug_id.get(drug.lower())
        if drug_id:
            gt[disease_id].add(drug_id)

    return dict(gt)


def load_graph_neighbors() -> Dict[str, Set[str]]:
    """Load graph to get direct neighbors."""
    graph_path = GRAPHS_DIR / "unified_graph.gpickle"
    if not graph_path.exists():
        return {}

    with open(graph_path, 'rb') as f:
        G = pickle.load(f)

    neighbors: Dict[str, Set[str]] = defaultdict(set)
    for node in G.nodes():
        if 'Disease' in str(node):
            for neighbor in G.neighbors(node):
                neighbors[str(node)].add(str(neighbor))

    return dict(neighbors)


def load_disease_genes() -> Dict[str, Set[str]]:
    """Load disease-gene associations from DRKG."""
    disease_genes_path = REFERENCE_DIR / "disease_genes.json"
    if disease_genes_path.exists():
        with open(disease_genes_path) as f:
            return {k: set(v) for k, v in json.load(f).items()}
    return {}


def compute_drug_overlap(gt: Dict[str, Set[str]], disease1: str, disease2: str) -> float:
    """Compute Jaccard similarity of drugs treating both diseases."""
    drugs1 = gt.get(disease1, set())
    drugs2 = gt.get(disease2, set())

    if not drugs1 or not drugs2:
        return 0.0

    intersection = len(drugs1 & drugs2)
    union = len(drugs1 | drugs2)

    return intersection / union if union > 0 else 0.0


def compute_gene_overlap(disease_genes: Dict[str, Set[str]], disease1: str, disease2: str) -> float:
    """Compute Jaccard similarity of genes associated with both diseases."""
    genes1 = disease_genes.get(disease1, set())
    genes2 = disease_genes.get(disease2, set())

    if not genes1 or not genes2:
        return 0.0

    intersection = len(genes1 & genes2)
    union = len(genes1 | genes2)

    return intersection / union if union > 0 else 0.0


def compute_neighbor_overlap(neighbors: Dict[str, Set[str]], disease1: str, disease2: str) -> float:
    """Compute Jaccard similarity of graph neighbors."""
    n1 = neighbors.get(disease1, set())
    n2 = neighbors.get(disease2, set())

    if not n1 or not n2:
        return 0.0

    intersection = len(n1 & n2)
    union = len(n1 | n2)

    return intersection / union if union > 0 else 0.0


def compute_common_neighbor_count(neighbors: Dict[str, Set[str]], disease1: str, disease2: str) -> int:
    """Compute number of common graph neighbors."""
    n1 = neighbors.get(disease1, set())
    n2 = neighbors.get(disease2, set())
    return len(n1 & n2)


def main():
    print("h124: Disease Embedding Interpretability")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    emb_dict = load_node2vec_embeddings()
    name_to_drug_id = load_drugbank_lookup()
    gt = load_ground_truth(name_to_drug_id)
    neighbors = load_graph_neighbors()
    disease_genes = load_disease_genes()

    print(f"  Embeddings: {len(emb_dict)}")
    print(f"  Ground truth diseases: {len(gt)}")
    print(f"  Diseases with neighbors: {len(neighbors)}")
    print(f"  Diseases with genes: {len(disease_genes)}")

    # Get diseases in both GT and embeddings
    diseases = [d for d in gt if d in emb_dict]
    print(f"  Common diseases: {len(diseases)}")

    # Compute all pairwise similarities
    print("\nComputing pairwise similarities...")
    disease_embs = np.array([emb_dict[d] for d in diseases])
    embedding_sims = cosine_similarity(disease_embs)

    # Sample pairs for analysis
    np.random.seed(42)
    n_pairs = 5000
    results = []

    # Get random pairs and top similar pairs
    pair_indices = []

    # Random pairs
    for _ in range(n_pairs // 2):
        i, j = np.random.choice(len(diseases), 2, replace=False)
        pair_indices.append((i, j))

    # High similarity pairs
    for i in range(len(diseases)):
        top_j = np.argsort(embedding_sims[i])[-21:-1]  # Top 20 excluding self
        for j in top_j[:5]:  # Top 5 per disease
            if (i, j) not in pair_indices and (j, i) not in pair_indices:
                pair_indices.append((i, j))

    print(f"  Analyzing {len(pair_indices)} disease pairs...")

    for i, j in pair_indices:
        d1, d2 = diseases[i], diseases[j]

        emb_sim = float(embedding_sims[i, j])
        drug_sim = compute_drug_overlap(gt, d1, d2)
        gene_sim = compute_gene_overlap(disease_genes, d1, d2)
        neighbor_sim = compute_neighbor_overlap(neighbors, d1, d2)
        common_neighbors = compute_common_neighbor_count(neighbors, d1, d2)

        results.append({
            'disease1': d1,
            'disease2': d2,
            'embedding_similarity': emb_sim,
            'drug_jaccard': drug_sim,
            'gene_jaccard': gene_sim,
            'neighbor_jaccard': neighbor_sim,
            'common_neighbors': common_neighbors,
        })

    df = pd.DataFrame(results)

    # Correlation analysis
    print("\n" + "=" * 70)
    print("Correlation Analysis: What Explains Embedding Similarity?")
    print("=" * 70)

    features = ['drug_jaccard', 'gene_jaccard', 'neighbor_jaccard', 'common_neighbors']

    print(f"\n{'Feature':<20} {'Pearson r':<12} {'Spearman ρ':<12} {'P-value':<15}")
    print("-" * 60)

    correlations = {}
    for feat in features:
        pearson_r, pearson_p = stats.pearsonr(df['embedding_similarity'], df[feat])
        spearman_r, spearman_p = stats.spearmanr(df['embedding_similarity'], df[feat])
        correlations[feat] = {
            'pearson': pearson_r,
            'spearman': spearman_r,
            'p_value': pearson_p
        }
        print(f"{feat:<20} {pearson_r:>10.4f}  {spearman_r:>10.4f}  {pearson_p:>12.2e}")

    # Stratified analysis
    print("\n" + "=" * 70)
    print("Stratified Analysis: High vs Low Embedding Similarity")
    print("=" * 70)

    high_sim = df[df['embedding_similarity'] >= df['embedding_similarity'].quantile(0.9)]
    low_sim = df[df['embedding_similarity'] <= df['embedding_similarity'].quantile(0.1)]

    print(f"\nHigh similarity pairs (top 10%): {len(high_sim)}")
    print(f"Low similarity pairs (bottom 10%): {len(low_sim)}")

    print(f"\n{'Feature':<20} {'High Sim Mean':<15} {'Low Sim Mean':<15} {'Ratio':<10}")
    print("-" * 60)

    for feat in features:
        high_mean = high_sim[feat].mean()
        low_mean = low_sim[feat].mean()
        ratio = high_mean / low_mean if low_mean > 0 else float('inf')
        print(f"{feat:<20} {high_mean:>12.4f}  {low_mean:>12.4f}  {ratio:>8.1f}x")

    # Best predictor analysis
    print("\n" + "=" * 70)
    print("Feature Importance for Predicting Embedding Similarity")
    print("=" * 70)

    # Simple linear regression R² for each feature
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    r2_scores = {}
    for feat in features:
        X = df[[feat]].values
        y = df['embedding_similarity'].values
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        r2_scores[feat] = r2
        print(f"  {feat}: R² = {r2:.4f} ({r2*100:.1f}% variance explained)")

    # Combined model
    print("\n  Combined model (all features):")
    X = df[features].values
    y = df['embedding_similarity'].values
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    combined_r2 = r2_score(y, y_pred)
    print(f"    R² = {combined_r2:.4f} ({combined_r2*100:.1f}% variance explained)")

    # Feature weights
    print("\n  Feature coefficients (standardized):")
    feature_importance = pd.Series(model.coef_, index=features).abs().sort_values(ascending=False)
    for feat, coef in zip(features, model.coef_):
        print(f"    {feat}: {coef:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    best_feature = max(r2_scores, key=r2_scores.get)
    best_r2 = r2_scores[best_feature]

    print(f"\nBest single predictor: {best_feature} (R² = {best_r2:.4f})")
    print(f"Combined model: R² = {combined_r2:.4f}")

    if combined_r2 >= 0.5:
        print(f"\n✓ SUCCESS: {combined_r2*100:.1f}% of embedding similarity explained by biological features")
    else:
        print(f"\n✗ Below 50% threshold: Only {combined_r2*100:.1f}% explained")
        print("  → Node2Vec captures additional graph structure not in these features")

    # Save results
    results_file = PROJECT_ROOT / "data" / "analysis" / "h124_embedding_interpretability.json"
    with open(results_file, 'w') as f:
        json.dump({
            'correlations': correlations,
            'r2_scores': r2_scores,
            'combined_r2': combined_r2,
            'best_feature': best_feature,
            'high_sim_means': {f: float(high_sim[f].mean()) for f in features},
            'low_sim_means': {f: float(low_sim[f].mean()) for f in features},
            'n_pairs': len(df),
        }, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
