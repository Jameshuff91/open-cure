#!/usr/bin/env python3
"""
h97: Mechanism-kNN Hybrid Confidence

PURPOSE:
    Test whether kNN predictions that ALSO have mechanism support (drug targets
    disease genes) have higher precision than pure pattern-based predictions.

APPROACH:
    1. Run kNN to get top-30 drug predictions per disease
    2. For each prediction, check if drug has ANY gene overlap with disease
    3. Compare: precision of "mechanism-supported" vs "pattern-only" predictions
    4. If mechanism support improves precision, integrate into confidence model

SUCCESS CRITERIA:
    Mechanism-supported predictions have 10%+ higher precision than pattern-only.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from disease_name_matcher import DiseaseMatcher, load_mesh_mappings

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"

SEEDS = [42, 123, 456, 789, 1024]


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


def load_drugbank_lookup() -> Tuple[Dict[str, str], Dict[str, str]]:
    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)
    name_to_id = {name.lower(): f"drkg:Compound::{db_id}" for db_id, name in id_to_name.items()}
    id_to_name_drkg = {f"drkg:Compound::{db_id}": name for db_id, name in id_to_name.items()}
    return name_to_id, id_to_name_drkg


def load_mesh_mappings_from_file() -> Dict[str, str]:
    mesh_path = REFERENCE_DIR / "mesh_mappings_from_agents.json"
    if not mesh_path.exists():
        return {}
    with open(mesh_path) as f:
        mesh_data = json.load(f)
    mesh_mappings: Dict[str, str] = {}
    for batch_data in mesh_data.values():
        if isinstance(batch_data, dict):
            for disease_name, mesh_id in batch_data.items():
                if mesh_id:
                    mesh_str = str(mesh_id)
                    if mesh_str.startswith("D") or mesh_str.startswith("C"):
                        mesh_mappings[disease_name.lower()] = f"drkg:Disease::MESH:{mesh_str}"
    return mesh_mappings


def load_ground_truth(
    mesh_mappings: Dict[str, str],
    name_to_drug_id: Dict[str, str],
) -> Dict[str, Set[str]]:
    df = pd.read_excel(REFERENCE_DIR / "everycure" / "indicationList.xlsx")
    fuzzy_mappings = load_mesh_mappings()
    matcher = DiseaseMatcher(fuzzy_mappings)

    gt: Dict[str, Set[str]] = defaultdict(set)

    for _, row in df.iterrows():
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
            gt[disease_id].add(drug_id)

    return dict(gt)


def load_disease_genes() -> Dict[str, Set[str]]:
    """Load disease-gene associations. Returns disease MESH ID -> set of gene IDs."""
    with open(REFERENCE_DIR / "disease_genes.json") as f:
        data = json.load(f)
    # data is MESH:DXXXXXX -> list of gene IDs
    return {k: set(v) for k, v in data.items()}


def load_drug_targets() -> Dict[str, Set[str]]:
    """Load drug-target associations. Returns drug ID -> set of gene IDs."""
    with open(REFERENCE_DIR / "drug_targets.json") as f:
        data = json.load(f)
    return {k: set(v) for k, v in data.items()}


def normalize_drug_id(drug_id: str) -> str:
    """Extract core drug ID from drkg format."""
    if "::" in drug_id:
        return drug_id.split("::")[-1]
    return drug_id


def normalize_disease_id(disease_id: str) -> str:
    """Extract MESH ID from drkg format."""
    if "::" in disease_id:
        return disease_id.split("::")[-1]
    return disease_id


def has_mechanism_support(drug_id: str, disease_id: str,
                          drug_targets: Dict[str, Set[str]],
                          disease_genes: Dict[str, Set[str]]) -> bool:
    """Check if drug targets any genes associated with disease."""
    drug_core = normalize_drug_id(drug_id)
    disease_mesh = normalize_disease_id(disease_id)

    drug_genes = drug_targets.get(drug_core, set())
    disease_gene_set = disease_genes.get(disease_mesh, set())

    return bool(drug_genes & disease_gene_set)


def run_knn_and_analyze(
    emb_dict: Dict[str, np.ndarray],
    train_gt: Dict[str, Set[str]],
    test_gt: Dict[str, Set[str]],
    drug_targets: Dict[str, Set[str]],
    disease_genes: Dict[str, Set[str]],
    k: int = 20,
) -> Dict:
    """Run kNN and analyze mechanism support vs pattern-only precision."""
    train_disease_list = [d for d in train_gt if d in emb_dict]
    train_disease_embs = np.array([emb_dict[d] for d in train_disease_list], dtype=np.float32)

    results = {
        'mechanism_supported': {'hits': 0, 'total': 0},
        'pattern_only': {'hits': 0, 'total': 0},
        'no_mechanism_data': {'hits': 0, 'total': 0},  # Disease or drug lacks gene data
    }

    for disease_id in test_gt:
        if disease_id not in emb_dict:
            continue
        gt_drugs = {d for d in test_gt[disease_id] if d in emb_dict}
        if not gt_drugs:
            continue

        # Find k nearest training diseases
        test_emb = emb_dict[disease_id].reshape(1, -1)
        sims = cosine_similarity(test_emb, train_disease_embs)[0]
        top_k_disease_idx = np.argsort(sims)[-k:]

        # Count drug frequency among nearest diseases
        drug_counts: Dict[str, float] = defaultdict(float)
        for idx in top_k_disease_idx:
            neighbor_disease = train_disease_list[idx]
            neighbor_sim = sims[idx]
            for drug_id in train_gt[neighbor_disease]:
                if drug_id in emb_dict:
                    drug_counts[drug_id] += neighbor_sim

        # Get top 30 predictions
        if not drug_counts:
            continue
        sorted_drugs = sorted(drug_counts.items(), key=lambda x: x[1], reverse=True)
        top_30 = [(d, score) for d, score in sorted_drugs[:30]]

        # Check each prediction: mechanism-supported or pattern-only?
        disease_mesh = normalize_disease_id(disease_id)
        disease_has_genes = disease_mesh in disease_genes

        for drug_id, score in top_30:
            drug_core = normalize_drug_id(drug_id)
            drug_has_targets = drug_core in drug_targets

            is_hit = drug_id in gt_drugs

            if not disease_has_genes or not drug_has_targets:
                results['no_mechanism_data']['total'] += 1
                if is_hit:
                    results['no_mechanism_data']['hits'] += 1
            elif has_mechanism_support(drug_id, disease_id, drug_targets, disease_genes):
                results['mechanism_supported']['total'] += 1
                if is_hit:
                    results['mechanism_supported']['hits'] += 1
            else:
                results['pattern_only']['total'] += 1
                if is_hit:
                    results['pattern_only']['hits'] += 1

    return results


def main():
    print("h97: Mechanism-kNN Hybrid Confidence Analysis")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    emb_dict = load_node2vec_embeddings()
    name_to_drug_id, id_to_name = load_drugbank_lookup()
    mesh_mappings = load_mesh_mappings_from_file()
    ground_truth = load_ground_truth(mesh_mappings, name_to_drug_id)
    drug_targets = load_drug_targets()
    disease_genes = load_disease_genes()

    print(f"  Embeddings: {len(emb_dict)}")
    print(f"  Ground truth diseases: {len(ground_truth)}")
    print(f"  Diseases with genes: {len(disease_genes)}")
    print(f"  Drugs with targets: {len(drug_targets)}")

    # Multi-seed evaluation
    print("\n" + "=" * 70)
    print("Multi-Seed Evaluation (5 seeds)")
    print("=" * 70)

    all_results = []
    for seed in SEEDS:
        # Split GT into train/test by disease
        np.random.seed(seed)
        diseases = list(ground_truth.keys())
        np.random.shuffle(diseases)
        n_test = len(diseases) // 5  # 20% test
        test_diseases = set(diseases[:n_test])
        train_diseases = set(diseases[n_test:])

        train_gt = {d: ground_truth[d] for d in train_diseases}
        test_gt = {d: ground_truth[d] for d in test_diseases}

        results = run_knn_and_analyze(
            emb_dict, train_gt, test_gt,
            drug_targets, disease_genes, k=20
        )
        all_results.append(results)

    # Aggregate results
    agg = {
        'mechanism_supported': {'hits': 0, 'total': 0},
        'pattern_only': {'hits': 0, 'total': 0},
        'no_mechanism_data': {'hits': 0, 'total': 0},
    }
    for r in all_results:
        for key in agg:
            agg[key]['hits'] += r[key]['hits']
            agg[key]['total'] += r[key]['total']

    print("\nAggregate Results (5 seeds):")
    print("-" * 50)
    for key in ['mechanism_supported', 'pattern_only', 'no_mechanism_data']:
        hits = agg[key]['hits']
        total = agg[key]['total']
        precision = hits / total * 100 if total > 0 else 0
        print(f"  {key:25s}: {hits:5d}/{total:5d} = {precision:5.2f}% precision")

    # Compare mechanism-supported vs pattern-only
    mech_prec = agg['mechanism_supported']['hits'] / agg['mechanism_supported']['total'] * 100 \
        if agg['mechanism_supported']['total'] > 0 else 0
    pattern_prec = agg['pattern_only']['hits'] / agg['pattern_only']['total'] * 100 \
        if agg['pattern_only']['total'] > 0 else 0

    print("\n" + "=" * 70)
    print("KEY COMPARISON")
    print("=" * 70)
    print(f"  Mechanism-supported precision: {mech_prec:.2f}%")
    print(f"  Pattern-only precision:        {pattern_prec:.2f}%")
    diff = mech_prec - pattern_prec
    if diff > 0:
        print(f"  Difference: +{diff:.2f} pp (mechanism support HELPS)")
    else:
        print(f"  Difference: {diff:.2f} pp (mechanism support doesn't help)")

    # Statistical breakdown
    print("\n" + "=" * 70)
    print("Breakdown of Top-30 Predictions")
    print("=" * 70)
    total_predictions = sum(agg[k]['total'] for k in agg)
    for key in ['mechanism_supported', 'pattern_only', 'no_mechanism_data']:
        pct = agg[key]['total'] / total_predictions * 100 if total_predictions > 0 else 0
        print(f"  {key:25s}: {agg[key]['total']:5d} predictions ({pct:.1f}%)")

    # Success criteria check
    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 70)
    success = diff >= 10
    if success:
        print(f"  ✓ Mechanism-supported has +{diff:.1f} pp precision (>= 10 pp threshold)")
        print("  → VALIDATED: Use mechanism support as confidence booster")
    else:
        print(f"  ✗ Mechanism-supported has only +{diff:.1f} pp precision (< 10 pp threshold)")
        print("  → INVALIDATED: Mechanism support doesn't meaningfully improve precision")

    # Save results
    results_file = PROJECT_ROOT / "data" / "analysis" / "h97_mechanism_knn_hybrid.json"
    with open(results_file, 'w') as f:
        json.dump({
            'aggregate': agg,
            'mechanism_precision': mech_prec,
            'pattern_precision': pattern_prec,
            'difference_pp': diff,
            'success': success,
        }, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    return {
        'mechanism_precision': mech_prec,
        'pattern_precision': pattern_prec,
        'difference': diff,
        'success': success,
    }


if __name__ == '__main__':
    main()
