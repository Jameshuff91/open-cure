#!/usr/bin/env python3
"""
h114: Drug Frequency Mechanism - Why Do High-Frequency Drugs Generalize Better?

PURPOSE:
    h111 found drug training frequency is the strongest predictor of hits (r=0.187).
    This experiment investigates WHY. Possible explanations:

    (a) POLYPHARMACOLOGY: High-freq drugs have more targets → broader coverage
    (b) DISEASE CENTRALITY: High-freq drugs treat common diseases with many analogs
    (c) LITERATURE BIAS: High-freq drugs are over-studied, more likely to be discovered
    (d) GT SELECTION BIAS: Every Cure GT favors well-characterized drugs

    Understanding the mechanism could inform feature engineering.

SUCCESS CRITERIA:
    Identify 1+ causal factor explaining the frequency-hit correlation.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, List

import numpy as np
import pandas as pd
from scipy import stats
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


def load_drugbank_lookup():
    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)
    name_to_id = {name.lower(): f"drkg:Compound::{db_id}" for db_id, name in id_to_name.items()}
    id_to_name_full = {f"drkg:Compound::{db_id}": name for db_id, name in id_to_name.items()}
    return name_to_id, id_to_name_full


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


def load_ground_truth(mesh_mappings, name_to_drug_id):
    """Load ground truth and disease names."""
    df = pd.read_excel(REFERENCE_DIR / "everycure" / "indicationList.xlsx")
    fuzzy_mappings = load_mesh_mappings()
    matcher = DiseaseMatcher(fuzzy_mappings)

    gt: Dict[str, Set[str]] = defaultdict(set)
    disease_names: Dict[str, str] = {}
    drug_to_diseases: Dict[str, Set[str]] = defaultdict(set)

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

        disease_names[disease_id] = disease
        drug_id = name_to_drug_id.get(drug.lower())
        if drug_id:
            gt[disease_id].add(drug_id)
            drug_to_diseases[drug_id].add(disease_id)

    return dict(gt), disease_names, dict(drug_to_diseases)


def load_drug_targets() -> Dict[str, Set[str]]:
    """Load drug -> target genes mapping."""
    targets_path = REFERENCE_DIR / "drug_targets.json"
    if not targets_path.exists():
        return {}
    with open(targets_path) as f:
        drug_targets = json.load(f)
    return {f"drkg:Compound::{k}": set(v) for k, v in drug_targets.items()}


def compute_drug_frequencies(gt: Dict[str, Set[str]]) -> Dict[str, int]:
    """Count how many diseases each drug treats in the ground truth."""
    drug_freq = defaultdict(int)
    for disease_id, drugs in gt.items():
        for drug_id in drugs:
            drug_freq[drug_id] += 1
    return dict(drug_freq)


def compute_disease_centrality(emb_dict: Dict[str, np.ndarray], diseases: Set[str]) -> Dict[str, float]:
    """Compute centrality as average cosine similarity to other diseases."""
    disease_list = [d for d in diseases if d in emb_dict]
    if len(disease_list) < 2:
        return {}

    disease_embs = np.array([emb_dict[d] for d in disease_list], dtype=np.float32)
    sim_matrix = cosine_similarity(disease_embs)

    centrality = {}
    for i, disease in enumerate(disease_list):
        # Average similarity to other diseases (exclude self)
        mask = np.ones(len(disease_list), dtype=bool)
        mask[i] = False
        centrality[disease] = float(np.mean(sim_matrix[i, mask]))

    return centrality


def main():
    print("h114: Drug Frequency Mechanism Analysis")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    emb_dict = load_node2vec_embeddings()
    name_to_drug_id, id_to_name = load_drugbank_lookup()
    mesh_mappings = load_mesh_mappings_from_file()
    ground_truth, disease_names, drug_to_diseases = load_ground_truth(mesh_mappings, name_to_drug_id)
    drug_targets = load_drug_targets()

    # Compute drug frequencies
    drug_freq = compute_drug_frequencies(ground_truth)

    print(f"  Embeddings: {len(emb_dict)}")
    print(f"  Ground truth diseases: {len(ground_truth)}")
    print(f"  Unique drugs in GT: {len(drug_freq)}")
    print(f"  Drugs with target data: {len(drug_targets)}")

    # Split drugs into high-freq and low-freq groups
    freq_values = list(drug_freq.values())
    median_freq = np.median(freq_values)
    high_freq_threshold = np.percentile(freq_values, 75)
    low_freq_threshold = np.percentile(freq_values, 25)

    print(f"\nFrequency distribution:")
    print(f"  Median: {median_freq}")
    print(f"  25th percentile: {low_freq_threshold}")
    print(f"  75th percentile: {high_freq_threshold}")
    print(f"  Max: {max(freq_values)}")

    high_freq_drugs = {d for d, f in drug_freq.items() if f >= high_freq_threshold}
    low_freq_drugs = {d for d, f in drug_freq.items() if f <= low_freq_threshold}

    print(f"\n  High-freq drugs (≥{high_freq_threshold}): {len(high_freq_drugs)}")
    print(f"  Low-freq drugs (≤{low_freq_threshold}): {len(low_freq_drugs)}")

    # === HYPOTHESIS A: POLYPHARMACOLOGY (more targets) ===
    print("\n" + "=" * 70)
    print("HYPOTHESIS A: POLYPHARMACOLOGY")
    print("Do high-frequency drugs have more targets?")
    print("=" * 70)

    high_freq_target_counts = []
    low_freq_target_counts = []

    for drug_id in high_freq_drugs:
        targets = drug_targets.get(drug_id, set())
        high_freq_target_counts.append(len(targets))

    for drug_id in low_freq_drugs:
        targets = drug_targets.get(drug_id, set())
        low_freq_target_counts.append(len(targets))

    print(f"\nTarget counts:")
    print(f"  HIGH-freq drugs: mean = {np.mean(high_freq_target_counts):.2f}, median = {np.median(high_freq_target_counts)}")
    print(f"  LOW-freq drugs: mean = {np.mean(low_freq_target_counts):.2f}, median = {np.median(low_freq_target_counts)}")

    # Statistical test
    stat, p_value = stats.mannwhitneyu(high_freq_target_counts, low_freq_target_counts, alternative='greater')
    print(f"\n  Mann-Whitney U test (high > low): p = {p_value:.4e}")

    if p_value < 0.05 and np.mean(high_freq_target_counts) > np.mean(low_freq_target_counts):
        print("  → HIGH-freq drugs have significantly MORE targets ✓")
        poly_confirmed = True
    else:
        print("  → No significant difference in target count ✗")
        poly_confirmed = False

    # === HYPOTHESIS B: DISEASE CENTRALITY ===
    print("\n" + "=" * 70)
    print("HYPOTHESIS B: DISEASE CENTRALITY")
    print("Do high-frequency drugs treat more 'central' diseases?")
    print("=" * 70)

    # Compute centrality for all diseases
    all_diseases = set(ground_truth.keys())
    disease_centrality = compute_disease_centrality(emb_dict, all_diseases)

    # Average centrality of diseases treated by high-freq vs low-freq drugs
    high_freq_centralities = []
    low_freq_centralities = []

    for drug_id in high_freq_drugs:
        for disease_id in drug_to_diseases.get(drug_id, set()):
            if disease_id in disease_centrality:
                high_freq_centralities.append(disease_centrality[disease_id])

    for drug_id in low_freq_drugs:
        for disease_id in drug_to_diseases.get(drug_id, set()):
            if disease_id in disease_centrality:
                low_freq_centralities.append(disease_centrality[disease_id])

    print(f"\nDisease centrality (avg cosine similarity to other diseases):")
    print(f"  Diseases treated by HIGH-freq drugs: mean = {np.mean(high_freq_centralities):.4f}")
    print(f"  Diseases treated by LOW-freq drugs: mean = {np.mean(low_freq_centralities):.4f}")

    stat, p_value = stats.mannwhitneyu(high_freq_centralities, low_freq_centralities, alternative='greater')
    print(f"\n  Mann-Whitney U test (high > low): p = {p_value:.4e}")

    if p_value < 0.05 and np.mean(high_freq_centralities) > np.mean(low_freq_centralities):
        print("  → HIGH-freq drugs treat more CENTRAL diseases ✓")
        centrality_confirmed = True
    else:
        print("  → No significant difference in disease centrality ✗")
        centrality_confirmed = False

    # === HYPOTHESIS C: EMBEDDING PROPERTIES ===
    print("\n" + "=" * 70)
    print("HYPOTHESIS C: EMBEDDING PROPERTIES")
    print("Are high-frequency drugs more central in the embedding space?")
    print("=" * 70)

    # Compute drug centrality (avg similarity to other drugs)
    drugs_with_emb = [d for d in drug_freq if d in emb_dict]
    drug_embs = np.array([emb_dict[d] for d in drugs_with_emb], dtype=np.float32)
    drug_sim_matrix = cosine_similarity(drug_embs)

    drug_centrality = {}
    for i, drug in enumerate(drugs_with_emb):
        mask = np.ones(len(drugs_with_emb), dtype=bool)
        mask[i] = False
        drug_centrality[drug] = float(np.mean(drug_sim_matrix[i, mask]))

    high_freq_drug_centralities = [drug_centrality.get(d, 0) for d in high_freq_drugs if d in drug_centrality]
    low_freq_drug_centralities = [drug_centrality.get(d, 0) for d in low_freq_drugs if d in drug_centrality]

    print(f"\nDrug embedding centrality:")
    print(f"  HIGH-freq drugs: mean = {np.mean(high_freq_drug_centralities):.4f}")
    print(f"  LOW-freq drugs: mean = {np.mean(low_freq_drug_centralities):.4f}")

    stat, p_value = stats.mannwhitneyu(high_freq_drug_centralities, low_freq_drug_centralities, alternative='greater')
    print(f"\n  Mann-Whitney U test (high > low): p = {p_value:.4e}")

    if p_value < 0.05 and np.mean(high_freq_drug_centralities) > np.mean(low_freq_drug_centralities):
        print("  → HIGH-freq drugs are more central in embedding space ✓")
        drug_centrality_confirmed = True
    else:
        print("  → No significant difference in drug embedding centrality ✗")
        drug_centrality_confirmed = False

    # === HYPOTHESIS D: CORRELATION ANALYSIS ===
    print("\n" + "=" * 70)
    print("HYPOTHESIS D: DIRECT CORRELATIONS")
    print("What properties correlate with drug frequency?")
    print("=" * 70)

    # For each drug, compute properties
    drug_data = []
    for drug_id, freq in drug_freq.items():
        n_targets = len(drug_targets.get(drug_id, set()))
        n_diseases = len(drug_to_diseases.get(drug_id, set()))
        emb_centrality = drug_centrality.get(drug_id, np.nan)

        # Average centrality of diseases this drug treats
        disease_cents = [disease_centrality.get(d, np.nan) for d in drug_to_diseases.get(drug_id, set())]
        avg_disease_cent = np.nanmean(disease_cents) if disease_cents else np.nan

        drug_data.append({
            'drug_id': drug_id,
            'frequency': freq,
            'n_targets': n_targets,
            'emb_centrality': emb_centrality,
            'avg_disease_centrality': avg_disease_cent,
        })

    df = pd.DataFrame(drug_data)

    print("\nCorrelation with drug frequency:")
    for col in ['n_targets', 'emb_centrality', 'avg_disease_centrality']:
        valid_mask = ~df[col].isna()
        if valid_mask.sum() > 10:
            r, p = stats.spearmanr(df.loc[valid_mask, 'frequency'], df.loc[valid_mask, col])
            sig = '*' if p < 0.01 else ''
            print(f"  {col}: ρ = {r:.4f}{sig} (p = {p:.4e})")

    # === TOP DRUGS ANALYSIS ===
    print("\n" + "=" * 70)
    print("TOP 20 HIGH-FREQUENCY DRUGS")
    print("=" * 70)

    top_drugs = sorted(drug_freq.items(), key=lambda x: x[1], reverse=True)[:20]
    print(f"\n{'Drug':<30} {'Freq':>6} {'Targets':>8} {'Cent':>8}")
    print("-" * 55)
    for drug_id, freq in top_drugs:
        drug_name = id_to_name.get(drug_id, drug_id.split("::")[-1])[:29]
        n_targets = len(drug_targets.get(drug_id, set()))
        cent = drug_centrality.get(drug_id, 0)
        print(f"{drug_name:<30} {freq:>6} {n_targets:>8} {cent:>8.4f}")

    # === SUMMARY ===
    print("\n" + "=" * 70)
    print("SUMMARY: Drug Frequency Mechanism")
    print("=" * 70)

    findings = []
    if poly_confirmed:
        findings.append("POLYPHARMACOLOGY: High-freq drugs have more targets")
    if centrality_confirmed:
        findings.append("DISEASE CENTRALITY: High-freq drugs treat more central diseases")
    if drug_centrality_confirmed:
        findings.append("EMBEDDING CENTRALITY: High-freq drugs are more central in embedding space")

    if findings:
        print("\nCONFIRMED MECHANISMS:")
        for f in findings:
            print(f"  ✓ {f}")
    else:
        print("\nNo mechanisms confirmed at p < 0.05")

    # Save results
    results = {
        'polypharmacology': {
            'confirmed': poly_confirmed,
            'high_freq_mean_targets': float(np.mean(high_freq_target_counts)),
            'low_freq_mean_targets': float(np.mean(low_freq_target_counts)),
        },
        'disease_centrality': {
            'confirmed': centrality_confirmed,
            'high_freq_mean_centrality': float(np.mean(high_freq_centralities)),
            'low_freq_mean_centrality': float(np.mean(low_freq_centralities)),
        },
        'drug_embedding_centrality': {
            'confirmed': drug_centrality_confirmed,
            'high_freq_mean': float(np.mean(high_freq_drug_centralities)),
            'low_freq_mean': float(np.mean(low_freq_drug_centralities)),
        },
        'n_high_freq_drugs': len(high_freq_drugs),
        'n_low_freq_drugs': len(low_freq_drugs),
        'frequency_threshold_high': float(high_freq_threshold),
        'frequency_threshold_low': float(low_freq_threshold),
    }

    results_file = PROJECT_ROOT / "data" / "analysis" / "h114_drug_frequency_mechanism.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
