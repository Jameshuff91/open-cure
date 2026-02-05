#!/usr/bin/env python3
"""
h102: Hybrid Zero-Shot Recommendations (Mechanism + ATC Ensemble)

Combines:
1. Mechanism traversal: Disease -> Genes -> Drugs (from h93)
2. ATC class transfer: Similar diseases -> Drug classes -> Drugs (from h98)

The hypothesis is that these capture complementary signals:
- Mechanism: biological targets
- ATC: therapeutic class relationships
"""

import json
import csv
import numpy as np
from collections import defaultdict


def load_atc_mappings(edges_file='data/processed/unified_edges_clean.csv'):
    """Load drug-ATC code mappings from DRKG."""
    drug_atc = defaultdict(set)
    atc_drugs = defaultdict(set)

    with open(edges_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['relation'] == 'HAS_ATC_CODE':
                drug_id = row['source'].replace('drkg:Compound::', '')
                atc_code = row['target'].replace('drkg:Atc::', '')
                drug_atc[drug_id].add(atc_code)
                atc_drugs[atc_code].add(drug_id)

    return drug_atc, atc_drugs


def load_disease_genes(edges_file='data/processed/unified_edges_clean.csv'):
    """Load disease-gene associations."""
    disease_genes = defaultdict(set)

    with open(edges_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['relation'] == 'ASSOCIATED_GENE':
                source = row['source']
                target = row['target']
                if 'Disease::' in source and 'Gene::' in target:
                    mesh_id = source.replace('drkg:Disease::', '').replace('MESH:', '')
                    gene_id = target.replace('drkg:Gene::', '')
                    disease_genes[mesh_id].add(gene_id)

    return disease_genes


def load_gene_drugs(edges_file='data/processed/unified_edges_clean.csv'):
    """Load gene-drug target relationships."""
    gene_drugs = defaultdict(set)

    with open(edges_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['relation'] == 'TARGETS':
                source = row['source']
                target = row['target']
                if 'Compound::' in source and 'Gene::' in target:
                    drug_id = source.replace('drkg:Compound::', '')
                    gene_id = target.replace('drkg:Gene::', '')
                    gene_drugs[gene_id].add(drug_id)

    return gene_drugs


def load_disease_treatments(edges_file='data/processed/unified_edges_clean.csv'):
    """Load known disease-drug treatment relationships."""
    treatments = defaultdict(set)

    with open(edges_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['relation'] == 'TREATS':
                drug = row['source'].replace('drkg:Compound::', '')
                disease = row['target'].replace('drkg:Disease::', '')
                treatments[disease].add(drug)

    return treatments


def load_disease_embeddings():
    """Load Node2Vec disease embeddings from CSV."""
    emb_file = 'data/embeddings/node2vec_256_named.csv'
    embeddings = {}

    with open(emb_file) as f:
        reader = csv.reader(f)
        next(reader)  # Skip header

        for row in reader:
            node_id = row[0]
            if 'Disease::' in node_id:
                if '::' in node_id:
                    parts = node_id.split('::')
                    if len(parts) >= 2:
                        mesh_part = '::'.join(parts[1:])
                        embeddings[mesh_part] = np.array([float(x) for x in row[1:]], dtype=np.float32)

    return embeddings


def get_atc_level(atc_code, level):
    """Get ATC code at specified level."""
    if level == 1:
        return atc_code[0] if len(atc_code) >= 1 else None
    elif level == 2:
        return atc_code[:3] if len(atc_code) >= 3 else None
    elif level == 3:
        return atc_code[:4] if len(atc_code) >= 4 else None
    elif level == 4:
        return atc_code[:5] if len(atc_code) >= 5 else None
    elif level == 5:
        return atc_code if len(atc_code) >= 7 else None
    return None


def find_similar_diseases(target_disease, disease_embeddings, k=10):
    """Find k most similar diseases using cosine similarity."""
    if target_disease not in disease_embeddings:
        return []

    target_emb = disease_embeddings[target_disease]
    target_norm = np.linalg.norm(target_emb)
    if target_norm == 0:
        return []
    target_emb = target_emb / target_norm

    similarities = []
    for disease, emb in disease_embeddings.items():
        if disease == target_disease:
            continue
        if not disease.startswith('MESH:'):
            continue

        emb_norm = np.linalg.norm(emb)
        if emb_norm == 0:
            continue
        emb_normalized = emb / emb_norm
        sim = float(np.dot(target_emb, emb_normalized))
        similarities.append((disease, sim))

    similarities.sort(key=lambda x: -x[1])
    return similarities[:k]


def get_mechanism_scores(mesh_id, disease_genes, gene_drugs):
    """Get mechanism traversal scores for a disease."""
    genes = disease_genes.get(mesh_id, set())

    if not genes:
        return {}

    drug_scores = defaultdict(int)
    for gene in genes:
        for drug in gene_drugs.get(gene, set()):
            drug_scores[drug] += 1

    return dict(drug_scores)


def get_atc_scores(mesh_id, disease_embeddings, disease_treatments,
                   drug_atc, atc_drugs, k_similar=10, atc_level=3):
    """Get ATC-based transfer scores for a disease."""
    disease_id = f"MESH:{mesh_id}"

    similar = find_similar_diseases(disease_id, disease_embeddings, k=k_similar)
    if not similar:
        return {}

    # Collect ATC codes from similar diseases' treatments
    atc_scores = defaultdict(float)
    reference_drugs = set()

    for similar_disease, sim_score in similar:
        drugs = disease_treatments.get(similar_disease, set())
        for drug in drugs:
            reference_drugs.add(drug)
            atc_codes = drug_atc.get(drug, set())
            for atc in atc_codes:
                atc_at_level = get_atc_level(atc, atc_level)
                if atc_at_level:
                    atc_scores[atc_at_level] += sim_score

    if not atc_scores:
        return {}

    # Find drugs with these ATC codes
    drug_scores = defaultdict(float)
    for atc_code, atc_score in atc_scores.items():
        for full_atc, drugs in atc_drugs.items():
            if get_atc_level(full_atc, atc_level) == atc_code:
                for drug in drugs:
                    if drug not in reference_drugs:
                        drug_scores[drug] += atc_score

    return dict(drug_scores)


def normalize_scores(scores: dict) -> dict:
    """Normalize scores to [0, 1] range."""
    if not scores:
        return {}

    values = list(scores.values())
    min_val = min(values)
    max_val = max(values)

    if max_val == min_val:
        return {k: 1.0 for k in scores}

    return {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}


def combine_scores(mechanism_scores, atc_scores, weight_mechanism=0.5):
    """Combine mechanism and ATC scores with given weights."""
    # Normalize both
    norm_mechanism = normalize_scores(mechanism_scores)
    norm_atc = normalize_scores(atc_scores)

    # Get all drugs
    all_drugs = set(norm_mechanism.keys()) | set(norm_atc.keys())

    # Combine
    combined = {}
    for drug in all_drugs:
        mech_score = norm_mechanism.get(drug, 0)
        atc_score = norm_atc.get(drug, 0)
        combined[drug] = weight_mechanism * mech_score + (1 - weight_mechanism) * atc_score

    return combined


def evaluate_hybrid(benchmark_file='data/analysis/zero_shot_benchmark.json',
                    weight_mechanism=0.5, atc_level=3, k_similar=10):
    """Evaluate hybrid approach on zero-shot benchmark."""
    print("Loading resources...")

    # Load all data
    disease_embeddings = load_disease_embeddings()
    print(f"Disease embeddings: {len(disease_embeddings)}")

    disease_genes = load_disease_genes()
    print(f"Diseases with genes: {len(disease_genes)}")

    gene_drugs = load_gene_drugs()
    print(f"Genes with drug targets: {len(gene_drugs)}")

    disease_treatments = load_disease_treatments()
    print(f"Diseases with treatments: {len(disease_treatments)}")

    drug_atc, atc_drugs = load_atc_mappings()
    print(f"Drugs with ATC: {len(drug_atc)}")

    # Load MESH mappings
    with open('data/reference/mesh_mappings_from_agents.json') as f:
        mesh_data = json.load(f)
    all_mesh = {}
    for k, v in mesh_data.items():
        if k != 'metadata' and isinstance(v, dict):
            all_mesh.update(v)

    # Load DrugBank lookup
    with open('data/reference/drugbank_lookup.json') as f:
        drugbank = json.load(f)

    # Load benchmark
    with open(benchmark_file) as f:
        benchmark = json.load(f)

    diseases_in_drkg = benchmark['diseases_in_drkg']
    disease_potential_treatments = {}
    for entry in benchmark['benchmark_diseases']:
        disease_name = entry['disease'].lower()
        disease_potential_treatments[disease_name] = [
            t.lower() for t in entry.get('potential_treatments', [])
        ]

    # Evaluate
    results = []
    hits_hybrid = 0
    hits_mech = 0
    hits_atc = 0
    total = 0

    print(f"\n=== Evaluating Hybrid (weight_mech={weight_mechanism}) ===")

    for disease_name in diseases_in_drkg:
        disease_lower = disease_name.lower()
        mesh_id = all_mesh.get(disease_lower)

        if not mesh_id:
            continue
        if f"MESH:{mesh_id}" not in disease_embeddings:
            continue

        total += 1

        # Get scores from both methods
        mech_scores = get_mechanism_scores(mesh_id, disease_genes, gene_drugs)
        atc_scores = get_atc_scores(mesh_id, disease_embeddings, disease_treatments,
                                    drug_atc, atc_drugs, k_similar=k_similar, atc_level=atc_level)

        # Combine
        hybrid_scores = combine_scores(mech_scores, atc_scores, weight_mechanism)

        # Get potential treatments
        actuals = disease_potential_treatments.get(disease_lower, [])

        # Check hits for each method
        def check_hit(scores_dict, top_n=30):
            if not scores_dict:
                return False, None
            ranked = sorted(scores_dict.items(), key=lambda x: -x[1])[:top_n]
            for i, (drug_id, _) in enumerate(ranked):
                drug_name = drugbank.get(drug_id, drug_id).lower()
                for actual in actuals:
                    if drug_name == actual or drug_name in actual or actual in drug_name:
                        return True, (i + 1, drug_name)
            return False, None

        hit_mech, mech_info = check_hit(mech_scores)
        hit_atc, atc_info = check_hit(atc_scores)
        hit_hybrid, hybrid_info = check_hit(hybrid_scores)

        if hit_mech:
            hits_mech += 1
        if hit_atc:
            hits_atc += 1
        if hit_hybrid:
            hits_hybrid += 1

        # Report
        status = '✓' if hit_hybrid else '✗'
        details = []
        if hit_hybrid:
            details.append(f"hybrid@{hybrid_info[0]}")
        if hit_mech:
            details.append(f"mech@{mech_info[0]}")
        if hit_atc:
            details.append(f"atc@{atc_info[0]}")

        print(f"{status} {disease_name}: {', '.join(details) if details else 'no hits'}")

        results.append({
            'disease': disease_name,
            'hit_hybrid': hit_hybrid,
            'hit_mech': hit_mech,
            'hit_atc': hit_atc
        })

    # Summary
    print(f"\n=== Summary (weight_mechanism={weight_mechanism}, ATC Level {atc_level}) ===")
    print(f"Diseases evaluated: {total}")
    print(f"Mechanism only: {hits_mech}/{total} = {hits_mech/total*100:.1f}%")
    print(f"ATC only: {hits_atc}/{total} = {hits_atc/total*100:.1f}%")
    print(f"Hybrid: {hits_hybrid}/{total} = {hits_hybrid/total*100:.1f}%")

    return {
        'weight_mechanism': weight_mechanism,
        'atc_level': atc_level,
        'total': total,
        'hits_mech': hits_mech,
        'hits_atc': hits_atc,
        'hits_hybrid': hits_hybrid,
        'recall_mech': hits_mech / total * 100 if total > 0 else 0,
        'recall_atc': hits_atc / total * 100 if total > 0 else 0,
        'recall_hybrid': hits_hybrid / total * 100 if total > 0 else 0
    }


if __name__ == '__main__':
    # Test different weights with ATC Level 2 (best from h98)
    all_results = []

    for weight in [0.0, 0.3, 0.5, 0.7, 1.0]:
        print(f"\n{'='*60}")
        result = evaluate_hybrid(weight_mechanism=weight, atc_level=2)
        all_results.append(result)

    # Summary table
    print(f"\n{'='*60}")
    print("=== SUMMARY TABLE ===")
    print(f"{'Weight':<10} {'Mech%':<10} {'ATC%':<10} {'Hybrid%':<10}")
    print("-" * 40)
    for r in all_results:
        print(f"{r['weight_mechanism']:<10.1f} {r['recall_mech']:<10.1f} {r['recall_atc']:<10.1f} {r['recall_hybrid']:<10.1f}")
