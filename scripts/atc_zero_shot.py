#!/usr/bin/env python3
"""
h98: ATC-Based Zero-Shot Drug Recommendations

For zero-shot diseases (no known treatments), this approach:
1. Finds similar diseases using Node2Vec embeddings
2. Gets drugs that treat those similar diseases
3. Extracts ATC codes of those drugs
4. Recommends other drugs with same/similar ATC codes

This leverages drug class similarity rather than direct gene targeting.
"""

import json
import csv
import pickle
import numpy as np
from collections import defaultdict
from pathlib import Path


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


def get_atc_level(atc_code, level):
    """Get ATC code at specified level (1-5).

    Level 1: Anatomical main group (A)
    Level 2: Therapeutic subgroup (A01)
    Level 3: Pharmacological subgroup (A01A)
    Level 4: Chemical subgroup (A01AA)
    Level 5: Chemical substance (A01AA01)
    """
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


def load_disease_embeddings():
    """Load Node2Vec disease embeddings from CSV."""
    try:
        emb_file = 'data/embeddings/node2vec_256_named.csv'
        embeddings = {}

        with open(emb_file) as f:
            reader = csv.reader(f)
            next(reader)  # Skip header

            for row in reader:
                node_id = row[0]
                if 'Disease::' in node_id:
                    # Key format: Disease::MESH:D000123 -> MESH:D000123
                    # Remove the prefix before Disease::
                    if '::' in node_id:
                        parts = node_id.split('::')
                        if len(parts) >= 2:
                            # Get the MESH part after Disease::
                            mesh_part = '::'.join(parts[1:])  # Disease::MESH:D123 -> MESH:D123
                    else:
                        mesh_part = node_id

                    emb = np.array([float(x) for x in row[1:]], dtype=np.float32)
                    embeddings[mesh_part] = emb

        print(f"Loaded {len(embeddings)} disease embeddings")
        return embeddings
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return None


def load_disease_treatments():
    """Load known disease-drug treatment relationships."""
    treatments = defaultdict(set)

    with open('data/processed/unified_edges_clean.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['relation'] == 'TREATS':
                drug = row['source'].replace('drkg:Compound::', '')
                disease = row['target'].replace('drkg:Disease::', '')
                treatments[disease].add(drug)

    return treatments


def find_similar_diseases(target_disease, disease_embeddings, k=10):
    """Find k most similar diseases using cosine similarity."""
    # target_disease format: MESH:D123456
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

        emb_norm = emb / np.linalg.norm(emb)
        sim = np.dot(target_emb, emb_norm)
        similarities.append((disease, sim))

    similarities.sort(key=lambda x: -x[1])
    return similarities[:k]


def predict_drugs_atc(mesh_id, disease_embeddings, disease_treatments,
                      drug_atc, atc_drugs, k_similar=10, atc_level=3, top_n=30):
    """
    Predict drugs for a disease using ATC-based transfer.

    1. Find k similar diseases
    2. Get drugs treating those diseases
    3. Extract ATC codes of those drugs
    4. Find other drugs with same ATC codes
    5. Rank by weighted frequency
    """
    disease_id = f"MESH:{mesh_id}"

    # Find similar diseases
    similar = find_similar_diseases(disease_id, disease_embeddings, k=k_similar)

    if not similar:
        return [], {}

    # Collect ATC codes from similar diseases' treatments
    atc_scores = defaultdict(float)
    reference_drugs = set()

    for similar_disease, sim_score in similar:
        drugs = disease_treatments.get(similar_disease, set())
        for drug in drugs:
            reference_drugs.add(drug)
            # Get ATC codes at specified level
            atc_codes = drug_atc.get(drug, set())
            for atc in atc_codes:
                atc_at_level = get_atc_level(atc, atc_level)
                if atc_at_level:
                    atc_scores[atc_at_level] += sim_score

    if not atc_scores:
        return [], {}

    # Find drugs with these ATC codes (excluding reference drugs)
    drug_scores = defaultdict(float)
    drug_atc_evidence = defaultdict(set)

    for atc_code, atc_score in atc_scores.items():
        # Get all drugs with this ATC code at the same level
        for full_atc, drugs in atc_drugs.items():
            if get_atc_level(full_atc, atc_level) == atc_code:
                for drug in drugs:
                    if drug not in reference_drugs:  # Exclude training drugs
                        drug_scores[drug] += atc_score
                        drug_atc_evidence[drug].add(atc_code)

    # Rank drugs
    ranked = sorted(drug_scores.items(), key=lambda x: -x[1])

    results = []
    for drug_id, score in ranked[:top_n]:
        results.append({
            'drug_id': drug_id,
            'score': score,
            'atc_evidence': list(drug_atc_evidence[drug_id])
        })

    debug_info = {
        'similar_diseases': similar[:5],
        'reference_drugs': len(reference_drugs),
        'atc_codes_found': len(atc_scores),
        'candidate_drugs': len(drug_scores)
    }

    return results, debug_info


def evaluate_on_benchmark(benchmark_file='data/analysis/zero_shot_benchmark.json',
                          atc_level=3, k_similar=10):
    """Evaluate ATC-based recommendations on zero-shot benchmark."""
    print("Loading resources...")

    # Load embeddings
    disease_embeddings = load_disease_embeddings()
    if disease_embeddings is None:
        print("Could not load embeddings")
        return None

    # Filter to disease embeddings only
    disease_emb = {k: v for k, v in disease_embeddings.items() if 'MESH:' in k}
    print(f"Disease embeddings: {len(disease_emb)}")

    # Load ATC mappings
    drug_atc, atc_drugs = load_atc_mappings()
    print(f"Drugs with ATC: {len(drug_atc)}")

    # Load treatments
    disease_treatments = load_disease_treatments()
    print(f"Diseases with treatments: {len(disease_treatments)}")

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

    # Get potential treatments lookup
    disease_potential_treatments = {}
    for entry in benchmark['benchmark_diseases']:
        disease_name = entry['disease'].lower()
        disease_potential_treatments[disease_name] = [
            t.lower() for t in entry.get('potential_treatments', [])
        ]

    # Evaluate
    results = []
    hits = 0
    total = 0

    diseases_in_drkg = benchmark['diseases_in_drkg']

    print(f"\n=== Evaluating on {len(diseases_in_drkg)} diseases ===")

    for disease_name in diseases_in_drkg:
        disease_lower = disease_name.lower()
        mesh_id = all_mesh.get(disease_lower)

        if not mesh_id:
            continue

        # Check if disease has embeddings
        if f"MESH:{mesh_id}" not in disease_emb:
            continue

        total += 1

        # Get predictions
        predictions, debug = predict_drugs_atc(
            mesh_id, disease_emb, disease_treatments,
            drug_atc, atc_drugs, k_similar=k_similar, atc_level=atc_level
        )

        # Get potential treatments
        actuals = disease_potential_treatments.get(disease_lower, [])

        # Check for hits
        hit = False
        hit_info = None
        pred_names = []

        for i, pred in enumerate(predictions):
            drug_name = drugbank.get(pred['drug_id'], pred['drug_id']).lower()
            pred_names.append(drug_name)

            for actual in actuals:
                if drug_name == actual or drug_name in actual or actual in drug_name:
                    hit = True
                    hit_info = {'rank': i + 1, 'drug': drug_name, 'matched': actual}
                    break
            if hit:
                break

        if hit:
            hits += 1

        result = {
            'disease': disease_name,
            'mesh_id': mesh_id,
            'hit': hit,
            'hit_info': hit_info,
            'num_predictions': len(predictions),
            'top_5_predictions': pred_names[:5],
            'potential_treatments': actuals,
            'debug': debug
        }
        results.append(result)

        symbol = '✓' if hit else '✗'
        if hit:
            print(f"{symbol} {disease_name}: HIT at rank {hit_info['rank']} ({hit_info['drug']})")
        else:
            sim_count = debug.get('similar_diseases', [])
            ref_count = debug.get('reference_drugs', 0)
            print(f"{symbol} {disease_name}: {len(sim_count)} similar diseases, {ref_count} ref drugs")

    # Summary
    recall = hits / total * 100 if total > 0 else 0
    print(f"\n=== Summary (ATC Level {atc_level}, k={k_similar}) ===")
    print(f"Diseases evaluated: {total}")
    print(f"Hits@30: {hits}")
    print(f"Recall@30: {recall:.1f}%")

    return {
        'atc_level': atc_level,
        'k_similar': k_similar,
        'diseases_evaluated': total,
        'hits': hits,
        'recall_at_30': recall,
        'detailed_results': results
    }


if __name__ == '__main__':
    # Test with different ATC levels
    results = {}
    for level in [2, 3, 4]:
        print(f"\n{'='*60}")
        print(f"Testing ATC Level {level}")
        print('='*60)
        results[level] = evaluate_on_benchmark(atc_level=level)

    # Save results
    with open('data/analysis/atc_zero_shot_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to data/analysis/atc_zero_shot_results.json")
