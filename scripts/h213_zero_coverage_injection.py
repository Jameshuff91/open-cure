#!/usr/bin/env python3
"""
h213: Zero-Coverage Drug Injection Layer

264 drugs have zero neighbor coverage for specific diseases but ARE in DRKG.
This analysis:
1. Identifies drugs that COULD be predicted (have DRKG embeddings) but AREN'T (zero kNN coverage)
2. Tests whether mechanism/ATC matching can rescue these predictions
3. Evaluates precision of the rescue approach
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def load_data():
    """Load required data for analysis."""
    from production_predictor import DrugRepurposingPredictor
    from atc_features import ATCMapper, DISEASE_ATC_RELEVANCE

    predictor = DrugRepurposingPredictor(project_root)
    atc_mapper = ATCMapper()

    return predictor, atc_mapper, DISEASE_ATC_RELEVANCE


def get_knn_neighbors(
    disease_id: str,
    predictor,
    k: int = 20
) -> List[Tuple[str, float]]:
    """Get k nearest neighbor diseases with similarity scores."""
    if disease_id not in predictor.embeddings:
        return []

    test_emb = predictor.embeddings[disease_id].reshape(1, -1)
    sims = cosine_similarity(test_emb, predictor.train_embeddings)[0]
    top_k_idx = np.argsort(sims)[-k:][::-1]

    neighbors = []
    for idx in top_k_idx:
        neighbor_id = predictor.train_diseases[idx]
        if neighbor_id != disease_id:
            neighbors.append((neighbor_id, float(sims[idx])))

    return neighbors[:k]


def get_knn_drug_coverage(disease_id: str, predictor, k: int = 20) -> Set[str]:
    """Get all drugs from kNN neighbors."""
    neighbors = get_knn_neighbors(disease_id, predictor, k)
    neighbor_ids = [n[0] for n in neighbors]

    all_neighbor_drugs = set()
    for n_id in neighbor_ids:
        all_neighbor_drugs.update(predictor.ground_truth.get(n_id, set()))

    return all_neighbor_drugs


def analyze_zero_coverage_injection(predictor, atc_mapper, disease_atc_relevance) -> Dict:
    """Analyze zero-coverage drug injection potential."""

    ground_truth = predictor.ground_truth
    disease_names = predictor.disease_names

    # Load h209 blocked pairs for reference
    with open(project_root / "data" / "analysis" / "h209_gt_coverage_analysis.json") as f:
        h209_data = json.load(f)

    # Focus on diseases with embeddings
    diseases_to_analyze = [d for d in ground_truth if d in predictor.embeddings]
    print(f"Analyzing {len(diseases_to_analyze)} diseases")

    # For each disease, identify zero-coverage GT drugs
    zero_coverage_pairs = []  # (disease_id, drug_id, has_mechanism, atc_match)

    for i, disease_id in enumerate(diseases_to_analyze):
        if i % 100 == 0:
            print(f"  Processing disease {i+1}/{len(diseases_to_analyze)}")

        disease_name = disease_names.get(disease_id, '')
        gt_drugs = ground_truth.get(disease_id, set())
        knn_drugs = get_knn_drug_coverage(disease_id, predictor)

        # Disease genes for mechanism matching
        disease_genes = predictor.disease_genes.get(disease_id, set())

        # Get disease ATC relevance
        disease_atc_codes = []
        for pattern, codes in disease_atc_relevance.items():
            if pattern in disease_name.lower():
                disease_atc_codes.extend(codes)
        disease_atc_codes = set(disease_atc_codes)

        # Check each GT drug
        for drug_id in gt_drugs:
            if drug_id not in predictor.embeddings:
                continue  # No embedding, can't inject

            if drug_id in knn_drugs:
                continue  # Has kNN coverage, not zero-coverage

            # This is a zero-coverage drug - can we rescue it?
            drug_name = predictor.drug_id_to_name.get(drug_id, '')
            drug_targets = predictor.drug_targets.get(drug_id, set())

            # Check mechanism overlap
            has_mechanism = len(drug_targets & disease_genes) > 0

            # Check ATC match
            drug_atc_l1 = set(atc_mapper.get_atc_level1(drug_name))
            atc_match = len(drug_atc_l1 & disease_atc_codes) > 0

            # Get training frequency
            train_freq = predictor.drug_train_freq.get(drug_id, 0)

            zero_coverage_pairs.append({
                'disease_id': disease_id,
                'disease_name': disease_name,
                'drug_id': drug_id,
                'drug_name': drug_name,
                'has_mechanism': has_mechanism,
                'atc_match': atc_match,
                'train_freq': train_freq,
            })

    print(f"Found {len(zero_coverage_pairs)} zero-coverage GT pairs")

    # Calculate rescue precision by criteria
    criteria_stats = {
        'mechanism_only': {'hits': 0, 'total': 0},
        'atc_only': {'hits': 0, 'total': 0},
        'both': {'hits': 0, 'total': 0},
        'either': {'hits': 0, 'total': 0},
        'neither': {'hits': 0, 'total': 0},
        'freq_10+': {'hits': 0, 'total': 0},
        'freq_5+': {'hits': 0, 'total': 0},
    }

    # All zero-coverage pairs are GT hits by definition
    for pair in zero_coverage_pairs:
        has_mech = pair['has_mechanism']
        atc = pair['atc_match']
        freq = pair['train_freq']

        if has_mech and atc:
            criteria_stats['both']['hits'] += 1
            criteria_stats['both']['total'] += 1
        elif has_mech:
            criteria_stats['mechanism_only']['hits'] += 1
            criteria_stats['mechanism_only']['total'] += 1
        elif atc:
            criteria_stats['atc_only']['hits'] += 1
            criteria_stats['atc_only']['total'] += 1

        if has_mech or atc:
            criteria_stats['either']['hits'] += 1
            criteria_stats['either']['total'] += 1
        else:
            criteria_stats['neither']['hits'] += 1
            criteria_stats['neither']['total'] += 1

        if freq >= 10:
            criteria_stats['freq_10+']['hits'] += 1
            criteria_stats['freq_10+']['total'] += 1

        if freq >= 5:
            criteria_stats['freq_5+']['hits'] += 1
            criteria_stats['freq_5+']['total'] += 1

    # Now simulate injection: for criteria, how many FALSE positives would we inject?
    # For each disease, count drugs that match criteria but are NOT in GT
    injection_simulation = {
        'mechanism_only': {'true_pos': 0, 'false_pos': 0},
        'atc_only': {'true_pos': 0, 'false_pos': 0},
        'both': {'true_pos': 0, 'false_pos': 0},
        'mechanism_and_freq10': {'true_pos': 0, 'false_pos': 0},
    }

    # Sample diseases for simulation (full simulation would be expensive)
    sample_diseases = diseases_to_analyze[:50]

    for disease_id in sample_diseases:
        disease_name = disease_names.get(disease_id, '')
        gt_drugs = ground_truth.get(disease_id, set())
        knn_drugs = get_knn_drug_coverage(disease_id, predictor)

        disease_genes = predictor.disease_genes.get(disease_id, set())

        disease_atc_codes = []
        for pattern, codes in disease_atc_relevance.items():
            if pattern in disease_name.lower():
                disease_atc_codes.extend(codes)
        disease_atc_codes = set(disease_atc_codes)

        # Check all drugs NOT in kNN coverage
        for drug_id, drug_name in predictor.drug_id_to_name.items():
            if drug_id not in predictor.embeddings:
                continue
            if drug_id in knn_drugs:
                continue  # Already covered by kNN

            drug_targets = predictor.drug_targets.get(drug_id, set())
            has_mech = len(drug_targets & disease_genes) > 0

            drug_atc_l1 = set(atc_mapper.get_atc_level1(drug_name))
            atc = len(drug_atc_l1 & disease_atc_codes) > 0

            freq = predictor.drug_train_freq.get(drug_id, 0)

            is_gt = drug_id in gt_drugs

            # mechanism_only
            if has_mech:
                if is_gt:
                    injection_simulation['mechanism_only']['true_pos'] += 1
                else:
                    injection_simulation['mechanism_only']['false_pos'] += 1

            # atc_only
            if atc:
                if is_gt:
                    injection_simulation['atc_only']['true_pos'] += 1
                else:
                    injection_simulation['atc_only']['false_pos'] += 1

            # both
            if has_mech and atc:
                if is_gt:
                    injection_simulation['both']['true_pos'] += 1
                else:
                    injection_simulation['both']['false_pos'] += 1

            # mechanism_and_freq10
            if has_mech and freq >= 10:
                if is_gt:
                    injection_simulation['mechanism_and_freq10']['true_pos'] += 1
                else:
                    injection_simulation['mechanism_and_freq10']['false_pos'] += 1

    # Calculate precision for injection criteria
    injection_precision = {}
    for criteria, stats in injection_simulation.items():
        total = stats['true_pos'] + stats['false_pos']
        precision = stats['true_pos'] / total if total > 0 else 0
        injection_precision[criteria] = {
            'true_pos': stats['true_pos'],
            'false_pos': stats['false_pos'],
            'total': total,
            'precision': round(precision * 100, 2),
        }

    results = {
        'summary': {
            'zero_coverage_pairs': len(zero_coverage_pairs),
            'diseases_analyzed': len(diseases_to_analyze),
            'diseases_sampled_for_injection': len(sample_diseases),
        },
        'zero_coverage_characteristics': {
            'has_mechanism': sum(1 for p in zero_coverage_pairs if p['has_mechanism']),
            'has_atc_match': sum(1 for p in zero_coverage_pairs if p['atc_match']),
            'has_both': sum(1 for p in zero_coverage_pairs if p['has_mechanism'] and p['atc_match']),
            'has_neither': sum(1 for p in zero_coverage_pairs if not p['has_mechanism'] and not p['atc_match']),
            'freq_10+': sum(1 for p in zero_coverage_pairs if p['train_freq'] >= 10),
            'freq_5+': sum(1 for p in zero_coverage_pairs if p['train_freq'] >= 5),
        },
        'injection_precision': injection_precision,
        'sample_zero_coverage_pairs': zero_coverage_pairs[:50],
    }

    return results


def main():
    print("h213: Zero-Coverage Drug Injection Layer")
    print("=" * 60)

    # Load data
    print("\n1. Loading data...")
    predictor, atc_mapper, disease_atc_relevance = load_data()

    # Run analysis
    print("\n2. Analyzing zero-coverage injection potential...")
    results = analyze_zero_coverage_injection(predictor, atc_mapper, disease_atc_relevance)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    s = results['summary']
    print(f"Zero-coverage GT pairs: {s['zero_coverage_pairs']}")
    print(f"Diseases analyzed: {s['diseases_analyzed']}")
    print(f"Diseases sampled for injection simulation: {s['diseases_sampled_for_injection']}")

    print("\n" + "-" * 60)
    print("ZERO-COVERAGE PAIR CHARACTERISTICS")
    print("-" * 60)
    c = results['zero_coverage_characteristics']
    total = s['zero_coverage_pairs']
    print(f"Has mechanism overlap:  {c['has_mechanism']:4d} ({c['has_mechanism']/total*100:5.1f}%)")
    print(f"Has ATC match:          {c['has_atc_match']:4d} ({c['has_atc_match']/total*100:5.1f}%)")
    print(f"Has both:               {c['has_both']:4d} ({c['has_both']/total*100:5.1f}%)")
    print(f"Has neither:            {c['has_neither']:4d} ({c['has_neither']/total*100:5.1f}%)")
    print(f"Train freq >= 10:       {c['freq_10+']:4d} ({c['freq_10+']/total*100:5.1f}%)")
    print(f"Train freq >= 5:        {c['freq_5+']:4d} ({c['freq_5+']/total*100:5.1f}%)")

    print("\n" + "-" * 60)
    print("INJECTION PRECISION SIMULATION (50 diseases)")
    print("-" * 60)
    print(f"{'Criteria':<25} {'TP':>6} {'FP':>6} {'Total':>6} {'Precision':>10}")
    for criteria, stats in results['injection_precision'].items():
        print(f"{criteria:<25} {stats['true_pos']:>6} {stats['false_pos']:>6} {stats['total']:>6} {stats['precision']:>9.1f}%")

    # Save results
    output_path = project_root / "data" / "analysis" / "h213_zero_coverage_injection.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    results = main()
