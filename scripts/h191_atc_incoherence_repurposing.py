#!/usr/bin/env python3
"""
h191: ATC L1 Incoherence as Novel Repurposing Signal

PURPOSE:
    h110 found counter-intuitively that ATC-incoherent predictions (drug class doesn't
    match disease category) had HIGHER precision (11.24%) than coherent (6.69%).

    This suggests incoherence might signal TRUE repurposing - drugs being used for
    completely unexpected indications, which is the core goal of drug repurposing.

APPROACH:
    1. Reproduce h110 incoherence analysis at ATC L1 level
    2. Examine specific examples of incoherent high-precision predictions
    3. Check if these match known repurposing successes (e.g., thalidomide for cancer)
    4. Test if incoherence + high kNN score = true repurposing signal

SUCCESS CRITERIA:
    Identify a pattern where incoherence predicts novel validated discoveries.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from disease_name_matcher import DiseaseMatcher, load_mesh_mappings
from atc_features import ATCMapper, ATC_LEVEL1
from disease_categorizer import categorize_disease

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"

SEEDS = [42, 123, 456, 789, 1024]

# Disease category to expected ATC codes mapping
DISEASE_CATEGORY_ATC_MAP = {
    'autoimmune': {'L', 'M', 'H'},  # Antineoplastic/immunomod, Musculoskeletal, Hormones
    'cancer': {'L'},  # Antineoplastic
    'cardiovascular': {'C', 'B'},  # Cardiovascular, Blood
    'dermatological': {'D', 'L'},  # Dermatological, Immunomodulating
    'infectious': {'J', 'P'},  # Antiinfectives, Antiparasitic
    'metabolic': {'A', 'H'},  # Alimentary, Hormones
    'neurological': {'N'},  # Nervous system
    'ophthalmic': {'S'},  # Sensory organs
    'psychiatric': {'N'},  # Nervous system
    'respiratory': {'R'},  # Respiratory
    'other': set(),  # No expectation
}


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
    """Load DrugBank ID to name mappings."""
    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)
    name_to_id = {name.lower(): f"drkg:Compound::{db_id}" for db_id, name in id_to_name.items()}
    id_to_name_full = {f"drkg:Compound::{db_id}": name for db_id, name in id_to_name.items()}
    return name_to_id, id_to_name_full


def load_ground_truth(
    name_to_drug_id: Dict[str, str],
) -> Tuple[Dict[str, Set[str]], Dict[str, str], Dict[str, str]]:
    """Load ground truth with disease categories."""
    mesh_mappings = load_mesh_mappings()
    matcher = DiseaseMatcher(mesh_mappings)

    df = pd.read_excel(REFERENCE_DIR / "everycure" / "indicationList.xlsx")

    gt: Dict[str, Set[str]] = defaultdict(set)
    disease_categories: Dict[str, str] = {}
    disease_names: Dict[str, str] = {}  # disease_id -> disease_name

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
            disease_names[disease_id] = disease
            # Use keyword-based categorizer
            category = categorize_disease(disease)
            if category:
                disease_categories[disease_id] = category
            else:
                disease_categories[disease_id] = 'other'

    return dict(gt), disease_categories, disease_names


def compute_atc_coherence(
    drug_name: str,
    disease_category: str,
    atc_mapper: ATCMapper,
) -> Tuple[Optional[bool], List[str]]:
    """
    Check if drug's ATC L1 codes are coherent with disease category.

    Returns (is_coherent, atc_codes).
    """
    atc_l1 = atc_mapper.get_atc_level1(drug_name)
    if not atc_l1:
        return None, []  # No ATC data

    expected_atc = DISEASE_CATEGORY_ATC_MAP.get(disease_category, set())
    if not expected_atc:
        return None, atc_l1  # 'other' category, no expectation

    # Check if ANY of drug's ATC codes match expected
    is_coherent = bool(set(atc_l1) & expected_atc)
    return is_coherent, atc_l1


def run_knn_with_incoherence(
    emb_dict: Dict[str, np.ndarray],
    train_gt: Dict[str, Set[str]],
    test_gt: Dict[str, Set[str]],
    disease_categories: Dict[str, str],
    id_to_name: Dict[str, str],
    atc_mapper: ATCMapper,
    k: int = 20,
) -> List[Dict]:
    """Run kNN and compute ATC incoherence for each prediction."""
    train_disease_list = [d for d in train_gt if d in emb_dict]
    train_disease_embs = np.array([emb_dict[d] for d in train_disease_list], dtype=np.float32)

    results = []

    for disease_id in test_gt:
        if disease_id not in emb_dict:
            continue
        gt_drugs = {d for d in test_gt[disease_id] if d in emb_dict}
        if not gt_drugs:
            continue

        disease_cat = disease_categories.get(disease_id, 'other')

        # Find k nearest training diseases
        test_emb = emb_dict[disease_id].reshape(1, -1)
        sims = cosine_similarity(test_emb, train_disease_embs)[0]
        top_k_idx = np.argsort(sims)[-k:]

        # Count drug frequency
        drug_counts: Dict[str, float] = defaultdict(float)
        for idx in top_k_idx:
            neighbor_disease = train_disease_list[idx]
            neighbor_sim = sims[idx]
            for drug_id in train_gt[neighbor_disease]:
                if drug_id in emb_dict:
                    drug_counts[drug_id] += neighbor_sim

        if not drug_counts:
            continue

        # Get top 30
        sorted_drugs = sorted(drug_counts.items(), key=lambda x: x[1], reverse=True)[:30]

        for rank, (drug_id, score) in enumerate(sorted_drugs, 1):
            drug_name = id_to_name.get(drug_id, "")
            is_coherent, atc_codes = compute_atc_coherence(drug_name, disease_cat, atc_mapper)

            is_hit = drug_id in gt_drugs

            results.append({
                'disease': disease_id,
                'disease_category': disease_cat,
                'drug': drug_id,
                'drug_name': drug_name,
                'rank': rank,
                'score': float(score),
                'is_coherent': is_coherent,
                'atc_codes': atc_codes,
                'is_hit': is_hit,
            })

    return results


def analyze_incoherence_patterns(results: List[Dict]) -> Dict:
    """Analyze patterns in incoherent vs coherent predictions."""
    # Separate by coherence
    coherent = [r for r in results if r['is_coherent'] is True]
    incoherent = [r for r in results if r['is_coherent'] is False]
    no_atc = [r for r in results if r['is_coherent'] is None]

    def calc_precision(group):
        if not group:
            return 0.0
        return sum(1 for r in group if r['is_hit']) / len(group)

    analysis = {
        'n_coherent': len(coherent),
        'n_incoherent': len(incoherent),
        'n_no_atc': len(no_atc),
        'coherent_precision': calc_precision(coherent),
        'incoherent_precision': calc_precision(incoherent),
        'no_atc_precision': calc_precision(no_atc),
    }

    # Analyze by disease category
    category_analysis = {}
    for cat in DISEASE_CATEGORY_ATC_MAP:
        cat_coherent = [r for r in coherent if r['disease_category'] == cat]
        cat_incoherent = [r for r in incoherent if r['disease_category'] == cat]
        if cat_coherent or cat_incoherent:
            category_analysis[cat] = {
                'coherent_n': len(cat_coherent),
                'incoherent_n': len(cat_incoherent),
                'coherent_precision': calc_precision(cat_coherent),
                'incoherent_precision': calc_precision(cat_incoherent),
            }
    analysis['by_category'] = category_analysis

    # Analyze by rank (high score = low rank)
    top10_coherent = [r for r in coherent if r['rank'] <= 10]
    top10_incoherent = [r for r in incoherent if r['rank'] <= 10]
    analysis['top10_coherent_precision'] = calc_precision(top10_coherent)
    analysis['top10_incoherent_precision'] = calc_precision(top10_incoherent)

    return analysis


def find_incoherent_hits(results: List[Dict]) -> List[Dict]:
    """Find incoherent predictions that are true hits - potential repurposing signals."""
    incoherent_hits = [
        r for r in results
        if r['is_coherent'] is False and r['is_hit']
    ]
    # Sort by rank (best first) then score
    incoherent_hits.sort(key=lambda x: (x['rank'], -x['score']))
    return incoherent_hits


def analyze_cross_category_patterns(incoherent_hits: List[Dict]) -> Dict[str, Dict]:
    """Analyze which ATC→disease category patterns work."""
    patterns = defaultdict(lambda: {'hits': 0, 'total': 0, 'examples': []})

    for hit in incoherent_hits:
        for atc in hit['atc_codes']:
            atc_name = ATC_LEVEL1.get(atc, 'unknown')
            pattern_key = f"{atc}:{atc_name} -> {hit['disease_category']}"
            patterns[pattern_key]['hits'] += 1
            if len(patterns[pattern_key]['examples']) < 5:
                patterns[pattern_key]['examples'].append({
                    'drug': hit['drug_name'],
                    'disease': hit['disease'],
                    'rank': hit['rank'],
                })

    return dict(patterns)


def main():
    print("h191: ATC L1 Incoherence as Novel Repurposing Signal")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    emb_dict = load_node2vec_embeddings()
    name_to_drug_id, id_to_name = load_drugbank_lookup()
    ground_truth, disease_categories, disease_names = load_ground_truth(name_to_drug_id)
    atc_mapper = ATCMapper()

    print(f"  Embeddings: {len(emb_dict)}")
    print(f"  Ground truth diseases: {len(ground_truth)}")
    print(f"  Disease categories: {len(set(disease_categories.values()))}")

    # Show category distribution
    cat_counts = defaultdict(int)
    for cat in disease_categories.values():
        cat_counts[cat] += 1
    print(f"  Category distribution: {dict(sorted(cat_counts.items(), key=lambda x: -x[1])[:10])}")

    # Multi-seed evaluation
    print("\n" + "=" * 70)
    print("Multi-Seed Evaluation (5 seeds)")
    print("=" * 70)

    all_results = []

    for seed in SEEDS:
        np.random.seed(seed)
        diseases = list(ground_truth.keys())
        np.random.shuffle(diseases)
        n_test = len(diseases) // 5
        test_diseases = set(diseases[:n_test])
        train_diseases = set(diseases[n_test:])

        train_gt = {d: ground_truth[d] for d in train_diseases}
        test_gt = {d: ground_truth[d] for d in test_diseases}

        seed_results = run_knn_with_incoherence(
            emb_dict, train_gt, test_gt, disease_categories, id_to_name, atc_mapper, k=20
        )
        all_results.extend(seed_results)
        print(f"  Seed {seed}: {len(seed_results)} predictions")

    # Analyze overall patterns
    print("\n" + "=" * 70)
    print("Overall Incoherence Analysis")
    print("=" * 70)

    analysis = analyze_incoherence_patterns(all_results)

    print(f"\nPrediction counts:")
    print(f"  COHERENT (ATC matches disease category): {analysis['n_coherent']}")
    print(f"  INCOHERENT (ATC doesn't match): {analysis['n_incoherent']}")
    print(f"  NO ATC data: {analysis['n_no_atc']}")

    print(f"\nPrecision comparison:")
    print(f"  COHERENT precision:   {100*analysis['coherent_precision']:.2f}%")
    print(f"  INCOHERENT precision: {100*analysis['incoherent_precision']:.2f}%")
    print(f"  NO ATC precision:     {100*analysis['no_atc_precision']:.2f}%")

    diff = analysis['incoherent_precision'] - analysis['coherent_precision']
    print(f"\n  Difference (incoherent - coherent): {100*diff:+.2f} pp")

    # Top 10 analysis
    print(f"\nTop-10 predictions only:")
    print(f"  COHERENT top-10 precision:   {100*analysis['top10_coherent_precision']:.2f}%")
    print(f"  INCOHERENT top-10 precision: {100*analysis['top10_incoherent_precision']:.2f}%")

    # By category
    print("\n" + "=" * 70)
    print("Incoherence by Disease Category")
    print("=" * 70)

    print(f"\n{'Category':<15} {'Coh N':<8} {'Coh Prec':<10} {'Incoh N':<8} {'Incoh Prec':<10} {'Delta':<8}")
    print("-" * 65)
    for cat, data in sorted(analysis['by_category'].items()):
        coh_prec = 100*data['coherent_precision']
        incoh_prec = 100*data['incoherent_precision']
        delta = incoh_prec - coh_prec
        print(f"{cat:<15} {data['coherent_n']:<8} {coh_prec:<10.1f}% {data['incoherent_n']:<8} {incoh_prec:<10.1f}% {delta:+.1f}pp")

    # Find incoherent hits (repurposing signals)
    print("\n" + "=" * 70)
    print("Incoherent Hits (Potential Repurposing Signals)")
    print("=" * 70)

    incoherent_hits = find_incoherent_hits(all_results)
    print(f"\nFound {len(incoherent_hits)} incoherent hits")

    # Show top examples
    print("\nTop 20 examples (best rank first):")
    print(f"{'Rank':<6} {'Drug':<25} {'ATC':<10} {'Disease Category':<15}")
    print("-" * 60)
    for hit in incoherent_hits[:20]:
        atc_str = ','.join(hit['atc_codes'])
        print(f"{hit['rank']:<6} {hit['drug_name'][:24]:<25} {atc_str:<10} {hit['disease_category']:<15}")

    # Cross-category patterns
    print("\n" + "=" * 70)
    print("Cross-Category Repurposing Patterns")
    print("=" * 70)

    patterns = analyze_cross_category_patterns(incoherent_hits)

    # Sort by number of hits
    sorted_patterns = sorted(patterns.items(), key=lambda x: x[1]['hits'], reverse=True)

    print(f"\n{'Pattern (ATC -> Disease)':<45} {'Hits':<8}")
    print("-" * 55)
    for pattern, data in sorted_patterns[:15]:
        print(f"{pattern:<45} {data['hits']:<8}")

    # Key patterns to investigate
    print("\n" + "=" * 70)
    print("Key Pattern Investigation")
    print("=" * 70)

    # Find patterns with 5+ hits
    strong_patterns = [(p, d) for p, d in sorted_patterns if d['hits'] >= 5]

    if strong_patterns:
        print(f"\nPatterns with 5+ hits:")
        for pattern, data in strong_patterns:
            print(f"\n  {pattern}: {data['hits']} hits")
            print("    Examples:")
            for ex in data['examples'][:3]:
                print(f"      - {ex['drug']} (rank {ex['rank']})")

    # Summary statistics
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    print(f"\nIncoherence effect: {100*diff:+.2f} pp")
    if diff > 0:
        print("  → CONFIRMS h110: Incoherent predictions have HIGHER precision")
        print("  → Suggests incoherence may signal true repurposing potential")
    else:
        print("  → Does not confirm h110 finding")

    # Check if high-rank incoherent predictions are better
    top5_incoh = [r for r in all_results if r['is_coherent'] is False and r['rank'] <= 5]
    top5_incoh_prec = sum(1 for r in top5_incoh if r['is_hit']) / len(top5_incoh) if top5_incoh else 0

    print(f"\nTop-5 incoherent precision: {100*top5_incoh_prec:.2f}%")
    print(f"  vs overall incoherent: {100*analysis['incoherent_precision']:.2f}%")

    # Save results
    results_file = ANALYSIS_DIR / "h191_atc_incoherence_repurposing.json"
    with open(results_file, 'w') as f:
        json.dump({
            'analysis': analysis,
            'incoherence_effect_pp': float(diff * 100),
            'n_incoherent_hits': len(incoherent_hits),
            'top_patterns': {p: d['hits'] for p, d in sorted_patterns[:20]},
            'top5_incoherent_precision': float(top5_incoh_prec),
        }, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
