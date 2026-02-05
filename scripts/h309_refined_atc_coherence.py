#!/usr/bin/env python3
"""
h309: Refine ATC-Category Coherence Map for Corticosteroids

PURPOSE:
    h308 found ATC coherence fails because corticosteroids (ATC A/H) are used
    across many disease categories. Refine the DISEASE_CATEGORY_ATC_MAP to
    include H02 (corticosteroids) for autoimmune, respiratory, dermatological,
    ophthalmological categories.

SUCCESS CRITERIA:
    Coherent precision > incoherent precision after map refinement
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, List

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from disease_name_matcher import DiseaseMatcher, load_mesh_mappings
from atc_features import ATCMapper
from disease_categorizer import categorize_disease

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"

SEEDS = [42, 123, 456, 789, 1024]

# ORIGINAL Disease category to expected ATC L1 codes mapping
ORIGINAL_MAP = {
    'autoimmune': {'L', 'M', 'H'},
    'cancer': {'L'},
    'cardiovascular': {'C', 'B'},
    'dermatological': {'D', 'L'},
    'infectious': {'J', 'P'},
    'metabolic': {'A', 'H'},
    'neurological': {'N'},
    'ophthalmic': {'S'},
    'psychiatric': {'N'},
    'respiratory': {'R'},
    'gastrointestinal': {'A'},
    'hematological': {'B', 'L'},
    'renal': {'C'},
    'musculoskeletal': {'M'},
    'other': set(),
}

# REFINED mapping - add H (systemic hormonal, includes corticosteroids) to inflammatory disease categories
# Also add A (alimentary, where some corticosteroid formulations are classified)
REFINED_MAP = {
    'autoimmune': {'L', 'M', 'H', 'A'},  # Added A (corticosteroid formulations)
    'cancer': {'L'},
    'cardiovascular': {'C', 'B'},
    'dermatological': {'D', 'L', 'H', 'A'},  # Added H, A (corticosteroids)
    'infectious': {'J', 'P'},
    'metabolic': {'A', 'H'},
    'neurological': {'N'},
    'ophthalmic': {'S', 'H', 'A'},  # Added H, A (corticosteroids for inflammation)
    'ophthalmological': {'S', 'H', 'A'},  # Same for alternate spelling
    'psychiatric': {'N'},
    'respiratory': {'R', 'H', 'A'},  # Added H, A (corticosteroids for asthma/inflammation)
    'gastrointestinal': {'A'},
    'hematological': {'B', 'L', 'H', 'A'},  # Added H, A (corticosteroids for blood disorders)
    'renal': {'C'},
    'musculoskeletal': {'M', 'H', 'A'},  # Added H, A (corticosteroids for inflammation)
    'genetic': {'H', 'A'},  # Add for genetic disorders treated with corticosteroids
    'other': set(),
}


def load_node2vec_embeddings() -> Dict[str, np.ndarray]:
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


def load_ground_truth(name_to_drug_id):
    mesh_mappings = load_mesh_mappings()
    matcher = DiseaseMatcher(mesh_mappings)

    df = pd.read_excel(REFERENCE_DIR / "everycure" / "indicationList.xlsx")

    gt: Dict[str, Set[str]] = defaultdict(set)
    disease_names: Dict[str, str] = {}
    disease_categories: Dict[str, str] = {}

    for _, row in df.iterrows():
        disease = str(row.get("disease name", "")).strip()
        drug = str(row.get("final normalized drug label", "")).strip()
        if not disease or not drug:
            continue

        disease_id = matcher.get_mesh_id(disease)
        if not disease_id:
            continue

        disease_names[disease_id] = disease
        cat = categorize_disease(disease)
        if cat:
            disease_categories[disease_id] = cat
        drug_id = name_to_drug_id.get(drug.lower())
        if drug_id:
            gt[disease_id].add(drug_id)

    return dict(gt), disease_names, disease_categories


def is_coherent(atc_code: str, disease_category: str, atc_map: Dict) -> bool:
    if not atc_code or not disease_category:
        return False

    expected_atc = atc_map.get(disease_category, set())
    if not expected_atc:
        return False

    atc_l1 = atc_code[0] if atc_code else ""
    return atc_l1 in expected_atc


def get_simple_tier(rank: int, train_freq: int, mechanism_support: bool) -> str:
    if rank > 20:
        return "FILTER"
    if train_freq <= 2 and not mechanism_support:
        return "FILTER"

    if train_freq >= 15 and mechanism_support:
        return "GOLDEN"

    if train_freq >= 10 and mechanism_support:
        return "HIGH"
    if rank <= 5 and train_freq >= 5:
        return "HIGH"

    if train_freq >= 5 and mechanism_support:
        return "MEDIUM"
    if train_freq >= 10:
        return "MEDIUM"

    return "LOW"


def run_analysis(
    emb_dict, train_gt, test_gt, drug_atc, disease_names, disease_categories, id_to_name, atc_map, k=20
) -> List[Dict]:
    """Run kNN and compute coherence with given ATC map."""
    train_disease_list = [d for d in train_gt if d in emb_dict]
    train_disease_embs = np.array([emb_dict[d] for d in train_disease_list], dtype=np.float32)

    drug_train_freq: Dict[str, int] = defaultdict(int)
    for disease_id, drugs in train_gt.items():
        for drug_id in drugs:
            drug_train_freq[drug_id] += 1

    results = []

    for disease_id in test_gt:
        if disease_id not in emb_dict:
            continue
        gt_drugs = {d for d in test_gt[disease_id] if d in emb_dict}
        if not gt_drugs:
            continue

        disease_name = disease_names.get(disease_id, "")
        disease_cat = disease_categories.get(disease_id, "")
        if not disease_cat:
            continue

        test_emb = emb_dict[disease_id].reshape(1, -1)
        sims = cosine_similarity(test_emb, train_disease_embs)[0]
        top_k_idx = np.argsort(sims)[-k:]

        drug_counts: Dict[str, float] = defaultdict(float)
        for idx in top_k_idx:
            neighbor_disease = train_disease_list[idx]
            neighbor_sim = sims[idx]
            for drug_id in train_gt[neighbor_disease]:
                if drug_id in emb_dict:
                    drug_counts[drug_id] += neighbor_sim

        if not drug_counts:
            continue

        sorted_drugs = sorted(drug_counts.items(), key=lambda x: x[1], reverse=True)[:30]
        max_score = sorted_drugs[0][1] if sorted_drugs else 1.0

        for rank, (drug_id, score) in enumerate(sorted_drugs, 1):
            atc = drug_atc.get(drug_id, "")
            if not atc:
                continue

            train_freq = drug_train_freq.get(drug_id, 0)
            coherent = is_coherent(atc, disease_cat, atc_map)
            mechanism_support = train_freq >= 3
            tier = get_simple_tier(rank, train_freq, mechanism_support)

            is_hit = drug_id in gt_drugs

            results.append({
                'tier': tier,
                'is_coherent': coherent,
                'is_hit': 1 if is_hit else 0,
                'disease_category': disease_cat,
                'atc_l1': atc[0],
            })

    return results


def main():
    print("h309: Refine ATC-Category Coherence Map for Corticosteroids")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    emb_dict = load_node2vec_embeddings()
    name_to_drug_id, id_to_name = load_drugbank_lookup()
    ground_truth, disease_names, disease_categories = load_ground_truth(name_to_drug_id)

    atc_mapper = ATCMapper()
    drug_atc = {}
    for drug_id, drug_name in id_to_name.items():
        codes = atc_mapper.get_atc_codes(drug_name)
        if codes:
            drug_atc[drug_id] = codes[0]

    print(f"  Embeddings: {len(emb_dict)}")
    print(f"  Ground truth diseases: {len(ground_truth)}")

    # Collect predictions with ORIGINAL map
    print("\n" + "=" * 70)
    print("Collecting predictions with ORIGINAL ATC map")
    print("=" * 70)

    orig_results = []
    for seed in SEEDS:
        np.random.seed(seed)
        diseases = list(ground_truth.keys())
        np.random.shuffle(diseases)
        n_test = len(diseases) // 5
        test_diseases = set(diseases[:n_test])
        train_diseases = set(diseases[n_test:])

        train_gt = {d: ground_truth[d] for d in train_diseases}
        test_gt = {d: ground_truth[d] for d in test_diseases}

        seed_results = run_analysis(
            emb_dict, train_gt, test_gt, drug_atc, disease_names, disease_categories, id_to_name, ORIGINAL_MAP, k=20
        )
        orig_results.extend(seed_results)
        print(f"  Seed {seed}: {len(seed_results)} predictions")

    # Collect predictions with REFINED map
    print("\n" + "=" * 70)
    print("Collecting predictions with REFINED ATC map")
    print("=" * 70)

    ref_results = []
    for seed in SEEDS:
        np.random.seed(seed)
        diseases = list(ground_truth.keys())
        np.random.shuffle(diseases)
        n_test = len(diseases) // 5
        test_diseases = set(diseases[:n_test])
        train_diseases = set(diseases[n_test:])

        train_gt = {d: ground_truth[d] for d in train_diseases}
        test_gt = {d: ground_truth[d] for d in test_diseases}

        seed_results = run_analysis(
            emb_dict, train_gt, test_gt, drug_atc, disease_names, disease_categories, id_to_name, REFINED_MAP, k=20
        )
        ref_results.extend(seed_results)
        print(f"  Seed {seed}: {len(seed_results)} predictions")

    # Convert to DataFrames
    df_orig = pd.DataFrame(orig_results)
    df_ref = pd.DataFrame(ref_results)

    print(f"\nTotal predictions: {len(df_orig)}")

    # === Compare ORIGINAL vs REFINED ===
    print(f"\n{'='*70}")
    print("COMPARISON: Original vs Refined ATC Map")
    print("=" * 70)

    tier_order = ['GOLDEN', 'HIGH', 'MEDIUM', 'LOW', 'FILTER']

    print(f"\n{'Tier':<10} {'Orig Coh%':>10} {'Orig Coh Prec':>14} {'Orig Incoh Prec':>16}")
    print("-" * 55)

    orig_tier_results = []
    for tier in tier_order:
        tier_data = df_orig[df_orig['tier'] == tier]
        coh = tier_data[tier_data['is_coherent']]
        incoh = tier_data[~tier_data['is_coherent']]

        if len(coh) > 0 and len(incoh) > 0:
            coh_pct = len(coh) / len(tier_data) * 100
            coh_prec = coh['is_hit'].mean() * 100
            incoh_prec = incoh['is_hit'].mean() * 100
            print(f"{tier:<10} {coh_pct:>9.1f}% {coh_prec:>13.2f}% {incoh_prec:>15.2f}%")
            orig_tier_results.append({
                'tier': tier,
                'coherent_pct': coh_pct,
                'coherent_prec': coh_prec,
                'incoherent_prec': incoh_prec,
            })

    print(f"\n{'Tier':<10} {'Ref Coh%':>10} {'Ref Coh Prec':>14} {'Ref Incoh Prec':>16}")
    print("-" * 55)

    ref_tier_results = []
    for tier in tier_order:
        tier_data = df_ref[df_ref['tier'] == tier]
        coh = tier_data[tier_data['is_coherent']]
        incoh = tier_data[~tier_data['is_coherent']]

        if len(coh) > 0 and len(incoh) > 0:
            coh_pct = len(coh) / len(tier_data) * 100
            coh_prec = coh['is_hit'].mean() * 100
            incoh_prec = incoh['is_hit'].mean() * 100
            print(f"{tier:<10} {coh_pct:>9.1f}% {coh_prec:>13.2f}% {incoh_prec:>15.2f}%")
            ref_tier_results.append({
                'tier': tier,
                'coherent_pct': coh_pct,
                'coherent_prec': coh_prec,
                'incoherent_prec': incoh_prec,
            })

    # === Focus on GOLDEN tier ===
    print(f"\n{'='*70}")
    print("GOLDEN TIER FOCUS: Did refinement help?")
    print("=" * 70)

    golden_orig = df_orig[df_orig['tier'] == 'GOLDEN']
    golden_ref = df_ref[df_ref['tier'] == 'GOLDEN']

    orig_coh = golden_orig[golden_orig['is_coherent']]
    orig_incoh = golden_orig[~golden_orig['is_coherent']]
    ref_coh = golden_ref[golden_ref['is_coherent']]
    ref_incoh = golden_ref[~golden_ref['is_coherent']]

    print(f"\nORIGINAL GOLDEN:")
    print(f"  Coherent: {len(orig_coh)} predictions, {orig_coh['is_hit'].mean()*100:.2f}% precision")
    print(f"  Incoherent: {len(orig_incoh)} predictions, {orig_incoh['is_hit'].mean()*100:.2f}% precision")

    print(f"\nREFINED GOLDEN:")
    print(f"  Coherent: {len(ref_coh)} predictions, {ref_coh['is_hit'].mean()*100:.2f}% precision")
    print(f"  Incoherent: {len(ref_incoh)} predictions, {ref_incoh['is_hit'].mean()*100:.2f}% precision")

    # Calculate if coherent > incoherent after refinement
    orig_gap = orig_coh['is_hit'].mean() - orig_incoh['is_hit'].mean()
    ref_gap = ref_coh['is_hit'].mean() - ref_incoh['is_hit'].mean()

    print(f"\nGap (coherent - incoherent):")
    print(f"  Original: {orig_gap*100:+.2f} pp")
    print(f"  Refined: {ref_gap*100:+.2f} pp")

    # === Summary ===
    print(f"\n{'='*70}")
    print("SUMMARY: h309 Findings")
    print("=" * 70)

    success = ref_coh['is_hit'].mean() > ref_incoh['is_hit'].mean()

    print(f"\nSuccess criterion: Coherent precision > Incoherent precision after refinement")
    print(f"  GOLDEN Coherent precision (refined): {ref_coh['is_hit'].mean()*100:.2f}%")
    print(f"  GOLDEN Incoherent precision (refined): {ref_incoh['is_hit'].mean()*100:.2f}%")

    if success:
        print("  ✓ SUCCESS: Coherent > Incoherent after refinement")
    else:
        print("  ✗ FAIL: Incoherent still beats coherent even after refinement")

    print("\nCONCLUSION:")
    if not success:
        print("  The ATC coherence signal cannot be fixed by expanding the ATC map.")
        print("  The fundamental issue is that high-confidence predictions (GOLDEN tier)")
        print("  are dominated by broad-spectrum drugs that work across many categories.")
        print("  Expanding the map just reclassifies predictions without improving signal.")

    # Save results
    results = {
        'hypothesis': 'h309',
        'original_golden_coherent_prec': float(orig_coh['is_hit'].mean() * 100),
        'original_golden_incoherent_prec': float(orig_incoh['is_hit'].mean() * 100),
        'refined_golden_coherent_prec': float(ref_coh['is_hit'].mean() * 100),
        'refined_golden_incoherent_prec': float(ref_incoh['is_hit'].mean() * 100),
        'original_gap_pp': float(orig_gap * 100),
        'refined_gap_pp': float(ref_gap * 100),
        'success': success,
        'original_tier_results': orig_tier_results,
        'refined_tier_results': ref_tier_results,
    }

    results_file = ANALYSIS_DIR / "h309_refined_atc_coherence.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
