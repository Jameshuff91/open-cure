#!/usr/bin/env python3
"""
h287: ATC Coherence as Positive Confidence Tier Signal

PURPOSE:
    h193 showed coherent predictions have 9.0% precision vs 5.0% incoherent (4 pp lift).
    h191 showed 11.1% vs 6.4% (4.7 pp lift).

    Rather than filtering, use ATC coherence as a positive tiering signal.

    Test: within each confidence tier, do coherent predictions have higher precision?
    If so, we can boost coherent predictions by one tier.

SUCCESS CRITERIA:
    Coherent HIGH tier has >25% precision (would justify boosting)
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
from atc_features import ATCMapper
from disease_categorizer import categorize_disease

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"

SEEDS = [42, 123, 456, 789, 1024]

# Disease category to expected ATC L1 codes mapping (from h191)
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
    'gastrointestinal': {'A'},  # Alimentary
    'hematological': {'B', 'L'},  # Blood, Antineoplastic
    'renal': {'C'},  # Cardiovascular (diuretics)
    'musculoskeletal': {'M'},  # Musculoskeletal
    'other': set(),  # No expectation
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
        disease_categories[disease_id] = categorize_disease(disease)
        drug_id = name_to_drug_id.get(drug.lower())
        if drug_id:
            gt[disease_id].add(drug_id)

    return dict(gt), disease_names, disease_categories


def is_atc_coherent(atc_code: str, disease_category: str) -> bool:
    """Check if drug's ATC L1 code is expected for the disease category."""
    if not atc_code or disease_category == 'other':
        return False  # Can't determine

    expected_atc = DISEASE_CATEGORY_ATC_MAP.get(disease_category, set())
    if not expected_atc:
        return False  # No expectation for this category

    atc_l1 = atc_code[0] if atc_code else ""
    return atc_l1 in expected_atc


def get_simple_tier(rank: int, train_freq: int, mechanism_support: bool) -> str:
    """Simplified tier assignment (approximating production_predictor.py rules)."""
    # FILTER tier
    if rank > 20:
        return "FILTER"
    if train_freq <= 2 and not mechanism_support:
        return "FILTER"

    # GOLDEN tier (simplified - just high freq + mechanism)
    if train_freq >= 15 and mechanism_support:
        return "GOLDEN"

    # HIGH tier
    if train_freq >= 10 and mechanism_support:
        return "HIGH"
    if rank <= 5 and train_freq >= 5:
        return "HIGH"

    # MEDIUM tier
    if train_freq >= 5 and mechanism_support:
        return "MEDIUM"
    if train_freq >= 10:
        return "MEDIUM"

    return "LOW"


def run_analysis(
    emb_dict, train_gt, test_gt, drug_atc, disease_names, disease_categories, id_to_name, k=20
) -> List[Dict]:
    """Run kNN and compute coherence for each prediction."""
    train_disease_list = [d for d in train_gt if d in emb_dict]
    train_disease_embs = np.array([emb_dict[d] for d in train_disease_list], dtype=np.float32)

    # Drug training frequency
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
        disease_cat = disease_categories.get(disease_id, "other")

        # Find k nearest training diseases
        test_emb = emb_dict[disease_id].reshape(1, -1)
        sims = cosine_similarity(test_emb, train_disease_embs)[0]
        top_k_idx = np.argsort(sims)[-k:]

        # Count drug frequency from neighbors
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
            train_freq = drug_train_freq.get(drug_id, 0)

            # Check ATC coherence
            coherent = is_atc_coherent(atc, disease_cat)

            # Approximate tier (without full mechanism analysis)
            mechanism_support = train_freq >= 3  # Simplified
            tier = get_simple_tier(rank, train_freq, mechanism_support)

            is_hit = drug_id in gt_drugs
            drug_name = id_to_name.get(drug_id, drug_id.split("::")[-1])

            results.append({
                'disease': disease_id,
                'drug': drug_id,
                'drug_name': drug_name,
                'disease_name': disease_name,
                'disease_category': disease_cat,
                'atc': atc,
                'atc_l1': atc[0] if atc else "",
                'is_coherent': coherent,
                'train_frequency': train_freq,
                'rank': rank,
                'norm_score': score / max_score if max_score > 0 else 0,
                'tier': tier,
                'is_hit': 1 if is_hit else 0,
            })

    return results


def main():
    print("h287: ATC Coherence as Positive Confidence Tier Signal")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    emb_dict = load_node2vec_embeddings()
    name_to_drug_id, id_to_name = load_drugbank_lookup()
    ground_truth, disease_names, disease_categories = load_ground_truth(name_to_drug_id)

    # Build drug_atc mapping
    atc_mapper = ATCMapper()
    drug_atc = {}
    for drug_id, drug_name in id_to_name.items():
        codes = atc_mapper.get_atc_codes(drug_name)
        if codes:
            drug_atc[drug_id] = codes[0]

    print(f"  Embeddings: {len(emb_dict)}")
    print(f"  Ground truth diseases: {len(ground_truth)}")
    print(f"  Drugs with ATC codes: {len(drug_atc)}")

    # Collect predictions across seeds
    print("\n" + "=" * 70)
    print("Collecting predictions across 5 seeds")
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

        seed_results = run_analysis(
            emb_dict, train_gt, test_gt, drug_atc, disease_names, disease_categories, id_to_name, k=20
        )
        all_results.extend(seed_results)
        print(f"  Seed {seed}: {len(seed_results)} predictions")

    df = pd.DataFrame(all_results)
    print(f"\nTotal predictions: {len(df)}")
    print(f"Base hit rate: {df['is_hit'].mean()*100:.2f}%")

    # Filter to predictions with ATC data and non-other categories
    has_atc = df[(df['atc'] != '') & (df['disease_category'] != 'other')].copy()
    print(f"With ATC and non-other category: {len(has_atc)} predictions")

    # === Overall coherence effect ===
    print(f"\n{'='*70}")
    print("OVERALL COHERENCE EFFECT")
    print("=" * 70)

    coherent = has_atc[has_atc['is_coherent']]
    incoherent = has_atc[~has_atc['is_coherent']]

    coh_prec = coherent['is_hit'].mean() * 100
    incoh_prec = incoherent['is_hit'].mean() * 100

    print(f"\nCoherent: {len(coherent)} predictions, {coh_prec:.2f}% precision")
    print(f"Incoherent: {len(incoherent)} predictions, {incoh_prec:.2f}% precision")
    print(f"Coherence lift: {coh_prec - incoh_prec:+.2f} pp")

    # === Within-tier coherence effect ===
    print(f"\n{'='*70}")
    print("WITHIN-TIER COHERENCE EFFECT (Key test)")
    print("=" * 70)

    tier_order = ['GOLDEN', 'HIGH', 'MEDIUM', 'LOW', 'FILTER']
    tier_results = []

    print(f"\n{'Tier':<10} {'Coherent N':>12} {'Coh Prec':>10} {'Incoh N':>12} {'Incoh Prec':>10} {'Lift':>8}")
    print("-" * 70)

    for tier in tier_order:
        tier_coh = coherent[coherent['tier'] == tier]
        tier_incoh = incoherent[incoherent['tier'] == tier]

        if len(tier_coh) > 0 and len(tier_incoh) > 0:
            coh_p = tier_coh['is_hit'].mean() * 100
            incoh_p = tier_incoh['is_hit'].mean() * 100
            lift = coh_p - incoh_p
            print(f"{tier:<10} {len(tier_coh):>12} {coh_p:>9.2f}% {len(tier_incoh):>12} {incoh_p:>9.2f}% {lift:>+7.2f} pp")
            tier_results.append({
                'tier': tier,
                'coherent_n': len(tier_coh),
                'coherent_prec': coh_p,
                'incoherent_n': len(tier_incoh),
                'incoherent_prec': incoh_p,
                'lift_pp': lift,
            })
        else:
            print(f"{tier:<10} {len(tier_coh):>12} {'N/A':>10} {len(tier_incoh):>12} {'N/A':>10} {'N/A':>8}")

    # === By category ===
    print(f"\n{'='*70}")
    print("COHERENCE EFFECT BY CATEGORY")
    print("=" * 70)

    category_results = []

    print(f"\n{'Category':<15} {'Coh N':>8} {'Coh Prec':>10} {'Incoh N':>8} {'Incoh Prec':>10} {'Lift':>8}")
    print("-" * 70)

    for cat in sorted(has_atc['disease_category'].unique()):
        cat_data = has_atc[has_atc['disease_category'] == cat]
        cat_coh = cat_data[cat_data['is_coherent']]
        cat_incoh = cat_data[~cat_data['is_coherent']]

        if len(cat_coh) > 10 and len(cat_incoh) > 10:
            coh_p = cat_coh['is_hit'].mean() * 100
            incoh_p = cat_incoh['is_hit'].mean() * 100
            lift = coh_p - incoh_p
            print(f"{cat:<15} {len(cat_coh):>8} {coh_p:>9.2f}% {len(cat_incoh):>8} {incoh_p:>9.2f}% {lift:>+7.2f} pp")
            category_results.append({
                'category': cat,
                'coherent_n': len(cat_coh),
                'coherent_prec': coh_p,
                'incoherent_n': len(cat_incoh),
                'incoherent_prec': incoh_p,
                'lift_pp': lift,
            })

    # === Simulated tier boost ===
    print(f"\n{'='*70}")
    print("SIMULATED TIER BOOST: What if coherent predictions are boosted one tier?")
    print("=" * 70)

    # Simulate boosting coherent predictions
    has_atc_sim = has_atc.copy()
    tier_boost_map = {
        'FILTER': 'LOW',
        'LOW': 'MEDIUM',
        'MEDIUM': 'HIGH',
        'HIGH': 'GOLDEN',
        'GOLDEN': 'GOLDEN',  # Already at top
    }

    has_atc_sim['boosted_tier'] = has_atc_sim.apply(
        lambda row: tier_boost_map.get(row['tier'], row['tier']) if row['is_coherent'] else row['tier'],
        axis=1
    )

    print(f"\n{'Tier':<10} {'Original N':>12} {'Original Prec':>14} {'Boosted N':>12} {'Boosted Prec':>14} {'Change':>8}")
    print("-" * 80)

    for tier in tier_order:
        orig_tier = has_atc_sim[has_atc_sim['tier'] == tier]
        boosted_tier = has_atc_sim[has_atc_sim['boosted_tier'] == tier]

        if len(orig_tier) > 0 and len(boosted_tier) > 0:
            orig_p = orig_tier['is_hit'].mean() * 100
            boost_p = boosted_tier['is_hit'].mean() * 100
            change = boost_p - orig_p
            print(f"{tier:<10} {len(orig_tier):>12} {orig_p:>13.2f}% {len(boosted_tier):>12} {boost_p:>13.2f}% {change:>+7.2f} pp")

    # === Summary ===
    print(f"\n{'='*70}")
    print("SUMMARY: h287 Findings")
    print("=" * 70)

    avg_within_tier_lift = np.mean([r['lift_pp'] for r in tier_results if r['lift_pp'] is not None])

    print(f"\nOverall coherence lift: {coh_prec - incoh_prec:+.2f} pp")
    print(f"Average within-tier lift: {avg_within_tier_lift:+.2f} pp")

    # Check if HIGH tier coherent meets success criteria
    high_tier_result = [r for r in tier_results if r['tier'] == 'HIGH']
    if high_tier_result:
        high_coh_prec = high_tier_result[0]['coherent_prec']
        print(f"\nHIGH tier coherent precision: {high_coh_prec:.2f}%")
        if high_coh_prec > 25:
            print("  ✓ SUCCESS: HIGH tier coherent > 25% precision")
        else:
            print("  ✗ FAIL: HIGH tier coherent < 25% precision")

    print("\nCONCLUSION:")
    if avg_within_tier_lift > 3:
        print(f"  ✓ Within-tier coherence lift is significant ({avg_within_tier_lift:.1f} pp)")
        print("  → RECOMMEND: Implement coherence-based tier boost")
    else:
        print(f"  ✗ Within-tier coherence lift is small ({avg_within_tier_lift:.1f} pp)")
        print("  → NOT RECOMMENDED: Marginal gain doesn't justify complexity")

    # Save results
    results = {
        'hypothesis': 'h287',
        'overall_coherent_precision': float(coh_prec),
        'overall_incoherent_precision': float(incoh_prec),
        'overall_lift_pp': float(coh_prec - incoh_prec),
        'n_coherent': int(len(coherent)),
        'n_incoherent': int(len(incoherent)),
        'tier_results': tier_results,
        'category_results': category_results,
        'avg_within_tier_lift': float(avg_within_tier_lift),
    }

    results_file = ANALYSIS_DIR / "h287_atc_coherence_tier_signal.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
