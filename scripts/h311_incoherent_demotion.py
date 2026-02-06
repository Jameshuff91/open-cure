#!/usr/bin/env python3
"""
h311: ATC Coherence as Negative Signal for Incoherent Predictions

PURPOSE:
    h309 found that incoherent GOLDEN predictions have only 18.7% precision
    (vs 35.5% for coherent). This hypothesis tests whether demoting incoherent
    predictions improves tier calibration.

APPROACH:
    1. For GOLDEN tier predictions, check ATC coherence
    2. If incoherent, demote to HIGH tier
    3. Measure precision by tier to verify improved calibration

SUCCESS CRITERIA:
    - Improved tier separation (coherent tiers > incoherent)
    - Better calibration (tier precision closer to expected)
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.disease_name_matcher import DiseaseMatcher, load_mesh_mappings
from src.disease_categorizer import categorize_disease
from src.atc_features import ATCMapper

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"

SEEDS = [42, 123, 456, 789, 1024]

# From h309/h310 - refined ATC category map
DISEASE_CATEGORY_ATC_MAP: Dict[str, Set[str]] = {
    'autoimmune': {'L', 'M', 'H', 'A'},
    'cancer': {'L'},
    'cardiovascular': {'C', 'B'},
    'dermatological': {'D', 'L', 'H', 'A'},
    'infectious': {'J', 'P'},
    'metabolic': {'A', 'H'},
    'neurological': {'N'},
    'ophthalmic': {'S', 'H', 'A'},
    'ophthalmological': {'S', 'H', 'A'},
    'psychiatric': {'N'},
    'respiratory': {'R', 'H', 'A'},
    'gastrointestinal': {'A'},
    'hematological': {'B', 'L', 'H', 'A'},
    'renal': {'C'},
    'musculoskeletal': {'M', 'H', 'A'},
    'genetic': {'H', 'A'},
    'immunological': {'L', 'H', 'A'},
    'endocrine': {'H', 'A'},
    'reproductive': {'G', 'H'},
    'other': set(),
}


def load_node2vec_embeddings() -> Dict[str, np.ndarray]:
    """Load Node2Vec embeddings."""
    df = pd.read_csv(EMBEDDINGS_DIR / "node2vec_256_named.csv")
    dim_cols = [c for c in df.columns if c.startswith("dim_")]
    embeddings: Dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        entity = f"drkg:{row['entity']}"
        embeddings[entity] = row[dim_cols].values.astype(np.float32)
    return embeddings


def load_drugbank_lookup() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Load DrugBank ID mappings."""
    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)
    name_to_id = {name.lower(): f"drkg:Compound::{db_id}" for db_id, name in id_to_name.items()}
    id_to_name_drkg = {f"drkg:Compound::{db_id}": name for db_id, name in id_to_name.items()}
    return name_to_id, id_to_name_drkg


def load_ground_truth(name_to_drug_id: Dict[str, str]) -> Tuple[Dict[str, Set[str]], Dict[str, str], Dict[str, str]]:
    """Load Every Cure ground truth."""
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


def is_atc_coherent(drug_name: str, category: str, atc_mapper: ATCMapper) -> bool:
    """Check if drug's ATC code matches expected for disease category."""
    expected_atc = DISEASE_CATEGORY_ATC_MAP.get(category, set())
    if not expected_atc:
        return False

    try:
        atc_codes = atc_mapper.get_atc_codes(drug_name)
        if not atc_codes:
            return False

        for code in atc_codes:
            if code and code[0] in expected_atc:
                return True
        return False
    except Exception:
        return False


def get_base_tier(rank: int, train_freq: int, mechanism_support: bool) -> str:
    """Get base confidence tier without coherence adjustment."""
    if rank > 20:
        return "FILTER"
    if train_freq <= 2 and not mechanism_support:
        return "FILTER"

    if train_freq >= 15 and mechanism_support:
        return "GOLDEN"
    if rank <= 5 and train_freq >= 10 and mechanism_support:
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


def adjust_tier_for_coherence(base_tier: str, is_coherent: bool, demote: bool) -> str:
    """Adjust tier based on ATC coherence.

    If demote=True and prediction is incoherent:
    - GOLDEN -> HIGH
    - HIGH -> MEDIUM
    """
    if not demote or is_coherent:
        return base_tier

    if base_tier == "GOLDEN":
        return "HIGH"
    elif base_tier == "HIGH":
        return "MEDIUM"
    return base_tier


def run_evaluation(
    emb_dict: Dict[str, np.ndarray],
    train_gt: Dict[str, Set[str]],
    test_gt: Dict[str, Set[str]],
    disease_categories: Dict[str, str],
    id_to_name: Dict[str, str],
    atc_mapper: ATCMapper,
    demote_incoherent: bool = False,
    k: int = 20
) -> Dict[str, List[Dict]]:
    """Run kNN and collect tier/coherence/hit data."""
    train_disease_list = [d for d in train_gt if d in emb_dict]
    train_disease_embs = np.array([emb_dict[d] for d in train_disease_list], dtype=np.float32)

    # Drug train frequency
    drug_train_freq: Dict[str, int] = defaultdict(int)
    for disease_id, drugs in train_gt.items():
        for drug_id in drugs:
            drug_train_freq[drug_id] += 1

    predictions = []

    for disease_id in test_gt:
        if disease_id not in emb_dict:
            continue
        gt_drugs = {d for d in test_gt[disease_id] if d in emb_dict}
        if not gt_drugs:
            continue

        category = disease_categories.get(disease_id, "other")

        # kNN
        test_emb = emb_dict[disease_id].reshape(1, -1)
        sims = cosine_similarity(test_emb, train_disease_embs)[0]
        top_k_idx = np.argsort(sims)[-k:]

        drug_scores: Dict[str, float] = defaultdict(float)
        for idx in top_k_idx:
            neighbor = train_disease_list[idx]
            sim = sims[idx]
            for drug_id in train_gt[neighbor]:
                if drug_id in emb_dict:
                    drug_scores[drug_id] += sim

        if not drug_scores:
            continue

        # Rank drugs
        sorted_drugs = sorted(drug_scores.items(), key=lambda x: x[1], reverse=True)[:30]

        for rank, (drug_id, score) in enumerate(sorted_drugs, 1):
            drug_name = id_to_name.get(drug_id, "")
            train_freq = drug_train_freq.get(drug_id, 0)
            mechanism_support = train_freq >= 3  # Simple proxy

            # Get base tier
            base_tier = get_base_tier(rank, train_freq, mechanism_support)

            # Check coherence
            is_coherent = is_atc_coherent(drug_name, category, atc_mapper)

            # Adjust tier
            final_tier = adjust_tier_for_coherence(base_tier, is_coherent, demote_incoherent)

            is_hit = drug_id in gt_drugs

            predictions.append({
                'base_tier': base_tier,
                'final_tier': final_tier,
                'is_coherent': is_coherent,
                'is_hit': 1 if is_hit else 0,
                'category': category,
                'rank': rank,
            })

    return predictions


def main():
    print("h311: ATC Coherence as Negative Signal for Incoherent Predictions")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    emb_dict = load_node2vec_embeddings()
    name_to_drug_id, id_to_name = load_drugbank_lookup()
    ground_truth, disease_names, disease_categories = load_ground_truth(name_to_drug_id)
    atc_mapper = ATCMapper()

    print(f"  Embeddings: {len(emb_dict)}")
    print(f"  Ground truth diseases: {len(ground_truth)}")

    # Run evaluations
    print("\n" + "=" * 70)
    print("Running 5-seed evaluation: Baseline vs Incoherent Demotion")
    print("=" * 70)

    all_baseline = []
    all_demoted = []

    for seed in SEEDS:
        np.random.seed(seed)
        diseases = list(ground_truth.keys())
        np.random.shuffle(diseases)
        n_test = len(diseases) // 5
        test_diseases = set(diseases[:n_test])
        train_diseases = set(diseases[n_test:])

        train_gt = {d: ground_truth[d] for d in train_diseases}
        test_gt = {d: ground_truth[d] for d in test_diseases}

        # Baseline (no demotion)
        preds_baseline = run_evaluation(
            emb_dict, train_gt, test_gt, disease_categories, id_to_name, atc_mapper,
            demote_incoherent=False
        )
        all_baseline.extend(preds_baseline)

        # With demotion
        preds_demoted = run_evaluation(
            emb_dict, train_gt, test_gt, disease_categories, id_to_name, atc_mapper,
            demote_incoherent=True
        )
        all_demoted.extend(preds_demoted)

        print(f"  Seed {seed}: {len(preds_baseline)} predictions")

    # Analyze results
    df_baseline = pd.DataFrame(all_baseline)
    df_demoted = pd.DataFrame(all_demoted)

    print(f"\nTotal predictions: {len(df_baseline)}")

    # Tier precision - BASELINE
    print("\n" + "=" * 70)
    print("BASELINE (no demotion) - Precision by tier")
    print("=" * 70)

    tier_order = ['GOLDEN', 'HIGH', 'MEDIUM', 'LOW', 'FILTER']
    baseline_prec = {}

    for tier in tier_order:
        tier_data = df_baseline[df_baseline['final_tier'] == tier]
        if len(tier_data) > 0:
            prec = tier_data['is_hit'].mean() * 100
            n = len(tier_data)
            baseline_prec[tier] = prec
            print(f"  {tier}: {prec:.2f}% ({n} predictions)")

    # Tier precision - WITH DEMOTION
    print("\n" + "=" * 70)
    print("WITH DEMOTION - Precision by tier")
    print("=" * 70)

    demoted_prec = {}
    tier_changes = {'GOLDEN_to_HIGH': 0, 'HIGH_to_MEDIUM': 0}

    for tier in tier_order:
        tier_data = df_demoted[df_demoted['final_tier'] == tier]
        if len(tier_data) > 0:
            prec = tier_data['is_hit'].mean() * 100
            n = len(tier_data)
            demoted_prec[tier] = prec
            print(f"  {tier}: {prec:.2f}% ({n} predictions)")

    # Count demotions
    for _, row in df_demoted.iterrows():
        if row['base_tier'] == 'GOLDEN' and row['final_tier'] == 'HIGH':
            tier_changes['GOLDEN_to_HIGH'] += 1
        elif row['base_tier'] == 'HIGH' and row['final_tier'] == 'MEDIUM':
            tier_changes['HIGH_to_MEDIUM'] += 1

    print(f"\n  Demotions: GOLDEN→HIGH: {tier_changes['GOLDEN_to_HIGH']}, HIGH→MEDIUM: {tier_changes['HIGH_to_MEDIUM']}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Did demotion improve GOLDEN precision?
    if 'GOLDEN' in baseline_prec and 'GOLDEN' in demoted_prec:
        golden_diff = demoted_prec['GOLDEN'] - baseline_prec['GOLDEN']
        print(f"\nGOLDEN precision change: {golden_diff:+.2f} pp")
        print(f"  Baseline: {baseline_prec['GOLDEN']:.2f}%")
        print(f"  With demotion: {demoted_prec['GOLDEN']:.2f}%")

        if golden_diff > 5:
            print("  → SUCCESS: Removing incoherent predictions improved GOLDEN precision >5pp")
            success = True
        elif golden_diff > 0:
            print("  → MARGINAL: Small improvement in GOLDEN precision")
            success = False
        else:
            print("  → FAIL: No improvement in GOLDEN precision")
            success = False
    else:
        success = False
        golden_diff = 0

    # Save results
    results = {
        "hypothesis": "h311",
        "baseline_tier_precision": baseline_prec,
        "demoted_tier_precision": demoted_prec,
        "golden_precision_change_pp": float(golden_diff) if 'GOLDEN' in baseline_prec else None,
        "demotions": tier_changes,
        "total_predictions": len(df_baseline),
        "success": success,
    }

    output_file = ANALYSIS_DIR / "h311_incoherent_demotion.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
