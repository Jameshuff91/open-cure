#!/usr/bin/env python3
"""
h306: Low-Frequency Unique Demotion Rule

PURPOSE:
    h303 found low-freq unique is 2.3-3.4 pp worse than low-freq not-unique.
    Test if adding unique + freq<=5 -> demote one tier improves precision.

SUCCESS CRITERIA:
    >1 pp precision improvement at MEDIUM tier from demotion rule.
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

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"

SEEDS = [42, 123, 456, 789, 1024]


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

    for _, row in df.iterrows():
        disease = str(row.get("disease name", "")).strip()
        drug = str(row.get("final normalized drug label", "")).strip()
        if not disease or not drug:
            continue

        disease_id = matcher.get_mesh_id(disease)
        if not disease_id:
            continue

        disease_names[disease_id] = disease
        drug_id = name_to_drug_id.get(drug.lower())
        if drug_id:
            gt[disease_id].add(drug_id)

    return dict(gt), disease_names


def load_drug_targets() -> Dict[str, Set[str]]:
    """Load drug->targets mapping for mechanism support calculation.

    Note: drug_targets.json uses raw DrugBank IDs (DB00661),
    but we need drkg format (drkg:Compound::DB00661).
    """
    targets_path = REFERENCE_DIR / "drug_targets.json"
    if targets_path.exists():
        with open(targets_path) as f:
            data = json.load(f)
        # Convert to drkg format for compatibility
        result = {}
        for drug_id, targets in data.items():
            drkg_id = f"drkg:Compound::{drug_id}"
            result[drkg_id] = set(targets)
        return result
    return {}


def load_disease_genes() -> Dict[str, Set[str]]:
    """Load disease->genes mapping for mechanism support calculation."""
    genes_path = REFERENCE_DIR / "disease_genes.json"
    if genes_path.exists():
        with open(genes_path) as f:
            data = json.load(f)
        return {k: set(v) for k, v in data.items()}
    return {}


def get_atc_class(atc_code: str) -> str:
    if atc_code and len(atc_code) >= 1:
        return atc_code[0]
    return "unknown"


# h135/h136: Category definitions for tier determination
TIER_1_CATEGORIES = {'autoimmune', 'dermatological', 'psychiatric', 'ophthalmic'}
TIER_2_CATEGORIES = {'cardiovascular', 'other', 'cancer'}

# h169/h148: Category keywords for disease classification
CATEGORY_KEYWORDS = {
    'autoimmune': ['autoimmune', 'lupus', 'rheumatoid', 'arthritis', 'scleroderma', 'myasthenia',
                   'multiple sclerosis', 'crohn', 'colitis', 'psoriasis', 'sjögren', 'behcet',
                   'spondylitis', 'vasculitis', 'dermatomyositis', 'polymyositis', 'still disease'],
    'infectious': ['infection', 'bacterial', 'viral', 'fungal', 'hiv', 'aids', 'hepatitis',
                   'tuberculosis', 'malaria', 'pneumonia', 'sepsis', 'meningitis'],
    'cancer': ['cancer', 'carcinoma', 'tumor', 'leukemia', 'lymphoma', 'melanoma',
               'neoplasm', 'oncology', 'sarcoma', 'myeloma', 'glioma', 'adenocarcinoma'],
    'cardiovascular': ['cardiac', 'heart', 'coronary', 'hypertension', 'arrhythmia',
                       'atherosclerosis', 'stroke', 'vascular', 'myocardial', 'angina'],
    'neurological': ['neurological', 'alzheimer', 'parkinson', 'epilepsy', 'neuropathy',
                     'dementia', 'huntington', 'brain', 'seizure', 'ataxia', 'dystonia'],
    'metabolic': ['diabetes', 'metabolic', 'obesity', 'thyroid', 'hyperlipidemia',
                  'hypercholesterolemia', 'gout', 'porphyria', 'glycogen storage'],
    'respiratory': ['respiratory', 'asthma', 'copd', 'bronchitis', 'fibrosis',
                    'emphysema', 'lung'],
    'psychiatric': ['psychiatric', 'depression', 'anxiety', 'bipolar', 'schizophrenia',
                    'adhd', 'autism', 'ocd', 'ptsd', 'insomnia', 'phobia', 'personality'],
    'dermatological': ['dermatological', 'skin', 'eczema', 'acne', 'rosacea',
                       'dermatitis', 'urticaria', 'pruritus', 'alopecia', 'vitiligo'],
    'ophthalmic': ['ophthalmic', 'eye', 'glaucoma', 'macular', 'retinal', 'cataracts',
                   'conjunctivitis', 'uveitis', 'keratitis', 'dry eye'],
    'gastrointestinal': ['gastrointestinal', 'gastric', 'bowel', 'intestinal', 'stomach',
                         'ibd', 'gerd', 'gastritis', 'hepatic', 'liver', 'cirrhosis'],
}


def categorize_disease(disease_name: str) -> str:
    """Categorize a disease by name."""
    name_lower = disease_name.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in name_lower:
                return category
    return 'other'


def get_disease_tier(category: str) -> int:
    """Determine disease tier from category (h135 validated)."""
    if category.lower() in TIER_1_CATEGORIES:
        return 1
    elif category.lower() in TIER_2_CATEGORIES:
        return 2
    return 3


def simulate_tier_assignment(
    rank: int,
    train_freq: int,
    mechanism_support: bool,
    has_targets: bool,
    is_class_unique: bool,
    disease_tier: int = 3,
    demotion_threshold: int = 5,
    apply_demotion: bool = False,
) -> str:
    """
    Simulate simplified tier assignment to understand the impact of uniqueness demotion.

    Returns tier as string: 'GOLDEN', 'HIGH', 'MEDIUM', 'LOW', 'FILTER'
    """
    # FILTER tier (baseline)
    if rank > 20:
        return 'FILTER'
    if not has_targets:
        return 'FILTER'
    if train_freq <= 2 and not mechanism_support:
        return 'FILTER'

    # h306: Apply demotion for low-freq unique predictions
    low_freq_unique_demotion = 0
    if apply_demotion and is_class_unique and train_freq <= demotion_threshold:
        low_freq_unique_demotion = 1  # Demote one tier

    # GOLDEN tier (Tier1 + freq>=10 + mechanism)
    if disease_tier == 1 and train_freq >= 10 and mechanism_support:
        tier = 'GOLDEN'
        if low_freq_unique_demotion:
            tier = 'HIGH'  # Demote GOLDEN -> HIGH
        return tier

    # HIGH tier
    if train_freq >= 15 and mechanism_support:
        tier = 'HIGH'
        if low_freq_unique_demotion:
            tier = 'MEDIUM'
        return tier
    if rank <= 5 and train_freq >= 10 and mechanism_support:
        tier = 'HIGH'
        if low_freq_unique_demotion:
            tier = 'MEDIUM'
        return tier

    # MEDIUM tier
    if train_freq >= 5 and mechanism_support:
        tier = 'MEDIUM'
        if low_freq_unique_demotion:
            tier = 'LOW'
        return tier
    if train_freq >= 10:
        tier = 'MEDIUM'
        if low_freq_unique_demotion:
            tier = 'LOW'
        return tier

    # LOW tier (default)
    tier = 'LOW'
    # LOW can't be demoted further (or could demote to FILTER if needed)
    return tier


def run_analysis(
    emb_dict: Dict[str, np.ndarray],
    train_gt: Dict[str, Set[str]],
    test_gt: Dict[str, Set[str]],
    drug_atc: Dict[str, str],
    drug_targets: Dict[str, Set[str]],
    disease_genes: Dict[str, Set[str]],
    id_to_name: Dict[str, str],
    disease_names: Dict[str, str],
    k: int = 20,
) -> List[Dict]:
    """Run kNN and compute tier assignments with and without demotion."""
    train_disease_list = [d for d in train_gt if d in emb_dict]
    train_disease_embs = np.array([emb_dict[d] for d in train_disease_list], dtype=np.float32)

    # Drug training frequency
    drug_train_freq: Dict[str, int] = defaultdict(int)
    for disease_id, drugs in train_gt.items():
        for drug_id in drugs:
            drug_train_freq[drug_id] += 1

    # Build ATC class -> drugs mapping from training data
    atc_class_drugs: Dict[str, Set[str]] = defaultdict(set)
    for disease_id, drugs in train_gt.items():
        for drug_id in drugs:
            atc = drug_atc.get(drug_id, "")
            if atc:
                atc_class = get_atc_class(atc)
                atc_class_drugs[atc_class].add(drug_id)

    results = []

    for disease_id in test_gt:
        if disease_id not in emb_dict:
            continue
        gt_drugs = {d for d in test_gt[disease_id] if d in emb_dict}
        if not gt_drugs:
            continue

        disease_name = disease_names.get(disease_id, "")
        disease_genes_set = disease_genes.get(disease_id, set())

        # Compute disease category and tier
        disease_category = categorize_disease(disease_name)
        disease_tier = get_disease_tier(disease_category)

        # Find k nearest training diseases
        test_emb = emb_dict[disease_id].reshape(1, -1)
        sims = cosine_similarity(test_emb, train_disease_embs)[0]
        top_k_idx = np.argsort(sims)[-k:]
        neighbor_diseases = set(train_disease_list[idx] for idx in top_k_idx)

        # Get drugs from neighbors
        neighbor_drugs = set()
        for neighbor in neighbor_diseases:
            neighbor_drugs.update(train_gt[neighbor])

        # Count drug frequency from neighbors (with weighting)
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
            atc_class = get_atc_class(atc) if atc else ""

            # Class uniqueness: is this the ONLY drug from its ATC class in neighbors?
            classmate_drugs = atc_class_drugs.get(atc_class, set())
            classmates_in_neighbors = classmate_drugs & neighbor_drugs
            n_classmates_in_neighbors = len(classmates_in_neighbors)
            is_class_unique = (n_classmates_in_neighbors <= 1)

            # Training frequency
            train_freq = drug_train_freq.get(drug_id, 0)

            # Mechanism support
            drug_targets_set = drug_targets.get(drug_id, set())
            mechanism_support = len(drug_targets_set & disease_genes_set) > 0
            has_targets = len(drug_targets_set) > 0

            is_hit = drug_id in gt_drugs
            drug_name = id_to_name.get(drug_id, drug_id.split("::")[-1])

            # Compute tiers with different demotion thresholds
            tier_baseline = simulate_tier_assignment(
                rank, train_freq, mechanism_support, has_targets, is_class_unique,
                disease_tier=disease_tier, demotion_threshold=5, apply_demotion=False
            )
            tier_demote_5 = simulate_tier_assignment(
                rank, train_freq, mechanism_support, has_targets, is_class_unique,
                disease_tier=disease_tier, demotion_threshold=5, apply_demotion=True
            )
            tier_demote_3 = simulate_tier_assignment(
                rank, train_freq, mechanism_support, has_targets, is_class_unique,
                disease_tier=disease_tier, demotion_threshold=3, apply_demotion=True
            )
            tier_demote_10 = simulate_tier_assignment(
                rank, train_freq, mechanism_support, has_targets, is_class_unique,
                disease_tier=disease_tier, demotion_threshold=10, apply_demotion=True
            )

            results.append({
                'disease': disease_id,
                'drug': drug_id,
                'drug_name': drug_name,
                'disease_name': disease_name,
                'disease_category': disease_category,
                'disease_tier': disease_tier,
                'atc_class': atc_class,
                'is_class_unique': is_class_unique,
                'n_classmates': n_classmates_in_neighbors,
                'train_frequency': train_freq,
                'mechanism_support': mechanism_support,
                'has_targets': has_targets,
                'knn_score': score,
                'norm_score': score / max_score if max_score > 0 else 0,
                'rank': rank,
                'is_hit': 1 if is_hit else 0,
                'tier_baseline': tier_baseline,
                'tier_demote_3': tier_demote_3,
                'tier_demote_5': tier_demote_5,
                'tier_demote_10': tier_demote_10,
            })

    return results


def compute_tier_precision(df: pd.DataFrame, tier_col: str) -> Dict[str, Dict]:
    """Compute precision by tier for a given tier column."""
    tiers = ['GOLDEN', 'HIGH', 'MEDIUM', 'LOW', 'FILTER']
    results = {}
    for tier in tiers:
        tier_df = df[df[tier_col] == tier]
        if len(tier_df) > 0:
            precision = tier_df['is_hit'].mean() * 100
            results[tier] = {
                'n': len(tier_df),
                'hits': int(tier_df['is_hit'].sum()),
                'precision': precision,
            }
        else:
            results[tier] = {'n': 0, 'hits': 0, 'precision': 0}
    return results


def main():
    print("h306: Low-Frequency Unique Demotion Rule")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    emb_dict = load_node2vec_embeddings()
    name_to_drug_id, id_to_name = load_drugbank_lookup()
    ground_truth, disease_names = load_ground_truth(name_to_drug_id)
    drug_targets = load_drug_targets()
    disease_genes = load_disease_genes()

    # Build drug_atc mapping using ATCMapper
    atc_mapper = ATCMapper()
    drug_atc = {}
    for drug_id, drug_name in id_to_name.items():
        codes = atc_mapper.get_atc_codes(drug_name)
        if codes:
            drug_atc[drug_id] = codes[0]

    print(f"  Embeddings: {len(emb_dict)}")
    print(f"  Ground truth diseases: {len(ground_truth)}")
    print(f"  Drugs with ATC codes: {len(drug_atc)}")
    print(f"  Drug targets: {len(drug_targets)}")
    print(f"  Disease genes: {len(disease_genes)}")

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
            emb_dict, train_gt, test_gt, drug_atc, drug_targets, disease_genes,
            id_to_name, disease_names, k=20
        )
        all_results.extend(seed_results)
        print(f"  Seed {seed}: {len(seed_results)} predictions")

    df = pd.DataFrame(all_results)
    print(f"\nTotal predictions: {len(df)}")
    print(f"Base hit rate: {df['is_hit'].mean()*100:.2f}%")

    # Filter to drugs with ATC
    has_atc = df[df['atc_class'] != ''].copy()
    print(f"With ATC data: {len(has_atc)} predictions")

    # === VERIFY h303 inverse finding ===
    print(f"\n{'='*70}")
    print("VERIFY h303: Low-frequency unique vs not-unique precision")
    print("=" * 70)

    for thresh in [2, 3, 5, 10]:
        low_freq = has_atc[has_atc['train_frequency'] <= thresh]
        unique = low_freq[low_freq['is_class_unique']]
        not_unique = low_freq[~low_freq['is_class_unique']]

        if len(unique) > 0 and len(not_unique) > 0:
            unique_prec = unique['is_hit'].mean() * 100
            not_unique_prec = not_unique['is_hit'].mean() * 100
            gap = unique_prec - not_unique_prec
            print(f"freq <= {thresh}: Unique {unique_prec:.2f}% ({len(unique)}) vs Not-unique {not_unique_prec:.2f}% ({len(not_unique)}) | Gap: {gap:+.2f} pp")

    # === MAIN ANALYSIS: Tier precision comparison ===
    print(f"\n{'='*70}")
    print("MAIN ANALYSIS: Tier precision with and without demotion")
    print("=" * 70)

    tier_cols = ['tier_baseline', 'tier_demote_3', 'tier_demote_5', 'tier_demote_10']
    tier_names = ['Baseline', 'Demote freq≤3', 'Demote freq≤5', 'Demote freq≤10']

    tier_results = {}
    for col in tier_cols:
        tier_results[col] = compute_tier_precision(has_atc, col)

    # Print comparison table
    print(f"\n{'Tier':<10}", end="")
    for name in tier_names:
        print(f" {name:>20}", end="")
    print()
    print("-" * 90)

    tiers = ['GOLDEN', 'HIGH', 'MEDIUM', 'LOW', 'FILTER']
    for tier in tiers:
        print(f"{tier:<10}", end="")
        for col in tier_cols:
            r = tier_results[col][tier]
            if r['n'] > 0:
                print(f" {r['precision']:>6.2f}% (n={r['n']:>4})", end="")
            else:
                print(f" {'N/A':>20}", end="")
        print()

    # === Count affected predictions ===
    print(f"\n{'='*70}")
    print("AFFECTED PREDICTIONS: How many predictions are demoted?")
    print("=" * 70)

    for thresh, col in [(3, 'tier_demote_3'), (5, 'tier_demote_5'), (10, 'tier_demote_10')]:
        demoted = has_atc[has_atc['tier_baseline'] != has_atc[col]]
        n_demoted = len(demoted)
        demoted_hits = demoted['is_hit'].sum()
        demoted_prec = demoted['is_hit'].mean() * 100 if n_demoted > 0 else 0
        print(f"\nDemotion threshold freq <= {thresh}:")
        print(f"  Total demoted: {n_demoted} predictions")
        print(f"  Demoted hits: {int(demoted_hits)}")
        print(f"  Demoted precision: {demoted_prec:.2f}%")

        # Show before/after tiers
        if n_demoted > 0:
            tier_changes = demoted.groupby(['tier_baseline', col]).size().reset_index(name='count')
            print(f"  Tier changes:")
            for _, row in tier_changes.iterrows():
                print(f"    {row['tier_baseline']} -> {row[col]}: {row['count']}")

    # === PRECISION CHANGE ANALYSIS ===
    print(f"\n{'='*70}")
    print("PRECISION IMPACT: Change from baseline by tier")
    print("=" * 70)

    baseline = tier_results['tier_baseline']

    for thresh, col in [(3, 'tier_demote_3'), (5, 'tier_demote_5'), (10, 'tier_demote_10')]:
        demote = tier_results[col]
        print(f"\nDemotion threshold freq <= {thresh}:")
        for tier in tiers:
            b = baseline[tier]
            d = demote[tier]
            if b['n'] > 0 and d['n'] > 0:
                diff = d['precision'] - b['precision']
                n_change = d['n'] - b['n']
                print(f"  {tier}: {b['precision']:.2f}% -> {d['precision']:.2f}% ({diff:+.2f} pp) | n: {b['n']} -> {d['n']} ({n_change:+d})")

    # === OVERALL WEIGHTED PRECISION ===
    print(f"\n{'='*70}")
    print("OVERALL WEIGHTED PRECISION (excluding FILTER)")
    print("=" * 70)

    for name, col in zip(tier_names, tier_cols):
        non_filter = has_atc[has_atc[col] != 'FILTER']
        if len(non_filter) > 0:
            overall_prec = non_filter['is_hit'].mean() * 100
            print(f"{name}: {overall_prec:.3f}% (n={len(non_filter)})")

    # === SUMMARY ===
    print(f"\n{'='*70}")
    print("SUMMARY: h306 Findings")
    print("=" * 70)

    # Calculate MEDIUM tier precision change (success criterion)
    baseline_medium_prec = tier_results['tier_baseline']['MEDIUM']['precision']
    demote5_medium_prec = tier_results['tier_demote_5']['MEDIUM']['precision']
    medium_change = demote5_medium_prec - baseline_medium_prec

    print(f"\nMEDIUM tier precision change (freq <= 5 demotion):")
    print(f"  Baseline: {baseline_medium_prec:.2f}%")
    print(f"  With demotion: {demote5_medium_prec:.2f}%")
    print(f"  Change: {medium_change:+.2f} pp")

    if medium_change > 1.0:
        print(f"\n✓ SUCCESS: MEDIUM tier precision improved by {medium_change:.2f} pp (>1 pp threshold)")
        conclusion = "VALIDATED"
    elif medium_change > 0:
        print(f"\n~ MARGINAL: MEDIUM tier precision improved by {medium_change:.2f} pp (below 1 pp threshold)")
        conclusion = "MARGINAL"
    else:
        print(f"\n✗ FAILED: MEDIUM tier precision did not improve (change: {medium_change:+.2f} pp)")
        conclusion = "INVALIDATED"

    print(f"\nCONCLUSION: {conclusion}")

    # Check if there are few low-freq unique predictions
    low_freq_unique = has_atc[(has_atc['is_class_unique']) & (has_atc['train_frequency'] <= 5)]
    print(f"\nNote: Low-frequency unique predictions (freq<=5): {len(low_freq_unique)} ({len(low_freq_unique)/len(has_atc)*100:.1f}%)")

    # Save results
    results = {
        'hypothesis': 'h306',
        'conclusion': conclusion,
        'n_predictions': len(has_atc),
        'tier_precision_baseline': {k: v for k, v in tier_results['tier_baseline'].items()},
        'tier_precision_demote_3': {k: v for k, v in tier_results['tier_demote_3'].items()},
        'tier_precision_demote_5': {k: v for k, v in tier_results['tier_demote_5'].items()},
        'tier_precision_demote_10': {k: v for k, v in tier_results['tier_demote_10'].items()},
        'medium_tier_change_5': medium_change,
        'success_criterion': '>1 pp MEDIUM tier improvement',
    }

    results_file = ANALYSIS_DIR / "h306_low_freq_unique_demotion.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
