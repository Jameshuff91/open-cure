#!/usr/bin/env python3
"""
Validate the target overlap boost strategy rigorously.

Tests:
1. Statistical significance (McNemar's test for paired data)
2. Per-disease analysis (how many diseases improved vs hurt)
3. Bootstrap confidence intervals
4. Analysis of which predictions changed
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Set, List, Tuple
import numpy as np
import torch
from tqdm import tqdm
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
MODELS_DIR = PROJECT_ROOT / "models"


def load_data():
    """Load all required data."""
    print("Loading data...")

    # Load baseline model
    with open(MODELS_DIR / "drug_repurposing_gb_enhanced.pkl", "rb") as f:
        model = pickle.load(f)

    # Load embeddings
    checkpoint = torch.load(MODELS_DIR / "transe.pt", map_location="cpu", weights_only=False)
    embeddings = None
    if "entity_embeddings" in checkpoint:
        embeddings = checkpoint["entity_embeddings"].numpy()
    elif "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
        for key in ["entity_embeddings.weight", "ent_embeddings.weight"]:
            if key in state:
                embeddings = state[key].numpy()
                break
    entity2id = checkpoint.get("entity2id", {})

    # Load target data
    with open(REFERENCE_DIR / "drug_targets.json") as f:
        drug_targets_raw = json.load(f)
    drug_targets = {k: set(v) for k, v in drug_targets_raw.items()}

    with open(REFERENCE_DIR / "disease_genes.json") as f:
        disease_genes_raw = json.load(f)
    disease_genes = {k: set(v) for k, v in disease_genes_raw.items()}

    # Load mappings
    with open(REFERENCE_DIR / "mesh_mappings_from_agents.json") as f:
        mesh_data = json.load(f)

    mesh_mappings = {}
    for batch_data in mesh_data.values():
        if isinstance(batch_data, dict):
            for disease_name, mesh_id in batch_data.items():
                if mesh_id:
                    mesh_str = str(mesh_id)
                    if mesh_str.startswith("D"):
                        mesh_mappings[disease_name.lower()] = f"drkg:Disease::MESH:{mesh_str}"

    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)
    name_to_id = {name.lower(): f"drkg:Compound::{db_id}" for db_id, name in id_to_name.items()}

    with open(REFERENCE_DIR / "everycure_gt_for_txgnn.json") as f:
        gt = json.load(f)

    return model, embeddings, entity2id, drug_targets, disease_genes, mesh_mappings, name_to_id, gt, id_to_name


def evaluate_disease(
    disease_name: str,
    disease_data: dict,
    model,
    embeddings: np.ndarray,
    entity2id: dict,
    drug_targets: dict,
    disease_genes: dict,
    mesh_mappings: dict,
    name_to_id: dict,
    valid_drug_ids: List[str],
    drug_embs: np.ndarray,
    drug_id_to_local_idx: dict,
):
    """
    Evaluate a single disease and return baseline/boosted top-30 sets.

    Returns: (baseline_top30, boosted_top30, gt_local_indices, mesh_id) or (None, None, 0, None)
    """
    mesh_id = mesh_mappings.get(disease_name.lower())
    if not mesh_id:
        return None, None, 0, None

    disease_idx = entity2id.get(mesh_id)
    if disease_idx is None:
        return None, None, 0, None

    disease_emb = embeddings[disease_idx]
    mesh_id_short = mesh_id.split("MESH:")[-1]

    # Get disease genes
    dis_genes = disease_genes.get(f"MESH:{mesh_id_short}", set())

    # Get GT drugs
    gt_local_indices = set()
    for drug_info in disease_data['drugs']:
        drug_name = drug_info['name'].lower()
        drug_id = name_to_id.get(drug_name)
        if drug_id and drug_id in drug_id_to_local_idx:
            gt_local_indices.add(drug_id_to_local_idx[drug_id])

    if not gt_local_indices:
        return None, None, 0, None

    # Score drugs
    n_drugs = len(drug_embs)
    disease_emb_tiled = np.tile(disease_emb, (n_drugs, 1))

    concat_feats = np.hstack([drug_embs, disease_emb_tiled])
    product_feats = drug_embs * disease_emb_tiled
    diff_feats = drug_embs - disease_emb_tiled
    base_features = np.hstack([concat_feats, product_feats, diff_feats])

    base_scores = model.predict_proba(base_features)[:, 1]

    # Compute target overlap for each drug
    overlaps = []
    for drug_id in valid_drug_ids:
        db_id = drug_id.split("::")[-1]
        drug_genes = drug_targets.get(db_id, set())
        overlap = len(drug_genes & dis_genes)
        overlaps.append(overlap)
    overlaps = np.array(overlaps)

    # Baseline top-30
    baseline_top30 = set(np.argsort(base_scores)[-30:])

    # Boosted scores (boost_by_overlap strategy)
    boosted_scores = base_scores * (1 + 0.01 * np.minimum(overlaps, 10))
    boosted_top30 = set(np.argsort(boosted_scores)[-30:])

    return baseline_top30, boosted_top30, gt_local_indices, mesh_id


def main():
    print("=" * 70)
    print("TARGET BOOST VALIDATION")
    print("=" * 70)

    # Load data
    model, embeddings, entity2id, drug_targets, disease_genes, mesh_mappings, name_to_id, gt, id_to_name = load_data()

    # Get all drug IDs
    all_drug_ids = [eid for eid in entity2id.keys() if "Compound" in eid]

    # Pre-compute drug embeddings
    valid_drug_ids = []
    valid_drug_indices = []
    for drug_id in all_drug_ids:
        drug_idx = entity2id.get(drug_id)
        if drug_idx is not None:
            valid_drug_ids.append(drug_id)
            valid_drug_indices.append(drug_idx)

    drug_embs = embeddings[valid_drug_indices]
    drug_id_to_local_idx = {did: i for i, did in enumerate(valid_drug_ids)}

    # Collect per-disease results
    print("\n1. Evaluating all diseases...")
    results = []

    for disease_name, disease_data in tqdm(gt.items(), desc="Diseases"):
        baseline_top30, boosted_top30, gt_indices, mesh_id = evaluate_disease(
            disease_name, disease_data, model, embeddings, entity2id,
            drug_targets, disease_genes, mesh_mappings, name_to_id,
            valid_drug_ids, drug_embs, drug_id_to_local_idx
        )

        if baseline_top30 is None:
            continue

        # Count hits
        baseline_hits = len(baseline_top30 & gt_indices)
        boosted_hits = len(boosted_top30 & gt_indices)

        results.append({
            'disease': disease_name,
            'mesh_id': mesh_id,
            'n_gt': len(gt_indices),
            'baseline_hits': baseline_hits,
            'boosted_hits': boosted_hits,
            'diff': boosted_hits - baseline_hits,
            'baseline_top30': baseline_top30,
            'boosted_top30': boosted_top30,
            'gt_indices': gt_indices,
        })

    n_diseases = len(results)
    print(f"\nEvaluated {n_diseases} diseases")

    # =========================================================================
    # TEST 1: McNemar's Test for Statistical Significance
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 1: McNemar's Test (paired comparison)")
    print("=" * 70)

    # For each GT drug, track if it was hit by baseline only, boost only, both, or neither
    baseline_only = 0  # Hit by baseline, missed by boost
    boost_only = 0     # Hit by boost, missed by baseline
    both = 0           # Hit by both
    neither = 0        # Missed by both

    for r in results:
        for gt_idx in r['gt_indices']:
            in_baseline = gt_idx in r['baseline_top30']
            in_boosted = gt_idx in r['boosted_top30']

            if in_baseline and in_boosted:
                both += 1
            elif in_baseline and not in_boosted:
                baseline_only += 1
            elif not in_baseline and in_boosted:
                boost_only += 1
            else:
                neither += 1

    print(f"\nContingency table (per GT drug):")
    print(f"  Both hit:       {both}")
    print(f"  Baseline only:  {baseline_only}")
    print(f"  Boost only:     {boost_only}")
    print(f"  Neither:        {neither}")

    # McNemar's test: compares baseline_only vs boost_only
    # Null hypothesis: boost_only = baseline_only (no difference)
    if baseline_only + boost_only > 0:
        # Exact binomial test
        n = baseline_only + boost_only
        k = boost_only  # successes for boost
        result = stats.binomtest(k, n, 0.5, alternative='greater')
        p_value = result.pvalue

        print(f"\nMcNemar's test (one-sided, boost > baseline):")
        print(f"  Discordant pairs: {n}")
        print(f"  Boost wins: {boost_only}, Baseline wins: {baseline_only}")
        print(f"  P-value: {p_value:.6f}")

        if p_value < 0.05:
            print("  ✓ Statistically significant (p < 0.05)")
        else:
            print("  ✗ NOT statistically significant (p >= 0.05)")

    # =========================================================================
    # TEST 2: Per-Disease Analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 2: Per-Disease Analysis")
    print("=" * 70)

    improved = sum(1 for r in results if r['diff'] > 0)
    hurt = sum(1 for r in results if r['diff'] < 0)
    unchanged = sum(1 for r in results if r['diff'] == 0)

    print(f"\nDisease-level changes:")
    print(f"  Improved:   {improved} ({100*improved/n_diseases:.1f}%)")
    print(f"  Hurt:       {hurt} ({100*hurt/n_diseases:.1f}%)")
    print(f"  Unchanged:  {unchanged} ({100*unchanged/n_diseases:.1f}%)")

    # Sign test
    if improved + hurt > 0:
        result_sign = stats.binomtest(improved, improved + hurt, 0.5, alternative='greater')
        p_value_sign = result_sign.pvalue
        print(f"\nSign test (one-sided):")
        print(f"  P-value: {p_value_sign:.6f}")
        if p_value_sign < 0.05:
            print("  ✓ More diseases improved than hurt (p < 0.05)")
        else:
            print("  ✗ No significant difference in disease counts")

    # Show top improved and hurt diseases
    sorted_by_diff = sorted(results, key=lambda x: x['diff'], reverse=True)

    print("\nTop 10 IMPROVED diseases:")
    for r in sorted_by_diff[:10]:
        if r['diff'] > 0:
            print(f"  {r['disease'][:40]:40} +{r['diff']} hits ({r['baseline_hits']}→{r['boosted_hits']})")

    print("\nTop 10 HURT diseases:")
    for r in sorted_by_diff[-10:]:
        if r['diff'] < 0:
            print(f"  {r['disease'][:40]:40} {r['diff']} hits ({r['baseline_hits']}→{r['boosted_hits']})")

    # =========================================================================
    # TEST 3: Bootstrap Confidence Intervals
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 3: Bootstrap Confidence Intervals")
    print("=" * 70)

    n_bootstrap = 1000
    baseline_recalls = []
    boosted_recalls = []
    diffs = []

    np.random.seed(42)
    for _ in tqdm(range(n_bootstrap), desc="Bootstrap"):
        # Sample diseases with replacement
        sample_indices = np.random.choice(len(results), len(results), replace=True)

        baseline_total = 0
        boosted_total = 0
        gt_total = 0

        for idx in sample_indices:
            r = results[idx]
            baseline_total += r['baseline_hits']
            boosted_total += r['boosted_hits']
            gt_total += r['n_gt']

        baseline_recall = baseline_total / gt_total if gt_total > 0 else 0
        boosted_recall = boosted_total / gt_total if gt_total > 0 else 0

        baseline_recalls.append(baseline_recall)
        boosted_recalls.append(boosted_recall)
        diffs.append(boosted_recall - baseline_recall)

    baseline_ci = np.percentile(baseline_recalls, [2.5, 97.5])
    boosted_ci = np.percentile(boosted_recalls, [2.5, 97.5])
    diff_ci = np.percentile(diffs, [2.5, 97.5])

    print(f"\n95% Confidence Intervals (bootstrap, n={n_bootstrap}):")
    print(f"  Baseline R@30: {np.mean(baseline_recalls):.1%} [{baseline_ci[0]:.1%}, {baseline_ci[1]:.1%}]")
    print(f"  Boosted R@30:  {np.mean(boosted_recalls):.1%} [{boosted_ci[0]:.1%}, {boosted_ci[1]:.1%}]")
    print(f"  Difference:    {np.mean(diffs):.2%} [{diff_ci[0]:.2%}, {diff_ci[1]:.2%}]")

    if diff_ci[0] > 0:
        print("  ✓ CI does not include 0 - improvement is robust")
    else:
        print("  ✗ CI includes 0 - improvement may be noise")

    # =========================================================================
    # TEST 4: Analysis of Changed Predictions
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 4: Analysis of Changed Predictions")
    print("=" * 70)

    # Count drugs that entered/left top-30 due to boost
    entered_top30 = []
    left_top30 = []

    for r in results:
        entered = r['boosted_top30'] - r['baseline_top30']
        left = r['baseline_top30'] - r['boosted_top30']

        for idx in entered:
            is_gt = idx in r['gt_indices']
            drug_id = valid_drug_ids[idx]
            db_id = drug_id.split("::")[-1]
            drug_name = id_to_name.get(db_id, db_id)
            mesh_short = r['mesh_id'].split("MESH:")[-1]
            targets = drug_targets.get(db_id, set())
            genes = disease_genes.get(f"MESH:{mesh_short}", set())
            overlap = len(targets & genes)
            entered_top30.append({
                'disease': r['disease'],
                'drug': drug_name,
                'is_gt': is_gt,
                'overlap': overlap,
                'n_targets': len(targets),
                'n_genes': len(genes),
            })

        for idx in left:
            is_gt = idx in r['gt_indices']
            drug_id = valid_drug_ids[idx]
            db_id = drug_id.split("::")[-1]
            drug_name = id_to_name.get(db_id, db_id)
            left_top30.append({
                'disease': r['disease'],
                'drug': drug_name,
                'is_gt': is_gt,
            })

    print(f"\nDrugs that ENTERED top-30 due to boost: {len(entered_top30)}")
    gt_entered = sum(1 for x in entered_top30 if x['is_gt'])
    print(f"  Of which are GT drugs: {gt_entered} ({100*gt_entered/len(entered_top30):.1f}%)")

    print(f"\nDrugs that LEFT top-30 due to boost: {len(left_top30)}")
    gt_left = sum(1 for x in left_top30 if x['is_gt'])
    print(f"  Of which are GT drugs: {gt_left} ({100*gt_left/len(left_top30) if left_top30 else 0:.1f}%)")

    print(f"\nNet GT drug change: +{gt_entered - gt_left}")

    # Show examples of GT drugs that entered
    print("\nExamples of GT drugs that ENTERED top-30:")
    for x in [e for e in entered_top30 if e['is_gt']][:10]:
        print(f"  {x['drug'][:25]:25} → {x['disease'][:30]:30} (overlap={x['overlap']})")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_baseline = sum(r['baseline_hits'] for r in results)
    total_boosted = sum(r['boosted_hits'] for r in results)
    total_gt = sum(r['n_gt'] for r in results)

    print(f"\nBaseline: {total_baseline}/{total_gt} = {100*total_baseline/total_gt:.1f}%")
    print(f"Boosted:  {total_boosted}/{total_gt} = {100*total_boosted/total_gt:.1f}%")
    print(f"Diff:     +{total_boosted - total_baseline} hits (+{100*(total_boosted-total_baseline)/total_gt:.2f}%)")


if __name__ == "__main__":
    main()
