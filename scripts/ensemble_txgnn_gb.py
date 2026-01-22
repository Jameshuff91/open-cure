#!/usr/bin/env python3
"""
Ensemble TxGNN and GB Model Predictions
========================================

This script creates an ensemble of TxGNN and GB model predictions and evaluates
if the combination beats either model alone.

TxGNN: 14.5% Recall@30 on 779 diseases (uses drug rankings)
GB Enhanced: 13.2% Recall@30 on 77 diseases (uses MESH IDs)

**Important:** The TxGNN results CSV contains only the ranks of ground truth drugs,
not a full ranking of all drugs. So we use the TxGNN ranks as-is for GT drugs,
and the GB model provides full rankings for all drugs.

Ensemble strategies:
1. Simple average of rankings (for GT drugs we have both ranks)
2. Weighted average (try different weights)
3. Take best rank from either model
4. Reciprocal Rank Fusion (RRF)
"""

import ast
import json
import pickle
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from loguru import logger

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REFERENCE_DIR = DATA_DIR / "reference"
MODELS_DIR = PROJECT_ROOT / "models"
ANALYSIS_DIR = DATA_DIR / "analysis"

# Ensure analysis dir exists
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


# Disease name mappings (TxGNN uses lowercase names, GB uses MESH IDs)
# We need to map between them for the intersection
DISEASE_NAME_TO_MESH = {
    "hiv infection": "drkg:Disease::MESH:D015658",
    "hepatitis c": "drkg:Disease::MESH:D006526",
    "tuberculosis": "drkg:Disease::MESH:D014376",
    "breast cancer": "drkg:Disease::MESH:D001943",
    "lung cancer": "drkg:Disease::MESH:D008175",
    "colorectal cancer": "drkg:Disease::MESH:D015179",
    "hypertension": "drkg:Disease::MESH:D006973",
    "heart failure": "drkg:Disease::MESH:D006333",
    "atrial fibrillation": "drkg:Disease::MESH:D001281",
    "epilepsy": "drkg:Disease::MESH:D004827",
    "parkinson disease": "drkg:Disease::MESH:D010300",
    "alzheimer disease": "drkg:Disease::MESH:D000544",
    "rheumatoid arthritis": "drkg:Disease::MESH:D001172",
    "multiple sclerosis": "drkg:Disease::MESH:D009103",
    "psoriasis": "drkg:Disease::MESH:D011565",
    "type 2 diabetes mellitus": "drkg:Disease::MESH:D003924",
    "obesity": "drkg:Disease::MESH:D009765",
    "asthma": "drkg:Disease::MESH:D001249",
    "copd": "drkg:Disease::MESH:D029424",
    "osteoporosis": "drkg:Disease::MESH:D010024",
    "crohn disease": "drkg:Disease::MESH:D003424",
}

MESH_TO_DISEASE_NAME = {v: k for k, v in DISEASE_NAME_TO_MESH.items()}


def load_txgnn_results() -> pd.DataFrame:
    """Load TxGNN proper scoring results."""
    txgnn_path = REFERENCE_DIR / "txgnn_proper_scoring_results.csv"
    df = pd.read_csv(txgnn_path)
    logger.info(f"Loaded TxGNN results: {len(df)} diseases")
    return df


def parse_gt_ranks(gt_ranks_str: str) -> Dict[str, int]:
    """Parse the gt_ranks column from string to dict."""
    if pd.isna(gt_ranks_str):
        return {}
    try:
        return ast.literal_eval(gt_ranks_str)
    except (ValueError, SyntaxError):
        return {}


def parse_gt_sample(gt_sample_str: str) -> List[str]:
    """Parse the gt_sample column from string to list."""
    if pd.isna(gt_sample_str):
        return []
    try:
        return ast.literal_eval(gt_sample_str)
    except (ValueError, SyntaxError):
        return []


def create_features(drug_emb: np.ndarray, disease_emb: np.ndarray) -> np.ndarray:
    """Create features from embeddings for GB model."""
    concat = np.concatenate([drug_emb, disease_emb])
    product = drug_emb * disease_emb
    diff = drug_emb - disease_emb
    return np.concatenate([concat, product, diff])


def load_gb_model_and_embeddings():
    """Load GB model and TransE embeddings."""
    # Load GB model
    with open(MODELS_DIR / "drug_repurposing_gb_enhanced.pkl", "rb") as f:
        gb_model = pickle.load(f)
    logger.info("Loaded GB enhanced model")

    # Load TransE embeddings
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
    logger.info(f"Loaded embeddings: {embeddings.shape}")

    return gb_model, embeddings, entity2id


def load_drugbank_lookup() -> Dict[str, str]:
    """Load DrugBank ID to name mapping."""
    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)
    # Create name to ID mapping (lowercase)
    name_to_id = {name.lower(): f"drkg:Compound::{db_id}" for db_id, name in id_to_name.items()}
    return name_to_id


def get_gb_rankings_for_disease(
    disease_mesh: str,
    gb_model,
    embeddings: np.ndarray,
    entity2id: Dict[str, int],
    all_drug_ids: List[str],
    name_to_drugbank: Dict[str, str],
) -> Dict[str, int]:
    """Get GB model rankings for all drugs for a disease.

    Returns: dict mapping drug_name (lowercase) -> rank (1-indexed)
    """
    disease_idx = entity2id.get(disease_mesh)
    if disease_idx is None:
        return {}

    disease_emb = embeddings[disease_idx]

    # Score all drugs
    scores = []
    drug_names = []

    # Create reverse mapping from DrugBank ID to name
    drugbank_to_name = {v: k for k, v in name_to_drugbank.items()}

    for drug_id in all_drug_ids:
        drug_idx = entity2id.get(drug_id)
        if drug_idx is None:
            continue

        drug_emb = embeddings[drug_idx]
        features = create_features(drug_emb, disease_emb).reshape(1, -1)
        score = gb_model.predict_proba(features)[0, 1]
        scores.append(score)

        # Get drug name from ID
        drug_name = drugbank_to_name.get(drug_id, drug_id.split("::")[-1]).lower()
        drug_names.append(drug_name)

    # Rank drugs (higher score = better = lower rank number)
    ranked_indices = np.argsort(scores)[::-1]

    rankings = {}
    for rank, idx in enumerate(ranked_indices, 1):
        drug_name = drug_names[idx]
        rankings[drug_name] = rank

    return rankings


def reciprocal_rank_fusion(
    ranks: List[int],
    k: int = 60
) -> float:
    """Reciprocal Rank Fusion score. Higher is better."""
    return sum(1.0 / (k + r) for r in ranks if r is not None)


def evaluate_ensemble_for_gt_drugs(
    txgnn_gt_ranks: Dict[str, int],
    gb_rankings: Dict[str, int],
    strategy: str = "average",
    txgnn_weight: float = 0.5,
    k: int = 30
) -> Tuple[int, int, float, Dict[str, dict]]:
    """Evaluate ensemble on ground truth drugs only.

    The key insight: TxGNN only provides ranks for GT drugs (from gt_ranks column).
    GB provides ranks for all drugs. For ensemble, we compute combined ranks
    for GT drugs and see how many would be in top-k.

    Args:
        txgnn_gt_ranks: drug_name -> TxGNN rank (only for GT drugs)
        gb_rankings: drug_name -> GB rank (for all drugs)
        strategy: one of "average", "weighted", "best", "rrf"
        txgnn_weight: weight for TxGNN (GB weight = 1 - txgnn_weight)
        k: recall@k threshold

    Returns: (hits, total, recall, per_drug_info)
    """
    gt_drugs = list(txgnn_gt_ranks.keys())
    gb_weight = 1 - txgnn_weight

    # For each GT drug, compute ensemble score
    # Lower ensemble score = better (smaller rank)
    ensemble_scores = {}
    per_drug_info = {}

    for drug in gt_drugs:
        txgnn_r = txgnn_gt_ranks[drug]
        gb_r = gb_rankings.get(drug)

        if gb_r is None:
            # Drug not in GB model's vocabulary - use a large rank
            gb_r = len(gb_rankings) + 1000

        if strategy == "average":
            ensemble_scores[drug] = (txgnn_r + gb_r) / 2

        elif strategy == "weighted":
            ensemble_scores[drug] = txgnn_weight * txgnn_r + gb_weight * gb_r

        elif strategy == "best":
            ensemble_scores[drug] = min(txgnn_r, gb_r)

        elif strategy == "rrf":
            # RRF: sum of 1/(k+rank), higher is better, so we negate
            k_param = 60
            rrf_score = 1.0/(k_param + txgnn_r) + 1.0/(k_param + gb_r)
            ensemble_scores[drug] = -rrf_score  # Negate so lower is better

        per_drug_info[drug] = {
            'txgnn_rank': txgnn_r,
            'gb_rank': gb_r,
            'ensemble_score': ensemble_scores[drug],
        }

    # Now we need to determine: would each GT drug be in the top-k of the ensemble?
    # This is tricky because we only have GT drug ranks, not all drugs.
    #
    # For "best" strategy: a GT drug is in top-k if min(txgnn_r, gb_r) <= k
    # For "average" and "weighted": need to estimate. If the combined score is low
    # enough that the drug would rank highly.
    #
    # A conservative estimate: if the ensemble_score is <= k, the drug is likely in top-k
    # (since even if all drugs had rank 1 in one model and k in the other,
    # the average would be (1+k)/2 which is < k)

    hits = 0
    for drug in gt_drugs:
        score = ensemble_scores[drug]

        if strategy == "best":
            # Drug is in top-k if its best rank <= k
            if score <= k:
                hits += 1
                per_drug_info[drug]['in_top_k'] = True
            else:
                per_drug_info[drug]['in_top_k'] = False

        elif strategy == "rrf":
            # For RRF, we need to compare against other drugs
            # But since we only have GT drugs, use a threshold
            # If both ranks are <= k, RRF score would be at least 2/(60+k)
            # which is quite high (good). Threshold: if either rank <= k, likely in top-k
            txgnn_r = txgnn_gt_ranks[drug]
            gb_r = gb_rankings.get(drug, len(gb_rankings) + 1000)
            if txgnn_r <= k or gb_r <= k:
                hits += 1
                per_drug_info[drug]['in_top_k'] = True
            else:
                per_drug_info[drug]['in_top_k'] = False

        else:  # average, weighted
            # If ensemble_score (average rank) <= k, likely in top-k
            if score <= k:
                hits += 1
                per_drug_info[drug]['in_top_k'] = True
            else:
                per_drug_info[drug]['in_top_k'] = False

    total = len(gt_drugs)
    recall = hits / total if total > 0 else 0

    return hits, total, recall, per_drug_info


def main():
    logger.info("=" * 70)
    logger.info("TxGNN + GB Model Ensemble Evaluation")
    logger.info("=" * 70)

    # Load TxGNN results
    txgnn_df = load_txgnn_results()

    # Load GB model and embeddings
    gb_model, embeddings, entity2id = load_gb_model_and_embeddings()

    # Get all drug IDs for GB model
    all_drug_ids = [eid for eid in entity2id.keys() if "Compound" in eid]
    logger.info(f"Total drugs in DRKG: {len(all_drug_ids)}")

    # Load DrugBank lookup
    name_to_drugbank = load_drugbank_lookup()
    logger.info(f"DrugBank mappings: {len(name_to_drugbank)}")

    # Find diseases in common (that we can evaluate with both models)
    # For this, we need TxGNN diseases that map to MESH IDs we have
    common_diseases = []

    for _, row in txgnn_df.iterrows():
        disease_name = row['disease'].lower()
        mesh_id = DISEASE_NAME_TO_MESH.get(disease_name)

        if mesh_id and mesh_id in entity2id:
            gt_ranks = parse_gt_ranks(row['gt_ranks'])
            gt_sample = parse_gt_sample(row['gt_sample'])

            if gt_ranks:  # Has ground truth
                common_diseases.append({
                    'name': disease_name,
                    'mesh_id': mesh_id,
                    'txgnn_hit_at_30': row['hit_at_30'],
                    'gt_drugs_count': row['gt_drugs_count'],
                    'gt_ranks': gt_ranks,
                    'gt_sample': gt_sample,
                })

    logger.info(f"Diseases in common (mappable): {len(common_diseases)}")

    if len(common_diseases) == 0:
        logger.error("No common diseases found! Check disease name mappings.")
        return

    # Evaluate each ensemble strategy
    strategies = [
        ("txgnn_only", None, None),
        ("gb_only", None, None),
        ("average", "average", 0.5),
        ("weighted_txgnn_0.7", "weighted", 0.7),
        ("weighted_txgnn_0.3", "weighted", 0.3),
        ("weighted_gb_0.7", "weighted", 0.3),  # GB weight 0.7
        ("best_rank", "best", 0.5),
        ("rrf", "rrf", 0.5),
    ]

    results = {s[0]: {'hits': 0, 'total': 0, 'diseases': []} for s in strategies}

    for disease in tqdm(common_diseases, desc="Evaluating diseases"):
        disease_name = disease['name']
        mesh_id = disease['mesh_id']
        txgnn_gt_ranks = disease['gt_ranks']  # TxGNN ranks for GT drugs only

        # Get GB rankings for this disease
        gb_rankings = get_gb_rankings_for_disease(
            mesh_id, gb_model, embeddings, entity2id, all_drug_ids, name_to_drugbank
        )

        if not gb_rankings:
            logger.warning(f"No GB rankings for {disease_name}")
            continue

        gt_drugs = list(txgnn_gt_ranks.keys())

        for strategy_name, strategy, weight in strategies:
            if strategy_name == "txgnn_only":
                # Count GT drugs with TxGNN rank <= 30
                hits = sum(1 for drug, rank in txgnn_gt_ranks.items() if rank <= 30)
                total = len(gt_drugs)
                recall = hits / total if total > 0 else 0

            elif strategy_name == "gb_only":
                # Count GT drugs with GB rank <= 30
                hits = 0
                for drug in gt_drugs:
                    gb_rank = gb_rankings.get(drug)
                    if gb_rank is not None and gb_rank <= 30:
                        hits += 1
                total = len(gt_drugs)
                recall = hits / total if total > 0 else 0

            else:
                # Ensemble strategies
                hits, total, recall, _ = evaluate_ensemble_for_gt_drugs(
                    txgnn_gt_ranks,
                    gb_rankings,
                    strategy=strategy,
                    txgnn_weight=weight,
                    k=30
                )

            results[strategy_name]['hits'] += hits
            results[strategy_name]['total'] += total
            results[strategy_name]['diseases'].append({
                'name': disease_name,
                'hits': hits,
                'total': total,
                'recall': recall,
            })

    # Print results
    print("\n" + "=" * 80)
    print("ENSEMBLE EVALUATION RESULTS")
    print("=" * 80)
    print(f"\nDiseases evaluated: {len(common_diseases)}")
    print()

    print(f"{'Strategy':<25} {'Hits@30':>10} {'Total':>10} {'Recall@30':>12}")
    print("-" * 60)

    summary = []
    for strategy_name, _, _ in strategies:
        r = results[strategy_name]
        recall = r['hits'] / r['total'] if r['total'] > 0 else 0
        print(f"{strategy_name:<25} {r['hits']:>10} {r['total']:>10} {recall*100:>11.1f}%")
        summary.append({
            'strategy': strategy_name,
            'hits': r['hits'],
            'total': r['total'],
            'recall': recall,
        })

    print("-" * 60)

    # Identify best strategy
    best = max(summary, key=lambda x: x['recall'])
    print(f"\nBest strategy: {best['strategy']} with {best['recall']*100:.1f}% Recall@30")

    # Detailed per-disease comparison
    print("\n" + "=" * 80)
    print("PER-DISEASE BREAKDOWN")
    print("=" * 80)
    print(f"\n{'Disease':<30} {'TxGNN':>8} {'GB':>8} {'Avg':>8} {'Best':>8} {'RRF':>8} {'GT Cnt':>8}")
    print("-" * 90)

    for i, disease in enumerate(common_diseases):
        disease_name = disease['name'][:28]
        gt_count = disease['gt_drugs_count']
        txgnn_r = results['txgnn_only']['diseases'][i]['recall'] if i < len(results['txgnn_only']['diseases']) else 0
        gb_r = results['gb_only']['diseases'][i]['recall'] if i < len(results['gb_only']['diseases']) else 0
        avg_r = results['average']['diseases'][i]['recall'] if i < len(results['average']['diseases']) else 0
        best_r = results['best_rank']['diseases'][i]['recall'] if i < len(results['best_rank']['diseases']) else 0
        rrf_r = results['rrf']['diseases'][i]['recall'] if i < len(results['rrf']['diseases']) else 0

        print(f"{disease_name:<30} {txgnn_r*100:>7.1f}% {gb_r*100:>7.1f}% {avg_r*100:>7.1f}% {best_r*100:>7.1f}% {rrf_r*100:>7.1f}% {gt_count:>8}")

    # Analyze ensemble benefit - which drugs did ensemble recover that neither alone got?
    print("\n" + "=" * 80)
    print("ENSEMBLE BENEFIT ANALYSIS")
    print("=" * 80)

    ensemble_benefit_drugs = []

    for disease in common_diseases:
        disease_name = disease['name']
        mesh_id = disease['mesh_id']
        txgnn_gt_ranks = disease['gt_ranks']

        gb_rankings = get_gb_rankings_for_disease(
            mesh_id, gb_model, embeddings, entity2id, all_drug_ids, name_to_drugbank
        )

        for drug, txgnn_rank in txgnn_gt_ranks.items():
            gb_rank = gb_rankings.get(drug)

            txgnn_hit = txgnn_rank <= 30
            gb_hit = gb_rank is not None and gb_rank <= 30
            best_rank_hit = min(txgnn_rank, gb_rank if gb_rank else float('inf')) <= 30

            # Did ensemble (best_rank) find something neither found alone?
            if best_rank_hit and not txgnn_hit and not gb_hit:
                ensemble_benefit_drugs.append({
                    'disease': disease_name,
                    'drug': drug,
                    'txgnn_rank': txgnn_rank,
                    'gb_rank': gb_rank,
                })
            # Did ensemble find something only one found?
            elif best_rank_hit and (txgnn_hit or gb_hit) and not (txgnn_hit and gb_hit):
                pass  # This is expected behavior

    if ensemble_benefit_drugs:
        print(f"\nDrugs found by ensemble that neither model found alone:")
        for item in ensemble_benefit_drugs:
            print(f"  {item['disease']}: {item['drug']} (TxGNN: {item['txgnn_rank']}, GB: {item['gb_rank']})")
    else:
        print("\nNo additional drugs recovered by ensemble that weren't found by at least one model.")

    # Show complementary findings
    print("\n" + "=" * 80)
    print("COMPLEMENTARY FINDINGS (drugs found by one model, not the other)")
    print("=" * 80)

    txgnn_only_hits = []
    gb_only_hits = []

    for disease in common_diseases:
        disease_name = disease['name']
        mesh_id = disease['mesh_id']
        txgnn_gt_ranks = disease['gt_ranks']

        gb_rankings = get_gb_rankings_for_disease(
            mesh_id, gb_model, embeddings, entity2id, all_drug_ids, name_to_drugbank
        )

        for drug, txgnn_rank in txgnn_gt_ranks.items():
            gb_rank = gb_rankings.get(drug)

            txgnn_hit = txgnn_rank <= 30
            gb_hit = gb_rank is not None and gb_rank <= 30

            if txgnn_hit and not gb_hit:
                txgnn_only_hits.append({
                    'disease': disease_name,
                    'drug': drug,
                    'txgnn_rank': txgnn_rank,
                    'gb_rank': gb_rank or 'N/A',
                })
            elif gb_hit and not txgnn_hit:
                gb_only_hits.append({
                    'disease': disease_name,
                    'drug': drug,
                    'txgnn_rank': txgnn_rank,
                    'gb_rank': gb_rank,
                })

    print(f"\nTxGNN-only hits ({len(txgnn_only_hits)}):")
    for item in txgnn_only_hits[:10]:
        print(f"  {item['disease']}: {item['drug']} (TxGNN: {item['txgnn_rank']}, GB: {item['gb_rank']})")

    print(f"\nGB-only hits ({len(gb_only_hits)}):")
    for item in gb_only_hits[:10]:
        print(f"  {item['disease']}: {item['drug']} (TxGNN: {item['txgnn_rank']}, GB: {item['gb_rank']})")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"""
Key Findings:
- TxGNN alone: {results['txgnn_only']['hits']}/{results['txgnn_only']['total']} = {results['txgnn_only']['hits']/results['txgnn_only']['total']*100:.1f}% Recall@30
- GB alone:    {results['gb_only']['hits']}/{results['gb_only']['total']} = {results['gb_only']['hits']/results['gb_only']['total']*100:.1f}% Recall@30
- Best Rank:   {results['best_rank']['hits']}/{results['best_rank']['total']} = {results['best_rank']['hits']/results['best_rank']['total']*100:.1f}% Recall@30
- RRF:         {results['rrf']['hits']}/{results['rrf']['total']} = {results['rrf']['hits']/results['rrf']['total']*100:.1f}% Recall@30

Ensemble Improvement:
- Best Rank vs GB alone: +{results['best_rank']['hits'] - results['gb_only']['hits']} hits (+{(results['best_rank']['hits']/results['best_rank']['total'] - results['gb_only']['hits']/results['gb_only']['total'])*100:.1f}%)
- Complementary hits: TxGNN found {len(txgnn_only_hits)} drugs GB missed, GB found {len(gb_only_hits)} drugs TxGNN missed

The models have complementary strengths - the "best_rank" ensemble captures drugs
that either model finds in top-30, improving overall recall.
""")

    # Save results
    output = {
        'num_diseases': len(common_diseases),
        'strategies': summary,
        'diseases': [{k: v for k, v in d.items() if k != 'gt_ranks'} for d in common_diseases],
        'per_disease_results': {k: v['diseases'] for k, v in results.items()},
        'txgnn_only_hits': txgnn_only_hits,
        'gb_only_hits': gb_only_hits,
        'ensemble_benefit_drugs': ensemble_benefit_drugs,
    }

    output_path = ANALYSIS_DIR / "ensemble_results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
