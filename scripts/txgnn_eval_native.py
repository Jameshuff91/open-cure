#!/usr/bin/env python3
"""
Evaluate TxGNN models using proper model.predict() with dataframes.
Compares original vs fine-tuned model on Every Cure ground truth.

Usage (on Vast.ai GPU):
    python3 txgnn_eval_native.py
"""

import json
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from txgnn import TxData, TxGNN

def load_ground_truth(filepath='everycure_gt_for_txgnn.json'):
    """Load Every Cure ground truth data."""
    with open(filepath, 'r') as f:
        return json.load(f)

def build_name_to_idx_mappings(tx_data):
    """
    Build mappings from names to TxGNN PER-TYPE indices.

    Two-step mapping for drugs:
    1. name -> DrugBank ID (from node.csv)
    2. DrugBank ID -> per-type idx (from tx_data.df)

    Two-step mapping for diseases:
    1. name -> MONDO ID (from node.csv)
    2. MONDO ID -> per-type idx (from tx_data.df)
    """
    df = tx_data.df

    # Load node.csv for name -> ID mappings
    node_df = pd.read_csv('./data/node.csv', sep='\t', on_bad_lines='skip')

    # === DRUG MAPPING ===
    # Step 1: name -> DrugBank ID
    drug_nodes = node_df[node_df['node_type'] == 'drug']
    drug_name_to_dbid = dict(zip(
        drug_nodes['node_name'].str.lower(),
        drug_nodes['node_id']
    ))

    # Step 2: DrugBank ID -> per-type idx
    drug_df = df[df['x_type'] == 'drug'][['x_id', 'x_idx']].drop_duplicates()
    drug_df2 = df[df['y_type'] == 'drug'][['y_id', 'y_idx']].drop_duplicates()
    drug_df2.columns = ['x_id', 'x_idx']
    drug_df = pd.concat([drug_df, drug_df2]).drop_duplicates()

    dbid_to_idx = dict(zip(drug_df['x_id'], drug_df['x_idx'].astype(int)))

    # Combined: name -> per-type idx
    drug_name_to_idx = {}
    for name, dbid in drug_name_to_dbid.items():
        if dbid in dbid_to_idx:
            drug_name_to_idx[name] = dbid_to_idx[dbid]

    # === DISEASE MAPPING ===
    # Step 1: name -> MONDO ID
    disease_nodes = node_df[node_df['node_type'] == 'disease']
    disease_name_to_mondoid = dict(zip(
        disease_nodes['node_name'].str.lower(),
        disease_nodes['node_id']
    ))

    # Step 2: MONDO ID -> per-type idx
    disease_df = df[df['x_type'] == 'disease'][['x_id', 'x_idx']].drop_duplicates()
    disease_df2 = df[df['y_type'] == 'disease'][['y_id', 'y_idx']].drop_duplicates()
    disease_df2.columns = ['x_id', 'x_idx']
    disease_df = pd.concat([disease_df, disease_df2]).drop_duplicates()

    mondoid_to_idx = dict(zip(disease_df['x_id'], disease_df['x_idx'].astype(int)))

    # Combined: name -> per-type idx
    disease_name_to_idx = {}
    for name, mondoid in disease_name_to_mondoid.items():
        if mondoid in mondoid_to_idx:
            disease_name_to_idx[name] = mondoid_to_idx[mondoid]

    # Get all drug per-type indices
    all_drug_indices = sorted(dbid_to_idx.values())

    return disease_name_to_idx, drug_name_to_idx, all_drug_indices


def evaluate_model(model, gt_data, disease_name_to_idx, drug_name_to_idx, all_drug_indices, model_name="Model"):
    """
    Evaluate model by scoring disease-drug pairs using model.predict().
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")

    hits_at_30 = 0
    total_drugs = 0
    detailed_results = []
    all_ranks = []

    diseases_evaluated = 0
    diseases_skipped = 0

    for disease_name, disease_data in gt_data.items():
        gt_drugs = [d['name'].lower() for d in disease_data['drugs']]
        name_lower = disease_name.lower().strip()

        # Get disease index
        disease_idx = disease_name_to_idx.get(name_lower)
        if disease_idx is None:
            # Try partial match
            for known_name, idx in disease_name_to_idx.items():
                if name_lower in known_name or known_name in name_lower:
                    disease_idx = idx
                    break

        if disease_idx is None:
            diseases_skipped += 1
            continue

        try:
            # Create dataframe with all disease-drug pairs for this disease
            df_pairs = pd.DataFrame({
                'x_idx': [disease_idx] * len(all_drug_indices),
                'relation': ['indication'] * len(all_drug_indices),
                'y_idx': all_drug_indices
            })

            # Get predictions
            with torch.no_grad():
                result = model.predict(df_pairs)

            # model.predict() returns a dict keyed by edge type
            # Get the indication scores
            if isinstance(result, dict):
                # Find the indication key
                scores = None
                for key in result.keys():
                    if 'indication' in str(key).lower():
                        scores = result[key]
                        break
                if scores is None:
                    # Just use the first value
                    scores = list(result.values())[0]
            else:
                scores = result

            # Convert to numpy if tensor
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()

            scores = np.array(scores).flatten()

            # Rank drugs (higher score = better)
            ranked_indices = np.argsort(-scores)

            # Map drug idx -> rank
            idx_to_rank = {all_drug_indices[i]: rank + 1 for rank, i in enumerate(ranked_indices)}

            # Check each GT drug
            for drug_name in gt_drugs:
                drug_idx = drug_name_to_idx.get(drug_name)
                if drug_idx is None:
                    continue

                total_drugs += 1
                rank = idx_to_rank.get(drug_idx, len(all_drug_indices) + 1)
                all_ranks.append(rank)

                if rank <= 30:
                    hits_at_30 += 1

                detailed_results.append({
                    'disease': disease_name,
                    'drug': drug_name,
                    'rank': rank,
                    'hit_at_30': rank <= 30
                })

            diseases_evaluated += 1

            if diseases_evaluated % 20 == 0:
                print(f"  Evaluated {diseases_evaluated} diseases...")

        except Exception as e:
            print(f"  Error on {disease_name}: {e}")
            import traceback
            traceback.print_exc()
            diseases_skipped += 1

    recall_at_30 = hits_at_30 / total_drugs if total_drugs > 0 else 0

    print(f"\nResults for {model_name}:")
    print(f"  Diseases evaluated: {diseases_evaluated}")
    print(f"  Diseases skipped: {diseases_skipped}")
    print(f"  Total GT drugs evaluated: {total_drugs}")
    print(f"  Hits@30: {hits_at_30}")
    print(f"  Recall@30: {recall_at_30:.1%}")

    if all_ranks:
        print(f"  Mean rank of GT drugs: {sum(all_ranks)/len(all_ranks):.0f}")
        sorted_ranks = sorted(all_ranks)
        print(f"  Median rank: {sorted_ranks[len(sorted_ranks)//2]}")
        print(f"  Best rank: {min(all_ranks)}")
        print(f"  Worst rank: {max(all_ranks)}")

    return recall_at_30, detailed_results


def main():
    print("Loading TxGNN data...")
    tx_data = TxData(data_folder_path='./data')
    tx_data.prepare_split(split='random', seed=42)

    print("\nInitializing model...")
    model = TxGNN(data=tx_data, device='cuda:0', weight_bias_track=False)
    model.model_initialize(n_hid=100, n_inp=100, n_out=100, proto=True, proto_num=3)

    # Load ground truth
    gt_data = load_ground_truth('everycure_gt_for_txgnn.json')
    print(f"Loaded ground truth for {len(gt_data)} diseases")

    # Build mappings using per-type indices from tx_data
    print("\nBuilding name to per-type index mappings...")
    disease_name_to_idx, drug_name_to_idx, all_drug_indices = build_name_to_idx_mappings(tx_data)
    print(f"  Disease mappings: {len(disease_name_to_idx)}")
    print(f"  Drug mappings: {len(drug_name_to_idx)}")
    print(f"  Total drugs: {len(all_drug_indices)}")
    print(f"  Drug idx range: {min(all_drug_indices)} - {max(all_drug_indices)}")

    # Verify a few mappings
    print("\nSample drug mappings:")
    for name in ['donepezil', 'methotrexate', 'infliximab']:
        idx = drug_name_to_idx.get(name)
        print(f"  {name}: {idx}")

    results = {}

    # Evaluate original model
    print("\n" + "="*60)
    print("Loading ORIGINAL model (txgnn_500epochs.pt)")
    print("="*60)
    state_dict = torch.load('./txgnn_500epochs.pt', map_location='cuda:0')
    model.model.load_state_dict(state_dict)
    model.model.eval()
    r1, details1 = evaluate_model(model, gt_data, disease_name_to_idx, drug_name_to_idx, all_drug_indices, "ORIGINAL (500 epochs)")
    results['original'] = {'recall_at_30': r1, 'details': details1}

    # Evaluate fine-tuned model
    print("\n" + "="*60)
    print("Loading FINE-TUNED model (txgnn_finetuned_lr3e-05.pt)")
    print("="*60)
    state_dict = torch.load('./txgnn_finetuned_lr3e-05.pt', map_location='cuda:0')
    model.model.load_state_dict(state_dict)
    model.model.eval()
    r2, details2 = evaluate_model(model, gt_data, disease_name_to_idx, drug_name_to_idx, all_drug_indices, "FINE-TUNED (Every Cure)")
    results['finetuned'] = {'recall_at_30': r2, 'details': details2}

    # Summary comparison
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"Original model Recall@30:   {r1:.1%}")
    print(f"Fine-tuned model Recall@30: {r2:.1%}")

    if r1 > 0:
        pct_change = ((r2 - r1) / r1) * 100
        print(f"Relative change: {pct_change:+.1f}%")

    print(f"Absolute change: {(r2-r1)*100:+.1f} percentage points")

    if r2 > r1:
        print("\n>>> Fine-tuning IMPROVED performance!")
    elif r2 < r1:
        print("\n>>> Fine-tuning DEGRADED performance (catastrophic forgetting?)")
    else:
        print("\n>>> No change in performance")

    # Save results
    with open('eval_comparison_results.json', 'w') as f:
        save_results = {
            'original_recall_at_30': r1,
            'finetuned_recall_at_30': r2,
            'improvement': r2 - r1,
            'original_details': details1[:100],
            'finetuned_details': details2[:100]
        }
        json.dump(save_results, f, indent=2)
    print("\nResults saved to eval_comparison_results.json")

    # Show some example comparisons
    print("\n" + "="*60)
    print("EXAMPLE COMPARISONS (first 10 diseases)")
    print("="*60)

    # Group by disease
    orig_by_disease = defaultdict(list)
    ft_by_disease = defaultdict(list)

    for d in details1:
        orig_by_disease[d['disease']].append(d)
    for d in details2:
        ft_by_disease[d['disease']].append(d)

    for disease in list(orig_by_disease.keys())[:10]:
        print(f"\n{disease}:")
        orig_drugs = orig_by_disease.get(disease, [])
        ft_drugs = ft_by_disease.get(disease, [])

        for od in orig_drugs:
            fd = next((f for f in ft_drugs if f['drug'] == od['drug']), None)
            if fd:
                change = od['rank'] - fd['rank']
                symbol = "+" if change > 0 else ("-" if change < 0 else "=")
                print(f"  {od['drug']}: {od['rank']} -> {fd['rank']} ({symbol}{abs(change)})")


if __name__ == '__main__':
    main()
