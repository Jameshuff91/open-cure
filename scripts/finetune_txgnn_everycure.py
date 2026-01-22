#!/usr/bin/env python3
"""
Fine-tune TxGNN on Every Cure Ground Truth Data
================================================

This script adds Every Cure ground truth drug-disease pairs as training signal
to improve TxGNN's performance on clinically-validated treatments.

Strategy:
1. Load pre-trained TxGNN model (500 epochs)
2. Add Every Cure indication edges to the knowledge graph
3. Fine-tune for additional epochs with our ground truth
4. Evaluate on held-out portion of our ground truth

Usage (on Vast.ai GPU):
    python finetune_txgnn_everycure.py --epochs 100 --lr 1e-4
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import torch


def load_everycure_ground_truth(gt_path: str) -> pd.DataFrame:
    """Load Every Cure ground truth drug-disease pairs."""
    # Load the expanded ground truth
    with open(gt_path) as f:
        gt_data = json.load(f)

    pairs = []
    for disease_id, info in gt_data.items():
        disease_name = info.get('disease_name', disease_id)
        for drug in info.get('drugs', []):
            drug_name = drug.get('name', drug) if isinstance(drug, dict) else drug
            pairs.append({
                'disease_id': disease_id,
                'disease_name': disease_name.lower(),
                'drug_name': drug_name.lower(),
            })

    return pd.DataFrame(pairs)


def map_everycure_to_txgnn(gt_df: pd.DataFrame, tx_data) -> pd.DataFrame:
    """Map Every Cure names to TxGNN node indices."""

    # Get TxGNN disease and drug mappings
    df = tx_data.df

    # Disease name to internal index
    disease_rows = df[df['x_type'] == 'disease'][['x_id', 'x_idx']].drop_duplicates()

    # Load node.csv for name mappings
    nodes_df = pd.read_csv('./data/node.csv', sep='\t')
    disease_nodes = nodes_df[nodes_df['node_type'] == 'disease']
    drug_nodes = nodes_df[nodes_df['node_type'] == 'drug']

    # Create name -> node_id mapping
    disease_name_to_nodeid = {
        row['node_name'].lower(): row['node_id']
        for _, row in disease_nodes.iterrows()
    }
    drug_name_to_nodeid = {
        row['node_name'].lower(): row['node_id']
        for _, row in drug_nodes.iterrows()
    }

    # Create node_id -> internal_idx mapping
    disease_id_to_idx = {}
    for _, row in disease_rows.iterrows():
        xid = str(row['x_id'])
        # Handle .0 suffix
        for variant in [xid, xid.replace('.0', ''), f"{xid}.0"]:
            disease_id_to_idx[variant] = int(row['x_idx'])

    drug_rows = df[df['x_type'] == 'drug'][['x_id', 'x_idx']].drop_duplicates()
    drug_id_to_idx = {}
    for _, row in drug_rows.iterrows():
        xid = str(row['x_id'])
        for variant in [xid, xid.replace('.0', ''), f"{xid}.0"]:
            drug_id_to_idx[variant] = int(row['x_idx'])

    # Map Every Cure pairs to TxGNN indices
    mapped_pairs = []
    unmapped_diseases = set()
    unmapped_drugs = set()

    for _, row in gt_df.iterrows():
        disease_name = row['disease_name']
        drug_name = row['drug_name']

        # Find disease index
        disease_nodeid = disease_name_to_nodeid.get(disease_name)
        disease_idx = None
        if disease_nodeid:
            disease_idx = disease_id_to_idx.get(str(disease_nodeid))

        # Find drug index
        drug_nodeid = drug_name_to_nodeid.get(drug_name)
        drug_idx = None
        if drug_nodeid:
            drug_idx = drug_id_to_idx.get(str(drug_nodeid))

        if disease_idx is not None and drug_idx is not None:
            mapped_pairs.append({
                'disease_name': disease_name,
                'drug_name': drug_name,
                'disease_idx': disease_idx,
                'drug_idx': drug_idx,
            })
        else:
            if disease_idx is None:
                unmapped_diseases.add(disease_name)
            if drug_idx is None:
                unmapped_drugs.add(drug_name)

    print(f"Mapped {len(mapped_pairs)} drug-disease pairs to TxGNN indices")
    print(f"Unmapped diseases: {len(unmapped_diseases)}")
    print(f"Unmapped drugs: {len(unmapped_drugs)}")

    return pd.DataFrame(mapped_pairs)


def add_indication_edges(tx_data, mapped_pairs: pd.DataFrame):
    """Add Every Cure indication edges to TxGNN's knowledge graph."""

    # Create new edges dataframe
    new_edges = []
    for _, row in mapped_pairs.iterrows():
        new_edges.append({
            'x_type': 'drug',
            'x_id': row['drug_idx'],
            'x_idx': row['drug_idx'],
            'relation': 'indication',
            'y_type': 'disease',
            'y_id': row['disease_idx'],
            'y_idx': row['disease_idx'],
        })

    new_edges_df = pd.DataFrame(new_edges)

    # Add to training data
    # Note: This modifies tx_data.df which feeds into training
    original_len = len(tx_data.df)
    tx_data.df = pd.concat([tx_data.df, new_edges_df], ignore_index=True)
    print(f"Added {len(new_edges_df)} indication edges (total: {len(tx_data.df)}, was {original_len})")

    return tx_data


def main():
    parser = argparse.ArgumentParser(description='Fine-tune TxGNN on Every Cure data')
    parser.add_argument('--epochs', type=int, default=100, help='Number of fine-tuning epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--gt-path', type=str, default='./everycure_gt.json',
                        help='Path to Every Cure ground truth JSON')
    parser.add_argument('--pretrained', type=str, default='./txgnn_500epochs.pt',
                        help='Path to pre-trained TxGNN weights')
    parser.add_argument('--output', type=str, default='./txgnn_finetuned.pt',
                        help='Output path for fine-tuned model')
    args = parser.parse_args()

    print("=" * 70)
    print("TxGNN Fine-tuning on Every Cure Ground Truth")
    print("=" * 70)
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Ground truth: {args.gt_path}")
    print(f"Pre-trained weights: {args.pretrained}")

    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This script requires a GPU.")
        sys.exit(1)

    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Import TxGNN (after CUDA check)
    from txgnn import TxData, TxGNN

    # Load TxGNN data
    print("\nLoading TxGNN data...")
    tx_data = TxData(data_folder_path='./data')
    tx_data.prepare_split(split='random', seed=42)

    # Load Every Cure ground truth
    print("\nLoading Every Cure ground truth...")
    gt_df = load_everycure_ground_truth(args.gt_path)
    print(f"Loaded {len(gt_df)} drug-disease pairs")

    # Map to TxGNN indices
    print("\nMapping to TxGNN indices...")
    mapped_pairs = map_everycure_to_txgnn(gt_df, tx_data)

    if len(mapped_pairs) == 0:
        print("ERROR: No pairs could be mapped. Check disease/drug name formats.")
        sys.exit(1)

    # Split into train (80%) and validation (20%)
    np.random.seed(42)
    indices = np.random.permutation(len(mapped_pairs))
    split_idx = int(0.8 * len(indices))
    train_pairs = mapped_pairs.iloc[indices[:split_idx]]
    val_pairs = mapped_pairs.iloc[indices[split_idx:]]
    print(f"Train pairs: {len(train_pairs)}, Validation pairs: {len(val_pairs)}")

    # Add training edges to knowledge graph
    print("\nAdding indication edges to knowledge graph...")
    tx_data = add_indication_edges(tx_data, train_pairs)

    # Initialize model
    print("\nInitializing TxGNN model...")
    model = TxGNN(data=tx_data, device='cuda:0', weight_bias_track=False)
    model.model_initialize(n_hid=100, n_inp=100, n_out=100, proto=True, proto_num=3)

    # Load pre-trained weights
    if os.path.exists(args.pretrained):
        print(f"Loading pre-trained weights from {args.pretrained}...")
        state_dict = torch.load(args.pretrained, map_location='cuda:0')
        model.model.load_state_dict(state_dict)
        print("Pre-trained weights loaded successfully")
    else:
        print(f"WARNING: Pre-trained weights not found at {args.pretrained}")
        print("Training from scratch...")

    # Fine-tune
    print(f"\nFine-tuning for {args.epochs} epochs...")
    start_time = datetime.now()

    model.finetune(
        n_epoch=args.epochs,
        learning_rate=args.lr,
        train_print_per_n=10,
    )

    elapsed = datetime.now() - start_time
    print(f"Fine-tuning completed in {elapsed}")

    # Save fine-tuned model
    print(f"\nSaving fine-tuned model to {args.output}...")
    torch.save(model.model.state_dict(), args.output)

    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    model.model.eval()

    hits_at_30 = 0
    total = 0

    for _, row in val_pairs.iterrows():
        disease_idx = row['disease_idx']
        drug_idx = row['drug_idx']

        # Create prediction dataframe for all drugs
        all_drug_indices = tx_data.df[tx_data.df['x_type'] == 'drug']['x_idx'].unique()
        pred_data = [{'x_idx': int(d), 'relation': 'indication', 'y_idx': int(disease_idx)}
                     for d in all_drug_indices]
        pred_df = pd.DataFrame(pred_data)

        with torch.no_grad():
            scores = model.predict(pred_df)
            if ('drug', 'indication', 'disease') in scores:
                indication_scores = scores[('drug', 'indication', 'disease')].cpu().numpy()

                # Find rank of GT drug
                drug_score_idx = list(all_drug_indices).index(drug_idx)
                drug_score = indication_scores[drug_score_idx]
                rank = (indication_scores > drug_score).sum() + 1

                if rank <= 30:
                    hits_at_30 += 1
                total += 1

    val_recall = hits_at_30 / total if total > 0 else 0
    print(f"Validation Recall@30: {val_recall*100:.1f}% ({hits_at_30}/{total})")

    # Save results
    results = {
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'train_pairs': len(train_pairs),
        'val_pairs': len(val_pairs),
        'val_recall_at_30': val_recall,
        'val_hits': hits_at_30,
        'training_time': str(elapsed),
    }

    results_path = args.output.replace('.pt', '_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    print("\n" + "=" * 70)
    print("Fine-tuning complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
