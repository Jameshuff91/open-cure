#!/usr/bin/env python3
"""
Run TxGNN predictions for Open-Cure target diseases.
This script is meant to run on a GPU instance with TxGNN installed.
"""

import json
import os
from pathlib import Path

# Target diseases from our evaluation
TARGET_DISEASES = [
    "asthma",
    "atrial fibrillation",
    "breast cancer",
    "chronic obstructive pulmonary disease",
    "colorectal cancer",
    "epilepsy",
    "HIV infection",
    "heart failure",
    "hepatitis C",
    "hypertension",
    "lung cancer",
    "multiple sclerosis",
    "obesity",
    "osteoporosis",
    "psoriasis",
    "rheumatoid arthritis",
    "tuberculosis",
    "type 2 diabetes mellitus",
]

def main():
    print("=" * 70)
    print("TxGNN Drug Repurposing Predictions")
    print("=" * 70)

    # Import TxGNN
    print("\nLoading TxGNN...")
    from txgnn import TxData, TxGNN, TxEval

    # Initialize data
    print("Initializing TxGNN data...")
    tx_data = TxData(data_folder_path='./data')
    tx_data.prepare_split(split='random', seed=42)

    # Initialize model
    print("Initializing TxGNN model...")
    model = TxGNN(
        data=tx_data,
        weight_bias_track=False,
        proj_name='TxGNN',
        exp_name='open_cure_eval',
        device='cuda:0'
    )

    # Try to load pretrained model or initialize new one
    pretrained_path = './model_ckpt'
    if os.path.exists(pretrained_path):
        print(f"Loading pretrained model from {pretrained_path}...")
        model.load_pretrained(pretrained_path)
    else:
        print("No pretrained model found, initializing new model...")
        model.model_initialize(
            n_hid=100,
            n_inp=100,
            n_out=100,
            proto=True,
            proto_num=3,
            attention=False,
            sim_measure='all_nodes_profile',
            agg_measure='rarity'
        )

        # Fine-tune the model
        print("Fine-tuning model (this may take a while)...")
        model.finetune(
            n_epoch=500,
            learning_rate=5e-4,
            train_print_per_n=50,
            valid_per_n=100
        )

    # Get predictions for each target disease
    print("\n" + "=" * 70)
    print("Getting predictions for target diseases...")
    print("=" * 70)

    results = {}
    evaluator = TxEval(model)

    # Get all disease names in the knowledge graph
    disease_names = list(tx_data.disease_idx_to_name.values()) if hasattr(tx_data, 'disease_idx_to_name') else []

    print(f"\nTxGNN has {len(disease_names)} diseases in its knowledge graph")

    # Match our target diseases to TxGNN disease names
    for target in TARGET_DISEASES:
        print(f"\nSearching for: {target}")

        # Find matching disease
        matches = [d for d in disease_names if target.lower() in d.lower()]

        if matches:
            print(f"  Found matches: {matches[:5]}")

            for match in matches[:1]:  # Take first match
                try:
                    # Get predictions for this disease
                    preds = evaluator.predict_drug_for_disease(match)

                    if preds is not None and len(preds) > 0:
                        results[target] = {
                            'txgnn_disease_name': match,
                            'top_30_drugs': preds[:30] if len(preds) >= 30 else preds,
                            'top_50_drugs': preds[:50] if len(preds) >= 50 else preds,
                            'top_100_drugs': preds[:100] if len(preds) >= 100 else preds,
                            'total_predictions': len(preds)
                        }
                        print(f"  Got {len(preds)} predictions")
                    else:
                        print(f"  No predictions returned")

                except Exception as e:
                    print(f"  Error: {e}")
        else:
            print(f"  No match found in TxGNN knowledge graph")

    # Save results
    output_path = './txgnn_predictions.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n" + "=" * 70)
    print(f"Results saved to {output_path}")
    print(f"Diseases with predictions: {len(results)}/{len(TARGET_DISEASES)}")
    print("=" * 70)

    # Print summary
    print("\nSummary:")
    for disease, data in results.items():
        n_preds = data.get('total_predictions', 0)
        print(f"  {disease}: {n_preds} predictions")

if __name__ == '__main__':
    main()
