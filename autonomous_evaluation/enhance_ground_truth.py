#!/usr/bin/env python3
"""
Autonomous Ground Truth Enhancement

Strategy:
1. Start with Every Cure as high-quality baseline
2. For each disease, identify model predictions NOT in Every Cure
3. Research each candidate to determine if it's:
   - A known drug (add to GT)
   - In clinical trials (add with lower confidence)
   - No evidence (true novel candidate)
4. Save enhanced ground truth with evidence
"""

import json
import pandas as pd
import os
from pathlib import Path

# Config
DISEASES_TO_ENHANCE = [
    ('HIV infection', 'drkg:Disease::MESH:D015658'),
    ('Osteoporosis', 'drkg:Disease::MESH:D010024'),
    ('Epilepsy', 'drkg:Disease::MESH:D004827'),
]

STATE_FILE = Path('autonomous_evaluation/.enhancement_state.json')

def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        'diseases_processed': [],
        'enhanced_gt': {},
        'research_log': []
    }

def save_state(state):
    STATE_FILE.parent.mkdir(exist_ok=True)
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

def get_top_predictions(disease_mesh_id, n=50):
    """Get top N model predictions for a disease"""
    import torch
    import numpy as np
    import pickle
    
    transe = torch.load('models/transe.pt', map_location='cpu', weights_only=False)
    entity2id = transe['entity2id']
    entity_emb = transe['model_state_dict']['entity_embeddings.weight'].numpy()
    
    with open('data/reference/drugbank_lookup.json') as f:
        drugbank = json.load(f)
    
    with open('models/drug_repurposing_gb.pkl', 'rb') as f:
        gb = pickle.load(f)['model']
    
    disease_emb = entity_emb[entity2id[disease_mesh_id]]
    
    scores = []
    for drug_id in [e for e in entity2id if 'drkg:Compound::DB' in e]:
        drug_emb = entity_emb[entity2id[drug_id]]
        feat = np.concatenate([np.concatenate([drug_emb, disease_emb]),
                              drug_emb * disease_emb,
                              drug_emb - disease_emb]).reshape(1, -1)
        prob = gb.predict_proba(feat)[0, 1]
        db_id = drug_id.split('::')[1]
        name = drugbank.get(db_id, db_id)
        scores.append({'drug_id': drug_id, 'db_id': db_id, 'name': name, 'score': prob})
    
    scores.sort(key=lambda x: -x['score'])
    return scores[:n]

if __name__ == '__main__':
    state = load_state()
    print("Ground Truth Enhancement System")
    print("="*50)
    print(f"Diseases to process: {len(DISEASES_TO_ENHANCE)}")
    print(f"Already processed: {len(state['diseases_processed'])}")
    save_state(state)
    print("\nRun with Claude to enhance each disease's ground truth.")
