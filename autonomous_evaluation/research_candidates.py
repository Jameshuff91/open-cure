#!/usr/bin/env python3
"""
Autonomous Ground Truth Enhancement via Research

For each top model prediction NOT in Every Cure:
1. Search for evidence that it treats the disease
2. Classify as: CONFIRMED (add to GT), EXPERIMENTAL (in trials), or NOVEL (no evidence)
3. Save enhanced ground truth with evidence citations
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

STATE_FILE = Path('autonomous_evaluation/.research_state.json')
ENHANCED_GT_FILE = Path('autonomous_evaluation/enhanced_ground_truth.json')

def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        'current_disease': None,
        'candidates_researched': {},  # drug_id -> classification
        'enhanced_gt': {},  # disease -> list of {drug_id, name, classification, evidence}
        'research_log': []
    }

def save_state(state):
    STATE_FILE.parent.mkdir(exist_ok=True)
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

    # Also save enhanced GT separately for easy use
    with open(ENHANCED_GT_FILE, 'w') as f:
        json.dump(state['enhanced_gt'], f, indent=2)

def get_next_candidate(state, disease_name):
    """Get the next drug candidate that needs research"""
    candidates_file = Path(f'autonomous_evaluation/{disease_name.lower().replace(" ", "_")}_candidates.json')
    if not candidates_file.exists():
        return None

    with open(candidates_file) as f:
        data = json.load(f)

    # Combine likely drugs and needs_research
    all_candidates = data.get('likely_hiv_drugs', []) + data.get('needs_research', [])

    for candidate in all_candidates:
        drug_id = candidate['drug_id']
        if drug_id not in state['candidates_researched']:
            return candidate

    return None  # All done

def add_research_result(state, disease_name, drug_info, classification, evidence):
    """Add a research result to the state"""
    drug_id = drug_info['drug_id']

    # Record the classification
    state['candidates_researched'][drug_id] = {
        'classification': classification,
        'evidence': evidence,
        'timestamp': datetime.now().isoformat()
    }

    # Add to enhanced GT if confirmed or experimental
    if classification in ['CONFIRMED', 'EXPERIMENTAL']:
        if disease_name not in state['enhanced_gt']:
            state['enhanced_gt'][disease_name] = []

        state['enhanced_gt'][disease_name].append({
            'drug_id': drug_id,
            'db_id': drug_info['db_id'],
            'name': drug_info['name'],
            'classification': classification,
            'evidence': evidence,
            'model_score': drug_info['score']
        })

    # Log
    state['research_log'].append({
        'disease': disease_name,
        'drug': drug_info['name'],
        'classification': classification,
        'timestamp': datetime.now().isoformat()
    })

    save_state(state)
    return state

def print_status(state):
    """Print current research status"""
    print("\n" + "="*60)
    print("GROUND TRUTH ENHANCEMENT STATUS")
    print("="*60)

    total_researched = len(state['candidates_researched'])
    confirmed = sum(1 for v in state['candidates_researched'].values() if v['classification'] == 'CONFIRMED')
    experimental = sum(1 for v in state['candidates_researched'].values() if v['classification'] == 'EXPERIMENTAL')
    novel = sum(1 for v in state['candidates_researched'].values() if v['classification'] == 'NOVEL')

    print(f"\nTotal drugs researched: {total_researched}")
    print(f"  CONFIRMED (known treatments): {confirmed}")
    print(f"  EXPERIMENTAL (in trials): {experimental}")
    print(f"  NOVEL (no evidence): {novel}")

    print("\nEnhanced ground truth by disease:")
    for disease, drugs in state['enhanced_gt'].items():
        print(f"  {disease}: {len(drugs)} drugs added")

    print("="*60)

if __name__ == '__main__':
    state = load_state()
    print_status(state)

    # Check for HIV candidates
    hiv_candidates = Path('autonomous_evaluation/hiv_candidates.json')
    if hiv_candidates.exists():
        next_candidate = get_next_candidate(state, 'HIV infection')
        if next_candidate:
            print(f"\nNext candidate to research: {next_candidate['name']} ({next_candidate['db_id']})")
            print(f"Model score: {next_candidate['score']:.4f}")
        else:
            print("\nAll HIV candidates have been researched!")
