#!/usr/bin/env python3
"""Fix misplaced AF drug entries in research state."""

import json
from pathlib import Path

def main():
    base_dir = Path(__file__).parent
    state_file = base_dir / ".research_state.json"

    # Load current state
    with open(state_file, 'r') as f:
        state = json.load(f)

    # Find misplaced entries in enhanced_gt
    drug_ids_to_move = []
    for key in state['enhanced_gt'].keys():
        if key.startswith('drkg:Compound::'):
            drug_ids_to_move.append(key)

    print(f'Found {len(drug_ids_to_move)} misplaced drug entries in enhanced_gt:')

    # Move them to candidates_researched
    for drug_id in drug_ids_to_move:
        entry = state['enhanced_gt'].pop(drug_id)
        print(f'  Moving {drug_id}: {entry["classification"]}')

        # Remove _AF suffix if present
        clean_id = drug_id.replace('_AF', '')

        # Add to candidates_researched
        state['candidates_researched'][clean_id] = entry

    # Save updated state
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)

    print(f'\n✓ Fixed research state structure')

    # Verify AF candidates
    with open(base_dir / "atrial_fibrillation_candidates.json", 'r') as f:
        candidates_data = json.load(f)
        candidate_ids = [c['drug_id'] for c in candidates_data['candidates']]

    researched_count = 0
    for cand_id in candidate_ids:
        if cand_id in state['candidates_researched']:
            researched_count += 1

    print(f'\nAF Candidates: {len(candidate_ids)}')
    print(f'Researched: {researched_count}')
    print(f'Remaining: {len(candidate_ids) - researched_count}')

if __name__ == "__main__":
    main()
