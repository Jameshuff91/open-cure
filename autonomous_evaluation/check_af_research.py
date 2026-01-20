#!/usr/bin/env python3
"""Check which AF drugs have been researched."""

import json
from pathlib import Path

def main():
    base_dir = Path(__file__).parent

    # Load candidates
    with open(base_dir / "atrial_fibrillation_candidates.json", 'r') as f:
        candidates_data = json.load(f)
        candidates = {c["drug_id"]: c["name"] for c in candidates_data["candidates"]}

    # Load research state
    with open(base_dir / ".research_state.json", 'r') as f:
        state = json.load(f)

    # Get researched candidates
    researched = state["candidates_researched"]

    # Find AF drugs in research log
    af_log = [log for log in state["research_log"] if log["disease"] == "Atrial fibrillation"]

    print(f"Total candidates: {len(candidates)}")
    print(f"Total researched (in candidates_researched): {len([k for k in researched.keys() if k in candidates])}")
    print(f"Total in research log: {len(af_log)}")

    # Check which candidates are not researched
    not_researched = []
    for drug_id, name in candidates.items():
        if drug_id not in researched:
            not_researched.append((drug_id, name))

    if not_researched:
        print(f"\n{len(not_researched)} candidates NOT YET RESEARCHED:")
        for drug_id, name in not_researched:
            print(f"  {drug_id}: {name}")
    else:
        print("\n✓ All candidates have been researched!")

    # Show AF research log entries
    print(f"\n{len(af_log)} AF drugs in research log:")
    for i, log in enumerate(af_log, 1):
        print(f"  {i}. [{log['classification']}] {log['drug']}")

if __name__ == "__main__":
    main()
