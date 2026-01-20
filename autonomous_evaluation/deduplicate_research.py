#!/usr/bin/env python3
"""Remove duplicate entries from research state."""

import json
from pathlib import Path

def main():
    base_dir = Path(__file__).parent
    state_file = base_dir / ".research_state.json"

    # Load current state
    with open(state_file, 'r') as f:
        state = json.load(f)

    # Get research log
    research_log = state["research_log"]

    # Track unique entries by (disease, drug) tuple
    seen = set()
    deduped_log = []

    for entry in research_log:
        key = (entry["disease"], entry["drug"])
        if key not in seen:
            seen.add(key)
            deduped_log.append(entry)

    removed_count = len(research_log) - len(deduped_log)
    print(f"Removed {removed_count} duplicate entries")
    print(f"Kept {len(deduped_log)} unique entries")

    # Update state
    state["research_log"] = deduped_log

    # Save updated state
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)

    # Count AF drugs
    af_drugs = [log for log in deduped_log if log["disease"] == "Atrial fibrillation"]
    confirmed = len([d for d in af_drugs if d["classification"] == "CONFIRMED"])
    experimental = len([d for d in af_drugs if d["classification"] == "EXPERIMENTAL"])
    novel = len([d for d in af_drugs if d["classification"] == "NOVEL"])

    print(f"\nAtrial fibrillation drugs researched: {len(af_drugs)}")
    print(f"  - CONFIRMED: {confirmed}")
    print(f"  - EXPERIMENTAL: {experimental}")
    print(f"  - NOVEL: {novel}")

    # Check consecutive novel count
    consecutive_novel = 0
    for log in reversed(af_drugs):
        if log["classification"] == "NOVEL":
            consecutive_novel += 1
        else:
            break

    print(f"  Consecutive NOVEL at end: {consecutive_novel}")

if __name__ == "__main__":
    main()
