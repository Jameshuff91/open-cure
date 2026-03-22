"""h901: Integrate new MeSH mappings into the production pipeline.

Merges Phase 1 (NLM API) and Phase 2 (broader matching) results into
mesh_mappings_from_agents.json, then verifies coverage improvement.
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent
REF_DIR = DATA_DIR / "data" / "reference"
ANALYSIS_DIR = DATA_DIR / "data" / "analysis"


def main() -> None:
    print("h901: Integrating new MeSH mappings")
    print("=" * 60)

    # Load existing mappings
    mesh_path = REF_DIR / "mesh_mappings_from_agents.json"
    with open(mesh_path) as f:
        mesh_data = json.load(f)

    # Count existing
    existing_count = 0
    for batch in mesh_data.values():
        if isinstance(batch, dict):
            existing_count += sum(1 for v in batch.values() if v)
    print(f"Existing mappings: {existing_count}")

    # Load Phase 1 results (NLM API)
    with open(ANALYSIS_DIR / "h901_resolved_mappings.json") as f:
        phase1 = json.load(f)
    phase1_mappings = phase1.get('mappings', {})
    print(f"Phase 1 (NLM API) mappings: {len(phase1_mappings)}")

    # Load Phase 2 results (broader matching)
    with open(ANALYSIS_DIR / "h901_broader_matches.json") as f:
        phase2 = json.load(f)
    phase2_mappings = phase2.get('mappings', {})
    print(f"Phase 2 (broader) mappings: {len(phase2_mappings)}")

    # Build new batch entries
    # Phase 1: high/medium confidence → add directly
    new_batch_phase1: dict[str, str] = {}
    for disease_name, info in phase1_mappings.items():
        mesh_id = info['mesh_id']
        new_batch_phase1[disease_name.lower()] = mesh_id

    # Phase 2: medium → add, approximate → add with tracking
    new_batch_phase2_medium: dict[str, str] = {}
    new_batch_phase2_approx: dict[str, str] = {}
    for disease_name, info in phase2_mappings.items():
        mesh_id = info['mesh_id']
        if info['confidence'] == 'medium':
            new_batch_phase2_medium[disease_name.lower()] = mesh_id
        else:
            new_batch_phase2_approx[disease_name.lower()] = mesh_id

    # Check for conflicts with existing mappings
    existing_names: dict[str, str] = {}
    for batch in mesh_data.values():
        if isinstance(batch, dict):
            for name, mid in batch.items():
                if mid:
                    existing_names[name.lower()] = str(mid)

    conflicts = 0
    for name, mesh_id in {**new_batch_phase1, **new_batch_phase2_medium, **new_batch_phase2_approx}.items():
        if name in existing_names and existing_names[name] != mesh_id:
            conflicts += 1
    print(f"Conflicts with existing: {conflicts} (will keep existing)")

    # Skip conflicts — existing mappings take priority
    for name in existing_names:
        new_batch_phase1.pop(name, None)
        new_batch_phase2_medium.pop(name, None)
        new_batch_phase2_approx.pop(name, None)

    # Add new batches to mesh_data
    mesh_data['h901_nlm_api'] = new_batch_phase1
    mesh_data['h901_broader_medium'] = new_batch_phase2_medium
    mesh_data['h901_broader_approximate'] = new_batch_phase2_approx

    # Save updated mappings
    with open(mesh_path, 'w') as f:
        json.dump(mesh_data, f, indent=2)

    # Verify
    total_new = len(new_batch_phase1) + len(new_batch_phase2_medium) + len(new_batch_phase2_approx)
    print(f"\n{'=' * 60}")
    print(f"INTEGRATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Added to mesh_mappings_from_agents.json:")
    print(f"  h901_nlm_api: {len(new_batch_phase1)} mappings")
    print(f"  h901_broader_medium: {len(new_batch_phase2_medium)} mappings")
    print(f"  h901_broader_approximate: {len(new_batch_phase2_approx)} mappings")
    print(f"  Total new: {total_new}")
    print(f"  Total in file: {existing_count + total_new}")

    # Verify by reloading
    with open(mesh_path) as f:
        verify = json.load(f)
    verify_count = 0
    for batch in verify.values():
        if isinstance(batch, dict):
            verify_count += sum(1 for v in batch.values() if v)
    print(f"  Verified total: {verify_count}")


if __name__ == '__main__':
    main()
