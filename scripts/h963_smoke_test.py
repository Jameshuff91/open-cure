"""h963 smoke test: verify predict(disease_id) fast-path returns identical
predictions to predict(disease_name) on diseases where name-resolution works,
and that it recovers diseases where name-resolution silently failed pre-h952.
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from production_predictor import DrugRepurposingPredictor  # noqa: E402


def predictions_equal(a, b):
    if len(a.predictions) != len(b.predictions):
        return False, f"pred count differs: {len(a.predictions)} vs {len(b.predictions)}"
    for pa, pb in zip(a.predictions, b.predictions):
        if pa.drug_id != pb.drug_id:
            return False, f"drug_id mismatch at rank {pa.rank}: {pa.drug_id} vs {pb.drug_id}"
        if pa.confidence_tier != pb.confidence_tier:
            return False, f"tier mismatch at rank {pa.rank}"
        if pa.rank != pb.rank:
            return False, f"rank mismatch: {pa.rank} vs {pb.rank}"
    return True, "ok"


def main():
    print("Loading predictor...")
    predictor = DrugRepurposingPredictor()

    # Equivalence test as used by h393 evaluator:
    # pre-h963 evaluator passed canonical disease_name from disease_names[did].
    # post-h963 evaluator passes disease_id directly. id-path internally sets
    # disease_name = disease_names[did], so the downstream pipeline sees the
    # same disease_name string in both cases -> predictions must match.
    eligible = [
        did for did in predictor.disease_names
        if did in predictor.embeddings and predictor.disease_names[did]
    ]

    rng = random.Random(42)
    rng.shuffle(eligible)
    sample = eligible[:25]

    print(f"Testing canonical-name path vs id-path equivalence on {len(sample)} diseases...")
    mismatches = []
    for did in sample:
        canonical_name = predictor.disease_names[did]
        res_name = predictor.predict(canonical_name, top_n=30, include_filtered=True)
        res_id = predictor.predict(did, top_n=30, include_filtered=True)
        ok, msg = predictions_equal(res_name, res_id)
        if not ok:
            mismatches.append((did, canonical_name, msg))

    if mismatches:
        print(f"FAIL — {len(mismatches)} mismatches:")
        for did, name, msg in mismatches[:5]:
            print(f"  {did} ({name}): {msg}")
        sys.exit(1)
    print(f"PASS — all {len(sample)} diseases match between canonical-name path and id-path.")

    # Verify fast-path recovers diseases that name-resolution cannot handle.
    # Pick diseases whose canonical disease_names are NOT in mesh_mappings
    # (these are exactly the h952-class failures).
    name_set = set(predictor.mesh_mappings.keys())
    id_only = [
        did for did, n in predictor.disease_names.items()
        if n and n.lower() not in name_set and did in predictor.embeddings
    ]
    rng.shuffle(id_only)
    test_ids = id_only[:10]
    print(f"\nTesting id-path recovery on {len(test_ids)} name-unresolvable diseases...")

    id_path_ok = 0
    name_path_ok = 0
    for did in test_ids:
        disease_name = predictor.disease_names[did]
        res_id = predictor.predict(did, top_n=30, include_filtered=True)
        res_name = predictor.predict(disease_name, top_n=30, include_filtered=True)
        if res_id.predictions:
            id_path_ok += 1
        if res_name.predictions:
            name_path_ok += 1

    print(f"  id-path produced predictions: {id_path_ok}/{len(test_ids)}")
    print(f"  name-path produced predictions: {name_path_ok}/{len(test_ids)}")
    print(f"  recovered diseases: {id_path_ok - name_path_ok}")

    # Summary
    print("\n=== h963 SMOKE TEST ===")
    print(f"Equivalence: {len(sample) - len(mismatches)}/{len(sample)} match")
    print(f"Recovery: id-path {id_path_ok}, name-path {name_path_ok}")
    if id_path_ok < name_path_ok:
        print("WARN: id-path underperformed name-path — investigate")
        sys.exit(2)


if __name__ == "__main__":
    main()
