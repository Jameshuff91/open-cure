#!/usr/bin/env python3
"""h1002: Does neighbor-augmented bio_gt rescue the 16 unique-target hits?

h995b LOO de-biased analysis found that 16 of 32 autoimmune biologic hits
have target_match_loo=False — they are target-unique in their disease's
bio_gt (first-in-family mAbs). These are exactly the hits the h995 filter
cannot keep.

Hypothesis: if we union the bio_gt of the disease's top-3 kNN neighbor
diseases (instead of using just the disease's own bio_gt), the extended
target universe may cover these unique-target drugs. If so, a neighbor-
augmented filter could retain >=12/16 of the current flip-to-False hits
while still dropping genuine mispicks.

Specificity check: we also measure what fraction of the 199
target_match_loo=False MISSES are rescued (i.e. incorrectly flipped
back to True) by the neighbor augmentation. If the augmentation rescues
>80% of misses as well, the filter becomes permissive and loses its
discriminating power.

Ship gate for follow-up h1003:
    Hit rescue rate >= 75%  AND  miss rescue rate <= 40%
    (i.e. at least 2:1 selective lift on the LOO=False bucket)
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from production_predictor import DrugRepurposingPredictor  # noqa: E402
from h939_biologic_target_overlap_audit import is_biologic  # noqa: E402
from h995_autoimmune_biologic_family_audit import load_expanded_gt  # noqa: E402


K_NEIGHBORS = 3


def main():
    print("=" * 78)
    print("h1002: Neighbor-augmented target_match rescue of unique-target hits")
    print("=" * 78)

    with open(ROOT / "data/analysis/h995_slot_records.json") as f:
        records = json.load(f)
    print(f"Loaded {len(records)} slot records from h995")

    predictor = DrugRepurposingPredictor()
    expanded_gt = load_expanded_gt(
        predictor.reference_dir / "expanded_ground_truth.json"
    )
    biologic_pool = {
        d for d in predictor.drug_id_to_name
        if is_biologic(predictor.drug_id_to_name.get(d))
    }
    print(f"Biologic pool: {len(biologic_pool)}")

    # For each unique disease in records, compute top-K kNN neighbors
    # from TRAIN-diseases only for the relevant seed. Unfortunately h995
    # slot records don't carry the train-set membership per seed, so we
    # reconstruct by re-computing: for each seed, split and record kNN.
    from h393_holdout_tier_validation import split_diseases

    all_diseases = [
        d for d in predictor.ground_truth if d in predictor.embeddings
    ]

    # Group records by (seed, disease_id)
    by_seed_dis = defaultdict(list)
    for r in records:
        by_seed_dis[(r["seed"], r["disease_id"])].append(r)

    seeds = sorted({r["seed"] for r in records})
    print(f"Seeds in records: {seeds}")

    # For each seed, compute kNN structure (train-only)
    neighbor_cache: dict = {}  # (seed, disease_id) -> [train_neighbor_disease_ids]

    for seed in seeds:
        train_ids, _ = split_diseases(all_diseases, seed)
        train_ids = [d for d in train_ids if d in predictor.embeddings]
        train_embs = np.array(
            [predictor.embeddings[d] for d in train_ids], dtype=np.float32
        )

        # For every disease in records for this seed, query top-K neighbors
        holdout_ids = {
            dis for (s, dis) in by_seed_dis if s == seed
        }
        for dis in holdout_ids:
            if dis not in predictor.embeddings:
                continue
            q = predictor.embeddings[dis].astype(np.float32)
            # Euclidean distance, take top-K
            diffs = train_embs - q
            dists = np.einsum("ij,ij->i", diffs, diffs)
            idx = np.argsort(dists)[:K_NEIGHBORS]
            neighbors = [train_ids[i] for i in idx]
            neighbor_cache[(seed, dis)] = neighbors

    print(f"Cached neighbors for {len(neighbor_cache)} (seed, disease) pairs")

    # For each record, compute neighbor-augmented target union and match
    def bio_gt_target_union(bio_gt_set):
        u = set()
        for g in bio_gt_set:
            tgts = predictor.drug_targets.get(g)
            if tgts:
                u |= tgts
        return u

    rescued_hit = 0
    flipped_hit = 0
    rescued_miss = 0
    flipped_miss = 0

    # For breakdown by orig LOO=False
    hit_loo_false = []   # records with hit=True AND target_match_loo=False (the 16)
    miss_loo_false = []  # records with hit=False AND target_match_loo=False

    # Recompute target_match_loo here (we don't have it in the slot records,
    # but we have it in h995b_debias_results.json). Rebuild from scratch.
    for r in records:
        if r["category"] != "autoimmune":
            continue
        dis = r["disease_id"]
        cand = r["drug_id"]
        # Own-disease bio_gt
        own_bio_gt = (expanded_gt.get(dis, set()) & biologic_pool)
        # Leave-one-out own target union
        loo_ref = own_bio_gt - {cand}
        loo_union = bio_gt_target_union(loo_ref)
        cand_tgts = predictor.drug_targets.get(cand, set())
        target_match_loo = bool(
            cand_tgts and loo_union and (cand_tgts & loo_union)
        )
        r["target_match_loo"] = target_match_loo
        if target_match_loo:
            continue  # not in the flip bucket

        # Build neighbor-augmented bio_gt target union
        neighbors = neighbor_cache.get((r["seed"], dis), [])
        nbr_bio_gt = set()
        for n in neighbors:
            nbr_bio_gt |= (expanded_gt.get(n, set()) & biologic_pool)
        # Leave candidate out of neighbor set too
        nbr_ref = (loo_ref | nbr_bio_gt) - {cand}
        nbr_union = bio_gt_target_union(nbr_ref)
        target_match_neighbor = bool(
            cand_tgts and nbr_union and (cand_tgts & nbr_union)
        )
        r["target_match_neighbor"] = target_match_neighbor

        if r["hit"]:
            hit_loo_false.append(r)
            if target_match_neighbor:
                rescued_hit += 1
            else:
                flipped_hit += 1
        else:
            miss_loo_false.append(r)
            if target_match_neighbor:
                rescued_miss += 1
            else:
                flipped_miss += 1

    n_hit_loo_false = len(hit_loo_false)
    n_miss_loo_false = len(miss_loo_false)

    print()
    print("=" * 78)
    print("AUTOIMMUNE — LOO=False slots rescued by neighbor-augmentation")
    print("=" * 78)
    print(f"HIT  bucket (LOO=False AND hit=True): n={n_hit_loo_false}")
    print(f"  rescued (flip False→True under neighbor aug): {rescued_hit} "
          f"({100*rescued_hit/max(n_hit_loo_false,1):.1f}%)")
    print(f"  stays False: {flipped_hit}")
    print(f"MISS bucket (LOO=False AND hit=False): n={n_miss_loo_false}")
    print(f"  rescued (flip False→True under neighbor aug): {rescued_miss} "
          f"({100*rescued_miss/max(n_miss_loo_false,1):.1f}%)")
    print(f"  stays False: {flipped_miss}")

    # Ship gate
    hit_rate = rescued_hit / max(n_hit_loo_false, 1)
    miss_rate = rescued_miss / max(n_miss_loo_false, 1)
    print()
    print(f"Hit rescue rate : {100*hit_rate:.1f}%  (gate: ≥75%)")
    print(f"Miss rescue rate: {100*miss_rate:.1f}%  (gate: ≤40%)")
    ok = hit_rate >= 0.75 and miss_rate <= 0.40
    print(f"Ship gate for h1003: {'PASS' if ok else 'FAIL'}")

    # Per-hit detail for the 16
    print()
    print("=" * 78)
    print("PER-HIT DETAIL (LOO=False hits)")
    print("=" * 78)
    for r in hit_loo_false:
        dis_name = predictor.disease_names.get(r["disease_id"], r["disease_id"])
        print(f"  [{'RESCUED' if r.get('target_match_neighbor') else 'STILL_FALSE'}] "
              f"{r['drug_name']:<25s} -> {dis_name} (seed={r['seed']})")

    # Save
    out = ROOT / "data/analysis/h1002_neighbor_augmented_rescue.json"
    payload = {
        "hypothesis": "h1002",
        "K_NEIGHBORS": K_NEIGHBORS,
        "n_hit_loo_false": n_hit_loo_false,
        "n_miss_loo_false": n_miss_loo_false,
        "rescued_hit": rescued_hit,
        "flipped_hit": flipped_hit,
        "rescued_miss": rescued_miss,
        "flipped_miss": flipped_miss,
        "hit_rescue_rate": hit_rate,
        "miss_rescue_rate": miss_rate,
        "ship_gate_pass": ok,
        "hit_loo_false_detail": [
            {
                "drug_name": r["drug_name"],
                "disease_id": r["disease_id"],
                "disease_name": predictor.disease_names.get(
                    r["disease_id"], r["disease_id"]
                ),
                "seed": r["seed"],
                "rescued": r.get("target_match_neighbor", False),
            }
            for r in hit_loo_false
        ],
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
