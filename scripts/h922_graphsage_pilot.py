#!/usr/bin/env python3
"""
h922: GraphSAGE scaffolding pilot for h909's legitimately-blocked diseases.

h909 identified 71 diseases blocked by missing data in the DRKG 2x2:
- 24 have GT but no DRKG embedding (this pilot's targets)
- 27 have embedding but no GT
- 20 have neither

GraphSAGE's inductive property lets us generate an embedding for a novel
disease by aggregating its graph neighbours. For the 24 "gt-yes, embed-no"
diseases we don't need to train GraphSAGE — we can test the *core idea*
with a zero-training neighbour-mean baseline first:

1. Find MeSH tree siblings (diseases sharing the longest tree-number prefix)
   that DO have DRKG embeddings.
2. Average those sibling embeddings.
3. Feed the averaged embedding through the same kNN scoring path production
   uses.
4. Measure R@30 against the GT we have for the target.

If this simple neighbour-mean gives usable predictions, full GraphSAGE
training (on GPU, with learned aggregation) is worth pursuing. If the
mean doesn't even work, the approach is structurally weak.

This is the scaffold — a ready-to-run pilot. No PyTorch Geometric install
required, runs on CPU in minutes. Full GraphSAGE training would be a
follow-up hypothesis (h922-v2) with PyG + GPU.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"
DRKG_DIR = PROJECT_ROOT / "data" / "raw" / "drkg"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"


def load_drkg_entities() -> Set[str]:
    """Disease MeSH D-code entities in DRKG's entity table.

    Matches h909's definition: the first column of entities.tsv restricted to
    entries starting with 'Disease::MESH:'. Returns the drkg-prefixed form.
    """
    entities: Set[str] = set()
    ents_path = DRKG_DIR / "embed" / "entities.tsv"
    with open(ents_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if not parts:
                continue
            ent = parts[0]
            if ent.startswith("Disease::MESH:"):
                entities.add(f"drkg:{ent}")
    return entities


def load_mesh_mappings() -> Dict[str, str]:
    """disease_name -> drkg_disease_id (via MeSH D-codes only).

    Matches h909: only MeSH 'D' descriptors are carried by DRKG. 'C' codes
    (supplementary concepts) are valid MeSH but absent from DRKG.
    """
    out: Dict[str, str] = {}
    mp = json.load(open(REFERENCE_DIR / "mesh_mappings_from_agents.json"))
    for batch in mp.values():
        if isinstance(batch, dict):
            for name, mid in batch.items():
                if not isinstance(mid, str):
                    continue
                if not mid.startswith("D"):
                    continue
                out[name.lower()] = f"drkg:Disease::MESH:{mid}"
    ap = REFERENCE_DIR / "h961_disease_name_aliases.json"
    if ap.exists():
        ad = json.load(open(ap))
        for name, did in ad.get("disease_names_backfill", {}).items():
            # Restrict to D-codes
            mid = did.rsplit("MESH:", 1)[-1]
            if mid.startswith("D"):
                out.setdefault(name.lower(), did)
    return out


def load_mesh_tree_cache() -> Dict[str, Dict]:
    return json.load(open(PROJECT_ROOT / "data" / "cache" / "mesh_tree_cache.json"))


def load_node2vec_embeddings() -> Dict[str, np.ndarray]:
    """Match production_predictor's Node2Vec loader."""
    entities_path = EMBEDDINGS_DIR / "node2vec_256_entities.npy"
    embs_path = EMBEDDINGS_DIR / "node2vec_256_embeddings.npy"
    if entities_path.exists() and embs_path.exists():
        ents = np.load(entities_path, allow_pickle=True)
        embs = np.load(embs_path)
        return {f"drkg:{e}": embs[i] for i, e in enumerate(ents)}
    raise FileNotFoundError("node2vec embeddings not found")


def mesh_id_from_drkg(drkg_id: str) -> str:
    return drkg_id.rsplit("MESH:", 1)[-1]


def build_blocked_sets(
    mesh_mappings: Dict[str, str],
    drkg_entities: Set[str],
    exp_gt: Dict[str, list],
) -> Tuple[List[str], List[str], List[str]]:
    """Reconstruct h909's 2x2 'blocked' buckets."""
    all_mapped = set(mesh_mappings.values())

    has_embed = {d for d in all_mapped if d in drkg_entities}
    has_gt = {d for d in all_mapped if d in exp_gt and exp_gt[d]}

    gt_yes_embed_no = sorted(has_gt - has_embed)   # GraphSAGE primary targets
    embed_yes_gt_no = sorted(has_embed - has_gt)   # Secondary targets
    no_embed_no_gt = sorted(all_mapped - has_embed - has_gt)  # Hardest

    return gt_yes_embed_no, embed_yes_gt_no, no_embed_no_gt


def tree_siblings_with_embeddings(
    target_drkg_id: str,
    mesh_tree: Dict[str, Dict],
    drkg_entities: Set[str],
    *,
    min_prefix_depth: int = 3,
) -> List[str]:
    """Find DRKG-embedded MeSH diseases sharing the longest tree prefix.

    Returns up to ~20 closest neighbours. Falls back to broader prefixes if
    the most-specific prefix has no embedded neighbours, down to
    `min_prefix_depth` characters (e.g. 'C04' for Neoplasms). Returns empty
    list if nothing is found even at the root level.
    """
    target_mesh_id = mesh_id_from_drkg(target_drkg_id)
    entry = mesh_tree.get(target_mesh_id)
    if not entry or not entry.get("tree_numbers"):
        return []

    # Strip the URL prefix and drop the final segment each iteration
    target_trees = [
        t.rsplit("/", 1)[-1] for t in entry["tree_numbers"]
    ]

    # Build an index: tree_number -> list of DRKG ids with embeddings
    # (done once per call for simplicity; cache outside in the driver)
    by_tree: Dict[str, List[str]] = defaultdict(list)
    for mid, info in mesh_tree.items():
        drkg = f"drkg:Disease::MESH:{mid}"
        if drkg not in drkg_entities or drkg == target_drkg_id:
            continue
        for t in info.get("tree_numbers", []):
            t = t.rsplit("/", 1)[-1]
            # Index all prefixes so we can match by-descending-specificity
            parts = t.split(".")
            for i in range(1, len(parts) + 1):
                by_tree[".".join(parts[:i])].append(drkg)

    # Find siblings at the longest shared prefix
    seen: Set[str] = set()
    neighbours: List[str] = []
    for target_t in target_trees:
        parts = target_t.split(".")
        while parts:
            prefix = ".".join(parts)
            if len(prefix) < min_prefix_depth:
                break
            for drkg in by_tree.get(prefix, []):
                if drkg not in seen:
                    seen.add(drkg)
                    neighbours.append(drkg)
            if len(neighbours) >= 20:
                break
            parts = parts[:-1]
        if len(neighbours) >= 20:
            break
    return neighbours[:20]


def mean_embedding(
    neighbours: List[str], embeddings: Dict[str, np.ndarray]
) -> Optional[np.ndarray]:
    arrs = [embeddings[n] for n in neighbours if n in embeddings]
    if not arrs:
        return None
    return np.mean(arrs, axis=0)


def knn_top30(
    query: np.ndarray, embeddings: Dict[str, np.ndarray], k: int = 30
) -> List[str]:
    """Cosine top-k drugs for a given query embedding."""
    drug_ids: List[str] = []
    drug_arr: List[np.ndarray] = []
    for eid, emb in embeddings.items():
        if "Compound::" in eid:
            drug_ids.append(eid)
            drug_arr.append(emb)
    if not drug_arr:
        return []
    M = np.stack(drug_arr)
    q = query / (np.linalg.norm(query) + 1e-8)
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
    sims = Mn @ q
    top_idx = np.argsort(-sims)[:k]
    return [drug_ids[i] for i in top_idx]


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("Loading DRKG entities ...")
    drkg_entities = load_drkg_entities()
    print(f"  {len(drkg_entities):,} entities")

    print("Loading mesh mappings + expanded GT + tree cache ...")
    mesh_mappings = load_mesh_mappings()
    exp_gt = json.load(open(REFERENCE_DIR / "expanded_ground_truth.json"))
    mesh_tree = load_mesh_tree_cache()
    print(f"  {len(mesh_mappings):,} mappings; {len(exp_gt):,} GT diseases; "
          f"{len(mesh_tree):,} tree entries")

    print("Loading Node2Vec embeddings ...")
    embeddings = load_node2vec_embeddings()
    print(f"  {len(embeddings):,} entities embedded")

    gt_yes_embed_no, embed_yes_gt_no, neither = build_blocked_sets(
        mesh_mappings, drkg_entities, exp_gt
    )
    print(f"\nh909 reconstruction:")
    print(f"  gt_yes_embed_no (primary GraphSAGE targets): {len(gt_yes_embed_no)}")
    print(f"  embed_yes_gt_no (secondary):                 {len(embed_yes_gt_no)}")
    print(f"  no_embed_no_gt (hardest):                    {len(neither)}")

    # Evaluate each primary target with a neighbour-mean embedding
    results: List[Dict] = []
    name_lookup = {v: k for k, v in mesh_mappings.items()}
    for i, target in enumerate(gt_yes_embed_no, 1):
        mesh_id = mesh_id_from_drkg(target)
        target_name = name_lookup.get(target, "?")
        gt_drugs = set(exp_gt.get(target, []))

        neighbours = tree_siblings_with_embeddings(target, mesh_tree, drkg_entities)
        if not neighbours:
            results.append({
                "target": target,
                "name": target_name,
                "mesh_id": mesh_id,
                "status": "no_neighbours",
                "n_neighbours": 0,
                "n_gt_drugs": len(gt_drugs),
                "r30": 0.0,
                "hits": 0,
            })
            continue

        avg = mean_embedding(neighbours, embeddings)
        if avg is None:
            results.append({
                "target": target, "name": target_name, "mesh_id": mesh_id,
                "status": "empty_avg", "n_neighbours": len(neighbours),
                "n_gt_drugs": len(gt_drugs), "r30": 0.0, "hits": 0,
            })
            continue

        top30 = knn_top30(avg, embeddings)
        hits = len(set(top30) & gt_drugs)
        r30 = hits / min(len(gt_drugs), 30) if gt_drugs else 0.0
        results.append({
            "target": target,
            "name": target_name,
            "mesh_id": mesh_id,
            "status": "scored",
            "n_neighbours": len(neighbours),
            "n_gt_drugs": len(gt_drugs),
            "hits": int(hits),
            "r30": float(r30),
            "top30_sample": top30[:5],
        })
        if i % 5 == 0 or i == len(gt_yes_embed_no):
            print(f"  [{i}/{len(gt_yes_embed_no)}] {target_name[:40]} "
                  f"n={len(neighbours)} R@30={r30:.3f}")

    # Summary
    scored = [r for r in results if r["status"] == "scored"]
    non_empty = [r for r in scored if r["hits"] > 0]
    r30s = [r["r30"] for r in scored]
    print("\n" + "=" * 60)
    print("  h922 neighbour-mean pilot — results")
    print("=" * 60)
    print(f"  Primary targets:            {len(gt_yes_embed_no)}")
    print(f"  With tree-sibling neighbours: {len(scored)}")
    print(f"  At least one top-30 hit:     {len(non_empty)}")
    if r30s:
        print(f"  Mean R@30:   {np.mean(r30s):.3f}")
        print(f"  Median R@30: {float(np.median(r30s)):.3f}")
        print(f"  Max R@30:    {max(r30s):.3f}")

    # Decision heuristic
    production_r30 = 0.30  # approx production R@30 baseline
    if r30s and np.mean(r30s) >= 0.10:
        decision = "PROCEED_TO_FULL_GRAPHSAGE"
    elif r30s and np.mean(r30s) >= 0.05:
        decision = "SIGNAL_PRESENT_BUT_WEAK"
    else:
        decision = "NEIGHBOUR_MEAN_INSUFFICIENT"
    print(f"\n  Decision: {decision}")
    print(f"  Reference: production kNN R@30 ~{production_r30:.2f} on reachable diseases")

    out_path = ANALYSIS_DIR / "h922_pilot.json"
    out = {
        "n_primary_targets": len(gt_yes_embed_no),
        "n_with_neighbours": len(scored),
        "n_with_any_hit": len(non_empty),
        "mean_r30": float(np.mean(r30s)) if r30s else None,
        "median_r30": float(np.median(r30s)) if r30s else None,
        "max_r30": float(max(r30s)) if r30s else None,
        "decision": decision,
        "wall_time_seconds": float(time.time() - t0),
        "per_target": results,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
