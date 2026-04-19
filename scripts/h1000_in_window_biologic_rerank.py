#!/usr/bin/env python3
"""h1000: In-window biologic family-match re-rank (no in/out exchange).

Test whether bumping target_match_LOO=True biologics up by 1 rank and
demoting target_match_LOO=False biologics down by 1 rank — within the
existing top-30 — lifts tier precision without dropping bio_r30.

target_match_LOO is computed from the k=3 kNN neighbors' bio_gt (NOT the
disease's own bio_gt — that leaks at inference time). Per h1002, the
filter-form of this rule drops 50% of autoimmune biologic hits because
they are target-unique; h1000 is the softer in-window form.

Ship gate:
    bio_r30 drop <= 0.5pp AND at least one tier lifts >= 1pp
    AND no other tier drops >= 1pp.

The experiment:
1. Load predictor + expanded GT + biologic pool.
2. For each of 5 seeds: recompute GT structures for train-only; predict
   on holdout; capture baseline top-30 preds.
3. For each biologic pred: compute target_match_loo_neighbor using
   k=3 kNN train-disease neighbors' bio_gt target union.
4. Re-sort top-30 using score-perturbation key:
       key(pred) = orig_rank + (-1.5 if bio & match else +1.5 if bio & !match else 0)
5. For each pred at new rank: re-tier via _assign_confidence_tier + replay
   tier-dependent mutations that are rank-independent (target_overlap
   promotion, hematological/infectious CS demotions, literature_strong /
   literature_moderate / literature_high promotions). Rank-dependent
   inline mutations (TransE rank<=5; lit_weak rank<=5 skip) ARE replayed
   with the new rank.
6. Aggregate: baseline vs shifted tier precision; bio-tier distribution.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from production_predictor import (  # noqa: E402
    ConfidenceTier,
    DrugRepurposingPredictor,
    TARGET_OVERLAP_PROMOTE_HIGH_TO_GOLDEN,
    TARGET_OVERLAP_PROMOTE_LOW_TO_MEDIUM,
    TARGET_OVERLAP_GOLDEN_ELIGIBLE_RULES,
    _CORTICOSTEROID_LOWER,
)
from h393_holdout_tier_validation import split_diseases, recompute_gt_structures, restore_gt_structures  # noqa: E402
from h939_biologic_target_overlap_audit import is_biologic  # noqa: E402
from h995_autoimmune_biologic_family_audit import load_expanded_gt  # noqa: E402


K_NEIGHBORS = 3
SEEDS = [42, 123, 456, 789, 2024]
import os as _os
RANK_SHIFT_MAGNITUDE = float(_os.environ.get("H1000_RANK_SHIFT", "1.5"))   # default: cross one integer rank boundary
_OUT_SUFFIX = _os.environ.get("H1000_OUT_SUFFIX", "")


def compute_neighbors(
    predictor: DrugRepurposingPredictor,
    holdout_ids: List[str],
    train_ids: List[str],
    k: int = K_NEIGHBORS,
) -> Dict[str, List[str]]:
    """For each holdout disease, return its top-k train-disease neighbors."""
    train_emb_arr = np.array(
        [predictor.embeddings[d] for d in train_ids], dtype=np.float32
    )
    neighbors: Dict[str, List[str]] = {}
    for dis in holdout_ids:
        if dis not in predictor.embeddings:
            continue
        q = predictor.embeddings[dis].astype(np.float32)
        diffs = train_emb_arr - q
        dists = np.einsum("ij,ij->i", diffs, diffs)
        idx = np.argsort(dists)[:k]
        neighbors[dis] = [train_ids[i] for i in idx]
    return neighbors


def bio_gt_target_union(
    predictor: DrugRepurposingPredictor,
    bio_gt_set: Set[str],
) -> Set[str]:
    u: Set[str] = set()
    for g in bio_gt_set:
        tgts = predictor.drug_targets.get(g)
        if tgts:
            u |= tgts
    return u


def compute_neighbor_target_union(
    predictor: DrugRepurposingPredictor,
    neighbors: List[str],
    expanded_gt: Dict[str, Set[str]],
    biologic_pool: Set[str],
) -> Set[str]:
    bio_gt = set()
    for n in neighbors:
        bio_gt |= (expanded_gt.get(n, set()) & biologic_pool)
    return bio_gt_target_union(predictor, bio_gt)


def rerank_top30_with_shift(
    preds_sorted: List,  # DrugPrediction list sorted by rank asc
    match_flags: Dict[str, bool],  # drug_id -> target_match_loo_neighbor (bio only)
    bio_pool: Set[str],
) -> List[int]:
    """Return new_rank list aligned with preds_sorted.

    Uses score-perturbation: sort key = orig_rank + shift where
        shift = -1.5 if biologic & match=True
        shift = +1.5 if biologic & match=False
        shift = 0 otherwise

    Ties broken by original rank (stable).
    """
    keyed = []
    for p in preds_sorted:
        is_bio = p.drug_id in bio_pool
        if is_bio and p.drug_id in match_flags:
            shift = -RANK_SHIFT_MAGNITUDE if match_flags[p.drug_id] else RANK_SHIFT_MAGNITUDE
        else:
            shift = 0.0
        keyed.append((p.rank + shift, p.rank, p.drug_id))
    # Sort by (shifted_key, orig_rank, drug_id) — deterministic
    order = sorted(range(len(preds_sorted)), key=lambda i: keyed[i])
    new_ranks = [0] * len(preds_sorted)
    for new_pos, orig_idx in enumerate(order, start=1):
        new_ranks[orig_idx] = new_pos
    return new_ranks


def replay_tier_for_shifted_rank(
    predictor: DrugRepurposingPredictor,
    pred,
    new_rank: int,
    disease_id: str,
    disease_name: str,
    category: str,
    disease_tier: int,
) -> Tuple[ConfidenceTier, str]:
    """Replicate the tier logic in predict() (lines 4861–5033) for a given rank.

    Uses the pred's knn_score, mechanism_support, train_frequency, has_targets,
    drug_id, drug_name, literature_evidence_level, transe_consilience, and
    category. Recomputes target_overlap_count (cheap lookup).
    """
    drug_id = pred.drug_id
    drug_name = pred.drug_name
    knn_score = pred.knn_score
    mech_support = pred.mechanism_support
    train_freq = pred.train_frequency
    has_targets = pred.has_targets
    in_transe_top30 = pred.transe_consilience
    lit_level = pred.literature_evidence_level

    # Base tier from _assign_confidence_tier
    tier, rescue_applied, cat_specific = predictor._assign_confidence_tier(
        new_rank,
        train_freq,
        mech_support,
        has_targets,
        disease_tier,
        category,
        drug_name,
        disease_name,
        drug_id,
        knn_score,
    )

    # h388 target_overlap_promotion
    target_overlap = predictor._get_target_overlap_count(drug_id, disease_id)
    if (tier == ConfidenceTier.HIGH
            and target_overlap >= TARGET_OVERLAP_PROMOTE_HIGH_TO_GOLDEN
            and cat_specific in TARGET_OVERLAP_GOLDEN_ELIGIBLE_RULES):
        tier = ConfidenceTier.GOLDEN
    elif (tier == ConfidenceTier.LOW
            and target_overlap >= TARGET_OVERLAP_PROMOTE_LOW_TO_MEDIUM
            and category not in {'gastrointestinal', 'immunological', 'reproductive',
                                 'neurological', 'cancer', 'cardiovascular',
                                 'hematological', 'metabolic'}
            and cat_specific != 'incoherent_demotion'
            and cat_specific != 'antimicrobial_pathogen_mismatch'
            and cat_specific != 'infectious_hierarchy_pneumonia'
            and cat_specific != 'local_anesthetic_procedural'
            and cat_specific != 'default_nomech_low_score'
            and cat_specific != 'default_nomech_r1_5_low_score'):
        tier = ConfidenceTier.MEDIUM
        cat_specific = cat_specific or 'target_overlap_promotion'

    # h522 hematological corticosteroid demotion
    if (tier == ConfidenceTier.MEDIUM
            and category == 'hematological'
            and drug_name.lower() in _CORTICOSTEROID_LOWER
            and not predictor._is_immune_mediated_hematological(disease_name)):
        tier = ConfidenceTier.LOW
        cat_specific = 'hematological_corticosteroid_demotion'

    # h559 CS→TB demotion (HIGH→MEDIUM)
    if (tier == ConfidenceTier.HIGH
            and category == 'infectious'
            and drug_name.lower() in _CORTICOSTEROID_LOWER
            and cat_specific and 'tuberculosis' in cat_specific):
        tier = ConfidenceTier.MEDIUM
        cat_specific = 'infectious_cs_tb_demotion'

    # h557 CS→infectious demotion (MEDIUM→LOW)
    if (tier == ConfidenceTier.MEDIUM
            and category == 'infectious'
            and drug_name.lower() in _CORTICOSTEROID_LOWER
            and cat_specific != 'infectious_cs_tb_demotion'):
        tier = ConfidenceTier.LOW
        cat_specific = 'infectious_corticosteroid_demotion'

    # h630 TransE MEDIUM→HIGH promotion — RANK-DEPENDENT via rank<=5
    if (tier == ConfidenceTier.MEDIUM
            and in_transe_top30
            and drug_name.lower() not in _CORTICOSTEROID_LOWER
            and (mech_support or new_rank <= 5)):
        tier = ConfidenceTier.HIGH
        cat_specific = 'transe_medium_promotion'

    # literature_strong_promotion MEDIUM→GOLDEN
    if tier == ConfidenceTier.MEDIUM and lit_level == 'STRONG_EVIDENCE':
        tier = ConfidenceTier.GOLDEN
        cat_specific = 'literature_strong_promotion'

    # literature_strong_low_promotion LOW/FILTER→HIGH
    if (tier in (ConfidenceTier.LOW, ConfidenceTier.FILTER)
            and lit_level == 'STRONG_EVIDENCE'
            and cat_specific not in ('inverse_indication', 'non_therapeutic',
                                     'non_therapeutic_compound')
            and category != 'other'):
        tier = ConfidenceTier.HIGH
        cat_specific = 'literature_strong_low_promotion'

    # literature_moderate_low_promotion LOW→MEDIUM
    if (tier == ConfidenceTier.LOW
            and lit_level == 'MODERATE_EVIDENCE'
            and cat_specific not in ('inverse_indication', 'non_therapeutic',
                                     'non_therapeutic_compound')):
        tier = ConfidenceTier.MEDIUM
        cat_specific = 'literature_moderate_low_promotion'

    # h791 literature_high_demotion HIGH→MEDIUM
    _orig_cat_specific = cat_specific
    if (tier == ConfidenceTier.HIGH
            and lit_level in ('NO_EVIDENCE', 'WEAK_EVIDENCE')):
        tier = ConfidenceTier.MEDIUM
        cat_specific = 'literature_high_demotion'

    # h732/h984 literature_weak_demotion MEDIUM→LOW — RANK-DEPENDENT via rank<=5 skip
    _LIT_DEMOTION_PROTECTED = {'default_freq10_nomech_r1_5'}
    _skip_lit_demotion = (
        cat_specific == 'literature_high_demotion'
        and _orig_cat_specific in _LIT_DEMOTION_PROTECTED
    ) or (lit_level == 'WEAK_EVIDENCE' and new_rank <= 5)
    if (tier == ConfidenceTier.MEDIUM
            and lit_level in ('NO_EVIDENCE', 'WEAK_EVIDENCE')
            and not _skip_lit_demotion):
        tier = ConfidenceTier.LOW
        cat_specific = 'literature_weak_demotion'

    return tier, cat_specific


def evaluate_seed(
    predictor: DrugRepurposingPredictor,
    seed: int,
    all_diseases: List[str],
    expanded_gt_sets: Dict[str, Set[str]],
    biologic_pool: Set[str],
) -> Dict:
    train_ids, holdout_ids = split_diseases(all_diseases, seed)
    # Filter holdout to embedded diseases
    holdout_ids = [d for d in holdout_ids if d in predictor.embeddings]
    train_ids = [d for d in train_ids if d in predictor.embeddings]

    # Recompute GT structures for training
    originals = recompute_gt_structures(predictor, set(train_ids))

    try:
        # Compute k-NN neighbors for each holdout disease (train-only)
        neighbor_cache = compute_neighbors(predictor, holdout_ids, train_ids, k=K_NEIGHBORS)

        # For each holdout disease, build neighbor target union once
        nbr_target_union_cache: Dict[str, Set[str]] = {}
        for dis in holdout_ids:
            nbrs = neighbor_cache.get(dis, [])
            nbr_target_union_cache[dis] = compute_neighbor_target_union(
                predictor, nbrs, expanded_gt_sets, biologic_pool
            )

        # Run baseline predictions and capture them
        per_disease: Dict[str, Dict] = {}
        for dis in holdout_ids:
            try:
                result = predictor.predict(dis, top_n=30, include_filtered=True)
            except Exception:
                continue
            per_disease[dis] = {
                "disease_name": result.disease_name,
                "category": result.category,
                "disease_tier": result.disease_tier,
                "preds": list(result.predictions),
            }

        # Aggregate stats
        stats = {
            "baseline_tier": defaultdict(lambda: {"hits": 0, "total": 0}),
            "shifted_tier": defaultdict(lambda: {"hits": 0, "total": 0}),
            "baseline_bio_tier": Counter(),
            "shifted_bio_tier": Counter(),
            "baseline_bio_tier_hits": Counter(),
            "shifted_bio_tier_hits": Counter(),
            "bio_r30_per_disease": [],   # list of per-disease bio_r30 (same for both)
            "bio_p30_per_disease": [],
            "overall_p30_per_disease": [],
            "n_bio_slots": 0,
            "n_bio_hits": 0,
            "n_bio_match_true": 0,
            "n_bio_match_false": 0,
            "tier_moves": Counter(),     # (orig_tier, new_tier) -> count for biologics
            "tier_moves_hits": Counter(),
        }

        for dis, rec in per_disease.items():
            preds = rec["preds"]
            category = rec["category"]
            disease_tier = rec["disease_tier"]
            disease_name = rec["disease_name"]

            gt_drugs = expanded_gt_sets.get(dis, set())
            bio_gt_count = len(gt_drugs & biologic_pool)
            nbr_target_union = nbr_target_union_cache.get(dis, set())

            # Compute match flags for biologics
            match_flags: Dict[str, bool] = {}
            for p in preds:
                if p.drug_id in biologic_pool:
                    cand_tgts = predictor.drug_targets.get(p.drug_id, set())
                    m = bool(cand_tgts and nbr_target_union and (cand_tgts & nbr_target_union))
                    match_flags[p.drug_id] = m

            # Sort predictions by rank asc (should already be, but ensure)
            preds_sorted = sorted(preds, key=lambda p: p.rank)
            new_ranks = rerank_top30_with_shift(preds_sorted, match_flags, biologic_pool)

            # For each pred: replay tier with new_rank
            bio_hits_in_top30 = 0
            for i, p in enumerate(preds_sorted):
                new_rank = new_ranks[i]
                is_hit = p.drug_id in gt_drugs
                is_bio = p.drug_id in biologic_pool

                # Baseline tier: from pred
                base_tier = p.confidence_tier.name

                # Shifted tier: replay
                new_tier_enum, new_cat_spec = replay_tier_for_shifted_rank(
                    predictor, p, new_rank, dis, disease_name, category, disease_tier
                )
                shift_tier = new_tier_enum.name

                stats["baseline_tier"][base_tier]["total"] += 1
                stats["shifted_tier"][shift_tier]["total"] += 1
                if is_hit:
                    stats["baseline_tier"][base_tier]["hits"] += 1
                    stats["shifted_tier"][shift_tier]["hits"] += 1

                if is_bio:
                    stats["n_bio_slots"] += 1
                    stats["baseline_bio_tier"][base_tier] += 1
                    stats["shifted_bio_tier"][shift_tier] += 1
                    if is_hit:
                        bio_hits_in_top30 += 1
                        stats["n_bio_hits"] += 1
                        stats["baseline_bio_tier_hits"][base_tier] += 1
                        stats["shifted_bio_tier_hits"][shift_tier] += 1
                    if match_flags.get(p.drug_id):
                        stats["n_bio_match_true"] += 1
                    else:
                        stats["n_bio_match_false"] += 1
                    stats["tier_moves"][(base_tier, shift_tier)] += 1
                    if is_hit:
                        stats["tier_moves_hits"][(base_tier, shift_tier)] += 1

            # bio_r30 (same for baseline and shifted since top-30 membership unchanged)
            if bio_gt_count > 0:
                stats["bio_r30_per_disease"].append(bio_hits_in_top30 / bio_gt_count)
            # overall p30
            n_hits_top30 = sum(1 for p in preds_sorted if p.drug_id in gt_drugs)
            stats["overall_p30_per_disease"].append(n_hits_top30 / max(len(preds_sorted), 1))
    finally:
        restore_gt_structures(predictor, originals)

    return stats


def summarize_stats(label: str, stats: Dict) -> Dict:
    tier_prec = {}
    for tier, s in stats.items():
        if s["total"] > 0:
            tier_prec[tier] = {
                "precision": round(100 * s["hits"] / s["total"], 2),
                "hits": s["hits"],
                "total": s["total"],
            }
    return tier_prec


def main():
    print("=" * 78)
    print("h1000: In-window biologic family-match re-rank")
    print("=" * 78)

    predictor = DrugRepurposingPredictor()
    expanded_gt_raw = load_expanded_gt(
        predictor.reference_dir / "expanded_ground_truth.json"
    )
    # load_expanded_gt returns dict[str] -> set[str]
    expanded_gt_sets: Dict[str, Set[str]] = {
        k: (set(v) if not isinstance(v, set) else v)
        for k, v in expanded_gt_raw.items()
    }

    biologic_pool = {
        d for d in predictor.drug_id_to_name
        if is_biologic(predictor.drug_id_to_name.get(d))
    }
    print(f"Biologic pool: {len(biologic_pool)}")

    all_diseases = [
        d for d in predictor.ground_truth if d in predictor.embeddings
    ]
    print(f"Diseases with GT + embeddings: {len(all_diseases)}")

    seed_stats_list = []
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        st = evaluate_seed(predictor, seed, all_diseases, expanded_gt_sets, biologic_pool)
        seed_stats_list.append(st)

        b_prec = summarize_stats("baseline", st["baseline_tier"])
        s_prec = summarize_stats("shifted", st["shifted_tier"])
        print(f"  baseline tier precision: {b_prec}")
        print(f"  shifted tier precision:  {s_prec}")
        print(f"  n_bio_slots: {st['n_bio_slots']}, n_bio_hits: {st['n_bio_hits']}")
        print(f"  bio match=True: {st['n_bio_match_true']}, match=False: {st['n_bio_match_false']}")

    # Aggregate across seeds (micro)
    def merge_tier(agg_key: str) -> Dict:
        agg: Dict[str, Dict[str, int]] = defaultdict(lambda: {"hits": 0, "total": 0})
        for st in seed_stats_list:
            for tier, s in st[agg_key].items():
                agg[tier]["hits"] += s["hits"]
                agg[tier]["total"] += s["total"]
        return dict(agg)

    baseline_agg = merge_tier("baseline_tier")
    shifted_agg = merge_tier("shifted_tier")

    # Per-seed tier precision lists for mean/std
    def per_seed_prec(agg_key: str) -> Dict[str, List[float]]:
        out: Dict[str, List[float]] = defaultdict(list)
        for st in seed_stats_list:
            for tier in ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]:
                s = st[agg_key].get(tier)
                if s and s["total"] > 0:
                    out[tier].append(100 * s["hits"] / s["total"])
                else:
                    out[tier].append(0.0)
        return out

    b_ps = per_seed_prec("baseline_tier")
    s_ps = per_seed_prec("shifted_tier")

    print()
    print("=" * 78)
    print("AGGREGATE TIER PRECISION (5 seeds)")
    print("=" * 78)
    print(f"{'tier':<8} {'base mean±std':<18} {'shift mean±std':<18} {'delta':<10}")
    deltas = {}
    for tier in ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]:
        b_arr = b_ps.get(tier, [0.0])
        s_arr = s_ps.get(tier, [0.0])
        b_m, b_s = float(np.mean(b_arr)), float(np.std(b_arr, ddof=1)) if len(b_arr) > 1 else 0.0
        s_m, s_s = float(np.mean(s_arr)), float(np.std(s_arr, ddof=1)) if len(s_arr) > 1 else 0.0
        d = s_m - b_m
        deltas[tier] = d
        print(f"{tier:<8} {b_m:6.2f}±{b_s:5.2f}      {s_m:6.2f}±{s_s:5.2f}      {d:+6.2f}pp")

    # Bio-tier distribution (aggregated)
    print()
    print("BIOLOGIC TIER DISTRIBUTION (5 seeds pooled)")
    print(f"{'tier':<8} {'base':<12} {'shift':<12} {'Δ':<8}")
    base_bt = Counter()
    shift_bt = Counter()
    base_bt_hits = Counter()
    shift_bt_hits = Counter()
    for st in seed_stats_list:
        base_bt += st["baseline_bio_tier"]
        shift_bt += st["shifted_bio_tier"]
        base_bt_hits += st["baseline_bio_tier_hits"]
        shift_bt_hits += st["shifted_bio_tier_hits"]

    for tier in ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]:
        b_n = base_bt.get(tier, 0)
        b_h = base_bt_hits.get(tier, 0)
        s_n = shift_bt.get(tier, 0)
        s_h = shift_bt_hits.get(tier, 0)
        b_p = 100 * b_h / b_n if b_n > 0 else 0.0
        s_p = 100 * s_h / s_n if s_n > 0 else 0.0
        print(f"{tier:<8} {b_h:4d}/{b_n:4d} ({b_p:5.1f}%)  {s_h:4d}/{s_n:4d} ({s_p:5.1f}%)  Δn={s_n-b_n:+5d}")

    # bio_r30
    bio_r30_all = []
    for st in seed_stats_list:
        bio_r30_all.extend(st["bio_r30_per_disease"])
    bio_r30_mean = float(np.mean(bio_r30_all)) * 100 if bio_r30_all else 0.0
    bio_r30_std = float(np.std(bio_r30_all, ddof=1)) * 100 if len(bio_r30_all) > 1 else 0.0
    print()
    print(f"bio_r30 (macro): {bio_r30_mean:.2f}% ± {bio_r30_std:.2f}% on n={len(bio_r30_all)} disease-seeds")
    print("  (baseline == shifted since top-30 membership unchanged)")

    # Tier-move matrix for biologics (most informative)
    print()
    print("BIOLOGIC TIER MOVES (5 seeds pooled)")
    print(f"{'from_tier':<10} {'to_tier':<10} {'count':<8} {'hits':<8} {'prec':<8}")
    tier_moves_agg = Counter()
    tier_moves_hits_agg = Counter()
    for st in seed_stats_list:
        tier_moves_agg += st["tier_moves"]
        tier_moves_hits_agg += st["tier_moves_hits"]
    for (ft, tt), cnt in sorted(tier_moves_agg.items()):
        if ft == tt:
            continue  # only show actual moves
        h = tier_moves_hits_agg.get((ft, tt), 0)
        prec = 100 * h / cnt if cnt > 0 else 0.0
        print(f"{ft:<10} {tt:<10} {cnt:<8} {h:<8} {prec:5.1f}%")

    # SHIP GATE
    bio_r30_drop = 0.0  # always zero by construction
    tier_lifts = {t: d for t, d in deltas.items() if d >= 1.0}
    tier_drops = {t: d for t, d in deltas.items() if d <= -1.0}

    print()
    print("=" * 78)
    print("SHIP GATE")
    print("=" * 78)
    print(f"bio_r30 drop: {bio_r30_drop:.2f}pp  (gate ≤0.5pp)  [{'PASS' if bio_r30_drop <= 0.5 else 'FAIL'}]")
    print(f"tier lifts ≥1pp: {tier_lifts if tier_lifts else 'NONE'}")
    print(f"tier drops ≥1pp: {tier_drops if tier_drops else 'NONE'}")
    passed = (bio_r30_drop <= 0.5) and (len(tier_lifts) >= 1) and (len(tier_drops) == 0)
    print(f"SHIP: {'PASS' if passed else 'FAIL'}")

    # Save
    payload = {
        "hypothesis": "h1000",
        "seeds": SEEDS,
        "K_NEIGHBORS": K_NEIGHBORS,
        "RANK_SHIFT_MAGNITUDE": RANK_SHIFT_MAGNITUDE,
        "baseline_tier_mean": {t: float(np.mean(b_ps[t])) for t in b_ps},
        "shifted_tier_mean": {t: float(np.mean(s_ps[t])) for t in s_ps},
        "baseline_tier_std": {
            t: float(np.std(b_ps[t], ddof=1)) if len(b_ps[t]) > 1 else 0.0 for t in b_ps
        },
        "shifted_tier_std": {
            t: float(np.std(s_ps[t], ddof=1)) if len(s_ps[t]) > 1 else 0.0 for t in s_ps
        },
        "tier_deltas": deltas,
        "bio_r30_mean_pct": bio_r30_mean,
        "bio_r30_std_pct": bio_r30_std,
        "bio_r30_n_disease_seeds": len(bio_r30_all),
        "bio_tier_baseline": {t: {"n": base_bt.get(t, 0), "hits": base_bt_hits.get(t, 0)} for t in ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]},
        "bio_tier_shifted": {t: {"n": shift_bt.get(t, 0), "hits": shift_bt_hits.get(t, 0)} for t in ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]},
        "bio_tier_moves": [
            {"from": ft, "to": tt, "count": cnt, "hits": tier_moves_hits_agg.get((ft, tt), 0)}
            for (ft, tt), cnt in sorted(tier_moves_agg.items()) if ft != tt
        ],
        "ship_pass": passed,
        "ship_gate_lifts": tier_lifts,
        "ship_gate_drops": tier_drops,
    }
    out = ROOT / f"data/analysis/h1000_in_window_bio_rerank{_OUT_SUFFIX}.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
