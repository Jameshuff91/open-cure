#!/usr/bin/env python3
"""h952: Diagnose the ~4pp recall regression between plain-kNN and the
production pipeline (predict()).

Baseline: h940 plain kNN bio_r30 = 31.42%, overall_r30 = 20.30% (5-seed).
Observed: h951 production bio_r30 = 27.06%, overall_r30 = 16.39% (5-seed).

Goal:
  For seed 42 only (low effort, per-roadmap step), for every holdout disease:
    - production_top30 = predict(top_n=30, include_filtered=True)
    - plain_knn_top30  = top 30 drugs by plain-kNN score (same setup as h940
      baseline — no SELECTIVE_BOOST, no supplements)
  Diff: lost = plain - production, added = production - plain
  Stratify lost-prediction counts by:
    - disease category
    - biologic vs small-molecule (h939 proxy)
    - SELECTIVE_BOOST eligibility (neurological, respiratory, metabolic, renal,
      hematological, immunological)
    - supplement eligibility (neurological, gastrointestinal)
  Per-category recall delta = overall_r30(production) - overall_r30(plain_kNN)

Decision:
  If SELECTIVE_BOOST categories dominate the lost-recall, the boost (h170) is
  a net negative on expanded-GT 5-seed holdout and we propose disabling it
  (or fusing scores instead of replacing neighbors). Follow-up hypothesis for
  the full 5-seed sweep is generated regardless of seed-42 signal.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from production_predictor import (  # noqa: E402
    DrugRepurposingPredictor,
    SELECTIVE_BOOST_CATEGORIES,
)
from h393_holdout_tier_validation import (  # noqa: E402
    split_diseases,
    recompute_gt_structures,
    restore_gt_structures,
)
from h939_biologic_target_overlap_audit import is_biologic  # noqa: E402


SUPPLEMENT_CATEGORIES = {"neurological", "gastrointestinal"}


def load_expanded_gt(path: Path) -> Dict[str, Set[str]]:
    with open(path) as f:
        raw = json.load(f)
    out: Dict[str, Set[str]] = {}
    for dis_id, drugs in raw.items():
        s: Set[str] = set()
        for d in drugs:
            if isinstance(d, str):
                s.add(d)
            elif isinstance(d, dict):
                did = d.get("drug_id") or d.get("drug")
                if did:
                    s.add(did)
        out[dis_id] = s
    return out


def plain_knn_top30(
    predictor: DrugRepurposingPredictor,
    disease_id: str,
    k: int = 20,
    top_n: int = 30,
) -> List[str]:
    """Plain kNN top-30 — no SELECTIVE_BOOST, no supplements, no MinRank.

    Uses predictor.train_embeddings / predictor.train_diseases which have
    already been recomputed by recompute_gt_structures.
    """
    test_emb = predictor.embeddings[disease_id].reshape(1, -1)
    sims = cosine_similarity(test_emb, predictor.train_embeddings)[0]
    top_k_idx = np.argsort(sims)[-k:]
    drug_scores: Dict[str, float] = defaultdict(float)
    for idx in top_k_idx:
        neighbor = predictor.train_diseases[idx]
        neighbor_sim = float(sims[idx])
        for drug_id in predictor.ground_truth.get(neighbor, set()):
            if drug_id in predictor.embeddings:
                drug_scores[drug_id] += neighbor_sim
    sorted_drugs = sorted(drug_scores.items(), key=lambda x: (-x[1], x[0]))
    return [drug_id for drug_id, _ in sorted_drugs[:top_n]]


def production_top30(
    predictor: DrugRepurposingPredictor,
    disease_name: str,
    top_n: int = 30,
) -> List[str]:
    result = predictor.predict(disease_name, top_n=top_n, include_filtered=True)
    return [p.drug_id for p in result.predictions[:top_n]]


def main():
    seed = 42
    print("=" * 78)
    print(f"h952: Plain kNN vs Production top-30 diagnostic (seed={seed})")
    print("=" * 78)

    print("Loading predictor ...")
    predictor = DrugRepurposingPredictor()

    expanded_gt = load_expanded_gt(
        predictor.reference_dir / "expanded_ground_truth.json"
    )

    biologic_pool = {
        d for d in predictor.drug_targets
        if is_biologic(predictor.drug_id_to_name.get(d))
    }
    print(f"Biologic pool: {len(biologic_pool)}")

    all_diseases = [
        d for d in predictor.ground_truth if d in predictor.embeddings
    ]
    train_ids, holdout_ids = split_diseases(all_diseases, seed)
    print(f"Train: {len(train_ids)}  Holdout: {len(holdout_ids)}")

    originals = recompute_gt_structures(predictor, set(train_ids))

    per_disease: List[Dict] = []
    n_skipped = 0

    try:
        for i, dis_id in enumerate(holdout_ids, 1):
            if i % 25 == 0:
                print(f"  ... {i}/{len(holdout_ids)} diseases")
            gt_drugs = expanded_gt.get(dis_id, set())
            if not gt_drugs:
                n_skipped += 1
                continue
            dis_name = predictor.disease_names.get(dis_id)
            if not dis_name:
                n_skipped += 1
                continue
            try:
                prod_top = production_top30(predictor, dis_name)
                plain_top = plain_knn_top30(predictor, dis_id)
            except Exception as e:
                n_skipped += 1
                continue

            prod_set = set(prod_top)
            plain_set = set(plain_top)
            lost = plain_set - prod_set
            added = prod_set - plain_set

            category = predictor.categorize_disease(dis_name)

            # Lost GT hits = drugs that were in plain-kNN top-30 AND GT, but missing from prod top-30
            lost_gt_hits = lost & gt_drugs
            added_gt_hits = added & gt_drugs

            bio_gt = gt_drugs & biologic_pool
            sm_gt = gt_drugs - biologic_pool

            overall_r30_prod = len(prod_set & gt_drugs) / len(gt_drugs)
            overall_r30_plain = len(plain_set & gt_drugs) / len(gt_drugs)

            bio_r30_prod = (
                len(prod_set & bio_gt) / len(bio_gt) if bio_gt else None
            )
            bio_r30_plain = (
                len(plain_set & bio_gt) / len(bio_gt) if bio_gt else None
            )

            sm_r30_prod = (
                len(prod_set & sm_gt) / len(sm_gt) if sm_gt else None
            )
            sm_r30_plain = (
                len(plain_set & sm_gt) / len(sm_gt) if sm_gt else None
            )

            per_disease.append({
                "disease_id": dis_id,
                "disease_name": dis_name,
                "category": category,
                "boost_eligible": category in SELECTIVE_BOOST_CATEGORIES,
                "supplement_eligible": category in SUPPLEMENT_CATEGORIES,
                "gt_size": len(gt_drugs),
                "bio_gt_size": len(bio_gt),
                "sm_gt_size": len(sm_gt),
                "n_lost": len(lost),
                "n_added": len(added),
                "n_lost_gt_hits": len(lost_gt_hits),
                "n_lost_bio": len(lost & biologic_pool),
                "n_lost_sm": len(lost - biologic_pool),
                "n_lost_bio_gt": len(lost_gt_hits & biologic_pool),
                "n_lost_sm_gt": len(lost_gt_hits - biologic_pool),
                "n_added_gt_hits": len(added_gt_hits),
                "overall_r30_prod": overall_r30_prod,
                "overall_r30_plain": overall_r30_plain,
                "overall_delta": overall_r30_prod - overall_r30_plain,
                "bio_r30_prod": bio_r30_prod,
                "bio_r30_plain": bio_r30_plain,
                "sm_r30_prod": sm_r30_prod,
                "sm_r30_plain": sm_r30_plain,
                "lost_drug_ids": sorted(lost),
                "added_drug_ids": sorted(added),
                "lost_gt_hit_ids": sorted(lost_gt_hits),
                "added_gt_hit_ids": sorted(added_gt_hits),
            })
    finally:
        restore_gt_structures(predictor, originals)

    print(f"\nEvaluated {len(per_disease)} diseases  (skipped {n_skipped})")

    # ---- Aggregates ----
    def agg(recs):
        bios = [r["bio_r30_prod"] for r in recs if r["bio_r30_prod"] is not None]
        bios_plain = [r["bio_r30_plain"] for r in recs if r["bio_r30_plain"] is not None]
        sms = [r["sm_r30_prod"] for r in recs if r["sm_r30_prod"] is not None]
        sms_plain = [r["sm_r30_plain"] for r in recs if r["sm_r30_plain"] is not None]
        return {
            "n": len(recs),
            "overall_prod": float(np.mean([r["overall_r30_prod"] for r in recs])) if recs else 0.0,
            "overall_plain": float(np.mean([r["overall_r30_plain"] for r in recs])) if recs else 0.0,
            "bio_prod": float(np.mean(bios)) if bios else None,
            "bio_plain": float(np.mean(bios_plain)) if bios_plain else None,
            "sm_prod": float(np.mean(sms)) if sms else None,
            "sm_plain": float(np.mean(sms_plain)) if sms_plain else None,
            "mean_lost": float(np.mean([r["n_lost"] for r in recs])) if recs else 0.0,
            "mean_lost_gt_hits": float(np.mean([r["n_lost_gt_hits"] for r in recs])) if recs else 0.0,
            "mean_added_gt_hits": float(np.mean([r["n_added_gt_hits"] for r in recs])) if recs else 0.0,
        }

    total = agg(per_disease)
    print("\n--- Overall (seed 42) ---")
    for k, v in total.items():
        if isinstance(v, float):
            print(f"  {k:22s} = {v:.4f}" if "overall" in k or "bio" in k or "sm" in k else f"  {k:22s} = {v:.3f}")
        else:
            print(f"  {k:22s} = {v}")

    # Per-category delta
    print("\n--- Per-category delta (production - plain kNN) ---")
    by_cat = defaultdict(list)
    for r in per_disease:
        by_cat[r["category"]].append(r)
    print(f"{'category':<18s}  {'n':>4s}  {'boost':>5s}  {'supp':>5s}  "
          f"{'prod_ovr':>9s}  {'plain_ovr':>9s}  {'delta':>7s}  "
          f"{'mean_lost':>9s}  {'mean_lost_hit':>13s}")
    cat_aggregate = {}
    for cat in sorted(by_cat.keys()):
        recs = by_cat[cat]
        a = agg(recs)
        delta_pp = 100.0 * (a["overall_prod"] - a["overall_plain"])
        cat_aggregate[cat] = {
            **a,
            "delta_pp": delta_pp,
            "boost_eligible": cat in SELECTIVE_BOOST_CATEGORIES,
            "supplement_eligible": cat in SUPPLEMENT_CATEGORIES,
        }
        print(f"  {cat:<16s}  {a['n']:>4d}  "
              f"{'Y' if cat in SELECTIVE_BOOST_CATEGORIES else '-':>5s}  "
              f"{'Y' if cat in SUPPLEMENT_CATEGORIES else '-':>5s}  "
              f"{100*a['overall_prod']:>8.2f}%  "
              f"{100*a['overall_plain']:>8.2f}%  "
              f"{delta_pp:>+6.2f}pp  "
              f"{a['mean_lost']:>9.2f}  "
              f"{a['mean_lost_gt_hits']:>13.2f}")

    # Group strata
    print("\n--- Grouped strata ---")
    def grp(pred_filter):
        recs = [r for r in per_disease if pred_filter(r)]
        return len(recs), agg(recs)

    for label, pf in [
        ("boost_eligible", lambda r: r["boost_eligible"]),
        ("NOT boost_eligible", lambda r: not r["boost_eligible"]),
        ("supplement_eligible", lambda r: r["supplement_eligible"]),
        ("neither boost nor supp", lambda r: not r["boost_eligible"] and not r["supplement_eligible"]),
    ]:
        n, a = grp(pf)
        dpp = 100.0 * (a["overall_prod"] - a["overall_plain"])
        print(f"  {label:<25s}  n={n:<4d}  "
              f"prod={100*a['overall_prod']:5.2f}%  "
              f"plain={100*a['overall_plain']:5.2f}%  delta={dpp:+5.2f}pp  "
              f"mean_lost={a['mean_lost']:.2f}  mean_lost_gt={a['mean_lost_gt_hits']:.2f}  "
              f"mean_added_gt={a['mean_added_gt_hits']:.2f}")

    # Biologic vs SM: aggregate lost drugs
    print("\n--- Lost-prediction class breakdown ---")
    total_lost_bio = sum(r["n_lost_bio"] for r in per_disease)
    total_lost_sm = sum(r["n_lost_sm"] for r in per_disease)
    total_lost_bio_gt = sum(r["n_lost_bio_gt"] for r in per_disease)
    total_lost_sm_gt = sum(r["n_lost_sm_gt"] for r in per_disease)
    print(f"  total lost biologic drugs     = {total_lost_bio}")
    print(f"  total lost biologic GT hits   = {total_lost_bio_gt}")
    print(f"  total lost small-mol drugs    = {total_lost_sm}")
    print(f"  total lost small-mol GT hits  = {total_lost_sm_gt}")
    if (total_lost_bio + total_lost_sm) > 0:
        bio_lost_rate_of_bio_in_pool = total_lost_bio / max(1, len(biologic_pool))
        print(f"  biologic fraction of all lost = "
              f"{total_lost_bio/(total_lost_bio+total_lost_sm):.3f}")

    out = {
        "hypothesis": "h952",
        "seed": seed,
        "biologic_pool_size": len(biologic_pool),
        "n_evaluated": len(per_disease),
        "n_skipped": n_skipped,
        "overall": total,
        "per_category": cat_aggregate,
        "per_disease": per_disease,
    }
    out_path = ROOT / "data/analysis/h952_prod_vs_plain_knn.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
