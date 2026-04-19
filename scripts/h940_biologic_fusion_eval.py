#!/usr/bin/env python3
"""h940: Biologic target-overlap + kNN fusion — holdout per-drug R@30 sweep.

Hypothesis (h939 follow-up):
    h939 showed target-overlap ranks biologic drugs 3-11x better than small
    molecules within several disease categories. But biologic precision@30 is
    similar to SM (2.24% vs 2.63%) because the biologic pool is small. If we
    fuse target-overlap with kNN *only for biologic candidates*, can we move
    the biologic R@30 baseline (27.3%) up by >=5pp without overall regression?

Design:
    Reuse h393's 80/20 disease holdout framework (5 seeds).
    For each holdout disease (has embedding + gene set + GT):
      - Compute kNN scores over training-only neighbors (k=20, no category boost,
        no MinRank — we want a clean baseline)
      - Compute target-overlap for biologic pool
      - Normalize per-disease:
          knn_norm[d] = knn_score[d] / max_knn
          overlap_norm[d] = overlap[d] / max_overlap (biologics only)
      - Scoring schemes:
          baseline:          rank by knn_norm
          bio_fusion(alpha): biologics -> alpha*knn_norm + (1-alpha)*overlap_norm
                             small molecules -> knn_norm
    Metrics (per disease):
      - bio_r30  = |top30 ∩ bio_GT| / |bio_GT|      (requires |bio_GT|>=1)
      - sm_r30   = |top30 ∩ sm_GT| / |sm_GT|        (requires |sm_GT|>=1)
      - overall_r30 = |top30 ∩ GT| / |GT|

    The top-30 set for each metric is from a single unified ranking over all
    drugs (biologics scored via fusion, SMs scored via kNN_norm); bio/sm_r30
    count hits among all top-30 that fall in the biologic/SM GT respectively.

Decision:
    Success: fusion biologic_r30 increases >=5pp vs baseline biologic_r30,
    AND overall_r30 |delta| < 1pp.
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

from production_predictor import DrugRepurposingPredictor  # noqa: E402

BIOLOGIC_SUFFIXES = (
    "mab", "cept", "kinra", "parin", "tide", "plase", "tropin",
    "relin", "genase", "nase", "pase", "lase", "streptim", "ferax",
)
BIOLOGIC_KEYWORDS = (
    "insulin", "interferon", "erythropoietin", "antibody", "immunoglobulin",
    "globulin", "vaccine", "heparin", "filgrastim", "streptokinase",
    "urokinase", "factor viii", "factor ix", "factor xiii", "von willebrand",
    "interleukin", "alteplase", "darbepoetin", "botulinum", "glucagon",
    "somatropin", "somatostatin", "hyaluronidase", "chymotrypsin",
    "collagenase", "trypsin", "pancrelipase", "pancreatin",
)


def is_biologic(drug_name: str | None) -> bool:
    if not drug_name:
        return False
    nm = drug_name.lower().strip()
    for suf in BIOLOGIC_SUFFIXES:
        if nm.endswith(suf):
            return True
    for kw in BIOLOGIC_KEYWORDS:
        if kw in nm:
            return True
    return False


def split_diseases(all_diseases: List[str], seed: int, train_ratio: float = 0.8):
    rng = np.random.RandomState(seed)
    shuffled = list(all_diseases)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def compute_knn_scores(
    predictor: DrugRepurposingPredictor,
    disease_id: str,
    train_diseases: List[str],
    train_embeddings: np.ndarray,
    k: int,
) -> Dict[str, float]:
    """Plain kNN scoring: sum of neighbor similarities for each drug in neighbor GT."""
    test_emb = predictor.embeddings[disease_id].reshape(1, -1)
    sims = cosine_similarity(test_emb, train_embeddings)[0]
    top_k_idx = np.argsort(sims)[-k:]
    drug_scores: Dict[str, float] = defaultdict(float)
    for idx in top_k_idx:
        neighbor = train_diseases[idx]
        neighbor_sim = sims[idx]
        for drug_id in predictor.ground_truth.get(neighbor, set()):
            if drug_id in predictor.embeddings:
                drug_scores[drug_id] += float(neighbor_sim)
    return dict(drug_scores)


def compute_target_overlap(
    predictor: DrugRepurposingPredictor,
    disease_id: str,
    pool: Set[str],
) -> Dict[str, int]:
    """Raw target-overlap counts restricted to a drug pool."""
    disease_genes = predictor.disease_genes.get(disease_id, set())
    if not disease_genes:
        return {}
    overlap: Dict[str, int] = {}
    for drug_id in pool:
        d_tgt = predictor.drug_targets.get(drug_id)
        if not d_tgt:
            continue
        c = len(d_tgt & disease_genes)
        if c > 0:
            overlap[drug_id] = c
    return overlap


def evaluate_seed(
    predictor: DrugRepurposingPredictor,
    expanded_gt: Dict[str, Set[str]],
    biologic_pool: Set[str],
    seed: int,
    alphas: List[float],
    k: int = 20,
    top_n: int = 30,
) -> Dict:
    all_diseases = [
        d for d in predictor.ground_truth
        if d in predictor.embeddings and expanded_gt.get(d)
    ]
    train_ids, holdout_ids = split_diseases(all_diseases, seed)
    train_set = set(train_ids)

    # Build training-only kNN index
    train_disease_ids = [d for d in train_ids if d in predictor.embeddings]
    train_embeddings = np.array(
        [predictor.embeddings[d] for d in train_disease_ids], dtype=np.float32
    )

    # Temporarily swap ground_truth used by kNN aggregation to training-only
    original_gt = predictor.ground_truth
    train_only_gt = {d: v for d, v in original_gt.items() if d in train_set}
    predictor.ground_truth = train_only_gt

    # Per-alpha aggregators
    # key = scheme name ("baseline" or f"alpha_{a:.1f}")
    per_scheme: Dict[str, Dict[str, List[float]]] = {
        s: {"bio_r30": [], "sm_r30": [], "overall_r30": [], "bio_p30": []}
        for s in ["baseline"] + [f"alpha_{a:.1f}" for a in alphas]
    }

    n_eval = 0
    n_with_bio_gt = 0
    n_with_sm_gt = 0

    try:
        for dis_id in holdout_ids:
            if dis_id not in predictor.embeddings:
                continue
            gt_drugs = expanded_gt.get(dis_id, set())
            if not gt_drugs:
                continue
            bio_gt = gt_drugs & biologic_pool
            sm_gt = gt_drugs - biologic_pool

            knn_scores = compute_knn_scores(
                predictor, dis_id, train_disease_ids, train_embeddings, k=k
            )
            if not knn_scores:
                # Zero-coverage disease; still evaluate (overlap alone can score)
                max_knn = 0.0
            else:
                max_knn = max(knn_scores.values())

            # Union of candidates: any drug that has some signal (knn or overlap
            # target if biologic). This keeps eval compact.
            overlap = compute_target_overlap(predictor, dis_id, biologic_pool)
            max_overlap = max(overlap.values()) if overlap else 0

            # Candidate set: all drugs with kNN score + biologics with overlap
            candidates = set(knn_scores) | set(overlap)
            if not candidates:
                continue

            n_eval += 1
            if bio_gt:
                n_with_bio_gt += 1
            if sm_gt:
                n_with_sm_gt += 1

            # Normalized per-disease
            knn_norm = {
                d: (knn_scores.get(d, 0.0) / max_knn) if max_knn > 0 else 0.0
                for d in candidates
            }
            overlap_norm = {
                d: (overlap.get(d, 0) / max_overlap) if max_overlap > 0 else 0.0
                for d in candidates
            }

            # Baseline: pure kNN
            baseline_scored = [(d, knn_norm[d]) for d in candidates]
            baseline_scored.sort(key=lambda x: (-x[1], x[0]))
            top30_base = [d for d, _ in baseline_scored[:top_n]]
            _record(per_scheme["baseline"], top30_base, gt_drugs, bio_gt, sm_gt, biologic_pool)

            # Fusion schemes per alpha (biologics fused; SMs kNN-only)
            for a in alphas:
                scored = []
                for d in candidates:
                    if d in biologic_pool:
                        s = a * knn_norm[d] + (1 - a) * overlap_norm[d]
                    else:
                        s = knn_norm[d]
                    scored.append((d, s))
                scored.sort(key=lambda x: (-x[1], x[0]))
                top30 = [d for d, _ in scored[:top_n]]
                _record(
                    per_scheme[f"alpha_{a:.1f}"], top30, gt_drugs, bio_gt, sm_gt, biologic_pool
                )
    finally:
        predictor.ground_truth = original_gt

    return {
        "seed": seed,
        "n_holdout": len(holdout_ids),
        "n_evaluated": n_eval,
        "n_with_bio_gt": n_with_bio_gt,
        "n_with_sm_gt": n_with_sm_gt,
        "schemes": {
            scheme: {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}
            for scheme, metrics in per_scheme.items()
        },
    }


def _record(bucket, top30, gt, bio_gt, sm_gt, biologic_pool):
    top30_set = set(top30)
    hits = top30_set & gt
    # per-drug R@30 (overall)
    bucket["overall_r30"].append(len(hits) / len(gt) if gt else 0.0)
    # biologic R@30 (among biologic GT, how many are in the top-30 (which may be mixed))
    if bio_gt:
        bio_hits = top30_set & bio_gt
        bucket["bio_r30"].append(len(bio_hits) / len(bio_gt))
        # also track biologic precision within biologic subset of top30
        bio_in_top = top30_set & biologic_pool
        bucket["bio_p30"].append(len(bio_hits) / len(bio_in_top) if bio_in_top else 0.0)
    if sm_gt:
        sm_hits = top30_set & sm_gt
        bucket["sm_r30"].append(len(sm_hits) / len(sm_gt))


def main():
    alphas = [0.3, 0.5, 0.7, 0.9]
    seeds = [42, 123, 456, 789, 2024]

    print("=" * 78)
    print("h940: Biologic target-overlap + kNN fusion — 5-seed holdout R@30")
    print("=" * 78)

    print("Loading predictor ...")
    predictor = DrugRepurposingPredictor()

    gt_path = predictor.reference_dir / "expanded_ground_truth.json"
    with open(gt_path) as f:
        raw = json.load(f)
    expanded_gt: Dict[str, Set[str]] = {}
    for dis_id, drugs in raw.items():
        s: Set[str] = set()
        for d in drugs:
            if isinstance(d, str):
                s.add(d)
            elif isinstance(d, dict):
                did = d.get("drug_id") or d.get("drug")
                if did:
                    s.add(did)
        expanded_gt[dis_id] = s

    biologic_pool = {
        d for d in predictor.drug_targets
        if is_biologic(predictor.drug_id_to_name.get(d))
    }
    print(f"Biologic pool size: {len(biologic_pool)} of "
          f"{len(predictor.drug_targets)} drugs with targets")

    # Aggregators
    agg = defaultdict(lambda: {"bio_r30": [], "sm_r30": [], "overall_r30": [], "bio_p30": []})
    seed_results = []
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        res = evaluate_seed(predictor, expanded_gt, biologic_pool, seed, alphas)
        seed_results.append(res)
        print(f"  evaluated={res['n_evaluated']}  "
              f"bio_gt_present={res['n_with_bio_gt']}  "
              f"sm_gt_present={res['n_with_sm_gt']}")
        for scheme, metrics in res["schemes"].items():
            print(f"  {scheme:<12s} "
                  f"bio_r30={100*metrics['bio_r30']:5.2f}%  "
                  f"sm_r30={100*metrics['sm_r30']:5.2f}%  "
                  f"overall_r30={100*metrics['overall_r30']:5.2f}%")
            for m, v in metrics.items():
                agg[scheme][m].append(v)

    # Aggregate
    print("\n" + "=" * 78)
    print("AGGREGATE (mean ± std across 5 seeds)")
    print("=" * 78)
    print(f"{'scheme':<12s}  {'bio_r30':>16s}  {'sm_r30':>16s}  "
          f"{'overall_r30':>16s}  {'bio_p30':>16s}")
    summary = {}
    for scheme in ["baseline"] + [f"alpha_{a:.1f}" for a in alphas]:
        metrics = agg[scheme]
        row = {}
        parts = [scheme]
        for m in ["bio_r30", "sm_r30", "overall_r30", "bio_p30"]:
            vals = metrics[m]
            mean = float(np.mean(vals)) if vals else 0.0
            std = float(np.std(vals)) if vals else 0.0
            row[m] = {"mean": mean, "std": std}
            parts.append(f"{100*mean:6.2f}% ± {100*std:4.2f}%")
        summary[scheme] = row
        print(f"  {parts[0]:<10s}  {parts[1]:>16s}  {parts[2]:>16s}  "
              f"{parts[3]:>16s}  {parts[4]:>16s}")

    # Decision
    print("\n" + "=" * 78)
    print("DECISION")
    print("=" * 78)
    base_bio = summary["baseline"]["bio_r30"]["mean"]
    base_overall = summary["baseline"]["overall_r30"]["mean"]
    winners = []
    for a in alphas:
        key = f"alpha_{a:.1f}"
        bio = summary[key]["bio_r30"]["mean"]
        overall = summary[key]["overall_r30"]["mean"]
        d_bio = 100 * (bio - base_bio)
        d_over = 100 * (overall - base_overall)
        verdict = "WIN" if (d_bio >= 5.0 and abs(d_over) < 1.0) else "NO"
        print(f"  {key}: Δbio_r30={d_bio:+5.2f}pp  Δoverall_r30={d_over:+5.2f}pp  [{verdict}]")
        if verdict == "WIN":
            winners.append((a, d_bio, d_over))

    out_path = ROOT / "data/analysis/h940_biologic_fusion_eval.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "hypothesis": "h940",
                "title": "Biologic target-overlap + kNN fusion",
                "seeds": seeds,
                "alphas": alphas,
                "biologic_pool_size": len(biologic_pool),
                "seed_results": seed_results,
                "summary": summary,
                "winners": winners,
            },
            f,
            indent=2,
        )
    print(f"\nSaved summary -> {out_path}")


if __name__ == "__main__":
    main()
