#!/usr/bin/env python3
"""h939: Audit biologics-only subset — is target-overlap actually a biologic signal?

Context:
    h906 (DrugBank target features) motivates itself with "biologics fail at 27.3%
    R@30 vs 41.8% overall." h912 showed target-overlap ranking is a weak signal
    overall (5.96% prec@30). h916 showed the cancer concentration is not a
    drug-target-density artifact.

    Question: does target-overlap work *for biologics specifically*? Each mAb has
    one-to-few high-affinity targets, and target identity IS the mechanism —
    unlike small molecules whose polypharmacy dilutes the overlap signal. If
    biologic_target_r30 >> small_molecule_target_r30 in multiple categories,
    h906 is still viable. If not, h906 should be archived and effort diverted
    to h920 (PubMedBERT) / h921 (ESM2).

Design:
    For every disease with embedding + non-empty disease_genes + any GT:
      - Partition candidate drug pool into biologic vs small-molecule
      - Compute target-overlap top-30 within each pool
      - Compute target_r30 against the matching-pool GT subset
    Stratify by disease category and report biologic_target_r30 vs
    small_molecule_target_r30 delta.

Decision rule (from research_roadmap.json h939):
    If biologic_target_r30 >= 3x small_molecule_target_r30 in >=3 categories
    (each with n>=10 diseases contributing >=1 biologic GT), h906 motivation
    holds. Otherwise mark h906 as likely-to-fail.

Biologic flag:
    USAN-suffix + protein-name keywords (see is_biologic()). No DrugBank
    academic license available, so we use a naming-convention proxy. 266
    biologics in drug_targets (of 11,656 total).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

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


def target_topk_within_pool(
    disease_genes: set,
    drug_targets: dict,
    pool: set,
    top_n: int = 30,
) -> tuple[list[str], int, float]:
    """Target-overlap ranking restricted to a drug pool."""
    scored = []
    for drug_id in pool:
        d_tgt = drug_targets.get(drug_id)
        if not d_tgt:
            continue
        overlap = len(d_tgt & disease_genes)
        if overlap > 0:
            scored.append((drug_id, overlap))
    scored.sort(key=lambda x: (-x[1], x[0]))
    top = scored[:top_n]
    drug_ids = [d for d, _ in top]
    max_o = top[0][1] if top else 0
    avg_o = float(np.mean([o for _, o in top])) if top else 0.0
    return drug_ids, max_o, avg_o


def main():
    print("=" * 70)
    print("h939: Biologic vs small-molecule target-overlap audit")
    print("=" * 70)

    predictor = DrugRepurposingPredictor()
    gt_path = predictor.reference_dir / "expanded_ground_truth.json"
    with open(gt_path) as f:
        expanded_gt = json.load(f)

    # Normalize expanded GT to disease -> set[drug_id]
    gt_by_disease: dict[str, set] = {}
    for dis_id, drugs in expanded_gt.items():
        s = set()
        for d in drugs:
            if isinstance(d, str):
                s.add(d)
            elif isinstance(d, dict):
                did = d.get("drug_id") or d.get("drug")
                if did:
                    s.add(did)
        gt_by_disease[dis_id] = s

    # Build biologic / small-molecule partitions of drug_targets candidate pool
    biologic_pool = {
        d for d in predictor.drug_targets
        if is_biologic(predictor.drug_id_to_name.get(d))
    }
    sm_pool = {d for d in predictor.drug_targets if d not in biologic_pool}
    print(f"Biologic pool size: {len(biologic_pool)} of {len(predictor.drug_targets)} "
          f"({100*len(biologic_pool)/len(predictor.drug_targets):.1f}%)")

    # Candidate diseases
    candidates = [
        d for d in predictor.disease_genes
        if d in predictor.embeddings
        and predictor.disease_genes[d]
        and gt_by_disease.get(d)
    ]
    print(f"Candidate diseases (embed ∩ genes ∩ expanded GT): {len(candidates)}")

    records = []
    for i, dis_id in enumerate(candidates, 1):
        if i % 500 == 0:
            print(f"  processed {i}/{len(candidates)}")
        dis_genes = predictor.disease_genes[dis_id]
        gt_drugs = gt_by_disease[dis_id]
        bio_gt = gt_drugs & biologic_pool
        sm_gt = gt_drugs & sm_pool

        # Biologic pool ranking
        bio_top30, bio_max_o, bio_avg_o = target_topk_within_pool(
            dis_genes, predictor.drug_targets, biologic_pool, top_n=30
        )
        bio_hits = sum(1 for d in bio_top30 if d in bio_gt)
        bio_r30 = bio_hits / len(bio_gt) if bio_gt else None
        bio_p30 = bio_hits / 30.0 if bio_top30 else 0.0

        # Small-molecule pool ranking
        sm_top30, sm_max_o, sm_avg_o = target_topk_within_pool(
            dis_genes, predictor.drug_targets, sm_pool, top_n=30
        )
        sm_hits = sum(1 for d in sm_top30 if d in sm_gt)
        sm_r30 = sm_hits / len(sm_gt) if sm_gt else None
        sm_p30 = sm_hits / 30.0 if sm_top30 else 0.0

        dis_name = predictor.disease_names.get(dis_id, dis_id)
        category = predictor.categorize_disease(dis_name)

        records.append({
            "disease_id": dis_id,
            "disease_name": dis_name,
            "category": category,
            "gene_set_size": len(dis_genes),
            "gt_size": len(gt_drugs),
            "bio_gt_size": len(bio_gt),
            "sm_gt_size": len(sm_gt),
            "bio_hits30": bio_hits,
            "bio_r30": bio_r30,
            "bio_p30": bio_p30,
            "bio_max_overlap": int(bio_max_o),
            "sm_hits30": sm_hits,
            "sm_r30": sm_r30,
            "sm_p30": sm_p30,
            "sm_max_overlap": int(sm_max_o),
        })

    print(f"Collected {len(records)} per-disease records")

    # Summaries (restrict to diseases with >=1 biologic GT for bio stats)
    bio_recs = [r for r in records if r["bio_gt_size"] >= 1]
    sm_recs = [r for r in records if r["sm_gt_size"] >= 1]
    bio_recs_5 = [r for r in records if r["bio_gt_size"] >= 5]

    def mean(lst):
        lst = [x for x in lst if x is not None]
        return float(np.mean(lst)) if lst else 0.0

    def median(lst):
        lst = [x for x in lst if x is not None]
        return float(np.median(lst)) if lst else 0.0

    print()
    print("=" * 70)
    print("OVERALL")
    print("=" * 70)
    print(f"Diseases with >=1 biologic GT: {len(bio_recs)}")
    print(f"  mean bio_r30    = {100*mean(r['bio_r30'] for r in bio_recs):.2f}%")
    print(f"  median bio_r30  = {100*median(r['bio_r30'] for r in bio_recs):.2f}%")
    print(f"  mean bio_p30    = {100*mean(r['bio_p30'] for r in bio_recs):.2f}%")
    print(f"Diseases with >=5 biologic GT: {len(bio_recs_5)}")
    print(f"  mean bio_r30    = {100*mean(r['bio_r30'] for r in bio_recs_5):.2f}%")
    print(f"  median bio_r30  = {100*median(r['bio_r30'] for r in bio_recs_5):.2f}%")
    print(f"Diseases with >=1 SM GT: {len(sm_recs)}")
    print(f"  mean sm_r30     = {100*mean(r['sm_r30'] for r in sm_recs):.2f}%")
    print(f"  median sm_r30   = {100*median(r['sm_r30'] for r in sm_recs):.2f}%")
    print(f"  mean sm_p30     = {100*mean(r['sm_p30'] for r in sm_recs):.2f}%")

    # Category stratification
    print()
    print("=" * 70)
    print("PER-CATEGORY (require >=10 diseases with >=1 biologic GT)")
    print("=" * 70)
    cats = defaultdict(list)
    for r in records:
        cats[r["category"]].append(r)

    cat_rows = []
    for cat, recs in sorted(cats.items()):
        bio_sub = [r for r in recs if r["bio_gt_size"] >= 1]
        sm_sub = [r for r in recs if r["sm_gt_size"] >= 1]
        if not bio_sub:
            continue
        bio_r = mean(r["bio_r30"] for r in bio_sub)
        sm_r = mean(r["sm_r30"] for r in sm_sub)
        bio_p = mean(r["bio_p30"] for r in bio_sub)
        sm_p = mean(r["sm_p30"] for r in sm_sub)
        ratio = bio_r / sm_r if sm_r > 0 else float("inf")
        cat_rows.append({
            "category": cat,
            "n_diseases_bio": len(bio_sub),
            "n_diseases_sm": len(sm_sub),
            "bio_r30_mean": bio_r,
            "sm_r30_mean": sm_r,
            "bio_p30_mean": bio_p,
            "sm_p30_mean": sm_p,
            "ratio": ratio if ratio != float("inf") else None,
        })

    cat_rows.sort(key=lambda x: -x["bio_r30_mean"])
    print(f"{'category':<18} {'n_bio':>6} {'n_sm':>6} {'bio_r30':>9} {'sm_r30':>8} "
          f"{'ratio':>6} {'bio_p30':>9} {'sm_p30':>8}")
    for row in cat_rows:
        ratio_s = f"{row['ratio']:.2f}x" if row["ratio"] is not None else "inf"
        print(f"  {row['category']:<16} {row['n_diseases_bio']:>6} "
              f"{row['n_diseases_sm']:>6} {100*row['bio_r30_mean']:>7.2f}% "
              f"{100*row['sm_r30_mean']:>6.2f}% {ratio_s:>6} "
              f"{100*row['bio_p30_mean']:>7.2f}% {100*row['sm_p30_mean']:>6.2f}%")

    # Decision rule: biologic_r30 >= 3x sm_r30 in >=3 categories with n>=10 bio diseases
    decisive_cats = [
        r for r in cat_rows
        if r["n_diseases_bio"] >= 10
        and r["ratio"] is not None
        and r["ratio"] >= 3.0
    ]
    print()
    print("=" * 70)
    print("DECISION")
    print("=" * 70)
    print(f"Categories with n_bio>=10 AND bio_r30 >= 3x sm_r30: {len(decisive_cats)}")
    for r in decisive_cats:
        print(f"  {r['category']}: n={r['n_diseases_bio']}, "
              f"bio_r30={100*r['bio_r30_mean']:.2f}%, "
              f"sm_r30={100*r['sm_r30_mean']:.2f}%, ratio={r['ratio']:.2f}x")

    if len(decisive_cats) >= 3:
        verdict = (
            "h906 motivation HOLDS: biologic target-overlap >> small-molecule "
            "in multiple categories."
        )
    else:
        verdict = (
            "h906 motivation WEAK: biologic target-overlap does NOT substantially "
            "exceed small-molecule signal in 3+ categories. Recommend archiving "
            "h906 in favor of h920 (PubMedBERT) / h921 (ESM2)."
        )
    print()
    print("VERDICT:", verdict)

    # Top diseases where biologic target-overlap wins
    print()
    print("Top 15 diseases where biologic target-overlap hits (bio_r30 ranked):")
    winners = sorted(
        [r for r in records if r["bio_gt_size"] >= 3 and r["bio_hits30"] > 0],
        key=lambda r: -(r["bio_r30"] or 0.0),
    )[:15]
    print(f"  {'disease':<42} {'cat':<14} {'bio_gt':>6} {'bio_hits':>8} "
          f"{'bio_r30':>8} {'sm_r30':>7}")
    for r in winners:
        nm = (r["disease_name"] or r["disease_id"])[:41]
        sm_r = r["sm_r30"] if r["sm_r30"] is not None else 0.0
        print(f"  {nm:<42} {r['category']:<14} {r['bio_gt_size']:>6} "
              f"{r['bio_hits30']:>8} {100*(r['bio_r30'] or 0):>6.1f}% "
              f"{100*sm_r:>5.1f}%")

    # Write outputs
    out_dir = ROOT / "data/analysis"
    out_dir.mkdir(exist_ok=True, parents=True)
    summary = {
        "hypothesis": "h939",
        "title": "Audit biologics-only subset target-overlap signal",
        "n_evaluable": len(records),
        "n_biologic_pool": len(biologic_pool),
        "n_sm_pool": len(sm_pool),
        "n_dis_bio_gt_ge1": len(bio_recs),
        "n_dis_bio_gt_ge5": len(bio_recs_5),
        "overall_bio_r30_mean": mean(r["bio_r30"] for r in bio_recs),
        "overall_bio_r30_median": median(r["bio_r30"] for r in bio_recs),
        "overall_bio_p30_mean": mean(r["bio_p30"] for r in bio_recs),
        "overall_sm_r30_mean": mean(r["sm_r30"] for r in sm_recs),
        "overall_sm_r30_median": median(r["sm_r30"] for r in sm_recs),
        "overall_sm_p30_mean": mean(r["sm_p30"] for r in sm_recs),
        "category_rows": cat_rows,
        "decisive_categories_ratio_ge3_n_ge10": decisive_cats,
        "verdict": verdict,
    }
    with open(out_dir / "h939_biologic_target_overlap_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "h939_biologic_target_overlap_records.json", "w") as f:
        json.dump(records, f, indent=2)
    print()
    print(f"Wrote: {out_dir / 'h939_biologic_target_overlap_summary.json'}")
    print(f"Wrote: {out_dir / 'h939_biologic_target_overlap_records.json'}")


if __name__ == "__main__":
    main()
