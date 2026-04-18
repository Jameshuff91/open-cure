#!/usr/bin/env python3
"""h912: Is target-overlap ranking a valid repurposing signal at all?

Follow-up to h903. The mechanism-only fallback at production_predictor.py:4663
ranks drugs by |drug_targets ∩ disease_genes|. h903 measured 5.96% prec@30
pooled across all 722 evaluable diseases (below FILTER 9.2%). But that pooled
number hides heterogeneity: certain diseases (e.g. monogenic, small/specific
gene sets) might have much higher signal than pathway-diffuse diseases.

Design:
    - For every disease with a DRKG embedding + non-empty disease_genes,
      compute target-overlap top-30 and measure per-disease R@30 + prec@30
      against expanded ground truth.
    - Attach covariates: disease_genes size, max overlap, top-30 avg overlap,
      disease category, in train_diseases, kNN baseline R@30 (same disease,
      no-tricks top-30 kNN), expanded-GT size.
    - Regress/segment to look for subpopulations with R@30 >= 15% (threshold
      would beat LOW tier 11.3% and make a 'mechanism_strong' annotation viable).
    - If no covariate isolates a usable subset, close the mechanism-ranking
      direction definitively.
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
from sklearn.metrics.pairwise import cosine_similarity  # noqa: E402


def knn_topk_drugs(predictor, disease_id, k=20, top_n=30):
    """Run plain top-k kNN, return ordered list of top_n drug_ids."""
    if disease_id not in predictor.embeddings:
        return []
    test_emb = predictor.embeddings[disease_id].reshape(1, -1)
    sims = cosine_similarity(test_emb, predictor.train_embeddings)[0]
    top_k_idx = np.argsort(sims)[-k:]

    drug_scores = defaultdict(float)
    for idx in top_k_idx:
        neighbor = predictor.train_diseases[idx]
        sim = sims[idx]
        for drug_id in predictor.ground_truth.get(neighbor, []):
            if drug_id in predictor.embeddings:
                drug_scores[drug_id] += sim

    sorted_drugs = sorted(drug_scores.items(), key=lambda x: (-x[1], x[0]))
    return [d for d, _ in sorted_drugs[:top_n]]


def target_topk_drugs(predictor, disease_id, top_n=30):
    """Target-overlap ranking, return (top_n drug list, max_overlap, avg_overlap_top30)."""
    dis_genes = predictor.disease_genes.get(disease_id, set())
    if not dis_genes:
        return [], 0, 0.0
    scored = []
    for drug_id, drug_genes in predictor.drug_targets.items():
        overlap = len(drug_genes & dis_genes)
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
    print("h912: Target-overlap ranking subpopulation analysis")
    print("=" * 70)

    predictor = DrugRepurposingPredictor()
    gt_path = predictor.reference_dir / "expanded_ground_truth.json"
    with open(gt_path) as f:
        expanded_gt = json.load(f)

    # Normalize expanded GT to disease -> set[drug_id]
    gt_by_disease: dict[str, set[str]] = {}
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

    # Candidate diseases: have embedding + have disease_genes + have any GT in expanded
    candidates = [
        d for d in predictor.disease_genes
        if d in predictor.embeddings
        and predictor.disease_genes[d]
        and gt_by_disease.get(d)
    ]
    print(f"Evaluable diseases (embed ∩ genes ∩ expanded GT): {len(candidates)}")

    train_set = set(predictor.train_diseases)

    records = []
    for i, dis_id in enumerate(candidates, 1):
        if i % 100 == 0:
            print(f"  processed {i}/{len(candidates)}")
        dis_genes = predictor.disease_genes[dis_id]
        gt_drugs = gt_by_disease[dis_id]
        # Target-overlap ranking
        target_top30, max_o, avg_o = target_topk_drugs(predictor, dis_id, top_n=30)
        t_hits30 = sum(1 for d in target_top30 if d in gt_drugs)
        t_r30 = t_hits30 / len(gt_drugs) if gt_drugs else 0.0
        t_p30 = t_hits30 / 30.0

        # kNN baseline R@30 — ONLY meaningful if disease in train_diseases, else also valid
        knn_top30 = knn_topk_drugs(predictor, dis_id, k=20, top_n=30)
        k_hits30 = sum(1 for d in knn_top30 if d in gt_drugs)
        k_r30 = k_hits30 / len(gt_drugs) if gt_drugs else 0.0

        # Disease category
        dis_name = predictor.disease_names.get(dis_id, dis_id)
        category = predictor.categorize_disease(dis_name)

        records.append({
            "disease_id": dis_id,
            "disease_name": dis_name,
            "category": category,
            "in_train": dis_id in train_set,
            "gene_set_size": len(dis_genes),
            "gt_size": len(gt_drugs),
            "max_overlap": int(max_o),
            "avg_overlap_top30": avg_o,
            "target_hits30": t_hits30,
            "target_r30": t_r30,
            "target_p30": t_p30,
            "knn_hits30": k_hits30,
            "knn_r30": k_r30,
        })

    n = len(records)
    print(f"Collected {n} per-disease records")

    # Aggregate
    def stats(recs, key="target_r30"):
        vals = [r[key] for r in recs]
        if not vals:
            return {"n": 0, "mean": 0.0, "median": 0.0, "ge15": 0, "ge20": 0}
        return {
            "n": len(vals),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "ge15": sum(1 for v in vals if v >= 0.15),
            "ge20": sum(1 for v in vals if v >= 0.20),
        }

    overall = stats(records)
    print()
    print(f"OVERALL target_r30: mean={overall['mean']*100:.2f}% "
          f"median={overall['median']*100:.2f}% "
          f"ge15%={overall['ge15']}/{n} ({overall['ge15']/n*100:.1f}%) "
          f"ge20%={overall['ge20']}/{n}")

    # Compare to kNN baseline on same records
    k_stats = stats(records, "knn_r30")
    print(f"OVERALL   knn_r30: mean={k_stats['mean']*100:.2f}% "
          f"median={k_stats['median']*100:.2f}% "
          f"ge15%={k_stats['ge15']}/{n}")

    # Segmentation analyses
    print("\n" + "=" * 70)
    print("SEGMENTATION")
    print("=" * 70)

    def segment(records, segname, groups):
        print(f"\n[{segname}]")
        rows = []
        for gname, filt in groups:
            sub = [r for r in records if filt(r)]
            s = stats(sub)
            rows.append((gname, s))
            if s["n"] > 0:
                print(f"  {gname:<30} n={s['n']:4d}  mean={s['mean']*100:5.2f}%  "
                      f"median={s['median']*100:5.2f}%  >=15% {s['ge15']:3d} "
                      f"({(s['ge15']/s['n']*100 if s['n']>0 else 0):4.1f}%)")
        return rows

    # Gene-set size segments
    segment(records, "Gene-set size", [
        ("genes 1-5", lambda r: 1 <= r["gene_set_size"] <= 5),
        ("genes 6-20", lambda r: 6 <= r["gene_set_size"] <= 20),
        ("genes 21-50", lambda r: 21 <= r["gene_set_size"] <= 50),
        ("genes 51-100", lambda r: 51 <= r["gene_set_size"] <= 100),
        ("genes 101-300", lambda r: 101 <= r["gene_set_size"] <= 300),
        ("genes 301+", lambda r: r["gene_set_size"] >= 301),
    ])

    # Max overlap segments
    segment(records, "Max target overlap (top-1)", [
        ("max_ov 1", lambda r: r["max_overlap"] == 1),
        ("max_ov 2", lambda r: r["max_overlap"] == 2),
        ("max_ov 3", lambda r: r["max_overlap"] == 3),
        ("max_ov 4-5", lambda r: 4 <= r["max_overlap"] <= 5),
        ("max_ov 6-10", lambda r: 6 <= r["max_overlap"] <= 10),
        ("max_ov 11+", lambda r: r["max_overlap"] >= 11),
    ])

    # Disease category
    cats = sorted({r["category"] for r in records})
    segment(records, "Category", [
        (c, (lambda r, c=c: r["category"] == c)) for c in cats
    ])

    # In train vs not
    segment(records, "In train_diseases", [
        ("in_train", lambda r: r["in_train"]),
        ("not_in_train", lambda r: not r["in_train"]),
    ])

    # kNN baseline buckets (in-train only for fair kNN comparison)
    in_train_recs = [r for r in records if r["in_train"]]
    segment(in_train_recs, "kNN R@30 bucket (in-train)", [
        ("kNN_r30==0", lambda r: r["knn_r30"] == 0),
        ("kNN_r30 (0,0.1]", lambda r: 0 < r["knn_r30"] <= 0.1),
        ("kNN_r30 (0.1,0.3]", lambda r: 0.1 < r["knn_r30"] <= 0.3),
        ("kNN_r30 (0.3,0.6]", lambda r: 0.3 < r["knn_r30"] <= 0.6),
        ("kNN_r30 >0.6", lambda r: r["knn_r30"] > 0.6),
    ])

    # GT size buckets
    segment(records, "GT size bucket", [
        ("gt 1-3", lambda r: 1 <= r["gt_size"] <= 3),
        ("gt 4-10", lambda r: 4 <= r["gt_size"] <= 10),
        ("gt 11-30", lambda r: 11 <= r["gt_size"] <= 30),
        ("gt 31-100", lambda r: 31 <= r["gt_size"] <= 100),
        ("gt 101+", lambda r: r["gt_size"] >= 101),
    ])

    # Cross: small gene-set AND high max_overlap
    segment(records, "Cross: small genes + high top-1 overlap", [
        ("genes<=20 & max_ov>=3", lambda r: r["gene_set_size"] <= 20 and r["max_overlap"] >= 3),
        ("genes<=20 & max_ov>=5", lambda r: r["gene_set_size"] <= 20 and r["max_overlap"] >= 5),
        ("genes 21-100 & max_ov>=5", lambda r: 21 <= r["gene_set_size"] <= 100 and r["max_overlap"] >= 5),
        ("genes>100 & max_ov>=10", lambda r: r["gene_set_size"] > 100 and r["max_overlap"] >= 10),
    ])

    # Top-10 diseases by target R@30
    print("\nTop 15 diseases by target_r30 (hits>0):")
    top = sorted([r for r in records if r["target_hits30"] > 0],
                 key=lambda r: -r["target_r30"])[:15]
    print(f"  {'disease':<48} {'cat':<14} {'genes':>5} {'gt':>4} {'max_ov':>6} "
          f"{'t_r30':>6} {'k_r30':>6}")
    for r in top:
        nm = (r["disease_name"] or r["disease_id"])[:47]
        print(f"  {nm:<48} {r['category']:<14} {r['gene_set_size']:>5} "
              f"{r['gt_size']:>4} {r['max_overlap']:>6} "
              f"{r['target_r30']*100:>5.1f}% {r['knn_r30']*100:>5.1f}%")

    # Conclusion: does any segment cross R@30 >= 15% with n >= 30?
    actionable = []
    segments_tested = {
        "genes 6-20": [r for r in records if 6 <= r["gene_set_size"] <= 20],
        "genes 21-50": [r for r in records if 21 <= r["gene_set_size"] <= 50],
        "max_ov 4-5": [r for r in records if 4 <= r["max_overlap"] <= 5],
        "max_ov 6-10": [r for r in records if 6 <= r["max_overlap"] <= 10],
        "max_ov 11+": [r for r in records if r["max_overlap"] >= 11],
        "genes<=20 & max_ov>=3": [r for r in records if r["gene_set_size"] <= 20 and r["max_overlap"] >= 3],
        "genes<=20 & max_ov>=5": [r for r in records if r["gene_set_size"] <= 20 and r["max_overlap"] >= 5],
        "genes 21-100 & max_ov>=5": [r for r in records if 21 <= r["gene_set_size"] <= 100 and r["max_overlap"] >= 5],
    }
    for name, sub in segments_tested.items():
        if not sub:
            continue
        s = stats(sub)
        if s["n"] >= 30 and s["mean"] >= 0.15:
            actionable.append({"segment": name, **s})

    # Save outputs
    out = {
        "hypothesis": "h912",
        "title": "Is target-overlap ranking a valid repurposing signal at all?",
        "n_evaluable_diseases": n,
        "overall_target": overall,
        "overall_knn": k_stats,
        "actionable_segments_R30_ge15_n30": actionable,
        "conclusion": (
            "No actionable subpopulation (R@30>=15% @ n>=30)"
            if not actionable
            else f"Found {len(actionable)} actionable segment(s)"
        ),
    }
    out_dir = ROOT / "data/analysis"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "h912_subpopulation_summary.json", "w") as f:
        json.dump(out, f, indent=2)

    # Also dump raw per-disease records for reuse
    with open(out_dir / "h912_per_disease_records.json", "w") as f:
        json.dump(records, f, indent=2)

    print()
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    if actionable:
        print("Actionable segments (mean R@30 >= 15%, n >= 30):")
        for a in actionable:
            print(f"  {a['segment']}: n={a['n']}, mean R@30={a['mean']*100:.1f}%")
    else:
        print("No covariate isolates a usable subpopulation for target-overlap ranking.")
        print("Target-overlap ranking is NOT a valid standalone repurposing signal.")

    print(f"\nWrote: {out_dir / 'h912_subpopulation_summary.json'}")
    print(f"Wrote: {out_dir / 'h912_per_disease_records.json'}")


if __name__ == "__main__":
    main()
