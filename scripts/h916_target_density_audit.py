"""h916: Audit whether target-overlap concentration in cancer is explained by
drug-target density rather than biological signal.

For each disease category, compute:
  1. mean |drug_targets| per drug indicated in that category (expanded GT)
  2. mean target_r30 from h912 per-disease records

Correlate (1) vs (2) across categories. If Pearson r > 0.7, target-overlap is
primarily a drug-target-density artifact, not a category-specific biological
signal. Use result to scope h906 DrugBank priority.
"""

import json
import math
from collections import defaultdict
from pathlib import Path


def extract_drug_id(drkg_id: str) -> str:
    prefix = "drkg:Compound::"
    return drkg_id[len(prefix):] if drkg_id.startswith(prefix) else drkg_id


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def spearman(xs, ys):
    def rank(vals):
        s = sorted(enumerate(vals), key=lambda t: t[1])
        r = [0.0] * len(vals)
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and s[j + 1][1] == s[i][1]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[s[k][0]] = avg
            i = j + 1
        return r

    return pearson(rank(xs), rank(ys))


def main():
    repo = Path(__file__).resolve().parents[1]

    with open(repo / "data/reference/drug_targets.json") as f:
        drug_targets = json.load(f)
    with open(repo / "data/reference/expanded_ground_truth.json") as f:
        expanded_gt = json.load(f)
    with open(repo / "data/analysis/h912_per_disease_records.json") as f:
        records = json.load(f)

    by_cat = defaultdict(list)
    for r in records:
        cat = r.get("category") or "unknown"
        by_cat[cat].append(r)

    rows = []
    for cat, rs in by_cat.items():
        n = len(rs)
        target_vals = [r["target_r30"] for r in rs]
        knn_vals = [r["knn_r30"] for r in rs]
        gene_sizes = [r.get("gene_set_size") or 0 for r in rs]

        drugs_seen = set()
        for r in rs:
            for d in expanded_gt.get(r["disease_id"], []):
                drugs_seen.add(extract_drug_id(d))

        drugs_with_targets = [d for d in drugs_seen if d in drug_targets]
        target_counts = [len(drug_targets[d]) for d in drugs_with_targets]

        rows.append({
            "category": cat,
            "n_diseases": n,
            "mean_target_r30": sum(target_vals) / n,
            "median_target_r30": sorted(target_vals)[n // 2],
            "frac_ge15_target_r30": sum(1 for v in target_vals if v >= 0.15) / n,
            "mean_knn_r30": sum(knn_vals) / n,
            "mean_gene_set_size": sum(gene_sizes) / n if gene_sizes else 0,
            "n_gt_drugs": len(drugs_seen),
            "n_gt_drugs_with_targets": len(drugs_with_targets),
            "coverage_gt_drugs": (len(drugs_with_targets) / len(drugs_seen))
                                 if drugs_seen else 0,
            "mean_drug_targets": (sum(target_counts) / len(target_counts))
                                 if target_counts else 0,
            "median_drug_targets": (sorted(target_counts)[len(target_counts) // 2])
                                   if target_counts else 0,
            "p90_drug_targets": (sorted(target_counts)[int(0.9 * len(target_counts))])
                                if target_counts else 0,
        })

    rows.sort(key=lambda r: -r["n_diseases"])

    # Correlations — restrict to categories with n_diseases >= 10 for stability
    stable = [r for r in rows if r["n_diseases"] >= 10]
    xs_targets = [r["mean_drug_targets"] for r in stable]
    ys_r30 = [r["mean_target_r30"] for r in stable]
    xs_genes = [r["mean_gene_set_size"] for r in stable]
    xs_knn = [r["mean_knn_r30"] for r in stable]

    corr = {
        "n_categories_used": len(stable),
        "categories_used": [r["category"] for r in stable],
        "pearson_target_r30_vs_mean_drug_targets": pearson(xs_targets, ys_r30),
        "spearman_target_r30_vs_mean_drug_targets": spearman(xs_targets, ys_r30),
        "pearson_target_r30_vs_mean_gene_set_size": pearson(xs_genes, ys_r30),
        "spearman_target_r30_vs_mean_gene_set_size": spearman(xs_genes, ys_r30),
        "pearson_target_r30_vs_knn_r30": pearson(xs_knn, ys_r30),
        "pearson_drug_targets_vs_gene_set": pearson(xs_targets, xs_genes),
    }

    output = {
        "hypothesis": "h916",
        "title": "Is target-overlap concentration in cancer a drug-target-density artifact?",
        "per_category": rows,
        "correlations_n_ge10": corr,
        "interpretation": {
            "primary_test": "pearson_target_r30_vs_mean_drug_targets",
            "threshold_density_artifact": 0.7,
            "threshold_biology": 0.4,
        },
    }

    out_path = repo / "data/analysis/h916_target_density_audit.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {out_path}")

    print("\n=== Per-category summary (sorted by n_diseases) ===")
    print(f"{'category':<18} {'n':>5} {'target_r30':>11} {'knn_r30':>8} "
          f"{'mean_tgt':>9} {'med_tgt':>8} {'n_drugs':>8} {'cov':>6}")
    for r in rows:
        print(f"{r['category']:<18} {r['n_diseases']:>5} "
              f"{r['mean_target_r30']:>11.4f} {r['mean_knn_r30']:>8.4f} "
              f"{r['mean_drug_targets']:>9.2f} {r['median_drug_targets']:>8.0f} "
              f"{r['n_gt_drugs_with_targets']:>8} {r['coverage_gt_drugs']:>6.2f}")

    print("\n=== Correlations (categories n>=10) ===")
    for k, v in corr.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
