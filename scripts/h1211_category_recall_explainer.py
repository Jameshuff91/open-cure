#!/usr/bin/env python3
"""h1211: Per-category R@30 explanation.

Decomposes the 3x per-category R@30 spread observed in h1199
(endocrine 41%, ophthalmic 38% ... psychiatric 10%, hematological 10%)
into three mechanistic diagnostics per category:

  (a) category density       — mean pairwise drug-Jaccard between GT diseases in the category
  (b) category isolation     — mean fraction of top-20 kNN neighbours (over eligible diseases)
                               that share the category
  (c) GT completeness proxy  — mean / median GT drugs per disease in the category

Then correlates each diagnostic with per-category R@30 from
data/analysis/clean_benchmark_node2vec_256.json.

Outputs:
  data/analysis/h1211_category_recall_explainer.json
  data/analysis/h1211_category_recall_explainer.md
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Set, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from production_predictor import CATEGORY_KEYWORDS  # noqa: E402


def categorize(name: str) -> str:
    n = name.lower()
    for cat, kws in CATEGORY_KEYWORDS.items():
        for kw in kws:
            if kw in n:
                return cat
    return "other"


def load_disease_names() -> Dict[str, str]:
    with open(PROJECT_ROOT / "data" / "cache" / "ground_truth_cache.json") as f:
        cache = json.load(f)
    raw = cache.get("disease_names", {})
    return {k if k.startswith("drkg:") else f"drkg:{k}": v for k, v in raw.items()}


def load_node2vec() -> Tuple[Dict[str, np.ndarray], np.ndarray, List[str]]:
    emb_dir = PROJECT_ROOT / "data" / "embeddings"
    entities = np.load(emb_dir / "node2vec_256_entities.npy", allow_pickle=True)
    matrix = np.load(emb_dir / "node2vec_256_embeddings.npy").astype(np.float32)
    keys = [f"drkg:{e}" for e in entities]
    lookup = {k: matrix[i] for i, k in enumerate(keys)}
    return lookup, matrix, keys


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    u = len(a | b)
    if u == 0:
        return 0.0
    return len(a & b) / u


def pearson(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    """Return (r, n) — uses numpy.corrcoef if enough points."""
    if len(xs) < 3:
        return 0.0, len(xs)
    arr = np.array([xs, ys])
    c = np.corrcoef(arr)[0, 1]
    return float(c), len(xs)


def main() -> None:
    print("[h1211] loading GT, names, Node2Vec embeddings ...", flush=True)
    with open(PROJECT_ROOT / "data" / "reference" / "expanded_ground_truth.json") as f:
        expanded_gt: Dict[str, List[str]] = json.load(f)

    names = load_disease_names()
    lookup, _, keys = load_node2vec()
    emb_set = set(keys)

    # Eligible diseases: have GT ∧ have embedding (matches h1199 filter)
    eligible = [d for d in expanded_gt if d in emb_set]
    print(f"[h1211] eligible diseases (GT ∩ embedding): {len(eligible)}", flush=True)

    # Assign category per disease
    dis_cat: Dict[str, str] = {
        d: categorize(names.get(d, d)) for d in eligible
    }

    # GT drug sets per eligible disease
    dis_drugs: Dict[str, Set[str]] = {
        d: set(expanded_gt.get(d, [])) for d in eligible
    }

    # Bucket by category
    by_cat: Dict[str, List[str]] = defaultdict(list)
    for d, c in dis_cat.items():
        by_cat[c].append(d)

    # ============= (a) density: mean pairwise drug-Jaccard within category =============
    density: Dict[str, Dict[str, float]] = {}
    for cat, ds in by_cat.items():
        if len(ds) < 2:
            density[cat] = {
                "mean_jaccard": 0.0,
                "median_jaccard": 0.0,
                "n_pairs": 0,
            }
            continue
        js: List[float] = []
        # Cap at 40k pairs to keep runtime bounded; random shuffle is not needed
        # because all eligible diseases are considered and we're averaging.
        # For categories with >=300 diseases we subsample deterministically.
        rng = np.random.RandomState(0)
        if len(ds) > 300:
            idx = rng.choice(len(ds), 300, replace=False)
            ds_use = [ds[i] for i in idx]
        else:
            ds_use = ds
        for i in range(len(ds_use)):
            for j in range(i + 1, len(ds_use)):
                js.append(jaccard(dis_drugs[ds_use[i]], dis_drugs[ds_use[j]]))
        density[cat] = {
            "mean_jaccard": float(mean(js)) if js else 0.0,
            "median_jaccard": float(median(js)) if js else 0.0,
            "n_pairs": len(js),
        }

    # ============= (b) isolation: mean fraction of top-20 kNN neighbours sharing the category =============
    # Restrict nearest-neighbour search to eligible diseases only (excludes the
    # query itself). This matches the "how often does the kNN land inside the
    # same category?" question.
    print("[h1211] computing kNN isolation ...", flush=True)
    elig_emb = np.stack([lookup[d] for d in eligible])
    # cosine_similarity is O(n²) in memory; eligible ≈ 1011 rows × 256 dims — ~1M entries, fine
    sim = cosine_similarity(elig_emb, elig_emb)
    # Self-similarity mask
    np.fill_diagonal(sim, -np.inf)
    top_k = 20
    # top_k indices per row
    top_idx = np.argpartition(-sim, top_k, axis=1)[:, :top_k]
    # Ensure sorted by similarity descending (not strictly needed for isolation but nice)
    # — partition doesn't guarantee order; for isolation fraction this is fine
    cat_array = np.array([dis_cat[d] for d in eligible])
    isolation: Dict[str, Dict[str, float]] = {}
    for cat, ds in by_cat.items():
        row_indices = [eligible.index(d) for d in ds] if len(ds) <= 300 else None
        if row_indices is None:
            # For very large categories, sample 300 diseases
            rng = np.random.RandomState(0)
            chosen_ds = [ds[i] for i in rng.choice(len(ds), 300, replace=False)]
            row_indices = [eligible.index(d) for d in chosen_ds]
        fracs: List[float] = []
        same_cat_sets: List[float] = []
        for r in row_indices:
            neighbours = top_idx[r]
            frac = float(np.mean(cat_array[neighbours] == cat))
            fracs.append(frac)
            same_cat_sets.append(frac)
        # Category-level baseline: the "chance" fraction if kNN were uniform would be
        # (n_category - 1) / (n_eligible - 1)
        chance = (len(ds) - 1) / (len(eligible) - 1)
        isolation[cat] = {
            "mean_same_cat_frac": float(mean(fracs)) if fracs else 0.0,
            "chance_frac": float(chance),
            "lift_over_chance": (float(mean(fracs)) - float(chance)) if fracs else 0.0,
            "n_diseases_evaluated": len(row_indices),
        }

    # ============= (c) GT completeness proxy =============
    completeness: Dict[str, Dict[str, float]] = {}
    for cat, ds in by_cat.items():
        sizes = [len(dis_drugs[d]) for d in ds]
        completeness[cat] = {
            "n_diseases": len(ds),
            "mean_gt_drugs": float(mean(sizes)) if sizes else 0.0,
            "median_gt_drugs": float(median(sizes)) if sizes else 0.0,
            "total_gt_pairs": int(sum(sizes)),
        }

    # ============= pull h1199 per-category R@30 =============
    with open(PROJECT_ROOT / "data" / "analysis" / "clean_benchmark_node2vec_256.json") as f:
        bench = json.load(f)
    per_cat_r30 = {
        cat: vals["r30_mean"] for cat, vals in bench["per_category_aggregate"].items()
    }

    # ============= Combine + correlate =============
    combined_rows: List[Dict] = []
    cats_with_all = sorted(
        set(per_cat_r30) & set(density) & set(isolation) & set(completeness)
    )
    for cat in cats_with_all:
        row = {
            "category": cat,
            "r30": per_cat_r30[cat],
            "n_diseases": completeness[cat]["n_diseases"],
            "mean_gt_drugs": completeness[cat]["mean_gt_drugs"],
            "median_gt_drugs": completeness[cat]["median_gt_drugs"],
            "total_gt_pairs": completeness[cat]["total_gt_pairs"],
            "density_mean_jaccard": density[cat]["mean_jaccard"],
            "density_median_jaccard": density[cat]["median_jaccard"],
            "isolation_same_cat_frac": isolation[cat]["mean_same_cat_frac"],
            "isolation_chance_frac": isolation[cat]["chance_frac"],
            "isolation_lift": isolation[cat]["lift_over_chance"],
        }
        combined_rows.append(row)

    # Ceiling-adjusted R@30: average per-disease ceiling = E[min(30, |GT|)] / E[|GT|].
    # A category with median 64 GT drugs can hit at most 30/64 = 47% on R@30, even
    # under perfect ranking. Reporting r30 / ceiling normalises the denominator
    # effect out and isolates the "how well does retrieval actually work?" signal.
    # We compute the ceiling per category using that category's GT size distribution.
    for row in combined_rows:
        cat = row["category"]
        ds = by_cat[cat]
        per_dis_ceilings = [
            min(30, len(dis_drugs[d])) / max(len(dis_drugs[d]), 1) for d in ds
        ]
        ceil_mean = float(np.mean(per_dis_ceilings)) if per_dis_ceilings else 1.0
        row["r30_ceiling"] = ceil_mean
        row["r30_over_ceiling"] = (
            row["r30"] / ceil_mean if ceil_mean > 0 else 0.0
        )

    # Sort by R@30 descending
    combined_rows.sort(key=lambda r: -r["r30"])

    # Pearson correlations
    r30_vals = [r["r30"] for r in combined_rows]
    correlations = {
        "density_mean_jaccard_vs_r30": pearson(
            [r["density_mean_jaccard"] for r in combined_rows], r30_vals
        ),
        "isolation_same_cat_frac_vs_r30": pearson(
            [r["isolation_same_cat_frac"] for r in combined_rows], r30_vals
        ),
        "isolation_lift_vs_r30": pearson(
            [r["isolation_lift"] for r in combined_rows], r30_vals
        ),
        "mean_gt_drugs_vs_r30": pearson(
            [r["mean_gt_drugs"] for r in combined_rows], r30_vals
        ),
        "median_gt_drugs_vs_r30": pearson(
            [r["median_gt_drugs"] for r in combined_rows], r30_vals
        ),
        "n_diseases_vs_r30": pearson(
            [float(r["n_diseases"]) for r in combined_rows], r30_vals
        ),
        "log_n_diseases_vs_r30": pearson(
            [float(np.log(max(r["n_diseases"], 1))) for r in combined_rows], r30_vals
        ),
        "ceiling_vs_r30": pearson(
            [r["r30_ceiling"] for r in combined_rows], r30_vals
        ),
        "density_vs_r30_over_ceiling": pearson(
            [r["density_mean_jaccard"] for r in combined_rows],
            [r["r30_over_ceiling"] for r in combined_rows],
        ),
        "isolation_vs_r30_over_ceiling": pearson(
            [r["isolation_same_cat_frac"] for r in combined_rows],
            [r["r30_over_ceiling"] for r in combined_rows],
        ),
        "mean_gt_drugs_vs_r30_over_ceiling": pearson(
            [r["mean_gt_drugs"] for r in combined_rows],
            [r["r30_over_ceiling"] for r in combined_rows],
        ),
    }

    # Rank-based: does density + isolation jointly explain spread?
    # Simple OLS via numpy on (density, isolation) -> r30
    if len(combined_rows) >= 4:
        X = np.array(
            [
                [r["density_mean_jaccard"], r["isolation_same_cat_frac"]]
                for r in combined_rows
            ]
        )
        y = np.array(r30_vals)
        X1 = np.hstack([X, np.ones((len(X), 1))])
        # lstsq
        coef, *_ = np.linalg.lstsq(X1, y, rcond=None)
        y_hat = X1 @ coef
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        ols_joint = {
            "coef_density": float(coef[0]),
            "coef_isolation": float(coef[1]),
            "intercept": float(coef[2]),
            "r2": float(r2),
        }
    else:
        ols_joint = None

    out = {
        "hypothesis": "h1211",
        "description": "Per-category R@30 explanation via drug-Jaccard density, kNN isolation, and GT completeness.",
        "n_eligible_diseases": len(eligible),
        "category_rows": combined_rows,
        "correlations": {k: {"r": v[0], "n": v[1]} for k, v in correlations.items()},
        "ols_joint_density_isolation_to_r30": ols_joint,
    }
    out_json = PROJECT_ROOT / "data" / "analysis" / "h1211_category_recall_explainer.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[h1211] wrote {out_json}")

    # ============= Markdown summary =============
    lines = []
    lines.append("# h1211 — Per-category R@30 explainer (Node2Vec kNN)")
    lines.append("")
    lines.append(f"- Eligible diseases (GT ∩ Node2Vec embeddings): **{len(eligible)}**")
    lines.append("- Per-category R@30: sourced from `clean_benchmark_node2vec_256.json` (5-seed mean)")
    lines.append("")
    lines.append("## Per-category diagnostics")
    lines.append("")
    lines.append("| Category | n_dis | R@30 | Ceiling | R@30/Ceil | Density (J̄) | Isolation | Iso chance | Iso lift | mean GT | median GT |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in combined_rows:
        lines.append(
            "| {cat} | {n} | {r30:.2%} | {ceil:.2%} | {roc:.2%} | {dens:.4f} | {iso:.2%} | {ch:.2%} | {lift:+.2%} | {meang:.1f} | {medg:.0f} |".format(
                cat=r["category"],
                n=r["n_diseases"],
                r30=r["r30"],
                ceil=r["r30_ceiling"],
                roc=r["r30_over_ceiling"],
                dens=r["density_mean_jaccard"],
                iso=r["isolation_same_cat_frac"],
                ch=r["isolation_chance_frac"],
                lift=r["isolation_lift"],
                meang=r["mean_gt_drugs"],
                medg=r["median_gt_drugs"],
            )
        )
    lines.append("")
    lines.append("## Univariate Pearson correlations vs per-category R@30")
    lines.append("")
    lines.append("| Diagnostic | r | n |")
    lines.append("|---|---:|---:|")
    for k, v in correlations.items():
        lines.append(f"| {k} | {v[0]:.3f} | {v[1]} |")
    lines.append("")
    if ols_joint is not None:
        lines.append("## OLS (density + isolation → R@30)")
        lines.append("")
        lines.append("| Coefficient | Value |")
        lines.append("|---|---:|")
        lines.append(f"| density (mean drug-Jaccard within category) | {ols_joint['coef_density']:.4f} |")
        lines.append(f"| isolation (same-category fraction of top-20 kNN neighbours) | {ols_joint['coef_isolation']:.4f} |")
        lines.append(f"| intercept | {ols_joint['intercept']:.4f} |")
        lines.append(f"| R² | {ols_joint['r2']:.3f} |")
        lines.append("")

    # Headline interpretation
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- **Density (drug-Jaccard)** measures whether diseases within a category "
        "share drugs. High density → kNN should transfer well."
    )
    lines.append(
        "- **Isolation (same-category kNN fraction)** measures whether cosine-nearest "
        "neighbours in Node2Vec space land inside the same category. High isolation → "
        "the embedding respects category structure. We report the chance baseline "
        "(n_category-1)/(n_eligible-1) and the lift above chance."
    )
    lines.append(
        "- **GT completeness proxy** uses mean / median drugs per disease in the "
        "category. Very small mean values (e.g. cancer with biomarker-stratified GT) "
        "are known failure modes; very large means (e.g. infectious with 10+ "
        "antibiotics per disease) inflate R@30 because even a broad transfer hits."
    )
    lines.append("")
    out_md = PROJECT_ROOT / "data" / "analysis" / "h1211_category_recall_explainer.md"
    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[h1211] wrote {out_md}")


if __name__ == "__main__":
    main()
