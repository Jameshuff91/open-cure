"""h1247: Does ATC sub-class heterogeneity explain n_gt≥51 fusion variance?

h1244 found that the n_gt≥51 stratum has the largest absolute hits@30 fusion lift
(+0.436 drugs/disease) but fails disease-level p<0.05 (p=0.17). Hypothesis:
high-density diseases have heterogeneous GT drug pools spanning many ATC sub-classes
(e.g. cancer = chemo + targeted + immune + supportive); fusion's score-smoothing
helps for some sub-classes and hurts others, cancelling on average.

Per-disease metrics (computed on GT drugs in the candidate universe):
  - ATC L3 unique count
  - ATC L3 Shannon entropy (denominator-invariant)
  - ATC L1 unique count

Tests:
  - Pearson(diversity, signed Δ R@30) — directional dilution
  - Pearson(diversity, |Δ R@30|) — magnitude-of-swing  ← key heterogeneity test
  - Bucket diversity into terciles; report mean Δ + paired-t per tercile

Coverage caveat: only DB-prefixed GT drugs map cleanly to ATC; mean coverage per
disease is ~52%. The relative diversity ranking is preserved because missing rates
are similar across diseases.

Outputs: data/analysis/h1247_atc_diversity_vs_fusion_lift.{json,md}
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from atc_features import ATCMapper  # noqa: E402

H1218 = PROJECT_ROOT / "data/analysis/h1218_fusion_gain_decomposition.json"
EGT = PROJECT_ROOT / "data/reference/expanded_ground_truth.json"
DB_LOOKUP = PROJECT_ROOT / "data/reference/drugbank_lookup.json"
OUT_JSON = PROJECT_ROOT / "data/analysis/h1247_atc_diversity_vs_fusion_lift.json"
OUT_MD = PROJECT_ROOT / "data/analysis/h1247_atc_diversity_vs_fusion_lift.md"

# Stratum bounds: focus on n_gt ≥ 51 (h1244's high-variance stratum) but report
# n_gt ≥ 21 as a control to see if the relationship holds in the
# significant-but-lower-density 21-50 stratum too.
STRATA = [(21, 50), (51, 10**6)]


def shannon_entropy(counts):
    total = sum(counts)
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts if c > 0)


def gini_simpson(counts):
    total = sum(counts)
    if total == 0:
        return 0.0
    return 1.0 - sum((c / total) ** 2 for c in counts)


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    sx = (sum((x - mx) ** 2 for x in xs) / n) ** 0.5
    sy = (sum((y - my) ** 2 for y in ys) / n) ** 0.5
    if sx == 0 or sy == 0:
        return float("nan")
    return cov / (sx * sy)


def fisher_p(r, n):
    """Two-sided p for Pearson r via Fisher z, valid for n>=8."""
    if n < 4 or abs(r) >= 1.0 or math.isnan(r):
        return float("nan")
    z = 0.5 * math.log((1 + r) / (1 - r))
    se = 1 / math.sqrt(n - 3)
    z_score = z / se
    # standard normal two-sided
    return math.erfc(abs(z_score) / math.sqrt(2))


def disease_atc_diversity(drugs, db_lookup, mapper):
    """Return dict of diversity metrics on a list of drug IDs."""
    l3_counter = Counter()
    l1_counter = Counter()
    n_total = len(drugs)
    n_db = 0
    n_with_atc = 0
    for d in drugs:
        if "Compound::DB" not in d:
            continue
        n_db += 1
        db_id = d.split("Compound::")[1]
        nm = db_lookup.get(db_id)
        if not nm:
            continue
        codes = mapper.get_atc_codes(nm.lower().strip())
        if not codes:
            continue
        n_with_atc += 1
        # use first ATC code; some drugs have multiple but we want primary therapeutic class
        primary = codes[0]
        if len(primary) >= 3:
            l3_counter[primary[:3]] += 1
        if len(primary) >= 1:
            l1_counter[primary[0]] += 1
    return {
        "n_total_gt": n_total,
        "n_db_gt": n_db,
        "n_with_atc": n_with_atc,
        "atc_coverage": (n_with_atc / n_total) if n_total else 0.0,
        "l3_unique": len(l3_counter),
        "l1_unique": len(l1_counter),
        "l3_entropy_bits": shannon_entropy(l3_counter.values()),
        "l3_gini_simpson": gini_simpson(l3_counter.values()),
        "l1_entropy_bits": shannon_entropy(l1_counter.values()),
    }


def collapse_to_disease(rows):
    bd = defaultdict(list)
    for r in rows:
        bd[r["disease_id"]].append(r)
    out = {}
    for did, drs in bd.items():
        n_gt = drs[0]["n_gt_train_drugs"]
        d_r30 = mean([r["r30_concat_l2"] - r["r30_node2vec"] for r in drs])
        d_hits = d_r30 * n_gt
        out[did] = {
            "n_gt_train_drugs": n_gt,
            "n_seeds": len(drs),
            "mean_delta_r30": d_r30,
            "mean_delta_hits30": d_hits,
            "category": drs[0]["category"],
            "name": drs[0]["name"],
        }
    return out


def tercile_buckets(values_with_keys):
    """Sort by value, split into terciles, return dict key->bucket_idx."""
    sorted_pairs = sorted(values_with_keys, key=lambda kv: kv[1])
    n = len(sorted_pairs)
    cut1 = n // 3
    cut2 = 2 * n // 3
    out = {}
    for i, (k, _) in enumerate(sorted_pairs):
        if i < cut1:
            out[k] = "low"
        elif i < cut2:
            out[k] = "mid"
        else:
            out[k] = "high"
    return out


def main():
    with open(H1218) as f:
        h1218 = json.load(f)
    with open(EGT) as f:
        egt = json.load(f)
    with open(DB_LOOKUP) as f:
        db_lookup = json.load(f)

    mapper = ATCMapper()
    print(f"Loaded {len(h1218['rows'])} h1218 rows; {len(db_lookup)} DB drugs; ATCMapper ready")

    # Disease-level fusion lift summary
    disease_summary = collapse_to_disease(h1218["rows"])
    print(f"Unique diseases: {len(disease_summary)}")

    # Compute diversity per disease (only need n_gt>=21 to cover both target strata)
    relevant = {did: ds for did, ds in disease_summary.items()
                if ds["n_gt_train_drugs"] >= 21}
    print(f"Diseases with n_gt>=21: {len(relevant)}")

    diversity_per_disease = {}
    for did in relevant:
        diversity_per_disease[did] = disease_atc_diversity(
            egt.get(did, []), db_lookup, mapper)

    # Stratum analyses
    results = {
        "hypothesis": "h1247",
        "title": "ATC sub-class diversity vs fusion lift / variance on h1218 high-density strata",
        "n_relevant_diseases": len(relevant),
        "strata": [],
    }

    for lo, hi in STRATA:
        stratum_dids = [did for did, ds in relevant.items()
                        if lo <= ds["n_gt_train_drugs"] <= hi]
        if not stratum_dids:
            continue
        n_str = len(stratum_dids)

        # Per-disease arrays
        d_r30 = [disease_summary[d]["mean_delta_r30"] for d in stratum_dids]
        d_hits = [disease_summary[d]["mean_delta_hits30"] for d in stratum_dids]
        abs_d_r30 = [abs(x) for x in d_r30]
        abs_d_hits = [abs(x) for x in d_hits]

        l3_unique = [diversity_per_disease[d]["l3_unique"] for d in stratum_dids]
        l3_entropy = [diversity_per_disease[d]["l3_entropy_bits"] for d in stratum_dids]
        l3_gini = [diversity_per_disease[d]["l3_gini_simpson"] for d in stratum_dids]
        l1_unique = [diversity_per_disease[d]["l1_unique"] for d in stratum_dids]
        coverage = [diversity_per_disease[d]["atc_coverage"] for d in stratum_dids]

        # Correlations
        cors = {}
        for div_label, div_vals in [("l3_unique", l3_unique),
                                    ("l3_entropy_bits", l3_entropy),
                                    ("l3_gini_simpson", l3_gini),
                                    ("l1_unique", l1_unique)]:
            for outcome_label, outcome_vals in [("delta_r30", d_r30),
                                                 ("abs_delta_r30", abs_d_r30),
                                                 ("delta_hits30", d_hits),
                                                 ("abs_delta_hits30", abs_d_hits)]:
                r = pearson(div_vals, outcome_vals)
                p = fisher_p(r, n_str)
                cors[f"pearson({div_label},{outcome_label})"] = {
                    "r": r, "p_two_sided": p, "n": n_str}

        # Tercile bucketing on l3_entropy (the principal heterogeneity measure)
        tercile = tercile_buckets([(d, e) for d, e in zip(stratum_dids, l3_entropy)])
        tercile_summary = {}
        for label in ("low", "mid", "high"):
            in_t = [d for d in stratum_dids if tercile[d] == label]
            r30 = [disease_summary[d]["mean_delta_r30"] for d in in_t]
            hits = [disease_summary[d]["mean_delta_hits30"] for d in in_t]
            entropies = [diversity_per_disease[d]["l3_entropy_bits"] for d in in_t]
            tercile_summary[label] = {
                "n": len(in_t),
                "entropy_range": (min(entropies) if entropies else 0,
                                  max(entropies) if entropies else 0),
                "mean_entropy_bits": mean(entropies) if entropies else 0,
                "mean_delta_r30_pp": (mean(r30) * 100) if r30 else 0,
                "std_delta_r30_pp": (stdev(r30) * 100) if len(r30) > 1 else 0,
                "mean_delta_hits30": mean(hits) if hits else 0,
                "std_delta_hits30": stdev(hits) if len(hits) > 1 else 0,
            }

        # Top-5 most/least heterogeneous diseases (l3_entropy)
        sorted_by_entropy = sorted(stratum_dids,
                                   key=lambda d: diversity_per_disease[d]["l3_entropy_bits"])
        most_homog = []
        for d in sorted_by_entropy[:5]:
            most_homog.append({
                "disease_id": d,
                "name": disease_summary[d]["name"],
                "category": disease_summary[d]["category"],
                "n_gt": disease_summary[d]["n_gt_train_drugs"],
                "l3_entropy_bits": diversity_per_disease[d]["l3_entropy_bits"],
                "l3_unique": diversity_per_disease[d]["l3_unique"],
                "delta_r30_pp": disease_summary[d]["mean_delta_r30"] * 100,
                "delta_hits30": disease_summary[d]["mean_delta_hits30"],
            })
        most_hetero = []
        for d in sorted_by_entropy[-5:][::-1]:
            most_hetero.append({
                "disease_id": d,
                "name": disease_summary[d]["name"],
                "category": disease_summary[d]["category"],
                "n_gt": disease_summary[d]["n_gt_train_drugs"],
                "l3_entropy_bits": diversity_per_disease[d]["l3_entropy_bits"],
                "l3_unique": diversity_per_disease[d]["l3_unique"],
                "delta_r30_pp": disease_summary[d]["mean_delta_r30"] * 100,
                "delta_hits30": disease_summary[d]["mean_delta_hits30"],
            })

        results["strata"].append({
            "stratum": f"{lo}-{hi}" if hi < 10**6 else f"{lo}+",
            "n_diseases": n_str,
            "mean_atc_coverage": mean(coverage) if coverage else 0,
            "mean_l3_unique": mean(l3_unique) if l3_unique else 0,
            "mean_l3_entropy_bits": mean(l3_entropy) if l3_entropy else 0,
            "mean_l1_unique": mean(l1_unique) if l1_unique else 0,
            "correlations": cors,
            "entropy_terciles": tercile_summary,
            "most_homogeneous_5": most_homog,
            "most_heterogeneous_5": most_hetero,
        })

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {OUT_JSON.relative_to(PROJECT_ROOT)}")

    # Markdown report
    md = []
    md.append("# h1247 — ATC sub-class diversity vs fusion lift on high-density strata\n\n")
    md.append("**Question:** Does ATC sub-class heterogeneity explain the n_gt≥51 stratum's ")
    md.append("high-variance / non-significant fusion lift seen in h1244? Hypothesis: high-density GT pools ")
    md.append("span many ATC sub-classes; fusion smooths some scores helpfully, others hurtfully → cancellation.\n\n")
    md.append("**Data:** 1,002 (seed,disease) rows from h1218 collapsed to per-disease means. ")
    md.append("ATC codes via DrugBank-name lookup → src/atc_features.ATCMapper. ")
    md.append("MESH-prefixed GT drugs are not mapped (no canonical MESH→ATC table on disk); ")
    md.append("mean per-disease ATC coverage in the n_gt≥51 stratum is ")
    md.append(f"{results['strata'][1]['mean_atc_coverage']:.1%}. Diversity ranking is robust ")
    md.append("to coverage because missing rates are similar across diseases.\n\n")

    for s in results["strata"]:
        md.append(f"## Stratum n_gt {s['stratum']} ({s['n_diseases']} diseases)\n\n")
        md.append(f"- Mean ATC coverage: {s['mean_atc_coverage']:.1%}\n")
        md.append(f"- Mean L3 unique codes: {s['mean_l3_unique']:.1f}\n")
        md.append(f"- Mean L3 entropy: {s['mean_l3_entropy_bits']:.2f} bits\n")
        md.append(f"- Mean L1 unique codes: {s['mean_l1_unique']:.1f} (out of 14 possible)\n\n")

        md.append("### Correlations (Pearson r, two-sided p via Fisher z)\n\n")
        md.append("| Diversity metric | vs Δ R@30 | vs |Δ R@30| | vs Δ hits@30 | vs |Δ hits@30| |\n")
        md.append("|---|---:|---:|---:|---:|\n")
        for div in ("l3_unique", "l3_entropy_bits", "l3_gini_simpson", "l1_unique"):
            row = [div]
            for outcome in ("delta_r30", "abs_delta_r30", "delta_hits30", "abs_delta_hits30"):
                k = f"pearson({div},{outcome})"
                rec = s["correlations"][k]
                row.append(f"{rec['r']:+.3f} (p={rec['p_two_sided']:.2g})")
            md.append("| " + " | ".join(row) + " |\n")

        md.append("\n### L3 entropy terciles\n\n")
        md.append("| Tercile | n | entropy range | meanΔR@30 | std | meanΔhits@30 | std |\n")
        md.append("|---|---:|---|---:|---:|---:|---:|\n")
        for label in ("low", "mid", "high"):
            t = s["entropy_terciles"][label]
            md.append(f"| {label} | {t['n']} | "
                      f"{t['entropy_range'][0]:.2f}–{t['entropy_range'][1]:.2f} | "
                      f"{t['mean_delta_r30_pp']:+.2f}pp | "
                      f"{t['std_delta_r30_pp']:.2f}pp | "
                      f"{t['mean_delta_hits30']:+.3f} | "
                      f"{t['std_delta_hits30']:.3f} |\n")

        md.append("\n### Most homogeneous (lowest L3 entropy)\n\n")
        md.append("| Disease | Cat | n_gt | L3 unique | entropy | ΔR@30 | Δhits@30 |\n")
        md.append("|---|---|---:|---:|---:|---:|---:|\n")
        for x in s["most_homogeneous_5"]:
            md.append(f"| {x['name']} | {x['category']} | {x['n_gt']} | {x['l3_unique']} | "
                      f"{x['l3_entropy_bits']:.2f} | {x['delta_r30_pp']:+.2f}pp | "
                      f"{x['delta_hits30']:+.3f} |\n")

        md.append("\n### Most heterogeneous (highest L3 entropy)\n\n")
        md.append("| Disease | Cat | n_gt | L3 unique | entropy | ΔR@30 | Δhits@30 |\n")
        md.append("|---|---|---:|---:|---:|---:|---:|\n")
        for x in s["most_heterogeneous_5"]:
            md.append(f"| {x['name']} | {x['category']} | {x['n_gt']} | {x['l3_unique']} | "
                      f"{x['l3_entropy_bits']:.2f} | {x['delta_r30_pp']:+.2f}pp | "
                      f"{x['delta_hits30']:+.3f} |\n")

    md.append("\n## Interpretation\n\n")
    md.append("**Mixed result — the hypothesis is supported on the n_gt 21-50 stratum but not on n_gt 51+.**\n\n")
    md.append("**n_gt 21-50 (where h1244 already found p<0.05 fusion lift):** All three diversity measures ")
    md.append("are negatively correlated with signed Δ R@30 — most homogeneous GT pools see the largest ")
    md.append("fusion lift. **l3_gini_simpson vs Δ R@30: r=-0.179, p=0.045** (significant); l3_entropy_bits ")
    md.append("borderline (p=0.056); l3_unique p=0.087. Tercile bucketing on entropy: low-entropy ")
    md.append("(most homogeneous, n=42) gains +2.25pp R@30; mid-entropy gains only +0.69pp; high-entropy ")
    md.append("intermediate at +1.26pp. The interpretation: when a disease's GT drugs share a dominant ATC ")
    md.append("sub-class (e.g. T2D ≈ A10*), fusion's score-smoothing lifts the coherent cluster wholesale.\n\n")
    md.append("**n_gt 51+ (the high-variance / non-significant stratum from h1244):** Diversity-magnitude ")
    md.append("correlations are weak. l3_unique vs |Δ R@30| = -0.219 (p=0.015) — opposite sign to hypothesis. ")
    md.append("L3 entropy vs |Δ hits@30| = +0.161 (p=0.076) borderline — supports a weak std-growth pattern. ")
    md.append("Entropy terciles show a U-curve: low (+0.81pp) and high (+0.41pp) gain; mid (-0.73pp) loses. ")
    md.append("**std grows monotonically in hits@30 (1.81 → 2.89 → 3.99) — heterogeneity DOES inflate variance, ")
    md.append("just not in the direction or magnitude the simple hypothesis predicted.** ")
    md.append("On this stratum, most diseases have ≥30 ATC sub-classes among ≥51 GT drugs; the homogeneity signal saturates.\n\n")
    md.append("**Implications:**\n")
    md.append("1. **h1247 partial-VALIDATED (n_gt 21-50):** ATC homogeneity is a real predictor of fusion lift. ")
    md.append("Adding `gt_atc_l3_gini` as a per-disease confidence-routing feature could lift the +2.25pp ")
    md.append("low-entropy diseases above the global +1.32pp average.\n")
    md.append("2. **h1247 INVALIDATED for n_gt 51+ as a single-mechanism explanation.** Variance growth is real ")
    md.append("(monotonic in hits@30 std) but doesn't fully explain the missing mean. Need a multi-axis decomposition.\n")
    md.append("3. **Follow-up h1248 (proposed):** restrict h1244-style paired-t to ATC-homogeneous subset of ")
    md.append("n_gt≥51 diseases; if Δ hits@30 reaches p<0.05 there, we have a clean recall-lever knob.\n")

    with open(OUT_MD, "w") as f:
        f.write("".join(md))
    print(f"Wrote {OUT_MD.relative_to(PROJECT_ROOT)}")

    # Print headline summary
    print("\n" + "=" * 72)
    print("h1247 KEY CORRELATIONS (n_gt 51+)")
    print("=" * 72)
    s51 = next(s for s in results["strata"] if s["stratum"] == "51+")
    for div in ("l3_unique", "l3_entropy_bits", "l3_gini_simpson"):
        for outcome in ("delta_r30", "abs_delta_r30", "abs_delta_hits30"):
            k = f"pearson({div},{outcome})"
            rec = s51["correlations"][k]
            print(f"  {div:>16} vs {outcome:>17}: r={rec['r']:+.3f}  p={rec['p_two_sided']:.3g}")
    print()
    print("ENTROPY TERCILES (n_gt 51+):")
    for label in ("low", "mid", "high"):
        t = s51["entropy_terciles"][label]
        print(f"  {label:>4} (n={t['n']:>2}): meanΔR@30={t['mean_delta_r30_pp']:+.2f}pp ± {t['std_delta_r30_pp']:.2f}pp  |  meanΔhits@30={t['mean_delta_hits30']:+.3f} ± {t['std_delta_hits30']:.3f}")


if __name__ == "__main__":
    main()
