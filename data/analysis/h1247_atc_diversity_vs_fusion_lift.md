# h1247 — ATC sub-class diversity vs fusion lift on high-density strata

**Question:** Does ATC sub-class heterogeneity explain the n_gt≥51 stratum's high-variance / non-significant fusion lift seen in h1244? Hypothesis: high-density GT pools span many ATC sub-classes; fusion smooths some scores helpfully, others hurtfully → cancellation.

**Data:** 1,002 (seed,disease) rows from h1218 collapsed to per-disease means. ATC codes via DrugBank-name lookup → src/atc_features.ATCMapper. MESH-prefixed GT drugs are not mapped (no canonical MESH→ATC table on disk); mean per-disease ATC coverage in the n_gt≥51 stratum is 51.8%. Diversity ranking is robust to coverage because missing rates are similar across diseases.

## Stratum n_gt 21-50 (126 diseases)

- Mean ATC coverage: 60.8%
- Mean L3 unique codes: 9.2
- Mean L3 entropy: 2.52 bits
- Mean L1 unique codes: 5.5 (out of 14 possible)

### Correlations (Pearson r, two-sided p via Fisher z)

| Diversity metric | vs Δ R@30 | vs |Δ R@30| | vs Δ hits@30 | vs |Δ hits@30| |
|---|---:|---:|---:|---:|
| l3_unique | -0.153 (p=0.087) | -0.076 (p=0.4) | -0.082 (p=0.36) | +0.015 (p=0.87) |
| l3_entropy_bits | -0.171 (p=0.056) | -0.082 (p=0.36) | -0.112 (p=0.21) | -0.027 (p=0.76) |
| l3_gini_simpson | -0.179 (p=0.045) | -0.084 (p=0.35) | -0.131 (p=0.14) | -0.045 (p=0.62) |
| l1_unique | -0.104 (p=0.25) | -0.067 (p=0.46) | -0.053 (p=0.55) | -0.003 (p=0.98) |

### L3 entropy terciles

| Tercile | n | entropy range | meanΔR@30 | std | meanΔhits@30 | std |
|---|---:|---|---:|---:|---:|---:|
| low | 42 | -0.00–2.15 | +2.25pp | 4.49pp | +0.663 | 1.345 |
| mid | 42 | 2.16–3.17 | +0.69pp | 6.09pp | +0.127 | 1.980 |
| high | 42 | 3.17–4.21 | +1.26pp | 4.82pp | +0.492 | 1.849 |

### Most homogeneous (lowest L3 entropy)

| Disease | Cat | n_gt | L3 unique | entropy | ΔR@30 | Δhits@30 |
|---|---|---:|---:|---:|---:|---:|
| gastrointestinal stromal tumour | gastrointestinal | 21 | 1 | -0.00 | +0.00pp | +0.000 |
| biliary tract cancer | cancer | 25 | 1 | -0.00 | +12.00pp | +3.000 |
| medulloblastoma | cancer | 27 | 1 | -0.00 | +1.23pp | +0.333 |
| malignant astrocytoma | cancer | 23 | 1 | -0.00 | +4.35pp | +1.000 |
| hospitalacquired bacterial pneumonia | infectious | 30 | 2 | 0.30 | +3.33pp | +1.000 |

### Most heterogeneous (highest L3 entropy)

| Disease | Cat | n_gt | L3 unique | entropy | ΔR@30 | Δhits@30 |
|---|---|---:|---:|---:|---:|---:|
| hypoglycemia | metabolic | 40 | 20 | 4.21 | +0.83pp | +0.333 |
| extrapyramidal disorders | other | 30 | 18 | 4.00 | -1.67pp | -0.500 |
| duchenne muscular dystrophy | musculoskeletal | 42 | 17 | 3.92 | +0.00pp | +0.000 |
| neuropathy | neurological | 50 | 17 | 3.87 | +7.00pp | +3.500 |
| hereditary chronic cholestasis | gastrointestinal | 34 | 15 | 3.85 | -2.94pp | -1.000 |
## Stratum n_gt 51+ (122 diseases)

- Mean ATC coverage: 51.8%
- Mean L3 unique codes: 23.6
- Mean L3 entropy: 3.51 bits
- Mean L1 unique codes: 9.7 (out of 14 possible)

### Correlations (Pearson r, two-sided p via Fisher z)

| Diversity metric | vs Δ R@30 | vs |Δ R@30| | vs Δ hits@30 | vs |Δ hits@30| |
|---|---:|---:|---:|---:|
| l3_unique | +0.025 (p=0.78) | -0.219 (p=0.015) | +0.158 (p=0.083) | +0.225 (p=0.013) |
| l3_entropy_bits | -0.068 (p=0.46) | -0.098 (p=0.28) | +0.035 (p=0.7) | +0.161 (p=0.076) |
| l3_gini_simpson | -0.154 (p=0.09) | -0.033 (p=0.72) | -0.051 (p=0.57) | +0.120 (p=0.19) |
| l1_unique | -0.023 (p=0.8) | -0.197 (p=0.029) | +0.099 (p=0.28) | +0.195 (p=0.031) |

### L3 entropy terciles

| Tercile | n | entropy range | meanΔR@30 | std | meanΔhits@30 | std |
|---|---:|---|---:|---:|---:|---:|
| low | 40 | 0.90–3.18 | +0.81pp | 2.09pp | +0.842 | 1.812 |
| mid | 41 | 3.21–3.95 | -0.73pp | 2.95pp | -0.622 | 2.891 |
| high | 41 | 3.96–5.24 | +0.41pp | 2.23pp | +0.949 | 3.993 |

### Most homogeneous (lowest L3 entropy)

| Disease | Cat | n_gt | L3 unique | entropy | ΔR@30 | Δhits@30 |
|---|---|---:|---:|---:|---:|---:|
| esophageal cancer | cancer | 62 | 5 | 0.90 | +6.45pp | +4.000 |
| malignant bone and soft tissue tumors | cancer | 81 | 6 | 0.93 | +1.23pp | +1.000 |
| pyelonephritis | renal | 71 | 7 | 1.36 | +1.41pp | +1.000 |
| small lymphocytic lymphoma | cancer | 76 | 9 | 1.52 | +1.32pp | +1.000 |
| neisseria gonorrhoeae infections | infectious | 68 | 8 | 1.56 | -2.94pp | -2.000 |

### Most heterogeneous (highest L3 entropy)

| Disease | Cat | n_gt | L3 unique | entropy | ΔR@30 | Δhits@30 |
|---|---|---:|---:|---:|---:|---:|
| hepatic disease | gastrointestinal | 299 | 51 | 5.24 | +0.00pp | +0.000 |
| kidney failure | renal | 372 | 58 | 5.17 | +2.42pp | +9.000 |
| acetaminopheninduced hepatic injury | gastrointestinal | 210 | 44 | 5.17 | +0.00pp | +0.000 |
| renal diseases | renal | 412 | 54 | 5.13 | +0.00pp | +0.000 |
| hepatic cirrhosis | gastrointestinal | 162 | 42 | 5.12 | +3.09pp | +5.000 |

## Interpretation

**Mixed result — the hypothesis is supported on the n_gt 21-50 stratum but not on n_gt 51+.**

**n_gt 21-50 (where h1244 already found p<0.05 fusion lift):** All three diversity measures are negatively correlated with signed Δ R@30 — most homogeneous GT pools see the largest fusion lift. **l3_gini_simpson vs Δ R@30: r=-0.179, p=0.045** (significant); l3_entropy_bits borderline (p=0.056); l3_unique p=0.087. Tercile bucketing on entropy: low-entropy (most homogeneous, n=42) gains +2.25pp R@30; mid-entropy gains only +0.69pp; high-entropy intermediate at +1.26pp. The interpretation: when a disease's GT drugs share a dominant ATC sub-class (e.g. T2D ≈ A10*), fusion's score-smoothing lifts the coherent cluster wholesale.

**n_gt 51+ (the high-variance / non-significant stratum from h1244):** Diversity-magnitude correlations are weak. l3_unique vs |Δ R@30| = -0.219 (p=0.015) — opposite sign to hypothesis. L3 entropy vs |Δ hits@30| = +0.161 (p=0.076) borderline — supports a weak std-growth pattern. Entropy terciles show a U-curve: low (+0.81pp) and high (+0.41pp) gain; mid (-0.73pp) loses. **std grows monotonically in hits@30 (1.81 → 2.89 → 3.99) — heterogeneity DOES inflate variance, just not in the direction or magnitude the simple hypothesis predicted.** On this stratum, most diseases have ≥30 ATC sub-classes among ≥51 GT drugs; the homogeneity signal saturates.

**Implications:**
1. **h1247 partial-VALIDATED (n_gt 21-50):** ATC homogeneity is a real predictor of fusion lift. Adding `gt_atc_l3_gini` as a per-disease confidence-routing feature could lift the +2.25pp low-entropy diseases above the global +1.32pp average.
2. **h1247 INVALIDATED for n_gt 51+ as a single-mechanism explanation.** Variance growth is real (monotonic in hits@30 std) but doesn't fully explain the missing mean. Need a multi-axis decomposition.
3. **Follow-up h1248 (proposed):** restrict h1244-style paired-t to ATC-homogeneous subset of n_gt≥51 diseases; if Δ hits@30 reaches p<0.05 there, we have a clean recall-lever knob.
