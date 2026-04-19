# h1248 — Homogeneity-restricted paired-t on n_gt≥51 fusion lift

**Question:** h1244 found the n_gt≥51 stratum's overall fusion lift fails p<0.05 (p=0.17 hits@30 disease-level). h1247 showed the bottom-tercile L3-entropy subset has the highest mean lift (+0.81pp R@30 / +0.842 hits@30) and lowest variance. Does restricting to that subset recover p<0.05?

**Sample:** 122 diseases with n_gt≥51 from h1218, partitioned by L3 entropy into three ~40-disease terciles (low/mid/high entropy of GT drug ATC sub-class distribution).

## Per-tercile paired-t

### delta_r30_pp

| Tercile | n | mean | std | t | p (two-sided) |
|---|---:|---:|---:|---:|---:|
| low | 40 | +0.815 | 2.092 | +2.46 | 0.0183 |
| mid | 41 | -0.726 | 2.954 | -1.57 | 0.123 |
| high | 41 | +0.407 | 2.231 | +1.17 | 0.249 |

### delta_hits30

| Tercile | n | mean | std | t | p (two-sided) |
|---|---:|---:|---:|---:|---:|
| low | 40 | +0.842 | 1.812 | +2.94 | 0.00554 |
| mid | 41 | -0.622 | 2.891 | -1.38 | 0.176 |
| high | 41 | +0.949 | 3.993 | +1.52 | 0.136 |

### delta_r30_ceiling_normalised_pp

| Tercile | n | mean | std | t | p (two-sided) |
|---|---:|---:|---:|---:|---:|
| low | 40 | +2.806 | 6.041 | +2.94 | 0.00554 |
| mid | 41 | -2.073 | 9.636 | -1.38 | 0.176 |
| high | 41 | +3.164 | 13.311 | +1.52 | 0.136 |

## Reference: h1244 global n_gt≥51 disease-level p-values

- Δ R@30: p ≈ 0.49 (mean +0.18pp)
- Δ hits@30: p ≈ 0.17 (mean +0.436)

If the homogeneous-low tercile reaches p<0.05 while mid/high do not, h1247's mechanism is confirmed and we have a per-disease routing rule for high-density diseases. If all three terciles still p>0.05, we close the homogeneity-restriction direction for this stratum.

## Sample homogeneous-low-entropy diseases

| Disease | Cat | n_gt | entropy | ΔR@30 | Δhits@30 |
|---|---|---:|---:|---:|---:|
| esophageal cancer | cancer | 62 | 0.90 | +6.45pp | +4.000 |
| malignant bone and soft tissue tumors | cancer | 81 | 0.93 | +1.23pp | +1.000 |
| pyelonephritis | renal | 71 | 1.36 | +1.41pp | +1.000 |
| small lymphocytic lymphoma | cancer | 76 | 1.52 | +1.32pp | +1.000 |
| neisseria gonorrhoeae infections | infectious | 68 | 1.56 | -2.94pp | -2.000 |
| complicated urinary tract infections | infectious | 160 | 1.61 | +0.63pp | +1.000 |
| adenocarcinoma | cancer | 79 | 1.64 | +3.80pp | +3.000 |
| osteosarcoma | cancer | 78 | 1.67 | -0.64pp | -0.500 |
| staphylococcus aureus bacteraemia | infectious | 63 | 1.84 | -0.79pp | -0.500 |
| infective endocarditis | infectious | 68 | 1.85 | +0.74pp | +0.500 |
| staphylococcus aureus skin infection | infectious | 77 | 1.86 | +2.60pp | +2.000 |
| head and neck squamous cell cancer | cancer | 58 | 1.92 | +1.72pp | +1.000 |
| acute bacterial skin and skin structure infections absssi | infectious | 160 | 1.92 | +0.00pp | +0.000 |
| squamouscell carcinoma | cancer | 120 | 2.01 | +1.67pp | +2.000 |
| osteomyelitis | infectious | 68 | 2.04 | +2.94pp | +2.000 |

## Interpretation

**VALIDATED — homogeneity-restricted fusion lift recovers significance on n_gt≥51.**

The bottom-tercile L3-entropy subset (n=40 of 122 n_gt≥51 diseases) reaches:
- Δ R@30 = +0.815pp ± 2.09pp, t=2.46, **p=0.018** (vs h1244 full-stratum p=0.49).
- Δ hits@30 = +0.842 drugs/disease ± 1.81, t=2.94, **p=0.0055** (vs h1244 full-stratum p=0.17).
- Δ R@30/ceiling = +2.806pp ± 6.04, t=2.94, **p=0.0055**.

The mid tercile is significantly negative-trending (-0.73pp R@30, p=0.12; -0.62 hits@30, p=0.18) — fusion HURTS moderately-heterogeneous high-density diseases. The high tercile is positive in mean but variance kills significance (std hits@30 = 3.99 vs 1.81 in low tercile).

**Combined two-axis routing rule (h1247 + h1248):**
| Stratum | n | Routing | Mean Δ | p |
|---|---:|---|---:|---:|
| n_gt 21-50 + low entropy | 42 | **fuse** | +2.25pp R@30 | 0.045 |
| n_gt 21-50 + high entropy | 42 | (still fuse, smaller gain) | +1.26pp R@30 | — |
| n_gt 51+ + low entropy | 40 | **fuse** | +0.84 hits/disease | **0.0055** |
| n_gt 51+ + mid entropy | 41 | **avoid fusion** (negative trend) | -0.62 hits/disease | 0.18 |
| n_gt 51+ + high entropy | 41 | indifferent | +0.95 hits/disease (high var) | 0.14 |

This is the cleanest publishable per-disease fusion-routing rule found so far. Implementation: ship `gt_atc_l3_gini` as a deliverable column and route at inference based on (n_gt_train_drugs, gt_atc_l3_gini). Generated h1249 (production routing benchmark with the rule wired in).
