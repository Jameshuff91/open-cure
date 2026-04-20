# h1255 — Score-scale-normalised soft-blend fusion

**Premise:** h1228 + h1249 both lifted R@30 on a targeted subset but regressed AUPRC/AUROC at p<0.1, traceable to per-disease score-scale mismatch when swapping embedding spaces. This script tests whether per-disease z-normalisation + soft blend (instead of hard switch) recovers the targeted-subset lift WITHOUT the global pooled-AUPRC regression.

**Modes:**

- `concat_l2_raw` — production baseline (h1249 anchor)
- `concat_l2_znorm` — z-normalise concat_l2 scores per-disease; tests the AUPRC effect of z-norm alone
- `soft_blend_w*` — on (n_gt≥51 + mid-entropy) flipped subset only: `score = w·z(node2vec) + (1-w)·z(concat_l2)`. Other diseases use `z(concat_l2)`. Sweep w ∈ {0.0, 0.25, 0.5, 0.75, 1.0}.

## Aggregate (mean ± std across 5 seeds)

| Mode | R@30 | H@30 drug | MRR | AUPRC | AUROC |
|---|---|---|---|---|---|
| `concat_l2_raw` | 20.87%±0.91% | 20.87%±0.91% | 0.0296±0.0036 | 0.0642±0.0033 | 0.5851±0.0086 |
| `concat_l2_znorm` | 20.87%±0.91% | 20.87%±0.91% | 0.0296±0.0036 | 0.0526±0.0042 | 0.4380±0.0347 |
| `soft_blend_w000` | 20.87%±0.91% | 20.87%±0.91% | 0.0296±0.0036 | 0.0526±0.0042 | 0.4380±0.0347 |
| `soft_blend_w025` | 20.92%±0.90% | 20.92%±0.90% | 0.0297±0.0036 | 0.0530±0.0043 | 0.4416±0.0354 |
| `soft_blend_w050` | 20.92%±0.90% | 20.92%±0.90% | 0.0298±0.0037 | 0.0531±0.0043 | 0.4423±0.0357 |
| `soft_blend_w075` | 20.92%±0.90% | 20.92%±0.90% | 0.0298±0.0037 | 0.0532±0.0043 | 0.4444±0.0359 |
| `soft_blend_w100` | 20.92%±0.89% | 20.92%±0.89% | 0.0297±0.0036 | 0.0525±0.0042 | 0.4432±0.0358 |

## Per-seed paired-t vs `concat_l2_raw` (n=5)

| Mode | ΔR@30 | t (R@30) | p | ΔAUPRC | t (AUPRC) | p | ΔAUROC | t (AUROC) | p |
|---|---|---|---|---|---|---|---|---|---|
| `concat_l2_znorm` | +0.0000pp | +inf | 0 | -0.01160 | -13.31 | 0.000184 | -0.14704 | -9.80 | 0.000607 |
| `soft_blend_w000` | +0.0000pp | +inf | 0 | -0.01160 | -13.31 | 0.000184 | -0.14704 | -9.80 | 0.000607 |
| `soft_blend_w025` | +0.0423pp | +6.48 | 0.00292 | -0.01116 | -12.66 | 0.000224 | -0.14346 | -9.50 | 0.000686 |
| `soft_blend_w050` | +0.0485pp | +4.92 | 0.00795 | -0.01112 | -12.11 | 0.000266 | -0.14279 | -9.41 | 0.000712 |
| `soft_blend_w075` | +0.0439pp | +2.49 | 0.0678 | -0.01099 | -11.63 | 0.000313 | -0.14067 | -9.31 | 0.000742 |
| `soft_blend_w100` | +0.0490pp | +2.18 | 0.0948 | -0.01170 | -12.62 | 0.000227 | -0.14187 | -9.41 | 0.000711 |

## Per-disease paired-t (n=1002): best-R@30 mode `soft_blend_w100` vs `concat_l2_raw`

- R@30: Δ_mean = +0.0491pp, t=+2.027, p=0.0429
- hits@30: Δ_mean = +0.0399, t=+1.713, p=0.0869

### Restricted to flipped subset (n=65 disease-seed rows)

- R@30: Δ_mean = +0.7571pp, t=+2.075, p=0.042
- hits@30: Δ_mean = +0.6154, t=+1.738, p=0.087

## Per-seed details

| Seed | n_test | flipped | low_cut | high_cut |
|---|---|---|---|---|
| 42 | 202 | 7 | 3.212 | 3.994 |
| 123 | 200 | 14 | 3.168 | 4.007 |
| 456 | 200 | 12 | 3.212 | 4.008 |
| 789 | 200 | 16 | 3.168 | 4.015 |
| 2024 | 200 | 16 | 3.168 | 4.006 |
