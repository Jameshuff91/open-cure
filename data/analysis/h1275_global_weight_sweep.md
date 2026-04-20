# h1275 — 20-seed weight sweep on SUBSET_D_GLOBAL (soft-blend)

**Premise:** h1269's 20-seed extension locked `SUBSET_D_GLOBAL` with w=0.5 as the canonical fusion recipe. h1268 preregistered a weight sweep at 5 seeds; h1269 showed 5-seed paired-t over-estimates effect sizes by 27-54%. This script runs the 20-seed sweep on the locked subset with w ∈ [0.3, 0.4, 0.5, 0.6, 0.7].

`blended(d) = w * z(n2v_score_d) + (1 - w) * z(concat_l2_score_d)` on every disease.

## Aggregate (mean ± std across 20 seeds)

| Mode | R@30 | per-dis-AUPRC | per-dis-AUROC | pooled-AUPRC | pooled-AUROC |
|---|---|---|---|---|---|
| `concat_l2_raw` | 21.35%±1.13% | 0.1241±0.0094 | 0.6240±0.0054 | 0.0647±0.0052 | 0.5856±0.0064 |
| `W030` | 21.66%±1.11% | 0.1272±0.0098 | 0.6345±0.0049 | 0.0592±0.0053 | 0.4581±0.0369 |
| `W040` | 21.68%±1.15% | 0.1278±0.0100 | 0.6345±0.0049 | 0.0596±0.0053 | 0.4552±0.0357 |
| `W050` | 21.54%±1.12% | 0.1275±0.0102 | 0.6345±0.0049 | 0.0595±0.0052 | 0.4531±0.0350 |
| `W060` | 21.33%±1.09% | 0.1267±0.0101 | 0.6344±0.0049 | 0.0589±0.0052 | 0.4517±0.0336 |
| `W070` | 21.15%±1.06% | 0.1260±0.0101 | 0.6344±0.0049 | 0.0579±0.0050 | 0.4513±0.0325 |

## Paired-t vs `concat_l2_raw` (n=20 seeds)

| Mode (w) | ΔR@30 (p) | Δper-dis-AUPRC (p) | Δper-dis-AUROC (p) |
|---|---|---|---|
| `W030` (w=0.3) | +0.3087pp (0.000594) | +0.00306 (0.000106) | +0.01044 (6.71e-12) |
| `W040` (w=0.4) | +0.3248pp (0.00282) | +0.00368 (0.00027) | +0.01043 (7.2e-12) |
| `W050` (w=0.5) | +0.1848pp (0.0848) | +0.00343 (0.00223) | +0.01041 (7.94e-12) |
| `W060` (w=0.6) | -0.0205pp (0.877) | +0.00263 (0.0204) | +0.01039 (9.43e-12) |
| `W070` (w=0.7) | -0.2017pp (0.189) | +0.00195 (0.0738) | +0.01036 (9.83e-12) |

## Paired-t vs `W050` (current canonical w=0.5, n=20)

| Mode (w) | ΔR@30 (p) | Δper-dis-AUPRC (p) | Δper-dis-AUROC (p) |
|---|---|---|---|
| `W030` (w=0.3) | +0.1239pp (0.0632) | -0.00037 (0.39) | +0.00003 (0.00544) |
| `W040` (w=0.4) | +0.1400pp (0.0141) | +0.00025 (0.244) | +0.00002 (0.0215) |
| `W060` (w=0.6) | -0.2053pp (0.024) | -0.00080 (0.000511) | -0.00003 (0.012) |
| `W070` (w=0.7) | -0.3865pp (0.000893) | -0.00148 (3.42e-05) | -0.00006 (4.1e-05) |

## Promotion gate (beat w=0.5 on Δ per-dis AUPRC at p<0.05, Δ>+0.0003)

| Mode (w) | Δ vs w=0.5 | p | Decision |
|---|---|---|---|
| `W030` (w=0.3) | -0.00037 | 0.39 | **STAY with w=0.5** |
| `W040` (w=0.4) | +0.00025 | 0.244 | **STAY with w=0.5** |
| `W060` (w=0.6) | -0.00080 | 0.000511 | **STAY with w=0.5** |
| `W070` (w=0.7) | -0.00148 | 3.42e-05 | **STAY with w=0.5** |

## Best-w per metric

| Metric | Best w | Mean |
|---|---|---|
| R@30 | 0.4 | 0.2168 |
| per-dis-AUPRC | 0.4 | 0.1278 |
| per-dis-AUROC | 0.3 | 0.6345 |
