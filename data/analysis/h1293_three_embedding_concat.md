# h1293 — Three-embedding concat_l2 (N2V + FastRP + TransE) (20-seed SUBSET_D_GLOBAL)

**Universe:** 1,011 diseases (3-way embedding intersection ∩ knn_gt)

## Aggregate (mean ± std across 20 seeds)

| Mode | R@30 | per-dis-AUPRC | per-dis-AUROC | pooled-AUPRC | pooled-AUROC |
|---|---|---|---|---|---|
| `concat_l2_2way` | 21.35%±1.13% | 0.1241±0.0094 | 0.6240±0.0054 | 0.0647±0.0052 | 0.5856±0.0064 |
| `concat_l2_3way` | 21.10%±1.24% | 0.1230±0.0097 | 0.6243±0.0055 | 0.0666±0.0056 | 0.5877±0.0065 |
| `soft_blend_w050_2way` | 21.54%±1.12% | 0.1275±0.0102 | 0.6345±0.0049 | 0.0595±0.0052 | 0.4531±0.0350 |
| `soft_blend_w050_3way` | 21.50%±1.16% | 0.1278±0.0107 | 0.6350±0.0053 | 0.0593±0.0051 | 0.4411±0.0310 |

## Paired-t comparisons

| Comparison | ΔR@30 (p) | Δper-dis-AUPRC (p) | Δper-dis-AUROC (p) |
|---|---|---|---|
| 3-way concat vs 2-way anchor | -0.2511pp (0.0671) | -0.00108 (0.099) | +0.00026 (0.679) |
| 3-way soft-blend vs 2-way canonical | -0.0433pp (0.64) | +0.00028 (0.537) | +0.00056 (0.19) |
| 3-way soft-blend vs 2-way anchor | +0.1415pp (0.235) | +0.00371 (0.00181) | +0.01097 (9.96e-11) |
| 3-way soft-blend vs 3-way raw | +0.3926pp (0.0125) | +0.00479 (3.08e-05) | +0.01071 (2.54e-12) |

## Promotion gate

3-way concat_l2 vs 2-way concat_l2 anchor; pass if ΔR@30>+0.3pp AND ΔAUPRC>+0.001 both at p<0.05.

- ΔR@30 = -0.2511pp (p=0.0671)  → R@30 gate: **FAIL**
- ΔAUPRC = -0.00108 (p=0.099)  → AUPRC gate: **FAIL**
- **Decision: STAY with 2-way concat**
