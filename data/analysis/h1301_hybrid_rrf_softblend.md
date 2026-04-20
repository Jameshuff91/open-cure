# h1301 — Hybrid 4-ranker RRF with soft-blend as 4th voter (20-seed SUBSET_D_GLOBAL)

## Aggregate (mean ± std across 20 seeds)

| Mode | R@30 | per-dis-AUPRC | per-dis-AUROC |
|---|---|---|---|
| `soft_blend_w050_2way` | 21.54%±1.12% | 0.1275±0.0102 | 0.6345±0.0049 |
| `rrf_k60_3ranker` | 21.42%±1.26% | 0.1339±0.0103 | 0.6429±0.0052 |
| `rrf_k60_4ranker` | 21.57%±1.20% | 0.1335±0.0103 | 0.6429±0.0052 |
| `rrf_k60_softblend_only` | 21.54%±1.12% | 0.1290±0.0101 | 0.6345±0.0049 |

## Paired-t comparisons

| Comparison | ΔR@30 (p) | Δper-dis-AUPRC (p) | Δper-dis-AUROC (p) |
|---|---|---|---|
| 4-ranker vs 3-ranker (R@30-recovery test) | +0.1468pp (0.00201) | -0.00037 (0.187) | +0.00002 (0.00713) |
| 4-ranker vs soft-blend canonical (dual-gate) | +0.0319pp (0.668) | +0.00599 (1.08e-07) | +0.00843 (1.45e-12) |
| sanity check — RRF(softblend) ≡ softblend | +0.0000pp (0) | +0.00152 (6.02e-09) | +0.00002 (6.28e-12) |

## Gates

**Gate A** (4-ranker beats 3-ranker on ΔR@30>+0.1pp at p<0.05 AND AUPRC Δ≥0 at p<0.05): R@30 PASS, AUPRC-no-regress FAIL

**Gate B** (4-ranker clears h1299 dual-arm gate vs soft-blend ΔR@30>+0.15pp AND ΔAUPRC>+0.0005 both p<0.05): R@30 FAIL, AUPRC PASS, overall FAIL — triple-recipe stays
