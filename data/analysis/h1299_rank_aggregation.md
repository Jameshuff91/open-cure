# h1299 — RRF + Borda rank-aggregation on N2V + FastRP (20-seed SUBSET_D_GLOBAL)

**k_rrf = 60**, `N = 1,011` diseases

## Aggregate (mean ± std across 20 seeds)

| Mode | R@30 | per-dis-AUPRC | per-dis-AUROC | pooled-AUPRC | pooled-AUROC |
|---|---|---|---|---|---|
| `concat_l2_2way` | 21.35%±1.13% | 0.1241±0.0094 | 0.6240±0.0054 | 0.0647±0.0052 | 0.5856±0.0064 |
| `soft_blend_w050_2way` | 21.54%±1.12% | 0.1275±0.0102 | 0.6345±0.0049 | 0.0595±0.0052 | 0.4531±0.0350 |
| `rrf_k60_n2v_fastrp` | 20.99%±1.34% | 0.1313±0.0099 | 0.6404±0.0056 | 0.0600±0.0046 | 0.5964±0.0071 |
| `rrf_k60_n2v_concat` | 21.37%±1.08% | 0.1291±0.0101 | 0.6345±0.0049 | 0.0559±0.0049 | 0.5933±0.0067 |
| `rrf_k60_3ranker` | 21.42%±1.26% | 0.1339±0.0103 | 0.6429±0.0052 | 0.0615±0.0049 | 0.5989±0.0074 |
| `borda_n2v_fastrp` | 20.94%±1.35% | 0.1300±0.0100 | 0.6404±0.0056 | 0.0597±0.0046 | 0.5964±0.0071 |
| `borda_3ranker` | 21.36%±1.25% | 0.1326±0.0103 | 0.6429±0.0052 | 0.0614±0.0049 | 0.5989±0.0074 |

## Paired-t vs `soft_blend_w050_2way` (canonical, n=20)

| Mode | ΔR@30 (p) | Δper-dis-AUPRC (p) | Δper-dis-AUROC (p) |
|---|---|---|---|
| `concat_l2_2way` | -0.1848pp (0.0848) | -0.00343 (0.00223) | -0.01041 (7.94e-12) |
| `rrf_k60_n2v_fastrp` | -0.5481pp (0.000398) | +0.00373 (0.000156) | +0.00595 (1.74e-08) |
| `rrf_k60_n2v_concat` | -0.1667pp (0.0179) | +0.00162 (0.00297) | +0.00002 (0.000949) |
| `rrf_k60_3ranker` | -0.1149pp (0.201) | +0.00636 (7.66e-07) | +0.00841 (1.77e-12) |
| `borda_n2v_fastrp` | -0.5981pp (0.000136) | +0.00252 (0.0042) | +0.00595 (1.77e-08) |
| `borda_3ranker` | -0.1821pp (0.057) | +0.00511 (1.4e-05) | +0.00840 (1.81e-12) |

## Promotion gate

Pass if ΔR@30>+0.15pp AND ΔAUPRC>+0.0005 both at p<0.05 vs canonical soft_blend_w050_2way.

| Mode | ΔR@30 | R@30 pass | ΔAUPRC | AUPRC pass | Decision |
|---|---|---|---|---|---|
| `rrf_k60_n2v_fastrp` | -0.5481pp (p=0.000398) | FAIL | +0.00373 (p=0.000156) | PASS | **STAY with soft_blend_w050_2way** |
| `rrf_k60_n2v_concat` | -0.1667pp (p=0.0179) | FAIL | +0.00162 (p=0.00297) | PASS | **STAY with soft_blend_w050_2way** |
| `rrf_k60_3ranker` | -0.1149pp (p=0.201) | FAIL | +0.00636 (p=7.66e-07) | PASS | **STAY with soft_blend_w050_2way** |
| `borda_n2v_fastrp` | -0.5981pp (p=0.000136) | FAIL | +0.00252 (p=0.0042) | PASS | **STAY with soft_blend_w050_2way** |
| `borda_3ranker` | -0.1821pp (p=0.057) | FAIL | +0.00511 (p=1.4e-05) | PASS | **STAY with soft_blend_w050_2way** |
