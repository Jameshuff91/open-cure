# h1273 — Refined GLOBAL recipe (drop n_gt=1 singletons from soft-blend)

**Premise:** h1272's per-disease audit found n_gt=1 singletons (n=58/1002 rows) have mean Δ per-dis AUPRC = -0.0043 under the GLOBAL soft-blend — fusion HURTS them.

Two refined subsets tested (blend_w=0.5, n_seeds=20):

- `SUBSET_D_GLOBAL` — current canonical (every disease blended)
- `SUBSET_E_NOSINGLE` — n_gt ≥ 2 (58 singletons dropped)
- `SUBSET_F_NGT6` — n_gt ≥ 6 (singletons + 2-5 bucket dropped)

## Aggregate (mean ± std across 20 seeds)

| Mode | R@30 | per-dis-AUPRC | per-dis-AUROC | pooled-AUPRC | pooled-AUROC | in_subset_mean |
|---|---|---|---|---|---|---|
| `concat_l2_raw` | 21.35%±1.13% | 0.1241±0.0094 | 0.6240±0.0054 | 0.0647±0.0052 | 0.5856±0.0064 | 0.0 |
| `SUBSET_D_GLOBAL` | 21.54%±1.12% | 0.1275±0.0102 | 0.6345±0.0049 | 0.0595±0.0052 | 0.4531±0.0350 | 199.7 |
| `SUBSET_E_NOSINGLE` | 21.56%±1.10% | 0.1275±0.0090 | 0.6343±0.0049 | 0.0610±0.0055 | 0.4396±0.0330 | 188.7 |
| `SUBSET_F_NGT6` | 21.42%±1.14% | 0.1268±0.0091 | 0.6316±0.0047 | 0.0674±0.0065 | 0.4028±0.0263 | 148.7 |

## Paired-t vs `concat_l2_raw` (n=20)

| Subset | ΔR@30 (p) | Δper-dis-AUPRC (p) | Δper-dis-AUROC (p) | Δpooled-AUPRC (p) | Δpooled-AUROC (p) |
|---|---|---|---|---|---|
| `SUBSET_D_GLOBAL` | +0.1848pp (0.0848) | +0.00343 (0.00223) | +0.01041 (7.94e-12) | -0.00522 (1.3e-09) | -0.13251 (1.97e-13) |
| `SUBSET_E_NOSINGLE` | +0.2098pp (0.0418) | +0.00345 (3.73e-05) | +0.01028 (6.84e-12) | -0.00370 (1.41e-07) | -0.14592 (1.24e-14) |
| `SUBSET_F_NGT6` | +0.0661pp (0.345) | +0.00272 (2.28e-05) | +0.00759 (2.77e-14) | +0.00272 (0.00041) | -0.18280 (1e-18) |

## Paired-t vs `SUBSET_D_GLOBAL` (current canonical, n=20)

| Subset | ΔR@30 (p) | Δper-dis-AUPRC (p) | Δper-dis-AUROC (p) |
|---|---|---|---|
| `SUBSET_E_NOSINGLE` | +0.0250pp (0.33) | +0.00002 (0.967) | -0.00013 (0.435) |
| `SUBSET_F_NGT6` | -0.1187pp (0.235) | -0.00070 (0.291) | -0.00282 (1.15e-05) |

## Promotion gate (beat `SUBSET_D_GLOBAL` on Δ per-dis AUPRC at p<0.05)

| Subset | Δ vs GLOBAL | p | Decision |
|---|---|---|---|
| `SUBSET_E_NOSINGLE` | +0.00002 | 0.967 | **STAY with GLOBAL** |
| `SUBSET_F_NGT6` | -0.00070 | 0.291 | **STAY with GLOBAL** |
