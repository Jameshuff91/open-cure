# h1264 — Soft-blend subset sweep (does h1259's win generalise?)

**Premise:** h1259 validated soft_blend_w=0.5 ONLY on the n_gt>=51 ∩ mid-entropy 'flipped' subset (~7-16 diseases per seed): per-dis AUPRC Δ=+0.00054 p=0.013. Outside the subset, scores were concat_l2_znorm.

h1264 tests 4 progressively broader subsets, with concat_l2_RAW outside (not znorm), so cross-disease pooled AUPRC remains a valid sanity check.

## Aggregate (mean ± std across 5 seeds)

| Mode | R@30 | per-dis-AUPRC | per-dis-AUROC | pooled-AUPRC | pooled-AUROC | in_subset_mean |
|---|---|---|---|---|---|---|
| `concat_l2_raw` | 21.35%±1.13% | 0.1241±0.0094 | 0.6240±0.0054 | 0.0647±0.0052 | 0.5856±0.0064 | 0.0 |
| `SUBSET_A_FLIPPED` | 21.37%±1.12% | 0.1243±0.0094 | 0.6245±0.0054 | 0.0642±0.0071 | 0.5365±0.0213 | 10.9 |
| `SUBSET_B_HIGHDENS` | 21.40%±1.11% | 0.1248±0.0094 | 0.6252±0.0053 | 0.0779±0.0079 | 0.3723±0.0238 | 34.5 |
| `SUBSET_C_MODHIGH` | 21.44%±1.12% | 0.1254±0.0093 | 0.6267±0.0053 | 0.0752±0.0073 | 0.3594±0.0227 | 71.0 |
| `SUBSET_D_GLOBAL` | 21.54%±1.12% | 0.1275±0.0102 | 0.6345±0.0049 | 0.0595±0.0052 | 0.4531±0.0350 | 199.7 |

## Per-seed paired-t vs `concat_l2_raw` (n=5)

| Subset | ΔR@30 (p) | Δper-dis-AUPRC (p) | Δper-dis-AUROC (p) | Δpooled-AUPRC (p) | Δpooled-AUROC (p) |
|---|---|---|---|---|---|
| `SUBSET_A_FLIPPED` | +0.0130pp (0.106) | +0.00025 (0.00158) | +0.00041 (4.22e-07) | -0.00053 (0.493) | -0.04908 (3.39e-09) |
| `SUBSET_B_HIGHDENS` | +0.0423pp (0.00242) | +0.00068 (1.87e-07) | +0.00117 (1.54e-13) | +0.01317 (1.47e-12) | -0.21322 (1.49e-21) |
| `SUBSET_C_MODHIGH` | +0.0887pp (0.0039) | +0.00130 (2.84e-08) | +0.00270 (8.8e-15) | +0.01047 (9.48e-11) | -0.22614 (2.28e-22) |
| `SUBSET_D_GLOBAL` | +0.1848pp (0.0848) | +0.00343 (0.00223) | +0.01041 (7.94e-12) | -0.00522 (1.3e-09) | -0.13251 (1.97e-13) |

## Promotion gate decisions (Δ per-dis AUPRC ≥ +0.0005, p < 0.05)

| Subset | Δ per-dis AUPRC | p | Decision |
|---|---|---|---|
| `SUBSET_A_FLIPPED` | +0.00025 | 0.00158 | **STAY** |
| `SUBSET_B_HIGHDENS` | +0.00068 | 1.87e-07 | **PROMOTE** |
| `SUBSET_C_MODHIGH` | +0.00130 | 2.84e-08 | **PROMOTE** |
| `SUBSET_D_GLOBAL` | +0.00343 | 0.00223 | **PROMOTE** |
