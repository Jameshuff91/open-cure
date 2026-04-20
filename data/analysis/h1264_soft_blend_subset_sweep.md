# h1264 — Soft-blend subset sweep (does h1259's win generalise?)

**Premise:** h1259 validated soft_blend_w=0.5 ONLY on the n_gt>=51 ∩ mid-entropy 'flipped' subset (~7-16 diseases per seed): per-dis AUPRC Δ=+0.00054 p=0.013. Outside the subset, scores were concat_l2_znorm.

h1264 tests 4 progressively broader subsets, with concat_l2_RAW outside (not znorm), so cross-disease pooled AUPRC remains a valid sanity check.

## Aggregate (mean ± std across 5 seeds)

| Mode | R@30 | per-dis-AUPRC | per-dis-AUROC | pooled-AUPRC | pooled-AUROC | in_subset_mean |
|---|---|---|---|---|---|---|
| `concat_l2_raw` | 20.87%±0.91% | 0.1230±0.0088 | 0.6211±0.0045 | 0.0642±0.0033 | 0.5851±0.0086 | 0.0 |
| `SUBSET_A_FLIPPED` | 20.92%±0.90% | 0.1235±0.0089 | 0.6217±0.0045 | 0.0637±0.0056 | 0.5285±0.0293 | 13.0 |
| `SUBSET_B_HIGHDENS` | 20.95%±0.89% | 0.1239±0.0088 | 0.6223±0.0044 | 0.0778±0.0068 | 0.3659±0.0322 | 35.6 |
| `SUBSET_C_MODHIGH` | 20.97%±0.90% | 0.1241±0.0083 | 0.6238±0.0042 | 0.0748±0.0077 | 0.3521±0.0283 | 72.8 |
| `SUBSET_D_GLOBAL` | 21.30%±1.12% | 0.1277±0.0070 | 0.6319±0.0036 | 0.0582±0.0042 | 0.4362±0.0397 | 200.4 |

## Per-seed paired-t vs `concat_l2_raw` (n=5)

| Subset | ΔR@30 (p) | Δper-dis-AUPRC (p) | Δper-dis-AUROC (p) | Δpooled-AUPRC (p) | Δpooled-AUROC (p) |
|---|---|---|---|---|---|
| `SUBSET_A_FLIPPED` | +0.0485pp (0.00795) | +0.00054 (0.013) | +0.00059 (0.00134) | -0.00054 (0.773) | -0.05661 (0.0196) |
| `SUBSET_B_HIGHDENS` | +0.0799pp (0.00199) | +0.00094 (0.00613) | +0.00123 (0.000139) | +0.01361 (0.00244) | -0.21916 (5.63e-05) |
| `SUBSET_C_MODHIGH` | +0.1010pp (0.109) | +0.00107 (0.0456) | +0.00274 (0.000542) | +0.01064 (0.0124) | -0.23298 (2.38e-05) |
| `SUBSET_D_GLOBAL` | +0.4281pp (0.1) | +0.00469 (0.108) | +0.01080 (0.00549) | -0.00600 (0.0035) | -0.14886 (0.000951) |

## Promotion gate decisions (Δ per-dis AUPRC ≥ +0.0005, p < 0.05)

| Subset | Δ per-dis AUPRC | p | Decision |
|---|---|---|---|
| `SUBSET_A_FLIPPED` | +0.00054 | 0.013 | **PROMOTE** |
| `SUBSET_B_HIGHDENS` | +0.00094 | 0.00613 | **PROMOTE** |
| `SUBSET_C_MODHIGH` | +0.00107 | 0.0456 | **PROMOTE** |
| `SUBSET_D_GLOBAL` | +0.00469 | 0.108 | **STAY** |
