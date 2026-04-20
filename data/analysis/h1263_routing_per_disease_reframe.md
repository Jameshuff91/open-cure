# h1263 — Re-evaluate cat_gated + entropy_routed under per-disease AUPRC

**Premise:** h1259 showed pooled AUPRC/AUROC are scale-confounded across embedding spaces. Per-disease AUPRC is rank-equivariant (immune to per-disease score scaling). h1228 (category-gated fusion) and h1249 (entropy-routed fusion) were both INVALIDATED on pooled-AUPRC regression; this script re-tests them under the corrected metric.

**Promotion gate:** per-disease AUPRC Δ ≥ 0 with p < 0.1 over 5 seeds.

## Aggregate (mean ± std across 5 seeds)

| Mode | R@30 | per-dis-AUPRC | per-dis-AUROC | pooled-AUPRC | pooled-AUROC |
|---|---|---|---|---|---|
| `node2vec` | 19.55%±1.18% | 0.1195±0.0086 | 0.6133±0.0052 | 0.0569±0.0023 | 0.5766±0.0067 |
| `concat_l2_raw` | 20.87%±0.91% | 0.1230±0.0088 | 0.6211±0.0045 | 0.0642±0.0033 | 0.5851±0.0086 |
| `cat_gated` | 21.06%±1.01% | 0.1233±0.0095 | 0.6209±0.0052 | 0.0623±0.0027 | 0.5826±0.0080 |
| `entropy_routed` | 20.92%±0.89% | 0.1231±0.0089 | 0.6207±0.0045 | 0.0628±0.0033 | 0.5834±0.0083 |

## Per-seed paired-t vs `concat_l2_raw` (n=5)

| Mode | ΔR@30 (p) | Δper-dis-AUPRC (p) | Δper-dis-AUROC (p) | Δpooled-AUPRC (p) | Δpooled-AUROC (p) |
|---|---|---|---|---|---|
| `node2vec` | -1.3260pp (0.0654) | -0.00348 (0.148) | -0.00778 (0.0328) | -0.00728 (0.000908) | -0.00850 (0.00102) |
| `cat_gated` | +0.1860pp (0.227) | +0.00031 (0.502) | -0.00021 (0.819) | -0.00188 (0.019) | -0.00244 (0.0067) |
| `entropy_routed` | +0.0490pp (0.0948) | +0.00008 (0.763) | -0.00040 (0.112) | -0.00140 (0.0536) | -0.00167 (0.0694) |

## Promotion gate decisions

| Mode | Δ per-disease AUPRC | p | Decision |
|---|---|---|---|
| `cat_gated` | +0.00031 | 0.502 | **STAY_INVALIDATED** |
| `entropy_routed` | +0.00008 | 0.763 | **STAY_INVALIDATED** |
