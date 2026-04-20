# h1259 — Per-disease AUPRC reframe (rank-equivariant metric)

**Premise:** h1255 found per-disease z-norm collapses pooled AUROC by 0.147 even when the per-disease ranking is byte-identical to the raw scores (R@30 unchanged). This script tests whether per-disease AUPRC (mean MAP across diseases) — which IS rank-equivariant — confirms that the h1228/h1249/h1255 'AUPRC regression' is a pooling artifact rather than a real ranking degradation.

If the artifact thesis is correct, we expect:
- `per-disease AUPRC` for `concat_l2_znorm` ≈ `concat_l2_raw` (z-norm preserves per-disease rank)
- `per-disease AUPRC` for `soft_blend_w0.50` ≥ `concat_l2_raw` (R@30 lift translates to AUPRC lift)
- `pooled AUPRC` collapses on the z-norm modes (the h1255 finding)

## Aggregate (mean ± std across 5 seeds)

| Mode | R@30 | per-dis-AUPRC | per-dis-AUROC | pooled-AUPRC | pooled-AUROC |
|---|---|---|---|---|---|
| `concat_l2_raw` | 20.87%±0.91% | 0.1230±0.0088 | 0.6211±0.0045 | 0.0642±0.0033 | 0.5851±0.0086 |
| `concat_l2_znorm` | 20.87%±0.91% | 0.1230±0.0088 | 0.6211±0.0045 | 0.0526±0.0042 | 0.4380±0.0347 |
| `soft_blend_w000` | 20.87%±0.91% | 0.1230±0.0088 | 0.6211±0.0045 | 0.0526±0.0042 | 0.4380±0.0347 |
| `soft_blend_w025` | 20.92%±0.90% | 0.1234±0.0089 | 0.6217±0.0045 | 0.0530±0.0043 | 0.4416±0.0354 |
| `soft_blend_w050` | 20.92%±0.90% | 0.1235±0.0089 | 0.6217±0.0045 | 0.0531±0.0043 | 0.4423±0.0357 |
| `soft_blend_w075` | 20.92%±0.90% | 0.1235±0.0090 | 0.6217±0.0045 | 0.0532±0.0043 | 0.4444±0.0359 |
| `soft_blend_w100` | 20.92%±0.89% | 0.1231±0.0089 | 0.6207±0.0045 | 0.0525±0.0042 | 0.4432±0.0358 |

## Per-seed paired-t vs `concat_l2_raw` (n=5)

| Mode | ΔR@30 (p) | Δper-dis-AUPRC (p) | Δper-dis-AUROC (p) | Δpooled-AUPRC (p) | Δpooled-AUROC (p) |
|---|---|---|---|---|---|
| `concat_l2_znorm` | +0.0000pp (0) | +0.00000 (0) | +0.00000 (0) | -0.01160 (0.000184) | -0.14704 (0.000607) |
| `soft_blend_w000` | +0.0000pp (0) | +0.00000 (0) | +0.00000 (0) | -0.01160 (0.000184) | -0.14704 (0.000607) |
| `soft_blend_w025` | +0.0423pp (0.00292) | +0.00044 (0.0113) | +0.00059 (0.00121) | -0.01116 (0.000224) | -0.14346 (0.000686) |
| `soft_blend_w050` | +0.0485pp (0.00795) | +0.00054 (0.013) | +0.00059 (0.00134) | -0.01112 (0.000266) | -0.14279 (0.000712) |
| `soft_blend_w075` | +0.0439pp (0.0678) | +0.00055 (0.0233) | +0.00059 (0.0014) | -0.01099 (0.000313) | -0.14067 (0.000742) |
| `soft_blend_w100` | +0.0490pp (0.0948) | +0.00008 (0.763) | -0.00040 (0.112) | -0.01170 (0.000227) | -0.14187 (0.000711) |
