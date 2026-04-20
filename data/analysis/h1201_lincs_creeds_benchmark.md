# h1201 Phase E — LINCS × CREEDS reversal-connectivity standalone

- Pool: **72 diseases / 420 treatment edges** (LINCS drug sig ∧ CREEDS disease sig ∧ DRKG treatment edge)
- LINCS drugs candidate pool: 1,593
- Seeds: [42, 123, 456, 789, 2024]  |  holdout disease pct: 0.2  |  human_only: True

## Headline

| Metric | Mean | Std |
|---|---|---|
| R@30 per-drug | 1.08% | ±0.79% |
| MRR | 0.0310 | ±0.0275 |
| AUPRC | 0.0079 | ±0.0029 |
| AUROC | 0.4847 | ±0.0630 |

## Hits@K per-drug

| K | mean | std |
|---|---|---|
| 1 | 0.12% | ±0.24% |
| 5 | 0.26% | ±0.32% |
| 10 | 0.40% | ±0.25% |
| 30 | 1.08% | ±0.79% |
| 100 | 3.47% | ±3.17% |

## Comparison baselines

| Approach | R@30 (on full 1011-disease pool, not this subset) |
|---|---|
| Node2Vec 256 (h1199) | 19.55% ± 1.18% |
| h1215 ensemble (best DRKG) | 20.87% ± 0.91% |
| h1200 supervised GNN (invalidated) | 11.47% (best variant) |

**Ship gate:** ≥15% R@30 standalone on this subset → expand coverage. <15% → close h1201 inconclusive.