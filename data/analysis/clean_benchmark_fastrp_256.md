# Clean Embedding Benchmark — `fastrp_256`

- Hypothesis: **h1199** (tier-free multi-metric kNN baseline)
- k = 20, seeds = [42, 123, 456, 789, 2024]
- eval_gt: `data/reference/expanded_ground_truth.json`
- knn_gt (aggregation): `data/cache/ground_truth_cache.json`
- Eligible diseases (GT + embeddings): **1,011**

## Headline metrics (mean ± std across seeds)

| Metric | Value |
|---|---|
| R@30 per-drug | 18.79% ± 0.92% |
| R@30 per-drug / ceiling (h1240) | 23.28% ± 0.76% |
| MRR (per-triple) | 0.0267 ± 0.0030 |
| AUPRC (per-triple) | 0.0584 ± 0.0032 |
| AUROC (per-triple) | 0.5790 ± 0.0071 |

## Hits@K — per-drug (fraction of GT drugs recovered in top-K, averaged over test diseases)

| K | mean | std | ceiling-adjusted mean (hits/min(K,|GT|)) | ceiling-adj std |
|---|---|---|---|---|
| 1 | 2.45% | 0.39% | 32.54% | 2.10% |
| 5 | 7.81% | 0.66% | 27.41% | 1.33% |
| 10 | 12.23% | 1.16% | 25.30% | 1.77% |
| 30 | 18.79% | 0.92% | 23.28% | 0.76% |
| 100 | 24.85% | 1.16% | 25.78% | 0.98% |

## Hits@K — per-test-triple (fraction of held-out (disease, drug) pairs ranked ≤ K)

| K | mean | std |
|---|---|---|
| 1 | 0.91% | 0.13% |
| 5 | 3.73% | 0.45% |
| 10 | 6.29% | 0.76% |
| 30 | 11.05% | 1.03% |
| 100 | 17.11% | 1.42% |

## Per-category breakdown (averaged over seeds)

| Category | n_dis | R@30 | Ceiling | R@30/Ceiling | Hits@30 triple | MRR |
|---|---|---|---|---|---|---|
| other | 64 | 16.28%±1.30% | 94.68% | 17.10%±1.47% | 11.78% | 0.0365 |
| infectious | 30 | 30.50%±2.66% | 88.84% | 35.84%±2.57% | 22.49% | 0.0495 |
| cancer | 23 | 17.54%±1.94% | 76.71% | 26.13%±3.33% | 7.64% | 0.0155 |
| cardiovascular | 13 | 8.50%±1.95% | 64.62% | 17.28%±3.71% | 6.83% | 0.0130 |
| metabolic | 11 | 10.90%±4.31% | 91.02% | 12.06%±4.06% | 9.22% | 0.0216 |
| neurological | 10 | 11.23%±2.25% | 87.46% | 14.70%±1.90% | 7.65% | 0.0167 |
| dermatological | 10 | 24.86%±4.83% | 83.16% | 31.00%±4.02% | 18.91% | 0.0501 |
| gastrointestinal | 8 | 16.05%±5.87% | 86.03% | 19.61%±2.72% | 15.03% | 0.0307 |
| autoimmune | 6 | 27.48%±4.40% | 70.37% | 39.21%±4.19% | 14.34% | 0.0379 |
| renal | 5 | 16.69%±6.18% | 66.60% | 27.88%±1.94% | 11.14% | 0.0296 |
| ophthalmic | 4 | 37.31%±10.77% | 97.69% | 37.97%±10.24% | 32.76% | 0.1056 |
| respiratory | 4 | 16.18%±9.47% | 82.56% | 23.01%±11.91% | 13.06% | 0.0295 |
| musculoskeletal | 4 | 24.65%±12.90% | 88.32% | 26.14%±12.23% | 25.10% | 0.0776 |
| psychiatric | 4 | 9.48%±2.78% | 49.55% | 29.88%±10.07% | 8.72% | 0.0221 |
| hematological | 3 | 16.26%±9.92% | 90.58% | 18.95%±9.11% | 13.85% | 0.0221 |
| immunological | 2 | 13.26%±21.25% | 94.82% | 14.65%±20.71% | 13.68% | 0.0102 |
| endocrine | 2 | 31.25%±19.94% | 100.00% | 31.25%±19.94% | 25.26% | 0.0537 |
| reproductive | 1 | 33.33%±0.00% | 100.00% | 33.33%±0.00% | 33.33% | 0.0250 |

## Per-seed detail

| Seed | n_dis | n_triples | R@30 | H@10 drug | H@30 triple | MRR | AUPRC | AUROC |
|---|---|---|---|---|---|---|---|---|
| 42 | 202 | 6477 | 18.16% | 10.75% | 11.12% | 0.0273 | 0.0574 | 0.5792 |
| 123 | 200 | 7530 | 18.50% | 12.00% | 10.98% | 0.0263 | 0.0604 | 0.5756 |
| 456 | 200 | 6583 | 19.12% | 13.00% | 12.18% | 0.0292 | 0.0629 | 0.5887 |
| 789 | 200 | 6447 | 20.40% | 14.02% | 11.77% | 0.0293 | 0.0577 | 0.5835 |
| 2024 | 200 | 9309 | 17.76% | 11.38% | 9.17% | 0.0212 | 0.0534 | 0.5679 |
