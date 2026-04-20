# Clean Embedding Benchmark — `graphsage_h1200_s42_256`

- Hypothesis: **h1199** (tier-free multi-metric kNN baseline)
- k = 20, seeds = [42, 123, 456, 789, 2024]
- eval_gt: `data/reference/expanded_ground_truth.json`
- knn_gt (aggregation): `data/cache/ground_truth_cache.json`
- Eligible diseases (GT + embeddings): **956**

## Headline metrics (mean ± std across seeds)

| Metric | Value |
|---|---|
| R@30 per-drug | 10.81% ± 0.96% |
| R@30 per-drug / ceiling (h1240) | 13.64% ± 1.06% |
| MRR (per-triple) | 0.0152 ± 0.0008 |
| AUPRC (per-triple) | 0.0300 ± 0.0034 |
| AUROC (per-triple) | 0.5462 ± 0.0024 |

## Hits@K — per-drug (fraction of GT drugs recovered in top-K, averaged over test diseases)

| K | mean | std | ceiling-adjusted mean (hits/min(K,|GT|)) | ceiling-adj std |
|---|---|---|---|---|
| 1 | 0.88% | 0.23% | 16.82% | 1.49% |
| 5 | 4.23% | 0.74% | 16.51% | 1.89% |
| 10 | 6.65% | 0.86% | 14.65% | 1.68% |
| 30 | 10.81% | 0.96% | 13.64% | 1.06% |
| 100 | 16.47% | 1.26% | 17.18% | 1.23% |

## Hits@K — per-test-triple (fraction of held-out (disease, drug) pairs ranked ≤ K)

| K | mean | std |
|---|---|---|
| 1 | 0.44% | 0.03% |
| 5 | 2.13% | 0.17% |
| 10 | 3.46% | 0.24% |
| 30 | 6.04% | 0.27% |
| 100 | 11.26% | 0.74% |

## Per-category breakdown (averaged over seeds)

| Category | n_dis | R@30 | Ceiling | R@30/Ceiling | Hits@30 triple | MRR |
|---|---|---|---|---|---|---|
| other | 55 | 11.43%±3.99% | 95.10% | 12.04%±3.85% | 8.91% | 0.0238 |
| infectious | 23 | 16.86%±3.41% | 84.95% | 19.60%±3.11% | 7.85% | 0.0202 |
| cancer | 20 | 10.86%±2.11% | 73.28% | 18.56%±3.50% | 6.16% | 0.0140 |
| metabolic | 15 | 3.94%±2.03% | 85.88% | 5.60%±2.34% | 2.99% | 0.0098 |
| cardiovascular | 12 | 6.09%±1.83% | 64.80% | 12.60%±5.37% | 4.54% | 0.0094 |
| neurological | 9 | 5.66%±2.29% | 95.54% | 5.93%±2.22% | 5.24% | 0.0145 |
| gastrointestinal | 9 | 9.15%±6.59% | 76.65% | 10.88%±6.76% | 2.90% | 0.0081 |
| hematological | 7 | 7.45%±4.33% | 82.81% | 9.50%±4.62% | 6.79% | 0.0193 |
| dermatological | 6 | 20.14%±6.85% | 88.00% | 23.56%±6.94% | 17.03% | 0.0447 |
| autoimmune | 6 | 20.37%±5.59% | 79.90% | 25.73%±6.52% | 14.68% | 0.0327 |
| respiratory | 6 | 16.68%±10.71% | 67.94% | 24.69%±16.58% | 4.74% | 0.0100 |
| renal | 4 | 6.15%±1.25% | 58.87% | 16.54%±4.62% | 4.77% | 0.0098 |
| ophthalmic | 3 | 13.55%±13.74% | 95.42% | 14.39%±13.54% | 17.96% | 0.0569 |
| musculoskeletal | 3 | 7.71%±9.53% | 93.18% | 7.98%±9.44% | 7.63% | 0.0246 |
| immunological | 3 | 5.38%±4.73% | 91.85% | 6.19%±4.22% | 10.03% | 0.0094 |
| psychiatric | 3 | 1.91%±1.44% | 54.40% | 6.30%±5.13% | 1.91% | 0.0060 |
| endocrine | 2 | 6.68%±4.62% | 100.00% | 6.68%±4.62% | 7.74% | 0.0321 |
| reproductive | 2 | 0.40%±0.70% | 87.10% | 0.83%±1.44% | 0.40% | 0.0018 |

## Per-seed detail

| Seed | n_dis | n_triples | R@30 | H@10 drug | H@30 triple | MRR | AUPRC | AUROC |
|---|---|---|---|---|---|---|---|---|
| 42 | 184 | 7362 | 9.52% | 5.48% | 5.76% | 0.0143 | 0.0309 | 0.5416 |
| 123 | 190 | 6624 | 10.10% | 6.41% | 5.86% | 0.0147 | 0.0257 | 0.5468 |
| 456 | 187 | 6169 | 10.82% | 6.12% | 6.42% | 0.0160 | 0.0268 | 0.5482 |
| 789 | 186 | 8590 | 11.35% | 7.32% | 5.84% | 0.0146 | 0.0350 | 0.5462 |
| 2024 | 187 | 6973 | 12.28% | 7.89% | 6.31% | 0.0162 | 0.0314 | 0.5481 |
