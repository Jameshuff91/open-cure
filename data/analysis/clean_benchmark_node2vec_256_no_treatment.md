# Clean Embedding Benchmark — `node2vec_256_no_treatment`

- Hypothesis: **h1199** (tier-free multi-metric kNN baseline)
- k = 20, seeds = [42, 123, 456, 789, 2024]
- eval_gt: `data/reference/expanded_ground_truth.json`
- knn_gt (aggregation): `data/cache/ground_truth_cache.json`
- Eligible diseases (GT + embeddings): **850**

## Headline metrics (mean ± std across seeds)

| Metric | Value |
|---|---|
| R@30 per-drug | 8.46% ± 0.94% |
| MRR (per-triple) | 0.0153 ± 0.0009 |
| AUPRC (per-triple) | 0.0300 ± 0.0032 |
| AUROC (per-triple) | 0.5555 ± 0.0037 |

## Hits@K — per-drug (fraction of GT drugs recovered in top-K, averaged over test diseases)

| K | mean | std |
|---|---|---|
| 1 | 0.96% | 0.53% |
| 5 | 2.80% | 0.61% |
| 10 | 4.35% | 0.62% |
| 30 | 8.46% | 0.94% |
| 100 | 14.15% | 1.01% |

## Hits@K — per-test-triple (fraction of held-out (disease, drug) pairs ranked ≤ K)

| K | mean | std |
|---|---|---|
| 1 | 0.45% | 0.05% |
| 5 | 1.90% | 0.15% |
| 10 | 3.29% | 0.25% |
| 30 | 6.97% | 0.46% |
| 100 | 12.94% | 0.87% |

## Per-category breakdown (averaged over seeds)

| Category | n_dis | R@30 | Hits@30 triple | MRR |
|---|---|---|---|---|
| other | 46 | 6.92%±1.37% | 6.81% | 0.0163 |
| cancer | 22 | 10.42%±2.39% | 6.65% | 0.0150 |
| infectious | 17 | 12.54%±2.63% | 9.50% | 0.0187 |
| cardiovascular | 13 | 6.71%±1.00% | 6.22% | 0.0139 |
| neurological | 12 | 4.86%±1.56% | 4.33% | 0.0082 |
| metabolic | 12 | 6.01%±1.53% | 5.73% | 0.0101 |
| gastrointestinal | 7 | 5.56%±1.82% | 6.61% | 0.0127 |
| dermatological | 6 | 17.87%±3.24% | 18.83% | 0.0592 |
| autoimmune | 6 | 14.04%±5.05% | 14.45% | 0.0336 |
| hematological | 6 | 6.29%±0.79% | 7.09% | 0.0180 |
| respiratory | 5 | 7.35%±3.35% | 7.63% | 0.0214 |
| renal | 4 | 4.07%±2.16% | 3.66% | 0.0069 |
| ophthalmic | 3 | 7.98%±5.18% | 7.14% | 0.0128 |
| psychiatric | 3 | 10.07%±6.46% | 7.70% | 0.0117 |
| endocrine | 3 | 13.60%±3.29% | 17.86% | 0.0419 |
| immunological | 2 | 7.02%±7.48% | 1.89% | 0.0051 |
| musculoskeletal | 1 | 16.91%±11.08% | 20.18% | 0.0544 |
| reproductive | 1 | 5.56%±5.56% | 5.56% | 0.0055 |

## Per-seed detail

| Seed | n_dis | n_triples | R@30 | H@10 drug | H@30 triple | MRR | AUPRC | AUROC |
|---|---|---|---|---|---|---|---|---|
| 42 | 167 | 6760 | 8.80% | 4.96% | 7.00% | 0.0152 | 0.0329 | 0.5570 |
| 123 | 169 | 7048 | 7.43% | 3.70% | 7.70% | 0.0168 | 0.0339 | 0.5566 |
| 456 | 165 | 6673 | 9.99% | 5.22% | 6.97% | 0.0155 | 0.0303 | 0.5598 |
| 789 | 165 | 6499 | 8.57% | 3.93% | 6.94% | 0.0146 | 0.0261 | 0.5556 |
| 2024 | 162 | 7591 | 7.49% | 3.92% | 6.24% | 0.0143 | 0.0266 | 0.5488 |
