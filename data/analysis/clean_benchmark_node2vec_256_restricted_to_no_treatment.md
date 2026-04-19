# Clean Embedding Benchmark — `node2vec_256`

- Hypothesis: **h1199** (tier-free multi-metric kNN baseline)
- k = 20, seeds = [42, 123, 456, 789, 2024]
- eval_gt: `data/reference/expanded_ground_truth.json`
- knn_gt (aggregation): `data/cache/ground_truth_cache.json`
- Eligible diseases (GT + embeddings): **838**

## Headline metrics (mean ± std across seeds)

| Metric | Value |
|---|---|
| R@30 per-drug | 17.09% ± 0.75% |
| MRR (per-triple) | 0.0246 ± 0.0033 |
| AUPRC (per-triple) | 0.0531 ± 0.0080 |
| AUROC (per-triple) | 0.5686 ± 0.0076 |

## Hits@K — per-drug (fraction of GT drugs recovered in top-K, averaged over test diseases)

| K | mean | std |
|---|---|---|
| 1 | 2.10% | 0.36% |
| 5 | 7.01% | 0.53% |
| 10 | 10.87% | 0.66% |
| 30 | 17.09% | 0.75% |
| 100 | 22.36% | 0.94% |

## Hits@K — per-test-triple (fraction of held-out (disease, drug) pairs ranked ≤ K)

| K | mean | std |
|---|---|---|
| 1 | 0.85% | 0.14% |
| 5 | 3.36% | 0.47% |
| 10 | 5.64% | 0.74% |
| 30 | 10.38% | 1.28% |
| 100 | 15.86% | 1.78% |

## Per-category breakdown (averaged over seeds)

| Category | n_dis | R@30 | Hits@30 triple | MRR |
|---|---|---|---|---|
| other | 47 | 15.39%±2.44% | 12.07% | 0.0314 |
| cancer | 21 | 16.17%±2.59% | 6.76% | 0.0163 |
| infectious | 17 | 24.87%±4.47% | 16.39% | 0.0370 |
| cardiovascular | 13 | 14.16%±3.03% | 10.10% | 0.0240 |
| metabolic | 11 | 13.41%±3.28% | 10.57% | 0.0274 |
| neurological | 10 | 10.24%±2.99% | 5.74% | 0.0123 |
| gastrointestinal | 7 | 10.80%±3.36% | 8.96% | 0.0225 |
| dermatological | 7 | 27.00%±3.39% | 25.82% | 0.0779 |
| autoimmune | 6 | 28.88%±6.77% | 24.31% | 0.0627 |
| hematological | 5 | 9.02%±4.96% | 9.37% | 0.0262 |
| respiratory | 5 | 19.11%±6.63% | 16.57% | 0.0370 |
| ophthalmic | 3 | 34.86%±8.20% | 29.07% | 0.0833 |
| renal | 3 | 9.58%±6.24% | 7.71% | 0.0191 |
| psychiatric | 3 | 17.38%±7.61% | 14.63% | 0.0261 |
| immunological | 2 | 12.97%±8.40% | 9.34% | 0.0094 |
| endocrine | 2 | 42.26%±32.28% | 41.70% | 0.0488 |
| musculoskeletal | 2 | 14.91%±2.72% | 14.82% | 0.0414 |
| reproductive | 1 | 16.67%±16.67% | 16.67% | 0.0182 |

## Per-seed detail

| Seed | n_dis | n_triples | R@30 | H@10 drug | H@30 triple | MRR | AUPRC | AUROC |
|---|---|---|---|---|---|---|---|---|
| 42 | 165 | 5874 | 17.84% | 11.85% | 11.41% | 0.0279 | 0.0566 | 0.5752 |
| 123 | 168 | 7494 | 17.94% | 11.26% | 11.98% | 0.0273 | 0.0666 | 0.5769 |
| 456 | 160 | 5914 | 16.42% | 10.17% | 10.72% | 0.0266 | 0.0515 | 0.5717 |
| 789 | 168 | 8683 | 17.18% | 10.94% | 8.96% | 0.0203 | 0.0449 | 0.5606 |
| 2024 | 166 | 7874 | 16.05% | 10.12% | 8.81% | 0.0210 | 0.0460 | 0.5586 |
