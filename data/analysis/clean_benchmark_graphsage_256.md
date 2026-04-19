# Clean Embedding Benchmark — `graphsage_256`

- Hypothesis: **h1199** (tier-free multi-metric kNN baseline)
- k = 20, seeds = [42, 123, 456, 789, 2024]
- eval_gt: `data/reference/expanded_ground_truth.json`
- knn_gt (aggregation): `data/cache/ground_truth_cache.json`
- Eligible diseases (GT + embeddings): **850**

## Headline metrics (mean ± std across seeds)

| Metric | Value |
|---|---|
| R@30 per-drug | 8.17% ± 0.53% |
| R@30 per-drug / ceiling (h1240) | 11.43% ± 0.40% |
| MRR (per-triple) | 0.0126 ± 0.0010 |
| AUPRC (per-triple) | 0.0322 ± 0.0026 |
| AUROC (per-triple) | 0.5529 ± 0.0046 |

## Hits@K — per-drug (fraction of GT drugs recovered in top-K, averaged over test diseases)

| K | mean | std | ceiling-adjusted mean (hits/min(K,|GT|)) | ceiling-adj std |
|---|---|---|---|---|
| 1 | 0.74% | 0.15% | 14.37% | 1.69% |
| 5 | 2.97% | 0.27% | 13.57% | 1.41% |
| 10 | 4.70% | 0.44% | 12.27% | 1.22% |
| 30 | 8.17% | 0.53% | 11.43% | 0.40% |
| 100 | 13.68% | 1.12% | 14.54% | 1.05% |

## Hits@K — per-test-triple (fraction of held-out (disease, drug) pairs ranked ≤ K)

| K | mean | std |
|---|---|---|
| 1 | 0.34% | 0.03% |
| 5 | 1.58% | 0.19% |
| 10 | 2.74% | 0.34% |
| 30 | 5.47% | 0.44% |
| 100 | 10.88% | 0.84% |

## Per-category breakdown (averaged over seeds)

| Category | n_dis | R@30 | Ceiling | R@30/Ceiling | Hits@30 triple | MRR |
|---|---|---|---|---|---|---|
| other | 46 | 7.30%±0.68% | 92.39% | 8.26%±0.66% | 6.02% | 0.0132 |
| cancer | 22 | 10.91%±2.94% | 67.06% | 20.66%±4.02% | 5.91% | 0.0136 |
| infectious | 17 | 9.07%±1.15% | 85.25% | 11.85%±1.92% | 6.56% | 0.0138 |
| cardiovascular | 13 | 6.84%±0.62% | 66.86% | 13.91%±3.19% | 5.23% | 0.0107 |
| neurological | 12 | 4.61%±1.09% | 85.11% | 5.46%±1.24% | 2.91% | 0.0066 |
| metabolic | 12 | 6.60%±2.93% | 90.48% | 7.53%±3.39% | 3.80% | 0.0078 |
| gastrointestinal | 7 | 2.44%±1.10% | 75.88% | 4.80%±2.29% | 2.93% | 0.0059 |
| dermatological | 6 | 12.32%±7.43% | 93.39% | 13.34%±7.91% | 14.56% | 0.0433 |
| autoimmune | 6 | 15.09%±5.93% | 72.91% | 21.98%±5.78% | 15.29% | 0.0420 |
| hematological | 6 | 8.07%±6.24% | 77.90% | 11.65%±8.29% | 6.15% | 0.0127 |
| respiratory | 5 | 9.31%±7.05% | 78.78% | 11.76%±8.07% | 6.54% | 0.0207 |
| renal | 4 | 5.99%±2.76% | 60.70% | 13.09%±3.18% | 3.03% | 0.0081 |
| ophthalmic | 3 | 18.58%±7.46% | 90.72% | 18.81%±7.24% | 10.87% | 0.0215 |
| psychiatric | 3 | 5.38%±2.06% | 67.08% | 8.41%±3.18% | 3.10% | 0.0055 |
| endocrine | 3 | 12.63%±6.66% | 100.00% | 12.63%±6.66% | 12.43% | 0.0584 |
| immunological | 2 | 1.17%±1.69% | 83.61% | 2.63%±2.63% | 0.70% | 0.0021 |
| musculoskeletal | 1 | 6.47%±8.17% | 100.00% | 6.47%±8.17% | 6.47% | 0.0105 |
| reproductive | 1 | 0.00%±0.00% | 100.00% | 0.00%±0.00% | 0.00% | 0.0034 |

## Per-seed detail

| Seed | n_dis | n_triples | R@30 | H@10 drug | H@30 triple | MRR | AUPRC | AUROC |
|---|---|---|---|---|---|---|---|---|
| 42 | 167 | 6760 | 8.63% | 4.83% | 5.19% | 0.0117 | 0.0311 | 0.5520 |
| 123 | 169 | 7048 | 8.31% | 4.92% | 5.66% | 0.0142 | 0.0360 | 0.5512 |
| 456 | 165 | 6673 | 8.42% | 5.11% | 5.93% | 0.0131 | 0.0348 | 0.5618 |
| 789 | 165 | 6499 | 8.35% | 4.79% | 5.80% | 0.0128 | 0.0298 | 0.5512 |
| 2024 | 162 | 7591 | 7.14% | 3.84% | 4.76% | 0.0114 | 0.0296 | 0.5483 |
