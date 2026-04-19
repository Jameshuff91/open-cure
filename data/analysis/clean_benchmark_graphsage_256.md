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
| MRR (per-triple) | 0.0126 ± 0.0010 |
| AUPRC (per-triple) | 0.0322 ± 0.0026 |
| AUROC (per-triple) | 0.5529 ± 0.0046 |

## Hits@K — per-drug (fraction of GT drugs recovered in top-K, averaged over test diseases)

| K | mean | std |
|---|---|---|
| 1 | 0.74% | 0.15% |
| 5 | 2.97% | 0.27% |
| 10 | 4.70% | 0.44% |
| 30 | 8.17% | 0.53% |
| 100 | 13.68% | 1.12% |

## Hits@K — per-test-triple (fraction of held-out (disease, drug) pairs ranked ≤ K)

| K | mean | std |
|---|---|---|
| 1 | 0.34% | 0.03% |
| 5 | 1.58% | 0.19% |
| 10 | 2.74% | 0.34% |
| 30 | 5.47% | 0.44% |
| 100 | 10.88% | 0.84% |

## Per-category breakdown (averaged over seeds)

| Category | n_dis | R@30 | Hits@30 triple | MRR |
|---|---|---|---|---|
| other | 46 | 7.30%±0.68% | 6.02% | 0.0132 |
| cancer | 22 | 10.91%±2.94% | 5.91% | 0.0136 |
| infectious | 17 | 9.07%±1.15% | 6.56% | 0.0138 |
| cardiovascular | 13 | 6.84%±0.62% | 5.23% | 0.0107 |
| neurological | 12 | 4.61%±1.09% | 2.91% | 0.0066 |
| metabolic | 12 | 6.60%±2.93% | 3.80% | 0.0078 |
| gastrointestinal | 7 | 2.44%±1.10% | 2.93% | 0.0059 |
| dermatological | 6 | 12.32%±7.43% | 14.56% | 0.0433 |
| autoimmune | 6 | 15.09%±5.93% | 15.29% | 0.0420 |
| hematological | 6 | 8.07%±6.24% | 6.15% | 0.0127 |
| respiratory | 5 | 9.31%±7.05% | 6.54% | 0.0207 |
| renal | 4 | 5.99%±2.76% | 3.03% | 0.0081 |
| ophthalmic | 3 | 18.58%±7.46% | 10.87% | 0.0215 |
| psychiatric | 3 | 5.38%±2.06% | 3.10% | 0.0055 |
| endocrine | 3 | 12.63%±6.66% | 12.43% | 0.0584 |
| immunological | 2 | 1.17%±1.69% | 0.70% | 0.0021 |
| musculoskeletal | 1 | 6.47%±8.17% | 6.47% | 0.0105 |
| reproductive | 1 | 0.00%±0.00% | 0.00% | 0.0034 |

## Per-seed detail

| Seed | n_dis | n_triples | R@30 | H@10 drug | H@30 triple | MRR | AUPRC | AUROC |
|---|---|---|---|---|---|---|---|---|
| 42 | 167 | 6760 | 8.63% | 4.83% | 5.19% | 0.0117 | 0.0311 | 0.5520 |
| 123 | 169 | 7048 | 8.31% | 4.92% | 5.66% | 0.0142 | 0.0360 | 0.5512 |
| 456 | 165 | 6673 | 8.42% | 5.11% | 5.93% | 0.0131 | 0.0348 | 0.5618 |
| 789 | 165 | 6499 | 8.35% | 4.79% | 5.80% | 0.0128 | 0.0298 | 0.5512 |
| 2024 | 162 | 7591 | 7.14% | 3.84% | 4.76% | 0.0114 | 0.0296 | 0.5483 |
