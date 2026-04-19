# Clean Embedding Benchmark — `node2vec_256`

- Hypothesis: **h1199** (tier-free multi-metric kNN baseline)
- k = 20, seeds = [42, 123, 456, 789, 2024]
- eval_gt: `data/reference/expanded_ground_truth.json`
- knn_gt (aggregation): `data/cache/ground_truth_cache.json`
- Eligible diseases (GT + embeddings): **1,011**

## Headline metrics (mean ± std across seeds)

| Metric | Value |
|---|---|
| R@30 per-drug | 19.55% ± 1.18% |
| MRR (per-triple) | 0.0284 ± 0.0027 |
| AUPRC (per-triple) | 0.0569 ± 0.0023 |
| AUROC (per-triple) | 0.5766 ± 0.0067 |

## Hits@K — per-drug (fraction of GT drugs recovered in top-K, averaged over test diseases)

| K | mean | std |
|---|---|---|
| 1 | 2.88% | 0.40% |
| 5 | 8.65% | 1.09% |
| 10 | 12.82% | 0.94% |
| 30 | 19.55% | 1.18% |
| 100 | 24.86% | 1.11% |

## Hits@K — per-test-triple (fraction of held-out (disease, drug) pairs ranked ≤ K)

| K | mean | std |
|---|---|---|
| 1 | 0.98% | 0.10% |
| 5 | 4.03% | 0.45% |
| 10 | 6.75% | 0.64% |
| 30 | 11.91% | 0.98% |
| 100 | 17.33% | 1.65% |

## Per-category breakdown (averaged over seeds)

| Category | n_dis | R@30 | Hits@30 triple | MRR |
|---|---|---|---|---|
| other | 64 | 18.08%±2.35% | 14.62% | 0.0392 |
| infectious | 30 | 30.45%±2.97% | 22.37% | 0.0511 |
| cancer | 23 | 14.42%±2.82% | 6.53% | 0.0138 |
| cardiovascular | 13 | 12.94%±1.04% | 10.23% | 0.0220 |
| metabolic | 11 | 14.16%±4.62% | 12.57% | 0.0324 |
| neurological | 10 | 11.35%±3.74% | 8.80% | 0.0208 |
| dermatological | 10 | 26.00%±3.78% | 20.69% | 0.0518 |
| gastrointestinal | 8 | 13.81%±4.89% | 12.42% | 0.0315 |
| autoimmune | 6 | 26.66%±3.77% | 14.93% | 0.0391 |
| renal | 5 | 16.48%±4.48% | 10.57% | 0.0274 |
| ophthalmic | 4 | 38.33%±10.52% | 34.71% | 0.1115 |
| respiratory | 4 | 21.64%±6.85% | 15.79% | 0.0315 |
| musculoskeletal | 4 | 21.99%±9.85% | 23.39% | 0.0752 |
| psychiatric | 4 | 10.14%±1.98% | 9.62% | 0.0203 |
| hematological | 3 | 10.27%±6.14% | 6.06% | 0.0206 |
| immunological | 2 | 13.55%±21.11% | 14.73% | 0.0293 |
| endocrine | 2 | 41.07%±14.40% | 38.18% | 0.0639 |
| reproductive | 1 | 33.33%±0.00% | 33.33% | 0.0317 |

## Per-seed detail

| Seed | n_dis | n_triples | R@30 | H@10 drug | H@30 triple | MRR | AUPRC | AUROC |
|---|---|---|---|---|---|---|---|---|
| 42 | 202 | 6477 | 18.77% | 12.51% | 12.51% | 0.0302 | 0.0565 | 0.5796 |
| 123 | 200 | 7530 | 21.07% | 12.37% | 11.50% | 0.0272 | 0.0552 | 0.5698 |
| 456 | 200 | 6583 | 18.16% | 11.72% | 12.85% | 0.0297 | 0.0590 | 0.5827 |
| 789 | 200 | 6447 | 20.85% | 14.53% | 12.52% | 0.0312 | 0.0601 | 0.5835 |
| 2024 | 200 | 9309 | 18.88% | 12.97% | 10.16% | 0.0237 | 0.0538 | 0.5673 |
