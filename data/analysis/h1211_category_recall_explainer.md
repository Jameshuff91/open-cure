# h1211 — Per-category R@30 explainer (Node2Vec kNN)

- Eligible diseases (GT ∩ Node2Vec embeddings): **3566**
- Per-category R@30: sourced from `clean_benchmark_node2vec_256.json` (5-seed mean)

## Per-category diagnostics

| Category | n_dis | R@30 | Ceiling | R@30/Ceil | Density (J̄) | Isolation | Iso chance | Iso lift | mean GT | median GT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| endocrine | 5 | 41.07% | 100.00% | 41.07% | 0.0405 | 3.00% | 0.11% | +2.89% | 14.0 | 14 |
| ophthalmic | 21 | 38.33% | 92.18% | 41.58% | 0.0616 | 2.38% | 0.56% | +1.82% | 21.3 | 12 |
| reproductive | 3 | 33.33% | 82.54% | 40.38% | 0.0194 | 0.00% | 0.06% | -0.06% | 25.3 | 10 |
| infectious | 142 | 30.45% | 89.86% | 33.89% | 0.0307 | 28.31% | 3.96% | +24.35% | 26.4 | 12 |
| autoimmune | 35 | 26.66% | 74.13% | 35.96% | 0.0843 | 6.86% | 0.95% | +5.90% | 57.4 | 25 |
| dermatological | 40 | 26.00% | 88.59% | 29.35% | 0.0445 | 4.75% | 1.09% | +3.66% | 27.1 | 16 |
| musculoskeletal | 16 | 21.99% | 84.89% | 25.90% | 0.0389 | 5.00% | 0.42% | +4.58% | 27.8 | 14 |
| respiratory | 28 | 21.64% | 79.75% | 27.14% | 0.0351 | 5.36% | 0.76% | +4.60% | 44.3 | 22 |
| other | 2858 | 18.08% | 97.51% | 18.54% | 0.0036 | 88.92% | 80.14% | +8.78% | 9.1 | 3 |
| renal | 21 | 16.48% | 66.89% | 24.64% | 0.0333 | 2.86% | 0.56% | +2.30% | 79.7 | 42 |
| cancer | 113 | 14.42% | 74.39% | 19.39% | 0.0467 | 16.99% | 3.14% | +13.85% | 70.2 | 23 |
| metabolic | 65 | 14.16% | 89.38% | 15.84% | 0.0118 | 4.62% | 1.80% | +2.82% | 32.8 | 12 |
| gastrointestinal | 45 | 13.81% | 84.15% | 16.41% | 0.0191 | 5.11% | 1.23% | +3.88% | 38.0 | 13 |
| immunological | 11 | 13.55% | 84.41% | 16.05% | 0.0130 | 1.36% | 0.28% | +1.08% | 42.5 | 2 |
| cardiovascular | 62 | 12.94% | 68.55% | 18.88% | 0.0365 | 13.39% | 1.71% | +11.68% | 70.8 | 37 |
| neurological | 55 | 11.35% | 83.59% | 13.58% | 0.0204 | 6.64% | 1.51% | +5.12% | 37.9 | 17 |
| hematological | 29 | 10.27% | 88.50% | 11.60% | 0.0269 | 3.62% | 0.79% | +2.84% | 25.6 | 14 |
| psychiatric | 17 | 10.14% | 60.74% | 16.69% | 0.0770 | 7.94% | 0.45% | +7.49% | 84.4 | 64 |

## Univariate Pearson correlations vs per-category R@30

| Diagnostic | r | n |
|---|---:|---:|
| density_mean_jaccard_vs_r30 | 0.248 | 18 |
| isolation_same_cat_frac_vs_r30 | -0.083 | 18 |
| isolation_lift_vs_r30 | -0.041 | 18 |
| mean_gt_drugs_vs_r30 | -0.513 | 18 |
| median_gt_drugs_vs_r30 | -0.323 | 18 |
| n_diseases_vs_r30 | -0.078 | 18 |
| log_n_diseases_vs_r30 | -0.328 | 18 |
| ceiling_vs_r30 | 0.491 | 18 |
| density_vs_r30_over_ceiling | 0.389 | 18 |
| isolation_vs_r30_over_ceiling | -0.159 | 18 |
| mean_gt_drugs_vs_r30_over_ceiling | -0.311 | 18 |

## OLS (density + isolation → R@30)

| Coefficient | Value |
|---|---:|
| density (mean drug-Jaccard within category) | 1.1104 |
| isolation (same-category fraction of top-20 kNN neighbours) | -0.0024 |
| intercept | 0.1688 |
| R² | 0.062 |

## Notes

- **Density (drug-Jaccard)** measures whether diseases within a category share drugs. High density → kNN should transfer well.
- **Isolation (same-category kNN fraction)** measures whether cosine-nearest neighbours in Node2Vec space land inside the same category. High isolation → the embedding respects category structure. We report the chance baseline (n_category-1)/(n_eligible-1) and the lift above chance.
- **GT completeness proxy** uses mean / median drugs per disease in the category. Very small mean values (e.g. cancer with biomarker-stratified GT) are known failure modes; very large means (e.g. infectious with 10+ antibiotics per disease) inflate R@30 because even a broad transfer hits.

