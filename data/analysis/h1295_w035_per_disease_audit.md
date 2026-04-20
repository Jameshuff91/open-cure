# h1295 — Per-disease W035 vs W050 ΔR@30 audit (20 seeds, SUBSET_D_GLOBAL)

**Aggregate:** mean Δ R@30 = +0.1741pp  std = 4.5705pp  n = 3994  (408+ / 317- / 3269=0)

**Non-zero rows only (Δ≠0, n=725):** mean Δ R@30 = +0.9592pp

## Per-n_gt bucket

| n_gt bucket | n_rows | mean Δ R@30 | nonzero mean Δ | frac+ | frac- |
|---|---|---|---|---|---|
| 1 | 221 | +0.4525pp | +100.0000pp | 0.5% | 0.0% |
| 2-5 | 800 | +0.2188pp | +4.1667pp | 3.1% | 2.1% |
| 6-20 | 1552 | +0.2177pp | +1.5787pp | 8.2% | 5.5% |
| 21-50 | 730 | +0.0907pp | +0.3244pp | 15.9% | 12.1% |
| 51+ | 691 | +0.0237pp | +0.0621pp | 20.0% | 18.2% |

## Per-category (sorted by mean Δ R@30)

| Category | n_rows | mean Δ R@30 | nonzero mean Δ | frac+ | frac- |
|---|---|---|---|---|---|
| `musculoskeletal` | 51 | +1.1483pp | +6.5072pp | 15.7% | 2.0% |
| `cancer` | 431 | +0.7953pp | +2.2115pp | 22.3% | 13.7% |
| `ophthalmic` | 92 | +0.7944pp | +4.0601pp | 12.0% | 7.6% |
| `hematological` | 85 | +0.6609pp | +2.9565pp | 15.3% | 7.1% |
| `gastrointestinal` | 156 | +0.5903pp | +2.3614pp | 14.1% | 10.9% |
| `infectious` | 571 | +0.4554pp | +2.7964pp | 10.0% | 6.3% |
| `autoimmune` | 120 | +0.2316pp | +0.8174pp | 17.5% | 10.8% |
| `cardiovascular` | 252 | +0.1939pp | +0.6603pp | 16.7% | 12.7% |
| `psychiatric` | 60 | +0.1820pp | +0.9930pp | 11.7% | 6.7% |
| `reproductive` | 10 | +0.1587pp | +1.5873pp | 10.0% | 0.0% |
| `renal` | 84 | +0.0408pp | +0.2143pp | 9.5% | 9.5% |
| `metabolic` | 225 | +0.0241pp | +0.1933pp | 8.0% | 4.4% |
| `other` | 1342 | -0.0932pp | -0.8622pp | 5.1% | 5.7% |
| `respiratory` | 80 | -0.2149pp | -1.1463pp | 7.5% | 11.2% |
| `immunological` | 37 | -0.2286pp | -1.6917pp | 2.7% | 10.8% |
| `dermatological` | 166 | -0.2585pp | -1.3841pp | 9.0% | 9.6% |
| `neurological` | 211 | -0.3841pp | -2.7012pp | 6.2% | 8.1% |
| `endocrine` | 21 | -0.5102pp | -5.3571pp | 0.0% | 9.5% |

## Cross-tab (category × n_gt bucket) mean Δ R@30 (pp)

| Category | 1 | 2-5 | 6-20 | 21-50 | 51+ |
|---|---|---|---|---|---|
| `musculoskeletal` | +0.000 | +2.000 | +1.742 | -0.265 | +0.373 |
| `cancer` | +0.000 | +3.153 | +0.199 | +0.597 | +0.187 |
| `ophthalmic` | — | +0.000 | +1.746 | -1.235 | +0.000 |
| `hematological` | +0.000 | +0.784 | +0.874 | +0.393 | +0.509 |
| `gastrointestinal` | +0.000 | +2.031 | +0.789 | -0.524 | -0.077 |
| `infectious` | +0.000 | +1.506 | +0.366 | +0.285 | +0.038 |
| `autoimmune` | +0.000 | +0.000 | +1.287 | -0.725 | +0.027 |
| `cardiovascular` | +0.000 | +0.000 | +0.440 | +0.722 | -0.172 |
| `psychiatric` | — | — | +0.619 | +0.340 | -0.044 |
| `reproductive` | — | +0.000 | +0.000 | — | +0.794 |
| `renal` | — | +0.000 | -0.505 | +0.395 | -0.065 |
| `metabolic` | +0.000 | +0.000 | -0.083 | +0.585 | -0.249 |
| `other` | +0.641 | -0.461 | -0.001 | -0.277 | +0.011 |
| `respiratory` | — | +0.000 | -0.290 | -0.378 | -0.018 |
| `immunological` | +0.000 | +0.000 | -1.923 | — | -0.077 |
| `dermatological` | +0.000 | +0.000 | -0.437 | +0.052 | +0.057 |
| `neurological` | +0.000 | -2.000 | +0.092 | +0.247 | +0.037 |
| `endocrine` | +0.000 | — | +0.000 | -1.339 | — |

## Top 15 best diseases

| Mean Δ R@30 | n_seeds | n_gt_mean | Category | Name |
|---|---|---|---|---|
| +42.857pp | 1 | 7.0 | `infectious` | eye infections |
| +37.500pp | 2 | 8.0 | `other` | recurrent herpes labialis |
| +33.333pp | 3 | 1.0 | `other` | beriberi |
| +25.000pp | 2 | 4.0 | `cancer` | small intestine cancer |
| +19.048pp | 3 | 7.0 | `hematological` | paroxysmal nocturnal hemoglobinuria pnh |
| +13.846pp | 5 | 13.0 | `cancer` | relapsed or refractory diffuse large bcell lymphoma |
| +13.333pp | 3 | 5.0 | `gastrointestinal` | hepatic venoocclusive disease |
| +12.500pp | 4 | 4.0 | `infectious` | fusariosis |
| +12.500pp | 4 | 4.0 | `cancer` | merkel cell carcinoma |
| +12.000pp | 5 | 5.0 | `cancer` | lymphangioma |
| +11.111pp | 3 | 6.0 | `other` | blind loop syndrome |
| +10.714pp | 7 | 4.0 | `infectious` | secondary bacterial infections |
| +10.417pp | 3 | 16.0 | `infectious` | tonsillitis |
| +10.000pp | 1 | 10.0 | `cardiovascular` | coronary spasm |
| +10.000pp | 2 | 5.0 | `musculoskeletal` | osteolytic bone metastases |

## Bottom 15 worst diseases

| Mean Δ R@30 | n_seeds | n_gt_mean | Category | Name |
|---|---|---|---|---|
| -27.273pp | 2 | 11.0 | `other` | primary dysbetalipoproteinemia fredrickson type iii |
| -20.000pp | 5 | 3.0 | `other` | gastrinoma |
| -20.000pp | 5 | 4.0 | `neurological` | renal angiomyolipoma and tuberous sclerosis complex tsc |
| -18.750pp | 2 | 8.0 | `hematological` | pernicious anemia |
| -13.333pp | 3 | 5.0 | `other` | relapsing polychrondritis |
| -12.500pp | 3 | 8.0 | `metabolic` | hyperinsulinemic hypoglycemia |
| -12.500pp | 2 | 4.0 | `other` | tropical sprue |
| -12.245pp | 7 | 7.0 | `other` | dacryoadenitis |
| -11.429pp | 5 | 7.0 | `cancer` | ta papillary tumors |
| -10.256pp | 3 | 26.0 | `other` | flutter |
| -9.524pp | 3 | 7.0 | `other` | heroin addiction |
| -8.333pp | 6 | 12.0 | `cancer` | choriocarcinoma |
| -8.333pp | 4 | 6.0 | `other` | plasmodium vivax |
| -8.333pp | 6 | 6.0 | `infectious` | herpes simplex virus infection |
| -8.333pp | 6 | 2.0 | `other` | spasmodic dysphonia |
