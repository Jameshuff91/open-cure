# h1218: Fusion Gain Decomposition — per-disease concat_l2 vs node2vec

- seeds = [42, 123, 456, 789, 2024], k = 20
- Rows (seed × holdout disease): **1,002**
- Pearson(Δ R@30, 1 − Jaccard top-20 train neighbours) = **-0.0774**  p=0.0142
- Pearson(Δ R@30, −Spearman of full train-sim rankings) = **+0.0222**  p=0.483

## Fusion gain by neighbour-agreement quartile

| Jaccard bucket | n | mean Δ R@30 | std |
|---|---|---|---|
| Q1_low | 330 | +0.876pp | 13.505pp |
| Q2 | 190 | +0.598pp | 9.424pp |
| Q3 | 279 | +1.408pp | 9.633pp |
| Q4_high | 203 | +2.631pp | 15.577pp |

## Per-category fusion gain

| Category | n | mean Δ R@30 | std | note |
|---|---|---|---|---|
| musculoskeletal | 18 | +6.201pp | 7.418pp |  |
| cancer | 116 | +4.000pp | 9.888pp |  |
| gastrointestinal | 38 | +3.824pp | 16.965pp |  |
| hematological | 17 | +2.476pp | 8.030pp |  |
| psychiatric | 18 | +2.149pp | 4.984pp |  |
| infectious | 149 | +2.090pp | 13.396pp |  |
| other | 320 | +1.419pp | 15.226pp |  |
| ophthalmic | 22 | +0.815pp | 7.569pp |  |
| neurological | 50 | +0.344pp | 8.900pp |  |
| renal | 23 | +0.034pp | 5.199pp |  |
| reproductive | 1 | +0.000pp | 0.000pp |  |
| autoimmune | 31 | -0.334pp | 5.496pp |  |
| dermatological | 48 | -0.452pp | 6.956pp |  |
| immunological | 7 | -0.568pp | 1.392pp | REGRESS |
| respiratory | 21 | -0.988pp | 5.906pp | REGRESS |
| metabolic | 54 | -1.117pp | 10.456pp | REGRESS |
| cardiovascular | 63 | -1.967pp | 11.038pp | REGRESS |
| endocrine | 6 | -5.357pp | 5.740pp | REGRESS |

## Gain concentration

- Rows gaining: 293 (29.2%)
- Rows losing: 185 (18.5%)
- Rows neutral: 524 (52.3%)
- Net mean Δ R@30 per row: **+1.327pp**
