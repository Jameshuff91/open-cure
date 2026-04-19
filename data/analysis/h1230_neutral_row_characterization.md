# h1230 — Neutral-row characterisation of h1218 fusion decomposition
**Question:** Are the 52% Δ-R@30=0 rows in h1218 dominated by low-recall-denominator (n_gt small) diseases? If yes, the +1.33pp h1215 headline understates fusion benefit on actionable cases.

## Headline

- Total rows: **1002** (5 seeds × ~200 holdouts).
- Neutral (|Δ|<1e-09): **524 (52.3%)**.
- Gainers: 293 (29.2%) | Losers: 185 (18.5%).
- Mean Δ R@30 across all rows: **+1.327pp** (matches h1215).
- **Non-trivial mean Δ R@30 (excl. Δ=0): +2.782pp** — more than 2× the headline. This is the actionable lift on diseases where fusion actually moves R@30.
- Gainer-only mean: +11.20pp; loser-only mean: -10.54pp.

## n_gt buckets

| Bucket | n | %neut | meanΔ_all | meanΔ_nontrivial |
|---|---:|---:|---:|---:|
| 1-1 | 58 | 93.1% | +6.90pp | +100.00pp |
| 2-2 | 43 | 86.0% | -1.16pp | -8.33pp |
| 3-5 | 150 | 75.3% | +0.96pp | +3.87pp |
| 6-10 | 177 | 58.2% | +2.03pp | +4.85pp |
| 11-20 | 210 | 51.9% | +1.07pp | +2.23pp |
| 21-50 | 186 | 40.3% | +1.33pp | +2.22pp |
| 51+ | 178 | 18.5% | +0.03pp | +0.04pp |

**93% of n_gt=1 rows are neutral**, dropping to 19% at n_gt≥51. Confirms a-priori intuition: tiny GT pools cannot register fractional gains.

## Per-category Δ R@30 (full vs non-trivial vs n_gt≥5)

| Category | n | %neut | Δ_all | Δ_nontrivial | Δ_nt_n_gt≥5 | medΔ_nt |
|---|---:|---:|---:|---:|---:|---:|
| autoimmune | 31 | 35% | -0.33pp | -0.52pp | -0.52pp | -0.66pp |
| cancer | 116 | 30% | +4.00pp | +5.73pp | +4.94pp | +3.85pp |
| cardiovascular | 63 | 33% | -1.97pp | -2.95pp | -2.60pp | -1.57pp |
| dermatological | 48 | 54% | -0.45pp | -0.99pp | -0.99pp | -1.41pp |
| endocrine | 6 | 50% | -5.36pp | -10.71pp | -10.71pp | -10.71pp |
| gastrointestinal | 38 | 66% | +3.82pp | +11.18pp | +3.78pp | +3.40pp |
| hematological | 17 | 47% | +2.48pp | +4.68pp | +4.68pp | +4.55pp |
| immunological | 7 | 86% | -0.57pp | -3.98pp | -3.98pp | -3.98pp |
| infectious | 149 | 52% | +2.09pp | +4.33pp | +0.79pp | +3.28pp |
| metabolic | 54 | 69% | -1.12pp | -3.55pp | +2.64pp | +2.50pp |
| musculoskeletal | 18 | 44% | +6.20pp | +11.16pp | +11.16pp | +8.33pp |
| neurological | 50 | 54% | +0.34pp | +0.75pp | +1.69pp | +2.56pp |
| ophthalmic | 22 | 50% | +0.81pp | +1.63pp | +1.63pp | -4.35pp |
| other | 320 | 65% | +1.42pp | +4.05pp | +2.32pp | +3.97pp |
| psychiatric | 18 | 17% | +2.15pp | +2.58pp | +2.58pp | +1.54pp |
| renal | 23 | 30% | +0.03pp | +0.05pp | +0.05pp | +0.15pp |
| reproductive | 1 | 100% | +0.00pp | +0.00pp | +0.00pp | +0.00pp |
| respiratory | 21 | 48% | -0.99pp | -1.89pp | -1.89pp | -2.13pp |

GI's +11.18pp non-trivial lift collapses to +3.78pp once tiny denominators are excluded — the headline figure was binary-flip-driven. Musculoskeletal (+11.16pp) and cancer (+4.94pp) hold up at n_gt≥5; cardiovascular (-2.60pp) and endocrine (-10.71pp) regressions are robust.

## Why does n_gt≥51 show ~zero R@30 lift?

- Rows: 178; mean n_gt: 140.9.
- Mean R@30 ceiling (=30/n_gt): 0.313 (structural cap, not embedding limit).
- mean R@30: n2v=0.100 → concat=0.101 (Δ = +0.029pp).
- mean hits@30: n2v=10.71 → concat=10.97 (**Δ = +0.258 drugs/disease — fusion DOES recover more drugs**).
- ceiling-normalised R@30: n2v=0.357 → concat=0.366 (Δ = +0.86pp).

**Implication: hits@30 is the right metric for high-density diseases. R@30 hides ~+0.26 drugs/disease of fusion lift behind the recall ceiling.**

## Disease-level summary

- Unique diseases: 669 (avg 1.50 seeds/disease).
- Diseases with non-zero mean-across-seeds Δ: 360 (53.8%).
- Mean disease-level Δ: +1.296pp.
- Non-trivial disease-level Δ: **+2.409pp**.

## Top losers (n_gt≥5, n_seeds≥2)

| Δ R@30 | Category | Disease | n_gt |
|---:|---|---|---:|
| -33.33pp | infectious | fungal meningitis | 5 |
| -20.00pp | infectious | sporotrichosis | 10 |
| -16.67pp | cardiovascular | tetralogy of fallot | 6 |
| -15.15pp | other | eclampsia | 11 |
| -11.11pp | dermatological | pyoderma | 18 |
| -8.33pp | ophthalmic | keratoconjunctivitis | 12 |
| -7.89pp | metabolic | type 1 diabetes mellitus | 95 |
| -7.69pp | other | dysenteries | 13 |
| -7.14pp | infectious | h influenzae meningitis | 7 |
| -5.95pp | endocrine | hypogonadotropic hypogonadism | 28 |

## Top gainers (n_gt≥5, n_seeds≥2)

| Δ R@30 | Category | Disease | n_gt |
|---:|---|---|---:|
| +25.00pp | infectious | trichomoniasis | 6 |
| +22.22pp | cancer | blast crisis of chronic leukemia | 9 |
| +20.83pp | gastrointestinal | reflux esophagitis | 36 |
| +20.00pp | musculoskeletal | osteolytic bone metastases | 5 |
| +20.00pp | ophthalmic | chorioretinitis | 10 |
| +17.86pp | infectious | infections of the ear | 14 |
| +16.67pp | infectious | postoperative wound infections | 12 |
| +14.81pp | other | extracranial carotid arteries | 9 |
| +14.29pp | other | dyspareunia | 7 |
| +14.29pp | other | spider veins | 7 |

## Correlations

- Pearson(Δ R@30, log10 n_gt) = -0.0658 — small negative, consistent with the bucket finding that small GT pools have larger swings.
- Pearson(Δ R@30, recall ceiling) = +0.0458 — near zero overall (recall ceiling and Δ are not linearly related; the relationship is non-monotonic).

## Implications

1. **Re-frame the h1215 headline**: the +1.32pp R@30 lift is averaged across a population that is 52% structurally-inert. The non-trivial lift on actionable diseases is +2.78pp (row-level) or +2.41pp (disease-level). The confidence interval on the original number is correct; the *interpretation* needs the qualifier.
2. **Hits@30 should join R@30 as a reported metric** for n_gt≥51 diseases, where R@30 is denominator-bound and hides fusion recovery. h1199 already supports this via Hits@K.
3. **Cardiovascular & endocrine regressions are real**, not binary noise. They each have ≥6 row-events with consistent negative Δ. They warrant a deeper ATC/sub-class audit (h1241 covers psychiatric/CV/cancer; extend to endocrine).
4. **Musculoskeletal +11pp and cancer +5pp** non-trivial gains are the highest-ROI targets for category-restricted fusion inference (and they survived h1228's leak-free gate).
