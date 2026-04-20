# h1281 — Per-category soft-blend weight (inner-fit, outer-eval)

**Premise.** h1275 locked global w=0.5 on the flat plateau. h1272 showed strong category heterogeneity in per-disease fusion lift. If each category has a different optimal w, a per-category sweep could unlock gains the global sweep misses.

**Design.** Inner 80/20 split on outer train fold → fit `best_w[cat]` (argmax mean Δ per-dis AUPRC vs w=0.5 on inner-fit rows, min 5 rows per category); apply fitted weights on outer holdout using full outer-train basis.

Grid: w ∈ [0.0, 0.25, 0.5, 0.75, 1.0], reference w=0.5, seeds=20.

## Aggregate (mean ± std across 20 seeds)

| Mode | R@30 | per-dis-AUPRC | per-dis-AUROC |
|---|---|---|---|
| `W050` | 21.54%±1.12% | 0.1275±0.0102 | 0.6345±0.0049 |
| `W_PER_CAT` | 21.41%±1.07% | 0.1265±0.0091 | 0.6305±0.0049 |
| `W000` | 21.35%±1.13% | 0.1241±0.0094 | 0.6240±0.0054 |
| `W025` | 21.66%±1.12% | 0.1269±0.0097 | 0.6345±0.0049 |
| `W075` | 21.14%±1.05% | 0.1256±0.0101 | 0.6344±0.0049 |
| `W100` | 20.13%±1.17% | 0.1196±0.0104 | 0.6152±0.0049 |

## Paired-t: W_PER_CAT vs W050 (n seeds, outer holdout)

| Metric | Δ | p |
|---|---|---|
| R@30 | -0.1306pp | 0.0988 |
| per-dis-AUPRC | -0.00102 | 0.159 |
| per-dis-AUROC | -0.00397 | 7.93e-05 |

## Per-category outer-holdout summary (sorted by mean Δ per-dis AUPRC)

| Category | rows | mean Δ AUPRC | std | mean Δ R@30 | modal_w |
|---|---|---|---|---|---|
| `cancer` | 431 | +0.00218 | 0.00776 | +1.491pp | 0.25 |
| `autoimmune` | 120 | +0.00149 | 0.00622 | +0.000pp | 0.50 |
| `renal` | 84 | +0.00140 | 0.00332 | -0.071pp | 0.50 |
| `psychiatric` | 60 | +0.00021 | 0.00090 | +0.016pp | 0.50 |
| `endocrine` | 21 | +0.00000 | 0.00000 | +0.000pp | 0.50 |
| `immunological` | 37 | +0.00000 | 0.00000 | +0.000pp | 0.50 |
| `reproductive` | 10 | +0.00000 | 0.00000 | +0.000pp | 0.50 |
| `cardiovascular` | 252 | -0.00003 | 0.00266 | +0.006pp | 0.75 |
| `neurological` | 211 | -0.00031 | 0.01201 | -0.634pp | 0.50 |
| `metabolic` | 225 | -0.00076 | 0.01003 | -1.423pp | 1.00 |
| `other` | 1342 | -0.00129 | 0.00781 | -0.268pp | 0.00 |
| `musculoskeletal` | 51 | -0.00136 | 0.00560 | -0.108pp | 0.50 |
| `respiratory` | 80 | -0.00170 | 0.00430 | -0.245pp | 0.50 |
| `gastrointestinal` | 156 | -0.00201 | 0.00602 | +0.024pp | 0.25 |
| `ophthalmic` | 92 | -0.00221 | 0.00521 | +0.058pp | 0.50 |
| `dermatological` | 166 | -0.00302 | 0.00510 | -0.044pp | 0.50 |
| `infectious` | 571 | -0.00350 | 0.00915 | -0.315pp | 0.75 |
| `hematological` | 85 | -0.00703 | 0.01248 | -1.685pp | 0.50 |

## Preregistered promotion gate

- Δ per-dis AUPRC ≥ +0.001 AND p<0.05: **FAIL** (Δ=-0.00102, p=0.159)
- No category regresses > -0.005 Δ AUPRC: **FAIL** (worst category Δ=-0.00703)

**Decision:** STAY with global W050
