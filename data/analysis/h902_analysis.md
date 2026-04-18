# h902: h901 MeSH Expansion Impact — Coverage & Precision

**Status:** VALIDATED (coverage) + INVALIDATED (precision) — mapping-expansion loop is at diminishing returns; pivot authorized.

## Method

- Pre-h901 holdout: `data/analysis/h900_holdout_run.txt` (5 seeds, scripts/h393-style validation)
- Post-h901 holdout: `data/analysis/h901_full_eval_output.txt` (5 seeds, same methodology)
- Mapping diff: `git diff 80a068e^ 80a068e -- data/reference/mesh_mappings_from_agents.json`

## Coverage (disease-name mapping → evaluable predictions)

| Metric | Pre-h901 | Post-h901 | Delta |
|---|---|---|---|
| Total MeSH mappings (flattened) | 917 | 1555 | +638 (+70%) |
| Evaluable diseases (GT + embeddings) | 965 | 1034 | +69 (+7.2%) |
| Predictions (full-data) | 21,164 | 24,713 | +3,549 (+16.8%) |

**Conversion rate: 638 new mappings → 69 new evaluable diseases = 10.8%.**
Most new mappings name symptoms/findings/clinical qualifiers that do not exist as MeSH treatable targets with DRKG embeddings + GT drugs.

## Tier Precision (holdout, 5-seed mean ± std)

| Tier | Pre-h901 | Post-h901 | Δ | σ-units |
|---|---|---|---|---|
| GOLDEN | 87.1% ± 2.7% | 86.7% ± 2.4% | -0.4pp | 0.11σ |
| HIGH | 83.4% ± 4.0% | 80.8% ± 2.8% | -2.6pp | 0.53σ |
| MEDIUM | 38.5% ± 3.6% | 39.7% ± 4.7% | +1.2pp | 0.20σ |
| LOW | 11.3% ± 0.5% | 10.8% ± 0.3% | -0.5pp | 0.86σ |
| FILTER | 9.2% ± 0.5% | 7.9% ± 0.7% | -1.3pp | 1.51σ |

All tier precisions are within noise (|Δ| < 1.6σ, most < 1σ). No significant precision gain or loss.

## Tier Volume (predictions per holdout seed)

| Tier | Pre | Post | Δ |
|---|---|---|---|
| GOLDEN | 123 | 118 | **-4.1%** |
| HIGH | 198 | 222 | +12.1% |
| MEDIUM | 203 | 225 | +10.8% |
| LOW | 1,824 | 2,249 | +23.3% |
| FILTER | 1,854 | 2,266 | +22.2% |

**Critical finding: zero growth in GOLDEN-tier volume.** The +69 evaluable diseases generate predictions almost entirely in LOW and FILTER. No new high-confidence discoveries were unlocked.

## New-Mapping Category Breakdown (638 total)

| Category | Count | Share |
|---|---|---|
| Other (symptoms/findings/qualifiers) | 346 | 54% |
| Cancer | 82 | 13% |
| Pain/symptom | 41 | 6% |
| Metabolic | 40 | 6% |
| Rare genetic | 36 | 6% |
| Infectious | 35 | 5% |
| Cardiovascular | 30 | 5% |
| Autoimmune | 22 | 3% |
| Neurological | 6 | 1% |

Samples of "other" (symptoms, not treatable diseases): `agitation`, `anorexia`, `back pain`, `bradycardia`, `ascites`, `heartburn`, `apnea`, `hypovolemia`, `vomiting`, `arthralgia`. These should NOT generate drug-repurposing predictions at all — they are reactions, signs, or acute presentations.

## Recommendation

1. **STOP** further name-mapping expansion. The 10.8% conversion rate and 0 new GOLDEN predictions confirm that the remaining ~1,200 unmapped Every Cure names are not the bottleneck.
2. **PIVOT** to external-data hypotheses (h905 LINCS pilot, h906 DrugBank targets, h907 Ryland blinded review). DRKG embedding coverage is the real constraint for rare diseases.
3. **FILTER** symptom/sign mappings. The 346 "other" entries risk surfacing nonsense predictions (e.g., "drug X → vomiting") in the deliverable. Audit needed (see h908).

## Files

- Coverage diff: this doc
- Holdout evals: `data/analysis/h900_holdout_run.txt`, `data/analysis/h901_full_eval_output.txt`
- Mapping diff: git commit 80a068e
