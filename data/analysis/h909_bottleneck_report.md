# h909: Bottleneck Diagnosis — Embedding vs GT Coverage

**Status:** VALIDATED (finding is surprising and strategically decisive)
**Date:** 2026-04-18

## Question
h902 showed only 69 of 638 new MeSH mappings (10.8%) became pipeline-evaluable.
Where did the other 569 go — missing DRKG embeddings (fixable by LINCS/h905) or
missing GT drugs (fixable by DrugBank/h906)?

## Method
2x2 coverage table over {embedding present, GT drugs present} for:
- all 1,534 valid MeSH mappings
- the 638 h901-new mappings in isolation

Data sources: `data/raw/drkg/embed/entities.tsv` (DRKG nodes),
`data/reference/expanded_ground_truth.json` (GT pairs),
`data/reference/mesh_mappings_from_agents.json` (disease-name -> MeSH).

## Findings

### All 1,534 mappings
| | GT yes | GT no |
|-|-|-|
| **embed yes** | 1,463 (95.4%) | 27 (1.8%) |
| **embed no**  | 24 (1.6%)     | 20 (1.3%) |

Only **71 total diseases (4.6%)** are blocked by missing embedding and/or GT.

### h901-new mappings only (638 diseases)
| | GT yes | GT no |
|-|-|-|
| **embed yes** | 626 (98.1%) | 12 (1.9%) |
| **embed no**  | 0            | 0         |

**Every new mapping has a DRKG embedding. 626/638 also have GT drugs.
Yet only 69 (h902) became evaluable.**

## The 569-diseases bottleneck is not external data

- LINCS (h905) would unblock 0 of the 638 new mappings (embedding is never missing).
- DrugBank (h906) would unblock at most 12 of the 638 new mappings.
- **557 mappings have both axes present but still fail the pipeline** — blocked by
  symptom/finding exclusion (h902 reports 54% of new mappings are symptoms),
  name-resolution, holdout sampling (disease must survive the disease split to be
  counted as evaluable), or other filters.

## Strategic recommendation

1. **Neither h905 (LINCS) nor h906 (DrugBank) should be justified as a coverage
   expansion.** The pool of diseases they can unblock is tiny (<60 across all
   1,555 mappings).
2. **Justify h905/h906 only by precision gains on already-evaluable diseases**
   (e.g., h906 / DrugBank targets for the biologic failure class noted in
   research_spec.md: mAbs 27.3% R@30, oncology mAbs 0–17%).
3. **The real coverage bottleneck is pipeline hygiene, not external data.**
   h908 (symptom/finding filter) is the correct first action; a follow-up
   hypothesis should audit why 557 data-complete mappings never reach the
   evaluator.
4. **h907 (Ryland external labels) is the highest-ROI external pivot** because
   it targets precision on the existing evaluable pool and does not depend on
   unblocking new diseases.

## Outputs
- `data/analysis/h909_bottleneck_2x2.json` — raw counts per cell
- `scripts/h909_bottleneck_analysis.py` — reproducible analysis
