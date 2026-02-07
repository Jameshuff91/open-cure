# h669: CS HIGH Novel Prediction Quality Assessment

## Summary
CS HIGH novel predictions are 97.2% medically acceptable, dramatically outperforming
non-CS (62.3%). CS domination of HIGH novel is a quality feature, not an artifact.

## CS Novel Quality (144 predictions, 46 diseases)
| Verdict | Diseases | Predictions | % |
|---------|----------|-------------|---|
| GENUINE | 27 (58.7%) | 92 (64.3%) | Established/first-line CS treatment |
| PLAUSIBLE | 15 (32.6%) | 47 (32.9%) | Some evidence, not first-line |
| IMPLAUSIBLE | 4 (8.7%) | 4 (2.8%) | No medical basis |

## Non-CS Novel Quality (114 predictions)
| Verdict | Count | % |
|---------|-------|---|
| GENUINE | ~38 | 33.3% |
| PLAUSIBLE | ~33 | 28.9% |
| IMPLAUSIBLE | ~32 | 28.1% |

## Bugs Found & Fixed
1. **Diabetes insipidus comp_to_base bug**: `_is_comp_to_base()` matched "diabetes" substring
   in "diabetes insipidus". 9 wrong HIGH predictions (insulin, glimepiride, etc. → DI).
   Fixed with COMP_TO_BASE_EXCLUSIONS list.

2. **False GT in indicationList.xlsx**: 6 lipid drugs (fenofibrate, gemfibrozil, lovastatin,
   cholestyramine, lomitapide, omega-3 FA) → hypothyroidism. FDA labels mention hypothyroidism
   as a "secondary cause to exclude before starting therapy" — NLP incorrectly extracts as
   indication. Fixed with FALSE_GT_PAIRS exclusion + removal from expanded GT.

## GT Changes
- Removed: 6 false GT entries (lipid drugs → hypothyroidism)
- Added: 12 CS GT gaps (established treatments: EGPA, GVHD, PAN, etc.)
- Net: +6 pairs. Expanded GT: 59,644

## Holdout Impact
| Tier | Before (h668) | After (h669) | Change |
|------|---------------|--------------|--------|
| GOLDEN | 71.6% ± 4.3% | 71.9% ± 4.7% | +0.3pp |
| HIGH | 58.0% ± 7.7% | 61.5% ± 7.2% | **+3.5pp** |
| MEDIUM | 43.3% ± 2.9% | 43.4% ± 2.9% | +0.1pp |
| LOW | 15.3% ± 1.8% | 15.3% ± 1.9% | +0.0pp |
| FILTER | 10.7% ± 1.2% | 10.7% ± 1.2% | +0.0pp |

## Key Insight
Every Cure's indicationList.xlsx has NLP extraction errors where "secondary causes such as
hypothyroidism should be excluded before starting therapy" is misinterpreted as an indication.
This pattern likely affects other metabolic drugs too (h670 follow-up).
