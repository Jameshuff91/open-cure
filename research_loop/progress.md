# Research Loop Progress

## Current Session: h244, h248, h251, h249 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** In Progress
**Hypotheses Tested: 4**
- h244: Pulmonary Hypertension Drug Transfer to Heart Failure - **VALIDATED** (critical safety findings)
- h248: Endothelin Antagonist + Prostacyclin Heart Failure Safety Filter - **VALIDATED** (23.3% FP removed)
- h251: SGLT2 Inhibitor Cross-Category Transfer Analysis - **VALIDATED** (63.9% precision!)
- h249: sGC Stimulator Validation Beyond HF/PAH - **VALIDATED** (27-50% precision)

### Cumulative Statistics (2026-02-05)
| Status | Count |
|--------|-------|
| Validated | 140 |
| Invalidated | 50 |
| Inconclusive | 8 |
| Blocked | 18 |
| Deprioritized | 3 |
| Pending | 34 |
| **Total** | **253** |

### Session Key Learnings

1. **h244: PAH→HF Transfer - CRITICAL SAFETY FINDINGS**
   - sGC stimulators VALIDATED (Vericiguat FDA approved for HF)
   - Endothelin antagonists CONTRAINDICATED (fluid retention)
   - Prostacyclin analogs CONTRAINDICATED (INCREASED MORTALITY in FIRST trial!)
   - 7/10 PAH drug predictions for HF were FALSE POSITIVES

2. **h248: PAH-HF Safety Filter Implemented**
   - Added endothelin antagonist + prostacyclin filters to confidence system
   - 23.3% of heart failure HIGH predictions were HARMFUL
   - 7 predictions removed from HIGH tier

3. **h251: SGLT2 Inhibitors - BEST DRUG CLASS (63.9% precision)**
   - 23/36 predictions validated by clinical trials
   - Validated cross-category: Hypertension, atherosclerosis, MI, CKD, obesity
   - False positive patterns: hypoglycemia (SGLT2i CAUSE it), uremia (too advanced)
   - Model correctly captures pleiotropic cardiovascular/renal/metabolic effects

4. **h249: sGC Stimulators - Mixed Results**
   - 27.3% strict precision, 50% lenient (including mechanistically plausible)
   - CRITICAL: Riociguat + pregnancy = CONTRAINDICATED (teratogenic!)
   - Validated: CKD, diabetic nephropathy, proteinuria (preclinical evidence)
   - Plausible: Peripheral arterial disease, stroke, CAD (need clinical trials)

### Session Theme: Drug Class Safety Validation & Precision Analysis

**Key Safety Filters Added:**
1. Endothelin antagonists + heart failure = EXCLUDED (fluid retention)
2. Prostacyclin analogs + heart failure = EXCLUDED (INCREASED MORTALITY)
3. [Pending h253] sGC stimulators + pregnancy = EXCLUDED (teratogenic)

**Precision by Drug Class:**
| Drug Class | Precision | Notes |
|------------|-----------|-------|
| SGLT2 inhibitors | 63.9% | BEST - pleiotropic effects validated |
| sGC stimulators | 27-50% | HF/PAH validated, many FPs in other areas |
| PAH drugs overall | ~30% | 7/10 predictions for HF were contraindicated |

**New Hypotheses Generated: 6**
- h248: PAH-HF safety filter (COMPLETED)
- h249: sGC validation (COMPLETED)
- h250: Systematic CV drug class safety review
- h251: SGLT2 analysis (COMPLETED)
- h252: SGLT2 false positive filter (hypoglycemia/uremia)
- h253: sGC pregnancy safety filter

**Recommended Next Steps:**
1. Implement h253 (sGC pregnancy filter) - critical safety
2. Continue with h250 (CV safety review) - systematic approach
3. Implement h252 (SGLT2 filter) - improve precision

---

## Archive

See previous entries in git history.
