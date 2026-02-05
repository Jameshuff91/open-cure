# Research Loop Progress

## Current Session: h199, h203 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** In Progress
**Hypotheses Tested:**
- h199: Solid vs Hematologic Cancer Gap Analysis - **VALIDATED**
- h203: GT-Density Weighted Confidence Scoring - **VALIDATED**

---

### h203: GT-Density Weighted Confidence Scoring - VALIDATED

**Objective:** Test if disease GT density predicts precision.

**KEY RESULT:** GT density is a HIGHLY SIGNIFICANT confidence signal.

| GT Density | Precision | Lift vs Baseline |
|------------|-----------|------------------|
| Low (1-5)  | 0.28%     | 0.1x (7x WORSE)  |
| Medium (6-10) | 3.28%  | 1.7x             |
| High (11-20) | 6.53%   | 3.3x             |
| Very High (>20) | 8.57%| 4.4x             |

**Statistics:**
- Correlation: r = 0.1814 (p < 1e-175)
- Low GT vs High GT: **31x precision difference**

**Actionable Recommendation:**
- Add GT density as confidence feature
- Downgrade predictions for Low GT diseases (1-5 drugs)
- Boost predictions for High GT diseases (>10 drugs)

**Output:** `data/analysis/h203_gt_density_confidence.json`, `data/analysis/disease_gt_density.json`

---

### h199: Solid vs Hematologic Cancer Gap Analysis - VALIDATED

**Objective:** Analyze why solid tumors vastly outperform hematologic in precision.

**Precision Gap:**
- Hematologic: 5.4% (n=630)
- Solid: 13.1% (n=420)
- Ratio: 2.4x

**ROOT CAUSE: Disease Fragmentation**
- Leukemia: 49 diseases, 45% with only 1 drug
- Myeloma: 3 diseases, 14 drugs/disease → 23.3% precision

**Output:** `data/analysis/h199_solid_vs_hematologic_gap.json`

---

### Cumulative Statistics (2026-02-05)
| Status | Count |
|--------|-------|
| Validated | 97 |
| Invalidated | 43 |
| Inconclusive | 8 |
| Blocked | 18 |
| Deprioritized | 2 |
| Pending | 32 |
| **Total Tested** | **150** |

### Key Session Learnings

1. **h199:** Disease fragmentation causes hematologic low precision, not embedding quality
2. **h203:** GT density = strong confidence signal (31x precision difference)
3. Both findings point to same solution: favor high-GT diseases in confidence scoring

### Recommended Next Steps

1. **h202: Subtype-Specific Leukemia Rules** (priority 3) - Implement AML/CML/ALL/CLL rules
2. **h204: Lymphoma Subtype Stratification** (priority 4) - Analyze lymphoma patterns
3. **Integrate GT density** into production confidence scoring

---

## Previous Sessions

See previous entries in git history or archived progress.md.
