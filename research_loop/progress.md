# Research Loop Progress

## Current Session: h199, h203, h204, h195 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested:**
- h199: Solid vs Hematologic Cancer Gap Analysis - **VALIDATED**
- h203: GT-Density Weighted Confidence Scoring - **VALIDATED**
- h204: Lymphoma Subtype Stratification - **VALIDATED**
- h195: Metabolic Exception Analysis - **VALIDATED**

---

### h195: Metabolic Exception Analysis - VALIDATED

**Objective:** Why does CV→Metabolic work (incoherent > coherent)?

**KEY FINDING:** CV→Metabolic success is COMORBIDITY MANAGEMENT, not true repurposing.

**Evidence:**
- 6 statin+diabetes entries, all FDA approved
- Indication texts: "multiple risk factor intervention for atherosclerotic disease"
- Statins treat CV risk IN diabetic patients, not diabetes itself

**Interpretation:**
- NOT a data quality issue - reflects real clinical practice
- NOT novel repurposing - no new mechanism discovery
- IS comorbidity management - diabetes + dyslipidemia co-occur

**Output:** `data/analysis/h195_cv_metabolic_analysis.json`

---

### h204: Lymphoma Subtype Stratification - VALIDATED

**KEY RESULT:** Lymphoma/leukemia have IDENTICAL drug overlap (6%).

**BUT: Mechanism-based groupings identified:**
- **CD30+**: Adcetris covers 9 diseases (Hodgkin, ALCL, PTCL, CTCL)
- **CD20+**: Rituximab covers 6 diseases (B-cell NHL types)

**Recommendation:** Use mechanism-based rules (h205) not subtype overlap.

**Output:** `data/analysis/h204_lymphoma_stratification.json`

---

### h203: GT-Density Weighted Confidence Scoring - VALIDATED

**KEY RESULT:** GT density is a HIGHLY SIGNIFICANT confidence signal.

| GT Density | Precision | Lift |
|------------|-----------|------|
| Low (1-5)  | 0.28%     | 0.1x |
| High (>20) | 8.57%     | 4.4x |

**Statistics:** r = 0.1814 (p < 1e-175), 31x precision difference

**Output:** `data/analysis/h203_gt_density_confidence.json`

---

### h199: Solid vs Hematologic Cancer Gap Analysis - VALIDATED

**ROOT CAUSE: Disease Fragmentation**
- Leukemia: 49 diseases, 45% with only 1 drug → LOW precision
- Myeloma: 3 diseases, concentrated coverage → HIGH precision (23.3%)

**Output:** `data/analysis/h199_solid_vs_hematologic_gap.json`

---

### Cumulative Statistics (2026-02-05)
| Status | Count |
|--------|-------|
| Validated | 99 |
| Invalidated | 43 |
| Inconclusive | 8 |
| Blocked | 18 |
| Deprioritized | 2 |
| Pending | 31 |
| **Total Tested** | **152** |

### Key Session Learnings

1. **h199:** Disease fragmentation causes hematologic low precision
2. **h203:** GT density = strong confidence signal (31x precision difference)
3. **h204:** Lymphoma/leukemia have identical drug overlap; use mechanism-based rules
4. **h195:** CV→Metabolic is comorbidity management, not true repurposing

### Recommended Next Steps

1. **h202: Subtype-Specific Leukemia Rules** (priority 3)
2. **h205: Lymphoma Mechanism Rules (CD30+/CD20+)** (priority 3)
3. **Integrate GT density** into production confidence scoring

---

## Previous Sessions

See previous entries in git history or archived progress.md.
