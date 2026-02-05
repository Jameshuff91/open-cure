# Research Loop Progress

## Current Session: h199, h203, h204, h195, h200 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested:**
- h199: Solid vs Hematologic Cancer Gap Analysis - **VALIDATED**
- h203: GT-Density Weighted Confidence Scoring - **VALIDATED**
- h204: Lymphoma Subtype Stratification - **VALIDATED**
- h195: Metabolic Exception Analysis - **VALIDATED**
- h200: Brain Tumor Zero Hit Investigation - **VALIDATED**

---

### h200: Brain Tumor Zero Hit Investigation - VALIDATED

**Objective:** Why does brain tumor have 0% precision (n=30)?

**ROOT CAUSE:** DRKG Drug Coverage Gap

**Key Findings:**
- Actual precision: 1.8% (5/278), not 0%
- **Temozolomide (CHEBI:72564) NOT IN DRKG**
- Brain tumor drug coverage: **0/47 drugs have embeddings**
- Model can only recommend drugs in its pool

**Recommendation:**
- Exclude brain tumors from high-confidence tiers
- OR add manual rules for known drugs (Temozolomide→glioma)

**Output:** `data/analysis/h200_brain_tumor_investigation.json`

---

### h195: Metabolic Exception Analysis - VALIDATED

**KEY FINDING:** CV→Metabolic success is COMORBIDITY MANAGEMENT, not true repurposing.
- Statins treat CV risk IN diabetic patients, not diabetes itself

---

### h204: Lymphoma Subtype Stratification - VALIDATED

**KEY RESULT:** Lymphoma/leukemia have IDENTICAL drug overlap (6%).
- Mechanism-based rules (CD30+, CD20+) are the solution

---

### h203: GT-Density Weighted Confidence Scoring - VALIDATED

**KEY RESULT:** GT density = 31x precision difference (Low 0.28% vs High 8.57%)

---

### h199: Solid vs Hematologic Cancer Gap Analysis - VALIDATED

**ROOT CAUSE:** Disease Fragmentation (not embedding quality)

---

### Cumulative Statistics (2026-02-05)
| Status | Count |
|--------|-------|
| Validated | 100 |
| Invalidated | 43 |
| Inconclusive | 8 |
| Blocked | 18 |
| Deprioritized | 2 |
| Pending | 30 |
| **Total Tested** | **153** |

### Key Session Learnings

1. **h199:** Disease fragmentation causes hematologic low precision
2. **h203:** GT density = strong confidence signal (31x difference)
3. **h204:** Use mechanism-based rules (CD30+/CD20+), not subtype overlap
4. **h195:** CV→Metabolic is comorbidity management, not novel repurposing
5. **h200:** Brain tumor failure is DRKG drug coverage gap, not model failure

### Recommended Next Steps

1. **h202: Subtype-Specific Leukemia Rules** (priority 3)
2. **h205: Lymphoma Mechanism Rules (CD30+/CD20+)** (priority 3)
3. **Integrate GT density** into production confidence scoring
4. **Add manual brain tumor rules** or exclude from high-confidence

---

## Previous Sessions

See previous entries in git history or archived progress.md.
