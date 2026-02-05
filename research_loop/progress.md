# Research Loop Progress

## Current Session: h199 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypothesis Tested:** h199 - Solid vs Hematologic Cancer Gap Analysis

### h199: Solid vs Hematologic Cancer Gap Analysis - VALIDATED

**Objective:** Analyze why solid tumors (ovarian 27%, colorectal 18%) vastly outperform hematologic (leukemia 5.5%, lymphoma 3.3%) in precision.

**Precision Gap:**
- Hematologic: 5.4% (n=630)
- Solid: 13.1% (n=420)
- Ratio: 2.4x

**ROOT CAUSE IDENTIFIED: Disease Fragmentation**

NOT the cause (ruled out):
- Embedding quality: Hematologic clusters BETTER (0.56 vs 0.52 within-class similarity)
- GT density: Hematologic has MORE entries/disease (4.7 vs 3.6)
- Drug transferability: Leukemia drugs transfer MORE across subtypes (2.06 diseases/drug)

IS the cause:
- **Disease fragmentation**: Hematologic cancers have many specific subtypes with sparse GT
- Leukemia: 49 diseases, **45% with only 1 drug**
- Lymphoma: 35 diseases, 37% with only 1 drug
- Myeloma: 3 diseases, 14 drugs/disease → **23.3% precision (exception)**

**Why This Matters:**
When kNN predicts drugs for leukemia subtypes:
- Hits disease with 1-2 GT drugs → likely MISS
- Myeloma hits disease with 39 GT drugs → likely HIT

**Output:** `data/analysis/h199_solid_vs_hematologic_gap.json`

### New Hypotheses Generated

1. **h202: Subtype-Specific Leukemia Production Rules** (priority 3)
   - Expand h201 approach: AML, CML, ALL, CLL specific rules

2. **h203: GT-Density Weighted Confidence Scoring** (priority 3)
   - Add GT density as confidence feature

3. **h204: Lymphoma Subtype Stratification** (priority 4)
   - Analyze if lymphoma behaves differently than leukemia

### Cumulative Statistics (2026-02-05)
| Status | Count |
|--------|-------|
| Validated | 96 |
| Invalidated | 43 |
| Inconclusive | 8 |
| Blocked | 18 |
| Deprioritized | 2 |
| Pending | 33 |
| **Total Tested** | **149** |

### Key Session Learning

**Disease fragmentation is the structural cause of hematologic cancer low precision.**
- Leukemia/lymphoma: many sparse diseases → predictions miss
- Myeloma: few concentrated diseases → predictions hit
- Solution: subtype-specific rules, not general category rules

### Recommended Next Steps

1. **h202: Subtype-Specific Leukemia Rules** (priority 3) - Implement AML/CML/ALL/CLL rules
2. **h203: GT-Density Confidence** (priority 3) - Weight by GT coverage
3. **h195: Metabolic Exception Analysis** (priority 3, low effort) - Quick win

---

## Previous Sessions

See previous entries in git history or archived progress.md.
