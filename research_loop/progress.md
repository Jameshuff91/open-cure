# Research Loop Progress

## Current Session: h279, h277 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** In Progress
**Hypotheses Tested: 2**
- h279: Disease Specificity Scoring - **VALIDATED**
- h277: Cross-Category Hierarchy Matching - **INVALIDATED**

### Cumulative Statistics
| Status | Count |
|--------|-------|
| Validated | 160 |
| Invalidated | 53 |
| Inconclusive | 8 |
| Blocked | 18 |
| Deprioritized | 3 |
| Pending | 39 |
| **Total** | **284** (4 new hypotheses added)

### KEY SESSION FINDINGS

#### h279: Disease Specificity Scoring - VALIDATED

**Hypothesis:** Specificity direction correlates with prediction precision.

| Relationship | Precision | N |
|--------------|-----------|---|
| Exact disease | 78.7% | 277 |
| Generic→Specific | 19.3% | 88 |
| Same-level different | 19.1% | 94 |
| Within-generic different | 19.2% | 214 |
| **Specific→Generic** | **4.1%** | 98 |

**Key Finding:** Specific→Generic predictions have **4.7x LOWER precision** (15.2 pp difference).

**Caveat:** Must distinguish true SUBTYPES from COMPLICATIONS - "diabetic neuropathy" is a complication, not a subtype of diabetes.

#### h277: Cross-Category Hierarchy Matching - INVALIDATED

**Hypothesis:** Drugs treating one category would have precision for cross-category diseases (e.g., diabetes drugs → diabetic nephropathy).

| Category Match Type | Precision | N |
|---------------------|-----------|---|
| Within-category | 10.4% | 10,124 |
| Cross-category match | **0.0%** | 187 |
| Cross-category no match | 0.0% | 3,105 |

**Key Finding:** Cross-category matching has **0% precision** because it's a RECALL problem, not precision.
- 75% of diabetic nephropathy drugs also treat diabetes
- But model predicts single-category diseases (e.g., "chronic kidney disease") instead of cross-category ones (e.g., "diabetic nephropathy")

### New Hypotheses Generated
- **h280**: Complication vs Subtype Classification for Confidence
- **h281**: Bidirectional Treatment Analysis (base disease → complications)
- **h282**: Disease Hierarchy Depth as Confidence Signal
- **h283**: Cross-Category Disease Recall Enhancement

### Recommended Next Steps
1. **h280**: Distinguish complications from true subtypes (medium effort)
2. **h193**: Combined ATC Coherence Signals (medium effort)
3. **h91/h269**: Literature mining or cancer targeting (high effort)

---

## Previous Session: h273, h276, h278, h271, h275 (2026-02-05)

**Hypotheses Tested: 5** - All VALIDATED
- h273: Disease Hierarchy Matching - 4.5x precision lift
- h276: GOLDEN tier for metabolic/neurological hierarchy
- h278: Infectious hierarchy gap analysis
- h271: Domain-isolated drug filter (0% cross-domain precision)
- h275: Subtype refinements have 99.2% fuzzy precision

---

## Previous Sessions

See git history for detailed session notes.
