# Research Loop Progress

## Current Session: h279, h277, h282 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 3**
- h279: Disease Specificity Scoring - **VALIDATED**
- h277: Cross-Category Hierarchy Matching - **INVALIDATED**
- h282: Hierarchy Depth Delta - **VALIDATED**

### Cumulative Statistics
| Status | Count |
|--------|-------|
| Validated | 161 |
| Invalidated | 53 |
| Inconclusive | 8 |
| Blocked | 18 |
| Deprioritized | 3 |
| Pending | 38 |
| **Total** | **284** (4 new hypotheses added)

### KEY SESSION FINDINGS

#### h279: Disease Specificity Scoring - VALIDATED

| Relationship | Precision | N |
|--------------|-----------|---|
| Exact disease | 78.7% | 277 |
| Generic→Specific | 19.3% | 88 |
| **Specific→Generic** | **4.1%** | 98 |

**Key Finding:** Specific→Generic predictions have **4.7x LOWER precision** (15.2 pp difference).

#### h277: Cross-Category Hierarchy Matching - INVALIDATED

| Category Match Type | Precision | N |
|---------------------|-----------|---|
| Within-category | 10.4% | 10,124 |
| Cross-category match | **0.0%** | 187 |

**Key Finding:** Cross-category is a RECALL problem, not precision. Model predicts single-category diseases instead of cross-category ones.

#### h282: Hierarchy Depth Delta - VALIDATED ⭐

| Delta Magnitude | Precision | N |
|-----------------|-----------|---|
| |delta| = 0 (same level) | **51.6%** | 440 |
| |delta| = 1 (off by 1) | 12.0% | 125 |
| |delta| = 2 (off by 2) | 5.4% | 37 |

**Major Finding:** Same-level predictions have **39.6 pp higher precision** than off-by-1! Clear precision gradient by delta magnitude.

**Actionable:** Can use delta magnitude as confidence signal:
- |delta| = 0 → HIGH tier (51.6% precision)
- |delta| = 1 → MEDIUM tier (12.0% precision)
- |delta| ≥ 2 → LOW tier (5.4% precision)

### New Hypotheses Generated
- **h280**: Complication vs Subtype Classification for Confidence
- **h281**: Bidirectional Treatment Analysis (base disease → complications)
- **h283**: Cross-Category Disease Recall Enhancement

### Recommended Next Steps
1. **Implement h282** in production_predictor.py - use delta magnitude for tiering
2. **h280**: Distinguish complications from true subtypes (medium effort)
3. **h193**: Combined ATC Coherence Signals (medium effort)

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
