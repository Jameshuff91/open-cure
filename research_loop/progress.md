# Research Loop Progress

## Current Session: h281, h193, h280, h290 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 4**
- h281: Bidirectional Treatment Analysis - **VALIDATED**
- h193: Combined ATC Coherence Signals - **INVALIDATED**
- h280: Complication vs Subtype Classification - **VALIDATED**
- h290: Implement Relationship Type Filter - **VALIDATED**

### Cumulative Statistics
| Status | Count |
|--------|-------|
| Validated | 165 |
| Invalidated | 54 |
| Inconclusive | 8 |
| Blocked | 18 |
| Deprioritized | 3 |
| Pending | 41 |
| **Total** | **290** (6 new hypotheses added this session)

### KEY SESSION FINDINGS

#### h281: Bidirectional Treatment Analysis - VALIDATED

| Direction | Predictions | Correct | Precision |
|-----------|-------------|---------|-----------|
| Base → Complication | 11 | 1 | **9.1%** |
| Complication → Base | 47 | 17 | **36.2%** |

**Key Finding:** Treating a complication is **4x more predictive** of treating the base disease than vice versa.

#### h193: Combined ATC Coherence Signals - INVALIDATED

| Quadrant | N | Precision |
|----------|---|-----------|
| Coherent + Not Unique | 2,632 | **9.0%** (BEST) |
| Incoherent + Unique | 291 | **4.5%** (WORST) |

**Key Finding:** Opposite of hypothesis! Incoherent+unique is WORST, not best.

#### h280: Complication vs Subtype Classification - VALIDATED ⭐

| Relationship Type | N | Precision |
|-------------------|---|-----------|
| Subtype relationships | 54 | **42.6%** |
| Complication relationships | 36 | **13.9%** |
| Base → Complication | 18 | **0.0%** |

**Major Finding:** Subtype relationships have **28.7 pp higher precision** than complication relationships!

#### h290: Implement Relationship Type Filter - VALIDATED ✅

**Implementation complete:**
- Added BASE_TO_COMPLICATIONS mapping (32 complications)
- Added _is_base_to_complication() filter
- Integrated into production_predictor.py

**Test Results:**
- Diabetic nephropathy: 10 predictions filtered
- Type 2 diabetes: 0 filtered (correct)
- Chronic kidney disease: 0 filtered (correct)

### New Hypotheses Generated
- **h284-h290**: Complication scoring, relationship classification, pathway overlap, ATC coherence tiering, etc.

### Recommended Next Steps
1. Continue testing pending hypotheses
2. Consider h284 (Complication Specialization Score) for follow-up
3. Run full evaluation to measure precision improvement

---

## Previous Session: h279, h277, h282 (2026-02-05)

**Hypotheses Tested: 3**
- h279: Disease Specificity Scoring - **VALIDATED**
- h277: Cross-Category Hierarchy Matching - **INVALIDATED**
- h282: Hierarchy Depth Delta - **VALIDATED**

---

## Previous Session: h273, h276, h278, h271, h275 (2026-02-05)

**Hypotheses Tested: 5** - All VALIDATED
- h273: Disease Hierarchy Matching - 4.5x precision lift
- h276: GOLDEN tier for metabolic/neurological hierarchy
- h278: Infectious hierarchy gap analysis
- h271: Domain-isolated drug filter
- h275: Subtype refinements have 99.2% fuzzy precision

---

## Previous Sessions

See git history for detailed session notes.
