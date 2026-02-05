# Research Loop Progress

## Current Session: h281, h193, h280 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** In Progress
**Hypotheses Tested: 3**
- h281: Bidirectional Treatment Analysis - **VALIDATED**
- h193: Combined ATC Coherence Signals - **INVALIDATED**
- h280: Complication vs Subtype Classification - **VALIDATED**

### Cumulative Statistics
| Status | Count |
|--------|-------|
| Validated | 163 |
| Invalidated | 54 |
| Inconclusive | 8 |
| Blocked | 18 |
| Deprioritized | 3 |
| Pending | 43 |
| **Total** | **293** (9 new hypotheses added this session)

### KEY SESSION FINDINGS

#### h281: Bidirectional Treatment Analysis - VALIDATED

| Direction | Predictions | Correct | Precision |
|-----------|-------------|---------|-----------|
| Base → Complication | 11 | 1 | **9.1%** |
| Complication → Base | 47 | 17 | **36.2%** |

**Key Finding:** Treating a complication is **4x more predictive** of treating the base disease than vice versa.
- Diabetes → diabetic nephropathy: 0% precision
- DKA → diabetes: 62.5% precision

#### h193: Combined ATC Coherence Signals - INVALIDATED

| Quadrant | N | Precision |
|----------|---|-----------|
| Coherent + Not Unique | 2,632 | **9.0%** (BEST) |
| Coherent + Unique | 129 | 7.8% |
| Incoherent + Not Unique | 1,373 | 5.0% |
| Incoherent + Unique | 291 | **4.5%** (WORST) |

**Key Finding:** Opposite of hypothesis! Category-incoherent + class-unique does NOT identify true repurposing. Following ATC conventions + classmate support = highest precision.

#### h280: Complication vs Subtype Classification - VALIDATED ⭐

| Relationship Type | N | Precision |
|-------------------|---|-----------|
| Exact match | 756 | 100.0% |
| Subtype relationships | 54 | **42.6%** |
| Complication relationships | 36 | **13.9%** |
| Base → Complication | 18 | **0.0%** |

**Major Finding:** Subtype relationships have **28.7 pp higher precision** than complication relationships!
- Base→complication predictions should be FILTERED (0% precision)
- Subtype predictions should be BOOSTED (42.6% precision)

### New Hypotheses Generated
- **h284**: Complication Specialization Score for Confidence
- **h285**: Disease Relationship Type Classification
- **h286**: Mechanistic Pathway Overlap for Complication Predictions
- **h287**: ATC Coherence as Positive Confidence Tier Signal
- **h288**: ATC Class-Supported Predictions as GOLDEN Tier Candidate
- **h289**: Why Does Class Uniqueness Hurt Precision?
- **h290**: Implement Relationship Type Filter in Production Predictor

### Recommended Next Steps
1. **h290**: Implement relationship type filter (priority 2, low effort)
2. **h284**: Complication transferability scoring (priority 3, medium effort)
3. Continue testing pending hypotheses

---

## Previous Session: h279, h277, h282 (2026-02-05)

**Hypotheses Tested: 3**
- h279: Disease Specificity Scoring - **VALIDATED** (specific→generic = 4.1% precision)
- h277: Cross-Category Hierarchy Matching - **INVALIDATED** (0% cross-category precision)
- h282: Hierarchy Depth Delta - **VALIDATED** (51.6% same-level vs 12% off-by-1)

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
