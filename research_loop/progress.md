# Research Loop Progress

## Current Session: h309, h310, h261 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** In Progress
**Hypotheses Tested: 3**
- h309: Refine ATC-Category Coherence Map - **VALIDATED** (already committed)
- h310: Implement Coherence Boost with Refined ATC Map - **VALIDATED**
- h261: Pathway-Weighted PPI Scoring - **INVALIDATED**

### Cumulative Statistics
| Status | Count |
|--------|-------|
| Validated | 180 |
| Invalidated | 59 |
| Inconclusive | 10 |
| Blocked | 21 |
| Deprioritized | 3 |
| Pending | 37 |
| **Total** | **310**

### KEY SESSION FINDINGS

#### h309/h310: ATC Coherence Map and Boost - VALIDATED

**h309 Finding:**
- Original ATC map: coherent GOLDEN 15.0% vs incoherent 28.5% (wrong direction!)
- Refined ATC map: coherent GOLDEN 35.5% vs incoherent 18.7% (correct +16.8pp gap)
- Key fix: Add H (corticosteroids) and A (alimentary) to inflammatory categories

**h310 Implementation:**
- Added `DISEASE_CATEGORY_ATC_MAP` with refined mapping
- Added `_is_atc_coherent()` method
- Integrated coherence boost: LOW→MEDIUM for coherent + rank<=10 + evidence
- All 10 unit tests pass

#### h261: Pathway-Weighted PPI Scoring - INVALIDATED

**Results:**
- Raw PPI R@30: 6.21% ± 0.79%
- Pathway-weighted R@30: 6.46% ± 0.87%
- Improvement: +0.25 pp (below 1pp threshold)

**Root cause:** Only 50% of PPI genes have pathway annotations, and the signal is redundant with PPI connectivity.

### Recommended Next Steps
1. h181 (Drug-Level Cross-Category Transfer) builds on ATC coherence work
2. h272 (GT Expansion: Cancer Drug Non-Cancer Uses) requires manual research
3. Low-effort hypotheses available (h178, h183, h196, etc.)

---

## Previous Session: h297, h298, h293, h286, h299, h162 (2026-02-05)

**Hypotheses Tested: 6**
- h297: Mechanism-Specific Disease Categories - **VALIDATED**
- h298: Implement Mechanism-Specificity Confidence Signal - **VALIDATED**
- h293: Inverse Complication Filter Analysis - **VALIDATED**
- h286: Mechanistic Pathway Overlap - **BLOCKED** (ID format mismatch)
- h299: Alternative Methods for Mechanism-Specific Diseases - **BLOCKED** (ID format mismatch)
- h162: Precision-Coverage Trade-off Quantification - **VALIDATED**

---

## Previous Sessions

See git history for detailed session notes.
