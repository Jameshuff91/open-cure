# Research Loop Progress

## Current Session: h279 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 1**
- h279: Disease Specificity Scoring - **VALIDATED** (with nuances)

### Cumulative Statistics
| Status | Count |
|--------|-------|
| Validated | 160 |
| Invalidated | 52 |
| Inconclusive | 8 |
| Blocked | 18 |
| Deprioritized | 3 |
| Pending | 39 |
| **Total** | **283** (3 new hypotheses added)

### KEY SESSION FINDINGS

#### h279: Disease Specificity Scoring - VALIDATED

**Hypothesis:** Specificity direction (generic→specific vs specific→generic) correlates with prediction precision.

**Results:**
| Relationship | Precision | N | Description |
|--------------|-----------|---|-------------|
| Exact disease | 78.7% | 277 | Same disease match |
| Generic→Specific | 19.3% | 88 | Subtype refinement |
| Same-level different | 19.1% | 94 | Same specificity, different disease |
| Within-generic different | 19.2% | 214 | Both L1, different disease |
| **Specific→Generic** | **4.1%** | 98 | Over-generalization |

**Key Insight:** Specific→Generic predictions have **4.7x LOWER precision** than Generic→Specific!
- 15.2 pp difference (4.1% vs 19.3%)
- This makes clinical sense: treating a specific subtype (diabetic neuropathy) doesn't mean the drug works for the generic parent (diabetes)

**Important Caveat:**
The original hierarchy definition conflated TRUE SUBTYPES with COMPLICATIONS:
- True subtype: "plaque psoriasis" IS a subtype of "psoriasis"
- Complication: "diabetic neuropathy" is a COMPLICATION of diabetes, not a subtype

When properly restricted to true subtypes only, specific→generic precision increases to ~32%.

**Actionable:** Could implement specificity-based confidence scoring, but requires careful ontological classification of subtype vs complication relationships.

### New Hypotheses Generated
1. **h280**: Complication vs Subtype Classification for Confidence
2. **h281**: Bidirectional Treatment Analysis (base disease → complications)
3. **h282**: Disease Hierarchy Depth as Confidence Signal

### Recommended Next Steps
1. **h280**: Distinguish true subtypes from complications for better confidence scoring (medium effort)
2. **h277**: Cross-category hierarchy matching (medium effort)
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
