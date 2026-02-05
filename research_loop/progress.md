# Research Loop Progress

## Current Session: h273, h276, h278, h271, h275 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 5**
- h273: Disease Hierarchy Matching for All Categories - **VALIDATED**
- h276: Extend Hierarchy to GOLDEN Tier - **VALIDATED (then corrected)**
- h278: Infectious Disease Hierarchy Gap Analysis - **VALIDATED**
- h271: Domain-Isolated Drug Detection - **VALIDATED**
- h275: Subtype Refinement vs Novel Discovery - **VALIDATED**

### Cumulative Statistics
| Status | Count |
|--------|-------|
| Validated | 159 |
| Invalidated | 52 |
| Inconclusive | 8 |
| Blocked | 18 |
| Deprioritized | 3 |
| Pending | 42 |
| **Total** | **282** |

### KEY SESSION FINDINGS

#### h273: Disease Hierarchy Matching - VALIDATED
Subtype refinement predictions (e.g., drug treats "psoriasis", predict for "plaque psoriasis") have high precision.

**Full evaluation precision:**
| Category | Hierarchy Precision | Tier |
|----------|---------------------|------|
| Metabolic | 65.2% | GOLDEN |
| Neurological | 63.3% | GOLDEN |
| Autoimmune | 44.7% | HIGH |
| Respiratory | 40.4% | HIGH |
| Cardiovascular | 22.6% | HIGH |
| Infectious | 22.1% | HIGH |

#### h271: Cross-Domain Isolated Drug Filter - VALIDATED
828 domain-isolated drugs identified. Cross-domain predictions have 0% precision.
- 273 predictions filtered (2.1% of total)
- 99.3% filter accuracy

#### h275: Subtype Refinements are Clinically Correct - VALIDATED
**MAJOR FINDING:** Hierarchy-matched predictions have **99.2% FUZZY precision**!

They are NOT novel predictions - they're subtype refinements of known indications.
- "chronic heart failure" when GT has "heart failure" = SAME disease
- "plaque psoriasis" when GT has "psoriasis" = SAME condition

**Prediction Type Distribution (13,412 total):**
- exact_match: 12.0% - Already known
- hierarchy_match: 5.7% - Subtype refinements (99.2% correct)
- category_match: 63.2% - Same category, different disease
- novel_repurposing: 21.4% - TRUE novel predictions

### Production Changes
Modified `src/production_predictor.py`:
1. `DISEASE_HIERARCHY_GROUPS` constant
2. `_build_disease_hierarchy_mapping()` method
3. `_check_disease_hierarchy_match()` method
4. `HIERARCHY_GOLDEN_CATEGORIES` = {metabolic, neurological}
5. `_build_domain_isolation_mapping()` method
6. `_is_cross_domain_isolated()` method
7. Cross-domain isolated FILTER rule

### Recommended Next Steps
1. **Add prediction_type field** to deliverables (h275 follow-up)
2. **h277**: Cross-category hierarchy matching (medium effort)
3. **h91/h269**: Literature mining or cancer targeting (high effort)

---

## Previous Sessions

See git history for detailed session notes.
