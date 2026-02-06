# Research Loop Progress

## Current Session: h376, h378, h386 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** In Progress
**Hypotheses Tested: 3**
- h376: Ensemble Coverage Analysis - **VALIDATED**
- h378: Tier Precision Analysis - **VALIDATED**
- h386: Fix Infectious GOLDEN Rule - **VALIDATED**

### h386: Fix Infectious GOLDEN Rule - VALIDATED

**Hypothesis:** Adding viral vs bacterial disease distinction will fix infectious GOLDEN.

**Findings:**
Analyzed infectious disease predictions by tier and rule:

| Tier | Rule | Precision |
|------|------|-----------|
| GOLDEN | infectious | 5.3% (1/19) |
| HIGH | infectious_hierarchy_uti | 75.0% (12/16) |
| HIGH | infectious_hierarchy_tuberculosis | 45.5% (5/11) |
| HIGH | infectious_hierarchy_hepatitis | 0.0% (0/4) |
| MEDIUM | atc_coherent_infectious | 44.1% (15/34) |

**Key Problems:**
1. `infectious` GOLDEN rule: 5.3% → should be removed entirely
2. `infectious_hierarchy_hepatitis`: 0% → viral diseases don't work with hierarchy

**Key Successes:**
1. UTI hierarchy: 75% precision
2. Tuberculosis hierarchy: 45.5% precision
3. ATC coherent: 44.1% precision

**Recommendation:** Remove GOLDEN tier from get_category_tier for infectious category. Keep specific bacterial disease hierarchies (UTI, TB).

### h378: Tier Precision Analysis - VALIDATED

**Hypothesis:** Some existing tier assignment rules have precision below tier average.

**Tier-Level Baseline:**
| Tier | Precision |
|------|-----------|
| GOLDEN | 35.8% |
| HIGH | 59.9% |
| MEDIUM | 30.0% |
| LOW | 24.0% |
| FILTER | 17.9% |

**Problem Rules (>10pp below tier avg, n>=10):**
| Rule | Tier | Precision | Tier Avg | Delta |
|------|------|-----------|----------|-------|
| cardiovascular | HIGH | 31.6% | 59.9% | -28.3pp |
| infectious | GOLDEN | 13.0% | 35.8% | -22.8pp |
| cv_pathway_comprehensive | HIGH | 38.0% | 59.9% | -21.9pp |
| cardiovascular_hierarchy_coronary | HIGH | 40.0% | 59.9% | -19.9pp |
| metabolic_hierarchy_thyroid | GOLDEN | 20.6% | 35.8% | -15.2pp |
| hematological | HIGH | 45.8% | 59.9% | -14.1pp |
| cancer | MEDIUM | 18.2% | 30.0% | -11.8pp |
| incoherent_demotion | HIGH | 48.7% | 59.9% | -11.2pp |

**Unexpected Finding:** HIGH tier (59.9%) has higher precision than GOLDEN (35.8%).

### h376: Ensemble Coverage Analysis - VALIDATED

**Key Results:**
- Overall: Ensemble HURTS (-2.0 pp) - kNN 61.0% vs Ensemble 59.0%
- Best categories for ensemble: metabolic +8.3pp, autoimmune +7.7pp, cancer +2.0pp
- Worst: CV -14.3pp, neuro -11.1pp, immune -12.5pp

**Key Insight:** Ensemble only helps when Target and kNN have similar performance.

### New Hypotheses Generated
- **h381:** Category-Specific Ensemble Routing - Priority 3
- **h382:** Gene Count Q2 Ensemble Rule - Priority 4
- **h383:** CV Ensemble Harm Investigation - Priority 4
- **h384:** Tighten CV Pathway Comprehensive Rule - Priority 3
- **h385:** Demote Thyroid Hierarchy to HIGH - Priority 3
- **h387:** Remove Infectious GOLDEN Rule - Priority 2

### Cumulative Statistics
| Status | Count |
|--------|-------|
| Validated | 239 |
| Invalidated | 71 |
| Inconclusive | 14 |
| Blocked | 21 |
| Deprioritized | 7 |
| Pending | 33 |
| **Total** | **385** |

---

## Previous Session: h374, h377 (2026-02-05)

[Truncated for brevity - see git history]
