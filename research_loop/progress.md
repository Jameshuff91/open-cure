# Research Loop Progress

## Current Session: h381, h388 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** In Progress
**Hypotheses Tested: 2**
- h381: Category-Specific Ensemble Routing - **INVALIDATED**
- h388: Target Overlap Tier Promotion - **VALIDATED** (GOLDEN +2.2pp, MEDIUM +1.5pp)

### h381: Category-Specific Ensemble Routing - INVALIDATED

**Hypothesis:** Apply MinRank ensemble only to autoimmune/metabolic/cancer categories.

**Findings:**
- Raw LOO: +1.5pp overall (aggressive routing rescues 11 diseases, hurts 4)
- Production context: -2.9pp (tier rules already capture the same signal)
- Confirms h374: ensemble is redundant in production context

**Key Learning:** Never evaluate rank-changing interventions with raw LOO alone. Production tier rules capture the same signal, and rank changes disrupt tier assignment.

### h388: Target Overlap Tier Promotion - VALIDATED

**Hypothesis:** Use target overlap to PROMOTE tier (adjust confidence) without changing rankings.

**Rules Implemented in production_predictor.py:**
1. HIGH + overlap≥3 → GOLDEN (64.6% promoted precision vs 38.4% baseline)
2. LOW + overlap≥1 → MEDIUM (37.9% promoted precision vs 19.8% baseline)

**Results:**
- R@30: 79.2% → 79.2% (UNCHANGED)
- GOLDEN: 38.4% → 40.6% (+2.2 pp)
- MEDIUM: 19.8% → 21.3% (+1.5 pp)

### h395: Demote Remaining Below-Tier Rules - VALIDATED

**Hypothesis:** Demoting rules performing below their tier average will improve tier precision.

**Demotions Applied:**
1. Metabolic TZD/statin GOLDEN → MEDIUM (6.9% vs 41.7% avg)
2. Cancer cross-type MEDIUM → LOW (0.9% vs 21.2%, n=349)
3. Hematological corticosteroids HIGH → MEDIUM (19.1% vs 47.1%)
4. CV generic rescue HIGH → MEDIUM (26.5% vs 47.1%)
5. Respiratory generic HIGH → MEDIUM (14.3% vs 47.1%)
6. Class-injected HIGH → capped at MEDIUM (0% at HIGH)
7. ATC coherent metabolic/neurological excluded (4.3-10.8% vs 21.2%)

**Results:**
- R@30: 79.4% (UNCHANGED)
- GOLDEN: 41.7% → 42.2% (+0.5 pp)
- HIGH: 47.1% → 55.0% (+7.9 pp)
- MEDIUM: 21.2% → 22.5% (+1.3 pp)

### Recommended Next Steps
1. h396: Resolve GOLDEN vs HIGH tier precision inversion (GOLDEN 42.2% < HIGH 55.0%)
2. h394: Fix training frequency label leakage
3. h391: MEDIUM Tier Overlap Anomaly

---

## Previous Session: h376, h378, h386, h387, h385 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 5**
- h376: Ensemble Coverage Analysis - **VALIDATED**
- h378: Tier Precision Analysis - **VALIDATED**
- h386: Fix Infectious GOLDEN Rule - **VALIDATED**
- h387: Remove Infectious GOLDEN Rule - **VALIDATED** (+4.6pp GOLDEN precision)
- h385: Demote Thyroid Hierarchy to HIGH - **VALIDATED** (+0.3pp GOLDEN)

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

### h387: Remove Infectious GOLDEN Rule - VALIDATED

**Implementation:**
1. Removed infectious GOLDEN/HIGH case from get_category_tier()
2. Removed 'hepatitis' and 'hiv' from DISEASE_HIERARCHY_GROUPS (viral = 0% precision)
3. Kept bacterial groups: UTI, tuberculosis, pneumonia, sepsis, skin_infection, respiratory_infection

**Impact on Tier Precision:**
| Tier | Before | After | Delta |
|------|--------|-------|-------|
| GOLDEN | 35.8% | 40.4% | **+4.6pp** |
| HIGH | 59.9% | 58.1% | -1.8pp |
| MEDIUM | 30.0% | 30.9% | +0.9pp |

**Success:** GOLDEN tier precision improved by +4.6 pp.

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

### h385: Demote Thyroid Hierarchy to HIGH - VALIDATED

**Implementation:**
Added HIERARCHY_DEMOTE_TO_HIGH check before GOLDEN assignment.

**Impact:**
| Tier | Before | After | Delta |
|------|--------|-------|-------|
| GOLDEN | 40.4% | 40.7% | +0.3pp |
| HIGH | 58.1% | 55.2% | -2.9pp |

**Rationale:** Thyroid at 20.6% precision was incorrectly in GOLDEN (40% avg).

### Cumulative Statistics
| Status | Count |
|--------|-------|
| Validated | 241 |
| Invalidated | 71 |
| Inconclusive | 14 |
| Blocked | 21 |
| Deprioritized | 7 |
| Pending | 31 |
| **Total** | **385** |

---

## Previous Session: h374, h377 (2026-02-05)

[Truncated for brevity - see git history]
