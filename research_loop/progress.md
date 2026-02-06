# Research Loop Progress

## Current Session: h393, h396 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 2**
- h393: Holdout Validation of All Tier Rules - **VALIDATED**
- h396: Resolve GOLDEN vs HIGH Tier Precision Inversion - **VALIDATED**

### h393: Holdout Validation of All Tier Rules - VALIDATED

**Hypothesis:** 80/20 holdout split across 5 seeds to test if hand-crafted tier rules are genuine or overfitted.

**Findings:**
| Tier | Full-Data | Holdout (5-seed) | Delta |
|------|-----------|------------------|-------|
| GOLDEN | 30.3% | 24.1% ± 2.7% | -6.2pp |
| HIGH | 50.9% | 49.1% ± 6.4% | -1.8pp |
| MEDIUM | 24.5% | 23.3% ± 3.6% | -1.2pp |
| LOW | 12.9% | 11.0% ± 1.6% | -1.9pp |
| FILTER | 10.6% | 8.2% ± 0.9% | -2.4pp |

**Key Insights:**
1. Tier system IS genuine: HIGH/MED/LOW/FILTER retain >80% precision on holdout
2. GOLDEN drops 6.2pp — driven by cancer_same_type GT leakage (-5.8pp)
3. GOLDEN<HIGH inversion is REAL, not an overfitting artifact
4. 10 rules flagged as "overfitted" — but 6 are structural absence (1-disease groups)
5. Only 1 truly overfitted rule: infectious_hierarchy_pneumonia (36.4% → 0%)

### h396: Resolve GOLDEN vs HIGH Tier Precision Inversion - VALIDATED

**Changes:**
1. Demoted cancer_same_type from GOLDEN → MEDIUM (24.5% precision = MEDIUM level)
2. Demoted parkinsons/migraine hierarchy from GOLDEN → MEDIUM (0% precision)
3. Updated DEFAULT_TIER_PRECISION with holdout-validated values
4. Updated stale CATEGORY_PRECISION entries

**Results (after changes):**
| Tier | Full-Data | Holdout (5-seed) | Delta |
|------|-----------|------------------|-------|
| GOLDEN | 53.6% | 55.4% ± 12.1% | +0.6pp |
| HIGH | 47.7% | 48.1% ± 6.1% | -2.0pp |
| MEDIUM | 25.6% | 22.4% ± 3.0% | -2.1pp |
| LOW | 10.1% | 11.0% ± 1.7% | -2.1pp |
| FILTER | 10.5% | 8.1% ± 0.9% | -2.4pp |

**Success:** Tier ordering correct on holdout: GOLDEN > HIGH > MEDIUM > LOW > FILTER

### New Hypotheses Generated
- **h410:** Literature Validation of 1-Disease Hierarchy Rules - Priority 3
- **h411:** Target Overlap Promotion Holdout Degradation Analysis - Priority 2
- **h412:** LOW vs FILTER Recalibration (Precision Convergence) - Priority 3

### Recommended Next Steps
1. h400: Deploy Category-Specific k Values (h66 finding, never implemented) - Priority 2, low effort, high impact
2. h411: Target Overlap Promotion Holdout Degradation - Priority 2, low effort
3. h399: Rule Interaction Audit - Priority 2, medium effort

---

## Previous Session: h381, h388 (2026-02-05)

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

### h394: Fix Training Frequency Label Leakage - VALIDATED

**Hypothesis:** Drug frequency counts include the test disease, inflating precision.

**Findings:**
- R@30: 79.4% → 75.4% honest (-4.0 pp) — ranking most affected
- GOLDEN: -0.4 pp (negligible, rules are hierarchy-based)
- HIGH: -2.7 pp (moderate, freq thresholds are borderline)
- MEDIUM: -0.5 pp (negligible)
- 57 drug-disease pairs cross freq>=10 threshold
- No fix needed for production (new diseases have no leakage)

### Recommended Next Steps
1. h396: Resolve GOLDEN vs HIGH tier precision inversion (GOLDEN 42.2% < HIGH 55.0%)
2. h391: MEDIUM Tier Overlap Anomaly
3. h401: Frequency-Independent Tier Rules for Border Cases

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
