# Research Loop Progress

## Current Session: h427, h402 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 2**
- h427: Psychiatric Target Overlap → GOLDEN Promotion - **INCONCLUSIVE** (signal real but n too small)
- h402: Simplify Production Predictor - **VALIDATED** (3 demotions, holdout stable)

### h427: Psychiatric Target Overlap → GOLDEN Promotion - INCONCLUSIVE

**Hypothesis:** Psychiatric MEDIUM has 55.3% precision. Promote to HIGH or GOLDEN.

**Key Findings:**
1. Only 10 psychiatric diseases total — too few for reliable holdout validation
2. Holdout: 40.9% ± 20.9% (massive variance from seed 456 having only 1 disease in holdout)
3. Per-rule: atc_coherent_psychiatric 42.0% ± 23.8%, target_overlap_promotion 36.4% ± 22.2%
4. Per-disease: schizophrenia 90%, social_anxiety 78%, hyperactive_children 0%
5. Signal IS real (4/5 seeds >30%) but sample too small to distinguish from MEDIUM baseline (22.4%)

**NOT IMPLEMENTED:** Psychiatric category too small (n=10) for reliable holdout validation.

### h402: Simplify Production Predictor - VALIDATED

**Hypothesis:** Prune to top validated rules only, reducing code by 50%+.

**Key Findings:**
1. Comprehensive audit of 83 rule-tier pairs across 51 unique rule names
2. **Only 1 rule clearly fails holdout** (pneumonia hierarchy: 36.4% full → 6.7% holdout)
3. **21 rules are TOO_SMALL to evaluate** (each covers 1-2 diseases)
4. **18 rules are KEEP** with solid holdout validation
5. **5 rules are MARGINAL** (borderline)
6. Volume: top 3 rules = 82% of predictions, 41 rules handle only 7%
7. **Hypothesis PARTIALLY WRONG:** Can't prune 50% of rules. Code complexity is from many small rules, not bad rules.

**Demotions Implemented:**
1. pneumonia hierarchy: HIGH → MEDIUM (6.7% holdout)
2. diabetes hierarchy: GOLDEN → HIGH (31.5% ± 13.8% holdout, below GOLDEN threshold)
3. cv_pathway_comprehensive: HIGH → MEDIUM (26.0% ± 4.9% holdout, below HIGH threshold)

**Holdout Results:**
| Tier | h396 Baseline | h402 After | Delta |
|------|---------------|------------|-------|
| GOLDEN | 55.4% ± 12.1% | 52.9% ± 6.0% | -2.5pp (lower variance!) |
| HIGH | 48.1% ± 6.1% | 50.6% ± 10.4% | **+2.5pp** |
| MEDIUM | 22.4% ± 3.0% | 21.2% ± 1.9% | -1.2pp |
| LOW | 11.0% ± 1.7% | 12.2% ± 1.9% | +1.2pp |
| FILTER | 8.1% ± 0.9% | 7.0% ± 1.5% | -1.1pp |

Tier ordering MAINTAINED: GOLDEN > HIGH > MEDIUM > LOW > FILTER.

### New Hypotheses Generated
- **h432:** Consolidate Small Hierarchy Rules into Generic Category Groups - Priority 4
- **h433:** Full-Data vs Holdout Precision Degradation Predictors - Priority 4

### Recommended Next Steps
1. **h430:** Narrow Diabetes Hierarchy Match to T2D/DM Only - Priority 4, low effort
2. **h428:** Category-Specific Incoherent Demotion Tiers - Priority 4
3. **h407:** Build Comprehensive Drug/Disease ID Mapping Infrastructure - Priority 2 (high effort but unblocks many)

---

## Previous Session: h417, h421, h423 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 3**
- h417: Rank 21-30 Rule Coverage Gap Analysis - **VALIDATED** (findings useful, no implementation)
- h421: Demote base_to_complication from FILTER to MEDIUM - **INVALIDATED** (15.6% precision, not 28.1%)
- h423: Category-Specific Rank Cutoffs - **INVALIDATED** (full-data +1.6pp, holdout -8.7pp)

### h417: Rank 21-30 Rule Coverage Gap Analysis - VALIDATED

**Hypothesis:** Beyond hierarchy matches, are there other high-precision signals at rank 21-30 that could rescue predictions from FILTER?

**Key Findings:**
1. Overall rank 21-30 precision: 10.3% (481/4690 GT hits)
2. Best non-hierarchy signal: **target_overlap>=3 = 32.7% full-data** (n=346, 113 GT hits)
3. Target overlap>=3 implies mechanism_support (identical results)
4. freq>=10 + mechanism: 27.3% full-data (n=128)
5. Category variation: psychiatric 28%, immunological 22% vs neurological 1.8%

**Holdout Validation (5-seed, 80/20):**

| Rule | Full-Data | Holdout (5-seed) | Delta |
|------|-----------|------------------|-------|
| overlap>=3 | 32.7% | 23.6% ± 5.1% | -9.1pp |
| freq10_mech | 31.4% | 27.6% ± 17.3% | -3.8pp (n≈15) |
| Current MEDIUM | 24.7% | 20.9% ± 1.8% | — |

**Decision:** NOT IMPLEMENTED. While overlap>=3 holdout (23.6%) exceeds MEDIUM avg (20.9%), the 9.1pp full-to-holdout drop and small effect size (2.7pp above MEDIUM, within std) doesn't justify adding another rank>20 exception given h418 precedent.

**Key Insight:** Target overlap is MORE generalizable than hierarchy for rank>20 rescue (23.6% holdout vs hierarchy failure), but still not strong enough to overcome the rank>20 filter. The filter is doing its job.

### h421: Demote base_to_complication from FILTER to MEDIUM - INVALIDATED

**Hypothesis:** h412 found base_to_complication has 28.1% precision. Demote from FILTER to MEDIUM.

**Key Findings:**
1. Actual precision is **15.6% (5/32)**, NOT 28.1% as h412 reported
2. Signal is ENTIRELY from diabetic nephropathy: **5/12 = 41.7%**
3. Other complications have 0% precision: DKA 0/7, retinopathy 0/6, uremia 0/7
4. Nephropathy hits are all statins/fibrates/ARBs (known renoprotective drugs)
5. Blanket demotion to MEDIUM would be wrong since 4/5 diseases have 0%

### h423: Category-Specific Rank Cutoffs - INVALIDATED

**Hypothesis:** Use category-specific rank cutoffs instead of global rank>20 FILTER.

**Full-Data Analysis:**
Category precision at rank 21-30 varies enormously:
- Consistent EXTEND (all 5 seeds): psychiatric 28-30%, autoimmune 16-24%, CV 17-19%
- Consistent TIGHTEN: neurological 1.8%, musculoskeletal 1.3%, reproductive 2%
- Mechanism_support at rank 21-30 in extend cats: 40.9% full-data, 35.9% ± 3.3% holdout

**Implementation attempted:** Skip rank>20 for psych/autoimmune/CV with mechanism_support.

**Full-data precision (WITH h423):**
| Tier | Before | After | Delta |
|------|--------|-------|-------|
| GOLDEN | 53.6% | 55.2% | +1.6pp |
| HIGH | 47.7% | 48.0% | +0.3pp |

**Holdout validation FAILED:**
| Tier | h396 Baseline | h423 Holdout | Delta |
|------|---------------|--------------|-------|
| GOLDEN | 55.4% | 50.9% ± 6.4% | -4.5pp |
| HIGH | 48.1% | 39.4% ± 9.3% | **-8.7pp** |

**Root cause:** Same as h399/h418 — drug features (freq, mechanism) are INFLATED on full data. At rank 21-30, drugs appear high-quality because they're counted across ALL 497 diseases. On holdout (80% diseases), their freq drops below tier thresholds, getting assigned to lower tiers and diluting HIGH/GOLDEN.

**CRITICAL LEARNING:** The rank>20 filter compensates for feature inflation, not just noise. THREE separate attempts (h399, h418, h423) confirm this is a fundamental boundary. Do NOT attempt further rank>20 rescue without solving the underlying feature inflation.

### New Hypotheses Generated
- **h422:** Expand Top-N from 30 to 50 with Target Overlap Rescue - Priority 4
- **h424:** Fix h412 Precision Discrepancy for base_to_complication - Priority 4
- **h425:** Nephropathy-Specific Renoprotective Drug Rescue - Priority 4
- **h426:** Holdout-Aware Feature Computation for Rank>20 Rescue - Priority 4

### Recommended Next Steps
1. **h402:** Simplify Production Predictor - high impact, prune to validated rules only
2. **h390:** Production Tier Rule Coverage Analysis - understand rule interaction patterns
3. **h414:** h170 Boosting Strength Optimization - tune alpha for category-specific boosts

---

## Previous Session: h399, h418, h415, h420, h412 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 5**
- h399: Rule Interaction Audit - **VALIDATED** (findings valid, implementation REVERTED)
- h418: Holdout Validation of h399 - **VALIDATED** (showed hierarchy reorder fails holdout)
- h415: Zero-Precision ATC Mismatch Refinement - **VALIDATED** (removed 9 rules, holdout stable)
- h420: Deliverable File Regeneration - **VALIDATED** (14,150 predictions regenerated)
- h412: LOW vs FILTER Recalibration - **VALIDATED** (found base_to_complication at 28.1%)

### h399: Rule Interaction Audit - VALIDATED

**Hypothesis:** The _assign_confidence_tier method has 15+ return paths. A drug can match multiple rules but only the first fires. Systematically audit all interactions to find cases where rule ordering is suboptimal.

**Key Findings:**
1. **88.2% of predictions match 2+ rules** (12,014/13,622) - interactions are pervasive
2. **rank>20 filter shadows 332 hierarchy-matched predictions** with 60.5% precision (201 GT hits!)
   - hierarchy_rheumatoid_arthritis: 92.1% precision at rank 21-30
   - hierarchy_colitis: 100% precision at rank 21-30
   - hierarchy_hypertension: 81.8% precision at rank 21-30
   - hierarchy_multiple_sclerosis: 100% precision at rank 21-30
3. **CV pathway-comprehensive at rank>20**: 44.6% precision (25/56 GT)
4. **zero_precision_mismatch catches 126 GT hits** - potentially too aggressive (h415 for follow-up)
5. **mechanism_specific cap at LOW is appropriate** (14.5% precision)
6. **cancer_same_type → HIGH**: marginal +0.8pp, deferred (h416)

**Implementation: Moved hierarchy + CV pathway checks BEFORE rank>20 filter**

| Tier | Before | After | Delta |
|------|--------|-------|-------|
| GOLDEN | 61.8% | 63.9% | **+2.1pp** |
| HIGH | 54.6% | 54.8% | +0.2pp |
| MEDIUM | 26.4% | 26.2% | -0.2pp |
| LOW | 12.2% | 12.7% | +0.5pp |
| FILTER | 12.2% | 9.6% | -2.6pp |
| R@30 | 79.1% | 79.1% | unchanged |

**GOLDEN improvement explained:** h388 target overlap promotions now find HIGH predictions (from hierarchy rescue) and promote some to GOLDEN (90% precision on the 30 new GOLDEN predictions).

### New Hypotheses Generated
- **h415:** Zero-Precision ATC Mismatch Refinement (126 GT hits caught) - Priority 3
- **h416:** Cancer Same-Type + HIGH Criteria Promotion - Priority 4
- **h417:** Rank 21-30 Rule Coverage Gap Analysis - Priority 3
- **h418:** Holdout Validation of h399 Changes - Priority 2

### h418: Holdout Validation of h399 Changes - VALIDATED (showed regression)

**Hypothesis:** Verify h399 hierarchy-before-rank reordering holds on 80/20 holdout.

**Results:**
| Tier | Full (h399 code) | Holdout (h399 code) | h396 Holdout Baseline | Delta vs h396 |
|------|-------------------|--------------------|-----------------------|---------------|
| GOLDEN | 57.4% | 52.6% ± 14.0% | 55.4% ± 12.1% | -2.8pp |
| HIGH | 44.2% | 41.9% ± 6.4% | 48.1% ± 6.1% | **-6.2pp** |
| MEDIUM | 24.6% | 22.6% ± 3.3% | 22.4% ± 3.0% | +0.2pp |
| LOW | 13.3% | 10.9% ± 1.6% | 11.0% ± 1.7% | -0.1pp |
| FILTER | 9.0% | 7.4% ± 0.8% | 8.1% ± 0.9% | -0.7pp |

**Decision:** REVERTED h399 implementation. HIGH -6.2pp on holdout is unacceptable.
Hierarchy rules at rank>20 don't generalize. Many are 1-disease groups.

**Key Learning:** Full-data precision can be misleading for rule changes. Always validate with holdout before deploying.

### h415: Zero-Precision ATC Mismatch Refinement - VALIDATED

Removed 9 mismatch rules with >10% GT hit rate + clear medical justification:
1. (R, other): 100% - opioids for pain
2. (M, other): 58.3% - dantrolene for malignant hyperthermia
3. (V, other): 30.0% - antidotes for poisoning
4. (B, renal): 23.1% - EPO for CKD anemia
5. (A, renal): 18.7% - corticosteroids for nephrotic syndrome
6-9. Various with 10-20% precision and clear clinical justification

Holdout validation: HIGH -0.5pp (within noise), GOLDEN -4.0pp (within std=15.2%).
Tier ordering maintained. All tiers within noise of h396 baselines.

### Recommended Next Steps
1. **h420:** Deliverable file regeneration - Priority 2, low effort
2. **h417:** Rank 21-30 coverage gap - Priority 3
3. **h402:** Simplify Production Predictor - Priority 3

---

## Previous Session: h393, h396 (2026-02-05)

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

### h400: Deploy Category-Specific k Values - INVALIDATED

h66's category-specific k values hurt R@30 by -2.4pp. h170 selective boosting already provides
category focus. Only cancer k=30 helps (+4.9pp) but is not statistically significant (h413, p=0.20).

### h411: Target Overlap Promotion Holdout Degradation - VALIDATED

Both promotions are genuine:
- HIGH→GOLDEN (overlap≥3): 78.5% precision (n=93), well above GOLDEN avg
- LOW→MEDIUM (overlap≥1): 35.7% holdout ± 2.7% (13.3pp above MEDIUM avg despite -12.6pp drop)

### h413: Cancer-Only k=30 - INCONCLUSIVE

Cancer k=30: +4.9pp (p=0.20), Overall +0.7pp (p=0.21). Not significant, not deployed.

### h397: Remove Dead Code - VALIDATED
Removed 3 dead constants: ATC_HIGH_PRECISION_DERMATOLOGICAL, CV_PATHWAY_EXCLUDE, TARGET_DOMINANT_CATEGORIES.

### h398: Fix CATEGORY_PRECISION Impossible Values - VALIDATED
Replaced 68 stale h165 estimates with actual measured values. Worst corrections: psychiatric FILTER 90→26%, ophthalmic LOW 68→6%.

### Cumulative Statistics (Session Total)
| Status | Count |
|--------|-------|
| Validated (this session) | 6 (h393, h396, h397, h398, h400→invalidated, h411) |
| Invalidated | 1 (h400) |
| Inconclusive | 1 (h413) |

### Recommended Next Steps
1. h399: Rule Interaction Audit - Priority 2, medium effort
2. h91: Literature Mining - Priority 2, high effort (but high potential)
3. h407: ID Mapping Infrastructure - Priority 2, high effort (unblocks many hypotheses)

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
