# Research Loop Progress

## Current Session: h490 - CV ATC Coherent Full-to-Holdout Gap Investigation (2026-02-06)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 1**
- h490: Cardiovascular ATC Coherent Full-to-Holdout Gap Investigation - **VALIDATED**

### h490: Cardiovascular Full-to-Holdout Gap - VALIDATED

**Objective:** Investigate why cardiovascular ATC coherent predictions have a large full-to-holdout precision gap.

**Key findings:**

1. **CV MEDIUM sub-rule holdout analysis:**
   - cv_pathway_comprehensive: 21.4% ± 12.7% (n=14/seed) — legitimate signal
   - target_overlap_promotion: 16.2% ± 14.9% (n=8/seed) — OK
   - atc_coherent_cardiovascular: 8.4% ± 10.4% (n=5/seed) — below LOW avg
   - standard: 2.0% ± 4.0% (n=17/seed) — essentially random

2. **cv_pathway_comprehensive 44pp gap explained:**
   - NOT overfitting (hardcoded set, not recomputed on holdout)
   - Disease selection variance: only 5/39 CV diseases have high PC drug overlap
   - When CHF/MI/stroke in holdout: high precision. When PAH/cardiac arrest: near 0%

3. **Self-referential diseases discovered:**
   - PAH: 100% self-referential (ALL 26 GT drugs from self in kNN)
   - Angioedema: 90%, Hereditary angioedema: 80%, PVD: 80%

4. **Action taken:** Added 'cardiovascular' to MEDIUM_DEMOTED_CATEGORIES
   - 114 predictions moved MEDIUM→LOW (84 standard + 66 ATC coherent - 36 rescued by target_overlap)
   - MEDIUM holdout: +0.4pp (31.7% → 32.1%)
   - cv_pathway_comprehensive and target_overlap preserved (return before demotion check)

### New Hypotheses Generated (3)
- h504: PAH Self-Referential Filter (Priority 4)
- h505: CV Target Overlap Rescue Precision (Priority 5)
- h506: CV Hierarchy Rules Holdout Validation (Priority 4)

### h504: PAH Self-Referential Filter - VALIDATED

**Objective:** Systematically identify diseases with >80% self-referential GT.

**Key findings:**
1. ALL 455 diseases are their own #1 kNN neighbor
2. 144 (31.6%) are 100% self-referential → FLOOR of kNN performance
3. 27.5pp MEDIUM holdout gap: <50% self-ref (40.8%) vs 100% (13.3%)
4. NOT actionable as demotion (requires GT knowledge, already handled indirectly by low freq)
5. Self-referentiality is the DOMINANT source of full-to-holdout gap

### New Hypotheses Generated (5 total)
- h504-h506 from h490, h507-h508 from h504

### Recommended Next Steps
1. **h506:** CV Hierarchy Rules Holdout Validation - check if CV hierarchy HIGH predictions are genuine
2. **h507:** Predictable Self-Referentiality - can we detect self-ref from embedding features?
3. **h503:** Seed 42 Failure Mode - low effort, could explain variance

---

## Previous Session: h497/h501/h498 - GOLDEN Validation + Determinism Fix + Precision Recalibration (2026-02-06)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 3**
- h497: GOLDEN Standard vs Hierarchy Holdout Validation - **VALIDATED** (no demotion needed)
- h501: kNN Tie-Breaking Determinism Fix - **VALIDATED** (predictions now reproducible)
- h498: Full-Data Precision Recalibration - **VALIDATED** (constants updated to h478 holdout values)

### h497: GOLDEN Standard vs Hierarchy Holdout - VALIDATED

**Objective:** Test whether non-hierarchy GOLDEN predictions ("standard" rule) have significantly lower holdout precision than hierarchy-based GOLDEN predictions.

**Key findings:**
- Standard GOLDEN holdout: 62.2% ± 31.3% (n=26/seed)
- Hierarchy GOLDEN holdout: 70.3% ± 19.1% (n=19/seed)
- Difference: +8.1pp (NOT significant, t=1.01, p>0.35)
- Excluding seed 42 outlier (0/4 standard): standard 77.7% vs hierarchy 78.6% (+0.9pp)
- Full-data gap (standard 55.7% vs hierarchy 89.5%) is mostly hierarchy overfitting

**Decision:** No demotion needed. Standard GOLDEN holdout (62.2%) exceeds HIGH avg (60.8%).

### h501: kNN Tie-Breaking Determinism Fix - VALIDATED

**Objective:** Fix non-reproducible predictions across Python processes.

**Root cause:** `sorted(drug_scores.items(), key=lambda x: x[1], reverse=True)` relied on dict iteration order for tied scores. Python hash randomization causes dict order to vary between processes.

**Fix:** Added drug_id as secondary sort key: `key=lambda x: (-x[1], x[0])`
Applied to 4 sort operations in predict().

**Verification:** 3 independent Python processes produce identical hash (970d4f6e16b8c82dc5dd438466117732). Tier counts: GOLDEN=286, HIGH=507, MEDIUM=3566, LOW=2341, FILTER=6922 — identical across all runs.

### h498: Full-Data Precision Recalibration - VALIDATED

**Objective:** Update all precision constants from stale h402 values to h478 holdout-validated values.

**Changes:**
- DEFAULT_TIER_PRECISION: GOLDEN 53→67, HIGH 51→61, MEDIUM 21→31, LOW 12→15, FILTER 7→10
- get_category_holdout_precision() tier_defaults: same update
- CATEGORY_MEDIUM_HOLDOUT_PRECISION: updated from h462 to h499 corrected values
  - Major changes: dermatological 23→48, musculoskeletal 30→56
  - Demoted categories confirmed: neuro 10→6, GI 11→5
- Module docstring updated

### New Hypotheses Generated (3)
- h501: kNN Tie-Breaking Determinism [COMPLETED]
- h502: GOLDEN Full-Data vs Holdout Gap: Hierarchy Overfitting Quantification (Priority 5)
- h503: Seed 42 Failure Mode: Why Does One Seed Have 0% Standard GOLDEN? (Priority 5)

### Recommended Next Steps
1. **h492:** GT Expansion for psychiatric drug-disease pairs (Priority 4, medium effort)
2. **h486:** Systematic SIDER-based adverse effect mining (Priority 3, high effort)
3. **h257:** IV vs Oral formulation safety distinction (Priority 4, medium effort)

---

## Previous Session: h489/h494/h484/h495 - Safety Audit & Meta-Science (2026-02-06)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 4**
- h489: Mechanism-Required ATC Coherent for Psychiatric/Respiratory - **INVALIDATED**
- h494: Systematic Small-n Holdout Audit - **VALIDATED** (no reversals needed)
- h484: CCB Cardiac Arrest Audit - **VALIDATED** (4 harmful predictions removed)
- h495: Confidence Filter Integration - **VALIDATED** (5 more harmful predictions removed)

### Combined Impact (h484 + h495)

| Tier | Before | After | Δ |
|------|--------|-------|---|
| GOLDEN | 284, 64.1% | 284, 64.1% | unchanged |
| HIGH | 516, 56.6% | 504, 57.7% | -12, **+1.1pp** |
| MEDIUM | 3620, 29.5% | 3613, 29.5% | -7, +0.0pp |
| LOW | 2439, 11.4% | 2426, 11.5% | -13, +0.1pp |
| FILTER | 7288, 11.3% | 7323, 11.3% | +35, 0.0pp |

9 harmful predictions removed from HIGH/MEDIUM tiers.

---

## Previous Session: h493/h499 - Respiratory ATC + Category Re-Validation (2026-02-06)

### Session Summary
- h493: Respiratory ATC coherent literature validation - **VALIDATED** (8 corticosteroid→IPF filtered)
- h499: Category MEDIUM demotions re-validated with corrected GT - **VALIDATED** (all justified)

---

[Earlier sessions: see git history]
