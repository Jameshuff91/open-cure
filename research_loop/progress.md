# Research Loop Progress

## Current Session: h329, h331, h330, h315, h335 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 5**
- h329: Drug Class Width Analysis - **VALIDATED** (class width NOT predictive, r=0.33)
- h331: Platinum Agent Isolation Boost - **VALIDATED** (justified but not implemented, 0.21% impact)
- h330: Formulation-Specific Prediction Analysis - **VALIDATED** (combos not a significant error source)
- h315: Category-Specific Coherence Thresholds - **VALIDATED** (gaps from +25pp to -7pp)
- h335: Cancer Coherence Investigation - **VALIDATED** (non-L cancer drugs are legitimate)

### Cumulative Statistics
| Status | Count |
|--------|-------|
| Validated | 199 |
| Invalidated | 64 |
| Inconclusive | 12 |
| Blocked | 21 |
| Deprioritized | 4 |
| Pending | 35 |
| **Total** | **335**

### KEY SESSION FINDINGS

#### h329: Drug Class Width Analysis - VALIDATED

**Hypothesis:** Class width (avg GT diseases per drug) predicts isolation signal quality.

**Result:** Correlation is WEAK (r=0.33, p=0.26 - not significant)

**Key Finding:** STATINS EXCEPTION WAS WRONG
- h307/h327 suggested statins have isolation=good (37.5% vs 29.3%)
- Corrected analysis: statins show cohesion helps (+7.8 pp gap)
- Small n (8 isolated) made original finding unreliable

**Classes where COHESION helps (9/16):**
- local_anesthetics: +19.1 pp (width=93.5)
- corticosteroids: +18.3 pp (width=61.2)
- tnf_inhibitors: +22.1 pp (width=10.2)
- antihistamines: +33.3 pp (width=8.2)
- beta_blockers: +16.0 pp (width=8.7)

**TRUE EXCEPTIONS where isolation helps (2/16):**
- platinum_agents: -10.3 pp (width=7.0)
- anticonvulsants: -6.2 pp (insufficient data, n=2)

**Conclusion:** Class width does NOT predict isolation signal. Most classes benefit from cohesion.

#### h331: Platinum Agent Isolation Boost - VALIDATED (but not implemented)

**Findings:**
- Isolated platinum: 17.1% precision (7/41)
- Non-isolated: 9.6% precision (5/52)
- Gap: +7.5 pp (isolation helps)

**MEDIUM tier analysis:**
- Isolated MEDIUM: 21.4% precision (6/28)
- Current HIGH tier: 21.4% precision
- EXACT MATCH - boost justified!

**DECISION: Do not implement**
- 28 predictions = 0.21% of total
- Complexity of implementation outweighs marginal gain
- Pattern documented for future reference

#### h330: Formulation-Specific Prediction Analysis - VALIDATED

**Key Findings:**
- GT: 12.5% combination products (297/2367 drugs)
- Predictions: 99.9% single-agent (kNN from DRKG)
- Component coverage: 68.8% of combo GT covered by single-agent predictions

**Conclusion:** Combination products NOT a significant error source. No special handling needed.

#### h315: Category-Specific Coherence Thresholds - VALIDATED

**Coherence gaps vary dramatically by category:**

| Gap | Categories |
|-----|------------|
| >+10 pp | ophthalmic +25, psychiatric +21, dermatological +18, cardiovascular +12, hematological +12 |
| +3 to +10 pp | autoimmune +8, respiratory +8, neurological +7, infectious +7 |
| <+3 pp | musculoskeletal +1.5, **cancer +0.8** |
| **<0 pp** | GI -1.9, metabolic -2.4, endocrine -3.5, **renal -7.3** |

**Key Insight:** Cancer has nearly ZERO coherence signal (+0.8 pp). Renal is INVERTED (-7.3 pp).

**Implications:**
- Current uniform demotion suboptimal for cancer/metabolic/renal
- Category-specific rules could improve precision
- Implementation deferred (complexity vs gain)

#### h335: Cancer Coherence Investigation - VALIDATED

**Question:** Why does cancer have nearly zero coherence signal (+0.8 pp)?

**Answer:** Non-L cancer drugs ARE legitimate cancer treatments!

**Non-L cancer hits breakdown (71 total):**
- Corticosteroids (26): dexamethasone/prednisone in treatment protocols (ALL, lymphoma)
- Diagnostic agents (23): FDG-PET for staging
- Novel biologics (14): ATC classification lag
- Other legitimate (9): thyroid hormone for thyroid cancer, allopurinol for tumor lysis

**Conclusion:** Cancer is multi-modal (chemo + steroids + hormones + diagnostics).
Don't demote non-L cancer predictions - both L and non-L are legitimate.

### New Hypotheses Added
- h331: Platinum Isolation Boost (tested this session)
- h332: Cancer-Selective Drug Class Analysis
- h333: Statin Broad Class Re-evaluation
- h334: Renal Disease Incoherence Boost
- h335: Cancer Coherence Investigation (tested this session)

### Recommended Next Steps
1. h334: Renal Incoherence Analysis (understand inverted signal, -7.3 pp gap)
2. h272: GT Expansion (medium effort potential win)
3. h332: Cancer-Selective Drug Class Analysis (follow-up to h329 platinum finding)

---

## Previous Session: h307, h326, h327, h328 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** In Progress
**Hypotheses Tested: 6**
- h307: Lidocaine-Specific Boosting Analysis - **VALIDATED** (discovered broader class cohesion pattern)
- h304: Lidocaine-Specific Pattern Analysis - **VALIDATED** (via h307)
- h326: Broad Class Isolation Demotion Rule - **VALIDATED** (151 demoted, 0% HIGH tier cleaned)
- h327: Statin Isolation Boost Analysis - **INCONCLUSIVE** (n=8 too small)
- h196: Gene-Augmented Disease Similarity - **INVALIDATED** (prior findings h41/h124 definitive)
- h328: Class Cohesion Boost Analysis - **VALIDATED** (cohesion=demotion signal, not boost; IL inhibitors added)

### KEY SESSION FINDINGS

#### h307/h304: Lidocaine Pattern → Broad Class Isolation - VALIDATED

**Original Hypothesis:** Lidocaine dominates unique correct predictions (21/44 hits). Is this pattern reliable?

**Key Discovery:** Lidocaine-specific boosting NOT warranted (13.3% precision is below average).
BUT: A broader pattern was discovered: **CLASS COHESION is a positive signal**.

**Findings:**
- Lidocaine alone (no Bupivacaine): 2.1% precision
- Lidocaine with Bupivacaine: 17.9% precision (+15.8 pp!)

**Generalized to all "broad" drug classes:**
| Class | Alone | With Classmates | Difference |
|-------|-------|-----------------|------------|
| TNF Inhibitors | 3.4% | 27.3% | **+23.8 pp** |
| Local Anesthetics | 1.8% | 15.0% | **+13.2 pp** |
| Corticosteroids | 0.0% | 12.6% | **+12.6 pp** |
| NSAIDs | 2.4% | 7.1% | **+4.7 pp** |
| Statins (EXCEPTION) | 37.5% | 29.3% | **-8.2 pp** |

**Insight:** When a drug from a "broad therapeutic class" is predicted ALONE (no classmates),
it's likely noise. Class cohesion = multiple drugs from same class recommended = positive signal.

#### h326: Broad Class Isolation Demotion - VALIDATED

**Implementation:** Added to production_predictor.py:
1. `BROAD_THERAPEUTIC_CLASSES`: anesthetics, steroids, TNFi, NSAIDs
2. `_is_broad_class_isolated()`: checks if drug has no classmates predicted
3. Post-processing: demotes isolated broad-class drugs HIGH→LOW, MEDIUM→LOW

**Impact:**
- 151 predictions demoted from HIGH/MEDIUM to LOW
- HIGH tier: 36 predictions with 0% precision → ALL correctly demoted
- MEDIUM tier: 115 predictions with 3.48% precision → below baseline
- Combined precision: 2.65%

**Result:** HIGH tier now cleaner - 0% precision predictions removed.

---

## Previous Sessions

See git history for detailed session notes.
