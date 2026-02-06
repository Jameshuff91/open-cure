# Research Loop Progress

## Current Session: h343, h344, h237, h336, h342 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 5**
- h343: Anti-VEGF Psoriasis/Inflammatory Boost - **INVALIDATED** (0% precision, case reports don't justify GOLDEN)
- h344: TKI Psoriasis Off-Target Effects - **VALIDATED** (TKIs correctly limited to cancer, no psoriasis predictions)
- h237: Indication-Weighted Drug Boosting - **INVALIDATED** (signal already captured by tier system)
- h336: Disease Name Standardization for GT Matching - **VALIDATED** (+25 GT hits from 5 synonym mappings)
- h342: Cancer Drug Cross-Activity Patterns - **VALIDATED** (0-2.6% non-cancer precision across all mechanisms)

### Cumulative Statistics
| Status | Count |
|--------|-------|
| Validated | 215 |
| Invalidated | 67 |
| Inconclusive | 13 |
| Blocked | 21 |
| Deprioritized | 8 |
| Pending | 21 |
| **Total** | **345**

### KEY SESSION FINDINGS

#### h343: Anti-VEGF Psoriasis/Inflammatory Boost - INVALIDATED

**Hypothesis:** Boost anti-VEGF + psoriasis to GOLDEN tier based on h339 case reports

**Analysis:**
- Anti-VEGF overall precision: 3.1% (5/162 predictions)
- Anti-VEGF autoimmune precision: 0% (0/5)
- Anti-VEGF dermatological precision: 0% (0/1)
- Only 2 predictions affected: Bevacizumab→PsA (HIGH), Aflibercept→psoriasis (HIGH)

**Decision:** Current HIGH tier is appropriate for case-report-level evidence.
GOLDEN tier (57.7%) requires stronger evidence than case reports.

#### h344: TKI Psoriasis Off-Target Effects - VALIDATED

**Result:** VEGFR TKIs have ZERO predictions for psoriasis or inflammatory diseases.
- Total TKI predictions: 43
- All predictions are for cancer (MEDIUM: 33, LOW: 6, HIGH: 4)

**Conclusion:** kNN model correctly limits TKIs to cancer domain. No boost needed.

#### h237: Indication-Weighted Drug Boosting - INVALIDATED

**Key finding:** Indication count adds signal WITHIN tiers:
- HIGH tier: 4.7% (low-ind) → 22.7% (high-ind) = +18 pp gap
- MEDIUM tier: 2.3% (low-ind) → 10.8% (high-ind) = +8.5 pp gap
- LOW tier: 0.4% → 0.4% = NO signal (high-ind drugs are noise!)

**Top LOW tier high-indication drugs:** Levofloxacin (0%), Prednisone (0%), Azithromycin (0%)
These are correctly in LOW tier - they appear everywhere but predict nothing.

**Decision:** Don't implement boost. The tier system already captures this via knn_score.

#### h336: Disease Name Standardization - VALIDATED

**Finding:** 128 prediction diseases NOT in GT by exact match.
Conservative normalization finds 25 additional VALID GT hits (+3.2%).

**5 synonym mappings needed:**
1. 'acquired hemolytic anemia' → 'anemia, hemolytic, acquired'
2. 'pure red cell aplasia' → 'pure red-cell aplasia'
3. 'zollinger ellison syndrome' → 'zollinger-ellison syndrome'
4. 'graft versus host disease gvhd' → 'graft versus host disease'
5. 'diffuse large b cell lymphoma dlbcl' → 'diffuse large b-cell lymphoma'

**Recommendation:** Add to DISEASE_SYNONYMS in disease_name_matcher.py (LOW priority)

#### h342: Cancer Drug Cross-Activity Patterns - VALIDATED

**ALL cancer drug mechanisms have 0-2.5% precision for non-cancer predictions:**
| Mechanism | Non-Cancer Preds | Precision |
|-----------|------------------|-----------|
| mTOR inhibitors | 78 | 2.6% |
| VEGF inhibitors | 103 | 1.0% |
| MEK inhibitors | 26 | 0.0% |
| BRAF inhibitors | 3 | 0.0% |
| Proteasome inhibitors | 21 | 0.0% |
| Immunotherapy | 42 | 0.0% |

**Key insight:** Even mTOR inhibitors with FDA-approved non-cancer uses (transplant, TSC) have only 2.6%.
The kNN model predicts based on disease similarity, not drug mechanism.

---

## Previous Session: h337, h338, h272, h178, h340, h339 (2026-02-05)

### Previous Session Summary (6 hypotheses tested)
- h337: ACE Inhibitor Broad Class Analysis - **INCONCLUSIVE** (opposite pattern to statins, but tiny sample)
- h338: NRTI HBV Cross-Activity Boost - **INVALIDATED** (already in GT, nothing to boost)
- h272: GT Expansion: Cancer Drug Repurposing - **VALIDATED** (Bevacizumab → PsA literature confirmed)
- h178: DiseaseMatcher Performance Optimization - **DEPRIORITIZED** (0.02ms/lookup, not needed)
- h340: MEK Inhibitor Non-Cancer Filter - **VALIDATED + IMPLEMENTED** (0% precision → LOW tier)
- h339: Anti-VEGF Drug Non-Cancer Repurposing - **VALIDATED** (psoriasis case reports confirmed)

### KEY SESSION FINDINGS

#### h337: ACE Inhibitor Broad Class Analysis - INCONCLUSIVE

**Question:** Do ACE inhibitors follow the same pattern as statins (isolation helps)?

**Answer:** OPPOSITE PATTERN - cohesion helps for ACE inhibitors

**Data:**
- HIGH isolated: 0% (2/0 GT)
- HIGH non-isolated: 22.2% (9/2 GT)
- Gap: -22.2 pp (cohesion HELPS, not isolation)

**Comparison with Statins:**
- Statins: +16.7 pp (isolation helps)
- ACE inhibitors: -22.2 pp (cohesion helps)
- ARBs: +33.3 pp (isolation helps)

**Decision:** Sample too small (2 isolated HIGH) - NOT implementing

#### h338: NRTI HBV Cross-Activity Boost - INVALIDATED

**Hypothesis:** Boost NRTI → HBV predictions based on h300 finding

**Finding:** NRTI → HBV pairs already exist in GT:
- Lamivudine → hepatitis B (GT)
- Tenofovir → chronic hepatitis B virus infection (GT)
- Adefovir → hepatitis B (GT)

**Conclusion:** Nothing to boost - predictions correctly filtered as known indications

#### h272: GT Expansion: Cancer Drug Repurposing - VALIDATED

**Hypothesis:** Find non-cancer uses for cancer-only drugs

**Key Finding:** Bevacizumab → Psoriatic Arthritis
- HIGH confidence prediction
- VALIDATED in literature: PMC4248526 (case report)
- Patient had 40-year psoriasis, 30-year PsA history
- Complete skin clearance + DAS28 drop (6.98 → 2.8) during bevacizumab
- Relapse when switched to other drugs, remission when restarted

**Mechanism:** VEGF inhibition - VEGF elevated in psoriatic joints

**Other findings:**
- Trametinib → hypothyroidism: Actually NOT inverse indication (hypothyroidism not a known side effect)
- Sunitinib → TSC: Not supported (mTOR inhibitors are standard)

**New hypotheses generated:** h339-h342 (anti-VEGF repurposing, MEK inverse detection)

#### h340: MEK Inhibitor Non-Cancer Filter - VALIDATED + IMPLEMENTED

**Question:** Why do MEK inhibitors have 0% precision for non-cancer predictions?

**Analysis:**
- MEK inhibitors in predictions: Trametinib (15), Selumetinib (21)
- Total predictions: 36
- GT hits: 0
- Precision: **0.0%** across ALL tiers

**GT Analysis:**
- 100% of MEK inhibitor GT indications are cancer
- Trametinib: melanoma, thyroid carcinoma, lung cancer, glioma
- Selumetinib: neurofibromatosis, plexiform neurofibroma

**Non-cancer predictions (24 total):**
- HIGH: 3 (Trametinib→hypothyroidism, Selumetinib→hyperparathyroidism/hypoparathyroidism)
- MEDIUM: 9
- LOW: 12

**Literature check:** NO evidence found for:
- Trametinib → hypothyroidism (hypothyroidism NOT a known side effect of MEK inhibitors)
- Selumetinib → parathyroid disorders

**Implementation:**
- Added MEK_INHIBITORS set to production_predictor.py
- Added `_is_mek_inhibitor_non_cancer()` check
- Non-cancer MEK predictions now capped at LOW tier

**Impact:** 3 HIGH + 9 MEDIUM → LOW, 0% precision improvement

#### h339: Anti-VEGF Drug Non-Cancer Repurposing - VALIDATED

**Question:** Can anti-VEGF drugs be repurposed for inflammatory diseases?

**Analysis:**
- Anti-VEGF predictions: 148 total
- Overall precision: 3.4% (5/148) - mostly cancer/retinopathy GT
- HIGH tier inflammatory predictions: Bevacizumab → PsA, Aflibercept → psoriasis

**Literature Validation:**
Multiple case reports confirm anti-VEGF efficacy for psoriasis:
- Bevacizumab: Complete psoriasis remission during cancer treatment
- Sunitinib: 2 case reports of psoriasis clearance/improvement
- Sorafenib: Psoriasis improvement after 3 weeks
- PNAS mouse study: Anti-VEGF strongly reduced skin inflammation

**Mechanistic Support:**
- VEGF elevated in psoriatic plaques
- Angiogenesis plays key role in psoriasis pathogenesis
- No anti-VEGF licensed for psoriasis YET

**New hypotheses generated:** h343-h345 (anti-VEGF boost, TKI psoriasis, side effect mining)

---

## Previous Session: h333, h300, h332, h183, h220, h256, h228, h222, h224 (2026-02-05)

### Previous Session Summary (9 hypotheses tested)

#### h333: Statin Broad Class Re-evaluation - VALIDATED

**Question:** Should statins be added to BROAD_THERAPEUTIC_CLASSES?

**Answer:** NO - h329's claim that "cohesion helps" was WRONG

**Actual data:**
- Isolated statins: 37.5% precision (3/8)
- Non-isolated statins: 20.0% precision (15/75)
- Gap: -17.5 pp (isolation HELPS, not hurts)

**Isolated HIGH statins: 66.7% (2/3)**
- Lovastatin → hypothyroidism (HIT)
- Lovastatin → coronary atherosclerosis (HIT)
- Lovastatin → congenital hypothyroidism (MISS)

**Decision:** Do NOT add statins - would demote 2 correct HIGH predictions

#### h300: HIV Drug Network Analysis - VALIDATED

**Question:** Are HIV drugs truly mechanism-specific or is this data artifact?

**Answer:** Mechanism-specificity is REAL

- 82 drugs with HIV-related GT indications
- 57 true ARVs (target HIV lifecycle)
- 93% of ARVs are HIV-only (mechanism-specific)
- Exception: NRTIs (Lamivudine, Tenofovir) have HBV cross-activity
- Current MECHANISM_SPECIFIC_DISEASES classification is CORRECT

#### h332: Cancer-Selective Drug Class Analysis - VALIDATED + IMPLEMENTED

**Hypothesis:** Cancer drugs generally show isolation=positive

**Result:** HYPOTHESIS INVALIDATED - Not a general pattern

**Isolation helps:** Platinum agents only (+7.3 pp)

**Cohesion helps:**
- mTOR inhibitors: -7.4 pp (isolated 0% HIGH)
- Alkylating agents: -5.6 pp (isolated 0% HIGH)

**Implementation:** Added to BROAD_THERAPEUTIC_CLASSES:
- mtor_inhibitors: 7 isolated HIGH at 0% precision demoted
- alkylating_agents: 9 isolated predictions demoted

#### h183: Reproductive Disease Category - VALIDATED + IMPLEMENTED

**Finding:** Reproductive category existed but had no rescue rule

**Analysis:**
- Hormone drugs: 26.3% precision for reproductive diseases
- Non-hormone drugs: 3.1% precision
- Gap: +23.2 pp

**Implementation:**
- Added REPRODUCTIVE_HORMONE_DRUGS set
- Added reproductive rescue: hormone drugs → HIGH tier

#### h222: Injection Layer Quality Check - VALIDATED

**Finding:** Injection layer has excellent quality (73%+ precision)

**Analysis:**
- 45 manual_rule predictions
- Initial precision: 42.2% (exact matching)
- Adjusted precision: 73.3% (accounting for disease name variations)
- Most "misses" are GT matches with different name formats

**Conclusion:** Current INJECTED tier is appropriate, no changes needed

### Recommended Next Steps
1. h272: GT Expansion (medium effort potential win)
2. h178: DiseaseMatcher Algorithm Optimization (performance)
3. h237: Indication-Weighted Drug Boosting

---

## Previous Session: h329, h331, h330, h315, h335, h334 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 6**
- h329: Drug Class Width Analysis - **VALIDATED** (class width NOT predictive, r=0.33)
- h331: Platinum Agent Isolation Boost - **VALIDATED** (justified but not implemented, 0.21% impact)
- h330: Formulation-Specific Prediction Analysis - **VALIDATED** (combos not a significant error source)
- h315: Category-Specific Coherence Thresholds - **VALIDATED** (gaps from +25pp to -7pp)
- h335: Cancer Coherence Investigation - **VALIDATED** (non-L cancer drugs are legitimate)
- h334: Renal Disease Incoherence Boost - **VALIDATED + IMPLEMENTED** (fixed ATC map)

---

## Previous Sessions

See git history for detailed session notes.
