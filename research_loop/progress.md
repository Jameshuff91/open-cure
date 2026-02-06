# Research Loop Progress

## Current Session: h333, h300, h332, h183 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** In Progress
**Hypotheses Tested: 4**
- h333: Statin Broad Class Re-evaluation - **VALIDATED** (statins are exception - do NOT demote)
- h300: HIV Drug Network Analysis - **VALIDATED** (HIV mechanism-specificity is REAL)
- h332: Cancer-Selective Drug Class Analysis - **VALIDATED + IMPLEMENTED** (mTOR, alkylating added)
- h183: Reproductive Disease Category - **VALIDATED + IMPLEMENTED** (hormone rescue added)

### Cumulative Statistics
| Status | Count |
|--------|-------|
| Validated | 204 |
| Invalidated | 64 |
| Inconclusive | 12 |
| Blocked | 21 |
| Deprioritized | 4 |
| Pending | 30 |
| **Total** | **335**

### KEY SESSION FINDINGS

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

### Recommended Next Steps
1. h272: GT Expansion (medium effort potential win)
2. h178: DiseaseMatcher Algorithm Optimization (performance)
3. h220: Expand MESH Mappings (low effort cleanup)

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
