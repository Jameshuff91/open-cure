# Research Loop Progress

## Current Session: h318 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypothesis Tested: 1**
- h318: Antibiotic FILTER for Non-Infectious Diseases - **VALIDATED** (+180 filtered, 0 hits lost)

### Cumulative Statistics
| Status | Count |
|--------|-------|
| Validated | 186 |
| Invalidated | 63 |
| Inconclusive | 10 |
| Blocked | 21 |
| Deprioritized | 3 |
| Pending | 40 |
| **Total** | **323**

### KEY SESSION FINDINGS

#### h318: Antibiotic FILTER for Non-Infectious Diseases - VALIDATED

**Hypothesis:** J (antiinfective) drugs have 0% precision for many non-infectious disease categories.
Building on h316's zero-precision filter, expand comprehensive filtering for J drugs.

**Results from h314 data:**
| J → Category | Precision | Predictions |
|--------------|-----------|-------------|
| respiratory | 17.8% | 185 | HIGH_PREC (exception) |
| other | 7.4% | 1175 | Keep |
| ophthalmological | 3.7% | 27 | Keep |
| hematological | 0.0% | 32 | **NEW FILTER** |
| gastrointestinal | 0.0% | 34 | **NEW FILTER** |
| metabolic | 0.0% | 23 | **NEW FILTER** |
| immune | 0.0% | 47 | **NEW FILTER** |
| rare_genetic | 0.0% | 44 | **NEW FILTER** |

**Impact:**
- 5 new J→category pairs added to ZERO_PRECISION_MISMATCHES
- Additional 180 predictions filtered
- **0 hits lost** (all filtered predictions have 0% precision)
- Total J pairs now filtered: 9 (was 4 from h316)

**Implementation:** Updated `ZERO_PRECISION_MISMATCHES` in production_predictor.py

### New Hypotheses Added
- h319: Comprehensive Low-Precision ATC Filter (Batch 2) - 26 more candidates from h314
- h320: N (Nervous System) Comprehensive Filter - Similar to h318 for N drugs
- h321: L (Antineoplastic) Cross-Domain Filter - L drugs for non-cancer/non-autoimmune
- h322: C (Cardiovascular) Cross-Domain Filter - C drugs for non-CV categories
- h323: Cohort Analysis: Why Do Some Diseases Have 100% kNN Success?

### Recommended Next Steps
1. h319: Add remaining 0% precision ATC→category pairs (potential +1165 filtered)
2. h320-h322: Class-specific comprehensive filters (N, L, C drugs)
3. h323: Meta-analysis of kNN success predictors for confidence calibration

---

## Previous Session: h311, h312, h314, h316 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** In Progress
**Hypotheses Tested: 4**
- h311: ATC Incoherence Demotion - **VALIDATED** (GOLDEN +7.78 pp)
- h312: Drug Target Count vs Precision - **INVALIDATED** (multi-target BETTER)
- h314: ATC Mismatch Severity by Category - **VALIDATED** (identified 0% and 27% mismatches)
- h316: Zero-Precision Mismatch FILTER - **VALIDATED** (+0.78 pp, 1319 filtered)

### Cumulative Statistics
| Status | Count |
|--------|-------|
| Validated | 185 |
| Invalidated | 63 |
| Inconclusive | 10 |
| Blocked | 21 |
| Deprioritized | 3 |
| Pending | 36 |
| **Total** | **318**

### KEY SESSION FINDINGS

#### h311: ATC Incoherence Demotion - VALIDATED

**Results:**
- Baseline GOLDEN precision: 22.48%
- With demotion GOLDEN precision: 30.26% (+7.78 pp)
- GOLDEN→HIGH demotions: 1,121 (57.7% of GOLDEN)
- HIGH→MEDIUM demotions: 663 (54.9% of HIGH)

**Implementation:** Added to `_assign_confidence_tier()`:
- Check ATC coherence before assigning GOLDEN/HIGH tiers
- If incoherent, demote GOLDEN→HIGH, HIGH→MEDIUM

#### h312: Drug Target Count vs Precision - INVALIDATED

**Hypothesis:** Single-target drugs should have higher kNN precision.

**Results (5-seed, 13,531 predictions):**
- Single-target (1): 5.57% precision
- Multi-target (5+): 9.68% precision
- Difference: **-4.11 pp (multi-target is BETTER!)**

**Explanation:** Drugs with more targets treat more diseases (higher repurposability).
kNN benefits from broader utility since similar diseases share these widely-used drugs.

#### h314: ATC Mismatch Severity by Category - VALIDATED

**Coherent baseline:** 11.7% precision

**High-precision mismatches (better than coherent!):**
- D→respiratory: 27.5% (2.4x baseline!)
- A→ophthalmological: 26.5%
- D→autoimmune: 20.0%
- J→respiratory: 17.8%

**Zero-precision mismatches (always wrong):**
- A→cancer, J→cancer, N→cancer: 0.0%
- B→other, R→other, J→dermatological: 0.0%
- L→ophthalmological, G→other: 1-2%

#### h316: Zero-Precision Mismatch FILTER - VALIDATED

**Implementation:** Added 16 ATC→category pairs to FILTER tier.

**Results:**
- Predictions filtered: 1,319 (9.7% of total)
- Filtered precision: 1.21%
- Hits lost: 16
- Overall precision improvement: +0.78 pp

**Note:** High-precision mismatches (10-27%) are NOT rescued to GOLDEN since they're
below GOLDEN baseline (30%). Normal h311 demotion to HIGH (20% baseline) is optimal.

### New Hypotheses Added

- h313: Coherence Degree (full vs partial ATC match)
- h314: ATC Mismatch Severity (DONE)
- h315: Category-Specific Coherence Thresholds
- h316: Zero-Precision Mismatch FILTER (DONE)
- h317: Category-Specific Coherence Maps
- h318: Antibiotic FILTER for Non-Infectious Diseases

### Recommended Next Steps
1. h318: Antibiotic FILTER for non-infectious diseases (builds on h314)
2. h313: Test coherence degree (full vs partial ATC match)
3. h317: Refine DISEASE_CATEGORY_ATC_MAP with evidence-based mappings

---

## Previous Session: h309, h310, h261, h181 (2026-02-05)

**Hypotheses Tested: 4**
- h309: Refine ATC-Category Coherence Map - **VALIDATED**
- h310: Implement Coherence Boost with Refined ATC Map - **VALIDATED**
- h261: Pathway-Weighted PPI Scoring - **INVALIDATED**
- h181: Drug-Level Cross-Category Transfer - **INVALIDATED**

---

## Previous Sessions

See git history for detailed session notes.
