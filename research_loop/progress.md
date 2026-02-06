# Research Loop Progress

## Current Session: h307, h326, h327, h328 (2026-02-05)

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

### Cumulative Statistics
| Status | Count |
|--------|-------|
| Validated | 197 |
| Invalidated | 65 |
| Inconclusive | 12 |
| Blocked | 21 |
| Deprioritized | 4 |
| Pending | 33 |
| **Total** | **332**

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

### New Hypotheses Added
- h327: Statin Isolation Boost Analysis (statins are EXCEPTION - alone is better)

---

## Previous Session: h318, h319 (2026-02-05)

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

#### h319: Comprehensive Low-Precision ATC Filter (Batch 2) - VALIDATED

**Hypothesis:** h314 analysis revealed 26 more ATC→category pairs with <3% precision not yet filtered.

**Implementation:** Added all 22 pairs with 0% precision (skipped >0% to avoid overfitting).

**New pairs (0% precision, all 0 hits):**
- C: neurological, cancer, musculoskeletal
- L: infectious, storage, neurological, endocrine, cardiovascular
- N: cardiovascular, gastrointestinal, metabolic, rare_genetic
- M/P/V: other
- H/R: cancer
- R: autoimmune
- A: renal, rare_genetic
- B: renal
- D: gastrointestinal

**Impact:**
- 22 new ATC→category pairs added
- Additional 703 predictions filtered
- **0 hits lost** (all 0% precision)
- Total pairs now: 43 (was 21 after h318)

**Combined h316+h318+h319 impact:**
- Total predictions filtered: ~2,202
- Total hits lost: ~16 (from h316 borderline cases)
- Overall filtered precision: ~0.7%

### New Hypotheses Added
- h319: Comprehensive Low-Precision ATC Filter (Batch 2) - 26 more candidates from h314
- h320/h321/h322: Class-specific filters - Subsumed by h319
- h323: Cohort Analysis - kNN success is bimodal and category-driven
- h324: Promote Endocrine to Tier 2
- h325: Cancer Tier Promotion Analysis

#### h323: Cohort Analysis of kNN Success Predictors - VALIDATED

**Key Findings:**
1. **Bimodal distribution**: 58.9% diseases have 100% success, 37.7% have 0%
2. **Category is highly predictive**: AUC=0.649
   - Best: dermatological (+1.29), autoimmune (+1.00), endocrine (+0.64)
   - Worst: hematological (-0.83), cardiovascular (-0.58)
3. **Pool size NOT predictive**: correlation only 0.035
4. **Current tier system well-aligned**: Tier 1 categories have positive coefficients

**Recommendations:**
- Consider promoting 'endocrine' to Tier 2 (h324)
- Consider promoting 'cancer' to Tier 1 (h325)
- Existing category precision (h165) already captures this signal

#### h324: Endocrine Tier Promotion - INCONCLUSIVE
- Endocrine: 75% (n=4) - sample too small
- Need more endocrine disease data

#### h325: Cancer Tier Promotion - INVALIDATED
- Cancer: 68.6% (35/51)
- Tier 1 minimum: 93.8%
- Gap: 25pp - too large for promotion
- Lymphoma shows promise (80%) but other subtypes are 62-67%

### Recommended Next Steps
1. h272: GT Expansion (medium effort) - potential quick win
2. High-effort hypotheses: h91 (literature mining), h55 (gene expression)

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
