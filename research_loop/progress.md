# Research Loop Progress

## Current Session: h442-h449 (2026-02-06)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 9**
- h442: kNN Score Margin as Confidence Signal - **INVALIDATED** (margin vs R@30 = -0.04, no signal)
- h444: Sub-Tier Precision Reporting - **VALIDATED** (MEDIUM rank 1-5 = 21.7% holdout, monotonic)
- h445: TransE Score Distribution Paradox - **VALIDATED** (TransE is subset selector, not within-tier ranker)
- h391: MEDIUM Tier Overlap Anomaly - **VALIDATED** (anomaly resolved, overlap now +10.6pp)
- h416: Cancer Same-Type → HIGH Promotion - **INVALIDATED** (61.6% full → 16.1% holdout, 45.5pp gap)
- h446: Add Rank-Bucket Precision to Deliverable - **VALIDATED** (implemented + regenerated)
- h432: Consolidate Small Hierarchy Rules - **INVALIDATED** (small groups have 87% precision, do NOT consolidate)
- h449: Corticosteroid MEDIUM Demotion - **INVALIDATED** (20.8% ≈ MEDIUM holdout 21.2%)
- h448: HIGH Tier Rank Gradient Anomaly - **VALIDATED** (flat gradient due to hierarchy rescue at rank 16-20)
- h425: Nephropathy Drug Rescue - **INCONCLUSIVE** (n too small)
- h383: CV Ensemble Harm - **DEPRIORITIZED** (ensemble not used)

### h442: kNN Score Margin - INVALIDATED

Score margin (gap between rank-20 and rank-21 drug scores) is not a useful signal.
- Margin vs R@30 correlation: -0.04
- Score space is too flat (median margin = 0.0)
- n_drugs has strongest correlation with R@30 (-0.37) but confounded by GT size

### h444: Sub-Tier Precision Reporting - VALIDATED

**MEDIUM tier has a strong, monotonic rank gradient that holds on holdout:**

| Tier | Rank 1-5 | 6-10 | 11-15 | 16-20 | Gap | Monotonic? |
|------|----------|------|-------|-------|-----|------------|
| GOLDEN | 73.0% ± 23 | 52.1% ± 30 | 55.7% ± 27 | 44.8% ± 35 | +28pp | NO (noisy) |
| HIGH | 43.0% ± 12 | 39.6% ± 11 | 20.4% ± 10 | 21.0% ± 15 | +22pp | NO (flat 16-20) |
| **MEDIUM** | **21.7% ± 6** | **18.1% ± 4** | **10.7% ± 1** | **8.1% ± 4** | **+14pp** | **YES** |
| LOW | 7.9% ± 4 | 5.6% ± 2 | 5.8% ± 2 | 5.4% ± 2 | +3pp | NO (flat) |
| FILTER | 6.1% ± 6 | 6.2% ± 3 | 3.5% ± 1 | 2.9% ± 2 | +3pp | NO |

**Clinical recommendation:** Add rank_bucket_precision to deliverable for MEDIUM/HIGH tiers.
MEDIUM rank 1-5 (~22% holdout) is clinically useful; rank 11-20 (~9%) is not.

### h445: TransE Paradox - VALIDATED

**Resolved why TransE is positive across tiers (h405: +13.6pp) but negative within tiers (h443: -3.7pp):**

1. TransE=True has LOWER precision than TransE=False within every tier below GOLDEN
   - MEDIUM rank 1-5: TransE 33.3% vs No-TransE 37.3% (-4.0pp)
   - LOW rank 1-5: 12.1% vs 25.5% (-13.4pp)

2. Per-disease, TransE is net negative (more diseases hurt than helped):
   - MEDIUM: 24% better, 35% worse

3. NOT a Simpson's paradox - global and per-disease precision nearly identical

**Resolution:** TransE value is at the TIER level (subset selection), not within-tier (ranking).
h439's implementation as a boolean flag was correct. Do NOT use TransE for within-tier ranking.

### h391: MEDIUM Overlap Anomaly - VALIDATED (Anomaly Resolved)

The h388 anomaly (overlap hurts MEDIUM) no longer exists with current tier rules:
- Current: MEDIUM + overlap = 32.3% vs no-overlap = 21.7% (+10.6pp)
- h388 reported: 14.8% vs 20.7% (-5.9pp)

Root cause: tier composition changed due to cancer_same_type demotion and target_overlap_promotion rule.

Key finding: Broad-target drugs (>36 targets) have 26.8% vs narrow (≤36) at 37.8% (-11.0pp).
Corticosteroids are the main low-precision overlap drugs (Dexamethasone 10.8%, Betamethasone 7.7%).

### h416: Cancer Same-Type → HIGH Promotion - INVALIDATED

**Cancer same-type has the largest full-to-holdout gap in the system:**
- rank<=5: 61.6% full-data → 16.1% ± 11.3% holdout (45.5pp gap!)
- rank<=5 + mech: 67.3% → 17.3% ± 14.3%
- ALL thresholds < 20% holdout precision

Root cause: Cancer subtype matching is almost entirely GT leakage.
Confirms h393/h396 decision to demote cancer_same_type from GOLDEN to MEDIUM.

### h446: Add Rank-Bucket Precision to Deliverable - VALIDATED

Added `rank_bucket_precision` column to production predictor and deliverable:
- RANK_BUCKET_PRECISION constant with holdout-validated values
- Deliverable now has 18 columns (was 16)
- Also added `transe_consilience` column to deliverable

### h432: Consolidate Small Hierarchy Rules - INVALIDATED

Small groups (<=2 diseases) have HIGHER precision than large groups:
- Autoimmune: small 88.2% vs large 73.4%
- Infectious: small 86.2% vs large 47.4%
- 1-disease groups encode specific medical knowledge. Do NOT consolidate.

### h449: Corticosteroid MEDIUM Demotion - INVALIDATED

Corticosteroids at 20.8% MEDIUM precision ≈ MEDIUM holdout 21.2%. Not below tier threshold.
Most individual steroids (Prednisolone 23.6%, Dexamethasone 22.6%) are at/above average.

### h448: HIGH Tier Rank Gradient Anomaly - VALIDATED

HIGH rank 16-20 has 56.5% precision because hierarchy rules RESCUE low-ranked drugs:
- RA hierarchy: 94.7% at rank 16-20, Spondylitis/Colitis: 100%
- Default rule at rank 1-5 = 16% precision (dilutes rank 1-5)
- On holdout, gradient re-emerges (43.0% rank 1-5 vs 21.0% rank 16-20)

### New Hypotheses Generated
- **h446-h449:** (tested this session - see above)

### Recommended Next Steps
1. **h447:** Cancer subtype leakage mitigation (Priority 4, medium effort)
2. **h410:** Literature validation of 1-disease hierarchy rules (Priority 3, medium effort)
3. **h441:** Drug-level embedding stability for within-tier ranking (Priority 4, medium effort)

---

## Previous Session: h435, h434, h437, h405, h439, h440 (2026-02-06)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 6**
- h435: Deliverable Regeneration with h402 Tier Demotions - **VALIDATED** (14,150 predictions, tier ordering correct)
- h434: Feature Inflation Root Cause: LOO Frequency - **INVALIDATED** (LOO changes <0.5pp; kNN neighborhood changes are the real driver)
- h437: kNN Neighborhood Overlap Analysis - **VALIDATED** (Jaccard 0.664, rank-20 crossings 4.1/disease)
- h405: Multi-Method Consilience Ensemble - **VALIDATED** (TransE top-30 = +13.6pp lift on holdout!)
- h439: Implement TransE Consilience in Production - **VALIDATED** (annotation flag, not tier promotion)
- h440: TransE Threshold Optimization - **VALIDATED** (top-30 optimal at 38.9% precision)

### h435: Deliverable Regeneration - VALIDATED

Regenerated production deliverable with h402 tier demotions. 14,150 predictions across 473 diseases.
Tier ordering correct: GOLDEN 64.1% > HIGH 52.5% > MEDIUM 27.0% > LOW 12.0% > FILTER 11.3%.

### h434: Feature Inflation Root Cause - INVALIDATED

LOO frequency is a negligible effect (0-0.5pp per tier). The real driver of holdout degradation is kNN neighborhood changes (5-10pp effect), not frequency inflation.

### h437: kNN Neighborhood Overlap - VALIDATED

Mean Jaccard overlap: 0.664 (66.4% of k=20 neighbors retained in holdout). Mean 4.1 drugs cross rank-20 boundary per disease.

### h405: Multi-Method Consilience Ensemble - VALIDATED (BREAKTHROUGH)

TransE top-30 agreement = +13.6pp holdout lift for MEDIUM tier. Implemented as boolean flag on DrugPrediction.

### h439-h440: TransE Implementation and Optimization - VALIDATED

TransE consilience as annotation flag (not tier promotion). Top-30 optimal threshold (38.9% precision).

### Recommended Next Steps
1. **h436:** kNN Bootstrap for rank stability (Priority 3, medium effort)
2. **h407:** Build Comprehensive Drug/Disease ID Mapping Infrastructure (Priority 2, high effort)

---

## Previous Session: h427, h402, h430, h429, h428, h433 (2026-02-05)

[See git history for older sessions]
