# Research Loop Progress

## Current Session: h410 (2026-02-06)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 1**
- h410: Literature Validation of 1-Disease Hierarchy Rules - **VALIDATED**

### h410: Literature Validation of Small Hierarchy Rules - VALIDATED

**Objective:** Validate whether hierarchy rules with <=3 GT diseases encode genuine medical knowledge or are memorization artifacts.

**Method:** Literature review of 20 small hierarchy groups against treatment guidelines (GINA, GOLD, ACR, AAN, ACC/AHA, IDSA, etc.)

**Results:**
- 14/20 CONFIRMED by treatment guidelines (70%)
- 5/20 PARTIAL (heterogeneous groups or string-matching contamination)
- 0/20 pure memorization
- 1/20 N/A (gout - no GT diseases)

**CRITICAL FINDING: 3 String-Matching Bugs Discovered and Fixed:**
1. `'sle'` in lupus variants matched `'sleep'`/`'sleepiness'` → obstructive sleep apnea, hypersomnia falsely in lupus group
2. `'cystitis'` in UTI matched cholecystitis, dacryocystitis, interstitial cystitis (not UTIs)
3. bare `'fibrosis'` in pulmonary_fibrosis matched `'cystic fibrosis'` (different disease)
4. `'bronchitis'` in respiratory_infection matched `'chronic bronchitis'` (which is COPD, not infectious)

**Fixes Applied:**
- Removed `'sle'` from lupus variants (too short, causes false positives)
- Removed bare `'fibrosis'` from pulmonary_fibrosis variants
- Added `HIERARCHY_EXCLUSIONS` dict for remaining false matches (cystitis, bronchitis, cystic fibrosis)
- Updated both `_build_disease_hierarchy_mapping()` and `_check_disease_hierarchy_match()` to use exclusions
- Fixed same bug in `scripts/h402_rule_precision_audit_v2.py`

**Impact (5-seed holdout validated):**
| Tier | Before | After | Δ |
|------|--------|-------|---|
| GOLDEN | 52.9% ± 6.0% | 53.9% ± 7.1% | +1.0pp (noise) |
| HIGH | 50.6% ± 10.4% | 49.9% ± 8.2% | -0.7pp (noise) |
| **MEDIUM** | **21.2% ± 1.9%** | **22.3% ± 2.0%** | **+1.1pp** |
| LOW | 12.2% ± 1.9% | 12.0% ± 1.8% | -0.2pp (noise) |
| FILTER | 7.0% ± 1.5% | 6.9% ± 1.6% | -0.1pp (noise) |

271 low-quality predictions moved from MEDIUM to LOW. Total predictions preserved (13,472).

**Key Conclusions:**
1. **Hierarchy rules = genuine medical knowledge** - 14/20 confirmed by clinical guidelines
2. **0% holdout = structural absence** (confirmed by h432: small groups have 87% precision)
3. **Substring matching is fragile** - 3 bugs found with short variant strings
4. **HIERARCHY_EXCLUSIONS pattern** - reusable for preventing future false matches
5. **MEDIUM +1.1pp** is a genuine data quality improvement from fixing contamination

### New Hypotheses Generated
- h467: Systematic Substring Matching Audit for All Hierarchy Variants (Priority 4, low effort)
- h468: Neuropathy Hierarchy Group Decomposition (Priority 5, low effort)
- h469: Word-Boundary-Aware Hierarchy Matching (Priority 4, medium effort)

### Recommended Next Steps
1. **h467:** Systematic audit of ALL variant substrings (Priority 4, low effort) - may find more bugs
2. **h469:** Word-boundary matching (Priority 4, medium effort) - systematic fix vs manual exclusions
3. **h461:** Sparse neighborhood disease classification (Priority 5, low effort)

---

## Previous Session: h462, h463, h466, h464, h465 (2026-02-06)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 5**
- h462: Category-Specific MEDIUM→HIGH Promotion - **PARTIALLY VALIDATED** (3 categories demoted, none promoted)
- h463: GI MEDIUM Demotion to LOW - **VALIDATED** (confirmed by h462)
- h466: Category Holdout Precision as Deliverable Column - **VALIDATED** (19-column deliverable)
- h464: Psychiatric MEDIUM→HIGH Promotion - **INVALIDATED** (p=0.232, not significant)
- h465: Immunological Deep Dive - **VALIDATED** (100% self-referential GT hits)

### h462: Category-Specific MEDIUM Promotion/Demotion - PARTIALLY VALIDATED

Comprehensive 5-seed holdout validation of MEDIUM tier precision by disease category.

**No categories qualify for HIGH promotion:**
| Category | Full-Data | Holdout | ±std | Decision |
|----------|-----------|---------|------|----------|
| Psychiatric | 54.8% | 45.7% | 5.4 | KEEP (close but below 50.8%) |
| Renal | 31.4% | 43.5% | 25.2 | KEEP (extreme variance) |
| GI | 22.9% | 31.8% | 17.9 | KEEP (already demoted by h463) |
| Musculoskeletal | 33.3% | 29.8% | 13.5 | KEEP |

**Three categories demoted MEDIUM→LOW:**
| Category | Full-Data | Holdout | ±std | Overfitting Gap | n_diseases |
|----------|-----------|---------|------|-----------------|------------|
| Immunological | 38.9% | 2.5% | 3.5 | **36.4pp** | 5 |
| Neurological | 15.7% | 10.2% | 11.1 | 5.5pp | 24 |
| Reproductive | 4.5% | 0.0% | 0.0 | 4.5pp | 5 |

**Impact:**
- MEDIUM precision: 25.3% → 26.7% full-data (+1.4pp)
- 113 predictions moved from MEDIUM to LOW (3% of MEDIUM)
- Immunological = most overfitted MEDIUM category (36pp gap from 5 diseases)

**Implementation:**
- Added `MEDIUM_DEMOTED_CATEGORIES` set in `_assign_confidence_tier()`
- Updated target overlap promotion guard to exclude demoted categories
- Removed stale CATEGORY_PRECISION entries for demoted category MEDIUM tiers

### h463: GI MEDIUM Demotion - VALIDATED
- GI-as-LOW has 10.9% full-data precision (matches LOW tier avg)
- Prior "0% holdout" claim was inaccurate but demotion still justified
- Only 3 residual GI MEDIUM predictions survive via incoherent_demotion path

### Key Conclusions

1. **Small-n categories are massively overfitted:** Immunological (5 diseases) has 36pp full-to-holdout gap
2. **Psychiatric is the strongest MEDIUM category** (45.7% holdout) but still 5pp below HIGH threshold
3. **Category-specific precision varies enormously** even within same tier (0-47% holdout for MEDIUM)
4. **Prior h462 claims about renal/musculoskeletal promotion were overoptimistic** based on too few seeds

### New Hypotheses Generated
- h464: Psychiatric MEDIUM→HIGH Promotion with Additional Evidence (Priority 4)
- h465: Immunological Category Deep Dive - Why 36pp Overfitting Gap? (Priority 5)
- h466: Category-Specific Holdout Precision as Deliverable Column (Priority 4)

### Recommended Next Steps
1. **h466:** Add category holdout precision to deliverable (Priority 4, low effort)
2. **h464:** Investigate psychiatric HIGH promotion (Priority 4, medium effort)
3. **h410:** Literature validation of 1-disease hierarchy rules (Priority 3, medium effort)

---

## Previous Session: h453, h456, h457, h458, h450 (2026-02-06)

### Session Summary

**Agent Role:** Research Executor
**Status:** In Progress
**Hypotheses Tested: 5**
- h453: External-Signal-Only Within-Tier Ranking - **INVALIDATED** (no external signal adds >0pp to kNN rank)
- h456: Train Frequency as Disease-Level Difficulty Predictor - **INVALIDATED** (confounded by category)
- h457: Within-Tier Rank Calibration Curve - **VALIDATED** (MEDIUM monotonic, HIGH broken)
- h458: HIGH Tier Rank Instability Diagnosis - **VALIDATED** (hierarchy rescue causes non-monotonicity)
- h450: Weighted kNN by Neighborhood Stability Score - **INVALIDATED** (direction varies by tier)

### h453: External-Signal-Only Within-Tier Ranking - INVALIDATED

External signals (target overlap, mechanism, ATC, train frequency) do NOT improve within-tier ranking beyond raw kNN rank.

Key results (5-seed holdout, MEDIUM tier):
- rank_only: +8.8 ± 3.7pp (SIGNIFICANT, confirms h443)
- freq_only: +6.7 ± 2.2pp (significant independently)
- rank + freq composite: +7.8pp (additive = -1.0pp — WORSE than rank alone)
- All composites perform worse than rank alone

**Collider effect discovered:** Within a tier, kNN rank and external signals are ANTI-CORRELATED because tier assignment conditions on both. High kNN rank + low external support → MEDIUM. Low kNN rank + high external support → MEDIUM. Adding external signals to rank introduces noise.

**Practical conclusion:** Current system (kNN rank order within each tier) is already optimal.

### h456: Train Frequency as Disease-Level Difficulty Predictor - INVALIDATED

Mean train_frequency of top-20 predictions appears to predict disease-level R@30 (+16.0pp Q1 vs Q4) but is ENTIRELY confounded by disease category.

- Q1 (highest freq): autoimmune, dermatological, infectious → easy categories
- Q4 (lowest freq): metabolic, neurological, cancer → hard categories
- Within-category analysis: gap REVERSES to -5.1pp (not significant)

**Lesson:** Always control for confounders before reporting signals.

### h457: Within-Tier Rank Calibration Curve - VALIDATED

Systematic analysis of tier × rank bucket precision on holdout:
- MEDIUM: **MONOTONIC and reliable** (R1-5=20.5%, R6-10=16.3%, R11-15=10.8%, R16-20=9.2%)
- FILTER: approximately monotonic on holdout
- HIGH: **BROKEN calibration** — hierarchy rescue creates 15-38pp full-to-holdout gaps
- LOW: rank calibration REVERSES on holdout (collider effect)

Updated RANK_BUCKET_PRECISION constants with new holdout-validated values.

### Key Conclusions

1. **Within-tier ranking is SOLVED:** kNN rank is optimal, external signals cannot improve it (h453)
2. **Disease difficulty is category-driven:** mean drug frequency is a confound, not a signal (h456)
3. **Rank calibration varies drastically by tier:** MEDIUM=reliable, HIGH=broken, LOW=reversed (h457)
4. **Collider bias is fundamental:** tier assignment conditions on rank + signals, making them anti-correlated within tiers

### h458: HIGH Tier Rank Instability Diagnosis - VALIDATED

Hierarchy rescue rules promote drugs at ALL rank positions, creating non-monotonic rank calibration.
- Hierarchy HIGH: flat ~31% precision across all rank buckets (hierarchy IS the signal)
- Non-hierarchy HIGH: clean gradient 31.3% → 10.7% (rank IS the signal)
- R16-20 precision spike (38.1%) explained by hierarchy rescue at high ranks

### h450: Weighted kNN by Neighborhood Stability Score - INVALIDATED

Mean neighbor similarity is NOT a useful meta-confidence signal.
- LOW-similarity diseases have HIGHER precision in most tiers (GOLDEN, HIGH, FILTER)
- LOW-sim diseases are "island diseases" with unique signatures, rescued by hierarchy rules
- MEDIUM inverts: HIGH-sim Q1=26.0% > LOW-sim Q4=13.4%, but not significant within-category
- The tier system already compensates for neighborhood quality

### Key Conclusions from This Session

1. **Within-tier ranking is SOLVED:** kNN rank is optimal, no external signal improves it (h453)
2. **Collider effect is real and fundamental:** Tier assignment makes rank and signals anti-correlated within tiers (h453)
3. **Category confounding is pervasive:** Mean drug frequency (h456) and mean neighbor similarity (h450) are both confounded by category
4. **MEDIUM rank calibration is reliable:** monotonic on holdout, 20.5% → 9.2% across rank buckets (h457)
5. **HIGH rank calibration is broken:** hierarchy rescue at all ranks, not rank-dependent (h458)
6. **Sparse neighborhoods ≠ unreliable:** Low-sim diseases actually get BETTER predictions via hierarchy rescue (h450)

### New Hypotheses Generated
- h455: Collider Bias Decomposition (Priority 5)
- h458: HIGH Rank Instability (tested, validated)
- h459: Category-Adjusted Rank Calibration (Priority 5)
- h460: Split HIGH into Hierarchy vs Default (Priority 5)
- h461: Sparse Neighborhood Disease Classification (Priority 5)

### Recommended Next Steps
1. **h410:** Literature Validation of 1-Disease Hierarchy Rules (Priority 3, medium effort)
2. **h373:** Weighted Rank Fusion by Method Confidence (Priority 4, medium effort)
3. **h367:** Disease-Specific Gene Weighting (Priority 4, medium effort)

---

## Previous Session: h436, h451, h441, h422, h424, h454 (2026-02-06)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 6**
- h436: kNN Bootstrap Ensemble for Rank Stability - **INVALIDATED** (bootstrap hurts R@30 -1.17pp, drift +30.3% worse)
- h451: k-Expansion for Rank Stabilization - **INVALIDATED** (multi-k only +0.31pp, no drift reduction)
- h441: Drug-Level Embedding Stability - **INVALIDATED** (97.2% full-data → 7.4% holdout, circular signal)
- h422: Expand Top-N from 30 to 50 - **INVALIDATED** (rank 31-50 overlap>=3 only 4.4% holdout)
- h424: base_to_complication Precision Discrepancy - **VALIDATED** (DKA self-prediction = 65% of hits)
- h454: Circularity Audit of Tier Rules - **VALIDATED** (7 structural-absence rules, 19 genuine rules)

### h436: kNN Bootstrap Ensemble - INVALIDATED

Subsampling 80% of training diseases HURTS performance because it removes actual nearest neighbors.
- Full-data: Standard 87.47% vs Bootstrap 84.56% (-2.91pp)
- Holdout (5-seed): Standard 39.22% ± 2.26% vs Bootstrap 38.05% ± 1.49% (-1.17pp)
- Rank drift: +30.3% WORSE (7.19 vs 5.54 crossings)
- Root cause: kNN subsampling degrades neighborhood quality, unlike classical bagging

### h451: k-Expansion - INVALIDATED

Using multiple k values (15,20,25,30) with Borda count or score averaging does not help.
- k=30 holdout: +0.31pp over k=20 (within noise)
- Multi-k: +0.15 to +0.20pp (within noise)
- Rank drift: identical (5.41-5.59 crossings vs 5.54 for standard)
- Combined with h436: kNN instability is inherent to embedding space, not fixable by parameter tuning

### h441: Drug-Level Embedding Stability - INVALIDATED

Drug rank consistency (CV across diseases) is entirely circular with GT.
- Full-data: Q1 (most stable) = 97.2% vs Q4 (least stable) = 17.6% — MASSIVE signal
- Holdout: REVERSES to 7.4% vs 9.4% (-2.1pp, wrong direction!)
- Root cause: CV=0 drugs (always same rank) are drugs in GT for ALL their kNN neighbors
- kNN-derived metrics are inherently confounded by GT overlap

### h422: Top-N Expansion to 50 - INVALIDATED

Target overlap IS a valid non-circular signal (~2x lift) but rank 31-50 base precision is too low.
- Rank 31-40 overlap>=3: 8.8% full → 4.4% ± 1.3% holdout (below 20% MEDIUM threshold)
- Rank 41-50 overlap>=3: 8.8% full → 8.3% ± 2.6% holdout
- 7th independent confirmation that rank>20 filter is correct

### h424: base_to_complication Discrepancy - VALIDATED

Root cause: DKA (diabetic ketoacidosis) is its own #1 nearest neighbor (sim=1.0).
ALL 8 GT drugs come only from DKA itself, no other neighbors contribute.
- Full-data: 6/7 DKA hits inflate precision to 28.1% (h412)
- Holdout: 0/7 DKA hits → ~15.6% (h421 was correct)
- 65% of base_to_complication hits are self-referential

### h454: Circularity Audit - VALIDATED

Systematic audit of all tier rules for circularity:
- 7 hierarchy rules with 0% holdout = structural absence (drug_disease_groups disappears when disease held out)
- 19 rules with gap ≤10pp = GENUINE (including default n=9698, RA n=78, arrhythmia n=46)
- No hidden circular rules beyond known small-group limitation and DKA self-prediction
- Min n≥30 for reliable holdout validation (confirmed)

### Key Conclusions from this Session

1. **kNN instability is INHERENT** (h436+h451): Neither subsampling nor expansion can fix it. The rank>20 filter is the correct compensation. This research direction is now CLOSED after 7 independent confirmations.

2. **kNN-derived metrics are circular** (h441): Any metric computed from kNN predictions (drug rank CV, frequency, stability) correlates with GT on full-data but fails holdout. Within-tier ranking must use EXTERNAL signals only.

3. **Self-referential leakage is real but contained** (h424+h454): DKA self-prediction is the main example. 19 of the large-n rules are genuine. The system's circularity is well-characterized.

### New Hypotheses Generated
- h450: Weighted kNN by Neighborhood Stability Score (Priority 4)
- h451: k-Expansion (tested and invalidated this session)
- h452: Per-Disease k Optimization by Category (Priority 5)
- h453: External-Signal-Only Within-Tier Ranking (Priority 4)
- h454: Circularity Audit (tested and validated this session)

### Recommended Next Steps
1. **h453:** External-Signal-Only Within-Tier Ranking (Priority 4, medium effort) - use target overlap, mechanism, ATC as within-tier ranking
2. **h410:** Literature Validation of 1-Disease Hierarchy Rules (Priority 3, medium effort) - validate small rules via clinical evidence
3. **h389:** Rescued Disease Analysis (Priority 4, low effort) - understand ensemble rescue patterns

---

## Previous Session: h442-h449 (2026-02-06)

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

### h447: Cancer Subtype Leakage Quantification - VALIDATED

Cancer same-type holdout = 10.7% ± 3.1% (below MEDIUM avg 21.2%). Represents 28% of MEDIUM predictions.
No drug class achieves >30% holdout. Taxane/Vinca best at 28.1% ± 12.2%. "Other" at 7.5%.
Cancer predictions are a known MEDIUM limitation. No further demotion possible without losing 36.7% of GT hits.

### New Hypotheses Generated
- **h446-h449:** (tested this session - see above)

### Recommended Next Steps
1. **h410:** Literature validation of 1-disease hierarchy rules (Priority 3, medium effort)
2. **h441:** Drug-level embedding stability for within-tier ranking (Priority 4, medium effort)
3. **h422:** Expand Top-N from 30 to 50 with target overlap rescue (Priority 4, medium effort)

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
