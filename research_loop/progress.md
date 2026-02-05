# Research Loop Progress

## Current Session: h175 Cross-Category Knowledge Transfer (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested:**
- h175: Cross-Category Knowledge Transfer - **INVALIDATED** (hurts performance)
- h176: Production Predictor Initialization Speedup - **VALIDATED** (33x speedup)

### h175: Cross-Category Knowledge Transfer - INVALIDATED

Tested whether boosting related-category neighbors (based on drug overlap) improves kNN.

**Related Categories Tested (by drug overlap):**
- dermatological ↔ ophthalmic (26.9% overlap)
- respiratory ↔ ophthalmic (17.2%)
- cardiovascular ↔ metabolic (15.4%)
- metabolic ↔ renal (13.6%)
- infectious ↔ ophthalmic/respiratory

**Results:**
| Method | R@30 | Delta | p-value |
|--------|------|-------|---------|
| Baseline (h170 same-cat only) | 36.39% | - | - |
| Same + Related boost | 35.32% | -1.07pp | 0.15 |

**Key Finding:** All alpha values hurt or are neutral. Even 26.9% drug overlap isn't sufficient for knowledge transfer.

**Per-Category Harm:**
- Infectious: -13.8pp (related neighbors bring wrong drugs)
- Dermatological: -6.7pp

**Learning:** Category boundaries matter more than drug statistics suggest. kNN needs near-complete drug overlap (same-category), not partial (related-category).

**New Hypotheses:**
- h181: Drug-Level Cross-Category Transfer
- h182: Category Boundary Refinement

### Cumulative Statistics (2026-02-05)
| Status | Count |
|--------|-------|
| Validated | 83 |
| Invalidated | 41 |
| Inconclusive | 8 |
| Blocked | 18 |
| Deprioritized | 2 |
| Pending | 30 |
| **Total Tested** | **132** |

---

## Previous Session: h176 Production Predictor Initialization Speedup (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested:**
- h176: Production Predictor Initialization Speedup - **VALIDATED** (33x speedup achieved!)

### h176: Production Predictor Initialization Speedup - VALIDATED

The production predictor took ~210 seconds to initialize due to fuzzy disease matching on 10K+ rows of ground truth data.

**Root Cause Analysis:**
- `DiseaseMatcher.get_mesh_id()` has O(n) complexity for steps 3-4 (linear search through mappings)
- Called once per row in the ground truth Excel file (~10K+ rows)
- Total complexity: O(n×m) where n=rows, m=mappings

**Solution: Caching**
Implemented a caching mechanism that:
1. Pre-computes ground truth mappings on first run
2. Saves to `data/cache/ground_truth_cache.json`
3. Uses MD5 hash of source file modification times as cache key
4. Automatically invalidates when source files change

**Performance Results:**
| Scenario | Time | Speedup |
|----------|------|---------|
| Cold start (no cache) | 212s | 1x (baseline) |
| **Warm start (from cache)** | **6.5s** | **33x** |

**Files Monitored for Cache Invalidation:**
- `data/reference/everycure/indicationList.xlsx`
- `data/reference/mesh_mappings_from_agents.json`
- `data/reference/mondo_to_mesh.json`
- `data/reference/drugbank_lookup.json`
- `src/disease_name_matcher.py`

**Implementation:** Updated `src/production_predictor.py`:
- Added `_get_cache_key()` method for cache invalidation
- Modified `_load_ground_truth()` to use cache

**New Hypotheses Generated:**
- h178: DiseaseMatcher Algorithm Optimization (O(1) lookup instead of O(n))
- h179: Embedding Loading Optimization (binary .npy format)
- h180: Batch Prediction API for Web Service

### Cumulative Statistics (2026-02-05)
| Status | Count |
|--------|-------|
| Validated | 83 |
| Invalidated | 40 |
| Inconclusive | 8 |
| Blocked | 18 |
| Deprioritized | 2 |
| Pending | 29 |
| **Total Tested** | **131** |

---

## Previous Session: h170 Selective Category Boosting (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested:**
- h170: Category-Aware kNN with Same-Category Boost - **VALIDATED** (selective boost works!)

### h170: Selective Category Boosting - VALIDATED

Tested whether boosting same-category neighbor weights improves kNN predictions for isolated categories (identified in h168).

**Key Discovery: Universal boost HURTS, Selective boost HELPS**

| Method | R@30 | Delta | p-value |
|--------|------|-------|---------|
| Baseline (no boost) | 38.62% | - | - |
| Universal boost (all categories) | 38.04% | -0.58pp | hurts |
| **Selective boost (isolated only)** | **41.03%** | **+2.40pp** | **0.009** |

**Why universal boost fails:**
- Infectious diseases: -11.2pp (n=32) — same-category neighbors are too sparse
- Other category: -4.8pp (n=215) — "other" is heterogeneous catch-all

**Selective boost categories (isolated):** neurological, respiratory, metabolic, renal, hematological, immunological

**Per-category gains from selective boost:**
| Category | Baseline | Boosted | Delta |
|----------|----------|---------|-------|
| immunological | 0.0% | 40.0% | +40.0pp |
| hematological | 15.9% | 54.4% | +38.5pp |
| respiratory | 9.2% | 26.0% | +16.8pp |
| neurological | 57.1% | 71.4% | +14.3pp |
| metabolic | 19.9% | 33.8% | +13.9pp |
| renal | 18.3% | 18.8% | +0.5pp |

**Statistical validation:**
- All 5 seeds show improvement (no lucky seed)
- Cohen's d = 2.375 (huge effect size)
- Highly significant (p=0.009)

**Implementation:** Added to `production_predictor.py`:
- `SELECTIVE_BOOST_CATEGORIES` = {neurological, respiratory, metabolic, renal, hematological, immunological}
- `SELECTIVE_BOOST_ALPHA` = 0.5 (1.5x weight for same-category neighbors)

**New Hypotheses Generated:**
- h175: Cross-Category Knowledge Transfer (could psychiatric help neurological?)
- h176: Production Predictor Initialization Speedup (~210s init is slow)

### Cumulative Statistics (2026-02-05)
| Status | Count |
|--------|-------|
| Validated | 82 |
| Invalidated | 40 |
| Inconclusive | 8 |
| Blocked | 18 |
| Deprioritized | 2 |
| Pending | 27 |
| **Total Tested** | **130** |

---

## Previous Session: h167/h168 Category Precision + Neurological Gap (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested:**
- h167: Add Category-Specific Precision to Production Output - **VALIDATED** (completed)
- h168: Neurological Disease Performance Gap Analysis - **VALIDATED** (root cause identified)

### h167: Category-Specific Precision - VALIDATED (continued from previous)

Completed the implementation of category-specific precision in production predictor.

**Final Implementation:**
- Added GOLDEN/HIGH tier values from validated rescue criteria (h136/h144/h150/h154/h157)
- Key precision values: Autoimmune GOLDEN 75.4%, Infectious GOLDEN 55.6%, Metabolic GOLDEN 60.0%
- All tests passing - category lookup working correctly

### h168: Neurological Disease Performance Gap Analysis - VALIDATED

**ROOT CAUSE IDENTIFIED:** Embedding isolation + low GT drug coverage

**Key Metrics:**
- Only 6 neurological diseases in training (vs 62 cancer, 337 other)
- **3.3% same-category neighbors** (vs 61% for cancer, 87% for other)
- 92% of neurological neighbors are from 'other' category
- Average drug coverage in neighbors: 37% (vs 62.5% for autoimmune)

**Specific Failures:**
| Disease | GT Drugs | Drugs in Pool | Recall@10 |
|---------|----------|---------------|-----------|
| Alzheimer's | 3 | 0 | **0.0%** |
| Parkinson's | 19 | 2 | 5.3% |
| Epilepsy | 17 | 2 | 11.8% |

**Key Finding:** kNN CANNOT find neurological drugs because neighbors are from wrong categories.
- Alzheimer's top recommendations: Fluoxetine, Eculizumab (NOT GT drugs)
- Parkinson's: Only Amantadine and Droxidopa reachable (2/19)
- Epilepsy: Only Carbamazepine and Diazepam reachable (2/17)

**Drug Class Rescue NOT Feasible:**
- Neurological drugs have low training frequency (most freq=1-4)
- Anticonvulsants used cross-category (psychiatric, pain)
- No single dominant class like statins for metabolic

**New Hypotheses Generated:**
- h170: Category-Weighted kNN for Isolated Categories (priority 3)
- h171: Neurological Drug Catalog Injection (priority 2)
- h172: Disease Graph Distance for Category Similarity (priority 4)
- h173: Epilepsy-Specific Analysis (priority 4)

### Cumulative Statistics (2026-02-05)
| Status | Count |
|--------|-------|
| Validated | 81 |
| Invalidated | 40 |
| Inconclusive | 8 |
| Blocked | 18 |
| Deprioritized | 2 |
| Pending | 25 |
| **Total Tested** | **129** |

---

## Previous Session: h163/h165/h167 Drug Class + Calibration + Production (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested:**
- h163: Drug Class Precision Ranking - **VALIDATED** (no new classes found, confirms existing)
- h165: Per-Disease-Category Precision Calibration - **VALIDATED** (massive miscalibration found)
- h167: Add Category-Specific Precision to Production - **VALIDATED** (implemented in production_predictor.py)

### h163: Drug Class Precision Ranking - VALIDATED

Systematically analyzed 26 drug classes across all disease categories to find hidden high-precision pockets.

**Key Findings:**
1. **No NEW high-precision classes** meeting >35% precision + n>=10 threshold
2. All existing production classes confirmed:
   - Statins rank<=5: 38.5% (n=13)
   - Tetracyclines rank<=5: 29.5% (n=281)
   - Beta-blockers rank<=10: 26.5% (n=34)
3. Small-sample high-precision classes (need more data):
   - SGLT2 inhibitors for cardiovascular: 71.4% (n=7 too small)
   - Thiazolidinediones for metabolic: 66.7% (n=6 too small)

**Conclusion:** Existing production rescue criteria are already optimal. No actionable new drug classes.

### h165: Per-Disease-Category Precision Calibration - VALIDATED

Computed precision by (disease_category, confidence_tier) to identify calibration issues.

**Analysis:** 5 seeds, 101,939 predictions, 2,455 diseases

**MASSIVE MISCALIBRATION FOUND:**

| Category | MEDIUM Tier Precision | vs Overall 19.3% |
|----------|----------------------|------------------|
| Psychiatric | 85.0% | +65.7 pp |
| Autoimmune | 77.8% | +58.5 pp |
| Respiratory | 54.2% | +34.9 pp |
| Dermatological | 49.0% | +29.7 pp |
| Metabolic | 47.6% | +28.3 pp |
| Cancer | 45.7% | +26.4 pp |
| **Other (uncategorized)** | **17.3%** | **-2.0 pp** |
| **Neurological** | **26.1%** | **+6.8 pp (lowest)** |

**Even FILTER tier has high precision for some categories:**
- Psychiatric FILTER: 90.0% (!!)
- Autoimmune FILTER: 45.9%
- Respiratory FILTER: 35.7%

**Implication:** Current tier-only calibration under-reports precision for ~10 categories and over-reports for 'other'. Category-specific calibration would dramatically improve prediction communication.

### New Hypotheses Generated
- **h167**: Add Category-Specific Precision to Production Output (priority 2, low effort)
- **h168**: Neurological Disease Performance Gap Analysis (priority 3, medium effort)
- **h169**: Other Category Disease Re-Classification (priority 3, low effort)

### h167: Add Category-Specific Precision to Production Output - VALIDATED

Implemented category-specific precision in production predictor based on h165 findings.

**Implementation:**
1. Added `CATEGORY_PRECISION` lookup table with:
   - MEDIUM/LOW/FILTER values from h165 (5-seed analysis)
   - GOLDEN/HIGH values from h136/h144/h150/h154/h157 rescue validations
2. Added `get_category_precision(category, tier)` function with fallback
3. Updated `PredictionResult.summary()` to include `category_precision_by_tier`
4. Updated CLI display to show category-adjusted precision

**Key precision values now available:**
- Autoimmune GOLDEN: 75.4% (vs generic 58%)
- Psychiatric MEDIUM: 85.0% (vs generic 14%)
- Neurological LOW: 15.0% (vs generic 6%)

Users now see accurate precision expectations based on their disease category.

### Cumulative Statistics (2026-02-05)
| Status | Count |
|--------|-------|
| Validated | 80 |
| Invalidated | 40 |
| Inconclusive | 8 |
| Blocked | 18 |
| Deprioritized | 2 |
| Pending | 23 |
| **Total Tested** | **128** |

---

## Previous Session: h154/h155/h157 Drug Class Analysis (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested:**
- h154: Cardiovascular Beta-Blocker Combined Criteria - **VALIDATED**
- h155: GI Drug Coverage Gap Analysis - **VALIDATED**
- h157: Autoimmune Drug Class Analysis - **VALIDATED**

### h154: Cardiovascular Beta-Blocker Combined Criteria - VALIDATED

Tested if beta-blocker + additional criteria could achieve >30% precision for cardiovascular diseases.

**Key Results (18 CV diseases, 450 predictions, 5-seed):**

| Criteria | N | Precision |
|----------|---|-----------|
| beta_blocker (all) | 53 | 17.0% |
| beta_blocker + rank<=10 | 34 | 23.5% |
| beta_blocker + rank<=5 | 15 | **33.3%** |
| beta_blocker + mechanism | 18 | 16.7% |

**Critical Finding:** Mechanism support HURTS precision for beta-blockers (17% → 16.7%). Rank constraint is the dominant signal.

**Production Update:** beta_blocker + rank<=5 added as HIGH tier for cardiovascular.

### h155: GI Drug Coverage Gap Analysis - VALIDATED

Investigated why GI drugs (PPIs, H2 blockers) appeared to have 0% precision in h150.

**Key Findings:**
1. GI drugs actually have HIGH precision when predicted:
   - PPI precision: 38.5% (5/13 hits)
   - H2 blocker precision: 80.0% (12/15 hits)
2. Only 4/27 GI diseases have GI-specific drugs in GT (duodenal ulcer, functional gastric disease, ulcer disease, short bowel syndrome)
3. Other GI diseases (IBD, liver) are correctly treated with immunosuppressants/corticosteroids

**Root Cause:** "GI failure" is actually CORRECT behavior - most GI diseases aren't treated with PPIs/H2 blockers. The model correctly predicts Dexamethasone, Azathioprine, Methotrexate for IBD.

**Production:** No rescue criteria needed - GI drugs already have high precision when applicable.

### h157: Autoimmune Drug Class Analysis - VALIDATED

Tested if biologics (anti-TNF, IL inhibitors) or DMARDs show class-specific precision for autoimmune diseases.

**Key Results:**

| Drug Class | Mean N/seed | Precision |
|------------|-------------|-----------|
| DMARDs | 21.6 | **75.4% ± 4.7%** |
| Anti-TNF | 0 | N/A (not predicted) |
| IL inhibitors | 0 | N/A (not predicted) |
| JAK inhibitors | 0.8 | 0% |

**Root Cause of Biologic Failure - Training Frequency:**
- Methotrexate: 275 (dominates)
- Cyclosporine: 232
- Azathioprine: 120
- Adalimumab: 4 (!)
- Most IL inhibitors: 0-2

Biologics simply don't appear in enough training diseases. This is DATA SPARSITY, not model failure.

**Production Update:** Added DMARD + rank<=10 as GOLDEN tier for autoimmune (75.4% precision).

### Cumulative Statistics (2026-02-05)
| Status | Count |
|--------|-------|
| Validated | 77 |
| Invalidated | 40 |
| Inconclusive | 8 |
| Blocked | 18 |
| Deprioritized | 2 |
| Pending | 21 |
| **Total Tested** | **125** |

---

## Previous Session: h153/h156 Safety & Combination Criteria (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested:**
- h153: Corticosteroid Metabolic Contraindication - **VALIDATED**
- h156: Combined Multi-Class Rescue Criteria - **INVALIDATED**

### h153: Corticosteroid Metabolic Contraindication - VALIDATED

Corticosteroids (prednisone, dexamethasone, etc.) were being predicted for diabetes but they CAUSE hyperglycemia.

**Implementation:**
1. Added CORTICOSTEROID_PATTERNS to confidence_filter.py
2. Added Rule 2b to exclude corticosteroids for metabolic diseases
3. Added safety check in production_predictor.py to assign FILTER tier

**Verification:**
- Type 2 diabetes: No corticosteroids in predictions (properly filtered)
- Corticosteroids still allowed for appropriate uses (hematological, inflammatory)

### h156: Combined Multi-Class Rescue Criteria - INVALIDATED

Tested whether combining drug classes (e.g., "antibiotic OR steroid") maintains precision while improving coverage.

**Results (5-seed evaluation):**
| Category | Combined Criteria | Precision | vs Single Class |
|----------|------------------|-----------|-----------------|
| Ophthalmic | antibiotic OR steroid + rank<=15 | 32.8% | -2.2 pp vs steroid |
| Cancer | taxane OR alkylating + rank<=10 | 37.5% | -12.5 pp vs alkylating |
| Dermatological | topical_steroid OR biologic + rank<=10 | 30.8% | +20.8 pp vs topical |

**Key Finding:** Drug classes within a category are NOT equally precise. Combining them averages out the signal. Single-class criteria (alkylating=50%, steroid=35%) are more precise than combinations.

### New Hypotheses Generated
- h163: Drug Class Precision Ranking - Find hidden high-precision classes
- h164: Contraindication Database - Systematic safety filter expansion
- h165: Per-Disease-Category Precision Calibration
- h166: Drug-Disease Mechanism Path Tracing

### Cumulative Statistics (2026-02-05)
| Status | Count |
|--------|-------|
| Validated | 74 |
| Invalidated | 40 |
| Inconclusive | 8 |
| Blocked | 18 |
| Deprioritized | 2 |
| Pending | 27 |
| **Total Tested** | **122** |

---

## Previous Session: h144 Metabolic Disease Rescue (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested:**
- h144: Metabolic Disease Rescue - **VALIDATED** (statin + rank<=10 = 60% precision)

### h144: Metabolic Disease Rescue - VALIDATED

Type 2 diabetes and other metabolic diseases previously showed 0 GOLDEN and 0 HIGH predictions in production. This hypothesis investigated alternative confidence signals.

**Key Results (720 predictions, 5 seeds, 6.1% base rate):**

| Drug Class | N | Precision | Notes |
|------------|---|-----------|-------|
| Statin | 21 | 47.6% | Best overall class |
| Statin + rank<=10 | 10 | **60.0%** | RESCUE CRITERIA |
| Insulin | 3 | 66.7% | Small n |
| Fibrate | 11 | 36.4% | - |
| Thiazolidinedione | 6 | 33.3% | - |
| Sulfonylurea | 6 | 33.3% | - |
| GLP-1/SGLT2 | 14 | 0.0% | Not in DRKG |

**Critical Finding:** Generic mechanism support FAILS for metabolic (freq>=10+mech = 0% precision).
Drug class is the dominant signal.

**Production Update:**
- Added `STATIN_DRUGS` set to production_predictor.py
- Statin + rank<=10 → GOLDEN tier for metabolic diseases
- Verified: hyperlipidemia now shows 4 GOLDEN statin predictions

**New Hypotheses Generated:**
- h150: Drug Class Rescue for Other Categories
- h151: Modern Drug Gap Analysis (GLP-1/SGLT2 not in DRKG)
- h152: ATC Code Integration for Precision

---

## Previous Session: h145, h146, h147, h149 Production & Validation (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested:**
- h145: Production Novel Prediction Export - **VALIDATED**
- h146: Minocycline Repurposing Validation - **VALIDATED** (20% validated, 10% false positive)
- h149: Mechanistic Contraindication Filter - **VALIDATED** (4+ false positives filtered)
- h147: Biologic Drug Prioritization - **VALIDATED** (124 biologics analyzed)

### Key Findings This Session

1. **h145 Novel Prediction Export:**
   - 2,391 novel predictions exported
   - 1,457 truly novel, 934 broad-spectrum
   - 39% are broad-spectrum drugs (corticosteroids, NSAIDs)

2. **h146 Validation:**
   - Minocycline -> malaria: VALIDATED
   - Minocycline -> pneumonia: VALIDATED
   - Adalimumab -> SLE: **FALSE POSITIVE** (TNF inhibitors contraindicated)

3. **h149 Contraindication Filter:**
   - Added TNF inhibitor contraindication filter
   - Filters adalimumab/infliximab/etanercept for SLE, MS, heart failure, AIH
   - 4+ false positives now correctly excluded

4. **h147 Biologic Prioritization:**
   - 124 biologic predictions (2 GOLDEN, 26 HIGH, 96 MEDIUM)
   - Pembrolizumab -> cholangiocarcinoma is FDA-APPROVED (GT gap)
   - Adalimumab -> autoimmune hepatitis: FALSE POSITIVE (added to filter)

### Cumulative Statistics (2026-02-05)
| Status | Count |
|--------|-------|
| Validated | 69 |
| Invalidated | 38 |
| Inconclusive | 8 |
| Blocked | 18 |
| Deprioritized | 2 |
| Pending | 18 |
| **Total Tested** | **117** |

---

## Previous Session: h69 Production Pipeline (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested:**
- h69: Production Pipeline Integration - **VALIDATED**
- h145: Production Novel Prediction Export - **VALIDATED**

### h69: Production Pipeline Integration - VALIDATED

Implemented unified production predictor integrating all validated research findings:

**Components Integrated:**
1. **kNN Collaborative Filtering (h39)** - Best method at 37.04% R@30
2. **Tiered Confidence System (h135)** - 9.1x precision separation
3. **Category-Specific Rescue (h136)** - Infectious 55.6%, Cardiovascular 38.2%

**Implementation:**
- `src/production_predictor.py` - DrugRepurposingPredictor class
- CLI: `python -m src.production_predictor "disease name"`
- JSON output: `--json` flag for programmatic use

**Tier System (validated):**
| Tier   | Precision | Criteria |
|--------|-----------|----------|
| GOLDEN | ~58%      | Tier1 + freq>=10 + mech OR category-rescued |
| HIGH   | ~21%      | freq>=15 + mech OR rank<=5 + freq>=10 + mech |
| MEDIUM | ~14%      | freq>=5 + mech OR freq>=10 |
| LOW    | ~6%       | All else passing filter |
| FILTER | ~3%       | rank>20 OR no_targets OR (freq<=2 AND no_mech) |

**Test Results:**
- rheumatoid arthritis (Tier 1): 13 GOLDEN, 6 MEDIUM, 1 LOW
- hepatitis C (Tier 3): 6 GOLDEN [rescued], 1 HIGH, 6 MEDIUM, 4 LOW
- type 2 diabetes (Tier 3): 5 MEDIUM, 15 LOW (no rescue criteria for metabolic)

### h145: Novel Prediction Export - VALIDATED

**Export Results (2026-02-05):**
- Total diseases evaluated: 448
- Novel predictions exported: 2,391
  - Truly novel (specific drugs): 1,457
  - Broad-spectrum (corticosteroids, NSAIDs): 934
- FDA-approved filtered: 958
- Safety filter exclusions: 45

**By Confidence Tier:**
| Tier | Total | Truly Novel | Broad-Spectrum |
|------|-------|-------------|----------------|
| GOLDEN | 72 | 16 | 56 |
| HIGH | 265 | 85 | 180 |
| MEDIUM | 2,054 | 1,356 | 698 |

**Top Truly Novel GOLDEN Predictions:**
1. Minocycline -> bacterial meningitis (score 0.99, freq 19, mech+)
2. Minocycline -> malaria (score 0.82, freq 19, mech+)
3. Doxycycline/Minocycline -> CF Pseudomonas (score 0.70)
4. Adalimumab -> SLE (score 0.44, freq 11, mech+)
5. Corticotropin -> autoimmune hepatitis (score 0.43, freq 15, mech+)

**Key Insight:** 39% of predictions are broad-spectrum drugs. The `is_broad_spectrum` flag helps collaborators prioritize truly novel predictions.

**By Confidence Tier (old format for compatibility):**
| Tier | Count | Expected Precision |
|------|-------|-------------------|
| GOLDEN | 72 | ~58% |
| HIGH | 266 | ~21% |
| MEDIUM | 2,046 | ~14% |

**Output Files:**
- `data/deliverables/novel_predictions_20260205.json`
- `data/deliverables/novel_predictions_20260205.xlsx`

**Top GOLDEN Predictions:**
1. Methylprednisolone -> systemic myasthenia gravis
2. Methylprednisolone -> Crohn's disease
3. Methylprednisolone -> hepatitis B [rescued]
4. Dexamethasone -> urticaria
5. Minocycline -> bacterial meningitis [rescued]

### New Hypotheses Added

- **h144**: Metabolic Disease Rescue - Alternative confidence signals for metabolic category

### Session Statistics
- Hypotheses tested: 2
- Validated: 2 (h69, h145)
- New hypotheses added: 1

### Cumulative Statistics (2026-02-05)
| Status | Count |
|--------|-------|
| Validated | 65 |
| Invalidated | 38 |
| Inconclusive | 7 |
| Blocked | 17 |
| Deprioritized | 2 |
| Pending | 14 |
| **Total Tested** | **130** |

### Key Learnings

1. **Production predictor unifies all validated findings** - kNN, tiered confidence, category rescue
2. **Different disease tiers produce different confidence distributions** - Tier 1 gets more GOLDEN
3. **Metabolic diseases remain unrescued** - No criteria found for >30% precision
4. **Category rescue applied correctly** - Infectious diseases get GOLDEN tier when criteria met

### Next Steps
1. **h144: Metabolic Disease Rescue** - Find alternative signals for this category
2. **h85: Metabolic Disease Alternative Similarity** - Different approach to hard category
3. **h91: Literature Mining** - PubMed drug-disease hypothesis extraction

---

## Previous Session: 9 Hypotheses Resolved (2026-02-05 earlier)

### Summary
- h122: Category Misclassification - INCONCLUSIVE
- h139: Hybrid Confidence - INVALIDATED
- h129: Mechanism-Category Interaction - VALIDATED (infectious 2.68x)
- h133: Non-Tier1 Golden Criteria - VALIDATED
- h138: Other Category Sub-Stratification - INVALIDATED
- h140: Bimodality Predictor - VALIDATED
- h141: Infectious Mechanism Weighting - VALIDATED
- h143: Zero-Hit Disease Predictor - VALIDATED (81.6%)
- h142: prob_h52 Feature Decomposition - INCONCLUSIVE

---

## Earlier Sessions (2026-02-05)

### h126, h128, h132, h137, h136 Session
- h126: XGBoost Feature Interaction - **VALIDATED**
- h128: High-Confidence Subset - **INVALIDATED**
- h132: Golden Predictions - **VALIDATED** (57.9% precision)
- h137: Why Tier 1 Succeeds - **VALIDATED** (13x drug overlap)
- h136: Tier 2/3 Category Rescue - **VALIDATED** (Infectious 55.6%)

---

## Older Sessions

See git history for detailed session logs.
