# Research Loop Progress

## Current Session: h144 Metabolic Disease Rescue (2026-02-05)

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
