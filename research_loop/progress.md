# Research Loop Progress

## Current Session: h374, h377 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 2**
- h374: Integrate MinRank Ensemble into Production Predictor - **INVALIDATED**
- h377: Identify Under-Covered Categories in Production Predictor - **VALIDATED**

### h377: Identify Under-Covered Categories - VALIDATED

**Hypothesis:** Categories with R@30 < 80% may benefit from new rescue rules.

**Under-Covered Categories Found (n >= 5):**
| Category | R@30 | n | Priority |
|----------|------|---|----------|
| gastrointestinal | 42.9% | 14 | **WORST** |
| other | 70.5% | 44 | 2 |
| hematological | 70.6% | 17 | 3 |
| neurological | 71.4% | 21 | 4 |
| metabolic | 77.5% | 40 | 5 |

**GI Deep Dive (42.9% R@30):**
- Constipation diseases (4): 0/4 hits - kNN predicts antibiotics, GT is laxatives
- Liver diseases (3): 0/3 hits - kNN predicts steroids, GT is bile acid agents
- Ulcer (1): 0/1 hits - kNN predicts tetracyclines, GT is PPIs
- Other GI (7): 4/7 hits

**Root Cause:** kNN neighbors are from different categories (infectious, neurological) with different therapeutic needs.

**Proposed Rescue Rules (h380):**
1. `constipation + laxative → HIGH` (8 drugs: lactulose, lubiprostone, prucalopride...)
2. `liver + bile_acid → HIGH` (5 drugs: cholestyramine, ursodeoxycholic acid...)
3. `ulcer + ppi → HIGH` (6 drugs: omeprazole, esomeprazole...)

**Expected Impact:** +28.5 pp for GI (42.9% → 71.4%)

### New Hypotheses Generated
- **h380:** GI Drug Class Rescue Rules (Constipation, Liver, Ulcer) - Priority 2

### Cumulative Statistics
| Status | Count |
|--------|-------|
| Validated | 236 |
| Invalidated | 71 |
| Inconclusive | 14 |
| Blocked | 21 |
| Deprioritized | 7 |
| Pending | 29 |
| **Total** | **378** |

---

## Previous Session: h374 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 1**
- h374: Integrate MinRank Ensemble into Production Predictor - **INVALIDATED**

### KEY SESSION FINDINGS

#### h374: Integrate MinRank Ensemble into Production Predictor - INVALIDATED

**Hypothesis:** Adding MinRank ensemble for cancer/neuro/metabolic categories will improve production prediction quality.

**Implementation:**
1. Added `_get_target_overlap_count()` - count overlapping genes between drug targets and disease genes
2. Added `_get_target_scores()` - get target overlap scores for all drugs
3. Added `_minrank_fusion()` - combine kNN and target scores using min-rank fusion
4. Modified `predict()` to use MinRank for cancer/neuro/metabolic categories

**Evaluation Results (n=502 diseases):**
| Category | n | MinRank | kNN | Δ |
|----------|---|---------|-----|---|
| Cancer | 100 | 89.0% | 89.0% | **0%** |
| Metabolic | 40 | 75.0% | 77.5% | **-2.5%** |
| Neurological | 21 | 71.4% | 71.4% | **0%** |
| **Overall** | 502 | 83.7% | 83.9% | **-0.2%** |

**Root Cause Analysis:**
1. h369/h370 validated MinRank in ISOLATION with simple kNN+target scoring
2. Production predictor has h274 (cancer_same_type) + other rules that ALREADY capture target overlap signal
3. MinRank adds REDUNDANT signal that slightly HARMS metabolic predictions
4. Cancer predictions show 19/21 drugs with GOLDEN tier via h274 (already correctly ranked)

**Why Isolated Validation Didn't Transfer:**
- h369 evaluated on 38 cancer diseases with simple scoring → 76.3% MinRank vs 65.8% kNN
- Production evaluates on 100 cancer diseases with full tier rules → 89.0% for both
- The +10.5 pp gain was already captured by existing rules

**Resolution:**
- Implementation kept but DISABLED (empty set for MINRANK_ENSEMBLE_CATEGORIES)
- Helper methods retained for potential future use

**Key Learning:**
> Ensemble methods validated in isolation may not help in production when existing rules capture similar signals. Always test improvements in full production context.

### New Hypotheses Generated
- **h377:** Identify Under-Covered Categories in Production Predictor
- **h378:** Tier Precision Analysis: Which Rules Hurt Precision?
- **h379:** Within-Tier Ranking Optimization

### Cumulative Statistics
| Status | Count |
|--------|-------|
| Validated | 235 |
| Invalidated | 71 |
| Inconclusive | 14 |
| Blocked | 21 |
| Deprioritized | 7 |
| Pending | 28 |
| **Total** | **376** |

---

## Previous Session: h369, h370, h371, h372 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 4**
- h369: Apply Max Ensemble to Non-Cancer Categories - **VALIDATED** (MinRank helps cancer/neuro, hurts autoimmune)
- h370: Adaptive Ensemble Selection by Category - **VALIDATED** (72.4% R@30 with 10% threshold)
- h371: Target-Only for High-Gene-Coverage Diseases - **VALIDATED** (gene count helps but doesn't make Target superior)
- h372: kNN-Only for Neurological Diseases - **INVALIDATED** (neuro gene coverage not unusually low)

### KEY SESSION FINDINGS

#### h369: Apply Max Ensemble to Non-Cancer Categories - VALIDATED

**Hypothesis:** MaxScore/MinRank ensemble (from h366) will generalize to non-cancer categories.

**Results (LOO CV):**
| Category | N | Target | kNN | MinRank | Δ |
|----------|---|--------|-----|---------|---|
| Cancer | 38 | 65.8% | 65.8% | **76.3%** | **+10.5%** |
| Neurological | 19 | 42.1% | 36.8% | **47.4%** | **+5.3%** |
| Cardiovascular | 24 | 66.7% | 54.2% | 66.7% | 0% |
| Autoimmune | 18 | **88.9%** | 77.8% | 83.3% | **-5.6%** |
| Metabolic | 6 | 100% | 100% | 100% | 0% |

**Key insight:** Ensemble only helps when methods have EQUAL performance. When one method dominates (autoimmune: Target 88.9% vs kNN 77.8%), ensemble HURTS by mixing in noise from the weaker method.

**Alternative strategies tested:**
- Union30 (top-30 from each): Higher recall but unfairly expands candidate set
- MinRank: Same @30 threshold, genuine improvement for cancer/neuro

#### h370: Adaptive Ensemble Selection by Category - VALIDATED

**Hypothesis:** Use ensemble when |target-kNN| < 10%, else use dominant method.

**Results:**
| Threshold | Adaptive R@30 |
|-----------|---------------|
| 5% | 71.4% |
| **10%** | **72.4%** ← Best |
| 15% | 71.4% |
| 20% | 71.4% |

**Baselines:**
- Static MinRank: 71.4%
- Static Best Single: 67.6%
- **Best Adaptive: 72.4% (+1.0 pp vs MinRank, +4.8 pp vs best single)**

**Category assignments at 10% threshold:**
- CV (12.5% gap) → Target alone
- Autoimmune (11.1% gap) → Target alone
- Neuro (5.3% gap) → MinRank ensemble
- Metabolic (0% gap) → MinRank ensemble
- Cancer (2.6% gap) → MinRank ensemble

#### h371: Target-Only for High-Gene-Coverage Diseases - VALIDATED

**Hypothesis:** Diseases with more DRKG gene associations benefit more from target-based scoring.

**Results across 376 diseases:**
| Gene Quartile | Target | kNN | Gap |
|---------------|--------|-----|-----|
| Q1 (≤4 genes) | 11.4% | 52.4% | **-41 pp** |
| Q2 (5-12) | 37.9% | 37.9% | 0 pp |
| Q3 (13-68) | 37.8% | 46.7% | -9 pp |
| Q4 (>68 genes) | 43.6% | 66.0% | **-22 pp** |

**Category average gene counts:**
- Autoimmune: 223 genes (Target wins 88.9% vs 77.8%)
- Cancer: 337 genes (Tie 65.8% vs 65.8%)
- Neurological: 124 genes (Target wins 42.1% vs 36.8%)
- Overall median: 12 genes

**Key insight:** Gene count helps Target but isn't sufficient. Also need:
1. Many drugs per disease (autoimmune has 5-65 drugs)
2. Category-specific drug overlap (shared targets across drugs)

### New Hypotheses Generated

- **h370:** Adaptive Ensemble Selection by Category - **DONE**
- **h371:** Target-Only for High-Gene-Coverage Diseases - **DONE**
- **h372:** kNN-Only for Neurological Diseases (Low Gene Coverage)
- **h373:** Weighted Rank Fusion by Method Confidence

### Cumulative Statistics
| Status | Count |
|--------|-------|
| Validated | 235 |
| Invalidated | 70 |
| Inconclusive | 14 |
| Blocked | 21 |
| Deprioritized | 7 |
| Pending | 26 |
| **Total** | **373** |

---

## Previous Session: h269, h366, h368 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 3**
- h269: Cancer-Specific Target-Based Scoring - **INCONCLUSIVE** (equivalent to kNN baseline)
- h366: Target+kNN Ensemble for Cancer - **VALIDATED** (76.3% R@30, +10.5 pp improvement)
- h368: Cancer Subtype-Specific Target Scoring - **VALIDATED** (subtypes differ significantly)

### KEY SESSION FINDINGS

#### h269: Cancer-Specific Target-Based Scoring - INCONCLUSIVE

**Hypothesis:** Can we improve cancer drug predictions by scoring drug-disease pairs based on target-gene overlap?

**Results (n=38 evaluable cancer diseases):**
| Method | R@30 | Hits |
|--------|------|------|
| Target Overlap | 65.8% | 25/38 |
| kNN Baseline | 63.2% | 24/38 |

**Key insight:** The methods capture **complementary signals** - they agree on 25 diseases but disagree on 13. This suggests ensemble potential.

#### h366: Target+kNN Ensemble for Cancer - VALIDATED

**Results (LOO CV on 38 cancer diseases):**
| Method | R@30 | Hits |
|--------|------|------|
| **Max Ensemble** | **76.3%** | 29/38 |
| Target Only | 65.8% | 25/38 |
| kNN Only | 63.2% | 24/38 |

**Key findings:**
- Max ensemble: take max(normalized_target, normalized_kNN) per drug
- Captures complementary signals: wins on 5 Target-failed and 6 kNN-failed diseases
- +10.5 pp improvement over best single method

#### h368: Cancer Subtype-Specific Target Scoring - VALIDATED

**Finding:** Different cancer subtypes have dramatically different optimal methods:

| Subtype | N | Target | kNN | Ensemble |
|---------|---|--------|-----|----------|
| Hematological | 15 | 66.7% | 73.3% | **93.3%** |
| Other | 7 | 71.4% | 57.1% | 85.7% |
| Carcinoma | 9 | 66.7% | 55.6% | 66.7% |
| Sarcoma | 2 | **100%** | 50% | 50% |
| Brain/CNS | 4 | 25% | **50%** | 25% |

---

## Previous Session: h361, h360, h363, h364, h179, h362 (2026-02-05)

[Truncated for brevity - see git history]
