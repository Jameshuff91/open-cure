# Research Loop Progress

## Current Session: h376 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 1**
- h376: Ensemble Coverage Analysis - **VALIDATED**

### h376: Ensemble Coverage Analysis - VALIDATED

**Hypothesis:** Diseases with moderate gene coverage and multiple drug classes benefit most from ensemble.

**Methodology:**
- Leave-one-out evaluation on 300 sampled diseases
- Compared kNN, Target, and MinRank ensemble methods
- Analyzed by gene count quartile, drug count, category, and method gap

**Key Results:**
| Metric | kNN | Target | Ensemble | Delta |
|--------|-----|--------|----------|-------|
| Overall R@30 | 61.0% | 25.0% | 59.0% | **-2.0 pp** |
| Diseases rescued | - | - | 0 | - |
| Diseases hurt | - | - | 21 | - |

**Categories that BENEFIT from ensemble:**
| Category | kNN | Ensemble | Delta |
|----------|-----|----------|-------|
| Metabolic | 66.7% | 75.0% | **+8.3 pp** |
| Autoimmune | 84.6% | 92.3% | **+7.7 pp** |
| Cancer | 69.4% | 71.4% | **+2.0 pp** |

**Categories HURT by ensemble:**
| Category | kNN | Ensemble | Delta |
|----------|-----|----------|-------|
| Cardiovascular | 71.4% | 57.1% | **-14.3 pp** |
| Immune | 62.5% | 50.0% | **-12.5 pp** |
| Neurological | 33.3% | 22.2% | **-11.1 pp** |
| Other | 56.3% | 50.5% | **-5.8 pp** |

**Gene Count Analysis:**
| Quartile | Genes | Ensemble Delta |
|----------|-------|----------------|
| Q1 (low) | 0-2 | -4.5 pp |
| Q2 | 2-8 | **+1.6 pp** |
| Q3 | 8-67 | -1.4 pp |
| Q4 (high) | >67 | -2.7 pp |

**Key Insight:** Ensemble only helps categories where Target and kNN have similar performance. When kNN dominates (CV, neuro, immune), adding Target noise hurts rankings.

**Implication:** Reinforces h369/h370/h374 - MinRank ensemble should be disabled globally or applied ONLY to autoimmune/metabolic/cancer.

### New Hypotheses Generated
- **h381:** Category-Specific Ensemble Routing (autoimmune/metabolic/cancer only) - Priority 3
- **h382:** Gene Count Q2 Ensemble Rule (2-8 genes only) - Priority 4
- **h383:** Cardiovascular Ensemble Harm Investigation - Priority 4

### Cumulative Statistics
| Status | Count |
|--------|-------|
| Validated | 237 |
| Invalidated | 71 |
| Inconclusive | 14 |
| Blocked | 21 |
| Deprioritized | 7 |
| Pending | 31 |
| **Total** | **381** |

---

## Previous Session: h374, h377 (2026-02-05)

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

**Evaluation Results (n=502 diseases):**
| Category | n | MinRank | kNN | Δ |
|----------|---|---------|-----|---|
| Cancer | 100 | 89.0% | 89.0% | **0%** |
| Metabolic | 40 | 75.0% | 77.5% | **-2.5%** |
| Neurological | 21 | 71.4% | 71.4% | **0%** |
| **Overall** | 502 | 83.7% | 83.9% | **-0.2%** |

**Root Cause:** Production predictor rules (h274 cancer_same_type etc.) already capture target overlap signal. MinRank adds redundant/harmful signal.

**Resolution:** Implementation kept but DISABLED (empty set for MINRANK_ENSEMBLE_CATEGORIES)

**Key Learning:** Ensemble methods validated in isolation may not help in production when existing rules capture similar signals.

---

## Previous Session: h369, h370, h371, h372 (2026-02-05)

[Truncated for brevity - see git history]
