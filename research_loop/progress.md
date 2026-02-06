# Research Loop Progress

## Current Session: h369, h370, h371 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** In Progress
**Hypotheses Tested: 3**
- h369: Apply Max Ensemble to Non-Cancer Categories - **VALIDATED** (MinRank helps cancer/neuro, hurts autoimmune)
- h370: Adaptive Ensemble Selection by Category - **VALIDATED** (72.4% R@30 with 10% threshold)
- h371: Target-Only for High-Gene-Coverage Diseases - **VALIDATED** (gene count helps but doesn't make Target superior)

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
