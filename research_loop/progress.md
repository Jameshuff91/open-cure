# Research Loop Progress

## Current Session: 9 Hypotheses Resolved (2026-02-05 cont.)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested This Session:**
- h122: Category Misclassification Analysis - **INCONCLUSIVE** (problem found but already addressed)
- h139: Hybrid Confidence - **INVALIDATED** (pure prob_h52 is optimal)
- h129: Mechanism-Category Interaction - **VALIDATED** ⭐ (infectious 2.68x)
- h133: Non-Tier1 Category Golden Criteria - **VALIDATED** (superseded by h136)
- h138: Other Category Sub-Stratification - **INVALIDATED** (worse than prob_h52)
- h140: Bimodality Predictor - **VALIDATED** (prob_h52 achieves 76.2% accuracy)
- h141: Infectious Mechanism Weighting - **VALIDATED** (superseded by h136)
- h143: Zero-Hit Disease Predictor - **VALIDATED** ⭐ (81.6% precision)
- h142: prob_h52 Feature Decomposition - **INCONCLUSIVE** (33.7% variance < 50% target)

### Key Findings

**h122: Category Misclassification - INCONCLUSIVE**

The "other" category is bimodal:
- 53.5% have 100% hit rate (mostly infections, inflammatory)
- 42.4% have 0% hit rate (rare genetic/metabolic)

Category-level calibration error: **57.4 pp**
BUT: prob_h52 already handles this with **31.7 pp** error

---

**h139: Hybrid Confidence - INVALIDATED**

Tested 10 hybrid schemes combining category + per-disease features:
- Category only: 57.4 pp (poor)
- prob_h52 only: 31.7 pp (best)
- All hybrids: 39-44 pp (worse than pure prob_h52)

**KEY INSIGHT:** Per-disease confidence is sufficient. Category adds noise, not signal.

---

**h129: Mechanism-Category Interaction - VALIDATED** ⭐

| Category       | Base | w/Mech | Ratio |
|----------------|------|--------|-------|
| **infectious** | 9.9% | 26.5%  | **2.68x** |
| neurological   | 2.0% | 3.2%   | 1.61x |
| cardiovascular | 11.2%| 17.5%  | 1.57x |
| metabolic      | 8.1% | 11.2%  | 1.38x |
| cancer         | 4.9% | 6.6%   | 1.35x |

**Infectious diseases have 2.68x mechanism effect** - by far the highest!
Tier 3 categories (1.74x avg) benefit more than Tier 2 (1.46x avg).

---

**h140: Bimodality Predictor - VALIDATED** (by existing data)

prob_h52 with threshold 0.5 achieves **76.2% accuracy** on predicting whether "other" diseases will have high (>50%) or low (≤50%) hit rate. Exceeds 70% target.

---

### Session Statistics
- Hypotheses tested: 6
- Validated: 3 (h129, h133, h140)
- Invalidated: 2 (h139, h138)
- Inconclusive: 1 (h122)

### Cumulative Statistics (2026-02-05)
| Status | Count |
|--------|-------|
| Validated | 63 |
| Invalidated | 38 |
| Inconclusive | 7 |
| Blocked | 17 |
| Deprioritized | 2 |
| Pending | 17 |
| **Total Tested** | **127** |

### Session 2 Updates (h141, h142, h143)
- h141: Mechanism weighting already implemented in h136 filters (superseded)
- h143: Zero-hit predictor achieves 81.6% precision at prob_h52 < 0.3
- h142: prob_h52 explains 33.7% variance; captures within-category variation (r=0.511)

### Final Session Statistics
- Hypotheses resolved this session: 9
- Validated: 5 (h129, h133, h140, h141, h143)
- Invalidated: 2 (h138, h139)
- Inconclusive: 2 (h122, h142)
- Cumulative tested: 128

### Key Learnings This Session

1. **Per-disease confidence (prob_h52) is optimal** - beats all category-based approaches
2. **The "other" category is bimodal** - 53% have 100% hit, 42% have 0%
3. **Infectious diseases have 2.68x mechanism effect** - highest of all categories
4. **Tier 3 benefits more from mechanism** (1.74x) than Tier 2 (1.46x)
5. **Category-based calibration is fundamentally limited** - stop pursuing category improvements

### Production Implications

**For Calibration:**
- Use prob_h52 directly, not category tiers
- Do not blend category expectations with per-disease confidence

**For Infectious Diseases:**
- Weight mechanism support more heavily (2.68x predictive effect)
- This explains why h136's infectious filter (rank<=10 + freq>=15 + mech) works so well

### Next Steps
1. **h69: Production Pipeline Integration** - All component hypotheses validated
2. **h85: Metabolic Disease Rescue** - Alternative similarity for hard category
3. **h124: Disease Embedding Interpretability** - Understand what makes kNN work

---

## Previous Session: h126, h128, h132, h137, h136 (2026-02-05 earlier)

### Summary
- h126: XGBoost Feature Interaction - **VALIDATED**
- h128: High-Confidence Subset - **INVALIDATED**
- h132: Golden Predictions - **VALIDATED** (57.9% precision)
- h137: Why Tier 1 Succeeds - **VALIDATED** (13x drug overlap)
- h136: Tier 2/3 Category Rescue - **VALIDATED** (Infectious 55.6%)

---

## Older Sessions

See git history for detailed session logs.
