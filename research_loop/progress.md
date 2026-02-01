# Research Loop Progress

## Current Session: h68, h72 (2026-01-31)

### Session Summary

**Agent Role:** Research Executor
**Status:** Completed (2 hypotheses tested)
**Hypotheses Tested:**
- h68: Unified Confidence-Weighted Predictions - **VALIDATED**
- h72: Production Deliverable with Confidence Tiers - **VALIDATED**

### Key Findings

**h68: Unified Confidence Scoring EXCEEDS TARGET (88% precision at 0.7)**

Combined three confidence signals:
1. h65 disease success predictor (RF)
2. h52 meta-confidence model (XGBoost)
3. Category-based priors (h58/h59)

Multi-seed Results (5 seeds):
| Signal | AUC | AP | Precision@0.7 | Coverage |
|--------|-----|-----|---------------|----------|
| h65 (success predictor) | 0.698 | 0.771 | 81.5% | 19.8 |
| h52 (meta-confidence) | 0.816 | 0.823 | 82.6% | 41.0 |
| Category prior | 0.593 | 0.661 | 70.0% | 23.4 |
| **Combined avg** | **0.826** | **0.856** | **88.4%** | 26.0 |

**Key Insight:** Simple average achieves 88% precision (exceeds 75% target). h52 alone achieves 82.6% with 2x coverage - may be simpler for production.

**h72: Production Deliverable Generated**

Output: `data/deliverables/drug_repurposing_predictions_with_confidence.xlsx`

| Tier | Diseases | Predictions | Novel Predictions |
|------|----------|-------------|-------------------|
| HIGH | 110 (24.6%) | 3,288 | 2,797 |
| MEDIUM | 236 (52.7%) | 7,078 | 6,569 |
| LOW | 102 (22.8%) | 3,050 | 3,004 |

**Validation of Top Predictions:**
- Sirolimus → Tuberous Sclerosis Complex: FDA-APPROVED (2022)
- Lovastatin → Atherosclerosis: MARS & AFCAPS trials validated
- Adalimumab → SLE: Complex (needs careful review)

### New Hypotheses Generated

| Priority | ID | Title |
|----------|-----|-------|
| 1 | h72 | Production Deliverable (completed) |
| 2 | h70 | Threshold Optimization by Use Case |
| 2 | h73 | h52 Model Simplification |
| 3 | h71 | Per-Category Calibration |

### Updated Pending Hypotheses

| Priority | ID | Title | Effort |
|----------|-----|-------|--------|
| 1 | h69 | Production Pipeline Integration | high |
| 2 | h66 | Category-Specific k Values | low |
| 2 | h67 | Drug Class Boosting | medium |
| 2 | h70 | Threshold Optimization | low |
| 2 | h73 | h52 Simplification Analysis | low |
| 3 | h55 | GEO Gene Expression | high |
| 3 | h71 | Per-Category Calibration | medium |
| 4 | h64 | ARCHS4 Real Expression | high |

### Session Statistics

- Hypotheses tested: 2
- Validated: 2 (h68, h72)
- Invalidated: 0
- New hypotheses: 4 (h70-h73)
- Deliverables: 1 (Excel + JSON)

### Recommended Next Steps

1. **h73**: Analyze whether h52-only is sufficient (simpler deployment)
2. **h70**: Optimize thresholds for different use cases
3. **h69**: Full production pipeline integration

---

## Previous Sessions

### h61, h57, h65, h62, h63 (2026-01-31)
- 5 hypotheses tested, 2 validated
- Gene-based approaches consistently fail vs Node2Vec
- h65 success predictor achieves 70% precision

### h49-h59 (2026-01-31)
- 9 hypotheses tested, 8 validated
- Major finding: GI diseases are critical blind spot (5% hit rate)
- Extended categories identified clear performance tiers
