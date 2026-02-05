# Research Loop Progress

## Current Session: h115, h116, h99 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested This Session:**
- h115: Ensemble Simplification - **VALIDATED**
- h116: Category Tier 2.0: Disease-Specific Calibration - **INVALIDATED**
- h99: Phenotype-Based Drug Transfer - **INCONCLUSIVE**

### Key Findings

**h115: Ensemble Simplification - VALIDATED**

Tested whether removing redundant features from the 5-feature confidence ensemble preserves precision.

**EXPERIMENT RESULTS (13,522 predictions, 5 seeds):**

| Ensemble              | Features | Top 10% | Top 20% | Top 33% |
|-----------------------|----------|---------|---------|---------|
| Full (5 features)     | 5        | 22.04%  | 18.31%  | 13.80%  |
| Without norm_score    | 4        | 21.82%  | 18.38%  | 13.67%  |
| **Without inv_rank**  | 4        | **22.04%** | 17.97%  | 13.51%  |

**KEY FINDINGS:**
1. Removing inv_rank: IDENTICAL precision at top 10% (22.04%)
2. 4-feature model is sufficient: [mechanism_support, train_frequency, tier_inv, norm_score]
3. inv_rank and norm_score are redundant (r=0.665 from h111)
4. Simpler model is more interpretable for production

---

**h116: Category Tier 2.0 - INVALIDATED**

Tested whether disease-specific calibration (using category-level training hit rates) improves on coarse tier system.

**CORRELATION WITH HITS:**
| Signal               | r     |
|----------------------|-------|
| Category Tier (inv)  | 0.082 |
| Calibrated Cat Rate  | 0.072 |

**KEY FINDINGS:**
1. Calibrated rate is WORSE than coarse tier (0.072 < 0.082)
2. Calibration errors are large (20-40% for most categories)
3. Training hit rates don't transfer well to test diseases within same category
4. Coarse tier system captures the broad pattern as well as fine-grained calibration

---

**h99: Phenotype-Based Drug Transfer - INCONCLUSIVE**

Tested symptom/phenotype similarity as alternative to Node2Vec for kNN.

**DATA COVERAGE:**
- GT diseases with symptom data: 58/472 (12.3%)
- Test predictions: 43 total (across 5 seeds)

**RESULTS (on diseases WITH symptoms):**
| Method              | R@30   |
|---------------------|--------|
| Symptom-based kNN   | 21.34% |
| Node2Vec kNN        | 42.44% |

**KEY FINDINGS:**
1. Symptom-based kNN underperforms Node2Vec by 21 pp
2. Only 12.3% of diseases have symptom data (cannot scale)
3. Diseases WITH symptoms are easier to predict (42% vs ~26% overall)
4. Symptom similarity doesn't capture drug treatment patterns well

---

### Session Statistics
- Hypotheses tested: 3 (h115, h116, h99)
- Validated: 1 (h115)
- Invalidated: 1 (h116)
- Inconclusive: 1 (h99)
- New hypotheses added: 5 (h121-h125)

### Cumulative Statistics (2026-02-05)
| Status | Count |
|--------|-------|
| Validated | 49 |
| Invalidated | 32 |
| Inconclusive | 6 |
| Blocked | 15 |
| Pending | 19 |
| **Total Tested** | **87** |

### Pending Hypotheses: 19

### Next Steps
1. **h123: Negative Confidence Signal** - What predicts MISSES? (priority 2)
2. **h125: Drug-Level Success Prediction** - Drug reliability score (priority 2)
3. **h121: Minimal 3-Feature Ensemble** - Further simplification (priority 3)

---

## Previous Session: h111, h114, h117, h112 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested:**
- h111: Confidence Feature Independence Analysis - **VALIDATED**
- h114: Drug Frequency Mechanism Analysis - **VALIDATED**
- h117: Target Breadth as Confidence Feature - **VALIDATED**
- h112: Cross-Class Drug Discovery - **INCONCLUSIVE**

### Key Findings

**h111 VALIDATED:** Confidence signals are mostly independent (7/10 pairs have |r| < 0.3)
- Mechanism Support + Drug Frequency: 20.04% precision (orthogonal, r=0.071)
- Drug Frequency: r=0.187 (strongest predictor)

**h114 VALIDATED:** Drug frequency works via polypharmacology + disease centrality
- High-freq drugs have 2.3x MORE targets (48.9 vs 21.5)
- High-freq drugs treat more central diseases

**h117 VALIDATED:** Target breadth is independent confidence feature (+5.58 pp)
- HIGH targets (≥31): 10.41% precision
- LOW targets (≤8): 4.83% precision

---

## Previous Session: h104, h110, h106, h113 (2026-02-04)

### Key Findings

**h106 VALIDATED:** Multi-signal ensemble achieves 22.56% precision @ top 10%
**h108 VALIDATED:** Drug training frequency = strongest confidence signal (+9.4 pp)
**h113 VALIDATED:** Fixed mechanism support data loading (+0.52 pp)
**h104, h110 INVALIDATED:** ATC coherence is weak/inverted signal
