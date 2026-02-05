# Research Loop Progress

## Current Session: h115, h118 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** In Progress
**Hypotheses Tested This Session:**
- h115: Ensemble Simplification - **VALIDATED**
- h118: Minimal 2-Feature Confidence Score - **INVALIDATED**

### Key Findings

**h115: Ensemble Simplification - VALIDATED**

Tested whether removing redundant features from the 5-feature confidence ensemble preserves precision.

**EXPERIMENT RESULTS (13,522 predictions, 5 seeds):**

| Ensemble              | Features | Top 10% | Top 20% |
|-----------------------|----------|---------|---------|
| Full (5 features)     | 5        | 22.12%  | 17.90%  |
| Without norm_score    | 4        | 21.23%  | 18.23%  |
| **Without inv_rank**  | 4        | **22.04%** | 17.38%  |

**KEY FINDINGS:**
1. Removing inv_rank loses only -0.07 pp at top 10% (within noise)
2. 4-feature model is sufficient: [mechanism_support, train_frequency, tier_inv, norm_score]
3. inv_rank and norm_score are redundant (r=0.665 from h111)
4. Simpler model is more interpretable for production

**RECOMMENDED SIMPLIFIED ENSEMBLE:**
- Features: mechanism_support, train_frequency, tier_inv, norm_score
- Precision: 22.04% at top 10%

---

**h118: Minimal 2-Feature Confidence Score - INVALIDATED**

Tested if mechanism_support + train_frequency alone could match 4-feature ensemble.

| Model                    | Features | Top 10% | Top 20% |
|--------------------------|----------|---------|---------|
| 4-feature (h115 best)    | 4        | 21.89%  | 17.83%  |
| 2-feature (mech + freq)  | 2        | 19.38%  | 16.09%  |
| Frequency only           | 1        | 18.79%  | 15.90%  |
| Mechanism only           | 1        | 10.36%  | 12.24%  |

**KEY FINDINGS:**
1. 2-feature model loses -2.51 pp vs 4-feature (exceeds 2 pp tolerance)
2. Frequency alone (18.79%) is nearly as good as 2-feature
3. Mechanism alone (10.36%) performs poorly as standalone
4. tier_inv and norm_score contribute ~2.5 pp to the ensemble

**CONCLUSION:** 4-feature model remains the recommended production model.

### New Hypotheses Added
- h118: Minimal 2-Feature Confidence Score
- h119: Non-Linear Feature Interactions for Confidence
- h120: 3-Feature Confidence Model (Remove Mechanism)

### Session Statistics
- Hypotheses tested: 2 (h115, h118)
- Validated: 1
- Invalidated: 1
- New hypotheses added: 3

### Cumulative Statistics (2026-02-05)
| Status | Count |
|--------|-------|
| Validated | 49 |
| Invalidated | 32 |
| Inconclusive | 5 |
| Blocked | 15 |
| Pending | 17 |
| **Total Tested** | **86** |

### Next Steps
1. **h119**: Test non-linear feature interactions
2. **h120**: Test 3-feature model (remove mechanism)
3. **h116**: Per-disease calibration for category tier

---

## Previous Session: h111, h114, h117, h112 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested This Session:**
- h111: Confidence Feature Independence Analysis - **VALIDATED**
- h114: Drug Frequency Mechanism Analysis - **VALIDATED**
- h117: Target Breadth as Confidence Feature - **VALIDATED**
- h112: Cross-Class Drug Discovery - **INCONCLUSIVE** (missing ATC data)

### Key Findings

**h111: Confidence Feature Independence Analysis - VALIDATED**

Analyzed correlation structure between 5 confidence signals across 13,522 predictions:

**CORRELATION ANALYSIS:**

| Signal Pair | Pearson r | Independence |
|-------------|-----------|--------------|
| Mechanism ↔ Drug Freq | 0.071 | **INDEPENDENT** |
| Mechanism ↔ Category Tier | -0.005 | **INDEPENDENT** |
| Mechanism ↔ kNN Score | 0.118 | **INDEPENDENT** |
| Drug Freq ↔ Category Tier | 0.031 | **INDEPENDENT** |
| Category Tier ↔ kNN Score | -0.042 | **INDEPENDENT** |
| Drug Freq ↔ kNN Score | 0.404 | Correlated |
| kNN Score ↔ Inv Rank | 0.665 | Correlated |

**7 out of 10 pairs are independent (|r| < 0.3)!**

**HIT PREDICTION POWER (point-biserial correlation):**
1. Drug Frequency (h108): r = 0.187 ⭐ **Strongest**
2. Inverse Rank: r = 0.160
3. kNN Score: r = 0.156
4. Mechanism Support (h97): r = 0.098
5. Category Tier (h71): r = 0.082 ⭐ **Weakest**

**BEST COMBINATION:**
- Mechanism Support + Drug Frequency: **20.04% precision** on 1,013 predictions
- These signals are truly orthogonal (r = 0.071)
- Nearly matches h106 ensemble top 10% (22%)

**KEY INSIGHT:**
Mechanism support captures **mechanistic plausibility** (drug targets disease genes), while drug frequency captures **empirical reliability** (drugs that work for many diseases). These are ORTHOGONAL signals — combining them provides additive gain.

### Session Statistics
- Hypotheses tested: 1 (h111)
- Validated: 1
- New hypotheses added: 3 (h114, h115, h116)

---

## h114: Drug Frequency Mechanism (2026-02-05, continued)

### Key Findings

**h114: Drug Frequency Mechanism - VALIDATED**

Investigated WHY drug training frequency (h108) is the strongest predictor of hits.

**CONFIRMED MECHANISMS (p < 0.001):**

| Hypothesis | Finding | Evidence |
|------------|---------|----------|
| **POLYPHARMACOLOGY** | High-freq drugs have 2.3x MORE targets | 48.9 vs 21.5 targets, p = 2e-14 |
| **DISEASE CENTRALITY** | High-freq drugs treat more central diseases | 0.367 vs 0.325, p = 2e-15 |
| Drug embedding centrality | NOT a factor | ρ = -0.12 (negative!) |

**DIRECT CORRELATIONS WITH FREQUENCY:**
- Number of targets: ρ = 0.24* (strongest)
- Disease centrality: ρ = 0.10*
- Drug embedding centrality: ρ = -0.12* (negative!)

**TOP HIGH-FREQUENCY DRUGS:**
Corticosteroids dominate (Dexamethasone, Prednisolone, Prednisone) due to broad anti-inflammatory mechanisms.

**INTERPRETATION:**
Drugs generalize better when they have:
1. More targets → more chances to match disease pathways
2. Treat central diseases → more similar diseases in test set

---

## h117: Target Breadth Confidence Feature (2026-02-05, continued)

### Key Findings

**h117: Target Breadth as Confidence Feature - VALIDATED**

| Target Level | Precision | Count |
|--------------|-----------|-------|
| HIGH (≥31) | 10.41% | 4,314 |
| MEDIUM | 7.10% | 3,999 |
| LOW (≤8) | 4.83% | 4,725 |
| NO targets | 1.24% | 484 |

**Difference: +5.58 pp (exceeds 5 pp threshold)**

**Independence from Frequency:**
- Correlation with train_frequency: ρ = 0.27 (independent, |r| < 0.3)
- Effect persists when controlling for frequency

---

## h112: Cross-Class Drug Discovery (2026-02-05, continued)

**h112: Cross-Class Drug Discovery - INCONCLUSIVE**

Attempted to analyze why incoherent predictions (drugs from ATC classes that never treat similar diseases) have HIGHER precision than coherent ones (11.24% vs 6.69%).

**BLOCKED:** Missing drug_atc_codes.json file. ATC mappings require extraction from unified_edges_clean.csv.

**PRELIMINARY HYPOTHESES (untested):**
1. Selection pressure: Incoherent hits need stronger kNN signals
2. Polypharmacology: Incoherent hits may have more targets
3. Confounding: Coherent predictions capture obvious/known associations

### Session Statistics
- Hypotheses tested: 4 (h111, h114, h117, h112)
- Validated: 3
- Inconclusive: 1
- New hypotheses added: 4 (h114, h115, h116, h117)

### Cumulative Statistics (2026-02-05)
| Status | Count |
|--------|-------|
| Validated | 48 |
| Invalidated | 31 |
| Inconclusive | 5 |
| Blocked | 15 |
| Pending | 16 |
| **Total Tested** | **84** |

### Next Steps
1. **h115**: Test simplified ensemble (remove redundant kNN score/rank)
2. **h116**: Per-disease calibration for category tier
3. **h69**: Production Pipeline Integration

---

## Previous Session: h104, h110, h106, h113 (2026-02-04, continued)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested This Session:**
- h104: Confidence Feature - Drug Class Coherence - **INVALIDATED** (+1.3 pp < 5 pp threshold)
- h110: ATC Incoherence as Negative Signal - **INVALIDATED** (counter-intuitive!)
- h106: Multi-Signal Confidence Ensemble - **VALIDATED** (22.56% precision @ top 10%)
- h113: Fix Mechanism Support Data Loading - **VALIDATED** (+0.52 pp ensemble improvement)

### Key Findings

**h104: Drug Class Coherence - INVALIDATED**
- HIGH coherence: 8.94% precision
- LOW coherence: 7.77% precision
- Difference: +1.17 pp (below 5 pp threshold)
- Class membership too coarse to predict precision

**h107: Rank Stability - INVALIDATED**
- STABLE predictions (low CV): 6.56% precision
- UNSTABLE predictions (high CV): 6.62% precision
- Difference: -0.06 pp (no signal)
- Rank stability doesn't predict precision beyond rank itself

**h108: Drug Training Frequency - VALIDATED (+9.4 pp)**
- HIGH frequency drugs: 12.87% precision
- LOW frequency drugs: 3.46% precision
- Difference: +9.40 pp (3.7x improvement!)
- Drugs with more training indications generalize better
- **STRONGEST confidence signal found**

**h106: Multi-Signal Ensemble - VALIDATED**
- Top 10%: 21.75% precision (exceeds 15% target)
- Top 20%: 16.90% precision (exceeds 15% target)
- Top 33%: 13.42% precision
- Feature importance: train_frequency > tier_inv > norm_score ≈ inv_rank
- Ensemble provides 45% more high-confidence predictions than Tier 1 alone

### Confidence Feature Summary

| Signal | Precision Diff | Status |
|--------|---------------|--------|
| h108 Drug frequency | +9.40 pp | **VALIDATED** (strongest) |
| h97 Mechanism support | +6.48 pp | **VALIDATED** |
| h71 Category tier | varies | **VALIDATED** |
| h106 Ensemble | 21.75% @ top 10% | **VALIDATED** |
| h104 ATC coherence | +1.17 pp | INVALIDATED |
| h107 Rank stability | -0.06 pp | INVALIDATED |
| h105 Coverage strength | -0.45 pp | INVALIDATED |
| h110 ATC incoherence | -4.55 pp | INVALIDATED (inverted!) |

**h110: ATC Incoherence - COUNTER-INTUITIVE RESULT!**
- INCOHERENT (no classmate treats similar): 11.24% precision
- COHERENT (classmate treats similar): 6.69% precision
- **Incoherent predictions perform BETTER** (opposite of hypothesis)
- Interpretation: Drugs from "irrelevant" ATC classes that rank highly must have strong kNN signal from independent sources

**h113: Fix Mechanism Support Data Loading - VALIDATED**
- Fixed disease ID format mismatch in h106_multi_signal_ensemble.py
- Mechanism support now working: 2,718 predictions (20.1%) have support
- WITH support: 12.10% precision vs WITHOUT: 5.96% (+6.14 pp)
- Top 10% ensemble: 22.04% → 22.56% (+0.52 pp)
- Top 20% ensemble: 17.42% → 18.23% (+0.81 pp)

### Session Statistics
- Hypotheses tested: 4 (h104, h110, h106, h113)
- Validated: 2 (h106, h113)
- Invalidated: 2 (h104, h110)
- New hypotheses added: 4 (h110, h111, h112, h113)

### Cumulative Statistics (2026-02-04)
| Status | Count |
|--------|-------|
| Validated | 44 |
| Invalidated | 31 |
| Inconclusive | 4 |
| Blocked | 15 |
| Pending | 16 |
| **Total Tested** | **79** |

### Pending Hypotheses: 16

---

## Previous Session: h93, h95, h97, h105 (2026-02-04)

### Session Summary

**Hypotheses Tested:**
- h93: Direct Mechanism Traversal - **INVALIDATED** (3.53% R@30)
- h95: Pathway-Level Traversal - **INVALIDATED** (3.57% R@30)
- h97: Mechanism-kNN Hybrid Confidence - **VALIDATED** (+6.5 pp)
- h105: Disease Coverage Strength - **INVALIDATED** (predicts recall, not precision)

### Key Takeaway

**Learned representations >> explicit graph traversal for drug repurposing**

The 26% kNN vs 3.5% traversal gap quantifies the value of embeddings.

---

## Cumulative Statistics

| Status | Count |
|--------|-------|
| Validated | 41 |
| Invalidated | 29 |
| Inconclusive | 4 |
| Blocked | 14 |
| Pending | 17 |
| In Progress | 0 |
| **Total Tested** | **74** |

---

## Recommended Next Steps

1. **h109: Chemical Fingerprint Similarity** - Test if structural similarity to known treatments predicts precision
2. **h111: Confidence Feature Independence** - Check if signals (frequency, tier, mechanism) are correlated or orthogonal
3. **h91: Literature Mining** - Extract drug-disease hypotheses from PubMed for zero-shot diseases
