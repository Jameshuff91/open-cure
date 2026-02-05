# Research Loop Progress

## Current Session: h111 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** In Progress
**Hypotheses Tested This Session:**
- h111: Confidence Feature Independence Analysis - **VALIDATED**

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

### Session Statistics
- Hypotheses tested: 2 (h111, h114)
- Validated: 2
- New hypotheses added: 4 (h114, h115, h116, h117)

### Next Steps
1. **h117**: Add target breadth as confidence feature
2. **h115**: Test simplified ensemble (remove redundant kNN score/rank)
3. **h112**: Cross-class drug discovery analysis

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
