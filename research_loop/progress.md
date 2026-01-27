# Research Loop Progress

## Current Session: h3 Evaluation (2026-01-26)

### Session Summary

**Agent Role:** Research Executor
**Status:** Completed
**Hypothesis Tested:** h3 (Infectious Disease Specialist Model)
**Outcome:** INVALIDATED

### Experiment Details

**Objective:** Test if a specialist XGBoost model trained only on infectious disease pairs can outperform the general GB model.

**Reported Baseline:** 13.6% R@30 for infectious diseases (from CLAUDE.md)
**Actual Baseline:** 52.0% R@30 on 47 mappable infectious diseases

### Critical Finding: Baseline Discrepancy

The 13.6% figure in CLAUDE.md was based on **antibiotic CLASS performance** (e.g., fluoroquinolones 0%, macrolides 6%), not disease-level evaluation.

**Actual General Model Performance:**
- 52.0% R@30 on infectious diseases (104/200 hits)
- 47 diseases evaluated with proper EC-to-DRKG mapping
- Best performers: E. coli infections 100%, Herpes zoster 100%
- Worst performers: Diabetic foot infections 0%, Cutaneous candidiasis 0%

### Specialist Model Results

| Model | R@30 | Test Diseases | Notes |
|-------|------|---------------|-------|
| General GB | 63.6% | 12 | Baseline |
| Specialist | 36.4% | 12 | **Underperforms by 27.3%** |

**Root Cause of Specialist Underperformance:**
1. Insufficient training data: 294 positive pairs vs ~3000 for general model
2. Disease-level split left only 12 test diseases
3. General model's broader training data provides better feature learning

### Key Insights

1. **The "infectious disease problem" was mischaracterized.** The real issue is antibiotics being predicted for NON-infectious diseases (spurious predictions), not poor recall ON infectious diseases.

2. **General model already performs well (52% R@30)** on infectious diseases when evaluated properly.

3. **Specialist approach is unnecessary.** The confidence_filter.py already handles spurious antibiotic predictions.

### Hypotheses Updated

| ID | Title | Status | Change |
|----|-------|--------|--------|
| h3 | Infectious Disease Specialist | **invalidated** | General model outperforms |
| h26 | Antibiotic Prediction Filtering Analysis | **added** | Low priority |
| h27 | Per-Category Baseline Documentation | **added** | Verify other categories |

### Files Created/Modified

| File | Action |
|------|--------|
| `scripts/evaluate_infectious_specialist.py` | Created |
| `data/analysis/h3_infectious_specialist_results.json` | Created |
| `data/analysis/infectious_baseline_evaluation.json` | Created |
| `data/analysis/infectious_antimicrobial_analysis.json` | Created |
| `research_roadmap.json` | Updated |

### Recommended Next Hypothesis

**h4: Expand Ground Truth with DrugBank/ChEMBL Indications** (Priority 4)

**Why:**
- Medium impact, medium effort
- Addresses data sparsity root cause
- Can directly increase training examples
- Validation found 4 FDA-approved drugs missing from GT

**Alternative:** h5 (Hard Negative Mining) - addresses model discrimination quality

---

## Previous Session: h1 Evaluation (2026-01-26)

### Session Summary

**Agent Role:** Research Executor
**Status:** Completed
**Hypothesis Tested:** h1 (GB + TxGNN Best-Rank Ensemble)
**Outcome:** INVALIDATED

### Experiment Details

**Objective:** Test if taking min(GB_rank, TxGNN_rank) improves R@30 beyond either model alone.

**Method:**
1. Loaded TxGNN predictions from `data/reference/txgnn_predictions_final.csv`
2. Loaded GB model and generated predictions for all drug-disease pairs
3. Computed ensemble ranks as min(GB_rank, TxGNN_rank)
4. Evaluated on 603 diseases with MESH mappings

**Results:**

| Model | R@30 | Diseases | Notes |
|-------|------|----------|-------|
| GB model | 42.04% | 567 | Baseline |
| TxGNN | 0.00% | 602 | **CRITICAL: Zero hits** |
| Ensemble | 42.03% | 566 | No improvement |

### Critical Finding

**TxGNN predictions file limitation:** The pre-computed `txgnn_predictions_final.csv` only contains the **top 50 drugs per disease**. Ground truth drugs almost never appear in TxGNN's top-50:

| Disease | GT Drugs | In TxGNN Top-50 |
|---------|----------|-----------------|
| Psoriasis | 4 | 0 |
| Hypertension | 3 | 0 |
| Breast cancer | 4 | 0 |
| Rheumatoid arthritis | 3 | 0 |
| All tested | 27 | 0 (0%) |

**Root Cause:** TxGNN ranks most GT drugs in the 1000s-7000s range (out of 7954 drugs total). Only ~14.5% of GT drugs rank within top-30 (per archive). The pre-computed file misses these because it only stores top-50.

**Implication:** TxGNN ensembles require **live GPU inference** to rank all drugs, not pre-computed files.

### Hypotheses Updated

| ID | Title | Status | Change |
|----|-------|--------|--------|
| h1 | GB + TxGNN Best-Rank Ensemble | **invalidated** | Blocked by data format |
| h2 | Category-Routed Ensemble | **blocked** | Same blocker |
| h23 | TxGNN Full Ranking Storage | **added** | Requires GPU |
| h24 | GB Error Analysis by Drug Class | **added** | Next priority |
| h25 | Embedding Distance Calibration | **added** | Alternative approach |

### Current Baseline Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| **R@30** | **42.04%** | Per-drug Recall@30 (slightly higher than 41.8% in CLAUDE.md due to evaluation subset) |
| Diseases | 567 | With GB embeddings and MESH mapping |
| Pairs | 1,125 | Drug-disease ground truth pairs |
| Model | GB + Fuzzy Matcher | Gradient Boosting with fuzzy disease name matching |

### Key Learning

> TxGNN pre-computed predictions contain only top-50 drugs per disease. GT drugs are NOT in top-50 for most diseases (0% coverage). TxGNN ensembles require live GPU inference, not pre-computed files.

### Files Created/Modified

| File | Action |
|------|--------|
| `scripts/evaluate_best_rank_ensemble.py` | Created |
| `data/analysis/h1_ensemble_results.json` | Created |
| `research_roadmap.json` | Updated (h1 invalidated, h2 blocked, 3 new hypotheses) |
| `research_loop/progress.md` | Updated (this file) |

### Recommended Next Hypothesis

**h3: Infectious Disease Specialist Model** (Priority 3)

**Why:**
- Does not require GPU or TxGNN
- High expected impact
- Addresses known model weakness (13.6% recall for infectious diseases)
- Can be implemented with existing data

**Alternative:** h24 (GB Error Analysis by Drug Class) - low effort, provides insights for future work.

---

## Previous Session: Initialization (2026-01-26)

### Session Summary

**Agent Role:** Research Initializer
**Status:** Completed
**Output:** `research_roadmap.json` with 22 prioritized hypotheses

### Current Baseline Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| **R@30** | **41.8%** | Per-drug Recall@30 |
| Diseases | 1,236 | 30.9% of Every Cure diseases mapped |
| Pairs | 3,618 | Drug-disease ground truth pairs |
| Model | GB + Fuzzy Matcher | Gradient Boosting with fuzzy disease name matching |
| Validation Precision | 22.5% | Top predictions (batches 1+2) |

### Key Findings from Analysis

1. **TxGNN Opportunity:** Achieves 83.3% R@30 on storage diseases vs 6.7% overall. Category-specific routing is promising.

2. **Biologic Gap:** mAbs have only 2.1 diseases/drug vs 11.1 for small molecules. Data sparsity is root cause.

3. **Infectious Disease Paradox:** More training data correlates with WORSE performance for antibiotics. Model learns wrong patterns.

4. **Circular Feature Warning:** Quad Boost features (target overlap, ATC, chemical similarity) were found to be circular and inflate metrics when evaluated on training diseases.

5. **Validation Reality:** Only 22.5% of top predictions validate against literature. Strong need for better filtering.

---

*Last updated: 2026-01-26*
*Agent: Research Executor*
*Hypothesis tested: h1 (invalidated)*
