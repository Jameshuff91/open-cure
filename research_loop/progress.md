# Research Loop Progress

## Current Session: h1 Evaluation (2026-01-26)

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
