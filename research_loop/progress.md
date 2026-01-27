# Research Loop Progress

## Current Session: Initialization (2026-01-26)

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

### Research Roadmap Created

**Total Hypotheses:** 22
**Categories:**
- Ensemble: 3 hypotheses (h1, h2, h20)
- Feature: 8 hypotheses (h6, h7, h8, h11, h14, h16, h17, h19, h21)
- Data: 4 hypotheses (h4, h5, h9, h13, h18)
- Evaluation: 3 hypotheses (h10, h22)
- Architecture: 4 hypotheses (h3, h12, h15)

### Recommended First Hypothesis

**h1: GB + TxGNN Best-Rank Ensemble**

**Why Start Here:**
- Low effort (predictions already exist)
- Medium expected impact
- No retraining required
- Quick validation of ensemble approach

**Steps:**
1. Load TxGNN predictions from `data/reference/txgnn_predictions.csv`
2. Compute `min(GB_rank, TxGNN_rank)` for each drug-disease pair
3. Evaluate R@30 on held-out disease set
4. Success criteria: >43% R@30

**Files Needed:**
- `models/drug_repurposing_gb_enhanced.pkl`
- `data/reference/txgnn_predictions.csv`
- `scripts/evaluate_pathway_boost.py` (for evaluation framework)

### Next Agent Instructions

The next research agent should:

1. **Pick up h1** (or h2 if h1 is blocked)
2. **Load existing predictions** - don't retrain anything yet
3. **Use disease-level split** for fair evaluation (see archive/experiment_history.md)
4. **Update research_roadmap.json** with findings
5. **Commit results** before context fills

### Files Modified This Session

| File | Action |
|------|--------|
| `research_roadmap.json` | Created (22 hypotheses) |
| `research_loop/progress.md` | Created (this file) |

### Git Status

```
commit 52080ad - Initialize research roadmap with 22 hypotheses
```

---

*Last updated: 2026-01-26*
*Agent: Research Initializer*
