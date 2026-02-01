# Research Loop Progress

## Current Session: h61, h57, h65, h62 (2026-01-31)

### Session Summary

**Agent Role:** Research Executor
**Status:** Completed (4 hypotheses tested)
**Hypotheses Tested:**
- h61: Bio Foundation Model (Geneformer) Pseudo-Expression - **INVALIDATED**
- h57: Metabolic Disease Deep Dive - **VALIDATED**
- h65: Meta-Learning Disease Success Predictor - **VALIDATED**
- h62: Weighted Gene Jaccard - **INVALIDATED**

### Key Findings

**h61: Geneformer Pseudo-Expression FAILS (-21 pp vs Node2Vec)**
- Used binary disease-gene associations as fake expression counts
- 8.03% ± 1.06% R@30 vs 29.31% Node2Vec
- Root cause: Foundation models require REAL expression data, not binary

**h57: Metabolic Disease Bifurcated Failure**
- Common metabolic (T2D, gout): 20.3% coverage
- Rare storage diseases (Gaucher, Fabry): 0.0% coverage
- 43% of metabolic drugs are disease-specific

**h65: Meta-Learning Success Predictor Achieves 70% Precision**
- Threshold 0.59: 70.7% precision, 54.5% recall (27% coverage)
- Top features: neighbors_with_gt (0.31), pool_size (0.25)
- Model saved: `models/disease_success_predictor.pkl`

**h62: Weighted Gene Jaccard Doesn't Help (+1.22 pp)**
- IDF-weighted: 15.56% ± 1.27%
- Binary: 14.34% ± 1.14%
- Still -21 pp behind Node2Vec

### Deliverables Created

1. **Disease Success Predictor**: `models/disease_success_predictor.pkl`
2. **Geneformer embeddings (failed)**: `data/analysis/h61/`
3. **kNN comparison results**: `data/analysis/h61/knn_comparison_results.json`

### Updated Remaining Hypotheses

| Priority | ID | Title | Expected Impact |
|----------|-----|-------|-----------------|
| 3 | h55 | GEO Gene Expression Integration | high |
| 3 | h63 | Ensemble kNN (Node2Vec + Gene) | medium |
| 4 | h64 | Real Gene Expression via ARCHS4 | high |
| 20 | h16 | Clinical Trial Phase Features | low |

### Key Learnings

**Gene-based approaches consistently fail:**
- h19 (HPO Phenotype): 14.20% R@30
- h51 (Gene Jaccard): 22.21% R@30
- h61 (Geneformer pseudo): 8.03% R@30
- h62 (Weighted Jaccard): 15.56% R@30
- Node2Vec: 37.04% R@30

**Root cause:** Graph STRUCTURE provides more signal than node attributes. Multi-hop paths (gene→drug→disease) captured by Node2Vec are more informative than gene set overlaps.

### Session Statistics

- Hypotheses tested: 4
- Validated: 2 (h57, h65)
- Invalidated: 2 (h61, h62)
- New hypotheses generated: 4 (h62-h65 from previous session)
- Models saved: 1 (disease_success_predictor.pkl)

---

## Previous Sessions

### h49-h59 (2026-01-31)
- 9 hypotheses tested, 8 validated
- Major finding: GI diseases are critical blind spot (5% hit rate)
- Extended categories identified clear performance tiers
