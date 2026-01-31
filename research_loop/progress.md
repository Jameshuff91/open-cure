# Research Loop Progress

## Current Session: h49, h50, h51, h52 (2026-01-31)

### Session Summary

**Agent Role:** Research Executor
**Status:** Completed (4 hypotheses)
**Hypotheses Tested:**
- h49: Gene Expression → Drug Mapping Pipeline - **VALIDATED**
- h50: Rare Skin Disease Baseline Evaluation - **VALIDATED**
- h51: Gene Module Similarity for Disease Matching - **INVALIDATED**
- h52: Meta-Confidence Model for Prediction Reliability - **VALIDATED**

### Results Summary

| Hypothesis | Status | Key Finding |
|---|---|---|
| h49 | **VALIDATED** | Gene→Drug mapping pipeline created. Known drugs rank 88.1 percentile on average. |
| h50 | **VALIDATED** | Skin diseases: 54.87% R@30 vs 32.60% non-skin (+22 pp improvement) |
| h51 | **INVALIDATED** | Gene Jaccard kNN 17.09% vs Node2Vec 31.80% (-14.71 pp). Node2Vec is better. |
| h52 | **VALIDATED** | Meta-confidence AUC 0.733. High-confidence (>80%) predictions achieve 89.7% actual hit rate. |

### Key Deliverables Created

1. **Gene Expression → Drug Mapping Pipeline** (`scripts/gene_expression_drug_mapping.py`)
   - Maps dysregulated genes to candidate drugs
   - 19,089 drugs, 19,565 genes, 155,765 edges
   - Usage: `python scripts/gene_expression_drug_mapping.py --disease "MESH:D011565"`

2. **Skin Disease Performance Baseline**
   - Skin diseases achieve 54.87% ± 9.83% R@30 (vs 32.60% baseline)
   - Rare skin diseases: 62.18% ± 37.70% (high variance, small sample)
   - kNN method is particularly effective for dermatological conditions

3. **Meta-Confidence Model**
   - Predicts kNN success probability using training-only features
   - AUC 0.733, calibration: 80%+ predicted → 89.7% actual
   - Top features: disease category, neighbor GT sizes, similarity metrics

### Current Metrics

| Model | R@30 | Notes |
|-------|------|-------|
| kNN Collaborative Filtering k=20 | 37.04% ± 5.81% | Best overall (honest 5-seed) |
| kNN on Skin Diseases | 54.87% ± 9.83% | +22 pp vs baseline |
| kNN on Rare Skin Diseases | 62.18% ± 37.70% | High variance |

### Remaining Pending Hypotheses

1. h16: Clinical Trial Phase Features (Priority 20, low impact)

### Recommended Next Steps

Based on session findings, high-ROI next directions:

1. **Improve Disease Name Mapping for Skin Diseases**
   - Only 39/141 (28%) skin diseases have embeddings
   - Mapping is the bottleneck, not the model
   - Expected gain: +50% skin disease coverage

2. **Save and Package Meta-Confidence Model**
   - Train production model on full dataset
   - Create prediction tiering system (high/medium/low confidence)
   - Integrate into prediction pipeline

3. **Find External Transcriptomic Data for Rare Skin Diseases**
   - GEO/GTEx for gene expression signatures
   - Use gene→drug mapping pipeline (h49)
   - Bypass DRKG coverage limitations

4. **Expand Ground Truth for Rare Skin Diseases**
   - Query literature/clinical trials for additional drug-disease pairs
   - Improve kNN pool for rare conditions
