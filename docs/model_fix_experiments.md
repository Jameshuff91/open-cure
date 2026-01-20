# Model Fix Experiments: Addressing Disease-Specific Recall Issues

**Date:** January 19, 2025
**Objective:** Investigate and fix why certain diseases (HIV, Osteoporosis, Epilepsy) have 0% recall@30 while others (Rheumatoid Arthritis) perform well.

## Executive Summary

We identified the **root cause** of poor recall for HIV/Osteoporosis: the baseline model uses 67% drug embedding features and 33% disease embedding features with **zero interaction features**, causing ~108 drugs to score universally high and crowd out disease-specific drugs.

**Key Finding:** All attempted fixes that deviated from the baseline made performance significantly worse. The baseline model, despite its flaws, contains essential signal that is lost with alternative approaches.

## Root Cause Analysis

### Problem Identification
| Disease | EC Approved | In Top 30 | Best Rank | Issue |
|---------|-------------|-----------|-----------|-------|
| HIV infection | 17 | 0 | #109 | 108 drugs score higher |
| Osteoporosis | 10 | 0 | #90 | 89 drugs score higher |
| Epilepsy | 18 | 1 | #24 | 23 drugs score higher |
| Rheumatoid arthritis | 55 | 15 | #2 | Works well |

### Feature Importance Analysis
The GB classifier feature importance revealed:
- **Drug embedding:** 67% importance (397/400 non-zero features)
- **Disease embedding:** 33% importance (112/400 non-zero features)
- **Product features:** 0% importance
- **Difference features:** 0% importance

This means the model essentially computes: `score = f(drug) + g(disease)` with no true drug-disease interaction learning.

### Disease Embedding Collapse
All disease embeddings have >0.98 cosine similarity - they're nearly identical! This explains why the model relies heavily on drug features.

### Universal High-Scorers
~107 drugs score >0.95 against ANY disease (tested against average disease embedding), filling top ranking slots regardless of the target disease.

## Experiments Conducted

### Baseline (Current GB Model)
| Metric | Value |
|--------|-------|
| Avg R@30 | 20.5% |
| Avg R@100 | 54.1% |
| Avg R@200 | 82.7% |
| Avg Mean Rank | 114 |

### Fix 1: Interaction Features Only
**Hypothesis:** Use only drug-disease interaction features (product, diff, abs_diff, sq_product) instead of raw embeddings.

**Results:**
| Metric | Baseline | Fix 1 | Change |
|--------|----------|-------|--------|
| Avg R@30 | 20.5% | 0.5% | **-20.0%** |
| Avg R@100 | 54.1% | 1.7% | **-52.4%** |
| Mean Rank | 114 | 1524 | **+1410** |

**Conclusion:** Dramatically worse. Raw embeddings contain essential signal.

### Fix 2: Neural Network
**Hypothesis:** A neural network with learnable projections and interaction layers can better capture drug-disease relationships.

**Architecture:**
- Drug/Disease projection layers (128 → 128)
- Interaction: concat + product + diff + abs_diff
- MLP decoder with dropout

**Results:**
| Metric | Baseline | Fix 2 | Change |
|--------|----------|-------|--------|
| Avg R@30 | 20.5% | 0.5% | **-20.0%** |
| Avg R@100 | 54.1% | 1.6% | **-52.5%** |
| Mean Rank | 114 | 1680 | **+1566** |

**Conclusion:** Also dramatically worse, despite 93.6% test accuracy on link prediction.

### Fix 3: Score Calibration
**Hypothesis:** Subtract each drug's "universal score" to remove bias toward always-high-scoring drugs.

**Methods tested:**
1. **Subtract:** `calibrated = raw - universal + 0.5`
2. **Divide:** `calibrated = raw / (universal + 0.01)`
3. **Ratio:** `calibrated = (raw - universal) / (1 - universal + 0.01)`

**Results (Ratio method - best):**
| Disease | Baseline R@30 | Ratio R@30 | Change |
|---------|--------------|------------|--------|
| HIV infection | 0.0% | **5.9%** | +5.9% |
| Osteoporosis | 0.0% | 0.0% | same |
| Rheumatoid arthritis | 27.3% | 12.7% | -14.6% |
| Breast cancer | 56.4% | 28.2% | -28.2% |
| **Average** | 20.5% | 12.0% | **-8.5%** |

**Conclusion:** Helps hard cases (HIV) but significantly hurts easy cases.

### Ensemble Approach
**Hypothesis:** Combine baseline and calibration with a mixing parameter α.

`ensemble = α × raw + (1-α) × calibrated`

**Results:**
| Alpha | Avg R@30 | HIV R@30 |
|-------|----------|----------|
| 0.0 (pure calibration) | 12.0% | **5.9%** |
| 0.5 | 13.6% | 0.0% |
| 1.0 (pure baseline) | **20.5%** | 0.0% |

**Conclusion:** Fundamental trade-off - no single α optimizes both average and hard cases.

## Key Insights

1. **The baseline model contains essential signal** that is lost when we change the feature representation or model architecture.

2. **The problem is disease-specific:** Some diseases (RA, breast cancer) work well with the current approach; others (HIV, osteoporosis) don't.

3. **Universal high-scorers are a symptom, not the cause.** Calibrating them away loses important information.

4. **High test accuracy ≠ good ranking.** Both Fix 1 (92.6%) and Fix 2 (93.6%) had excellent classification accuracy but terrible ranking performance.

## Recommendations

### Short-term (Use Now)
1. **Accept disease-specific performance:** Use R@200 instead of R@30 for hard diseases
2. **Disease-aware thresholds:** HIV needs top 200, RA needs top 30

### Medium-term (Research Needed)
1. **Retrain TransE embeddings** with more focus on disease differentiation
2. **Disease-specific models:** Train separate models for disease clusters
3. **Ensemble per disease:** Use calibration for HIV, baseline for RA

### Long-term
1. **Improve knowledge graph** with more drug-disease edges for underrepresented diseases
2. **Use heterogeneous GNN** (like TxGNN) that can better capture disease-drug specificity

## Files Generated

- `models/baseline_metrics.json` - Baseline evaluation results
- `models/fix1_metrics.json` - Interaction-only GB results
- `models/fix2_metrics.json` - Neural network results
- `models/fix3_metrics.json` - Calibration results
- `models/drug_repurposing_gb_interaction.pkl` - Fix 1 model
- `models/drug_repurposing_nn.pt` - Fix 2 neural network
- `images/analysis/gb_importance.png` - Feature importance visualization
- `images/analysis/universal_scores.png` - Universal score distribution
- `images/analysis/disease_comparison.png` - Per-disease ranking visualization
