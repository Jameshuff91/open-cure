# Research Loop Progress

## Current Session: h61 (2026-01-31)

### Session Summary

**Agent Role:** Research Executor
**Status:** Completed (1 hypothesis tested)
**Hypothesis Tested:**
- h61: Bio Foundation Model Disease Embeddings (helicalAI Integration) - **INVALIDATED**

### Key Findings

**h61: Geneformer Pseudo-Expression Approach FAILS Dramatically**

- **Geneformer kNN: 8.03% ± 1.06% R@30**
- **Node2Vec kNN: 29.31% ± 1.92% R@30**
- **Difference: -21.28 pp (p=0.0001)**

Root Cause Analysis:
1. Geneformer expects REAL gene expression counts (varying values 0-1000s)
2. Our pseudo-expression (binary 0/100 from disease-gene associations) lacks variation
3. 14,080 NaN values in embeddings (4% of total)
4. 190 near-identical disease pairs (similarity > 0.99) = embedding collapse
5. Foundation models cannot work with fabricated input

### Deliverables Created

1. **Geneformer disease embeddings** (invalidated): `data/analysis/h61/geneformer_disease_embeddings.npy`
2. **Comparison results**: `data/analysis/h61/knn_comparison_results.json`
3. **Analysis documentation** in research_roadmap.json

### New Hypotheses Generated (4)

1. **h62** (Priority 2): Weighted Gene Association for Disease Similarity
   - Use edge weights from DRKG instead of binary associations

2. **h63** (Priority 3): Ensemble kNN - Node2Vec + Weighted Gene Jaccard
   - Combine similarity measures at the similarity level

3. **h64** (Priority 4): Real Gene Expression via ARCHS4
   - Download actual expression profiles from ARCHS4/GEO
   - HIGH effort but could properly test h61's premise

4. **h65** (Priority 2): Meta-Learning - Predict Which Diseases Will Succeed
   - Train classifier to predict >30% hit rate per disease

### Updated Remaining Hypotheses

| Priority | ID | Title | Expected Impact |
|----------|-----|-------|-----------------|
| 2 | h57 | Metabolic Disease Deep Dive | medium |
| 2 | h62 | Weighted Gene Association | medium |
| 2 | h65 | Meta-Learning Disease Success | medium |
| 3 | h63 | Ensemble kNN | medium |
| 3 | h55 | GEO Gene Expression Integration | high |
| 4 | h64 | Real Gene Expression via ARCHS4 | high |
| 20 | h16 | Clinical Trial Phase Features | low |

### Key Learning

Gene-based approaches consistently underperform graph-based similarity:
- h19 (HPO Phenotype): 14.20% R@30 (-22.71 pp vs Node2Vec)
- h51 (Gene Jaccard): 22.21% R@30 (-14.71 pp vs Node2Vec)
- h61 (Geneformer pseudo): 8.03% R@30 (-21.28 pp vs Node2Vec)

The graph structure captured by Node2Vec provides more signal than gene associations alone. Breaking the 37% ceiling requires either:
1. REAL gene expression data (not pseudo-expression)
2. Different architectures (not similarity-based kNN)
3. Better ground truth coverage

### Session Statistics

- Hypotheses tested: 1
- Validated: 0
- Invalidated: 1
- New hypotheses generated: 4 (h62-h65)
- Critical finding: Foundation models require proper input format

---

## Previous Session: h49-h59 (2026-01-31)

- Hypotheses tested: 9
- Validated: 8, Invalidated: 1
- Major finding: GI diseases are a critical blind spot (5% hit rate)
- Extended categories identified clear performance tiers
