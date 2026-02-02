# Methodology Limitations

This document describes fundamental limitations of our drug repurposing evaluation methodology that cannot be easily fixed and must be disclosed when sharing results.

## 1. Transductive vs Inductive Evaluation

### The Issue

Our evaluation is **transductive**: test diseases retain their presence in the knowledge graph through gene associations, pathway connections, and side-effect edges. Only the treatment edges are removed.

This differs from TxGNN's **inductive** evaluation where test diseases are completely removed from the graph before training embeddings.

### Implications

- Our 26% R@30 may not represent true zero-shot performance
- Diseases can still be found similar to training diseases through non-treatment edges
- Direct comparison to TxGNN's 6.7-14.5% is not strictly apples-to-apples

### Why We Can't Fix It

Removing diseases entirely from the graph would require:
1. Retraining Node2Vec without any test disease edges
2. Losing the ability to compute disease similarity for test diseases
3. A fundamentally different evaluation paradigm

### Fair Comparison Interpretation

The 10.5 percentage point drop from 36.59% (original) to 26.06% (honest) quantifies the contribution of treatment edges. The remaining 71% of performance comes from indirect similarity signals (genes, pathways, side effects). Our method still outperforms TxGNN (~2x better), but the gap is smaller than originally reported.

### Inductive Evaluation Option (NEW - 2026-02-01)

We developed an alternative **inductive evaluation** using KEGG pathway similarity instead of graph embeddings:

| Metric | Transductive (Node2Vec) | Inductive (KEGG) | TxGNN |
|--------|-------------------------|------------------|-------|
| R@30 | 26.06% | **15.73%** | 6.7-14.5% |
| Test disease in graph | Yes (non-treatment edges) | No | No |
| Features | Node2Vec embeddings | KEGG pathway Jaccard | GNN |

This provides a fair apples-to-apples comparison: KEGG pathway kNN achieves **15.73% R@30**, competitive with TxGNN's 6.7-14.5% under the same inductive paradigm.

Script: `scripts/evaluate_kegg_pathway_knn.py`

---

## 2. Coverage-Dependent Performance (Bimodal Distribution)

### The Issue

kNN can only recommend drugs that appear in similar training diseases' ground truth. Our analysis shows:

- **84.7%** of test diseases have at least one GT drug in the kNN pool → **24.2% R@30**
- **15.3%** of test diseases have zero GT drug coverage → **0.0% R@30**

### Implications

- Our 26% R@30 is an **average over a bimodal distribution**
- For diseases with coverage, performance is ~24%
- For diseases without coverage, performance is effectively random (0%)
- The overall metric obscures this stark divide

### Why We Can't Fix It

This is a fundamental limitation of the kNN paradigm. The model transfers treatments from similar diseases. If no similar disease in training shares any treatments with the test disease, recall must be zero.

Potential mitigations (not implemented):
- Hybrid approaches combining kNN with embedding-based scoring
- Extending the candidate pool beyond kNN neighbors
- Multi-hop drug transfer

---

## 3. Rare Disease Limitation

### The Issue

Performance varies dramatically by number of known treatments:

| GT Drugs | R@30 | Interpretation |
|----------|------|----------------|
| 1 | 13.5% | Ultra-rare diseases |
| 2-5 | 19-21% | Rare diseases |
| 6-10 | 32.2% | Well-studied |
| 11+ | 27.8% | Very well-studied |

Diseases with only 1-2 known treatments perform 2.4x worse than those with 6-10.

### Implications

- The "similar diseases share treatments" paradigm fails for rare diseases
- If a disease has few known treatments, fewer neighbors can contribute
- If a disease is dissimilar to training diseases, kNN finds poor matches
- **The hardest, most important cases (orphan diseases) show worst performance**

### Why We Can't Fix It

This is architectural. kNN fundamentally requires:
1. Similar diseases to exist in training
2. Those similar diseases to have known treatments

For rare orphan diseases, both conditions often fail.

---

## 4. Ground Truth Circularity

### The Issue

32% of our Every Cure ground truth pairs also exist as treatment edges in DRKG.

| Metric | Value |
|--------|-------|
| GT pairs in DRKG | 1,184 (32%) |
| GT pairs NOT in DRKG | 2,511 (68%) |
| Diseases with 100% overlap | 66 |
| Diseases with 0% overlap | 222 |

### Implications

- ~1/3 of our "ground truth" was in the training graph (as treatment edges, which we removed)
- Our honest evaluation correctly removes these edges before training
- However, the remaining signal (32%) was still available through correlated edges

### Assessment

This is **low circularity** (32%). Most GT pairs are genuinely novel to DRKG treatment edges, making our evaluation primarily a test of prediction rather than recall.

---

## 5. Selection Bias (Funnel Attrition)

### The Issue

Our evaluation set is heavily filtered from the original Every Cure data:

| Stage | Diseases | Drop |
|-------|----------|------|
| Raw Every Cure | 3,996 | - |
| After MESH mapping | 454 | 88.6% |
| In original embeddings | 435 | 4.2% |
| In honest embeddings | 385 | 11.5% |
| With ≥1 evaluable GT drug | 368 | 4.4% |

**Total attrition: 90.8%** (3,996 → 368 diseases)

### Implications

- We evaluate on only ~9% of Every Cure diseases
- The 88.6% drop at MESH mapping is the biggest bottleneck
- Diseases without MESH IDs may be systematically different (rarer, newer)
- Results may not generalize to the full disease universe

### Why We Can't Fix It

MESH ID mapping is required to link disease names to DRKG entities. Improving the matcher would help but cannot eliminate the fundamental vocabulary mismatch between Every Cure disease names and DRKG MESH IDs.

---

## 6. Disconnected Diseases

### The Issue

51 diseases in our ground truth became disconnected (lost embeddings) after removing treatment edges:

- These include important diseases like Parkinson's (19 GT drugs)
- 15 diseases with ≥5 GT drugs were lost
- 86% had other edge types but became disconnected anyway

### Implications

- Some well-studied diseases are excluded from honest evaluation
- These diseases were primarily connected through treatment edges
- Results may not reflect performance on diseases with sparse non-treatment connectivity

### Assessment

This is a necessary tradeoff for honest evaluation. Including these diseases would mean evaluating on partially leaked information.

---

## 7. Hyperparameter Choices

### The Issue

Key parameters were inherited or chosen pragmatically, not optimized:

| Parameter | Value | Source |
|-----------|-------|--------|
| Node2Vec dim | 256 | Inherited |
| Walk length | 80 | Inherited |
| Num walks | 10 | Inherited |
| p, q | 1.0, 1.0 | Inherited |
| kNN k | 20 | Grid search (h43) |
| Test fraction | 20% | Standard |

### Implications

- Node2Vec parameters may not be optimal for this task
- Different parameters might yield different results
- However, not tuning avoids overfitting to our benchmark

### Assessment

This is both a limitation (may not be optimal) and a strength (not overfit). Our kNN k=20 was validated as optimal through systematic testing (72 configurations in h43).

---

## Summary Table

| Limitation | Severity | Addressable? | Key Number |
|------------|----------|--------------|------------|
| Transductive evaluation | HIGH | **ADDRESSED** | KEGG kNN: 15.73% inductive |
| Coverage dependence | HIGH | Architectural | 15% zero coverage |
| Rare disease failure | HIGH | Fundamental | 13.5% vs 32.2% R@30 |
| GT circularity | LOW | Already addressed | 32% overlap |
| Selection bias | MEDIUM | Partially | 90.8% attrition |
| Disconnected diseases | MEDIUM | Necessary | 51 diseases |
| Hyperparameters | LOW | By design | - |

---

## Recommendation for Sharing

When presenting these results, disclose:

1. **Transductive claim**: 26.06% R@30 with honest embeddings (no treatment edges)
2. **Inductive claim**: 15.73% R@30 using KEGG pathways only (fair TxGNN comparison)
3. **Statistical validity**: p=0.025 (paired t-test), Cohen's d=2.44 (large effect)
4. **Bimodal nature**: ~24% for diseases with kNN coverage, 0% without
5. **TxGNN comparison**: Inductive 15.7% vs 6.7-14.5% (fair); Transductive 26% (not directly comparable)
6. **Fundamental limit**: kNN fails for rare diseases without similar training examples
7. **Novel discovery**: 100% of validated predictions have NO direct DRKG treatment edge
8. **Biological interpretability**: All validated predictions have traceable mechanisms

The method works well for diseases similar to training data with shared treatments. It fails for rare, isolated diseases—arguably the cases we most want to solve.

---

## Additional Evidence (2026-02-01)

See `docs/impressive_evidence_report.md` for:
- Full inductive evaluation methodology
- Novel discovery classification for all validated predictions
- Mechanism tracing with biological hypotheses
- Detailed case studies (Dantrolene→HF, Rituximab→MS, etc.)
