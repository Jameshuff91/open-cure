# Research Loop Progress

## Current Session: h191, h194, h124 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested:**
- h191: ATC L1 Incoherence as Novel Repurposing Signal - **INVALIDATED** (but revealed complementary insight)
- h194: Validated Cross-Category Repurposing Database - **VALIDATED** (11 patterns)
- h124: Disease Embedding Interpretability - **VALIDATED** (treatment edges explain similarity)

### h191: ATC L1 Incoherence - INVALIDATED

Tested whether ATC L1 incoherence predicts novel repurposing.

**Key Finding:** Two DIFFERENT coherence signals exist:
- h110: Within-class uniqueness → higher precision for unique
- h191: Category match → higher precision for coherent

**Result:** Coherent 11.1% > Incoherent 6.4% (-4.7pp)

**Cross-category patterns identified:** CV→metabolic, steroids→respiratory, SGLT2→CV

### h194: Cross-Category Repurposing Database - VALIDATED

Created validated database of 11 cross-category repurposing patterns.

**High confidence patterns (10):**
1. Statins → metabolic (FDA, ADA guidelines)
2. Azithromycin → respiratory (NEJM 2011)
3. SGLT2i → cardiovascular (FDA, EMPA-REG)
4. Doxycycline → dermatological (FDA for rosacea)
5. Corticosteroids → respiratory/infectious
6. Antihistamines → dermatological
7. And 4 more...

**Output:** `data/reference/validated_cross_category_repurposing.json`

### h124: Disease Embedding Interpretability - VALIDATED

**KEY FINDING: Node2Vec learns from TREATMENT EDGES.**

**Correlations with embedding similarity:**
| Feature | Pearson r | Notes |
|---------|-----------|-------|
| Drug Jaccard (graph) | **0.384** | Strongest |
| Drug Jaccard (shared pairs) | **0.512** | Very strong |
| Gene Jaccard | 0.030 | Not significant |

**Graph structure:**
- 85,849 treatment edges drive similarity
- Gene edges (123K) don't contribute much
- Only 543 direct disease-disease edges

**Implication:** kNN works because diseases sharing treatments end up close in embedding space. "No-treatment" embeddings (26% R@30) are fair; original (37%) include treatment leakage.

### Cumulative Statistics (2026-02-05)
| Status | Count |
|--------|-------|
| Validated | 95 |
| Invalidated | 43 |
| Inconclusive | 8 |
| Blocked | 18 |
| Deprioritized | 2 |
| Pending | 30 |
| **Total Tested** | **148** |

### Key Session Learnings

1. **h110 and h191 measure different signals** - both valid and complementary
2. **11 cross-category patterns validated** via literature (SGLT2, azithromycin, statins, etc.)
3. **Node2Vec = treatment similarity** - Drug Jaccard r=0.51 for shared pairs
4. **Gene associations don't explain embeddings** - r=0.03 only
5. **Treatment edges are the primary signal** - 85K edges in DRKG

### Recommended Next Steps

1. **h193: Combined ATC Coherence Signals** (priority 3) - Test both coherence signals together
2. **h160: Cancer Targeted Therapy Specificity** (priority 3) - Stratify cancer drugs
3. **h196: Gene-Augmented Disease Similarity** (priority 4) - Test if gene overlap helps

---

## Previous Sessions

See previous entries in git history or archived progress.md.
