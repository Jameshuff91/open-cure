# Research Loop Progress

## Current Session: h70, h75, h77, h78, h67 (2026-01-31)

### Session Summary

**Agent Role:** Research Executor
**Status:** Completed (5 hypotheses tested)
**Hypotheses Tested:**
- h70: Threshold Optimization by Use Case - **VALIDATED**
- h75: Coverage Gap Analysis - **VALIDATED**
- h77: Category-Specific Confidence Thresholds - **INCONCLUSIVE** (methodology issue)
- h78: Known Indication Density as Confidence Proxy - **INVALIDATED**
- h67: Drug Class (ATC) Boosting for kNN - **INVALIDATED**

### Key Findings

**h70: Use Case Thresholds Defined**
- Discovery: combined_avg @ 0.3 (57% precision, 88 diseases)
- Validation: prob_h52 @ 0.5 (75% precision, 56 diseases)
- Clinical: combined_avg @ 0.8 (100% precision, 5 diseases)
- Key insight: Different methods optimal for different use cases

**h75: Category Dominates Confidence**
- Autoimmune: 68x enriched in CLINICAL tier (63% of clinical diseases)
- Endocrine: 34x enriched, Dermatological: 8.5x enriched
- Cancer, metabolic, respiratory: ZERO in CLINICAL tier
- Known indications 3.7x higher in clinical diseases

**h77: Methodology Gap Identified**
- Cannot calculate per-category precision without held-out evaluation
- h68 only saved aggregates, not per-disease results
- Need to modify h68 to unblock per-category analysis

**h78: Conceptual Issue Found**
- Known indication count correlates with confidence (r=0.558)
- But this is EFFECT not CAUSE - can't use as feature
- Training GT is already captured by h52 model features

**h67: Embedding Superiority Confirmed**
- ATC boosting HURTS kNN: 57.57% → 57.11% (-0.46 pp)
- Only 29.1% drugs have ATC mappings
- Node2Vec embeddings already capture drug similarity
- Don't layer ontology on learned embeddings

### New Hypotheses Generated (this session)

| Priority | ID | Title |
|----------|-----|-------|
| 2 | h79 | Expand h68 to Save Per-Disease Results |
| 3 | h80 | Autoimmune-Only Production Model |
| 3 | h81 | GI Disease Alternative Strategy |

### Updated Pending Hypotheses

| Priority | ID | Title | Effort |
|----------|-----|-------|--------|
| 1 | h69 | Production Pipeline Integration | high |
| 2 | h74 | Use Case-Aware Production API | medium |
| 2 | h79 | Expand h68 for Per-Disease Results | low |
| 3 | h55 | GEO Gene Expression Integration | high |
| 3 | h80 | Autoimmune-Only Production Model | low |
| 3 | h81 | GI Disease Alternative Strategy | low |
| 4 | h64 | ARCHS4 Gene Expression | high |
| 20 | h16 | Clinical Trial Phase Features | medium |

### Session Statistics

- Hypotheses tested: 5
- Validated: 2 (h70, h75)
- Inconclusive: 1 (h77 - blocked by data format)
- Invalidated: 2 (h78, h67)
- New hypotheses: 3 (h79-h81)

### Recommended Next Steps

1. **h79**: Quick win - expand h68 to save per-disease results, unblocks h71/h76
2. **h80**: Autoimmune-focused analysis - strongest category for precision
3. **h74**: Use case-aware API based on h70 findings

### Key Learnings to Archive

1. **Methodology matters:** Can't use known indications as GT proxy - need held-out evaluation
2. **Embeddings are sufficient:** ATC/ontology boosting adds noise on top of learned embeddings
3. **Category predicts confidence:** Autoimmune dominates, cancer/metabolic fail
4. **Use case differentiation:** Different methods optimal for different precision/coverage tradeoffs

---

## Previous Sessions

### h68, h72, h73, h66 (2026-01-31)
- 4 hypotheses tested, 4 validated
- Unified confidence scoring (88% precision at 0.7)
- Production deliverable: 13K predictions, 2.8K HIGH confidence

### h61, h57, h65, h62, h63 (2026-01-31)
- 5 hypotheses tested, 2 validated
- Gene-based approaches fail vs Node2Vec
- h65 success predictor: 70% precision

### h49-h59 (2026-01-31)
- 9 hypotheses tested, 8 validated
- GI diseases: 5% hit rate (blind spot)
