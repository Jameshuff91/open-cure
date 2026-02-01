# Research Loop Progress

## Current Session: h71, h82, h83, h86, h80, h81, h88 (2026-01-31, continued)

### Session Summary

**Agent Role:** Research Executor
**Status:** In Progress
**Hypotheses Tested This Session:**
- h71: Per-Category Calibration - **VALIDATED**
- h82: Category-Specific k + Thresholds Combined - **INCONCLUSIVE**
- h83: Why Is Respiratory So Poorly Calibrated - **VALIDATED**
- h86: Same-Category Neighbor Ratio as Confidence Feature - **INVALIDATED**
- h80: Autoimmune-Only Production Model - **VALIDATED**
- h81: GI Disease Alternative Strategy - **VALIDATED**
- h88: Confidence Explanation Generation - **VALIDATED**

### Key Findings

**h71: Per-Category Calibration Reveals Reliability Tiers**
- Tier 1 (HIGH): autoimmune, dermatological, psychiatric, ophthalmic (93-100% precision at any threshold)
- Tier 2 (MEDIUM): cardiovascular, other, cancer (75-92% precision at 0.6+ threshold)
- Tier 3 (LOW/EXCLUDE): metabolic, respiratory, GI, hematological (<50% precision)
- Calibration errors: other (4.5pp), cancer (8.6pp), autoimmune (8.8pp), respiratory (30.8pp!)
- Category-specific thresholds: +4.2pp precision (93.5% vs 89.2%)

**h82: Combined k + Thresholds Are Orthogonal**
- h66's category-specific k improves hit rate for challenging categories (cancer +3.9pp, metabolic +9.1pp)
- But h71 EXCLUDES those same categories
- For high-tier categories, k optimization provides no benefit
- Verdict: h71 alone is sufficient for production

**h83: Respiratory Root Cause Identified**
- Respiratory has lowest same-category neighbor ratio (8.8% vs 45.3% for cancer)
- 65% of neighbors are from "other" category
- kNN recommends drugs for those other diseases → high confidence + low hit rate = 30.8pp overconfidence
- Node2Vec fundamentally doesn't capture respiratory disease similarity

**h86: Same-Category Neighbor Ratio Doesn't Help**
- Correlation with hit rate: -0.019 (effectively zero)
- Filtering by ratio >= 0.1: -0.3% precision, -35 diseases
- h71's direct category tiering is more effective than derived metrics

**h80: Autoimmune Is Highest-Confidence Category**
- 480 predictions, 100% HIGH confidence
- 76% novel (367/480)
- Top drug classes: corticosteroids (5-7 diseases), immunomodulators (5 diseases)
- Notable: Baricitinib→SLE, Adalimumab→MS/SLE, Hydroxychloroquine→MS

**h81: GI Correctly Handled by Exclusion**
- Only 3 GI diseases, 40% hit rate, 0% same-category neighbors
- Low confidence (0.41) correctly signals unreliability
- h71's exclusion is the right approach

**h88: Confidence Explanation Framework Created**
- 3 tier-based explanation templates for user-facing output
- TIER 1: "93-100% precision, strong mechanistic overlap"
- TIER 2: "~80% precision at 0.6+, recommend literature validation"
- TIER 3: "Exploratory only, consider specialized databases"

### Session Statistics

- Hypotheses tested: 7
- Validated: 5 (h71, h83, h80, h81, h88)
- Inconclusive: 1 (h82)
- Invalidated: 1 (h86)
- New hypotheses added: h82-h89

### Pending Hypotheses

| Priority | ID | Title | Effort |
|----------|-----|-------|--------|
| 1 | h69 | Production Pipeline Integration | high |
| 2 | h74 | Use Case-Aware Production API | medium |
| 2 | h84 | Tier-Based User Interface Design | medium |
| 2 | h89 | Validation Priority Scoring | medium |
| 3 | h55 | GEO Gene Expression Data Integration | high |
| 3 | h85 | Metabolic Disease Rescue via Alternative Similarity | medium |
| 3 | h87 | Drug Mechanism Clustering | medium |
| 4 | h64 | ARCHS4 Gene Expression | high |
| 20 | h16 | Clinical Trial Phase Features | medium |

### Recommended Next Steps

1. **h84**: Tier-based UI design based on h71's calibration findings
2. **h74**: Use case-aware API leveraging h70's threshold recommendations
3. **h89**: Validation priority scoring for clinical partners
4. **h69**: Full production pipeline integration (high effort but high value)

### Key Learnings

1. **Category is king:** Disease category is the strongest predictor of model reliability
2. **Orthogonal optimizations:** k-optimization and threshold-optimization address different problems
3. **Derived metrics don't help:** Same-category neighbor ratio doesn't add value over direct category tiering
4. **Node2Vec limitations:** Respiratory/GI diseases aren't well-captured by graph embeddings
5. **Autoimmune excellence:** Shared mechanisms make autoimmune the "safe" category for predictions
6. **Tier-based explanations:** Users understand category tiers better than numeric scores

---

## Previous Sessions

### h79, h76 (2026-01-31)
- 2 hypotheses tested, 2 validated
- Per-disease results enabled category calibration
- Category subsetting: 3.8x coverage gain at 93.5% precision

### h70, h75, h77, h78, h67 (2026-01-31)
- 5 hypotheses tested, 2 validated, 1 inconclusive, 2 invalidated
- Use case thresholds defined: discovery (0.3), validation (0.5), clinical (0.8)
- Category dominates confidence: autoimmune 68x enriched in clinical tier

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

---

## Cumulative Statistics (2026-01-31)

| Status | Count |
|--------|-------|
| Validated | 35 |
| Invalidated | 25 |
| Inconclusive | 4 |
| Blocked | 14 |
| Pending | 9 |
| **Total Tested** | **64** |
