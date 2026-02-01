# Research Loop Progress

## Current Session: h71, h82, h83, h86, h80, h81, h88, h84, h89, h74 (2026-01-31, continued)

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
- h84: Tier-Based User Interface Design - **VALIDATED**
- h89: Validation Priority Scoring - **VALIDATED**
- h74: Use Case-Aware Production API - **VALIDATED**

### Key Findings

**h71: Per-Category Calibration Reveals Reliability Tiers**
- Tier 1 (HIGH): autoimmune, dermatological, psychiatric, ophthalmic (93-100% precision)
- Tier 2 (MEDIUM): cardiovascular, other, cancer (75-92% precision at 0.6+)
- Tier 3 (LOW): metabolic, respiratory, GI, hematological (<50% precision)

**h83: Respiratory Root Cause Identified**
- Respiratory has 8.8% same-category neighbor ratio (lowest)
- 65% of neighbors are from "other" category

**h80: Autoimmune Excellence Confirmed**
- 480 predictions, 100% HIGH confidence
- Top drugs: corticosteroids, immunomodulators

**h88: Confidence Explanation Framework**
- 3 tier-based templates for user-facing explanations
- Tier 1: "93-100% precision" / Tier 2: "~80% at 0.6+" / Tier 3: "Exploratory only"

**h84: Tier-Based UI Design**
- 🟢 Tier 1: 1,020 predictions (7.6%)
- 🟡 Tier 2: 10,928 predictions (81.5%)
- 🔴 Tier 3: 1,468 predictions (10.9%)

**h89: Validation Priority Scoring**
- Formula: priority = confidence × novelty × tier_weight × rarity
- Top priorities: JIA, RA, Crohn's, atherosclerosis

### Session Statistics

- Hypotheses tested: 10
- Validated: 8 (h71, h83, h80, h81, h88, h84, h89, h74)
- Inconclusive: 1 (h82)
- Invalidated: 1 (h86)
- New hypotheses added: h82-h89

### Pending Hypotheses

| Priority | ID | Title | Effort |
|----------|-----|-------|--------|
| 1 | h69 | Production Pipeline Integration | high |

| 3 | h55 | GEO Gene Expression Data Integration | high |
| 3 | h85 | Metabolic Disease Rescue | medium |
| 3 | h87 | Drug Mechanism Clustering | medium |
| 4 | h64 | ARCHS4 Gene Expression | high |
| 20 | h16 | Clinical Trial Phase Features | medium |

### Key Learnings

1. **Category is king:** Disease category is the strongest predictor of reliability
2. **Tier-based UI:** Users understand tiers better than numeric scores
3. **Priority scoring:** Combine confidence × novelty × tier × rarity for validation prioritization
4. **Exclusion is valid:** For poorly-calibrated categories, exclusion is better than rescue attempts

---

## Previous Sessions

### h79, h76 (2026-01-31)
- 2 hypotheses tested, 2 validated
- Category subsetting: 3.8x coverage gain at 93.5% precision

### h70, h75, h77, h78, h67 (2026-01-31)
- 5 hypotheses tested, 2 validated
- Use case thresholds: discovery (0.3), validation (0.5), clinical (0.8)

### h68, h72, h73, h66 (2026-01-31)
- 4 hypotheses tested, 4 validated
- Production deliverable: 13K predictions, 2.8K HIGH confidence

---

## Cumulative Statistics (2026-01-31)

| Status | Count |
|--------|-------|
| Validated | 38 |
| Invalidated | 25 |
| Inconclusive | 4 |
| Blocked | 14 |
| Pending | 6 |
| **Total Tested** | **67** |
