# Research Loop Progress

## Current Session: h229, h230, h231, h232 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** In Progress
**Hypotheses Tested: 4**
- h229: Drug-Class Prediction for Cardiovascular - **VALIDATED**
- h230: Integrate Full CV Drug-Class Prediction into Production - **VALIDATED**
- h231: CV Disease DRKG Coverage Gap Analysis - **VALIDATED**
- h232: Stroke Subtype Analysis - Why kNN Wins - **VALIDATED**

### Cumulative Statistics (2026-02-05)
| Status | Count |
|--------|-------|
| Validated | 125 |
| Invalidated | 46 |
| Inconclusive | 8 |
| Blocked | 18 |
| Deprioritized | 3 |
| Pending | 32 |
| **Total** | **232** |

### Session Key Learnings

1. **h229:** Drug-class prediction achieves +21.8pp vs kNN for CV diseases. Best for HF (+45pp), AFib (+42pp), MI (+38pp), CAD (+34pp).
2. **h230:** Integrated CV drug-class rescue rules. AFib and hypertension achieve 100% HIGH tier precision.
3. **h231:** Added 50 CV MESH mappings, improving coverage from 44.3% to 63.6% (+43.8%).
4. **h232:** Stroke has only 32.8% drug class coverage because treatment includes BP control, cognitive support, spasticity. kNN wins for stroke.

### Session Theme: Cardiovascular Drug-Class Integration

This session validated and integrated drug-class prediction for cardiovascular diseases:
- Validated approach (+21.8pp vs kNN)
- Integrated into production predictor
- Expanded MESH mappings for broader applicability
- Identified exception (stroke) where kNN remains superior

**Key insight:** Drug-class prediction works for diseases with well-defined treatment paradigms (HF, AFib, MI). It fails for multi-faceted conditions (stroke) where kNN's broader neighbor similarity provides better coverage.

---

### h229: Drug-Class Prediction for Cardiovascular - VALIDATED

**Results:**
| Category | kNN | DrugClass | Diff |
|----------|-----|-----------|------|
| Heart failure | 12.7% | 57.7% | +45.0pp |
| AFib | 23.9% | 65.6% | +41.7pp |
| MI | 10.2% | 47.9% | +37.8pp |
| CAD | 31.7% | 65.8% | +34.1pp |
| **Overall CV** | **22.3%** | **44.2%** | **+21.8pp** |

---

### h230: CV Drug-Class Production Integration - VALIDATED

**Implementation:**
- Added AFib, MI, CAD rescue rules to production_predictor.py
- Added drug class constants and disease keywords
- HIGH tier precision: 59.4% (19/32)
- AFib: 100%, Hypertension: 100%, CAD: 67%, MI: 12.5%

---

### h231: CV MESH Mapping Expansion - VALIDATED

**Results:**
- Before: 112/253 CV diseases matched (44.3%)
- After: 161/253 CV diseases matched (63.6%)
- **Improvement: +49 diseases (+43.8%)**
- GT drug coverage: 85.3% (769/901)

---

### h232: Stroke Analysis - VALIDATED

**Why drug-class fails for stroke:**
- Only 32.8% of stroke drugs covered by traditional classes
- Stroke treatment includes: BP control, cognitive support, spasticity
- kNN wins because stroke shares neighbors with HTN, dementia, heart disease
- **Implication:** Keep kNN for stroke category

---

## Previous Session: h210, h164, h166, h225 (2026-02-05)

**Hypotheses Tested: 4**
- h210: Manual Rule Injection Layer - VALIDATED
- h164: Contraindication Database Expansion - VALIDATED
- h166: Mechanism Path Tracing - VALIDATED
- h225: Mechanism Support in Production - VALIDATED

---

## Archive

See previous entries in git history or archived progress.md.
