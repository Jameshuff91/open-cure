# Research Loop Progress

## Current Session: h229, h230, h231, h232, h127 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 5**
- h229: Drug-Class Prediction for Cardiovascular - **VALIDATED**
- h230: Integrate Full CV Drug-Class Prediction into Production - **VALIDATED**
- h231: CV Disease DRKG Coverage Gap Analysis - **VALIDATED**
- h232: Stroke Subtype Analysis - Why kNN Wins - **VALIDATED**
- h127: Explicit Frequency-Score Interaction Feature - **VALIDATED**

### Cumulative Statistics (2026-02-05)
| Status | Count |
|--------|-------|
| Validated | 126 |
| Invalidated | 46 |
| Inconclusive | 8 |
| Blocked | 18 |
| Deprioritized | 3 |
| Pending | 30 |
| **Total** | **232** |

### Session Key Learnings

1. **h229:** Drug-class prediction achieves +21.8pp vs kNN for CV diseases. Best for HF (+45pp), AFib (+42pp), MI (+38pp), CAD (+34pp).
2. **h230:** Integrated CV drug-class rescue rules. AFib and hypertension achieve 100% HIGH tier precision.
3. **h231:** Added 50 CV MESH mappings, improving coverage from 44.3% to 63.6% (+43.8%).
4. **h232:** Stroke has only 32.8% drug class coverage. kNN wins for stroke due to broader treatment paradigm.
5. **h127:** Explicit freq_x_score interaction captures 43% of XGBoost advantage. Linear+interaction nearly matches XGBoost (25.37% vs 26.55%).

### Session Theme: Cardiovascular Integration & Model Interpretability

**CV Drug-Class Integration:**
- Validated approach (+21.8pp vs kNN)
- Integrated into production predictor
- Expanded MESH mappings for broader applicability
- Identified exception (stroke) where kNN remains superior

**Model Interpretability:**
- Simple linear model + freq_x_score interaction achieves 25.37% precision
- Only 1.18 pp below XGBoost's 26.55%
- Explicit interaction captures 43% of XGBoost's advantage

---

### h229: Drug-Class Prediction for Cardiovascular - VALIDATED

| Category | kNN | DrugClass | Diff |
|----------|-----|-----------|------|
| Heart failure | 12.7% | 57.7% | +45.0pp |
| AFib | 23.9% | 65.6% | +41.7pp |
| MI | 10.2% | 47.9% | +37.8pp |
| CAD | 31.7% | 65.8% | +34.1pp |
| **Overall** | **22.3%** | **44.2%** | **+21.8pp** |

---

### h230: CV Drug-Class Production Integration - VALIDATED

- AFib, MI, CAD rescue rules integrated
- HIGH tier precision: 59.4%
- AFib: 100%, Hypertension: 100%

---

### h231: CV MESH Mapping Expansion - VALIDATED

- Before: 112/253 (44.3%), After: 161/253 (63.6%)
- **+49 diseases matched (+43.8%)**
- GT drug coverage: 85.3%

---

### h232: Stroke Analysis - VALIDATED

- Only 32.8% drug class coverage
- Stroke treatment includes BP control, cognitive, spasticity
- kNN wins via neighbor overlap with HTN, dementia

---

### h127: Explicit Interaction Feature - VALIDATED

- Linear + freq_x_score: 25.37% precision
- XGBoost: 26.55% precision
- Difference: 1.18 pp
- freq_x_score captures 43% of XGBoost advantage

---

## Archive

See previous entries in git history.
