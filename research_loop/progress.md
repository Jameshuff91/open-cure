# Research Loop Progress

## Current Session: h229, h230 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** In Progress
**Hypotheses Tested: 2**
- h229: Drug-Class Prediction for Cardiovascular - **VALIDATED**
- h230: Integrate Full CV Drug-Class Prediction into Production - **VALIDATED**

### Cumulative Statistics (2026-02-05)
| Status | Count |
|--------|-------|
| Validated | 122 |
| Invalidated | 46 |
| Inconclusive | 8 |
| Blocked | 18 |
| Deprioritized | 3 |
| Pending | 32 |
| **Total** | **229** |

### Session Key Learnings

1. **h229:** Drug-class prediction achieves +21.8pp vs kNN for cardiovascular diseases (44.2% vs 22.3%). Best for HF (+45pp), AFib (+42pp), MI (+38pp), CAD (+34pp).
2. **h230:** Integrated CV drug-class rescue rules into production_predictor.py. Achieved 59.4% HIGH tier precision. AFib and hypertension hit 100% precision.

### Session Theme: Cardiovascular Drug-Class Integration

This session extended the drug-class prediction approach (validated in h174 for psychiatric) to cardiovascular diseases:
- **h229:** Validated that drug-class prediction outperforms kNN for CV diseases by +21.8pp
- **h230:** Integrated AFib, MI, CAD rescue rules into production predictor
- **Key drugs now rescued:** Rivaroxaban, Warfarin (AFib), Clopidogrel, Ticagrelor (MI/CAD), Nitroglycerin (CAD)

---

### h229: Drug-Class Prediction for Cardiovascular - VALIDATED

**Objective:** Evaluate whether drug-class prediction (like h174 for psychiatric) extends to cardiovascular diseases.

**KEY RESULTS:**
| Metric | kNN | DrugClass | Diff |
|--------|-----|-----------|------|
| Overall CV | 22.3% | 44.2% | +21.8pp |
| Heart failure | 12.7% | 57.7% | +45.0pp |
| AFib | 23.9% | 65.6% | +41.7pp |
| MI | 10.2% | 47.9% | +37.8pp |
| CAD | 31.7% | 65.8% | +34.1pp |

**DRUG CLASSES DEFINED:**
- ACE inhibitors, ARBs, beta-blockers, CCBs, diuretics (HTN/HF)
- Anticoagulants, DOACs (AFib)
- Antiplatelets, statins, nitrates (MI/CAD)
- Antiarrhythmics (arrhythmias)

**NEW HYPOTHESES GENERATED:**
- h230: Integrate Full CV Drug-Class Prediction into Production
- h231: CV Disease DRKG Coverage Gap Analysis
- h232: Stroke Subtype Analysis - Why kNN Wins

**Output:** `data/analysis/h229_cv_drug_class_prediction.json`

---

### h230: Full CV Drug-Class in Production - VALIDATED

**Objective:** Integrate h229's CV drug-class rules into production_predictor.py.

**IMPLEMENTATION:**
1. Added drug class constants (ANTICOAGULANT_DRUGS, ANTIPLATELET_DRUGS, NITRATE_DRUGS, etc.)
2. Added disease keywords (AFIB_KEYWORDS, MI_KEYWORDS, CAD_KEYWORDS)
3. Added 'atrial fibrillation', 'atrial flutter' to cardiovascular category keywords
4. Added rescue rules in _apply_category_rescue() for AFib, MI, CAD

**PRECISION RESULTS (8 CV diseases, top 20):**
| Tier | TP | Total | Precision |
|------|-------|-------|-----------|
| HIGH | 19 | 32 | 59.4% |

**PER-DISEASE HIGH TIER:**
- AFib: 7/7 = 100%
- Hypertension: 6/6 = 100%
- CAD: 2/3 = 67%
- MI: 1/8 = 12.5% (needs investigation)

**RESCUED DRUGS:**
- AFib: Propranolol, Verapamil, Carvedilol, Bisoprolol, Rivaroxaban, Diltiazem, Warfarin
- MI: Clopidogrel, Vorapaxar, Pravastatin, Lovastatin
- CAD: Propranolol, Ticagrelor, Nitroglycerin

---

## Previous Session: h210, h164, h166, h225 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 4**
- h210: Implement Manual Rule Injection Layer in Production Pipeline - **VALIDATED**
- h164: Contraindication Database: Systematic Safety Filter Expansion - **VALIDATED**
- h166: Drug-Disease Mechanism Path Tracing for Interpretability - **VALIDATED**
- h225: Add Mechanism Support to Production Deliverable - **VALIDATED**

### Session Key Learnings

1. **h210:** Manual rule injection adds 45 FDA-approved drug-disease pairs missing from DRKG (4.3% coverage improvement)
2. **h164:** Systematic contraindication expansion has diminishing returns; kNN model implicitly avoids harmful patterns. Added immunosuppressant + infection rule (+10 exclusions)
3. **h166:** Mechanism paths (drug->gene->disease) provide 2.2x precision lift. 22% of predictions have direct paths.
4. **h225:** mechanism_genes column successfully added to production deliverable for researcher prioritization

---

## Archive

See previous entries in git history or archived progress.md.
