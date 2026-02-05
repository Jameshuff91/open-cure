# Research Loop Progress

## Current Session: h205, h207, h206 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested:**
- h205: Lymphoma Mechanism-Based Production Rules (CD30+/CD20+) - **VALIDATED**
- h207: Rituximab Prediction Gap Analysis - **VALIDATED**
- h206: Manual Rule Injection for Missing DRKG Drugs - **VALIDATED**

---

### h205: Lymphoma Mechanism-Based Production Rules - VALIDATED

**Objective:** Implement mechanism-based rules matching lymphoma subtypes to appropriate targeted therapies.

**ROOT CAUSE FOUND:** Adcetris NOT IN DRKG drug pool

**Key Findings:**
- Adcetris (CD30 antibody-drug conjugate) is NOT in DRKG despite FDA approval for 10 lymphoma indications
- The model CANNOT predict Adcetris because it has no embeddings
- CD30+ lymphomas (Hodgkin, ALCL, PTCL, CTCL): 0% precision - drugs cannot be predicted
- CD20+ lymphomas: 10% precision - Rituximab predicted for only 3/6 diseases

**Implication:** Mechanism-based rules cannot help if target drug is missing from embedding space.

**Output:** `data/analysis/h205_lymphoma_mechanism_rules.json`

---

### h207: Rituximab Prediction Gap Analysis - VALIDATED

**Objective:** Why is Rituximab predicted for some CD20+ diseases but not others?

**ROOT CAUSE:** kNN neighbor GT coverage

**Key Findings:**
- Follicular lymphoma: **0/20 neighbors have Rituximab in GT** → cannot be predicted
- Burkitt lymphoma: **0/20 neighbors have Rituximab in GT** → cannot be predicted
- DLBCL: 1/20 neighbors but score (0.695) below threshold (0.758)
- Diseases WHERE Rituximab IS predicted all have 1+ neighbors with it in GT

**Implication:** kNN works correctly but GT coverage gap limits recommendations.

**Output:** `data/analysis/h207_rituximab_gap_analysis.json`

---

### h206: Manual Rule Injection for Missing DRKG Drugs - VALIDATED

**Objective:** Quantify DRKG drug coverage gap and create manual injection rules.

**Key Findings:**
- **62.5% of GT drugs have DRKG embeddings** (1480/2367)
- **37.5% (887 drugs) are MISSING** - cannot be predicted
- **40 biologics missing** with 75 blocked drug-disease pairs
- **2,429 total blocked drug-disease pairs**

**Top Missing Biologics:**
1. Certolizumab (6 indications) - RA, AS, psoriasis
2. Faricimab (5 indications) - macular degeneration
3. Epcoritamab (4 indications) - DLBCL, FL
4. Adcetris (10 indications) - CD30+ lymphomas

**Deliverable:** Created `data/reference/manual_drug_rules.json` with 30 manual injection rules.

**Output:** `data/analysis/h206_missing_drkg_drugs.json`

---

### Cumulative Statistics (2026-02-05)
| Status | Count |
|--------|-------|
| Validated | 107 |
| Invalidated | 43 |
| Inconclusive | 8 |
| Blocked | 17 |
| Deprioritized | 3 |
| Pending | 32 |
| **Total** | **210** |

### Key Session Learnings

1. **h205:** Adcetris missing from DRKG = fundamental coverage gap for CD30+ lymphomas
2. **h207:** kNN neighbor GT coverage determines drug recommendations - zero coverage = zero prediction
3. **h206:** 37.5% of GT drugs have no DRKG embeddings; newer biologics systematically absent

### Session Theme: DRKG Coverage Gaps

All three hypotheses converged on the same root cause: **DRKG drug coverage is the fundamental bottleneck**.

The kNN collaborative filtering approach works correctly, but:
- If a drug has no embedding → cannot be predicted
- If no kNN neighbors have a drug in GT → cannot be recommended

This is NOT a model failure - it's a data coverage gap.

### Recommended Next Steps

1. **h210: Implement Manual Rule Injection Layer** (priority 4) - integrate manual_drug_rules.json
2. **h208: DRKG Biologic Coverage Audit** (priority 3) - systematic gap analysis
3. **h209: GT Coverage Analysis** (priority 3) - identify all blocked predictions

---

## Previous Session: h199, h203, h204, h195, h200 (2026-02-05)

**Hypotheses Tested:**
- h199: Solid vs Hematologic Cancer Gap Analysis - **VALIDATED**
- h203: GT-Density Weighted Confidence Scoring - **VALIDATED**
- h204: Lymphoma Subtype Stratification - **VALIDATED**
- h195: Metabolic Exception Analysis - **VALIDATED**
- h200: Brain Tumor Zero Hit Investigation - **VALIDATED**

**Key Learnings:**
- h199: Disease fragmentation causes hematologic low precision
- h203: GT density = strong confidence signal (31x difference)
- h204: Use mechanism-based rules (CD30+/CD20+), not subtype overlap
- h195: CV→Metabolic is comorbidity management, not novel repurposing
- h200: Brain tumor failure is DRKG drug coverage gap, not model failure

---

## Archive

See previous entries in git history or archived progress.md.
