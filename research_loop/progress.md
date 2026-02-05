# Research Loop Progress

## Current Session: h250 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 1**
- h250: Systematic Drug Class Safety Review for Cardiovascular - **VALIDATED**

### Cumulative Statistics (2026-02-05)
| Status | Count |
|--------|-------|
| Validated | 143 |
| Invalidated | 50 |
| Inconclusive | 8 |
| Blocked | 18 |
| Deprioritized | 3 |
| Pending | 35 |
| **Total** | **257** |

### h250 Findings: CV Drug Class Safety Review

**EVIDENCE-BASED CONTRAINDICATIONS IDENTIFIED:**

1. **Non-DHP CCBs (Verapamil, Diltiazem) + Heart Failure**
   - Source: ACC/AHA 2022 Guidelines
   - Mechanism: Negative inotropes cause acute decompensation
   - Predictions excluded: 3 (1 HIGH tier)

2. **Class Ic Antiarrhythmics (Flecainide, Propafenone) + Structural Heart Disease**
   - Source: CAST trial (NEJM 1991)
   - Finding: 2.5x mortality increase in post-MI patients
   - Predictions excluded: 4

3. **Ganglionic Blockers (Mecamylamine)**
   - Status: Obsolete drug class
   - Issues: Severe orthostatic hypotension, multiple side effects
   - Predictions excluded: 37 (2 HIGH tier)

4. **Surgical/Diagnostic Dyes (Isosulfan blue, Methylene blue, etc.)**
   - Not therapeutic agents
   - Predictions excluded: 47 (6 HIGH tier)

**DRUGS REVIEWED BUT NOT CONTRAINDICATED:**

| Drug Class | Evidence | Conclusion |
|------------|----------|------------|
| Nitrates (isosorbide + hydralazine) | AAHEFT trial | BENEFICIAL for HF |
| PDE5 inhibitors (sildenafil) | RELAX trial (HFpEF failed) | Mixed - OK for HFrEF |
| Digoxin | DIG trial | Neutral on mortality, reduces hospitalizations |
| Aliskiren | ASTRONAUT trial | Only harmful in diabetics with HF |

**FILTER IMPACT:**
- New rules added: 4 (non-DHP CCB, Class Ic, ganglionic blockers, surgical dyes)
- Total predictions now excluded: 302 (2.2%)
- HIGH tier predictions excluded: 40
- Newly excluded from h250 rules: 91 (9 HIGH tier)

**NEW HYPOTHESES GENERATED (4):**
- h254: Aliskiren-Diabetes HF Subgroup Filter
- h255: Antiarrhythmic Safety Review Beyond Class Ic
- h256: Methylene Blue Therapeutic vs Diagnostic Distinction
- h257: IV vs Oral Formulation Safety Distinction

### Key Learning

CV drug contraindications require evidence-based trial data (CAST, PROMISE, ASTRONAUT). Drugs that SEEM mechanistically plausible (vasodilators for HF) can still be harmful. Non-DHP CCBs and Class Ic antiarrhythmics are commonly predicted for HF/arrhythmias but are HARMFUL based on clinical trial evidence.

### Recommended Next Steps (Priority Order)
1. **h255**: Antiarrhythmic Safety Review Beyond Class Ic
2. **h96**: PPI-Extended Drug Targets
3. **h91**: Literature Mining (high effort but critical for zero-treatment diseases)

---

## Previous Session: h244, h248, h251, h249, h253, h252 (2026-02-05)

See git history for detailed session notes.

### Key findings:
- SGLT2 inhibitors: 63.9% precision (BEST drug class)
- PAH drug safety filters: 23.3% of HF predictions were HARMFUL
- sGC stimulators: Teratogenic - pregnancy filter added

---

## Archive

See previous entries in git history.
