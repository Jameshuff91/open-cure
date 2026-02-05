# Research Loop Progress

## Current Session: h227, h134, h241, h243, h242 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** In Progress
**Hypotheses Tested: 5**
- h227: Hybrid Drug-Class/kNN Routing - **VALIDATED** (+9.4pp hybrid vs kNN)
- h134: Steroid Dominance Analysis in Golden Set - **VALIDATED** (non-steroids 17.7%, calcium blockers 87.5%)
- h241: Steroid-Free Disease Categories Deep Dive - **VALIDATED** (CV/psychiatric 100% for drug classes)
- h243: Psychiatric Contraindication Rules - **VALIDATED** (+12.6pp precision with SSRI/stimulant filters)
- h242: Biologic Precision Rescue via Mechanism Tiers - **INVALIDATED** (no class >=20%)

### Cumulative Statistics (2026-02-05)
| Status | Count |
|--------|-------|
| Validated | 134 |
| Invalidated | 49 |
| Inconclusive | 8 |
| Blocked | 18 |
| Deprioritized | 3 |
| Pending | 35 |
| **Total** | **247** |

### Session Key Learnings

1. **h227:** Hybrid drug-class/kNN routing achieves 39.3% vs 29.9% pure kNN (+9.4pp). Route psychiatric to drug-class, rare diseases to kNN.

2. **h134:** Model captures diverse pharmacology beyond steroids:
   - Calcium blockers: 87.5% precision (best non-steroid)
   - Antipsychotics: 75%
   - Beta blockers: 69.2%
   - ACE inhibitors: 63.6%
   - Cardiovascular and psychiatric have NO steroids but 47-48% precision

3. **h241:** Steroid-free categories (CV, psychiatric) show excellent drug class patterns:
   - CV: Beta blockers, calcium blockers, diuretics, vasodilators all 100% precision
   - Psychiatric: Atypical antipsychotics 90%, typical 80%
   - ~70% of CV novel predictions are clinically plausible
   - ~40% of psychiatric novel predictions are plausible (SSRI/stimulant issue identified)

4. **h243:** Psychiatric contraindication rules improve HIGH confidence from 48.3% → 61.0% (+12.6pp):
   - SSRI/SNRI for bipolar (mania risk)
   - Stimulants for bipolar/schizophrenia (psychosis risk)
   - Local anesthetics for psychiatric (no indication)
   - 5 GT hits lost but they're adjunct uses requiring combination therapy

5. **h242:** Biologic precision rescue via mechanism tiers FAILED:
   - No mechanism class achieves >=20% precision overall
   - But disease-specific routing COULD work (33% for RA, PsA, melanoma)
   - Root cause: GT sparsity (58 entries for 10 biologics vs 271 for prednisone alone)

### New Hypotheses Generated
- h239: Antifungal Drug Repurposing Analysis (from h134)
- h240: Calcium Blocker Cross-Category Repurposing (from h134)
- h244: PAH Drug Transfer to Heart Failure (from h241)
- h245: Emerging Treatments Validation (from h241)
- h246: Adjunct Therapy Detection for Psychiatric (from h243)
- h247: Disease-Specific Biologic Routing (from h242)

### Session Theme: Precision Optimization & Contraindication Rules

**Key Patterns Discovered:**
1. Drug-class routing beats kNN for specific categories (psychiatric +23pp, neurological +6pp)
2. Non-steroid drug classes achieve steroid-level precision (calcium blockers 87.5%)
3. Contraindication rules are essential for psychiatric safety (SSRIs/stimulants for bipolar)
4. Biologics need disease-specific routing, not mechanism tiering

**Production Recommendations:**
1. Integrate psychiatric contraindication rules into confidence filter
2. Use hybrid routing: drug-class for psychiatric/neurological, kNN for rare diseases
3. Prioritize cardiovascular novel predictions for validation (70% clinically plausible)
4. Consider disease-routing for biologics rather than excluding them

---

## Archive

See previous entries in git history.
