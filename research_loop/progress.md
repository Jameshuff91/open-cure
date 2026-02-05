# Research Loop Progress

## Current Session: h227, h134, h241, h243, h242, h247 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** In Progress
**Hypotheses Tested: 6**
- h227: Hybrid Drug-Class/kNN Routing - **VALIDATED** (+9.4pp hybrid vs kNN)
- h134: Steroid Dominance Analysis in Golden Set - **VALIDATED** (non-steroids 17.7%, calcium blockers 87.5%)
- h241: Steroid-Free Disease Categories Deep Dive - **VALIDATED** (CV/psychiatric 100% for drug classes)
- h243: Psychiatric Contraindication Rules - **VALIDATED** (+12.6pp precision with SSRI/stimulant filters)
- h242: Biologic Precision Rescue via Mechanism Tiers - **INVALIDATED** (no class >=20%)
- h247: Disease-Specific Biologic Routing - **VALIDATED** (mechanism matching 64.7%, 10x improvement!)

### Cumulative Statistics (2026-02-05)
| Status | Count |
|--------|-------|
| Validated | 135 |
| Invalidated | 49 |
| Inconclusive | 8 |
| Blocked | 18 |
| Deprioritized | 3 |
| Pending | 34 |
| **Total** | **247** |

### Session Key Learnings

1. **h227:** Hybrid drug-class/kNN routing achieves 39.3% vs 29.9% pure kNN (+9.4pp).

2. **h134:** Model captures diverse pharmacology beyond steroids:
   - Calcium blockers: 87.5% precision (best non-steroid)
   - Antipsychotics: 75%, Beta blockers: 69.2%, ACE inhibitors: 63.6%

3. **h241:** Steroid-free categories (CV, psychiatric) show excellent drug class patterns:
   - CV: Beta blockers, calcium blockers, diuretics, vasodilators all 100% precision
   - Psychiatric: Atypical antipsychotics 90%, typical 80%

4. **h243:** Psychiatric contraindication rules improve HIGH confidence from 48.3% → 61.0% (+12.6pp)

5. **h242:** Biologic mechanism tiering FAILED - no class >=20% precision overall

6. **h247 BREAKTHROUGH:** Biologic mechanism matching transforms precision:
   - Baseline: 5.9%
   - Tier 1 diseases: 30.4%
   - Mechanism matching: **64.7%** (11/17 hits)
   - 10x improvement - biologics become high-value predictions!

### New Hypotheses Generated
- h239-h247 (7 new hypotheses from findings)

### Session Theme: Precision Optimization via Rules & Routing

**Production Recommendations:**
1. Implement psychiatric contraindication rules (h243)
2. Implement biologic mechanism matching (h247) - 10x precision improvement
3. Use hybrid routing: drug-class for psychiatric/neurological, kNN for rare diseases (h227)
4. Prioritize cardiovascular novel predictions for validation (h241)

---

## Archive

See previous entries in git history.
