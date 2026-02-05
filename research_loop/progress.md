# Research Loop Progress

## Current Session: h227, h134, h241, h243, h242, h247, h240 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** In Progress
**Hypotheses Tested: 7**
- h227: Hybrid Drug-Class/kNN Routing - **VALIDATED** (+9.4pp hybrid vs kNN)
- h134: Steroid Dominance Analysis in Golden Set - **VALIDATED** (non-steroids 17.7%, calcium blockers 87.5%)
- h241: Steroid-Free Disease Categories Deep Dive - **VALIDATED** (CV/psychiatric 100% for drug classes)
- h243: Psychiatric Contraindication Rules - **VALIDATED** (+12.6pp precision with SSRI/stimulant filters)
- h242: Biologic Precision Rescue via Mechanism Tiers - **INVALIDATED** (no class >=20%)
- h247: Disease-Specific Biologic Routing - **VALIDATED** (mechanism matching 64.7%, 10x improvement!)
- h240: Calcium Blocker Cross-Category Repurposing - **VALIDATED** (75% neurological precision)

### Cumulative Statistics (2026-02-05)
| Status | Count |
|--------|-------|
| Validated | 136 |
| Invalidated | 49 |
| Inconclusive | 8 |
| Blocked | 18 |
| Deprioritized | 3 |
| Pending | 33 |
| **Total** | **247** |

### Session Key Learnings

1. **h227:** Hybrid drug-class/kNN routing achieves 39.3% vs 29.9% pure kNN (+9.4pp).

2. **h134:** Model captures diverse pharmacology beyond steroids (calcium blockers 87.5%)

3. **h241:** Steroid-free categories show excellent drug class patterns (CV/psychiatric 100%)

4. **h243:** Psychiatric contraindication rules +12.6pp (SSRI/stimulants for bipolar/schizophrenia)

5. **h242→h247:** Biologic mechanism tiering failed but disease-specific routing succeeded:
   - Mechanism matching: 5.9% → 64.7% (10x improvement!)
   - Biologics become high-value when matched to approved indications

6. **h240:** CCB cross-category repurposing validated:
   - Neurological (migraine, epilepsy): 75% precision
   - Verapamil → migraine is FDA-approved (model correctly predicts)

### Session Theme: Precision Optimization via Rules & Routing

**Production Recommendations:**
1. Implement psychiatric contraindication rules (h243) - +12.6pp
2. Implement biologic mechanism matching (h247) - 10x precision improvement
3. Use hybrid routing: drug-class for psychiatric/neurological (h227)
4. Prioritize CCB predictions for neurological validation (h240)
5. Cardiovascular and psychiatric are steroid-free with high precision (h241)

---

## Archive

See previous entries in git history.
