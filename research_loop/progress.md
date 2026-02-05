# Research Loop Progress

## Current Session: h96, h259, h260, h262, h263, h264 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 6**
- h96: PPI-Extended Drug Targets - **VALIDATED** (2.89x selectivity, 55% ceiling)
- h259: PPI Mechanism Support as Confidence Tier - **VALIDATED** (2.62x precision lift)
- h260: Hybrid kNN + Mechanism Gating - **INVALIDATED** (no R@30 improvement)
- h262: Drug Class PPI Patterns - **VALIDATED** (CV 33.7x lift, category-specific)
- h263: Category-Specific Mechanism Requirements - **VALIDATED** (CV +47%, Neuro +62%)
- h264: Mechanism-Only Predictions for CV - **INVALIDATED** (9.5% R@30 < 15% threshold)

### Cumulative Statistics (2026-02-05)
| Status | Count |
|--------|-------|
| Validated | 149 |
| Invalidated | 52 |
| Inconclusive | 8 |
| Blocked | 18 |
| Deprioritized | 3 |
| Pending | 36 |
| **Total** | **266** |

### Session Theme: PPI Mechanism Integration

### CRITICAL SESSION INSIGHT

**Mechanism support serves PRECISION, not RECALL:**
- h259: 2.62x precision lift (18.9% vs 7.2%) - USE THIS
- h262: Category-specific lifts (CV 33.7x, Neuro 13.3x) - USE THIS
- h263: CV precision +7.3pp (47%), Neuro +5.8pp (62%) - USE THIS
- h260: No R@30 improvement from mechanism boosting - DON'T USE
- h264: Mechanism-only R@30 = 9.5% (below threshold) - DON'T USE

### Production Recommendations

1. **ADD mechanism as tier modifier:**
   - WITH mechanism: +1-2 tier boost
   - Results in GOLDEN 32.3%, HIGH 17.8%, MEDIUM 11.3% precision

2. **REQUIRE mechanism for CV/Neuro:**
   - Exclude predictions without mechanism support
   - 236 predictions excluded, only 2 GT lost (0.8%)
   - CV precision: 15.3% → 22.6% (+47%)
   - Neuro precision: 9.5% → 15.3% (+62%)

3. **Keep kNN for ranking:**
   - Mechanism doesn't improve R@30
   - kNN captures collaborative filtering signal
   - Mechanism filters false positives from kNN output

### Key Findings Summary

| Finding | Metric | Implication |
|---------|--------|-------------|
| PPI selectivity | 2.89x | Use as confidence filter |
| Overall precision lift | 2.62x | Add mechanism tier |
| CV lift | 33.7x | Require mechanism |
| Neuro lift | 13.3x | Require mechanism |
| Infectious lift | 1.8x | Mechanism optional |
| Mechanism R@30 boost | +0pp | Don't use for ranking |
| CV mechanism-only R@30 | 9.5% | Below threshold |

### Recommended Next Steps (Priority Order)
1. **h261**: Pathway-Weighted PPI Scoring (medium effort)
2. **h91**: Literature Mining for zero-treatment diseases (high effort, high impact)
3. **h193**: Combined ATC Coherence Signals (medium effort)

---

## Previous Sessions

### 2026-02-05 (Earlier): h250, h255, h258
Safety filter enhancement session - see git history.
