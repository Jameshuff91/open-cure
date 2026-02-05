# Research Loop Progress

## Current Session: h96-h264, h101 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 7**
- h96: PPI-Extended Drug Targets - **VALIDATED** (2.89x selectivity)
- h259: PPI Mechanism as Confidence Tier - **VALIDATED** (2.62x precision lift)
- h260: Hybrid kNN + Mechanism - **INVALIDATED** (no R@30 improvement)
- h262: Drug Class PPI Patterns - **VALIDATED** (CV 33.7x lift)
- h263: Category-Specific Mechanism Rules - **VALIDATED** (CV +47%, Neuro +62%)
- h264: Mechanism-Only for CV - **INVALIDATED** (9.5% R@30 < 15% threshold)
- h101: Mechanism Class Annotation - **VALIDATED** (steroid+autoimmune=76.4%)

### Cumulative Statistics
| Status | Count |
|--------|-------|
| Validated | 150 |
| Invalidated | 52 |
| Inconclusive | 8 |
| Blocked | 18 |
| Deprioritized | 3 |
| Pending | 34 |
| **Total** | **265** |

### Session Theme: Precision Optimization

### KEY SESSION FINDINGS

**1. PPI Mechanism = PRECISION signal, NOT RECALL:**
- h259: 2.62x precision lift (18.9% vs 7.2%)
- h260: No R@30 improvement from mechanism boosting
- h264: Mechanism-only R@30 = 9.5% (below threshold)

**2. Category-Specific Rules (h262, h263):**
- CV: 33.7x lift, require mechanism → +47% precision
- Neuro: 13.3x lift, require mechanism → +62% precision
- Infectious: 1.8x lift, mechanism optional

**3. Drug Class × Disease Category (h101):**
- Steroid + autoimmune: 76.4% precision (BEST!)
- Statin + metabolic: 68.0%
- mAb + cancer: 1.6% (despite intuition)
- Kinase inhibitor + cancer: 4.6%

### Production Recommendations

1. **Add mechanism tier modifier:**
   - WITH mechanism: boost tier
   - GOLDEN 32.3%, HIGH 17.8%, MEDIUM 11.3%

2. **Require mechanism for CV/Neuro:**
   - 236 excluded, only 2 GT lost (0.8%)

3. **Drug class modifiers:**
   - BOOST: Steroid+autoimmune, Statin+metabolic
   - FILTER/WARN: mAbs, Kinase inhibitors

### Recommended Next Steps
1. **h265**: Drug Class-Based Tier Modifier (low effort)
2. **h261**: Pathway-Weighted PPI Scoring (medium effort)
3. **h91**: Literature Mining (high effort, high impact)

---

## Previous Sessions

See git history for detailed session notes.
