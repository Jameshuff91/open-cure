# Research Loop Progress

## Current Session: h96, h259, h260 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** In Progress
**Hypotheses Tested: 3**
- h96: PPI-Extended Drug Targets - **VALIDATED** (2.89x selectivity, 55% ceiling)
- h259: PPI Mechanism Support as Confidence Tier - **VALIDATED** (2.62x precision lift)
- h260: Hybrid kNN + Mechanism Gating - **INVALIDATED** (no R@30 improvement)

### Cumulative Statistics (2026-02-05)
| Status | Count |
|--------|-------|
| Validated | 147 |
| Invalidated | 51 |
| Inconclusive | 8 |
| Blocked | 18 |
| Deprioritized | 3 |
| Pending | 36 |
| **Total** | **263** |

### Session Theme: PPI Mechanism Integration

**h96: PPI-Extended Drug Targets**
- PPI extension doubles theoretical R@30 ceiling (22% → 55%)
- But actual gains modest (+1.43pp) due to false positive dilution
- Key finding: 2.89x selectivity ratio (83.8% GT vs 29.0% non-GT)
- Conclusion: Use as CONFIDENCE FILTER, not primary ranking

**h259: PPI Mechanism Support as Confidence Tier**
- WITH mechanism: 18.9% precision, WITHOUT: 7.2%
- 2.62x overall precision lift (p < 0.000001)
- By tier: HIGH+MECH=30.4%, MEDIUM+MECH=17.7%, HIGH=13.5%
- MEDIUM+MECH > HIGH - suggests tier reorganization
- Conclusion: Add mechanism as tier modifier

**h260: Hybrid kNN + Mechanism Gating**
- Continuous weighting: Best alpha=0.2, +0.23pp (p=0.78, not significant)
- Higher alpha = WORSE performance (up to -3pp)
- Binary filter: -4.1pp (p<0.05, significant HARM)
- Root cause: kNN already captures drug-disease relationships
- Conclusion: Mechanism useful for PRECISION, not RECALL

### Key Insight

**Mechanism support serves different purposes:**
- PRECISION: 2.62x lift when used for tier stratification (h259)
- RECALL: No improvement when used for re-ranking (h260)

PPI mechanism support should be used to identify HIGH-CONFIDENCE predictions, not to change rankings.

### Recommended Next Steps (Priority Order)
1. **h261**: Pathway-Weighted PPI Scoring (medium effort) - may reduce noise
2. **h262**: Drug Class PPI Patterns (low effort) - understand where mechanism works
3. **h91**: Literature Mining for zero-treatment diseases (high effort, high impact)

---

## Previous Sessions

### 2026-02-05 (Earlier): h250, h255, h258
Safety filter enhancement session - see git history.

### 2026-02-05 (Earlier): h244, h248, h251, h249, h253, h252
SGLT2 inhibitors, PAH safety filters, sGC teratogenicity - see git history.
