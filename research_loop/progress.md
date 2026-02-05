# Research Loop Progress

## Current Session: h284, h288, h292, h159, h177 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 5**
- h284: Complication Specialization Score - **VALIDATED**
- h288: ATC Class-Supported GOLDEN Tier - **INVALIDATED**
- h292: Cardiovascular Event Transferability - **VALIDATED**
- h159: Category Boundary Refinement - **INCONCLUSIVE**
- h177: Epilepsy-Specific Analysis - **VALIDATED**

### Cumulative Statistics
| Status | Count |
|--------|-------|
| Validated | 168 |
| Invalidated | 55 |
| Inconclusive | 10 |
| Blocked | 18 |
| Deprioritized | 3 |
| Pending | 43 |
| **Total** | **297** (7 new hypotheses added this session)

### KEY SESSION FINDINGS

#### h284: Complication Specialization Score - VALIDATED

| Direction | Tier | N | Precision |
|-----------|------|---|-----------|
| Comp→Base | HIGH transferability | 8 | **62.5%** |
| Comp→Base | UNKNOWN | 12 | **41.7%** |
| Base→Comp | ALL tiers | 9 | **0.0%** |

**Key Finding:** Transferability score predicts comp→base precision (62.5% vs 41.7%), but base→comp is ALWAYS 0% regardless. Confirms h290 filter is correct.

#### h292: Cardiovascular Event Transferability - VALIDATED

| Event Type | N | Transferability | Category |
|------------|---|-----------------|----------|
| Stroke/TIA | 83 | 50-72% | HIGH |
| MI | 92 | 18.5% | LOW |
| Angina | 16 | 12.5% | LOW |

**Key Finding:** MI has LOW transferability (18.5%) despite being THE classic atherosclerosis complication. Statins are the ONLY drugs that correctly predict event→atherosclerosis.

#### h177: Epilepsy-Specific Analysis - VALIDATED

| Category | N Drugs | Mean Breadth | kNN Expected |
|----------|---------|--------------|--------------|
| Diabetes | 175 | 8.1 | GOOD |
| Hypertension | 161 | 5.3 | GOOD |
| Epilepsy | 37 | 5.8 | GOOD |
| Alzheimer's | 15 | 3.4 | POOR |

**Key Finding:** Drug pool size and breadth predict kNN success. Epilepsy drugs have cross-neurological utility (psychiatric, pain, movement). Alzheimer's drugs are mechanism-specific (no repurposability).

#### h288: ATC Class GOLDEN Tier - INVALIDATED
Pre-check from h193: Coherent+classmate precision = 9.0%, far below 40% GOLDEN threshold.

#### h159: Category Boundary Refinement - INCONCLUSIVE
ITP/TTP are autoimmune-hematological but GT is tiny (2-4 drugs). Expansion would add ~1 correct prediction.

### New Hypotheses Generated
- **h291-h297**: Transferability implementation, CV event expansion, drug pool confidence signals, statin-only CV predictions, mechanism-specific categories

### Recommended Next Steps
1. h295 (Drug Pool Size as Confidence Signal) - builds on h177
2. h296 (Statin-Only CV Predictions) - builds on h292
3. h291 (Implement Comp→Base Boost) - builds on h284

---

## Previous Session: h281, h193, h280, h290 (2026-02-05)

**Hypotheses Tested: 4**
- h281: Bidirectional Treatment Analysis - **VALIDATED** (4x asymmetry)
- h193: Combined ATC Coherence Signals - **INVALIDATED**
- h280: Complication vs Subtype Classification - **VALIDATED** (42.6% vs 13.9%)
- h290: Implement Relationship Type Filter - **VALIDATED**

---

## Previous Session: h279, h277, h282 (2026-02-05)

**Hypotheses Tested: 3**
- h279: Disease Specificity Scoring - **VALIDATED**
- h277: Cross-Category Hierarchy Matching - **INVALIDATED**
- h282: Hierarchy Depth Delta - **VALIDATED**

---

## Previous Sessions

See git history for detailed session notes.
