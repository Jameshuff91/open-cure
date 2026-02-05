# Research Loop Progress

## Current Session: h267 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypothesis Tested: 1**
- h267: Biologic Sparse GT Root Cause Analysis - **VALIDATED**

### Cumulative Statistics
| Status | Count |
|--------|-------|
| Validated | 152 |
| Invalidated | 52 |
| Inconclusive | 8 |
| Blocked | 18 |
| Deprioritized | 3 |
| Pending | 41 |
| **Total** | **274** |

### KEY SESSION FINDINGS

**h267: Biologic Sparse GT Root Cause Analysis - VALIDATED**

**ROOT CAUSE IDENTIFIED:** Cancer drug performance gap is NOT due to sparse GT data

Initial hypothesis was that mAbs/biologics have low precision because GT is sparse.
Analysis revealed this is FALSE:
- mAbs: 4.4 GT entries/drug (same as small molecules)
- The real issue is **domain isolation**

**KEY FINDING: 241 cancer drugs (55%) treat ONLY cancer diseases**
- These drugs are INVISIBLE to kNN collaborative filtering
- kNN requires non-cancer "anchor" diseases to recommend drugs for cancer
- This creates a fundamental ceiling that no amount of GT data fixes

**Theoretical Ceiling for Cancer:**
| Metric | Value |
|--------|-------|
| Total cancer GT entries | 1,370 |
| kNN-reachable | 695 (50.7%) |
| kNN-unreachable | 675 (49.3%) |

**By Drug Class (kNN-reachable %):**
| Class | Reachable | Total | % |
|-------|-----------|-------|---|
| mAbs | 110 | 267 | 41.2% |
| Kinase inhibitors | 77 | 183 | 42.1% |
| Chemotherapy | 106 | 186 | 57.0% |

**IMPLICATION:**
- Low mAb+cancer precision (6.2%) is a fundamental kNN limitation
- Same for kinase inhibitors (2.8%)
- Cancer requires DIFFERENT methodology than kNN
- Target/mechanism-based approaches (like TxGNN) may be better

### New Hypotheses Generated
1. **h269**: Cancer-Specific Target-Based Scoring (priority 2, high impact)
2. **h270**: Cross-Domain Bridge Drug Analysis (priority 3, medium impact)
3. **h271**: Domain-Isolated Drug Detection for Confidence Tiering (priority 3, medium impact)
4. **h272**: GT Expansion: Add Non-Cancer Uses for Cancer Drugs (priority 3, medium impact)

### Recommended Next Steps
1. **h270**: Low-effort analysis of bridge drugs (can improve cancer precision)
2. **h271**: Low-effort confidence tiering improvement
3. **h269**: High-effort but potentially high-impact for oncology

---

## Previous Sessions

See git history for detailed session notes.
