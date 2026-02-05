# Research Loop Progress

## Current Session: h267, h270 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 2**
- h267: Biologic Sparse GT Root Cause Analysis - **VALIDATED**
- h270: Cross-Domain Bridge Drug Analysis - **VALIDATED**

### Cumulative Statistics
| Status | Count |
|--------|-------|
| Validated | 153 |
| Invalidated | 52 |
| Inconclusive | 8 |
| Blocked | 18 |
| Deprioritized | 3 |
| Pending | 44 |
| **Total** | **278** |

### KEY SESSION FINDINGS

#### h267: Biologic Sparse GT Root Cause Analysis - VALIDATED

**ROOT CAUSE IDENTIFIED:** Cancer drug performance gap is NOT due to sparse GT data

- mAbs: 4.4 GT entries/drug (same as small molecules)
- The real issue is **domain isolation**: 241 cancer drugs (55%) treat ONLY cancer

**Theoretical kNN Ceiling for Cancer:**
| Metric | Value |
|--------|-------|
| kNN-reachable | 695 (50.7%) |
| kNN-unreachable | 675 (49.3%) |

**Implication:** kNN fundamentally cannot recommend cancer-only drugs for cancer.

#### h270: Cross-Domain Bridge Drug Analysis - VALIDATED

**MAJOR DISCOVERY: GT Granularity Problem**

What appeared as 1.1% precision for cancer predictions is actually ~35% when accounting for disease hierarchy:
- GT uses generic terms ("lymphoma")
- Predictions use specific subtypes ("DLBCL")
- This is a MEASUREMENT ISSUE, not a model failure

**Precision Rules Validated:**
| Rule | Precision | Count |
|------|-----------|-------|
| Same cancer type + score > 1.0 | 100% | 177 |
| Drug has cancer GT (any type) | 56.8% | 1650 |
| Different cancer type | 30.6% | 1371 |
| Drug has NO cancer GT | 0% | 239 |

**Actionable Rules:**
- GOLDEN: Same cancer type + score > 1.0 → 100% precision
- FILTER: Drugs with no cancer GT → 0% precision
- Cross-type repurposing: ~30% precision (better than we thought!)

### New Hypotheses Generated
1. **h269**: Cancer-Specific Target-Based Scoring (priority 2)
2. **h270-273**: Multiple follow-ups from bridge drug analysis
4. **h274**: Cancer Type Confidence Tier Implementation (priority 2)
5. **h275**: Subtype Refinement vs Novel Discovery Classification (priority 3)

### Recommended Next Steps
1. **h274**: Implement cancer type confidence tiering (low effort, high precision gain)
2. **h273**: Apply hierarchy matching to all categories (medium effort, high impact)
3. **h271**: Domain-isolated drug detection (low effort)

---

## Previous Sessions

See git history for detailed session notes.
