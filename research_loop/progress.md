# Research Loop Progress

## Current Session: h226, h233, h221, h236 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** In Progress
**Hypotheses Tested: 4**
- h226: Two-Hop Mechanism Paths - **INVALIDATED** (6.3% precision vs 13.3% 1-hop)
- h233: Threshold-Based 2-Hop Paths - **VALIDATED** (marginal: >=15 achieves 10.2%)
- h221: Manual Rule Expansion - **VALIDATED** (existing 30 rules sufficient)
- h236: High-Indication Drug Ranking Gap - **VALIDATED** (kNN favors specialists)

### Cumulative Statistics (2026-02-05)
| Status | Count |
|--------|-------|
| Validated | 129 |
| Invalidated | 47 |
| Inconclusive | 8 |
| Blocked | 18 |
| Deprioritized | 3 |
| Pending | 32 |
| **Total** | **238** |

### Session Key Learnings

1. **h226:** 2-hop PPI paths extend coverage (+37.7%) but precision drops to 6.3% (0.48x of 1-hop). PPI interactions are too non-specific.

2. **h233:** Higher 2-hop thresholds improve precision: >=15 paths = 10.2%, >=20 paths = 11.7%. Marginal value as secondary signal.

3. **h221:** Current 30 manual rules are sufficient. The gap (Aspirin, Metformin not appearing) is a kNN ranking issue, not a data gap.

4. **h236 KEY INSIGHT:** kNN collaborative filtering favors "specialist" drugs over "generalist" drugs:
   - Empagliflozin (9 GT diseases) → 11 predictions
   - Metformin (123 GT diseases) → 0 predictions
   - This is a design property of frequency-based collaborative filtering

### Session Theme: Understanding kNN Limitations

**2-Hop Mechanism Paths:**
- Extending mechanism paths via PPI adds noise, not signal
- Even high thresholds (>=15 paths) only achieve 10.2% precision
- Direct 1-hop paths remain the gold standard for mechanism support

**Manual Rule Limitations:**
- Manual rules help drugs NOT in DRKG
- Common drugs like Metformin ARE in DRKG but don't rank highly
- This requires algorithm changes, not data additions

**kNN Specialist Bias (Major Finding):**
- Drugs treating many diverse diseases get diluted in kNN scoring
- Drugs treating few focused diseases cluster and score highly
- This explains why broadly-applicable drugs are systematically missed

### New Hypotheses Generated
- h233: Threshold-Based 2-Hop Paths (completed same session)
- h234: Weighted PPI Path Scoring (pending)
- h235: Same-Pathway 2-Hop filtering (pending)
- h237: Indication-Weighted Drug Boosting (pending)
- h238: Category-Restricted kNN for Generalist Drug Recovery (pending)

---

## Archive

See previous entries in git history.
