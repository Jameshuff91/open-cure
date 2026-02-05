# Research Loop Progress

## Current Session: h226, h233, h221, h236, h238, h100 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 6**
- h226: Two-Hop Mechanism Paths - **INVALIDATED** (6.3% precision vs 13.3% 1-hop)
- h233: Threshold-Based 2-Hop Paths - **VALIDATED** (marginal: >=15 achieves 10.2%)
- h221: Manual Rule Expansion - **VALIDATED** (existing 30 rules sufficient)
- h236: High-Indication Drug Ranking Gap - **VALIDATED** (kNN favors specialists)
- h238: Category-Restricted kNN - **VALIDATED** (recovers generalist drugs)
- h100: Pathway-Level Drug Matching - **INVALIDATED** (gene overlap is better)

### Cumulative Statistics (2026-02-05)
| Status | Count |
|--------|-------|
| Validated | 130 |
| Invalidated | 48 |
| Inconclusive | 8 |
| Blocked | 18 |
| Deprioritized | 3 |
| Pending | 30 |
| **Total** | **238** |

### Session Key Learnings

1. **h226:** 2-hop PPI paths extend coverage (+37.7%) but precision drops to 6.3% (0.48x of 1-hop). PPI interactions are too non-specific.

2. **h233:** Higher 2-hop thresholds improve precision: >=15 paths = 10.2%, >=20 paths = 11.7%. Marginal value as secondary signal.

3. **h221:** Current 30 manual rules are sufficient. The gap (Aspirin, Metformin not appearing) is a kNN ranking issue, not a data gap.

4. **h236 KEY INSIGHT:** kNN collaborative filtering favors "specialist" drugs over "generalist" drugs:
   - Empagliflozin (9 GT diseases) → 11 predictions
   - Metformin (123 GT diseases) → 0 predictions

5. **h238 SOLUTION:** Category-restricted kNN recovers generalist drugs:
   - Metformin ranks #16 in metabolic category (vs not in top 30 globally)

6. **h100:** Gene overlap is a better signal than pathway overlap:
   - Gene: 2.36x separation, 17.9% precision at threshold >=10
   - Pathway: 1.74x separation, 11.0% precision at threshold >=10

### Session Theme: Mechanism Path Analysis and kNN Limitations

**Mechanism Path Hierarchy:**
1. Direct gene overlap (h100, h166): Best signal, 17.9% precision
2. 1-hop mechanism paths: 13.3% precision (validated in h166)
3. Pathway overlap (h100): Lower precision (11.0%) but higher coverage
4. 2-hop PPI paths (h226): Poor precision (6.3%), adds noise

**kNN Algorithm Insights:**
- Favors specialist drugs over generalist drugs (h236)
- Category-restricted kNN can recover generalists (h238)
- Manual rules help drugs NOT in DRKG, not ranking issues (h221)

### New Hypotheses Generated
- h234: Weighted PPI Path Scoring (pending)
- h235: Same-Pathway 2-Hop filtering (pending)
- h237: Indication-Weighted Drug Boosting (pending)

---

## Archive

See previous entries in git history.
