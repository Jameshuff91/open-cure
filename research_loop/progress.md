# Research Loop Progress

## Current Session: h297, h298, h293, h286, h299, h162 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** In Progress
**Hypotheses Tested: 6**
- h297: Mechanism-Specific Disease Categories - **VALIDATED**
- h298: Implement Mechanism-Specificity Confidence Signal - **VALIDATED**
- h293: Inverse Complication Filter Analysis - **VALIDATED**
- h286: Mechanistic Pathway Overlap - **BLOCKED** (ID format mismatch)
- h299: Alternative Methods for Mechanism-Specific Diseases - **BLOCKED** (ID format mismatch)
- h162: Precision-Coverage Trade-off Quantification - **VALIDATED**

### Cumulative Statistics
| Status | Count |
|--------|-------|
| Validated | 176 |
| Invalidated | 55 |
| Inconclusive | 10 |
| Blocked | 20 |
| Deprioritized | 3 |
| Pending | 37 |
| **Total** | **301** (4 new hypotheses added this session)

### KEY SESSION FINDINGS

#### h297: Mechanism-Specific Disease Categories - VALIDATED

**Disease Categories by kNN Potential:**
| Category | Criteria | Count | kNN Expected |
|----------|----------|-------|--------------|
| MECHANISM_SPECIFIC | breadth<3, mono>70% | 50 (1.9%) | FAIL |
| HIGHLY_REPURPOSABLE | breadth≥5, mono<40% | 452 (16.8%) | SUCCESS |
| MODERATE | Neither | 846 (31.5%) | VARIABLE |

**Key Metric:** Mechanism-specific diseases have only **6.6%** repurposable GT drugs vs **87.9%** for other diseases (81.3 pp difference!)

#### h298: Implement Mechanism-Specificity Confidence Signal - VALIDATED

**Implementation in production_predictor.py:**
- Added `MECHANISM_SPECIFIC_DISEASES` set (53 diseases)
- Added `HIGHLY_REPURPOSABLE_DISEASES` set (45 diseases)
- Mechanism-specific → capped at LOW tier
- Highly repurposable → boosted to MEDIUM if otherwise LOW

#### h293: Inverse Complication Filter Analysis - VALIDATED

**Overall base→comp precision: 2.9%** (119/4092) - confirms h280 filter is correct

| Complication Type | Precision | Examples |
|-------------------|-----------|----------|
| Organ-specific | **0.0%** | Retinopathy, neuropathy, macular edema |
| Pathway-comprehensive | **16.1%** | MI (70%), stroke (60%), DKA (43.5%) |

#### h286 & h299: BLOCKED - ID Format Mismatch

**Critical Infrastructure Gap:**
- GT uses CHEBI/MONDO/UMLS IDs
- Reference data uses DrugBank/CHEMBL/MESH IDs
- **Zero overlap** between ID formats
- Blocks all target-based and pathway-based analysis
- Added h301 for ID mapping infrastructure

#### h162: Precision-Coverage Trade-off - VALIDATED

**Pareto-Optimal Finding:**
GOLDEN tier is Pareto-optimal for ALL categories.
Adding more tiers decreases precision (no free lunch).

**Recommended Operating Modes:**
1. **Clinical (>50%)**: GOLDEN only → 55-75% precision, 5% coverage
2. **Research (>30%)**: GOLDEN+HIGH → 29-43% precision, 20% coverage
3. **Exploration (>15%)**: GOLDEN+HIGH+MEDIUM → 20-63% precision, 60% coverage

**Anomalies:**
- Autoimmune MEDIUM: 77.8% precision (higher than blended!)
- Psychiatric MEDIUM: 85.0% precision (highest overall!)

### New Hypotheses Generated
- **h301**: Build Drug/Disease ID Mapping Infrastructure (high effort)

### Recommended Next Steps
1. h261 or h272 for next testable hypothesis
2. Consider h301 if ID mapping becomes strategic priority
3. Run full evaluation to measure cumulative impact of h297+h298+h291+h293

---

## Previous Session: h284, h288, h292, h159, h177, h295, h296, h291 (2026-02-05)

**Hypotheses Tested: 8**
- h284: Complication Specialization Score - **VALIDATED**
- h288: ATC Class-Supported GOLDEN Tier - **INVALIDATED**
- h292: Cardiovascular Event Transferability - **VALIDATED**
- h159: Category Boundary Refinement - **INCONCLUSIVE**
- h177: Epilepsy-Specific Analysis - **VALIDATED**
- h295: Drug Pool Size as Confidence Signal - **VALIDATED**
- h296: Statin-Only CV Predictions - **VALIDATED** (100% vs 0%)
- h291: Implement Comp→Base Confidence Boost - **VALIDATED**

---

## Previous Sessions

See git history for detailed session notes.
