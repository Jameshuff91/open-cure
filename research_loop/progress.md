# Research Loop Progress

## Current Session: h297, h298, h293 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** In Progress
**Hypotheses Tested: 3**
- h297: Mechanism-Specific Disease Categories - **VALIDATED**
- h298: Implement Mechanism-Specificity Confidence Signal - **VALIDATED**
- h293: Inverse Complication Filter Analysis - **VALIDATED**

### Cumulative Statistics
| Status | Count |
|--------|-------|
| Validated | 174 |
| Invalidated | 55 |
| Inconclusive | 10 |
| Blocked | 18 |
| Deprioritized | 3 |
| Pending | 38 |
| **Total** | **298** (3 new hypotheses added this session)

### KEY SESSION FINDINGS

#### h297: Mechanism-Specific Disease Categories - VALIDATED

**Disease Categories by kNN Potential:**
| Category | Criteria | Count | kNN Expected |
|----------|----------|-------|--------------|
| MECHANISM_SPECIFIC | breadth<3, mono>70% | 50 (1.9%) | FAIL |
| HIGHLY_REPURPOSABLE | breadth≥5, mono<40% | 452 (16.8%) | SUCCESS |
| MODERATE | Neither | 846 (31.5%) | VARIABLE |

**Key Metric:** Mechanism-specific diseases have only **6.6%** repurposable GT drugs vs **87.9%** for other diseases (81.3 pp difference!)

**Examples:**
- Mechanism-specific: ALS (breadth=1.0), Fabry (1.0), Huntington (2.0), HIV (1.9)
- Highly repurposable: Lupus (breadth=95.5), Nephrotic syndrome (89.7), Trichinellosis (92.6)

#### h298: Implement Mechanism-Specificity Confidence Signal - VALIDATED

**Implementation in production_predictor.py:**
- Added `MECHANISM_SPECIFIC_DISEASES` set (53 diseases)
- Added `HIGHLY_REPURPOSABLE_DISEASES` set (45 diseases)
- Mechanism-specific → capped at LOW tier
- Highly repurposable → boosted to MEDIUM if otherwise LOW

**Testing Results:**
- ALS predictions: ALL LOW (correct - mechanism-specific)
- Fabry predictions: ALL LOW (correct - mechanism-specific)
- Lupus predictions: HIGH tier (correct - highly repurposable)

#### h293: Inverse Complication Filter Analysis - VALIDATED

**Overall base→comp precision: 2.9%** (119/4092) - confirms h280 filter is correct

**Complication Types:**
| Type | Precision | Examples |
|------|-----------|----------|
| Organ-specific | **0.0%** | Retinopathy, neuropathy, macular edema |
| Pathway-comprehensive | **16.1%** | MI (70%), stroke (60%), DKA (43.5%) |

**Key Finding:** Organ-specific complications ALWAYS require specialized treatment. Base disease treatment never helps. Pathway-comprehensive exceptions are explained by comprehensive drugs (statins, insulin).

### New Hypotheses Generated
- **h298**: Implement mechanism-specificity signal (COMPLETED)
- **h299**: Alternative methods for mechanism-specific diseases
- **h300**: HIV drug network analysis

### Recommended Next Steps
1. Continue with h261 (Pathway-Weighted PPI Scoring) or h286 (Mechanistic Pathway Overlap)
2. Run full evaluation to measure cumulative impact of h297+h298+h291 implementations
3. Consider h299 - what methods work for mechanism-specific diseases where kNN fails?

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

## Previous Session: h281, h193, h280, h290 (2026-02-05)

**Hypotheses Tested: 4**
- h281: Bidirectional Treatment Analysis - **VALIDATED** (4x asymmetry)
- h193: Combined ATC Coherence Signals - **INVALIDATED**
- h280: Complication vs Subtype Classification - **VALIDATED** (42.6% vs 13.9%)
- h290: Implement Relationship Type Filter - **VALIDATED**

---

## Previous Sessions

See git history for detailed session notes.
