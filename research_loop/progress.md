# Research Loop Progress

## Current Session: h508/h481/h518/h516 - Self-Ref + Literature Status + SOC Holdout (2026-02-06)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 4**
- h508: Self-Referential Disease Characterization - **VALIDATED** (GT size is dominant predictor)
- h481: Deliverable Literature Validation Status Column - **VALIDATED** (+28.4pp precision for SOC)
- h518: SOC Status as Holdout Precision Signal - **VALIDATED** (annotation only, not promotable)
- h516: Expand SOC Drug Class Mappings - **INVALIDATED** (0% precision for all 7 proposed)

### Key Findings

#### 1. Self-Referentiality Characterization (h508)
- **GT size is the DOMINANT predictor**: 79.2% of self-ref diseases have GT ≤ 2 (OR=8.6x)
- Category modulates: GI/immunological 89-100% vs autoimmune/dermatological 20-33%
- 11 "Therapeutic Islands" (GT>5, 100% self-ref): immunodeficiency, PAH, CKD, HepC, opioid constipation
- Two distinct causes: small GT (79%) and dedicated drug classes (8%)

#### 2. Literature Status Classification (h481)
- Added `literature_status` + `soc_drug_class` columns to deliverable
- 17 drug class SOC mappings (184 drugs), 1,651 as LIKELY_GT_GAP (11.7%)
- Full-data: SOC +28.4pp vs NOVEL in HIGH tier (40.3% vs 11.9%)
- Also regenerated JSON deliverable (was stale from old script)

#### 3. SOC Holdout Validation (h518)
- MEDIUM: SOC 20.3% vs NOVEL 14.3% (+6.0pp, p=0.005)
- HIGH: SOC 25.5% ≈ NOVEL 25.7% (NO SIGNAL)
- MEDIUM SOC (20.3%) << HIGH avg (51.5%) → NOT promotable
- **Conclusion**: SOC is ANNOTATION signal, not tier promotion signal

#### 4. SOC Expansion Fails (h516)
- All 7 proposed new drug classes have 0% precision
- Tetracyclines→infectious: 0/248 (0%), macrolides: 0/129, etc.
- Current 17 SOC classes are well-calibrated; expansion dilutes signal
- **Insight**: SOC captures BROAD classes (corticosteroids, statins), not specific ones

### New Hypotheses Generated (5)
- h516: INVALIDATED
- h517: Therapeutic island annotation (P5, low)
- h518: VALIDATED as annotation
- h519: CV pathway holdout re-evaluation (P5, low)
- h520: SOC class-specific holdout precision (P4, medium)

### Recommended Next Steps
1. **h520**: Which SOC drug classes drive the +6pp MEDIUM signal? Could identify class-specific promotions
2. **h517**: Annotate therapeutic islands in deliverable
3. **h486**: Systematic adverse effect mining from SIDER database (high effort but high impact safety)

---

## Previous Session: h507/h492/h509/h515 - Self-Referentiality + GT Expansion + Baseline Re-Calibration (2026-02-06)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 5**
- h507: Predictable Self-Referentiality - **INVALIDATED** (GT-free features reverse on holdout)
- h492: GT Expansion for FDA-Approved Pairs - **VALIDATED** (15 pairs added, negligible impact)
- h509: CV Coronary Hierarchy Demotion - **INVALIDATED** (small-n, do not demote)
- h509 extended: HIGH Tier Per-Rule Holdout Audit - **VALIDATED** (identified underperforming rules)
- h515: Diabetes Hierarchy Split - **INVALIDATED** (only 8 complication predictions)

### Key Findings

#### 1. Self-Referentiality is NOT Predictable (h507)
- Best GT-free feature: same_cat_frac (AUC=0.734 on full data)
- Combined features: AUC=0.781
- **REVERSES on holdout**: -6.0pp ± 12.9pp gap (inconsistent direction)
- Root cause: low same_cat_frac captures TWO opposite populations
  - 58% truly self-referential (bad on holdout)
  - 42% genuine cross-category transfer (good on holdout)
- No GT-free feature can separate them

#### 2. GT is Already Complete (h492)
- Audited 20 major diseases across 7 categories (167 pairs checked)
- Only 15 FDA-approved pairs missing
- 14/15 NOT in model's top-30 predictions
- **GT incompleteness is NOT the precision bottleneck**
- Model limitations are structural (kNN coverage)

#### 3. Holdout Baseline Has Drifted (h492 discovery)
| Tier | Previous (CLAUDE.md) | Current (h492 re-baseline) | Delta |
|------|---------------------|---------------------------|-------|
| GOLDEN | 67.0% ± 20.6% | 63.3% ± 23.2% | -3.7pp |
| HIGH | 60.8% ± 7.2% | 51.5% ± 5.3% | -9.3pp |
| MEDIUM | 32.1% ± 3.6% | 29.9% ± 2.8% | -2.2pp |
| LOW | 12.9% ± 1.4% | 12.3% ± 1.4% | -0.6pp |
| FILTER | 10.3% ± 1.1% | 8.9% ± 1.1% | -1.4pp |

Drift caused by accumulated code changes since h478 (69 insertions, 45 deletions).

#### 4. CV Hierarchy Groups All Too Small (h509)
- Coronary: 5 diseases, arrhythmia: 3, hypertension: 4
- Full-data precision high (65-93%) but holdout unreliable (n≈1/seed)
- DO NOT demote — these encode genuine medical knowledge

#### 5. HIGH Tier Per-Rule Audit (h509 extended)
- Several rules underperforming: respiratory (20%), thyroid (26%), diabetes (21%)
- But most have n<15 across 5 seeds → too small for reliable demotion
- Diabetes hierarchy: only 8 complication predictions → not worth splitting (h515)

### New Hypotheses Generated (5)
- h510: Cross-Category Transfer Disease Identification (P5, low)
- h511: Embedding Norm as Disease Confidence Annotation (P5, low)
- h512: HPO/Gene External Similarity for Self-Referential Diseases (P3, high)
- h513: Periodic Holdout Re-Baseline Policy (P4, low)
- h514: Migraine Drug Coverage Gap Analysis (P5, low)
- h515: Diabetes Hierarchy Split [COMPLETED - INVALIDATED]

### Recommended Next Steps
1. **h512:** HPO phenotype similarity as alternative for 144 self-referential diseases (high impact, high effort)
2. **h481:** Deliverable annotation with literature validation status (medium impact, medium effort)
3. **h257:** IV vs oral formulation safety distinction (medium impact, medium effort)

---

## Previous Session: h490/h504/h503/h505 - CV Gap + Self-Referential + Seed Analysis (2026-02-06)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 4**
- h490: CV ATC Coherent Full-to-Holdout Gap - **VALIDATED** (CV standard MEDIUM→LOW, +0.4pp)
- h504: Self-Referential Disease Analysis - **VALIDATED** (31.6% diseases are 100% self-ref)
- h503: Seed 42 Failure Mode - **VALIDATED** (sampling variance, no fix needed)
- h505: CV Target Overlap Rescue Block - **VALIDATED** (56 preds MEDIUM→LOW)

### Combined Impact (h490 + h505)
| Tier | Before | After | Change |
|------|--------|-------|--------|
| GOLDEN | 68.3% | 68.3% | 0.0pp |
| HIGH | 55.3% | 55.3% | 0.0pp |
| **MEDIUM** | **31.7%** | **32.1%** | **+0.4pp** |
| LOW | 12.0% | 12.9% | +0.9pp |
| FILTER | 10.3% | 10.3% | 0.0pp |

170 predictions moved MEDIUM→LOW. Tier counts: MEDIUM 3566→3396, LOW 2341→2511.

---

[Earlier sessions: see git history]
