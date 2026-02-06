# Research Loop Progress

## Current Session: h533 - FILTER Tier Precision Audit (2026-02-06)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 1**
- h533: FILTER Tier Precision Audit - **VALIDATED** (FILTER well-calibrated, no rescue opportunity)

### Key Findings

#### 1. FILTER Tier Is Well-Calibrated
- 7,322 FILTER predictions, 10.2% ± 1.4% holdout precision
- No sub-population exceeds 15% holdout with sufficient n (>= 30/seed)
- The tier boundary is robust across all analyses

#### 2. Rank>20 Dominates FILTER (~80% of predictions)
- rank>20 predictions are reliably low-precision across ALL categories
- No category-specific rescue possible (confirms CLOSED status of rank>20 rescue)
- Best category within rank>20: respiratory 27.6% holdout — but driven by very few diseases

#### 3. Category Patterns Within FILTER
Holdout precision by category:
- respiratory: 27.6% ± 9.4% (55/seed) — best, but driven by rank>20
- cardiovascular: 20.3% ± 12.3% (117/seed) — high variance
- endocrine: 17.2% ± 5.1% (26/seed) — small n
- autoimmune: 15.2% ± 7.7% (91/seed) — borderline LOW-level
- Most other categories: 4-12% (well below LOW threshold)

#### 4. Standard Filter Sub-Reasons (Cross-Tabulation)
Only 1 sub-reason × category exceeds 15% holdout:
- **low_freq_no_mech × respiratory: 23.3% ± 9.2%** (19/seed, 22 full predictions)
  - Too few predictions to impact tier metrics
  - Mostly genuine drug-disease pairs (COPD drugs for COPD, sleep apnea drugs for OSA)
  - Holdout wildly variable (0-50% across seeds)

#### 5. TransE Consilience in FILTER (**Key Finding**)
- FILTER + TransE top-30: **16.3% ± 2.9%** holdout vs 10.0% without (+6.3pp)
- 264 full-data predictions (~53/seed)
- **Full-data shows NO signal** (16.7% vs 16.8%) — only holdout differentiates
- 16.3% ≈ LOW (15.6%) → marginal, not sufficient for tier promotion
- TransE consilience identifies better FILTER predictions but not enough to rescue

#### 6. FILTER Reason Breakdown
| Reason | Holdout | Full | n/seed |
|--------|---------|------|--------|
| standard_filter | 10.4% | 17.6% | 1213 |
| cross_domain_isolated | 10.7% | 11.3% | 65 |
| corticosteroid_iatrogenic | 39.6%* | 22.2% | 2** |
| base_to_complication | 10.0% | 37.5% | 6 |
| inverse_indication | 8.7% | 10.3% | 19 |
| cancer_no_gt | 7.1% | 5.8% | 72 |
| cancer_only_non_cancer | 3.6% | 14.3% | 8 |
| complication_non_validated | 3.4% | 21.1% | 10 |

*High variance (37.0% std), **too small for reliable measurement

#### 7. Verdict
- **FILTER tier is appropriately calibrated** — no over-filtering
- The ~755 GT hits in FILTER (10.2% × ~7400) are structural: these drugs DO treat the disease, but our model correctly identifies them as low-confidence
- TransE consilience annotated as a flag (already implemented), not promoted

### New Hypotheses Generated (4)
- h534: TransE FILTER annotation for manual review (P5, low)
- h535: FILTER category analysis — why respiratory/autoimmune perform better (P5, medium)
- h536: FILTER precision stability monitoring (P6, low)
- h537: Deliverable quality audit — sample-based validation of top 50 (P4, medium)

### Recommended Next Steps
1. **h537**: Deliverable quality audit — validate top 50 predictions against literature
2. **h408**: Ryland collaboration prep (approaching deadline)
3. **h521**: Cancer drug same-category SOC promotion

---

## Previous Session: h526/h529/h531/h257 - Inverse Indication Taxonomy + GT Audit (2026-02-06)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 4**
- h526: Drug-Induced Disease Classes: Systematic Taxonomy - **VALIDATED** (+10 new pairs, ordering bug fix)
- h529: GT Quality Audit: Remove Inverse Indication GT Entries - **VALIDATED** (-19 false GT pairs)
- h531: TCA/MAOI → Bipolar Extension - **INVALIDATED** (no predictions to filter)
- h257: IV vs Oral Formulation Safety Distinction - **INVALIDATED** (no impact on predictions)

### Key Findings

#### 1. Mechanism Taxonomy (10 classes, 135 pairs total)
Classified all inverse indication pairs into systematic mechanism classes:
- Cardiac toxicity (34): CCBs, Class Ic/III antiarrhythmics
- Metabolic disruption (28): Glucose-lowering drugs → hypoglycemia
- Steroid AEs (26): Glaucoma, osteoporosis, pancreatitis, TB, IPF
- Hormonal disruption (12): Thyroid, vitamin D, GnRH
- Immune-mediated (10): TEN/SLE/EM from NSAIDs, azathioprine
- Organ toxicity (7): Hepato/nephro/gonadotoxic
- Procarcinogenic (2→4): Estrogen → cancer
- CNS effects (2→7): SSRI/SNRI mania
- Vascular (2): COX-2 → stroke
- Bradykinin (0→3): ACEi → angioedema (NEW)

#### 2. Ten New Inverse Indication Pairs Implemented
- SSRIs/SNRIs → bipolar (5): fluoxetine, sertraline, escitalopram, venlafaxine, duloxetine
- Conjugated estrogens → breast/endometrial cancer (2): WHI carcinogenicity
- ACEi → angioedema (3): benazepril, quinapril (bradykinin class effect)
- Total: 55→63 drugs, 124→135 pairs

#### 3. Bug Fix: Inverse Indication Ordering
Moved inverse_indication check BEFORE cancer_same_type in _assign_confidence_tier.
Previously conjugated estrogens→breast cancer was getting MEDIUM (cancer_same_type)
instead of FILTER (inverse_indication). Safety filters must always come first.

#### 4. GT Quality Finding: 38 Erroneous GT Entries
38 GT entries are inverse indications (drug CAUSES the disease):
- conjugated estrogens → breast cancer (WHI: causes breast cancer)
- benazepril/quinapril → angioedema (ACEi cause angioedema)
- flecainide → cardiac arrest/MI/HF (CAST trial: 2.5x mortality)
- corticosteroids → osteoporosis/TB/IPF
Source: adverse effect/warning mentions confused with indications in data curation.

#### 5. Impact
- 7 MEDIUM → FILTER, 2 LOW → FILTER
- Holdout: unchanged (too few predictions to measure)
- Safety: 10 harmful predictions now correctly filtered

#### 6. h529: GT Quality Audit
- 38 GT entries are inverse indications; 14 from Every Cure (flagged), 24 from DRKG (removed)
- 19 unique (drug_id, disease_id) pairs removed from expanded_ground_truth.json
- FILTER precision dropped 0.2pp (16 fewer false hits)
- Key insight: DRKG associations ≠ treatments

#### 7. h531: No TCA/MAOI Bipolar Predictions
- Checked 24 antidepressants (10 TCAs, 7 MAOIs, 7 others)
- None have bipolar disorder predictions — SSRIs/SNRIs already fully covered

### New Hypotheses Generated (5)
- h529: GT quality audit (P4, completed)
- h530: Automatic inverse indication classifier (P5, high)
- h531: TCA/MAOI → bipolar expansion (P5, completed - invalidated)
- h532: Every Cure GT error report for 14 incorrect entries (P5, low)
- h533: FILTER tier precision audit for rescue opportunities (P4, medium)

### Recommended Next Steps
1. **h408**: Ryland collaboration prep (Feb 10 deadline approaching)
2. **h533**: FILTER tier precision audit — ~755 correct predictions may be recoverable
3. **h530**: Automatic inverse indication classifier (high effort, longer term)

---

## Previous Session: h486 + h525 - SIDER Mining + GT Expansion (2026-02-06)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 6**
- h486: Drug-Induced Disease Filter: Systematic Adverse Effect Mining - **VALIDATED**
- h525: SIDER Indication-Based GT Expansion - **VALIDATED** (+51 GT pairs, HIGH +1.2pp)
- h527: Corticosteroid Iatrogenic Audit - **VALIDATED** (0 unfiltered, already comprehensive)
- h528: NSAID Iatrogenic Audit - **VALIDATED** (+1 celecoxib→ischemic stroke)
- h519: CV Pathway-Comprehensive Re-evaluation - **VALIDATED** (40.8% holdout, MEDIUM confirmed)
- h523: Anticoagulant SOC Signal - **INCONCLUSIVE** (n=10/seed too small)

### Key Findings

#### 1. SIDER Mining Requires Strict Matching
- Original loose substring matching: 1,462 candidates — 80%+ false positives
- Strict matching + SIDER indication exclusion + manual audit: 307 → 47 genuine pairs
- False positive sources: generic AE terms ("ulcer" matching "ulcerative colitis"), drugs that TREAT the disease

#### 2. 47 New Inverse Indication Pairs Implemented
- 20 new drugs added to INVERSE_INDICATION_PAIRS (35 → 55 drugs total, 77 → 124 pairs)
- Key categories:
  - Corticosteroid iatrogenic: TB reactivation, glaucoma, osteoporosis, MG crisis
  - NSAID: TEN (Stevens-Johnson), drug-induced SLE, peptic ulcer, stroke (COX-2)
  - Estradiol: endometrial/uterine cancer, hereditary angioedema
  - Proarrhythmic: ibutilide/dofetilide/milrinone → VT
  - Immunosuppressant: azathioprine → TEN, hepatitis B reactivation
  - Metabolic: paricalcitol → hypoparathyroidism

#### 3. Safety Impact
- ~105 predictions now filtered by inverse indication rules
- 98 GT negatives correctly filtered, 7 GT positives filtered (medically justified)
- Filter precision: 93.3%

#### 4. Holdout Impact (vs h520/h522 baseline)
| Tier | Previous | Current | Delta |
|------|----------|---------|-------|
| GOLDEN | 62.6% ± 8.1% | 69.9% ± 17.9% | +7.3pp |
| HIGH | 53.8% ± 2.6% | 57.3% ± 8.1% | +3.5pp |
| MEDIUM | 31.3% ± 1.4% | 28.8% ± 2.6% | -2.5pp (NS) |
| LOW | 14.2% ± 0.5% | 15.6% ± 2.6% | +1.4pp |
| FILTER | 9.7% ± 0.6% | 10.5% ± 1.4% | +0.8pp |

Note: Comparison imprecise due to accumulated code changes since last baseline.

### Corrections Applied During Session
- Removed lidocaine → VT (lidocaine is Class Ib antiarrhythmic that TREATS VT)
- Removed azathioprine → interstitial pneumonia (azathioprine treats underlying myositis)

### New Hypotheses Generated (4)
- h525: SIDER indication-based GT expansion (P4, medium)
- h526: Drug-induced disease class taxonomy (P4, medium)
- h527: Systematic corticosteroid iatrogenic filter expansion (P5, low)
- h528: Systematic NSAID inverse indication expansion (P5, low)

#### 5. SIDER GT Expansion (h525)
- Used NLP_indication labels only (not text_mention/NLP_precondition)
- 153 exact-match candidates → 51 genuine missing GT pairs after audit
- Key additions: 15 ACEi/ARB/statin/beta-blocker → ACS, anticoagulants → VTE, Sildenafil → PAH
- HIGH: 57.3% → 58.5% (+1.2pp holdout)
- SIDER indications are ~66% noise even for NLP_indication

### Recommended Next Steps
1. **h526**: Classify inverse indications by mechanism for systematic expansion
2. **h527**: Systematic corticosteroid iatrogenic filter expansion
3. **h257**: IV vs oral formulation safety distinction

---

## Previous Session: h520 - SOC Drug Class Precision Heterogeneity (2026-02-06)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 1**
- h520: SOC Drug Class Precision Heterogeneity - **VALIDATED** (corticosteroid promotion implemented)

### Key Findings

#### 1. SOC Signal is Driven by Corticosteroids (h520)
- Per-class holdout analysis across all 17 SOC drug classes in MEDIUM tier
- **Corticosteroids**: 46.1% holdout (n=96/seed, p=0.0065) — dominant signal
- **Cancer drugs**: 34.0% holdout (n=63/seed, p=0.25) — not significant
- All other classes: tiny-n (<10/seed), not individually actionable

#### 2. Category Breakdown for Corticosteroid MEDIUM
- **Dermatological**: 58.0% holdout (strong)
- **Respiratory**: 61.1% holdout (strong, small n=9)
- **Autoimmune**: 45.5% holdout (solid)
- **Ophthalmic**: 34.2% holdout (moderate)
- **Hematological**: 19.1% holdout (weak — excluded from promotion)

#### 3. Promotion Implemented: Corticosteroid MEDIUM → HIGH
- Non-hematological corticosteroid MEDIUM predictions promoted to HIGH
- 333 predictions moved
- **HIGH**: 51.5% → 53.8% (+2.3pp)
- **MEDIUM**: 29.9% → 31.1% (+1.2pp)
- Both tiers improved — clean win
- Code: `_CORTICOSTEROID_SOC_PROMOTE_CATEGORIES` + `_CORTICOSTEROID_LOWER` in production_predictor.py

### New Hypotheses Generated (4)
- h521: Cancer drug same-category SOC promotion (P4, medium)
- h522: Hematological corticosteroid demotion MEDIUM→LOW (P4, low)
- h523: Anticoagulant SOC signal in LOW tier (P5, low)
- h524: DMARD SOC signal across tiers (P5, medium)

### Recommended Next Steps
1. **h522**: Demote hematological corticosteroid MEDIUM→LOW (quick, likely +0.5pp MEDIUM)
2. **h521**: Investigate cancer_drugs MEDIUM stratification by cancer subtype
3. **h486**: Systematic adverse effect mining from SIDER (high effort but high safety impact)

---

## Previous Session: h508/h481/h518/h516 - Self-Ref + Literature Status + SOC Holdout (2026-02-06)

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
