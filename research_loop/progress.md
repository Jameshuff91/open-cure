# Research Loop Progress

## Current Session: h637/h642/h643 - Closed Direction Re-evaluation & MEDIUM Optimization (2026-02-06)

### h637: Systematic CLOSED Direction Re-evaluation — VALIDATED
Reviewed all 16 CLOSED directions for GT-dependency. 12/16 not GT-dependent. 4 candidates re-evaluated: all remain closed. h633 was a special case (wrong GT + abundant expanded GT + strong signal). No other CLOSED direction meets all conditions for reopening.

### h642: MEDIUM Default Sub-Stratification — INCONCLUSIVE
ALL remaining non-CS MEDIUM predictions are rank 16-20. Mech+Rank<=10 bucket is EMPTY (fully captured by h630/hierarchy/cancer rules). Found cv_established_drug_rescue NoMech = 22.5% (borderline LOW), leading to h643.

### h643: CV Rescue Mechanism-Gating — VALIDATED
Require mechanism support for cv_established_drug_rescue. NoMech CV drugs (22.5%, DOACs/PCSK9i) → LOW.

**Tier Impact:**
| Tier | Before | After h643 | Delta |
|------|--------|------------|-------|
| GOLDEN | 71.6% ± 4.3% | 71.6% ± 4.2% | 0 |
| HIGH | 54.7% ± 9.3% | 54.6% ± 9.3% | -0.1pp |
| MEDIUM | 38.1% ± 2.5% | **40.8% ± 2.0%** | **+2.7pp** |
| LOW | 14.5% ± 2.0% | 14.5% ± 2.0% | 0 |

### Additional hypotheses tested this session:
- h645 INVALIDATED: Other rescue rules don't need mechanism gates
- h635 INCONCLUSIVE: Cytotoxic drug class too small n per class for tier rules
- h639 INVALIDATED: Multi-system drug rescue negligible impact (n=5/seed)
- h640 INVALIDATED: Lidocaine MEDIUM n=4.6/seed too small for promotion
- h641, h646 INVALIDATED: Superseded by h643 / flawed rank analysis

**Deliverable regenerated** with h643 changes: MEDIUM 1532 preds, LOW 4055 preds.

### Key Insight: h642 rank analysis bug
`knn_rank` attribute doesn't exist on DrugPrediction. The h642 finding "ALL MEDIUM are rank 16-20" was an artifact of defaulting to 99. Always verify attribute names before analysis.

### Recommended Next Steps
1. **h638**: MEDIUM target_overlap → HIGH for psychiatric subset (53.3%, n=18/seed)
2. **h644**: ATC coherent infectious quality investigation (42.4% NoMech, interesting)
3. Consider higher-effort external data integrations (LINCS, PubMed mining)

---

## Previous Session: h633 - Cancer Same-Type Expanded GT Re-evaluation (2026-02-06)

### h633: Cancer Same-Type + Mechanism + Rank≤10 → HIGH Promotion — VALIDATED

Reopened CLOSED direction #4. Original closure (h416/h447) used internal GT showing 10.7% holdout. h611/h629 showed expanded GT (59,584 pairs vs 3,070) dramatically changes the calculus.

**Key Results (5-seed holdout, expanded GT):**
| Signal | Holdout | ±std | N/seed |
|---|---|---|---|
| Mech+R<=5 | 64.2% | 12.6% | 17.6 |
| rank_1_5 | 58.7% | 12.1% | 20.8 |
| Mech+R<=10 | 56.6% | 9.7% | 30.0 |
| ALL cancer_same_type | 37.4% | 6.4% | 100.8 |
| No mechanism | 18.3% | 3.9% | 28.4 |

**CS artifact check:** Only 0.4% of cancer_same_type predictions are corticosteroids. GENUINE signal.

**Drug class breakdown:**
| Drug Class | Holdout | N/seed |
|---|---|---|
| Taxane | 76.0% | 7.8 |
| Anthracycline | 71.6% | 6.5 |
| Platinum | 50.5% | 7.4 |
| Antimetabolite | 48.3% | 22.6 |
| Alkylating | 24.2% | 8.8 |

**Implementation:** cancer_same_type + mechanism + rank≤10 → HIGH (cancer_same_type_mech_rank10)

**H393 evaluator results:**
- cancer_same_type_mech_rank10: Full=82.4%, Holdout=62.4% ± 10.7% (n=27.6/seed)

**Tier impact:**
| Tier | Before | After | Delta |
|------|--------|-------|-------|
| GOLDEN | 71.6% ± 4.3% (420) | 71.6% ± 4.3% (419) | unchanged |
| HIGH | 53.1% ± 12.2% (719) | 54.5% ± 9.0% (876) | +1.4pp, -3.2% var |
| MEDIUM | 38.7% ± 3.3% (2019) | 36.8% ± 2.5% (1972) | -1.9pp, -0.8% var |
| LOW | 14.2% ± 2.0% (3718) | 14.2% ± 2.0% (3622) | unchanged |
| FILTER | 10.6% ± 1.3% (7274) | 10.6% ± 1.3% (7274) | unchanged |

181 predictions promoted. Top drugs: Doxorubicin (26), Paclitaxel (17), Bevacizumab (14), Cyclophosphamide (11).

### New Hypotheses Generated (4)
- h634: Remaining cancer_same_type MEDIUM demotion (no-mech = 18.3%)
- h635: Cytotoxic drug class as quality signal (taxane 76%, anthracycline 72%)
- h636: Bevacizumab cross-cancer validation (14 promoted preds)
- h637: Systematic CLOSED direction re-evaluation with expanded GT

### Key Insights
1. **CLOSED directions MUST be re-evaluated when GT changes fundamentally** — internal GT (3,070 pairs) to expanded GT (59,584 pairs) is a 19x increase that changes the precision landscape.
2. Cancer same-type is NOT CS-driven (0.4%) — unlike most HIGH-tier improvements.
3. Mechanism + rank within a single tier rule can create a HIGH-quality subset, even when the overall rule is MEDIUM.
4. Drug class stratification reveals: broad-spectrum cytotoxics (taxane, anthracycline, platinum) transfer across cancer types much better than alkylating agents or vinca alkaloids.

### Recommended Next Steps
### h634: Cancer Same-Type No-Mechanism Demotion — VALIDATED

**Key Results:**
- cancer_same_type_no_mechanism: 23.6% ± 7.7% holdout (n=23/seed) → demoted to LOW
- 166 predictions demoted

**Tier impact (cumulative h633+h634):**
| Tier | Before | After h633+h634 | Net Delta |
|------|--------|-----------------|-----------|
| GOLDEN | 71.6% ± 4.3% | 71.6% ± 4.3% | 0 |
| HIGH | 53.1% ± 12.2% | 54.7% ± 9.3% | +1.6pp |
| MEDIUM | 38.7% ± 3.3% | 38.1% ± 2.5% | -0.6pp |
| LOW | 14.2% ± 2.0% | 14.5% ± 1.9% | +0.3pp |

### Recommended Next Steps
1. **h637**: Systematically check all 16 CLOSED directions for GT-dependency
2. Regenerate deliverable with h633+h634 updates
3. **h635**: Investigate cytotoxic drug class as quality signal

---

## Previous Session: h629/h631 - MEDIUM Quality Stratification (2026-02-06)

### h629: MEDIUM Precision Stratification by Multiple Signals — VALIDATED

Expanded GT resolves original TransE MEDIUM blocker (h405: 34.7% < HIGH 50.8%). With expanded GT, TransE within MEDIUM reaches HIGH-level precision.

**Key Results (5-seed holdout, expanded GT):**
| Signal Combination | Holdout | ±std | N/seed |
|---|---|---|---|
| TransE+Mechanism+Rank≤10 | 71.9% | 15.7% | 7 |
| cancer_same_type+Rank≤5 | 66.0% | 14.1% | 22 |
| TransE+Rank≤5 | 64.9% | 12.4% | 11 |
| TransE+Rank≤10 | 63.2% | 7.1% | 19 |
| TransE+Mechanism | 59.4% | 13.9% | 14 |
| TransE alone | 56.5% | 8.8% | 28 |
| Mechanism+Rank≤5 | 53.9% | 6.2% | 39 |
| Mechanism+Rank≤10 | 52.5% | 4.4% | 76 |
| All MEDIUM | 38.8% | 3.7% | 328 |

**CS artifact check:** TransE non-CS: 49.1% (GENUINE). Not driven by corticosteroids.

**Differential:** +19.3pp over non-TransE MEDIUM (constant regardless of GT used).

**Tier impact assessment (TransE MEDIUM non-CS → HIGH):**
- HIGH: 49.1% → 49.5%, variance 7.9% → 5.7% (IMPROVES), +34 preds/seed
- MEDIUM: 39.9% → 38.9% (-0.9pp)
- Decision: NOT promoted (borderline, existing CLOSED direction). Implemented as annotation instead.

### h631: MEDIUM Quality Quartile Annotation — VALIDATED

Added `medium_quality` column to deliverable based on h629 signal combinations:
- Q1 (TransE + mechanism/rank≤5): 138 preds, 60-72% expected holdout
- Q2 (TransE OR mechanism+rank≤10): 459 preds, 50-57% expected holdout
- Q3 (mechanism OR rank≤5): 931 preds, 44-54% expected holdout
- Q4 (none): 606 preds, ~31% expected holdout

Q1-Q4 spans a 41pp range — more informative than single MEDIUM label for Ryland/collaborators.

### h630: TransE MEDIUM → HIGH Promotion — VALIDATED

Implemented TransE + (mechanism OR rank≤5) non-CS MEDIUM → HIGH promotion.

**H393 evaluator results:**
- transe_medium_promotion: Full=68.8%, Holdout=56.1% ± 11.9% (n=15/seed)

**Tier impact:**
| Tier | Before | After | Delta |
|------|--------|-------|-------|
| GOLDEN | 71.6% ± 4.3% (420) | 71.6% ± 4.3% (420) | unchanged |
| HIGH | 52.8% ± 13.5% (604) | 53.1% ± 12.2% (719) | +0.3pp, -1.3% var |
| MEDIUM | 39.5% ± 3.5% (2134) | 38.7% ± 3.3% (2019) | -0.8pp, -0.2% var |
| LOW | 14.2% ± 2.0% (3718) | 14.2% ± 2.0% (3718) | unchanged |
| FILTER | 10.6% ± 1.3% (7274) | 10.6% ± 1.3% (7274) | unchanged |

115 preds promoted. Top drugs: Doxorubicin (23), Amphotericin B (20), Bleomycin (12).

### Session Tier Performance (post-h630)
| Tier | Holdout | Predictions |
|------|---------|-------------|
| GOLDEN | 71.6% ± 4.3% | 420 |
| HIGH | 53.1% ± 12.2% | 719 |
| MEDIUM | 38.7% ± 3.3% | 2019 |
| LOW | 14.2% ± 2.0% | 3718 |
| FILTER | 10.6% ± 1.3% | 7274 |

### New Hypotheses Generated (4)
- h630: TransE MEDIUM → HIGH promotion (VALIDATED)
- h631: MEDIUM quality quartile annotation (VALIDATED)
- h632: Mechanism + Rank ≤ 10 as independent HIGH signal
- (More pending from h629 analysis)

### Key Insights
1. Expanded GT resolves TransE MEDIUM blocker — 56.5% vs 34.7% (internal GT)
2. The 19.3pp TransE differential is GT-independent (constant lift)
3. TransE promotion: HIGH precision INCREASES while variance DECREASES — counter-intuitive but correct
4. Signal combination reveals 41pp quality spread within MEDIUM
5. CLOSED directions should be re-evaluated when evaluation methodology changes (GT expansion)

### Recommended Next Steps
1. **h632**: Validate mechanism+rank≤10 as independent promotion signal
2. External data integration for fundamentally new signals
3. Meeting prep for Ryland (Monday Feb 10)

---

## Previous Session: h618/h622/h614/h617/h624 - CV Rescue + Tier Calibration (2026-02-06)

### h618: CV Medium Demotion Reversal — VALIDATED

h462 demoted ALL cardiovascular MEDIUM→LOW based on internal GT (2.0% holdout). h615 found 25.1% ± 19.4% with expanded GT. This experiment stratified by drug class:

**Key Results (5-seed holdout, expanded GT):**
| Drug Class | Holdout | N/seed | Preds | Action |
|------------|---------|--------|-------|--------|
| CCB | 49.7% ± 34.6% | 3.4 | 11 | Rescued to MEDIUM |
| Diuretic | 33.8% ± 32.4% | 3.2 | 12 | Rescued to MEDIUM |
| Anticoagulant/antiplatelet | 32.6% ± 23.4% | 14.2 | 70 | Rescued to MEDIUM |
| ARB | 30.0% ± 40.0% | 3.0 | 10 | Rescued to MEDIUM |
| other_CV (antibiotics/biologics) | 18.3% ± 9.0% | 37.6 | 166 | Stay LOW |
| Corticosteroid | 2.9% ± 5.7% | 1.4 | 20 | Stay LOW |

**Implementation:** `_is_established_cv_drug()` method identifies genuine CV pharmacotherapy (anticoagulants, CCBs, diuretics, ARBs, statins, beta-blockers, ACE inhibitors, antiarrhythmics, nitrates, etc.). 201 predictions rescued LOW→MEDIUM.

**Holdout validation:** cv_established_drug_rescue = 30.9% ± 20.9% (n=44/seed, Δ=-1.6pp, GENUINE). Remaining cardiovascular_medium_demotion = 4.6% ± 3.8% (correctly LOW).

**Tier impact:**
| Tier | Before | After | Delta |
|------|--------|-------|-------|
| GOLDEN | 71.6% ± 4.3% (420) | 71.6% ± 4.3% (420) | unchanged |
| HIGH | 52.8% ± 13.5% (604) | 52.8% ± 13.5% (604) | unchanged |
| MEDIUM | 41.3% ± 2.8% (1874) | 38.9% ± 4.0% (2075) | -2.4pp, +201 preds |
| LOW | 15.1% ± 2.4% (3978) | 14.6% ± 2.4% (3777) | -0.5pp, -201 preds |
| FILTER | 10.6% ± 1.3% (7274) | 10.6% ± 1.3% (7274) | unchanged |

**Key insight:** Internal GT systematically underestimates CV drug precision. Expanded GT reveals drug-class stratification cleanly separates genuine CV drugs (30.9%) from non-CV drugs predicted for CV diseases (4.6%).

### h622: Expanded GT Recalibration of Other Demoted Categories — INVALIDATED
No other demoted category has a drug-class subset with both >=25% holdout AND n>=5/seed. Best candidate: heme antineoplastic_heme 31.9% but n=4.8/seed (marginal). The CV case was special due to anticoagulant dominance (n=14.2/seed).

### h614: MEDIUM Sub-Pathway Quality Map v2 — VALIDATED
No demotable MEDIUM sub-pathways with expanded GT. All sub-pathways with n>=3/seed above 25% holdout. MEDIUM overall: 51.8% ± 5.5%. Metabolic target_overlap leak (45 preds) is correct behavior (48.9% holdout).

### h617: HIGH Tier Stabilization — INCONCLUSIVE
HIGH variance (±13.5%) is structural — driven by disease-split randomness. Seed 42 has fewer hierarchy-matching diseases in holdout. comp_to_base_high_87 = 0% holdout (n=5/seed, too small). Cannot be fixed without stratified splitting.

### h624: Deliverable Regeneration — VALIDATED
Deliverable regenerated. GOLDEN 420, HIGH 604, MEDIUM 2075, LOW 3777, FILTER 7274.

### Session Tier Performance (post-h618)
| Tier | Holdout | Predictions |
|------|---------|-------------|
| GOLDEN | 71.6% ± 4.3% | 420 |
| HIGH | 52.8% ± 13.5% | 604 |
| MEDIUM | 38.9% ± 4.0% | 2075 |
| LOW | 14.6% ± 2.4% | 3777 |
| FILTER | 10.6% ± 1.3% | 7274 |

### New Hypotheses Generated (3)
- h622: Expanded GT recalibration of other demoted categories (INVALIDATED)
- h623: MEDIUM precision recovery: tighten CV rescue criteria
- h624: Deliverable regeneration (VALIDATED)

### Key Insights
1. Internal GT systematically underestimates CV drug precision; expanded GT reveals 30.9% holdout
2. Drug-class stratification can find quality subsets within category demotions
3. Category demotions well-calibrated for all non-CV categories
4. MEDIUM tier is fully optimized at sub-pathway level
5. HIGH variance is structural, irreducible with current methodology

### Recommended Next Steps
1. **h623**: Tighten CV rescue criteria to recover MEDIUM precision
2. **h534/h578**: TransE annotation for FILTER/LOW tiers (deliverable quality)
3. External data integration (h91/h92) for fundamentally new signals

---

## Previous Session: h615/h619/h620/h621/h616 - Expanded GT Analysis (2026-02-06)

### h616: Disease-Specific GT Completeness Score — VALIDATED
Added `gt_completeness_ratio` column to deliverable. 479 diseases scored. Median 6.0x, mean 11.5x. Weakly negative correlation with holdout precision (r=-0.198) — not predictive of quality, but informative annotation.

### h621: Disease Categorization Fix — VALIDATED
Fixed pleural mesothelioma (respiratory→cancer) and retinoblastoma (ophthalmic→cancer). ~28 predictions rescued from FILTER. Added `mesothelioma` to cancer keywords, moved `retinoblastoma` from ophthalmic to cancer.

### h620: Expanded GT Safety Filter Audit — VALIDATED
GT contamination is real but minimal: 37 clearly wrong entries out of 1131 FILTER GT hits (3.3%). FDG PET diagnostic associations exist in both internal and expanded GT. Inverse indication hits are mostly genuine dual-use drugs (18/22 from internal GT). Does NOT affect tier precision.

### h619: Deliverable Regeneration — VALIDATED
Deliverable regenerated with h615+h621 changes. GOLDEN: 420, HIGH: 604, MEDIUM: 1874, LOW: 3978, FILTER: 7274. 14,150 total predictions.

### h615: Expanded GT-Based Tier Recalibration — VALIDATED

Compared per-rule precision using internal GT (3,070 pairs) vs expanded GT (59,584 pairs, 19x more). Found 26 tier boundary crossings. 5-seed holdout validated 4 HIGH→GOLDEN hierarchy group promotions.

**Key Results:**
- Internal GT systematically underestimates hierarchy group precision by 15-30pp
- 4 groups have GOLDEN-level holdout precision but were assigned HIGH based on internal GT analysis:
  - autoimmune_hierarchy_rheumatoid_arthritis: 86.4% ± 8.7% holdout (n=23/seed)
  - autoimmune_hierarchy_colitis: 85.7% ± 0.0% holdout (n=7/seed)
  - cardiovascular_hierarchy_arrhythmia: 72.9% ± 1.5% holdout (n=11/seed)
  - cardiovascular_hierarchy_coronary: 65.5% ± 1.2% holdout (n=13/seed)

**Implementation:** Added `HIERARCHY_PROMOTE_TO_GOLDEN` set in `_assign_confidence_tier()`. 139 predictions promoted.

**Impact:**
| Tier | Before | After | Delta |
|------|--------|-------|-------|
| GOLDEN | 69.9% ± 17.9% (280 preds) | 71.6% ± 4.3% (419 preds) | +1.7pp, +139 preds, std -13.6pp |
| HIGH | 58.8% ± 6.1% (736 preds) | 52.8% ± 13.5% (597 preds) | -6.0pp, -139 preds |
| MEDIUM | 41.3% ± 2.8% | 41.3% ± 2.8% | unchanged |
| LOW | 15.1% ± 2.4% | 15.1% ± 2.4% | unchanged |
| FILTER | 10.6% ± 1.3% | 10.6% ± 1.3% | unchanged |

Tier ordering preserved. HIGH drop due to removing best predictions. Seed 42 outlier (HIGH=30%, n=40) drives HIGH variance.

**Other findings NOT acted on:**
- cardiovascular_medium_demotion: 25.1% ± 19.4% holdout (above MEDIUM boundary but too variable)
- FILTER rules (non_therapeutic_compound, inverse_indication): expanded GT shows higher precision but safety filters should remain regardless
- Many FILTER rules show elevated expanded GT precision — suggests expanded GT may include non-therapeutic associations

**New Hypotheses (4):** h617-h620 (HIGH stabilization, CV medium demotion stratification, deliverable regeneration, expanded GT safety audit)

### Session Tier Performance (h621 update)
| Tier | Holdout | Predictions |
|------|---------|-------------|
| GOLDEN | 71.6% ± 4.3% | 420 |
| HIGH | 52.8% ± 13.5% | 604 |
| MEDIUM | 41.3% ± 2.8% | 1874 |
| LOW | 15.1% ± 2.4% | 3978 |
| FILTER | 10.6% ± 1.3% | 7274 |

### New Hypotheses Generated (5 total)
- h617: HIGH tier stabilization after h615 promotions
- h618: CV medium demotion drug-class stratification
- h619: Deliverable regeneration (COMPLETED)
- h620: Expanded GT safety filter audit (COMPLETED)
- h621: Disease categorization fix (COMPLETED)

### Recommended Next Steps
1. **h618**: CV medium demotion drug-class stratification (25.1% ± 19.4% holdout)
2. **h617**: Investigate HIGH tier seed-42 outlier
3. External data integration (h91/h92) for fundamentally new signals
4. Literature mining (h91) for novel hypotheses

---

## Previous Session: h606/h611/h612/h613/h605 - ATC Coherent + GT Methodology (2026-02-06)

### h606: ATC Coherent Respiratory/Endocrine Validation — VALIDATED
Comprehensive ATC coherent category analysis. Found 292 ATC coherent MEDIUM predictions across 9 categories. Psychiatric ATC coherent holdout = 17.2% ± 5.8% (p=0.0006 below MEDIUM avg). Added psychiatric to ATC_COHERENT_EXCLUDED. 47 predictions MEDIUM→LOW. Tier-level impact unmeasurable.

### h612: Deliverable Regeneration — VALIDATED
Regenerated deliverable with all h598-h606 changes. 14,150 predictions, 473 diseases, 1,004 drugs. MEDIUM: 1,876 preds. All changes confirmed.

### h611: MEDIUM Sub-Pathway Quality Map — INVALIDATED (GT methodology bug)
**CRITICAL FINDING:** Initial analysis using predictor.ground_truth (3,070 pairs) showed 7 below-LOW sub-pathways. But this was WRONG — expanded_ground_truth.json (59,584 pairs, 19x more) should be used. With correct GT, ALL sub-pathways above LOW threshold. cancer_same_type: 11.8% → 37.7%, target_overlap: 11.5% → 31.7%. Code changes reverted.

**KEY LESSON:** ALWAYS use expanded_ground_truth.json for holdout evaluation. predictor.ground_truth only has DRKG-derived pairs.

### h613: Expanded GT Gap Analysis — VALIDATED
Mapped the internal-vs-expanded GT gap. Cancer has highest ratio (13.2x), endocrine lowest (3.0x). Per-tier gains from expanded GT: GOLDEN +12.9pp, HIGH +16.8pp, MEDIUM +15.4pp, LOW +6.3pp. No diseases have zero expanded GT.

### h605: Highly Repurposable MEDIUM Demotion — INVALIDATED
Only 4 predictions (all chronic pain). 0% internal GT but 25% expanded GT. Holdout with expanded GT = 29.2%. Not demotable.

### Session Tier Performance (unchanged from h603)
| Tier | Holdout | Predictions |
|------|---------|-------------|
| GOLDEN | 69.9% ± 17.9% | ~280 |
| HIGH | 58.9% ± 6.0% | ~732 |
| MEDIUM | 41.3% ± 2.8% | ~1876 |
| LOW | 15.1% ± 2.4% | ~3958 |
| FILTER | 10.6% ± 1.3% | ~7300 |

### New Hypotheses Generated (7 total)
- h610: ATC coherent infectious per-drug-class quality
- h611: MEDIUM sub-pathway quality map (INVALIDATED)
- h612: Deliverable regeneration (COMPLETED)
- h613: Expanded GT gap analysis (COMPLETED)
- h614: MEDIUM sub-pathway v2 with correct GT
- h615: Expanded GT tier recalibration
- h616: Disease GT completeness score

### Recommended Next Steps
1. **h614**: Re-run quality map with expanded GT and significance tests
2. **h616**: Add GT completeness annotation to deliverable
3. External data integration (h91/h92) for fundamentally new signals
4. Literature mining (h91) remains highest-priority unblocked direction

---

## Previous Session: h603/h604 - MEDIUM Standard Rule Refinement (2026-02-06)

### h604: Standard MEDIUM Infectious Drug-Class Stratification — INCONCLUSIVE

Per-drug-class analysis of the 314 standard MEDIUM infectious predictions reveals significant heterogeneity but no cleanly demotable group with sufficient n.

**Drug class holdout:** tetracycline CLASS 32.3% (genuine MEDIUM), fluoroquinolone 11.4% (below MEDIUM, n=9.6), macrolide 10.6% (LOW-N).

**Per-drug insight:** tetracycline-the-drug has 0% holdout (n=7/seed) while doxycycline (35.4%), minocycline (36.6%), demeclocycline (31.2%) are genuine MEDIUM. Legacy drugs (tetracycline, erythromycin) at 0% but per-drug n too small.

**Decision:** NOT implementing per-drug demotions. n too small, marginal impact, risk of overfitting.

---

### h603: Standard MEDIUM Category Analysis — VALIDATED (marginal)

Analyzed all 630 standard MEDIUM predictions by disease category. Found that metabolic (10.0% full-data, 8.3% holdout), respiratory (5.9% full-data, 2.0% holdout), and endocrine (23.1% full-data, 9.5% holdout) perform far below the MEDIUM average in the standard pathway.

**Key Results:**
- Pooled met+resp+endo standard: 5.2% ± 6.6% holdout (n=10.4/seed) vs 25.2% other standard
- 20.1pp gap between these categories and the rest
- MEDIUM_DEMOTED_CATEGORIES interaction: adding categories also blocks ATC coherent pathway
  - metabolic: already excluded from ATC coherent → clean demotion (10.3% holdout)
  - respiratory: NOT excluded from ATC coherent → demotion includes ATC rescue preds (22.3% holdout)
  - endocrine: NOT excluded from ATC coherent → demotion includes ATC rescue preds (24.5% holdout)

**Implementation:** Added metabolic only to MEDIUM_DEMOTED_CATEGORIES. Respiratory and endocrine NOT demoted (holdout precision above LOW when including ATC coherent predictions).

**Impact:** MEDIUM 41.3% ± 2.8% (vs 41.4% baseline). Within noise. 20 predictions MEDIUM→LOW.

**Critical learning:** MEDIUM_DEMOTED_CATEGORIES intercepts predictions BEFORE the ATC coherent rescue pathway. Only add categories that are already in ATC_COHERENT_EXCLUDED or where combined standard+ATC precision is clearly LOW.

### Session Tier Performance (h603 update)
| Tier | Holdout | Predictions |
|------|---------|-------------|
| GOLDEN | 69.9% ± 17.9% | ~280 |
| HIGH | 58.9% ± 6.0% | ~720 |
| MEDIUM | 41.3% ± 2.8% | ~1879 |
| LOW | 15.1% ± 2.4% | ~3955 |
| FILTER | 10.6% ± 1.3% | ~7300 |

### New Hypotheses Generated (4)
- h604: Standard MEDIUM infectious drug-class stratification (P4, medium)
- h605: Highly repurposable MEDIUM demotion (P5, low)
- h606: ATC coherent respiratory/endocrine validation (P4, low)
- h607: Standard MEDIUM autoimmune quality (P5, low)

### Recommended Next Steps
1. **h604**: Largest remaining standard MEDIUM category (infectious, 314 preds, 22.7% holdout). Per-drug analysis may find demotable drug classes.
2. **h606**: Quick check — are atc_coherent respiratory/endocrine worth the ATC rescue or should they be excluded?
3. External data integration (h91/h92) for fundamentally new signals.

---

## Previous Session: h550/h598 - Antibiotic Spectrum + Targeted Cancer Expansion (2026-02-06)

### h550: Antibiotic Spectrum Validation — INVALIDATED

Tested whether within-antibacterial spectrum mismatches (gram-positive drugs for gram-negative diseases and vice versa) could filter MEDIUM infectious predictions.

**Key Results:**
- Built spectrum classification for 48 antibacterial drugs and pathogen type mapping for 38 infectious diseases
- Only 22 spectrum mismatches found in MEDIUM+ tier (4.8% of antibacterial-infectious predictions)
- **53% false positive rate** — many "mismatches" are medically valid:
  - Azithromycin→CF Pseudomonas = standard of care (anti-inflammatory + biofilm disruption)
  - Gentamicin→S. aureus = synergistic with beta-lactams (used in bacteremia)
  - Cephalexin→UTI = first-line treatment (1st-gen ceph covers E. coli)
- Only 7 genuine mismatches — far below n≈30 threshold for reliable holdout
- Full-data precision of mismatches (27.3%) still above LOW (15.6%)

**Conclusion:** Within-antibacterial spectrum matching is too nuanced for rule-based classification. The broad antimicrobial-pathogen mismatch from h560 already catches clear biological errors.

### h598: Expand CANCER_TARGETED_THERAPY — VALIDATED (+3.3pp MEDIUM)

Error analysis of MEDIUM false positives revealed 15 targeted cancer drugs missing from the cancer_targeted_therapy demotion list, despite being target/biomarker-specific drugs that should NOT generalize across cancer subtypes.

**Drugs Added (15 total):**
| Category | Drugs | Mechanism |
|----------|-------|-----------|
| Anti-HER2 mAbs | trastuzumab, pertuzumab | HER2+ cancers only |
| Anti-EGFR mAb | cetuximab | KRAS wild-type CRC, SCCHN |
| Anti-VEGFR2 mAb | ramucirumab | Anti-angiogenic, target-specific |
| PARP inhibitors | olaparib, niraparib, rucaparib | BRCA/HRD-mutant only |
| BTK inhibitors | tirabrutinib, acalabrutinib, zanubrutinib | B-cell malignancies only |
| IDH1 inhibitor | ivosidenib | IDH1-mutant AML/cholangiocarcinoma |
| mTOR inhibitor | everolimus | Tumor-specific mTOR |
| Narrow cytotoxic | trabectedin, eribulin, lanreotide | Very narrow indications |

**Holdout Validation (5-seed):**
| Group | Holdout | n/seed |
|-------|---------|--------|
| New targeted drugs | 6.1% ± 5.2% | 32.6 |
| Existing cancer_same_type | 40.2% ± 6.5% | 85.8 |
| Gap | 34.1pp | — |

**Impact:**
| Tier | Before | After | Delta |
|------|--------|-------|-------|
| MEDIUM | 38.1% ± 2.1% | 41.4% ± 2.0% | **+3.3pp** |
| Predictions moved | — | 202 MEDIUM→LOW | — |

### New Hypotheses Generated (3)
- h599: Obsolete tetracycline demotion (demeclocycline/oxytetracycline) — P4, medium
- h600: Low-precision infectious drug demotion (cefuroxime/streptomycin) — P5, low
- h601: Cancer same-type precision by drug class (remaining drugs) — P5, medium

### Session Tier Performance (h598 update)
| Tier | Holdout | Predictions |
|------|---------|-------------|
| GOLDEN | 70.3% ± 17.8% | 280 |
| HIGH | 54.7% ± 4.5% | 736 |
| MEDIUM | 41.4% ± 2.0% | 1899 |
| LOW | 13.4% ± 1.8% | 3935 |
| FILTER | 10.4% ± 1.3% | 7300 |

### Key Learnings
1. CANCER_TARGETED_THERAPY was incomplete — missing anti-target mAbs, PARP inhibitors, BTK inhibitors. When building drug class lists, check ALL therapeutic classes in the area.
2. Within-antibacterial spectrum matching fails because many antibiotics have secondary activities (synergy, anti-inflammatory) that simple classification misses. Only broad-category mismatches are clean enough for filtering.
3. Error analysis by drug (not by rule/category) is an effective way to find improvement opportunities that rule-level analysis misses.

### Recommended Next Steps
1. **h599**: Obsolete tetracycline demotion — demeclocycline and oxytetracycline have 80+ FP in MEDIUM
2. **h601**: Check remaining cancer_same_type for more low-quality drug classes
3. Consider external data integration (LINCS, PubMed mining) for fundamentally new signals

---

## Previous Session: h592/h593 - Composite Quality + GT Gap Detection (2026-02-06)

### h592: Experimental Validation Priority List — VALIDATED

Computed a composite quality score combining all validated signals (kNN rank, norm_score, TransE consilience, gene overlap, mechanism support, disease holdout precision, non-self-referentiality) to prioritize MEDIUM predictions for experimental validation.

**Key Results:**

**Holdout Validation (5-seed, MEDIUM tier):**
| Ranking Method | Q1 | Q2 | Q3 | Q4 | Q1-Q4 Gap |
|---------------|-----|-----|-----|-----|-----------|
| Composite | 14.0% ± 1.2% | 10.5% ± 0.7% | 7.1% ± 0.7% | 6.0% ± 0.7% | 8.0pp |
| kNN Rank only | 11.5% ± 0.7% | 9.2% ± 0.6% | 10.0% ± 0.8% | 7.0% ± 0.5% | 4.5pp |

**Composite beats kNN rank by +2.6pp for Q1 and 78% better separation (8.0pp vs 4.5pp gap).**

**Formula:** `1.5*rank_score + norm_score + TransE + gene_overlap + 0.5*mechanism + disease_holdout + 0.5*non_self_ref`

**Novel Non-CS MEDIUM (holdout):**
- Q1: 6.8% ± 1.1% vs Q4: 1.7% ± 0.7% (4.0x lift)

**Full-Data (novel non-CS MEDIUM):**
- Q1: 34.5% vs Q4: 9.5% (3.6x lift)

**Medical Plausibility (top 20 novel):**
- 65% reasonable (45% validated + 20% plausible) vs 56% overall MEDIUM (+9pp)
- Key validated novel: doxorubicin→choriocarcinoma (FDA), clopidogrel→CAD (FDA), enoxaparin→DIC
- Key implausible: erythromycin→meningitis (poor BBB), phenobarbital→dry skin (no mechanism)

**Key Insight: Many "novel" predictions are GT gaps, not discoveries:**
- 4/4 GOLDEN novel = FDA-approved (clopidogrel→CAD, lovastatin→atherosclerosis, etc.)
- 5/7 HIGH novel = standard treatments (levofloxacin→sinusitis, verapamil→ACS)
- Truly novel repurposing: bortezomib→Burkitt lymphoma, montelukast→IPF, lovastatin→Fabry disease

**Output:** `data/analysis/h592_validation_priority_list.json` (top 100 prioritized novel non-CS predictions)

**Difference from h443 (CLOSED):** h443 tested TransE+kNN within-tier and found no improvement over rank alone. h592 adds disease-level signals (holdout precision, self-referentiality) which provide the +2.6pp lift. This is an annotation/prioritization signal, NOT for tier changes.

### New Hypotheses Generated (3)
- h593: GT gap auto-detection from ATC/category matching (P4, medium)
- h594: Add composite_quality_score to production deliverable (P5, low)
- h595: Composite weight optimization via grid search (P5, medium)

### h593: GT Gap Auto-Detection — VALIDATED

Systematically identified FDA-approved drug-disease pairs missing from GT by checking if high-ranked predictions (rank<=5) are for drugs that already treat >=3 other diseases in the same category.

**Method:**
- 320 same-category candidates found
- 71 non-CS non-antibiotic interesting candidates
- Top 20 manually assessed: 10/20 (50%) are FDA-approved

**9 Definitive GT Gaps Added:**
1. Doxorubicin → choriocarcinoma (EMA/EP regimen)
2. Paclitaxel → germ cell testicular cancer (TIP regimen)
3. Fluorouracil → tongue cancer (head/neck SCC)
4. Verapamil → acute coronary syndrome (angina)
5. Posaconazole → cryptococcal meningitis (ECIL salvage)
6. Posaconazole → chromomycosis (triazole antifungal)
7. Posaconazole → ringworm (triazole antifungal)
8. Posaconazole → cryptococcosis (IDSA alternative)
9. Posaconazole → cutaneous candidiasis (triazole antifungal)

**Holdout Impact:** MEDIUM 35.8% → 36.6% (+0.8pp). All other tiers within seed variance.

**Key Finding:** 50% of high-evidence novel predictions are actually GT gaps, not discoveries. Posaconazole alone had 5 missing fungal disease indications. This suggests systematic GT incompleteness in antifungal and cancer drug families.

### h596: Triazole Antifungal GT Expansion — VALIDATED (marginal)

Antifungal GT was 85% complete (23/27 pairs already present). Added 4 new pairs:
- Voriconazole → cutaneous/chronic mucocutaneous candidiasis, cryptococcosis
- Isavuconazonium → zygomycosis/mucormycosis

Holdout unchanged (36.6% MEDIUM). Posaconazole gaps from h593 were the exception.

### h597: Cancer Drug GT Expansion — VALIDATED

Added 5 FDA/guideline-approved cancer drug pairs:
- Paclitaxel → larynx cancer, vulva cancer
- Cisplatin → uterine cancer
- Bortezomib → Burkitt lymphoma, anaplastic large cell lymphoma

Holdout: MEDIUM 36.6% → 37.0% (+0.4pp). Cancer drug GT more complete than expected.

### Cumulative GT Expansion (h593+h596+h597)
| Source | Pairs Added | MEDIUM Impact |
|--------|-------------|---------------|
| h593: Auto-detection | 9 | +0.8pp |
| h596: Antifungals | 4 | +0.0pp |
| h597: Cancer drugs | 5 | +0.4pp |
| **Total** | **18** | **+1.2pp** |

### New Hypotheses Generated (5 total this session)
- h593-h597: GT gap detection arc (COMPLETED)
- h594: Add composite score to deliverable (P5, low)
- h595: Composite weight optimization (P5, medium)
- h596: Triazole antifungal GT expansion (COMPLETED)
- h597: Cancer drug GT expansion (COMPLETED)

### Recommended Next Steps
1. **h594**: Add composite score to deliverable (quick implementation)
2. Consider pivoting to external data integration (LINCS, PubMed) for fundamentally new signals
3. Remaining GT gaps have diminishing returns; focus on deliverable quality

### Session Tier Performance (h597 update)
| Tier | Holdout | Predictions |
|------|---------|-------------|
| GOLDEN | 69.9% ± 17.9% | 280 |
| HIGH | 58.9% ± 6.0% | 754 |
| MEDIUM | 37.0% ± 2.8% | 2083 |
| LOW | 15.6% ± 2.4% | 3733 |
| FILTER | 10.6% ± 1.3% | 7300 |

### Key Learnings
1. Disease-level signals (holdout precision, self-referentiality) add genuine value for prediction prioritization that prediction-level signals (TransE, kNN score) miss. The composite score is useful for practical experiment prioritization but NOT for tier reassignment.
2. 50% of high-evidence "novel" predictions are actually GT gaps (FDA-approved but missing from our GT). Posaconazole had 5 missing fungal indications. Drug families have correlated GT gaps — fixing one suggests checking the whole family.
3. GT incompleteness inflates the "novel prediction" count and deflates measured precision. Always check for GT gaps before claiming novel discoveries.

---

## Previous Session: h586/h588 - GT-Free Quality Signals (2026-02-06)

### h586: GT-Free Paradigm Mismatch via DRKG Edges — INVALIDATED

Tested whether DRKG non-treatment edges (gene associations, anatomy, symptoms) can approximate Drug Jaccard (treatment paradigm similarity) without GT knowledge.

**Key Finding: Biology ≠ Treatment Paradigm**
- Gene Jaccard: r=+0.079 with holdout (NS), r=+0.086 with drug Jaccard — too weak
- Combined DRKG Jaccard (genes+anatomy+symptoms): r=+0.077 (NS)
- GT-free mismatch (embed_sim - combined_jaccard): r=0.996 with embed_sim — just embedding similarity in disguise
- 63% of diseases share ZERO genes with their kNN neighbors (too sparse)
- After controlling for self-referentiality: partial_r=+0.030 (NS)

**Why genes fail:** Disease-gene associations capture molecular biology, but treatment decisions are driven by clinical phenotype, drug class availability, and treatment paradigms. Two diseases with identical genes can be treated with completely different drug classes (e.g., hypertension vs PAH).

**Symptom/anatomy edges (Hetionet):** Show promise (symptom r=+0.297) but only 39/312 diseases have coverage.

### h588: HPO Symptom Phenotype Similarity as Quality Signal — VALIDATED (annotation)

Tested HPO phenotype similarity as an extended version of the sparse Hetionet symptom signal. HPO matrix covers 799 diseases (82/312 holdout diseases, 2x Hetionet coverage).

**Key Results:**
| Signal | r with holdout | Partial r (ctrl GT) | Coverage |
|--------|---------------|-------------------|----------|
| HPO sim | +0.243* | +0.258* | 82 diseases |
| HPO→Drug Jaccard proxy | +0.390*** | +0.416*** | 82 diseases |
| Gene→Drug Jaccard proxy | +0.086 | +0.116* | 270 diseases |

*p<0.05, ***p<0.001

**Why HPO works better than genes:** Clinical phenotype (symptoms, signs, lab findings) captures treatment paradigm similarity 4.5x better than molecular biology (gene overlap).

**Practical limitation:** Adds only 0.9% incremental R² beyond GT size + embed_sim. HPO sim partial_r=+0.149 (NS) after controlling for embed_sim. Coverage still limited (26.3%).

**Quartile analysis:** Q4 (highest HPO sim) = 11.3% holdout vs Q1 = 6.4%. Within GT 1-20 band: 3.8% vs 1.7% (2.2x lift).

**Conclusion:** Annotation value for deliverable. NOT promotable for tiers.

### New Hypotheses Generated (3)
- h588: HPO similarity (COMPLETED - VALIDATED)
- h589: ATC hierarchy as GT-free treatment paradigm proxy (P4, medium)
- h590: Hetionet disease-resembles as augmented kNN signal (P5, low)

### Key Learning
Clinical phenotype (HPO) is the best GT-free proxy for treatment paradigm similarity. Molecular biology (gene overlap) fails to predict treatment similarity. The gap between biology and therapy is fundamental: diseases with shared genes may have completely different treatment paradigms. This is consistent with h571 (therapeutic islands) and h583 (paradigm mismatch).

### Session Tier Performance (unchanged from h560)
| Tier | Holdout | Predictions |
|------|---------|-------------|
| GOLDEN | 69.9% ± 17.9% | 280 |
| HIGH | 59.5% ± 6.2% | 754 |
| MEDIUM | 35.8% ± 2.8% | 2083 |
| LOW | 15.5% ± 2.4% | 3733 |
| FILTER | 10.6% ± 1.3% | 7300 |

### h589: ATC Hierarchy as GT-Free Treatment Paradigm Proxy — VALIDATED (circular)

ATC codes are the best proxy for drug Jaccard (treatment paradigm similarity):

| Proxy Signal | r with Drug Jaccard | r with Holdout | Partial r (ctrl GT) |
|-------------|-------------------|---------------|-------------------|
| ATC L5 | +0.848 | +0.213 | +0.266 |
| ATC L3 | +0.737 | +0.180 | +0.229 |
| ATC L2 | +0.665 | +0.157 | +0.206 |
| HPO sim (h588) | +0.390 | +0.243 | +0.258 |
| Gene overlap (h586) | +0.086 | +0.079 | +0.037 |

**BUT ATC adds ZERO signal beyond drug Jaccard:**
- ATC L2 | drug Jaccard: partial_r=-0.013 (p=0.82, NS)
- Incremental R²: 0.04%

**ATC is a noisy version of drug Jaccard, not an independent signal.** This is because ATC codes come from GT drugs — fully circular. 50% of zero-drug pairs have non-zero ATC L2 overlap, but this doesn't help holdout.

**This closes the GT-free treatment paradigm proxy search.** No DRKG-derived signal (genes, HPO, ATC) provides independent information beyond drug Jaccard. Treatment paradigm knowledge requires treatment data.

### h590: Hetionet Disease-Resembles as Augmented kNN Signal — INVALIDATED

Tested whether Hetionet DrD (disease resembles disease) edges can augment kNN neighborhoods with curated medical knowledge.

**Findings:**
- Only 33/312 holdout diseases have resembles edges (10.6%)
- 80.8% of resembles neighbors are already in kNN top-20
- Drug overlap: resembles 0.036 < kNN 0.045 (kNN finds BETTER drug neighbors)
- Only 4/33 diseases gain ANY new GT drugs from resembles (mean 0.2/disease)
- Embedding already captures resembles (trained on same DRKG graph)

**Conclusion:** No augmentation value. Node2Vec embeddings subsume Hetionet edges.

### Session Summary: GT-Free Quality Signal Arc (h586→h588→h589→h590)

This session systematically explored whether DRKG-derived signals can independently predict treatment paradigm similarity:

| Signal | r with Drug Jaccard | r with Holdout | Independent? |
|--------|-------------------|---------------|-------------|
| Drug Jaccard (oracle) | 1.000 | +0.251 | GT-dependent |
| ATC L3 (h589) | +0.737 | +0.180 | Circular (GT) |
| HPO phenotype (h588) | +0.390 | +0.243 | Modest (+0.9% R²) |
| Gene overlap (h586) | +0.086 | +0.079 | None |
| Resembles (h590) | — | — | Subsumed by kNN |

**Key insight:** Treatment paradigm information exists ONLY in treatment data. No biological (genes), phenotypic (HPO), or graph-structural (resembles) signal provides independent prediction of treatment similarity. ATC hierarchy is a strong proxy but fully circular with drug Jaccard.

**The only partially independent signal is HPO phenotype similarity**, but at +0.9% incremental R², it's not actionable for tier changes.

### h591: LOW-Tier Success Pattern Analysis — VALIDATED (characterization)

Full-data analysis of which LOW predictions hit GT (20.0% = 747/3733).

Top success patterns: cancer_targeted_therapy 39.0%, immunological demotion 34.0%, Mech+Rank<=5 39.0%. 67.5% of LOW GT hits are known indications. All demotion rules confirmed correct. Useful for deliverable annotation, not tier changes.

### Recommended Next Steps
1. **h534**: TransE FILTER annotation for manual review (low effort)
2. **h539**: Cancer drug class annotation (low effort, deliverable improvement)
3. Consider pivoting to entirely external data (clinical guidelines, RWD, LINCS)

---

## Previous Session: h571 - Therapeutic Island Rescue Analysis (2026-02-06)

### h571: Therapeutic Island Disease Rescue — INVALIDATED

Comprehensive analysis of 9 "therapeutic island" diseases (GT>=5, 0% holdout) to determine whether alternative prediction strategies could rescue them.

**Islands Analyzed:**
| Disease | GT | Self-Ref | MEDIUM+ Preds | Failure Mode |
|---------|-----|---------|---------------|-------------|
| Immunodeficiency | 268 | 100% | 0 | immunological demotion |
| ADHD | 87 | 100% | 18 | hierarchy+ATC works, kNN blind |
| HCV | 50 | 100% | 2 | disease-specific antivirals |
| PAH | 36 | 100% | 11 | different paradigm than hypertension |
| Migraine | 35 | 100% | 2 | triptans/CGRPs not in kNN |
| Agranulocytosis | 31 | 83% | 0 | hematological demotion |
| Narcolepsy | 26 | 75% | 2 | stimulants unique to cluster |
| DKA | 12 | 100% | 0 | base_to_complication filter |
| Scabies | 9 | 80% | 2 | antiparasitic drugs unique |

**Key Finding 1: NOT drug uniqueness, but neighbor drug mismatch**
- All 9 islands have 67-100% of GT drugs shared with other diseases
- But kNN neighbors have VERY low drug overlap: mean 0.3-7.2 drugs (vs 6.2-22.6 for high performers)
- Islands are embedded NEAR other diseases (sim 0.498-0.791) but treated with DIFFERENT drugs
- e.g., PAH is near hypertension (uses PDE5i/ERA/prostacyclins) but neighbors use ACEi/ARBs/CCBs

**Key Finding 2: Alternative signals cannot help**
- TransE consilience: 0% for most island GT predictions
- Gene overlap: Present for ADHD/immunodeficiency but circular with kNN
- Drug class: Already exploited by hierarchy rules and ATC coherence
- All signals annotate EXISTING kNN predictions, cannot generate NEW ones

**Key Finding 3: System already works for some islands via non-kNN paths**
- ADHD: 18 MEDIUM+ predictions (8 known GT drugs) via psychiatric ATC + target overlap
- PAH: 11 HIGH predictions (all known) via cardiovascular hierarchy
- These non-kNN paths work; kNN just adds no value for these diseases

**Key Finding 4: 0% holdout is misleading for self-referential diseases**
- PAH has 11 correct HIGH predictions but holdout = 0%
- Holdout penalizes self-referential diseases because GT contributions vanish when held out
- The deliverable is actually CORRECT for these diseases

**Conclusion: No rescue possible within kNN architecture. Need fundamentally different approach.**

### New Hypotheses Generated (4)
- h580: Drug class expansion for migraine (P3, high impact)
- h581: Holdout metric correction excluding self-ref diseases (P4, medium, low effort)
- h582: kNN neighbor drug overlap as quality signal (P4, low, medium effort)
- h583: Treatment paradigm mismatch detection (P4, medium)

### Session Tier Performance (unchanged from h560)
| Tier | Holdout | Predictions |
|------|---------|-------------|
| GOLDEN | 69.9% ± 17.9% | 280 |
| HIGH | 59.5% ± 6.2% | 754 |
| MEDIUM | 35.8% ± 2.8% | 2083 |
| LOW | 15.5% ± 2.4% | 3733 |
| FILTER | 10.6% ± 1.3% | 7300 |

### h581: Holdout Metric Correction for Self-Referential Diseases — VALIDATED (major meta-finding)

Excluding 100% self-referential diseases from holdout reveals significantly higher true discovery rates:

| Tier | All (reported) | Non-Self-Ref | Delta | <50% Self-Ref |
|------|----------------|-------------|-------|---------------|
| GOLDEN | 70.3% ± 17.8 | 72.0% ± 16.8 | +1.7pp | 72.0% ± 19.0 |
| HIGH | 54.6% ± 4.8 | 59.4% ± 5.8 | +4.8pp | 60.8% ± 6.6 |
| MEDIUM | 37.6% ± 2.1 | 40.5% ± 3.0 | +2.9pp | 44.0% ± 2.8 |
| LOW | 13.7% ± 1.8 | 17.2% ± 1.5 | +3.5pp | 19.2% ± 1.0 |
| FILTER | 10.4% ± 1.3 | 13.6% ± 2.3 | +3.2pp | 15.5% ± 2.8 |

**Self-ref disease contribution to holdout predictions:**
- GOLDEN: 2% (negligible), HIGH: 13%, MEDIUM: 16%, LOW: 29%, FILTER: 34%

**Critical insight: MEDIUM 40% target IS ACHIEVED for non-self-ref diseases (40.5%)!**
The target deemed unachievable in h561 is actually met when excluding structural zeros.
For <50% self-ref: MEDIUM = 44.0% — exceeds 40% target significantly.

### New Hypotheses Generated (2)
- h584: Deliverable non-self-ref precision annotation (P5, low)
- h585: Self-referentiality threshold optimization (P5, low)

### h583: Treatment Paradigm Mismatch Detection — VALIDATED (novel independent signal)

Paradigm mismatch = mean_embedding_sim - mean_drug_Jaccard (high = diseases near in embedding but far in drug space).

**Key Results:**
| Signal | r (all) | Partial r (ctrl GT) | Partial r (ctrl self-ref) |
|--------|---------|--------------------|--------------------------|
| Drug Jaccard | +0.398 | +0.428 | — |
| Paradigm mismatch | -0.388 | -0.303 | -0.408 |
| Embedding sim | -0.177 | — | — |
| Mismatch vs self-ref | +0.032 | — | — |

**Critical: NOT a self-referentiality proxy (r=0.032).** This is an INDEPENDENT signal.

**Quartile analysis (all diseases):** Q1=15.7%, Q2=7.5%, Q3=5.4%, Q4=3.9% holdout.
**Non-self-ref only:** Q1=19.3% vs Q4=6.4%. Signal persists after removing self-ref.

**Limitation:** Drug Jaccard requires GT knowledge → circular at prediction time. Valid as annotation only.

### New Hypotheses Generated (2)
- h586: GT-free paradigm mismatch via DRKG edges (P4, medium)
- h587: Paradigm mismatch deliverable annotation (P5, low)

### h585: Self-Referentiality Threshold Optimization — VALIDATED (confirms 100% boundary)

Self-ref is bimodal: 0% (n=112) and 100% (n=144) dominate. Band analysis:

| Band | n | Mean Holdout | GT Size |
|------|---|-------------|---------|
| 0% | 80 | 8.7% | 20.2 |
| 1-25% | 21 | 26.8% | 86.0* |
| 26-50% | 54 | 14.0% | ~35 |
| 51-75% | 43 | 9.6% | ~25 |
| 76-99% | 15 | 5.2% | ~20 |
| 100% | 99 | 0.3% | ~15 |

*1-25% band confounded by large-GT diseases (RA, ovarian cancer, COPD).

**Within non-100% diseases, self-ref has MINIMAL predictive power (r=-0.104).**
Only the 100% vs <100% boundary matters. This confirms h581's choice of cutoffs.

### Recommended Next Steps
1. **h586**: GT-free paradigm mismatch approximation (if DRKG edges proxy drug Jaccard, novel non-circular signal)
2. **h580**: Drug class expansion for islands (high effort, new predictions)
3. **h584**: Add corrected precision to deliverable metadata

### Key Learning
Therapeutic islands fail because kNN neighbors treat with different drug classes, not because drugs are unique. The embedding space captures disease similarity but NOT treatment paradigm similarity. This is a fundamental limitation of Node2Vec embeddings trained on DRKG: they capture knowledge graph structure but not clinical treatment patterns.

---

## Previous Session: h576/h577/h579 - LOW Promotion + CS Artifact + Novel Precision (2026-02-06)

### h576: LOW Tier Promotion Analysis — INVALIDATED

Comprehensive analysis of 3,733 LOW predictions to identify promotion candidates.

**Holdout by tier_rule (5-seed):**
| Rule | Holdout% | n/seed | Notes |
|------|----------|--------|-------|
| incoherent_demotion | 44.2% ± 14.0% | 34 | Driven by CS→TB artifact |
| cardiovascular_medium_demotion | 25.1% ± 19.4% | 69 | Too variable |
| local_anesthetic_procedural | 23.8% ± 5.9% | 52 | Stable but below MEDIUM |
| hematological_corticosteroid_demotion | 23.7% ± 21.5% | 24 | High variance |
| default | 18.5% ± 3.7% | 298 | Appropriately LOW |

**Compound signals:**
| Signal | Holdout% | n/seed | Notes |
|--------|----------|--------|-------|
| TransE+Mech+Rank<=10 | 51.6% | 8 | Too small |
| TransE+Mech | 41.5% ± 19.4% | 14 | 73% are CS |
| Rank<=5+TransE | 40.0% ± 15.3% | 24 | CS-inflated |
| Freq>=10+Mech | 36.4% ± 11.7% | 54 | Mixed population |
| Mechanism overall | 25.0% ± 6.3% | 154 | Heterogeneous |
| TransE overall | 21.4% ± 7.7% | 94 | +6.6pp vs no TransE |

**Key finding: incoherent_demotion deep dive**
- h488 originally found 3.6% holdout for MEDIUM-level incoherent → LOW
- But incoherent_demotion also demotes HIGH-level (freq>=15+mech) → LOW
- HIGH-level incoherent = 44.2%, driven by CS→infectious (45.1%)
- Non-CS incoherent = 11.7% → correctly at LOW
- If promoted to MEDIUM, h557 post-processing would re-demote CS→infectious to LOW
- Net effect: only 11 non-CS predictions at 11.7% would be promoted → WORSE for MEDIUM
- **Decision: DO NOT PROMOTE. All demotion rules are correctly calibrated.**

### h579: MEDIUM Novel-Only Precision — VALIDATED (structural finding)

**100% of predicted drugs treat at least one training disease.** Zero "novel drug" predictions exist.

This is structural: kNN collaborative filtering only recommends drugs from similar diseases. If a drug doesn't treat ANY training disease, it won't appear in kNN neighbors. The system is inherently a drug REPURPOSING engine — all predictions are cross-disease transfer.

All holdout hits represent genuine drug repurposing: drug known for training diseases, correctly predicted for held-out disease. The tier precision numbers represent true repurposing discovery rates.

### h577: Corticosteroid Holdout Artifact — VALIDATED (major meta-finding)

High-frequency corticosteroids (freq 30-42) inflate holdout precision:

| Tier | All | CS | Non-CS | CS % | Inflation |
|------|-----|-----|--------|------|-----------|
| GOLDEN | 69.9% | 100.0% | 65.2% | 39% | +34.8pp |
| HIGH | 58.8% | 61.7% | 48.5% | 69% | +13.2pp |
| MEDIUM | 36.5% | 65.1% | 34.8% | 6% | +30.3pp |
| LOW | 15.5% | 22.8% | 14.3% | 14% | +8.4pp |
| FILTER | 10.6% | 21.5% | 10.1% | 5% | +11.4pp |

**Key insights:**
1. **HIGH is 69% corticosteroids!** Non-CS HIGH precision is 48.5%, not 58.8%
2. **MEDIUM barely affected** (6% CS): non-CS 34.8% vs total 36.5%
3. **Tier ordering preserved for non-CS**: GOLDEN 65.2% > HIGH 48.5% > MEDIUM 34.8% > LOW 14.3% > FILTER 10.1%
4. Biggest category inflation: renal +58pp, metabolic/musculoskeletal +44pp
5. Cancer/cardiovascular: NO CS inflation

**Implication**: Report CS-free precision as supplemental "discovery potential" metric. The tier system is valid but its numbers overstate non-obvious discovery potential, especially for HIGH tier.

### New Hypotheses Generated (3)
- h577: CS holdout artifact (P4, medium) — COMPLETED
- h578: LOW TransE annotation (P5, low)
- h579: Novel-only precision (P4, low) — COMPLETED

### Session Tier Performance (unchanged from h559)
| Tier | Holdout | Non-CS Holdout |
|------|---------|----------------|
| GOLDEN | 69.9% ± 17.9% | 65.2% |
| HIGH | 58.8% ± 6.2% | 48.5% |
| MEDIUM | 36.5% ± 3.0% | 34.8% |
| LOW | 15.5% ± 2.4% | 14.3% |
| FILTER | 10.6% ± 1.3% | 10.1% |

### Recommended Next Steps
1. **h577 follow-up**: Add CS-free precision to deliverable metadata
2. **h578**: Flag best LOW predictions for manual review
3. Consider pivoting to external data integration (h545, h91) since internal improvements are exhausted

### Key Learning
LOW→MEDIUM promotion is NOT possible with current signals. All demotion rules are correctly calibrated. CS inflation is a meta-issue that inflates tier precision numbers but doesn't affect tier ordering. The system's "true" discovery potential for non-obvious drug repurposing is ~48.5% for HIGH (not 58.8%) and ~34.8% for MEDIUM (not 36.5%). Future improvement requires external data or fundamentally new signals.

---

## Previous Session: h563/h567/h572 - Promotion/Mismatch/Coherence Analysis (2026-02-06)

### h563: LA Procedural MEDIUM→HIGH Promotion — INCONCLUSIVE

LA procedural MEDIUM predictions are LA drugs demoted to LOW by h540 then rescued to MEDIUM by target_overlap.
Only 41 full-data predictions (6.6/seed holdout). 28/41 are bupivacaine.
- Full-data precision: 31.7% (at MEDIUM level, not HIGH)
- Holdout: 24.9% ± 16.4% — too noisy with n=6.6/seed
- Decision: KEEP AS-IS. Too few predictions to justify code change.

### h567: Drug Class × Disease Type Mismatch Matrix — VALIDATED (confirms demotion ceiling)

Comprehensive cross-tabulation of 18 SOC drug classes + 12 broad therapeutic classes × 14 disease categories for MEDIUM predictions.
- Only 1 candidate: DMARDs→cancer (19.5%, n=41, 2.0% holdout) — BUT all 41 are methotrexate, which IS a cancer drug
- Anti-thyroid→metabolic: 0% (n=10) — genuine but too small
- **CONCLUSION: Existing filters are comprehensive. No new demotion rules available.**
- Demotion ceiling at ~35.8% MEDIUM confirmed

### h572: kNN Neighborhood Category Coherence — INVALIDATED

Tested whether fraction of same-category among k=20 kNN neighbors predicts precision.
- r = -0.002 (coherence vs holdout precision) — ZERO signal
- r = -0.028 (coherence vs GT size) — ZERO signal
- Node2Vec embeddings cluster by drug-sharing patterns, NOT disease category
- 91% of diseases have <20% same-category neighbors (mean=0.064, median=0.000)
- **Key insight: kNN works via drug-pattern similarity, not category similarity**

### New Hypotheses Generated (3)
- h573: kNN score gap as prediction confidence (P4, medium)
- h574: Drug-sharing density as disease quality signal (P5, low)
- h575: Methotrexate cancer subtype specificity (P5, medium)

### h573: kNN Score Gap as Prediction Confidence — VALIDATED

kNN norm_score adds signal beyond rank for prediction quality:
- Within-rank: high-score vs low-score = +9.2pp for rank 1-5, +8.8pp for rank 6-10
- Within MEDIUM: Q4 (highest score) 28.2% vs Q1 (lowest) 12.6% (+15.6pp)
- Score gap Q4: 29.2% within MEDIUM
- NOT circular with GT size (r=-0.009)
- Q4 MEDIUM (28.2%) << HIGH (59.5%) — useful as annotation, not promotable
- norm_score already stored in deliverable; no code change needed

### h574: Drug-Sharing Density as Disease Quality Signal — VALIDATED (circular)

Mean drug overlap between disease and k=20 kNN neighbors independently predicts holdout precision:
- r=0.434 with holdout precision
- Partial r=0.448 AFTER controlling for GT size (independent!)
- r=0.182 with GT size (NOT a proxy)
- 93% of diseases have near-zero drug sharing — most holdout diseases share no drugs with neighbors
- CIRCULAR (uses GT) — annotation only, not for novel predictions
- Key insight: kNN generalizes when neighbors share drugs; fails structurally when they don't

### h559: CS→Infectious HIGH TB Hierarchy Demotion — VALIDATED (marginal)

CS→TB hierarchy predictions demoted from HIGH→MEDIUM:
- 18 predictions (CS drugs × 3 TB diseases)
- Full-data: CS→TB 33.3% vs non-CS infectious HIGH 76.9% (43.6pp gap)
- Holdout: HIGH -0.7pp, MEDIUM +0.8pp (both within seed variance)
- Medically justified (dexamethasone→TB meningitis is valid, but generic CS→TB is not SOC)
- Protected from h557 cascade (stays MEDIUM, not demoted to LOW)

### Session Tier Performance (h559 update)
| Tier | Holdout | Delta vs h560 |
|------|---------|---------------|
| GOLDEN | 69.9% ± 17.9% | 0.0pp |
| HIGH | 58.8% ± 6.1% | -0.7pp |
| MEDIUM | 36.6% ± 3.0% | +0.8pp |
| LOW | 15.5% ± 2.4% | 0.0pp |
| FILTER | 10.6% ± 1.3% | 0.0pp |

### Recommended Next Steps
1. **h571**: Therapeutic island rescue (P3, high impact but high effort)
2. **h545**: Gene-poor disease expansion (P4, medium)
3. **h573 follow-up**: Consider norm_score thresholds for deliverable prioritization

### Key Learning
MEDIUM demotion is exhausted at 35.8%. All major drug-class × category mismatches are filtered.
Future MEDIUM improvement requires: (1) promotions, (2) new signals, or (3) external data.
Embedding space clusters by drug sharing, not disease category — quality signals must exploit this structure.

---

## Previous Session: h560 - Antimicrobial-Pathogen Mismatch Filter (2026-02-06)

### h569: Disease-Level Precision Audit — VALIDATED

37% of diseases (121/325) have 0% holdout precision. 80% of these have GT≤2 (structural limit).
GT size strongly predicts disease-level precision (r=0.732):
- GT≤2: 1.1% | GT 3-5: 5.0% | GT 6-10: 12.4% | GT 11-20: 19.7% | GT 21-50: 28.3% | GT 51+: 70.0%
Notable therapeutic island failures: PAH (GT=26, 0%), HCV (GT=11, 0%), migraine (GT=9, 0%).
Top performers: RA (93.3%), UC (66.7%), AS (63.3%) — all large-GT autoimmune diseases.

### h570: Disease Confidence Annotation — VALIDATED

Added `disease_holdout_precision` column to deliverable (9336/14150 predictions annotated).
Per-disease holdout precision computed across 5 seeds. Fixed json import shadowing bug.

### h560: Antimicrobial-Pathogen Mismatch Filter — VALIDATED

Extended h556's antibiotic→viral filter to comprehensive antimicrobial-pathogen mismatch detection.

**Key Finding:** 0.0% holdout precision for ALL antimicrobial-pathogen mismatches across 5 seeds (132 total mismatches, 0 hits). Matched predictions = 27.8% ± 4.8%.

**Mismatch Types Detected:**
| Mismatch Type | n/seed | Notes |
|---------------|--------|-------|
| antibacterial → parasitic | 8.0 | Cephalosporins/FQs for malaria/toxo/leish |
| antibacterial → fungal | 7.4 | FQs/macrolides for candidiasis/aspergillosis |
| antifungal → parasitic | 4.8 | Azoles/echinocandins for schistosomiasis/Chagas |
| antibacterial → viral | 3.0 | Already partially covered by h556 |
| antifungal → viral | 0.8 | Amphotericin B for hepatitis C |
| Other | 2.4 | Mixed |

**Dual-Activity Drug Handling:**
- Metronidazole: antibacterial + antiparasitic (treats trichomoniasis, amebiasis)
- Doxycycline/tetracycline: antibacterial + antiparasitic (malaria prophylaxis)
- Amphotericin B: antifungal + antiparasitic (leishmaniasis first-line)
- Sulfadiazine: antibacterial + antiparasitic (toxoplasmosis first-line)

**10 Legitimate Cross-Pathogen Pairs Excluded** (e.g., doxycycline→malaria, amphotericin B→leishmaniasis, ketoconazole→Chagas)

**Bug Found:** Target overlap promotion was rescuing 11 mismatch predictions from LOW back to MEDIUM. Fixed by adding `antimicrobial_pathogen_mismatch` to the target_overlap block list.

**Implementation:**
- Replaced h556's `antibiotic_viral_mismatch` with comprehensive `antimicrobial_pathogen_mismatch` rule
- Drug classification: antibacterial (48 drugs), antifungal (15), antiparasitic (20), dual-activity (5)
- Disease classification: viral (14 keywords), fungal (15 keywords), parasitic (11 keywords)
- ~30 MEDIUM predictions demoted to LOW, 292 total predictions tagged

**Holdout Impact:**
| Tier | Before (h562) | After (h560) | Delta |
|------|--------------|-------------|-------|
| GOLDEN | 69.9% ± 17.9% | 69.9% ± 17.9% | 0.0pp |
| HIGH | 59.5% ± 6.2% | 59.5% ± 6.2% | 0.0pp |
| **MEDIUM** | **34.9% ± 3.1%** | **35.8% ± 2.8%** | **+0.9pp** |
| LOW | 16.0% ± 2.5% | 15.5% ± 2.4% | -0.5pp |
| FILTER | 10.4% ± 1.3% | 10.6% ± 1.3% | +0.2pp |

**Cumulative MEDIUM improvement since h553:** +5.7pp (30.1% → 35.8%)

**New Hypotheses Generated (3):**
- h565: Azole antifungal anti-parasitic activity validation (P5, low)
- h566: Infectious target_overlap quality audit (P5, medium)
- h567: Drug class × disease type matrix for all categories (P4, high)

**Recommended Next Steps:**
1. **h567**: Systematic drug-class × disease-subtype mismatch scan across all categories
2. **h559**: CS→infectious HIGH TB hierarchy review
3. **h563**: LA procedural MEDIUM→HIGH promotion

---

## Previous Session: h557 - Corticosteroid→Infectious Demotion (2026-02-06)

### h557: Corticosteroid→Infectious Disease Selective Demotion — VALIDATED

Analyzed all 174 corticosteroid→infectious disease predictions across all tiers.

**Medical Classification of 33 Infectious Diseases:**
| Validity | Diseases | Rationale |
|----------|----------|-----------|
| VALID (6) | ABPA, herpes zoster, leprosy, TB, extrapulmonary TB, proctitis | CS are established adjunctive therapy |
| QUESTIONABLE (11) | Cryptococcosis, fungal meningitis, influenza, HSE, aspergillosis, etc. | Some evidence but not standard |
| INVALID (16) | Hep B/C, CMV, rabies, smallpox, zygomycosis, candidiasis, etc. | CS harmful or useless |

**Holdout Precision (5-seed):**
| Group | Holdout | n/seed | vs MEDIUM avg |
|-------|---------|--------|--------------|
| ALL CS→infectious MEDIUM | 2.1% ± 2.5% | 11.6 | -31.8pp |
| VALID CS→infectious | 2.9% ± 3.5% | 8.0 | -31.0pp |
| QUESTIONABLE | 0.0% | 1.8 | -33.9pp |
| INVALID | 0.0% | 1.8 | -33.9pp |
| Non-CS infectious MEDIUM | 18.7% ± 5.2% | 113.4 | -15.2pp |

**Key Finding:** Medical validity does NOT predict holdout performance. Even ABPA/zoster/leprosy/TB (genuinely valid uses) have 2.9% holdout. The KG co-occurrence signal doesn't generalize when specific diseases are held out.

**Implementation:**
- Added `infectious_corticosteroid_demotion` rule: CS + infectious + MEDIUM → LOW
- 59 predictions demoted
- Rule classified as GENUINE (16.1% ± 7.4% holdout = LOW-level)

**Holdout Impact:**
| Tier | Before (h555) | After (h557) | Delta |
|------|--------------|-------------|-------|
| GOLDEN | 70.3% ± 17.8% | 69.9% ± 17.9% | -0.4pp |
| HIGH | 58.7% ± 6.1% | 59.5% ± 6.2% | +0.8pp |
| **MEDIUM** | **33.9% ± 2.5%** | **34.2% ± 2.7%** | **+0.3pp** |
| LOW | 16.2% ± 2.7% | 16.0% ± 2.5% | -0.2pp |
| FILTER | 10.5% ± 1.3% | 10.4% ± 1.3% | -0.1pp |

**Cumulative MEDIUM improvement since h553:** +4.1pp (30.1% → 34.2%)

**New Hypotheses Generated (3):**
- h559: CS→infectious HIGH (TB hierarchy) review (P5, low)
- h560: Antifungal↔bacterial cross-pathogen mismatch (P5, medium)
- h561: Cumulative MEDIUM precision analysis vs 40% target (P4, medium)

**Recommended Next Steps:**
1. **h561**: MEDIUM precision gap analysis — what's left to improve?
2. **h560**: Cross-pathogen drug-disease mismatch filter
3. **h532**: Every Cure GT error report

### h561: Cumulative MEDIUM Precision vs 40% Target — VALIDATED

**Comprehensive sub-reason analysis shows MEDIUM demotion ceiling reached at ~34.2%.**

**MEDIUM Sub-Reason Holdout Precision (5-seed, proper GT):**
| Sub-Reason | Holdout | n/seed | % of MEDIUM | Notes |
|------------|---------|--------|-------------|-------|
| cancer_same_type | 27.9% | 125 | 39.3% | Largest drag, genuine MEDIUM |
| default | 36.6% | 101 | 29.6% | At average |
| atc_coherent_infectious | 36.4% | 50 | 9.1% | At average |
| target_overlap_promotion | 43.0% | 31 | 5.9% | Above average |
| cv_pathway_comprehensive | 40.8% | 16 | 4.8% | Above average |
| local_anesthetic_procedural | 50.5% | 11 | 1.9% | Potential promote to HIGH |
| atc_coherent_psychiatric | 45.3% | 11 | 2.8% | Above average |

**Ceiling Analysis:**
- Only 1 remaining demotion candidate: infectious_hierarchy_pneumonia (16.7%, n=6) — too small
- Max from further demotions: ~35.0% (negligible improvement)
- Without cancer_same_type: 38.3% (but loses 846 genuine predictions)
- **40% NOT achievable via demotions**

**MEDIUM Precision Journey:**
30.1% → 31.7% (h553) → 32.1% (h556) → 33.9% (h555) → 34.2% (h557) = **+4.1pp from 770 demotions**

**New Hypotheses Generated (3):**
- h562: Cancer same-type subtype specificity (P4, medium) — highest impact remaining
- h563: LA procedural MEDIUM→HIGH promotion (P5, low)
- h564: Deliverable regeneration with updated tiers (P4, low)

### h562: Cancer Same-Type Subtype Specificity Analysis — VALIDATED (Bug Fix)

Expected to find cross-subtype contamination. Instead found a substring matching bug.

**Finding 1:** All 846 cancer_same_type predictions were already SAME_SUBTYPE (100%). No cross-subtype issue.

**Finding 2:** `extract_cancer_types()` had a substring bug with short abbreviations:
- `'ALL'` (Acute Lymphoblastic Leukemia) matched "sm**all**", "f**all**opian", "**all**ergic"
- 8 diseases falsely tagged as leukemia, inflating cancer_same_type count by 39 predictions

**Fix:** Word boundary regex (`\b`) for keywords <=4 chars (ALL, CLL, AML, CML, SCLC).

**Impact:**
| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| MEDIUM predictions | 2152 | 2113 | -39 |
| cancer_same_type | 846 | 807 | -39 |
| cancer_same_type holdout | 27.9% | 29.4% | +1.5pp |
| **Overall MEDIUM holdout** | **34.2%** | **34.9%** | **+0.7pp** |

**Recommended Next Steps:**
1. **h564**: Deliverable regeneration (practical impact for collaboration)
2. **h560**: Cross-pathogen mismatch filter (infectious sub-type)
3. **h532**: Every Cure GT error report

---

## Previous Session: h553+h554+h555+h556 - MEDIUM Precision Deep Dive (2026-02-06)

### h553: MEDIUM Tier Precision by Category Analysis — VALIDATED

Computed holdout precision for MEDIUM predictions broken down by disease category.

**Category-Level Holdout Precision (MEDIUM only, 5-seed mean):**
| Category | Holdout | ±std | n/seed | vs avg |
|----------|---------|------|--------|--------|
| psychiatric | 54.8% | 6.6% | 19 | +24.6pp |
| musculoskeletal | 53.8% | 28.6% | 12 | +23.7pp |
| cardiovascular | 35.4% | 9.9% | 21 | +5.3pp |
| dermatological | 31.2% | 4.6% | 28 | +1.1pp |
| autoimmune | 30.7% | 14.9% | 23 | +0.6pp |
| respiratory | 29.2% | 9.0% | 14 | -0.9pp |
| cancer | 27.9% | 4.5% | 125 | -2.2pp |
| infectious | 26.6% | 8.1% | 104 | -3.5pp |
| **metabolic** | **18.0%** | 6.7% | 16 | **-12.1pp** |
| **hematological** | **10.0%** | 20.0% | 8 | **-20.1pp** |

**Sub-Reason Analysis (MEDIUM):**
| Sub-reason | Holdout | n/seed | vs avg |
|------------|---------|--------|--------|
| target_overlap_promotion | 45.3% | 45 | +15.2pp |
| local_anesthetic_procedural | 44.2% | 13 | +14.1pp |
| cv_pathway_comprehensive | 40.8% | 16 | +10.6pp |
| cardiovascular | 36.4% | 8 | +6.3pp |
| cancer_same_type | 27.9% | 125 | -2.2pp |
| default | 26.9% | 184 | -3.3pp |
| **metabolic (statin/TZD rescue)** | **8.3%** | 4.2 | **-21.8pp** |

**Changes Implemented:**
1. Hematological MEDIUM→LOW demotion (default sub-reason: 0% holdout, n=6/seed × 4 seeds)
2. Hematological blocked from target_overlap LOW→MEDIUM promotion (n=1.2/seed)
3. Metabolic statin/TZD category rescue demoted MEDIUM→LOW (8.3% holdout)
4. Metabolic default sub-reason NOT demoted (32.9% holdout — at MEDIUM level)

**Holdout Impact:**
| Tier | Before | After | Delta |
|------|--------|-------|-------|
| GOLDEN | 69.9% | 70.3% | +0.4pp |
| HIGH | 58.7% | 54.5%* | -4.2pp* |
| **MEDIUM** | **30.1%** | **31.7%** | **+1.6pp** |
| LOW | 16.2% | 14.2%* | -2.0pp* |
| FILTER | 10.5% | 10.3% | -0.2pp |

*HIGH and LOW changes are seed variance — code changes only affect MEDIUM→LOW transitions.

**New Hypotheses Generated (3):**
- h554: Target overlap promotion to HIGH (45.3% within MEDIUM, P4)
- h555: MEDIUM default sub-reason deep dive (26.9%, P5)
- h556: Infectious MEDIUM precision gap (26.6%, P4)

**Recommended Next Steps:**
1. **h554**: Target overlap promotion MEDIUM→HIGH (potentially highest impact)
2. **h556**: Infectious MEDIUM precision gap analysis
3. **h550**: Antibiotic spectrum validation (overlaps with h556)

### h554: Target Overlap Promotion to HIGH — INCONCLUSIVE

target_overlap_promotion within MEDIUM has 43.0% ± 6.3% holdout (31/seed). Category-heterogeneous:
- psychiatric 53.7%, infectious 53.3% — at HIGH level
- metabolic 18.3%, autoimmune 11.9% — terrible

Best strategy (exclude worst categories): HIGH -0.1pp (neutral), 21 promoted. Net effect marginal.
All target_overlap_promotion predictions have overlap=1 (minimum). Signal is binary, not graded.
Existing deliverable annotation already allows manual prioritization. Not worth implementation complexity.

### h556: Infectious MEDIUM Precision Gap — VALIDATED

Antibiotic → viral disease mismatch identified and implemented:
- 35 predictions caught (antibiotics for influenza, HSV, CMV, smallpox, AIDS, etc.)
- Full-data: 3.3% (1/30), holdout: 5.0% (5.8/seed)
- MEDIUM +0.4pp (31.7% → 32.1%)
- Corticosteroid→infectious (12.1% holdout, 20.4/seed) NOT demoted — includes valid uses

See session summary above for cumulative impact.

---

## Previous Session: h542+h551+h552+h548+h549 - MEDIUM Quality + Gene Overlap (2026-02-06)

### h542: Deliverable Quality Audit Round 2: MEDIUM Tier Top 59 — VALIDATED

Literature validation of 59 diverse MEDIUM novel predictions against PubMed/clinical guidelines:

**Overall Results:**
| Rating | Count | % | Comparison (GOLDEN/HIGH h537) |
|--------|-------|---|------------------------------|
| VALIDATED | 15 | 25.4% | 58.0% |
| PLAUSIBLE | 18 | 30.5% | 30.0% |
| IMPLAUSIBLE | 26 | 44.1% | 12.0% |
| **Reasonable** | **33** | **55.9%** | **88.0%** |

**Error Patterns (26 implausible):**
1. **Wrong cancer type/mechanism** (7): cancer_same_type overgeneralizes (hematologic→solid, bleomycin→non-SCC)
2. **Wrong antibiotic spectrum** (6): tetracyclines for resistant Shigella, bacteriostatic for meningococcal, poor urinary excretion for pyelonephritis
3. **Wrong drug class** (6): PTU→acromegaly, phenobarbital→pain/agoraphobia
4. **Local anesthetic artifact** (5): already handled by h540
5. **Inverse indication** (1): betamethasone CAUSES adrenocortical insufficiency
6. **Non-therapeutic compound** (1): FDG PET tracer is diagnostic, not drug

**By Drug Group:**
| Group | n | Reasonable% |
|-------|---|------------|
| Corticosteroids | 6 | 83% |
| Other drugs | 9 | 67% |
| Antifungals | 7 | 57% |
| Tetracyclines | 16 | 56% |
| Cancer drugs | 15 | 53% |
| Local anesthetics | 6 | 17% |

**Fixes Implemented:**
1. **Corticosteroid→adrenocortical insufficiency inverse indication**: 6 predictions (1 HIGH + 5 MEDIUM) → FILTER. Long-acting CS cause HPA suppression. Hydrocortisone/cortisone/corticotropin preserved as legitimate replacement.
2. **Fludeoxyglucose (18F) non-therapeutic compound filter**: 55 predictions (29 MEDIUM + 20 LOW) → FILTER. PET radiotracer, not a drug.

**Holdout After Fixes:**
| Tier | Holdout | Change |
|------|---------|--------|
| GOLDEN | 69.9% ± 17.9% | 0.0pp |
| HIGH | 58.7% ± 6.1% | -0.2pp |
| MEDIUM | 29.9% ± 2.4% | -0.4pp |
| LOW | 16.2% ± 2.7% | 0.0pp |
| FILTER | 10.5% ± 1.3% | +0.2pp |

**New Hypotheses Generated (4):**
- h550: Antibiotic spectrum validation (wrong-pathogen filter) — P4, high effort
- h551: Cancer same-type hematologic vs solid drug specificity — P4, medium effort
- h552: Non-therapeutic compound audit (other diagnostic agents) — P5, low effort
- h553: MEDIUM precision by category analysis — P5, medium effort

### h551: Cancer Drug Hematologic vs Solid Specificity — INCONCLUSIVE

- Only 16 cross-type predictions (heme drug→solid cancer or vice versa)
- 0% known indication rate vs 36% for same-type
- Too small for reliable holdout measurement
- cancer_same_type holdout 27.4% is appropriate for MEDIUM tier

### h552: Non-Therapeutic Compound Audit — VALIDATED

- Found indocyanine green (diagnostic imaging dye): 10 MEDIUM preds → FILTER
- Combined with h542's FDG fix: 66 total non-therapeutic predictions removed
- No other diagnostic agents found in 1,004 unique drugs

### h548: Gene-Poor Disease kNN Quality — VALIDATED

- Gene overlap is independent of self-referentiality (r=0.047)
- Gene-poor diseases NOT more self-referential than gene-rich
- Validates gene_overlap_count as genuine molecular signal

### h549: Gene Overlap Dose-Response — INVALIDATED

- Signal is BINARY (>0 vs 0), NOT proportional to count
- kNN score actually decreases with higher overlap
- Category confound: overlap 51+ is 100% cancer
- Existing binary annotation is sufficient

**Recommended Next Steps:**
1. **h550**: Antibiotic spectrum validation (wrong-pathogen filter) — P4, high effort
2. **h532**: Every Cure GT error report (low effort, medium impact)
3. **h539**: Cancer drug class annotation (low effort)

---

## Previous Session: h408+h546+h544 - Ryland Brief + Gene Overlap + Anti-TNF Audit (2026-02-06)

### h544: Anti-TNF Paradoxical Autoimmunity Audit — VALIDATED

Literature review + VigiBase analysis of anti-TNF paradoxical effects:

**Class effects (all anti-TNF agents):**
| Condition | Evidence | Cases |
|-----------|----------|-------|
| Autoimmune hepatitis | STRONG | 389 VigiBase |
| Sarcoidosis | STRONG | 90+ cases |
| Vasculitis | STRONG | 113 cases |
| SLE | STRONG | 12,080 FAERS (h408) |
| MS/demyelination | STRONG | FDA warning |

**Drug-specific (adalimumab):**
- Polymyositis: 20 cases (MODERATE)
- Lichen planus: 21 cases (MODERATE)

**NOT paradoxical (correctly left in):** GVHD (treated by anti-TNF), TEN (86.8% response to anti-TNF), GCA (failed RCT, not harmful)

**Impact:** 15 new inverse indication pairs, 5 predictions → FILTER (2 GOLDEN + 2 MEDIUM + 1 LOW). Holdout unchanged.

---

## Previous Session: h408+h546 - Ryland Brief + Gene Overlap Signal (2026-02-06)

### h546: Drug-Target/Disease-Gene Overlap as Confidence Signal — VALIDATED

Gene overlap (shared genes between drug targets and disease-associated genes) is a strong
holdout-validated signal within every tier:

| Tier | Overlap | No Overlap | Delta |
|------|---------|------------|-------|
| GOLDEN | 81.7% | 72.0% | +9.7pp |
| HIGH | 71.9% | 61.5% | +10.4pp |
| **MEDIUM** | **57.0%** | **36.6%** | **+20.3pp** |
| LOW | 30.9% | 19.5% | +11.4pp |
| FILTER | 26.6% | 14.1% | +12.5pp |

**Confound analysis**: Signal partially inflated by known indication bias (40.9% vs 24.1%)
and disease gene count. After controlling for NOVEL-only:
- MEDIUM novel overlap: 27.1% vs 15.7% (+11.4pp) — signal persists
- Category-controlled: +5.9pp to +21.5pp — persists everywhere
- Gene-poor diseases: +16.7pp (strongest for diseases with <50 genes)

**NOT promotable**: MEDIUM novel overlap (27.1%) << HIGH (58.9%). Partially circular with kNN.
**Implemented**: `gene_overlap_count` annotation column in deliverable.

---

## Previous Session: h408 - Ryland Collaboration Brief + Anti-TNF Safety Filter (2026-02-06)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 1**
- h408: [RYLAND] Transcriptomic Validation of Top Predictions in Skin/Inflammatory Diseases - **VALIDATED**

### Key Findings

#### 1. Collaboration Brief Prepared
- **Output**: `data/analysis/h408_ryland_collaboration_brief.md` (comprehensive 5-section brief)
- **Output**: `data/analysis/h408_ryland_predictions.xlsx` (407 curated predictions with drug mechanisms, gene overlap)
- 227 novel GOLDEN/HIGH predictions across 40 derm/autoimmune diseases
- 93/407 predictions have drug-target/disease-gene overlap (molecular support)

#### 2. Corticosteroid Dominance
- **86% of novel GOLDEN/HIGH predictions are corticosteroids** — clinically valid but not novel
- Only 15 non-CS GOLDEN predictions: adalimumab (9), azathioprine (2), rituximab (2), corticotropin (2), methotrexate (1)

#### 3. Literature Validation of Top Non-CS Predictions
| Prediction | Status | Evidence |
|------------|--------|----------|
| Azathioprine → Alopecia Areata | **VALIDATED** | 10-year cohort, 92.7% regrowth |
| Corticotropin → Alopecia Areata | Mechanistic only | ACTH upregulated in lesions, no trials |
| Adalimumab → SLE | **HARMFUL** | Anti-TNF INDUCES lupus (12K FAERS reports) |
| Adalimumab → MG | **HARMFUL** | Anti-TNF CAUSES MG (case reports) |
| Adalimumab → MS | **HARMFUL** | Paradoxical demyelination |
| Adalimumab → GCA | Failed RCT | Phase 2: no benefit vs placebo |
| Etanercept → SLE | **HARMFUL** | Same drug-induced lupus class effect |

#### 4. Safety Fix: Anti-TNF Inverse Indications
Added to INVERSE_INDICATION_PAIRS in production_predictor.py:
- Adalimumab → SLE, MG, MS (3 pairs)
- Etanercept → SLE, MS (2 pairs)
- Infliximab → SLE, MS (2 pairs)
- **Impact**: 4 predictions GOLDEN/MEDIUM → FILTER
- **Holdout**: HIGH 58.9% (+0.1pp), MEDIUM 30.3% (+0.1pp)

#### 5. Collaboration Opportunities Identified
- Ryland's spatial transcriptomics can provide gene signatures for gene-poor diseases (ichthyosis=8, TEN=2, HS=2 genes)
- Drug-target database (11,656 pairs) can prioritize drugs for cell culture testing
- Azathioprine and ACTH for alopecia areata are strongest testable predictions

### Current Tier Performance (h408)
| Tier | Holdout | Predictions |
|------|---------|-------------|
| GOLDEN | 69.9% ± 17.9% | 282 |
| HIGH | 58.9% ± 6.2% | 791 |
| MEDIUM | 30.3% ± 2.5% | 2655 |
| LOW | 16.2% ± 2.7% | 3140 |
| FILTER | 10.3% ± 1.4% | 7282 |

### New Hypotheses Generated (4)
- h544: Anti-TNF paradoxical autoimmunity comprehensive audit (P4, medium)
- h545: Gene-poor disease expansion from DisGeNET/OMIM (P4, medium)
- h546: Drug-target/disease-gene overlap as confidence signal (P4, low)
- h547: Corticosteroid prediction deduplication for deliverable (P5, medium)

### Recommended Next Steps
1. **h544**: Anti-TNF paradoxical autoimmunity audit (could find more harmful predictions)
2. **h546**: Drug-target/disease-gene overlap as confidence signal (low effort, potentially useful)
3. **h542**: MEDIUM tier quality audit round 2

---

## Previous Session: h537+h540 - Deliverable Quality Audit + LA Demotion (2026-02-06)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 2**
- h537: Deliverable Quality Audit: Sample-Based Validation of Top 50 - **VALIDATED**
- h540: Local Anesthetic Non-Pain Demotion - **VALIDATED** (HIGH +0.3pp, 132 predictions demoted)

### Key Findings

#### 1. Literature Validation Results
Audited top 50 GOLDEN/HIGH novel predictions against PubMed, FDA, clinical guidelines:
- **Overall**: 29/50 (58%) VALIDATED, 15/50 (30%) PLAUSIBLE, 6/50 (12%) IMPLAUSIBLE
- **GOLDEN**: 100% reasonable (65% validated, 35% plausible, 0% implausible)
- **HIGH**: 80% reasonable (53% validated, 27% plausible, 20% implausible)

#### 2. Three Systematic Error Patterns
1. **Local Anesthetic Procedural Confusion** (2/6 errors, 27/29 GOLDEN/HIGH LA preds affected)
   - Lidocaine/bupivacaine predicted for non-pain diseases (TB, GVHD, JIA, MS, etc.)
   - Root cause: KG edges from procedural co-occurrence, not therapeutic use
   - Impact: 27 GOLDEN/HIGH predictions are artifacts
2. **Wrong Antibiotic Spectrum** (3/6 errors)
   - Erythromycin/minocycline → meningitis (poor BBB penetration)
   - Doxycycline → Pseudomonas CF (inherently resistant)
3. **Statin → Diabetes Inverse Indication** (1/6 errors)
   - Statins CAUSE diabetes (2024 Lancet meta-analysis: 10-36% increase)
   - **FIX APPLIED**: 7 statins → diabetes/hyperglycemia added to INVERSE_INDICATION_PAIRS
   - 12 predictions moved to FILTER

#### 3. Holdout Impact
Negligible — statin filter affects too few predictions to move tier averages.
| Tier | Holdout | Change |
|------|---------|--------|
| GOLDEN | 69.9% ± 17.9% | 0.0pp |
| HIGH | 58.5% ± 7.1% | 0.0pp |
| MEDIUM | 30.0% ± 2.8% | +1.2pp |
| LOW | 15.5% ± 2.7% | -0.1pp |
| FILTER | 10.3% ± 1.4% | 0.0pp |

### New Hypotheses Generated (4)
- h540: Local anesthetic non-pain demotion (P4, medium) — highest impact
- h541: Antibiotic spectrum annotation (P5, medium)
- h542: MEDIUM tier quality audit round 2 (P5, medium)
- h543: Corticosteroid prediction saturation analysis (P5, low)

#### 4. h540: Local Anesthetic Demotion (VALIDATED)
- Bupivacaine: demoted to LOW for ALL categories (no systemic therapeutic use)
- Lidocaine: demoted to LOW for non-therapeutic categories (neurological/CV/dermatological/psychiatric preserved)
- 132 predictions moved GOLDEN/HIGH/MEDIUM → LOW
- HIGH: 58.5% → 58.8% (+0.3pp holdout), MEDIUM: 30.0% → 30.2% (+0.2pp)
- `local_anesthetic_procedural` rule: 28.6% ± 4.9% holdout (GENUINE)

### Current Tier Performance (h540)
| Tier | Holdout | Predictions |
|------|---------|-------------|
| GOLDEN | 69.9% ± 17.9% | 285 |
| HIGH | 58.8% ± 6.2% | 791 |
| MEDIUM | 30.2% ± 2.4% | 2656 |
| LOW | 16.2% ± 2.7% | 3140 |
| FILTER | 10.3% ± 1.4% | 7278 |

### Recommended Next Steps
1. **h408**: Ryland collaboration prep (approaching deadline)
2. **h542**: MEDIUM tier quality audit
3. **h543**: Corticosteroid saturation analysis

---

## Previous Session: h533 - FILTER Tier Precision Audit (2026-02-06)

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
