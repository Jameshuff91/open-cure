# Open-Cure Patterns & Architecture

Reference for key algorithmic patterns, filters, and signal integrations.

## Ceiling-Adjusted R@K (h1240, builds on h1211)

Always report **R@K / ceiling** alongside raw R@K in every embedding benchmark.

- **Per-disease ceiling**: `min(K, |GT|) / |GT|` — the maximum achievable R@K under perfect ranking.
- **Per-disease ceiling-adjusted Hits@K**: `hits@K / min(K, |GT|)` ∈ [0, 1] — "fraction of ceiling recovered".
- **Why**: h1211 showed the 3x per-category R@30 spread (endocrine 41% vs hematological 10%) is ~25% driven by the recall-denominator artifact: categories with large GT drug sets (psychiatric median 64, CV 37, renal 42) have per-disease ceilings of 50-70%, not 100%. Raw R@30 under-credits those retrievals.
- **Headline effect** (Node2Vec 256, h1199 splits):
  - Raw R@30 = 19.55%, ceiling-adjusted R@30 = 24.53%.
  - Raw R@1 = 2.88%, ceiling-adjusted R@1 = 35.13%. The top-1 kNN neighbour actually recovers ~35% of the achievable top-1 bucket.
- **Per-category re-ordering**: psychiatric 10.1% → 33.0% of ceiling, CV 12.9% → 26.0%, autoimmune 26.7% → 38.9%.
- **Implementation**: `scripts/clean_embedding_benchmark.py` now emits `hits_at_k_drug_ceiling` and `per_category_aggregate[*].r30_over_ceiling_mean` on every run.
- **Gate for h1200 / h1202 promotion**: must beat Node2Vec on **both** raw R@30 **and** R@30/ceiling — catches "trick" improvements that shift the GT-size distribution rather than retrieval quality.

## TransE Consilience (h405/h439/h440)

TransE agreement is a strong, holdout-validated signal:
- MEDIUM + TransE top-30: 34.7% ± 4.2% holdout (+13.6pp over MEDIUM avg)
- Works across ALL tiers: GOLDEN +11.4pp, HIGH +6.1pp, LOW +6.5pp, FILTER +7.2pp
- **NOT a tier promotion** (37.4% full-data < HIGH 50.8%)
- Implemented as `transe_consilience` boolean flag on DrugPrediction
- `_load_transe_model()` + `_get_transe_top_n()` in production_predictor.py
- TransE top-30 optimal (38.9% precision) vs top-100 (38.2% but 2x coverage)

**Key learning (h434):** LOO frequency provides negligible improvement (0-0.5pp). The rank>20 filter compensates for kNN NEIGHBORHOOD INSTABILITY (5-10pp), not frequency inflation. Mean 4.1 drugs cross rank-20 boundary per disease.

## Mechanism & ATC Integration (h96, h259, h152, h189)

- **Mechanism = PRECISION signal** (2.62x lift), NOT recall signal
- **CV/Neuro:** REQUIRE mechanism (>10x lift, 236 excluded, 2 GT lost)
- **ATC rescue:** L04AX (82%), H02AB (77%); EXCLUDE biologics L04AB/L04AC (<17%)
- Details: `docs/archive/experiment_history.md`

## Disease Hierarchy Matching (h273/h276/h278)

Subtype refinements (psoriasis → plaque psoriasis):
- Metabolic/Neuro 63-65% → GOLDEN
- Autoimmune/Resp/CV/Inf 22-45% → HIGH
- Implementation: `DISEASE_HIERARCHY_GROUPS` + `_check_disease_hierarchy_match()`

## Key Filters (all validated 2026-02-05)

- **Domain-Isolated (h271):** 828 drugs treat ONE category. Cross-domain = 0% precision. `_is_cross_domain_isolated()`
- **Broad Class Isolation (h307/h326/h328):** IL/TNF/anesthetics/steroids alone = 0-3%. `_is_broad_class_isolated()` demotes to LOW
- **Cancer-Only (h346):** 69 drugs (BRAF,PD-1,BCL2,PARP,etc.) = 0% non-cancer. `CANCER_ONLY_DRUGS` → FILTER

## CV Pathway-Comprehensive Boost (h351/h354/h356)

Drugs with GT for BOTH CV base (hypertension/lipids) AND CV complications perform much better:
- **Pathway-comprehensive: 28.9%** vs Non-pathway: **1.1%** (+27.8 pp, 26x lift!)
- 129 CV pathway-comprehensive drugs identified (statins, ACEi, ARBs, anticoagulants, etc.)
- **Why CV is special:** Shared vascular pathology — statins treat atherosclerosis → also treat MI/stroke/HF
- Implementation: `CV_PATHWAY_COMPREHENSIVE_DRUGS` + `_is_cv_pathway_comprehensive()` → HIGH tier

## Complication Drug Class Filter (h353)

Complication diseases (nephropathy/retinopathy/cardiomyopathy): non-validated drug classes = 0%. `COMPLICATION_VALIDATED_DRUGS` → FILTER

## Key Finding: Organ Proximity Doesn't Transfer (h294)

Within-organ novel predictions have **1.2% precision**. Only **CV pathway-comprehensive** transfer works.

## Performance Gaps & Error Patterns

**Gaps:** Biologics (mAbs 17% vs small mol 32%), Antibiotics (wrong diseases), GI (5% kNN blind spot)
**Best:** ACE inhibitors 67%, Autoimmune 63%, Infectious 52% | **Worst:** mAbs 27%, Antibiotics 6-20%, PPIs 17%

## Confidence Filter (`src/confidence_filter.py`)

Excludes harmful patterns:
- Withdrawn drugs (Pergolide, Cisapride, etc.)
- Antibiotics for metabolic diseases
- Sympathomimetics for diabetes
- TCAs/PPIs for hypertension
- Alpha blockers for heart failure
- Non-DHP CCBs (Verapamil/Diltiazem) + HF (ACC/AHA 2022)
- Class Ic/Ia antiarrhythmics + structural heart (CAST/SWORD trials)
- Dronedarone + HF (ANDROMEDA trial: 2.13x mortality)
- **Inverse indications** (drug CAUSES condition): 67 drugs, 157 pairs
  - Corticosteroids → TB, glaucoma, osteoporosis, MG, pancreatitis
  - NSAIDs → TEN, SLE, peptic ulcer, stroke (COX-2)
  - Estradiol → endometrial/uterine cancer, hereditary angioedema
  - Proarrhythmic drugs → ventricular tachycardia
  - Azathioprine → TEN, hepatitis B, erythema multiforme
  - h486: 47 new pairs from SIDER mining (93.3% filter precision)
  - h408+h544: Anti-TNF → SLE/MG/MS/AIH/sarcoidosis/vasculitis/polymyositis/lichen planus
  - Ganglionic blockers (obsolete), surgical dyes (not therapeutic)

**Total inverse indication filters:** ~141 predictions (67 drugs, 157 pairs)

## Expert-Label Ingestion (h907 — Ryland blinded review)

**Flow:**
1. Ryland returns review (CSV/XLSX/JSON) with one row per prediction and a `verdict` in {plausible, known, implausible, adverse, unsure}.
2. `python scripts/import_ryland_review.py <file>` validates against `data/reference/ryland_review_schema.json`, resolves drug/disease names to DRKG IDs via the predictor's alias map, and writes `data/reference/expert_labels_ryland.json` keyed by `prediction_id = '<disease_id>||<drug_id>'`.
3. `src/expert_labels.py` loads those records into an `ExpertLabels` helper for evaluation-side lookup.
4. `python scripts/h907_eval_expert_labels.py` produces a parallel precision split per tier: `drkg_precision` (against `expanded_ground_truth.json`) vs `expert_precision` (against Ryland's verdicts).

**Leakage-safe rules (DO NOT BREAK):**
- Expert labels carry `provenance = 'expert_ryland'` and are NEVER merged into `predictor.ground_truth`, `expanded_ground_truth.json`, `drug_train_freq`, `drug_to_diseases`, or anything that feeds kNN/embedding training. Merging would contaminate the very predictions Ryland is judging.
- Predictions Ryland did not review are excluded from the expert-precision denominator (not counted as misses).
- Low-confidence verdicts (`reviewer_confidence < 3`) and `unsure` verdicts are excluded from expert precision by default; override with `--min-reviewer-confidence`.
- `adverse` and `implausible` verdicts count as expert misses; they are also candidate inputs for the inverse-indication and safety-filter rules but MUST be reviewed per-prediction before codifying — Ryland's sample is not a systematic adverse-event survey.

**When the review has not arrived yet:** `expert_labels_ryland.json` is absent, `load_expert_labels` returns an empty helper, and `h907_eval_expert_labels.py` leaves the `expert_precision` column null. The DRKG-GT column stays live so tier metrics keep flowing.

## Key Validated Predictions

| Drug | Disease | Evidence |
|------|---------|----------|
| **Dantrolene** | Heart Failure/VT | RCT P=0.034, 66% reduction |
| **Lovastatin** | Multiple Myeloma | RCT: improved OS/PFS |
| **Rituximab** | MS | WHO Essential Medicine 2023 |
| **Pitavastatin** | RA | Superior to MTX alone |
| **Empagliflozin** | Parkinson's | HR 0.80 in Korean study |
