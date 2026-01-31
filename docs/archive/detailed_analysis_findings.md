# Detailed Analysis Findings (Archived 2026-01-31)

This document contains detailed analysis findings that were pruned from CLAUDE.md to keep working memory lean.

## Biologic Gap Analysis (2026-01-25)

**Root Cause:** Data sparsity - mAbs have 5x fewer training examples
- mAbs: 2.13 diseases/drug in DRKG
- Small molecules: 11.08 diseases/drug

**Performance by mAb Class:**
| Class | Strong% | Notes |
|-------|---------|-------|
| Anti-TNF | 100% | Immunology works |
| Anti-CD20 | 100% | Immunology works |
| Anti-integrin | 100% | Immunology works |
| Checkpoint | 42% | Mixed |
| Anti-HER2 | 0% | **Oncology fails** |
| Anti-EGFR | 17% | Oncology fails |

**Fix Applied:** Filter 16 weak oncology mAb predictions (precision improvement)
**Future:** Mechanism-based boosting for recall improvement

## Infectious Disease Gap Analysis (2026-01-25, Updated 2026-01-26)

**CORRECTION (2026-01-26):** The 13.6% figure was antibiotic CLASS performance, not disease-level R@30.

**Actual Performance:**
- General model: **52.0% R@30** on 47 infectious diseases (104/200 hits)
- Specialist model: 36.4% R@30 (underperforms due to data scarcity)

**Antibiotic Class Performance (within their GT indications):**

| Antibiotic Class | Avg Rank | Hit@30 |
|------------------|----------|--------|
| Antivirals | 1,341 | 20% |
| Tetracyclines | 1,445 | 18% |
| Fluoroquinolones | 2,385 | **0%** |
| Macrolides | 4,324 | 6% |

**The Real Problem:** Model predicts antibiotics for NON-infectious diseases
- Levofloxacin → diabetes, arthritis
- Telithromycin → heart failure
- Azithromycin → stroke

**Fix Applied:** Filter 20 spurious antibiotic predictions for non-infectious diseases

## Validation False Positives (2026-01-25)

**CRITICAL:** High validation scores can be misleading. Deep dives required.

| Prediction | Validation Score | Status | Reason |
|------------|------------------|--------|--------|
| Digoxin → T2D | 0.88 | **FALSE POSITIVE** | Comorbidity confounding |
| Simvastatin → T2D | 0.96 | **FALSE POSITIVE** | Inverse indication (statins cause T2D) |

**Digoxin → T2D Deep Dive:**
- 8 trials were DDI studies, not treatment trials
- Spigset 1999: Digoxin WORSENS glucose (HbA1c 5-6% → 7-8%)
- Mechanism: Na+/K+-ATPase inhibition reduces glucose transport
- DIG trial (n=6,800): No difference in diabetes outcomes
- Related: Digoxin shows promise for NAFLD/NASH (preclinical only)

**Simvastatin → T2D Deep Dive:**
- Statins INCREASE T2D risk (Lancet 2024: HR 1.12-1.44)
- 33 trials are for CV protection IN diabetics, not treating T2D
- ADA recommends statins for diabetics despite metabolic risk

**Confounding Patterns to Watch:**
1. **Cardiac-Metabolic Comorbidity** - HF drugs appear connected to T2D (digoxin, furosemide, etc.)
2. **Polypharmacy Interactions** - Phase 1 PK studies miscounted as treatment trials
3. **Inverse Indication** - Drug CAUSES disease but prescribed for other benefits (statins → T2D)

## Confounding Detection (2026-01-25)

**Script:** `src/confounding_detector.py`
**Output:** `data/analysis/confounding_analysis.json`

Scans 568 validated predictions for confounding patterns. Found 9 suspicious predictions (1.6%).

**High-Confidence False Positives (7):**

| Drug | Disease | Type | Reason |
|------|---------|------|--------|
| Simvastatin | T2D | Inverse indication | Statins INCREASE T2D risk (HR 1.12-1.44) |
| Hydrochlorothiazide | T2D | Inverse indication | Thiazides cause hyperglycemia |
| Quetiapine | T2D | Inverse indication | Antipsychotics cause metabolic syndrome |
| Digoxin | T2D | Mechanism mismatch | Na+/K+-ATPase inhibition worsens glucose |
| Digitoxin | T2D | Mechanism mismatch | Same as digoxin |
| Pembrolizumab | UC | Mechanism mismatch | Checkpoint inhibitors CAUSE colitis (irAE) |
| Quetiapine | Parkinson's | Mechanism mismatch | Antipsychotics cause drug-induced parkinsonism |

**True Positives (drugs that actually help T2D):**
- ACE inhibitors (ramipril, etc.) - HOPE trial: 34% reduction in new T2D
- Verapamil - RCT: HbA1c reduction, beta-cell preservation

**Medium Confidence - Need Review:**
- Felodipine → T2D (cardiac-metabolic comorbidity)
- Amiloride → T2D (cardiac-metabolic comorbidity)

## External Validation Pipeline (2026-01-25)

**Script:** `src/external_validation.py`
**Output:** `data/validation/`

Validates predictions against ClinicalTrials.gov and PubMed. Results on top 100 predictions:

| Evidence Level | Count | Interpretation |
|----------------|-------|----------------|
| Strong (≥0.5) | 57% | Model learns real drug-disease relationships |
| Moderate (0.2-0.5) | 10% | **Best repurposing candidates** |
| Weak (<0.2) | 23% | Needs investigation |
| None | 10% | Truly novel or spurious |

**Key Findings:**
- 61% have active clinical trials
- 89% have PubMed publications
- Model correctly predicts approved indications (Cetuximab/CRC, etc.)

**Top Repurposing Candidates (moderate evidence, not yet approved):**
- Sirolimus → Psoriasis (mTOR inhibitor for autoimmune)
- Clopidogrel → RA (platelet-inflammation link)
- Metformin → Breast Cancer (known epidemiological signal)

## Disease Coverage Expansion (2026-01-25)

**Integrated `mondo_to_mesh.json`** mapping into disease matcher.

| Metric | Before | After |
|--------|--------|-------|
| Total mappings | 836 | 9,090 |
| EC diseases mapped | 17.2% | **44.7%** |
| EC pairs mapped | - | **63.4%** |

**Key insight:** Every Cure uses MONDO IDs, embeddings use MESH IDs.
MONDO→MESH mapping bridges this gap.

**Bug Found & Fixed:** Short synonyms ("ra", "as", "mm") caused false substring matches:
- "ab**ra**sions" → rheumatoid arthritis (wrong!)
- "metast**as**is" → ankylosing spondylitis (wrong!)

**Fix:** `src/disease_name_matcher.py`
- Only check synonyms ≥6 chars for substring matching
- Require long disease names (>20 chars) for fuzzy substring matching
- Normalizes whitespace, punctuation, possessives

## Disease Name Matching (2026-01-25)

**FIXED:** Fuzzy matching now improves R@30 from 37.4% → 41.8%

| Version | Diseases Mapped | Pairs | R@30 |
|---------|-----------------|-------|------|
| Exact match only | 397 (9.2%) | 2,212 | 37.6% |
| Fuzzy (fixed) | 1,236 (30.9%) | 3,618 | **41.8%** |
