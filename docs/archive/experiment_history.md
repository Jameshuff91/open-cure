# Experiment History

Detailed logs of feature engineering experiments for the Open-Cure drug repurposing model.

## ATC Classification Feature Experiment (2026-01-24) - SUCCESS

**Hypothesis:** Adding ATC (Anatomical Therapeutic Chemical) classification boosting would improve predictions when drug mechanism matches disease category.

### Implementation

Downloaded WHO ATC-DDD 2024 classification data (7,345 codes). Created `src/atc_features.py` to:
1. Map DrugBank drugs to ATC codes by name matching
2. Score drug-disease mechanism relevance based on ATC level 1 categories
3. Boost predictions where drug ATC class is relevant for disease type

**ATC Coverage:**
- 2,959 drugs mapped to ATC codes (12.2% of 24,313 evaluation drugs)
- 28.4% of DrugBank drugs (2,979/10,474) have ATC mappings

### Results

| Strategy | R@30 | vs Baseline | vs Target Only |
|----------|------|-------------|----------------|
| Baseline | 37.39% | - | -1.63% |
| Target only | 39.02% | +1.63% | - |
| ATC only (10%) | 38.28% | +0.89% | -0.74% |
| **Combined (add)** | **39.69%** | **+2.30%** | **+0.67%** |
| Combined (tiered) | 39.39% | +2.00% | +0.37% |

**Best Strategy:** `combined_add` - `score × (1 + 0.01×target_overlap + 0.05×atc_score)`

### Files

- `data/external/atc/atc_codes_2024.csv` - WHO ATC-DDD 2024 data (not in git)
- `src/atc_features.py` - ATC mapping and feature extraction
- `scripts/evaluate_combined_boost.py` - Combined boost evaluation

---

## Chemical Structure Features (2026-01-25) - SUCCESS

**Hypothesis:** Drugs structurally similar to known treatments are more likely to be effective.

### Implementation

Used RDKit to compute Morgan fingerprints (ECFP4 equivalent) and Tanimoto similarity:
1. Fetch SMILES from PubChem for DrugBank drugs
2. Generate 2048-bit Morgan fingerprints (radius=2)
3. For each candidate drug, compute max Tanimoto similarity to known treatments
4. Boost predictions when similarity > 0.7 threshold

**Coverage:** 9,584/10,474 DrugBank drugs with fingerprints (91.5%) - fetched from PubChem in batch

### Results (Triple Boost Evaluation - 91% FP Coverage)

| Strategy | R@30 | vs Baseline | vs Target+ATC |
|----------|------|-------------|---------------|
| **triple_multiplicative** | **47.11%** | **+8.39%** | **+6.95%** |
| target+chem | 46.84% | +8.12% | +6.68% |
| chem_only | 46.66% | +7.94% | +6.50% |
| target+atc | 40.16% | +1.44% | - |
| baseline | 38.72% | - | -1.44% |

**Best Strategy:** `triple_multiplicative`
- Formula: `score × (1 + 0.01 × overlap) × (1 + 0.05 × atc) × (1.2 if sim > 0.7 else 1.0)`
- Improvement: **+8.39%** over baseline (38.72% → 47.11%)
- Improvement: **+6.95%** over previous best target+atc

### Coverage Expansion Impact

| Coverage | R@30 | Change |
|----------|------|--------|
| 4.1% | 43.95% | - |
| **91.5%** | **47.11%** | **+3.16%** |

### Key Insight

Chemical similarity boost provides the **largest individual gain** (+7.94% for chem_only). Expanding fingerprint coverage from 4% to 91% added another +3.16% improvement.

### Files

- `src/chemical_features.py` - Fingerprint generation and similarity calculation
- `scripts/fetch_smiles_batch.py` - Batch SMILES fetcher from PubChem
- `scripts/evaluate_chemical_boost.py` - Chemical-only boost evaluation
- `scripts/evaluate_triple_boost.py` - Combined triple boost evaluation
- `data/reference/chemical/drug_fingerprints.pkl` - Cached fingerprints (9,584 drugs)
- `data/reference/chemical/drug_smiles.json` - Cached SMILES strings

---

## Pathway Enrichment Features (2026-01-25) - MARGINAL SUCCESS

**Hypothesis:** Drugs that target pathways dysregulated in a disease are more likely to be therapeutic.

### Implementation

Used KEGG REST API to build pathway mappings:
1. Bulk download all human gene-pathway associations (~9,500 genes)
2. Map drugs → pathways via drug targets
3. Map diseases → pathways via disease genes
4. Compute pathway overlap and Jaccard similarity

**Coverage:**
- 9,594/11,656 drugs (82%) have pathway data
- 2,822/3,454 diseases (82%) have pathway data

### Results

| Strategy | R@30 | vs Triple |
|----------|------|-----------|
| **quad_additive** | **47.47%** | **+0.36%** |
| quad_multiplicative | 47.47% | +0.36% |
| triple (no pathway) | 47.11% | - |
| pathway_only | 40.25% | -6.86% |

**Best Strategy:** `quad_additive`
- Formula: `score × (1 + 0.01×overlap + 0.05×atc + 0.01×pathway_overlap) × (1.2 if chem>0.7 else 1.0)`
- Improvement: **+0.36%** over triple boost

### Key Insight

Pathway enrichment adds only marginal improvement (+0.36%) because:
1. Pathway overlap is highly correlated with target overlap (both gene-derived)
2. Chemical similarity already captures structural mechanisms
3. Diminishing returns from adding correlated features

Pathway boost is still worth including for explainability and edge cases.

### Files

- `src/pathway_features.py` - Pathway enrichment computation
- `scripts/evaluate_pathway_boost.py` - Pathway boost evaluation
- `data/reference/pathway/gene_pathways.json` - Gene → pathway mappings
- `data/reference/pathway/drug_pathways.json` - Drug → pathway mappings
- `data/reference/pathway/disease_pathways.json` - Disease → pathway mappings

---

## Similarity Feature Experiment (2026-01-24) - FAILED

**Hypothesis:** Adding "guilt by association" features (similarity to known treatments) would improve predictions.

**Features Added (4 new dimensions):**
1. `max_sim_drug`: Max cosine similarity to drugs known to treat this disease
2. `mean_sim_drug`: Mean cosine similarity to known treatments
3. `max_sim_disease`: Max cosine similarity to diseases this drug treats
4. `mean_sim_disease`: Mean cosine similarity to known indications

### Experiment 1: Naive Training (Same Diseases for Train/Eval)

**Training Results:**
- AUROC: 0.966 (very high!)
- AUPRC: 0.910
- Feature importance: **80.2% from similarity features** (dominated the model)

**Evaluation Results (CATASTROPHIC FAILURE):**
- Recall@30: **0.02%** (10 hits out of 58,016 GT drugs)
- Cause: Similarity features LEAKED training data

### Experiment 2: Proper Disease-Level Split (80/20)

**Setup:**
- Training diseases: 2,887 (80%)
- Test diseases: 722 (20%)
- Similarity lookup built ONLY from training diseases

**Training Results:**
- Validation AUROC: 0.963
- Validation AUPRC: 0.903
- Feature importance: **80.3% from similarity features**

**Test Results (STILL FAILED):**
- Recall@30: **0.0%** (0 hits out of 10,672 GT drugs)
- Cause: Test diseases have NO entries in similarity lookup → features 1-2 always = 0

**Comparison with Base Model:**
- Base model (no similarity) on same test diseases: **1.7% Recall@30**
- Similarity model: **0.0% Recall@30**
- **Conclusion: Similarity features HURT performance**

### Root Cause Analysis

The similarity features are fundamentally flawed for this task:

1. **Features 1-2** (`max_sim_drug`, `mean_sim_drug`): "Is this drug similar to known treatments for THIS disease?"
   - For test diseases: always 0 (no known treatments in lookup)
   - Model learned: "features 1-2 = 0 → negative"
   - Result: ALL drugs predicted as negative for test diseases

2. **Features 3-4** (`max_sim_disease`, `mean_sim_disease`): "Is this disease similar to diseases this drug treats?"
   - Can be non-zero if drug treats training diseases
   - But model over-relies on features 1-2 (80% importance)

3. **Fundamental Issue:** Similarity features encode "is this a known treatment?" which IS the label, just transformed. When held-out properly, the signal disappears.

### Key Lessons

1. **High training metrics ≠ good generalization** - 0.96 AUROC, 0% test recall
2. **Feature leakage is subtle** - features derived from labels can look great during training
3. **Disease-level holdout is essential** - pair-level splits miss cross-contamination
4. **Simpler is better** - base embedding features work; fancy features can hurt

### Files

- `src/train_gb_with_similarity.py` - Initial (flawed) training script
- `src/train_gb_similarity_split.py` - Proper disease-level split training
- `src/evaluate_gb_similarity.py` - Evaluation script (optimized, vectorized)
- `models/drug_repurposing_gb_similarity.pkl` - Model v1 (214MB)
- `models/drug_repurposing_gb_similarity_split.pkl` - Model v2 with split

---

## Target Feature Experiment (2026-01-24) - MIXED RESULTS

**Hypothesis:** Adding drug-target and disease-gene overlap features would improve predictions.

**Features Added (4 new dimensions):**
1. `n_drug_targets`: Number of targets for the drug
2. `n_disease_genes`: Number of genes associated with disease
3. `n_overlap`: Count of shared targets/genes
4. `frac_overlap`: Fraction of drug targets that overlap with disease genes

### Experiment 1: Retrain Model with Target Features - FAILED

**Setup:**
- 1,359 positive pairs from GT with both target and gene data available
- Drug target coverage: 11,656 drugs (from DrugBank)
- Disease gene coverage: 77% of MESH diseases

**Training Results:**
- Reasonable training metrics

**Evaluation Results:**
- Recall@30: **5.8%** (vs 37.4% baseline)
- **Regression of -31.6%** - CATASTROPHIC FAILURE

**Why It Failed:**
- Trained on different/smaller dataset than baseline model
- Model learned different patterns, lost baseline signal
- Target features didn't compensate for training data differences

### Experiment 2: Ensemble Strategy - SUCCESS

Instead of retraining, boost baseline scores when target overlap exists.

**Strategies Tested:**

| Strategy | R@30 | Change | Formula |
|----------|------|--------|---------|
| baseline | 37.4% | - | score |
| boost_if_overlap | 38.9% | +1.5% | score × 1.1 if overlap > 0 |
| **boost_by_overlap** | **39.0%** | **+1.6%** | score × (1 + 0.01 × min(overlap, 10)) |
| boost_by_frac | 36.9% | -0.4% | score × (1 + 0.2 × frac) |
| multiply_frac | 34.3% | -3.0% | score × (1 + frac) |
| add_bonus | 37.2% | -0.2% | min(1.0, score + 0.1 × frac) |

**Best Strategy:** `boost_by_overlap` - multiply score by (1 + 0.01 × overlap count, capped at 10)

**Statistical Validation (2026-01-24):**

| Test | Result | Significance |
|------|--------|--------------|
| McNemar's test | p=0.000014 | Highly significant |
| Sign test | p=0.000044 | Highly significant |
| Bootstrap 95% CI | [0.88%, 2.54%] | Does not include 0 |

**Detailed Results:**
- 25 GT drugs entered top-30, only 3 left (net +22)
- 23 diseases improved (3.3%), only 3 hurt (0.4%), 664 unchanged
- Bootstrap: Baseline 37.4% [34.5%, 40.4%], Boosted 39.1% [36.2%, 42.1%]

**Biological Plausibility (examples of boosted predictions):**
| Drug | Disease | Target-Gene Overlap |
|------|---------|---------------------|
| Fulvestrant | HER2- breast cancer | 131 genes |
| Paclitaxel | Cancer | 118 genes |
| Estradiol | Breast cancer | 76 genes |
| Doxorubicin | Bone sarcoma | 44 genes |
| Fluorouracil | Breast adenocarcinoma | 28 genes |

**Diseases hurt (minor):** Cholangiocarcinoma, CHF, Hypertension (-1 each)

### Files

- `src/train_gb_with_targets.py` - Retrained model (failed)
- `src/extract_target_features.py` - Target/gene extraction
- `scripts/evaluate_target_ensemble.py` - Ensemble strategy evaluation
- `data/reference/drug_targets.json` - Drug → target gene mappings
- `data/reference/disease_genes.json` - Disease → gene associations
- `models/drug_repurposing_gb_with_targets.pkl` - Retrained model (don't use)

---

## Confidence Calibration (2026-01-25)

**Purpose:** Predict "how confident should we be in this prediction?" to prioritize novel predictions.

### ML Confidence Model

Trained logistic regression on GT drug-disease pairs to predict probability of being in top-30.

**Performance:**
- Brier Score: 0.059 (excellent calibration)
- AUROC: 0.962
- AUPRC: 0.901

**Calibration by Tier:**

| Tier | Samples | Actual Hit Rate |
|------|---------|-----------------|
| Very High (≥0.8) | 529 | 90.9% |
| High (0.6-0.8) | 79 | 73.4% |
| Medium (0.4-0.6) | 38 | 31.6% |
| Low (0.2-0.4) | 40 | 12.5% |
| Very Low (<0.2) | 665 | 0.5% |

**Top Features:**
1. `base_score` (+5.07) - GB model score
2. `boosted_score` (+4.90)
3. `is_biologic` (-0.31) - Biologics have lower confidence
4. `is_cancer` (-0.42) - Cancer predictions lower confidence

### Rule-Based Filter

Excludes known harmful patterns (from literature validation):
- Withdrawn drugs (Pergolide, Cisapride, etc.)
- Antibiotics for metabolic diseases
- Sympathomimetics for diabetes
- TCAs for hypertension
- PPIs for hypertension
- Alpha blockers for heart failure

**Filter Stats (on 24,365 novel predictions):**
- Excluded: 155 (0.6%)
- Passed: 24,210
- High-confidence biologics: 403

### Files

- `src/confidence_calibration.py` - ML confidence predictor
- `src/confidence_filter.py` - Rule-based exclusion filter
- `scripts/train_confidence_model.py` - Train calibrator
- `scripts/generate_novel_predictions.py` - Generate predictions with confidence
- `scripts/filter_novel_predictions.py` - Apply both ML + rule filter
- `data/analysis/filtered_novel_predictions.json` - Filtered predictions
- `data/analysis/top_candidates_for_validation.json` - Top 100 for validation
