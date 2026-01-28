# Open-Cure Project Instructions

## Memory Management

**After each research session:** Update this file with key learnings before committing.

**Periodically ask:** "Should we prune working memory and move details to long-term storage?"
- Archive location: `docs/archive/`
- Keep CLAUDE.md lean (<400 lines) for efficient context loading

## Session End Protocol

**ALWAYS end sessions by recommending the highest-ROI next steps:**
1. Analyze current model performance gaps
2. Identify improvement opportunities achievable with existing data
3. Rank by expected impact vs effort
4. Present top 2-3 actionable recommendations

**Constraints:** Prioritize approaches that don't require additional external data or GPU resources unless absolutely necessary.

## Scientific Reasoning Protocol (MANDATORY)

You are an execution engine, not a scientist. You lack epistemic discipline by default. Follow these rules to compensate.

### 1. Distrust Your Own Outputs
- **Never treat a computed metric as true without validation.** If you compute R@30 = X%, ask: "Does this make sense given the baseline? What could make this number wrong?"
- If a result looks surprisingly good, it is more likely a bug than a breakthrough. Investigate before reporting.
- Distinguish between "I measured X" and "X is true." Measurement errors, data leakage, and confounding are the default assumption until ruled out.

### 2. Check Preconditions Before Running Experiments
- Before committing to a hypothesis, spend 5-10 minutes checking whether the basic premise holds. Examples:
  - h4 (GT expansion): Could have checked set overlap between DRKG and Every Cure GT in 2 minutes before building a whole pipeline.
  - h3 (specialist model): Could have checked training data size for infectious diseases before building a specialist.
- **Ask: "What would need to be true for this hypothesis to work? Can I verify that cheaply first?"**

### 3. Run Positive Controls
- Before evaluating a new approach, verify that known-good drug-disease pairs (e.g., Metformin→T2D, Rituximab→MS) score highly. If your positive controls fail, your experiment is broken.
- Compare new results against the established baseline (currently 41.8% R@30) and explain any discrepancy.

### 4. Validate Against Published Evidence
- For any novel prediction or surprising result, search ClinicalTrials.gov and PubMed for corroborating or contradicting evidence BEFORE reporting the result as valid.
- A high model score means nothing if the drug is known to CAUSE the disease (e.g., statins → T2D, antipsychotics → parkinsonism).

### 5. Stop Early When Evidence Contradicts
- If initial data (first 10% of an experiment) contradicts the hypothesis, STOP. Do not complete the full experiment hoping it will turn around. Report the early negative signal and move on.
- A failed hypothesis identified in 5 minutes is more valuable than one identified after an hour of compute.

### 6. Question Your Methodology
- Before reporting results, ask:
  - "Am I evaluating on training data?" (data leakage)
  - "Are my features derived from the labels?" (circularity)
  - "Could this correlation be confounded by comorbidity, polypharmacy, or indication overlap?"
  - "Would a domain expert find this result plausible?"

### 7. Benchmark Against Known Drugs
- When evaluating predictions for a disease, check: does the model rank FDA-approved treatments highly? If Metformin doesn't appear in the top 30 for T2D, something is wrong with the evaluation, not insightful about Metformin.
- Use known drug-disease pairs as sanity checks, not just aggregate metrics.

### 8. Report Uncertainty and Limitations
- Never write "improvement achieved" without quantifying confidence. Include effect size, sample size, and whether the improvement exceeds noise.
- If an experiment is inconclusive, say so plainly. Do not dress up a null result as "promising direction for future work."

## Cloud GPU (Vast.ai)

```bash
source .venv/bin/activate
vastai search offers 'gpu_ram>=8 cuda_vers>=11.0 reliability>0.95' --order 'dph' --limit 10
vastai create instance <OFFER_ID> --image nvidia/cuda:11.7.1-runtime-ubuntu22.04 --disk 30 --ssh
vastai show instances
ssh -p <PORT> root@<SSH_ADDR>
vastai destroy instance <INSTANCE_ID>  # Stop billing!

# Automated setup
./scripts/vastai_txgnn_setup.sh <PORT> <HOST>
```

**Current instance**: None

## Models

**Default Model (use this):**
- `models/drug_repurposing_gb_enhanced.pkl` + Quad Boost ensemble
- Formula: `score × (1 + 0.01×overlap + 0.05×atc + 0.01×pathway) × (1.2 if chem_sim > 0.7 else 1.0)`
- Script: `scripts/evaluate_pathway_boost.py`
- Requires: `data/reference/drug_targets.json`, `data/reference/disease_genes.json`, `data/reference/chemical/`, `data/reference/pathway/`

**Other Models:**
- `models/drug_repurposing_gb.pkl` - Original baseline (7.0% R@30)
- `models/transe.pt` - TransE embeddings
- `models/confidence_calibrator.pkl` - Predicts top-30 probability

## Key Metrics

| Model | Per-Drug R@30 | Evaluation | Notes |
|-------|---------------|------------|-------|
| **Node2Vec+XGBoost cpd (disease holdout)** | **29.45%** | **HONEST** | Best generalizing model, 88 test diseases |
| Node2Vec+XGBoost concat (disease holdout) | 26.18% | Honest | Concat-only features |
| GB + Fuzzy Matcher (fixed) | 41.8% | Within-dist | 1,236 diseases, pair-level (inflated) |
| GB + TransE (existing, on test) | 45.9% | Pair-trained | Trained on ALL diseases, tested on subset |
| TransE+XGBoost (disease holdout) | 15.90% | Honest | TransE fails to generalize |
| Node2Vec+XGBoost (pair-level) | ~21.6% | Within-dist | Pair-level trained, evaluated on test |
| GB + Quad Boost (inflated) | 47.5%* | Circular | *Circular features - NOT real |
| Node2Vec Cosine (no ML) | 1.27% | Honest | ML model IS required |
| TxGNN | 6.7% | Unknown | Near-random for most diseases |

**CRITICAL (2026-01-27):** The honest generalization baseline is **29.45% R@30** (Node2Vec+XGBoost cpd on disease-level holdout). All higher numbers used pair-level splits or circular features.

**Progression:** 37.4% → 41.8% (fuzzy fix, pair-level) → Generalization crisis → **29.45% (honest, Node2Vec)**

## Key Learnings

### What Works
1. **Fuzzy Disease Matching** - 41.8% R@30 (up from 37.4% exact-only)
2. **Disease holdout splits** - Required for honest novel discovery evaluation
3. **DRKG graph embeddings** - 256-dim Node2Vec captures treatment relationships

### CRITICAL: Generalization Gap (2026-01-27)
- **GB + TransE does NOT generalize**: Retrained 3-12% R@30 on disease holdout (h5)
- **Node2Vec DOES partially generalize**: 29.45% R@30 on disease holdout (h29) — 1.85x better than TransE (15.90%)
- The "41.9% on held-out diseases" was INCORRECT: original code used pair-level split
- Node2Vec's random walk captures transferable neighborhood structure; TransE's translational model memorizes
- Concat+product+diff features help Node2Vec: 26.18% (concat) → 29.45% (cpd) (+3.3 pp)
- Cosine similarity alone is useless: 0-1.27% R@30
- All 4 positive controls pass for Node2Vec concat model (Metformin rank 22, Rituximab rank 21, Imatinib rank 12, Lisinopril rank 27)
- **Next priority**: Improve beyond 29.45% via graph features + Node2Vec hybrid (h34), or gene-based features (h35)

### What SEEMED to Work (but was data leakage)
1. **Boost features** - Target overlap, chemical similarity, ATC were circular
2. **Evaluating on training diseases** - Inflated recall from 41.9% to 47.5%
3. **All negative sampling strategies** - Hard negatives, random, drug-treats-other ALL fail under disease holdout

### What Fails
1. **Embedding Similarity** - TransE cosine similarity causes data leakage
2. **Retraining with Features** - Adding features and retraining: 37%→6%
3. **Correlated Features** - Pathway adds only +0.36% (correlates with target)
4. **Biologics** - mAbs achieve only 16.7% recall vs 32.1% small molecules
5. **Circular Boost Features** - Target overlap, chemical similarity, ATC codes are circular
6. **Biologic Naming Penalty** - WHO INN naming convention unreliable for filtering
7. **Specialist Models** - Infectious disease specialist (36%) underperforms general model (52%)
8. **GB Disease Generalization** - Model cannot generalize to unseen diseases (45.9% → 3-12% R@30)
9. **Hard Negative Mining** - All 5 strategies fail under disease holdout (not a sampling issue, architectural)

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

## Error Patterns

| Best Performance | Worst Performance |
|------------------|-------------------|
| ACE inhibitors: 66.7% | Monoclonal antibodies: 27.3% |
| Autoimmune: 63.0% | Antibiotics (class perf): 6-20% |
| Infectious: 52.0% | PPIs: 16.7% |

## Confidence Filter

Use `src/confidence_filter.py` to exclude harmful patterns:
- Withdrawn drugs (Pergolide, Cisapride, etc.)
- Antibiotics for metabolic diseases
- Sympathomimetics for diabetes
- TCAs/PPIs for hypertension
- Alpha blockers for heart failure

**Validation precision:** 20-25% for top predictions (batches 1+2)

## Key Validated Predictions

| Drug | Disease | Evidence |
|------|---------|----------|
| **Dantrolene** | Heart Failure/VT | RCT P=0.034, 66% reduction |
| **Lovastatin** | Multiple Myeloma | RCT: improved OS/PFS |
| **Rituximab** | MS | WHO Essential Medicine 2023 |
| **Pitavastatin** | RA | Superior to MTX alone |
| **Empagliflozin** | Parkinson's | HR 0.80 in Korean study |

## Data Sources

- Every Cure GT: `data/reference/everycure/indicationList.xlsx`
- Enhanced GT: `data/reference/expanded_ground_truth.json`
- DrugBank: `data/reference/drugbank_lookup.json`
- Disease mapping: `data/reference/disease_ontology_mapping.json`

## Critical Finding: Disease Name Matching (2026-01-25)

**FIXED:** Fuzzy matching now improves R@30 from 37.4% → 41.8%

| Version | Diseases Mapped | Pairs | R@30 |
|---------|-----------------|-------|------|
| Exact match only | 397 (9.2%) | 2,212 | 37.6% |
| Fuzzy (fixed) | 1,236 (30.9%) | 3,618 | **41.8%** |

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

## TxGNN Summary

- 14.5% R@30 (comparable to early GB model)
- Excels at storage diseases (83.3% R@30)
- Fine-tuning causes catastrophic forgetting
- Best use: ensemble with min(TxGNN_rank, GB_rank)
- **Details:** `docs/archive/txgnn_learnings.md`

## Validation Pipeline (2026-01-25)

**Script:** `src/external_validation.py` (with confounding integration)
**Extended validation:** `scripts/run_extended_validation.py`

The validation pipeline now:
1. Queries ClinicalTrials.gov and PubMed for evidence
2. Detects confounding patterns automatically
3. Computes adjusted_score that penalizes confounded predictions
4. Caches results for efficiency

**Example:** Simvastatin→T2D
- validation_score: 0.96 (high trial/pub count)
- adjusted_score: 0.10 (90% penalty for inverse indication)

## Archive Index

| Archive | Content |
|---------|---------|
| `docs/archive/experiment_history.md` | ATC, Chemical, Pathway, Similarity, Target experiments |
| `docs/archive/validation_sessions.md` | Literature validation batches 1+2, novel predictions |
| `docs/archive/txgnn_learnings.md` | TxGNN training, evaluation, fine-tuning experiments |
