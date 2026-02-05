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

**Current instance**: None | Balance: $4.41

### Quick Commands
```bash
# Search for GPU instances (RTX 3090/4090)
vastai search offers 'gpu_name in [RTX_3090, RTX_4090] disk_space >= 50 reliability > 0.95' -o 'dph_total' --limit 10

# Create instance from offer ID
vastai create instance <OFFER_ID> --image pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel --disk 50

# Get SSH connection details
vastai show instances
vastai ssh-url <INSTANCE_ID>

# Setup TxGNN (after getting PORT and HOST)
./scripts/vastai_txgnn_setup.sh <PORT> <HOST>

# IMPORTANT: Destroy when done
vastai destroy instance <INSTANCE_ID>
```

**Skill**: Use `/vastai-gpu` for detailed GPU provisioning instructions

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
| **kNN k=20 (original embeddings)** | **36.59% ± 3.90%** | Honest (5-seed) | Has treatment edge leakage |
| **kNN k=20 (no-treatment embeddings)** | **26.06% ± 3.84%** | **FAIR (5-seed)** | **Best fair transductive comparison** |
| **KEGG Pathway kNN** | **15.73% ± 1.82%** | **INDUCTIVE (5-seed)** | **Fair inductive comparison to TxGNN** |
| Node2Vec+XGBoost TUNED (disease holdout) | 25.85% ± 4.06% | Honest (5-seed) | md=6,ne=500,lr=0.1,alpha=1.0 (h38/h40) |
| Node2Vec+XGBoost default (disease holdout) | 23.73% ± 3.73% | Honest (5-seed) | md=6,ne=100,lr=0.1 (h40) |
| GB + Fuzzy Matcher (fixed) | 41.8% | Within-dist | 1,236 diseases, pair-level (inflated) |
| GB + TransE (existing, on test) | 45.9% | Pair-trained | Trained on ALL diseases, tested on subset |
| TransE+XGBoost (disease holdout) | ~16% | Honest | TransE fails to generalize |
| Node2Vec Cosine (no ML) | 1.27% | Honest | ML model IS required |
| TxGNN | 6.7-14.5% | Inductive | Zero-shot on unseen diseases |

**CRITICAL (2026-01-27):** Multi-seed evaluation (h40) revealed seed 42 was lucky. True means are lower than single-seed reports:
- Previously reported 31.09% (XGBoost tuned) → actual mean **25.85% ± 4.06%**
- Previously reported 28.73% (XGBoost default) → actual mean **23.73% ± 3.73%**

**BREAKTHROUGH (h39):** kNN collaborative filtering (k=20 nearest diseases by Node2Vec similarity) achieves **37.04% ± 5.81%** — a **+10.47 pp** improvement over the best ML model (p=0.002). No ML model needed.

**Progression:** 37.4% → 41.8% (fuzzy, pair-level) → Generalization crisis → 25.85% (honest XGBoost, 5-seed) → **37.04% (kNN collab filtering, 5-seed)**

**DRKG CEILING (2026-01-28):** 37% R@30 is the maximum achievable with DRKG-only approaches. Oracle ceiling is 60%.

**LEAKAGE QUANTIFIED (2026-02-01):** Retrained Node2Vec WITHOUT 64K treatment edges:
- Original embeddings: 36.59% ± 3.90% R@30 (includes leakage)
- **Honest embeddings: 26.06% ± 3.84% R@30** (fair comparison)
- 10.5 pp drop (29% was leakage), 71.2% retained from indirect paths
- Fair TxGNN comparison: **26.06%** vs 6.7-14.5% (gap ~12-19 pp, not ~23-30 pp)

**EXTERNAL DATA TESTED & FAILED (2026-01-28):** HPO/PPI features WORSE than Node2Vec — details in `docs/archive/experiment_history.md`

## Key Learnings

### What Works
1. **kNN Collaborative Filtering** (h39) - **37.04% ± 5.81% R@30** — BEST METHOD
   - k=20 nearest diseases by Node2Vec cosine, rank drugs by weighted frequency
   - No ML model needed — purely similarity-based
   - +10.47 pp over XGBoost (p=0.002, highly significant)
2. **Fuzzy Disease Matching** - 41.8% R@30 (pair-level, inflated but useful for within-dist)
3. **Disease holdout splits** - Required for honest novel discovery evaluation
4. **Multi-seed evaluation** (h40) - Single-seed has ±4 pp noise; must use 5+ seeds
5. **Node2Vec embeddings** - Best disease similarity measure for kNN (vs gene overlap, etc.)

### CRITICAL: Paradigm Shift (2026-01-27)
- **kNN collaborative filtering outperforms ALL ML models** by >10 pp (h39)
- "Similar diseases share treatments" is the dominant signal for drug repurposing
- XGBoost model adds ZERO value on top of kNN (h42 hybrid = pure kNN)
- kNN is limited: can only recommend drugs from similar training diseases' GT
- 44% of test diseases have 0% GT drug coverage in kNN pool
- Node2Vec cosine is the best fair disease similarity measure (h41; gene overlap hurts)
- kNN parameters already optimal: k=20, raw scores, linear weighting (h43)

### Generalization Gap
- **GB + TransE does NOT generalize**: 3-12% R@30 on disease holdout (h5)
- **Node2Vec XGBoost**: 25.85% ± 4.06% mean (5-seed honest, h40)
- Previously reported 31.09%/28.73% were from lucky seed 42

### What SEEMED to Work (but was data leakage)
1. **Boost features** - Target overlap, chemical similarity, ATC were circular
2. **Evaluating on training diseases** - Inflated recall from 41.9% to 47.5%
3. **All negative sampling strategies** - Hard negatives, random, drug-treats-other ALL fail under disease holdout

### What Fails (Key Patterns)
- **External data** (h17, h19) - HPO/PPI features WORSE than Node2Vec; already captured
- **Additional features** (h34, h35) - Gene/graph features add nothing (sparsity, leakage)
- **ML improvements** (h41-h45) - Gene similarity hurts, XGBoost doesn't help, learned similarity overfits
- **37% = DRKG ceiling** - kNN at 37%, oracle 60%; 23 pp gap needs external data or new architectures
- **Treatment edge leakage** - Original 36.59% → honest 26.06% (29% was leakage)
- **Explicit graph traversal** (h93, h95) - Gene: 3.53% R@30, Pathway: 3.57% R@30. Embeddings >> symbolic reasoning
- Details: `docs/archive/experiment_history.md`

### Confidence Features (h111 validated)

**VALIDATED:** Drug frequency (+9.4pp), Chemical similarity (+8.8pp), Mechanism support (+6.5pp), Category tier
**Multi-Signal Ensemble (h106):** 22% precision @ top 10%. Best combo: Mechanism+Frequency = 20% precision (orthogonal, r=0.07)

### Production Tiered Confidence System (h135 validated)

| Tier   | Criteria | Precision | Separation |
|--------|----------|-----------|------------|
| GOLDEN | Tier1 + freq>=10 + mechanism | 57.7% | 9.1x vs LOW |
| HIGH | freq>=15+mech OR rank<=5+freq>=10+mech | 20.9% | 3.3x |
| MEDIUM | freq>=5+mech OR freq>=10 | 14.3% | 2.2x |
| LOW | All else passing filter | 6.4% | baseline |
| FILTER | rank>20 OR no_targets OR (freq<=2 AND no_mech) | 3.2% | excluded |

**Key findings (2026-02-05):**
- h126: XGBoost +2.07pp from interactions (freq dominant at 35%)
- h130: Linear better for infectious/autoimmune/ophthalmic; ALL hits had Linear>XGBoost
- h132: Tier1+freq>=15+mech = 57.9% precision (8x baseline)

## Performance Gaps & Error Patterns

**Gaps:** Biologics (mAbs 17% vs small mol 32%), Antibiotics (wrong diseases), GI (5% kNN blind spot)
**Best:** ACE inhibitors 67%, Autoimmune 63%, Infectious 52% | **Worst:** mAbs 27%, Antibiotics 6-20%, PPIs 17%

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

## External Resources for Breaking the 37% Ceiling

| Resource | Type | Potential Use |
|----------|------|---------------|
| **helicalAI/helical** | Bio Foundation Models | Disease embeddings from gene expression |
| GEO/GTEx | Gene expression DB | Skin disease expression profiles |
| DrugBank indications | Drug-disease GT | Expand ground truth coverage |

**helicalAI/helical**: Install `source .venv-helical/bin/activate` (Python 3.11)

## Validation & Confounding (Summary)

**Scripts:** `src/external_validation.py`, `src/confounding_detector.py`

| Check | Finding | Action |
|-------|---------|--------|
| Validation pipeline | 57% strong evidence, 10% moderate (best candidates) | Auto-queries ClinicalTrials + PubMed |
| Confounding detection | 7 high-confidence false positives (statins→T2D, etc.) | Filter inverse indications |
| Disease name matching | Fuzzy improved 37.4%→41.8% R@30 | MONDO→MESH bridge |

**Key confounding patterns:** Inverse indication (drug causes disease), cardiac-metabolic comorbidity, polypharmacy
**Details:** `docs/archive/detailed_analysis_findings.md`

## Production Deployment (h68, h72, h73, h66)

**Confidence Scoring (h68):**
- Combined avg of h52+h65+category = 88% precision at 0.7 threshold
- h52-only at 0.8 threshold = 84% precision (simpler, recommended)
- Model: `models/meta_confidence_model.pkl`

**Production Deliverable (h72):**
- `data/deliverables/drug_repurposing_predictions_with_confidence.xlsx`
- 13,416 predictions, 2,797 HIGH confidence novel
- Validated: Sirolimus→TSC (FDA-approved), Lovastatin→atherosclerosis

**Category-Specific k (h66):**
- k=5: dermatological, cardiovascular, psychiatric, respiratory
- k=10: autoimmune, gastrointestinal
- k=30: cancer (+3.9 pp), metabolic (+9.1 pp), other
- k=20: infectious, neurological (default)

## TxGNN Summary

14.5% R@30 on our benchmark, excels at storage diseases (83.3%). **However, direct comparison to our 37% kNN is unfair** — TxGNN was designed for zero-shot on diseases with NO graph edges, while our kNN leverages test diseases' existing graph presence. Under equivalent transductive conditions, performance would likely be comparable. Details: `docs/archive/txgnn_learnings.md`

## Zero-Shot Benchmark (2026-02-03)

**470 diseases with NO FDA-approved treatments** — Every Cure's core mission
- 31 in DRKG (6.6%) — can use graph methods
- 439 NOT in DRKG (93.4%) — require literature mining (h91)
- Benchmark: `data/analysis/zero_shot_benchmark.json`

## Methodology & Evidence

**Full docs:** `docs/impressive_evidence_report.md`, `docs/methodology_limitations.md`

**Key limitations:** Transductive evaluation (test diseases in graph), bimodal performance (15% have zero coverage), selection bias (only 9% of Every Cure evaluable)

## Archive Index

| Archive | Content |
|---------|---------|
| `docs/impressive_evidence_report.md` | **Harvard-impressive evidence: inductive eval, novel discovery, mechanisms** |
| `docs/mechanism_report.md` | Biological mechanism tracings for validated predictions |
| `docs/methodology_limitations.md` | Full methodological limitations documentation |
| `docs/methodology_summary.md` | One-page executive summary for sharing |
| `docs/archive/experiment_history.md` | ATC, Chemical, Pathway, Similarity, Target experiments |
| `docs/archive/validation_sessions.md` | Literature validation batches 1+2, novel predictions |
| `docs/archive/txgnn_learnings.md` | TxGNN training, evaluation, fine-tuning experiments |
| `docs/archive/detailed_analysis_findings.md` | Biologic gap, infectious disease, confounding, validation details |
