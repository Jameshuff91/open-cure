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
| **kNN k=20 (original embeddings)** | **36.59% ± 3.90%** | Honest (5-seed) | Has treatment edge leakage |
| **kNN k=20 (no-treatment embeddings)** | **26.06% ± 3.84%** | **FAIR (5-seed)** | **Best fair comparison to TxGNN** |
| Node2Vec+XGBoost TUNED (disease holdout) | 25.85% ± 4.06% | Honest (5-seed) | md=6,ne=500,lr=0.1,alpha=1.0 (h38/h40) |
| Node2Vec+XGBoost default (disease holdout) | 23.73% ± 3.73% | Honest (5-seed) | md=6,ne=100,lr=0.1 (h40) |
| GB + Fuzzy Matcher (fixed) | 41.8% | Within-dist | 1,236 diseases, pair-level (inflated) |
| GB + TransE (existing, on test) | 45.9% | Pair-trained | Trained on ALL diseases, tested on subset |
| TransE+XGBoost (disease holdout) | ~16% | Honest | TransE fails to generalize |
| Node2Vec Cosine (no ML) | 1.27% | Honest | ML model IS required |
| TxGNN | 6.7% | Unknown | Near-random for most diseases |

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

**EXTERNAL DATA TESTED & FAILED (2026-01-28):**
- h19 (HPO Phenotype): 14.20% R@30 — WORSE than Node2Vec (36.93%)
- h17 (PPI Network): 16.18% R@30 — WORSE than Node2Vec (36.93%)

The 23 pp gap is NOT simply missing external data. Node2Vec already captures functional similarity. Breaking the ceiling requires **fundamentally different approaches** (GNN, meta-learning, attention) or **better ground truth coverage**.

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
- Details: `docs/archive/experiment_history.md`

## Performance Gaps (Summary)

| Gap | Issue | Fix | Details |
|-----|-------|-----|---------|
| Biologics | mAbs 16.7% vs small mol 32.1% | Filter oncology mAbs | `docs/archive/detailed_analysis_findings.md` |
| Antibiotics | Predicted for non-infectious | Filter spurious predictions | `docs/archive/detailed_analysis_findings.md` |
| GI diseases | 5% hit rate (kNN blind spot) | Flag/exclude in production | h59 findings |

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

## External Resources for Breaking the 37% Ceiling

The kNN method has hit a 37% R@30 ceiling with DRKG-only approaches. These external resources may help:

| Resource | Type | Potential Use | Hypothesis |
|----------|------|---------------|------------|
| **helicalAI/helical** | Bio Foundation Models | Dense disease embeddings from gene expression | h61 |
| GEO/GTEx | Gene expression DB | Skin disease expression profiles for Ryland | h61 |
| DrugBank indications | Drug-disease GT | Expand ground truth coverage | h4 |
| UMLS Metathesaurus | Ontology cross-refs | Improve disease name mapping | h9 |

**helicalAI/helical** (https://github.com/helicalAI/helical): Geneformer, scGPT, etc. for disease embeddings from gene expression. Install: `source .venv-helical/bin/activate` (Python 3.11)

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

## Methodological Critique Response (2026-02-01)

**Full documentation:** `docs/methodology_limitations.md`, `docs/methodology_summary.md`

### Key Findings

| Analysis | Result | Implication |
|----------|--------|-------------|
| **Statistical significance** | p=0.025, Cohen's d=2.44 | 10.5 pp difference is significant |
| **GT circularity** | 32% overlap with DRKG | Low - evaluation tests prediction, not recall |
| **Coverage dependence** | 85% with coverage: 24.2% R@30 / 15% without: 0% | Bimodal performance |
| **Rare disease gap** | 1 GT drug: 13.5% vs 6+ drugs: 30% | kNN fails for rare diseases |
| **Selection funnel** | 90.8% attrition (3996 → 368) | MESH mapping is bottleneck |
| **Disconnected diseases** | 51 GT diseases lost | Includes Parkinson's (19 drugs) |
| **Embedding quality** | No NaN/Inf/zero-norm | Clean embeddings |
| **Precision@30** | 5.9% | Low due to large drug pool |

### Known Limitations (Cannot Fix)

1. **Transductive evaluation** - Test diseases retain non-treatment graph presence
2. **Bimodal performance** - 15% of diseases have zero kNN coverage
3. **Rare disease failure** - No similar diseases → no treatment transfer
4. **Selection bias** - Only 9% of Every Cure diseases evaluable

### Fair Comparison Statement

> Our 26.06% R@30 (honest embeddings) vs TxGNN 6.7-14.5% represents ~2x improvement, but paradigms differ. Our evaluation is transductive (test diseases in graph); TxGNN is inductive (test diseases removed). The gap is real but smaller than naive comparison suggests.

## Archive Index

| Archive | Content |
|---------|---------|
| `docs/archive/experiment_history.md` | ATC, Chemical, Pathway, Similarity, Target experiments |
| `docs/archive/validation_sessions.md` | Literature validation batches 1+2, novel predictions |
| `docs/archive/txgnn_learnings.md` | TxGNN training, evaluation, fine-tuning experiments |
| `docs/archive/detailed_analysis_findings.md` | Biologic gap, infectious disease, confounding, validation details |
| `docs/methodology_limitations.md` | Full methodological limitations documentation |
| `docs/methodology_summary.md` | One-page executive summary for sharing |
