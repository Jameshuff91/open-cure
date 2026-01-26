# Open-Cure Project Instructions

## Memory Management

**After each research session:** Update this file with key learnings before committing.

**Periodically ask:** "Should we prune working memory and move details to long-term storage?"
- Archive location: `docs/archive/`
- Keep CLAUDE.md lean (<400 lines) for efficient context loading

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

| Model | Per-Drug R@30 | Notes |
|-------|---------------|-------|
| **GB + Fuzzy Matcher (fixed)** | **41.8%** | Best baseline, 1,236 diseases, 3,618 pairs |
| Node2Vec (held-out diseases) | 41.9% | Fair evaluation for novel discovery |
| GB + Quad Boost (inflated) | 47.5%* | *Circular features - NOT real improvement |
| TxGNN | 6.7% | Near-random for most diseases |

*The 47.5% was inflated by evaluating on training diseases with circular boost features

**Progression:** 37.4% → 41.8% (fuzzy matcher fix)

## Key Learnings

### What Works
1. **Fuzzy Disease Matching** - 41.8% R@30 (up from 37.4% exact-only)
2. **Node2Vec + XGBoost** - 41.9% R@30 on held-out diseases (fair evaluation)
3. **Disease holdout splits** - Required for honest novel discovery evaluation
4. **DRKG graph embeddings** - 256-dim Node2Vec captures treatment relationships

### What SEEMED to Work (but was data leakage)
1. **Boost features** - Target overlap, chemical similarity, ATC were circular
2. **Evaluating on training diseases** - Inflated recall from 41.9% to 47.5%

### What Fails
1. **Embedding Similarity** - TransE cosine similarity causes data leakage
2. **Retraining with Features** - Adding features and retraining: 37%→6%
3. **Correlated Features** - Pathway adds only +0.36% (correlates with target)
4. **Biologics** - mAbs achieve only 27.3% recall vs 47.5% average
5. **Infectious Diseases** - Only 13.6% recall (different mechanisms)
6. **Circular Boost Features** - Target overlap, chemical similarity, ATC codes are circular
7. **Biologic Naming Penalty** - WHO INN naming convention unreliable for filtering

## Error Patterns

| Best Performance | Worst Performance |
|------------------|-------------------|
| ACE inhibitors: 66.7% | Monoclonal antibodies: 27.3% |
| Autoimmune: 63.0% | Infectious: 13.6% |
| Psychiatric: 62.5% | PPIs: 16.7% |

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

## TxGNN Summary

- 14.5% R@30 (comparable to early GB model)
- Excels at storage diseases (83.3% R@30)
- Fine-tuning causes catastrophic forgetting
- Best use: ensemble with min(TxGNN_rank, GB_rank)
- **Details:** `docs/archive/txgnn_learnings.md`

## Archive Index

| Archive | Content |
|---------|---------|
| `docs/archive/experiment_history.md` | ATC, Chemical, Pathway, Similarity, Target experiments |
| `docs/archive/validation_sessions.md` | Literature validation batches 1+2, novel predictions |
| `docs/archive/txgnn_learnings.md` | TxGNN training, evaluation, fine-tuning experiments |
