# Research Loop Progress

## Current Session: h29 Evaluation (2026-01-27)

### Session Summary

**Agent Role:** Research Executor
**Status:** Completed
**Hypothesis Tested:** h29 (Verify Node2Vec Held-Out Disease Generalization)
**Outcome:** VALIDATED — **Node2Vec generalizes to unseen diseases (28.73% R@30)**

### Experiment Details

**Objective:** Verify whether Node2Vec+XGBoost genuinely generalizes to unseen diseases, or if the "41.9% on held-out diseases" claim was inflated by pair-level split.

**Method:**
1. Disease-level 80/20 holdout split (seed=42, same as h5)
2. Tested 7 configurations: existing models, retrained models, cosine similarity
3. Node2Vec (256-dim) vs TransE (128-dim) under identical conditions
4. XGBoost classifier with concat and concat+product+diff features
5. Positive controls: Metformin→T2D, Rituximab→MS, Imatinib→CML, Lisinopril→HTN

### Results

| Experiment | R@30 | Notes |
|---|---|---|
| **Existing GB+TransE (pair-trained)** | **45.89%** | Trained on ALL diseases (inflated) |
| Existing Node2Vec (pair-trained) | 21.64% | Trained on ALL diseases |
| **Node2Vec+XGBoost concat (disease holdout)** | **28.73%** | **HONEST BASELINE** |
| Node2Vec+XGBoost full feat (disease holdout) | 28.73% | Same as concat |
| TransE+XGBoost full feat (disease holdout) | 16.64% | TransE generalizes worse |
| Node2Vec Cosine (no ML) | 1.27% | ML model IS required |
| TransE Cosine (no ML) | 0.00% | ML model IS required |

### Key Findings

1. **Node2Vec DOES generalize** — 28.73% R@30 on 88 held-out diseases, vs 16.64% for TransE (1.73x better)
2. **"41.9% on held-out diseases" was INCORRECT** — original code used pair-level split, not disease-level
3. **Feature type doesn't matter for Node2Vec** — concat and concat+product+diff yield identical 28.73%
4. **Cosine similarity is useless** — 0-1.27% without ML model
5. **Embedding method is the critical factor** — Node2Vec random walks capture transferable patterns that TransE's translational model does not

### Positive Controls (Node2Vec concat, disease holdout)

| Drug→Disease | Rank | Hit@30 |
|---|---|---|
| Metformin→T2D | 118 | No |
| Rituximab→MS | 217 | No |
| Imatinib→CML | 20 | Yes |
| Lisinopril→HTN | 15 | Yes |

### Files Created/Modified

| File | Action |
|------|--------|
| `scripts/evaluate_node2vec_generalization.py` | Used (existed) |
| `data/analysis/h29_node2vec_generalization_results.json` | Created |
| `research_roadmap.json` | Updated (h29 validated, h32 invalidated, 4 new hypotheses) |
| `CLAUDE.md` | Updated with Node2Vec generalization finding |

### New Hypotheses Added

| ID | Title | Priority | Rationale |
|----|-------|----------|-----------|
| h34 | Node2Vec + Graph Topological Features Hybrid | 1 | Graph features may complement Node2Vec embeddings |
| h35 | Node2Vec + Gene-Disease Feature Hybrid | 2 | Gene-based features are inductive |
| h36 | Node2Vec Hyperparameter Tuning for Generalization | 6 | Different p/q params may improve generalization |
| h37 | Node2Vec Generalization Analysis by Disease Category | 3 | Understand per-category performance for targeted improvement |

### Recommended Next Hypothesis

**h37: Node2Vec Generalization Analysis by Disease Category** (Priority 3)

**Why:**
- Low effort, immediate insight from existing h29 results
- Identifies which disease categories Node2Vec generalizes well/poorly for
- Guides targeted improvement in h34/h35
- Can be completed quickly before investing in expensive graph feature computation

**Alternative:** h34 (Graph + Node2Vec Hybrid) — higher potential impact but much higher effort

---

## Previous Session: h5 Evaluation (2026-01-27)

### Session Summary

**Agent Role:** Research Executor
**Status:** Completed
**Hypothesis Tested:** h5 (Hard Negative Mining)
**Outcome:** INVALIDATED — **CRITICAL DISCOVERY: Model generalization failure**

### Experiment Details

**Objective:** Test whether training with hard negatives (high-scoring false positives, confounding patterns) improves R@30.

**Key Innovation (v2):** Used disease-level holdout (80/20 split) instead of pair-level split. This is the CORRECT methodology for evaluating novel disease generalization.

**Approach:**
1. Split 465 GT diseases into 354 train / 88 test (disease-level holdout)
2. Trained 5 strategies: random negatives, drug-treats-other, model-scored FP (25%, 50%), and confounding patterns
3. Evaluated ALL strategies + existing baseline model on held-out test diseases
4. Ran positive controls (Metformin→T2D, Rituximab→MS, Imatinib→CML, Lisinopril→HTN)

### Results

| Strategy | R@30 (test) | Delta |
|----------|-------------|-------|
| **Baseline (existing model)** | **45.89%** | **---** |
| A: Random negatives | 12.43% | -33.5% |
| B: Drug-treats-other | 3.29% | -42.6% |
| C: B + 50% model FP | 5.67% | -40.2% |
| D: B + 25% model FP | 8.59% | -37.3% |
| E: D + confounding | 7.31% | -38.6% |

### CRITICAL DISCOVERY

**The GB model with TransE embedding features CANNOT generalize to unseen diseases.**

Every freshly trained model collapses to 3-12% R@30 on held-out diseases, regardless of negative sampling strategy. The existing model works (45.89%) because it was trained on ALL diseases using pair-level split — it memorizes per-disease patterns.

**Root Cause Analysis:**
- TransE embeddings encode entity-specific information
- concat/product/diff features capture pairwise relationships that are disease-specific
- GB model learns to recognize specific disease embedding patterns, not transferable drug-disease rules
- With 20% of diseases removed from training, the model has no basis for scoring unseen diseases

**Implications:**
1. The reported 41.8% R@30 is within-distribution performance, NOT novel disease generalization
2. The Node2Vec "41.9% on held-out diseases" is UNVERIFIED (code uses pair-level split)
3. Hard negative mining is irrelevant — the bottleneck is architectural, not data quality
4. Future work must focus on inductive approaches: graph features, gene-based representations, or models that can score unseen diseases

**Positive Controls:**
| Drug | Baseline Rank | Strategy A Rank | Strategy B Rank |
|------|--------------|-----------------|-----------------|
| Rituximab→MS | 6 | 49 | 189 |
| Imatinib→CML | 1 | 15 | 93 |
| Lisinopril→HTN | 12 | 406 | 69 |
| Metformin→T2D | 3,426 | 317 | 485 |

### Files Created/Modified

| File | Action |
|------|--------|
| `scripts/evaluate_hard_negatives_v2.py` | Created |
| `data/analysis/h5_hard_negatives_v2_results.json` | Created |
| `research_roadmap.json` | Updated (h5 invalidated, 5 new hypotheses) |
| `CLAUDE.md` | Updated with generalization gap finding |

### New Hypotheses Added

| ID | Title | Priority | Rationale |
|----|-------|----------|-----------|
| h29 | Verify Node2Vec Held-Out Generalization | 1 | Must verify if Node2Vec actually generalizes |
| h30 | Graph Feature-Based Generalization | 2 | Topological features may transfer |
| h31 | Inductive Disease Representation via Genes | 3 | Gene-based features are inductive |
| h32 | Embedding Similarity Ranking (No ML) | 4 | Cosine similarity needs no training |
| h33 | Quantify Generalization Gap | 5 | Statistical verification with multiple seeds |

### Recommended Next Hypothesis

**h29: Verify Node2Vec Held-Out Disease Generalization** (Priority 1)

**Why:**
- Highest priority: must establish whether ANY approach generalizes before investing in new architectures
- Low effort: just re-run existing evaluation with disease-level split on Node2Vec embeddings
- Resolves whether the "41.9%" claim is valid
- If Node2Vec generalizes, we know the embedding method matters (not just the classifier)
- If it doesn't generalize, we need fundamentally new approaches (graph features, gene-based, inductive)

---

## Previous Session: h4 Evaluation (2026-01-26)

### Session Summary

**Agent Role:** Research Executor
**Status:** Completed
**Hypothesis Tested:** h4 (Expand Ground Truth with DrugBank/ChEMBL Indications)
**Outcome:** INCONCLUSIVE

### Experiment Details

**Objective:** Expand the ground truth dataset with additional FDA-approved indications from DrugBank/ChEMBL to improve evaluation accuracy and potentially R@30.

**Approach:**
1. Analyzed existing DrugBank data availability
2. Compared DRKG treatment edges with current GT
3. Identified missing FDA-approved pairs from validation
4. Evaluated impact of adding new pairs

### Key Findings

**1. GT Already Contains DRKG:**
- All 4,968 DRKG `DRUGBANK::treats` edges are in the current GT
- GT has 58,016 pairs (53K more than DRKG alone)
- Every Cure annotations provide comprehensive coverage

**2. Missing FDA Pairs Identified:**
| Drug | Disease | Rank | Hit@30 |
|------|---------|------|--------|
| Pembrolizumab | Breast Cancer | 7 | ✓ |
| Natalizumab | Multiple Sclerosis | 16 | ✓ |
| Erlotinib | Pancreatic Cancer | 16 | ✓ |
| Cetuximab | Colorectal Cancer | 1 | ✓ |
| Oxaliplatin | Colorectal Cancer | 37 | ✗ |
| Bevacizumab | Colorectal Cancer | 7 | ✓ |

**3. Impact Analysis:**
- 5/6 (83.3%) missing pairs already hit@30
- Adding 6 pairs: R@30 increases by **+0.22 pp** (42.04% → 42.26%)
- Impact is marginal because few pairs missing and most already hit

### Data Availability Blockers

| Data Source | Status | Notes |
|-------------|--------|-------|
| DrugBank lookup | ✓ Available | Only ID-name mappings, no indications |
| DrugBank XML | ✗ Missing | Requires license/download |
| ChEMBL API | ✗ Not implemented | Would need API integration |
| DRKG treats | ✓ Already in GT | 100% overlap with current GT |

### Conclusions

1. **The Every Cure GT is highly comprehensive** - already contains all DRKG treatment edges
2. **Missing FDA pairs are few** - manual search found only 6 significant gaps
3. **Model already learns relationships** - 83% of missing pairs would hit@30
4. **Marginal impact** - +0.22 pp improvement from 6 additions
5. **Systematic expansion blocked** - requires DrugBank XML license or ChEMBL API

### Hypotheses Updated

| ID | Title | Status | Change |
|----|-------|--------|--------|
| h4 | Expand GT with DrugBank/ChEMBL | **inconclusive** | Blocked by data access |
| h28 | DrugBank XML Indication Extraction | **added** | For future with proper data |

### Files Created/Modified

| File | Action |
|------|--------|
| `data/reference/expanded_ground_truth_h4.json` | Created (GT + 6 new pairs) |
| `data/analysis/h4_gt_expansion_results.json` | Created |
| `research_roadmap.json` | Updated |

### Recommended Next Hypothesis

**h5: Hard Negative Mining** (Priority 5)

**Why:**
- Medium impact, medium effort
- Addresses model discrimination quality
- Can use existing confounding patterns as negatives
- Does not require external data

**Alternative:** h8 (Confidence-Based Post-Filtering) - low effort, quick validation

---

## Previous Session: h3 Evaluation (2026-01-26)

### Session Summary

**Agent Role:** Research Executor
**Status:** Completed
**Hypothesis Tested:** h3 (Infectious Disease Specialist Model)
**Outcome:** INVALIDATED

### Experiment Details

**Objective:** Test if a specialist XGBoost model trained only on infectious disease pairs can outperform the general GB model.

**Reported Baseline:** 13.6% R@30 for infectious diseases (from CLAUDE.md)
**Actual Baseline:** 52.0% R@30 on 47 mappable infectious diseases

### Critical Finding: Baseline Discrepancy

The 13.6% figure in CLAUDE.md was based on **antibiotic CLASS performance** (e.g., fluoroquinolones 0%, macrolides 6%), not disease-level evaluation.

**Actual General Model Performance:**
- 52.0% R@30 on infectious diseases (104/200 hits)
- 47 diseases evaluated with proper EC-to-DRKG mapping
- Best performers: E. coli infections 100%, Herpes zoster 100%
- Worst performers: Diabetic foot infections 0%, Cutaneous candidiasis 0%

### Specialist Model Results

| Model | R@30 | Test Diseases | Notes |
|-------|------|---------------|-------|
| General GB | 63.6% | 12 | Baseline |
| Specialist | 36.4% | 12 | **Underperforms by 27.3%** |

**Root Cause of Specialist Underperformance:**
1. Insufficient training data: 294 positive pairs vs ~3000 for general model
2. Disease-level split left only 12 test diseases
3. General model's broader training data provides better feature learning

### Key Insights

1. **The "infectious disease problem" was mischaracterized.** The real issue is antibiotics being predicted for NON-infectious diseases (spurious predictions), not poor recall ON infectious diseases.

2. **General model already performs well (52% R@30)** on infectious diseases when evaluated properly.

3. **Specialist approach is unnecessary.** The confidence_filter.py already handles spurious antibiotic predictions.

### Hypotheses Updated

| ID | Title | Status | Change |
|----|-------|--------|--------|
| h3 | Infectious Disease Specialist | **invalidated** | General model outperforms |
| h26 | Antibiotic Prediction Filtering Analysis | **added** | Low priority |
| h27 | Per-Category Baseline Documentation | **added** | Verify other categories |

### Files Created/Modified

| File | Action |
|------|--------|
| `scripts/evaluate_infectious_specialist.py` | Created |
| `data/analysis/h3_infectious_specialist_results.json` | Created |
| `data/analysis/infectious_baseline_evaluation.json` | Created |
| `data/analysis/infectious_antimicrobial_analysis.json` | Created |
| `research_roadmap.json` | Updated |

### Recommended Next Hypothesis

**h4: Expand Ground Truth with DrugBank/ChEMBL Indications** (Priority 4)

**Why:**
- Medium impact, medium effort
- Addresses data sparsity root cause
- Can directly increase training examples
- Validation found 4 FDA-approved drugs missing from GT

**Alternative:** h5 (Hard Negative Mining) - addresses model discrimination quality

---

## Previous Session: h1 Evaluation (2026-01-26)

### Session Summary

**Agent Role:** Research Executor
**Status:** Completed
**Hypothesis Tested:** h1 (GB + TxGNN Best-Rank Ensemble)
**Outcome:** INVALIDATED

### Experiment Details

**Objective:** Test if taking min(GB_rank, TxGNN_rank) improves R@30 beyond either model alone.

**Method:**
1. Loaded TxGNN predictions from `data/reference/txgnn_predictions_final.csv`
2. Loaded GB model and generated predictions for all drug-disease pairs
3. Computed ensemble ranks as min(GB_rank, TxGNN_rank)
4. Evaluated on 603 diseases with MESH mappings

**Results:**

| Model | R@30 | Diseases | Notes |
|-------|------|----------|-------|
| GB model | 42.04% | 567 | Baseline |
| TxGNN | 0.00% | 602 | **CRITICAL: Zero hits** |
| Ensemble | 42.03% | 566 | No improvement |

### Critical Finding

**TxGNN predictions file limitation:** The pre-computed `txgnn_predictions_final.csv` only contains the **top 50 drugs per disease**. Ground truth drugs almost never appear in TxGNN's top-50:

| Disease | GT Drugs | In TxGNN Top-50 |
|---------|----------|-----------------|
| Psoriasis | 4 | 0 |
| Hypertension | 3 | 0 |
| Breast cancer | 4 | 0 |
| Rheumatoid arthritis | 3 | 0 |
| All tested | 27 | 0 (0%) |

**Root Cause:** TxGNN ranks most GT drugs in the 1000s-7000s range (out of 7954 drugs total). Only ~14.5% of GT drugs rank within top-30 (per archive). The pre-computed file misses these because it only stores top-50.

**Implication:** TxGNN ensembles require **live GPU inference** to rank all drugs, not pre-computed files.

### Hypotheses Updated

| ID | Title | Status | Change |
|----|-------|--------|--------|
| h1 | GB + TxGNN Best-Rank Ensemble | **invalidated** | Blocked by data format |
| h2 | Category-Routed Ensemble | **blocked** | Same blocker |
| h23 | TxGNN Full Ranking Storage | **added** | Requires GPU |
| h24 | GB Error Analysis by Drug Class | **added** | Next priority |
| h25 | Embedding Distance Calibration | **added** | Alternative approach |

### Current Baseline Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| **R@30** | **42.04%** | Per-drug Recall@30 (slightly higher than 41.8% in CLAUDE.md due to evaluation subset) |
| Diseases | 567 | With GB embeddings and MESH mapping |
| Pairs | 1,125 | Drug-disease ground truth pairs |
| Model | GB + Fuzzy Matcher | Gradient Boosting with fuzzy disease name matching |

### Key Learning

> TxGNN pre-computed predictions contain only top-50 drugs per disease. GT drugs are NOT in top-50 for most diseases (0% coverage). TxGNN ensembles require live GPU inference, not pre-computed files.

### Files Created/Modified

| File | Action |
|------|--------|
| `scripts/evaluate_best_rank_ensemble.py` | Created |
| `data/analysis/h1_ensemble_results.json` | Created |
| `research_roadmap.json` | Updated (h1 invalidated, h2 blocked, 3 new hypotheses) |
| `research_loop/progress.md` | Updated (this file) |

### Recommended Next Hypothesis

**h3: Infectious Disease Specialist Model** (Priority 3)

**Why:**
- Does not require GPU or TxGNN
- High expected impact
- Addresses known model weakness (13.6% recall for infectious diseases)
- Can be implemented with existing data

**Alternative:** h24 (GB Error Analysis by Drug Class) - low effort, provides insights for future work.

---

## Previous Session: Initialization (2026-01-26)

### Session Summary

**Agent Role:** Research Initializer
**Status:** Completed
**Output:** `research_roadmap.json` with 22 prioritized hypotheses

### Current Baseline Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| **R@30** | **41.8%** | Per-drug Recall@30 |
| Diseases | 1,236 | 30.9% of Every Cure diseases mapped |
| Pairs | 3,618 | Drug-disease ground truth pairs |
| Model | GB + Fuzzy Matcher | Gradient Boosting with fuzzy disease name matching |
| Validation Precision | 22.5% | Top predictions (batches 1+2) |

### Key Findings from Analysis

1. **TxGNN Opportunity:** Achieves 83.3% R@30 on storage diseases vs 6.7% overall. Category-specific routing is promising.

2. **Biologic Gap:** mAbs have only 2.1 diseases/drug vs 11.1 for small molecules. Data sparsity is root cause.

3. **Infectious Disease Paradox:** More training data correlates with WORSE performance for antibiotics. Model learns wrong patterns.

4. **Circular Feature Warning:** Quad Boost features (target overlap, ATC, chemical similarity) were found to be circular and inflate metrics when evaluated on training diseases.

5. **Validation Reality:** Only 22.5% of top predictions validate against literature. Strong need for better filtering.

---

*Last updated: 2026-01-26*
*Agent: Research Executor*
*Hypothesis tested: h4 (inconclusive)*
