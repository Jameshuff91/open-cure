# Research Loop Progress

## Current Session: h19 + h17 (External Data Hypotheses) (2026-01-28)

### Session Summary

**Agent Role:** Research Executor
**Status:** Completed (2 hypotheses tested)
**Hypotheses Tested:** h19 (HPO Phenotype Similarity), h17 (PPI Network Distance)
**Key Discovery:** External data sources tested - HPO INVALIDATED, PPI INCONCLUSIVE (informative but blocked by O(n²) computation)

### Results Summary

| Hypothesis | Status | Key Finding |
|---|---|---|
| h19: HPO Phenotype Similarity | **INVALIDATED** | Node2Vec 67.96% >> HPO 36.13% on same diseases. HPO adds no complementary signal. |
| h17: PPI Network Distance | **INCONCLUSIVE** | PPI is statistically informative (2.2x enrichment, p<1e-38) but O(n²) computation blocks practical use. |

### h17: PPI Network Distance Details

**Objective:** Test whether PPI network distance (drug targets → disease genes) can improve predictions.

**Method:**
1. Downloaded STRING PPI network (473K high-confidence edges, 15,757 genes)
2. Built NCBI Gene ID mappings from Ensembl protein IDs
3. Computed coverage: 93.5% drug targets, 54.8% disease genes in PPI
4. Compared PPI distances for GT pairs vs random pairs

**Results:**

| Metric | GT Pairs | Random Pairs |
|--------|----------|--------------|
| Mean distance | 1.15 | 1.91 |
| % with dist≤1 | 70% | 31% |
| % with dist=0 | 19.4% | 2.6% |

Mann-Whitney p-value: **5.42e-39** (highly significant)

**Blocking Issue:** Using PPI for drug ranking requires:
- Computing BFS from each drug's targets to all disease genes
- O(drugs × diseases × graph edges) = billions of operations
- Would require precomputation or architectural changes

### Files Created/Modified (h17)

| File | Action |
|------|--------|
| `data/reference/ppi/9606.protein.links.v12.0.txt.gz` | Downloaded (STRING PPI) |
| `data/reference/ppi/ppi_network_high_conf.json` | Created (458K mapped edges) |
| `data/analysis/h17_ppi_gt_vs_random.json` | Created |
| `data/analysis/h17_ppi_distance_sample.json` | Created |
| `research_roadmap.json` | Updated (h17 inconclusive) |

### Recommended Next Steps

Both top external data hypotheses tested. Results:
1. **Disease-side enrichment (HPO)** → No improvement, Node2Vec already captures
2. **Drug-side enrichment (PPI)** → Informative but computationally blocked

**Remaining options:**
1. **h28 (DrugBank XML)** - More GT data could help kNN
2. **h8 (Confidence Filtering)** - Low effort, improve precision
3. **Precompute PPI distances** - Would unlock h17's potential
4. **Production deployment** - kNN is ready (37% R@30, simple, interpretable)

---

## Previous Session: h19 (Disease Phenotype Similarity) (2026-01-28)

### Session Summary

**Agent Role:** Research Executor
**Status:** Completed
**Hypothesis Tested:** h19 (Disease Phenotype Similarity using HPO)
**Outcome:** INVALIDATED — HPO phenotype similarity provides NO value over Node2Vec

### Experiment Details

**Objective:** Test whether HPO (Human Phenotype Ontology) disease-disease similarity can improve kNN drug repurposing predictions.

**Method:**
1. Downloaded HPO phenotype annotations (280K annotations for 12,958 diseases)
2. Built OMIM→MESH mappings via MONDO ontology (4,576 mappings)
3. Identified 464 GT diseases (13.3%) with HPO annotations
4. Computed Jaccard similarity on HPO phenotype sets
5. Compared HPO-based kNN vs Node2Vec-based kNN on same disease subset

### Results

| Method | R@30 (5-seed) | Notes |
|--------|---------------|-------|
| HPO-only kNN | **36.13% ± 1.46%** | Jaccard similarity on phenotype sets |
| Node2Vec kNN | **67.96% ± 4.63%** | Cosine similarity on embeddings |
| Combined (50/50) | **64.73% ± 2.49%** | WORSE than Node2Vec alone |

### Per-Disease Analysis (seed=42)

| Category | Count | % |
|----------|-------|---|
| Both hit | 31 | 33.3% |
| Node2Vec only wins | 39 | **41.9%** |
| HPO only wins | 1 | **1.1%** |
| Neither hits | 22 | 23.7% |

### Key Findings

1. **Node2Vec is far superior**: 67.96% vs 36.13% (31.83 pp difference, p=0.0001)
2. **HPO provides no complementary signal**: Only 1 disease (1.1%) rescued by HPO when Node2Vec failed
3. **Combining hurts**: 64.73% combined vs 67.96% Node2Vec alone (-3.23 pp)
4. **Coverage limited**: Only 13.3% of GT diseases have HPO annotations via OMIM→MESH
5. **Phenotype overlap sparse**: Mean Jaccard = 0.06 (very little overlap between diseases)

### Root Cause Analysis

1. **Node2Vec already captures disease relationships** from DRKG structure
2. **HPO is focused on rare Mendelian diseases**, not common diseases in GT
3. **Phenotype sets are sparse** - diseases share few phenotypes
4. **13.3% coverage** is too limited for significant impact

### Files Created/Modified

| File | Action |
|------|--------|
| `data/reference/hpo/phenotype.hpoa` | Downloaded (HPO annotations) |
| `data/reference/hpo/mondo.json` | Downloaded (MONDO ontology) |
| `data/reference/hpo/omim_to_mesh.json` | Created (4,576 mappings) |
| `data/reference/hpo/drkg_mesh_to_omim.json` | Created (555 mappings) |
| `data/reference/hpo/disease_phenotypes.json` | Created (8,576 diseases) |
| `data/reference/hpo/gt_diseases_with_hpo.json` | Created (464 diseases) |
| `data/analysis/h19_hpo_similarity_results.json` | Created |
| `data/analysis/h19_hpo_vs_node2vec_results.json` | Created |
| `research_roadmap.json` | Updated (h19 invalidated) |

### Recommended Next Steps

**External phenotype data unlikely to help.** Node2Vec already captures disease relationships. Future external data efforts should focus on:

1. **Drug/target-side enrichment** (PPI networks, drug mechanisms) - h17
2. **More GT data** (DrugBank XML indications) - h28
3. **Production deployment** - kNN method is ready (simple, fast, interpretable)

---

## Previous Session: h40/h39/h42/h43/h41/h44/h45 Multi-Hypothesis (2026-01-27)

### Session Summary

**Agent Role:** Research Executor
**Status:** Completed (7 hypotheses tested)
**Hypotheses Tested:** h40, h39, h42, h43, h41, h44, h45
**Key Discovery:** kNN collaborative filtering achieves 37.04% R@30 — BEST method. DRKG ceiling identified at ~37%. Oracle ceiling 60%.

### Results Summary

| Hypothesis | Status | Key Finding |
|---|---|---|
| h40: Multi-Seed Stability | VALIDATED | Default mean 23.73% ± 3.73%, tuned 25.85% ± 4.06%. Seed 42 was lucky. |
| h39: Disease Similarity Transfer | VALIDATED | **kNN k=20: 37.04% ± 5.81% R@30 (+10.47 pp, p=0.002)** |
| h42: kNN + XGBoost Rescue | INVALIDATED | XGBoost rescue helps NO disease subset. kNN dominates everywhere. |
| h43: kNN Optimization | INVALIDATED | Default config (k=20, raw, linear) already optimal. 72 configs tested. |
| h41: Improved Similarity Measure | INVALIDATED | Gene overlap hurts (23.2%). Node2Vec cosine is best fair measure. |
| h44: Transductive kNN (upper bound) | VALIDATED | LOO = 37.07% (k=30). Oracle ceiling = 60.4%. 23 pp gap. |
| h45: Learned Disease Similarity | INVALIDATED | XGBoost regressor WORSE than cosine (-3.98 pp, p=0.008). |

### Critical Findings

1. **PARADIGM SHIFT**: kNN collaborative filtering (37.04%) >> XGBoost ML model (25.85%). Drug repurposing is fundamentally a similarity/recommendation problem, not a classification problem.

2. **Multi-seed evaluation is essential**: Previously reported 31.09% was from lucky seed=42. True tuned XGBoost mean is 25.85% ± 4.06%. Always use 5-seed evaluation.

3. **kNN is already optimized**: k=20, raw scores, linear similarity weighting is optimal. No normalization, weighting, or similarity measure improvements found.

4. **Ceiling identified**: ~37% R@30 appears to be the inductive ceiling for DRKG-only kNN. Breaking through requires external data sources or fundamentally different approaches.

### New Hypotheses Added

| ID | Title | Priority | Rationale |
|----|-------|----------|-----------|
| h41 | Improved Disease Similarity Measure | 1 | Tested, invalidated — Node2Vec already optimal |
| h42 | kNN + XGBoost Rescue | 2 | Tested, invalidated — XGBoost helps nowhere |
| h43 | kNN Optimization | 3 | Tested, invalidated — default already optimal |
| h44 | Transductive kNN (upper bound) | 4 | Test leave-one-out ceiling for collaborative filtering |

### Recommended Next Steps (Updated 2026-01-28)

**ROADMAP CLEANUP COMPLETED:** 12 DRKG-internal hypotheses blocked, external data hypotheses prioritized.

| Priority | Hypothesis | Data Source | Potential |
|----------|------------|-------------|-----------|
| 1 | h19: Disease Phenotype Similarity | HPO (external) | **TESTED - INVALIDATED** |
| 2 | h17: PPI Network Distance | STRING (external) | New similarity signal |
| 3 | h28: DrugBank Indication Extraction | DrugBank XML | More GT = better kNN |
| 4 | h9: Disease Coverage via UMLS | UMLS (external) | More disease mappings |
| 5 | h16: Clinical Trial Features | ClinicalTrials.gov | External evidence |

**Blocked (DRKG ceiling reached):** h6, h7, h11, h12, h15, h20, h21, h25, h36
**Production-ready:** kNN method can be deployed as-is (simple, fast, interpretable)

---

## Previous Session: h29/h37/h35/h34 Multi-Hypothesis (2026-01-27)

### Session Summary

**Agent Role:** Research Executor
**Status:** Completed (4 hypotheses tested)
**Hypotheses Tested:** h29, h37, h35, h34
**Key Discovery:** DRKG feature ceiling — gene, graph, and embedding features from same KG are redundant

### Results Summary

| Hypothesis | Status | Key Finding |
|---|---|---|
| h29: Node2Vec Generalization | VALIDATED | 28.73% R@30 on disease-level holdout (1.73x better than TransE 16.64%) |
| h37: Category Analysis | VALIDATED | Ophthalmological 100%, hematological 70%, infectious/GI/rare 0% |
| h35: Gene Feature Hybrid | INVALIDATED | +0.73 pp (negligible; sparsity, already captured by embeddings) |
| h34: Graph Feature Hybrid | INVALIDATED | 45.82% was leakage! Clean: -0.18 pp (NO improvement) |
| h32: Cosine Similarity | INVALIDATED | 0-1.27% without ML model (tested as part of h29) |

### Critical Learning: DRKG Feature Ceiling

Features derived from the SAME knowledge graph (DRKG) used for embeddings are **redundant**:
- Gene overlap features → already captured by Node2Vec walks through Gene nodes
- Graph topology features → already captured by Node2Vec random walks
- Treatment edges → circular (they ARE the labels)

**Implication:** Improvement beyond 28.73% requires:
1. **External data sources** (clinical trials, literature, gene expression)
2. **Different model architectures** (GNN, meta-learning, attention)
3. **Better training strategy** (XGBoost hyperparameter tuning, similarity-weighted training)

### New Hypotheses Added

| ID | Title | Priority | Rationale |
|----|-------|----------|-----------|
| h38 | XGBoost Hyperparameter Tuning | 1 | Lowest-effort improvement: shallower trees, more regularization |
| h39 | Disease Similarity Transfer Learning | 2 | Weight training examples by similarity to test disease |
| h40 | Multi-Seed Stability Check | 3 | Verify 28.73% is not a fluke of seed=42 |

### Recommended Next Steps

1. **h38 (XGBoost Hyperparameter Tuning)** — Lowest effort, may squeeze out a few more pp
2. **h39 (Disease Similarity Transfer)** — Novel approach, medium effort, high potential
3. **h40 (Multi-Seed Check)** — Quick validation of baseline stability

---

## Previous Session: h29 Evaluation (2026-01-27)

### Session Summary

**Agent Role:** Research Executor
**Status:** Completed
**Hypothesis Tested:** h29 (Verify Node2Vec Held-Out Disease Generalization)
**Outcome:** VALIDATED — **Node2Vec generalizes to unseen diseases (29.45% R@30)**

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
| Node2Vec+XGBoost concat (disease holdout) | 26.18% | HONEST, concat features only |
| **Node2Vec+XGBoost cpd (disease holdout)** | **29.45%** | **HONEST BASELINE (BEST)** |
| TransE+XGBoost cpd (disease holdout) | 15.90% | TransE generalizes worse |
| Node2Vec Cosine (no ML) | 1.27% | ML model IS required |
| TransE Cosine (no ML) | 0.00% | ML model IS required |

### Key Findings

1. **Node2Vec DOES generalize** — 29.45% R@30 on 88 held-out diseases, vs 15.90% for TransE (1.85x better)
2. **"41.9% on held-out diseases" was INCORRECT** — original code used pair-level split, not disease-level
3. **Concat+product+diff features help Node2Vec** — 26.18% (concat) → 29.45% (cpd), +3.3 pp improvement
4. **Cosine similarity is useless** — 0-1.27% without ML model
5. **Embedding method is the critical factor** — Node2Vec random walks capture transferable patterns that TransE's translational model does not

### Positive Controls (Node2Vec concat, disease holdout)

**Concat model (26.18% R@30):**
| Drug→Disease | Rank | Hit@30 |
|---|---|---|
| Metformin→T2D | 22 | Yes |
| Rituximab→MS | 21 | Yes |
| Imatinib→CML | 12 | Yes |
| Lisinopril→HTN | 27 | Yes |
All 4/4 positive controls pass!

**CPD model (29.45% R@30):**
| Drug→Disease | Rank | Hit@30 |
|---|---|---|
| Metformin→T2D | 45 | No |
| Rituximab→MS | 72 | No |
| Imatinib→CML | 11 | Yes |
| Lisinopril→HTN | 13 | Yes |
2/4 pass (higher overall R@30 but worse on some controls)

### Files Created/Modified

| File | Action |
|------|--------|
| `scripts/evaluate_node2vec_generalization.py` | Created |
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

*Last updated: 2026-01-28*
*Agent: Research Executor*
*Hypothesis tested: h19 (INVALIDATED)*

---

## Session: h19 Disease Phenotype Similarity (2026-01-28)

### Session Summary

**Agent Role:** Research Executor
**Status:** Completed
**Hypothesis Tested:** h19 (Disease Phenotype Similarity using HPO)
**Outcome:** INVALIDATED — HPO phenotype similarity does NOT improve drug repurposing

### Experiment Details

**Objective:** Test whether Human Phenotype Ontology (HPO) phenotype similarity can provide external (non-DRKG) information to break the 37% R@30 ceiling.

**Method:**
1. Downloaded HPO phenotype annotations (phenotype.hpoa - 280K annotations)
2. Created MESH→OMIM/ORPHA mapping via MONDO ontology
3. Built disease phenotype profiles (799 DRKG diseases mapped)
4. Computed Jaccard similarity on HPO term sets
5. Evaluated HPO-only kNN, hybrid kNN (α=0.3-0.7), and subset analysis
6. Multi-seed evaluation (42, 123, 456, 789, 1024) per h40 standard

### Results

| Method | R@30 (5-seed mean ± std) | Delta vs Node2Vec |
|--------|--------------------------|-------------------|
| Node2Vec kNN (baseline) | 36.91% ± 5.59% | --- |
| HPO kNN (all diseases) | 14.20% ± 5.20% | -22.71 pp |
| HPO kNN (HPO subset only) | 34.84% ± 15.99% | -2.07 pp |
| Hybrid α=0.5 (best) | 37.19% ± 5.63% | +0.28 pp |

### Per-Disease Analysis (seed=42)

| Subset | Node2Vec R@30 | HPO R@30 |
|--------|---------------|----------|
| Diseases WITH HPO (25) | 62.3% | 53.6% |
| Diseases WITHOUT HPO (63) | 40.7% | N/A |

**Insight:** HPO-covered diseases are 1.53x easier to predict — they're well-characterized rare/Mendelian diseases. But even on these, Node2Vec beats HPO.

### Key Findings

1. **HPO-only fails badly** (14.20% R@30) — coverage is too sparse (25.6% of GT diseases)
2. **On HPO-covered diseases, Node2Vec still wins** (62.3% vs 53.6%)
3. **Hybrid provides only +0.28 pp** — within noise, not significant
4. **Low correlation (0.126)** between HPO and Node2Vec similarity — different signals, but HPO's is weaker
5. **HPO is Mendelian-focused** — OMIM/Orphanet sources cover rare diseases, not common indications

### Coverage Analysis

| Mapping Step | Count | % of Previous |
|--------------|-------|---------------|
| Total DRKG diseases in GT | 3,492 | 100% |
| Mapped to HPO disease IDs | 1,130 | 32.4% |
| With actual HPO annotations | 799 | 22.9% |
| GT diseases with HPO + embeddings | 119 | 25.6% of GT |

### Files Created

| File | Description |
|------|-------------|
| `data/reference/phenotype.hpoa` | HPO disease-phenotype annotations |
| `data/reference/mondo.obo` | MONDO ontology with xrefs |
| `data/reference/mondo_to_omim_from_obo.json` | MONDO→OMIM mapping |
| `data/reference/mondo_to_orpha_from_obo.json` | MONDO→Orphanet mapping |
| `data/reference/mesh_to_hpo_disease_ids.json` | MESH→HPO disease ID mapping |
| `data/reference/drkg_disease_phenotypes.json` | DRKG disease phenotype profiles |
| `data/reference/hpo_similarity_matrix.npz` | Pre-computed phenotype similarity |
| `scripts/evaluate_hpo_phenotype_knn.py` | Evaluation script |
| `data/analysis/h19_hpo_phenotype_results.json` | Full results |

### Conclusion

**HPO phenotype similarity does NOT improve drug repurposing predictions.**

External phenotype ontology data is NOT the path to breaking the 37% ceiling. The 23 pp gap to oracle ceiling will not be closed by phenotype similarity. Focus should shift to:
1. Other external data (clinical trials, PPI networks, gene expression)
2. Fundamentally different approaches (GNN, attention, meta-learning)
3. Better disease mapping (more GT diseases with DRKG embeddings)

### Recommended Next Steps

1. **h17: PPI Network Distance** — STRING protein-protein interactions (different external source)
2. **h28: DrugBank XML Indication Extraction** — More GT data for better kNN coverage
3. **h9: UMLS Disease Mapping** — Improve coverage (currently 25.6%)
4. **h16: Clinical Trial Features** — Different external signal


---

## Session: h17 PPI Network Distance (2026-01-28)

### Session Summary

**Agent Role:** Research Executor
**Status:** Completed
**Hypothesis Tested:** h17 (PPI Network Distance Features)
**Outcome:** INVALIDATED — PPI similarity does NOT improve drug repurposing

### Experiment Details

**Objective:** Test whether protein-protein interaction (PPI) network proximity from drug targets to disease genes can improve predictions.

**Method:**
1. Loaded STRING PPI network (15,757 genes, high confidence edges)
2. Pre-computed 2-hop gene neighborhoods for 3,454 diseases
3. Computed Jaccard similarity on gene neighborhoods
4. Evaluated PPI-only kNN, Node2Vec kNN, and hybrid combinations
5. Multi-seed evaluation (42, 123, 456, 789, 1024)

### Results

| Method | R@30 (5-seed mean ± std) | Delta vs Node2Vec |
|--------|--------------------------|-------------------|
| Node2Vec kNN (baseline) | 36.93% ± 6.02% | --- |
| PPI kNN (2-hop Jaccard) | 16.18% ± 2.00% | -20.76 pp |
| Hybrid α=0.1 | 36.88% ± 4.68% | -0.05 pp |
| Hybrid α=0.2 | 35.72% ± 4.55% | -1.21 pp |
| Hybrid α=0.3 | 35.02% ± 4.91% | -1.91 pp |

### Key Findings

1. **PPI-only fails badly** (16.18% R@30) — far worse than Node2Vec
2. **Hybrid methods HURT** — even α=0.1 is slightly worse (-0.05 pp)
3. **2-hop neighborhoods too large** — mean 4,828 genes per disease
4. **High false positive overlap** — unrelated diseases share many genes at 2 hops
5. **DRKG already captures PPI** — Drug-Gene and Gene-Disease edges in DRKG

### Files Created

| File | Description |
|------|-------------|
| `data/reference/ppi/disease_ppi_neighborhoods_2hop.json` | Pre-computed disease gene neighborhoods |
| `scripts/evaluate_ppi_distance_knn.py` | Evaluation script |
| `data/analysis/h17_ppi_distance_results.json` | Full results |

### Conclusion

**External PPI data does NOT improve drug repurposing predictions.**

The DRKG knowledge graph already incorporates drug-gene and gene-disease relationships that capture PPI proximity. 2-hop neighborhoods are too coarse for meaningful disease similarity.

### Session Summary (h17 + h19 Combined)

Two external data sources tested this session:
- **h19 (HPO Phenotype)**: 14.20% R@30 — INVALIDATED
- **h17 (PPI Network)**: 16.18% R@30 — INVALIDATED

Both external data sources provide **weaker signals than Node2Vec** (36.93%). The 37% ceiling is NOT due to missing external data — it's a fundamental limitation of the kNN collaborative filtering approach.

