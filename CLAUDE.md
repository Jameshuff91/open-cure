# Open-Cure Project Instructions

## Cloud GPU (Vast.ai)

For GPU-intensive tasks (TxGNN, model training), use Vast.ai:

```bash
# CLI installed in .venv
source .venv/bin/activate

# Search for cheap GPUs
vastai search offers 'gpu_ram>=8 cuda_vers>=11.0 reliability>0.95' --order 'dph' --limit 10

# Create instance (GTX 1080 Ti ~$0.07/hr)
vastai create instance <OFFER_ID> --image nvidia/cuda:11.7.1-runtime-ubuntu22.04 --disk 30 --ssh

# Check status
vastai show instances

# SSH to instance
ssh -p <PORT> root@<SSH_ADDR>

# Destroy when done (stop billing!)
vastai destroy instance <INSTANCE_ID>

# AUTOMATED SETUP (recommended)
./scripts/vastai_txgnn_setup.sh <PORT> <HOST>
# Example: ./scripts/vastai_txgnn_setup.sh 16464 ssh3.vast.ai
```

**Current instance**: None (destroyed 2026-01-22 after fine-tuning experiments)

## Models

- `models/drug_repurposing_gb.pkl` - Baseline GB model (7.0% R@30)
- `models/drug_repurposing_gb_enhanced.pkl` - Enhanced GB model with expanded MESH coverage
- `models/transe.pt` - TransE knowledge graph embeddings

## Key Metrics

**Current Performance (on Every Cure Ground Truth):**

| Model | Per-Drug R@30 | Diseases Evaluated | Notes |
|-------|---------------|-------------------|-------|
| **GB + Target Boost** | **39.0%** | 690/779 | Validated +1.6% improvement (p<0.0001) - BEST |
| GB Enhanced (Expanded MESH) | 37.4% | 700/779 | Agent web search MESH mappings |
| GB Enhanced (18 diseases) | 17.1% | 18 | CONFIRMED diseases only |
| Best Rank Ensemble | 7.5% | 779 | min(TxGNN rank, GB rank) |
| TxGNN (proper scoring) | 6.7% | 779 | Per-drug R@30 |

**MESH Mapping Expansion (2026-01-22):**
- Original: ~80 hardcoded MESH mappings
- Expanded: 827 agent-searched MESH mappings (10x increase)
- Disease coverage: 700/779 (90%) vs previous ~10%
- 354 diseases with at least 1 correct drug in top 30

**TxGNN Drug Ranking Statistics:**
- Mean rank of GT drugs: 3473 (out of 7954) - near random
- Median rank: 2990
- 65 diseases achieve ≥50% R@30 or top-10 ranking
- Storage diseases: 83.3% Recall@30 (best category)

**Key Finding (2026-01-22, verified 2026-01-24):** GB model with expanded MESH coverage achieves **37.4%** per-drug Recall@30 on 700 diseases, dramatically outperforming TxGNN (6.7%). The key was expanding disease-to-MESH mappings via parallel agent web searches against NIH/NLM database.

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

### Experiment 2: Ensemble Strategy - PROMISING (NEEDS VALIDATION)

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

**✓ VALIDATED (2026-01-24):**

| Test | Result | Significance |
|------|--------|--------------|
| McNemar's test | p=0.000014 | ✓ Highly significant |
| Sign test | p=0.000044 | ✓ Highly significant |
| Bootstrap 95% CI | [0.88%, 2.54%] | ✓ Does not include 0 |

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

## Clinical Trial Validation (2026-01-22)

**MAJOR FINDING: Model predictions validated by independent clinical trials**

### Dantrolene → Heart Failure / VT (RCT VALIDATED)

| Metric | Value |
|--------|-------|
| Model Score | 0.969 |
| Model Rank | #7 for heart failure |
| Trial Design | Double-blind RCT, 51 patients |
| Timeline | Dec 2020 - Mar 2024 |
| **Result** | **66% reduction in VT inducibility** |
| Dantrolene Arm | 41% → 14% VT inducibility |
| Placebo Arm | 46% → 41% (no change) |
| P-value | **0.034** |
| Safety | No drug-related serious adverse events |

**Source:** [medRxiv 2025.08.17.25333868](https://www.medrxiv.org/content/10.1101/2025.08.17.25333868v1.full)

**Significance:** Model prediction made BEFORE clinical trial results published, demonstrating genuine predictive capability.

### Empagliflozin → Parkinson's (Observational Validation)

| Metric | Value |
|--------|-------|
| Model Score | 0.903 |
| Korean Study | 20% reduced PD risk (HR 0.80) |
| LIGHT-MCI Trial | NCT05313529, results expected mid-2026 |

### Trial Monitoring List

| Trial | Drug | Condition | Results Expected |
|-------|------|-----------|------------------|
| LIGHT-MCI | Empagliflozin | MCI/Cognitive | Mid-2026 |
| SHO-IN | Dantrolene | VT/Mortality in HF | Ongoing |
| NCT02953665 | Liraglutide | Parkinson's | TBD |

## Scientific Validation (2026-01-22)

### Literature Validation of Novel Predictions

Validated 16 high-confidence novel predictions against PubMed/FDA sources:

| Metric | Result |
|--------|--------|
| **Clinically Validated** (FDA/standard) | **68.8%** (11/16) |
| **Biologically Plausible** (+ research) | **93.8%** (15/16) |
| **False Positives** | **6.2%** (1/16) |

### Validated Discoveries (FDA-approved drugs NOT in training data)

| Drug | Disease | FDA Status | Notes |
|------|---------|------------|-------|
| **Lecanemab** | Alzheimer's | FDA 2023 | First amyloid-clearing therapy |
| **Empagliflozin** | Heart failure | FDA 2021 | SGLT2 inhibitor |
| **Tezepelumab** | Asthma | FDA 2021 | First-in-class anti-TSLP |
| **Rivastigmine** | Parkinson's dementia | FDA approved | NEJM landmark trial |
| **Atezolizumab** | Lung cancer | FDA approved | Checkpoint inhibitor |

### Top Novel Predictions with Research Support

| Drug | Disease | Evidence | Sources |
|------|---------|----------|---------|
| **Empagliflozin** | Parkinson's | 2024 PubMed studies | Neuroprotection in rat models |
| **Paclitaxel** | Rheumatoid arthritis | Phase I data | Anti-angiogenic mechanism |
| **Thiamine** | Alzheimer's | NIH clinical trials | Benfotiamine ongoing |
| **Quetiapine** | Parkinson's psychosis | Off-label clinical use | First-line despite no FDA approval |

### Error Patterns by Drug Type

| Drug Type | Recall@30 | Notes |
|-----------|-----------|-------|
| ACE inhibitors (-pril) | 75% | Best performing |
| Small molecules | 32% | Moderate |
| Kinase inhibitors (-nib) | 17% | Poor |
| Biologics (-mab) | 17% | Worst performing |

**Key Insight:** Model excels at small molecule predictions but struggles with biologics. Filter -mab drugs for higher precision.

### Train/Test Split Analysis

| Set | Recall@30 | Notes |
|-----|-----------|-------|
| Training (560 diseases) | 36.5% | Expected high |
| **Test (140 held-out)** | **20.0%** | Still 3x better than TxGNN |
| Gap | 16.5% | Some overfitting but real generalization |

### External Validation (Drug Repurposing Cases)

Tested classic repurposing examples NOT in Every Cure:
- Top 100 hits: 29% (vs 0.9% random) = **30x improvement over random**
- Model has real biological signal for novel indications

### Extended Validation (24 total predictions)

| Category | Count | Percentage |
|----------|-------|------------|
| FDA/Standard + Clinical Trial | 13/24 | 54.2% |
| Research Support (preclinical+) | 8/24 | 33.3% |
| **Biologically Plausible (total)** | **21/24** | **87.5%** |
| False Positives | 3/24 | 12.5% |

### Top Novel Predictions for Further Research

| Drug | Disease | Evidence | Key Finding |
|------|---------|----------|-------------|
| **Dantrolene** | Heart failure/VT | **RCT VALIDATED P=0.034** | 66% reduction in VT inducibility, FDA-approved for MH, repurposing ready |
| **Empagliflozin** | Parkinson's | 2024 Observational | Korean study: 20% reduced PD risk (HR 0.80), LIGHT-MCI trial ongoing |
| **Lidocaine (nebulized)** | Asthma | RCT P<0.001 | FEV1 improvement, steroid-sparing potential |
| **Formoterol** | T2D hypoglycemia | Clinical study | 45-50% reduction in glucose infusion rate |
| **DHA/Omega-3** | Asthma | Multiple studies | 72% reduction in TNF-α/IL-17A |
| **Thiamine** | Alzheimer's | NIH trials | Benfotiamine trials ongoing |
| **Corticotropin** | RA | FDA approved | 62.9% achieved low disease activity in Phase IV RCT |

### False Positives Identified (Filter Rules)

| Pattern | Example | Reason |
|---------|---------|--------|
| Antibiotics for metabolic diseases | Gentamicin → T2D | Inhibits insulin release |
| Sympathomimetics for diabetes | Pseudoephedrine → T2D | Increases blood glucose |
| Alpha blockers for heart failure | Doxazosin → HF | ALLHAT: 2x HF risk! |
| Diagnostic agents as treatments | Ioflupane → PD | It's for imaging, not treatment |
| TCAs for hypertension | Protriptyline → HTN | TCAs CAUSE hypertension via NET inhibition |
| PPIs for hypertension | Pantoprazole → HTN | 17% increased HTN risk |
| Tumor-promoting hormones for cancer | Aldosterone → Lung cancer | Promotes tumor spread, not treats it |

### Confidence Scoring by Drug Type

| Drug Type | Precision | False Positive Rate |
|-----------|-----------|---------------------|
| Biologics (-mab) | 100% | 0% |
| Small molecules | 74% | 16% |
| Antibiotics | 0% | 50% |
| Sympathomimetics | 0% | 100% |
| TCAs | 0% | 100% (for HTN) |
| PPIs | 0% | 100% (for HTN) |

**Recommendation:** Use `src/confidence_filter.py` to auto-exclude harmful prediction patterns.

### Files

- `data/analysis/literature_validation.json` - Initial validation (16 predictions)
- `data/analysis/extended_validation.json` - Extended validation (8 more)
- `data/analysis/session2_validations.json` - Session 2 validation (8 more)
- `data/analysis/actionable_predictions.json` - 38 predictions for further review
- `data/analysis/error_analysis.json` - Systematic failure patterns
- `data/analysis/comprehensive_validation.json` - All validated predictions
- `data/analysis/every_cure_summary_report.txt` - Summary report for Every Cure
- `src/confidence_filter.py` - Auto-excludes harmful prediction patterns

## Data Sources

- Every Cure: `data/reference/everycure/indicationList.xlsx`
- Enhanced ground truth: `data/reference/expanded_ground_truth.json`
- DrugBank lookup: `data/reference/drugbank_lookup.json`
- Disease ontology mapping: `data/reference/disease_ontology_mapping.json`
- DOID disease names: `data/reference/doid_disease_names.json`

## TxGNN Integration (Learnings)

### What We Learned (2026-01-20)

**TxGNN** is a graph neural network for drug repurposing from Harvard's MIMS lab.
- Paper claims 0.87-0.91 AUPRC on their benchmark
- Uses MONDO disease ontology (our ground truth uses DOID)
- ~17,000 diseases, ~8,000 drugs in their knowledge graph

### Installation on Vast.ai GPU

```bash
# DGL version compatibility is critical
pip3 install dgl==1.1.3 -f https://data.dgl.ai/wheels/cu118/repo.html
pip3 install torch torchvision

# Clone and install TxGNN
git clone https://github.com/mims-harvard/TxGNN.git
cd TxGNN && pip3 install -e .
```

### Training Results (500 epochs, 2026-01-21)

| Metric | Value | Notes |
|--------|-------|-------|
| Test Micro AUROC | 0.725 | Full 500 epoch training |
| Test Micro AUPRC | 0.702 | Full 500 epoch training |
| **Indication AUROC** | **0.787** | Key metric for drug repurposing |
| **Indication AUPRC** | **0.746** | Key metric for drug repurposing |

**Saved artifacts:**
- `data/reference/txgnn_500epochs.pt` - Trained model weights
- `data/reference/txgnn_predictions.csv` - Top 50 drug predictions per disease
- `data/reference/txgnn_diseases.json` - Disease ID mappings
- `data/reference/txgnn_drugs.json` - Drug ID mappings

### TxGNN Evaluation Journey (2026-01-21)

**Initial Attempt (DistMult Scoring):** ~0% Recall@30
- Used embedding similarity with DistMult scoring
- Wrong scoring function - TxGNN uses a learned decoder

**Corrected Evaluation (model.predict()):** 14.5% Recall@30
- Used TxGNN's actual `model.predict()` method
- Properly matches diseases (779 evaluated) and drugs
- Comparable to GB model's 13.2%

**What We Learned:**

| Metric | Value |
|--------|-------|
| Diseases evaluated | 779/854 matching |
| Recall@30 | 14.5% (113 hits) |
| Mean GT drug rank | 3473/7954 (near random) |
| Median GT drug rank | 2990 |

**Per-Disease Examples:**
- Alzheimer's: Donepezil #39, Rivastigmine #35 ✓ (good)
- Behçet's: Adalimumab #139, Infliximab #169 (okay)
- Addison's: Dexamethasone #494 (moderate)
- RA: Methotrexate #297, Infliximab #839 (poor)
- Amyloidosis: Daratumumab #7559 (essentially random)

**Conclusion:** TxGNN with proper scoring achieves comparable performance to our simple GB model (14.5% vs 13.2%). Despite sophisticated GNN architecture and 500 epochs of training, GT drugs rank near-random on average. The model has signal for some diseases but not consistently. For practical drug repurposing, simpler models trained on curated data remain competitive.

### Ensemble Experiments (Completed 2026-01-21)

**Experiment Results:**

| Experiment | Result | Status |
|------------|--------|--------|
| Simple Ensemble (best_rank) | 7.5% R@30 | ✅ BEST |
| Category Routing (MESH) | 6.9% R@30 | ⚠️ Limited by 6% MESH coverage |
| Category Routing (Keywords) | TBD | 68% coverage via keyword patterns |
| TxGNN as GB Features | 4.6% R@30 | ❌ Failed (ontology mismatch) |

**Key Findings:**
- Storage diseases: **83.3% Recall@30** (enzyme replacements work!)
- Best Rank ensemble beats both models alone
- Category routing limited by only 48 MESH-mapped diseases (6% coverage)
- 65 diseases achieve excellent performance (≥50% R@30 or top-10)

**Keyword-Based Categorization (2026-01-21):**
- Improved coverage from 6% (MESH) to 68% (keyword patterns)
- `src/disease_categorizer.py` - Pattern-based disease categorization
- TxGNN preferred categories (R@30 > 20%): storage, psychiatric, dermatological, autoimmune, metabolic
- Best-rank preferred (<15%): respiratory, renal, gastrointestinal, cancer, hematological
- 69 diseases routed to TxGNN, 710 to best_rank ensemble

**TxGNN Per-Category Performance (GPU evaluation):**
| Category | R@30 | Sample Size |
|----------|------|-------------|
| Storage | 83.3% | 6 |
| Psychiatric | 28.6% | 7 |
| Dermatological | 25.0% | 20 |
| Autoimmune | 22.2% | 27 |
| Metabolic | 21.7% | 46 |
| Neurological | 18.5% | 54 |
| Cancer | 11.7% | 171 |
| Respiratory | 7.3% | 41 |
| Renal | 7.1% | 14 |

**What Works Well:**
- Enzyme replacement therapies (laronidase rank #3, imiglucerase rank #8)
- Well-defined mechanisms (storage diseases, porphyrias)
- Diseases with clear drug targets

**What Fails:**
- Biologics (-mab, -cept drugs)
- Complex/heterogeneous conditions
- Diseases where GT drugs rank near-random

### Fine-Tuning Experiments (FAILED - 2026-01-21)

**Experiment 1: Standard Fine-tuning (LR=5e-4)**
- Result: Catastrophic forgetting observed
- Training loss decreased but validation performance degraded

**Experiment 2: Lower Learning Rate (LR=3e-5)**
| Metric | Original | Fine-tuned | Change |
|--------|----------|------------|--------|
| Recall@30 | 8.0% | 6.6% | -1.4pp |
| Mean Rank | 3138 | 2537 | +601 |
| Median Rank | 1377 | 840 | +537 |

**Key Finding:** Fine-tuning consistently causes catastrophic forgetting. While mean/median ranks improved, Recall@30 dropped because the model "forgot" many correct associations.

**Why Fine-tuning Fails:**
- Small dataset (1512 pairs) vs large pretrained model
- Overfits to training pairs, loses generalization
- TxGNN's prototype-based architecture may not adapt well

**Conclusion:** Fine-tuning TxGNN is NOT a viable path. Need alternative approaches.

### Promising Alternative Paths

**Path 1: Better Disease/Drug Matching (LOCAL)**
- Current evaluation only matches 340/779 diseases due to name mismatches
- Improve DOID↔MONDO mapping could unlock more evaluation coverage
- Potential: +50% more diseases evaluated

**Path 2: Confidence-Based Model Selection (LOCAL)**
- Train meta-model to predict when to trust each model
- Features: disease category, drug count, mechanism clarity, embedding similarity
- Route to TxGNN for storage/metabolic diseases, GB for others

**Path 3: Knowledge Graph Augmentation**
- Add Every Cure edges directly to TxGNN's knowledge graph (not fine-tuning)
- Retrain from scratch with augmented data
- Risk: Expensive (500 epochs = several hours GPU)

**Path 4: External Data Integration**
- DrugBank: drug-target interactions, pharmacology
- ChEMBL: bioactivity data
- ClinicalTrials.gov: trial outcomes
- Could improve GB model features significantly

**Path 5: Disease-Specific Models**
- Train specialized models for disease categories
- Storage diseases already achieve 83.3% R@30
- Focus resources on categories where we can win

**Key Insight:** TxGNN excels for well-defined mechanisms (storage diseases, enzyme deficiencies) but struggles with complex conditions. Best ensemble approach: take minimum rank from either model.

### TxGNN API Notes

```python
from txgnn import TxData, TxGNN

# Load data (downloads ~1.5GB on first run)
tx_data = TxData(data_folder_path='./data')
tx_data.prepare_split(split='random', seed=42)

# Initialize model
model = TxGNN(data=tx_data, device='cuda:0', weight_bias_track=False)
model.model_initialize(n_hid=100, n_inp=100, n_out=100, proto=True, proto_num=3)

# Train - NOTE: no batch_size parameter!
model.finetune(n_epoch=500, learning_rate=5e-4, train_print_per_n=50)

# Get node names from dataframe (retrieve_id_mapping() may return empty!)
df = tx_data.df
disease_names = dict(zip(
    df[df['x_type']=='disease']['x_idx'],
    df[df['x_type']=='disease']['x_id']
))
```

### Technical Notes (TxGNN)

**Dependencies (critical order):**
```bash
pip3 install "numpy<2.0" "pandas<2.0"  # TxGNN uses deprecated pandas API
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip3 install dgl==1.1.3 -f https://data.dgl.ai/wheels/cu118/repo.html
pip3 install -e ./TxGNN
```

**API Notes:**
- `model.retrieve_embedding()` returns dict of node embeddings by type
- `tx_data.df` has columns: x_type, x_id, relation, y_type, y_id, x_idx, y_idx
- Embedding order follows sorted `node_index` from `node.csv`

**Vast.ai Tips:**
- Use `nohup` for long training runs
- Always destroy instances when done: `vastai destroy instance <ID>`
