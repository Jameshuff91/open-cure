# TxGNN Integration Learnings

Deep dive into TxGNN integration experiments and lessons learned.

## Overview (2026-01-20)

**TxGNN** is a graph neural network for drug repurposing from Harvard's MIMS lab.
- Paper claims 0.87-0.91 AUPRC on their benchmark
- Uses MONDO disease ontology (our ground truth uses DOID)
- ~17,000 diseases, ~8,000 drugs in their knowledge graph

## Installation on Vast.ai GPU

```bash
# DGL version compatibility is critical
pip3 install dgl==1.1.3 -f https://data.dgl.ai/wheels/cu118/repo.html
pip3 install torch torchvision

# Clone and install TxGNN
git clone https://github.com/mims-harvard/TxGNN.git
cd TxGNN && pip3 install -e .
```

## Training Results (500 epochs, 2026-01-21)

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

## Evaluation Journey (2026-01-21)

### Initial Attempt (DistMult Scoring) - FAILED
- ~0% Recall@30
- Used embedding similarity with DistMult scoring
- Wrong scoring function - TxGNN uses a learned decoder

### Corrected Evaluation (model.predict()) - SUCCESS
- 14.5% Recall@30
- Used TxGNN's actual `model.predict()` method
- Properly matches diseases (779 evaluated) and drugs
- Comparable to GB model's 13.2%

### Evaluation Metrics

| Metric | Value |
|--------|-------|
| Diseases evaluated | 779/854 matching |
| Recall@30 | 14.5% (113 hits) |
| Mean GT drug rank | 3473/7954 (near random) |
| Median GT drug rank | 2990 |

**Per-Disease Examples:**
- Alzheimer's: Donepezil #39, Rivastigmine #35 (good)
- Behçet's: Adalimumab #139, Infliximab #169 (okay)
- Addison's: Dexamethasone #494 (moderate)
- RA: Methotrexate #297, Infliximab #839 (poor)
- Amyloidosis: Daratumumab #7559 (essentially random)

**Conclusion:** TxGNN with proper scoring achieves comparable performance to our simple GB model (14.5% vs 13.2%). Despite sophisticated GNN architecture and 500 epochs of training, GT drugs rank near-random on average. The model has signal for some diseases but not consistently. For practical drug repurposing, simpler models trained on curated data remain competitive.

## Ensemble Experiments (2026-01-21)

| Experiment | Result | Status |
|------------|--------|--------|
| Simple Ensemble (best_rank) | 7.5% R@30 | BEST |
| Category Routing (MESH) | 6.9% R@30 | Limited by 6% MESH coverage |
| Category Routing (Keywords) | TBD | 68% coverage via keyword patterns |
| TxGNN as GB Features | 4.6% R@30 | Failed (ontology mismatch) |

**Key Findings:**
- Storage diseases: **83.3% Recall@30** (enzyme replacements work!)
- Best Rank ensemble beats both models alone
- Category routing limited by only 48 MESH-mapped diseases (6% coverage)
- 65 diseases achieve excellent performance (≥50% R@30 or top-10)

### Keyword-Based Categorization (2026-01-21)
- Improved coverage from 6% (MESH) to 68% (keyword patterns)
- `src/disease_categorizer.py` - Pattern-based disease categorization
- TxGNN preferred categories (R@30 > 20%): storage, psychiatric, dermatological, autoimmune, metabolic
- Best-rank preferred (<15%): respiratory, renal, gastrointestinal, cancer, hematological
- 69 diseases routed to TxGNN, 710 to best_rank ensemble

### TxGNN Per-Category Performance (GPU evaluation)

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

### What Works Well
- Enzyme replacement therapies (laronidase rank #3, imiglucerase rank #8)
- Well-defined mechanisms (storage diseases, porphyrias)
- Diseases with clear drug targets

### What Fails
- Biologics (-mab, -cept drugs)
- Complex/heterogeneous conditions
- Diseases where GT drugs rank near-random

## Fine-Tuning Experiments (FAILED - 2026-01-21)

### Experiment 1: Standard Fine-tuning (LR=5e-4)
- Result: Catastrophic forgetting observed
- Training loss decreased but validation performance degraded

### Experiment 2: Lower Learning Rate (LR=3e-5)

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

## Alternative Paths Considered

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

## TxGNN API Notes

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

## Technical Notes

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
