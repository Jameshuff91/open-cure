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
```

**Current instance**: None (destroyed 2026-01-21)

## Models

- `models/drug_repurposing_gb.pkl` - Baseline GB model (7.0% R@30)
- `models/drug_repurposing_gb_enhanced.pkl` - Enhanced GB model (13.2% R@30)
- `models/transe.pt` - TransE knowledge graph embeddings

## Key Metrics

**Current Performance (on Every Cure Ground Truth):**

| Model | Recall@30 | Diseases Evaluated | Notes |
|-------|-----------|-------------------|-------|
| **TxGNN (proper scoring)** | **14.5%** | 779 | Using model.predict() method |
| GB Enhanced | 13.2% | 77 | Simple ML on curated features |

**TxGNN Drug Ranking Statistics:**
- Mean rank of GT drugs: 3473 (out of 7954) - near random
- Median rank: 2990
- Example: RA drugs rank #297 (methotrexate), #839 (infliximab)

**TxGNN Internal Metrics (on its own test set):**
- Test Indication AUROC: 0.787
- Test Indication AUPRC: 0.746

**Key Finding (2026-01-21):** TxGNN with proper scoring achieves 14.5% Recall@30, comparable to our simple GB model. Despite sophisticated GNN architecture, GT drugs rank near-random on average. Model excels at some diseases (Alzheimer's drugs rank ~35) but fails at others (RA drugs rank ~300-800).

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

### Recommended Next Steps

**Option 1: Ensemble Models (Recommended)**
- Combine TxGNN scores with GB model predictions
- TxGNN excels on some diseases (Alzheimer's), GB on others
- Ensemble could leverage complementary strengths
- Target: 20%+ Recall@30

**Option 2: Improve GB Model**
- Add more features from DRKG (protein targets, pathways, side effects)
- Ensemble with TransE embeddings
- Try other ML models (XGBoost, Random Forest, Neural Network)

**Option 3: Fine-tune TxGNN on Our Data**
- Use Every Cure ground truth as additional training signal
- Add indication edges for known drug-disease pairs
- Re-train with our labels to align with clinical practice

**Key Insight:** Neither sophisticated GNNs nor simple ML models alone achieve great performance. The path forward likely involves combining approaches and/or training directly on clinically-curated data.

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
