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
| **GB Enhanced** | **13.2%** | 77 | Best performer on our ground truth |
| TxGNN (500 epochs) | ~0% | 1,194 | Ontology mismatch - only 1 hit |

**TxGNN Internal Metrics (on its own test set):**
- Test Indication AUROC: 0.787
- Test Indication AUPRC: 0.746
- Published benchmark: 0.87-0.91 AUPRC

**Key Finding:** TxGNN achieves good metrics on its own benchmark but fails on our ground truth due to fundamental ontology/data incompatibility.

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

### Why TxGNN Failed on Our Ground Truth (2026-01-21)

**Root Cause: Fundamental Ontology Mismatch**

| Aspect | Every Cure (Our GT) | TxGNN |
|--------|---------------------|-------|
| Drug IDs | CHEBI | DrugBank |
| Disease IDs | MONDO | MONDO (grouped) |
| Drug types | Approved treatments | Experimental + approved |
| Drug overlap | ~1,500 by name match | 7,957 total |
| Disease overlap | ~1,200 by name match | 17,080 total |

**What We Tried:**
1. Trained TxGNN for 500 epochs → 0.787 AUROC on its test set ✓
2. Extracted embeddings using `model.retrieve_embedding()` ✓
3. Computed DistMult scores for drug-disease pairs ✓
4. Matched diseases by exact name (1,194 matches) ✓
5. Matched drugs by name (CHEBI→DrugBank via 1,500 name mappings) ✓

**Result:** Only **1 drug hit** (Diflunisal for Rheumatoid Arthritis) across 1,194 disease evaluations.

**Why This Happened:**
- TxGNN predicts experimental/investigational compounds (e.g., "4-methyl-umbelliferyl-N-acetyl-chitobiose")
- Every Cure ground truth has standard-of-care approved drugs (e.g., Methotrexate, Infliximab)
- Even when disease names match, the predicted drugs don't overlap with approved treatments
- TxGNN's knowledge graph encodes different drug-disease relationships than clinical practice

**Conclusion:** SOTA graph neural networks trained on biomedical KGs don't necessarily predict clinically-relevant drug repurposing candidates. Simple ML models (GB) trained on curated ground truth outperform them for practical applications.

### Recommended Next Steps

**Option 1: Improve GB Model (Recommended)**
- Add more features from DRKG (protein targets, pathways, side effects)
- Ensemble with TransE embeddings
- Try other ML models (XGBoost, Random Forest, Neural Network)
- Target: 20%+ Recall@30

**Option 2: Train on Our Ground Truth Directly**
- Build a custom KG from Every Cure + DrugBank + disease ontologies
- Train a GNN on this aligned data
- More work but ensures ontology compatibility

**Option 3: Different SOTA Models**
- Try DRKG's own pretrained embeddings (already aligned with our ground truth format)
- Explore other drug repurposing models: DTINet, DeepDTA, GraphDTA
- Look for models trained on clinical/approved drug data

**Not Recommended:**
- Further TxGNN integration (ontology gap too large)
- Building complex CHEBI↔DrugBank↔MONDO mappings (diminishing returns)

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
