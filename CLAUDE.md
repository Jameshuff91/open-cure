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

**Current Performance (GB Enhanced Model):**
- AUROC: 0.78
- AUPRC: 0.13
- Recall@30: 13.2%

**Targets:**
- AUPRC: > 0.80
- TxGNN published: 0.87-0.91 AUPRC (on their benchmark, not ours)

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

### Key Integration Challenges

1. **Ontology Mismatch**: Our ground truth uses DOID, TxGNN uses MONDO
   - Created mapping file: `data/reference/disease_ontology_mapping.json`
   - 77 diseases in our ground truth, 310 name mappings including synonyms

2. **Drug Name Differences**: TxGNN includes experimental compounds not in DrugBank
   - Top predictions include obscure compounds (Forodesine, Amylocaine)
   - Need drug name normalization for fair comparison

3. **Evaluation Mismatch**: Can't directly compare Recall@30
   - Our GB model: 13.2% Recall@30 on DRKG ground truth (MESH IDs)
   - TxGNN: 0.787 AUROC on indication task (MONDO IDs)
   - Ground truth uses `drkg:Disease::MESH:*` format
   - TxGNN uses `MONDO:*` IDs - requires cross-reference mapping to compare

### Next Steps for TxGNN

**Completed:**
- ✅ Map DOID → MONDO for 77 diseases (`data/reference/doid_to_mondo_mapping.json`)
- ✅ Train TxGNN for 500 epochs (0.787 indication AUROC)
- ✅ Extract predictions using embedding similarity

**Remaining:**
- Map DRKG ground truth (MESH IDs) to MONDO IDs for fair Recall@30 comparison
- Alternative: Build MESH → MONDO disease mapping using UMLS or BioPortal
- Consider using TxGNN's own test set metrics as the primary comparison

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

### Lessons Learned (2026-01-21)

**Dependency Hell:**
1. TxGNN requires `pandas<2.0` (uses deprecated `df.append()`)
2. Must use `numpy<2.0` for pandas 1.x compatibility
3. DGL version must match CUDA exactly: `dgl==1.1.3` for CUDA 11.8

**Critical install order:**
```bash
pip3 install "numpy<2.0" "pandas<2.0"
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip3 install dgl==1.1.3 -f https://data.dgl.ai/wheels/cu118/repo.html
pip3 install -e ./TxGNN
```

**API Gotchas:**
- `model.finetune()` does NOT accept `batch_size` parameter
- `tx_data.retrieve_id_mapping()` may return empty dicts - use `tx_data.df` instead
- `tx_data.disease_list` attribute doesn't exist - use `tx_data.G.num_nodes('disease')`

**SSH to Vast.ai:**
- Exit code 255 is often false negative - check if command actually ran
- The "Welcome to vast.ai" banner goes to stderr, can confuse exit codes
- Use `nohup` for long training runs, output may be buffered
