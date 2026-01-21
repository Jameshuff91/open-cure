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

**Current instance**: None (destroyed 2026-01-20)

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

### Quick Training Results (50-100 epochs)

| Metric | Value | Notes |
|--------|-------|-------|
| Test Micro AUROC | 0.56-0.58 | Lower than published (needs more epochs) |
| Test Micro AUPRC | 0.55-0.58 | Lower than published 0.87-0.91 |
| Indication AUROC | 0.68 | Disease-centric evaluation |
| Recall@10% (indication) | ~30% | On TxGNN's own test set |

### Key Integration Challenges

1. **Ontology Mismatch**: Our ground truth uses DOID, TxGNN uses MONDO
   - Created mapping file: `data/reference/disease_ontology_mapping.json`
   - 77 diseases in our ground truth, 310 name mappings including synonyms

2. **Drug Name Differences**: TxGNN includes experimental compounds not in DrugBank
   - Top predictions include obscure compounds (Forodesine, Amylocaine)
   - Need drug name normalization for fair comparison

3. **Evaluation Mismatch**: Can't directly compare Recall@30
   - Our GB model: 13.2% Recall@30 on DOID ground truth
   - TxGNN: ~30% Recall@10% on MONDO test set (different diseases/drugs)

### Next Steps for TxGNN

1. Map DOID → MONDO for our 77 diseases
2. Retrain TxGNN with longer epochs (500+)
3. Extract predictions using proper link prediction scoring (not embedding similarity)
4. Normalize drug names to compare against our ground truth
5. Calculate true Recall@30 on our evaluation set

### TxGNN API Notes

```python
from txgnn import TxData, TxGNN

# Load data (downloads ~1.5GB on first run)
tx_data = TxData(data_folder_path='./data')
tx_data.prepare_split(split='random', seed=42)

# Initialize model
model = TxGNN(data=tx_data, device='cuda:0', weight_bias_track=False)
model.model_initialize(n_hid=100, n_inp=100, n_out=100, proto=True, proto_num=3)

# Train
model.finetune(n_epoch=100, learning_rate=5e-4)

# predict() expects DataFrame with columns: x_idx, relation, y_idx
# Use model.retrieve_embedding() for node embeddings
```
