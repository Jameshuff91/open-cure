# TxGNN Research Notes

**Date:** 2026-01-21
**Status:** Evaluation complete, future exploration opportunities identified

## Executive Summary

TxGNN (Harvard MIMS lab) is a graph neural network for drug repurposing. After proper evaluation using the model's native scoring method, it achieves **14.5% Recall@30** on our Every Cure ground truth - comparable to our simple Gradient Boosting model (13.2%). Despite sophisticated architecture, GT drugs rank near-random on average (mean rank 3473/7954).

## Key Metrics

| Metric | Value |
|--------|-------|
| Recall@30 | 14.5% (113/779 diseases) |
| Mean GT drug rank | 3473/7954 |
| Median GT drug rank | 2990 |
| Training time | ~2 hours (500 epochs on Titan Xp) |
| Model size | 4MB |

## Evaluation Journey

### Attempt 1: DistMult Embedding Similarity (WRONG)
- Extracted embeddings via `model.retrieve_embedding()`
- Computed scores: `(disease_emb * relation_emb) @ drug_emb.T`
- **Result:** ~0% Recall@30, only 1 hit
- **Problem:** Wrong scoring function - TxGNN uses a learned decoder, not DistMult

### Attempt 2: Proper model.predict() (CORRECT)
- Used TxGNN's actual `model.predict(df)` method
- Fixed disease ID mapping (.0 suffix issue)
- **Result:** 14.5% Recall@30

## Per-Disease Performance Analysis

TxGNN excels on some diseases but fails on others:

| Disease | GT Drug | Rank | Assessment |
|---------|---------|------|------------|
| Alzheimer's | Donepezil | 39 | Excellent |
| Alzheimer's | Rivastigmine | 35 | Excellent |
| Behcet's | Adalimumab | 139 | Good |
| Behcet's | Infliximab | 169 | Good |
| Addison's | Dexamethasone | 494 | Moderate |
| RA | Methotrexate | 297 | Poor |
| RA | Infliximab | 839 | Poor |
| Amyloidosis | Daratumumab | 7559 | Random |

**Pattern:** Model performs better on neurological diseases, worse on autoimmune conditions.

## Technical Lessons Learned

### 1. Scoring Method Matters
```python
# WRONG - embedding similarity
scores = (disease_emb * relation_emb) @ drug_emb.T

# CORRECT - use model's predict method
scores = model.predict(pred_df)
```

### 2. ID Format Issues
TxGNN's `tx_data.df` uses IDs like `"8383.0"` while `node.csv` has `"8383"`. Must normalize:
```python
def try_float_formats(nid):
    try:
        return [nid, f"{nid}.0", str(float(nid))]
    except:
        return [nid]  # Non-numeric like "DB00563"
```

### 3. Grouped Disease IDs
TxGNN groups multiple MONDO IDs: `"13924_12592_14672_..."`. Some diseases map to grouped IDs, others to single IDs.

### 4. Drug Overlap
- 62.6% of Every Cure drugs exist in TxGNN by name
- RA: 84.6% of GT drugs present (55/65)
- Drugs ARE there, model just doesn't rank them highly

## Future Exploration Opportunities

### 1. Ensemble TxGNN + GB Model
**Hypothesis:** Models may have complementary strengths
- TxGNN: Good at neurological diseases
- GB: May be better at autoimmune conditions
- **Action:** Create weighted ensemble, optimize weights per disease category

### 2. Fine-tune TxGNN on Every Cure Data
**Hypothesis:** Adding our GT as training signal could align predictions with clinical practice
- Add indication edges for known drug-disease pairs
- Use Every Cure labels for supervised fine-tuning
- **Risk:** Overfitting to small GT dataset

### 3. Use TxGNN Scores as GB Features
**Hypothesis:** TxGNN captures KG structure info that GB doesn't
- Extract per-drug TxGNN scores for each disease
- Add as features to GB model
- **Benefit:** Leverages both approaches without full ensemble complexity

### 4. Analyze Disease Categories
**Hypothesis:** TxGNN may systematically succeed/fail on certain disease types
- Group diseases by category (neurological, autoimmune, cancer, etc.)
- Compute per-category Recall@30
- **Action:** Route predictions to best model per category

### 5. Train Custom GNN on Aligned Data
**Hypothesis:** Ontology mismatch hurts performance
- Build custom KG from Every Cure + DrugBank + MONDO
- Train GNN with aligned drug-disease identifiers
- **Risk:** Significant engineering effort

### 6. Investigate Why Alzheimer's Works
**Hypothesis:** Understanding successes could improve failures
- What makes Alzheimer's drugs rank highly?
- Are there specific KG patterns TxGNN captures?
- Can we identify similar patterns for other diseases?

## Data Artifacts

| File | Description |
|------|-------------|
| `data/reference/txgnn_500epochs.pt` | Trained model weights (4MB) |
| `data/reference/txgnn_nodes.csv` | Node mappings (8.8MB) |
| `data/reference/txgnn_proper_scoring_results.csv` | Full evaluation results |
| `data/reference/txgnn_diseases.json` | Disease ID mappings |
| `data/reference/txgnn_drugs.json` | Drug ID mappings |

## API Reference

### Loading TxGNN
```python
from txgnn import TxData, TxGNN

tx_data = TxData(data_folder_path='./data')
tx_data.prepare_split(split='random', seed=42)

model = TxGNN(data=tx_data, device='cuda:0', weight_bias_track=False)
model.model_initialize(n_hid=100, n_inp=100, n_out=100, proto=True, proto_num=3)

# Load trained weights
state_dict = torch.load('txgnn_500epochs.pt', map_location='cuda:0')
model.model.load_state_dict(state_dict)
model.model.eval()
```

### Scoring Drug-Disease Pairs
```python
# Create prediction dataframe
pred_data = [{'x_idx': drug_idx, 'relation': 'indication', 'y_idx': disease_idx}
             for drug_idx in all_drug_indices]
pred_df = pd.DataFrame(pred_data)

# Get scores
with torch.no_grad():
    scores = model.predict(pred_df)
    indication_scores = scores[('drug', 'indication', 'disease')].cpu().numpy()
```

### Mapping Names to Indices
```python
# Load node.csv for name mappings
nodes_df = pd.read_csv('./data/node.csv', sep='\t')
disease_name_to_nodeid = {row['node_name'].lower(): row['node_id'] for ...}

# tx_data.df has internal indices
disease_xid_to_internal = {str(row['x_id']): int(row['x_idx']) for ...}

# Combined: name -> internal_idx (with .0 suffix handling)
```

## Cost Analysis

| Resource | Cost |
|----------|------|
| Vast.ai Titan Xp | ~$0.05/hr |
| Total GPU time | ~1.5 hours |
| Total cost | ~$0.08 |

## Conclusion

TxGNN is a viable drug repurposing model that achieves comparable performance to simpler approaches on our ground truth. The model has clear strengths (neurological diseases) and weaknesses (autoimmune conditions). Future work should focus on:

1. **Short-term:** Ensemble with GB model for immediate improvement
2. **Medium-term:** Use TxGNN scores as features in enhanced ML pipeline
3. **Long-term:** Train custom GNN on properly aligned data

The key insight is that neither sophisticated GNNs nor simple ML models alone achieve great performance - the path forward is combining approaches.
