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

## Exploration Experiments (2026-01-21)

### Experiment 1: Disease Category Analysis ✅
**Status:** COMPLETED
**Result:** Strong category-specific patterns found

| Category | TxGNN R@30 | Assessment |
|----------|------------|------------|
| Metabolic/Storage | **66.7%** | Excellent |
| Psychiatric | 28.6% | Good |
| Dermatological | 25.0% | Good |
| Autoimmune | 22.2% | Good |
| Metabolic | 21.7% | Good |
| Neurological | 15.5% | Average |
| Cancer | 15.0% | Average |
| Gastrointestinal | 8.7% | Poor |
| Respiratory | 7.3% | Poor |
| Renal | 7.1% | Poor |

**GB Model Strengths:** Infectious (50%), Cardiovascular (72% for atrial fib)

**Routing Recommendation:**
- Use TxGNN for: Metabolic, dermatological, autoimmune
- Use GB for: Infectious, cardiovascular, respiratory

### Experiment 2: Alzheimer's Success Patterns ✅
**Status:** COMPLETED
**Result:** Drug class determines success

**What Works Well:**
- Tetracyclines (doxycycline, minocycline): 17 good rankings
- Enzyme replacements (laronidase): Rank #3
- Cholinesterase inhibitors (rivastigmine, donepezil): Rank ~35

**What Fails:**
- Biologics (-mab, -cept drugs): Consistently poor
- Monoclonal antibodies: TxGNN can't model them well

**Actionable Pattern:** More GT drugs → better performance
- Excellent performers: avg 7.5 GT drugs
- Poor performers: avg 2.9 GT drugs

### Experiment 3: Simple Ensemble ✅
**Status:** COMPLETED
**Result:** "Best Rank" ensemble beats both models

| Strategy | R@30 | vs GB alone |
|----------|------|-------------|
| GB only | 14.6% | baseline |
| TxGNN only | 2.1% | -12.5% |
| Average | 0.0% | -14.6% |
| Weighted | 0-2.1% | worse |
| **Best Rank** | **16.7%** | **+2.1%** |
| **RRF** | **16.7%** | **+2.1%** |

**Why Average Failed:** TxGNN ranks in thousands drag down combined scores.
**Why Best Rank Works:** Takes minimum rank, capturing each model's strengths.

**Unique Contributions:**
- TxGNN found: dantrolene for MS (rank 1 vs GB rank 219)
- GB found: lisinopril, pergolide, pramipexole, insulin human

### Experiment 4: TxGNN as GB Features ✅
**Status:** COMPLETED
**Result:** FAILED - ontology mismatch too severe

| Model | R@30 | Change |
|-------|------|--------|
| GB Enhanced | 13.2% | baseline |
| GB + TxGNN Features | 4.6% | **-64.9%** |

**Why It Failed:**
- Only 0.4% of training pairs had TxGNN coverage
- TxGNN predicts experimental compounds, not approved drugs
- Feature importance: TxGNN contributed only 0.016%

**Conclusion:** Don't use TxGNN scores as features - ontology gap too large.

## Experiment Summary

| Experiment | Hypothesis | Result | Actionable? |
|------------|------------|--------|-------------|
| Disease Categories | Route by type | ✅ Strong patterns | YES |
| Alzheimer's Patterns | Drug class matters | ✅ Confirmed | YES |
| Simple Ensemble | Best rank wins | ✅ 16.7% R@30 | YES |
| TxGNN as Features | Add to GB | ❌ Failed | NO |
| Category Routing | Route by category | ⚠️ No improvement | LIMITED |

### Experiment 5: Category-Based Routing Ensemble ✅
**Status:** COMPLETED
**Result:** No improvement over best_rank due to limited MESH coverage

| Model | Hits@30 | Total | Recall@30 |
|-------|---------|-------|-----------|
| TxGNN only | 101 | 1501 | 6.7% |
| GB only | 7 | 48 | 14.6% |
| Best Rank | 108 | 1501 | 7.2% |
| Category Routing | 104 | 1501 | 6.9% |

**Why It Didn't Help:**
- 85% of drugs (1276/1501) fell back to best_rank
- Only 21 diseases have MESH mappings for GB model
- Can't leverage GB's strengths without broader disease coverage

**Bright Spot - Storage Diseases: 83.3% Recall@30!**
- Fabry: migalastat rank 26, agalsidase beta rank 2970
- Gaucher: velaglucerase alfa rank 8, imiglucerase rank 13
- Hurler/Scheie: laronidase rank 3-6
- Laron: mecasermin rank 22

**Why Storage Diseases Work:**
1. Enzyme replacement therapies have clear mechanisms
2. TxGNN's KG captures enzyme-drug relationships well
3. Small drug space (fewer competitors for top ranks)
4. Strong literature signal in knowledge graph

## Next Steps (Prioritized)

### 1. Expand MESH Mappings (LOCAL - NEXT)
**Priority:** HIGH
**Effort:** Medium
Map more TxGNN diseases to MESH IDs using disease ontology crosswalks.
**Goal:** Enable GB model on 100+ diseases instead of 21

### 2. Fine-tune TxGNN on Every Cure (GPU NEEDED)
**Priority:** MEDIUM
**Effort:** Medium
Add Every Cure indication edges to TxGNN training.
**Risk:** Overfitting to small dataset

### 3. Confidence-Based Model Selection (LOCAL)
**Priority:** MEDIUM
**Effort:** Medium
Train a meta-model to predict which model will perform better for each disease.
Features: disease category, drug count, literature mentions

### 4. Train Custom GNN (GPU NEEDED)
**Priority:** LOW
**Effort:** HIGH
Build aligned KG from scratch.
**Skip for now:** Too much work for uncertain gain

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
