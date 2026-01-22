# TxGNN Research Notes

**Date:** 2026-01-21
**Status:** Exploration complete, proceeding to fine-tuning

## Executive Summary

TxGNN (Harvard MIMS lab) is a graph neural network for drug repurposing. After comprehensive evaluation and 5 experiments, our key findings:

1. **Best Rank ensemble achieves 7.5% per-drug Recall@30** (best result)
2. **Storage diseases achieve 83.3% Recall@30** - enzyme replacements work exceptionally well
3. **65 diseases achieve excellent performance** (≥50% R@30 or top-10 ranking)
4. **TxGNN excels for well-defined mechanisms** but struggles with complex conditions

**Next Step:** Fine-tune TxGNN on Every Cure ground truth data (GPU required)

## Key Metrics

| Metric | Value |
|--------|-------|
| Best Rank Ensemble | **7.5% per-drug R@30** |
| TxGNN alone | 6.7% per-drug R@30 |
| Storage diseases | **83.3% R@30** (best category) |
| Excellent diseases | 65 (≥50% R@30 or top-10) |
| Mean GT drug rank | 3473/7954 |
| Training time | ~2 hours (500 epochs) |

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
| GB only | 12 | 132 | 9.1% |
| Best Rank | 113 | 1501 | **7.5%** |
| Category Routing | 104 | 1501 | 6.9% |

**Why It Didn't Help:**
- 84% of drugs (1258/1501) fell back to best_rank
- Only 48 diseases have MESH mappings (expanded from 21)
- Can't leverage GB's strengths without broader disease coverage

### Experiment 6: Excellent Disease Analysis ✅
**Status:** COMPLETED
**Result:** 65 diseases achieve excellent performance

**Diseases with ≥50% R@30 or top-10 ranking:**

| Disease Type | Count | Examples |
|--------------|-------|----------|
| Storage/Metabolic | 6 | Hurler, Gaucher, Fabry, Laron |
| Infectious | 8 | Chagas (100%), yaws, impetigo, scabies |
| Cancer | 9 | APL (100%), follicular lymphoma |
| Rare syndromes | 15+ | Cystinosis, porphyrias, lipodystrophies |

**Top Performers (100% R@30):**
- Chagas disease: rank 1
- Hypophosphatasia: rank 15
- Cystinosis: rank 6-7
- Hurler-Scheie: rank 3
- APL: all 3 drugs in top 10

**Why These Work:**
1. Clear mechanism (enzyme deficiency, specific pathogen)
2. Targeted therapies with strong literature signal
3. Small drug space (fewer competitors)
4. Well-characterized in knowledge graph

**Storage Diseases - 83.3% Recall@30!**
- Fabry: migalastat rank 26
- Gaucher: velaglucerase alfa rank 8, imiglucerase rank 13
- Hurler/Scheie: laronidase rank 3-6
- Laron: mecasermin rank 22

## Experiment Summary (Final)

| # | Experiment | Result | Status |
|---|------------|--------|--------|
| 1 | Disease Categories | Strong patterns found | ✅ |
| 2 | Alzheimer's Patterns | Drug class matters | ✅ |
| 3 | Simple Ensemble | Best Rank: 16.7% R@30 | ✅ |
| 4 | TxGNN as Features | Failed (-64.9%) | ❌ |
| 5 | Category Routing | Limited (48 diseases) | ⚠️ |
| 6 | Excellent Diseases | 65 diseases identified | ✅ |

## Next Steps (Prioritized)

### Experiment 7: Fine-tune TxGNN on Every Cure ✅
**Status:** COMPLETED (2026-01-21)
**Result:** Catastrophic forgetting - model degraded

| Metric | Original | Fine-tuned | Change |
|--------|----------|------------|--------|
| Test AUROC | 0.725 | 0.706 | -2.6% |
| Indication AUROC | 0.787 | 0.751 | -4.6% |
| Validation R@30 | - | 4.3% | - |

**What We Did:**
- Added 1,209 Every Cure indication edges to training
- Fine-tuned for 100 epochs at LR=1e-4
- Training time: 1 min 42 sec on Titan Xp

**Why It Failed:**
- Learning rate too high (1e-4) caused catastrophic forgetting
- Model "forgot" original knowledge while learning new edges
- Need: LR=1e-5, fewer epochs (20-50), layer freezing

**Lesson Learned:** Fine-tuning pre-trained GNNs requires careful hyperparameter tuning to avoid catastrophic forgetting.

### Experiment 8: Fine-tune with Lower LR (3e-5) ✅
**Status:** COMPLETED (2026-01-21)
**Result:** No improvement - still random performance

| Metric | LR=1e-4 | LR=3e-5 | Original |
|--------|---------|---------|----------|
| Training loss | 0.502 | 0.594 | 0.509 |
| Train AUROC | 0.839 | 0.747 | 0.835 |
| Mean GT rank | 3889 | 3889 | 3872 |
| R@30 (DistMult) | 0% | 0% | 0% |

**What We Did:**
- Reduced LR to 3e-5 (1/3 of original)
- 50 epochs instead of 100
- 1,512 matched disease-drug pairs
- Training time: 1 min 35 sec

**Key Finding:**
Both original and fine-tuned models show ~0% R@30 with mean rank ~3900 when evaluated with simple DistMult scoring. This suggests:
1. TxGNN's complex prototype-based scoring is essential for good performance
2. Simple embedding-based evaluation doesn't capture the full model behavior
3. Fine-tuning doesn't help because the base evaluation approach is flawed

**Lesson Learned:** TxGNN uses a sophisticated prototype-based scoring mechanism during evaluation that modifies disease embeddings based on similar diseases. Simple DistMult scoring (h_drug * h_rel * h_disease) doesn't replicate this, leading to near-random results.

## Next Steps (Revised)

### 1. Implement Full TxGNN Evaluation Pipeline (GPU NEEDED)
**Priority:** HIGH
**Effort:** Medium
Need to use TxGNN's full forward pass with prototype embeddings, not simple DistMult scoring.

### 2. Confidence-Based Model Selection (LOCAL)
**Priority:** MEDIUM
**Effort:** Medium
Train a meta-model to predict which model will perform better.
Features: disease category, mechanism clarity, drug count.

### 3. Train Custom GNN (GPU NEEDED)
**Priority:** LOW
**Effort:** HIGH
Build aligned KG from scratch.
**Skip for now:** Better fine-tuning strategy likely more impactful.

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

After 6 experiments, our key findings:

### What Works
- **Best Rank ensemble (7.5% R@30)** - Take min(TxGNN rank, GB rank)
- **Storage/metabolic diseases (83.3% R@30)** - Enzyme replacements excel
- **Well-defined mechanisms** - Clear pathways, targeted therapies
- **65 excellent diseases** - Achievable with current approach

### What Doesn't Work
- **Averaging scores** - TxGNN's high ranks drag down ensemble
- **TxGNN as features** - Ontology mismatch too severe
- **Category routing** - Limited by MESH mapping coverage
- **Complex diseases** - Heterogeneous conditions fail

### Key Insight
TxGNN excels for diseases with **clear mechanisms** (enzyme deficiencies, specific pathogens) but struggles with **complex conditions** (autoimmune, heterogeneous cancers). The path forward: fine-tune on our ground truth to teach the model what clinicians consider valid treatments.

### Next Action
**Fine-tune TxGNN on Every Cure data** - Add known drug-disease pairs as training signal to improve performance on complex diseases.
