# Open-Cure Drug Repurposing Research Specification

## Project Goal
Improve drug repurposing predictions using the DRKG knowledge graph and machine learning models.

## Current Baseline
- **Model:** Gradient Boosting + Fuzzy Disease Matcher
- **Performance:** 41.8% Recall@30 (per-drug)
- **Evaluation:** 1,236 diseases, 3,618 drug-disease pairs

## Key Files & Scripts

### Models
- `models/drug_repurposing_gb_enhanced.pkl` - Main GB model
- `models/transe.pt` - TransE embeddings
- `models/confidence_calibrator.pkl` - Confidence predictor

### Evaluation Scripts
- `scripts/evaluate_pathway_boost.py` - Main evaluation with Quad Boost
- `src/disease_name_matcher.py` - Fuzzy disease name matching
- `src/external_validation.py` - Clinical trials & PubMed validation
- `src/confounding_detector.py` - Detects false positive patterns

### Data
- `data/reference/everycure/indicationList.xlsx` - Ground truth
- `data/reference/disease_ontology_mapping.json` - DRKG disease mappings
- `data/reference/expanded_ground_truth.json` - Enhanced ground truth
- `data/reference/mondo_to_mesh.json` - MONDO→MESH ID mapping

## Known Performance Patterns

### What Works Well
| Category | Recall@30 |
|----------|-----------|
| ACE inhibitors | 66.7% |
| Autoimmune diseases | 63.0% |
| Psychiatric conditions | 62.5% |

### What Fails
| Category | Recall@30 | Root Cause |
|----------|-----------|------------|
| Monoclonal antibodies | 27.3% | Data sparsity (2.1 vs 11.1 diseases/drug) |
| Infectious diseases | 13.6% | Model predicts antibiotics for wrong diseases |
| Oncology mAbs | 0-17% | Weak knowledge graph connections |

## Identified But Unexplored Opportunities

1. **TxGNN Ensemble** - TxGNN excels at storage diseases (83.3%), could ensemble with GB
2. **Disease-class specific models** - Train separate models for infectious vs non-infectious
3. **Mechanism-based boosting** - Use drug mechanism to boost/filter predictions
4. **Negative sampling improvements** - Current random negatives may be suboptimal
5. **Graph structure features** - Path-based features between drugs and diseases

## Constraints
- Prefer approaches using existing data
- Prioritize interpretable improvements over black-box gains
- Validate improvements on held-out disease sets (not training diseases)

## GPU Resources (Vast.ai)

When a hypothesis requires GPU (model training, embedding retraining, TxGNN inference, etc.), you have access to Vast.ai cloud GPUs. Current balance: ~$4.41.

### Quick Commands
```bash
# Search for GPU instances (RTX 3090/4090)
vastai search offers 'gpu_name in [RTX_3090, RTX_4090] disk_space >= 50 reliability > 0.95' -o 'dph_total' --limit 10

# Create instance from offer ID
vastai create instance <OFFER_ID> --image pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel --disk 50

# Get SSH connection details
vastai show instances
vastai ssh-url <INSTANCE_ID>

# Setup TxGNN (after getting PORT and HOST)
./scripts/vastai_txgnn_setup.sh <PORT> <HOST>

# IMPORTANT: Always destroy instance when done to avoid burning balance
vastai destroy instance <INSTANCE_ID>
```

### Rules
- Always pick the cheapest instance that meets requirements
- Destroy the instance as soon as the job finishes — do not leave it running
- Log the instance ID, cost, and duration in your findings
- If balance is insufficient, note it in findings and mark hypothesis as blocked

## Success Metrics
- Primary: Per-drug Recall@30 on held-out diseases
- Secondary: Precision of top-100 predictions (via external validation)
- Tertiary: Calibration quality (does confidence predict success?)
- Avoid: Circular features, data leakage, evaluation on training set

## Research Directions (When Primary Metrics Plateau)

If you hit a fundamental ceiling on R@30, pivot to these directions:

### 1. Precision & Calibration
- Meta-confidence models: predict "will this disease hit@30?"
- Prediction tiering by confidence
- Per-category confidence thresholds

### 2. Error Analysis
- Which drugs are systematically missed?
- Which disease categories fail and why?
- What patterns predict failure?

### 3. Production Optimization
- Prediction prioritization for maximum value
- Category-specific strategies
- Negative prediction value (what to exclude)

### 4. Meta-Science
- What predicts whether a hypothesis will succeed?
- Which research directions have highest ROI?
- How to allocate effort across disease categories?

### 5. Inverse Problems
- What drug-disease pairs can we confidently EXCLUDE?
- Where is the model most reliable for "no effect" predictions?

**Key principle: Science never ends. If recall is capped, improve precision. If precision is capped, improve calibration. If calibration is capped, improve interpretability. There is always more to explore.**
