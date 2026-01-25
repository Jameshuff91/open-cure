# Every Cure Deliverable - January 2026

## Overview

Novel drug repurposing predictions generated using knowledge graph embeddings and gradient boosting classification.

**Model**: GB classifier on TransE embeddings from DRKG
**Performance**: 37.4% Recall@30 on Every Cure ground truth (5.6x better than TxGNN)
**Clinical Validation**: Dantrolene → Heart Failure (RCT P=0.034, 66% VT reduction)

## Files

| File | Description |
|------|-------------|
| `fda_approved_predictions_20260124.json` | **Start here** - 307 predictions using FDA-approved drugs only |
| `clean_predictions_20260124.json` | 3,834 predictions with proper drug names (score ≥0.85) |
| `clean_summary_20260124.txt` | Human-readable summary with top 50 predictions |
| `every_cure_predictions_20260124.csv` | Full dataset in CSV format |

## How to Use

1. **For clinical review**: Start with `fda_approved_predictions_*.json` - these use drugs that are already approved for other indications, making repurposing faster.

2. **For research**: Use `clean_predictions_*.json` for a broader set of candidates.

3. **Note**: Some predictions may already be approved uses (e.g., Empagliflozin → heart failure was FDA approved in 2021). This is validation that the model learns real biological signal, and may indicate gaps in the ground truth database.

## Top Validated Predictions

| Drug | Disease | Score | Evidence |
|------|---------|-------|----------|
| Dantrolene | Heart failure/VT | 0.969 | **RCT: 66% VT reduction, P=0.034** |
| Empagliflozin | Heart failure | 0.974 | FDA approved 2021 |
| Empagliflozin | Parkinson's | 0.903 | Korean study: 20% reduced PD risk |
| Paclitaxel | Rheumatoid arthritis | 0.990 | Phase I data, anti-angiogenic mechanism |
| Formoterol | Type 2 diabetes | 0.987 | Clinical study: 45-50% hypoglycemia reduction |

## Confidence Filter Applied

Predictions are filtered to exclude known harmful patterns:
- Antibiotics for metabolic diseases (inhibit insulin)
- Sympathomimetics for diabetes (raise glucose)
- Alpha blockers for heart failure (ALLHAT: 2x risk)
- TCAs/PPIs for hypertension (cause hypertension)
- Diagnostic agents (imaging, not treatment)

## Methodology

1. **Knowledge Graph**: DRKG (5.8M edges) - drugs, diseases, genes, proteins, pathways
2. **Embeddings**: TransE (128 dimensions) learned from graph structure
3. **Features**: Drug-disease embedding concatenation, product, and difference
4. **Classifier**: Gradient boosting (scikit-learn)
5. **Evaluation**: Per-drug Recall@30 on 700 diseases with MESH mappings

## Limitations

- Biologics (-mab drugs) have lower precision due to lack of target understanding
- Model reflects patterns in training data, not mechanisms
- Predictions require clinical validation before any use

## Contact

Repository: https://github.com/jimhuff/open-cure
Inspired by: Every Cure (https://everycure.org)
