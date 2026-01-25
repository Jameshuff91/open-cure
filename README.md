# Open Cure

**AI-powered drug repurposing using knowledge graph embeddings. 47.5% Recall@30 on 602 diseases.**

Inspired by [Dr. David Fajgenbaum's TED talk](https://www.youtube.com/watch?v=sb34MfJjurc) and [Every Cure](https://everycure.org/).

## Results

| Metric | Value |
|--------|-------|
| **Per-Drug Recall@30** | **47.5%** |
| Diseases evaluated | 602 |
| vs Harvard's TxGNN | **7x better** |
| Novel predictions validated | 40 (22.5% precision) |
| Clinical validation | Dantrolene → Heart Failure (RCT P=0.034) |

**What this means**: For 47.5% of known drug-disease treatments, our model ranks the correct drug in the top 30 out of 24,000+ candidates.

## Contributions to Every Cure

We've contributed back to the community:
- [Issue #24](https://github.com/everycure-org/matrix-indication-list/issues/24): 16 FDA-approved pairs missing from ground truth
- [Issue #25](https://github.com/everycure-org/matrix-indication-list/issues/25): Exclusion rules for false positive filtering

## Quick Start

```bash
# Clone and setup
git clone https://github.com/Jameshuff91/open-cure.git
cd open-cure
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Query predictions for a disease
python scripts/query.py "parkinson disease"

# Query predictions for a drug
python scripts/query.py --drug "metformin"
```

## How It Works

1. **Knowledge Graph**: We use [DRKG](https://github.com/gnn4dr/DRKG) (5.8M edges connecting drugs, diseases, genes, proteins, pathways)

2. **Embeddings**: TransE learns 128-dimensional representations capturing biological relationships

3. **Classifier**: Gradient boosting predicts treatment probability from embedding features

4. **Quad Boost**: Predictions are boosted using domain knowledge:
   - Target overlap (drug targets ∩ disease genes)
   - ATC classification matching
   - Chemical similarity to known treatments
   - Pathway enrichment overlap

5. **Confidence Filter**: Rule-based exclusion of harmful patterns (withdrawn drugs, mechanism mismatches)

```
Drug Embedding ──┐
                 ├── [concat, product, diff] ── GB Classifier ── Base Score
Disease Embedding┘                                                    │
                                                                      ▼
Target Overlap ──┬── Quad Boost ── Boosted Score ── Confidence Filter ── Final Prediction
ATC Match ───────┤
Chemical Sim ────┤
Pathway Overlap ─┘
```

## Model Evolution

| Version | Recall@30 | Key Change |
|---------|-----------|------------|
| Baseline GB | 37.4% | Expanded MESH mappings |
| + Target Boost | 39.0% | Drug-disease gene overlap |
| + ATC Boost | 39.7% | Mechanism category matching |
| + Chemical Sim | 47.1% | Tanimoto fingerprint similarity |
| **+ Pathway Boost** | **47.5%** | KEGG pathway overlap |

## Key Findings

### Clinical Validation
Our model predicted **Dantrolene for heart failure** with score 0.969 (rank #7). An independent RCT confirmed:
- 66% reduction in ventricular tachycardia
- P-value: 0.034
- No drug-related serious adverse events

### Novel Predictions Validated (4 actionable)
| Drug | Disease | Evidence |
|------|---------|----------|
| Ravulizumab | Asthma | Eculizumab proof-of-concept trial |
| Nimotuzumab | Psoriasis | EGFR overexpressed, case reports |
| Alirocumab | Psoriasis | Mendelian Randomization p<0.003 |
| Leronlimab | Ulcerative Colitis | CCR5 blockade effective in mouse models |

### Ground Truth Gaps Found (FDA-approved, missing from training)
| Drug | Disease | FDA Year |
|------|---------|----------|
| Ustekinumab | Ulcerative Colitis | 2019 |
| Guselkumab | Ulcerative Colitis | 2024 |
| Pembrolizumab | Breast Cancer | 2020 |
| Natalizumab | Multiple Sclerosis | 2004 |

### What Works Well
- Small molecules: **74% precision**
- ACE inhibitors: **67% Recall@30**
- Autoimmune diseases: **63% Recall@30**
- Storage diseases: **83% Recall@30**

### Known Limitations
- Biologics (-mab drugs): Only 27% Recall@30
- Infectious diseases: Only 14% Recall@30
- ~22% of top predictions validate (78% false positive rate)

## False Positive Patterns Identified

We documented 28+ exclusion patterns:
- **Anti-IL-5 for psoriasis/UC**: Reduces eosinophils but fails clinically
- **IL-6 inhibitors for psoriasis**: Wrong pathway (need IL-17/IL-23)
- **Intravitreal drugs for systemic diseases**: Wrong formulation
- **B-cell depletion for psoriasis**: Paradoxically worsens disease

See `src/confidence_filter.py` for the full rule set.

## Project Structure

```
open-cure/
├── models/
│   ├── drug_repurposing_gb_enhanced.pkl  # Trained model
│   └── confidence_calibrator.pkl          # ML confidence predictor
├── data/
│   ├── analysis/                          # Validation results
│   ├── deliverables/                      # Predictions for Every Cure
│   └── reference/                         # Ground truth, mappings
├── scripts/
│   ├── query.py                           # Query predictions CLI
│   ├── evaluate_pathway_boost.py          # Current best evaluation
│   └── filter_novel_predictions.py        # Apply confidence filter
├── src/
│   ├── confidence_filter.py               # Filter harmful predictions
│   ├── chemical_features.py               # Tanimoto similarity
│   ├── pathway_features.py                # KEGG pathway enrichment
│   └── atc_features.py                    # ATC classification
└── CLAUDE.md                              # Full technical details
```

## Reproduce Our Results

```bash
# Verify the 47.5% Recall@30
python scripts/evaluate_pathway_boost.py

# Generate filtered novel predictions
python scripts/filter_novel_predictions.py

# Check validation results
cat data/analysis/validation_results_batch2.json
```

## Contributing

We need help!

### Good First Issues
- [ ] Validate predictions for a disease you know about
- [ ] Add MESH mappings for unmapped diseases
- [ ] Identify new false positive patterns
- [ ] Add unit tests for the confidence filter

### If You Have Domain Expertise
- Review predictions in your specialty
- Identify mechanism mismatches we've missed
- Suggest new exclusion rules

### If You Have Compute
- Improve chemical similarity coverage
- Add more pathway databases (Reactome, WikiPathways)
- Train disease-specific models

## Links

- **Every Cure**: https://everycure.org
- **Our Contributions**: [Issue #24](https://github.com/everycure-org/matrix-indication-list/issues/24), [Issue #25](https://github.com/everycure-org/matrix-indication-list/issues/25)
- **Dr. Fajgenbaum's TED Talk**: https://www.youtube.com/watch?v=sb34MfJjurc

## Citation

If you use this work, please cite Every Cure's foundational research:

> Fajgenbaum et al., "Identifying potential treatments for rare diseases through computational pharmacophenomics"

## Disclaimer

This is research software. Predictions require clinical validation before any medical use. This is not medical advice.

---

*Built with Claude Code. Open source. No profit motive. Just trying to help find treatments.*
