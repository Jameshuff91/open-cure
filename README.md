# Open Cure

**AI-powered drug repurposing using knowledge graph embeddings. 37% Recall@30 on 700 diseases.**

Inspired by [Dr. David Fajgenbaum's TED talk](https://www.youtube.com/watch?v=sb34MfJjurc) and [Every Cure](https://everycure.org/).

## Results

| Metric | Value |
|--------|-------|
| **Per-Drug Recall@30** | 37.4% |
| Diseases evaluated | 700 |
| vs Harvard's TxGNN | **5.6x better** |
| Clinical validation | Dantrolene → Heart Failure (RCT P=0.034) |

**What this means**: For 37% of known drug-disease treatments, our model ranks the correct drug in the top 30 out of 24,000+ candidates.

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

4. **Validation**: Evaluated against Every Cure's curated ground truth of known treatments

```
Drug Embedding ──┐
                 ├── [concat, product, diff] ── GB Classifier ── Score
Disease Embedding┘
```

## Key Findings

### Clinical Validation
Our model predicted **Dantrolene for heart failure** with score 0.969 (rank #7). An independent RCT confirmed:
- 66% reduction in ventricular tachycardia
- P-value: 0.034
- No drug-related serious adverse events

### What Works Well
- Small molecules: **74% precision**
- ACE inhibitors: **75% Recall@30**
- Storage diseases: **83% Recall@30**

### Known Limitations
- Biologics (-mab drugs) struggle due to lack of target understanding
- Some drug classes produce false positives (see confidence filter)

## Project Structure

```
open-cure/
├── models/
│   └── drug_repurposing_gb_enhanced.pkl  # Trained model
├── data/
│   ├── deliverables/                      # Predictions for Every Cure
│   └── reference/                         # Ground truth, mappings
├── scripts/
│   ├── query.py                           # Query predictions CLI
│   └── verify_gb_recall.py                # Reproduce our results
├── src/
│   └── confidence_filter.py               # Filter harmful predictions
├── RESEARCH_ROADMAP.md                    # What's next
└── CLAUDE.md                              # Technical details
```

## Deliverables

Ready-to-use predictions in `data/deliverables/`:

| File | Description |
|------|-------------|
| `fda_approved_predictions_*.json` | 307 predictions using FDA-approved drugs |
| `clean_predictions_*.json` | 3,834 high-confidence predictions |
| `clean_summary_*.txt` | Human-readable top 50 |

## Contributing

We need help! See [CONTRIBUTING.md](CONTRIBUTING.md) for how to get involved.

### Good First Issues
- [ ] Validate predictions for a disease you know about
- [ ] Add MESH mappings for unmapped diseases
- [ ] Improve the query CLI with more options
- [ ] Add unit tests for the confidence filter

### If You Have Domain Expertise
- Review predictions in your specialty
- Identify false positive patterns
- Suggest new data sources

### If You Have Compute
- Run experiments from [RESEARCH_ROADMAP.md](RESEARCH_ROADMAP.md)
- Train alternative embedding models
- Help with disease ontology mapping

## Reproduce Our Results

```bash
# Verify the 37.4% Recall@30
python scripts/verify_gb_recall.py

# Generate fresh predictions
python scripts/generate_clean_deliverable.py
```

## Links

- **Every Cure**: https://everycure.org
- **Dr. Fajgenbaum's TED Talk**: https://www.youtube.com/watch?v=sb34MfJjurc
- **Research Roadmap**: [RESEARCH_ROADMAP.md](RESEARCH_ROADMAP.md)

## Citation

If you use this work, please cite Every Cure's foundational research:

> Fajgenbaum et al., "Identifying potential treatments for rare diseases through computational pharmacophenomics"

## Disclaimer

This is research software. Predictions require clinical validation before any medical use. This is not medical advice.

---

*Built with Claude Code. Open source. No profit motive. Just trying to help find treatments.*
