# Open Cure

**AI-powered drug repurposing using knowledge graph embeddings.**

Inspired by [Dr. David Fajgenbaum's TED talk](https://www.youtube.com/watch?v=sb34MfJjurc) and [Every Cure](https://everycure.org/).

## Important Limitation

**This project solves a different problem than Every Cure's core mission.**

| Problem | Description | Our Performance |
|---------|-------------|-----------------|
| **Transductive** (what we solve) | Find *additional* drugs for diseases that already have treatments | 37% Recall@30 |
| **Inductive/Zero-shot** (what Every Cure needs) | Find *first* drugs for diseases with NO known treatments | ~15% Recall@30 |

Our kNN method works by transferring treatments from similar diseases. For diseases with no similar treated diseases, this fundamentally cannot work. See [Limitations](#limitations--honest-assessment) for details.

## Results

| Metric | Value | Context |
|--------|-------|---------|
| **kNN Recall@30** | **37.04% ± 5.81%** | Diseases with existing treatments (transductive) |
| **KEGG Pathway kNN** | **15.73% ± 1.82%** | Inductive (no graph embeddings) |
| Diseases evaluated | 368 | After MESH mapping filter |
| Confidence precision | 88% at high threshold | For production use |
| Clinical validation | Dantrolene → Heart Failure | RCT P=0.034 |

**Comparison to TxGNN**: Harvard's TxGNN achieves 6.7-14.5% on true zero-shot (diseases removed from graph). Our 37% is on transductive evaluation (diseases remain in graph). **These are not directly comparable.** Under fair inductive conditions, our KEGG method (15.7%) is comparable to TxGNN.

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

### Current Approach: kNN Collaborative Filtering

Our best method doesn't use ML at all. It's based on a simple insight: **similar diseases share treatments**.

```
1. Embed diseases using Node2Vec on DRKG
2. For a test disease, find k=20 nearest training diseases
3. Rank drugs by weighted frequency in neighbors' treatments
4. Apply confidence scoring and safety filters
```

This achieves 37% Recall@30 on diseases with existing treatments — but fails completely for diseases with no similar treated neighbors.

### Architecture

```
Disease ──[Node2Vec]──> Embedding ──[Cosine Similarity]──> k Nearest Diseases
                                                                    │
                                                                    ▼
                                                    Neighbor Treatment Drugs
                                                                    │
                                                                    ▼
                                        Confidence Scoring ──> Safety Filter ──> Predictions
```

## Limitations & Honest Assessment

### What We Learned (81 Hypotheses Tested)

Our autonomous research agent tested 81 hypotheses. Key findings:

**What Works:**
- kNN collaborative filtering (37% R@30) — best method, no ML needed
- Confidence scoring (88% precision at high threshold)
- Per-category calibration (autoimmune/dermatological achieve 93%+ precision)

**What Failed:**
| Approach | Why It Failed |
|----------|---------------|
| PPI network features | Already captured in Node2Vec embeddings |
| HPO phenotype similarity | Too sparse (25% coverage), no improvement |
| Gene overlap similarity | 13.56 pp WORSE than Node2Vec cosine |
| Learned similarity metrics | Overfits to training pairs |
| Bio foundation models (Geneformer) | Domain mismatch (needs real expression data) |
| XGBoost hybrid | Adds nothing to kNN; can't rescue failed cases |

**The Fundamental Limitation:**
- 44% of test diseases have zero GT drug coverage in kNN neighbors
- For these diseases, performance is 0%
- No algorithm can conjure knowledge that isn't in the graph

### The 37% Ceiling

This isn't a model limitation — it's a **data limitation**. The DRKG knowledge graph simply doesn't contain treatment information for many disease neighborhoods. Breaking this ceiling requires external data sources.

## What Actually Helps Every Cure

For diseases with **no known treatments** (Every Cure's core mission), we need fundamentally different approaches:

### Planned Next Steps

1. **Literature Mining with LLMs**
   - Extract drug-disease hypotheses from PubMed
   - Use Claude to identify mechanism-based connections
   - Not dependent on DRKG graph structure

2. **Connectivity Map / LINCS**
   - Find drugs that reverse disease gene expression signatures
   - Mechanistically grounded, doesn't require similar diseases

3. **Direct Mechanism Traversal**
   - Disease → Gene → Drug paths in DRKG
   - Pure logical inference, no ML training needed

4. **Zero-Shot Benchmark**
   - Define test set of Every Cure diseases with ZERO known treatments
   - Evaluate approaches on this true benchmark

## Key Findings

### Clinical Validation
Our model predicted **Dantrolene for heart failure** with high confidence. An independent RCT confirmed:
- 66% reduction in ventricular tachycardia
- P-value: 0.034
- No drug-related serious adverse events

### Novel Predictions Validated
| Drug | Disease | Evidence |
|------|---------|----------|
| Ravulizumab | Asthma | Eculizumab proof-of-concept trial |
| Nimotuzumab | Psoriasis | EGFR overexpressed, case reports |
| Alirocumab | Psoriasis | Mendelian Randomization p<0.003 |
| Leronlimab | Ulcerative Colitis | CCR5 blockade effective in mouse models |

### Performance by Category
| Category | Performance | Notes |
|----------|-------------|-------|
| Autoimmune | 63% R@30 | Best category |
| Storage diseases | 83% R@30 | Small sample |
| Infectious | 14% R@30 | Poor — antibiotics confound |
| Biologics (-mab) | 27% R@30 | Worse than small molecules |

## False Positive Patterns

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
│   ├── drug_repurposing_gb_enhanced.pkl  # Legacy GB model
│   └── meta_confidence_model.pkl         # Confidence predictor
├── data/
│   ├── analysis/                         # Validation results
│   ├── deliverables/                     # Predictions with confidence
│   └── reference/                        # Ground truth, mappings
├── scripts/
│   ├── query.py                          # Query predictions CLI
│   ├── evaluate_knn_collab.py            # kNN evaluation
│   └── evaluate_kegg_pathway_knn.py      # Inductive evaluation
├── src/
│   ├── confidence_filter.py              # Safety filter
│   └── knn_repurposing.py                # Core kNN method
├── research_loop/                        # Autonomous research agent
│   └── research_roadmap.json             # 81 hypotheses tested
└── CLAUDE.md                             # Full technical details
```

## Reproduce Our Results

```bash
# Verify kNN performance (transductive)
python scripts/evaluate_knn_collab.py

# Verify inductive performance
python scripts/evaluate_kegg_pathway_knn.py

# Generate predictions with confidence tiers
python scripts/generate_production_predictions.py
```

## Contributing

We need help — especially with **zero-shot approaches**!

### High-Impact Contributions
- [ ] Build literature mining pipeline for untreated diseases
- [ ] Integrate LINCS drug expression signatures
- [ ] Define benchmark of diseases with zero known treatments
- [ ] Validate predictions for rare diseases you know about

### Good First Issues
- [ ] Add MESH mappings for unmapped diseases
- [ ] Identify new false positive patterns
- [ ] Review predictions in your medical specialty

## Links

- **Every Cure**: https://everycure.org
- **Our Contributions**: [Issue #24](https://github.com/everycure-org/matrix-indication-list/issues/24), [Issue #25](https://github.com/everycure-org/matrix-indication-list/issues/25)
- **Dr. Fajgenbaum's TED Talk**: https://www.youtube.com/watch?v=sb34MfJjurc
- **Research Roadmap**: See `research_loop/research_roadmap.json` for all 81 hypotheses

## Citation

If you use this work, please cite Every Cure's foundational research:

> Fajgenbaum et al., "Identifying potential treatments for rare diseases through computational pharmacophenomics"

## Disclaimer

This is research software. Predictions require clinical validation before any medical use. This is not medical advice.

**Honest assessment**: Our current methods are best suited for finding *additional* treatments for diseases that already have *some* treatments. For truly untreated diseases, we're still working on it.

---

*Built with Claude Code. Open source. No profit motive. Just trying to help find treatments.*
