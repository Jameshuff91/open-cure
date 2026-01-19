# Open Cure

**Complementary open-source drug repurposing research using AI and biomedical knowledge graphs.**

Inspired by the groundbreaking work of [Dr. David Fajgenbaum](https://davidfajgenbaum.com/) and [Every Cure](https://everycure.org/), this project aims to contribute to the mission of finding new treatments by exploring computational approaches to drug repurposing.

> "Over 300 million people globally have a disease with no FDA-approved treatment. Of the approximately 18,000 recognized diseases, only 4,000 have approved treatments." — Every Cure

## Mission

Use open-source tools, publicly available biomedical knowledge graphs, and AI/ML to:

1. **Detect novel patterns** in drug-disease relationships that existing approaches may miss
2. **Improve explainability** — not just predict, but explain *why* a drug might work
3. **Focus on rare diseases** where data sparsity makes traditional approaches struggle
4. **Sub-phenotype analysis** — find which patient subgroups might respond to which drugs
5. **Validate and replicate** findings from other drug repurposing efforts

## How This Complements Every Cure

Every Cure has built an incredible platform scoring 74 million drug-disease pairs. This project aims to:

- Apply **different algorithms** that may find patterns their approach misses
- Incorporate **different data sources** (clinical notes, international literature, patient communities)
- Add **explainability layers** to predictions
- Focus on **edge cases** and rare diseases with sparse data
- Provide **independent validation** of high-scoring candidates

## Data Sources

We leverage open biomedical knowledge graphs:

| Source | Description | Nodes | Edges |
|--------|-------------|-------|-------|
| [DRKG](https://github.com/gnn4dr/DRKG) | Drug Repurposing Knowledge Graph | 97K | 5.8M |
| [Hetionet](https://het.io) | Integrative network for drug repurposing | 47K | 2.2M |
| [PrimeKG](https://github.com/mims-harvard/PrimeKG) | Precision medicine knowledge graph | 129K | 8M+ |
| [RTX-KG2](https://github.com/RTXteam/RTX-KG2) | NCATS Translator backbone | 6.4M | 39M |
| [PubMed Central](https://www.ncbi.nlm.nih.gov/pmc/) | Open access biomedical literature | - | - |
| [OpenAlex](https://openalex.org/) | Open catalog of scholarly works | 250M+ | - |

## Approaches

### 1. Graph Neural Networks
- Node2Vec, GraphSAGE, GAT for learning embeddings
- Link prediction for drug-disease associations

### 2. Large Language Models
- Extract relationships from literature not yet in knowledge graphs
- Generate explanations for predictions
- Reason over sparse data for rare diseases

### 3. Ensemble Methods
- Combine predictions from multiple algorithms
- Consensus scoring to reduce false positives

### 4. Causal Inference
- Move beyond correlation to causal mechanisms
- Identify confounders and mediators

## Project Structure

```
open-cure/
├── data/
│   ├── raw/              # Downloaded knowledge graphs
│   ├── processed/        # Cleaned, unified format
│   └── graphs/           # Graph database exports
├── src/
│   ├── ingest/           # Data ingestion pipelines
│   ├── models/           # ML/AI models
│   ├── analysis/         # Analysis utilities
│   └── api/              # API for querying results
├── notebooks/            # Jupyter notebooks for exploration
├── scripts/              # Utility scripts
├── docs/                 # Documentation
└── tests/                # Test suite
```

## Getting Started

```bash
# Clone the repo
git clone https://github.com/yourusername/open-cure.git
cd open-cure

# Install dependencies
pip install -r requirements.txt

# Download knowledge graphs
python scripts/download_graphs.py

# Run initial analysis
python src/ingest/build_unified_graph.py
```

## References & Acknowledgments

This work stands on the shoulders of giants:

- **Every Cure** — [everycure.org](https://everycure.org/) — Pioneering computational pharmacophenomics
- **NCATS Biomedical Data Translator** — [ncats.nih.gov/translator](https://ncats.nih.gov/translator)
- **Hetionet** — Himmelstein et al., "Systematic integration of biomedical knowledge prioritizes drugs for repurposing"
- **DRKG** — "Drug Repurposing Knowledge Graph for COVID-19"

### Key Papers

1. Fajgenbaum et al., "Pioneering a new field of computational pharmacophenomics" — *Lancet Haematology* 2025
2. "Biomedical knowledge graph learning for drug repurposing" — *Nature Communications* 2023
3. "Knowledge Graphs for drug repurposing: a review" — *Briefings in Bioinformatics* 2024

## Contributing

We welcome contributions! See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

## License

Apache 2.0 — See [LICENSE](LICENSE) for details.

## Disclaimer

This project is for research purposes only. Any drug repurposing candidates identified require rigorous clinical validation before any medical application. This is not medical advice.

---

*"The best time to plant a tree was 20 years ago. The second best time is now."*
