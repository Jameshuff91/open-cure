# Contributing to Open Cure

Thanks for your interest in helping find treatments for diseases! This guide will help you get started.

## Ways to Contribute

### 1. Validate Predictions (No coding required)

The most valuable contribution is validating our predictions against real medical knowledge.

**How to help:**
1. Pick a disease you know about
2. Run `python scripts/query.py "your disease"`
3. Review the top 10-20 predictions
4. Create an issue reporting:
   - Which predictions make sense biologically
   - Which are false positives (and why)
   - Any known treatments we missed

### 2. Add Disease Mappings (Beginner-friendly)

We're missing MESH mappings for ~80 diseases. Adding mappings directly improves our evaluation.

**How to help:**
1. Look at `data/reference/everycure_gt_for_txgnn.json` for unmapped diseases
2. Search [NIH MeSH Browser](https://meshb.nlm.nih.gov/search) for the disease
3. Add the mapping to `data/reference/mesh_mappings_from_agents.json`
4. Submit a PR

### 3. Improve the Code (Intermediate)

**Good first issues:**
- Add unit tests for `src/confidence_filter.py`
- Add more options to `scripts/query.py` (e.g., filter by drug type)
- Improve error messages and edge case handling
- Add type hints to existing code

**Bigger projects:**
- Build a web interface for queries
- Add ClinicalTrials.gov integration
- Implement new embedding methods

### 4. Run Experiments (Advanced)

See [RESEARCH_ROADMAP.md](RESEARCH_ROADMAP.md) for open research questions:
- Add drug-target features from DrugBank
- Train disease-specific models
- Integrate external data (ChEMBL, PubMed)

## Setup

```bash
# Clone
git clone https://github.com/Jameshuff91/open-cure.git
cd open-cure

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Verify setup
python scripts/query.py "heart failure" --top 5
```

**Required files** (not in git due to size):
- `models/transe.pt` - TransE embeddings (~100MB)
- `models/drug_repurposing_gb_enhanced.pkl` - Trained model (~50MB)

Contact the maintainers if you need access to these files.

## Code Style

- Python 3.9+
- Type hints encouraged
- Docstrings for public functions
- No `# type: ignore` without explanation

## Pull Request Process

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Run existing scripts to verify nothing broke
5. Submit PR with clear description

## Reporting Issues

When reporting bugs, please include:
- What you were trying to do
- What happened vs what you expected
- Python version and OS
- Full error traceback

## Questions?

- Open a GitHub issue
- Check [CLAUDE.md](CLAUDE.md) for technical details
- Check [RESEARCH_ROADMAP.md](RESEARCH_ROADMAP.md) for project direction

## Code of Conduct

Be respectful. We're all here to help find treatments for diseases. Disagreements about methodology are fine; personal attacks are not.

---

*Every contribution matters. Even fixing a typo helps.*
