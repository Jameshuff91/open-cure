# Contributing to Open Cure

Thank you for your interest in contributing to Open Cure! This project aims to find new treatments for diseases through computational drug repurposing, and every contribution helps move us closer to that goal.

## Our Mission

We believe that finding cures should be a collaborative, open effort. Over 300 million people globally have diseases with no FDA-approved treatment. By combining open-source tools, publicly available biomedical data, and community contributions, we can help identify promising drug candidates faster.

## Ways to Contribute

### 1. Code Contributions
- Implement new ML models for drug-disease prediction
- Improve data ingestion pipelines
- Add new knowledge graph integrations
- Enhance model explainability
- Fix bugs and improve performance

### 2. Data Contributions
- Help curate and validate ground truth datasets
- Map disease/drug identifiers across ontologies (DOID, MONDO, MESH, etc.)
- Identify new open data sources

### 3. Research Contributions
- Analyze model predictions for specific diseases
- Validate findings against clinical literature
- Document failure cases and edge cases
- Write research notes and experiment summaries

### 4. Documentation
- Improve setup instructions
- Document model architectures and training procedures
- Add examples and tutorials

## Getting Started

### Prerequisites
- Python 3.10+
- Git

### Local Development Setup

```bash
# Fork and clone the repo
git clone https://github.com/YOUR_USERNAME/open-cure.git
cd open-cure

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests to verify setup
pytest tests/
```

### GPU Resources

For GPU-intensive tasks (training TxGNN, large embedding models), we recommend **[Vast.ai](https://vast.ai)** for affordable GPU rentals:

- GTX 1080 Ti / Titan Xp: ~$0.05-0.10/hr
- RTX 3090 / A6000: ~$0.20-0.40/hr

See the [GPU Setup Guide](#gpu-setup-with-vastai) below for details.

## Contribution Workflow

### 1. Find or Create an Issue
- Check [existing issues](https://github.com/yourusername/open-cure/issues) for something to work on
- For new features or bugs, create an issue first to discuss the approach
- Issues labeled `good first issue` are great for newcomers

### 2. Create a Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### 3. Make Your Changes
- Write clear, documented code
- Follow existing code style
- Add tests for new functionality
- Update documentation as needed

### 4. Test Your Changes
```bash
# Run the test suite
pytest tests/

# For model changes, validate against ground truth
python scripts/evaluate_model.py
```

### 5. Submit a Pull Request
- Push your branch and open a PR against `main`
- Fill out the PR template
- Link any related issues
- Be responsive to review feedback

## Code Style

### Python
- Use type hints where practical
- Follow PEP 8 conventions
- Use descriptive variable names
- Document functions with docstrings

```python
def predict_drug_disease_score(
    drug_id: str,
    disease_id: str,
    model: torch.nn.Module
) -> float:
    """
    Predict the repurposing score for a drug-disease pair.

    Args:
        drug_id: DrugBank ID (e.g., "DB00945")
        disease_id: DOID disease ID (e.g., "DOID:10652")
        model: Trained prediction model

    Returns:
        Score between 0 and 1, higher means more likely to treat
    """
    ...
```

### Commits
- Write clear, descriptive commit messages
- Use present tense ("Add feature" not "Added feature")
- Reference issues when relevant: `Fix #123: Handle missing embeddings`

## GPU Setup with Vast.ai

For contributors who need GPU access for training models:

### 1. Install Vast.ai CLI
```bash
pip install vastai
vastai set api-key YOUR_API_KEY  # Get key from vast.ai/console
```

### 2. Find an Instance
```bash
# Search for affordable GPUs
vastai search offers 'gpu_ram>=8 cuda_vers>=11.0 reliability>0.95' --order 'dph' --limit 10
```

### 3. Create and Connect
```bash
# Create instance (note the OFFER_ID from search)
vastai create instance OFFER_ID --image nvidia/cuda:11.7.1-runtime-ubuntu22.04 --disk 30 --ssh

# Check status
vastai show instances

# SSH to instance
ssh -p PORT root@SSH_ADDR
```

### 4. Important: Destroy When Done
```bash
# Stop billing - ALWAYS do this when finished!
vastai destroy instance INSTANCE_ID
```

**Cost tips:**
- Titan Xp and GTX 1080 Ti are excellent value (~$0.05-0.10/hr)
- Use `nohup` for long training runs so you can disconnect
- Spot instances are cheaper but may be interrupted

## Project Areas

### High Priority
- **Ontology Mapping**: Improving DOID/MONDO/MESH disease mappings
- **Model Ensembles**: Combining predictions from multiple models
- **Rare Disease Focus**: Improving performance on diseases with sparse data

### Good First Issues
- Adding drug/disease name mappings
- Writing evaluation scripts
- Documenting existing models
- Adding unit tests

### Advanced
- Implementing new GNN architectures
- Fine-tuning TxGNN on our ground truth
- Building explanation/reasoning modules

## Research Notes

When running experiments, please document your findings:
- What you tried
- What worked and what didn't
- Key metrics and comparisons
- Lessons learned

See `docs/txgnn_research_notes.md` for an example.

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Assume good intentions
- Help newcomers get started
- Celebrate contributions of all sizes

## Questions?

- Open a [Discussion](https://github.com/yourusername/open-cure/discussions) for general questions
- Check existing issues and docs first
- Tag maintainers if you're stuck

## Recognition

All contributors will be acknowledged in our release notes and contributor list. Significant contributions may be recognized in any publications that result from this work.

---

*Together, we can help find treatments for the millions of people waiting for cures.*
