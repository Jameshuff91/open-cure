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

### Literature Mining Tools
- `src/literature_miner.py` - Automated PubMed + ClinicalTrials.gov mining with optional Claude Haiku abstract classification
- `scripts/run_literature_mining.py` - CLI runner for batch literature mining
- `scripts/validate_literature_evidence.py` - Holdout validation of literature evidence levels
- Cache: `data/validation/literature_mining_cache.json` (reuses existing `validation_cache.json` with 1,052 entries)

**Usage for hypothesis validation:**
```bash
# Mine literature for NOVEL predictions in a tier
python scripts/run_literature_mining.py --tier MEDIUM --status NOVEL --top 200

# Mine with LLM abstract classification (detects adverse effects)
python scripts/run_literature_mining.py --tier HIGH --status NOVEL --use-llm

# View cached results summary
python scripts/run_literature_mining.py --summary

# Validate evidence levels against holdout precision
python scripts/validate_literature_evidence.py --tier MEDIUM
```

**When to use:** Run literature mining when investigating GT gaps, validating novel predictions, or detecting adverse effects. Evidence levels (STRONG/MODERATE/WEAK/ADVERSE/NO_EVIDENCE) are added to the deliverable via `literature_evidence_level` and `literature_evidence_score` columns.

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

## External Data Acquisition

You are authorized to download, curate, cleanse, and integrate publicly available datasets. Do not wait for human intervention — if a dataset is public, go get it.

### Approved Public Data Sources

| Dataset | URL / Method | Format | Store In |
|---------|-------------|--------|----------|
| **LINCS L1000 Phase I** | `wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/` | HDF5 (~3GB) | `data/external/lincs/` |
| **LINCS L1000 Phase II** | `wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl/` | HDF5 | `data/external/lincs/` |
| **Edge2Vec** | `pip install edge2vec` or `git clone https://github.com/RoyZhengGao/edge2vec` | Python package | N/A (installed) |
| **PharMeBINet** | Download from Zenodo (search "PharMeBINet") | TSV/CSV | `data/external/pharmebinet/` |
| **PubChem** | REST API: `https://pubchem.ncbi.nlm.nih.gov/rest/pug/` | JSON | `data/reference/pubchem/` |
| **ClinicalTrials.gov** | API: `https://clinicaltrials.gov/api/v2/studies` | JSON | API calls (no storage needed) |
| **GEO/ARCHS4** | `https://maayanlab.cloud/archs4/download.html` | HDF5 | `data/external/archs4/` |
| **UniProt** | REST API: `https://rest.uniprot.org/` | JSON | `data/reference/uniprot/` |
| **KEGG** | REST API: `https://rest.kegg.jp/` | Text | Already using |

### Data Acquisition Protocol

When a hypothesis requires external data:

1. **Check if already downloaded**: Look in `data/external/` and `data/reference/` first
2. **Download to correct location**: Use `data/external/<source_name>/` for new datasets
3. **Add .gitignore**: Large files (>50MB) must be gitignored. Create `data/external/<source>/.gitignore` with `*` and `!.gitignore`
4. **Curate and clean**:
   - Map IDs to our format (DrugBank drug IDs, MESH disease IDs where possible)
   - Document ID mapping coverage (what % of our drugs/diseases are covered?)
   - Remove duplicates, handle missing values, normalize formats
   - Save processed version as JSON or pickle alongside raw data
5. **Integration script**: Save to `scripts/integrate_<source>.py` with clear documentation
6. **Log coverage stats** in hypothesis findings: how many of our 10K+ drugs and 3K+ diseases are covered?

### ID Mapping (Critical Blocker)

Our system uses mixed IDs. When integrating external data, you will need to map between:
- **Drugs**: DrugBank IDs ↔ PubChem CID ↔ CHEMBL ↔ CHEBI ↔ drug names
- **Diseases**: MESH IDs ↔ MONDO ↔ UMLS ↔ disease names
- **Genes**: Entrez Gene IDs ↔ UniProt ↔ HGNC symbols

Use `data/reference/drugbank_lookup.json` and `data/reference/disease_ontology_mapping.json` as starting points. If coverage is <50%, build a broader mapping using PubChem/UniProt APIs and save to `data/reference/id_mappings/`.

### Large Downloads

For datasets >1GB:
- Use `wget` with `--continue` flag (resume on failure)
- Run downloads with `nohup` to survive disconnections
- Verify checksums if available
- Consider downloading on Vast.ai GPU instance if local disk is constrained

## Collaboration: Ryland Mortlock (Yale, Meeting Monday Feb 10)

We are partnering with Ryland Mortlock, an MD-PhD student at Yale (Genetics dept, Choate lab). His expertise:
- **Spatial transcriptomics** and single-cell RNA sequencing
- **Genomics** and gene expression analysis (R, Python)
- **Genetic skin diseases** — inflammation mechanisms, EGFR signaling
- **Published**: PNAS, Nature Computational Science, Cell Stem Cell, JID, BJD
- **Nucleate Activator 2026** participant (biotech startup program)

### What Ryland Brings
- Wet-lab validation capabilities (can test predictions in cell models)
- Transcriptomic data analysis (LINCS, GEO, ARCHS4 expertise)
- Clinical genetics perspective (which predictions are clinically meaningful?)
- Access to Yale computational resources and expertise

### Prep Work for Monday Meeting
When working on hypotheses, flag any findings that would benefit from Ryland's expertise:
- Gene expression signatures that need transcriptomic validation
- Predictions for genetic/dermatological diseases (his specialty area)
- ID mapping problems that genomics databases could solve
- Predictions that could be tested in cell culture models
- Any results where a geneticist's interpretation would add value

Tag relevant findings with `[RYLAND]` in the hypothesis findings field.

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
