# Open-Cure: Drug Repurposing Analysis for Rare Genetic Skin Diseases

**Prepared for:** Ryland Mortlock (Yale MD-PhD, Choate Lab)
**Date:** February 2026
**Contact:** Jim Huff

---

## Executive Summary

Open-Cure provides computational drug repurposing predictions using knowledge graph embeddings and collaborative filtering. This analysis package supports your **Gene → Pathway → Drug** workflow for identifying treatment candidates for rare genetic skin diseases.

### Key Deliverables

| File | Description | Records |
|------|-------------|---------|
| `nfkb_pathway_drugs.xlsx` | All drugs targeting NF-kB pathway genes (for SLURP1) | 1,022 drugs |
| `egfr_direct_drugs.xlsx` | All drugs targeting EGFR (validation) | 185 drugs |
| `egfr_pathway_drugs.xlsx` | All drugs targeting ErbB/EGFR pathway genes | 924 drugs |
| `skin_disease_predictions.xlsx` | Dermatological disease predictions | 330 predictions |
| `ichthyosis_predictions.xlsx` | Ichthyosis-specific predictions | 150 predictions |
| `atopic_dermatitis_predictions.xlsx` | Atopic dermatitis predictions | 30 predictions |

---

## SLURP1 Analysis

### Background
SLURP1 mutations cause Mal de Meleda, a rare palmoplantar keratoderma. The gene is associated with the NF-kB signaling pathway and neuroactive ligand-receptor interactions.

### Direct Drug Targets
SLURP1 has only **2 direct drug interactions** in DRKG:
- Acetylcholine (DB03128)
- One other compound

### Pathway-Based Candidates (Recommended Approach)

Since direct targets are limited, we recommend exploring drugs that target genes in SLURP1-related pathways:

**NF-kB Pathway (hsa04064):**
- 105 genes in pathway
- 86 genes with known drug targets
- **1,022 DrugBank drugs** targeting these genes
- Top drugs by target coverage: Dexamethasone (20 targets), Doxorubicin (19), Staurosporine (19)

See `nfkb_pathway_drugs.xlsx` for the complete list with:
- Drug name and DrugBank ID
- Target genes within the pathway
- Number of pathway targets per drug
- Gene functional roles (for key NF-kB genes)

### Neuroactive Ligand-Receptor Pathway (hsa04080)
SLURP1 is also in this pathway. Use the query script to explore:
```bash
python scripts/query_by_gene.py --pathway hsa04080 --top 100
```

---

## EGFR Validation

To validate our data against your erlotinib/EMP2 discovery:

### Erlotinib in Our Database
**Confirmed:** Erlotinib (DB00530) is present and correctly linked to EGFR (Gene ID: 1956)

### EGFR Drug Coverage
- **392 total compounds** targeting EGFR
- **185 DrugBank drugs** with clinical relevance
- Includes known EGFR inhibitors: Erlotinib, Gefitinib, Afatinib, Osimertinib

See `egfr_direct_drugs.xlsx` for the complete list.

---

## Ichthyosis Predictions

### Diseases Analyzed
1. Ichthyosis vulgaris (MESH:D016114, D016112)
2. Lamellar ichthyosis (MESH:D017490)
3. Recessive X-linked ichthyosis (MESH:D016113)
4. Ichthyosis general (MESH:D007057)

### Top Drug Candidates (Novel Predictions)

| Drug | Disease | kNN Score | Notes |
|------|---------|-----------|-------|
| Calcium | Ichthyosis vulgaris | 1.93 | Skin barrier function |
| Dexamethasone | Ichthyosis vulgaris | 1.92 | Anti-inflammatory |
| Cyclosporine | Ichthyosis vulgaris | 1.92 | Immunomodulator |
| Doxycycline | Ichthyosis vulgaris | 1.92 | Multiple skin conditions |
| Tretinoin | Lamellar ichthyosis | 1.71 | Retinoid, keratinocyte regulation |

**Note:** These are computational predictions ranked by similarity to diseases with known treatments. Clinical validation required.

---

## How to Use the Query Scripts

### Gene → Drug Lookup
```bash
# Direct gene targets
python scripts/query_by_gene.py SLURP1

# With pathway expansion
python scripts/query_by_gene.py SLURP1 --include-pathway

# Any gene
python scripts/query_by_gene.py EGFR --top 30
```

### Pathway → Drug Lookup
```bash
# NF-kB pathway
python scripts/query_by_gene.py --pathway hsa04064

# List pathways for a gene
python scripts/query_by_gene.py --list-pathways SLURP1
```

### Disease → Drug Predictions
```bash
# Query by disease name
python scripts/query.py "atopic dermatitis" --top 20

# Query by drug (find diseases)
python scripts/query.py --drug "erlotinib"
```

### Mechanism Tracing
```bash
# Trace drug-disease mechanisms
python scripts/trace_mechanism_paths.py
```

---

## Data Sources

| Source | Description |
|--------|-------------|
| DRKG | Drug-target interactions, disease-gene associations |
| DrugBank | Drug metadata and identifiers |
| KEGG | Pathway annotations |
| Every Cure | Ground truth disease-drug relationships |

---

## Methodology

### kNN Collaborative Filtering
Our main prediction method achieves **26% Recall@30** on held-out diseases:
1. Embed diseases using Node2Vec on DRKG
2. For each disease, find k=20 nearest neighbor diseases
3. Rank drugs by frequency in neighbors' treatments
4. Filter by confidence scoring

### Pathway Analysis
For genes with few direct drug targets, pathway analysis identifies:
1. All genes in the same KEGG pathway as the target gene
2. All drugs targeting those pathway genes
3. Ranked by number of pathway targets (higher = more pathway coverage)

---

## Limitations

1. **Transductive Evaluation:** Predictions are most reliable for diseases with existing graph presence
2. **Rare Disease Gap:** Very rare diseases with no similar neighbors perform poorly
3. **Computational Only:** All predictions require experimental/clinical validation
4. **Coverage Dependent:** kNN fails if no similar training diseases have known treatments

---

## Next Steps

1. **SLURP1 Prioritization:** Review NF-kB pathway drugs for biological plausibility
2. **Choate Lab Diseases:** Provide additional gene symbols for pathway analysis
3. **Spatial Transcriptomics:** Could integrate expression data to improve rare disease predictions
4. **Novel Disease Pipeline:** Develop workflow for diseases not in our database

---

## Contact

For questions about the data or analysis:
- Jim Huff
- Open-Cure Project

For custom queries or additional diseases, the query scripts can be run locally.
