# Open Cure Data Cleansing Specification

## Overview

This project cleanses the unified knowledge graph to improve data quality for drug repurposing ML models. The goal is to transform messy, merged data from multiple sources into a clean, standardized dataset comparable to PrimeKG quality.

## Current Data State

**Location:** `/Users/jimhuff/github/open-cure/data/processed/`

| File | Records | Issues |
|------|---------|--------|
| `unified_nodes.csv` | 273,581 | Duplicate types, ID-only names, no external mappings |
| `unified_edges.csv` | 16,224,956 | Fragmented relations, inconsistent naming |

## Known Data Quality Issues

### Issue 1: Duplicate Entity Types (Different names for same concept)
```
Gene (60,165) + gene/protein (27,610) → Should merge to "Gene"
Drug (25,865) + drug (7,957) + Compound → Should merge to "Drug"
disease (17,080) + Disease (5,240) → Should merge to "Disease"
biological_process (28,642) + BiologicalProcess (22,762) → Should merge to "BiologicalProcess"
molecular_function (11,169) + MolecularFunction (5,768) → Should merge to "MolecularFunction"
cellular_component (4,176) + CellularComponent (2,782) → Should merge to "CellularComponent"
pathway (2,516) + Pathway (3,644) → Should merge to "Pathway"
anatomy (14,033) + Anatomy → Should merge to "Anatomy"
```

### Issue 2: 30,803 Drugs Have ID-Only Names
```
Current:  "name": "Compound::DB02573"
Should be: "name": "Tobramycin"

The DrugBank IDs (DB#####) are embedded in the names - need to resolve to actual drug names.
```

### Issue 3: Treatment Relations Are Fragmented
```
Current relations that mean "treats/indicates":
- indication (18,776)
- DRUGBANK::treats::Compound:Disease (4,968)
- treats (755)

Current relations that mean "contraindication":
- contraindication (61,350)

Should unify to: INDICATION, CONTRAINDICATION
```

### Issue 4: Relation Names Have Source Prefixes
```
Current: "Hetionet::GpBP::Gene:Biological Process"
Should be: "participates_in" or similar clean name

Current: "DRUGBANK::ddi-interactor-in::Compound:Compound"
Should be: "drug_drug_interaction"
```

### Issue 5: No External ID Mappings
```
The "external_source" column is empty.
Should map entities to standard ontologies:
- Drugs → DrugBank ID, ChEMBL ID
- Diseases → MONDO, DOID, MeSH
- Genes → NCBI Gene ID, Ensembl, HGNC
```

## Cleansing Tasks

### Task 1: Normalize Entity Types [PRIORITY: HIGH]
**Status:** Not started
**Estimated Impact:** High - Reduces confusion, improves model learning

Steps:
1. Read unified_nodes.csv
2. Create type mapping dictionary
3. Apply case-insensitive normalization
4. Write normalized file
5. Validate counts match expected totals

Acceptance criteria:
- [ ] All entity types use consistent PascalCase
- [ ] No duplicate type names (case-insensitive)
- [ ] Total node count unchanged
- [ ] Validation report generated

### Task 2: Resolve Drug Names from DrugBank IDs [PRIORITY: HIGH]
**Status:** Not started
**Estimated Impact:** High - Enables human interpretation of results

Steps:
1. Download DrugBank vocabulary file (or use existing)
2. Extract DrugBank ID → Name mapping
3. Update drug nodes with resolved names
4. Keep original ID in separate column
5. Log drugs that couldn't be resolved

Acceptance criteria:
- [ ] >95% of DrugBank IDs resolved to names
- [ ] Original IDs preserved in `drugbank_id` column
- [ ] Unresolved drugs logged for review
- [ ] Validation: spot-check 20 random drugs

### Task 3: Resolve Gene Names from NCBI IDs [PRIORITY: MEDIUM]
**Status:** Not started
**Estimated Impact:** Medium - Improves interpretability

Steps:
1. Download NCBI gene_info file for human genes
2. Extract Gene ID → Symbol mapping
3. Update gene nodes with symbols
4. Keep original ID in separate column

Acceptance criteria:
- [ ] >90% of genes have human-readable symbols
- [ ] Original IDs preserved
- [ ] Gene symbols follow HGNC conventions

### Task 4: Unify Relation Types [PRIORITY: HIGH]
**Status:** Not started
**Estimated Impact:** High - Cleaner model training

Steps:
1. Create relation mapping table
2. Remove source prefixes (Hetionet::, DRUGBANK::, etc.)
3. Standardize to consistent naming convention
4. Merge semantically equivalent relations
5. Update unified_edges.csv

Target relation types:
```
TREATS / INDICATION
CONTRAINDICATES / CONTRAINDICATION
TARGETS (drug-gene)
INTERACTS_WITH (drug-drug, protein-protein)
ASSOCIATED_WITH (gene-disease)
PARTICIPATES_IN (gene-biological_process)
EXPRESSED_IN (gene-anatomy)
CAUSES (drug-side_effect)
```

Acceptance criteria:
- [ ] <50 unique relation types (currently 143+)
- [ ] No source prefixes in relation names
- [ ] Semantically equivalent relations merged
- [ ] Edge count unchanged

### Task 5: Deduplicate Entities [PRIORITY: MEDIUM]
**Status:** Not started
**Estimated Impact:** High but complex

Steps:
1. Identify potential duplicates by name similarity
2. Use external IDs to confirm duplicates
3. Create canonical entity mapping
4. Merge duplicate nodes
5. Update edge references
6. Log all merge decisions

Acceptance criteria:
- [ ] Duplicate detection report generated
- [ ] Merge decisions logged with confidence scores
- [ ] No orphaned edges after deduplication
- [ ] Validation: check known entities aren't incorrectly merged

### Task 6: Add External ID Mappings [PRIORITY: LOW]
**Status:** Not started
**Estimated Impact:** Medium - Enables cross-referencing

Steps:
1. For drugs: extract/add DrugBank, ChEMBL IDs
2. For diseases: map to MONDO, DOID
3. For genes: add NCBI, Ensembl, HGNC IDs
4. Store in structured columns

### Task 7: Generate Data Quality Report [PRIORITY: HIGH]
**Status:** Not started
**Estimated Impact:** Enables tracking improvement

Generate report with:
- Total nodes by type
- Total edges by relation
- Coverage statistics (% with names, % with external IDs)
- Duplicate detection summary
- Comparison to baseline (before cleaning)

## Output Files

After cleansing, create:
```
data/processed/
├── unified_nodes_clean.csv      # Cleaned nodes
├── unified_edges_clean.csv      # Cleaned edges
├── cleansing_log.json           # All transformations applied
├── data_quality_report.md       # Quality metrics
├── entity_mappings/
│   ├── drug_names.json          # DrugBank ID → Name
│   ├── gene_symbols.json        # Gene ID → Symbol
│   └── relation_mappings.json   # Old relation → New relation
└── validation/
    ├── spot_checks.json         # Manual validation samples
    └── before_after_stats.json  # Comparison metrics
```

## Success Metrics

| Metric | Before | Target |
|--------|--------|--------|
| Unique entity types | 30+ | <15 |
| Unique relation types | 143 | <50 |
| Drugs with real names | ~10% | >95% |
| Genes with symbols | ~40% | >90% |
| Entities with external IDs | 0% | >80% |

## Technical Notes

- All scripts should be idempotent (can re-run safely)
- Create backups before modifying files
- Log all transformations for auditability
- Use pandas for CSV processing
- Validate row counts before/after each step
