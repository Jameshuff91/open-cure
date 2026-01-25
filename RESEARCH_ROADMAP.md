# Open-Cure Research Roadmap

## Mission
Use AI and open-source tools to accelerate drug repurposing research, making it accessible to non-scientists who want to contribute to finding treatments for rare diseases.

## Current Status (January 2026)

### What Works
- **37.4% Recall@30** on 700 diseases using GB + TransE embeddings
- **5.6x better** than Harvard's TxGNN on same evaluation
- **Clinical validation**: Dantrolene → Heart Failure (RCT P=0.034, 66% VT reduction)
- **72% biological plausibility** on manually reviewed predictions
- **74% precision** for small molecule predictions

### Known Limitations
- Biologics (-mab drugs) fail due to lack of target understanding
- Certain drug classes produce systematic false positives (TCAs for HTN, sympathomimetics for diabetes)
- 79 diseases still unmapped (10% of ground truth)

### Key Learnings
1. **Simple beats complex** - GB + embeddings outperforms sophisticated GNNs
2. **Disease coverage is the bottleneck** - MESH mapping expansion was our biggest win
3. **Similarity features leak** - "Similar to known treatments" encodes the label
4. **Validation matters** - Easy to fool yourself with training metrics

---

## Research Priorities

### Priority 1: Actionable Deliverables ✅ COMPLETE
**Goal**: Produce curated predictions Every Cure can actually use

- [x] Generate high-confidence novel predictions (3,834 clean predictions)
- [x] Filter out known false positive patterns (confidence filter)
- [x] Add evidence/rationale for each candidate
- [x] Format for clinical review (JSON + CSV + summary)
- [ ] Submit to Every Cure for feedback

**Output**: `data/deliverables/` - 307 FDA-approved drug predictions, 3,834 total clean predictions

### Priority 2: Make It Contributor-Friendly ✅ COMPLETE
**Goal**: Enable others to join and contribute

- [x] Improve README with clear setup instructions
- [x] Add "good first issues" in README
- [x] Create simple query interface (CLI): `scripts/query.py`
- [x] Document methodology in README
- [x] Add contribution guidelines: `CONTRIBUTING.md`

### Priority 3: Add Drug-Target Features
**Goal**: Fix biologic prediction failures

- [ ] Download DrugBank target data (academic license)
- [ ] Create target-disease compatibility features
- [ ] Add target family features (kinase, GPCR, etc.)
- [ ] Retrain model with new features
- [ ] Re-evaluate biologic predictions

### Priority 4: Expand Disease Coverage
**Goal**: Evaluate more diseases

- [ ] Map remaining 79 diseases via alternative ontologies
- [ ] Try DOID → MONDO → MESH pathway
- [ ] Add manual mappings for high-priority diseases
- [ ] Re-run evaluation with expanded coverage

### Priority 5: Clinical Trial Monitoring
**Goal**: Ongoing validation of predictions

- [ ] Build ClinicalTrials.gov API integration
- [ ] Flag predictions that match active trials
- [ ] Track trial completions for validation
- [ ] Create automated monitoring pipeline

### Priority 6: External Data Integration
**Goal**: Improve model with more biological signal

- [ ] ChEMBL bioactivity data
- [ ] PubMed co-occurrence features
- [ ] OMIM disease genetics
- [ ] Pathway databases (Reactome, KEGG)

---

## Completed Work

### Phase 1: Baseline Model (January 2026)
- [x] Train GB classifier on TransE embeddings
- [x] Evaluate on Every Cure ground truth
- [x] Compare against TxGNN baseline

### Phase 2: MESH Expansion (January 2026)
- [x] Expand disease-to-MESH mappings via web search
- [x] Increase coverage from ~10% to 90%
- [x] Achieve 37.4% R@30 (up from ~7%)

### Phase 3: Validation (January 2026)
- [x] Manual literature validation of 54 predictions
- [x] Identify clinical trial validation (Dantrolene)
- [x] Document false positive patterns
- [x] Create confidence filter

### Phase 4: Failed Experiments (January 2026)
- [x] Similarity features experiment (FAILED - data leakage)
- [x] TxGNN fine-tuning (FAILED - catastrophic forgetting)
- [x] Document learnings for future reference

### Phase 5: Deliverable Generation (January 2026)
- [x] Create clean prediction pipeline with proper drug name filtering
- [x] Generate 307 FDA-approved drug predictions
- [x] Generate 3,834 total clean predictions (score ≥0.85)
- [x] Package with README and documentation
- [x] Verify model performance (37.4% R@30 confirmed)

---

## Key Lessons Learned

1. **Simple models can outperform complex ones** - Our GB + TransE beats Harvard's GNN by 5.6x
2. **Data quality > model complexity** - MESH mapping expansion was 5x more impactful than any model change
3. **Similarity features are a trap** - They encode the label and cause catastrophic test performance
4. **Validation is essential** - 0.96 training AUROC meant nothing when test R@30 was 0%
5. **Clinical validation exists** - Dantrolene prediction was independently confirmed by RCT
6. **Ground truth is incomplete** - FDA-approved uses (Empagliflozin/HF) appear as "novel"
7. **Biologics need special handling** - Antibody-target specificity isn't captured by embeddings

---

## How to Contribute

### If you have compute
- Run experiments from the roadmap
- Help with MESH/ontology mapping
- Train alternative embedding models

### If you have domain expertise
- Validate predictions in your disease area
- Identify false positive patterns
- Suggest feature engineering ideas

### If you have engineering skills
- Build web interface for queries
- Improve documentation
- Set up CI/CD pipeline

### If you want to learn
- Start with the README
- Run the evaluation scripts
- Pick a "good first issue"

---

## Links

- **Every Cure**: https://everycure.org
- **Dr. Fajgenbaum's TED Talk**: https://www.youtube.com/watch?v=sb34MfJjurc
- **Every Cure GitHub**: https://github.com/everycure
- **This Repo**: https://github.com/jimhuff/open-cure
