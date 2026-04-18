# Open-Cure Project Instructions

## Memory Management

**After each research session:** Update this file with key learnings before committing.

**Periodically ask:** "Should we prune working memory and move details to long-term storage?"
- Long-term storage: `docs/claude/` (patterns, confidence history)
- Archive location: `docs/archive/` (experiment history)
- Keep CLAUDE.md lean (<150 lines) for efficient context loading

## Session End Protocol

**ALWAYS end sessions by recommending the highest-ROI next steps:**
1. Analyze current model performance gaps
2. Identify improvement opportunities achievable with existing data
3. Rank by expected impact vs effort
4. Present top 2-3 actionable recommendations

**Constraints:** Prioritize approaches that don't require additional external data or GPU resources unless absolutely necessary.

## Scientific Reasoning Protocol (MANDATORY)

You are an execution engine, not a scientist. You lack epistemic discipline by default. Follow these rules to compensate.

### 1. Distrust Your Own Outputs
- **Never treat a computed metric as true without validation.** If you compute R@30 = X%, ask: "Does this make sense given the baseline? What could make this number wrong?"
- If a result looks surprisingly good, it is more likely a bug than a breakthrough. Investigate before reporting.
- Distinguish between "I measured X" and "X is true." Measurement errors, data leakage, and confounding are the default assumption until ruled out.

### 2. Check Preconditions Before Running Experiments
- Before committing to a hypothesis, spend 5-10 minutes checking whether the basic premise holds.
- **Ask: "What would need to be true for this hypothesis to work? Can I verify that cheaply first?"**

### 3. Run Positive Controls
- Before evaluating a new approach, verify that known-good drug-disease pairs (e.g., Metformin→T2D, Rituximab→MS) score highly. If your positive controls fail, your experiment is broken.
- Compare new results against the established baseline and explain any discrepancy.

### 4. Validate Against Published Evidence
- For any novel prediction or surprising result, search ClinicalTrials.gov and PubMed for corroborating or contradicting evidence BEFORE reporting the result as valid.

### 5. Stop Early When Evidence Contradicts
- If initial data (first 10% of an experiment) contradicts the hypothesis, STOP. Report the early negative signal and move on.

### 6. Question Your Methodology
- Before reporting results, ask: "Am I evaluating on training data?" / "Are my features derived from the labels?" / "Could this correlation be confounded?"

### 7. Report Uncertainty and Limitations
- Never write "improvement achieved" without quantifying confidence. Include effect size, sample size, and whether the improvement exceeds noise.

## Cloud GPU (Vast.ai)

**Current instance**: None | Balance: $4.41
**Skill**: Use `/vastai-gpu` for detailed GPU provisioning instructions

```bash
vastai search offers 'gpu_name in [RTX_3090, RTX_4090] disk_space >= 50 reliability > 0.95' -o 'dph_total' --limit 10
vastai create instance <OFFER_ID> --image pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel --disk 50
vastai show instances && vastai ssh-url <INSTANCE_ID>
./scripts/vastai_txgnn_setup.sh <PORT> <HOST>
vastai destroy instance <INSTANCE_ID>  # IMPORTANT: Destroy when done
```

## Models

**Default Model (use this):**
- `models/drug_repurposing_gb_enhanced.pkl` + Quad Boost ensemble
- Formula: `score × (1 + 0.01×overlap + 0.05×atc + 0.01×pathway) × (1.2 if chem_sim > 0.7 else 1.0)`
- Script: `scripts/evaluate_pathway_boost.py`

**Other Models:** `models/drug_repurposing_gb.pkl` (baseline 7.0%), `models/transe.pt`, `models/confidence_calibrator.pkl`

## Key Metrics

| Model | Per-Drug R@30 | Evaluation | Notes |
|-------|---------------|------------|-------|
| **kNN k=20 (original embeddings)** | **36.59% ± 3.90%** | Honest (5-seed) | Has treatment edge leakage |
| **kNN k=20 (no-treatment embeddings)** | **26.06% ± 3.84%** | **FAIR (5-seed)** | **Best fair transductive comparison** |
| **KEGG Pathway kNN** | **15.73% ± 1.82%** | **INDUCTIVE (5-seed)** | **Fair inductive comparison to TxGNN** |
| Node2Vec+XGBoost TUNED (disease holdout) | 25.85% ± 4.06% | Honest (5-seed) | md=6,ne=500,lr=0.1,alpha=1.0 |
| TxGNN | 6.7-14.5% | Inductive | Zero-shot on unseen diseases |

**DRKG CEILING:** 37% R@30 is the maximum achievable with DRKG-only approaches. Oracle ceiling is 60%.
**LEAKAGE:** Honest embeddings (no treatment edges): 26.06% vs 36.59%. 71.2% retained from indirect paths.
**COVERAGE BOTTLENECK (h909):** External-data pivots (h905 LINCS, h906 DrugBank) CANNOT be justified as coverage expansions. 95.4% of 1,534 MeSH mappings already have both DRKG embedding and GT drugs; of h901's 638 new mappings, 100% have an embedding and 98.1% have GT drugs. The 557 data-complete-but-non-evaluable mappings are blocked by pipeline hygiene (symptom filter, holdout sampling, name-resolution), not external data. h905/h906 must be re-justified as **precision** pivots on already-evaluable diseases (biologics, rare-disease features).

## Confidence Tiers (current)

| Tier | Holdout | Preds | Details |
|------|---------|-------|---------|
| GOLDEN | 87.1% ± 2.7% | 991 | See `docs/claude/confidence_system_history.md` |
| HIGH | 83.4% ± 4.0% | 1168 | |
| MEDIUM | 38.5% ± 3.6% | 914 | |
| LOW | 11.3% ± 0.5% | 9113 | |
| FILTER | 9.2% ± 0.5% | 8978 | |

**Rules:** Full-data is inflated; use HOLDOUT only. Always use `expanded_ground_truth.json` (19x more pairs).

## Reference Docs

- **Patterns & filters:** `docs/claude/patterns.md` (TransE, mechanism, filters, validated predictions)
- **Confidence history:** `docs/claude/confidence_system_history.md` (all h-number experiments)
- **Experiment archive:** `docs/archive/experiment_history.md`
- **TxGNN notes:** `docs/archive/txgnn_learnings.md`

## Data Sources

- Every Cure GT: `data/reference/everycure/indicationList.xlsx`
- Enhanced GT: `data/reference/expanded_ground_truth.json`
- DrugBank: `data/reference/drugbank_lookup.json`
- Disease mapping: `data/reference/disease_ontology_mapping.json`

## Production

**Deliverable:** `data/deliverables/drug_repurposing_predictions_with_confidence.xlsx` — 13,416 predictions
