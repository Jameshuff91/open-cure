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
**H900 FALLBACK REMOVED (h903 → h931):** Mechanism-only fallback at the old `production_predictor.py:4634` was verified dead code (0/1034 triggers at full data, 0/207 at seed-42 holdout) because `self.train_diseases` is pre-filtered to require GT+embeddings. h931 deleted ~75 lines (lines 4663-4737); the `if not drug_scores:` branch now sets a coverage warning and exits. `scripts/test_mechanism_fallback.py` rewritten as a regression marker that asserts no production prediction carries the removed sub_reason tags (`mechanism_only_fallback`, `mechanism_fallback_literature_strong`, `mechanism_fallback_literature_moderate`). When the fallback was *forced* in h903, target-overlap-only ranking scored 5.96% prec@30 (below FILTER 9.2% / LOW 11.3%) with 0% median per-disease R@30 — confirming the removal was safe.
**H939 BIOLOGIC TARGET-OVERLAP (VALIDATED):** Target-overlap IS a biologic-specific signal even though h912/h916 showed it is not general. Restricting the candidate pool to biologics (266 of 11,656 drugs; USAN suffix + keyword proxy) gives **bio_r30=26.5%** overall vs **sm_r30=5.5%** — 4.8x ratio. Diseases with ≥5 biologic GT (n=35): mean bio_r30=40.7%, median 41.7%. Four categories clear the n≥10, ratio≥3x bar: CV 11.3x, hematological 6.3x, 'other' 4.6x, autoimmune 3.3x. Cancer narrowly misses at 2.74x (shared cancer-gene vocabulary per h916). **Caveat:** precision@30 is similar to SM (2.24% vs 2.63%) because the 266-drug pool is small.
**H940 BIOLOGIC FUSION (INVALIDATED):** Fusing target-overlap into kNN for biologic candidates in a mixed biologic+SM pool HURTS biologic R@30 at every alpha. 5-seed holdout (k=20, no MinRank): baseline bio_r30=**31.42% ± 2.32%** (pure kNN); alpha=0.3→0.9 produced 22.58%/25.47%/25.51%/27.42% (−8.83 to −4.00pp). h939's 3-11x within-biologic-pool ratio does NOT transfer to a unified ranking — kNN already encodes which biologics are used in similar diseases. Precision silver lining: alpha=0.9 bio_p30 17.58%→21.32% (+3.74pp) — fusion tightens the biologic top-30 but misses rare biologic hits absent from neighbor GT. **Secondary finding:** baseline bio_r30=31.4% is ABOVE the 27.3% biologic-failure baseline quoted as motivation for h906/h921/h924 — filed as h951 reality-check before further biologic-pivot investment. Use target-overlap as annotation / audit signal for biologics, not as a re-ranker. Follow-ups: h948 (coverage stratification), h949 (zero-overlap safety filter), h950 (fusion-as-annotation), h951 (baseline reality-check).
**H951 BIOLOGIC FAILURE NARRATIVE DISSOLVED (VALIDATED):** Production-pipeline 5-seed holdout (`predictor.predict()` top-30 by rank, expanded GT, 80/20 disease split, h393 train-only GT structures): bio_r30=**27.06% ± 3.12%**, overall_r30=**16.39% ± 1.10%**, sm_r30=15.91% ± 1.12%, bio_p30=6.37% ± 2.06%. **Biologics OUTPERFORM overall by +10.67pp.** The historical 27.3% biologic R@30 quoted in research_spec.md / RESEARCH_ROADMAP.md / h906/h920/h921/h924 rationales was numerically correct (we measured 27.06%) — but the comparison "27.3% biologics vs 41.8% overall" combined two incompatible eval frameworks. On apples-to-apples production holdout, biologics are not the failure class. Per-category bio:sm ratios where biologics dominate: gastrointestinal 7.9x, hematological 7.4x, metabolic 3.8x, neurological 3.2x, musculoskeletal 3.0x, respiratory 2.7x, autoimmune 1.9x. Inverted in tiny-n strata only (infectious 0.17 n=2.2, psychiatric 0 n=1.5). **Implications:** h906 (DrugBank), h920 (PubMedBERT), h921 (ESM2), h924 (LINCS VAE) all need re-justification — recall motivation is gone. Precision is the genuine remaining gap (h953). Production pipeline loses 4.4pp bio_r30 vs h940 plain kNN (31.42→27.06) and 3.91pp overall_r30 vs h940 plain kNN (20.30→16.39) — diagnose in h952 (suspect SELECTIVE category boost or tier safety filters). research_spec.md 41.8% baseline must be corrected (h954). **NOTE:** h952 (below) identifies the 4pp gap root cause — the h951 numbers are artificially suppressed by a find_disease_id name-resolution bug; corrected seed-42 post-fix numbers are overall_r30=18.72%, bio_r30=31.96%. h958 re-runs the 5-seed baseline post-fix.
**H952 FIND_DISEASE_ID NAME-RESOLUTION BUG (VALIDATED & FIXED):** The ~4pp production-vs-plain-kNN recall regression (h940 20.30% → h951 16.39%) is NOT caused by SELECTIVE_BOOST, supplements, MinRank, or tier rules. **Root cause:** `DrugRepurposingPredictor.find_disease_id(disease_name)` silently returned `None` for 40 of 203 holdout diseases (19.7%) at seed 42, producing zero predictions for those diseases. 668 of 1146 `disease_names` entries (58%) were absent from `mesh_mappings`; the fuzzy matcher covered most but not all. Failing names: `cerebrotendinous xanthomatosis`, `squamouscell carcinoma`, `ewings sarcoma`, `autoimmune haemolytic anaemia`, `obstructive sleep apnoea` (British spellings, missing spaces, possessive drops). **Diagnostic:** 40 name-fail diseases contribute −16.28pp each → −3.22pp of the −3.29pp observed seed-42 regression; the other 162 diseases show production == plain kNN (delta −0.08pp). Per-stratum: `neither boost nor supp` n=125 delta=0.00pp, `supp_only` (GI) n=7 delta=0.00pp, `boost_only` n=17 delta=+1.41pp (SELECTIVE_BOOST actually helps!), `boost+supp` (neurological) n=13 delta=−2.85pp (supplement hurts — filed as h960). **Fix:** added a reverse-index fallback in `find_disease_id` (src/production_predictor.py:4586) — lowercased-name lookup against `disease_names` before fuzzy matching. **Post-fix seed 42:** overall_r30 15.42→18.72% (+3.30pp, matches plain kNN 18.71%); bio_r30 27.23→31.96% (+4.73pp, exceeds plain kNN 30.17%); sm_r30 14.57→17.81% (+3.24pp). **Implications:** h951 baselines are artificially low; h958 re-runs 5-seed post-fix. Any prior eval that called `predict(disease_name)` may be suspect — h959 audits call sites, h962 regenerates the 13,416-row deliverable. h961 proposes principled US↔UK and hyphenation aliasing for mesh_mappings.

## Confidence Tiers (current — post-h904+h908)

| Tier | Holdout | Preds | Details |
|------|---------|-------|---------|
| GOLDEN | 83.7% ± 1.3% | 94/seed | See `docs/claude/confidence_system_history.md` |
| HIGH | 78.5% ± 2.4% | 230/seed | |
| MEDIUM | 42.1% ± 3.0% | 200/seed | |
| LOW | 11.2% ± 1.1% | 2062/seed | |
| FILTER | 8.2% ± 0.6% | 2196/seed | |

**h904 (VALIDATED):** 10 overfitted rules demoted after h393 per-rule audit —
cancer_targeted_therapy LOW→FILTER, cv_pathway_comprehensive MEDIUM→LOW, hierarchy_uti
off-GOLDEN, hierarchy_multiple_sclerosis/lupus/asthma HIGH→MEDIUM, hierarchy_skin_infection
HIGH→LOW, hierarchy_diabetes LOW→FILTER, highly_repurposable LOW→FILTER, metabolic rescue
LOW→FILTER. Net: MEDIUM +1.1pp, GOLDEN +0.4pp, no regressions.
**h908 (VALIDATED):** 45-name MeSH C23 symptom blocklist removes 300 deliverable rows
(0 GOLDEN/HIGH, 44 MEDIUM antibiotic→symptom artifacts, 96 LOW, 160 FILTER). All tier
shifts ≤1.5σ vs h904-demoted baseline. GOLDEN/HIGH nominal drops are population-shuffle
from evaluable-disease pool 1034→1016 (zero rows actually removed from GOLDEN/HIGH).

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
