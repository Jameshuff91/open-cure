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
**H952 FIND_DISEASE_ID NAME-RESOLUTION BUG (VALIDATED & FIXED):** The ~4pp production-vs-plain-kNN recall regression (h940 20.30% → h951 16.39%) is NOT caused by SELECTIVE_BOOST, supplements, MinRank, or tier rules. **Root cause:** `DrugRepurposingPredictor.find_disease_id(disease_name)` silently returned `None` for 40 of 203 holdout diseases (19.7%) at seed 42, producing zero predictions for those diseases. 668 of 1146 `disease_names` entries (58%) were absent from `mesh_mappings`; the fuzzy matcher covered most but not all. Failing names: `cerebrotendinous xanthomatosis`, `squamouscell carcinoma`, `ewings sarcoma`, `autoimmune haemolytic anaemia`, `obstructive sleep apnoea` (British spellings, missing spaces, possessive drops). **Diagnostic:** 40 name-fail diseases contribute −16.28pp each → −3.22pp of the −3.29pp observed seed-42 regression; the other 162 diseases show production == plain kNN (delta −0.08pp). Per-stratum: `neither boost nor supp` n=125 delta=0.00pp, `supp_only` (GI) n=7 delta=0.00pp, `boost_only` n=17 delta=+1.41pp (SELECTIVE_BOOST actually helps!), `boost+supp` (neurological) n=13 delta=−2.85pp (supplement hurts — filed as h960). **Fix:** added a reverse-index fallback in `find_disease_id` (src/production_predictor.py:4586) — lowercased-name lookup against `disease_names` before fuzzy matching.
**H958 POST-FIX 5-SEED BASELINE (VALIDATED):** 5-seed production holdout with h952 fix in place: overall_r30=**19.49% ± 1.42%** (was 16.39%, +3.10pp); bio_r30=**30.31% ± 3.57%** (was 27.06%, +3.25pp); sm_r30=**18.99% ± 1.34%** (was 15.91%, +3.08pp); bio_p30=5.61% ± 1.97%. Biologics outperform overall by +10.82pp (strengthening h951 dissolution). Production bio_r30 is now +3.01pp above the research_spec 27.3% historical baseline (was -0.24pp below). Residual gap vs h940 plain kNN (bio 31.42%) is only -1.11pp, consistent with the h960 neurological supplement/boost interaction (-2.85pp on n=13). **Implications:** any prior eval that called `predict(disease_name)` may be suspect — h959 audits call sites, h962 regenerates the 13,416-row deliverable. h961 proposes principled US↔UK and hyphenation aliasing for mesh_mappings. h940 plain kNN remains the fair in-pipeline recall ceiling.
**H957 ZERO-OVERLAP BIOLOGIC FILTER (INVALIDATED — global form):** 5-seed h393 holdout with h949 implementation: drop biologic predictions where `drug_targets ∩ disease_genes == 0`. Global filter MEETS the +3pp ship target on bio_p30 (5.64%→8.85%, +3.22pp) but VIOLATES the -2pp recall cap by 7.8x on bio_r30 (30.31%→14.78%, **-15.54pp**). 91 biologic GT hits per 5-seed run dropped from top-30. sm_r30 ~flat (+0.41pp), overall_r30 ~flat (-0.20pp) — filter only touches biologics as designed. **Per-category split (sorted by Δp30):** musculoskeletal +26.1/+0 (n=6 too small to ship alone), hematological +25.9/-25.0, CV +13.2/-8.1, autoimmune +9.3/-28.6, metabolic +9.1/-19.1, **cancer +2.6/-2.1 (n=115/54, essentially within ship cap)**, ophthalmic +1.2/-5.6, neurological -5.8/-21.2, respiratory -6.4/-58.3, immunological -4.8/-100. **Mechanistic pattern:** filter is mechanism-aware in oncology (TCGA-dense disease_genes captures tumor-cell receptors that biologics target — HER2, VEGF, CD20, PD1) but mechanism-blind in inflammatory disease (anti-TNF/anti-IL6/anti-IL17/anti-IL23 bind cytokines whose genes are absent from disease etiology gene sets — HLA, complement, structural). **Tier-shift cost:** 7 GOLDEN, 40 HIGH, 5 MEDIUM, 1087 LOW, 2797 FILTER demotions across 5 seeds. The 47 GOLDEN/HIGH demotions are the highest-cost potential false-positives. **Pivots filed:** h965 (cancer-restricted variant — easy ship test), h966 (KEGG-pathway-extended disease_genes for inflammatory rescue), h967 (GOLDEN/HIGH zero-overlap audit — GT-gap vs genuine FP). Use target-overlap as audit signal, not as a global biologic demoter.
**H960 NEURO SUPPLEMENT IS BENIGN FOR R@30 (INVALIDATED):** 5-seed h393 holdout. Pass A = production. Pass B = `_supplement_neurological_predictions` monkey-patched to no-op. Result: neuro_overall_r30 Δ = +0.00pp on every seed (n=50 neuro diseases summed); non-neuro control Δ = +0.00pp (no leakage); 0/50 neuro diseases helped or hurt. **Mechanism:** the supplement function early-returns at `if not missing_drugs:` (production_predictor.py:4346) without re-sorting when no class-matched drug is missing from kNN top-N. On the holdout neuro pool, kNN already covers all class-matched drugs in the top-30, so the supplement is a no-op. **Implication:** the h952 −2.85pp boost+supp neurological regression is NOT caused by the supplement; the search shifts to SELECTIVE_BOOST itself behaving differently on neurological than on metabolic/renal/hematological/respiratory/immunological (the boost_only stratum that gave +1.41pp). Filed h972 (boost ablation on neuro) and h973 (per-disease set-diff localization on the n=13 boost+supp stratum). **Methodology lesson:** ablate pipeline components independently before naming a culprit — h173's name suggested the supplement was active, but its short-circuit makes it inert on the holdout. h171's class-coverage gap was measured on FULL data, not holdout.
**H965 CANCER-RESTRICTED BIOLOGIC FILTER (INVALIDATED — global lift too small):** 5-seed h393 holdout, h957 filter restricted to `category=='cancer'`. Cancer-cohort: bio_p30 7.51→10.04 (+2.54pp), bio_r30 24.65→22.41 (-2.24pp), overall +0.40pp — reproduces h957's cancer slice (+2.6/-2.1) within noise, confirming the per-category result. Non-cancer cohort Δ=0.00pp on every metric (clean filter isolation control). **Global aggregate:** bio_p30 5.64→5.92 (+0.28pp, BELOW +0.5pp ship target), bio_r30 30.31→29.86 (-0.46pp, within cap), overall +0.04pp. Cancer is ~12% of evaluable diseases (~23/200/seed), so +2.5pp lift on ~70-90 dropped preds dilutes to +0.3pp at the global tier-system level. **Implication:** the biologic-overlap signal is genuine in oncology but too narrow to register as a tier rule. Right surface area is per-prediction annotation (h968 — `biologic_low_mechanism_evidence` deliverable column) or further-narrowed subclass rule (h969 — anti-HER2/VEGF/CD20/checkpoint canonical-target biologics where target identity == MoA).
**H959 PREDICT() CALL-SITE AUDIT (VALIDATED):** End-to-end impact of the h952 find_disease_id bug measured. 287 of 1146 disease_names (25.0%) failed pre-fix resolution; post-fix 0 fail. In h393 evaluator pool (1011 diseases = ground_truth ∩ embeddings), 213 (21.1%) were silently zero-predicted pre-fix → ~43/202 per-seed holdout fails expected, consistent with h952's seed-42 observation (40). Across the codebase 111 scripts call `predictor.predict(disease_name)`; only 1 (h771) calls with disease_id. **Deliverable NOT affected:** `scripts/generate_production_deliverable.py` iterates disease IDs and calls its own `knn_predict(disease_id, …)` — bypasses predict() entirely. h939/h940 also bypass via direct disease_id kNN. h904/h908/most h393-derived validations (pre-h952): affected but directional conclusions remain valid — magnitudes dampened by ~3pp recall suppression. **Still valid unchanged:** h939/h940 pure-kNN numbers, h958 post-fix 5-seed, the 13,416-row XLSX. **Needs re-verification for magnitude:** tier-precision numbers derived from h393 pre-h952. Full audit at `data/analysis/h959_predict_audit.json`.
**H975 PER-RULE STATUS DIFF PRE- vs POST-H952 (VALIDATED):** Diffing h904_h393_with_demotions.json (pre-fix) vs h393_holdout_validation.json (post-fix): 30 of 80 rules changed classification. Actionable candidates: (1) **infectious_hierarchy_uti** [GOLDEN] flipped OVERFITTED?→GENUINE 0%→90.9% at n=11/seed — h904 off-GOLDEN demotion may be reversible (h977). (2) **immunological_medium_demotion** [LOW] flipped degraded→OVERFITTED? 8.3%→2.6% at n=23, now 7.4pp below LOW baseline 10.0% — candidate for LOW→FILTER demotion (h978). (3) **cancer_same_type_mech_rank10** [HIGH] flipped degraded→OVERFITTED? 76.1%→72.5% at n=42, 1.8σ below HIGH mean 80.0% — borderline HIGH→MEDIUM call (h979). Two rules with Δ−100pp (metabolic_hierarchy_lipid, infectious_hierarchy_sepsis) had post_mean_n=0 — untestable. Healthy rehabilitations: transe_medium_promotion +16.4pp, literature_high_demotion +4.8pp, default_freq10_nomech_r1_5 +1.5pp all flipped degraded→GENUINE. Raw: `data/analysis/h975_rule_status_diff.json`.
**H980 h771 RE-RUN POST-H963 (VALIDATED):** h963 surfaced that `scripts/h771_literature_coverage_analysis.py:101` was silently broken pre-fix (live check: `find_disease_id("drkg:Disease::MESH:D014141")` → None). h771's log file was 0 bytes — confirming no output ever captured. Post-h963 rerun produces well-formed tier precisions on h771's 1078-disease pool with seeds [42,123,456,789,1337]: GOLDEN 90.3%±3.1%, HIGH 81.9%±2.9%, MEDIUM 43.0%±4.5%, LOW 14.8%±0.9%, FILTER 12.6%±1.0% (higher than h964's 1011-pool because h771 drops the `∩ embeddings` filter and uses seed 1337 instead of 2024). Literature-modified MEDIUM splits: MODERATE 36.7% (n=65), NOT_ASSESSED 46.2% (99), NO_EVIDENCE 44.9% (21), WEAK 57.5% (16); MEDIUM→LOW(lit_weak) 19.1% (53). **Two calibration signals:** (a) `NOT_ASSESSED` at 46.2% > MEDIUM avg 43.0% — literature-cache absence is a coverage gap, not a negative signal (h985); (b) `MEDIUM→LOW(lit_weak)` at 19.1% sits 4.3pp above LOW (14.8%), possibly over-demoted (h984). Also filed h983 for an in-script disease_groups leak (h771:73/153 restores drug_disease_groups at end but never recomputes during the holdout loop). CLAUDE.md h731/h768 memory entries reference cache-level (GT-independent) numbers so they remain trustworthy. Raw: `data/analysis/h980_h771_rerun.txt`.
**H963 PREDICT(DISEASE_ID) FAST-PATH (VALIDATED):** `predict()` now detects inputs with `drkg:Disease::` prefix and skips `find_disease_id` entirely (src/production_predictor.py:4636–4645), looking up the canonical disease_name from `self.disease_names`. h393 evaluator migrated to pass disease_id directly (scripts/h393_holdout_tier_validation.py:167–175). **Zero regression:** 5-seed h393 post-h963 tier precisions match h964 post-fix to the tenth (GOLDEN 78.5%±6.0%, HIGH 80.0%±3.3%, MEDIUM 39.9%±5.1%, LOW 10.0%±0.7%, FILTER 6.8%±0.7%); per-seed values identical. Smoke test (scripts/h963_smoke_test.py): 25/25 diseases produce identical prediction lists between canonical-name path and id-path. **Side finding:** h771 (`scripts/h771_literature_coverage_analysis.py:101`) was calling `predict(disease_id)` pre-h963 — verified broken on `drkg:Disease::MESH:D014141`: `find_disease_id` returned None (disease_id strings match neither mesh_mappings keys nor disease_names values), so predict() produced ZERO predictions for every call. h771 literature-coverage output may have been empty; filed h980 to re-run and diff. h952 remains the safety net for legitimate name-only callers; h963 is the fast, deterministic path for callers that already hold a disease_id.
**H964 POST-FIX TIER PRECISION RE-RUN (VALIDATED):** 5-seed h393 holdout post-h952-fix: GOLDEN **78.5% ± 6.0%** (was 83.7% ± 1.3%, **−5.2pp**, std 4.6x wider), HIGH **80.0% ± 3.3%** (was 78.5% ± 2.4%, **+1.5pp**), MEDIUM **39.9% ± 5.1%** (was 42.1% ± 3.0%, **−2.2pp**), LOW **10.0% ± 0.7%** (was 11.2% ± 1.1%, −1.2pp), FILTER **6.8% ± 0.7%** (was 8.2% ± 0.6%, −1.4pp). All five tiers drifted |Δ|>1pp. Per-seed tier totals grew substantially (GOLDEN +7, HIGH +33, MEDIUM +65, LOW +502, FILTER +670 per seed — ~1277 extra preds/seed), consistent with ~40 previously-zero-predicted diseases/seed × ~32 preds/disease. **Tier ordering:** GOLDEN (78.5) < HIGH (80.0) at mean, but CIs overlap (GOLDEN 72.5–84.5 vs HIGH 76.7–83.3) — inversion is within noise. GOLDEN std widened from 1.3→6.0 because newly-resolving diseases vary per split. **h-number re-verification:** h904 (rule demotions net MEDIUM +1.1pp, GOLDEN +0.4pp) and h908 (MeSH C23 blocklist, all tier shifts ≤1.5σ) measured pre vs post within the SAME buggy evaluation framework, so their relative Δs survive the bug-fix shift. No prior validation decisions flip. CLAUDE.md tier table updated. Raw results: `data/analysis/h393_holdout_validation.json` + `data/analysis/h964_h393_postfix_run.txt`.

## Confidence Tiers (current — post-h904+h908, post-h952-fix per h964)

| Tier | Holdout | Preds | Details |
|------|---------|-------|---------|
| GOLDEN | 78.5% ± 6.0% | 101/seed | h964 post-fix |
| HIGH | 80.0% ± 3.3% | 263/seed | h964 post-fix |
| MEDIUM | 39.9% ± 5.1% | 265/seed | h964 post-fix |
| LOW | 10.0% ± 0.7% | 2564/seed | h964 post-fix |
| FILTER | 6.8% ± 0.7% | 2866/seed | h964 post-fix |

**Legacy (pre-h952-fix, for reference):** GOLDEN 83.7%±1.3%, HIGH 78.5%±2.4%, MEDIUM 42.1%±3.0%, LOW 11.2%±1.1%, FILTER 8.2%±0.6%. Deltas (post-fix − pre-fix): GOLDEN −5.2pp, HIGH +1.5pp, MEDIUM −2.2pp, LOW −1.2pp, FILTER −1.4pp. All |Δ|>1pp, tier ordering preserved within noise (GOLDEN/HIGH confidence intervals overlap: GOLDEN 72.5–84.5, HIGH 76.7–83.3). See h964.

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
