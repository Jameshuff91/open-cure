# Open-Cure Project Instructions

## Research Goal

**Build a hybrid drug-repurposing system that meets or exceeds published SOTA on standard metrics, with calibrated confidence tiers validated by expert review.**

Current state (2026-04-19):
- kNN on Node2Vec embeddings: ~26% per-drug R@30 honest, ~37% with treatment-edge leakage
- Published SOTA (TxGNN Nature Med 2024): AUPRC 0.913 standard eval, +49.2% on zero-shot indications
- Goal: match and beat SOTA on Hits@K, MRR, AUPRC, AUROC, and R@30 simultaneously

Target architecture (pivot committed 2026-04-19):
1. **Supervised GNN on DRKG** (h1200) — replace unsupervised Node2Vec with GNN trained on treatment edges as explicit labels. Expected: 40-50% R@30.
2. **LINCS L1000 reverse-connectivity** (h1201) — orthogonal transcriptomic signal. Expected: +5-10pp in fusion.
3. **Expert-calibrated confidence tiers** (h1203) — classifier trained on Ryland Mortlock's blinded review labels. Decouples calibration from embedding choice.
4. **Hybrid fusion** (h1202) — primary Nature-paper claim.

Evaluation standard: **every experiment reports all five metrics** — R@30 per-drug, Hits@K, MRR, AUPRC, AUROC. h1199 is the prerequisite multi-metric benchmark.

Hypothesis labels: **Recall lever** (Paths A, B), **Calibration lever** (Path C), **Infrastructure** (h1199, h907), **Complementary** (lower priority).

**Honest probability:** Path A + Path B fused, ~55% chance of reaching 40% R@30, ~25% of 50%, ~8% of 55%+.

**See also:** `research_loop/prompts/research_spec.md`, `~/.claude/projects/-Users-jimhuff/memory/project_open_cure_pivot_37_60.md`.

## Memory Management

**After each research session:** Update this file with key learnings before committing.

**Periodically prune:** move detail to `docs/claude/` (patterns, confidence history) and archive to `docs/archive/`. Keep CLAUDE.md <150 lines for efficient context loading.

## Session End Protocol

**ALWAYS end sessions by recommending the highest-ROI next steps:**
1. Analyze current model performance gaps
2. Identify improvement opportunities achievable with existing data
3. Rank by expected impact vs effort
4. Present top 2-3 actionable recommendations

**Constraints:** Prioritize approaches that don't require additional external data or GPU resources unless absolutely necessary.

## Scientific Reasoning Protocol (MANDATORY)

You are an execution engine, not a scientist. Follow these rules to compensate.

1. **Distrust your own outputs.** Never treat a computed metric as true without validation. Surprisingly-good results are more likely bugs than breakthroughs.
2. **Check preconditions before running experiments.** Spend 5-10 min verifying the basic premise holds. Ask: "what would need to be true for this hypothesis to work?"
3. **Run positive controls.** Verify known-good drug-disease pairs (Metformin→T2D, Rituximab→MS) score highly before trusting anything else.
4. **Validate against published evidence.** Search ClinicalTrials.gov and PubMed for corroborating or contradicting evidence BEFORE reporting a novel result.
5. **Stop early when evidence contradicts.** If first 10% of an experiment contradicts the hypothesis, STOP and report the early negative signal.
6. **Question your methodology.** "Am I evaluating on training data? Are features derived from labels? Could this correlation be confounded?"
7. **Report uncertainty and limitations.** Never write "improvement achieved" without effect size, sample size, and noise comparison.

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

**Ceilings:** DRKG-only ceiling 37% R@30; Oracle ceiling 60%. Honest-embedding leakage: 26.06% vs 36.59% retains 71.2% via indirect paths.

**Baselines (post-h952-fix, h958):** overall_r30 19.49%±1.42%, bio_r30 30.31%±3.57%, sm_r30 18.99%±1.34%. Biologics OUTPERFORM overall by +10.82pp — h951 dissolved the "biologic failure narrative". h940 plain kNN bio_r30=31.42% remains the fair in-pipeline recall ceiling.

**Coverage (h909):** 95.4% of 1,534 MeSH mappings already have DRKG embedding + GT drugs. External-data pivots (LINCS, DrugBank) must be re-justified as **precision** pivots, not coverage expansions.

**Closed pivot families (do NOT re-propose):**
- In/out biologic-filter precision (h953, h957, h965, h990, h995, h1002) — structurally precision-neutral because kNN rank 31-200 tail is uniform-quality.
- Score-perturbation in-window re-rank (h1000, h1005) — bottleneck is signal density (22% match=True), not shift magnitude.
- SM low-freq no-mech tier demote (h996) — 99.3% already at LOW/FILTER; would destroy 94% precision HIGH survivors.

**h1100 VALIDATED (2026-04-19):** Known-indication carve-out on 3 FILTER-demoting rules. `is_known_indication = drug_id in ground_truth.get(disease_id)` exempts paper-crown-jewel indications (metformin→T2D FILTER→LOW, tetrabenazine→HD FILTER→LOW, trastuzumab→breast FILTER→MEDIUM). Known-indication FILTER@rank≤30 misfires: 1,366→779 (-43%). FILTER precision 6.8→6.1% (within 1pp gate).

**h1101 VALIDATED (2026-04-19):** Dantrolene paper crown-jewel is a cohort/outcome conflation — model predicts VT rank 34 FILTER (rank_over_20), NOT heart failure. Section 3.5 row E1 errata shipped. No code change needed.

**h1107 VALIDATED (2026-04-19, paper credibility):** Full Section 3.5 audit — only 1 of 5 claims (Rituximab→MS) corroborated by current model. Dantrolene, Lovastatin, Pitavastatin, Empagliflozin all ABSENT from their claimed disease's top predictions. E2 errata shipped with recommended Option A revision (collapse table to 1 row + supplementary "prior art").

**h1103 VALIDATED (2026-04-19):** 543 rank>20 known-indication misfires clustered. 74% mech=False (canonical h1200 target — newly-approved drugs with narrow DRKG edges). 62% at ranks 21-25 (small ranker lift recovers most). Top cluster: biologic_mab × autoimmune (n=28, natural Ryland packet).

**h1199 VALIDATED (2026-04-19, infrastructure):** Shipped `scripts/clean_embedding_benchmark.py` — tier-free 5-seed benchmark reporting all five metrics. Node2Vec baseline: **R@30=19.55%±1.18%, MRR=0.0284, AUPRC=0.0569, AUROC=0.5766** on 1,011 eligible diseases; matches h958 overall_r30=19.49% (independent pipeline → convention correct). GraphSAGE loses on all five metrics (R@30=8.17%, -11.4pp). Every h1200/h1201 run drops into the same table via `OPEN_CURE_EMBEDDINGS_PREFIX`. Per-category spread: endocrine 41% / ophthalmic 38% (high) → psychiatric 10% / hematological 10% (low).

**h1212 VALIDATED (2026-04-19, infrastructure + recalibration):** Extended h1199 to 4 embeddings + `--restrict-to-embedding` flag. DRKG-only ceilings: R@30 ≤ 19.55% (node2vec), MRR ≤ 0.0284 (node2vec), AUPRC ≤ 0.0584 (fastrp), AUROC ≤ 0.5790 (fastrp). **FastRP nearly matches Node2Vec** (R@30 18.79% vs 19.55%, slightly beats on AUPRC/AUROC) → h1215 FastRP fusion candidate. **Treatment-edge leakage is ~50%, not ~29%** (apples-to-apples 850 diseases: full Node2Vec 17.09% → no_treatment 8.46% = 49.5% retained vs CLAUDE.md's 71.2% claim) → h1214 reconcile before external citation.

**h1218 VALIDATED (2026-04-19, falsifies disagreement-hypothesis):** Per-disease decomposition of h1215's +1.32pp concat_l2 gain over Node2Vec, across 1,002 (seed×holdout) rows. Net Δ = +1.327pp (cross-validates h1215 exactly). **Pearson(Δ R@30, 1 − Jaccard top-20 neighbours) = −0.077 (p=0.014) — gain is NEGATIVELY correlated with embedding disagreement.** Jaccard quartile gains: Q1_low +0.88pp, Q2 +0.60pp, Q3 +1.41pp, Q4_high +2.63pp — fusion is score-smoothing, not disagreement-exploitation. **Category splits: gainers musculoskeletal +6.2pp, cancer +4.0pp, GI +3.8pp, hematological +2.5pp, psychiatric/infectious +2.1pp; REGRESSORS endocrine −5.4pp, cardiovascular −2.0pp, metabolic −1.1pp, respiratory −1.0pp, immunological −0.6pp.** 52.3% of rows are neutral (Δ=0, mostly n_gt_train≤1 diseases). The 1.3pp headline is driven by a minority. → h1228 (category-gated fusion) is the most actionable follow-up.

**h1216 INVALIDATED as recipe-improver (2026-04-19, closes linear-weight axis):** Full 11-weight × 5-seed sweep of weighted concat_l2 (weight_a ∈ {0.0, 0.1, …, 1.0}). R@30/MRR form a BROAD PLATEAU across w=0.3–0.8 (paired |t|<2 vs w=0.5; differences <0.1pp). **AUPRC and AUROC are uniquely maximized at w=0.5** — every other weight is significantly worse (w=0.8 AUPRC t=-3.45; w=1.0 AUPRC t=-8.83). Best-R@30 weight (0.70, 20.91%±1.19%) is +0.034pp over anchor, well under the 0.56pp noise floor. No weight beats w=0.5 on ≥3 metrics → preregistered promotion gate NOT met. **w=0.5 equal-weight concat_l2 is locked in as the canonical fusion recipe.** Future fusion gains must come from non-linear combiners (h1225 RRF/Borda), per-disease adaptive weights (h1226), or a third embedding (h1227).

**h1215 VALIDATED (2026-04-19, recall lever):** L2-normalised concat of Node2Vec+FastRP (512-dim joint) beats BOTH parents on ALL 5 metrics over the 1,011-disease intersection (5-seed 80/20). **R@30 19.55%→20.87%±0.91% (+1.32pp), MRR 0.0284→0.0296, AUPRC 0.0569/0.0584→0.0642, AUROC 0.5766/0.5790→0.5851.** Paired per-seed R@30 lift: +1.91, -0.21, +2.84, +1.49, +0.61pp — t≈2.65 on df=4 (one-sided p≈0.028). Random-walk and random-projection embeddings sample partially orthogonal DRKG structure; ensemble is additive at zero marginal training cost. `score_mean` (per-disease z-norm of drug scores then mean) is a CLEAN NEGATIVE — ties on R@30 (19.55%) but tanks AUPRC (0.0500) and AUROC (0.5294) by redistributing mass onto never-scored drugs. **Recalibrated DRKG ceilings per metric: R@30 ≤ 20.87%, MRR ≤ 0.0296, AUPRC ≤ 0.0642, AUROC ≤ 0.5851 — h1200 (supervised GNN) must exceed all four.** `build_concat_lookup` primitive transfers directly to h1202 (DRKG + LINCS) via h1220.

**Remaining viable surfaces:** boundary-targeted re-rank (h1006), per-category adaptive (h1008), deliverable annotation columns (h1001, h1003, h1007, h1102), inverse positive controls (h1104), h1200 loss-weighting on h1103 residuals (h1110), expert-label calibration (h1203).

**See `docs/claude/confidence_system_history.md` for full h900+ experiment detail.**

## Confidence Tiers (current — post-h904+h908, post-h952-fix per h964)

| Tier | Holdout | Preds |
|------|---------|-------|
| GOLDEN | 78.5% ± 6.0% | 101/seed |
| HIGH | 80.0% ± 3.3% | 263/seed |
| MEDIUM | 39.9% ± 5.1% | 265/seed |
| LOW | 10.0% ± 0.7% | 2564/seed |
| FILTER | 6.8% ± 0.7% | 2866/seed |

GOLDEN/HIGH mean inversion is within noise (CIs overlap). Legacy pre-h952-fix: GOLDEN 83.7%, HIGH 78.5%, MEDIUM 42.1%, LOW 11.2%, FILTER 8.2%. See h964 for the bug-fix shift and h975/h979/h980 for the rule-status re-diff. h904+h908 demotions documented in `docs/claude/confidence_system_history.md`.

**Rules:** Full-data is inflated; use HOLDOUT only. Always use `expanded_ground_truth.json` (19x more pairs).

## Reference Docs

- **Patterns & filters:** `docs/claude/patterns.md`
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
