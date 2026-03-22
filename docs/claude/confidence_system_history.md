# Confidence System History

Detailed experiment log for the confidence tier system. Referenced from CLAUDE.md for historical context.

## Current Tier Performance (h815 update, 2026-02-25)

| Tier | Holdout Precision | Predictions |
|------|-------------------|-------------|
| GOLDEN | 87.1% ± 2.7% | 991 |
| HIGH | 83.4% ± 4.0% | 1168 |
| MEDIUM | 38.5% ± 3.6% | 914 |
| LOW | 11.3% ± 0.5% | 9113 |
| FILTER | 9.2% ± 0.5% | 8978 |

## Experiment Log (newest first)

- **h814+h815:** CS SOC revert + MODERATE LOW promotion. **HIGH +2.1pp, MEDIUM +2.2pp**. h817 WEAK/NO split INVALIDATED.
- **h811+h808:** RA GOLDEN→HIGH + lit double-demotion. **GOLDEN +1.6pp, MEDIUM +1.5pp**.
- **h798:** Excluded 'other' from lit_strong_low. HIGH +2.3pp. h797 cancer GOLDEN INVALIDATED (69.5% holdout).
- **h795:** lit_strong→GOLDEN (88.1% holdout). GOLDEN +5.2pp, variance halved.
- **h789+h791:** STRONG LOW→HIGH (80.4%), NO/WEAK HIGH→MEDIUM (31.1%). Combined HIGH +3.9pp.
- **h757:** Comprehensive sub-reason audit. Demoted weak HIGH rules (fluoroquinolones, neuro class match, reproductive hormones, comp_to_base_high) to MEDIUM. Demoted freq10_nomech_r6_10 MEDIUM→LOW. UTI→GOLDEN, diabetes/skin_infection→MEDIUM, epilepsy/gout→LOW. Combined: GOLDEN +9.2pp, HIGH +5.7pp, MEDIUM +2.6pp.
- **h686:** Drug name aliasing: 34 new aliases, +85 GT pairs, +10 diseases. Key drugs: piperacillin (F=16), HCTZ (F=11), clopidogrel (F=9).
- **h718/h730:** Cancer targeted therapy confirmed LOW across ALL sub-classes. Holdout=6.1% (full-data=36%, 5.9x inflation). Checkpoint inhibitors=10.1%, kinase=5.9%, PARP=0%. h598 demotion CORRECT.
- **h677:** GT quality audit: 6% error rate in EC data. 82 false lidocaine/bupivacaine + 3 B12 GT entries removed (combo product drug mismatch). Blocked LA rescue via target_overlap. GT: 59,626→59,541.
- **h673/h670/h671:** Safety fixes: CS→TEN/PAP/OSA filtered (15 preds), 18 false GT removed (NLP extraction errors), AmB antiparasitic spectrum narrowed (3 HIGH→LOW). Fixed duplicate dict key bug in INVERSE_INDICATION_PAIRS. HIGH +0.3pp.
- **h669:** CS HIGH novel quality audit: 97.2% medically acceptable. Fixed DI comp_to_base bug (9 wrong HIGH), removed 6 false GT (NLP errors), +12 CS GT gaps. HIGH +3.5pp (58.0→61.5%).
- **h658/h636/h668:** Literature validation + GT gap search: 54 pairs added. HIGH +3.2pp (54.8→58.0%). Key: DOACs→atrial flutter, cancer drugs→subtypes, antibiotics→prescribing info uses.
- **h661:** Ryland collaboration prep: 230 derm predictions, EGFR gap identified, Montelukast→IPF top wet-lab candidate.
- **h649/h648/h647/h643:** MEDIUM optimization: pneumonia→LOW, cancer R21+→LOW, metabolic leak fix, CV mech gate. Combined +4.8pp (38.1→42.9%).
- **h633/h634:** Cancer same-type: mech+R≤10→HIGH (62.4%), no-mech→LOW (23.6%). Reopened CLOSED #4.
- **h630:** TransE→HIGH promotion: TransE+(mech OR R≤5) non-CS. 56.1% holdout.
- **h629:** MEDIUM quartiles: Q1 60-72%, Q2 50-57%, Q3 44-54%, Q4 ~31%. TransE +19.3pp.
- **h625/h618:** Rescues: hematological immune-mediated (+0.6pp), CV drug-class w/mech gate (+2.7pp).
- **h615:** GT recalibration: 4 groups HIGH→GOLDEN (+139 preds, GOLDEN std 17.9→4.3%).
- **h606:** Psychiatric ATC coherent exclusion: 17.2% holdout (p=0.0006 < MEDIUM). 47 preds MEDIUM→LOW.
- **h611:** CRITICAL: Always use expanded_ground_truth.json for holdout eval (19x more pairs than internal GT).
- **h613:** Expanded GT adds +15pp across tiers (MEDIUM: 38.8% internal → 54.2% expanded).
- **h598:** Expanded CANCER_TARGETED_THERAPY: +15 drugs. 6.1% holdout vs 40.2% existing cancer_same_type. 202 preds MEDIUM→LOW. **MEDIUM +3.3pp**.
- **h592:** Composite quality score (rank+TransE+gene_overlap+mechanism+disease_holdout+non_SR) beats kNN rank by +2.6pp for Q1 MEDIUM. Added to deliverable as `composite_quality_score`.
- **h593+h596+h597:** GT gap expansion: 18 FDA-approved pairs added (antifungals, cancer drugs). MEDIUM +1.2pp.
- **h560:** Antimicrobial-pathogen mismatch filter: 0% holdout for all mismatches. ~30 MEDIUM→LOW. +0.9pp MEDIUM.
- **h553-h562:** MEDIUM precision improvements: cancer_types bug fix (+0.7pp), CS→infectious demotion (+0.3pp), sub-rule demotions (+3.8pp).
- **h542+h544+h546:** Safety audits + gene overlap annotation: non-therapeutic→FILTER, anti-TNF paradoxical autoimmunity, gene overlap +11.4pp (circular, annotation only).
- **h537+h540:** Quality audits, LA procedural demotion. Details in experiment_history.md.
- **h520:** Corticosteroid SOC promotion: 333 MEDIUM→HIGH for autoimmune/dermatological/respiratory/ophthalmic. HIGH +2.3pp, MEDIUM +1.2pp.
- **h486:** SIDER adverse effect mining: 47 new inverse indication pairs (55 drugs, 124 total). 105 predictions → FILTER, 93.3% precision.
- **h526:** Inverse indication taxonomy (10 mechanism classes). +10 new pairs. Bug fix: moved inverse_indication before cancer_same_type. Total: 63 drugs, 135 pairs.
- **h529:** GT quality audit: removed 19 false DRKG-derived GT entries (drug CAUSES disease). 14 Every Cure errors flagged.
- **h478:** GT sync: expanded_ground_truth.json was missing 1503 pairs from production GT. All holdout numbers improved ~7-8pp.
- **h497:** Standard GOLDEN (62.2% holdout) ≈ Hierarchy GOLDEN (70.3%), NOT significant (p>0.35). No demotion needed.
- **h501:** Fixed kNN non-determinism: drug_id tiebreaker for tied scores. Predictions now reproducible across processes.
- **h498:** Updated all precision constants to h478 holdout values. Full-data is misleading — use holdout only.
- **h490:** CV standard MEDIUM demoted to LOW (2.0% holdout), ATC coherent CV also demoted (8.4%). MEDIUM +0.4pp.
- **h479-h495:** Safety audits: 10 harmful preds → FILTER. CCBs+HF, antiarrhythmics+VT (CAST), inverse indications.
- **h487/h488:** Demotions (+1.8pp), h485 cross-type (+1.4pp), h462 category demotions, h393 holdout validation.

## Key Learnings

- Min n≈30 for reliable holdout precision.
- Full-data inflated; use HOLDOUT only.
- `confidence_filter.py` is separate from `production_predictor.py`.
- Always use `expanded_ground_truth.json` for holdout eval (19x more pairs than internal GT).
