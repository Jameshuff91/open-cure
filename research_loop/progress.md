# Research Loop Progress

## Current Session: h1216 — Weighted fusion sweep (INVALIDATED as recipe-improver; confirms w=0.5 is Pareto-optimal) (2026-04-19)

**Status:** Complete | **Hypothesis:** h1216 (INVALIDATED, recall lever)

### What was built
`scripts/h1216_weighted_fusion_sweep.py` — parameter sweep over 11 weights
w ∈ {0.0, 0.1, …, 1.0} × 5 seeds on the h1215 intersection (1,011
diseases). Uses the mathematical identity that weighted L2-concat with
scales (√w, √(1−w)) on pre-L2-normalised halves has cosine similarity
equal to w·cos_a + (1−w)·cos_b — i.e. weighted concat ≡ weighted sim_mean.
This lets one kNN call per weight reuse the h1215 `score_disease_single`
primitive. Setup verified: w=0.5 seed=42 reproduces h1215's anchor
exactly (R@30 20.68%, MRR 0.0321, AUPRC 0.0653, AUROC 0.5897).

### Headline
**No weight beats the equal-weight anchor on ≥3 metrics — preregistered promotion gate NOT met.**

| weight_a | R@30 | MRR | AUPRC | AUROC |
|---|---|---|---|---|
| 0.00 (pure FastRP)  | 18.79% ±0.92% | 0.0267 | 0.0584 | 0.5790 |
| 0.30 | 20.53% ±1.08% | 0.0286 | 0.0630 | 0.5839 |
| 0.40 | 20.90% ±0.96% | 0.0292 | 0.0640 | 0.5848 |
| **0.50 (h1215 anchor)** | **20.87% ±0.91%** | **0.0296** | **0.0642** | **0.5851** |
| 0.60 | 20.90% ±1.09% | 0.0297 | 0.0634 | 0.5839 |
| 0.70 | 20.91% ±1.19% | 0.0296 | 0.0630 | 0.5828 |
| 0.80 | 20.84% ±1.23% | 0.0297 | 0.0624 | 0.5823 |
| 1.00 (pure Node2Vec) | 19.55% ±1.18% | 0.0284 | 0.0569 | 0.5766 |

### Paired t-tests vs w=0.5 (df=4)
- **R@30 / MRR plateau:** weights 0.4–0.8 all indistinguishable from w=0.5 (|t|<2; differences <0.1pp).
- **AUPRC:** uniquely maximized at w=0.5; w=0.7 (-0.00121, t=-1.82), w=0.8 (-0.00186, t=-3.45), w=1.0 (-0.00728, t=-8.83).
- **AUROC:** same pattern; w=0.5 uniquely best; w=0.6 t=-2.63, w=0.8 t=-3.06.
- **Endpoints:** w=0 and w=1 lose on all 4 metrics with strong t-stats (>2.5).
- **Best-R@30 weight 0.70 is +0.034pp over anchor — noise floor ~0.56pp.**

### Scientific takeaway
The R@30/MRR landscape over fusion weight is **flat and symmetric from w=0.3 to w=0.8**
(both embeddings carry near-equal rank-relevant signal). AUPRC/AUROC
(probabilistic discrimination) has a **sharp peak at w=0.5** — the
probabilistic separability metric prefers exact-equal weighting even when
the rank-based metric is insensitive. This is a generalisable pattern for
two-embedding concat: linear interpolation does not discriminate on rank,
it discriminates on probability calibration.

h1215's w=0.5 concat_l2 recipe is **locked in** as the canonical fusion
primitive. h1216 closes the linear-weight axis of exploration. Any further
fusion gain will have to come from non-linear combiners (RRF, learned),
per-disease adaptive weighting, or adding a third orthogonal embedding.

### Shipped
- `scripts/h1216_weighted_fusion_sweep.py` (267 lines; reuses h1199/h1215 primitives)
- `data/analysis/h1216_weighted_fusion_sweep.json` (full per-seed metrics)
- `data/analysis/h1216_weighted_fusion_sweep.md` (markdown aggregate + per-metric optima)
- `data/analysis/h1216_run.log` (full sweep stdout)

### New hypotheses (3 added)
- **h1225 (P3, recall lever):** Rank-based fusion combiners (RRF, Borda) — orthogonal to linear weighting axis h1216 just closed.
- **h1226 (P3, recall lever):** Per-disease adaptive fusion weight — does disease-level w-selection beat global w=0.5? Picks up the signal averaging washes out.
- **h1227 (P3, recall lever):** Three-embedding concat (Node2Vec + FastRP + TransE/GraphSAGE) — tests whether additive-fusion thesis scales to 3 embeddings or saturates.

### Recommended next hypothesis
**h1218 (P2, low effort)** — fusion gain decomposition. h1216 established
that weight doesn't matter in aggregate; h1218 asks WHERE (which diseases)
the fusion gains land. If gains correlate with Node2Vec-FastRP neighbour
disagreement, we have a disease-level routing feature that connects
directly to h1226. Natural next step.


## Current Session: h1215 — Node2Vec + FastRP L2-concat fusion (VALIDATED, 2026-04-19)

**Status:** Complete | **Hypothesis:** h1215 (VALIDATED, recall lever)

### What was built
`scripts/h1215_fusion_benchmark.py` — reuses the h1199 scoring pipeline
(`compute_metrics_for_seed`, split, category) and adds three fusion modes:

- `concat_l2` — L2-normalise each embedding, concatenate → cosine similarity
  in the 512-dim joint space.
- `sim_mean` — average cosine similarities before top-k neighbour selection
  (mathematically equivalent to equal-weight concat_l2 when halves are
  L2-unit — the two modes gave identical numbers on the 1-seed sanity
  run, confirming the equivalence).
- `score_mean` — independent kNN scoring per embedding, z-normalise each
  drug-score vector per disease, average element-wise.

### Headline (5-seed, 1,011-disease intersection)

| Mode | R@30 | MRR | AUPRC | AUROC |
|---|---|---|---|---|
| `node2vec` | 19.55%±1.18% | 0.0284 | 0.0569 | 0.5766 |
| `fastrp` | 18.79%±0.92% | 0.0267 | 0.0584 | 0.5790 |
| **`concat_l2`** | **20.87%±0.91%** | **0.0296** | **0.0642** | **0.5851** |
| `score_mean` | 19.55%±1.52% | 0.0285 | 0.0500 | 0.5294 |

### Findings
**concat_l2 beats BOTH parents on ALL FIVE metrics.** Paired per-seed R@30
lift over Node2Vec: +1.91, −0.21, +2.84, +1.49, +0.61pp — 4 of 5 seeds
directionally positive, t≈2.65 on df=4 (one-sided p≈0.028). Random-walk
(Node2Vec) and random-projection (FastRP) embeddings sample partially
orthogonal DRKG structure; their ensemble is additive at zero marginal
training cost (FastRP trains in seconds; Node2Vec takes hours).

`score_mean` is a **clean negative control**: ties on R@30 but per-disease
z-normalisation of a sparse score vector redistributes probability mass
onto never-scored drugs, tanking AUPRC (0.0500) and AUROC (0.5294).

### DRKG ceiling recalibration
- R@30 ≤ 20.87% (was 19.55%)
- MRR ≤ 0.0296 (was 0.0284)
- AUPRC ≤ 0.0642 (was 0.0584, FastRP)
- AUROC ≤ 0.5851 (was 0.5790, FastRP)

**h1200 (supervised GNN) must exceed all four.**

### Outputs
- `scripts/h1215_fusion_benchmark.py`
- `data/analysis/h1215_fusion_benchmark.{json,md}`
- `data/analysis/h1215_run.txt`

### New hypotheses (5 added)
- **h1216 (P2):** Weighted fusion sweep — search for non-equal weights
  that further shift R@30 or AUPRC. FastRP leads on AUPRC/AUROC,
  Node2Vec on R@30/MRR — per-metric optima may differ.
- **h1217 (P3):** Three-way L2-concat of Node2Vec + FastRP + no_treatment.
  Test whether a third cheap embedding adds independent signal.
- **h1218 (P2):** Per-disease fusion gain decomposition — correlate
  ΔR@30 with neighbour-agreement; if fusion wins concentrate where the
  two embeddings disagree, gives a routing signal.
- **h1219 (P2):** Learnable fusion — logistic regression over
  (score_a, score_b, rank_a, rank_b) trained on training-disease GT,
  evaluated on the outer holdout. Gated-supervision version of
  concat_l2 — could exceed unsupervised by ≥2pp R@30.
- **h1220 (P2, infrastructure):** Fusion recipe portability to h1202
  (DRKG + LINCS). Declares `build_concat_lookup` as the canonical
  fusion primitive so h1201 (LINCS pilot) ships with a drop-in fusion
  target, not a separate design experiment.

### Recommended next hypothesis
**h1218 (P2)** — per-disease gain decomposition. Cheapest way to convert
the 1.32pp average lift into a larger per-disease lift via routing, and
the per-disease ΔR@30 table is a reusable artefact for h1200 loss-weighting.
Alternatively **h1219 (P2)** — if supervised fusion is materially better
than unsupervised concat_l2, it's the strongest pre-h1200 recall surface.

---

## Previous Session: h1199 — Clean multi-metric embedding benchmark (VALIDATED, 2026-04-19)

**Status:** Complete | **Hypothesis:** h1199 (VALIDATED, infrastructure)

### What was built
`scripts/clean_embedding_benchmark.py` — tier-free kNN evaluation producing
all five standard metrics on 5-seed disease-level 80/20 splits:

1. R@30 per-drug (macro-avg across test diseases)
2. Hits@K for K ∈ {1, 5, 10, 30, 100}, both per-drug and per-test-triple
3. MRR per-test-triple — KG-completion standard
4. AUPRC per-test-triple — TxGNN headline metric
5. AUROC per-test-triple

Uses production convention: internal indicationList GT (5,629 pairs, 1,078
diseases) for kNN neighbour aggregation, expanded_ground_truth.json
(57,805 pairs) for evaluation hits. Driven by `OPEN_CURE_EMBEDDINGS_PREFIX`
env var so any future embedding drops in without code edits.

### Headline results

| Embedding | n_dis | R@30 | Hits@10/drug | MRR | AUPRC | AUROC |
|---|---|---|---|---|---|---|
| **node2vec_256** | 1,011 | **19.55%±1.18%** | 12.82% | **0.0284** | **0.0569** | **0.5766** |
| graphsage_256 | 850 | 8.17%±0.53% | 4.70% | 0.0126 | 0.0322 | 0.5529 |

Node2Vec dominates GraphSAGE on ALL FIVE metrics (Δ R@30 +11.4pp, Δ MRR
+0.016, Δ AUPRC +0.025, Δ AUROC +0.024). Confirms h922-v2 invalidation was
a real ranking regression, not a tier-system coupling artifact.

### Sanity-check anchors
- Per-drug R@30 = 19.55% matches h958 overall_r30 = 19.49%±1.42% (independent
  pipeline) → benchmark convention is correct.
- Random Hits@1 baseline over 1,345-drug pool ≈ 0.074%. Node2Vec Hits@1
  per-drug = 2.88% → 39x better than random at rank 1.

### Per-category Node2Vec R@30 spread
endocrine 41%, ophthalmic 38%, reproductive 33%, infectious 30%, autoimmune
27%, dermatological 26% (high). cancer 14%, gastrointestinal 14%,
metabolic 14%, immunological 14%, cardiovascular 13%, neurological 11%,
psychiatric 10%, hematological 10% (low). 3x spread is a targeting signal
for h1200 loss-weighting.

### Outputs
- scripts/clean_embedding_benchmark.py
- data/analysis/clean_benchmark_node2vec_256.{json,md}
- data/analysis/clean_benchmark_graphsage_256.{json,md}

### New hypotheses (4 added)
- **h1210 (P2):** Negative-sample-aware AUROC — the 0.577 flat AUROC is
  likely deflated by all-vs-all negatives; filtered / 1:10 ratio negatives
  will give a publishable number comparable to TxGNN's 0.913 AUPRC.
- **h1211 (P2):** Per-category R@30 explanation — decompose the 3x
  category spread into density, isolation, GT completeness. Flags where
  h1200 should invest loss weight.
- **h1212 (P2):** Pre-h1200 ceiling probe — run h1199 on fastrp_256,
  node2vec_256_no_treatment, and TransE; declare per-metric DRKG-only
  ceilings as the h1200 minimum-bar.
- **h1213 (P3):** Hits@1 sanity audit — manual review of top-1 predictions
  under relaxed definitions.

### Recommended next hypothesis
**h1212 (P2)** — finish the embedding ceiling sweep before committing
compute to h1200.

---

## Previous Session (continued): h1212 — Embedding ceiling sweep (VALIDATED, with recalibration, 2026-04-19)

**Status:** Complete | **Hypothesis:** h1212 (VALIDATED + surfaces recalibration need)

### What was tested
Extended h1199 benchmark to 4 embeddings. Added `--restrict-to-embedding`
flag to `clean_embedding_benchmark.py` for apples-to-apples comparisons
when embeddings have different disease-vocabulary coverage.

### Headline table (restricted comparisons on same disease universe)

| Embedding | n_dis | R@30 | MRR | AUPRC | AUROC |
|---|---|---|---|---|---|
| node2vec_256 (full) | 1011 | **19.55%±1.18%** | **0.0284** | 0.0569 | 0.5766 |
| fastrp_256 (full) | 1011 | 18.79%±0.92% | 0.0267 | **0.0584** | **0.5790** |
| node2vec_256 (restricted to no_treatment dis) | 838 | 17.09%±0.75% | 0.0246 | 0.0531 | 0.5686 |
| node2vec_256_no_treatment | 850 | 8.46%±0.94% | 0.0153 | 0.0300 | 0.5555 |
| graphsage_256 | 850 | 8.17%±0.53% | 0.0126 | 0.0322 | 0.5529 |

### Three findings
1. **FastRP nearly matches Node2Vec** (R@30 18.79% vs 19.55%, within 1σ)
   and slightly BEATS it on AUPRC and AUROC. A simple random projection
   captures most of Node2Vec's repurposing signal. → h1215 (FastRP fusion).
2. **Treatment-edge leakage is ~50%, not ~29%.** Apples-to-apples on 850
   diseases: full Node2Vec 17.09% → no_treatment 8.46%. CLAUDE.md claims
   36.59% → 26.06% = 71.2% retained. On this benchmark, only 49.5% is
   retained. Material recalibration of the "honest transductive" claim.
   → h1214 (reconcile discrepancy before external citation).
3. **GraphSAGE's disease universe is narrower.** GraphSAGE and no_treatment
   share a 94,247-entity preprocessing (vs 49,616 for node2vec/fastrp) and
   miss 161 GT diseases that the smaller preprocessing retains. The h922-v2
   GraphSAGE regression is partly coverage, not purely ranking quality.

### DRKG-only ceilings per metric (full 1011-disease universe)
R@30 ≤ 19.55%, MRR ≤ 0.0284, AUPRC ≤ 0.0584 (FastRP), AUROC ≤ 0.5790 (FastRP).
**h1200 must exceed all four to qualify as improvement.**

### Outputs
- data/analysis/clean_benchmark_fastrp_256.{json,md}
- data/analysis/clean_benchmark_node2vec_256_no_treatment.{json,md}
- data/analysis/clean_benchmark_node2vec_256_restricted_to_no_treatment.{json,md}
- scripts/clean_embedding_benchmark.py (+ `--restrict-to-embedding` flag)

### New hypotheses (2 added)
- **h1214 (P2):** Reconcile the 71.2% vs 49.5% retention discrepancy —
  either find the original no_treatment build, fix CLAUDE.md, or
  retrain on matching vocabulary. Blocks external citation of the
  leakage-retention number.
- **h1215 (P2):** FastRP fusion — if a random projection is within 1σ of
  Node2Vec on R@30 and beats it on AUPRC/AUROC, score-fusion may give
  additive gains. Also a candidate h1200 warm-start initializer.

### Recommended next hypothesis
**h1200 (P1)** — benchmark infrastructure is now complete with per-metric
DRKG ceilings declared. The supervised GNN has a clear success bar.
Alternatively, **h1214 (P2, low effort)** — resolve the CLAUDE.md
discrepancy before it shows up in the paper.

---

## Current Session: h1002 — neighbor-augmented rescue test (2026-04-19)

### Status: INVALIDATED — closes filter-form biologic precision family

### Key Finding
Only 1/16 (6.2%) of h995b unique-target autoimmune hits are rescued by k=3 kNN
neighbor-bio_gt target-union augmentation. The other 15 (Tocilizumab→temporal
arteritis, Anakinra/Canakinumab→FMF, Abatacept→SLE, Rituximab→microscopic
polyangiitis/scleroderma, Adalimumab→non-behcets uveitis, etc.) are target-unique
across their entire DRKG disease neighborhood. **Structural floor:** ≥50% of
autoimmune biologic HITS are neighborhood-unique. Any filter on autoimmune
biologics is CAPPED at halving bio_r30 (-50pp). This closes the filter-form
biologic precision family: h953/h957/h990/h995/h1002 all fail at the same
structural wall.

### Shipped
- `scripts/h1002_neighbor_augmented_rescue.py`
- `data/analysis/h1002_neighbor_augmented_rescue.json`
- `data/analysis/h1002_run.txt`

### Rejected
- Filter-form biologic precision pivots (entire family now closed)

### New Hypotheses (1)
- **h1003** (P3, low): Surface the 15 unrescued unique-target hits as a
  `biologic_novel_target_family` deliverable annotation column. These are
  the highest-value triage signals for expert (Ryland) review — biologics
  that appear in bio_gt with no target-family peer in their neighborhood
  are either first-in-class discoveries or GT-gap candidates.

---

## Prior Session (same day): h995 — autoimmune biologic family-mis-selection audit (2026-04-19)

### Status: INVALIDATED at ship gate (filter form fails -50% bio_r30)

### Key Finding
Biologic target-family-match IS a genuine cross-drug precision signal — after
de-biasing the self-inclusion tautology via leave-one-out (LOO), the autoimmune
match-vs-miss ratio is 4.22x (31.37% vs 7.44% hit rate), with cancer 10.91x,
cv 11.57x, hematological 24.22x. But the proposed in/out filter form CANNOT
ship: in autoimmune, target_match_loo=True retains only 16/32 pooled 5-seed
hits, so applying the filter drops 50% of bio_r30 — violating the -2pp
ship cap by 25x. Half the biologic HITS are target-unique in bio_gt (first-
in-family mAbs like anti-IL4Rα or anti-IL5) and the filter cannot see them.
Same failure mode as h957 zero-overlap filter + the closed h953/h990 in/out
shuffle family. Methodological lesson: the naive diagnostic showed ratio=inf
(hr_miss=0%) across every category — classic self-inclusion tautology —
because any candidate IN bio_gt with annotated targets trivially satisfies
target_match via set-inclusion. LOO exposed that 49.5% of that signal
(96/194 naive hits) was tautological.

### Shipped
- `scripts/h995_autoimmune_biologic_family_audit.py` — 5-seed autoimmune
  audit over top-30 biologic slots with USAN suffix, 2-letter substem, and
  target-set family-match rules
- `scripts/h995b_debias_target_match.py` — leave-one-out re-analysis
  exposing the self-inclusion tautology
- `data/analysis/h995_autoimmune_family_audit.json`
- `data/analysis/h995_slot_records.json` (3,521 per-slot records, 5 seeds)
- `data/analysis/h995b_debias_results.json`
- `data/analysis/h995_run.txt`, `data/analysis/h995b_run.txt`

### Rejected
- In-out filter form for biologic family-match (fails -50% bio_r30)
- USAN-suffix match as a family proxy (LOO ratio 0.74x — ANTI-predictive;
  -mab/-cept suffix alone is too coarse)
- USAN-substem match (LOO ratio 1.58x, below 3x gate)

### New Hypotheses (3)
- **h1000** (P2, medium): In-window biologic family-match re-rank (no in/out
  exchange). Keep all 30 slots, bump target_match_LOO=True biologics up 1
  rank, demote LOO=False biologics down 1. Aligns with h994 in-window
  re-rank family. Uses k=3 kNN neighbor bio_gt (not own disease) to avoid
  inference-time leak.
- **h1001** (P3, low): Add `biologic_target_family_match` + shared-genes
  list columns to the 13,416-row XLSX deliverable. Mirrors h968 annotation
  philosophy. Low cost, high triage value for Ryland review.
- **h1002** (P3, low): Audit the 16 autoimmune unique-target hits. If
  their disease's k=3 kNN neighbors' bio_gt targets would cover them,
  propose a neighbor-augmented filter (h1003) that could rescue most of
  the 50% hit loss while keeping filter form.

### Next
Highest-ROI remaining pending:
1. **h905** (P2, LINCS L1000 pilot) — still unclaimed, but high effort and
   external-data dependent.
2. **h994** (P2, medium) — in-window re-ranking for precision; h1000 is
   the biologic-specialized cousin, so advancing h994 first may de-risk.
3. **h1000** (P2, medium) — follow-up from this session.
4. **h1002** (P3, low) — cheap diagnostic that can unlock h1003 filter
   variant.

---

## Prior Session: h961 — MeSH disease-name aliasing (2026-04-19)

### Status: VALIDATED (but original mechanism repudiated)

### Key Finding
Principled alias generation (British spelling, hyphen insertion, possessive restoration, plural stripping, trailing-abbrev drop) covers only **7.9%** of the 668 disease_names entries missing from mesh_mappings — well below the 90% target. The bulk of the gap is name-granularity mismatch (mm has `acne`, disease_names has `acne vulgaris`), not spelling variation.

### Shipped
- `scripts/h961_alias_generator.py` — two-layer resolver (transformations + disease_names backfill)
- `data/reference/h961_disease_name_aliases.json` — 668 backfill entries + 114 reverse British variants
- `src/production_predictor.py:2221-2244` — load path integration
- mesh_mappings grew 1489 → 2265; 100% of the 668 missing now resolve directly

### Rejected
- Less-specific-prefix fallback (43% of its hits produced wrong ids)

### New Hypotheses (3)
- h987 (P3): Measure h961 downstream impact on h393 5-seed tier precisions
- h988 (P4): Retire h952 runtime fallback once h961 canonicalization covers all disease_names
- h989 (P4): Provenance audit for the 260 "true new entities" in disease_names

### Next
Highest-ROI remaining pending: h961 done, h953 (biologic precision pivot) still P2. h984 (lit_weak over-demotion) in_progress.

---

## Prior Session: h980 — h771 literature-coverage re-run post-h963 (2026-04-19)

### Hypothesis
h963 revealed that `scripts/h771_literature_coverage_analysis.py:101` was
silently broken pre-h963 (passing `disease_id` through `find_disease_id`
returned None for every call). h980 re-runs h771 to measure what was
masked and check whether CLAUDE.md's h731/h768 memory entries need caveats.

### Status: VALIDATED — h771 was indeed broken; post-fix numbers reveal two calibration signals

### Evidence of prior breakage
- `data/analysis/h771_coverage_analysis.log` is 0 bytes — script never
  produced stdout to capture.
- Live verification in h963: `find_disease_id("drkg:Disease::MESH:D014141")`
  → `None`, confirming `predict(disease_id)` returned zero predictions for
  100% of holdout diseases pre-change.
- `h771_medium_origin_predictions.json` exists but is produced by a
  *different* script (`h771_mine_remaining_medium.py`) that keys off
  h768 data, not `predict()` — unaffected.
- CLAUDE.md h731/h768 entries cite `literature_mining_cache.json` numbers
  that are cache-level (GT-independent) — not affected.

### h771 post-h963 results (1078 diseases, seeds [42,123,456,789,1337])

| Tier   | Precision (h771 pool) | Note |
|--------|------------------------|------|
| GOLDEN | 90.3% ± 3.1% (n=140)   | h771 pool is broader than h393 (1078 vs 1011) |
| HIGH   | 81.9% ± 2.9% (n=272)   | |
| MEDIUM | 43.0% ± 4.5% (n=201)   | |
| LOW    | 14.8% ± 0.9% (n=2447)  | |
| FILTER | 12.6% ± 1.0% (n=2991)  | |

Gap vs h964 (1011-disease pool, seed 2024 instead of 1337) is expected —
h771's pool drops the `∩ embeddings` filter so it includes ~67 more
diseases whose predictions pass through the pipeline differently.

### Two calibration signals surfaced

1. **`MEDIUM(lit=NOT_ASSESSED)` at 46.2% ABOVE the MEDIUM baseline 43.0%.**
   Literature cache absence is not a negative signal — it's a coverage
   gap. Filed **h985** to verify by batch-mining the NOT_ASSESSED bucket.

2. **`MEDIUM→LOW(lit_weak)` at 19.1% sits 4.3pp above LOW average (14.8%).**
   The demotion pushes these preds out of MEDIUM (43.0%) but they land
   closer to the LOW/MEDIUM boundary than squarely in LOW. Over-aggressive
   demotion? Filed **h984**.

### Minor script bug filed as h983
`h771_literature_coverage_analysis.py:73` declares `new_disease_groups`
but never populates it in the per-seed training loop; line 153 restores
`orig_disease_groups` — so `drug_disease_groups` stayed at full-data
during holdout, inflating any rule keyed on disease_groups.

### New Hypotheses Generated (3)
- **h983** (P3, low): patch h771 disease_groups leak, re-measure tier precision
- **h984** (P2, low): MEDIUM→LOW(lit_weak) over-demotion audit
- **h985** (P3, low): NOT_ASSESSED bucket literature re-mining

### Recommended Next Steps
1. **h984** (P2, low): single highest-value follow-up — if the demotion
   rule is over-aggressive, relaxing it could rescue MEDIUM recall.
2. **h962** (P2, medium): deliverable regen — still independent of h963.
3. **h981** (P3, low): repo-wide `.predict(` call-site audit (from h963 set).

### Artifacts
- `data/analysis/h980_h771_rerun.txt` (h771 stdout post-h963)

---

## Previous Session: h963 — predict(disease_id) fast-path (2026-04-19)

### Hypothesis
h952/h959 established that `find_disease_id(disease_name)` is a fragile
name-resolution bottleneck that silently drops diseases. Callers that
already have a disease_id (h393 evaluator, h771 literature mining, future
deliverable regen) should not be round-tripping through name resolution.
h963 extends `predict()` to detect `drkg:Disease::` prefix and skip
`find_disease_id` entirely, and updates the h393 evaluator accordingly.

### Status: VALIDATED — zero regression, repairs a silently-broken caller

### Code changes
- `src/production_predictor.py:4636-4645` — `predict()` now checks for
  `drkg:Disease::` prefix; when matched, sets `disease_id = arg` and looks up
  canonical `disease_name` from `self.disease_names`. Otherwise falls back to
  the original `find_disease_id` path.
- `scripts/h393_holdout_tier_validation.py:167-175` — evaluator now passes
  `disease_id` directly instead of the canonical name.

### Equivalence test (`scripts/h963_smoke_test.py`)
- 25/25 diseases produced identical predictions between canonical-name path
  and id-path (drug_id order, tier, and rank all match per-rank).
- Justification: when h393 passed a canonical disease_name pre-change, the
  h952 reverse-index fallback in `find_disease_id` resolved it to the same
  disease_id, and the pipeline used the same canonical `disease_name`
  downstream. Post-change, the id-path produces the same (disease_id,
  canonical_name) pair. Pipeline state is identical.

### 5-seed h393 holdout (post-h963)

| Tier   | h964 post-fix | h963 id-path | Δ   |
|--------|---------------|--------------|-----|
| GOLDEN | 78.5% ± 6.0%  | 78.5% ± 6.0% | 0pp |
| HIGH   | 80.0% ± 3.3%  | 80.0% ± 3.3% | 0pp |
| MEDIUM | 39.9% ± 5.1%  | 39.9% ± 5.1% | 0pp |
| LOW    | 10.0% ± 0.7%  | 10.0% ± 0.7% | 0pp |
| FILTER | 6.8%  ± 0.7%  | 6.8%  ± 0.7% | 0pp |

Exact per-seed match (GOLDEN seeds: 71.4 / 78.8 / 85.1 / 71.9 / 85.2%).

### h771 side benefit
`scripts/h771_literature_coverage_analysis.py:101` was calling
`predictor.predict(disease_id)` — which pre-h963 routed through
`find_disease_id(disease_id)`. Live verification on `drkg:Disease::MESH:D014141`
shows that call returns `None` (disease_id strings are neither in mesh_mappings
nor disease_names-as-values), so `predict()` fell through to
"No kNN coverage" and produced zero predictions for every disease. h771's
literature-coverage output was therefore suspect. Post-h963 the same call
returns the expected 30 predictions. Filed **h980** to re-run h771 and diff.

### New Hypotheses Generated (3)
- **h980** (P2, low effort): re-run h771 literature coverage analysis on
  post-h963 predictor; diff vs prior output to quantify what was masked.
- **h981** (P3, low effort): audit all `.predict(` call sites for accidental
  disease_id-in-name-string usages; add a lint / load-time warning.
- **h982** (P3, low effort): add an explicit `predict_by_id()` method as the
  documented production interface, replacing the prefix-sniffing heuristic in
  callers that already hold a disease_id.

### Recommended Next Steps
1. **h980** (P2, low effort): quickest follow-on — if h771's cache is live
   it may invalidate h731/h768 STRONG/MODERATE attribution.
2. **h962** (P2, medium effort): deliverable regen — still open, unrelated
   to h963 (deliverable uses its own `knn_predict(disease_id, ...)` not
   `predict()`), so unaffected by the bug. Pure bug-fix refresh.
3. **h961** (P2, low effort): mesh_mappings aliasing — the remaining piece
   of the name-resolution triad.

### Artifacts
- `data/analysis/h393_holdout_validation.json` (post-h963, identical to
  h964 to the tenth of a percent)
- `data/analysis/h963_h393_run.txt` (full stdout)
- `scripts/h963_smoke_test.py` (regression marker for future changes)

---

## Previous Session: h964 — Post-h952-fix tier precision re-run (2026-04-19)

### Hypothesis
h959 flagged that h904/h908 tier precisions (GOLDEN 83.7%, HIGH 78.5%, MEDIUM 42.1%, LOW 11.2%, FILTER 8.2%) were measured pre-h952-fix. h964 re-runs the 5-seed h393 evaluator on current main to quantify magnitude drift and check whether any h-number decision flips.

### Status: VALIDATED — all five tiers drifted |Δ|>1pp; no h-number decisions flip.

### Findings (5-seed h393 holdout, post-h952-fix)

| Tier   | Pre-fix (CLAUDE.md) | Post-fix (h964)    | Δ mean | Δ std | Preds/seed |
|--------|---------------------|--------------------|--------|-------|------------|
| GOLDEN | 83.7% ± 1.3%        | 78.5% ± 6.0%       | -5.2pp | +4.7  | 94 → 101   |
| HIGH   | 78.5% ± 2.4%        | 80.0% ± 3.3%       | +1.5pp | +0.9  | 230 → 263  |
| MEDIUM | 42.1% ± 3.0%        | 39.9% ± 5.1%       | -2.2pp | +2.1  | 200 → 265  |
| LOW    | 11.2% ± 1.1%        | 10.0% ± 0.7%       | -1.2pp | -0.4  | 2062 → 2564|
| FILTER | 8.2% ± 0.6%         | 6.8% ± 0.7%        | -1.4pp | +0.1  | 2196 → 2866|

### Why

h952 added a reverse-index fallback to `find_disease_id` that recovered ~40 holdout diseases/seed from silent-zero prediction. These newly-resolving diseases add ~32 preds/disease (top-30 + filtered), explaining the per-seed total increases across all tiers. Their hit/miss mix differs from the always-resolved subset, which is why every tier drifted. The most striking effect is on GOLDEN: std widened 4.6x (1.3 → 6.0) because seeds 42 and 789 dropped to 71–72% GOLDEN precision vs seeds 123/456/2024 at 78–85% — a split-dependent signal worth investigating (filed as h974).

### Tier ordering

GOLDEN (78.5) < HIGH (80.0) at mean, but CIs overlap (GOLDEN 72.5–84.5 vs HIGH 76.7–83.3). Tier inversion is within noise. Filed h976 (10-seed h393 re-run) to resolve.

### No h-number decisions flip

h904 (10 overfitted-rule demotions) and h908 (45-name MeSH C23 blocklist) measured *relative* Δ within the same (buggy) framework, so their deltas survive the bug-fix shift. The absolute tier numbers quoted in CLAUDE.md needed correction but the per-rule ordering and the validated demotion decisions still hold.

### New Hypotheses Generated (3)

- h974 (P3): GOLDEN std 4.6x widening — decompose seed-42 / seed-789 dip
- h975 (P3): Post-fix per-rule overfit audit — any rule flipping GENUINE↔OVERFITTED?
- h976 (P3): 10-seed h393 GOLDEN/HIGH boundary reality check

### Recommended Next Steps
1. **h975** (P3, low effort): Diff per-rule status between pre-fix and post-fix h393 — quick win for tier-rule health.
2. **h976** (P3, low effort): 10-seed run resolves GOLDEN/HIGH inversion question with no code changes.
3. **h962** (P2, medium effort): Regenerate 13,416-row deliverable on post-fix predictor (separate from h964's scope but motivated by the bug-fix sweep).

### Artifacts
- `data/analysis/h393_holdout_validation.json` (5-seed tier + per-rule breakdown)
- `data/analysis/h964_h393_postfix_run.txt` (full stdout)

---

## Previous Session: h960 — Neurological Supplement Ablation (2026-04-19)

### Hypothesis
h952 found a -2.85pp production-vs-plain-kNN regression on n=13 neurological
holdout diseases (the "boost+supp" stratum). h960 hypothesized the h173
neurological supplement (`_supplement_neurological_predictions`) was the cause:
class-injected drugs displace correct kNN predictions when boost is also
active.

### Status: INVALIDATED — supplement is benign for R@30

### Findings (5-seed aggregate)
| scope             | baseline       | ablated        | Δ        |
|-------------------|----------------|----------------|----------|
| Neurological      | 10.74% ± 4.75% | 10.74% ± 4.75% | +0.00pp  |
| Non-neurological  | 19.90% ± 1.49% | 19.90% ± 1.49% | +0.00pp  |
| All holdout       | 19.49% ± 1.42% | 19.49% ± 1.42% | +0.00pp  |

EXACT 0pp delta on every seed × every metric. 0/50 neuro diseases helped or
hurt. Non-neuro control passes cleanly (no monkey-patch leakage).

### Why
The supplement function early-returns at `if not missing_drugs:`
(production_predictor.py:4346) without re-sorting when no class-matched drug
is missing from the kNN top-N. On the holdout neuro pool, kNN top-30 already
contains all class-matched drugs (anticonvulsants for epilepsy, dopaminergics
for Parkinson's, etc.), so missing_drugs is empty and the supplement is a
no-op. The h171 motivating coverage gap (60.4% class vs 18% kNN) was measured
on FULL data, not holdout.

### Implication
- h960 hypothesis ruled out — supplement is not the regression source.
- h972 (next): ablate SELECTIVE_BOOST for neurological category specifically.
  Boost helps on metabolic/renal/hematological/respiratory/immunological
  (boost_only stratum +1.41pp) — must be behaving differently on neuro to
  produce the -2.85pp on the boost+supp stratum.
- h973: per-disease set-diff localization on the n=13 boost+supp neuro
  diseases — name the specific drugs being swapped in/out by boost.

### Methodology lesson
Ablate pipeline components independently before naming a culprit. The
supplement's name suggested it was active but its short-circuit logic makes
it inert on the holdout — measurement, not assumption, identifies the cause.

---

## Previous Session: h965 — Cancer-Restricted Variant of h957 (2026-04-19)

### Hypothesis
Apply the h957 zero-overlap biologic filter ONLY to cancer-category diseases,
where TCGA-dense disease_genes captures the actual MoA target space (HER2, VEGF,
CD20, PD1). Hypothesis: localized filter recovers most of h957's precision lift
without the catastrophic recall cost that hit autoimmune/neuro/respiratory.

### Status: INVALIDATED — global lift too small to ship as tier rule

### Findings (5-seed aggregate)
| scope        | Δ bio_p30 | Δ bio_r30 | Δ overall |
|--------------|----------:|----------:|----------:|
| GLOBAL       |  +0.28pp  |  -0.46pp  |  +0.04pp  |
| CANCER ONLY  |  +2.54pp  |  -2.24pp  |  +0.40pp  |
| NON-CANCER   |  +0.00pp  |  +0.00pp  |  +0.00pp  |

Decision criterion: global Δ bio_p30 >= +0.5pp AND Δ bio_r30 >= -1.0pp.
Filter fails the precision threshold globally (+0.28 < +0.5). Non-cancer
control passes cleanly (Δ=0.00 everywhere → no leakage).

### Why
Cancer cohort effect (+2.54/-2.24) reproduces h957's cancer slice exactly
(+2.6/-2.1), confirming the per-category result is real. But cancer is
~12% of evaluable diseases (~23/200 per seed). A localized +2.5pp lift on
~75-90 dropped predictions per seed dilutes to +0.3pp at the global tier-
system level. The biologic-overlap signal is genuine in oncology but too
narrow to register as a tier rule.

### Implication
- INVALIDATED as a global tier rule (cancer-restricted form).
- h968: per-prediction `biologic_low_mechanism_evidence` annotation (ship
  the signal in the deliverable column without moving global metrics).
- h969: subclass-restricted variant on ~20 canonical anti-target biologics
  (anti-HER2/VEGF/CD20/checkpoint) where target identity == MoA.

---

## Previous Session: h957 — Zero-Overlap Biologic Safety Filter (2026-04-19)

### Hypothesis
h957 implements h949: drop biologic predictions where the drug's targets share zero
genes with the disease's gene set, on the premise that biologic efficacy almost
always requires direct target engagement (mAbs, cytokines, fusion proteins).
Filed by h955 as the precision pivot freed by closing h906/h920/h921/h924.

### Status: INVALIDATED (global form) / pivots filed as h965, h966, h967

### Experiment
5-seed (42/123/456/789/2024) h393 80/20 disease holdout. For each holdout disease,
`predictor.predict(disease_name, top_n=200, include_filtered=True)`. Apply filter
to drop predictions where:
  is_biologic(drug) AND drug_targets[d] non-empty AND disease_genes[D] non-empty
  AND |drug_targets[d] ∩ disease_genes[D]| == 0
Then take top-30 of survivors. Compare bio_p30, bio_r30, sm_r30, overall_r30
baseline vs filtered.

### Key findings (5-seed aggregate)
| metric        | baseline       | filtered       | Δ        |
|---------------|----------------|----------------|----------|
| bio_p30       | 5.64% ± 1.99%  | 8.85% ± 2.63%  | +3.22pp  |
| bio_r30       | 30.31% ± 3.57% | 14.78% ± 2.42% | -15.54pp |
| sm_r30        | 19.29% ± 1.51% | 19.70% ± 1.65% |  +0.41pp |
| overall_r30   | 19.79% ± 1.59% | 19.59% ± 1.57% |  -0.20pp |

Ship criterion: bio_p30 +>=3pp AND bio_r30 drop <=2pp.
Filter MEETS the precision target but VIOLATES the recall cap by ~7.8x.

### Per-category split (sorted by Δp30 desc)
| category         | n_p30 | n_r30 | Δ p30   | Δ r30   |
|------------------|------:|------:|--------:|--------:|
| musculoskeletal  |    17 |     6 | +26.1pp |  +0.0pp |
| hematological    |    17 |     6 | +25.9pp | -25.0pp |
| cardiovascular   |    55 |    37 | +13.2pp |  -8.1pp |
| autoimmune       |    29 |    25 |  +9.3pp | -28.6pp |
| metabolic        |    54 |    24 |  +9.1pp | -19.1pp |
| **cancer**       |   115 |    54 | **+2.6pp** | **-2.1pp** |
| ophthalmic       |    20 |     9 |  +1.2pp |  -5.6pp |
| neurological     |    43 |    11 |  -5.8pp | -21.2pp |
| respiratory      |    21 |     3 |  -6.4pp | -58.3pp |
| immunological    |     7 |     1 |  -4.8pp | -100pp  |

Cancer is the only category with sufficient n that essentially meets the ship
cap. Musculoskeletal looks great but n=6 too small. Autoimmune/neuro/respiratory
fail because disease_genes captures etiology genes (HLA, complement, structural)
not the cytokine signaling targets that anti-TNF/anti-IL6/anti-IL17 biologics
bind. Disease_genes is mechanism-aware in oncology (TCGA-dense), mechanism-blind
in inflammatory disease.

### Tier-shift impact
Across 5 seeds, filter would demote 7 GOLDEN, 40 HIGH, 5 MEDIUM, 1087 LOW, 2797
FILTER biologic predictions. The 47 GOLDEN/HIGH demotions are the highest-cost
false positives; filed as h967 audit.

### Why
Target-overlap is a real biologic-correctness signal (precision lift confirmed)
but disease_genes is too narrow a definition of "mechanism" to use as a global
demoter. The per-category split is exactly what you'd predict from biology:
cancer biologics target tumor-cell receptors directly; inflammatory biologics
target soluble cytokines whose genes are absent from disease etiology gene sets.

### Implication
- Global filter REJECTED for deployment.
- h965: cancer-restricted variant — easy ship test.
- h966: pathway-aware overlap (KEGG-extended disease_genes) — fixes the
  inflammatory false-positive class.
- h967: GOLDEN/HIGH zero-overlap audit — characterizes whether high-tier
  zero-overlap preds are GT gaps or genuine FPs.

---

## Previous Session: h951 — Biologic Failure-Class Reality Check (2026-04-19)

### Hypothesis
h951 was filed by h940 as a sanity check before committing GPU/license effort
to h906 (DrugBank), h920 (PubMedBERT), h921 (ESM2), h924 (LINCS VAE) — all of
which cited "biologics fail at 27.3% R@30 vs 41.8% overall" as motivation. Is
that gap still real on the current production pipeline + expanded GT?

### Status: VALIDATED — motivation dissolved

### Experiment
5-seed (42/123/456/789/2024) 80/20 disease holdout. h393 train-only GT
recompute (drug_train_freq, drug_to_diseases, drug_cancer_types,
drug_disease_groups, train_diseases, train_embeddings, train_disease_categories).
Then `predictor.predict(disease_name, top_n=30, include_filtered=True)` on each
holdout disease — top-30 by rank from the FULL production pipeline (kNN +
selective category boost + tier rules). Per-disease bio_r30, sm_r30,
overall_r30 against expanded_ground_truth.json. Biologic pool = 266 of 11,656
drugs (USAN suffix + keyword proxy from h939/h940).

### Key findings (5-seed aggregate)
| metric            | mean ± std       |
|-------------------|------------------|
| bio_r30           | 27.06% ± 3.12%   |
| overall_r30       | 16.39% ± 1.10%   |
| sm_r30            | 15.91% ± 1.12%   |
| bio_p30           |  6.37% ± 2.06%   |

**Biologics OUTPERFORM overall by +10.67pp.** Per-category bio:sm ratios:
gastrointestinal 7.9x, hematological 7.4x, metabolic 3.8x, neurological 3.2x,
musculoskeletal 3.0x, respiratory 2.7x, autoimmune 1.9x, dermatological 1.7x,
'other' 1.7x, renal 1.4x, immunological 1.0x, ophthalmic 0.75x (n=2.2),
endocrine 0.71x (n=1.0), infectious 0.17x (n=2.2), psychiatric 0.0x (n=1.5).

### Why
The historical "27.3% biologic R@30" was numerically correct (we measured
27.06%). But the comparison number — "41.8% overall R@30" cited in
research_spec.md — came from a different evaluation framework. CLAUDE.md
already lists honest-baseline numbers (26.06% no-treatment kNN, 15.73% KEGG
pathway kNN). Production today gives 16.39% overall. There is no biologic-
recall failure. There IS a biologic-precision gap (bio_p30 = 6.37%).

### Implication
- h906 (DrugBank), h920 (PubMedBERT), h921 (ESM2), h924 (LINCS VAE) all
  need re-justification or closure. Recall premise is dead. Filed as h955.
- Biologic work pivots from RECALL to PRECISION (h953).
- Production loses 4.4pp bio_r30 / 3.91pp overall_r30 vs h940 plain kNN.
  Suspect SELECTIVE_BOOST_CATEGORIES or tier safety filters demoting correct
  top-30 predictions. Filed as h952 (P1, low effort).
- research_spec.md 41.8% baseline must be corrected — it is silently
  poisoning every motivation-baseline calculation. Filed as h954 (P1).

### New hypotheses generated (5)
- h952 (P1): production-pipeline -4pp recall regression diagnosis
- h953 (P2): biologic precision pivot (bio_p30 6.37% → ?)
- h954 (P1): reconcile 41.8% historical vs 16.39% production overall_r30
- h955 (P1): re-justify or close h906/h920/h921/h924
- (h951 itself moved from pending → validated)

### Recommended next steps
1. **h952** (P1, low effort): diagnose where production pipeline loses 4pp.
   Plain-kNN vs production top-30 diff per-disease, stratified by category +
   biologic-vs-SM. If SELECTIVE_BOOST is the cause, decide revert vs accept.
2. **h955** (P1, low effort): walk h906/h920/h921/h924 rationales and re-write
   on a precision motivation, or mark inconclusive-no-longer-motivated.
3. **h954** (P1, low effort): find the commit that produced 41.8% overall
   R@30 and document the methodology delta in research_spec.md.

---

## Previous Session: h940 — Biologic Target-Overlap + kNN Fusion (2026-04-19)

### Hypothesis
h940 is the direct follow-up to h939 (VALIDATED — target-overlap is a biologic-
specific signal, 3-11x stronger than for small molecules in several categories).
Here we ask: if we fuse target-overlap with kNN for biologic candidates in a
mixed pool, can we raise biologic R@30 by >=5pp without overall regression? If
yes, it is a DrugBank-free delivery path for h906's motivation.

### Status: INVALIDATED — fusion hurts biologic R@30 at every alpha

### Experiment
5-seed (42/123/456/789/2024) 80/20 disease holdout, plain kNN (k=20, no category
boost, no MinRank) vs bio-fusion schemes. For each holdout disease:
- kNN_norm[d] = kNN_score[d] / max(kNN_scores)
- overlap_norm[d] = target_overlap[d] / max(overlap) (biologic pool only)
- Fusion: biologic score = alpha*kNN_norm + (1-alpha)*overlap_norm; SM = kNN_norm
- Unified top-30 ranking; bio_r30 = (top30 ∩ bio_GT)/|bio_GT|

### Key findings (aggregate, 5 seeds)
| Scheme      | bio_r30         | sm_r30          | overall_r30     | bio_p30         |
|-------------|-----------------|-----------------|-----------------|-----------------|
| baseline    | 31.42% ± 2.32%  | 19.78% ± 1.79%  | 20.30% ± 1.69%  | 17.58% ± 1.83%  |
| alpha=0.3   | 22.58% ± 2.89%  | 19.36% ± 1.84%  | 19.37% ± 1.75%  |  9.52% ± 2.25%  |
| alpha=0.5   | 25.47% ± 2.50%  | 19.46% ± 1.83%  | 19.56% ± 1.72%  | 12.94% ± 2.21%  |
| alpha=0.7   | 25.51% ± 3.47%  | 19.73% ± 1.80%  | 19.82% ± 1.71%  | 17.90% ± 3.37%  |
| alpha=0.9   | 27.42% ± 2.08%  | 20.10% ± 1.76%  | 20.35% ± 1.68%  | 21.32% ± 2.62%  |

Every alpha REDUCED biologic R@30: deltas -4.00pp to -8.83pp. All fail the +5pp bar.

### Why (with some confidence, not certainty)
- h939's 3-11x biologic/SM ratio was measured RESTRICTED to the 266-drug biologic
  pool. That within-pool signal does not transfer when biologics compete against
  small molecules in a unified top-30: kNN-transfer already encodes that a biologic
  is used in similar diseases, and target-overlap introduces noise for biologics
  whose kNN evidence is strong.
- Secondary finding: baseline biologic R@30 = 31.4% in this framework is ABOVE the
  27.3% biologic-failure baseline quoted in h906/h921/h924 rationales. The "biologic
  failure class" narrative may be narrower or outdated on expanded GT.
- Precision silver lining: alpha=0.9 bio_p30 = 21.3% vs baseline 17.6% (+3.7pp).
  Fusion concentrates biologic top-30 (higher hit-fraction among biologics returned)
  but misses many rare biologic hits not covered by kNN.

### Implication
- Target-overlap is an ANNOTATION / AUDIT signal, not a re-ranker, for biologics.
- h906 (DrugBank target features) motivation should be re-validated — the 27.3%
  baseline may no longer bind. Filed as h951.
- Follow-up directions (h948 coverage stratification, h949 zero-overlap filter,
  h950 fusion-as-annotation) offer precision-side reframings.

### New hypotheses generated
- h948: Biologic coverage stratification — does kNN-only bio_r30 mask rare-biologic failures?
- h949: Biologic zero-overlap safety filter (precision via demotion)
- h950: Biologic fusion-as-annotation feature for confidence calibrator
- h951: Reality-check the 27.3% biologic baseline motivating h906/h921/h924

### Recommended next steps
1. **h951** (P2, low effort): before spending GPU on h921 (ESM2) or license on
   h906 (DrugBank), re-measure biologic R@30 on current production pipeline.
2. **h948** (P3, low effort): biologic train_freq quartile R@30; cheap diagnostic.
3. **h949** (P3, low effort): zero-overlap biologic demotion is a natural precision
   play that inverts h940's re-ranking failure.

---

## Previous Session: h939 — Biologic Target-Overlap Audit (2026-04-19)

### Hypothesis
h939 asks whether target-overlap ranking is actually a biologic signal (as h906
motivates) even though h912/h916 showed it is not a general signal. Test: restrict
the candidate pool to biologics (mAbs, fusion proteins, cytokines — 266 of 11,656
drugs via USAN-suffix/keyword proxy) and compare per-category bio_r30 vs sm_r30.

### Status: VALIDATED — h906 motivation preserved

### Key findings
- **bio_r30 = 26.5%** (mean, n=381 diseases with >=1 biologic GT), vs
  **sm_r30 = 5.5%** overall. Median bio_r30 = 0% (signal is highly concentrated).
- Diseases with >=5 biologic GT: mean **bio_r30 = 40.7%**, median **41.7%** — a
  very clean signal in the dense-biologic-GT subset (n=35).
- 4 categories exceed the pre-registered n>=10, ratio>=3x bar:
  - cardiovascular **11.3x** (41.9% / 3.7%, n=32)
  - hematological **6.3x** (41.7% / 6.7%, n=14)
  - 'other' **4.6x** (23.2% / 5.0%, n=146)
  - autoimmune **3.3x** (20.9% / 6.3%, n=24)
- Cancer narrowly misses at **2.74x** (52.3% / 19.1%, n=56) — within-category SM
  is lifted by shared cancer-gene vocabulary (per h916), but biologics still win.
- **CAVEAT:** precision@30 is NOT higher for biologics (2.24% vs 2.63% SM).
  The biologic pool is small, so top-30 within 266 drugs contains many FPs.
  Recall (if you want biologics, target-overlap finds them) is strong; precision
  (of those top-30 biologics) is not.

### Implication for h906
h906 motivation HOLDS but deployment should be via a *fusion* feature rather than
a replacement ranker: use target-overlap to re-rank biologic-only candidates,
keep kNN score as the primary signal. Follow-up h940 tests fused score on
5-seed holdout with alpha sweep. h921 (ESM2 target embeddings) becomes more
attractive than h906 binary-overlap as a principled successor.

### Hypotheses generated (4)
- **h940** (P2): Biologic-only target+kNN fusion score + alpha sweep
- **h941** (P4): DrugBank biotech flag vs USAN-suffix robustness check
- **h942** (P4): Polypharmacy dilution test for SM target-overlap
- **h943** (P5): Inverse problem — biologic zero-overlap as FILTER signal

### Recommended next step
Run h940 — the only medium-effort, high-impact follow-up. It converts the h939
signal into an actual holdout R@30 gain for biologics without needing DrugBank
registration.

---

## Previous Session: h916 — Target-Overlap Density Audit (2026-04-19)

### Hypothesis
h912 found target-overlap ranking concentrates in cancer (mean target_r30=0.119 vs
0.032 baseline) but is not actionable as a standalone signal. h916 asks whether the
cancer concentration is a drug-target-density artifact (cancer drugs simply have more
targets) or a biological signal. Test: pearson(mean_target_r30, mean_drug_targets)
across 16 categories (n>=10 diseases). Threshold: r>0.7 → density artifact.

### Status: INVALIDATED — cancer signal is NOT a density artifact

### Key findings
- pearson(target_r30, mean_drug_targets) = **0.136** (spearman = -0.07). Drug-target
  count per drug does not explain target_r30 variation across categories.
- pearson(target_r30, mean_gene_set_size) = **0.868** cross-category — but entirely
  driven by cancer being an outlier (gene_set_size=371 vs next <=135). Excluding
  cancer, r drops to **0.353**. Within-category per-disease correlations are all near
  zero (|r|<0.24).
- Cancer's **max_overlap=28.5** vs baseline 2.6 — >10x — and avg_overlap_top30=21.8
  vs 1.9. Both drug targets and cancer disease-gene sets share a common cancer-gene
  vocabulary (oncogenes, TSGs, kinases, cell-cycle). Overlap is biologically
  meaningful but category-specific and does not generalize.

### Implication for h906 (DrugBank)
Raw target-overlap is a cancer-semantic signal, not a general density feature. h906's
premise (target features fix biologic failures across categories) is weakened. New
P2 hypothesis **h939** (biologic-only target_r30 stratification) will confirm or
refute h906 before investing medium-effort DrugBank work.

### New hypotheses
- **h937** (P3): cancer-gene-signature de-biasing for target-overlap
- **h938** (P3): Jaccard vs raw-count overlap for non-cancer categories
- **h939** (P2): biologics-only subset audit — is target-overlap a biologic signal?
- **h940** (P3): cancer target-overlap as within-cancer subtype discriminator

### Files
- scripts/h916_target_density_audit.py
- data/analysis/h916_target_density_audit.json

### Recommended next step
**h939** — same-day priority-2, low-effort. Stratify target_r30 by biologic vs
small-molecule to decide whether h906 DrugBank work is justified or should be
archived.

---

## Previous Session: h931 — Remove h900 Mechanism Fallback Dead Code (2026-04-19)

### Hypothesis
h903 verified the h900 mechanism-only fallback at production_predictor.py:4634 never
fires (0/1034 diseases at full data, 0/207 in seed-42 holdout) because
self.train_diseases is pre-filtered to require GT + embeddings, so the kNN aggregation
always populates drug_scores. h931 (formerly the second-occurrence h910) executes the
removal: ~75 lines of dead code (4663-4737), the misleading smoke test, and stale
sub-reason tag handling. Expected delta: 0pp tier precision since the branch never
fired.

### Status: VALIDATED — code removed, regression marker passes, h393 5-seed re-run
confirms 0pp delta (see data/analysis/h910_h393_postremoval.txt).

### Key findings
- production_predictor.py: replaced 75-line `if not drug_scores:` block (h900 fallback,
  including sub-tier override + literature promotion) with a single coverage_warning
  line. The `else:` branch (kNN/MinRank fusion) is unchanged.
- scripts/test_mechanism_fallback.py: rewritten as a regression marker that asserts no
  production prediction carries the removed sub_reason tags. Passes for 6 rare diseases
  (Niemann-Pick, Gaucher, Huntington, ALS, sickle cell, cystic fibrosis).
- mechanism_only_fallback / mechanism_fallback_literature_strong /
  mechanism_fallback_literature_moderate sub_reason tags are no longer written. No
  deliverable schema enforcement existed for these tags, so no downstream cleanup was
  needed beyond production_predictor.py.
- Discovered + fixed three duplicate hypothesis IDs in research_roadmap.json
  (h910/h911/h912 each appeared twice). Renamed the later occurrences to
  h931/h932/h933. h592, h795, h806 also have unfixed duplicates — flagged as h934.

### Files
- src/production_predictor.py:4663-4669 — replaces lines 4663-4737 (~75 lines deleted)
- scripts/test_mechanism_fallback.py — regression marker
- data/analysis/h910_h393_postremoval.txt — 5-seed holdout output
- research_roadmap.json — h931/h932/h933 renamed; h934/h935/h936 added

### New hypotheses generated (3)
- h934 (P4, low/low): Audit and dedupe stale collisions in research_roadmap.json IDs
- h935 (P3, medium/medium): Backwards-search for other never-fired branches in
  production_predictor (coverage-driven dead-code sweep)
- h936 (P4, low/low): Why does kNN ALWAYS populate drug_scores when disease_id has an
  embedding? — formal-proof of the invariant that made h900 fallback dead

### Recommended next steps
1. **h920 (P2, high/medium)**: PubMedBERT dense embeddings for biologic precision —
   highest-impact pivot still pending; needs vast.ai GPU
2. **h928 (P3, low/low)**: Remap kept-but-symptom-level MeSH names to disorder-level
   IDs — quick follow-up to h908
3. **h935 (P3, medium/medium)**: Coverage-driven dead-code sweep generated from this
   session's finding

---

## Previous Session: h908 — MeSH C23 Symptom Blocklist Validation (2026-04-18)

### Hypothesis
The h901 MeSH expansion (+638 mappings) introduced C23 symptom-level IDs that generate
antibiotic→symptom artifacts in the deliverable (piperacillin→fever, metronidazole→
dysmenorrhea, gentamicin→inflammation). Audit by MeSH tree class, block pure symptom
names, and confirm no tier-precision regression.

### Status: VALIDATED — 45-name blocklist, 300 deliverable rows removed, ≤1.5σ shifts.

### Key findings
- Tree-class classification: 582/638 = 91.2% real diseases, 52/638 = 8.2% C23 symptoms
  (much lower than h902's 54% keyword-only estimate — most keyword-flagged names were real
  cancer descriptors like "advanced breast cancer").
- Conservative blocklist (45 names): pure symptoms with no disorder equivalent. Kept 11
  C23-classified names that have legitimate treatment status (emphysema, hyperuricemia,
  anxiety, recurrent depression); scheduled for remap in h928.
- Deliverable impact: 0 GOLDEN, 0 HIGH, 44 MEDIUM, 96 LOW, 160 FILTER rows removed = 300
  total. The 47 MEDIUM tier antibiotic→symptom artifacts flagged in h902 are eliminated.
- Holdout (5 seeds, vs h904-demoted baseline): MEDIUM +1.3pp (0.23σ), HIGH -2.4pp
  (0.75σ), GOLDEN -3.4pp (1.44σ — attributable to disease-pool change 1034→1016, not
  the blocklist itself since zero GOLDEN rows were removed).

### Files
- `data/reference/h908_symptom_blocklist.json` — 45 blocklist names with provenance.
- `data/analysis/h908_analysis.md` — full audit + validation writeup.
- `data/analysis/h908_classified.json` — tree-class classification for 638 mappings.
- `data/analysis/h908_leaks.json` — 188 non-FILTER deliverable leak predictions.
- `data/analysis/h908_holdout_run.txt` — h393 5-seed holdout output.
- `scripts/h908_classify_mesh.py` — classifier script.
- `src/disease_name_matcher.py:1889`, `src/production_predictor.py:2197` — blocklist loader/filter.
- `src/production_predictor.py:2356` — blocklist added to GT cache key.

### Follow-up hypotheses queued
- **h928** (P3): Remap the 11 kept C23 symptom-level names to their disorder-level MeSH IDs.
- **h929** (P3): Extend tree-class filter to the pre-h901 base set of 917 mappings.
- **h930** (P3): Generic detector for antibiotic/NSAID→non-infectious-non-inflammatory artifacts.

---

## Previous Session: h907 — Ryland Blinded-Review Integration Protocol (2026-04-18)

### Hypothesis
External expert labels (Ryland Mortlock's blinded review of ~855 GOLDEN-tier derm
predictions) are the cleanest signal the project has for calibrating tiers beyond
the DRKG ceiling. Build the leakage-safe ingestion scaffold NOW so the agent does
not stall when the review arrives.

### Status: INCONCLUSIVE — infrastructure complete, review not yet delivered

### Deliverables
- `data/reference/ryland_review_schema.json` — JSON Schema with 5 verdicts (plausible, known, implausible, adverse, unsure).
- `scripts/import_ryland_review.py` — accepts CSV/XLSX/JSON, validates schema, resolves drug/disease names via `production_predictor` alias map, synthesizes `prediction_id = <disease_id>||<drug_id>` when missing, writes `data/reference/expert_labels_ryland.json` with `provenance='expert_ryland'`.
- `src/expert_labels.py` — `ExpertLabels` lookup helper; loads labels but NEVER merges them into `predictor.ground_truth` or `expanded_ground_truth.json` (leakage guarantee).
- `scripts/h907_eval_expert_labels.py` — parallel precision split per tier (`drkg_precision` vs `expert_precision`) with review-coverage column. Runs cleanly today and emits `null` expert column until the review lands.
- `docs/claude/patterns.md` — "Expert-Label Ingestion" section documents flow + leakage-safe rules.

### Validation
- Dry-run on synthetic 5-row sample: 3/5 accepted, 2 correctly rejected (unresolvable name; invalid verdict).
- Eval scaffold on current deliverable (GOLDEN + HIGH): `drkg_precision` = 87.0% / 70.8%, `expert_precision` = null (expected).

### Follow-up hypotheses queued
- **h917** (P2): Run expert_precision split when review lands; flag tiers with |drkg - expert| >= 10pp.
- **h918** (P3): Mine Ryland "adverse" / "implausible" verdicts for new inverse-indication rules, gated on citations + confidence.
- **h919** (P3): Quantify coverage gap — does derm tier precision generalise to other categories?

### Files
- Schema: `data/reference/ryland_review_schema.json`
- Importer: `scripts/import_ryland_review.py`
- Loader: `src/expert_labels.py`
- Evaluator: `scripts/h907_eval_expert_labels.py`
- Baseline output: `data/analysis/h907_expert_label_precision.json`

---

## Previous Session: h912 — Is Target-Overlap Ranking a Valid Repurposing Signal? (2026-04-18)

### Hypothesis
Follow-up to h903. Forced target-overlap ranking gave 5.96% pooled prec@30, below FILTER.
But pooled hides heterogeneity. Segment per-disease R@30 by gene-set size, max overlap,
disease category, kNN baseline, GT size — is there ANY subpopulation where target-overlap
alone supports a usable annotation (mean R@30 ≥ 15% at n ≥ 30)?

### Status: INVALIDATED — no actionable subpopulation, but fusion niche found

### Key Findings
- **No subpopulation reaches R@30 ≥ 15% at n ≥ 30.** Across 2,253 evaluable diseases
  (embed ∩ disease_genes ∩ expanded GT), overall mean target_r30 = 3.47% (median 0%)
  vs kNN baseline 17.95% on the same diseases. kNN dominates 5.2×.
- **Category ordering**: cancer 11.89% (n=104, only category >10%), metabolic 4.41%,
  psychiatric 4.41%, gastrointestinal 2.09%, infectious 0.59%. Cardio, neuro, derm,
  immun all <5%. Cancer's lead is likely drug-target database density, not biology (see h916).
- **Gene-set size**: non-monotonic peak at 51–100 genes (7.36%), not near 15%. Monogenic
  diseases (1–5 genes, n=1,223) are the WORST (2.16%) — opposite of the intuitive
  "mechanism ranking rewards well-characterized pathways" prior.
- **Max overlap**: mean rises with top-1 overlap count (2.50% @ ov=1 → 7.84% @ ov=6–10)
  then plateaus. Even the "max_ov 11+" subpopulation (n=168) sits at 6.35%.
- **Complementary niche found**: 11/104 cancer diseases have target_r30 > knn_r30,
  and on those 11: target mean 25.32% vs kNN 11.22% — a concrete 2× reversal. This
  motivates MinRank fusion for weak-kNN cancer cases (h914), not a general fallback.

### Recommendation
- Close the "mechanism-only as a tier" direction permanently.
- Run **h914** (MinRank fusion for cancer diseases with weak kNN) — existing
  `_minrank_fusion` at `production_predictor.py:3327` is not used in production.
- Pending h914 result, run **h915** (remove fallback block ± helpers) and
  **h916** (audit cancer signal — drug-target density vs biology).

### Files
- Script: `scripts/h912_target_overlap_subpopulation.py`
- Summary: `data/analysis/h912_subpopulation_summary.json`
- Raw records: `data/analysis/h912_per_disease_records.json`

---

## Previous Session: h903 — h900 Mechanism-Only Fallback Dead-Code Audit (2026-04-18)

### Hypothesis
Measure holdout precision of the h900 mechanism-only fallback and decide whether to
formalize it as a tier (MECHANISM tier, promote to MEDIUM, keep as internal, or remove).

### Status: INVALIDATED — fallback is dead code

### Key Findings
- **Fallback never fires.** 0/1034 train_diseases at full data; 0/207 at seed-42 holdout.
  Root cause: `self.train_diseases` is pre-filtered to require both GT and embeddings,
  so every top-20 kNN neighborhood populates `drug_scores`. The `if not drug_scores:`
  guard at `src/production_predictor.py:4634` is unreachable.
- **Forced fallback is worse than FILTER.** Running target-overlap ranking directly on
  the 722 diseases with `disease_genes` ∩ expanded GT yields 5.96% prec@30, 4.21% mean
  per-disease R@30, 0% median. Compare FILTER 9.2%, LOW 11.3%. Not a viable tier.
- **Smoke test was misleading.** `scripts/test_mechanism_fallback.py` reported "MECHANISM
  FALLBACK ACTIVE" via substring match `'mechanism' in category_specific_tier`, which
  collides with the unrelated `mechanism_specific` tag (h297). The h900 fallback never
  actually ran in the smoke test either.

### Recommendation
- Proceed with **h910** (remove 75 lines of dead code at production_predictor.py:4634-4709).
- Do not extend the fallback mechanism — target-overlap alone is a weak repurposing signal
  (5.96% vs FILTER 9.2%). The "rare disease coverage gap" is not solvable by mechanism ranking.
- New hypotheses added: h910 (remove code), h911 (softer trigger audit), h912 (target-overlap
  signal validity by subpopulation), h913 (true zero-coverage rare-disease inventory).

### Files
- Analysis: `data/analysis/h903_fallback_analysis.json`
- Evaluator: `scripts/h903_mechanism_fallback_eval.py` (superseded by direct analysis; kept for reference)

---

## Previous Session: h909 - External-Data Bottleneck Diagnosis (2026-04-18)

### Hypothesis
Identify whether the 569 h901 mappings that failed to become evaluable (per h902) are blocked by
missing DRKG embeddings (fixable by LINCS/h905), missing GT drug pairs (fixable by DrugBank/h906),
or neither. A 2x2 table over {embedding present, GT drugs present} should locate the bottleneck.

### Status: VALIDATED (surprising — invalidates coverage motive for h905/h906)

### Key Findings
- All 1,534 valid MeSH mappings: 95.4% are evaluable (embed+GT). Only 71 (4.6%) blocked on either axis.
- **638 h901-new mappings: 100% have a DRKG embedding. 98.1% also have GT drugs. Yet only 69 (10.8%)
  became pipeline-evaluable per h902.** → 557 data-complete mappings fail for PIPELINE reasons,
  not data reasons (symptom/finding exclusion 54%, holdout sampling, name-resolution).
- LINCS/h905 could unblock 0 of the 638 new mappings; DrugBank/h906 could unblock ≤12. Neither
  can be justified as a **coverage** pivot — they must be justified as **precision** pivots.
- Full report: `data/analysis/h909_bottleneck_report.md`; raw counts: `data/analysis/h909_bottleneck_2x2.json`.

### Strategic Implications
1. **Reframe h905/h906** — precision pivots on biologics / rare-disease features, not coverage.
2. **Prioritize h907 (Ryland labels)** — the only external pivot that already targets precision.
3. **Pipeline hygiene is the real coverage lever** — h908 (symptom filter) and new h910 audit.

### New Hypotheses Generated (3)
- h910 (P2, low): Audit the 557 data-complete-but-non-evaluable h901 mappings; categorize drop points.
- h911 (P2, medium): Reframe h906 as biologic-precision fix (mAbs 27.3% → target ≥32%).
- h912 (P3, low): Quantify holdout-sampling loss per disease across 5 seeds.

### Recommended Next Steps
1. h903 (priority 1) — formalize h900 mechanism-only fallback with holdout eval.
2. h904 (priority 1) — but only demote `infectious_hierarchy_uti` (GOLDEN→HIGH, 0% holdout);
   other flagged rules are FILTER-tier or zero-n and not actionable.
3. h910 (priority 2) — trace the 557 non-evaluable mappings to their pipeline drop point.

---

## Previous Session: h902 - h901 MeSH Expansion Impact (2026-04-18)

### h902: Measure h901 MeSH Expansion Impact on Coverage and Precision — VALIDATED (coverage) + INVALIDATED (precision)

**Methodology:** Compared pre-h901 (commit 80a068e^, `data/analysis/h900_holdout_run.txt`) vs post-h901 (`data/analysis/h901_full_eval_output.txt`) 5-seed holdout evaluations. Diffed MeSH mappings (`git show 80a068e -- data/reference/mesh_mappings_from_agents.json`). Categorized 638 new mappings by keyword heuristics.

**Coverage results:**
- MeSH mappings: 917 → 1,555 (+638, +70%)
- Evaluable diseases (GT + embeddings): **965 → 1,034 (+69, +7.2%)**
- Full-data predictions: 21,164 → 24,713 (+3,549, +16.8%)
- Mapping conversion rate: **10.8%** (638 new mappings → 69 new evaluable diseases)

**Tier precision (holdout, 5-seed mean):**

| Tier | Pre | Post | Δ | σ-units |
|---|---|---|---|---|
| GOLDEN | 87.1% ± 2.7% | 86.7% ± 2.4% | -0.4pp | 0.11σ |
| HIGH | 83.4% ± 4.0% | 80.8% ± 2.8% | -2.6pp | 0.53σ |
| MEDIUM | 38.5% ± 3.6% | 39.7% ± 4.7% | +1.2pp | 0.20σ |
| LOW | 11.3% ± 0.5% | 10.8% ± 0.3% | -0.5pp | 0.86σ |
| FILTER | 9.2% ± 0.5% | 7.9% ± 0.7% | -1.3pp | 1.51σ |

All tier precision shifts are **within noise** (|Δ| < 1.6σ). No significant quality gain or loss.

**Tier volume (predictions per holdout seed):** GOLDEN **0 new preds** (-4%), HIGH +12%, MEDIUM +11%, LOW +23%, FILTER +22%. New diseases generate almost exclusively LOW/FILTER predictions.

**New-mapping category breakdown (638 total):**
- Other (symptoms/findings/qualifiers): **346 (54%)** — e.g., agitation, anorexia, back pain, bradycardia, ascites, vomiting
- Cancer: 82 (13%)
- Pain/symptom: 41 (6%)
- Metabolic: 40 (6%)
- Rare genetic: 36 (6%)
- Infectious: 35 (5%)
- Cardiovascular: 30 (5%)
- Autoimmune: 22 (3%)
- Neurological: 6 (1%)

**Key learnings:**
1. Mapping expansion is at diminishing returns. The 10.8% conversion rate plus zero new GOLDEN predictions disconfirms the hypothesis that disease-name normalization is the binding constraint on rare-disease coverage.
2. 54% of h901 additions are symptoms/findings that should never generate repurposing predictions. Quality-risk audit needed (h908).
3. DRKG embedding coverage is the actual bottleneck for rare diseases — validates the h905 LINCS pilot and h906 DrugBank target-features direction.

**Recommendation (per research_spec.md priority-1 decision point):** Pivot to external data. The DRKG-expansion loop is CLOSED.

### New Hypotheses Generated (5)
- h908: Audit + filter symptom/finding entries from h901 (priority 2, quality risk)
- h909: 2x2 table {embedding, GT} to identify real bottleneck (priority 2, informs h905 vs h906)
- h910: Per-disease predicted-coverage score for targeted mapping (priority 3)
- h911: Deliverable impact audit of h901 (priority 3)
- h912: Update research_spec.md + MEMORY.md CLOSED list (priority 4)

### Recommended Next Steps
1. **h903** (priority 1, low effort): Holdout eval of h900 mechanism-only fallback as formal tier
2. **h904** (priority 1, low effort): Demote 12 overfitted tier rules from h393 audit
3. **h909** (priority 2, low effort): Identify embedding-vs-GT bottleneck before committing to h905/h906
4. **h905** (priority 2, high effort): LINCS L1000 pilot for 5 DRKG-absent rare diseases

### Current Tier State (post-h901, unchanged from h815 within noise)
| Tier | Holdout | Volume |
|------|---------|--------|
| GOLDEN | 86.7% ± 2.4% | 927 preds |
| HIGH | 80.8% ± 2.8% | 1,261 preds |
| MEDIUM | 39.7% ± 4.7% | 895 preds |
| LOW | 10.8% ± 0.3% | 11,195 preds |
| FILTER | 7.9% ± 0.7% | 10,435 preds |

Coverage: 1,034 evaluable diseases.

---

## Previous Session: h814+h815+h817 - HIGH Sub-Reason Audit + MODERATE Promotion + Evidence Split (2026-02-25)

### h814: HIGH Sub-Reason Audit (CS SOC Revert) — VALIDATED

**Methodology:** Analyzed all HIGH sub-reasons for demotion candidates. Found corticosteroid_soc_promotion at 61.4% ± 22.5% holdout (per-seed: 37.5%, 96.2%, 64.0%, 36.4%, 72.7%), 19.9pp below HIGH average. Already implemented by previous session (CS SOC promotion code reverted).

**Impact:** HIGH +2.1pp (81.3% → 83.4%), HIGH variance reduced (±4.8% → ±4.0%).

### h815: MODERATE_EVIDENCE LOW → MEDIUM Promotion — VALIDATED

**Methodology:** Verified already-implemented promotion of LOW predictions with MODERATE literature evidence to MEDIUM. 36.7% holdout (n=41/seed, well above LOW avg of 11.3%). Excludes safety sub-reasons.

**Impact:** MEDIUM +2.2pp (36.3% → 38.5%), MEDIUM variance reduced (±4.6% → ±3.6%). ~213 predictions promoted LOW→MEDIUM.

### h817: NO_EVIDENCE vs WEAK_EVIDENCE Split — INVALIDATED

**Methodology:** Tested splitting h732 demotion: only demote NO_EVIDENCE (14.3%) to LOW, keep WEAK_EVIDENCE (34.1%) at MEDIUM. Implemented change and ran 5-seed holdout.

**Result:** MEDIUM -2.0pp (38.5% → 36.5%). WEAK_EVIDENCE at 34.1% is below the post-h815 MEDIUM average of 38.5%, diluting the tier. REVERTED.

**Key learning:** When baseline improves (h815 raised MEDIUM from 36.3% to 38.5%), previously borderline sub-groups become below-average. Always check against current tier averages.

### Current Tier State
| Tier | Holdout | Δ from h811 |
|------|---------|-------------|
| GOLDEN | 87.1% ± 2.7% | +0.2pp |
| HIGH | 83.4% ± 4.0% | +2.1pp |
| MEDIUM | 38.5% ± 3.6% | +2.2pp |
| LOW | 11.3% ± 0.5% | -0.6pp |
| FILTER | 9.2% ± 0.5% | 0pp |

### New Hypotheses Generated (3)
- h818: WEAK_EVIDENCE per-sub-reason rescue (selective, not blanket)
- h819: Literature evidence coverage gap analysis (% NOT_ASSESSED)
- h820: HIGH variance decomposition by rule

### Recommended Next Steps
1. h818: Selective WEAK rescue for high-holdout sub-reasons (low effort)
2. h819: Literature coverage gap analysis (low effort, informs mining strategy)
3. h812: Holdout-invisible HIGH hierarchy rules (medium effort)

---

## Previous Session: h811 - RA Hierarchy GOLDEN→HIGH Demotion (2026-02-25)

### h811: Autoimmune RA Hierarchy GOLDEN→HIGH Demotion — VALIDATED

**Methodology:** Ran 5-seed holdout evaluation with per-seed breakdown for autoimmune_hierarchy_rheumatoid_arthritis. Previously promoted to GOLDEN at h615 (86.4% ± 8.7%, n=23/seed). Current measurement: 69.0% ± 28.8% (n=16/seed).

**Key findings:**
1. Seed 42: 0 RA predictions in holdout (all RA diseases in training)
2. Seed 123: 22.2% (2/9) — terrible outlier
3. Seed 456: 72.2% (26/36) — below GOLDEN
4. Seed 789: 100.0% (9/9) — perfect but tiny n
5. Seed 2024: 81.8% (9/11) — borderline

**Root cause:** n dropped from 23→16/seed due to GT and rule changes since h615. Extreme variance (22-100%) indicates small-n instability. RA at 69.0% is well below GOLDEN (85.3%) and below HIGH average (81.3%).

**Implementation:** Removed rheumatoid_arthritis from HIERARCHY_PROMOTE_TO_GOLDEN and TARGET_OVERLAP_GOLDEN_ELIGIBLE_RULES.

**Tier impact:**
| Tier | Before | After | Delta |
|------|--------|-------|-------|
| GOLDEN | 85.3% ± 2.6% | 86.9% ± 2.7% | **+1.6pp** |
| HIGH | 81.8% ± 5.5% | 81.3% ± 4.8% | -0.5pp |
| MEDIUM | 36.3% ± 4.6% | 36.3% ± 4.6% | 0pp |
| LOW | 11.9% ± 0.5% | 11.9% ± 0.5% | 0pp |
| FILTER | 9.2% ± 0.5% | 9.2% ± 0.5% | 0pp |

**Key learning:** Hierarchy rules with <20 preds/seed are unreliable — always check per-seed breakdown before trusting aggregate. Small-n rules can show 86% one evaluation and 69% another due to GT/rule changes affecting which predictions appear.

### New Hypotheses Generated (3)
- h812: Holdout-invisible HIGH hierarchy rules (MS/sepsis/asthma all 0% holdout)
- h813: GOLDEN variance reduction post-RA demotion
- h814: HIGH sub-reason audit for weakest rules

### Recommended Next Steps
1. h814: HIGH sub-reason audit (low effort, directly builds on h811 findings)
2. h812: Holdout-invisible hierarchy rules (medium effort, important for tier integrity)
3. h807: Literature evidence score gradient (low effort, refine existing feature)

---

## Previous Session: h808 - Literature High Demotion Sub-Reason Audit (2026-02-25)

### h808: Literature High Demotion Sub-Reason Audit — VALIDATED

**Methodology:** Analyzed all predictions demoted HIGH→MEDIUM by literature evidence (h791), breaking down by their original HIGH sub-reason. Used h808 analysis script (pre-existing) which ran 5-seed holdout on each original sub-reason × literature level combination.

**Key findings:**
1. `default_freq10_nomech_r1_5` (n=29/seed): 36.4% holdout — correctly MEDIUM, PROTECTED
2. `cancer_same_type_mech_rank10` (n=8/seed): 13.4% holdout — LOW quality, DEMOTED
3. `default` misc HIGH (n=7/seed): 15.7% — LOW quality, DEMOTED
4. All other sub-reasons (n<5/seed): LOW quality but tiny sample, DEMOTED

**Implementation:** Modified production_predictor.py to only protect `default_freq10_nomech_r1_5` from the MEDIUM→LOW literature demotion. All other literature-demoted HIGH sub-reasons now fall through to LOW.

**Tier impact:**
| Tier | Before | After | Delta |
|------|--------|-------|-------|
| GOLDEN | 85.3% ± 2.6% | 85.3% ± 2.6% | 0pp |
| HIGH | 81.8% ± 5.5% | 81.8% ± 5.5% | 0pp |
| MEDIUM | 34.8% ± 5.1% | 36.3% ± 4.6% | **+1.5pp** |
| LOW | 11.8% ± 0.6% | 11.9% ± 0.5% | +0.1pp |
| FILTER | 9.2% ± 0.5% | 9.2% ± 0.5% | 0pp |

**Key learning:** Blanket protection rules should be audited for sub-group heterogeneity. The literature_high_demotion protection was treating all original HIGH sub-reasons equally, but cancer_same_type_mech_rank10 was 13.4% while default_freq10_nomech_r1_5 was 36.4%.

### New Hypotheses Generated (3)
- h809: Literature evidence coverage expansion (mine more HIGH/MEDIUM predictions)
- h810: MEDIUM variance reduction (identify seed-sensitive sub-reasons)
- h811: Autoimmune RA hierarchy GOLDEN→HIGH demotion (69.0% holdout)

### Recommended Next Steps
1. h811: Quick RA hierarchy demotion check (low effort, clear OVERFITTED signal)
2. h808 sub-analysis: Check other OVERFITTED rules for actionable demotions
3. h809: Expand literature mining coverage for better tier calibration

---

## Previous Session: h797+h798+h803 - Cancer GOLDEN Promotion + Other Category Exclusion + LOW Rescue (2026-02-25)

### h797: Cancer Same-Type Mech Rank10 → GOLDEN — INVALIDATED

cancer_same_type_mech_rank10 has 69.5% ± 13.5% holdout (full-data 85.4%, -15.9pp delta). Well below GOLDEN threshold (85.3%). Flagged as OVERFITTED. Cancer same-type predictions have large full-data/holdout gap because many cancer drugs treat the same subtypes in training. Reverted to HIGH where it achieves 86.2% ± 9.2% holdout (n=29/seed).

### h798: Literature Strong Low 'Other' Category Exclusion — VALIDATED

Excluding category='other' from literature_strong_low_promotion: other=39.0% ± 6.1% vs non-other=85.7% ± 6.0%. Uncategorized diseases lack kNN structure. 69 predictions moved HIGH→LOW. **HIGH +2.3pp (79.5% → 81.8%)**.

### h803: Literature-Free LOW Tier Rescue — INCONCLUSIVE

Best LOW subset: rich_disease(GT≥30)+high_score(≥3) at 37.8% ± 6.9% (n=78/seed, non-CS=32.5%). But would rescue predictions from correctly-calibrated safety demotion rules. Disease GT richness is strongest LOW signal (24% for GT≥50 vs 3% for GT≤4). No clean, safe promotion rule found.

### New Hypotheses Generated (3)
- h804: LOW rich-disease score-gate rescue with mechanism + safety exclusions
- h805: Literature high demotion tightening (NO vs WEAK evidence split)
- h806: Autoimmune RA hierarchy GOLDEN→HIGH demotion (69.0% holdout)

### Recommended Next Steps
1. h806: Quick RA hierarchy demotion check (low effort, clear signal)
2. h805: Literature demotion tightening (low effort)
3. h804: LOW rescue with safety gates (medium effort)

---

## Previous Session: h757 - Post-h744 MEDIUM/HIGH Sub-Reason Audit (2026-02-24)

### h757: Comprehensive Sub-Reason Holdout Audit — VALIDATED

**Methodology:** Ran 5-seed holdout evaluation on all MEDIUM and HIGH sub-reasons. Identified and demoted weak rules. Fixed pneumonia hierarchy bug.

**Key findings:**
1. **MEDIUM demotion:** `default_freq10_nomech_r6_10` = 26.5% ± 11.1% holdout (n=31/seed) — demoted to LOW (124 preds)
2. **HIGH demotions to MEDIUM:** comp_to_base_high (18.8%), fluoroquinolone respiratory (21.8%), neurological class match (8.6%), reproductive hormones (28.6%)
3. **Hierarchy fixes:** UTI→GOLDEN (80.0%), diabetes HIGH→MEDIUM (21.1%), skin_infection→MEDIUM (25.0%), epilepsy+gout→LOW
4. **Bug fix:** pneumonia was accidentally removed from HIERARCHY_DEMOTE_TO_LOW

**Combined tier impact:**
| Tier | Before | After | Delta |
|------|--------|-------|-------|
| GOLDEN | 68.9% | 78.1% | **+9.2pp** |
| HIGH | 53.7% | 59.4% | **+5.7pp** |
| MEDIUM | 33.4% | 36.0% | **+2.6pp** |
| LOW | 13.7% | 13.9% | +0.2pp |
| FILTER | — | 10.0% | — |

**Key learning:** Rules originally calibrated on full-data precision (h183, h265) are severely overfitted — full-data 26-67% but holdout 8-29%. All category-specific tier rules need holdout validation.

### New Hypotheses Generated (3)
- h758: target_overlap_promotion sub-analysis (35.2% with low variance — biggest reliable MEDIUM sub-reason)
- h759: Neurological/respiratory MEDIUM → LOW re-evaluation (very low holdout, small n)
- h760: Hierarchy rule holdout-invisible audit (many 0-n rules)

### Recommended Next Steps
1. h758: target_overlap_promotion investigation (medium effort, reliable signal)
2. h747: Continue literature mining expansion (in-progress, high effort)
3. h759: Quick neurological/respiratory MEDIUM check

---

## Previous Session: h731 - Automated Literature Mining Validation (2026-02-07)

### h731: Automated Literature Mining on GOLDEN/HIGH/MEDIUM NOVEL predictions — VALIDATED

**Methodology:** Analyzed 590 pre-mined predictions from automated literature mining pipeline (PubMed + ClinicalTrials.gov). Ran holdout evaluation by evidence level. Identified and validated GT gaps. Added confirmed pairs to expanded GT.

**Key findings:**
1. **Literature evidence independently predicts holdout precision** — strongest signal we've found
2. STRONG_EVIDENCE: 78.7% ± 2.6% holdout (after 31 GT fixes; 67.0% before)
3. MODERATE_EVIDENCE: 32.1% ± 7.5% (below MEDIUM)
4. WEAK/NO_EVIDENCE: 10-20% (LOW quality)
5. Evidence score quartile gradient: Q1=82.5%, Q2=56.7%, Q3=16.9%, Q4=19.5%

**Cross-tabulation (breakthrough):**
- MEDIUM + STRONG_EVIDENCE: **67.4% ± 5.1%** — clearly HIGH-tier quality (139 predictions)
- HIGH + STRONG_EVIDENCE: 91.2% ± 2.2% — GOLDEN-tier quality
- MEDIUM + NO_EVIDENCE: 16.5% ± 4.4% — LOW-tier quality

**GT gaps added:** 31 new pairs (13 from GOLDEN/HIGH + 19 from MEDIUM − 1 removed invalid)
- Key additions: bevacizumab→esophageal/uterine cancer, azathioprine/tacrolimus→alopecia areata, multiple chemotherapy→cancer subtype pairs, antifungals→specific infections, antibiotics→specific infections

**Inverse indication found:** Gentamicin → kidney failure/CKD/AKI (aminoglycoside nephrotoxicity)

**False positive patterns in STRONG_EVIDENCE:**
- Procedural co-occurrence (lidocaine→edema)
- Macrolide spectrum mismatch (erythromycin→TB, meningitis)
- Adverse effect mimicking indication (CS→edema)

### New Hypotheses Generated (4)
- h732: Literature evidence tier promotion MEDIUM+STRONG → HIGH (priority 3)
- h733: Mine remaining 13,500 LOW/FILTER predictions (priority 4)
- h734: MEDIUM NO_EVIDENCE demotion to LOW (priority 4)
- h735: False positive analysis in STRONG_EVIDENCE (priority 5)

### Recommended Next Steps
1. h732: Implement literature evidence as tier rule (HIGH impact but needs full mining)
2. h712: Disease name synonym expansion (high priority, in-progress)
3. h734: Quick check on MEDIUM NO_EVIDENCE demotion
4. Continue low-effort in-progress hypotheses (h681, h706, h727)

---

## Previous Session: h718+h730+h686 - Cancer Targeted Therapy Analysis + Drug Name Aliasing (2026-02-07)

### h718: Cancer Targeted Therapy LOW Re-evaluation — INVALIDATED

**Methodology:** Analyzed 496 cancer_targeted_therapy LOW predictions by rank, mechanism, and frequency. Full-data precision=36.0% suggested possible rescue for high-rank+mechanism subset. Ran 5-seed holdout evaluation across all subsets.

**Key findings:**
1. Full-data to holdout inflation is 5.9x (36.0% → 6.1%) — worst in entire tier system
2. Even best subset R1-5+mech: 19.3% ± 16.6% (n=10.6/seed, unreliable)
3. R1-10+mech: 13.6% ± 9.7% — z=-2.55 vs MEDIUM, consistent with LOW
4. Biological explanation: targeted therapies work via specific biomarkers (BRAF, HER2, BRCA) not captured in DRKG disease embeddings
5. h598 demotion CONFIRMED correct — no rescue possible via kNN

### h730: Immunotherapy vs Kinase Inhibitor Split — INVALIDATED

**Methodology:** Split CANCER_TARGETED_THERAPY into sub-classes and ran per-class holdout.

**Key findings:**
1. Checkpoint inhibitors: 10.1% ± 9.4% (n=24.2/seed) — LOW
2. Kinase inhibitors: 5.9% ± 4.3% (n=53.2/seed) — firmly LOW
3. Anti-target mAbs: 2.7% ± 3.3% — very LOW
4. PARP inhibitors: 0.0% — zero holdout hits
5. Even pan-cancer immunotherapy doesn't transfer via kNN

### h686: Drug Name Aliasing — VALIDATED

**Methodology:** Systematic discovery of EC→DrugBank drug name mismatches. Built 34 new aliases covering INN variants, salt forms, combo products, biologic variants.

**Key findings:**
1. +85 GT pairs recovered across +10 diseases (4884→4969 pairs, +1.7%)
2. Key drugs: piperacillin (F=16), HCTZ (F=11), clopidogrel (F=9), iodoquinol (F=9)
3. **MEDIUM: 42.8% ± 1.8% (was 38.3% ± 4.7%)** — recovered and more stable
4. **GOLDEN: 82.9% ± 10.6% (was 71.8% ± 8.0%)**
5. HIGH: 57.6% ± 6.3% (was 61.4%) — slight drop but within noise
6. LOW/FILTER: unchanged

### New Hypotheses Generated (3)
- h728: Biomarker-matched targeted therapy rescue via DRKG mutation edges (priority 4)
- h729: Full-data inflation index for deliverable quality auditing (priority 5)
- h730: Immunotherapy vs kinase inhibitor split (completed/invalidated)

### Recommended Next Steps
1. Continue with remaining in-progress hypotheses (h712, h717, h727)
2. h728: Biomarker-based targeted therapy rescue (fundamentally different from kNN)
3. Priority 4 medium-effort hypotheses (h367, h375)

---

## Previous Session: h720+h723 - Gene-Poor Disease Supplementation + Antimicrobial Spectrum Matching (2026-02-07)

### h720: Gene-Poor Disease Supplementation via External Databases — INVALIDATED

**Methodology:** Investigated whether external gene-disease databases (CTD, DisGeNET, Open Targets) could provide mechanism support for 41 zero-gene diseases (diseases with no DRKG gene associations). Downloaded and analyzed CTD gene-disease associations (122.7M records). Tested both curated (direct evidence) and computationally inferred gene sets.

**Key findings:**
1. 41 zero-gene diseases generate 1,174 predictions (117 standard LOW promotable to MEDIUM with mechanism)
2. 24/41 (59%) are infectious diseases — caused by pathogens, not human gene variants
3. CTD direct evidence: only 8/41 diseases have curated genes (1-7 genes each) → **0 mechanism overlaps**
4. CTD inferred genes: 6K-23K genes per disease → 60-85% of ALL drugs overlap (non-discriminative noise)
5. Comparison: DRKG asthma (189 genes) = 10.6% drug overlap (specific); CTD conjunctivitis (22,799 genes) = 84.8% drug overlap (noise)
6. Open Targets: similar breadth problem (500-2800 targets per disease)

**Root cause:** Gene-disease associations are fundamentally inappropriate for mechanism checking in infectious diseases. Their etiology is pathogen-based, not human-genetic.

### h723: Antimicrobial Spectrum as Mechanism Proxy — INVALIDATED

**Methodology:** Tested whether antimicrobial drug class matching pathogen type (antifungal→fungal, antibacterial→bacterial) could serve as a mechanism-equivalent signal for tier promotion of infectious disease predictions.

**Key findings:**
1. 77 infectious LOW predictions have spectrum match (39 zero-gene + 38 from other infectious diseases)
2. Full-data precision: 35.1% (match) vs 23.9% (non-antimicrobial) — promising +11.2pp
3. **HOLDOUT precision: 29.2% ± 14.7% (match) vs 31.0% ± 6.4% (non-antimicrobial) — NO significant difference**
4. n per seed only 2-15 (far below n≥30 minimum for reliable holdout)
5. Z vs MEDIUM = -0.59, Z vs LOW = +1.00 — inconclusive
6. Spectrum matching correctly used for demotion (h560, 0% mismatch) but positive signal too weak for promotion

### New Hypotheses Generated (5)
- h723: Antimicrobial spectrum matching (completed/invalidated)
- h724: Non-infectious gene-poor disease audit (priority 5)
- h725: CTD supplementation for gene-sparse (1-10 gene) diseases (priority 5)
- h726: Stratified holdout splitting for infectious diseases (priority 5)
- h727: Infectious GT gap audit for OTC antifungals (priority 5)

### Recommended Next Steps
1. h724: Non-infectious gene-poor disease audit (quick, low effort)
2. h727: GT gap audit for OTC antifungals (could rescue GT precision)
3. Continue with priority-4 medium-effort hypotheses

---

## Previous Session: h719 - DRKG Mechanism Path Coverage (2026-02-07)

### h719: DRKG Mechanism Path Coverage — VALIDATED (Meta-Analysis)

**Methodology:** Comprehensive analysis of why LOW-tier predictions lack mechanism support (drug-target ∩ disease-gene overlap). Scanned all 1057 diseases, collected 3186 LOW no-mechanism predictions at R1-20. Classified root causes and tested 2-hop indirect mechanism paths through DRKG gene-gene network.

**Key findings:**
1. **627 predictions would become MEDIUM if they had mechanism** (43% tier expansion potential)
2. **Root cause breakdown of missing mechanism:**
   - 22.2% (139 preds across 41 diseases): Disease has NO gene annotations
   - 77.8% (488 preds): Drug targets and disease genes exist but don't overlap
3. **2-hop indirect paths exist for 68.2%** (333/488) of no-overlap predictions
4. **2-hop gene count is a gradient signal on full-data:** 0→14.3%, 1→13.9%, 2-3→18.7%, 4-10→22.4%, 11+→30.1%
5. **BUT 2-hop fails on holdout:** 2hop≥4 = 25.5% ± 8.5% vs no-2hop = 23.2% ± 5.4% — NOT significant
6. **Promoting 2hop≥4 to MEDIUM** would dilute MEDIUM by -3.1pp (55.1→52.1%)
7. **Category variation large:** psychiatric 52.8%, dermatological 37.8%, ophthalmic 0%, autoimmune 8.1%

**Conclusion:** DRKG gene-gene network too dense (27K genes, 2.35M edges) — at 2 hops, specificity collapses. Direct mechanism (1-hop) is fundamentally different from indirect (2-hop). The binding constraint on MEDIUM tier is NOT fixable by expanding mechanism path length. Requires: (a) gene-poor disease supplementation with external databases, (b) category-specific mechanism approaches.

**Tier x Mechanism (R1-20, has_targets):**
| Tier | With Mech | No Mech | Total | Mech % |
|------|-----------|---------|-------|--------|
| GOLDEN | 328 | 132 | 460 | 71.3% |
| HIGH | 455 | 456 | 911 | 49.9% |
| MEDIUM | 525 | 941 | 1466 | 35.8% |
| LOW | 648 | 3186 | 3834 | 16.9% |
| FILTER | 224 | 1640 | 1864 | 12.0% |

### New Hypotheses Generated (3)
- h720: Gene-poor disease supplementation via DisGeNET/OMIM (medium impact)
- h721: 2-hop mechanism as deliverable annotation (low impact)
- h722: Category-specific 2-hop mechanism for psychiatric/dermatological (low impact)

### Recommended Next Steps
1. h720: Gene-poor disease supplementation (adds DIRECT mechanism evidence)
2. h722: Category-specific 2-hop for psychiatric/dermatological (52.8% full-data)
3. Continue with other pending priority-4 medium-effort hypotheses

---

## Previous Session: h710 - Hemangioendothelioma Cancer Type Fix (2026-02-07)

### h710: Hemangioendothelioma Category Fix — VALIDATED (Deliverable Quality)

**Methodology:** Added 'vascular_proliferative' cancer type to CANCER_TYPE_KEYWORDS for hemangioendothelioma, lymphangioleiomyomatosis, and lymphangioma. These benign/intermediate proliferative disorders are categorized as cancer but weren't recognized by extract_cancer_types(), causing their GT drug (sirolimus) to be blocked by cancer_no_gt filter.

**Key findings:**
1. Sirolimus → hemangioendothelioma: FILTER → LOW (cancer_same_type_no_mechanism)
2. Sirolimus → lymphangioma: FILTER → LOW (same)
3. Everolimus → LAM: correctly typed as vascular_proliferative
4. Holdout unchanged (all tiers within normal variance)

---

## Previous Session: h708 - Anti-VEGF Retinal Disease Whitelist (2026-02-07)

### h708: Anti-VEGF Retinal Disease Whitelist — VALIDATED (Deliverable Quality)

**Methodology:** Added validated complication drug bypass before freq<=2 filter and zero_precision_mismatch filter in `_assign_confidence_tier()`. Used existing `COMPLICATION_VALIDATED_DRUGS` infrastructure.

**Key findings:**
1. **3 predictions rescued from FILTER:**
   - Ranibizumab → PDR: FILTER → HIGH (R2, via diabetes hierarchy)
   - Ranibizumab → ROP: FILTER → LOW (R1, standard tier)
   - Aflibercept → ROP: FILTER → MEDIUM (R4, via ATC coherent ophthalmic)
2. **All 3 are genuine GT drugs at top ranks (R1-R4)**
3. **Root cause:** Anti-VEGF drugs have tiny GT footprint (2-3 diseases) → low kNN frequency. Aflibercept also caught by ATC L→ophthalmic mismatch (first ATC=L01, second=S01).
4. **Holdout unchanged:** All tier precisions within normal variance. Deliverable quality improvement only.

### Tier Status (post h708)
| Tier | Holdout | Std | Previous |
|------|---------|-----|----------|
| GOLDEN | 72.5% | ± 6.5% | 72.5% |
| HIGH | 61.0% | ± 7.7% | 61.0% |
| MEDIUM | 37.9% | ± 5.0% | 37.9% |
| LOW | 14.4% | ± 1.2% | 14.4% |
| FILTER | 9.5% | ± 1.0% | 9.5% |

### Recommended Next Steps
1. Continue with remaining low-effort hypotheses (h710, h704, h702)
2. Deliverable regeneration to capture all recent fixes

---

## Previous Session: h707 - Zero-Prediction Disease Rescue (2026-02-07)

### h707: Zero-Prediction Disease Rescue — VALIDATED (Meta-Analysis)

**Methodology:** Scanned all 497 diseases to identify the 28 with zero predictions. Classified by root cause (no DRKG embedding vs filtered predictions). Tested parent disease transfer for 9 diseases with clear DRKG parents. Checked TransE coverage. Assessed medical plausibility of all rescue approaches.

**Key findings:**
1. **24/28 diseases have no DRKG embedding** — 20 completely absent from DRKG (newer MESH terms post-2016), 4 have edges as wrong entity type
2. **4/28 have embeddings but all predictions filtered** — 3 of these have CORRECT predictions incorrectly filtered:
   - PDR: Ranibizumab at R2 (GT drug, standard of care) → filtered by complication_non_validated
   - ROP: Ranibizumab R1 + Aflibercept R4 (BOTH GT drugs) → filtered by mixed rules
   - Hemangioendothelioma: Sirolimus R1 (GT drug) → filtered by cancer_no_gt
3. **Parent disease transfer works for 5/9 subtypes:**
   - HoFH → FH: 5/5 GT drugs found (26.3% precision, GOLDEN quality)
   - Esophageal SCC → esophageal cancer: 3/4 GT drugs (11.5%)
   - MZL → NHL: 2/2 GT drugs (10.5%)
   - Ovarian carcinoma → ovarian cancer: 1/2 GT drugs
   - MDS → AML: 1/3 GT drugs
4. **TransE useless:** Only 1/24 missing diseases has TransE embedding

**Actionable findings:**
- Anti-VEGF whitelist for retinal diseases: HIGH impact, LOW effort → h708
- Parent disease mapping infrastructure: HIGH impact, HIGH effort → h709
- Hemangioendothelioma category fix: LOW impact, LOW effort → h710

### New Hypotheses Generated (4)
- h708: Anti-VEGF retinal disease whitelist (rescues PDR + ROP)
- h709: Disease subtype-to-parent mapping infrastructure
- h710: Hemangioendothelioma category reclassification
- h711: DRKG disease coverage gap quantification

### Recommended Next Steps
1. **h708**: Anti-VEGF whitelist (LOW effort, rescues 3 GT drugs across 2 diseases)
2. **h710**: Hemangioendothelioma fix (LOW effort, rescues 1 GT drug)
3. **h709**: Parent disease mapping (HIGH effort but rescues ~12 drugs across 5 diseases)

---

## Previous Session: h703 - Drug DRKG Coverage Gap (2026-02-07)

### h703: Drug DRKG Coverage Gap Analysis — VALIDATED (Meta-Analysis)

**Methodology:** Ran predictions for all 497 diseases using correct API (disease names, not IDs). Tracked which of 1,158 GT drugs appear in predictions. Analyzed prediction rate by GT size, drug class, and disease characteristics.

**Key findings:**
1. **730 unique drugs** generate predictions across all diseases (7% of 10,474 DRKG compounds, 63% of 1,158 GT drugs)
2. **478 GT drugs (41.3%) are invisible** — they never generate predictions for any disease
3. **Critical mass threshold:**
   - GT=1 disease: 31.8% prediction rate (386 drugs invisible)
   - GT=2: 59.1%
   - GT≥3: 83.6%
   - GT≥6: 100%
4. **Corticosteroid dominance:** Top 9 of 20 most-predicted drugs are CS. Prednisolone predicted for 143/469 diseases (30.5%)
5. **Largest invisible drug classes:** Biologics (54 drugs), antihypertensives (20), antibiotics (17), antidiabetics (10)
6. **28 diseases generate zero predictions** — mostly rare/specific: poisonings, rare cancers (mesothelioma, lip cancer), rare syndromes (Bardet-Biedl)
7. **Non-predicting drugs have mean GT=1.3** vs 3.6 for predicting drugs

**Actionable insight:** GT expansion for single-indication drugs is highest-ROI improvement. Adding just 1 indication to GT=1 drugs doubles prediction rate (31.8%→59.1%).

### New Hypotheses Generated (3)
- h705: GT expansion for single-indication drugs (highest-ROI GT expansion)
- h706: Corticosteroid prediction concentration analysis
- h707: Zero-prediction disease rescue via alternative methods

### Recommended Next Steps
1. **h705**: GT expansion for single-indication drugs (high impact, high effort)
2. **h707**: Zero-prediction disease rescue (medium impact, medium effort)
3. Continue with remaining low/medium-effort data quality hypotheses

---

## Previous Session: h700 - NLP Limitation-of-Use Boilerplate (2026-02-07)

### h700: NLP Limitation-of-Use Boilerplate — VALIDATED (Data Quality, Zero Metric Impact)

**Methodology:** Systematic scan of ALL 10,224 EC indicationList rows for negative patterns ("not indicated", "not recommended", "should not be used", "limitations of use", etc.). For each row with negative patterns, checked whether the listed disease appears ONLY in the negative/limitation context (false GT) or also in positive indication context (legitimate).

**Key findings:**
1. **31 false GT entries** identified from NLP limitation-of-use extraction across 8 drug classes:
   - Oral antidiabetics → T1D/DKA (27 entries, already handled by h675)
   - ICS → bronchospasm (7 entries: budesonide, beclomethasone, fluticasone, ciclesonide, formoterol/mometasone)
   - 5-ARIs → prostate cancer prevention (2 entries: finasteride, dutasteride+tamsulosin)
   - Becaplermin → pressure/venous ulcers (3 entries)
   - Anti-IL5/TSLP → status asthmaticus (2 entries: reslizumab, tezepelumab)
   - Triptans/ergots → specific migraine subtypes (3 entries: rizatriptan, dihydroergotamine)
   - Anifrolumab → severe lupus subtypes (2 entries: lupus nephritis, CNS lupus)
   - Individual cases: sitagliptin→pancreatitis, doxylamine/pyridoxine→hyperemesis gravidarum

2. **5 false pairs removed** from internal GT (3,060→3,055) and expanded GT (57,445→57,440):
   - finasteride → prostate cancer (not approved for cancer prevention; FDA rejected PCPT data)
   - sitagliptin → pancreatitis (DPP-4i CAUSES pancreatitis — INVERSE indication)
   - empagliflozin → DKA (SGLT2i CAUSES euglycemic DKA — INVERSE indication)
   - liraglutide → T1D (GLP-1 agonist requires beta cells; FDA: not for T1D)
   - semaglutide → T1D (GLP-1 agonist requires beta cells; FDA: not for T1D)

3. **ZERO prediction or holdout impact** — all affected drugs/diseases outside kNN prediction space:
   - Affected drugs have DRKG embeddings but generate zero novel predictions (100% self-referential diseases)
   - DPP-4i, SGLT2i, GLP-1 agonists all in DRKG but insular in disease neighborhoods (T2D, heart failure)
   - ICS, triptans, 5-ARIs, becaplermin, biologics: either not in DRKG or target diseases not in predictor

4. **Pattern insight:** ~3% of EC rows are false GT from limitation-of-use NLP extraction. FDA labels use standard templates per drug class: "Important Limitations of Use: [drug] should not be used for [X]" and NLP extracts X as an indication. Clusters by drug class.

### Tier Status (post h700 — unchanged)
| Tier | Holdout | Std | Previous |
|------|---------|-----|----------|
| GOLDEN | 72.5% | ± 6.5% | 72.5% |
| HIGH | 61.0% | ± 7.7% | 61.3% |
| MEDIUM | 37.9% | ± 5.0% | 37.9% |
| LOW | 14.3% | ± 1.2% | 14.4% |
| FILTER | 9.6% | ± 1.0% | 9.5% |
All changes within normal holdout variance.

### New Hypotheses Generated (3)
- h702: ICS bronchospasm safety filter (controller vs reliever mismatch)
- h703: Newer drug DRKG coverage gap quantification
- h704: DPP-4i pancreatitis inverse indication (class-wide safety check)

### Recommended Next Steps
1. **h703**: Newer drug coverage gap (medium impact, medium effort) — would reveal systematic blind spots
2. **h702/h704**: Safety audits (low impact but defensive)
3. Consider pivoting to higher-impact work: deliverable regeneration or external data integration

---

## Previous Session: h675 - FDA Label Contraindication Mining (2026-02-07)

### h675: Systematic FDA Label Contraindication Mining — VALIDATED (Safety Improvement)

**Methodology:** Mined EC indication text for 5 negative patterns: 'contraindicated in', 'should not be used', 'not indicated for', 'not recommended', 'not effective'. Matched disease names appearing in negative context against the disease listed as the indication.

**Key findings:**
1. **37 false GT entries** identified where NLP extracted disease names from "limitations of use" sections, not from indication context. Major pattern: ALL oral antidiabetic labels include "should not be used for type 1 diabetes mellitus or diabetic ketoacidosis" boilerplate.
2. **None of the 37 false GT drugs are in DRKG** — no prediction or holdout impact from GT cleanup alone.
3. **CRITICAL safety fix:** 7 oral antidiabetics predicted at GOLDEN/HIGH for T1D are medically incorrect:
   - 2 GOLDEN: Glimepiride, Rosiglitazone → T1D (sulfonylurea/TZD, require beta cells)
   - 5 HIGH: Glipizide, Pioglitazone, Glyburide, Nateglinide, Repaglinide → T1D
   - All moved to FILTER via inverse_indication
   - Sulfonylureas/meglitinides stimulate insulin release from beta cells; T1D has autoimmune beta cell destruction → zero efficacy
   - TZDs require endogenous insulin production → ineffective in T1D
4. **Conservative approach:** Did NOT filter SGLT2i (dapagliflozin), GLP-1 (liraglutide, semaglutide), DPP-4i (sitagliptin), metformin, or alpha-glucosidase inhibitors (miglitol) for T1D — these have some evidence for adjunctive use.

### Tier Status (post h675)
| Tier | Holdout | Std | Previous |
|------|---------|-----|----------|
| GOLDEN | 72.5% | ± 6.5% | 71.8% |
| HIGH | 61.3% | ± 7.7% | 61.4% |
| MEDIUM | 37.9% | ± 5.0% | 38.3% |
| LOW | 14.4% | ± 1.2% | 14.4% |
| FILTER | 9.5% | ± 1.0% | 9.5% |
Note: All changes within normal holdout variance. Safety improvement is the main value.

### New Hypotheses Generated (3)
- h699: Comp_to_base T1D exclusion (split T1D from diabetes hierarchy)
- h700: NLP limitation-of-use boilerplate (other drug classes)
- h701: Alpha-blocker uroselective/non-selective distinction

### Recommended Next Steps
1. **h699**: T1D hierarchy separation (medium impact, medium effort)
2. **h675 follow-up**: Check if other disease subtypes have similar hierarchy issues (e.g., gestational diabetes)
3. **h672**: CS GT gap expansion

---

## Previous Session: h678 - Combo Product Drug Mismatch Audit (2026-02-06)

### h678: Combo Product Drug Mismatch Audit — VALIDATED (Data Cleanliness, No Metric Impact)
Comprehensive audit extending h677 (lidocaine/bupivacaine combo product fix) to all drugs.

**Methodology:** Identified 71 shared-text groups in EC data, 35 involving DRKG drugs. Classified each as legitimate (combo therapy approval) vs contamination (excipient/preservative inheriting active drug indications).

**Key finding:** h677 already caught the ONLY two impactful cases (lidocaine and bupivacaine). Other contamination involves non-EC excipients (inert) or dual-use substances (handled by existing fixes):
- Edetic acid: 22 false expanded GT entries removed (preservative). Has 247 DRKG edges but only 3 predictions (2 correct).
- Chloride ion: 13 false expanded GT entries removed (ion, not a drug).
- Calcium carbonate: 6 false GT entries (from chemo combo text). 14 legitimate kept. Has 515 DRKG edges.
- Zinc oxide: 4 false GT entries (from hydroquinone cream). 2 legitimate kept. Has 87 DRKG edges.
- **NOTE (h696):** h678 initially reported "0 DRKG edges" for these — this was WRONG due to checking wrong DRKG path. Corrected in h696.

**Impact:** 45 false expanded GT entries removed (57,538→57,493). No holdout change.

**Additional findings:**
- 98 EC rows (1.0%) are from non-drug substances (croscarmellose=42, allergenic extracts, etc.)
- Combo products use separate names that don't cross-contaminate DRKG IDs
- Only varicella zoster vaccine overlaps between combo and individual drug DRKG IDs (legitimate)

### Tier Status (post h678 — unchanged from h685)
| Tier | Holdout | Std |
|------|---------|-----|
| GOLDEN | 71.8% | ± 7.1% |
| HIGH | 61.4% | ± 7.9% |
| MEDIUM | 38.4% | ± 4.3% |
| LOW | 14.4% | ± 1.2% |
| FILTER | 9.5% | ± 0.9% |

### New Hypotheses Generated (3)
- h694: Excipient Detection in EC Data (systematic non-drug filter)
- h695: Japanese PMDA Shared-Text GT Expansion (drug-class gaps)
- h696: Croscarmellose Signal (other excipients with DRKG edges)

### h696: Excipient DRKG Presence — VALIDATED (Data Cleanliness + DRKG Path Correction)
CRITICAL: h678 used wrong DRKG path (data/drkg/drkg.tsv → doesn't exist). Correct: data/raw/drkg/drkg.tsv.

15 excipients/preservatives have DRKG edges (up to 1665 each) and Node2Vec embeddings. Only 6 are in EC data. 11 are NOT in EC — their 48 expanded GT entries removed (inert). Dual-use substances handled by existing h677/h678 fixes. GT: 57,493 → 57,445.

### New Hypotheses Generated (5 total: 3 from h678, 2 from h696)
- h694: Excipient Detection in EC Data
- h695: Japanese PMDA Shared-Text GT Expansion
- h696: Croscarmellose Signal (completed)
- h697: DRKG Excipient Edge Source Analysis
- h698: Saccharin False GT Audit

### Recommended Next Steps
1. **h675**: FDA label contraindication mining (safety improvement)
2. **h672**: CS GT gap expansion for remaining PLAUSIBLE diseases
3. **h697**: DRKG excipient edge source analysis (understanding KG quality)

---

## Previous Session: h685-h691 - GT Quality Audit Series (2026-02-06)

### h679: Lidocaine GOLDEN Quality — INVALIDATED (already resolved by h677)
### h682: Mesothelioma GT Gap — INVALIDATED (adequate via disease hierarchy)
### h688: Two-Drug Disease GT Audit — VALIDATED (0% error rate, much better than single-drug)

### h691: Large-Gap Disease Analysis — VALIDATED (Structural Finding)
222 diseases (47%) have ≤2 internal GT drugs but 8.9x more in expanded GT. This is a structural DRKG limitation. Key patterns: missing standard treatments (OCD lacks SSRIs, PBC lacks UDCA), generic disease names, outdated drugs. Not actionable without model retraining (h687).

### h690: TB Drug Co-occurrence — VALIDATED (Minimal Impact)
Only 1 new false entry: pyrazinamide→immunodeficiency (HIV/TB co-treatment artifact). All other TB drug associations are legitimate. Isoniazid was unique in having 3 false associations (broader prophylactic use).

### h689: DRKG Diagnostic Agent Census — VALIDATED (Defensive)
Systematically searched 64 diagnostic/imaging agents in DrugBank. Found 3 additional agents in internal GT:
- Flortaucipir F-18 → Alzheimer's (tau PET tracer)
- Fluciclovine 18F → prostate cancer, glioma (amino acid PET)
- Pentagastrin → duodenal ulcer, pernicious anemia (stimulation test)

Expanded prediction filter from 6 to 12 agents. Correctly excluded 7 dual-use therapeutic agents. Expanded GT: 57,555 → 57,539 (-16 entries). No significant holdout change.

### h685: Disease GT Coverage Quality — VALIDATED (GT Honesty Improvement)
Audited all 169 diseases with only 1 drug in internal GT. Found 5.3% error rate (9/169 completely false). Also found 5 diagnostic imaging agents across multi-drug diseases (12 more false entries). Total: 19 false GT entries removed.

**Key findings:**
1. **Diagnostic agents dominate errors** — Tc-99m sestamibi, Ioflupane I-123, Florbetaben, Tc-99m sulfur colloid were all in DRKG from co-occurrence in clinical imaging contexts
2. **Iobenguane is DUAL-USE** — I-131 iobenguane (Azedra) is therapeutic for neuroblastoma/paraganglioma but diagnostic-only for Parkinson's/CHF. Handled selectively.
3. **Non-diagnostic false entries** — Diazoxide→carcinoma, Insulin→protein C deficiency, Isoniazid→3 diseases (TB co-occurrence), Chlorhexidine→2 coagulation disorders
4. **No significant holdout impact** — All tiers within normal variance (paired t-test p > 0.18)

**Changes:**
- Internal GT: 3,086 → 3,067 entries. 9 diseases dropped entirely.
- Expanded GT: 57,586 → 57,555 (-31 entries)
- Added 4 diagnostic agents to NON_THERAPEUTIC_COMPOUNDS prediction filter
- Added Iobenguane diagnostic-only disease handling
- Added 7 specific false GT entry removals

### Tier Status (post h685)
| Tier | Holdout | Std | Previous |
|------|---------|-----|----------|
| GOLDEN | 71.8% | ± 8.0% | 72.3% |
| HIGH | 61.4% | ± 8.9% | 62.8% |
| MEDIUM | 38.3% | ± 4.7% | 42.3% |
| LOW | 14.4% | ± 1.3% | 15.0% |
| FILTER | 9.5% | ± 1.0% | 9.9% |

Note: No changes are statistically significant. MEDIUM apparent drop (p=0.183) driven by high-variance seeds.

### New Hypotheses Generated (4)
- h688: Two-drug disease GT audit (62 diseases with 2 drugs)
- h689: DRKG diagnostic agent census (ATC V08/V09 systematic search)
- h690: TB drug co-occurrence pattern (rifampicin, pyrazinamide, ethambutol)
- h691: Large-gap diseases (1-2 internal GT but 20+ expanded GT)

### Recommended Next Steps
1. **h689**: DRKG diagnostic agent census — likely more imaging agents to find
2. **h691**: Large-gap diseases — understanding why internal GT is sparse
3. **h690**: TB drug co-occurrence — quick check, similar pattern to Isoniazid

---

## Previous Session: h674/h677 - Statin Safety + GT Quality Audit (2026-02-06)

### h674: Statin → Diabetes Inverse Indication Expansion — VALIDATED (No Impact)
All 7 major statins already covered in INVERSE_INDICATION_PAIRS for diabetes/hyperglycemia. Added cerivastatin and mevastatin defensively (no current predictions). All statin→diabetes predictions already at FILTER tier.

### h677: Every Cure GT Quality Audit — VALIDATED (Major Finding)
Random sample audit of 100 Every Cure indication rows. **6% strict error rate** (95% CI: 2.8%-12.5%).

**KEY FINDING: Drug mismatch from combo products**
- Lidocaine (78 rows) and bupivacaine (61 rows) have corticosteroid indication text wrongly assigned
- Source: NLP pipeline mapped combo product labels (lidocaine/hydrocortisone, bupivacaine/dexamethasone) to wrong component
- This propagated to expanded GT: 82 false lidocaine/bupivacaine entries removed, 3 B12 false entries removed
- GT: 59,626 → 59,541 (-85 entries)

**BUG FIX: Target overlap rescue leaking LA procedural demotions**
- ~57 bupivacaine/lidocaine predictions were being rescued from LOW→MEDIUM via target_overlap
- Same pattern as h560/h647 leakage. Fixed by blocking 'local_anesthetic_procedural' from rescue.

**Error taxonomy (100-sample audit):**
| Error Type | Count | Rate |
|------------|-------|------|
| Drug mismatch (combo product) | 4 | 4% |
| False indication (differential dx) | 2 | 2% |
| Risk reduction ≠ treatment | 1 | 1% |
| Diagnostic agent confusion | 2 | 2% |
| **Correct** | **91** | **91%** |

### Tier Status (post h674/h677)
| Tier | Previous (h671) | Current | Change |
|------|-----------------|---------|--------|
| GOLDEN | 71.6% ± 4.8% | 71.6% ± 4.8% | — |
| HIGH | 61.8% ± 7.5% | 61.8% ± 7.5% | — |
| MEDIUM | 43.5% ± 2.9% | 43.0% ± 3.4% | -0.5pp (honesty) |
| LOW | 15.3% ± 1.8% | 14.1% ± 1.7% | -1.2pp (honesty) |
| FILTER | 10.7% ± 1.2% | 10.7% ± 1.2% | — |

**Note:** MEDIUM/LOW decreases are HONESTY improvements. Previous numbers were inflated by false GT entries for lidocaine/bupivacaine. The remaining predictions are evaluated against a more accurate standard.

### New Hypotheses Generated (4)
- h678: Combo product drug mismatch audit beyond lidocaine/bupivacaine
- h679: Lidocaine GOLDEN prediction quality audit (psychiatric tier appropriateness)
- h680: DRKG internal GT false entry systematic scan
- h681: B12 supplement false GT audit (cyanocobalamin/hydroxocobalamin)

### Recommended Next Steps
1. **h680**: DRKG internal GT false entry scan — highest impact, affects kNN training
2. **h678**: Combo product drug mismatch audit — likely more errors beyond lidocaine/bupivacaine
3. **h679**: Lidocaine GOLDEN quality — quick check if psychiatric predictions are appropriate

---

## Previous Session: h673/h670/h671 - Safety Audit, False GT Cleanup, TransE Fix (2026-02-06)

### h673: CS Safety Audit — VALIDATED
4 implausible CS HIGH predictions assessed. 3 genuinely harmful, 1 legitimate adjunctive use:
- **Triamcinolone → TEN**: HARMFUL. No RCT evidence, 40% mortality on CS, infection risk. → FILTER
- **Budesonide → autoimmune PAP**: HARMFUL. 74% deteriorate on CS, macrophage suppression. → FILTER
- **Prednisolone → OSA**: HARMFUL. CS increase OSA risk (HR 1.40), weight gain. → FILTER
- **Methylprednisolone → dacryocystitis**: Legitimate adjunctive with antibiotics. No change.

**BUG FIX**: Duplicate dict keys in INVERSE_INDICATION_PAIRS silently lost IPF/glaucoma/osteoporosis filters for prednisolone/prednisone/methylprednisolone. 3 HIGH CS→IPF predictions were NOT being filtered despite PANTHER-IPF trial (increased mortality).

Total: 15 newly filtered + 3 bug-fix restored. HIGH +0.3pp (61.5→61.8%).

### h670: NLP Differential Diagnosis False GT Audit — VALIDATED
Systematic search for NLP extraction errors in Every Cure indicationList.xlsx. Found 18 new false GT entries from 2 patterns:
- **Pattern 1**: "secondary causes should be excluded" boilerplate in lipid drug labels (17 entries: 10 drugs→diabetes, 6→nephrotic syndrome, 1→hypothyroidism)
- **Pattern 2**: "must be excluded" differential diagnosis (nafarelin→CAH)

GT: 59,644 → 59,626. Minimal holdout impact but improves GT quality.

### h671: TransE Antimicrobial Mismatch Gate — VALIDATED
Fixed amphotericin B blanket 'antiparasitic' tag. AmB has narrow antiparasitic activity (Leishmania only), not against schistosomes/trypanosomes/toxoplasma.
- 3 implausible predictions HIGH→LOW (AmB→Chagas/schistosomiasis/toxoplasmosis)
- 2 genuine Leishmania predictions preserved at HIGH

### Tier Status (post h673/h670/h671)
| Tier | Holdout | Previous (h669) | Change |
|------|---------|-----------------|--------|
| GOLDEN | 71.6% ± 4.8% | 71.9% ± 4.7% | -0.3pp |
| HIGH | 61.8% ± 7.5% | 61.5% ± 7.2% | **+0.3pp** |
| MEDIUM | 43.5% ± 2.9% | 43.4% ± 2.9% | +0.1pp |
| LOW | 15.3% ± 1.8% | 15.3% ± 1.9% | — |
| FILTER | 10.7% ± 1.2% | 10.7% ± 1.2% | — |

### New Hypotheses Generated (4)
- h674: Statin→diabetes inverse indication expansion
- h675: Systematic FDA label contraindication mining
- h676: CS promotion rule disease-level exclusions
- h677: Every Cure GT quality: quantify total NLP error rate

### Recommended Next Steps
1. **h677**: Random sample audit of 100 Every Cure indication rows to estimate overall error rate
2. **h675**: Systematic contraindication mining from FDA labels
3. **h674**: Statin→diabetes explicit inverse indication (quick safety fix)

---

## Previous Session: h669 - CS HIGH Novel Prediction Quality Audit (2026-02-06)

### h669: CS HIGH Novel Prediction Quality Assessment — VALIDATED
144 CS truly novel HIGH predictions assessed against medical literature. **97.2% medically acceptable** (64.3% GENUINE first-line treatments, 32.9% PLAUSIBLE adjunctive uses, 2.8% IMPLAUSIBLE). CS novel quality dramatically outperforms non-CS (62.3% acceptable).

**Bugs found and fixed:**
1. **Diabetes insipidus comp_to_base bug**: "diabetes" substring matched "diabetes insipidus" in `_is_comp_to_base()`, promoting 9 completely wrong diabetes drug→DI predictions to HIGH. Fixed with COMP_TO_BASE_EXCLUSIONS.
2. **False GT from NLP extraction errors**: 6 lipid drugs (fenofibrate, gemfibrozil, lovastatin, cholestyramine, lomitapide, omega-3 FA) → hypothyroidism. FDA labels mention hypothyroidism as "secondary cause to exclude before starting therapy" — NLP incorrectly extracts as indication. Fixed with FALSE_GT_PAIRS exclusion.

**GT changes:** -6 false entries removed, +12 CS GT gaps added. Net +6. Expanded GT: 59,644.

### Tier Status (h669 update)
| Tier | Holdout | Previous | Change |
|------|---------|----------|--------|
| GOLDEN | 71.9% ± 4.7% | 71.6% | +0.3pp |
| HIGH | 61.5% ± 7.2% | 58.0% | **+3.5pp** |
| MEDIUM | 43.4% ± 2.9% | 43.3% | +0.1pp |
| LOW | 15.3% ± 1.9% | 15.3% | — |
| FILTER | 10.7% ± 1.2% | 10.7% | — |

### New Hypotheses Generated (4)
- h670: NLP differential diagnosis extraction error audit (systematic false GT search)
- h671: TransE antimicrobial mismatch gate for HIGH predictions
- h672: CS GT gap expansion for 15 PLAUSIBLE diseases
- h673: Safety audit of 4 implausible CS HIGH predictions (TEN, PAP, dacryocystitis, OSA)

### Recommended Next Steps
1. **h670**: Systematic NLP false GT audit — likely more false entries beyond hypothyroidism
2. **h673**: Safety audit — TEN CS predictions may be harmful
3. **h671**: TransE antimicrobial mismatch gate — quick fix for 5-7 implausible TransE-promoted predictions

---

## Previous Session: h658/h666/h636 - Literature Validation & GT Gap Expansion (2026-02-06)

### h658: Holdout-Invisible Prediction Validation via Literature Mining — VALIDATED
194 ATC coherent MEDIUM predictions assessed. 72.7% literature-validated precision (61.9% GT + 10.8% ESTABLISHED/CLINICAL). Exceeds GOLDEN holdout (71.6%). Confirms holdout blind spot is methodological, not quality issue. 13 WRONG_SPECTRUM predictions identified (echinocandin/cephalosporin spectrum mismatches).

### h666: GT Gap Expansion for Antibiotic Standard-of-Care Uses — VALIDATED
16 FDA-approved antibiotic/antifungal/dermatological pairs added to expanded GT. 0pp holdout impact (ATC coherent predictions are holdout-invisible). Value is in GT quality.

### h636: Bevacizumab Cross-Cancer Transferability Validation — VALIDATED
15 novel bevacizumab predictions: 66.7% literature-validated (10/15). KEY FINDING: lung cancer (FDA 2006) was NOT in GT! Extended to systematic cancer drug GT gap search:
- 16 GT gaps filled (4 anti-VEGF + 12 cancer drugs)
- **HIGH: 54.8% → 56.3% (+1.5pp)**
- **MEDIUM: 42.9% → 43.3% (+0.4pp)**
- **LOW: 14.8% → 15.2% (+0.4pp)**

Cancer drugs with GT gaps: cyclophosphamide→DLBCL, docetaxel→lung, cladribine→ALL/NHL/AML, bortezomib→ALL/NHL/DLBCL, gemcitabine→bladder, methotrexate→uterine, thiotepa→choriocarcinoma.

### h668: Systematic HIGH Novel GT Gap Search — VALIDATED
Extended to ALL HIGH novel predictions. 22 more pairs added:
- DOACs → atrial flutter (4 FDA-approved drugs, all gaps!)
- Erythromycin → pneumococcal/rosacea/osteomyelitis
- Levofloxacin → sinusitis/osteomyelitis/empyema/bronchiectasis
- Liothyronine → congenital hypothyroidism, Amphotericin B → mycetoma
- Prednisone + Montelukast → IPF

### Tier Status (final, h668 update)
| Tier | Holdout | Previous | Change |
|------|---------|----------|--------|
| GOLDEN | 71.6% ± 4.3% | 71.6% | — |
| HIGH | 58.0% ± 7.7% | 54.8% | **+3.2pp** |
| MEDIUM | 43.3% ± 2.9% | 42.9% | **+0.4pp** |
| LOW | 15.3% ± 1.8% | 14.8% | +0.5pp |
| FILTER | 10.7% ± 1.2% | 10.6% | +0.1pp |

Total GT additions this session: 54 pairs. Expanded GT: 59,638 (was 59,584).

### New Hypotheses Generated (5)
- h665: Antimicrobial spectrum-level demotion (echinocandin/cephalosporin mismatches)
- h666: GT gap expansion for antibiotics (COMPLETED)
- h667: Literature validation for other holdout-invisible sub-reasons
- h668: Systematic HIGH GT gap search (COMPLETED)
- h669: Corticosteroid novel HIGH prediction quality assessment

### Recommended Next Steps
1. **h669**: CS novel HIGH prediction quality assessment (144 predictions, 49% of HIGH novel)
2. Continue GT gap search: 200+ HIGH novel predictions remain
3. **h665**: Antimicrobial spectrum-level demotion (small impact but clean)
4. Regenerate deliverable with updated GT

---

## Previous Session: h657/h654/h653/h661 - MEDIUM Calibration & Ryland Prep (2026-02-06)

### h657: Default MEDIUM NoMech R6-10 Demotion — INVALIDATED
With expanded GT, NoMech R6-10 has 40.5% ± 9.4% holdout (n=14.6/seed). This is MEDIUM-quality (z=-0.4 vs MEDIUM avg 42.9%), not LOW-quality. The original 30.0% estimate from h555-era used internal GT, which systematically underestimates signal-rich predictions (h629). Demoting would game MEDIUM headline (+1.6pp to 44.5%) but misclassify genuinely MEDIUM-quality predictions as LOW. Code reverted.

**Key lesson:** Always re-evaluate tier decisions when GT changes. Expanded GT lifts signal-rich predictions more than weak ones, so internal-GT-based demotions may be wrong.

### h654: Train Frequency Threshold Sensitivity — INVALIDATED
22% of freq>=3 drugs (74.6/343) drop below threshold during holdout. This makes holdout a conservative lower-bound estimate (desirable). Production unaffected. The freq>=3 threshold is correctly calibrated.

### h653: ATC Coherent Remaining Categories Quality Map — VALIDATED
190 ATC coherent predictions at 61.6% full-data precision. Remaining categories: infectious (85%), ophthalmic (7%), dermatological (6%), autoimmune (2%). All have 59-86% full-data precision vs excluded categories at 0-19%. KEY FINDING: ATC coherent predictions are holdout-invisible (freq drops below 3 during holdout). No further exclusions needed.

### h661: Ryland Collaboration Prep — VALIDATED
Created comprehensive prep for Monday Feb 10 meeting. Key findings:
- 30 dermatological diseases, 230 GOLDEN/HIGH/MEDIUM predictions (mostly corticosteroids)
- Tetracycline→ichthyosis predictions are kNN artifacts (no literature support)
- EGFR drugs NOT predicted for skin diseases (DRKG cancer-only gap)
- Top novel wet-lab candidate: Montelukast→IPF (HIGH, Mech+TransE)
- 58 novel non-CS predictions for rare/genetic diseases identified
- Document: `data/analysis/h661_ryland_collaboration_prep.md`

### Tier Status (unchanged from h649)
| Tier | Holdout | Predictions |
|------|---------|-------------|
| GOLDEN | 71.6% ± 4.3% | 420 |
| HIGH | 54.8% ± 8.9% | ~858 |
| MEDIUM | 42.9% ± 2.9% | ~1363 |
| LOW | 14.8% ± 1.7% | ~4235 |
| FILTER | 10.6% ± 1.3% | 7274 |

### New Hypotheses Generated (4)
- h658: Holdout-invisible prediction validation via literature mining
- h659: Expanded GT impact on internal GT precision estimates
- h660: MEDIUM Default NoMech R1-5 quality characterization
- h661: Ryland collaboration prep (completed)

### h659: Expanded GT Impact on Internal GT Precision — VALIDATED
Expanded GT lifts all sub-reasons but disproportionately: infectious_cs_tb +55.6pp, transe_promotion +32.3pp, cv_rescue +29.4pp. 23 INVALIDATED hypotheses checked — none need re-evaluation.

### h660: MEDIUM Default NoMech R1-5 Quality — INCONCLUSIVE
Couldn't isolate from holdout. Estimated ~49-54% (correctly MEDIUM).

### h662: Per-Rule Holdout Tracking for Default MEDIUM — VALIDATED
Named reasons added. Results: freq5_mechanism 45.3% (n=14/seed), NoMech R1-5 **45.4%** (n=50/seed, above avg), NoMech R6-10 34.1% (n=30/seed, below avg but GENUINE).

### h664: Deliverable Regeneration — VALIDATED
14,150 predictions. MEDIUM: 1336 (h657 revert reflected).

### Recommended Next Steps
1. **h658**: Literature validation of holdout-invisible ATC coherent predictions
2. Ryland meeting prep review (`data/analysis/h661_ryland_collaboration_prep.md`)
3. Higher-effort external data integrations (LINCS, PubMed mining)

---

## Previous Session: h651/h650/h655 - ATC Coherent Exclusions & Cancer Rank Analysis (2026-02-06)

### h651: ATC Coherent Endocrine/Musculoskeletal/Respiratory/Renal Exclusion — VALIDATED
Added 4 new categories to ATC_COHERENT_EXCLUDED: endocrine (0% holdout), musculoskeletal (0%), respiratory (19.4%), renal (11.1%). All had n<5/seed, consistently below MEDIUM avg (42.9%). 27 predictions demoted from MEDIUM to LOW in deliverable.

**Key finding:** Holdout evaluator shows 0pp change because GT recomputation reduces train_frequency below the freq>=3 threshold for borderline drugs (Terbutaline 2→1, Formoterol 3→2, Calcitriol 3→2). The change only affects full-data precision and deliverable quality.

**Lesson:** ATC coherent exclusions have minimal holdout impact due to the freq>=3 natural filter during holdout evaluation.

### h650: Cancer Same-Type Rank 16-20 Demotion — INVALIDATED
Rank 16-20 cancer_same_type has 31.9% ± 4.9% holdout (n=10.2/seed). While below MEDIUM avg (42.9%), many predictions are medically legitimate (FDA-approved drugs like Apalutamide→prostate, Nelarabine→ALL). At 2x LOW average (14.8%), these are borderline MEDIUM, not genuinely LOW. +0.58pp improvement not worth misclassifying legitimate predictions.

### h655: Cancer Same-Type Rank 11-15 → HIGH Promotion — INVALIDATED
Initial mean of 55.8% was inflated by seed 42 (n=1, 100%). Excluding this outlier: 44.7% ± 8.6%, which is at MEDIUM average — NOT promotable to HIGH.

**Cancer rank gradient (final):** R1-10=HIGH (62.4%), R11-15=MEDIUM (44.7%), R16-20=MEDIUM-low (31.9%), R21+=LOW (26.9%)

### Comprehensive MEDIUM Quality Map
Ran detailed sub-reason × mechanism × rank analysis. Key finding: **Default MEDIUM NoMech R6-10 at 30.0% ± 9.3% (n=25.8/seed)** is the single largest low-quality bucket in MEDIUM. New hypothesis h657 created to investigate demotion.

### New Hypotheses Generated (5)
- h653: ATC coherent remaining categories quality map
- h654: Train frequency threshold sensitivity (freq>=3 vs freq>=2)
- h655: Cancer rank 11-15 → HIGH (INVALIDATED)
- h656: Deliverable-only quality metrics
- h657: Default MEDIUM NoMech R6-10 demotion (priority 3, medium impact)

### Recommended Next Steps
1. **h657**: Default MEDIUM NoMech R6-10 demotion — largest remaining MEDIUM quality lever
2. **h653**: ATC coherent remaining categories quality map
3. Higher-effort external data integrations (LINCS, PubMed mining)

---

## Previous Session: h638/h644/h647/h648 - Target Overlap Analysis & MEDIUM Optimization (2026-02-06)

### h638: MEDIUM Target Overlap → HIGH Promotion — INVALIDATED
No subset of target_overlap_promotion MEDIUM exceeds 55% holdout with n>=10/seed. ALL preds are Mech=Y, TransE=N, non-CS. Overall: 49.2% ± 5.9% (n=35/seed). Psychiatric: 53.3% ± 8.8% (n=18.2, near-promotable but 1.7pp below 55%). Rank 1-5: 58.7% (n=6.2, too few). Target_overlap correctly placed as high-quality MEDIUM.

### h644: ATC Coherent Infectious Quality — VALIDATED (informative)
ATC coherent infectious NoMech = 41.0% ± 7.9% holdout (n=34/seed). Driven by ANTIFUNGAL drugs (64.9% ± 20.4%) vs non-antifungal/antibacterial (33.3% ± 12.8%). Not actionable for tier changes (z=-0.76 vs MEDIUM avg), but informative: antifungal predictions are HIGH-quality, antibacterial (especially cefuroxime 6%) are lower.

### h647: Metabolic Target Overlap Rescue Leak — VALIDATED
FOUND BUG: 37 metabolic_medium_demotion predictions rescued back to MEDIUM via target_overlap. Metabolic was missing from the rescue blocklist (line 4115). Fixed. These had 10.3% ± 4.7% holdout (clearly LOW).

### h648: Cancer Same-Type Rank 21+ Demotion — VALIDATED
cancer_same_type rank gradient: R11-15 (55.8%), R16-20 (31.9%), R21-30 (25.5%). Demoted rank 21+ to LOW. 100 predictions affected. Remaining cancer_same_type MEDIUM (R11-20) improved from 31.2% to 40.3%.

**Tier Impact (cumulative h647+h648):**
| Tier | Before | After | Delta |
|------|--------|-------|-------|
| GOLDEN | 71.6% ± 4.2% | 71.6% ± 4.3% | 0 |
| HIGH | 54.6% ± 9.3% | 54.8% ± 8.9% | +0.2pp |
| MEDIUM | 40.8% ± 2.0% | **42.8% ± 3.1%** | **+2.0pp** |
| LOW | 14.5% ± 2.0% | 14.8% ± 1.7% | +0.3pp |
| FILTER | 10.6% ± 1.3% | 10.6% ± 1.3% | 0 |

### New Hypotheses Generated (4)
- h649: Infectious hierarchy pneumonia demotion (16.7% holdout)
- h650: Cancer same-type R16-20 further demotion analysis
- h651: ATC coherent endocrine/respiratory demotion (0%/23% holdout)
- h652: Systematic demoted category rescue leak audit

### Key Insights
1. **Target_overlap rescue leaks demoted predictions** — same pattern as h560. After adding categories to MEDIUM_DEMOTED_CATEGORIES, MUST also update target_overlap blocklist.
2. **Cancer rank gradient is steep** — within same sub-reason, rank 11-15 can be HIGH quality while rank 21+ is LOW quality. Always check rank gradients.
3. **Sub-reason map is the most productive analysis** — the comprehensive MEDIUM sub-reason holdout map (run in h647 investigation) directly revealed both h647 and h648 opportunities.

### Recommended Next Steps
1. **h652**: Audit ALL rescue pathways for demoted category leaks (ATC coherent, highly_repurposable)
2. **h649**: Infectious hierarchy pneumonia demotion (16.7% holdout, n=6/seed)
3. **h650**: Cancer same-type R16-20 analysis (31.9%, n=10.2/seed)

---

## Previous Session: h637/h642/h643 - Closed Direction Re-evaluation & MEDIUM Optimization (2026-02-06)

### h637: Systematic CLOSED Direction Re-evaluation — VALIDATED
Reviewed all 16 CLOSED directions for GT-dependency. 12/16 not GT-dependent. 4 candidates re-evaluated: all remain closed. h633 was a special case (wrong GT + abundant expanded GT + strong signal). No other CLOSED direction meets all conditions for reopening.

### h642: MEDIUM Default Sub-Stratification — INCONCLUSIVE
ALL remaining non-CS MEDIUM predictions are rank 16-20. Mech+Rank<=10 bucket is EMPTY (fully captured by h630/hierarchy/cancer rules). Found cv_established_drug_rescue NoMech = 22.5% (borderline LOW), leading to h643.

### h643: CV Rescue Mechanism-Gating — VALIDATED
Require mechanism support for cv_established_drug_rescue. NoMech CV drugs (22.5%, DOACs/PCSK9i) → LOW.

**Tier Impact:**
| Tier | Before | After h643 | Delta |
|------|--------|------------|-------|
| GOLDEN | 71.6% ± 4.3% | 71.6% ± 4.2% | 0 |
| HIGH | 54.7% ± 9.3% | 54.6% ± 9.3% | -0.1pp |
| MEDIUM | 38.1% ± 2.5% | **40.8% ± 2.0%** | **+2.7pp** |
| LOW | 14.5% ± 2.0% | 14.5% ± 2.0% | 0 |

### Additional hypotheses tested this session:
- h645 INVALIDATED: Other rescue rules don't need mechanism gates
- h635 INCONCLUSIVE: Cytotoxic drug class too small n per class for tier rules
- h639 INVALIDATED: Multi-system drug rescue negligible impact (n=5/seed)
- h640 INVALIDATED: Lidocaine MEDIUM n=4.6/seed too small for promotion
- h641, h646 INVALIDATED: Superseded by h643 / flawed rank analysis

**Deliverable regenerated** with h643 changes: MEDIUM 1532 preds, LOW 4055 preds.

### Key Insight: h642 rank analysis bug
`knn_rank` attribute doesn't exist on DrugPrediction. The h642 finding "ALL MEDIUM are rank 16-20" was an artifact of defaulting to 99. Always verify attribute names before analysis.

### Recommended Next Steps
1. **h638**: MEDIUM target_overlap → HIGH for psychiatric subset (53.3%, n=18/seed)
2. **h644**: ATC coherent infectious quality investigation (42.4% NoMech, interesting)
3. Consider higher-effort external data integrations (LINCS, PubMed mining)

---

## Previous Session: h633 - Cancer Same-Type Expanded GT Re-evaluation (2026-02-06)

### h633: Cancer Same-Type + Mechanism + Rank≤10 → HIGH Promotion — VALIDATED

Reopened CLOSED direction #4. Original closure (h416/h447) used internal GT showing 10.7% holdout. h611/h629 showed expanded GT (59,584 pairs vs 3,070) dramatically changes the calculus.

**Key Results (5-seed holdout, expanded GT):**
| Signal | Holdout | ±std | N/seed |
|---|---|---|---|
| Mech+R<=5 | 64.2% | 12.6% | 17.6 |
| rank_1_5 | 58.7% | 12.1% | 20.8 |
| Mech+R<=10 | 56.6% | 9.7% | 30.0 |
| ALL cancer_same_type | 37.4% | 6.4% | 100.8 |
| No mechanism | 18.3% | 3.9% | 28.4 |

**CS artifact check:** Only 0.4% of cancer_same_type predictions are corticosteroids. GENUINE signal.

**Drug class breakdown:**
| Drug Class | Holdout | N/seed |
|---|---|---|
| Taxane | 76.0% | 7.8 |
| Anthracycline | 71.6% | 6.5 |
| Platinum | 50.5% | 7.4 |
| Antimetabolite | 48.3% | 22.6 |
| Alkylating | 24.2% | 8.8 |

**Implementation:** cancer_same_type + mechanism + rank≤10 → HIGH (cancer_same_type_mech_rank10)

**H393 evaluator results:**
- cancer_same_type_mech_rank10: Full=82.4%, Holdout=62.4% ± 10.7% (n=27.6/seed)

**Tier impact:**
| Tier | Before | After | Delta |
|------|--------|-------|-------|
| GOLDEN | 71.6% ± 4.3% (420) | 71.6% ± 4.3% (419) | unchanged |
| HIGH | 53.1% ± 12.2% (719) | 54.5% ± 9.0% (876) | +1.4pp, -3.2% var |
| MEDIUM | 38.7% ± 3.3% (2019) | 36.8% ± 2.5% (1972) | -1.9pp, -0.8% var |
| LOW | 14.2% ± 2.0% (3718) | 14.2% ± 2.0% (3622) | unchanged |
| FILTER | 10.6% ± 1.3% (7274) | 10.6% ± 1.3% (7274) | unchanged |

181 predictions promoted. Top drugs: Doxorubicin (26), Paclitaxel (17), Bevacizumab (14), Cyclophosphamide (11).

### New Hypotheses Generated (4)
- h634: Remaining cancer_same_type MEDIUM demotion (no-mech = 18.3%)
- h635: Cytotoxic drug class as quality signal (taxane 76%, anthracycline 72%)
- h636: Bevacizumab cross-cancer validation (14 promoted preds)
- h637: Systematic CLOSED direction re-evaluation with expanded GT

### Key Insights
1. **CLOSED directions MUST be re-evaluated when GT changes fundamentally** — internal GT (3,070 pairs) to expanded GT (59,584 pairs) is a 19x increase that changes the precision landscape.
2. Cancer same-type is NOT CS-driven (0.4%) — unlike most HIGH-tier improvements.
3. Mechanism + rank within a single tier rule can create a HIGH-quality subset, even when the overall rule is MEDIUM.
4. Drug class stratification reveals: broad-spectrum cytotoxics (taxane, anthracycline, platinum) transfer across cancer types much better than alkylating agents or vinca alkaloids.

### Recommended Next Steps
### h634: Cancer Same-Type No-Mechanism Demotion — VALIDATED

**Key Results:**
- cancer_same_type_no_mechanism: 23.6% ± 7.7% holdout (n=23/seed) → demoted to LOW
- 166 predictions demoted

**Tier impact (cumulative h633+h634):**
| Tier | Before | After h633+h634 | Net Delta |
|------|--------|-----------------|-----------|
| GOLDEN | 71.6% ± 4.3% | 71.6% ± 4.3% | 0 |
| HIGH | 53.1% ± 12.2% | 54.7% ± 9.3% | +1.6pp |
| MEDIUM | 38.7% ± 3.3% | 38.1% ± 2.5% | -0.6pp |
| LOW | 14.2% ± 2.0% | 14.5% ± 1.9% | +0.3pp |

### Recommended Next Steps
1. **h637**: Systematically check all 16 CLOSED directions for GT-dependency
2. Regenerate deliverable with h633+h634 updates
3. **h635**: Investigate cytotoxic drug class as quality signal

---

## Previous Session: h629/h631 - MEDIUM Quality Stratification (2026-02-06)

### h629: MEDIUM Precision Stratification by Multiple Signals — VALIDATED

Expanded GT resolves original TransE MEDIUM blocker (h405: 34.7% < HIGH 50.8%). With expanded GT, TransE within MEDIUM reaches HIGH-level precision.

**Key Results (5-seed holdout, expanded GT):**
| Signal Combination | Holdout | ±std | N/seed |
|---|---|---|---|
| TransE+Mechanism+Rank≤10 | 71.9% | 15.7% | 7 |
| cancer_same_type+Rank≤5 | 66.0% | 14.1% | 22 |
| TransE+Rank≤5 | 64.9% | 12.4% | 11 |
| TransE+Rank≤10 | 63.2% | 7.1% | 19 |
| TransE+Mechanism | 59.4% | 13.9% | 14 |
| TransE alone | 56.5% | 8.8% | 28 |
| Mechanism+Rank≤5 | 53.9% | 6.2% | 39 |
| Mechanism+Rank≤10 | 52.5% | 4.4% | 76 |
| All MEDIUM | 38.8% | 3.7% | 328 |

**CS artifact check:** TransE non-CS: 49.1% (GENUINE). Not driven by corticosteroids.

**Differential:** +19.3pp over non-TransE MEDIUM (constant regardless of GT used).

**Tier impact assessment (TransE MEDIUM non-CS → HIGH):**
- HIGH: 49.1% → 49.5%, variance 7.9% → 5.7% (IMPROVES), +34 preds/seed
- MEDIUM: 39.9% → 38.9% (-0.9pp)
- Decision: NOT promoted (borderline, existing CLOSED direction). Implemented as annotation instead.

### h631: MEDIUM Quality Quartile Annotation — VALIDATED

Added `medium_quality` column to deliverable based on h629 signal combinations:
- Q1 (TransE + mechanism/rank≤5): 138 preds, 60-72% expected holdout
- Q2 (TransE OR mechanism+rank≤10): 459 preds, 50-57% expected holdout
- Q3 (mechanism OR rank≤5): 931 preds, 44-54% expected holdout
- Q4 (none): 606 preds, ~31% expected holdout

Q1-Q4 spans a 41pp range — more informative than single MEDIUM label for Ryland/collaborators.

### h630: TransE MEDIUM → HIGH Promotion — VALIDATED

Implemented TransE + (mechanism OR rank≤5) non-CS MEDIUM → HIGH promotion.

**H393 evaluator results:**
- transe_medium_promotion: Full=68.8%, Holdout=56.1% ± 11.9% (n=15/seed)

**Tier impact:**
| Tier | Before | After | Delta |
|------|--------|-------|-------|
| GOLDEN | 71.6% ± 4.3% (420) | 71.6% ± 4.3% (420) | unchanged |
| HIGH | 52.8% ± 13.5% (604) | 53.1% ± 12.2% (719) | +0.3pp, -1.3% var |
| MEDIUM | 39.5% ± 3.5% (2134) | 38.7% ± 3.3% (2019) | -0.8pp, -0.2% var |
| LOW | 14.2% ± 2.0% (3718) | 14.2% ± 2.0% (3718) | unchanged |
| FILTER | 10.6% ± 1.3% (7274) | 10.6% ± 1.3% (7274) | unchanged |

115 preds promoted. Top drugs: Doxorubicin (23), Amphotericin B (20), Bleomycin (12).

### Session Tier Performance (post-h630)
| Tier | Holdout | Predictions |
|------|---------|-------------|
| GOLDEN | 71.6% ± 4.3% | 420 |
| HIGH | 53.1% ± 12.2% | 719 |
| MEDIUM | 38.7% ± 3.3% | 2019 |
| LOW | 14.2% ± 2.0% | 3718 |
| FILTER | 10.6% ± 1.3% | 7274 |

### New Hypotheses Generated (4)
- h630: TransE MEDIUM → HIGH promotion (VALIDATED)
- h631: MEDIUM quality quartile annotation (VALIDATED)
- h632: Mechanism + Rank ≤ 10 as independent HIGH signal
- (More pending from h629 analysis)

### Key Insights
1. Expanded GT resolves TransE MEDIUM blocker — 56.5% vs 34.7% (internal GT)
2. The 19.3pp TransE differential is GT-independent (constant lift)
3. TransE promotion: HIGH precision INCREASES while variance DECREASES — counter-intuitive but correct
4. Signal combination reveals 41pp quality spread within MEDIUM
5. CLOSED directions should be re-evaluated when evaluation methodology changes (GT expansion)

### Recommended Next Steps
1. **h632**: Validate mechanism+rank≤10 as independent promotion signal
2. External data integration for fundamentally new signals
3. Meeting prep for Ryland (Monday Feb 10)

---

## Previous Session: h618/h622/h614/h617/h624 - CV Rescue + Tier Calibration (2026-02-06)

### h618: CV Medium Demotion Reversal — VALIDATED

h462 demoted ALL cardiovascular MEDIUM→LOW based on internal GT (2.0% holdout). h615 found 25.1% ± 19.4% with expanded GT. This experiment stratified by drug class:

**Key Results (5-seed holdout, expanded GT):**
| Drug Class | Holdout | N/seed | Preds | Action |
|------------|---------|--------|-------|--------|
| CCB | 49.7% ± 34.6% | 3.4 | 11 | Rescued to MEDIUM |
| Diuretic | 33.8% ± 32.4% | 3.2 | 12 | Rescued to MEDIUM |
| Anticoagulant/antiplatelet | 32.6% ± 23.4% | 14.2 | 70 | Rescued to MEDIUM |
| ARB | 30.0% ± 40.0% | 3.0 | 10 | Rescued to MEDIUM |
| other_CV (antibiotics/biologics) | 18.3% ± 9.0% | 37.6 | 166 | Stay LOW |
| Corticosteroid | 2.9% ± 5.7% | 1.4 | 20 | Stay LOW |

**Implementation:** `_is_established_cv_drug()` method identifies genuine CV pharmacotherapy (anticoagulants, CCBs, diuretics, ARBs, statins, beta-blockers, ACE inhibitors, antiarrhythmics, nitrates, etc.). 201 predictions rescued LOW→MEDIUM.

**Holdout validation:** cv_established_drug_rescue = 30.9% ± 20.9% (n=44/seed, Δ=-1.6pp, GENUINE). Remaining cardiovascular_medium_demotion = 4.6% ± 3.8% (correctly LOW).

**Tier impact:**
| Tier | Before | After | Delta |
|------|--------|-------|-------|
| GOLDEN | 71.6% ± 4.3% (420) | 71.6% ± 4.3% (420) | unchanged |
| HIGH | 52.8% ± 13.5% (604) | 52.8% ± 13.5% (604) | unchanged |
| MEDIUM | 41.3% ± 2.8% (1874) | 38.9% ± 4.0% (2075) | -2.4pp, +201 preds |
| LOW | 15.1% ± 2.4% (3978) | 14.6% ± 2.4% (3777) | -0.5pp, -201 preds |
| FILTER | 10.6% ± 1.3% (7274) | 10.6% ± 1.3% (7274) | unchanged |

**Key insight:** Internal GT systematically underestimates CV drug precision. Expanded GT reveals drug-class stratification cleanly separates genuine CV drugs (30.9%) from non-CV drugs predicted for CV diseases (4.6%).

### h622: Expanded GT Recalibration of Other Demoted Categories — INVALIDATED
No other demoted category has a drug-class subset with both >=25% holdout AND n>=5/seed. Best candidate: heme antineoplastic_heme 31.9% but n=4.8/seed (marginal). The CV case was special due to anticoagulant dominance (n=14.2/seed).

### h614: MEDIUM Sub-Pathway Quality Map v2 — VALIDATED
No demotable MEDIUM sub-pathways with expanded GT. All sub-pathways with n>=3/seed above 25% holdout. MEDIUM overall: 51.8% ± 5.5%. Metabolic target_overlap leak (45 preds) is correct behavior (48.9% holdout).

### h617: HIGH Tier Stabilization — INCONCLUSIVE
HIGH variance (±13.5%) is structural — driven by disease-split randomness. Seed 42 has fewer hierarchy-matching diseases in holdout. comp_to_base_high_87 = 0% holdout (n=5/seed, too small). Cannot be fixed without stratified splitting.

### h624: Deliverable Regeneration — VALIDATED
Deliverable regenerated. GOLDEN 420, HIGH 604, MEDIUM 2075, LOW 3777, FILTER 7274.

### Session Tier Performance (post-h618)
| Tier | Holdout | Predictions |
|------|---------|-------------|
| GOLDEN | 71.6% ± 4.3% | 420 |
| HIGH | 52.8% ± 13.5% | 604 |
| MEDIUM | 38.9% ± 4.0% | 2075 |
| LOW | 14.6% ± 2.4% | 3777 |
| FILTER | 10.6% ± 1.3% | 7274 |

### New Hypotheses Generated (3)
- h622: Expanded GT recalibration of other demoted categories (INVALIDATED)
- h623: MEDIUM precision recovery: tighten CV rescue criteria
- h624: Deliverable regeneration (VALIDATED)

### Key Insights
1. Internal GT systematically underestimates CV drug precision; expanded GT reveals 30.9% holdout
2. Drug-class stratification can find quality subsets within category demotions
3. Category demotions well-calibrated for all non-CV categories
4. MEDIUM tier is fully optimized at sub-pathway level
5. HIGH variance is structural, irreducible with current methodology

### Recommended Next Steps
1. **h623**: Tighten CV rescue criteria to recover MEDIUM precision
2. **h534/h578**: TransE annotation for FILTER/LOW tiers (deliverable quality)
3. External data integration (h91/h92) for fundamentally new signals

---

## Previous Session: h615/h619/h620/h621/h616 - Expanded GT Analysis (2026-02-06)

### h616: Disease-Specific GT Completeness Score — VALIDATED
Added `gt_completeness_ratio` column to deliverable. 479 diseases scored. Median 6.0x, mean 11.5x. Weakly negative correlation with holdout precision (r=-0.198) — not predictive of quality, but informative annotation.

### h621: Disease Categorization Fix — VALIDATED
Fixed pleural mesothelioma (respiratory→cancer) and retinoblastoma (ophthalmic→cancer). ~28 predictions rescued from FILTER. Added `mesothelioma` to cancer keywords, moved `retinoblastoma` from ophthalmic to cancer.

### h620: Expanded GT Safety Filter Audit — VALIDATED
GT contamination is real but minimal: 37 clearly wrong entries out of 1131 FILTER GT hits (3.3%). FDG PET diagnostic associations exist in both internal and expanded GT. Inverse indication hits are mostly genuine dual-use drugs (18/22 from internal GT). Does NOT affect tier precision.

### h619: Deliverable Regeneration — VALIDATED
Deliverable regenerated with h615+h621 changes. GOLDEN: 420, HIGH: 604, MEDIUM: 1874, LOW: 3978, FILTER: 7274. 14,150 total predictions.

### h615: Expanded GT-Based Tier Recalibration — VALIDATED

Compared per-rule precision using internal GT (3,070 pairs) vs expanded GT (59,584 pairs, 19x more). Found 26 tier boundary crossings. 5-seed holdout validated 4 HIGH→GOLDEN hierarchy group promotions.

**Key Results:**
- Internal GT systematically underestimates hierarchy group precision by 15-30pp
- 4 groups have GOLDEN-level holdout precision but were assigned HIGH based on internal GT analysis:
  - autoimmune_hierarchy_rheumatoid_arthritis: 86.4% ± 8.7% holdout (n=23/seed)
  - autoimmune_hierarchy_colitis: 85.7% ± 0.0% holdout (n=7/seed)
  - cardiovascular_hierarchy_arrhythmia: 72.9% ± 1.5% holdout (n=11/seed)
  - cardiovascular_hierarchy_coronary: 65.5% ± 1.2% holdout (n=13/seed)

**Implementation:** Added `HIERARCHY_PROMOTE_TO_GOLDEN` set in `_assign_confidence_tier()`. 139 predictions promoted.

**Impact:**
| Tier | Before | After | Delta |
|------|--------|-------|-------|
| GOLDEN | 69.9% ± 17.9% (280 preds) | 71.6% ± 4.3% (419 preds) | +1.7pp, +139 preds, std -13.6pp |
| HIGH | 58.8% ± 6.1% (736 preds) | 52.8% ± 13.5% (597 preds) | -6.0pp, -139 preds |
| MEDIUM | 41.3% ± 2.8% | 41.3% ± 2.8% | unchanged |
| LOW | 15.1% ± 2.4% | 15.1% ± 2.4% | unchanged |
| FILTER | 10.6% ± 1.3% | 10.6% ± 1.3% | unchanged |

Tier ordering preserved. HIGH drop due to removing best predictions. Seed 42 outlier (HIGH=30%, n=40) drives HIGH variance.

**Other findings NOT acted on:**
- cardiovascular_medium_demotion: 25.1% ± 19.4% holdout (above MEDIUM boundary but too variable)
- FILTER rules (non_therapeutic_compound, inverse_indication): expanded GT shows higher precision but safety filters should remain regardless
- Many FILTER rules show elevated expanded GT precision — suggests expanded GT may include non-therapeutic associations

**New Hypotheses (4):** h617-h620 (HIGH stabilization, CV medium demotion stratification, deliverable regeneration, expanded GT safety audit)

### Session Tier Performance (h621 update)
| Tier | Holdout | Predictions |
|------|---------|-------------|
| GOLDEN | 71.6% ± 4.3% | 420 |
| HIGH | 52.8% ± 13.5% | 604 |
| MEDIUM | 41.3% ± 2.8% | 1874 |
| LOW | 15.1% ± 2.4% | 3978 |
| FILTER | 10.6% ± 1.3% | 7274 |

### New Hypotheses Generated (5 total)
- h617: HIGH tier stabilization after h615 promotions
- h618: CV medium demotion drug-class stratification
- h619: Deliverable regeneration (COMPLETED)
- h620: Expanded GT safety filter audit (COMPLETED)
- h621: Disease categorization fix (COMPLETED)

### Recommended Next Steps
1. **h618**: CV medium demotion drug-class stratification (25.1% ± 19.4% holdout)
2. **h617**: Investigate HIGH tier seed-42 outlier
3. External data integration (h91/h92) for fundamentally new signals
4. Literature mining (h91) for novel hypotheses

---

## Previous Session: h606/h611/h612/h613/h605 - ATC Coherent + GT Methodology (2026-02-06)

### h606: ATC Coherent Respiratory/Endocrine Validation — VALIDATED
Comprehensive ATC coherent category analysis. Found 292 ATC coherent MEDIUM predictions across 9 categories. Psychiatric ATC coherent holdout = 17.2% ± 5.8% (p=0.0006 below MEDIUM avg). Added psychiatric to ATC_COHERENT_EXCLUDED. 47 predictions MEDIUM→LOW. Tier-level impact unmeasurable.

### h612: Deliverable Regeneration — VALIDATED
Regenerated deliverable with all h598-h606 changes. 14,150 predictions, 473 diseases, 1,004 drugs. MEDIUM: 1,876 preds. All changes confirmed.

### h611: MEDIUM Sub-Pathway Quality Map — INVALIDATED (GT methodology bug)
**CRITICAL FINDING:** Initial analysis using predictor.ground_truth (3,070 pairs) showed 7 below-LOW sub-pathways. But this was WRONG — expanded_ground_truth.json (59,584 pairs, 19x more) should be used. With correct GT, ALL sub-pathways above LOW threshold. cancer_same_type: 11.8% → 37.7%, target_overlap: 11.5% → 31.7%. Code changes reverted.

**KEY LESSON:** ALWAYS use expanded_ground_truth.json for holdout evaluation. predictor.ground_truth only has DRKG-derived pairs.

### h613: Expanded GT Gap Analysis — VALIDATED
Mapped the internal-vs-expanded GT gap. Cancer has highest ratio (13.2x), endocrine lowest (3.0x). Per-tier gains from expanded GT: GOLDEN +12.9pp, HIGH +16.8pp, MEDIUM +15.4pp, LOW +6.3pp. No diseases have zero expanded GT.

### h605: Highly Repurposable MEDIUM Demotion — INVALIDATED
Only 4 predictions (all chronic pain). 0% internal GT but 25% expanded GT. Holdout with expanded GT = 29.2%. Not demotable.

### Session Tier Performance (unchanged from h603)
| Tier | Holdout | Predictions |
|------|---------|-------------|
| GOLDEN | 69.9% ± 17.9% | ~280 |
| HIGH | 58.9% ± 6.0% | ~732 |
| MEDIUM | 41.3% ± 2.8% | ~1876 |
| LOW | 15.1% ± 2.4% | ~3958 |
| FILTER | 10.6% ± 1.3% | ~7300 |

### New Hypotheses Generated (7 total)
- h610: ATC coherent infectious per-drug-class quality
- h611: MEDIUM sub-pathway quality map (INVALIDATED)
- h612: Deliverable regeneration (COMPLETED)
- h613: Expanded GT gap analysis (COMPLETED)
- h614: MEDIUM sub-pathway v2 with correct GT
- h615: Expanded GT tier recalibration
- h616: Disease GT completeness score

### Recommended Next Steps
1. **h614**: Re-run quality map with expanded GT and significance tests
2. **h616**: Add GT completeness annotation to deliverable
3. External data integration (h91/h92) for fundamentally new signals
4. Literature mining (h91) remains highest-priority unblocked direction

---

## Previous Session: h603/h604 - MEDIUM Standard Rule Refinement (2026-02-06)

### h604: Standard MEDIUM Infectious Drug-Class Stratification — INCONCLUSIVE

Per-drug-class analysis of the 314 standard MEDIUM infectious predictions reveals significant heterogeneity but no cleanly demotable group with sufficient n.

**Drug class holdout:** tetracycline CLASS 32.3% (genuine MEDIUM), fluoroquinolone 11.4% (below MEDIUM, n=9.6), macrolide 10.6% (LOW-N).

**Per-drug insight:** tetracycline-the-drug has 0% holdout (n=7/seed) while doxycycline (35.4%), minocycline (36.6%), demeclocycline (31.2%) are genuine MEDIUM. Legacy drugs (tetracycline, erythromycin) at 0% but per-drug n too small.

**Decision:** NOT implementing per-drug demotions. n too small, marginal impact, risk of overfitting.

---

### h603: Standard MEDIUM Category Analysis — VALIDATED (marginal)

Analyzed all 630 standard MEDIUM predictions by disease category. Found that metabolic (10.0% full-data, 8.3% holdout), respiratory (5.9% full-data, 2.0% holdout), and endocrine (23.1% full-data, 9.5% holdout) perform far below the MEDIUM average in the standard pathway.

**Key Results:**
- Pooled met+resp+endo standard: 5.2% ± 6.6% holdout (n=10.4/seed) vs 25.2% other standard
- 20.1pp gap between these categories and the rest
- MEDIUM_DEMOTED_CATEGORIES interaction: adding categories also blocks ATC coherent pathway
  - metabolic: already excluded from ATC coherent → clean demotion (10.3% holdout)
  - respiratory: NOT excluded from ATC coherent → demotion includes ATC rescue preds (22.3% holdout)
  - endocrine: NOT excluded from ATC coherent → demotion includes ATC rescue preds (24.5% holdout)

**Implementation:** Added metabolic only to MEDIUM_DEMOTED_CATEGORIES. Respiratory and endocrine NOT demoted (holdout precision above LOW when including ATC coherent predictions).

**Impact:** MEDIUM 41.3% ± 2.8% (vs 41.4% baseline). Within noise. 20 predictions MEDIUM→LOW.

**Critical learning:** MEDIUM_DEMOTED_CATEGORIES intercepts predictions BEFORE the ATC coherent rescue pathway. Only add categories that are already in ATC_COHERENT_EXCLUDED or where combined standard+ATC precision is clearly LOW.

### Session Tier Performance (h603 update)
| Tier | Holdout | Predictions |
|------|---------|-------------|
| GOLDEN | 69.9% ± 17.9% | ~280 |
| HIGH | 58.9% ± 6.0% | ~720 |
| MEDIUM | 41.3% ± 2.8% | ~1879 |
| LOW | 15.1% ± 2.4% | ~3955 |
| FILTER | 10.6% ± 1.3% | ~7300 |

### New Hypotheses Generated (4)
- h604: Standard MEDIUM infectious drug-class stratification (P4, medium)
- h605: Highly repurposable MEDIUM demotion (P5, low)
- h606: ATC coherent respiratory/endocrine validation (P4, low)
- h607: Standard MEDIUM autoimmune quality (P5, low)

### Recommended Next Steps
1. **h604**: Largest remaining standard MEDIUM category (infectious, 314 preds, 22.7% holdout). Per-drug analysis may find demotable drug classes.
2. **h606**: Quick check — are atc_coherent respiratory/endocrine worth the ATC rescue or should they be excluded?
3. External data integration (h91/h92) for fundamentally new signals.

---

## Previous Session: h550/h598 - Antibiotic Spectrum + Targeted Cancer Expansion (2026-02-06)

### h550: Antibiotic Spectrum Validation — INVALIDATED

Tested whether within-antibacterial spectrum mismatches (gram-positive drugs for gram-negative diseases and vice versa) could filter MEDIUM infectious predictions.

**Key Results:**
- Built spectrum classification for 48 antibacterial drugs and pathogen type mapping for 38 infectious diseases
- Only 22 spectrum mismatches found in MEDIUM+ tier (4.8% of antibacterial-infectious predictions)
- **53% false positive rate** — many "mismatches" are medically valid:
  - Azithromycin→CF Pseudomonas = standard of care (anti-inflammatory + biofilm disruption)
  - Gentamicin→S. aureus = synergistic with beta-lactams (used in bacteremia)
  - Cephalexin→UTI = first-line treatment (1st-gen ceph covers E. coli)
- Only 7 genuine mismatches — far below n≈30 threshold for reliable holdout
- Full-data precision of mismatches (27.3%) still above LOW (15.6%)

**Conclusion:** Within-antibacterial spectrum matching is too nuanced for rule-based classification. The broad antimicrobial-pathogen mismatch from h560 already catches clear biological errors.

### h598: Expand CANCER_TARGETED_THERAPY — VALIDATED (+3.3pp MEDIUM)

Error analysis of MEDIUM false positives revealed 15 targeted cancer drugs missing from the cancer_targeted_therapy demotion list, despite being target/biomarker-specific drugs that should NOT generalize across cancer subtypes.

**Drugs Added (15 total):**
| Category | Drugs | Mechanism |
|----------|-------|-----------|
| Anti-HER2 mAbs | trastuzumab, pertuzumab | HER2+ cancers only |
| Anti-EGFR mAb | cetuximab | KRAS wild-type CRC, SCCHN |
| Anti-VEGFR2 mAb | ramucirumab | Anti-angiogenic, target-specific |
| PARP inhibitors | olaparib, niraparib, rucaparib | BRCA/HRD-mutant only |
| BTK inhibitors | tirabrutinib, acalabrutinib, zanubrutinib | B-cell malignancies only |
| IDH1 inhibitor | ivosidenib | IDH1-mutant AML/cholangiocarcinoma |
| mTOR inhibitor | everolimus | Tumor-specific mTOR |
| Narrow cytotoxic | trabectedin, eribulin, lanreotide | Very narrow indications |

**Holdout Validation (5-seed):**
| Group | Holdout | n/seed |
|-------|---------|--------|
| New targeted drugs | 6.1% ± 5.2% | 32.6 |
| Existing cancer_same_type | 40.2% ± 6.5% | 85.8 |
| Gap | 34.1pp | — |

**Impact:**
| Tier | Before | After | Delta |
|------|--------|-------|-------|
| MEDIUM | 38.1% ± 2.1% | 41.4% ± 2.0% | **+3.3pp** |
| Predictions moved | — | 202 MEDIUM→LOW | — |

### New Hypotheses Generated (3)
- h599: Obsolete tetracycline demotion (demeclocycline/oxytetracycline) — P4, medium
- h600: Low-precision infectious drug demotion (cefuroxime/streptomycin) — P5, low
- h601: Cancer same-type precision by drug class (remaining drugs) — P5, medium

### Session Tier Performance (h598 update)
| Tier | Holdout | Predictions |
|------|---------|-------------|
| GOLDEN | 70.3% ± 17.8% | 280 |
| HIGH | 54.7% ± 4.5% | 736 |
| MEDIUM | 41.4% ± 2.0% | 1899 |
| LOW | 13.4% ± 1.8% | 3935 |
| FILTER | 10.4% ± 1.3% | 7300 |

### Key Learnings
1. CANCER_TARGETED_THERAPY was incomplete — missing anti-target mAbs, PARP inhibitors, BTK inhibitors. When building drug class lists, check ALL therapeutic classes in the area.
2. Within-antibacterial spectrum matching fails because many antibiotics have secondary activities (synergy, anti-inflammatory) that simple classification misses. Only broad-category mismatches are clean enough for filtering.
3. Error analysis by drug (not by rule/category) is an effective way to find improvement opportunities that rule-level analysis misses.

### Recommended Next Steps
1. **h599**: Obsolete tetracycline demotion — demeclocycline and oxytetracycline have 80+ FP in MEDIUM
2. **h601**: Check remaining cancer_same_type for more low-quality drug classes
3. Consider external data integration (LINCS, PubMed mining) for fundamentally new signals

---

## Previous Session: h592/h593 - Composite Quality + GT Gap Detection (2026-02-06)

### h592: Experimental Validation Priority List — VALIDATED

Computed a composite quality score combining all validated signals (kNN rank, norm_score, TransE consilience, gene overlap, mechanism support, disease holdout precision, non-self-referentiality) to prioritize MEDIUM predictions for experimental validation.

**Key Results:**

**Holdout Validation (5-seed, MEDIUM tier):**
| Ranking Method | Q1 | Q2 | Q3 | Q4 | Q1-Q4 Gap |
|---------------|-----|-----|-----|-----|-----------|
| Composite | 14.0% ± 1.2% | 10.5% ± 0.7% | 7.1% ± 0.7% | 6.0% ± 0.7% | 8.0pp |
| kNN Rank only | 11.5% ± 0.7% | 9.2% ± 0.6% | 10.0% ± 0.8% | 7.0% ± 0.5% | 4.5pp |

**Composite beats kNN rank by +2.6pp for Q1 and 78% better separation (8.0pp vs 4.5pp gap).**

**Formula:** `1.5*rank_score + norm_score + TransE + gene_overlap + 0.5*mechanism + disease_holdout + 0.5*non_self_ref`

**Novel Non-CS MEDIUM (holdout):**
- Q1: 6.8% ± 1.1% vs Q4: 1.7% ± 0.7% (4.0x lift)

**Full-Data (novel non-CS MEDIUM):**
- Q1: 34.5% vs Q4: 9.5% (3.6x lift)

**Medical Plausibility (top 20 novel):**
- 65% reasonable (45% validated + 20% plausible) vs 56% overall MEDIUM (+9pp)
- Key validated novel: doxorubicin→choriocarcinoma (FDA), clopidogrel→CAD (FDA), enoxaparin→DIC
- Key implausible: erythromycin→meningitis (poor BBB), phenobarbital→dry skin (no mechanism)

**Key Insight: Many "novel" predictions are GT gaps, not discoveries:**
- 4/4 GOLDEN novel = FDA-approved (clopidogrel→CAD, lovastatin→atherosclerosis, etc.)
- 5/7 HIGH novel = standard treatments (levofloxacin→sinusitis, verapamil→ACS)
- Truly novel repurposing: bortezomib→Burkitt lymphoma, montelukast→IPF, lovastatin→Fabry disease

**Output:** `data/analysis/h592_validation_priority_list.json` (top 100 prioritized novel non-CS predictions)

**Difference from h443 (CLOSED):** h443 tested TransE+kNN within-tier and found no improvement over rank alone. h592 adds disease-level signals (holdout precision, self-referentiality) which provide the +2.6pp lift. This is an annotation/prioritization signal, NOT for tier changes.

### New Hypotheses Generated (3)
- h593: GT gap auto-detection from ATC/category matching (P4, medium)
- h594: Add composite_quality_score to production deliverable (P5, low)
- h595: Composite weight optimization via grid search (P5, medium)

### h593: GT Gap Auto-Detection — VALIDATED

Systematically identified FDA-approved drug-disease pairs missing from GT by checking if high-ranked predictions (rank<=5) are for drugs that already treat >=3 other diseases in the same category.

**Method:**
- 320 same-category candidates found
- 71 non-CS non-antibiotic interesting candidates
- Top 20 manually assessed: 10/20 (50%) are FDA-approved

**9 Definitive GT Gaps Added:**
1. Doxorubicin → choriocarcinoma (EMA/EP regimen)
2. Paclitaxel → germ cell testicular cancer (TIP regimen)
3. Fluorouracil → tongue cancer (head/neck SCC)
4. Verapamil → acute coronary syndrome (angina)
5. Posaconazole → cryptococcal meningitis (ECIL salvage)
6. Posaconazole → chromomycosis (triazole antifungal)
7. Posaconazole → ringworm (triazole antifungal)
8. Posaconazole → cryptococcosis (IDSA alternative)
9. Posaconazole → cutaneous candidiasis (triazole antifungal)

**Holdout Impact:** MEDIUM 35.8% → 36.6% (+0.8pp). All other tiers within seed variance.

**Key Finding:** 50% of high-evidence novel predictions are actually GT gaps, not discoveries. Posaconazole alone had 5 missing fungal disease indications. This suggests systematic GT incompleteness in antifungal and cancer drug families.

### h596: Triazole Antifungal GT Expansion — VALIDATED (marginal)

Antifungal GT was 85% complete (23/27 pairs already present). Added 4 new pairs:
- Voriconazole → cutaneous/chronic mucocutaneous candidiasis, cryptococcosis
- Isavuconazonium → zygomycosis/mucormycosis

Holdout unchanged (36.6% MEDIUM). Posaconazole gaps from h593 were the exception.

### h597: Cancer Drug GT Expansion — VALIDATED

Added 5 FDA/guideline-approved cancer drug pairs:
- Paclitaxel → larynx cancer, vulva cancer
- Cisplatin → uterine cancer
- Bortezomib → Burkitt lymphoma, anaplastic large cell lymphoma

Holdout: MEDIUM 36.6% → 37.0% (+0.4pp). Cancer drug GT more complete than expected.

### Cumulative GT Expansion (h593+h596+h597)
| Source | Pairs Added | MEDIUM Impact |
|--------|-------------|---------------|
| h593: Auto-detection | 9 | +0.8pp |
| h596: Antifungals | 4 | +0.0pp |
| h597: Cancer drugs | 5 | +0.4pp |
| **Total** | **18** | **+1.2pp** |

### New Hypotheses Generated (5 total this session)
- h593-h597: GT gap detection arc (COMPLETED)
- h594: Add composite score to deliverable (P5, low)
- h595: Composite weight optimization (P5, medium)
- h596: Triazole antifungal GT expansion (COMPLETED)
- h597: Cancer drug GT expansion (COMPLETED)

### Recommended Next Steps
1. **h594**: Add composite score to deliverable (quick implementation)
2. Consider pivoting to external data integration (LINCS, PubMed) for fundamentally new signals
3. Remaining GT gaps have diminishing returns; focus on deliverable quality

### Session Tier Performance (h597 update)
| Tier | Holdout | Predictions |
|------|---------|-------------|
| GOLDEN | 69.9% ± 17.9% | 280 |
| HIGH | 58.9% ± 6.0% | 754 |
| MEDIUM | 37.0% ± 2.8% | 2083 |
| LOW | 15.6% ± 2.4% | 3733 |
| FILTER | 10.6% ± 1.3% | 7300 |

### Key Learnings
1. Disease-level signals (holdout precision, self-referentiality) add genuine value for prediction prioritization that prediction-level signals (TransE, kNN score) miss. The composite score is useful for practical experiment prioritization but NOT for tier reassignment.
2. 50% of high-evidence "novel" predictions are actually GT gaps (FDA-approved but missing from our GT). Posaconazole had 5 missing fungal indications. Drug families have correlated GT gaps — fixing one suggests checking the whole family.
3. GT incompleteness inflates the "novel prediction" count and deflates measured precision. Always check for GT gaps before claiming novel discoveries.

---

## Previous Session: h586/h588 - GT-Free Quality Signals (2026-02-06)

### h586: GT-Free Paradigm Mismatch via DRKG Edges — INVALIDATED

Tested whether DRKG non-treatment edges (gene associations, anatomy, symptoms) can approximate Drug Jaccard (treatment paradigm similarity) without GT knowledge.

**Key Finding: Biology ≠ Treatment Paradigm**
- Gene Jaccard: r=+0.079 with holdout (NS), r=+0.086 with drug Jaccard — too weak
- Combined DRKG Jaccard (genes+anatomy+symptoms): r=+0.077 (NS)
- GT-free mismatch (embed_sim - combined_jaccard): r=0.996 with embed_sim — just embedding similarity in disguise
- 63% of diseases share ZERO genes with their kNN neighbors (too sparse)
- After controlling for self-referentiality: partial_r=+0.030 (NS)

**Why genes fail:** Disease-gene associations capture molecular biology, but treatment decisions are driven by clinical phenotype, drug class availability, and treatment paradigms. Two diseases with identical genes can be treated with completely different drug classes (e.g., hypertension vs PAH).

**Symptom/anatomy edges (Hetionet):** Show promise (symptom r=+0.297) but only 39/312 diseases have coverage.

### h588: HPO Symptom Phenotype Similarity as Quality Signal — VALIDATED (annotation)

Tested HPO phenotype similarity as an extended version of the sparse Hetionet symptom signal. HPO matrix covers 799 diseases (82/312 holdout diseases, 2x Hetionet coverage).

**Key Results:**
| Signal | r with holdout | Partial r (ctrl GT) | Coverage |
|--------|---------------|-------------------|----------|
| HPO sim | +0.243* | +0.258* | 82 diseases |
| HPO→Drug Jaccard proxy | +0.390*** | +0.416*** | 82 diseases |
| Gene→Drug Jaccard proxy | +0.086 | +0.116* | 270 diseases |

*p<0.05, ***p<0.001

**Why HPO works better than genes:** Clinical phenotype (symptoms, signs, lab findings) captures treatment paradigm similarity 4.5x better than molecular biology (gene overlap).

**Practical limitation:** Adds only 0.9% incremental R² beyond GT size + embed_sim. HPO sim partial_r=+0.149 (NS) after controlling for embed_sim. Coverage still limited (26.3%).

**Quartile analysis:** Q4 (highest HPO sim) = 11.3% holdout vs Q1 = 6.4%. Within GT 1-20 band: 3.8% vs 1.7% (2.2x lift).

**Conclusion:** Annotation value for deliverable. NOT promotable for tiers.

### New Hypotheses Generated (3)
- h588: HPO similarity (COMPLETED - VALIDATED)
- h589: ATC hierarchy as GT-free treatment paradigm proxy (P4, medium)
- h590: Hetionet disease-resembles as augmented kNN signal (P5, low)

### Key Learning
Clinical phenotype (HPO) is the best GT-free proxy for treatment paradigm similarity. Molecular biology (gene overlap) fails to predict treatment similarity. The gap between biology and therapy is fundamental: diseases with shared genes may have completely different treatment paradigms. This is consistent with h571 (therapeutic islands) and h583 (paradigm mismatch).

### Session Tier Performance (unchanged from h560)
| Tier | Holdout | Predictions |
|------|---------|-------------|
| GOLDEN | 69.9% ± 17.9% | 280 |
| HIGH | 59.5% ± 6.2% | 754 |
| MEDIUM | 35.8% ± 2.8% | 2083 |
| LOW | 15.5% ± 2.4% | 3733 |
| FILTER | 10.6% ± 1.3% | 7300 |

### h589: ATC Hierarchy as GT-Free Treatment Paradigm Proxy — VALIDATED (circular)

ATC codes are the best proxy for drug Jaccard (treatment paradigm similarity):

| Proxy Signal | r with Drug Jaccard | r with Holdout | Partial r (ctrl GT) |
|-------------|-------------------|---------------|-------------------|
| ATC L5 | +0.848 | +0.213 | +0.266 |
| ATC L3 | +0.737 | +0.180 | +0.229 |
| ATC L2 | +0.665 | +0.157 | +0.206 |
| HPO sim (h588) | +0.390 | +0.243 | +0.258 |
| Gene overlap (h586) | +0.086 | +0.079 | +0.037 |

**BUT ATC adds ZERO signal beyond drug Jaccard:**
- ATC L2 | drug Jaccard: partial_r=-0.013 (p=0.82, NS)
- Incremental R²: 0.04%

**ATC is a noisy version of drug Jaccard, not an independent signal.** This is because ATC codes come from GT drugs — fully circular. 50% of zero-drug pairs have non-zero ATC L2 overlap, but this doesn't help holdout.

**This closes the GT-free treatment paradigm proxy search.** No DRKG-derived signal (genes, HPO, ATC) provides independent information beyond drug Jaccard. Treatment paradigm knowledge requires treatment data.

### h590: Hetionet Disease-Resembles as Augmented kNN Signal — INVALIDATED

Tested whether Hetionet DrD (disease resembles disease) edges can augment kNN neighborhoods with curated medical knowledge.

**Findings:**
- Only 33/312 holdout diseases have resembles edges (10.6%)
- 80.8% of resembles neighbors are already in kNN top-20
- Drug overlap: resembles 0.036 < kNN 0.045 (kNN finds BETTER drug neighbors)
- Only 4/33 diseases gain ANY new GT drugs from resembles (mean 0.2/disease)
- Embedding already captures resembles (trained on same DRKG graph)

**Conclusion:** No augmentation value. Node2Vec embeddings subsume Hetionet edges.

### Session Summary: GT-Free Quality Signal Arc (h586→h588→h589→h590)

This session systematically explored whether DRKG-derived signals can independently predict treatment paradigm similarity:

| Signal | r with Drug Jaccard | r with Holdout | Independent? |
|--------|-------------------|---------------|-------------|
| Drug Jaccard (oracle) | 1.000 | +0.251 | GT-dependent |
| ATC L3 (h589) | +0.737 | +0.180 | Circular (GT) |
| HPO phenotype (h588) | +0.390 | +0.243 | Modest (+0.9% R²) |
| Gene overlap (h586) | +0.086 | +0.079 | None |
| Resembles (h590) | — | — | Subsumed by kNN |

**Key insight:** Treatment paradigm information exists ONLY in treatment data. No biological (genes), phenotypic (HPO), or graph-structural (resembles) signal provides independent prediction of treatment similarity. ATC hierarchy is a strong proxy but fully circular with drug Jaccard.

**The only partially independent signal is HPO phenotype similarity**, but at +0.9% incremental R², it's not actionable for tier changes.

### h591: LOW-Tier Success Pattern Analysis — VALIDATED (characterization)

Full-data analysis of which LOW predictions hit GT (20.0% = 747/3733).

Top success patterns: cancer_targeted_therapy 39.0%, immunological demotion 34.0%, Mech+Rank<=5 39.0%. 67.5% of LOW GT hits are known indications. All demotion rules confirmed correct. Useful for deliverable annotation, not tier changes.

### Recommended Next Steps
1. **h534**: TransE FILTER annotation for manual review (low effort)
2. **h539**: Cancer drug class annotation (low effort, deliverable improvement)
3. Consider pivoting to entirely external data (clinical guidelines, RWD, LINCS)

---

## Previous Session: h571 - Therapeutic Island Rescue Analysis (2026-02-06)

### h571: Therapeutic Island Disease Rescue — INVALIDATED

Comprehensive analysis of 9 "therapeutic island" diseases (GT>=5, 0% holdout) to determine whether alternative prediction strategies could rescue them.

**Islands Analyzed:**
| Disease | GT | Self-Ref | MEDIUM+ Preds | Failure Mode |
|---------|-----|---------|---------------|-------------|
| Immunodeficiency | 268 | 100% | 0 | immunological demotion |
| ADHD | 87 | 100% | 18 | hierarchy+ATC works, kNN blind |
| HCV | 50 | 100% | 2 | disease-specific antivirals |
| PAH | 36 | 100% | 11 | different paradigm than hypertension |
| Migraine | 35 | 100% | 2 | triptans/CGRPs not in kNN |
| Agranulocytosis | 31 | 83% | 0 | hematological demotion |
| Narcolepsy | 26 | 75% | 2 | stimulants unique to cluster |
| DKA | 12 | 100% | 0 | base_to_complication filter |
| Scabies | 9 | 80% | 2 | antiparasitic drugs unique |

**Key Finding 1: NOT drug uniqueness, but neighbor drug mismatch**
- All 9 islands have 67-100% of GT drugs shared with other diseases
- But kNN neighbors have VERY low drug overlap: mean 0.3-7.2 drugs (vs 6.2-22.6 for high performers)
- Islands are embedded NEAR other diseases (sim 0.498-0.791) but treated with DIFFERENT drugs
- e.g., PAH is near hypertension (uses PDE5i/ERA/prostacyclins) but neighbors use ACEi/ARBs/CCBs

**Key Finding 2: Alternative signals cannot help**
- TransE consilience: 0% for most island GT predictions
- Gene overlap: Present for ADHD/immunodeficiency but circular with kNN
- Drug class: Already exploited by hierarchy rules and ATC coherence
- All signals annotate EXISTING kNN predictions, cannot generate NEW ones

**Key Finding 3: System already works for some islands via non-kNN paths**
- ADHD: 18 MEDIUM+ predictions (8 known GT drugs) via psychiatric ATC + target overlap
- PAH: 11 HIGH predictions (all known) via cardiovascular hierarchy
- These non-kNN paths work; kNN just adds no value for these diseases

**Key Finding 4: 0% holdout is misleading for self-referential diseases**
- PAH has 11 correct HIGH predictions but holdout = 0%
- Holdout penalizes self-referential diseases because GT contributions vanish when held out
- The deliverable is actually CORRECT for these diseases

**Conclusion: No rescue possible within kNN architecture. Need fundamentally different approach.**

### New Hypotheses Generated (4)
- h580: Drug class expansion for migraine (P3, high impact)
- h581: Holdout metric correction excluding self-ref diseases (P4, medium, low effort)
- h582: kNN neighbor drug overlap as quality signal (P4, low, medium effort)
- h583: Treatment paradigm mismatch detection (P4, medium)

### Session Tier Performance (unchanged from h560)
| Tier | Holdout | Predictions |
|------|---------|-------------|
| GOLDEN | 69.9% ± 17.9% | 280 |
| HIGH | 59.5% ± 6.2% | 754 |
| MEDIUM | 35.8% ± 2.8% | 2083 |
| LOW | 15.5% ± 2.4% | 3733 |
| FILTER | 10.6% ± 1.3% | 7300 |

### h581: Holdout Metric Correction for Self-Referential Diseases — VALIDATED (major meta-finding)

Excluding 100% self-referential diseases from holdout reveals significantly higher true discovery rates:

| Tier | All (reported) | Non-Self-Ref | Delta | <50% Self-Ref |
|------|----------------|-------------|-------|---------------|
| GOLDEN | 70.3% ± 17.8 | 72.0% ± 16.8 | +1.7pp | 72.0% ± 19.0 |
| HIGH | 54.6% ± 4.8 | 59.4% ± 5.8 | +4.8pp | 60.8% ± 6.6 |
| MEDIUM | 37.6% ± 2.1 | 40.5% ± 3.0 | +2.9pp | 44.0% ± 2.8 |
| LOW | 13.7% ± 1.8 | 17.2% ± 1.5 | +3.5pp | 19.2% ± 1.0 |
| FILTER | 10.4% ± 1.3 | 13.6% ± 2.3 | +3.2pp | 15.5% ± 2.8 |

**Self-ref disease contribution to holdout predictions:**
- GOLDEN: 2% (negligible), HIGH: 13%, MEDIUM: 16%, LOW: 29%, FILTER: 34%

**Critical insight: MEDIUM 40% target IS ACHIEVED for non-self-ref diseases (40.5%)!**
The target deemed unachievable in h561 is actually met when excluding structural zeros.
For <50% self-ref: MEDIUM = 44.0% — exceeds 40% target significantly.

### New Hypotheses Generated (2)
- h584: Deliverable non-self-ref precision annotation (P5, low)
- h585: Self-referentiality threshold optimization (P5, low)

### h583: Treatment Paradigm Mismatch Detection — VALIDATED (novel independent signal)

Paradigm mismatch = mean_embedding_sim - mean_drug_Jaccard (high = diseases near in embedding but far in drug space).

**Key Results:**
| Signal | r (all) | Partial r (ctrl GT) | Partial r (ctrl self-ref) |
|--------|---------|--------------------|--------------------------|
| Drug Jaccard | +0.398 | +0.428 | — |
| Paradigm mismatch | -0.388 | -0.303 | -0.408 |
| Embedding sim | -0.177 | — | — |
| Mismatch vs self-ref | +0.032 | — | — |

**Critical: NOT a self-referentiality proxy (r=0.032).** This is an INDEPENDENT signal.

**Quartile analysis (all diseases):** Q1=15.7%, Q2=7.5%, Q3=5.4%, Q4=3.9% holdout.
**Non-self-ref only:** Q1=19.3% vs Q4=6.4%. Signal persists after removing self-ref.

**Limitation:** Drug Jaccard requires GT knowledge → circular at prediction time. Valid as annotation only.

### New Hypotheses Generated (2)
- h586: GT-free paradigm mismatch via DRKG edges (P4, medium)
- h587: Paradigm mismatch deliverable annotation (P5, low)

### h585: Self-Referentiality Threshold Optimization — VALIDATED (confirms 100% boundary)

Self-ref is bimodal: 0% (n=112) and 100% (n=144) dominate. Band analysis:

| Band | n | Mean Holdout | GT Size |
|------|---|-------------|---------|
| 0% | 80 | 8.7% | 20.2 |
| 1-25% | 21 | 26.8% | 86.0* |
| 26-50% | 54 | 14.0% | ~35 |
| 51-75% | 43 | 9.6% | ~25 |
| 76-99% | 15 | 5.2% | ~20 |
| 100% | 99 | 0.3% | ~15 |

*1-25% band confounded by large-GT diseases (RA, ovarian cancer, COPD).

**Within non-100% diseases, self-ref has MINIMAL predictive power (r=-0.104).**
Only the 100% vs <100% boundary matters. This confirms h581's choice of cutoffs.

### Recommended Next Steps
1. **h586**: GT-free paradigm mismatch approximation (if DRKG edges proxy drug Jaccard, novel non-circular signal)
2. **h580**: Drug class expansion for islands (high effort, new predictions)
3. **h584**: Add corrected precision to deliverable metadata

### Key Learning
Therapeutic islands fail because kNN neighbors treat with different drug classes, not because drugs are unique. The embedding space captures disease similarity but NOT treatment paradigm similarity. This is a fundamental limitation of Node2Vec embeddings trained on DRKG: they capture knowledge graph structure but not clinical treatment patterns.

---

## Previous Session: h576/h577/h579 - LOW Promotion + CS Artifact + Novel Precision (2026-02-06)

### h576: LOW Tier Promotion Analysis — INVALIDATED

Comprehensive analysis of 3,733 LOW predictions to identify promotion candidates.

**Holdout by tier_rule (5-seed):**
| Rule | Holdout% | n/seed | Notes |
|------|----------|--------|-------|
| incoherent_demotion | 44.2% ± 14.0% | 34 | Driven by CS→TB artifact |
| cardiovascular_medium_demotion | 25.1% ± 19.4% | 69 | Too variable |
| local_anesthetic_procedural | 23.8% ± 5.9% | 52 | Stable but below MEDIUM |
| hematological_corticosteroid_demotion | 23.7% ± 21.5% | 24 | High variance |
| default | 18.5% ± 3.7% | 298 | Appropriately LOW |

**Compound signals:**
| Signal | Holdout% | n/seed | Notes |
|--------|----------|--------|-------|
| TransE+Mech+Rank<=10 | 51.6% | 8 | Too small |
| TransE+Mech | 41.5% ± 19.4% | 14 | 73% are CS |
| Rank<=5+TransE | 40.0% ± 15.3% | 24 | CS-inflated |
| Freq>=10+Mech | 36.4% ± 11.7% | 54 | Mixed population |
| Mechanism overall | 25.0% ± 6.3% | 154 | Heterogeneous |
| TransE overall | 21.4% ± 7.7% | 94 | +6.6pp vs no TransE |

**Key finding: incoherent_demotion deep dive**
- h488 originally found 3.6% holdout for MEDIUM-level incoherent → LOW
- But incoherent_demotion also demotes HIGH-level (freq>=15+mech) → LOW
- HIGH-level incoherent = 44.2%, driven by CS→infectious (45.1%)
- Non-CS incoherent = 11.7% → correctly at LOW
- If promoted to MEDIUM, h557 post-processing would re-demote CS→infectious to LOW
- Net effect: only 11 non-CS predictions at 11.7% would be promoted → WORSE for MEDIUM
- **Decision: DO NOT PROMOTE. All demotion rules are correctly calibrated.**

### h579: MEDIUM Novel-Only Precision — VALIDATED (structural finding)

**100% of predicted drugs treat at least one training disease.** Zero "novel drug" predictions exist.

This is structural: kNN collaborative filtering only recommends drugs from similar diseases. If a drug doesn't treat ANY training disease, it won't appear in kNN neighbors. The system is inherently a drug REPURPOSING engine — all predictions are cross-disease transfer.

All holdout hits represent genuine drug repurposing: drug known for training diseases, correctly predicted for held-out disease. The tier precision numbers represent true repurposing discovery rates.

### h577: Corticosteroid Holdout Artifact — VALIDATED (major meta-finding)

High-frequency corticosteroids (freq 30-42) inflate holdout precision:

| Tier | All | CS | Non-CS | CS % | Inflation |
|------|-----|-----|--------|------|-----------|
| GOLDEN | 69.9% | 100.0% | 65.2% | 39% | +34.8pp |
| HIGH | 58.8% | 61.7% | 48.5% | 69% | +13.2pp |
| MEDIUM | 36.5% | 65.1% | 34.8% | 6% | +30.3pp |
| LOW | 15.5% | 22.8% | 14.3% | 14% | +8.4pp |
| FILTER | 10.6% | 21.5% | 10.1% | 5% | +11.4pp |

**Key insights:**
1. **HIGH is 69% corticosteroids!** Non-CS HIGH precision is 48.5%, not 58.8%
2. **MEDIUM barely affected** (6% CS): non-CS 34.8% vs total 36.5%
3. **Tier ordering preserved for non-CS**: GOLDEN 65.2% > HIGH 48.5% > MEDIUM 34.8% > LOW 14.3% > FILTER 10.1%
4. Biggest category inflation: renal +58pp, metabolic/musculoskeletal +44pp
5. Cancer/cardiovascular: NO CS inflation

**Implication**: Report CS-free precision as supplemental "discovery potential" metric. The tier system is valid but its numbers overstate non-obvious discovery potential, especially for HIGH tier.

### New Hypotheses Generated (3)
- h577: CS holdout artifact (P4, medium) — COMPLETED
- h578: LOW TransE annotation (P5, low)
- h579: Novel-only precision (P4, low) — COMPLETED

### Session Tier Performance (unchanged from h559)
| Tier | Holdout | Non-CS Holdout |
|------|---------|----------------|
| GOLDEN | 69.9% ± 17.9% | 65.2% |
| HIGH | 58.8% ± 6.2% | 48.5% |
| MEDIUM | 36.5% ± 3.0% | 34.8% |
| LOW | 15.5% ± 2.4% | 14.3% |
| FILTER | 10.6% ± 1.3% | 10.1% |

### Recommended Next Steps
1. **h577 follow-up**: Add CS-free precision to deliverable metadata
2. **h578**: Flag best LOW predictions for manual review
3. Consider pivoting to external data integration (h545, h91) since internal improvements are exhausted

### Key Learning
LOW→MEDIUM promotion is NOT possible with current signals. All demotion rules are correctly calibrated. CS inflation is a meta-issue that inflates tier precision numbers but doesn't affect tier ordering. The system's "true" discovery potential for non-obvious drug repurposing is ~48.5% for HIGH (not 58.8%) and ~34.8% for MEDIUM (not 36.5%). Future improvement requires external data or fundamentally new signals.

---

## Previous Session: h563/h567/h572 - Promotion/Mismatch/Coherence Analysis (2026-02-06)

### h563: LA Procedural MEDIUM→HIGH Promotion — INCONCLUSIVE

LA procedural MEDIUM predictions are LA drugs demoted to LOW by h540 then rescued to MEDIUM by target_overlap.
Only 41 full-data predictions (6.6/seed holdout). 28/41 are bupivacaine.
- Full-data precision: 31.7% (at MEDIUM level, not HIGH)
- Holdout: 24.9% ± 16.4% — too noisy with n=6.6/seed
- Decision: KEEP AS-IS. Too few predictions to justify code change.

### h567: Drug Class × Disease Type Mismatch Matrix — VALIDATED (confirms demotion ceiling)

Comprehensive cross-tabulation of 18 SOC drug classes + 12 broad therapeutic classes × 14 disease categories for MEDIUM predictions.
- Only 1 candidate: DMARDs→cancer (19.5%, n=41, 2.0% holdout) — BUT all 41 are methotrexate, which IS a cancer drug
- Anti-thyroid→metabolic: 0% (n=10) — genuine but too small
- **CONCLUSION: Existing filters are comprehensive. No new demotion rules available.**
- Demotion ceiling at ~35.8% MEDIUM confirmed

### h572: kNN Neighborhood Category Coherence — INVALIDATED

Tested whether fraction of same-category among k=20 kNN neighbors predicts precision.
- r = -0.002 (coherence vs holdout precision) — ZERO signal
- r = -0.028 (coherence vs GT size) — ZERO signal
- Node2Vec embeddings cluster by drug-sharing patterns, NOT disease category
- 91% of diseases have <20% same-category neighbors (mean=0.064, median=0.000)
- **Key insight: kNN works via drug-pattern similarity, not category similarity**

### New Hypotheses Generated (3)
- h573: kNN score gap as prediction confidence (P4, medium)
- h574: Drug-sharing density as disease quality signal (P5, low)
- h575: Methotrexate cancer subtype specificity (P5, medium)

### h573: kNN Score Gap as Prediction Confidence — VALIDATED

kNN norm_score adds signal beyond rank for prediction quality:
- Within-rank: high-score vs low-score = +9.2pp for rank 1-5, +8.8pp for rank 6-10
- Within MEDIUM: Q4 (highest score) 28.2% vs Q1 (lowest) 12.6% (+15.6pp)
- Score gap Q4: 29.2% within MEDIUM
- NOT circular with GT size (r=-0.009)
- Q4 MEDIUM (28.2%) << HIGH (59.5%) — useful as annotation, not promotable
- norm_score already stored in deliverable; no code change needed

### h574: Drug-Sharing Density as Disease Quality Signal — VALIDATED (circular)

Mean drug overlap between disease and k=20 kNN neighbors independently predicts holdout precision:
- r=0.434 with holdout precision
- Partial r=0.448 AFTER controlling for GT size (independent!)
- r=0.182 with GT size (NOT a proxy)
- 93% of diseases have near-zero drug sharing — most holdout diseases share no drugs with neighbors
- CIRCULAR (uses GT) — annotation only, not for novel predictions
- Key insight: kNN generalizes when neighbors share drugs; fails structurally when they don't

### h559: CS→Infectious HIGH TB Hierarchy Demotion — VALIDATED (marginal)

CS→TB hierarchy predictions demoted from HIGH→MEDIUM:
- 18 predictions (CS drugs × 3 TB diseases)
- Full-data: CS→TB 33.3% vs non-CS infectious HIGH 76.9% (43.6pp gap)
- Holdout: HIGH -0.7pp, MEDIUM +0.8pp (both within seed variance)
- Medically justified (dexamethasone→TB meningitis is valid, but generic CS→TB is not SOC)
- Protected from h557 cascade (stays MEDIUM, not demoted to LOW)

### Session Tier Performance (h559 update)
| Tier | Holdout | Delta vs h560 |
|------|---------|---------------|
| GOLDEN | 69.9% ± 17.9% | 0.0pp |
| HIGH | 58.8% ± 6.1% | -0.7pp |
| MEDIUM | 36.6% ± 3.0% | +0.8pp |
| LOW | 15.5% ± 2.4% | 0.0pp |
| FILTER | 10.6% ± 1.3% | 0.0pp |

### Recommended Next Steps
1. **h571**: Therapeutic island rescue (P3, high impact but high effort)
2. **h545**: Gene-poor disease expansion (P4, medium)
3. **h573 follow-up**: Consider norm_score thresholds for deliverable prioritization

### Key Learning
MEDIUM demotion is exhausted at 35.8%. All major drug-class × category mismatches are filtered.
Future MEDIUM improvement requires: (1) promotions, (2) new signals, or (3) external data.
Embedding space clusters by drug sharing, not disease category — quality signals must exploit this structure.

---

## Previous Session: h560 - Antimicrobial-Pathogen Mismatch Filter (2026-02-06)

### h569: Disease-Level Precision Audit — VALIDATED

37% of diseases (121/325) have 0% holdout precision. 80% of these have GT≤2 (structural limit).
GT size strongly predicts disease-level precision (r=0.732):
- GT≤2: 1.1% | GT 3-5: 5.0% | GT 6-10: 12.4% | GT 11-20: 19.7% | GT 21-50: 28.3% | GT 51+: 70.0%
Notable therapeutic island failures: PAH (GT=26, 0%), HCV (GT=11, 0%), migraine (GT=9, 0%).
Top performers: RA (93.3%), UC (66.7%), AS (63.3%) — all large-GT autoimmune diseases.

### h570: Disease Confidence Annotation — VALIDATED

Added `disease_holdout_precision` column to deliverable (9336/14150 predictions annotated).
Per-disease holdout precision computed across 5 seeds. Fixed json import shadowing bug.

### h560: Antimicrobial-Pathogen Mismatch Filter — VALIDATED

Extended h556's antibiotic→viral filter to comprehensive antimicrobial-pathogen mismatch detection.

**Key Finding:** 0.0% holdout precision for ALL antimicrobial-pathogen mismatches across 5 seeds (132 total mismatches, 0 hits). Matched predictions = 27.8% ± 4.8%.

**Mismatch Types Detected:**
| Mismatch Type | n/seed | Notes |
|---------------|--------|-------|
| antibacterial → parasitic | 8.0 | Cephalosporins/FQs for malaria/toxo/leish |
| antibacterial → fungal | 7.4 | FQs/macrolides for candidiasis/aspergillosis |
| antifungal → parasitic | 4.8 | Azoles/echinocandins for schistosomiasis/Chagas |
| antibacterial → viral | 3.0 | Already partially covered by h556 |
| antifungal → viral | 0.8 | Amphotericin B for hepatitis C |
| Other | 2.4 | Mixed |

**Dual-Activity Drug Handling:**
- Metronidazole: antibacterial + antiparasitic (treats trichomoniasis, amebiasis)
- Doxycycline/tetracycline: antibacterial + antiparasitic (malaria prophylaxis)
- Amphotericin B: antifungal + antiparasitic (leishmaniasis first-line)
- Sulfadiazine: antibacterial + antiparasitic (toxoplasmosis first-line)

**10 Legitimate Cross-Pathogen Pairs Excluded** (e.g., doxycycline→malaria, amphotericin B→leishmaniasis, ketoconazole→Chagas)

**Bug Found:** Target overlap promotion was rescuing 11 mismatch predictions from LOW back to MEDIUM. Fixed by adding `antimicrobial_pathogen_mismatch` to the target_overlap block list.

**Implementation:**
- Replaced h556's `antibiotic_viral_mismatch` with comprehensive `antimicrobial_pathogen_mismatch` rule
- Drug classification: antibacterial (48 drugs), antifungal (15), antiparasitic (20), dual-activity (5)
- Disease classification: viral (14 keywords), fungal (15 keywords), parasitic (11 keywords)
- ~30 MEDIUM predictions demoted to LOW, 292 total predictions tagged

**Holdout Impact:**
| Tier | Before (h562) | After (h560) | Delta |
|------|--------------|-------------|-------|
| GOLDEN | 69.9% ± 17.9% | 69.9% ± 17.9% | 0.0pp |
| HIGH | 59.5% ± 6.2% | 59.5% ± 6.2% | 0.0pp |
| **MEDIUM** | **34.9% ± 3.1%** | **35.8% ± 2.8%** | **+0.9pp** |
| LOW | 16.0% ± 2.5% | 15.5% ± 2.4% | -0.5pp |
| FILTER | 10.4% ± 1.3% | 10.6% ± 1.3% | +0.2pp |

**Cumulative MEDIUM improvement since h553:** +5.7pp (30.1% → 35.8%)

**New Hypotheses Generated (3):**
- h565: Azole antifungal anti-parasitic activity validation (P5, low)
- h566: Infectious target_overlap quality audit (P5, medium)
- h567: Drug class × disease type matrix for all categories (P4, high)

**Recommended Next Steps:**
1. **h567**: Systematic drug-class × disease-subtype mismatch scan across all categories
2. **h559**: CS→infectious HIGH TB hierarchy review
3. **h563**: LA procedural MEDIUM→HIGH promotion

---

## Previous Session: h557 - Corticosteroid→Infectious Demotion (2026-02-06)

### h557: Corticosteroid→Infectious Disease Selective Demotion — VALIDATED

Analyzed all 174 corticosteroid→infectious disease predictions across all tiers.

**Medical Classification of 33 Infectious Diseases:**
| Validity | Diseases | Rationale |
|----------|----------|-----------|
| VALID (6) | ABPA, herpes zoster, leprosy, TB, extrapulmonary TB, proctitis | CS are established adjunctive therapy |
| QUESTIONABLE (11) | Cryptococcosis, fungal meningitis, influenza, HSE, aspergillosis, etc. | Some evidence but not standard |
| INVALID (16) | Hep B/C, CMV, rabies, smallpox, zygomycosis, candidiasis, etc. | CS harmful or useless |

**Holdout Precision (5-seed):**
| Group | Holdout | n/seed | vs MEDIUM avg |
|-------|---------|--------|--------------|
| ALL CS→infectious MEDIUM | 2.1% ± 2.5% | 11.6 | -31.8pp |
| VALID CS→infectious | 2.9% ± 3.5% | 8.0 | -31.0pp |
| QUESTIONABLE | 0.0% | 1.8 | -33.9pp |
| INVALID | 0.0% | 1.8 | -33.9pp |
| Non-CS infectious MEDIUM | 18.7% ± 5.2% | 113.4 | -15.2pp |

**Key Finding:** Medical validity does NOT predict holdout performance. Even ABPA/zoster/leprosy/TB (genuinely valid uses) have 2.9% holdout. The KG co-occurrence signal doesn't generalize when specific diseases are held out.

**Implementation:**
- Added `infectious_corticosteroid_demotion` rule: CS + infectious + MEDIUM → LOW
- 59 predictions demoted
- Rule classified as GENUINE (16.1% ± 7.4% holdout = LOW-level)

**Holdout Impact:**
| Tier | Before (h555) | After (h557) | Delta |
|------|--------------|-------------|-------|
| GOLDEN | 70.3% ± 17.8% | 69.9% ± 17.9% | -0.4pp |
| HIGH | 58.7% ± 6.1% | 59.5% ± 6.2% | +0.8pp |
| **MEDIUM** | **33.9% ± 2.5%** | **34.2% ± 2.7%** | **+0.3pp** |
| LOW | 16.2% ± 2.7% | 16.0% ± 2.5% | -0.2pp |
| FILTER | 10.5% ± 1.3% | 10.4% ± 1.3% | -0.1pp |

**Cumulative MEDIUM improvement since h553:** +4.1pp (30.1% → 34.2%)

**New Hypotheses Generated (3):**
- h559: CS→infectious HIGH (TB hierarchy) review (P5, low)
- h560: Antifungal↔bacterial cross-pathogen mismatch (P5, medium)
- h561: Cumulative MEDIUM precision analysis vs 40% target (P4, medium)

**Recommended Next Steps:**
1. **h561**: MEDIUM precision gap analysis — what's left to improve?
2. **h560**: Cross-pathogen drug-disease mismatch filter
3. **h532**: Every Cure GT error report

### h561: Cumulative MEDIUM Precision vs 40% Target — VALIDATED

**Comprehensive sub-reason analysis shows MEDIUM demotion ceiling reached at ~34.2%.**

**MEDIUM Sub-Reason Holdout Precision (5-seed, proper GT):**
| Sub-Reason | Holdout | n/seed | % of MEDIUM | Notes |
|------------|---------|--------|-------------|-------|
| cancer_same_type | 27.9% | 125 | 39.3% | Largest drag, genuine MEDIUM |
| default | 36.6% | 101 | 29.6% | At average |
| atc_coherent_infectious | 36.4% | 50 | 9.1% | At average |
| target_overlap_promotion | 43.0% | 31 | 5.9% | Above average |
| cv_pathway_comprehensive | 40.8% | 16 | 4.8% | Above average |
| local_anesthetic_procedural | 50.5% | 11 | 1.9% | Potential promote to HIGH |
| atc_coherent_psychiatric | 45.3% | 11 | 2.8% | Above average |

**Ceiling Analysis:**
- Only 1 remaining demotion candidate: infectious_hierarchy_pneumonia (16.7%, n=6) — too small
- Max from further demotions: ~35.0% (negligible improvement)
- Without cancer_same_type: 38.3% (but loses 846 genuine predictions)
- **40% NOT achievable via demotions**

**MEDIUM Precision Journey:**
30.1% → 31.7% (h553) → 32.1% (h556) → 33.9% (h555) → 34.2% (h557) = **+4.1pp from 770 demotions**

**New Hypotheses Generated (3):**
- h562: Cancer same-type subtype specificity (P4, medium) — highest impact remaining
- h563: LA procedural MEDIUM→HIGH promotion (P5, low)
- h564: Deliverable regeneration with updated tiers (P4, low)

### h562: Cancer Same-Type Subtype Specificity Analysis — VALIDATED (Bug Fix)

Expected to find cross-subtype contamination. Instead found a substring matching bug.

**Finding 1:** All 846 cancer_same_type predictions were already SAME_SUBTYPE (100%). No cross-subtype issue.

**Finding 2:** `extract_cancer_types()` had a substring bug with short abbreviations:
- `'ALL'` (Acute Lymphoblastic Leukemia) matched "sm**all**", "f**all**opian", "**all**ergic"
- 8 diseases falsely tagged as leukemia, inflating cancer_same_type count by 39 predictions

**Fix:** Word boundary regex (`\b`) for keywords <=4 chars (ALL, CLL, AML, CML, SCLC).

**Impact:**
| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| MEDIUM predictions | 2152 | 2113 | -39 |
| cancer_same_type | 846 | 807 | -39 |
| cancer_same_type holdout | 27.9% | 29.4% | +1.5pp |
| **Overall MEDIUM holdout** | **34.2%** | **34.9%** | **+0.7pp** |

**Recommended Next Steps:**
1. **h564**: Deliverable regeneration (practical impact for collaboration)
2. **h560**: Cross-pathogen mismatch filter (infectious sub-type)
3. **h532**: Every Cure GT error report

---

## Previous Session: h553+h554+h555+h556 - MEDIUM Precision Deep Dive (2026-02-06)

### h553: MEDIUM Tier Precision by Category Analysis — VALIDATED

Computed holdout precision for MEDIUM predictions broken down by disease category.

**Category-Level Holdout Precision (MEDIUM only, 5-seed mean):**
| Category | Holdout | ±std | n/seed | vs avg |
|----------|---------|------|--------|--------|
| psychiatric | 54.8% | 6.6% | 19 | +24.6pp |
| musculoskeletal | 53.8% | 28.6% | 12 | +23.7pp |
| cardiovascular | 35.4% | 9.9% | 21 | +5.3pp |
| dermatological | 31.2% | 4.6% | 28 | +1.1pp |
| autoimmune | 30.7% | 14.9% | 23 | +0.6pp |
| respiratory | 29.2% | 9.0% | 14 | -0.9pp |
| cancer | 27.9% | 4.5% | 125 | -2.2pp |
| infectious | 26.6% | 8.1% | 104 | -3.5pp |
| **metabolic** | **18.0%** | 6.7% | 16 | **-12.1pp** |
| **hematological** | **10.0%** | 20.0% | 8 | **-20.1pp** |

**Sub-Reason Analysis (MEDIUM):**
| Sub-reason | Holdout | n/seed | vs avg |
|------------|---------|--------|--------|
| target_overlap_promotion | 45.3% | 45 | +15.2pp |
| local_anesthetic_procedural | 44.2% | 13 | +14.1pp |
| cv_pathway_comprehensive | 40.8% | 16 | +10.6pp |
| cardiovascular | 36.4% | 8 | +6.3pp |
| cancer_same_type | 27.9% | 125 | -2.2pp |
| default | 26.9% | 184 | -3.3pp |
| **metabolic (statin/TZD rescue)** | **8.3%** | 4.2 | **-21.8pp** |

**Changes Implemented:**
1. Hematological MEDIUM→LOW demotion (default sub-reason: 0% holdout, n=6/seed × 4 seeds)
2. Hematological blocked from target_overlap LOW→MEDIUM promotion (n=1.2/seed)
3. Metabolic statin/TZD category rescue demoted MEDIUM→LOW (8.3% holdout)
4. Metabolic default sub-reason NOT demoted (32.9% holdout — at MEDIUM level)

**Holdout Impact:**
| Tier | Before | After | Delta |
|------|--------|-------|-------|
| GOLDEN | 69.9% | 70.3% | +0.4pp |
| HIGH | 58.7% | 54.5%* | -4.2pp* |
| **MEDIUM** | **30.1%** | **31.7%** | **+1.6pp** |
| LOW | 16.2% | 14.2%* | -2.0pp* |
| FILTER | 10.5% | 10.3% | -0.2pp |

*HIGH and LOW changes are seed variance — code changes only affect MEDIUM→LOW transitions.

**New Hypotheses Generated (3):**
- h554: Target overlap promotion to HIGH (45.3% within MEDIUM, P4)
- h555: MEDIUM default sub-reason deep dive (26.9%, P5)
- h556: Infectious MEDIUM precision gap (26.6%, P4)

**Recommended Next Steps:**
1. **h554**: Target overlap promotion MEDIUM→HIGH (potentially highest impact)
2. **h556**: Infectious MEDIUM precision gap analysis
3. **h550**: Antibiotic spectrum validation (overlaps with h556)

### h554: Target Overlap Promotion to HIGH — INCONCLUSIVE

target_overlap_promotion within MEDIUM has 43.0% ± 6.3% holdout (31/seed). Category-heterogeneous:
- psychiatric 53.7%, infectious 53.3% — at HIGH level
- metabolic 18.3%, autoimmune 11.9% — terrible

Best strategy (exclude worst categories): HIGH -0.1pp (neutral), 21 promoted. Net effect marginal.
All target_overlap_promotion predictions have overlap=1 (minimum). Signal is binary, not graded.
Existing deliverable annotation already allows manual prioritization. Not worth implementation complexity.

### h556: Infectious MEDIUM Precision Gap — VALIDATED

Antibiotic → viral disease mismatch identified and implemented:
- 35 predictions caught (antibiotics for influenza, HSV, CMV, smallpox, AIDS, etc.)
- Full-data: 3.3% (1/30), holdout: 5.0% (5.8/seed)
- MEDIUM +0.4pp (31.7% → 32.1%)
- Corticosteroid→infectious (12.1% holdout, 20.4/seed) NOT demoted — includes valid uses

See session summary above for cumulative impact.

---

## Previous Session: h542+h551+h552+h548+h549 - MEDIUM Quality + Gene Overlap (2026-02-06)

### h542: Deliverable Quality Audit Round 2: MEDIUM Tier Top 59 — VALIDATED

Literature validation of 59 diverse MEDIUM novel predictions against PubMed/clinical guidelines:

**Overall Results:**
| Rating | Count | % | Comparison (GOLDEN/HIGH h537) |
|--------|-------|---|------------------------------|
| VALIDATED | 15 | 25.4% | 58.0% |
| PLAUSIBLE | 18 | 30.5% | 30.0% |
| IMPLAUSIBLE | 26 | 44.1% | 12.0% |
| **Reasonable** | **33** | **55.9%** | **88.0%** |

**Error Patterns (26 implausible):**
1. **Wrong cancer type/mechanism** (7): cancer_same_type overgeneralizes (hematologic→solid, bleomycin→non-SCC)
2. **Wrong antibiotic spectrum** (6): tetracyclines for resistant Shigella, bacteriostatic for meningococcal, poor urinary excretion for pyelonephritis
3. **Wrong drug class** (6): PTU→acromegaly, phenobarbital→pain/agoraphobia
4. **Local anesthetic artifact** (5): already handled by h540
5. **Inverse indication** (1): betamethasone CAUSES adrenocortical insufficiency
6. **Non-therapeutic compound** (1): FDG PET tracer is diagnostic, not drug

**By Drug Group:**
| Group | n | Reasonable% |
|-------|---|------------|
| Corticosteroids | 6 | 83% |
| Other drugs | 9 | 67% |
| Antifungals | 7 | 57% |
| Tetracyclines | 16 | 56% |
| Cancer drugs | 15 | 53% |
| Local anesthetics | 6 | 17% |

**Fixes Implemented:**
1. **Corticosteroid→adrenocortical insufficiency inverse indication**: 6 predictions (1 HIGH + 5 MEDIUM) → FILTER. Long-acting CS cause HPA suppression. Hydrocortisone/cortisone/corticotropin preserved as legitimate replacement.
2. **Fludeoxyglucose (18F) non-therapeutic compound filter**: 55 predictions (29 MEDIUM + 20 LOW) → FILTER. PET radiotracer, not a drug.

**Holdout After Fixes:**
| Tier | Holdout | Change |
|------|---------|--------|
| GOLDEN | 69.9% ± 17.9% | 0.0pp |
| HIGH | 58.7% ± 6.1% | -0.2pp |
| MEDIUM | 29.9% ± 2.4% | -0.4pp |
| LOW | 16.2% ± 2.7% | 0.0pp |
| FILTER | 10.5% ± 1.3% | +0.2pp |

**New Hypotheses Generated (4):**
- h550: Antibiotic spectrum validation (wrong-pathogen filter) — P4, high effort
- h551: Cancer same-type hematologic vs solid drug specificity — P4, medium effort
- h552: Non-therapeutic compound audit (other diagnostic agents) — P5, low effort
- h553: MEDIUM precision by category analysis — P5, medium effort

### h551: Cancer Drug Hematologic vs Solid Specificity — INCONCLUSIVE

- Only 16 cross-type predictions (heme drug→solid cancer or vice versa)
- 0% known indication rate vs 36% for same-type
- Too small for reliable holdout measurement
- cancer_same_type holdout 27.4% is appropriate for MEDIUM tier

### h552: Non-Therapeutic Compound Audit — VALIDATED

- Found indocyanine green (diagnostic imaging dye): 10 MEDIUM preds → FILTER
- Combined with h542's FDG fix: 66 total non-therapeutic predictions removed
- No other diagnostic agents found in 1,004 unique drugs

### h548: Gene-Poor Disease kNN Quality — VALIDATED

- Gene overlap is independent of self-referentiality (r=0.047)
- Gene-poor diseases NOT more self-referential than gene-rich
- Validates gene_overlap_count as genuine molecular signal

### h549: Gene Overlap Dose-Response — INVALIDATED

- Signal is BINARY (>0 vs 0), NOT proportional to count
- kNN score actually decreases with higher overlap
- Category confound: overlap 51+ is 100% cancer
- Existing binary annotation is sufficient

**Recommended Next Steps:**
1. **h550**: Antibiotic spectrum validation (wrong-pathogen filter) — P4, high effort
2. **h532**: Every Cure GT error report (low effort, medium impact)
3. **h539**: Cancer drug class annotation (low effort)

---

## Previous Session: h408+h546+h544 - Ryland Brief + Gene Overlap + Anti-TNF Audit (2026-02-06)

### h544: Anti-TNF Paradoxical Autoimmunity Audit — VALIDATED

Literature review + VigiBase analysis of anti-TNF paradoxical effects:

**Class effects (all anti-TNF agents):**
| Condition | Evidence | Cases |
|-----------|----------|-------|
| Autoimmune hepatitis | STRONG | 389 VigiBase |
| Sarcoidosis | STRONG | 90+ cases |
| Vasculitis | STRONG | 113 cases |
| SLE | STRONG | 12,080 FAERS (h408) |
| MS/demyelination | STRONG | FDA warning |

**Drug-specific (adalimumab):**
- Polymyositis: 20 cases (MODERATE)
- Lichen planus: 21 cases (MODERATE)

**NOT paradoxical (correctly left in):** GVHD (treated by anti-TNF), TEN (86.8% response to anti-TNF), GCA (failed RCT, not harmful)

**Impact:** 15 new inverse indication pairs, 5 predictions → FILTER (2 GOLDEN + 2 MEDIUM + 1 LOW). Holdout unchanged.

---

## Previous Session: h408+h546 - Ryland Brief + Gene Overlap Signal (2026-02-06)

### h546: Drug-Target/Disease-Gene Overlap as Confidence Signal — VALIDATED

Gene overlap (shared genes between drug targets and disease-associated genes) is a strong
holdout-validated signal within every tier:

| Tier | Overlap | No Overlap | Delta |
|------|---------|------------|-------|
| GOLDEN | 81.7% | 72.0% | +9.7pp |
| HIGH | 71.9% | 61.5% | +10.4pp |
| **MEDIUM** | **57.0%** | **36.6%** | **+20.3pp** |
| LOW | 30.9% | 19.5% | +11.4pp |
| FILTER | 26.6% | 14.1% | +12.5pp |

**Confound analysis**: Signal partially inflated by known indication bias (40.9% vs 24.1%)
and disease gene count. After controlling for NOVEL-only:
- MEDIUM novel overlap: 27.1% vs 15.7% (+11.4pp) — signal persists
- Category-controlled: +5.9pp to +21.5pp — persists everywhere
- Gene-poor diseases: +16.7pp (strongest for diseases with <50 genes)

**NOT promotable**: MEDIUM novel overlap (27.1%) << HIGH (58.9%). Partially circular with kNN.
**Implemented**: `gene_overlap_count` annotation column in deliverable.

---

## Previous Session: h408 - Ryland Collaboration Brief + Anti-TNF Safety Filter (2026-02-06)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 1**
- h408: [RYLAND] Transcriptomic Validation of Top Predictions in Skin/Inflammatory Diseases - **VALIDATED**

### Key Findings

#### 1. Collaboration Brief Prepared
- **Output**: `data/analysis/h408_ryland_collaboration_brief.md` (comprehensive 5-section brief)
- **Output**: `data/analysis/h408_ryland_predictions.xlsx` (407 curated predictions with drug mechanisms, gene overlap)
- 227 novel GOLDEN/HIGH predictions across 40 derm/autoimmune diseases
- 93/407 predictions have drug-target/disease-gene overlap (molecular support)

#### 2. Corticosteroid Dominance
- **86% of novel GOLDEN/HIGH predictions are corticosteroids** — clinically valid but not novel
- Only 15 non-CS GOLDEN predictions: adalimumab (9), azathioprine (2), rituximab (2), corticotropin (2), methotrexate (1)

#### 3. Literature Validation of Top Non-CS Predictions
| Prediction | Status | Evidence |
|------------|--------|----------|
| Azathioprine → Alopecia Areata | **VALIDATED** | 10-year cohort, 92.7% regrowth |
| Corticotropin → Alopecia Areata | Mechanistic only | ACTH upregulated in lesions, no trials |
| Adalimumab → SLE | **HARMFUL** | Anti-TNF INDUCES lupus (12K FAERS reports) |
| Adalimumab → MG | **HARMFUL** | Anti-TNF CAUSES MG (case reports) |
| Adalimumab → MS | **HARMFUL** | Paradoxical demyelination |
| Adalimumab → GCA | Failed RCT | Phase 2: no benefit vs placebo |
| Etanercept → SLE | **HARMFUL** | Same drug-induced lupus class effect |

#### 4. Safety Fix: Anti-TNF Inverse Indications
Added to INVERSE_INDICATION_PAIRS in production_predictor.py:
- Adalimumab → SLE, MG, MS (3 pairs)
- Etanercept → SLE, MS (2 pairs)
- Infliximab → SLE, MS (2 pairs)
- **Impact**: 4 predictions GOLDEN/MEDIUM → FILTER
- **Holdout**: HIGH 58.9% (+0.1pp), MEDIUM 30.3% (+0.1pp)

#### 5. Collaboration Opportunities Identified
- Ryland's spatial transcriptomics can provide gene signatures for gene-poor diseases (ichthyosis=8, TEN=2, HS=2 genes)
- Drug-target database (11,656 pairs) can prioritize drugs for cell culture testing
- Azathioprine and ACTH for alopecia areata are strongest testable predictions

### Current Tier Performance (h408)
| Tier | Holdout | Predictions |
|------|---------|-------------|
| GOLDEN | 69.9% ± 17.9% | 282 |
| HIGH | 58.9% ± 6.2% | 791 |
| MEDIUM | 30.3% ± 2.5% | 2655 |
| LOW | 16.2% ± 2.7% | 3140 |
| FILTER | 10.3% ± 1.4% | 7282 |

### New Hypotheses Generated (4)
- h544: Anti-TNF paradoxical autoimmunity comprehensive audit (P4, medium)
- h545: Gene-poor disease expansion from DisGeNET/OMIM (P4, medium)
- h546: Drug-target/disease-gene overlap as confidence signal (P4, low)
- h547: Corticosteroid prediction deduplication for deliverable (P5, medium)

### Recommended Next Steps
1. **h544**: Anti-TNF paradoxical autoimmunity audit (could find more harmful predictions)
2. **h546**: Drug-target/disease-gene overlap as confidence signal (low effort, potentially useful)
3. **h542**: MEDIUM tier quality audit round 2

---

## Previous Session: h537+h540 - Deliverable Quality Audit + LA Demotion (2026-02-06)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 2**
- h537: Deliverable Quality Audit: Sample-Based Validation of Top 50 - **VALIDATED**
- h540: Local Anesthetic Non-Pain Demotion - **VALIDATED** (HIGH +0.3pp, 132 predictions demoted)

### Key Findings

#### 1. Literature Validation Results
Audited top 50 GOLDEN/HIGH novel predictions against PubMed, FDA, clinical guidelines:
- **Overall**: 29/50 (58%) VALIDATED, 15/50 (30%) PLAUSIBLE, 6/50 (12%) IMPLAUSIBLE
- **GOLDEN**: 100% reasonable (65% validated, 35% plausible, 0% implausible)
- **HIGH**: 80% reasonable (53% validated, 27% plausible, 20% implausible)

#### 2. Three Systematic Error Patterns
1. **Local Anesthetic Procedural Confusion** (2/6 errors, 27/29 GOLDEN/HIGH LA preds affected)
   - Lidocaine/bupivacaine predicted for non-pain diseases (TB, GVHD, JIA, MS, etc.)
   - Root cause: KG edges from procedural co-occurrence, not therapeutic use
   - Impact: 27 GOLDEN/HIGH predictions are artifacts
2. **Wrong Antibiotic Spectrum** (3/6 errors)
   - Erythromycin/minocycline → meningitis (poor BBB penetration)
   - Doxycycline → Pseudomonas CF (inherently resistant)
3. **Statin → Diabetes Inverse Indication** (1/6 errors)
   - Statins CAUSE diabetes (2024 Lancet meta-analysis: 10-36% increase)
   - **FIX APPLIED**: 7 statins → diabetes/hyperglycemia added to INVERSE_INDICATION_PAIRS
   - 12 predictions moved to FILTER

#### 3. Holdout Impact
Negligible — statin filter affects too few predictions to move tier averages.
| Tier | Holdout | Change |
|------|---------|--------|
| GOLDEN | 69.9% ± 17.9% | 0.0pp |
| HIGH | 58.5% ± 7.1% | 0.0pp |
| MEDIUM | 30.0% ± 2.8% | +1.2pp |
| LOW | 15.5% ± 2.7% | -0.1pp |
| FILTER | 10.3% ± 1.4% | 0.0pp |

### New Hypotheses Generated (4)
- h540: Local anesthetic non-pain demotion (P4, medium) — highest impact
- h541: Antibiotic spectrum annotation (P5, medium)
- h542: MEDIUM tier quality audit round 2 (P5, medium)
- h543: Corticosteroid prediction saturation analysis (P5, low)

#### 4. h540: Local Anesthetic Demotion (VALIDATED)
- Bupivacaine: demoted to LOW for ALL categories (no systemic therapeutic use)
- Lidocaine: demoted to LOW for non-therapeutic categories (neurological/CV/dermatological/psychiatric preserved)
- 132 predictions moved GOLDEN/HIGH/MEDIUM → LOW
- HIGH: 58.5% → 58.8% (+0.3pp holdout), MEDIUM: 30.0% → 30.2% (+0.2pp)
- `local_anesthetic_procedural` rule: 28.6% ± 4.9% holdout (GENUINE)

### Current Tier Performance (h540)
| Tier | Holdout | Predictions |
|------|---------|-------------|
| GOLDEN | 69.9% ± 17.9% | 285 |
| HIGH | 58.8% ± 6.2% | 791 |
| MEDIUM | 30.2% ± 2.4% | 2656 |
| LOW | 16.2% ± 2.7% | 3140 |
| FILTER | 10.3% ± 1.4% | 7278 |

### Recommended Next Steps
1. **h408**: Ryland collaboration prep (approaching deadline)
2. **h542**: MEDIUM tier quality audit
3. **h543**: Corticosteroid saturation analysis

---

## Previous Session: h533 - FILTER Tier Precision Audit (2026-02-06)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 1**
- h533: FILTER Tier Precision Audit - **VALIDATED** (FILTER well-calibrated, no rescue opportunity)

### Key Findings

#### 1. FILTER Tier Is Well-Calibrated
- 7,322 FILTER predictions, 10.2% ± 1.4% holdout precision
- No sub-population exceeds 15% holdout with sufficient n (>= 30/seed)
- The tier boundary is robust across all analyses

#### 2. Rank>20 Dominates FILTER (~80% of predictions)
- rank>20 predictions are reliably low-precision across ALL categories
- No category-specific rescue possible (confirms CLOSED status of rank>20 rescue)
- Best category within rank>20: respiratory 27.6% holdout — but driven by very few diseases

#### 3. Category Patterns Within FILTER
Holdout precision by category:
- respiratory: 27.6% ± 9.4% (55/seed) — best, but driven by rank>20
- cardiovascular: 20.3% ± 12.3% (117/seed) — high variance
- endocrine: 17.2% ± 5.1% (26/seed) — small n
- autoimmune: 15.2% ± 7.7% (91/seed) — borderline LOW-level
- Most other categories: 4-12% (well below LOW threshold)

#### 4. Standard Filter Sub-Reasons (Cross-Tabulation)
Only 1 sub-reason × category exceeds 15% holdout:
- **low_freq_no_mech × respiratory: 23.3% ± 9.2%** (19/seed, 22 full predictions)
  - Too few predictions to impact tier metrics
  - Mostly genuine drug-disease pairs (COPD drugs for COPD, sleep apnea drugs for OSA)
  - Holdout wildly variable (0-50% across seeds)

#### 5. TransE Consilience in FILTER (**Key Finding**)
- FILTER + TransE top-30: **16.3% ± 2.9%** holdout vs 10.0% without (+6.3pp)
- 264 full-data predictions (~53/seed)
- **Full-data shows NO signal** (16.7% vs 16.8%) — only holdout differentiates
- 16.3% ≈ LOW (15.6%) → marginal, not sufficient for tier promotion
- TransE consilience identifies better FILTER predictions but not enough to rescue

#### 6. FILTER Reason Breakdown
| Reason | Holdout | Full | n/seed |
|--------|---------|------|--------|
| standard_filter | 10.4% | 17.6% | 1213 |
| cross_domain_isolated | 10.7% | 11.3% | 65 |
| corticosteroid_iatrogenic | 39.6%* | 22.2% | 2** |
| base_to_complication | 10.0% | 37.5% | 6 |
| inverse_indication | 8.7% | 10.3% | 19 |
| cancer_no_gt | 7.1% | 5.8% | 72 |
| cancer_only_non_cancer | 3.6% | 14.3% | 8 |
| complication_non_validated | 3.4% | 21.1% | 10 |

*High variance (37.0% std), **too small for reliable measurement

#### 7. Verdict
- **FILTER tier is appropriately calibrated** — no over-filtering
- The ~755 GT hits in FILTER (10.2% × ~7400) are structural: these drugs DO treat the disease, but our model correctly identifies them as low-confidence
- TransE consilience annotated as a flag (already implemented), not promoted

### New Hypotheses Generated (4)
- h534: TransE FILTER annotation for manual review (P5, low)
- h535: FILTER category analysis — why respiratory/autoimmune perform better (P5, medium)
- h536: FILTER precision stability monitoring (P6, low)
- h537: Deliverable quality audit — sample-based validation of top 50 (P4, medium)

### Recommended Next Steps
1. **h537**: Deliverable quality audit — validate top 50 predictions against literature
2. **h408**: Ryland collaboration prep (approaching deadline)
3. **h521**: Cancer drug same-category SOC promotion

---

## Previous Session: h526/h529/h531/h257 - Inverse Indication Taxonomy + GT Audit (2026-02-06)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 4**
- h526: Drug-Induced Disease Classes: Systematic Taxonomy - **VALIDATED** (+10 new pairs, ordering bug fix)
- h529: GT Quality Audit: Remove Inverse Indication GT Entries - **VALIDATED** (-19 false GT pairs)
- h531: TCA/MAOI → Bipolar Extension - **INVALIDATED** (no predictions to filter)
- h257: IV vs Oral Formulation Safety Distinction - **INVALIDATED** (no impact on predictions)

### Key Findings

#### 1. Mechanism Taxonomy (10 classes, 135 pairs total)
Classified all inverse indication pairs into systematic mechanism classes:
- Cardiac toxicity (34): CCBs, Class Ic/III antiarrhythmics
- Metabolic disruption (28): Glucose-lowering drugs → hypoglycemia
- Steroid AEs (26): Glaucoma, osteoporosis, pancreatitis, TB, IPF
- Hormonal disruption (12): Thyroid, vitamin D, GnRH
- Immune-mediated (10): TEN/SLE/EM from NSAIDs, azathioprine
- Organ toxicity (7): Hepato/nephro/gonadotoxic
- Procarcinogenic (2→4): Estrogen → cancer
- CNS effects (2→7): SSRI/SNRI mania
- Vascular (2): COX-2 → stroke
- Bradykinin (0→3): ACEi → angioedema (NEW)

#### 2. Ten New Inverse Indication Pairs Implemented
- SSRIs/SNRIs → bipolar (5): fluoxetine, sertraline, escitalopram, venlafaxine, duloxetine
- Conjugated estrogens → breast/endometrial cancer (2): WHI carcinogenicity
- ACEi → angioedema (3): benazepril, quinapril (bradykinin class effect)
- Total: 55→63 drugs, 124→135 pairs

#### 3. Bug Fix: Inverse Indication Ordering
Moved inverse_indication check BEFORE cancer_same_type in _assign_confidence_tier.
Previously conjugated estrogens→breast cancer was getting MEDIUM (cancer_same_type)
instead of FILTER (inverse_indication). Safety filters must always come first.

#### 4. GT Quality Finding: 38 Erroneous GT Entries
38 GT entries are inverse indications (drug CAUSES the disease):
- conjugated estrogens → breast cancer (WHI: causes breast cancer)
- benazepril/quinapril → angioedema (ACEi cause angioedema)
- flecainide → cardiac arrest/MI/HF (CAST trial: 2.5x mortality)
- corticosteroids → osteoporosis/TB/IPF
Source: adverse effect/warning mentions confused with indications in data curation.

#### 5. Impact
- 7 MEDIUM → FILTER, 2 LOW → FILTER
- Holdout: unchanged (too few predictions to measure)
- Safety: 10 harmful predictions now correctly filtered

#### 6. h529: GT Quality Audit
- 38 GT entries are inverse indications; 14 from Every Cure (flagged), 24 from DRKG (removed)
- 19 unique (drug_id, disease_id) pairs removed from expanded_ground_truth.json
- FILTER precision dropped 0.2pp (16 fewer false hits)
- Key insight: DRKG associations ≠ treatments

#### 7. h531: No TCA/MAOI Bipolar Predictions
- Checked 24 antidepressants (10 TCAs, 7 MAOIs, 7 others)
- None have bipolar disorder predictions — SSRIs/SNRIs already fully covered

### New Hypotheses Generated (5)
- h529: GT quality audit (P4, completed)
- h530: Automatic inverse indication classifier (P5, high)
- h531: TCA/MAOI → bipolar expansion (P5, completed - invalidated)
- h532: Every Cure GT error report for 14 incorrect entries (P5, low)
- h533: FILTER tier precision audit for rescue opportunities (P4, medium)

### Recommended Next Steps
1. **h408**: Ryland collaboration prep (Feb 10 deadline approaching)
2. **h533**: FILTER tier precision audit — ~755 correct predictions may be recoverable
3. **h530**: Automatic inverse indication classifier (high effort, longer term)

---

## Previous Session: h486 + h525 - SIDER Mining + GT Expansion (2026-02-06)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 6**
- h486: Drug-Induced Disease Filter: Systematic Adverse Effect Mining - **VALIDATED**
- h525: SIDER Indication-Based GT Expansion - **VALIDATED** (+51 GT pairs, HIGH +1.2pp)
- h527: Corticosteroid Iatrogenic Audit - **VALIDATED** (0 unfiltered, already comprehensive)
- h528: NSAID Iatrogenic Audit - **VALIDATED** (+1 celecoxib→ischemic stroke)
- h519: CV Pathway-Comprehensive Re-evaluation - **VALIDATED** (40.8% holdout, MEDIUM confirmed)
- h523: Anticoagulant SOC Signal - **INCONCLUSIVE** (n=10/seed too small)

### Key Findings

#### 1. SIDER Mining Requires Strict Matching
- Original loose substring matching: 1,462 candidates — 80%+ false positives
- Strict matching + SIDER indication exclusion + manual audit: 307 → 47 genuine pairs
- False positive sources: generic AE terms ("ulcer" matching "ulcerative colitis"), drugs that TREAT the disease

#### 2. 47 New Inverse Indication Pairs Implemented
- 20 new drugs added to INVERSE_INDICATION_PAIRS (35 → 55 drugs total, 77 → 124 pairs)
- Key categories:
  - Corticosteroid iatrogenic: TB reactivation, glaucoma, osteoporosis, MG crisis
  - NSAID: TEN (Stevens-Johnson), drug-induced SLE, peptic ulcer, stroke (COX-2)
  - Estradiol: endometrial/uterine cancer, hereditary angioedema
  - Proarrhythmic: ibutilide/dofetilide/milrinone → VT
  - Immunosuppressant: azathioprine → TEN, hepatitis B reactivation
  - Metabolic: paricalcitol → hypoparathyroidism

#### 3. Safety Impact
- ~105 predictions now filtered by inverse indication rules
- 98 GT negatives correctly filtered, 7 GT positives filtered (medically justified)
- Filter precision: 93.3%

#### 4. Holdout Impact (vs h520/h522 baseline)
| Tier | Previous | Current | Delta |
|------|----------|---------|-------|
| GOLDEN | 62.6% ± 8.1% | 69.9% ± 17.9% | +7.3pp |
| HIGH | 53.8% ± 2.6% | 57.3% ± 8.1% | +3.5pp |
| MEDIUM | 31.3% ± 1.4% | 28.8% ± 2.6% | -2.5pp (NS) |
| LOW | 14.2% ± 0.5% | 15.6% ± 2.6% | +1.4pp |
| FILTER | 9.7% ± 0.6% | 10.5% ± 1.4% | +0.8pp |

Note: Comparison imprecise due to accumulated code changes since last baseline.

### Corrections Applied During Session
- Removed lidocaine → VT (lidocaine is Class Ib antiarrhythmic that TREATS VT)
- Removed azathioprine → interstitial pneumonia (azathioprine treats underlying myositis)

### New Hypotheses Generated (4)
- h525: SIDER indication-based GT expansion (P4, medium)
- h526: Drug-induced disease class taxonomy (P4, medium)
- h527: Systematic corticosteroid iatrogenic filter expansion (P5, low)
- h528: Systematic NSAID inverse indication expansion (P5, low)

#### 5. SIDER GT Expansion (h525)
- Used NLP_indication labels only (not text_mention/NLP_precondition)
- 153 exact-match candidates → 51 genuine missing GT pairs after audit
- Key additions: 15 ACEi/ARB/statin/beta-blocker → ACS, anticoagulants → VTE, Sildenafil → PAH
- HIGH: 57.3% → 58.5% (+1.2pp holdout)
- SIDER indications are ~66% noise even for NLP_indication

### Recommended Next Steps
1. **h526**: Classify inverse indications by mechanism for systematic expansion
2. **h527**: Systematic corticosteroid iatrogenic filter expansion
3. **h257**: IV vs oral formulation safety distinction

---

## Previous Session: h520 - SOC Drug Class Precision Heterogeneity (2026-02-06)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 1**
- h520: SOC Drug Class Precision Heterogeneity - **VALIDATED** (corticosteroid promotion implemented)

### Key Findings

#### 1. SOC Signal is Driven by Corticosteroids (h520)
- Per-class holdout analysis across all 17 SOC drug classes in MEDIUM tier
- **Corticosteroids**: 46.1% holdout (n=96/seed, p=0.0065) — dominant signal
- **Cancer drugs**: 34.0% holdout (n=63/seed, p=0.25) — not significant
- All other classes: tiny-n (<10/seed), not individually actionable

#### 2. Category Breakdown for Corticosteroid MEDIUM
- **Dermatological**: 58.0% holdout (strong)
- **Respiratory**: 61.1% holdout (strong, small n=9)
- **Autoimmune**: 45.5% holdout (solid)
- **Ophthalmic**: 34.2% holdout (moderate)
- **Hematological**: 19.1% holdout (weak — excluded from promotion)

#### 3. Promotion Implemented: Corticosteroid MEDIUM → HIGH
- Non-hematological corticosteroid MEDIUM predictions promoted to HIGH
- 333 predictions moved
- **HIGH**: 51.5% → 53.8% (+2.3pp)
- **MEDIUM**: 29.9% → 31.1% (+1.2pp)
- Both tiers improved — clean win
- Code: `_CORTICOSTEROID_SOC_PROMOTE_CATEGORIES` + `_CORTICOSTEROID_LOWER` in production_predictor.py

### New Hypotheses Generated (4)
- h521: Cancer drug same-category SOC promotion (P4, medium)
- h522: Hematological corticosteroid demotion MEDIUM→LOW (P4, low)
- h523: Anticoagulant SOC signal in LOW tier (P5, low)
- h524: DMARD SOC signal across tiers (P5, medium)

### Recommended Next Steps
1. **h522**: Demote hematological corticosteroid MEDIUM→LOW (quick, likely +0.5pp MEDIUM)
2. **h521**: Investigate cancer_drugs MEDIUM stratification by cancer subtype
3. **h486**: Systematic adverse effect mining from SIDER (high effort but high safety impact)

---

## Previous Session: h508/h481/h518/h516 - Self-Ref + Literature Status + SOC Holdout (2026-02-06)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 4**
- h508: Self-Referential Disease Characterization - **VALIDATED** (GT size is dominant predictor)
- h481: Deliverable Literature Validation Status Column - **VALIDATED** (+28.4pp precision for SOC)
- h518: SOC Status as Holdout Precision Signal - **VALIDATED** (annotation only, not promotable)
- h516: Expand SOC Drug Class Mappings - **INVALIDATED** (0% precision for all 7 proposed)

### Key Findings

#### 1. Self-Referentiality Characterization (h508)
- **GT size is the DOMINANT predictor**: 79.2% of self-ref diseases have GT ≤ 2 (OR=8.6x)
- Category modulates: GI/immunological 89-100% vs autoimmune/dermatological 20-33%
- 11 "Therapeutic Islands" (GT>5, 100% self-ref): immunodeficiency, PAH, CKD, HepC, opioid constipation
- Two distinct causes: small GT (79%) and dedicated drug classes (8%)

#### 2. Literature Status Classification (h481)
- Added `literature_status` + `soc_drug_class` columns to deliverable
- 17 drug class SOC mappings (184 drugs), 1,651 as LIKELY_GT_GAP (11.7%)
- Full-data: SOC +28.4pp vs NOVEL in HIGH tier (40.3% vs 11.9%)
- Also regenerated JSON deliverable (was stale from old script)

#### 3. SOC Holdout Validation (h518)
- MEDIUM: SOC 20.3% vs NOVEL 14.3% (+6.0pp, p=0.005)
- HIGH: SOC 25.5% ≈ NOVEL 25.7% (NO SIGNAL)
- MEDIUM SOC (20.3%) << HIGH avg (51.5%) → NOT promotable
- **Conclusion**: SOC is ANNOTATION signal, not tier promotion signal

#### 4. SOC Expansion Fails (h516)
- All 7 proposed new drug classes have 0% precision
- Tetracyclines→infectious: 0/248 (0%), macrolides: 0/129, etc.
- Current 17 SOC classes are well-calibrated; expansion dilutes signal
- **Insight**: SOC captures BROAD classes (corticosteroids, statins), not specific ones

### New Hypotheses Generated (5)
- h516: INVALIDATED
- h517: Therapeutic island annotation (P5, low)
- h518: VALIDATED as annotation
- h519: CV pathway holdout re-evaluation (P5, low)
- h520: SOC class-specific holdout precision (P4, medium)

### Recommended Next Steps
1. **h520**: Which SOC drug classes drive the +6pp MEDIUM signal? Could identify class-specific promotions
2. **h517**: Annotate therapeutic islands in deliverable
3. **h486**: Systematic adverse effect mining from SIDER database (high effort but high impact safety)

---

## Previous Session: h507/h492/h509/h515 - Self-Referentiality + GT Expansion + Baseline Re-Calibration (2026-02-06)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 5**
- h507: Predictable Self-Referentiality - **INVALIDATED** (GT-free features reverse on holdout)
- h492: GT Expansion for FDA-Approved Pairs - **VALIDATED** (15 pairs added, negligible impact)
- h509: CV Coronary Hierarchy Demotion - **INVALIDATED** (small-n, do not demote)
- h509 extended: HIGH Tier Per-Rule Holdout Audit - **VALIDATED** (identified underperforming rules)
- h515: Diabetes Hierarchy Split - **INVALIDATED** (only 8 complication predictions)

### Key Findings

#### 1. Self-Referentiality is NOT Predictable (h507)
- Best GT-free feature: same_cat_frac (AUC=0.734 on full data)
- Combined features: AUC=0.781
- **REVERSES on holdout**: -6.0pp ± 12.9pp gap (inconsistent direction)
- Root cause: low same_cat_frac captures TWO opposite populations
  - 58% truly self-referential (bad on holdout)
  - 42% genuine cross-category transfer (good on holdout)
- No GT-free feature can separate them

#### 2. GT is Already Complete (h492)
- Audited 20 major diseases across 7 categories (167 pairs checked)
- Only 15 FDA-approved pairs missing
- 14/15 NOT in model's top-30 predictions
- **GT incompleteness is NOT the precision bottleneck**
- Model limitations are structural (kNN coverage)

#### 3. Holdout Baseline Has Drifted (h492 discovery)
| Tier | Previous (CLAUDE.md) | Current (h492 re-baseline) | Delta |
|------|---------------------|---------------------------|-------|
| GOLDEN | 67.0% ± 20.6% | 63.3% ± 23.2% | -3.7pp |
| HIGH | 60.8% ± 7.2% | 51.5% ± 5.3% | -9.3pp |
| MEDIUM | 32.1% ± 3.6% | 29.9% ± 2.8% | -2.2pp |
| LOW | 12.9% ± 1.4% | 12.3% ± 1.4% | -0.6pp |
| FILTER | 10.3% ± 1.1% | 8.9% ± 1.1% | -1.4pp |

Drift caused by accumulated code changes since h478 (69 insertions, 45 deletions).

#### 4. CV Hierarchy Groups All Too Small (h509)
- Coronary: 5 diseases, arrhythmia: 3, hypertension: 4
- Full-data precision high (65-93%) but holdout unreliable (n≈1/seed)
- DO NOT demote — these encode genuine medical knowledge

#### 5. HIGH Tier Per-Rule Audit (h509 extended)
- Several rules underperforming: respiratory (20%), thyroid (26%), diabetes (21%)
- But most have n<15 across 5 seeds → too small for reliable demotion
- Diabetes hierarchy: only 8 complication predictions → not worth splitting (h515)

### New Hypotheses Generated (5)
- h510: Cross-Category Transfer Disease Identification (P5, low)
- h511: Embedding Norm as Disease Confidence Annotation (P5, low)
- h512: HPO/Gene External Similarity for Self-Referential Diseases (P3, high)
- h513: Periodic Holdout Re-Baseline Policy (P4, low)
- h514: Migraine Drug Coverage Gap Analysis (P5, low)
- h515: Diabetes Hierarchy Split [COMPLETED - INVALIDATED]

### Recommended Next Steps
1. **h512:** HPO phenotype similarity as alternative for 144 self-referential diseases (high impact, high effort)
2. **h481:** Deliverable annotation with literature validation status (medium impact, medium effort)
3. **h257:** IV vs oral formulation safety distinction (medium impact, medium effort)

---

## Previous Session: h490/h504/h503/h505 - CV Gap + Self-Referential + Seed Analysis (2026-02-06)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypotheses Tested: 4**
- h490: CV ATC Coherent Full-to-Holdout Gap - **VALIDATED** (CV standard MEDIUM→LOW, +0.4pp)
- h504: Self-Referential Disease Analysis - **VALIDATED** (31.6% diseases are 100% self-ref)
- h503: Seed 42 Failure Mode - **VALIDATED** (sampling variance, no fix needed)
- h505: CV Target Overlap Rescue Block - **VALIDATED** (56 preds MEDIUM→LOW)

### Combined Impact (h490 + h505)
| Tier | Before | After | Change |
|------|--------|-------|--------|
| GOLDEN | 68.3% | 68.3% | 0.0pp |
| HIGH | 55.3% | 55.3% | 0.0pp |
| **MEDIUM** | **31.7%** | **32.1%** | **+0.4pp** |
| LOW | 12.0% | 12.9% | +0.9pp |
| FILTER | 10.3% | 10.3% | 0.0pp |

170 predictions moved MEDIUM→LOW. Tier counts: MEDIUM 3566→3396, LOW 2341→2511.

---

[Earlier sessions: see git history]

---

## Current Session: h959 - predict() Call-Site Audit (2026-04-19)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypothesis Tested: h959 (VALIDATED)**

### Key Findings

**h952 find_disease_id bug had broad reach but the deliverable is safe:**

1. **Name-resolution coverage:** 287/1146 disease_names (25.0%) failed pre-fix; 0 fail post-fix.
2. **h393 evaluator impact:** 213/1011 (21.1%) of the eval pool was silently zero-predicted pre-fix. Extrapolates to ~43/202 per-seed holdout fails — matches h952's seed-42 observation of 40/203 within sampling error.
3. **Call-site distribution:** 111 of 112 scripts pass `disease_name` to `predictor.predict()`; only `h771_literature_coverage_analysis.py` passes `disease_id` (and that path actually fails silently because `find_disease_id` cannot resolve IDs).
4. **Deliverable NOT affected:** `scripts/generate_production_deliverable.py` iterates `train_disease_list` (disease_ids) and calls its own `knn_predict(disease_id, …)`, bypassing `predictor.predict()` entirely.
5. **h939/h940 NOT affected:** Both operate directly on disease embeddings by ID — their pure kNN numbers (bio_r30=31.42%, overall_r30=20.30%) are trustworthy.
6. **h904/h908/pre-h952 h393 results:** Directionally valid (tier-ordering preserved under ~uniform 3pp recall suppression) but magnitudes dampened.

### Bug-Impact Table

| Experiment | Affected? | Severity |
|---|---|---|
| 13,416-row XLSX | NO | Uses disease_id path |
| h939 biologic target-overlap | NO | Direct kNN bypasses predict() |
| h940 biologic fusion | NO | Direct kNN bypasses predict() |
| h958 post-fix 5-seed | NO | Ran with h952 fix |
| h908 MeSH C23 blocklist | NO | Classification-only |
| h904 rule tier demotions | YES (magnitude) | via h393; direction preserved |
| h951 biologic baseline | YES (magnitude) | surface-case of the bug |
| Most h393-derived tier precisions (pre-h952) | YES (magnitude) | ~3pp recall suppression |

### New Hypotheses Generated (3)
- h963 (P2): predict(disease_id) fast-path to bypass find_disease_id
- h964 (P2): Re-run h904/h908 tier precisions post-fix; update CLAUDE.md magnitudes if |Δ|>1pp
- h965 (P3): Regression test for silent-zero predict() outputs

### Recommended Next Steps
1. **h954** (P1): Reconcile 41.8% historical overall R@30 vs h951 16.39% — eval framework drift audit
2. **h957** (P1): h949 zero-overlap biologic safety filter — precision pivot
3. **h964** (P2): Re-run h904/h908 post-fix to validate quoted tier precisions

---

## Current Session: h954 - Baseline reconciliation (41.8% vs h951 16.39%) (2026-04-19)

### Session Summary

**Status:** Complete
**Hypothesis: h954 (VALIDATED)**

### Key Finding: Not a regression — an eval-framework mismatch

The 41.8% R@30 baseline cited in research_spec.md and RESEARCH_ROADMAP.md came from
`scripts/evaluate_hard_negatives_v2.py:evaluate_recall_at_k` (first reported in
`docs/archive/detailed_analysis_findings.md` 2026-01-25). It is:

- **Model:** GB-enhanced classifier (`drug_repurposing_gb_enhanced.pkl`) scoring drug×disease pair features (concat + product + diff)
- **Metric:** MICRO-averaged `total_hits / total_gt_drugs` across ALL test pairs
- **GT:** Every Cure only, 3,618 positive pairs / 442 unique diseases (1,236 after fuzzy mapping)
- **Split:** static train/test

h951/h958 production numbers (16.39% / 19.49%) are:

- **Model:** `production_predictor.predict()` — kNN(k=20) + tier-override rules
- **Metric:** MACRO-averaged per-disease R@30 (mean of `hits/gt_drugs` per disease)
- **GT:** expanded GT, 57,495 pairs / 1011 diseases
- **Split:** 5-seed 80/20 disease holdout

The 25pp gap is not a regression — it's a methodology mismatch.

### Documents updated
- `research_loop/prompts/research_spec.md` — "Current Baseline" section rewritten with labeled table of fair baselines and a clear "DO NOT cite 41.8% as production comparison" warning
- `research_loop/prompts/initializer_prompt.md` — `baseline_metric` updated to "19.49% per-disease R@30 (h958 production, post-h952 fix)"
- CLAUDE.md already had the fair baselines in "Key Metrics"; no changes needed there

### Drift audit
- **0 pending hypotheses** reference 41.8% in rationales or success criteria (grepped roadmap)
- Drift was contained to top-level project docs only

### New Hypotheses Generated (2)
- h966 (P3): Re-evaluate GB-enhanced model on production eval framework (apples-to-apples)
- h967 (P3): Document micro-vs-macro R@30 pitfall as CLAUDE.md metric-hygiene subsection

### Recommended Next Steps
1. **h957** (P1): h949 zero-overlap biologic safety filter — only remaining P1
2. **h964** (P2): Re-run h904/h908 post-fix to validate quoted tier precisions
3. **h960** (P2): Neurological supplement + SELECTIVE_BOOST interaction diagnosis

---

## Current Session: h953 — Biologic Precision Pivot (INVALIDATED) (2026-04-19)

**Status:** Complete | **Hypothesis:** h953 (INVALIDATED)

### What was tested
For each holdout disease, compute biologic_prior[category] = fraction of train diseases in that category with ≥1 biologic in expanded GT. If a holdout disease's category prior < threshold, drop biologics from its prediction list and refill top-30 from non-biologic ranks. Sweep thresholds {0.05, 0.10, 0.15, 0.20, 0.30}; 5-seed h393 holdout; top_n=200 fetched, top-30 evaluated.

### Headline result
| metric | baseline | best (any thr) | gate |
|---|---|---|---|
| p30 | 14.32% ± 0.94% | +0.05pp (thr=0.30) | +2.0pp required |
| r30 | 19.79% ± 1.59% | -0.13pp (thr=0.15) | – |
| bio_r30 | 30.31% ± 3.57% | -0.18pp (thr=0.10), -8.16pp (thr=0.30) | ≥ -5pp required |

**DO NOT SHIP — no threshold meets the +2pp p30 / ≤5pp bio_r30 gate.**

### Why it failed (3 reasons)
1. kNN already concentrates biologics in immunology/cancer/endocrine — few biologics are misplaced into low-prior categories to demote.
2. Rank-31+ non-biologic replacements hit at the same background rate as the demoted biologic, so the swap is precision-neutral.
3. Category-level prior is too coarse: dermatological (prior=0.183) has bio_r30=72.2%, so any demotion there destroys real biologic hits.

### Side finding
**Baseline bio_r30 (30.31%) > overall_r30 (19.79%) by +10.5pp** — biologics OVERPERFORM. The h906/h920/h921/h924 "biologic failure" narrative was already dissolved in h951; h953 reconfirms with explicit p30/r30 split. Precision lift for the deliverable must come from SMALL-MOLECULE slots, not biologic slots. Filed h991 to test that direction directly.

### New hypotheses (3 added)
- **h990 (P2):** Per-disease kNN-neighbor biologic prior (replace category-level prior with neighbor-derived prior — finer-grained signal that matches kNN's actual scoring mechanism).
- **h991 (P2):** Inverse pivot — small-molecule precision gap. Stratify p30 by sm vs bio rank-position per category; identify SM-precision-deficit categories for targeted SM demotion.
- **h992 (P3):** p30 ceiling = min(|GT|, 30) / 30 — reframe baseline 14.32% as fraction-of-achievable. Likely shows current p30 is closer to ceiling than it appears, bounding precision-pivot ambition.

### Recommended next hypothesis
**h990 (P2)** — quick to implement (reuse h953 plumbing, swap category prior for neighbor-derived prior). Most likely to produce a different result, since neighbor-derived priors directly diagnose whether biologics in top-30 are mechanism-supported.


## Current Session (continued): h990 — Per-Disease Neighbor Bio Prior (INVALIDATED) (2026-04-19)

**Status:** Complete | **Hypothesis:** h990 (INVALIDATED — closes the biologic in/out-shuffle family)

### What was tested
Replace h953's per-CATEGORY biologic prior with a per-DISEASE prior derived from the actual kNN signal: `prior(D) = (#{top-20 train kNN neighbors of D with ≥1 biologic in expanded GT}) / 20`. Sweep thresholds {0.00, 0.05, 0.10, 0.20}; demote biologics in top-30 for diseases below threshold; refill from rank 31+.

### Headline result
| metric | baseline | thr=0.00 (strict) | thr=0.10 | thr=0.20 | gate |
|---|---|---|---|---|---|
| p30 | 14.32% ± 0.94% | +0.00pp (avg 19 demoted) | -0.00pp (avg 43 demoted) | +0.01pp (avg 99 demoted) | +1.0pp |
| bio_r30 | 30.31% ± 3.57% | **+0.00pp** | -1.37pp | -6.94pp | ≥-3pp |

**DO NOT SHIP — no threshold meets the +1pp p30 / ≤3pp bio_r30 gate.**

### Crucial structural finding
**Strict prior=0.00 demotes 19 diseases per seed but bio_r30 doesn't move at all.** This proves the kNN structural invariant: `drug_scores` only sums over neighbor GT, so if ZERO top-20 neighbors have any biologic in GT, ZERO biologics can appear in top-N from kNN. The strict filter is a safe no-op.

### Joint h953 + h990 closure
Removing top-30 biologics and refilling from rank 31+ is FUNDAMENTALLY precision-neutral because the kNN rank 31-200 tail is uniform-quality. The "biologic in/out shuffle" precision-pivot family (h953, h990, conceptually h957's safety-filter variant too) is closed.

### New hypotheses (2 added)
- **h993 (P3):** Per-rank hit-rate audit — formalize the tail-uniformity claim by computing p(hit | rank=r) for r ∈ [1, 200].
- **h994 (P2):** In-window re-ranking via auxiliary features (target overlap, literature, ATC) — lift hits to higher ranks within top-30 for tier-precision gains. Most promising remaining biologic-precision direction.

### Recommended next hypothesis
**h994 (P2)** — operates inside top-30 (the only rank window where signal differentiation appears to exist), targets tier precision (the metric h964 actually cares about), and complements the h979/h986 tier work.



## Current Session (continued): h991 — SM Precision Gap (INVALIDATED but diagnostic) (2026-04-19)

**Status:** Complete | **Hypothesis:** h991 (INVALIDATED — framing backwards)

### What was tested
Stratified the h953 baseline p30=14.32% by drug-type within top-30. Computed `sm_p30|sm = sm_hits / n_sm_in_top30` and `bio_p30|bio = bio_hits / n_bio_in_top30` per disease, pooled across 5 seeds (h393 holdout, n≈200 holdout diseases/seed). Flagged any category with `sm_p30|sm ≤ bio_p30|bio - 5pp` for targeted SM demotion (per h991 spec).

### Headline result — direction INVERTED
| metric | value |
|---|---|
| sm_p30\|sm | **15.48%** |
| bio_p30\|bio | **5.64%** |
| gap (bio - sm) | **−9.85pp** (SM OUTPERFORMS bio slot-for-slot) |
| categories with sm < bio (n≥50, bio_n≥10) | **0** |
| categories with bio < sm | **17 of 18** (only endocrine n=6 inverts) |

The h991 framing assumed biologics outperform SM slot-for-slot; actually the opposite is true. The h953 "bio_r30 > overall_r30 by +10pp" signal was a MACRO-averaging artifact: small bio_gt per disease (often 1–2 drugs) means each bio hit contributes disproportionately to the per-disease r30 mean.

### Crucial side finding (diagnostic payoff)
SM top-30 slot hit rate by `(train_frequency, mechanism_support)` cross-tab (pooled 26,564 SM slots):

| freq bucket | mech | n slots | hit_rate |
|---|---|---|---|
| f=1 | False | 2138 | 7.86% |
| f=2 | False | 2645 | 7.33% |
| f=3-4 | False | 4089 | 8.36% |
| f=5-9 | False | 5208 | 9.62% |
| f≥10 | False | 8110 | 16.95% |
| f=1 | True  | 333 | 24.62% |
| f≥10 | True | 1359 | **46.06%** |

**6× precision range within SM top-30.** The ~4,783 SM slots with `f≤2 AND no mech` (18% of SM top-30 volume) hit at 7.57% — below current LOW tier (10.0%) and comparable to FILTER (6.8%). This is the actionable signal surfaced by h991 even though the framing failed: an explicit tier rule that demotes these slots should lift MEDIUM precision without regression elsewhere.

### Why the h991 step-4 ship test was not run
The decision rule `sm_p30|sm ≤ bio_p30|bio - 5pp` yielded zero qualifying categories, so the proposed SM demote rule had nothing to demote. Step 4 (ship gate measurement) was skipped because step 2 (qualifying categories) returned empty.

### New hypotheses (3 added)
- **h995 (P2):** Autoimmune biologic-mis-selection audit — in autoimmune (n=29), sm_p30|sm=39.15% but bio_p30|bio=13.91% (−25.2pp). Autoimmune is THE biologic-treatment category (anti-TNF, anti-IL6/17/23, anti-CD20); 13.9% means kNN is surfacing wrong biologic families. If confirmed, USAN-family-match filter could lift bio_p30 without global bio_r30 hit.
- **h996 (P2):** SM low-freq no-mech tier demote — explicit `SM AND f≤2 AND no mech → cap at LOW` rule. Pooled hit rate 7.57% on ~4,783 slots; expect MEDIUM +0.5–1.5pp.
- **h997 (P3):** Micro vs macro r30 — the h951/h953 bio_r30 headline is macro-inflated. Recompute micro-averaged r30 from h991 raw records; predict the +10pp bio>overall gap largely collapses.

### Recommended next hypothesis
**h996 (P2)** — small, self-contained tier-rule change with clean ship gate (MEDIUM +0.5pp, LOW drop ≤1pp). The freq×mech gradient is the strongest slot-level precision signal surfaced in the h953/h990/h991 trilogy.


## Current Session (continued): h996 — SM Low-Freq Tier Demote (INVALIDATED at precondition) (2026-04-19)

**Status:** Complete | **Hypothesis:** h996 (INVALIDATED — existing tier system already handles the signal)

### Precondition check
Before implementing the proposed `SM AND f≤2 AND NOT mech → cap LOW` rule, measured current tier distribution of target slots (5-seed h393, top-30, include_filtered).

| tier | target n | target hits | target hit_rate |
|---|---|---|---|
| GOLDEN | 0 | 0 | n/a |
| HIGH | 35 | 33 | **94.29%** |
| MEDIUM | 0 | 0 | n/a |
| LOW | 86 | 3 | 3.49% |
| FILTER | 4,619 | 278 | 6.02% |
| **total** | **4,740** | **314** | 6.62% |

**99.3% of SM f≤2 no-mech slots are already at LOW or FILTER.** The proposed rule would:
- Destroy 33 legitimate HIGH hits (94% precision survivors — explicit promotion rules override the default)
- Demote 0 MEDIUM slots (zero MEDIUM lift possible)
- Provide no signal to FILTER/LOW (already there)

### Key insight
The h991 7.57% "aggregate hit rate" for f≤2 no-mech SM slots was a cross-TIER average, dominated by the 4,619 FILTER slots (6.02%). Within HIGH tier, surface features (freq, mech) do NOT predict quality because explicit promotion rules (hierarchy, comp_to_base, cancer_same_type, TransE high promotion, etc.) beat the default heuristic — correctly — surfacing genuine 94% precision preds.

**Methodology lesson:** before adding a tier-rule based on a slot-level signal, always measure the tier distribution of target slots. If most already sit at the target tier (LOW/FILTER here), the rule is a no-op or worse.

### New hypotheses (2 added)
- **h998 (P3):** Identify which sub_reason paths promote the 35 HIGH survivors at 94.3% precision — if a single path dominates, it validates a pre-existing rule; if multiple weak paths composite, may indicate a new promotion target.
- **h999 (P3):** LOW tier SM f≤2 no-mech sub-population at 3.49% (vs LOW avg 10%) — small-volume LOW→FILTER demote candidate; test only if h998 yields clean paths.

### Recommended next hypothesis
**h994 (P2)** — in-window re-ranking (the only remaining P2 precision pivot that isn't closed by h953/h990/h991/h996). Or **h957/h965 follow-ups** if further biologic-tier work surfaces.


## Current Session: h1000 — In-Window Biologic Re-Rank (INVALIDATED at ship gate) (2026-04-19)

**Status:** Complete | **Hypothesis:** h1000 (INVALIDATED — magnitude too weak)

### What was tested
Implemented score-perturbation in-window re-rank: for each biologic in top-30 of each holdout disease, compute target_match_loo using k=3 kNN neighbor-aug bio_gt (inference-safe), then shift its sort key by (−1.5 if match else +1.5). Re-sort the 30-pred list, re-assign ranks 1..30, recompute tier via `_assign_confidence_tier` + replay of 9 tier-dependent inline mutations (target_overlap promotion, CS demotions, TransE MEDIUM→HIGH, literature strong/moderate/high promotions and weak demotion). Non-biologic ranks shift passively via swap with biologic neighbors.

### Headline result
| tier | baseline mean±std | shifted mean±std | Δ |
|---|---|---|---|
| GOLDEN | 82.58 ± 4.50 | 82.32 ± 4.67 | **−0.26pp** |
| HIGH | 80.44 ± 3.99 | 80.11 ± 3.93 | **−0.33pp** |
| MEDIUM | 39.89 ± 4.99 | 39.40 ± 4.90 | **−0.49pp** |
| LOW | 9.96 ± 0.83 | 9.91 ± 0.84 | −0.04pp |
| FILTER | 6.78 ± 0.81 | 6.76 ± 0.80 | −0.02pp |

Ship gate (bio_r30 drop ≤0.5pp AND ≥1 tier lift ≥1pp AND no tier drop ≥1pp): **bio_r30 preserved (0.00pp drop by construction)** but **zero tier lifts ≥1pp** → **SHIP FAIL**.

### Mechanistic diagnosis
Only 60 of 3,570 biologic slots (1.7%) crossed a tier boundary across 5 seeds. Rank-based tier rules sit at thresholds {5, 10, 15, 20}; a ±1 rank shift only crosses a boundary when the biologic's baseline rank is at r∈{5, 6, 10, 11, 15, 16, 20, 21}. Most biologics don't sit in those 8 positions.

Also: 78% of biologics are target_match_loo_neighbor=False (neighbor bio_gt too narrow universe, per h1002 — autoimmune structural floor generalizes). So the majority of biologics are demoted +1 rank, but this rarely moves the tier.

**Biologic tier moves (5 seeds pooled):**
- FILTER→GOLDEN: 1 (1/1 hit, 100%)
- FILTER→LOW: 6 (0 hits)
- FILTER→MEDIUM: 1 (1/1 hit, 100%)
- LOW→FILTER: 44 (2 hits, 4.5%) ← dominant move
- LOW→HIGH: 1 (1/1 hit)
- LOW→MEDIUM: 4 (1 hit, 25%)
- MEDIUM→FILTER: 3 (1 hit, 33%)

### Why h1000 is worth closing at −0.5pp rather than "inconclusive"
Per h1002/h995b structural-floor analysis, the neighbor target-match signal is too sparse on the majority of biologic slots. Even if we scaled ±1 to ±5, the demotion side (78% of bio) would move the wrong direction on target-unique first-in-family biologic hits. Score-perturbation mechanism is safe (bio_r30 preserved exactly) but magnitude is physically limited — the tier rule system samples rank at discrete thresholds, not continuously.

### New hypotheses (4 added)
- **h1005 (P3):** Larger rank-shift magnitude (±3, ±5) — tests whether more boundary crossings deliver a lift, accepting more non-bio disruption.
- **h1006 (P3):** Boundary-targeted rank shift — only move biologics within ±2 of rank 5/10/15/20 (surgical shift with 100% on-boundary rate).
- **h1007 (P3):** Audit the 44 biologic LOW→FILTER demotes — legitimate low-value or GT gaps? Decides whether the demote surface has annotation value.
- **h1008 (P3):** Per-category adaptive magnitude — ±1 for autoimmune (unique-target structural floor per h1002), ±3 for cancer/cv/hematological (10-24x LOO ratios per h995b).

### Recommended next hypothesis
**h1005** (P3) — cheapest follow-up, same script with `RANK_SHIFT_MAGNITUDE=3.0`. If it still doesn't lift, close the ±N score-perturbation family entirely and escalate to boundary-targeted (h1006) or annotation-only (h1003/h1007) surfaces. If it DOES lift, run h1006 to see if a surgical version preserves the lift at half the collateral damage.


## Current Session (continued): h1005 — ±3.5 Rank-Shift Magnitude (INVALIDATED — closes family) (2026-04-19)

**Status:** Complete | **Hypothesis:** h1005 (INVALIDATED)

### What was tested
Re-ran h1000 script with `RANK_SHIFT_MAGNITUDE=3.5` to test if larger shifts move the needle (h1000 used 1.5, i.e. ±1 integer rank).

### Result
| tier | baseline | shifted (±3.5) | Δ |
|---|---|---|---|
| GOLDEN | 82.58 | 82.41 | **−0.17pp** |
| HIGH | 80.44 | 80.07 | **−0.37pp** |
| MEDIUM | 39.89 | 39.40 | **−0.49pp** |
| LOW | 9.96 | 9.90 | −0.06pp |
| FILTER | 6.78 | 6.76 | −0.02pp |

Same SHIP FAIL pattern as h1000. Biologic movement amplified:
- LOW→FILTER: 44→153 demotes (3.5x at magnitude 2.3x)
- FILTER→GOLDEN+HIGH: 1→4 promotions (all 100% hit rate individually)

### Mechanism: magnitude is not the bottleneck — signal density is
- 22% of biologics are target_match_loo_neighbor=True (bump candidates)
- 78% are False (demote candidates, most already at LOW/FILTER heading to FILTER)
- High-precision match=True promotions are numerically tiny (4-6 hits across 5 seeds) — cannot lift a 53-slot GOLDEN or 213-slot HIGH
- Match=False demotions work as designed (LOW→FILTER legitimately tracks rank>20 rule at 4.6% hit rate) but cannot LIFT any tier

### Score-perturbation family CLOSED
Both magnitudes invalidated. Score-perturbation re-rank mechanism is mechanically safe (bio_r30 preserved exactly by top-30 membership invariant) but cannot lift tier precision because the bump side signal is too sparse.

Remaining viable surfaces:
- **h1006** (boundary-targeted): only shift biologics within ±2 of rank thresholds (5/10/15/20) → 100% boundary-crossing rate.
- **h1008** (per-category adaptive): exploit 10-24x LOO ratios in cancer/cv/hematological per h995b.
- **h1003/h1007** (annotation-only): surface match_loo as deliverable column, don't touch tier.

### Recommended next hypothesis
**h1006** (P3) — surgical version of h1000 that concentrates the shift effect at tier boundaries. If it fails too, the entire in-window re-rank family is closed and the biologic precision problem is STRUCTURALLY unsolvable via rank manipulation.


## Current Session (continued): h1100 — FILTER Known-Indication Carve-Outs (VALIDATED) (2026-04-19)

**Status:** Complete | **Hypothesis:** h1100 (VALIDATED)

### What was tested
Added a known-indication flag (`is_known_indication = drug_id in self.ground_truth.get(disease_id, set())`) to `_assign_confidence_tier` and wired three FILTER-demoting rules to carve out known indications:
1. `cancer_targeted_therapy` — fall through to cancer_same_type paths (lands MEDIUM/LOW) instead of FILTER.
2. `HIERARCHY_DEMOTE_TO_FILTER` — route to LOW via new sub_reason `<cat>_hierarchy_<group>_known_cap` instead of FILTER.
3. `freq<=2 AND not mech` — fall through to default paths.

Also named 5 previously-anonymous FILTER sub_reasons (`rank_over_20`, `no_targets`, `low_freq_no_mech`, `corticosteroid_metabolic_iatrogenic`) for audit clarity. Safety filters (`inverse_indication`, `corticosteroid_iatrogenic`, `non_therapeutic_compound`, `rank_over_20`, `no_targets`) were NOT carved out — they remain structural/safety signals.

### Headline result
| metric | before | after | Δ |
|---|---|---|---|
| Known-indication FILTER misfires @rank<=30 | 1,366 | 779 | **-43%** |
| Diseases with >=1 misfire | 492 | 270 | -222 |
| FILTER aggregate precision | 6.8% ± 0.7% | 6.1% ± 0.6% | -0.7pp (within 1pp tolerance) |
| GOLDEN / HIGH / MEDIUM / LOW | 78.5 / 80.0 / 39.9 / 10.0 | 82.6 / 80.5 / 40.4 / 10.5 | stable (within noise) |

### Positive-control impact (paper-visible)
| pair | rank | baseline tier | post-h1100 tier |
|---|---|---|---|
| metformin -> T2D | 5 | FILTER | **LOW** |
| tetrabenazine -> Huntington | 1 | FILTER | **LOW** |
| trastuzumab -> breast neoplasms | 20 | FILTER | **MEDIUM** |
| sildenafil -> PAH | 100 | FILTER | FILTER (rank>20 structural, unchanged — correct) |

Three paper crown-jewel FILTER mis-fires resolved. Sildenafil stays in FILTER because rank=100 is a genuine kNN weakness, not a rule mis-application (see h1103 follow-up).

### Remaining 779 known-indication FILTER cases — structural, not rule-based
| sub_reason | count | type |
|---|---|---|
| rank_over_20 | 543 | structural kNN |
| no_targets | 130 | structural kNN |
| corticosteroid_metabolic_iatrogenic | 28 | safety filter (CORRECT) |
| inverse_indication | 24 | safety filter (CORRECT) |
| other rules | 54 | smaller misc |

543/779 (70%) are rank>20 — kNN truly does not rank these FDA-approved pairs in the top 20. Not fixable by tier rules; only by improving the ranker (h1200).

### Methodology lesson
Before adding any FILTER-demoting rule, decide explicitly whether to carve out `is_known_indication=True`. Safety filters (inverse indication, iatrogenic effect) should NOT carve out. Blanket tier rules that assume "drug-class X is low quality for disease-type Y" probably should — the drug may be FDA-approved for this specific disease.

### New hypotheses (4 added)
- **h1102 (P2):** Deliverable annotation — surface `is_known_indication`, `filter_reason` (plain-English sub_reason), `known_but_low_rank`. Improves reader transparency without touching tiers.
- **h1103 (P2):** Audit the 543 rank>20 known-indication residuals. If they cluster by drug class or disease category, that guides h1200 supervised-GNN loss weighting.
- **h1104 (P3):** Inverse positive-control suite — 20 contraindication cases checked for HIGH/GOLDEN mis-assignment. Symmetric safety audit.
- **h1105 (P3):** Wire positive controls into CI/pre-commit. Prevents silent regression on the 20 fixed cases.

### Recommended next hypothesis
**h1102 (P2)** — low-effort, high-value annotation. The is_known_indication plumbing is already in the predictor; just surface it in the deliverable. Fully decouples from GNN work.

Or alternatively **h1103 (P2)** — diagnostic for h1200 (supervised GNN). Identifies which mis-rankings matter most for the upcoming training, so h1200 loss weighting has a principled prior.


## Current Session (continued): h1101 — Dantrolene Crown-Jewel Relabel (VALIDATED) (2026-04-19)

**Status:** Complete | **Hypothesis:** h1101 (VALIDATED — paper relabel, no code change)

### What was tested
Traced Dantrolene's predictions and tier assignments to validate Paper Section 3.5's "Dantrolene | Heart failure" row. Checked: (a) does the model predict Dantrolene for heart failure? (b) which rule demotes Dantrolene in the VT/VF/tachycardia/arrhythmia family? (c) is the rule over-broad? (d) is Dantrolene's drug-class assignment correct?

### Findings
**The model does NOT predict Dantrolene for heart failure** (D006333: 117 total preds, Dantrolene absent — no kNN neighbors surface it).

**The model DOES predict Dantrolene for the arrhythmia family:**
| Disease | Rank | Tier | sub_reason | freq | mech | in_gt |
|---|---|---|---|---|---|---|
| Ventricular tachycardia | 34 | FILTER | rank_over_20 | 5 | False | False |
| Ventricular fibrillation | 36 | FILTER | rank_over_20 | 5 | False | False |
| Tachycardia | 28 | FILTER | rank_over_20 | 5 | False | False |
| Arrhythmia | 67 | FILTER | rank_over_20 | 5 | **True** | True |

Tier is set by the **structural `rank_over_20` rule** — not a rule misfire. Per h1100 design principle, `rank_over_20` is NOT carved out for known indications because it is a structural kNN-floor signal (543/779 of all known-indication FILTER misfires hit it; demoting these would explode FILTER precision).

**Drug-class assignment is correct** — Dantrolene is not in CANCER_TARGETED_THERAPY, CORTICOSTEROID_DRUGS, or INVERSE_INDICATION_PAIRS. Targets: RyR1/2/3 (6261-6263) + CYP3A4 (1576), consistent with its MOA.

**Paper conflation root-cause:** Zamiri et al. 2014 *Circulation* randomized heart failure patients but the primary outcome was VT reduction (P=0.034, 66%). The paper conflated cohort (HF) with outcome (VT).

### Shipped
- **docs/claude/paper_v2_errata.md** created. First entry E1 corrects Section 3.5 row to: "Dantrolene | Ventricular tachycardia (in HF patients) | Drug -> Target (RyR2) -> Disease | RCT P=0.034, 66% VT reduction | model rank 34, FILTER (rank_over_20)".
- No code change to production_predictor.py — rank_over_20 is the correct rule.

### Latent artifact noted (not blocking)
`drug_cancer_types[Dantrolene] = {'solid_tumor'}` despite Dantrolene having no oncology role. Likely a DRKG co-occurrence edge. Queued as h1106 for broader audit.

### New hypotheses (2 added)
- **h1106 (P3):** Scan drug_cancer_types for similar artifacts in other non-cancer drugs (train_freq >= 3, 0% cancer GT, non-empty cancer_types).
- **h1107 (P2):** Audit every row of Paper Section 3.5 — same cohort/outcome conflation may apply to others. Run positive-control methodology on every claimed prediction.

### Recommended next hypothesis
**h1107 (P2)** — protects preprint credibility at low cost. Same positive-control methodology as h1100; 1-2h work. Alternatively **h1199 (P1, infrastructure)** unblocks h1200/h1201 on the 37->60 pivot.


## Current Session (continued): h1107 — Section 3.5 Full Audit (VALIDATED, high-impact) (2026-04-19)

**Status:** Complete | **Hypothesis:** h1107 (VALIDATED, paper credibility issue)

### What was tested
Mechanical audit of all 5 rows of Paper Section 3.5 ("retrospectively corroborated predictions"). For each row, query `p.predict(<disease>, top_n=500, include_filtered=True)` and check whether the claimed drug appears.

### Headline result
**Only 1 of 5 rows (20%) is corroborated by the current production predictor.**

| # | Drug | Paper disease | Model output | Verdict |
|---|------|---------------|-----------|---------|
| 1 | Dantrolene | Heart failure | absent (top 117) | FALSE (E1: relabel to VT, rank 34 FILTER) |
| 2 | Lovastatin | Multiple myeloma | absent (top 86) | FALSE (correct preds: hypercholesterolemia rank 1 GOLDEN) |
| 3 | Rituximab | Multiple sclerosis | rank 55, HIGH | **TRUE** |
| 4 | Pitavastatin | Rheumatoid arthritis | absent (top 125) | FALSE (correct preds: hypercholesterolemia rank 4 GOLDEN) |
| 5 | Empagliflozin | Parkinson's disease | absent (top 77/59) | FALSE — but model correctly predicts T2D rank 29, HF rank 52, CKD rank 23 (all in-GT, all FILTER via rank_over_20) |

### Empagliflozin is the cleanest h1103 case
Empagliflozin's 3 known-indication FILTER cases at rank 21-52 are the canonical "newly-approved second-in-class drug with narrow DRKG edge history" pattern. All 3 are in_gt=True. All 3 land FILTER via the structural rank_over_20 rule. This is exactly the signal class that:
- h1103 should quantify across the deliverable
- h1200 supervised GNN should upweight during training

### Shipped
- **docs/claude/paper_v2_errata.md** extended with E2 entry — full table + row-by-row detail.
- Recommended Option A revision: collapse Section 3.5 to Rituximab→MS only; move the other 4 rows to a supplementary "prior art / related repurposing literature" table with explicit note that the model does NOT surface these predictions.

### Paper credibility implications
Framing "five predictions that are retrospectively corroborated" is not supported. The honest framing is "one model prediction (Rituximab→MS) concurs with independent WHO Essential Medicines Listing; four additional clinical repurposing results are reported in the literature but are NOT surfaced by the current model — their inclusion here is as prior art, not model corroboration."

### New hypotheses (2 added)
- **h1108 (P3):** For the 4 uncorroborated rows, probe related/sibling/subtype diseases — find ANY related disease the model surfaces for each drug. Builds out the errata honestly.
- **h1109 (P3):** Separate the 5 Section 3.5 controls into their own positive-control category; add scripts/ci_check_section_3_5.py as a pre-merge guard for preprint-touching PRs.

### Recommended next hypothesis
**h1199 (P1, infrastructure)** — with Section 3.5 credibility surfaced, the infrastructure work to support a multi-metric comparison with TxGNN / HGTDR becomes even more urgent. A defensible paper v2 needs R@30 + Hits@K + MRR + AUPRC + AUROC all reported on the same 5-seed splits.

Alternatively, **h1103 (P2)** — now that Empagliflozin is known to be the canonical rank>20 known-indication residual, the 543-row audit should build on that. Clusters by drug-class (SGLT2 inhibitors, biologics, newly-approved narrow-edge drugs) will directly inform h1200's loss weighting.


## Current Session (continued): h1103 — Rank>20 Known-Indication Audit (VALIDATED) (2026-04-19)

**Status:** Complete | **Hypothesis:** h1103 (VALIDATED — clean h1200 training target)

### What was tested
Scanned all 543 known-indication FILTER cases at rank<=30 whose sub_reason is `rank_over_20` (identified by h1100). Clustered by drug class, disease category, mechanism support, train frequency, and rank bucket. Saved all 543 cases with metadata to `data/analysis/h1103_rank_over_20_audit.json`.

### Key distributions
| axis | top 3 |
|---|---|
| disease_category | infectious (111), autoimmune (82), cardiovascular (52) |
| drug_class | other_sm (343, undifferentiated SM), macrolide_or_other_abx (25), corticosteroid (19) |
| rank_bucket | 21-25 (334, 62%), 26-30 (209, 38%) |
| mechanism | mech=False (401, 74%), mech=True (142, 26%) |
| train_freq | f=3-9 (240), f<=2 (200), f>=10 (103) |

### Top (drug_class x disease_category) clusters
| drug_class | disease_category | n |
|---|---|---|
| other_sm | infectious | 66 |
| other_sm | autoimmune | 41 |
| other_sm | cardiovascular | 30 |
| other_sm | dermatological | 30 |
| **biologic_mab** | **autoimmune** | **28** |
| macrolide_or_other_abx | infectious | 20 |
| kinase_inhibitor | autoimmune | 10 |
| quinolone_abx | infectious | 9 |
| sglt2 | various | 7 |

### Structural insight
**74% of cases have mech=False** — the model has NO mechanism path for these drug-disease pairs in DRKG, yet the drugs are FDA-approved. This is the canonical "newly-approved drug with narrow DRKG edge history" pattern identified in h1107 (Empagliflozin → T2D/HF/CKD). A supervised GNN (h1200) that trains directly on treatment edges should learn these via embedding proximity rather than explicit mechanism paths.

**62% of cases are at ranks 21-25**, only slightly outside the tier threshold. A modest ranker lift (5 positions) recovers the majority.

### New hypotheses (2 added)
- **h1110 (P2, recall lever):** Apply 3x sample weight on these 543 pairs during h1200 supervised GNN training. Gate: >=100 pairs cross rank-20 threshold post-training.
- **h1111 (P3) [RYLAND]:** Deep-dive on the 28 biologic_mab x autoimmune cases. Packet for Feb 10 Ryland meeting.

### Recommended next hypothesis
**h1199 (P1, infrastructure)** — this and the three preceding hypotheses (h1100, h1101, h1107, h1103) have now established clear diagnostic + corrective targets for the h1200 supervised GNN. The infrastructure benchmark is the last gating step before the big GNN training run.
