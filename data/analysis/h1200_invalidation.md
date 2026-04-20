# h1200 INVALIDATION — Supervised GNN on DRKG (three variants)

**Date:** 2026-04-19
**Hypothesis:** h1200 — Supervised GraphSAGE on DRKG treatment edges as Path A of the 37→60 pivot
**Verdict:** **INVALIDATED** across three architectural variants
**Preregistered ship gate:** R@30 ≥ 35% on h1199 5-seed benchmark (stretch 45%)

## Headline

All three variants underperform the node2vec_256 prior they started from. The best variant (warm-start from Node2Vec) reached 11.47% R@30 vs the 19.55% Node2Vec baseline — a **−8pp regression**.

| Variant | R@30 | MRR | AUPRC | AUROC | Δ vs Node2Vec |
|---|---|---|---|---|---|
| Node2Vec 256 (starting prior) | **19.55%** | 0.0284 | 0.0569 | 0.5766 | — |
| h1215 ensemble (best DRKG ceiling) | **20.87%** | 0.0296 | 0.0642 | 0.5851 | +1.32pp |
| h1200 V1 (Xavier cold-start, supervised only) | 7.99% ± 1.07% | 0.0103 | 0.0208 | 0.5291 | **−11.56pp** |
| h1200 V2 (Node2Vec warm-start, supervised only) | 11.47% ± 0.90% | 0.0150 | 0.0311 | 0.5491 | **−8.08pp** |
| h1200 V3 (warm-start + 50% unsupervised mix) | 10.81% ± 0.96% | 0.0152 | 0.0300 | 0.5462 | **−8.74pp** |

Benchmark source: `data/analysis/clean_benchmark_graphsage_h1200_s42_256.{json,md}` (h1199 5-seed, 956 eligible diseases).

## Training setup (common to all variants)

- Architecture: 3-layer GraphSAGE, mean aggregator, 256-d hidden
- Message-passing graph: `drkg_no_treatment.tsv` (5.81M edges) + train-split treatment edges
- Supervision: 4,968 treatment edges (`DRUGBANK::treats::Compound:Disease`)
- Disease-level holdout: 20% of diseases held out (matching h393 convention)
- Negatives: degree-weighted random (drug, disease) pairs, 1:3 ratio
- Optimizer: AdamW, lr=1e-3, weight_decay=1e-5
- 50 epochs max, early stop patience=8 on in-training R@30
- Seed: 42
- Hardware: RTX 3090 (vast.ai), ~3 min / variant, ~$0.01 / variant

Differences between variants:
- **V1:** x_param Xavier-initialized; loss = supervised BCE only
- **V2:** x_param initialized from `node2vec_256_no_treatment` (99.7% node coverage; Xavier fallback for 281 missing); loss = supervised BCE only
- **V3:** V2 initialization + `--unsup-weight 0.5` (adds BCE on random message-passing edges alongside supervised treatment-edge BCE)

## In-training vs h1199 benchmark

In-training holdout R@30 (computed on 16 held-out diseases / 19 treatment edges per the train/holdout split) was consistently higher than the h1199 benchmark number:

| Variant | in-training R@30 | h1199 R@30 |
|---|---|---|
| V1 | 12.05% | 7.99% |
| V2 | 14.25% | 11.47% |
| V3 | 15.27% | 10.81% |

The in-training metric is on a tiny set and per-disease variance is huge. **h1199 is the honest comparison.** All three honest numbers are below Node2Vec by 8-12pp.

## Diagnosis

### Signal sparsity

4,968 treatment edges is **~0.05 supervision pairs per node** across a 94,528-node graph. V1's Xavier-init failure is unsurprising — the model has no prior information and the supervised signal alone cannot span the full representation space.

### Active degradation of the prior

The V2/V3 finding is the diagnostic killer. **Starting from a 19.55%-R@30 prior, 50 epochs of supervised gradient drives the representation to 11.47% / 10.81%.** The model IS learning — loss decreases monotonically, in-training holdout improves through mid-training — but what it learns is destructive to the downstream 1,011-disease benchmark.

Mechanism: the supervised loss objective is to make treatment-edge pairs score high and degree-weighted random pairs score low. With only 4,968 positives, the model can satisfy this objective by pulling embeddings into a narrow manifold that hits the positives while losing the broader similarity structure Node2Vec encoded. The result is an embedding that scores the train treatment edges well but generalizes poorly — classic overfitting, but to a tiny supervision set.

### Mixed loss doesn't save it

V3 added unsupervised BCE on random message-passing edges alongside the supervised term, 50/50 weighted — exactly the classical GraphSAGE training recipe. The unsupervised loss *did* stabilize training (slower degradation, higher in-training R@30 peak). But h1199 benchmark still landed at 10.81% — slightly WORSE than V2. Unsupervised loss on random edges is a weaker signal than Node2Vec's random-walk-based co-occurrence statistics; it partially preserves node similarity but not enough to offset the supervised pull.

### What a "proper" supervised GNN would need

Based on these three failures, for supervised GNN training to work on DRKG, one of these would need to be true:

1. **~100× more supervision** (~500,000 labeled positive drug-disease pairs). Not available; DRKG has 4,968.
2. **Heterogeneous edge types preserved in message passing** (HGT, R-GCN per-relation aggregators). Might help marginally; does not address the core sparsity problem.
3. **Self-supervised pretraining on the full graph, THEN supervised fine-tuning with very low LR**. Essentially: do Node2Vec/FastRP first, then a tiny supervised nudge. We tested V2 with LR=1e-3; a much lower LR (1e-5) with short epoch count might do less damage but is unlikely to IMPROVE beyond the prior — at best it would match Node2Vec.

All three are speculative rescues with low expected lift, and none map to a ≥35% R@30 outcome.

## Implications for the 37→60 pivot

The original pivot plan (memory: `project_open_cure_pivot_37_60.md`) assumed:
> Path A is the biggest single DRKG lever; supervised training optimises directly for drug-disease alignment (h922-v2 failed on unsupervised).

**This premise is wrong at DRKG scale.** The supervised signal is not a lever; it is a detractor. The 4,968-edge supervision set lacks the density to train a GNN from scratch and lacks the quality differential to refine a strong prior.

**What still works:**
- **h1215 ensemble pattern (+1.32pp at zero GPU cost)** — the proven DRKG recall lever. Further fusion work (h1225 RRF, h1226 per-disease weights, h1227 three-embedding concat) is a viable track.
- **h1201 LINCS reverse-connectivity** — unchanged; the orthogonal transcriptomic signal is independent of DRKG supervised training.
- **h1202 hybrid fusion** — unchanged; gated on h1201 landing.
- **h1203 Ryland-label calibration** — unchanged; a precision lever.

## Recommended next moves

1. **Skip h1200 entirely.** Do not retry with HGT, R-GCN, or different hyperparameters. The supervision signal is the bottleneck, and all architectural variations inherit that bottleneck.
2. **Elevate h1215-style ensemble work to the top recall lever.** The autonomous loop has already produced h1225/h1226/h1227 hypotheses in this family.
3. **Start h1201 (LINCS) as the next P1.** Its success would give us the orthogonal signal that makes h1202 fusion a genuine Nature-paper claim.
4. **Correct the pivot-plan memory.** `project_open_cure_pivot_37_60.md` needs the h1200 invalidation note and the recalibrated path (h1215 ensemble + h1201 LINCS as the two levers).

## Cost summary

| Item | Cost |
|---|---|
| vast.ai RTX 3090 provisioning + setup | ~$0.10 |
| V1 training + benchmark | ~$0.01 |
| V2 training + benchmark | ~$0.01 |
| V3 training + benchmark | ~$0.02 |
| **Total** | **~$0.14** |

Cheap to invalidate. The $0.14 bought a structural answer about Path A that closes a whole branch of the roadmap.
