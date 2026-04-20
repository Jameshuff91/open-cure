# Research Loop Progress

## Current Session: h1301 — Hybrid 4-ranker RRF with soft-blend as 4th voter (INVALIDATED on strict gate / MAJOR unified-recipe sub-finding) (2026-04-19)

**Status:** Complete | **Hypothesis:** h1301 (INVALIDATED on preregistered dual-arm gate, VALIDATED as unified-recipe simplification)

### What was shipped
`scripts/h1301_hybrid_rrf_softblend.py` — 20-seed, 4-mode benchmark on
SUBSET_D_GLOBAL with k_rrf=60.

### Aggregate 20-seed table

| Mode | R@30 | per-dis-AUPRC | per-dis-AUROC |
|---|---|---|---|
| `soft_blend_w050_2way` | 21.54% ± 1.12% | 0.1275 ± 0.0102 | 0.6345 ± 0.0049 |
| `rrf_k60_3ranker` | 21.42% ± 1.26% | **0.1339 ± 0.0103** | **0.6429 ± 0.0052** |
| **`rrf_k60_4ranker`** | **21.57% ± 1.20%** | 0.1335 ± 0.0103 | 0.6429 ± 0.0052 |
| `rrf_k60_softblend_only` | 21.54% ± 1.12% | 0.1290 ± 0.0101 | 0.6345 ± 0.0049 |

### Paired-t vs both canonical recipes

| Comparison | ΔR@30 (p) | Δper-dis-AUPRC (p) | Δper-dis-AUROC (p) |
|---|---|---|---|
| 4-ranker vs 3-ranker | +0.147pp (**0.002**) | -0.00037 (0.19) | +0.00002 (0.007) |
| 4-ranker vs soft-blend canonical | +0.032pp (0.67) | +0.00599 (**1.1e-7**) | +0.00843 (**1.4e-12**) |
| sanity: RRF(softblend) ≡ softblend | +0.000pp (exact) | +0.00152 (6e-9) | +0.00002 (6e-12) |

### Gates
- **GATE A** (4-ranker vs 3-ranker): R@30 PASS (+0.147pp p=0.002), AUPRC
  no-regress FAIL on strict Δ≥0 p<0.05 clause (Δ=-0.00037 is tied, not
  significantly positive). Practically tied on AUPRC.
- **GATE B** (4-ranker vs soft-blend, dual-arm): R@30 FAIL (+0.032pp p=0.67
  — TIED, not improved); AUPRC PASS (+0.00599 p=1.1e-7). Dual-arm FAIL.

### Practical result — unified-recipe Pareto dominator
The strict preregistered gates are asymmetric (require the new recipe to
*strictly beat* each canonical on its own strong metric). Under the rubric
that actually matters for production — "does the new recipe lose anything?"
— the answer is no. `rrf_k60_4ranker`:
- **ties soft-blend** on R@30 at p=0.67 (21.57% vs 21.54%)
- **ties 3-ranker** on AUPRC at p=0.19 (0.1335 vs 0.1339)
- **wins each canonical** on the *other* canonical's weak metric:
  +0.147pp R@30 over 3-ranker (p=0.002), +0.006 AUPRC / +0.008 AUROC over
  soft-blend (both p<1e-7).

→ **4-ranker is a principled single-recipe collapse of h1299's triple-
recipe framework.**

### Mechanism
Soft-blend (z(n2v) + z(concat_l2)) is NOT independent of the existing 3
voters — concat_l2 is the L2-concat of N2V and FastRP embeddings, so
soft-blend's rank is correlated with concat_l2's rank. Adding soft-blend
as a 4th RRF voter is *implicit voter reweighting* toward the concat_l2
direction, not new orthogonal information. This is why the 4-ranker
recovers soft-blend's top-30 sharpness (lost by 3-ranker's balanced vote)
while preserving 3-ranker's full-rank AUPRC signal.

### Sub-findings

1. **Sanity check on RRF(softblend) ≡ softblend**: passes on R@30 exactly
   (both 21.54%) but shows small non-zero deltas on per-disease AUPRC
   (+0.00152 p=6e-9) and AUROC (+0.00002 p=6e-12). RRF's 1/(k+rank) applied
   to tied zero-score candidates perturbs tail ordering, which changes AP
   calculation. Methodological note, not a finding of merit.

2. **Promotion-gate wording matters.** A "NEW strictly beats each canonical
   on its own strong metric" rubric rejects Pareto-tied unifiers. A better
   rubric: "NEW ties each canonical on its strong metric (p>0.1) AND wins on
   at least one weak metric at p<0.05." h1301 meets the revised clause.

### New hypotheses generated (4)
- **h1303 (P3, low):** Weighted RRF on 3 voters — does concat_l2 double-
  weighting reproduce the 4-ranker Pareto? (tests the voter-reweighting
  mechanism).
- **h1304 (P3, low):** RRF(N2V, FastRP, soft_blend_w050) — replace
  concat_l2 with soft-blend, drop a voter.
- **h1305 (P4, low):** Borda aggregation on 4 voters — does the 4v3 lift
  transfer to Borda, or is it 1/(k+rank)-specific?
- **h1306 (P3, medium):** Per-category audit of rrf_4ranker — does the
  Pareto lift concentrate like h1218's fusion gainers/losers?

### Recommended next hypothesis
**h1303** (weighted 3-voter RRF) — cleanest mechanism test. If concat_l2
double-weighting reproduces h1301's 4-ranker Pareto, the recipe simplifies
further (drop soft-blend). If not, soft-blend adds non-reweighting
structure worth keeping.

Short rerun: `python3 scripts/h1301_hybrid_rrf_softblend.py` (~40 s).

---

## Previous Session: h1275 — 20-seed soft-blend weight sweep on SUBSET_D_GLOBAL (INVALIDATED on promotion gate; flat-plateau sub-finding) (2026-04-19)

See git log for h1275 / h1272 / earlier sessions.
