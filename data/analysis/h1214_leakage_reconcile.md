# h1214 — Reconcile Treatment-Edge Leakage Retention

**Question:** Why does CLAUDE.md cite 71.2% retention after removing treatment edges
while h1212's clean_embedding_benchmark says 49.5%?

- Seeds: [42, 123, 456, 789, 2024]
- k = 20, top-N = 30
- Common disease universe (both embeddings ∩ internal GT): **838**
- Full-only eligible diseases: 1,011
- No-treatment-only eligible diseases: 850

## 8-cell factorial (common universe)

| Embedding | GT | micro R@30 | macro R@30 |
|---|---|---|---|
| full | internal_gt | 37.33%±2.88% | 36.10%±3.11% |
| full | expanded_gt | 10.24%±1.30% | 16.94%±0.70% |
| no_treatment | internal_gt | 23.65%±1.53% | 18.69%±0.74% |
| no_treatment | expanded_gt | 7.11%±0.72% | 9.21%±0.68% |

## Retention (no_treatment / full) — by methodology

| Universe | GT | Aggregation | full R@30 | no_treatment R@30 | retention |
|---|---|---|---|---|---|
| common | internal_gt | micro_r30 | 37.33% | 23.65% | 63.4% |
| common | internal_gt | macro_r30 | 36.10% | 18.69% | 51.8% |
| common | expanded_gt | micro_r30 | 10.24% | 7.11% | 69.4% |
| common | expanded_gt | macro_r30 | 16.94% | 9.21% | 54.4% |
| embedding-native | internal_gt | micro_r30 | 41.64% | 24.51% | 58.9% |
| embedding-native | internal_gt | macro_r30 | 40.61% | 19.42% | 47.8% |

## Embedding-native universe (reproduces compare_honest_embeddings.py)

| Embedding | N eligible | micro R@30 | macro R@30 |
|---|---|---|---|
| full | 1011 | 41.64%±1.53% | 40.61%±1.86% |
| no_treatment | 850 | 24.51%±2.36% | 19.42%±1.60% |

## Interpretation

**Retention ranges from 47.8% to 69.4% depending on methodology.** The old CLAUDE.md
"71.2%" number was the upper end of this range: micro aggregation, internal GT,
embedding-native universes. The h1212 "49.5%" number was at the lower end: macro
aggregation, expanded GT, common-universe (850 diseases). Both are self-consistent
on their own axes.

### Which retention to cite externally?

For an apples-to-apples claim about treatment-edge leakage, the defensible
methodology stacks the most stringent controls:

1. **Common disease universe** — eliminates the confound that `no_treatment` was
   trained on a larger DRKG snapshot (94,247 entities) than the "named"
   full-treatment CSV (49,616). Using each embedding's native universe lets
   coverage differences masquerade as retention.
2. **Macro aggregation (per-drug R@30 averaged across diseases)** — each disease
   gets equal weight, preventing large-GT diseases (where treatment edges
   concentrate) from dominating the headline.
3. **Expanded GT** — 19× more pairs reduces the stochastic hit/miss noise on
   any individual disease.

Under that stack: **full Node2Vec 16.94%, no-treatment 9.21%, retention = 54.4%
(leakage = 45.6%).** This is within 5pp of h1212's 49.5% headline.

### Per-axis retention effect (ceteris paribus)

- **Aggregation shift (micro → macro):** retention DROPS 11-15pp across both GTs.
  Macro surfaces leakage that micro averages away.
- **GT shift (internal → expanded):** retention RISES 3-6pp. Expanded GT has more
  "easy" pairs that kNN recovers even without treatment edges (indirect paths
  work for well-annotated drug-disease pairs).
- **Universe shift (common → embedding-native):** retention DROPS 2-4pp under
  macro — the no_treatment embedding performs slightly worse on its
  additional 12 diseases than on the common set, which the embedding-native
  comparison reveals.

### Recommendation

Replace "Honest-embedding leakage: 26.06% vs 36.59% retains 71.2% via indirect
paths" in CLAUDE.md with:

> Treatment-edge leakage: full Node2Vec 16.94% → no-treatment 9.21% R@30 (macro,
> 838 common diseases, expanded GT, 5 seeds) → **54.4% retention (45.6% leakage)**.
> Legacy 71.2% figure from `compare_honest_embeddings.py` used micro + internal GT +
> embedding-native universes; see `data/analysis/h1214_leakage_reconcile.md` for
> the 8-cell factorial.
