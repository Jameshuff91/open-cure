# h1249 — Entropy-routed fusion benchmark (production routing rule)

**Routing rule (from h1247 + h1248):**

| Stratum | Mode |
|---|---|
| n_gt < 21 | concat_l2 (default fuse) |
| n_gt 21-50 | concat_l2 (all terciles) |
| n_gt ≥ 51, low entropy | concat_l2 (h1248: +0.84 hits, p=0.0055) |
| n_gt ≥ 51, mid entropy | **node2vec** (h1248: -0.62 hits, avoid fuse) |
| n_gt ≥ 51, high entropy | concat_l2 (indifferent / +0.41pp) |

**Caveat:** entropy and n_gt are computed from `expanded_ground_truth.json`. For a held-out disease this is technically oracle metadata; production deployment would require a learned router on disease features (h1260 follow-up). Tercile cuts are leak-bounded — recomputed per-seed on training-side n_gt≥51 diseases only.

## Aggregate (mean ± std across seeds)

| Mode | R@30 | hits@30 (drug) | hits@30 (triple) | MRR | AUPRC | AUROC |
|---|---|---|---|---|---|---|
| `node2vec` | 19.55%±1.18% | 19.55%±1.18% | 11.91%±0.98% | 0.0284±0.0027 | 0.0569±0.0023 | 0.5766±0.0067 |
| `concat_l2` | 20.87%±0.91% | 20.87%±0.91% | 12.49%±1.32% | 0.0296±0.0036 | 0.0642±0.0033 | 0.5851±0.0086 |
| `entropy_routed` | 20.92%±0.89% | 20.92%±0.89% | 12.60%±1.34% | 0.0297±0.0036 | 0.0628±0.0033 | 0.5834±0.0083 |

## Per-seed paired-t — entropy_routed vs concat_l2 (n=5)

| Metric | Δ (routed − concat_l2) | t | p (two-sided) |
|---|---|---|---|
| R@30 | +0.0490 | +2.180 | 0.0948 |
| hits30_drug | +0.0490 | +2.180 | 0.0948 |
| MRR | +0.0001 | +1.175 | 0.305 |
| AUPRC | -0.0014 | -2.708 | 0.0536 |
| AUROC | -0.0017 | -2.464 | 0.0694 |

## Per-disease paired-t — entropy_routed vs concat_l2 (all rows)

| Metric | n | Δ_mean | t | p (two-sided) |
|---|---:|---|---|---|
| R@30 | 1002 | +0.0491 | +2.027 | 0.0429 |
| hits30 | 1002 | +0.0399 | +1.713 | 0.0869 |

## Restricted to flipped diseases (routed→node2vec, n=65)

Only the (n_gt≥51 + mid-entropy) subset is routed away from concat_l2; everything else is identical to the concat_l2 baseline. The restricted paired-t isolates the rule's contribution.

| Metric | n | Δ_mean (routed − concat_l2) | t | p |
|---|---:|---|---|---|
| R@30 | 65 | +0.7571 | +2.075 | 0.042 |
| hits30 | 65 | +0.6154 | +1.738 | 0.087 |

## Per-seed details

| Seed | n_test | low_cut | high_cut | concat_l2 R@30 | routed R@30 | Δ |
|---|---|---|---|---|---|---|
| 42 | 202 | 3.212 | 3.994 | 20.68% | 20.79% | +0.114pp |
| 123 | 200 | 3.168 | 4.007 | 20.86% | 20.87% | +0.008pp |
| 456 | 200 | 3.212 | 4.008 | 21.00% | 21.08% | +0.078pp |
| 789 | 200 | 3.168 | 4.015 | 22.34% | 22.33% | -0.008pp |
| 2024 | 200 | 3.168 | 4.006 | 19.49% | 19.55% | +0.053pp |

## Per-seed routing histogram

- seed 42: {'concat_l2|hd=False|t=n/a': 170, 'concat_l2|hd=True|t=high': 13, 'concat_l2|hd=True|t=low': 12, 'node2vec|hd=True|t=mid': 7}
- seed 123: {'concat_l2|hd=False|t=n/a': 166, 'node2vec|hd=True|t=mid': 14, 'concat_l2|hd=True|t=high': 10, 'concat_l2|hd=True|t=low': 10}
- seed 456: {'node2vec|hd=True|t=mid': 12, 'concat_l2|hd=False|t=n/a': 165, 'concat_l2|hd=True|t=low': 13, 'concat_l2|hd=True|t=high': 10}
- seed 789: {'concat_l2|hd=False|t=n/a': 167, 'concat_l2|hd=True|t=low': 9, 'concat_l2|hd=True|t=high': 8, 'node2vec|hd=True|t=mid': 16}
- seed 2024: {'concat_l2|hd=False|t=n/a': 156, 'concat_l2|hd=True|t=low': 13, 'concat_l2|hd=True|t=high': 15, 'node2vec|hd=True|t=mid': 16}
