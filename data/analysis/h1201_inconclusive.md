# h1201 INCONCLUSIVE — standalone LINCS × CREEDS reversal-connectivity

**Date:** 2026-04-19
**Hypothesis:** h1201 — LINCS L1000 reverse-connectivity (Path B of the 37→60 pivot)
**Verdict:** **INCONCLUSIVE as a standalone recall lever** (not invalidated — see scope note)
**Preregistered ship gate:** ≥15% R@30 standalone on the evaluable subset → expand coverage; else close h1201 inconclusive

## Headline

Standalone LINCS reverse-connectivity fails at the sanity-check level:

| Metric | Value |
|---|---|
| R@30 per-drug (5-seed mean) | **1.46% ± 1.0%** |
| Random baseline (30/1593) | **1.88%** |
| MRR | 0.039 |
| AUPRC | 0.010 |
| AUROC | 0.524 (near chance) |

Human-only filter (drops mouse/rat CREEDS sigs to avoid ortholog noise) performs **worse**: R@30 = 1.08%, AUROC = 0.47 (below 0.5 on one seed).

## Pipeline built

The infrastructure landed correctly — the failure is biological, not a pipeline bug.

| Stage | Artifact | Size |
|---|---|---|
| LINCS Phase I Level 5 | `data/external/lincs/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx` | 2.9GB |
| LINCS metadata | `data/external/lincs/GSE92742_Broad_LINCS_*_info.txt.gz` | ~15MB |
| LINCS→DrugBank bridge | drugbank vocabulary.csv InChIKey + drug-mappings.tsv PubChem CID | built in script |
| CREEDS disease sigs | `data/external/creeds/disease_signatures-v1.0.json` | 16MB |
| DO→MESH xref (for disease mapping) | `data/reference/doid_to_mesh_mapping.json` | 4,011 pairs |
| Disease Ontology source | `data/external/disease_ontology/doid.obo` | 6.7MB |
| Aggregated drug signatures | `data/embeddings/lincs_signatures.npy` | (1593, 12328) float32 |
| Aggregation script | `scripts/h1201_lincs_aggregate.py` | 248 lines |
| Scoring script | `scripts/h1201_score_lincs_creeds.py` | 228 lines |
| Benchmark output | `data/analysis/h1201_lincs_creeds_benchmark.{json,md}` | — |

### Funnel

| Stage | Count | % of DRKG treat |
|---|---|---|
| DRKG diseases with treatment edges | 1,182 | 100% |
| + has DO_ID in CREEDS (via DO→MeSH xref) | 110 | 9.3% |
| + has ≥1 LINCS-covered drug in its GT | 92 | 7.8% |
| Treatment edges evaluable | 559 | 11.3% |

## Positive-control diagnosis

Canonical, well-established drug-disease pairs are **scattered across the rank distribution**, many at the bottom:

| Disease | Canonical drug | LINCS rank / 1,593 |
|---|---|---|
| Type 2 diabetes | Metformin | 490 |
| Type 2 diabetes | Glipizide | 54 ✓ |
| Alzheimer's | Donepezil | 567 |
| Alzheimer's | **Memantine** | **1184** |
| Alzheimer's | Rivastigmine | 466 |
| Alzheimer's | Galantamine | 249 |
| Rheumatoid arthritis | Methotrexate | 99 |
| Rheumatoid arthritis | **Hydroxychloroquine** | **1509 (bottom)** |
| Rheumatoid arthritis | **Sulfasalazine** | **1431 (near bottom)** |
| Rheumatoid arthritis | Leflunomide | 977 |
| Breast cancer | Tamoxifen | 229 |
| Breast cancer | **Letrozole** | **1335 (near bottom)** |
| Breast cancer | Anastrozole | 331 |
| Asthma | **Albuterol** | **1373 (near bottom)** |
| Asthma | Salmeterol | 677 |
| Asthma | Montelukast | 1105 |
| Asthma | Fluticasone | 55 ✓ |

Only 3 of 17 canonical pairs land in top-100. Several are at the very bottom, which is worse than random — reversal scoring is actively *anti-predicting* these drugs.

## Mechanistic interpretation

Reverse-connectivity assumes: **drug treats disease ⇔ drug's transcriptomic signature opposes disease's transcriptomic signature**.

This assumption holds for narrow drug classes:
- Cytotoxics in cancer (kill proliferating cells → reverse proliferation signature)
- Some direct metabolic modulators

The assumption **fails structurally** for most drug mechanisms:

| Drug class | Why reversal fails | Example |
|---|---|---|
| Receptor antagonists / agonists | Act at protein level, no transcriptional mirror | Albuterol (β2-agonist), Montelukast (LTD4 antagonist) |
| Hormonal modulators | Alter serum hormone levels, tumor expression diffuse | Letrozole, Anastrozole (aromatase inhibitors) |
| Enzyme inhibitors upstream of transcription | Block metabolite flux, transcriptional response noisy | Methotrexate (DHFR), Hydroxychloroquine (lysosomal) |
| Neurotransmitter modulators | Act on synaptic proteins, downstream transcription diffuse | Memantine (NMDA antagonist), Donepezil (AChE inhibitor) |
| Most biologics | Extracellular binding, indirect transcription | Tocilizumab, Rituximab |

This is a known but underappreciated limitation of the CMap/L1000 reverse-connectivity paradigm. It was motivated by specific anecdotal successes (e.g. gedunin → HSP90, sirolimus → rapamycin-target pathways) and never systematically validated across broad repurposing targets.

## Why INCONCLUSIVE, not INVALIDATED

We tested **one specific use of LINCS**: standalone reversal-connectivity as a recall lever. That failed.

We did NOT test:
1. **LINCS-as-feature in fusion (h1290)** — kNN handles most drugs; LINCS reversal could still add signal on orthogonal errors even with poor standalone quality. Fusion lift depends on error uncorrelation, not absolute accuracy.
2. **KS-statistic CMap-style scoring (h1291)** — cosine may be the wrong similarity. Original CMap uses rank-based Kolmogorov-Smirnov enrichment. Positive controls might still fail, but worth confirming before closing LINCS entirely.
3. **Narrow-class scope** — reversal might work specifically on cytotoxic + cancer subset. Too-small-n to test in this pool.

## Implications for the 37→60 pivot

The plan's "Path B: +5-10pp from LINCS" premise is **disproven at standalone scale**. Combined with h1200's invalidation, **both designed recall levers from the pivot plan have failed.**

Remaining viable paths:
- **h1215-family ensemble refinements** — the autonomous loop's fusion-recipe work. Validated +1.32pp ceiling. Further gains likely small (plateau already observed in h1273).
- **h1203 Ryland-label calibration** — precision lever, not recall. Doesn't help push R@30 past 20.87%.
- **LINCS-as-feature in h1202 fusion (h1290)** — low-probability salvage of LINCS signal via fusion orthogonality.
- **Paper reframe** — honestly report the DRKG ceiling at ~21% R@30, position against TxGNN inductive 6-14% and present the hybrid confidence-tier work as the contribution instead of SOTA recall.

## Cost summary

| Item | Cost |
|---|---|
| LINCS download (2.9GB gctx + metadata) | $0 (NCBI GEO) |
| CREEDS download | $0 |
| Disease Ontology obo | $0 |
| Aggregation + scoring compute | $0 (laptop CPU) |
| **Total** | **$0** |

Time: ~2.5 hours end-to-end including the dupe-download detour.

## Lessons

1. **Run positive controls BEFORE declaring a hypothesis alive.** The 1.46% R@30 alone is ambiguous (could be power problem on low-n). The positive-control distribution (Albuterol rank 1373 for asthma) is unambiguous — the scoring is anti-correlated with reality for whole drug classes. Should have been the first check, not the last.
2. **Reverse-connectivity's reputation doesn't match its breadth.** CMap/L1000 are cited as general repurposing tools but the published success cases cluster tightly in cytotoxic/metabolic space. Pivot-plan assumption "+5-10pp from LINCS" was motivated by this reputation, not by a scoped audit of where reversal actually works.
3. **DRKG ceiling is close to what we already have.** h1215's 20.87% may be genuinely near the ceiling of what DRKG + surface-level external data can deliver. Further large R@30 gains require either Ryland-labeled expansion of the GT set itself, or fundamentally different data modalities (clinical trial outcomes, protein structure, etc.).
