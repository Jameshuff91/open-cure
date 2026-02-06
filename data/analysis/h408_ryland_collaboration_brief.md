# Open-Cure Drug Repurposing: Skin & Inflammatory Disease Predictions
## Prepared for Ryland Mortlock Collaboration Meeting (Feb 10, 2026)

---

## Executive Summary

Our DRKG-based drug repurposing model generates predictions across 14,150 drug-disease pairs
with a tiered confidence system validated on held-out data:

| Tier | Holdout Precision | Count | Interpretation |
|------|-------------------|-------|----------------|
| **GOLDEN** | 69.9% | 285 | Very high confidence - likely true |
| **HIGH** | 58.8% | 791 | High confidence - strong evidence |
| **MEDIUM** | 30.2% | 2,656 | Moderate - worth investigating |
| **LOW** | 16.2% | 3,140 | Low - speculative |
| **FILTER** | 10.3% | 7,278 | Filtered - likely noise |

For dermatological and autoimmune diseases, we have **227 novel GOLDEN/HIGH predictions**
across 40 diseases, plus 180 MEDIUM-tier predictions worth reviewing.

---

## Section 1: Top Novel Predictions for Ryland's Research Areas

### 1A. Highest-Confidence Non-Corticosteroid Predictions

These are the most interesting predictions — drugs with specific mechanisms predicted
for skin/autoimmune diseases, NOT already FDA-approved for those conditions.

#### Tier: GOLDEN (70% holdout precision)

| # | Drug | Disease | Mechanism | Key Target(s) | Gene Overlap | Notes |
|---|------|---------|-----------|---------------|--------------|-------|
| 1 | **Adalimumab** | Systemic Lupus Erythematosus | Anti-TNF-alpha mAb | TNF-alpha | None in DRKG | **HARMFUL**: Anti-TNF INDUCES SLE (12K FAERS reports). Drug-induced lupus, not treatment. |
| 2 | **Adalimumab** | Polymyositis | Anti-TNF-alpha mAb | TNF-alpha | None | **MIXED**: Case reports of success BUT also polymyositis flares on adalimumab |
| 3 | **Adalimumab** | Temporal Arteritis (GCA) | Anti-TNF-alpha mAb | TNF-alpha | None | **FAILED RCT**: 70-pt Phase 2 trial showed no benefit vs placebo (58.9% vs 50%) |
| 4 | **Rituximab** | Temporal Arteritis (GCA) | Anti-CD20 (B-cell depletion) | CD20/MS4A1 | None | **WEAK**: Case reports/series only, no RCTs. Tocilizumab is the proven biologic. |
| 5 | **Adalimumab** | Autoimmune Hepatitis | Anti-TNF-alpha mAb | TNF-alpha | None | **MIXED**: TNF in hepatic inflammation, but limited clinical evidence |
| 6 | **Adalimumab** | GVHD | Anti-TNF-alpha mAb | TNF-alpha | None | **NEGATIVE**: Used off-label, but limited efficacy + infection risk. No major RCTs. |
| 7 | **Adalimumab** | Myasthenia Gravis | Anti-TNF-alpha mAb | TNF-alpha | None | **HARMFUL**: Anti-TNF INDUCES MG (case reports of adalimumab-induced MG) |
| 8 | **Azathioprine** | Alopecia Areata | Purine analog / immunosuppressant | HPRT1 (via 6-MP) | None | **VALIDATED**: 10-year cohort (63 pts): 92.7% mean regrowth at 2mg/kg/day |
| 9 | **Corticotropin (ACTH)** | Alopecia Areata | ACTH receptor agonist | MC2R | MC2R overlap! | **MECHANISTIC**: ACTH upregulated in AA lesions, but no clinical trials |
| 10 | **Corticotropin (ACTH)** | Autoimmune Hepatitis | ACTH receptor agonist | MC2R | None | Steroid-independent anti-inflammatory; no clinical evidence |
| 11 | **Adalimumab** | Multiple Sclerosis | Anti-TNF-alpha mAb | TNF-alpha | None | **HARMFUL**: TNF blockade worsens MS (paradoxical demyelination) |
| 12 | **Methotrexate** | Actinic Keratosis | DHFR inhibitor / immunomod | DHFR + 204 targets | None | **OFF-TARGET**: Works for keratoacanthomas (92%), NOT actinic keratosis. 5-FU preferred for AK. |

#### Tier: HIGH (59% holdout precision)

| # | Drug | Disease | Mechanism | Key Target(s) | Gene Overlap | Notes |
|---|------|---------|-----------|---------------|--------------|-------|
| 13 | **Corticotropin** | Juvenile Idiopathic Arthritis | ACTH receptor agonist | MC2R | None | Melanocortin anti-inflammatory |
| 14 | **Azathioprine** | Juvenile Idiopathic Arthritis | Purine analog | HPRT1 | None | Standard immunosuppressant |
| 15 | **Rituximab** | Juvenile Idiopathic Arthritis | Anti-CD20 | CD20 | None | B-cell depletion for refractory JIA |

### Literature Validation Summary (Non-CS GOLDEN predictions)

| Prediction | Lit. Status | Evidence |
|------------|------------|---------|
| Azathioprine → Alopecia Areata | **VALIDATED** | 10-year cohort, 92.7% regrowth |
| Corticotropin → Alopecia Areata | Mechanistic only | ACTH upregulated in lesions, no trials |
| Adalimumab → SLE | **HARMFUL** | TNF inhibitors INDUCE lupus (12K FAERS reports) |
| Adalimumab → MG | **HARMFUL** | TNF inhibitors CAUSE MG (case reports) |
| Adalimumab → MS | **HARMFUL** | Paradoxical demyelination |
| Adalimumab → GCA | Failed RCT | Phase 2: no benefit vs placebo |
| Adalimumab → GVHD | Negative | Limited efficacy + infection risk |
| Adalimumab → Polymyositis | Mixed | Success cases AND flare cases |
| Rituximab → GCA | Weak | Case series only |
| Methotrexate → AK | Off-target | Works for keratoacanthomas, not AK |

**Key insight**: 4/12 non-CS GOLDEN predictions are harmful (anti-TNF → autoimmune).
This is a known model limitation: the KG captures drug-disease associations (including
adverse effects) without distinguishing "treats" from "causes."

### 1B. Predictions with Drug-Target / Disease-Gene Overlap

**Most relevant for spatial transcriptomics validation.** These predictions have
molecular-level support: the drug's known gene targets overlap with genes
associated with the disease.

| Drug → Disease | Shared Genes | Key Pathway |
|----------------|-------------|-------------|
| **Etanercept → SLE** (MEDIUM) | 9 genes: FCGR1A, FCGR2A, FCGR2B, FCGR2C, C1QA, C1R, C1S, TNF, TNFRSF1B | Fc gamma receptors + complement + TNF signaling |
| **Dexamethasone → Psoriasis** (HIGH) | 30 genes: VEGFA, JUN, GSTP1, NOTCH1, IRS2, SLC7A11, HMOX1, DUSP22... | NF-kB/AP-1 + oxidative stress + angiogenesis |
| **Betamethasone → Atopic Dermatitis** (GOLDEN) | 5 genes: NFKBIA, IL1B, DNMT1, CTSL, RELB | NF-kB pathway + epigenetic regulation |
| **Dexamethasone → Scleroderma** (GOLDEN) | 5 genes: ITGB1, LGALS1, TNFAIP3, TGFB2, TRIT1 | TGF-beta + integrin + A20 (NF-kB) |
| **Betamethasone → Scleroderma** (GOLDEN) | 5 genes: ITGB1, LGALS1, TNFAIP3, TGFB2, TRIT1 | TGF-beta + integrin fibrosis |
| **Dexamethasone → Osteoarthritis** (GOLDEN) | 15 genes: BMP2, SOX9, MYC, MMP2, IL1B, VEGFA, TIMP1, NRF2... | Cartilage remodeling + inflammation |

### 1C. Key Skin Disease Predictions (All Tiers)

#### Ichthyosis / Lamellar Ichthyosis
- **Known treatments**: Urea (emollient), Salicylic acid (keratolytic), Lactic acid
- **Novel predictions**: Corticosteroids (HIGH tier) — anti-inflammatory for ichthyosiform erythroderma
- **Ryland angle**: Ichthyosis has only 8 known disease genes. Spatial transcriptomics could dramatically expand this.
- **Drug targets to check**: TGM1 (Entrez 21816) — the ONLY known gene for lamellar ichthyosis in DRKG

#### Hidradenitis Suppurativa
- **Known treatments**: Adalimumab (GT), Secukinumab, Bimekizumab (all biologics)
- **Novel GOLDEN**: Methylprednisolone (rank 1), Dexamethasone (rank 2), Hydrocortisone (rank 4)
- **Gene overlap**: IL1B shared between corticosteroid targets and HS disease genes
- **Ryland angle**: IL-1beta and TNF-alpha are central. Could validate corticosteroid gene expression effects in HS tissue.

#### Alopecia Areata
- **Known treatments**: Baricitinib (JAK1/2 inhibitor), corticosteroids
- **Novel GOLDEN**: Azathioprine (immunosuppressant), Corticotropin/ACTH (melanocortin pathway)
- **Gene overlap**: MC2R (ACTH receptor) is a direct overlap for Corticotropin
- **Interesting**: ACTH has BOTH steroid-dependent and steroid-independent anti-inflammatory effects via melanocortin receptors (MC1R-MC5R). Hair follicle melanocytes express melanocortin receptors.

#### Psoriasis Vulgaris
- **Known treatments**: 27 drugs including adalimumab, methotrexate, cyclosporine, etanercept, infliximab, secukinumab, ustekinumab
- **Novel HIGH**: Methylprednisolone, Dexamethasone (30 shared genes!)
- **Gene overlap**: VEGFA, JUN, GSTP1, NOTCH1, HMOX1, DUSP22 — highly relevant to psoriasis pathogenesis

#### Toxic Epidermal Necrolysis (TEN)
- **Known treatments**: Only IVIG in our GT (1 drug!)
- **Novel GOLDEN**: Methylprednisolone (rank 1, score 8.9!), Dexamethasone, Betamethasone, Hydrocortisone
- **Gene overlap**: IL1B shared — IL-1 beta is key in TEN keratinocyte apoptosis
- **Note**: Steroid use in TEN is controversial (some evidence for early high-dose pulse)

#### Scleroderma / Systemic Sclerosis
- **Known treatments**: Bosentan (endothelin antagonist), Methylprednisolone, Rituximab
- **Novel GOLDEN**: Dexamethasone, Betamethasone, Prednisone — all with TGFB2, TNFAIP3, ITGB1 overlap
- **Ryland angle**: TGF-beta pathway is THE central driver of scleroderma fibrosis. Drug-target overlap with TGFB2 is highly relevant.

---

## Section 2: Summary Statistics

### Disease Coverage
| Category | Diseases with GOLDEN/HIGH Predictions | Total Predictions |
|----------|--------------------------------------|-------------------|
| Dermatological | 17 diseases | 84 predictions |
| Autoimmune | 23 diseases | 143 predictions |
| **Total** | **40 diseases** | **227 predictions** |

### Drug Class Distribution (novel GOLDEN/HIGH)
| Drug Class | Predictions | Key Drugs |
|------------|------------|-----------|
| Corticosteroids | 195 (86%) | Methylprednisolone, Dexamethasone, Betamethasone, etc. |
| Anti-TNF biologics | 9 (4%) | Adalimumab |
| Immunosuppressants | 5 (2%) | Azathioprine |
| Anti-CD20 | 3 (1%) | Rituximab |
| ACTH analogs | 3 (1%) | Corticotropin |
| Other | 12 (5%) | Methotrexate, Lidocaine, etc. |

### Model Observations
- **86% of novel GOLDEN/HIGH predictions are corticosteroids** — clinically reasonable (corticosteroids ARE used broadly for these conditions) but represents a "confirmation" rather than "discovery"
- **14% are mechanistically specific** — anti-TNF, anti-CD20, immunosuppressants. These are the most interesting for validation.
- **Etanercept → SLE** (MEDIUM) has the strongest molecular support: 9 shared genes across Fc receptor and complement pathways

---

## Section 3: Collaboration Opportunities

### What Ryland's Spatial Transcriptomics Can Provide

1. **Disease Gene Signature Validation**
   - Our model uses DRKG-derived gene associations (many diseases have very few: ichthyosis=8, TEN=2, HS=2)
   - Spatial transcriptomics from diseased skin tissue would provide **much richer** gene signatures
   - Priority diseases for gene expansion: ichthyosis, TEN, HS, eczema (only 2 genes each!)

2. **Drug-Target Expression Confirmation**
   - For predictions with gene overlap (Section 1B), Ryland can confirm whether shared genes are actually dysregulated in the target tissue
   - E.g., Is IL1B upregulated in hidradenitis suppurativa lesions? Is TGFB2 upregulated in scleroderma skin?

3. **Novel Drug Target Discovery**
   - If spatial transcriptomics reveals unexpected pathway activation in disease tissue, and that pathway is targetable by a known drug, we have a new repurposing hypothesis
   - Our KG has 11,656 drug-gene target associations that can be cross-referenced

### What Our Model Can Provide to Ryland

1. **Prioritized Drug Lists per Disease**
   - For any skin/inflammatory disease Ryland studies, we can generate a ranked list of candidate drugs with confidence tiers
   - Useful for experimental planning: which drugs to test in cell culture

2. **Drug-Gene-Disease Network Visualization**
   - For diseases in Ryland's portfolio, we can map the full DRKG neighborhood showing how drugs connect to diseases through genes/pathways

3. **Negative Controls**
   - Our FILTER tier (10% precision) and inverse indication database (drugs that CAUSE diseases) provide well-characterized negative controls for experiments

### Proposed Next Steps

1. **[LOW EFFORT]** Ryland reviews this document and identifies 3-5 diseases of highest interest
2. **[MEDIUM EFFORT]** We generate detailed drug-gene-disease network maps for those diseases
3. **[HIGH EFFORT]** Ryland provides spatial transcriptomics gene signatures for priority diseases → we integrate as features to improve predictions
4. **[HIGH EFFORT]** Identify 2-3 predictions testable in Ryland's cell culture models

---

## Section 4: Testable Predictions for Lab Validation

### Tier 1: Literature-Validated, High Confidence

| # | Drug | Disease | Confidence | Testability | Lit. Evidence |
|---|------|---------|------------|-------------|---------------|
| 1 | **Azathioprine** | Alopecia Areata | GOLDEN | High | 92.7% regrowth in 63-pt cohort. Hair follicle organoid testing possible. |
| 2 | **Corticotropin/ACTH** | Alopecia Areata | GOLDEN | High | Melanocortin receptors on hair follicle cells; ACTH upregulated in lesions. NO clinical trials yet — novel opportunity. |

### Tier 2: Mechanistic Support, Worth Investigating

| # | Drug | Disease | Confidence | Testability | Notes |
|---|------|---------|------------|-------------|-------|
| 3 | Corticosteroids | Scleroderma | GOLDEN | Medium | 5 shared genes incl. TGFB2, TNFAIP3 — central to fibrosis |
| 4 | Corticosteroids | TEN | GOLDEN | Low | IL1B overlap; controversial clinically (pulse steroids early) |
| 5 | Rituximab | JIA | HIGH | Low | In vivo model needed; used off-label in refractory cases |

### Tier 3: Interesting but Unproven

| # | Drug | Disease | Confidence | Testability | Notes |
|---|------|---------|------------|-------------|-------|
| 6 | Indomethacin | Sarcoidosis | MEDIUM | Medium | COX inhibition + AGTR1/FGF2 overlap |
| 7 | Rituximab | Eczema | MEDIUM | Medium | B-cell role in atopic inflammation? |

### HARMFUL Predictions (model errors — do NOT test)
Literature validation revealed **4 GOLDEN predictions that are actively harmful**:
- **Adalimumab → SLE**: Anti-TNF INDUCES lupus (12,080 FAERS reports, >90% serious)
- **Adalimumab → Myasthenia Gravis**: Anti-TNF CAUSES MG (case reports after 18mo RA treatment)
- **Adalimumab → Multiple Sclerosis**: Paradoxical demyelination (well-documented)
- **Etanercept → SLE** (MEDIUM): Same drug-induced lupus concern

These should be added to our inverse indication filter. Model confuses drug-disease KG associations (including adverse effects) with therapeutic relationships.

### Failed/Negative Predictions (model overestimates)
- **Adalimumab → GCA**: Failed Phase 2 RCT (no benefit). Tocilizumab is the proven biologic.
- **Adalimumab → GVHD**: Off-label use shows limited efficacy + infection risk.
- **Methotrexate → Actinic Keratosis**: Works for keratoacanthomas, not AK.

---

## Section 5: Technical Details

### Model Architecture
- **Knowledge Graph**: DRKG (Drug Repurposing Knowledge Graph) — 97K nodes, 5.8M edges, 107 edge types
- **Embeddings**: Node2Vec on DRKG → 128-dimensional vectors per drug/disease
- **Prediction**: kNN collaborative filtering (k=20 nearest diseases by embedding similarity)
- **Confidence**: 29+ hierarchy rules + safety filters + TransE cross-validation

### Evaluation
- **Holdout**: 20% of ground truth drug-disease pairs held out across 5 random seeds
- **R@30**: 37% recall at rank 30 (best fair model on DRKG)
- **Fair comparison**: Retrained without treatment edges → 26% R@30 (vs TxGNN 6.7-14.5%)

### Data Sources
- Ground Truth: Every Cure indication list (3,618 drug-disease pairs) + SIDER indications
- Knowledge Graph: DRKG (GNBR, Hetionet, DrugBank, STRING, etc.)
- Drug Targets: DrugBank + DRKG gene associations (11,656 drug-target pairs)
- Disease Genes: DRKG disease-gene associations (3,454 diseases)

### Known Limitations
1. **Corticosteroid dominance**: kNN collaborative filtering naturally amplifies broadly-used drug classes
2. **Self-referentiality**: 31.6% of diseases are 100% self-referential (predictions recapitulate training data)
3. **Small disease gene sets**: Many skin diseases have very few DRKG gene associations (ichthyosis=8, TEN=2)
4. **No transcriptomic data**: Current model uses only KG structure, not gene expression

---

*Generated: 2026-02-06*
*Model version: production_predictor.py (h540 — LA demotion)*
*Ground truth: expanded_ground_truth.json (post h529 GT quality audit)*
