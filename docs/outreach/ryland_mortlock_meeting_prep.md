# Ryland Mortlock Meeting Prep

*Meeting: Monday, February 10, 2026*
*Updated: February 7, 2026*

---

## Who is Ryland Mortlock?

**Position:** 4th year MD-PhD student at Yale School of Medicine, Genetics Department
**Lab:** Keith Choate Lab (Chair of Dermatology)
**Skills:** Data Science, R, Genomics, single-cell RNA-seq, spatial transcriptomics
**Nucleate Activator 2026:** Selected for biotech incubator program (New Haven branch) — entrepreneurial interest

### Research Focus
- **Genetic skin diseases** — inflammation in epidermal differentiation disorders
- Uses animal models, spatial transcriptomics, 3D cell culture systems
- Barrier dysfunction and cellular interactions in the epidermis
- Interested in AI/ML for health (reposted Nick Furlotte's ChatGPT Health article)

---

## His Key Publications

### 1. EMP2 → Erlotinib Discovery (PNAS, Aug 2025) — DRUG REPURPOSING SUCCESS

**Disease:** Progressive Symmetric Erythrokeratoderma (PSEK) — severely thickened, scaly skin

**The Discovery:**
- EMP2 gene variant causes aberrant EGFR/FAK/Src activation
- **Repurposed erlotinib (cancer drug) → marked clinical improvement**
- Patient's skin disease resolved at many sites

**Why This Matters:**
- They already did exactly what Open-Cure does — but manually, one disease at a time
- Workflow: spatial transcriptomics → pathway identification → drug match → clinical test
- **We can help them scale this to other genetic skin diseases computationally**

### 2. SLURP1 Paper (BJD, Feb 2025)

**Diseases:** Palmoplantar keratoderma, Progressive symmetric erythrokeratoderma (PSEK)

**Key Finding:** SLURP1 variants → increased NF-kB signaling and innate immune activity

**Treatment Challenge:** No curative treatment. Retinoids have side effects and recurrence.

**Opportunity:** We have **1,022 drugs targeting NF-kB pathway genes** in our database — far more candidates than manual review could identify.

### 3. Immune Biomarkers Paper (JID, 2023)

**Diseases:** Psoriasis, Atopic Dermatitis

**Key Message:** Despite revolutionary biologics, no biomarkers predict which therapy works for which patient. Our confidence tier system and mechanism annotations could help stratify patients.

---

## What's Changed Since Initial Outreach

Ryland received our initial message describing the 37% R@30 system. Since then, substantial improvements:

### Current System Performance (Feb 7, 2026)

| Tier | Holdout Precision | Predictions | Meaning |
|------|-------------------|-------------|---------|
| **GOLDEN** | 70.1% ± 5.6% | 507 | Strong multiple-signal evidence |
| **HIGH** | 51.5% ± 5.4% | 1,123 | Good evidence, validated at scale |
| **MEDIUM** | 40.6% ± 2.8% | 2,162 | Moderate — worth investigating |
| **LOW** | 13.1% ± 0.8% | 6,253 | Weak evidence |
| **FILTER** | 10.0% ± 0.5% | 8,672 | Known issues or contradictions |

**Total: 18,717 predictions** across 455 diseases, evaluated against 57,495 ground truth pairs.

### Key Advances Since Outreach
1. **Literature mining (h731):** Automated ClinicalTrials.gov + medical evidence mining. Predictions with STRONG_EVIDENCE achieve **78.7% holdout precision** — the strongest independent predictor we've found.
2. **Drug name aliasing (h686):** Recovered 85 ground truth pairs by resolving drug name variants (INN names, salt forms, combo products). Reduced MEDIUM tier variance from ±4.7% to ±1.8%.
3. **Known vs novel split (h736/h738):** Deliverable now distinguishes "known indication" from "novel prediction" — Ryland can filter directly to novel candidates.
4. **Safety filters:** 163 inverse indication pairs (drugs that CAUSE diseases), 66 non-therapeutic compounds filtered, GT quality audits removing NLP extraction errors.
5. **30 confidence rules:** Hierarchy matching, mechanism support, TransE consilience, cancer type matching, antimicrobial spectrum, drug class isolation — all holdout-validated.

---

## Dermatological Predictions (Current)

### Summary: 29 diseases, 855 predictions total

| Tier | Total | Novel | Key Drug Classes |
|------|-------|-------|-----------------|
| GOLDEN | 39 | 20 | Corticosteroids, azathioprine, ACTH |
| HIGH | 98 | 50 | Corticosteroids (SOC), tetracyclines |
| MEDIUM | 92 | 60 | Mixed — tetracyclines, salicylic acid, NSAIDs |
| LOW | 196 | 144 | Broader drug classes |
| FILTER | 430 | 356 | Filtered (noise, inverse indications) |

**Literature evidence on derm predictions:** 12 with STRONG_EVIDENCE, 4 MODERATE, 5 WEAK.

### Predictions for Choate Lab Diseases

| Disease | Top Novel Predictions | Tier | Notes |
|---------|----------------------|------|-------|
| **Atopic dermatitis** | Antihistamines, Azathioprine, Montelukast | HIGH | Well-studied disease, high GT coverage |
| **Psoriasis vulgaris** | Dexamethasone (30 shared genes!), Methylprednisolone | HIGH | VEGFA, JUN, NOTCH1 gene overlap |
| **Ichthyosis vulgaris** | Tetracycline, Doxycycline, Minocycline | MEDIUM | Likely kNN artifact (co-prescription) |
| **Lamellar ichthyosis** | Salicylic acid (already standard of care) | MEDIUM | Only 8 disease genes in DRKG |
| **Ichthyosis (general)** | Methotrexate (1970s case reports, abandoned) | MEDIUM | Failed historically |
| **Alopecia areata** | Azathioprine (GOLDEN), Corticotropin/ACTH (GOLDEN) | GOLDEN | Azathioprine: 92.7% regrowth in 63-pt cohort |
| **Hidradenitis suppurativa** | Corticosteroids with IL1B gene overlap | GOLDEN | IL-1beta central to HS pathogenesis |
| **Scleroderma** | Dex/Beta with TGFB2 + TNFAIP3 gene overlap | GOLDEN | TGF-beta is THE fibrosis driver |

### Honest Assessment: Ichthyosis Predictions Are Weak

Tetracycline-class drugs for ichthyosis are a **kNN co-occurrence artifact**: these drugs are co-prescribed for other dermatological conditions (acne, rosacea), and kNN picks up the dermatology pattern rather than biological relevance to cornification defects. This is important to be transparent about.

**This is actually the best case for collaboration:** our computational approach hits a wall where domain expertise is needed to distinguish true signal from co-prescription noise.

---

## The EGFR Gap — Primary Collaboration Hook

### The Problem
**No EGFR inhibitor is predicted for ANY dermatological disease** in our system.

**Why:** EGFR inhibitors (erlotinib, gefitinib, cetuximab, osimertinib) are cancer-only drugs in DRKG. Our cross-domain isolation filter (h271) correctly blocks them from non-cancer predictions because they have zero non-cancer GT. The kNN system can only recommend drugs that already treat at least one disease similar to the target — a structural limitation for drugs used exclusively in one domain.

### The Opportunity
Ryland proved erlotinib works for PSEK. His spatial transcriptomics identifies EGFR dysregulation in skin diseases. Our database has **392 compounds targeting EGFR** and **924 drugs targeting the broader ErbB pathway**. But we need his biological expertise to bridge the gap.

### What We Can Offer for EGFR
- **185 DrugBank drugs directly targeting EGFR** — exported in `egfr_direct_drugs.xlsx`
- **924 drugs targeting ErbB/EGFR pathway genes** — exported in `egfr_pathway_drugs.xlsx`
- Drug-gene-pathway network for any disease gene he identifies via spatial transcriptomics

### Discussion Points
1. Which genetic skin diseases show EGFR dysregulation beyond EMP2/PSEK?
2. Could anti-EGFR drugs work for specific ichthyosis subtypes with hyperproliferation?
3. Can spatial transcriptomics data help us score EGFR pathway predictions computationally?

---

## NF-kB Pathway Analysis (for SLURP1)

SLURP1 has only 2 direct drug interactions in DRKG. But expanding to the NF-kB pathway (hsa04064):

- **105 genes** in NF-kB pathway
- **86 genes** with known drug targets
- **1,022 DrugBank drugs** targeting these genes
- Top by target coverage: Dexamethasone (20 targets), Doxorubicin (19), Staurosporine (19)
- Key NF-kB genes: NFKB1, NFKB2, RELA, RELB, IKBKB, TNFRSF1A, TRAF6, MYD88, IRAK1

All exported in `nfkb_pathway_drugs.xlsx` with 3 sheets (Summary, Drugs, Drug-Gene Pairs).

---

## Testable Predictions for Wet-Lab Validation

### Tier 1: Literature-Validated, High Confidence

| Drug | Disease | Tier | Evidence | Cell Culture Feasibility |
|------|---------|------|----------|-------------------------|
| **Azathioprine** | Alopecia areata | GOLDEN | 10-year cohort, 92.7% regrowth | Hair follicle organoids |
| **Corticotropin/ACTH** | Alopecia areata | GOLDEN | ACTH upregulated in lesions, melanocortin receptors on follicle cells | Hair follicle organoids |
| **Montelukast** | Idiopathic pulmonary fibrosis | HIGH | Preclinical anti-fibrotic evidence | Fibroblast cultures |

### Tier 2: Mechanistic Support

| Drug | Disease | Tier | Gene Overlap | Notes |
|------|---------|------|-------------|-------|
| Corticosteroids | Scleroderma | GOLDEN | TGFB2, TNFAIP3, ITGB1 | Central fibrosis pathway |
| Corticosteroids | Hidradenitis suppurativa | GOLDEN | IL1B | Key HS cytokine |
| Dexamethasone | Psoriasis | HIGH | 30 shared genes | VEGFA, JUN, NOTCH1 |

### Known Harmful Predictions (Model Errors — Demonstrates Rigor)

We identified and filtered 4 predictions where the drug CAUSES the disease:
- **Adalimumab → SLE:** Anti-TNF induces lupus (12,080 FAERS reports)
- **Adalimumab → Myasthenia Gravis:** Anti-TNF causes MG (case reports)
- **Adalimumab → Multiple Sclerosis:** Paradoxical demyelination
- **Adalimumab → GCA:** Failed Phase 2 RCT (no benefit vs placebo)

Sharing these errors transparently demonstrates scientific rigor and the value of human expert review.

---

## Collaboration Opportunities

### What He Has That We Need
1. **Disease gene signatures** from spatial transcriptomics — many diseases have <10 genes in DRKG
2. **Domain expertise** to filter kNN artifacts from genuine candidates
3. **Cell culture validation** capability (3D models, organoids)
4. **EGFR/NF-kB pathway expertise** to bridge our prediction gap

### What We Have That He Needs
1. **Drug-target databases** at scale (11,656 drug-gene pairs, 392 EGFR compounds, 1,022 NF-kB drugs)
2. **Prioritized drug lists** with holdout-validated confidence tiers
3. **Automated pipeline** that runs 24/7 (autonomous research agent with 590+ hypotheses tested)
4. **Mechanism tracing** from drug → target → pathway → disease
5. **Literature evidence mining** automated via ClinicalTrials.gov

### Proposed Collaboration Structure

**Immediate (this meeting):**
- Share deliverable + derm-specific exports
- Ryland identifies 3-5 highest-priority diseases from Choate lab
- Discuss EGFR gap and SLURP1/NF-kB candidates

**Short-term (Feb-Mar):**
- We generate detailed drug-gene-disease networks for his priority diseases
- He provides spatial transcriptomics gene lists → we cross-reference against drug targets
- Identify 2-3 predictions testable in his cell culture models

**Strategic:**
- The "variant → pathway → drug" pipeline at scale — exactly the EMP2 story, automated
- Potential Nucleate Activator synergy — computational engine for rare disease drug repurposing
- Co-authored publication on computational + experimental validation workflow

---

## Questions to Ask

1. **Which diseases does the Choate lab most want drug candidates for right now?**
2. **What genes/pathways have you identified as causal but lack treatments?** (Beyond SLURP1/EMP2)
3. **Would spatial transcriptomics gene lists be something you could share?** We can cross-reference against 11,656 drug-target pairs.
4. **Interest in an automated "variant → pathway → drug" pipeline?** This is EMP2 at scale.
5. **What's the Nucleate project?** Could Open-Cure's computational platform be relevant?
6. **What validation capacity do you have?** Which cell models, what throughput?

---

## What He Already Knows

From the initial outreach message:
- Autonomous research agent concept
- 37% R@30 baseline, 60% theoretical ceiling (now with refined confidence tiers)
- "Similar diseases share treatments" approach
- Validated predictions: Rituximab→MS, Lovastatin→MM, Empagliflozin→Parkinson's
- The ceiling-breaking opportunity with external data

---

## Key Message

> "You proved drug repurposing works with EMP2 → erlotinib.
> We can help you do that systematically for other genetic skin diseases.
>
> You have the domain expertise, spatial transcriptomics, and cell culture models.
> We have the computational infrastructure, drug-target databases, and validated confidence system.
> Together: identify candidates for diseases that currently have no treatment."

---

## Files to Share

| File | Content |
|------|---------|
| `data/deliverables/drug_repurposing_predictions_with_confidence.xlsx` | All 18,717 predictions with confidence tiers, literature evidence |
| `data/exports/skin_disease_predictions.xlsx` | 855 dermatological predictions |
| `data/exports/ichthyosis_predictions.xlsx` | Ichthyosis-specific (5 disease types) |
| `data/exports/nfkb_pathway_drugs.xlsx` | 1,022 NF-kB pathway drugs (for SLURP1) |
| `data/exports/egfr_direct_drugs.xlsx` | 185 direct EGFR-targeting drugs |
| `data/exports/egfr_pathway_drugs.xlsx` | 924 ErbB/EGFR pathway drugs |
| `data/exports/ryland_analysis_summary.md` | Analysis guide and methodology |

---

## Supporting Analysis Documents

| File | Content |
|------|---------|
| `data/analysis/h661_ryland_collaboration_prep.md` | Detailed derm prediction analysis |
| `data/analysis/h408_ryland_collaboration_brief.md` | Full skin disease brief with gene overlap |

---

*Prepared 2026-02-01, updated 2026-02-07 with latest tier system, literature mining, and deliverable.*
