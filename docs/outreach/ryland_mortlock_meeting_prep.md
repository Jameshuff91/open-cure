# Ryland Mortlock Meeting Prep

*Meeting scheduled: Week of Feb 3, 2026*

---

## Who is Ryland Mortlock?

**Position:** 4th year MD-PhD student at Yale School of Medicine, Genetics Department
**Lab:** Keith Choate Lab (Chair of Dermatology)
**Skills:** Data Science, R, Genomics, single-cell RNA-seq, spatial transcriptomics

### Research Focus
- **Genetic skin diseases** - specifically inflammation in epidermal differentiation disorders
- Uses animal models, spatial transcriptomics, 3D cell culture systems
- Interested in barrier dysfunction and cellular interactions

### Recent Activity
- Selected for **2026 Nucleate Activator** program (biotech incubator) - indicates entrepreneurial interest
- Interested in AI/ML for health (reposted Nick Furlotte's ChatGPT Health post)

---

## His Key Publications (Highly Relevant to Open-Cure)

### 1. EMP2 → Erlotinib Discovery (PNAS, Aug 2025) ⭐ DRUG REPURPOSING SUCCESS

**Disease:** Progressive Symmetric Erythrokeratoderma (PSEK) - severely thickened, scaly skin

**The Discovery:**
- Found that EMP2 gene variant causes aberrant EGFR/FAK/Src activation
- **Repurposed erlotinib (cancer drug) → marked clinical improvement**
- Patient's skin disease resolved at many sites

**Why This Matters for Us:**
- They already did exactly what Open-Cure does - but manually
- Used spatial transcriptomics → identified pathway → matched to existing drug
- **We can help them scale this approach to other genetic skin diseases**

### 2. SLURP1 Paper (BJD, Feb 2025)

**Diseases:** Palmoplantar keratoderma, Progressive symmetric erythrokeratoderma

**Key Finding:** SLURP1 variants → increased NF-kB signaling and innate immune activity

**Treatment Challenge:**
- No curative treatment exists
- Retinoids have side effects and recurrence on cessation
- Topical calcipotriol shows some promise

**Opportunity:** We have **325 drugs targeting NF-kB** in our database. Could identify repurposing candidates.

### 3. Immune Biomarkers Paper (JID, 2023)

**Diseases:** Psoriasis, Atopic Dermatitis

**Key Message:** Despite revolutionary biologics, **no biomarkers exist to predict which therapy works for which patient**

**Opportunity:** Our model could help match patient molecular profiles to drug mechanisms.

---

## What Open-Cure Can Offer Ryland

### 1. Predictions for Diseases He Studies

| Disease | Our Predictions | Confidence | Novel Candidates |
|---------|-----------------|------------|------------------|
| **Atopic dermatitis** | 30 drugs | ALL HIGH | 18 novel |
| **Psoriasis vulgaris** | 30 drugs | ALL HIGH | 23 novel |
| **Ichthyosis** (Choate lab focus) | 30 drugs | MEDIUM | 29 novel |
| **Lamellar ichthyosis** | 30 drugs | MEDIUM | 29 novel |
| **Ichthyosis vulgaris** | 30 drugs | MEDIUM | 30 novel |
| Eczema | 30 drugs | ALL HIGH | 29 novel |
| Dermatitis | 30 drugs | ALL HIGH | 22 novel |

### 2. Top Novel Drug Predictions

**For ICHTHYOSIS (Choate Lab's Major Focus):**
- Salicylic acid (score: 2.42)
- Methotrexate (score: 1.43) - immunosuppressant
- Tetracyclines (Doxycycline, Minocycline) - anti-inflammatory
- Diclofenac (score: 0.77) - NSAID

**For ATOPIC DERMATITIS (HIGH confidence):**
- Betamethasone (score: 4.63)
- Levocetirizine, Desloratadine, Cetirizine (antihistamines)
- Azathioprine (score: 1.37) - immunosuppressant
- Montelukast, Pranlukast (leukotriene inhibitors)

### 3. Pathway-Based Drug Discovery (NEW)

Since Ryland's approach (EMP2 paper) was:
1. Identify dysregulated pathway in patients
2. Find drug targeting that pathway
3. Test in clinic

**We can help with step 2 at scale:**
- We have **395 drugs targeting EGFR** (like erlotinib they used)
- We have **325 drugs targeting NF-kB** (relevant for SLURP1)
- We can trace Drug → Target → Pathway → Disease mechanisms

### 4. Mechanism Tracing for Rare Diseases

For diseases NOT in our predictions (keratoderma, erythrokeratoderma), we can:
1. Input the disease genes he identifies (like EMP2, SLURP1)
2. Find drugs targeting those genes or pathways
3. Rank by biological plausibility

---

## Specific Collaboration Opportunities

### Immediate (This Week)
1. **Run predictions for his specific diseases** - any MESH-mapped genetic skin condition
2. **Generate EGFR/NF-kB drug candidates** for SLURP1 patients (similar to erlotinib success)
3. **Share our ichthyosis predictions** - directly relevant to Choate lab

### Short-term (This Month)
1. **Point the research agent at genetic skin diseases** - let it generate hypotheses 24/7
2. **Validate predictions against his clinical knowledge** - which are plausible?
3. **Integrate spatial transcriptomics data** - he has this, we could use disease gene expression for similarity

### Strategic (Nucleate Activator Potential)
The EMP2 → erlotinib success story is exactly what drug repurposing looks like in practice:
- Academic discovery of mechanism
- Matching to existing approved drug
- Clinical validation

**Open-Cure could be the computational engine for this approach applied to rare genetic skin diseases.**

---

## Questions to Ask Ryland

1. **Which specific diseases does the Choate lab most want drug candidates for?** We can run targeted predictions.

2. **What genes/pathways have they identified as causal but don't have treatments?** (Like SLURP1)

3. **Would spatial transcriptomics data from patients help improve disease similarity?** This could break our 37% ceiling.

4. **Is there interest in an automated pipeline for "variant → pathway → drug candidate"?** This is the EMP2 story at scale.

5. **What's the Nucleate project about?** Could Open-Cure be relevant?

---

## What He Already Knows

From the initial outreach message, Ryland knows:
- The autonomous research agent concept
- 37% R@30 baseline, 60% theoretical ceiling
- The "similar diseases share treatments" approach
- Validated predictions (Rituximab→MS, Lovastatin→MM, Empagliflozin→Parkinson's)
- The ceiling-breaking opportunity with external data

---

## Key Message for the Meeting

> "You already did drug repurposing successfully with EMP2 → erlotinib.
> We can help you do that systematically for other genetic skin diseases.
>
> You have the domain expertise and patient data.
> We have the computational infrastructure and drug-target databases.
> Together we can identify candidates for diseases that currently have no treatment."

---

## Files to Share

| File | Content |
|------|---------|
| `docs/impressive_evidence_report.md` | Full methodology + validated predictions |
| `docs/mechanism_report.md` | How we trace biological mechanisms |
| `data/deliverables/drug_repurposing_predictions_with_confidence.xlsx` | All 13K predictions |
| `scripts/trace_mechanism_paths.py` | The mechanism tracing code |

---

*Prepared 2026-02-01*
