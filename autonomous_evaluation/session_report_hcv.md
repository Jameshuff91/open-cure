# Evaluation Enhancement Research Session: Hepatitis C

**Date**: 2026-01-19
**Disease**: Hepatitis C (MESH:D006526)
**Status**: COMPLETE ✓

## Summary

Researched 10 drug candidates predicted by the model for Hepatitis C treatment. Found **0 CONFIRMED** drugs among the top candidates, triggering the consecutive novel threshold stopping criterion (10 consecutive NOVEL > 5 threshold).

## Research Methodology

1. **Auto-classification by name patterns** - Checked for antiviral naming patterns
2. **Web search for verification** - Searched clinical trial databases and medical literature
3. **Evidence-based classification** - Required clear evidence for CONFIRMED status

## Results

### Classification Breakdown
- **CONFIRMED**: 0
- **NOVEL**: 10 (100%)

### Drugs Researched (Ranked by Model Score)

1. **Amdoxovir (DB00718)** - Score: 0.9657 - **NOVEL**
   - Nucleotide analog approved for Hepatitis B (HBV), not HCV
   - All clinical trials were for HBV treatment
   - Evidence: NEJM, DrugBank

2. **Camicinal (DB12567)** - Score: 0.9846 - **NOVEL**
   - Motilin agonist for gastroparesis
   - Gastrointestinal motility agent, not antiviral
   - Evidence: DrugBank

3. **Coproporphyrin I (DB03727)** - Score: 0.9827 - **NOVEL**
   - Porphyrin metabolite that accumulates in HCV infection
   - Consequence of disease, not a treatment
   - Evidence: PLOS One, PubMed 29856826

4. **CYT007-TNFQb (DB05758)** - Score: 0.9737 - **NOVEL**
   - TNF-targeting vaccine for autoimmune diseases
   - Evidence: DrugBank

5. **Urea-14C (DB09513)** - Score: 0.9650 - **NOVEL**
   - Diagnostic agent for H. pylori breath test
   - Not therapeutic, wrong pathogen
   - Evidence: PMC8213946

6. **Benzylpenicillin (DB01053)** - Score: 0.9554 - **NOVEL**
   - Penicillin G antibiotic for bacterial infections
   - Not antiviral
   - Evidence: NCBI Bookshelf

7. **Methotrexate (DB00563)** - Score: 0.9547 - **NOVEL**
   - Immunosuppressant for RA and cancer
   - Studies only assess safety in HCV+ patients with RA
   - Evidence: PubMed 24429167

8. **Chlorambucil (DB00291)** - Score: 0.9544 - **NOVEL**
   - Alkylating chemotherapy agent
   - No antiviral properties
   - Evidence: HCV treatment guidelines

9. **Bedoradrine (DB05590)** - Score: 0.9528 - **NOVEL**
   - β2-agonist for asthma
   - Evidence: DrugBank, Wikipedia

10. **Raxatrigine (DB11706)** - Score: 0.9527 - **NOVEL**
    - Sodium channel blocker for bipolar disorder
    - Evidence: DrugBank

## Key Findings

### Model Performance
- The model incorrectly predicted multiple non-antiviral drugs for HCV
- High confidence scores (0.95-0.98) on incorrect predictions
- Systematic errors include:
  - Wrong viral target (Adefovir for HBV not HCV)
  - Non-antiviral mechanisms (antibiotics, chemotherapy, β-agonists)
  - Disease biomarkers confused as treatments (Coproporphyrin)
  - Diagnostics confused as therapeutics (Urea-14C)

### Comparison to Every Cure
- Every Cure has 12 known HCV drugs
- Model's top 10 candidates: 0 overlap with known treatments
- This suggests very poor model performance on HCV

### Notable Errors

1. **Adefovir (DB00718)** - Wrong hepatitis virus
   - Approved for HBV, not HCV
   - Despite being nucleotide analog, specificity matters

2. **Coproporphyrin (DB03727)** - Disease biomarker
   - HCV causes coproporphyrin accumulation
   - Model confused consequence with treatment

3. **Urea-14C (DB09513)** - Wrong diagnostic
   - H. pylori test, not HCV
   - Model confused diagnostic agents

## Stopping Criteria Met

✓ 10 consecutive NOVEL classifications (threshold: 5)
✓ Research complete for Hepatitis C
✗ No confirmed drugs found to add to ground truth

## Next Steps

Continue research with next disease: **Tuberculosis** (Task ID 3)

## Sources

All research backed by:
- [DrugBank](https://go.drugbank.com)
- [PubMed/PMC](https://pubmed.ncbi.nlm.nih.gov)
- [NEJM](https://www.nejm.org)
- [Clinical Trials](https://clinicaltrials.gov)
- Medical literature and FDA resources
