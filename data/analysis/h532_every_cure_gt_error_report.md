# Every Cure Indication List Error Report

## Summary

14 entries in the Every Cure indication list appear to be **inverse indications** — cases where the drug is known to CAUSE or WORSEN the listed disease, rather than treat it. These entries likely entered the dataset through automated extraction from drug labels, where adverse effects, contraindications, or warnings sections were confused with indication sections.

## Error Categories

### 1. Statins → Diabetes (6 entries)

Statins increase the risk of type 2 diabetes mellitus by 10-36% (2024 Lancet meta-analysis). The FDA added a diabetes warning to all statin labels in 2012.

| Drug | Disease | Evidence |
|------|---------|----------|
| Atorvastatin | type 2 diabetes mellitus | FDA label warning since 2012 |
| Amlodipine/atorvastatin combination | type 2 diabetes mellitus | Atorvastatin component |
| Lovastatin | diabetes mellitus | Class effect |
| Lovastatin | poorly controlled diabetes mellitus | Class effect |
| Simvastatin | Diabetes | Class effect |
| Ezetimibe/simvastatin | Diabetes | Simvastatin component |

**Source:** Reith C, et al. Lancet 2024. "Effect of statin therapy on muscle symptoms: an individual participant data meta-analysis."

### 2. ACE Inhibitors → Angioedema (2 entries)

ACE inhibitors cause angioedema through bradykinin accumulation. This is a well-known class adverse effect with FDA black box warning. Incidence: 0.1-0.7% of patients.

| Drug | Disease | Evidence |
|------|---------|----------|
| Benazepril | angioedema | FDA black box warning |
| Quinapril | angioedema | FDA black box warning |

### 3. Thyroid Hormones → Hyperthyroidism (3 entries)

Thyroid hormones (T3, T4) are indicated for hypothyroidism. Administering them to patients with hyperthyroidism would cause thyrotoxicosis.

| Drug | Disease | Evidence |
|------|---------|----------|
| Liothyronine | hyperthyroidism | T3 CAUSES hyperthyroidism |
| Liothyronine | Subclinical hyperthyroidism | Same mechanism |
| Levothyroxine/liothyronine | hyperthyroidism | Combination thyroid hormone |

### 4. Estrogens → Breast Cancer (1 entry)

The Women's Health Initiative (WHI) trial demonstrated that conjugated estrogens increase breast cancer risk by 26%. FDA black box warning on all estrogen products.

| Drug | Disease | Evidence |
|------|---------|----------|
| Conjugated estrogens | breast cancer | WHI trial (JAMA 2002) |

### 5. Corticosteroids → Adrenocortical Insufficiency (2 entries)

Long-acting corticosteroids cause secondary adrenocortical insufficiency through hypothalamic-pituitary-adrenal (HPA) axis suppression. While short-acting CS (hydrocortisone, cortisone) are legitimate replacement therapy for adrenal insufficiency, dexamethasone and methylprednisolone are not used for this purpose and instead cause it.

| Drug | Disease | Evidence |
|------|---------|----------|
| Dexamethasone | adrenocortical insufficiency | HPA suppression; NOT replacement therapy |
| Methylprednisolone | adrenocortical insufficiency | HPA suppression; NOT replacement therapy |

**Note:** Prednisolone and prednisone entries for adrenocortical insufficiency are CORRECT — they are FDA-approved for this indication as replacement therapy.

## Methodology

These errors were identified by cross-referencing the Every Cure indication list against:
1. Our curated inverse indication database (157 drug-disease pairs, 67 drugs)
2. FDA label warnings and contraindications
3. Published clinical evidence (RCTs, meta-analyses, FDA safety communications)

## Recommended Actions

1. **Remove** the 14 entries listed above from the indication list
2. **Audit** other statin, ACE inhibitor, and hormone-related entries for similar patterns
3. **Consider** adding an automated check for drug label "Warnings" vs "Indications" section during data curation

## Contact

This report was generated as part of the Open-Cure drug repurposing research project.
Date: 2026-02-06
