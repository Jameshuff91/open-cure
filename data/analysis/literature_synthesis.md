# Drug Repurposing Literature Synthesis
**Generated:** 2026-01-25

## Top Tier Candidates

### 1. Magnesium → Heart Failure ⭐ HIGHEST PRIORITY

**Summary:** Low-cost, safe intervention with compelling mortality data.

**Key Studies:**

1. **Veterans HF Cohort (2025)** - [JAHA](https://www.ahajournals.org/doi/10.1161/JAHA.124.038870)
   - n=19,800 (9,900 users, 9,900 matched controls)
   - **HR 0.81 [0.77-0.86], p<0.0001** for all-cause mortality
   - Median follow-up: 0.7 years

2. **HFpEF Critically Ill (2025)** - [Nature Sci Rep](https://www.nature.com/articles/s41598-025-85931-1)
   - Propensity-matched cohort from MIMIC-IV
   - n=1,970 (985 per group)
   - Primary: 28-day mortality
   - Secondary: ICU and 1-year mortality

3. **Mechanism Studies:**
   - Reduces isolated VPCs, couplets, and non-sustained VT
   - Lowers mean arterial pressure
   - Reduces systolic vascular resistance

**Limitations:**
- Observational designs
- March 2025 preprint suggests potential harm in acute HF
- Need prospective RCT

**Cost:** ~$10-20/month for supplements

---

### 2. Verapamil → Type 2 Diabetes ⭐ HIGH PRIORITY

**Summary:** CCB with RCT evidence for beta-cell preservation.

**Key Studies:**

1. **Meta-analysis (2025)** - [PubMed](https://pubmed.ncbi.nlm.nih.gov/40111679/)
   - 8 RCTs, n=1,100 patients
   - **HbA1c -0.45% [95% CI -0.66, -0.23], p<0.001**
   - C-peptide AUC +0.27 pmol/mL, p<0.0001
   - Works in both T1D and T2D

2. **R-Form Verapamil RCT (2022)** - [JCEM](https://academic.oup.com/jcem/article/107/10/e4063/6653488)
   - R-Vera 300mg/day + metformin
   - Greater HbA1c reduction vs placebo
   - Higher proportion achieving <7.0% target

3. **Mechanism:**
   - Reduces TXNIP (thioredoxin-interacting protein)
   - TXNIP causes beta-cell apoptosis
   - Also elevates CCK (incretin, improves mitochondrial respiration)

**Active Trials:** NCT04233034 (Ver-A-T1D)

---

### 3. Methotrexate → Systemic Lupus Erythematosus

**Summary:** Established off-label use with RCT support.

**Key Studies:**

1. **Systematic Review (2014)** - [Lupus](https://pubmed.ncbi.nlm.nih.gov/24399812/)
   - 9 studies (3 RCTs, 6 observational)
   - **SLEDAI reduction: OR 0.444, p=0.001**

2. **RCT (1999)** - [PubMed](https://pubmed.ncbi.nlm.nih.gov/10381042/)
   - MTX 15-20mg/week for 6 months
   - Articular complaints: 16 placebo vs 1 MTX (p<0.001)
   - Cutaneous lesions: 16 placebo vs 3 MTX (p<0.001)
   - Prednisone reduction: 13 MTX vs 1 placebo (p<0.001)

**Status:** Already used clinically, primarily for cutaneous/articular manifestations

---

## Confirmed False Positives

### Digoxin → Type 2 Diabetes ❌

**Why it's wrong:**
- Na+/K+-ATPase inhibition reduces GLUT4 translocation
- Spigset 1999: HbA1c worsened (5-6% → 7-8%)
- DIG trial (n=6,800): No diabetes benefit
- 8 "trials" were DDI studies, not treatment

### Simvastatin → Type 2 Diabetes ❌

**Why it's wrong:**
- Statins INCREASE T2D risk
- Lancet 2024: HR 1.12-1.44 for new-onset T2D
- 33 trials are for CV protection IN diabetics
- Inverse indication pattern

### Pembrolizumab → Ulcerative Colitis ❌

**Why it's wrong:**
- Checkpoint inhibitors CAUSE immune-related colitis
- Colitis is a known irAE of PD-1 inhibitors
- Model detected treatment for the adverse event, not the disease

---

## Methodology Notes

### Validation Score Formula
```
trial_score = min(trial_count / 50, 1.0) × 0.6
pub_score = min(pub_count / 500, 1.0) × 0.4
validation_score = trial_score + pub_score
```

### Evidence Categories
- **Strong (≥0.5):** 39.6% - Likely already approved/established
- **Moderate (0.2-0.5):** 22.8% - BEST repurposing candidates
- **Weak (<0.2):** 21.0% - Needs investigation
- **None:** 16.6% - Truly novel or spurious

### Confounding Patterns Detected
1. Inverse indication (drug causes disease)
2. Mechanism mismatch (drug worsens pathophysiology)
3. Cardiac-metabolic comorbidity (HF + T2D = 75% overlap)
