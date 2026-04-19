# Paper v2 Errata

Tracking incorrect claims in the preprint + the corrected versions to ship in v2.
Every entry: original claim → evidence of error → corrected claim → source of fix.

## E1 (h1101) — Dantrolene crown-jewel indication relabel

**Section 3.5, table of retrospectively corroborated predictions.**

**Original (incorrect):**

| Drug       | Disease        | Path                              | Evidence                  |
|------------|---------------|-----------------------------------|---------------------------|
| Dantrolene | Heart failure | Drug → Drug → Disease             | RCT P=0.034, 66% VT reduction |

**Evidence of error (positive-control + predict() audit, 2026-04-19):**
- The model does **not** predict Dantrolene for heart failure (D006333). Heart failure has 117 total predictions (include_filtered=True) and Dantrolene is not among them — no kNN neighbors of D006333 surface Dantrolene.
- The model **does** predict Dantrolene for the arrhythmia family:
  - Ventricular tachycardia (D017180): rank 34, FILTER (`rank_over_20`)
  - Ventricular fibrillation (D014693): rank 36, FILTER (`rank_over_20`)
  - Tachycardia (D013610): rank 28, FILTER (`rank_over_20`)
  - Arrhythmia (D001145): rank 67, FILTER (`rank_over_20`), `mech=True`
- The RCT cited (Zamiri et al. 2014, *Circulation*, PMID 25122731) randomized **heart failure patients** but the primary outcome was **VT reduction** (P=0.034, 66% reduction). The paper's Section 3.5 row conflated the cohort (heart failure) with the outcome (ventricular tachycardia).

**Corrected row (to ship in v2):**

| Drug       | Disease                                  | Path                              | Evidence                        | Model rank / tier |
|------------|------------------------------------------|-----------------------------------|----------------------------------|-------------------|
| Dantrolene | Ventricular tachycardia (in HF patients) | Drug → Target (RyR2) → Disease    | RCT Zamiri et al. 2014, P=0.034, 66% VT reduction | rank 34, FILTER (`rank_over_20`) |

**Why the model ranks it FILTER (not a rule misfire):**
The tier is set by the structural `rank_over_20` rule, which is a kNN floor — not a tier-rule bug (h1100 confirmed this rule is not carved out for known indications because it is a structural signal, not a safety filter). Dantrolene is correctly classified (no inverse indication, no targeted-therapy demotion, not a corticosteroid). The kNN ranker simply places it outside the top 20 because disease-similarity signals for VT/VF/tachycardia do not cluster strongly around Dantrolene's neighbors. This is a **ranker weakness**, not a rule error; the tier system correctly reflects low-confidence support.

**Paper framing change:**
Describe this prediction as "surfaced by the model within the top 100, filtered to LOW/FILTER tier by the structural rank-floor rule, corroborated retrospectively by RCT evidence" — honest about the weak model signal while preserving the scientific point that the model detects the signal at all.

**No code change required.** Dantrolene drug-class assignment is correct. The VT/VF/tachycardia disease category is `cardiovascular`, correctly handled. The `rank_over_20` rule should not be carved out per h1100's design principle.

**Residual noted for future work:**
`drug_cancer_types[DB01219] = {'solid_tumor'}` is a minor DRKG artifact (Dantrolene is not a cancer drug — the assignment likely traces to a DRKG edge from a cancer-adjacent paper). Does not affect VT-related predictions, but should be cleaned up in a later pass (see h1100 follow-ups).

---

## E2 (h1107) — Section 3.5 table audit: 4 of 5 claims uncorroborated

**Section 3.5, Table 3 (retrospectively corroborated predictions).**

**Audit methodology:** For each of the 5 rows, call `p.predict(<disease>, top_n=500, include_filtered=True)` with the production predictor and check whether the claimed drug appears in the returned list. For rows where the drug is absent from the claimed disease, probe related diseases in the same family/category and report where the drug IS surfaced.

**Headline finding:**

| # | Drug | Paper disease | Model rank / tier | Corroborated? |
|---|------|---------------|-------------------|---------------|
| 1 | Dantrolene | Heart failure | **not in top 117 preds** | **FALSE** (covered in E1; actual signal is VT rank 34 FILTER) |
| 2 | Lovastatin | Multiple myeloma | **not in top 86 preds** | **FALSE** |
| 3 | Rituximab | Multiple sclerosis | rank 55, HIGH | TRUE |
| 4 | Pitavastatin | Rheumatoid arthritis | **not in top 125 preds** | **FALSE** |
| 5 | Empagliflozin | Parkinson's disease | **not in top 77/59 preds (2 MESH IDs)** | **FALSE** |

**Only 1 of 5 (20%) is corroborated by the current production predictor.**

**Row-by-row detail:**

**1. Dantrolene → Heart failure** — see E1. Model predicts Dantrolene GOLDEN (rank 10) for malignant hyperthermia (its FDA-approved primary indication, in-GT) and FILTER for VT/VF/tachycardia/arrhythmia (rank 28-67). The RCT's outcome was VT reduction in an HF cohort; paper conflated cohort with outcome.

**2. Lovastatin → Multiple myeloma (D009101, 86 preds).** Lovastatin is absent from the top-86 preds. The model correctly surfaces Lovastatin for its FDA-approved indications: hypercholesterolemia (rank 1, GOLDEN, in-GT), hyperlipidemia (rank 1, GOLDEN, in-GT), atherosclerosis (rank 7, GOLDEN, in-GT, sub_reason `statin_cv_event`). The RCT improved OS/PFS in MM is a legitimate clinical repurposing signal, but **the model does not surface it** — MM shares no strong kNN neighborhood with lipid diseases.

**3. Rituximab → Multiple sclerosis (D009103, rank 55, HIGH, sub_reason `literature_strong_low_promotion`).** Corroborated. Model also correctly surfaces Rituximab for its FDA-approved lymphoma/CLL indications (rank 3-5, MEDIUM/LOW in-GT).

**4. Pitavastatin → Rheumatoid arthritis (D001172, 125 preds).** Pitavastatin is absent from top-125 for RA. The model correctly surfaces Pitavastatin for hypercholesterolemia, hyperlipidemia, dyslipidemia (all rank 4, GOLDEN, in-GT). The "superior to MTX alone" clinical claim is real but not corroborated by the model.

**5. Empagliflozin → Parkinson's disease.** Empagliflozin is absent from top-77 (D010300) and top-59 (D010301) for Parkinson's. The model DOES surface Empagliflozin for diseases in its FDA-approved indication cluster — **all FILTER due to `rank_over_20`**:
- T2D (rank 29, FILTER, in-GT=True)
- Chronic heart failure (rank 52, FILTER, in-GT=True)
- Chronic kidney disease (rank 23, FILTER, in-GT=True)
- Diabetes (generic, D003920, rank 59, FILTER)
- Diabetic nephropathy: not in top 92

This is the cleanest h1103 case: 3 in-GT FDA-approved indications all land in FILTER because kNN rank>20. The rank_over_20 rule fires correctly (rank IS >20), but the result hides 3 legitimate FDA indications from readers.

**Recommended paper v2 revision:**

Option A (conservative): Replace Section 3.5 with a single-row table (Rituximab → MS), and move the other 4 rows to a supplementary table with a clear caveat that the model does NOT surface those predictions — reframing as "prior art / related repurposing literature" rather than "model corroborated."

Option B (honest-but-expansive): Keep all 5 rows but add a "model rank / tier" column and replace "retrospectively corroborated" framing with "clinical literature consistent with model signal." Include the E1 Dantrolene relabel (VT, not HF) and the Empagliflozin → T2D/HF/CKD observations.

Either way: the current framing ("five predictions that are retrospectively corroborated") is not supported by the model's actual output. Recommend Option A for clarity.

**Residual for h1103 (follow-up):** Empagliflozin's 3 in-GT FILTER cases at rank 21-52 are exactly the drug-class the h1200 supervised GNN should prioritize — newly-approved second-in-class drugs (SGLT2 inhibitors post-2015) with narrow DRKG edge histories.

---

*Errata maintainer: research agent. Add new entries above this line, chronologically.*
