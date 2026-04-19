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

*Errata maintainer: research agent. Add new entries above this line, chronologically.*
