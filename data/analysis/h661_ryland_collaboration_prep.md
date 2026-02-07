# Ryland Collaboration Prep: Dermatological & Genetic Disease Predictions

## Meeting: Monday Feb 10, 2026
**Prepared:** Feb 6, 2026 | **Hypothesis:** h661

---

## System Overview for Ryland

Our drug repurposing system uses kNN collaborative filtering on DRKG (Drug Repurposing Knowledge Graph) to predict drug-disease pairs. Predictions are tiered by confidence:

| Tier | Holdout Precision | Predictions | Meaning |
|------|------------------|-------------|---------|
| GOLDEN | 71.6% | 420 | Strong multiple-signal evidence |
| HIGH | 54.8% | ~858 | Good evidence, some confirmed |
| MEDIUM | 42.9% | ~1,363 | Moderate evidence, needs validation |
| LOW | 14.8% | ~4,235 | Weak evidence |
| FILTER | 10.6% | 7,274 | Known issues or contradictions |

**Key signals:** kNN rank (how many neighbor diseases share the drug), mechanism support (DRKG pathway exists), TransE consilience (embedding model agrees), gene overlap (shared drug targets & disease genes).

---

## 1. Dermatological Predictions (30 diseases, 230 GOLDEN/HIGH/MEDIUM)

### GOLDEN Tier (40 predictions, 50% GT-confirmed)
Dominated by corticosteroids (methylprednisolone, dexamethasone, hydrocortisone) for:
- Atopic dermatitis, pemphigus foliaceus, urticaria, alopecia areata
- **Less expected:** Lidocaine for atopic dermatitis [GT-confirmed, M]

### HIGH Tier (92 predictions, 63% GT-confirmed)
Corticosteroid SOC promotion (h520). High GT hit rate confirms system calibration.
- Standard dermatological corticosteroids for inflammatory skin diseases
- Tetracycline for ichthyosis vulgaris [GT-confirmed, TransE]

### MEDIUM Tier (98 predictions, 39% GT-confirmed)
More diverse drug classes:
- Tetracyclines (doxycycline, minocycline) for ichthyosis varieties
- Salicylic acid for ichthyosis/keratosis pilaris [GT-confirmed]
- Doxycycline for acne vulgaris [GT-confirmed]

---

## 2. Genetic Skin Disease Focus (Ryland's Specialty)

### Ichthyosis Predictions

| Drug | Disease | Tier | Lit Evidence | Assessment |
|------|---------|------|-------------|------------|
| Salicylic acid | Lamellar ichthyosis | MEDIUM [GT] | Standard of care (keratolytic) | Already known |
| Tetracycline | Ichthyosis vulgaris | HIGH [GT, T] | No clinical evidence | Likely kNN artifact |
| Methotrexate | Ichthyosis | MEDIUM | 1970s case reports, failed | Known-failed (abandoned for retinoids) |
| Doxycycline | Ichthyosis vulgaris | MEDIUM | No clinical evidence | kNN artifact (anti-inflammatory ≠ cornification fix) |
| Minocycline | Ichthyosis | MEDIUM | No clinical evidence | Same class artifact as doxycycline |

**Key insight:** Tetracycline-class drugs appearing for ichthyosis is a kNN co-occurrence artifact. These drugs are co-prescribed for other skin conditions (acne, rosacea), and kNN picks up dermatology patterns rather than biological relevance to cornification defects.

**Discussion point for Ryland:** Can transcriptomic analysis of ichthyosis keratinocytes help distinguish genuine drug effects from co-occurrence artifacts?

### Other Genetic Diseases of Interest

| Disease | Top Novel Prediction | Tier | Signals | Notes |
|---------|---------------------|------|---------|-------|
| Familial hypercholesterolemia | Ezetimibe | GOLDEN | - | Already used clinically |
| Congenital hypothyroidism | Liothyronine | HIGH | - | Clinically logical |
| Hereditary chronic cholestasis | Obeticholic acid | MEDIUM | - | FXR agonist, plausible |
| Transthyretin amyloid polyneuropathy | Pregabalin/gabapentin | MEDIUM | - | Symptomatic (neuropathic pain) |
| Familial mediterranean fever | Levofloxacin | MEDIUM | - | Unlikely (genetic autoinflammatory) |
| Duchenne muscular dystrophy | Prednisolone | MEDIUM [GT, T] | - | Standard of care |

---

## 3. EGFR Connection (Ryland's Research Area)

### Current EGFR Drug Predictions
- Cetuximab: All cancer-related (colorectal, lung, head/neck) - expected
- Panitumumab: Colorectal cancer HIGH [GT] - expected
- **GAP: No EGFR inhibitor predicted for ANY dermatological disease**

### Literature Evidence
- EGFR inhibitors (cetuximab, erlotinib) show dramatic psoriasis improvement in cancer patients (case reports)
- EGFR overexpressed in active psoriatic epidermis
- Not used clinically due to severe skin toxicity (acneiform rash in ~90%)
- Paradoxical: EGFR inhibitors cause skin problems while treating psoriasis

### Discussion Points for Ryland
1. **Why no EGFR→dermatological predictions?** EGFR inhibitors are cancer-only in DRKG. Cross-domain isolated (h271 filter).
2. **Can spatial transcriptomics identify EGFR-driven skin diseases?** Beyond psoriasis, which genetic skin diseases have EGFR dysregulation?
3. **Anti-EGFR for specific ichthyosis subtypes?** Some ichthyosis involves hyperproliferation where EGFR inhibition might help.

---

## 4. Novel Predictions Amenable to Wet-Lab Validation

### Most Interesting for Cell Culture Testing

1. **Montelukast → Idiopathic pulmonary fibrosis** (HIGH, Mech+TransE)
   - Leukotriene receptor antagonist, anti-inflammatory
   - Published evidence: Some preclinical studies showing anti-fibrotic effects
   - Cell culture: Test on fibroblasts, measure collagen production

2. **Obeticholic acid → Hereditary chronic cholestasis** (MEDIUM)
   - FXR agonist, already used for primary biliary cholangitis
   - Logical extension: FXR pathway shared across cholestatic conditions
   - Not a typical Ryland target but mechanistically interesting

3. **Aflibercept → Autoimmune pulmonary alveolar proteinosis** (MEDIUM)
   - Anti-VEGF, used in ophthalmology
   - Novel connection: VEGF in pulmonary disease
   - Would need specialized cell models

### Lower Priority but Interesting
4. Pregabalin/gabapentin → TTR familial amyloid polyneuropathy (symptomatic only)
5. Bisoprolol → Congenital heart disease (clinical, not testable in culture)

---

## 5. Known Limitations to Discuss

1. **Self-referentiality:** 31.6% of diseases are 100% self-referential (predictions come from the disease's own GT). Less useful for truly novel drugs.

2. **Corticosteroid inflation:** HIGH tier is 69% corticosteroids. Non-CS HIGH = 48.5%. CS predictions are medically valid but less novel.

3. **kNN artifact patterns:** Co-prescribed drugs get predicted together (tetracyclines for ichthyosis). Need independent validation signals.

4. **DRKG coverage gaps:** Some modern biologics (dupilumab, secukinumab) are underrepresented. Predictions miss recent advances.

5. **ID mapping blocker:** DRKG uses CHEBI/MONDO/UMLS IDs while reference data uses DrugBank/CHEMBL/MESH. Zero overlap blocks some analyses.

---

## 6. What Ryland Could Contribute

1. **Transcriptomic validation:** Use LINCS/GEO data to check if predicted drugs have gene expression signatures consistent with disease reversal
2. **Genetic context:** Which predictions make sense given known disease genetics?
3. **Cell culture testing:** Priority predictions for wet-lab validation
4. **EGFR signaling expertise:** Identify EGFR-driven skin diseases not captured by DRKG
5. **ID mapping help:** Genomics databases may bridge our ID gap
6. **Publication potential:** Validated novel predictions → case reports or brief communications

---

## Quick Reference

| File | Contents |
|------|----------|
| `src/production_predictor.py` | Main prediction engine |
| `data/deliverables/drug_repurposing_predictions_with_confidence.xlsx` | Full deliverable |
| `scripts/h393_holdout_tier_validation.py` | Holdout evaluator |
| `data/reference/expanded_ground_truth.json` | Enhanced ground truth (59K pairs) |
