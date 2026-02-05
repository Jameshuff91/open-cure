# Research Loop Progress

## Current Session: h191 (2026-02-05)

### Session Summary

**Agent Role:** Research Executor
**Status:** Complete
**Hypothesis Tested:**
- h191: ATC L1 Incoherence as Novel Repurposing Signal - **INVALIDATED** (but revealed complementary insight)

### h191: ATC L1 Incoherence as Novel Repurposing Signal - INVALIDATED

Tested whether ATC L1 incoherence (drug category doesn't match disease category) predicts novel repurposing, as suggested by h110's counter-intuitive finding.

**Key Discovery: Two DIFFERENT Coherence Signals**

1. **h110's Signal (Within-Class Uniqueness):**
   - "Classmate treats similar" vs "no classmate treats similar"
   - INCOHERENT (unique) = 11.24% > COHERENT = 6.69%
   - Interpretation: If prediction is unique among classmates, more likely true

2. **h191's Signal (Category Match):**
   - "ATC L1 matches expected disease category" vs "doesn't match"
   - COHERENT = 11.1% > INCOHERENT = 6.4%
   - Interpretation: Expected drug-disease relationships are more reliable

**Results:**
| Metric | Coherent | Incoherent | Delta |
|--------|----------|------------|-------|
| Overall precision | 11.1% | 6.4% | -4.7pp |
| Top-5 rank precision | 22.4% | 11.0% | -11.4pp |
| Rank 6-10 precision | 14.3% | 11.9% | -2.4pp |
| Rank 16-20 precision | 4.8% | 5.1% | +0.3pp |

**Cross-Category Repurposing Patterns Identified:**
| Pattern | Hits | Drugs |
|---------|------|-------|
| Cardiovascular → metabolic | 20 | Statins, fibrates |
| Antiinfectives → respiratory | 19 | Azithromycin, levofloxacin |
| Systemic hormonal → respiratory | 17 | Methylprednisolone, hydrocortisone |
| Dermatologicals → respiratory | 13 | Corticosteroids |
| Systemic hormonal → dermatological | 12 | Prednisone, cortisone |

**Category-Specific Findings:**
- **Metabolic is the only exception**: +1.3pp for incoherent (CV drugs → metabolic)
- **Autoimmune shows largest coherence benefit**: -25.5pp
- **Respiratory**: -21.9pp coherence benefit

**Key Insight:**
h110 and h191 measure COMPLEMENTARY signals:
- h110: Drug uniqueness within its class → discovery signal
- h191: Drug-disease category match → reliability signal

A drug that is:
- INCOHERENT by category (unexpected drug class) BUT
- UNIQUE in its ATC class (no classmates treat similar)
= Potentially a true repurposing discovery

**New Hypotheses Generated:**
- h193: Combined ATC Coherence Signals for Discovery Prediction (priority 3)
- h194: Validated Cross-Category Repurposing Database (priority 2)
- h195: Metabolic Exception Analysis: Why CV→Metabolic Works (priority 3)

### Cumulative Statistics (2026-02-05)
| Status | Count |
|--------|-------|
| Validated | 93 |
| Invalidated | 42 |
| Inconclusive | 8 |
| Blocked | 18 |
| Deprioritized | 2 |
| Pending | 31 |
| **Total Tested** | **145** |

### Key Session Learnings

1. **ATC L1 coherence and h110 within-class uniqueness are DIFFERENT signals**
2. **Category coherence predicts known relationships** (+4.7pp for coherent)
3. **Within-class uniqueness predicts novel discoveries** (+4.5pp for unique, from h110)
4. **Top-ranked incoherent predictions match overall coherent precision** (11.0% vs 11.1%)
5. **Metabolic is the only category where incoherent beats coherent** (+1.3pp)
6. **Strong cross-category patterns exist**: CV→metabolic, steroids→respiratory

### Recommended Next Steps

1. **h194: Validated Cross-Category Repurposing Database** (priority 2) - Quick win to document validated patterns
2. **h193: Combined ATC Coherence Signals** (priority 3) - Test if combining both signals identifies true discoveries
3. **h124: Disease Embedding Interpretability** (priority 3) - Understand what makes diseases similar

---

## Previous Session: h152, h189, h190, h87 (2026-02-05)

### Session Summary
**Hypotheses Tested:**
- h152: ATC Code Integration for Precision - **VALIDATED** (+11.1pp mean precision)
- h189: ATC L4 Rescue Criteria Implementation - **VALIDATED** (+383% coverage)
- h190: ATC-Based Biologic Gap Analysis - **VALIDATED** (sparse GT is root cause)
- h87: Drug Mechanism Clustering - **VALIDATED** (mechanism breadth predicts transfer)

See previous progress.md for details.
