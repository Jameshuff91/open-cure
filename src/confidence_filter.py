#!/usr/bin/env python3
"""
Confidence Filter for Drug Repurposing Predictions.

Filters out false positive patterns identified through literature validation.
Assigns confidence scores based on drug type and disease category.

Based on validation of 30 predictions:
- Biologics: 100% precision, 0% FP rate
- Small molecules: 74% precision, 16% FP rate
- Antibiotics: 0% precision, 50% FP rate
- Sympathomimetics: 0% precision, 100% FP rate

Updated 2026-01-24: Added FDA approval checking and withdrawn drug filtering.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum


class ConfidenceLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    EXCLUDED = "excluded"


@dataclass
class FilteredPrediction:
    drug: str
    disease: str
    original_score: float
    confidence: ConfidenceLevel
    reason: str
    drug_type: Optional[str] = None
    adjusted_score: Optional[float] = None


# Drug type patterns
ANTIBIOTIC_PATTERNS = [
    r".*mycin$", r".*cillin$", r".*cycline$", r".*floxacin$",
    r".*azole$", r"^cef.*", r"^amox.*", r"gentamicin", r"vancomycin",
    r"rifampicin", r"rifampin", r"streptomycin", r"kanamycin",
    r"tobramycin", r"neomycin", r"erythromycin", r"azithromycin",
    r"clarithromycin", r"metronidazole", r"sulfamethoxazole",
]

SYMPATHOMIMETIC_PATTERNS = [
    r"pseudoephedrine", r"ephedrine", r"phenylephrine", r"epinephrine",
    r"norepinephrine", r"dopamine", r"dobutamine", r"isoproterenol",
    r"amphetamine", r"methamphetamine", r"phenylpropanolamine",
]

ALPHA_BLOCKER_PATTERNS = [
    r"doxazosin", r"prazosin", r"terazosin", r"tamsulosin",
    r"alfuzosin", r"silodosin",
]

TRICYCLIC_ANTIDEPRESSANT_PATTERNS = [
    r"amitriptyline", r"nortriptyline", r"protriptyline", r"imipramine",
    r"desipramine", r"clomipramine", r"doxepin", r"trimipramine",
]

PPI_PATTERNS = [
    r"omeprazole", r"pantoprazole", r"lansoprazole", r"esomeprazole",
    r"rabeprazole", r"dexlansoprazole",
]

TUMOR_PROMOTING_HORMONE_PATTERNS = [
    r"^aldosterone$", r"^estrogen$", r"^testosterone$",  # Context dependent
]

CARDIAC_STRESS_AGENT_PATTERNS = [
    r"arbutamine", r"regadenoson", r"adenosine",  # Stress testing agents
]

CHEMOTHERAPY_PATTERNS = [
    r"idarubicin", r"daunorubicin", r"epirubicin",  # Anthracyclines
    r"vincristine", r"vinblastine",  # Vinca alkaloids
    r"etoposide", r"topotecan",  # Topoisomerase inhibitors
]

DIAGNOSTIC_AGENT_PATTERNS = [
    r".*\s*i\s*123$", r".*\s*i-123$", r"ioflupane", r"fludeoxyglucose",
    r"^tc-99m.*", r"technetium", r"gadolinium", r"barium",
    r"iohexol", r"iopamidol", r"iodixanol", r".*contrast.*",
]

BIOLOGIC_PATTERNS = [
    r".*mab$", r".*cept$", r".*ase$",  # monoclonal antibodies, fusion proteins, enzymes
]

# Drugs withdrawn from US market due to safety concerns
WITHDRAWN_DRUG_PATTERNS = [
    r"pergolide",  # Cardiac valve regurgitation (2007)
    r"cisapride",  # Cardiac arrhythmias (2000)
    r"rofecoxib",  # Cardiovascular events (Vioxx, 2004)
    r"valdecoxib",  # Cardiovascular and skin reactions (Bextra, 2005)
    r"sibutramine",  # Cardiovascular events (2010)
    r"propoxyphene",  # Cardiac effects (Darvon, 2010)
    r"tegaserod",  # Cardiovascular events (2007, limited availability)
    r"troglitazone",  # Hepatotoxicity (Rezulin, 2000)
    r"cerivastatin",  # Rhabdomyolysis (Baycol, 2001)
    r"phenylpropanolamine",  # Hemorrhagic stroke (2000)
]

# Discontinued drugs (development stopped, not available)
DISCONTINUED_DRUG_PATTERNS = [
    r"aducanumab",  # Discontinued Jan 2024 (commercial factors)
    r"lexatumumab",  # Discontinued 2015 (insufficient efficacy)
    r"fontolizumab",  # Failed Phase II Crohn's/RA, discontinued
    r"volociximab",  # Failed Phase II oncology, discontinued
    r"bectumomab",  # Imaging agent only, not therapeutic
    r"enokizumab",  # Anti-IL-9 for asthma, development discontinued
    r"matuzumab",  # Anti-EGFR for cancer, development discontinued 2008
]

# Drugs with FDA approval REVOKED (confirmed to not work)
REVOKED_APPROVAL_PATTERNS = [
    r"olaratumab",  # FDA revoked Feb 2020 after Phase III failed (Lartruvo)
]

# Anti-IL-5 drugs (fail clinically despite reducing eosinophils)
ANTI_IL5_PATTERNS = [
    r"reslizumab",
    r"mepolizumab",
    r"benralizumab",
]

# Diseases where anti-IL-5 consistently fails (eosinophils are markers, not drivers)
ANTI_IL5_EXCLUDED_DISEASES = [
    "psoriasis", "ulcerative colitis", "crohn", "multiple sclerosis",
    "atopic dermatitis", "eosinophilic esophagitis",
]

# TRAIL agonists (induce apoptosis, harmful in inflammatory diseases)
TRAIL_AGONIST_PATTERNS = [
    r"lexatumumab",
    r"mapatumumab",
    r"conatumumab",
]

# Intravitreal formulations (designed for eye injection, not systemic use)
INTRAVITREAL_DRUG_PATTERNS = [
    r"brolucizumab",  # Anti-VEGF for wet AMD
    r"ranibizumab",  # Anti-VEGF for wet AMD
    r"aflibercept",  # VEGF trap for wet AMD (also has systemic formulation)
    r"faricimab",  # Anti-VEGF/Ang-2 for wet AMD
]

# Systemic diseases where intravitreal drugs are wrong formulation
SYSTEMIC_DISEASES = [
    "multiple sclerosis", "ulcerative colitis", "crohn", "rheumatoid arthritis",
    "lupus", "psoriasis", "asthma", "diabetes", "heart failure",
]

# Failed Phase III drugs for specific indications (don't repurpose)
FAILED_PHASE3_COMBINATIONS = [
    # (drug_pattern, disease_pattern, reason)
    (r"linsitinib", r"breast.*cancer", "IGF-1R inhibitors failed Phase III breast cancer trials"),
    (r"fontolizumab", r"ulcerative.*colitis", "Anti-IFN-gamma wrong pathway - UC is Th2-like, not Th1"),
    (r"fontolizumab", r"crohn", "Failed Phase II for Crohn's disease"),
    (r"volociximab", r"multiple.*sclerosis", "α5β1 integrin is PROTECTIVE in MS - inhibiting worsens disease"),
    (r"daclizumab", r"ulcerative.*colitis", "Failed Phase 2 RCT (2006) - 2-7% remission vs 10% placebo"),
    (r"otelixizumab", r"ulcerative.*colitis", "Anti-CD3 developed for T1D only, no IBD mechanism"),
]

# IL-6 inhibitors (WRONG pathway for psoriasis - need IL-17/IL-23)
IL6_INHIBITOR_PATTERNS = [
    r"vobarilizumab",  # IL-6R nanobody for RA
    r"sarilumab",  # IL-6R antibody for RA
    r"sirukumab",  # IL-6 antibody for RA
    r"olokizumab",  # IL-6 antibody for RA
    r"clazakizumab",  # IL-6 antibody for RA
]

# Cancer-specific antibodies that target tumor markers (not autoimmune targets)
CANCER_SPECIFIC_ANTIBODY_PATTERNS = [
    r"farletuzumab",  # Anti-FRα for ovarian cancer
    r"adecatumumab",  # Anti-EpCAM for cancer
    r"nebacumab",  # Anti-endotoxin for sepsis
    r"iratumumab",  # Anti-CD30 for Hodgkin lymphoma
]

# Bone metabolism drugs (no CNS or autoimmune mechanism)
BONE_DRUG_PATTERNS = [
    r"romosozumab",  # Sclerostin inhibitor for osteoporosis
    r"denosumab",  # RANKL inhibitor (has autoimmune uses though)
]

# B-cell depleting drugs that paradoxically worsen psoriasis
BCELL_DEPLETING_PATTERNS = [
    r"rituximab",
    r"ocrelizumab",
    r"ofatumumab",
    r"bectumomab",
]

# TNF inhibitors - contraindicated in SLE, MS, and heart failure (h146/h149)
# Can cause drug-induced lupus and worsen demyelinating diseases
TNF_INHIBITOR_PATTERNS = [
    r"adalimumab",
    r"infliximab",
    r"etanercept",
    r"certolizumab",
    r"golimumab",
]

# Diseases where TNF inhibitors are contraindicated
# h146/h147: TNF inhibitors can CAUSE drug-induced lupus, worsen MS, and induce AIH
TNF_CONTRAINDICATED_DISEASES = [
    "systemic lupus erythematosus", "sle", "lupus",
    "multiple sclerosis", "ms",
    "heart failure", "congestive heart failure", "cardiac failure",
    "demyelinating disease", "optic neuritis",
    "autoimmune hepatitis",  # h147: TNF inhibitors can INDUCE AIH
]

# JAK inhibitors - contraindicated in certain conditions
JAK_INHIBITOR_PATTERNS = [
    r"tofacitinib",
    r"baricitinib",
    r"upadacitinib",
    r"filgotinib",
    r"ruxolitinib",
]

# h248: Endothelin Receptor Antagonists - CONTRAINDICATED for heart failure
# ENABLE, MELODY, SERENADE trials: Fluid retention, no benefit, possible harm
ENDOTHELIN_ANTAGONIST_PATTERNS = [
    r"bosentan",
    r"ambrisentan",
    r"macitentan",
]

# h248: Prostacyclin Analogs - CONTRAINDICATED for heart failure
# FIRST trial: INCREASED MORTALITY - trial terminated early
PROSTACYCLIN_ANALOG_PATTERNS = [
    r"epoprostenol",
    r"treprostinil",
    r"iloprost",
    r"selexipag",
    r"beraprost",
]

# h253: sGC Stimulators - CONTRAINDICATED in pregnancy (TERATOGENIC)
# Riociguat: FDA Pregnancy Category X - contraindicated in pregnancy
# Can cause fetal harm based on animal studies
SGC_STIMULATOR_PATTERNS = [
    r"riociguat",
    r"vericiguat",
    r"cinaciguat",
    r"lificiguat",
]

# Pregnancy-related conditions where teratogenic drugs are HARMFUL
PREGNANCY_CONDITIONS = [
    "pregnancy", "pregnant", "gestational", "prenatal",
    "toxemia", "preeclampsia", "eclampsia",
    "hellp syndrome", "hyperemesis gravidarum",
    "placenta", "fetal", "maternal",
]

# h252: SGLT2 Inhibitors - FALSE POSITIVE patterns
# SGLT2i cause hypoglycemia (don't treat it) and are for CKD stages 2-4 (not uremia/ESRD)
SGLT2_INHIBITOR_PATTERNS = [
    r"canagliflozin",
    r"empagliflozin",
    r"dapagliflozin",
    r"ertugliflozin",
    r"sotagliflozin",
    r"ipragliflozin",
]

# Conditions where SGLT2 inhibitors are FALSE POSITIVES
SGLT2_FALSE_POSITIVE_CONDITIONS = [
    "hypoglycemia",  # SGLT2i can CAUSE hypoglycemia, not treat it
    "uremia",  # Too advanced (ESRD/CKD stage 5) - SGLT2i for stages 2-4
    "kidney failure",  # Same as uremia
    "end stage renal",  # Same as uremia
    "esrd",  # End-stage renal disease
]

# h250: Non-DHP Calcium Channel Blockers - CONTRAINDICATED for HFrEF
# Negative inotropes that can cause acute decompensation
# ACC/AHA 2022 guidelines: CCBs classified as HARMFUL in HFrEF
NON_DHP_CCB_PATTERNS = [
    r"verapamil",
    r"diltiazem",
]

# h250: Class Ic Antiarrhythmics - CONTRAINDICATED with structural heart disease
# CAST trial: 2.5x mortality increase, proarrhythmic in post-MI and HF patients
CLASS_IC_ANTIARRHYTHMIC_PATTERNS = [
    r"flecainide",
    r"propafenone",
    r"encainide",
]

# Structural heart disease patterns where Class Ic is dangerous
STRUCTURAL_HEART_PATTERNS = [
    "heart failure", "cardiomyopathy", "cardiac failure",
    "myocardial infarction", "post-mi", "post mi",
    "ventricular tachycardia", "ventricular fibrillation",
    "ischemic heart", "ischemic cardiomyopathy",
]

# h250: Oral Milrinone for chronic HF - CONTRAINDICATED
# PROMISE trial: 28% increase in all-cause mortality, 34% increase in CV mortality
# IV milrinone for acute decompensation is different (OK for short-term)
ORAL_INOTROPE_PATTERNS = [
    r"milrinone",  # Only oral chronic use is contraindicated
]

# h250: Aliskiren - increased mortality in diabetics with HF
# ASTRONAUT trial: Higher risk of death in diabetics
ALISKIREN_PATTERNS = [
    r"aliskiren",
]

# h250: Ganglionic blockers - obsolete, severe orthostatic hypotension
# No longer used due to severe side effects and better alternatives
GANGLIONIC_BLOCKER_PATTERNS = [
    r"mecamylamine",
    r"trimethaphan",
    r"hexamethonium",
]

# h250: Surgical dyes/agents - not therapeutic
SURGICAL_DYE_PATTERNS = [
    r"isosulfan blue",
    r"methylene blue",  # Can be therapeutic in some contexts but often diagnostic
    r"indocyanine green",
    r"patent blue",
]

# h164: Immunosuppressants - contraindicated for infectious diseases
# Immunosuppressants weaken the immune system, making infections WORSE
# Exception: Autoimmune conditions (e.g., autoimmune hepatitis) are NOT infections
IMMUNOSUPPRESSANT_PATTERNS = [
    r"tacrolimus", r"cyclosporine", r"mycophenolate", r"azathioprine",
    r"sirolimus", r"everolimus", r"basiliximab", r"belatacept",
    r"leflunomide", r"teriflunomide",
]

# Infectious disease patterns where immunosuppressants are harmful
INFECTIOUS_DISEASE_PATTERNS = [
    "tuberculosis", " tb ", "hepatitis b", "hepatitis c", "hiv",
    "cytomegalovirus", " cmv ", "herpes zoster", "shingles",
    "influenza", "pneumonia", "sepsis", "bacterial infection",
    "fungal infection", "candida", "aspergillus", "malaria",
    "foot infection", "cellulitis", "abscess",
]

# Exceptions: These contain "infection" keywords but ARE autoimmune/treated with immunosuppressants
INFECTIOUS_EXCEPTIONS = [
    "autoimmune hepatitis",  # Autoimmune, not viral
    "interstitial pneumonia associated with",  # Autoimmune lung disease
]

# Metabolic disease names
METABOLIC_DISEASES = [
    "diabetes", "type 2 diabetes", "type 1 diabetes", "metabolic syndrome",
    "obesity", "hyperglycemia", "hypoglycemia", "insulin resistance",
    "diabetic", "glycemic",
]

# h153: Corticosteroids cause hyperglycemia - contraindicated for diabetes
# These drugs are commonly predicted for metabolic diseases but are HARMFUL
CORTICOSTEROID_PATTERNS = [
    r"prednisone", r"prednisolone", r"methylprednisolone", r"dexamethasone",
    r"hydrocortisone", r"cortisone", r"betamethasone", r"triamcinolone",
    r"fluticasone", r"budesonide", r"beclomethasone", r"fludrocortisone",
]

# Cardiac conditions where alpha blockers are harmful
CARDIAC_CONDITIONS = [
    "heart failure", "congestive heart failure", "cardiac failure",
    "cardiomyopathy", "left ventricular dysfunction",
]

# FDA-approved drug-disease pairs (not truly novel predictions)
# These represent ground truth gaps, not model errors
FDA_APPROVED_PAIRS: Set[Tuple[str, str]] = {
    # Breast cancer treatments
    ("pembrolizumab", "breast cancer"),
    ("pembrolizumab", "triple-negative breast cancer"),
    ("doxorubicin", "breast cancer"),
    ("methotrexate", "breast cancer"),
    ("paclitaxel", "breast cancer"),
    ("trastuzumab", "breast cancer"),
    # MS treatments
    ("natalizumab", "multiple sclerosis"),
    ("ocrelizumab", "multiple sclerosis"),
    ("fingolimod", "multiple sclerosis"),
    ("dimethyl fumarate", "multiple sclerosis"),
    # Colorectal cancer
    ("cetuximab", "colorectal cancer"),
    ("oxaliplatin", "colorectal cancer"),
    ("bevacizumab", "colorectal cancer"),
    ("irinotecan", "colorectal cancer"),
    # Melanoma
    ("trametinib", "melanoma"),
    ("dabrafenib", "melanoma"),
    ("pembrolizumab", "melanoma"),
    ("nivolumab", "melanoma"),
    # Pancreatic cancer
    ("erlotinib", "pancreatic cancer"),
    ("gemcitabine", "pancreatic cancer"),
    # Lung cancer
    ("erlotinib", "lung cancer"),
    ("gefitinib", "lung cancer"),
    ("osimertinib", "lung cancer"),
    # RA treatments
    ("prednisone", "rheumatoid arthritis"),
    ("prednisolone", "rheumatoid arthritis"),
    ("methotrexate", "rheumatoid arthritis"),
    ("adalimumab", "rheumatoid arthritis"),
    ("etanercept", "rheumatoid arthritis"),
    # COPD
    ("umeclidinium", "copd"),
    ("tiotropium", "copd"),
    ("formoterol", "copd"),
    # Asthma
    ("tezepelumab", "asthma"),
    ("dupilumab", "asthma"),
    ("omalizumab", "asthma"),
    # UC treatments (added from batch 2 validation)
    ("ustekinumab", "ulcerative colitis"),
    ("guselkumab", "ulcerative colitis"),
    ("vedolizumab", "ulcerative colitis"),
    ("infliximab", "ulcerative colitis"),
    ("adalimumab", "ulcerative colitis"),
    # Psoriasis (added from batch 2 validation)
    ("ustekinumab", "psoriasis"),
    ("guselkumab", "psoriasis"),
    ("secukinumab", "psoriasis"),
    ("ixekizumab", "psoriasis"),
    ("risankizumab", "psoriasis"),
}


def is_fda_approved_pair(drug: str, disease: str) -> bool:
    """Check if a drug-disease pair is already FDA-approved."""
    drug_lower = drug.lower()
    disease_lower = disease.lower()

    for approved_drug, approved_disease in FDA_APPROVED_PAIRS:
        if approved_drug in drug_lower or drug_lower in approved_drug:
            if approved_disease in disease_lower or disease_lower in approved_disease:
                return True
    return False


def classify_drug_type(drug_name: str) -> str:
    """Classify a drug into a type category."""
    drug_lower = drug_name.lower()

    # Check specific patterns
    for pattern in BIOLOGIC_PATTERNS:
        if re.search(pattern, drug_lower):
            return "biologic"

    for pattern in ANTIBIOTIC_PATTERNS:
        if re.search(pattern, drug_lower):
            return "antibiotic"

    for pattern in SYMPATHOMIMETIC_PATTERNS:
        if re.search(pattern, drug_lower):
            return "sympathomimetic"

    for pattern in ALPHA_BLOCKER_PATTERNS:
        if re.search(pattern, drug_lower):
            return "alpha_blocker"

    for pattern in DIAGNOSTIC_AGENT_PATTERNS:
        if re.search(pattern, drug_lower):
            return "diagnostic_agent"

    for pattern in TRICYCLIC_ANTIDEPRESSANT_PATTERNS:
        if re.search(pattern, drug_lower):
            return "tricyclic_antidepressant"

    for pattern in PPI_PATTERNS:
        if re.search(pattern, drug_lower):
            return "ppi"

    for pattern in TUMOR_PROMOTING_HORMONE_PATTERNS:
        if re.search(pattern, drug_lower):
            return "tumor_promoting_hormone"

    for pattern in CARDIAC_STRESS_AGENT_PATTERNS:
        if re.search(pattern, drug_lower):
            return "cardiac_stress_agent"

    for pattern in CHEMOTHERAPY_PATTERNS:
        if re.search(pattern, drug_lower):
            return "chemotherapy"

    return "small_molecule"


def is_metabolic_disease(disease_name: str) -> bool:
    """Check if disease is metabolic."""
    disease_lower = disease_name.lower()
    return any(m in disease_lower for m in METABOLIC_DISEASES)


def is_cardiac_condition(disease_name: str) -> bool:
    """Check if disease is a cardiac condition."""
    disease_lower = disease_name.lower()
    return any(c in disease_lower for c in CARDIAC_CONDITIONS)


def is_hypertension(disease_name: str) -> bool:
    """Check if disease is hypertension."""
    disease_lower = disease_name.lower()
    return "hypertension" in disease_lower or "high blood pressure" in disease_lower


def is_cancer(disease_name: str) -> bool:
    """Check if disease is cancer."""
    disease_lower = disease_name.lower()
    return any(c in disease_lower for c in ["cancer", "carcinoma", "tumor", "neoplasm", "lymphoma", "leukemia", "melanoma"])


def filter_prediction(
    drug: str,
    disease: str,
    score: float,
) -> FilteredPrediction:
    """
    Filter a single prediction and assign confidence level.

    Returns FilteredPrediction with confidence level and reason.
    """
    drug_type = classify_drug_type(drug)
    disease_lower = disease.lower()

    # EXCLUSION RULES (known harmful patterns)

    # Rule 0a: Withdrawn drugs (safety concerns)
    drug_lower = drug.lower()
    for pattern in WITHDRAWN_DRUG_PATTERNS:
        if re.search(pattern, drug_lower):
            return FilteredPrediction(
                drug=drug,
                disease=disease,
                original_score=score,
                confidence=ConfidenceLevel.EXCLUDED,
                reason="Drug withdrawn from US market due to safety concerns",
                drug_type=drug_type,
                adjusted_score=0.0,
            )

    # Rule 0b: Failed Phase III combinations
    for drug_pattern, disease_pattern, reason in FAILED_PHASE3_COMBINATIONS:
        if re.search(drug_pattern, drug_lower) and re.search(disease_pattern, disease_lower):
            return FilteredPrediction(
                drug=drug,
                disease=disease,
                original_score=score,
                confidence=ConfidenceLevel.EXCLUDED,
                reason=reason,
                drug_type=drug_type,
                adjusted_score=0.0,
            )

    # Rule 0c: Discontinued drugs (development stopped)
    for pattern in DISCONTINUED_DRUG_PATTERNS:
        if re.search(pattern, drug_lower):
            return FilteredPrediction(
                drug=drug,
                disease=disease,
                original_score=score,
                confidence=ConfidenceLevel.EXCLUDED,
                reason="Drug development discontinued - not available",
                drug_type=drug_type,
                adjusted_score=0.0,
            )

    # Rule 0d: Anti-IL-5 drugs for non-eosinophilic diseases
    # These reduce eosinophils but fail to provide clinical benefit
    for pattern in ANTI_IL5_PATTERNS:
        if re.search(pattern, drug_lower):
            if any(d in disease_lower for d in ANTI_IL5_EXCLUDED_DISEASES):
                return FilteredPrediction(
                    drug=drug,
                    disease=disease,
                    original_score=score,
                    confidence=ConfidenceLevel.EXCLUDED,
                    reason="Anti-IL-5 reduces eosinophils but fails clinically for this disease",
                    drug_type=drug_type,
                    adjusted_score=0.0,
                )

    # Rule 0e: Intravitreal drugs for systemic diseases (wrong formulation)
    for pattern in INTRAVITREAL_DRUG_PATTERNS:
        if re.search(pattern, drug_lower):
            if any(d in disease_lower for d in SYSTEMIC_DISEASES):
                return FilteredPrediction(
                    drug=drug,
                    disease=disease,
                    original_score=score,
                    confidence=ConfidenceLevel.EXCLUDED,
                    reason="Intravitreal formulation - not suitable for systemic disease",
                    drug_type=drug_type,
                    adjusted_score=0.0,
                )

    # Rule 0f: B-cell depleting drugs for psoriasis (paradoxically worsen)
    for pattern in BCELL_DEPLETING_PATTERNS:
        if re.search(pattern, drug_lower):
            if "psoriasis" in disease_lower:
                return FilteredPrediction(
                    drug=drug,
                    disease=disease,
                    original_score=score,
                    confidence=ConfidenceLevel.EXCLUDED,
                    reason="B-cell depletion paradoxically induces/worsens psoriasis",
                    drug_type=drug_type,
                    adjusted_score=0.0,
                )

    # Rule 0f2: TNF inhibitors contraindicated for SLE, MS, heart failure (h146/h149)
    # TNF inhibitors can cause drug-induced lupus and worsen demyelinating diseases
    for pattern in TNF_INHIBITOR_PATTERNS:
        if re.search(pattern, drug_lower):
            if any(d in disease_lower for d in TNF_CONTRAINDICATED_DISEASES):
                return FilteredPrediction(
                    drug=drug,
                    disease=disease,
                    original_score=score,
                    confidence=ConfidenceLevel.EXCLUDED,
                    reason="TNF inhibitors contraindicated - can cause drug-induced lupus or worsen demyelinating/cardiac disease",
                    drug_type=drug_type,
                    adjusted_score=0.0,
                )

    # Rule 0f3 (h248): Endothelin receptor antagonists for heart failure
    # ENABLE, MELODY, SERENADE trials: Fluid retention, peripheral edema, no clinical benefit
    for pattern in ENDOTHELIN_ANTAGONIST_PATTERNS:
        if re.search(pattern, drug_lower):
            if is_cardiac_condition(disease):
                return FilteredPrediction(
                    drug=drug,
                    disease=disease,
                    original_score=score,
                    confidence=ConfidenceLevel.EXCLUDED,
                    reason="Endothelin antagonists cause fluid retention in heart failure (ENABLE/MELODY/SERENADE trials)",
                    drug_type=drug_type,
                    adjusted_score=0.0,
                )

    # Rule 0f4 (h248): Prostacyclin analogs for systolic heart failure
    # FIRST trial: INCREASED MORTALITY - trial terminated early due to harm
    for pattern in PROSTACYCLIN_ANALOG_PATTERNS:
        if re.search(pattern, drug_lower):
            if is_cardiac_condition(disease):
                return FilteredPrediction(
                    drug=drug,
                    disease=disease,
                    original_score=score,
                    confidence=ConfidenceLevel.EXCLUDED,
                    reason="Prostacyclin analogs INCREASE MORTALITY in heart failure (FIRST trial terminated for harm)",
                    drug_type=drug_type,
                    adjusted_score=0.0,
                )

    # Rule 0f5 (h253): sGC stimulators for pregnancy conditions
    # Riociguat is FDA Pregnancy Category X - TERATOGENIC
    for pattern in SGC_STIMULATOR_PATTERNS:
        if re.search(pattern, drug_lower):
            if any(preg in disease_lower for preg in PREGNANCY_CONDITIONS):
                return FilteredPrediction(
                    drug=drug,
                    disease=disease,
                    original_score=score,
                    confidence=ConfidenceLevel.EXCLUDED,
                    reason="sGC stimulators are TERATOGENIC - FDA Pregnancy Category X (CONTRAINDICATED)",
                    drug_type=drug_type,
                    adjusted_score=0.0,
                )

    # Rule 0f6 (h252): SGLT2 inhibitors for hypoglycemia/uremia (FALSE POSITIVES)
    # SGLT2i cause hypoglycemia (don't treat) and are for CKD 2-4 (not ESRD/uremia)
    for pattern in SGLT2_INHIBITOR_PATTERNS:
        if re.search(pattern, drug_lower):
            if any(fp in disease_lower for fp in SGLT2_FALSE_POSITIVE_CONDITIONS):
                return FilteredPrediction(
                    drug=drug,
                    disease=disease,
                    original_score=score,
                    confidence=ConfidenceLevel.EXCLUDED,
                    reason="SGLT2 inhibitors: CAUSE hypoglycemia (don't treat) and are for CKD 2-4 (not uremia/ESRD)",
                    drug_type=drug_type,
                    adjusted_score=0.0,
                )

    # Rule 0f7 (h250): Non-DHP CCBs for HFrEF - negative inotropes cause decompensation
    # ACC/AHA 2022: Non-DHP CCBs classified as HARMFUL in HFrEF
    for pattern in NON_DHP_CCB_PATTERNS:
        if re.search(pattern, drug_lower):
            if is_cardiac_condition(disease):
                return FilteredPrediction(
                    drug=drug,
                    disease=disease,
                    original_score=score,
                    confidence=ConfidenceLevel.EXCLUDED,
                    reason="Non-DHP CCBs (Verapamil/Diltiazem) are HARMFUL in HFrEF - negative inotrope causes decompensation",
                    drug_type=drug_type,
                    adjusted_score=0.0,
                )

    # Rule 0f8 (h250): Class Ic antiarrhythmics with structural heart disease
    # CAST trial: 2.5x mortality, proarrhythmic in post-MI and HF
    for pattern in CLASS_IC_ANTIARRHYTHMIC_PATTERNS:
        if re.search(pattern, drug_lower):
            if any(shd in disease_lower for shd in STRUCTURAL_HEART_PATTERNS):
                return FilteredPrediction(
                    drug=drug,
                    disease=disease,
                    original_score=score,
                    confidence=ConfidenceLevel.EXCLUDED,
                    reason="Class Ic antiarrhythmics INCREASE MORTALITY 2.5x in structural heart disease (CAST trial)",
                    drug_type=drug_type,
                    adjusted_score=0.0,
                )

    # Rule 0f9 (h250): Oral milrinone for chronic HF
    # PROMISE trial: 28% increase in all-cause mortality
    for pattern in ORAL_INOTROPE_PATTERNS:
        if re.search(pattern, drug_lower):
            if is_cardiac_condition(disease):
                return FilteredPrediction(
                    drug=drug,
                    disease=disease,
                    original_score=score,
                    confidence=ConfidenceLevel.EXCLUDED,
                    reason="Oral milrinone INCREASES MORTALITY 28% in chronic HF (PROMISE trial)",
                    drug_type=drug_type,
                    adjusted_score=0.0,
                )

    # Rule 0f10 (h250): Ganglionic blockers - obsolete, severe side effects
    for pattern in GANGLIONIC_BLOCKER_PATTERNS:
        if re.search(pattern, drug_lower):
            return FilteredPrediction(
                drug=drug,
                disease=disease,
                original_score=score,
                confidence=ConfidenceLevel.EXCLUDED,
                reason="Ganglionic blockers are obsolete - severe orthostatic hypotension and multiple side effects",
                drug_type=drug_type,
                adjusted_score=0.0,
            )

    # Rule 0f11 (h250): Surgical dyes - not therapeutic
    for pattern in SURGICAL_DYE_PATTERNS:
        if re.search(pattern, drug_lower):
            return FilteredPrediction(
                drug=drug,
                disease=disease,
                original_score=score,
                confidence=ConfidenceLevel.EXCLUDED,
                reason="Surgical/diagnostic dye - not a therapeutic agent",
                drug_type=drug_type,
                adjusted_score=0.0,
            )

    # Rule 0g: TRAIL agonists for inflammatory diseases
    for pattern in TRAIL_AGONIST_PATTERNS:
        if re.search(pattern, drug_lower):
            if any(d in disease_lower for d in ["colitis", "crohn", "psoriasis", "arthritis"]):
                return FilteredPrediction(
                    drug=drug,
                    disease=disease,
                    original_score=score,
                    confidence=ConfidenceLevel.EXCLUDED,
                    reason="TRAIL agonists worsen epithelial damage in inflammatory diseases",
                    drug_type=drug_type,
                    adjusted_score=0.0,
                )

    # Rule 0h: Drugs with FDA approval REVOKED (confirmed not to work)
    for pattern in REVOKED_APPROVAL_PATTERNS:
        if re.search(pattern, drug_lower):
            return FilteredPrediction(
                drug=drug,
                disease=disease,
                original_score=score,
                confidence=ConfidenceLevel.EXCLUDED,
                reason="FDA approval revoked after confirmatory trial failed",
                drug_type=drug_type,
                adjusted_score=0.0,
            )

    # Rule 0i: IL-6 inhibitors for psoriasis (wrong pathway - need IL-17/IL-23)
    for pattern in IL6_INHIBITOR_PATTERNS:
        if re.search(pattern, drug_lower):
            if "psoriasis" in disease_lower:
                return FilteredPrediction(
                    drug=drug,
                    disease=disease,
                    original_score=score,
                    confidence=ConfidenceLevel.EXCLUDED,
                    reason="IL-6 inhibition is WRONG pathway for psoriasis - need IL-17/IL-23",
                    drug_type=drug_type,
                    adjusted_score=0.0,
                )

    # Rule 0j: Bone drugs for neurological diseases (no CNS mechanism)
    for pattern in BONE_DRUG_PATTERNS:
        if re.search(pattern, drug_lower):
            if any(d in disease_lower for d in ["multiple sclerosis", "parkinson", "alzheimer", "dementia"]):
                return FilteredPrediction(
                    drug=drug,
                    disease=disease,
                    original_score=score,
                    confidence=ConfidenceLevel.EXCLUDED,
                    reason="Bone metabolism drug has no CNS mechanism for neurological diseases",
                    drug_type=drug_type,
                    adjusted_score=0.0,
                )

    # Rule 0k: Cancer-specific antibodies for autoimmune diseases (wrong target)
    for pattern in CANCER_SPECIFIC_ANTIBODY_PATTERNS:
        if re.search(pattern, drug_lower):
            if any(d in disease_lower for d in ["psoriasis", "colitis", "arthritis", "lupus", "sclerosis"]):
                return FilteredPrediction(
                    drug=drug,
                    disease=disease,
                    original_score=score,
                    confidence=ConfidenceLevel.EXCLUDED,
                    reason="Cancer-specific antibody targets tumor marker, not autoimmune pathway",
                    drug_type=drug_type,
                    adjusted_score=0.0,
                )

    # Rule 0m (h164): Immunosuppressants for infectious diseases
    # Immunosuppressants weaken immunity and make infections WORSE
    # Exception: Autoimmune conditions that contain "infection" keywords
    for pattern in IMMUNOSUPPRESSANT_PATTERNS:
        if re.search(pattern, drug_lower):
            # Check for infectious disease patterns
            is_infectious = any(inf in disease_lower for inf in INFECTIOUS_DISEASE_PATTERNS)
            # Check for exceptions (autoimmune conditions)
            is_exception = any(exc in disease_lower for exc in INFECTIOUS_EXCEPTIONS)
            if is_infectious and not is_exception:
                return FilteredPrediction(
                    drug=drug,
                    disease=disease,
                    original_score=score,
                    confidence=ConfidenceLevel.EXCLUDED,
                    reason="Immunosuppressants weaken immunity - contraindicated for infections (HARMFUL)",
                    drug_type=drug_type,
                    adjusted_score=0.0,
                )

    # Rule 0l: Already FDA-approved (not truly novel - ground truth gap)
    if is_fda_approved_pair(drug, disease):
        return FilteredPrediction(
            drug=drug,
            disease=disease,
            original_score=score,
            confidence=ConfidenceLevel.LOW,
            reason="Already FDA-approved for this indication (ground truth gap)",
            drug_type=drug_type,
            adjusted_score=score * 0.5,  # Downweight but don't exclude
        )

    # Rule 1: Antibiotics for metabolic diseases
    if drug_type == "antibiotic" and is_metabolic_disease(disease):
        return FilteredPrediction(
            drug=drug,
            disease=disease,
            original_score=score,
            confidence=ConfidenceLevel.EXCLUDED,
            reason="Antibiotics can inhibit insulin release or worsen glucose control",
            drug_type=drug_type,
            adjusted_score=0.0,
        )

    # Rule 2: Sympathomimetics for diabetes
    if drug_type == "sympathomimetic" and is_metabolic_disease(disease):
        return FilteredPrediction(
            drug=drug,
            disease=disease,
            original_score=score,
            confidence=ConfidenceLevel.EXCLUDED,
            reason="Sympathomimetics increase blood glucose (HARMFUL)",
            drug_type=drug_type,
            adjusted_score=0.0,
        )

    # Rule 2b (h153): Corticosteroids for metabolic diseases
    # Corticosteroids cause hyperglycemia and worsen diabetes
    for pattern in CORTICOSTEROID_PATTERNS:
        if re.search(pattern, drug_lower):
            if is_metabolic_disease(disease):
                return FilteredPrediction(
                    drug=drug,
                    disease=disease,
                    original_score=score,
                    confidence=ConfidenceLevel.EXCLUDED,
                    reason="Corticosteroids cause hyperglycemia - contraindicated for diabetes (HARMFUL)",
                    drug_type=drug_type,
                    adjusted_score=0.0,
                )
            break  # Only check once

    # Rule 3: Alpha blockers for heart failure
    if drug_type == "alpha_blocker" and is_cardiac_condition(disease):
        return FilteredPrediction(
            drug=drug,
            disease=disease,
            original_score=score,
            confidence=ConfidenceLevel.EXCLUDED,
            reason="ALLHAT trial: Alpha blockers increase HF risk 2x (HARMFUL)",
            drug_type=drug_type,
            adjusted_score=0.0,
        )

    # Rule 4: Diagnostic agents as treatments
    if drug_type == "diagnostic_agent":
        return FilteredPrediction(
            drug=drug,
            disease=disease,
            original_score=score,
            confidence=ConfidenceLevel.EXCLUDED,
            reason="Diagnostic agent (for imaging, not treatment)",
            drug_type=drug_type,
            adjusted_score=0.0,
        )

    # Rule 5: TCAs for hypertension (TCAs CAUSE hypertension via NET inhibition)
    if drug_type == "tricyclic_antidepressant" and is_hypertension(disease):
        return FilteredPrediction(
            drug=drug,
            disease=disease,
            original_score=score,
            confidence=ConfidenceLevel.EXCLUDED,
            reason="TCAs can CAUSE hypertension via NET inhibition (HARMFUL)",
            drug_type=drug_type,
            adjusted_score=0.0,
        )

    # Rule 6: PPIs for hypertension (PPIs associated with 17% increased HTN risk)
    if drug_type == "ppi" and is_hypertension(disease):
        return FilteredPrediction(
            drug=drug,
            disease=disease,
            original_score=score,
            confidence=ConfidenceLevel.EXCLUDED,
            reason="PPIs cause 17% increased hypertension risk (HARMFUL)",
            drug_type=drug_type,
            adjusted_score=0.0,
        )

    # Rule 7: Tumor-promoting hormones for cancer (opposite of treatment)
    if drug_type == "tumor_promoting_hormone" and is_cancer(disease):
        return FilteredPrediction(
            drug=drug,
            disease=disease,
            original_score=score,
            confidence=ConfidenceLevel.EXCLUDED,
            reason="Hormone promotes tumor growth; blocking it is therapeutic (OPPOSITE)",
            drug_type=drug_type,
            adjusted_score=0.0,
        )

    # Rule 8: Cardiac stress agents for hypertension (they RAISE BP, not treat it)
    if drug_type == "cardiac_stress_agent" and is_hypertension(disease):
        return FilteredPrediction(
            drug=drug,
            disease=disease,
            original_score=score,
            confidence=ConfidenceLevel.EXCLUDED,
            reason="Cardiac stress testing agent - RAISES BP, doesn't treat hypertension",
            drug_type=drug_type,
            adjusted_score=0.0,
        )

    # Rule 9: Chemotherapy drugs for metabolic diseases (no mechanism)
    if drug_type == "chemotherapy" and is_metabolic_disease(disease):
        return FilteredPrediction(
            drug=drug,
            disease=disease,
            original_score=score,
            confidence=ConfidenceLevel.EXCLUDED,
            reason="Chemotherapy drug has no mechanism for metabolic disease",
            drug_type=drug_type,
            adjusted_score=0.0,
        )

    # CONFIDENCE SCORING

    # High confidence: Biologics (100% precision in validation)
    if drug_type == "biologic":
        return FilteredPrediction(
            drug=drug,
            disease=disease,
            original_score=score,
            confidence=ConfidenceLevel.HIGH,
            reason="Biologic drugs: 100% precision in validation",
            drug_type=drug_type,
            adjusted_score=score * 1.0,  # No adjustment
        )

    # Medium confidence: Small molecules (74% precision)
    if drug_type == "small_molecule":
        # Specific high-confidence patterns
        if "cancer" in disease_lower and score > 0.9:
            return FilteredPrediction(
                drug=drug,
                disease=disease,
                original_score=score,
                confidence=ConfidenceLevel.HIGH,
                reason="Small molecule for cancer with high score",
                drug_type=drug_type,
                adjusted_score=score * 0.9,
            )

        return FilteredPrediction(
            drug=drug,
            disease=disease,
            original_score=score,
            confidence=ConfidenceLevel.MEDIUM,
            reason="Small molecule: 74% precision in validation",
            drug_type=drug_type,
            adjusted_score=score * 0.74,
        )

    # Low confidence: Antibiotics for non-metabolic diseases
    if drug_type == "antibiotic":
        return FilteredPrediction(
            drug=drug,
            disease=disease,
            original_score=score,
            confidence=ConfidenceLevel.LOW,
            reason="Antibiotic for non-metabolic disease: requires specific validation",
            drug_type=drug_type,
            adjusted_score=score * 0.5,
        )

    # Default: Medium confidence
    return FilteredPrediction(
        drug=drug,
        disease=disease,
        original_score=score,
        confidence=ConfidenceLevel.MEDIUM,
        reason="Default classification",
        drug_type=drug_type,
        adjusted_score=score * 0.7,
    )


def filter_predictions(
    predictions: List[Dict],
    exclude_low_confidence: bool = False,
) -> Tuple[List[FilteredPrediction], Dict]:
    """
    Filter a list of predictions and return filtered results with statistics.

    Args:
        predictions: List of dicts with 'drug', 'disease', 'score' keys
        exclude_low_confidence: If True, also exclude LOW confidence predictions

    Returns:
        Tuple of (filtered predictions, statistics dict)
    """
    results = []
    stats = {
        "total": len(predictions),
        "high_confidence": 0,
        "medium_confidence": 0,
        "low_confidence": 0,
        "excluded": 0,
        "exclusion_reasons": {},
    }

    for pred in predictions:
        filtered = filter_prediction(
            drug=pred["drug"],
            disease=pred["disease"],
            score=pred.get("score", pred.get("probability", 0.5)),
        )

        # Track statistics
        if filtered.confidence == ConfidenceLevel.HIGH:
            stats["high_confidence"] += 1
        elif filtered.confidence == ConfidenceLevel.MEDIUM:
            stats["medium_confidence"] += 1
        elif filtered.confidence == ConfidenceLevel.LOW:
            stats["low_confidence"] += 1
        elif filtered.confidence == ConfidenceLevel.EXCLUDED:
            stats["excluded"] += 1
            reason = filtered.reason
            stats["exclusion_reasons"][reason] = stats["exclusion_reasons"].get(reason, 0) + 1

        # Filter based on confidence
        if filtered.confidence == ConfidenceLevel.EXCLUDED:
            continue
        if exclude_low_confidence and filtered.confidence == ConfidenceLevel.LOW:
            continue

        results.append(filtered)

    return results, stats


def print_filter_report(stats: Dict) -> None:
    """Print a summary report of filtering."""
    print("\n" + "=" * 60)
    print("CONFIDENCE FILTER REPORT")
    print("=" * 60)
    print(f"\nTotal predictions: {stats['total']}")
    print(f"High confidence:   {stats['high_confidence']} ({stats['high_confidence']/stats['total']*100:.1f}%)")
    print(f"Medium confidence: {stats['medium_confidence']} ({stats['medium_confidence']/stats['total']*100:.1f}%)")
    print(f"Low confidence:    {stats['low_confidence']} ({stats['low_confidence']/stats['total']*100:.1f}%)")
    print(f"EXCLUDED:          {stats['excluded']} ({stats['excluded']/stats['total']*100:.1f}%)")

    if stats["exclusion_reasons"]:
        print("\nExclusion reasons:")
        for reason, count in sorted(stats["exclusion_reasons"].items(), key=lambda x: -x[1]):
            print(f"  - {reason}: {count}")


if __name__ == "__main__":
    # Test with actionable predictions
    import json
    from pathlib import Path

    data_dir = Path(__file__).parent.parent / "data" / "analysis"

    with open(data_dir / "actionable_predictions.json") as f:
        actionable = json.load(f)

    # Flatten predictions
    all_preds = []
    for disease, drugs in actionable["predictions_by_disease"].items():
        for drug_info in drugs:
            all_preds.append({
                "drug": drug_info["drug"],
                "disease": disease,
                "score": drug_info["probability"],
            })

    # Filter
    filtered, stats = filter_predictions(all_preds)
    print_filter_report(stats)

    print("\n" + "=" * 60)
    print("EXCLUDED PREDICTIONS")
    print("=" * 60)

    for pred in all_preds:
        result = filter_prediction(pred["drug"], pred["disease"], pred["score"])
        if result.confidence == ConfidenceLevel.EXCLUDED:
            print(f"  ❌ {result.drug} → {result.disease}")
            print(f"     Reason: {result.reason}")

    print("\n" + "=" * 60)
    print("HIGH CONFIDENCE PREDICTIONS")
    print("=" * 60)

    for result in filtered:
        if result.confidence == ConfidenceLevel.HIGH:
            print(f"  ✓ {result.drug} → {result.disease} (score: {result.original_score:.3f})")
