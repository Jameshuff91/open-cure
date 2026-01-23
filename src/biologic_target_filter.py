#!/usr/bin/env python3
"""
Biologic Target Filter - Validates monoclonal antibody predictions based on naming conventions.

The -mab naming convention encodes target information:
  Infix     Target                     Suitable For
  -------   ----------------------     ---------------------------
  -tu-      Tumor                      Cancer, not autoimmune
  -li(m)-   Immune system (lymphocyte) Autoimmune, MS, RA
  -ci(r)-   Circulatory system         Cardiovascular diseases
  -ne(r)-   Nervous system             Neurological conditions
  -ki-      Cytokines/interleukins     Inflammatory diseases
  -os-      Bone                       Osteoporosis, bone diseases
  -ba-      Bacterial                  Infections
  -vi-      Viral                      Viral infections

Source suffixes:
  -omab     Murine (highest immunogenicity)
  -ximab    Chimeric (~65-75% human)
  -zumab    Humanized (~90-95% human)
  -umab     Fully human (lowest immunogenicity)
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TargetCategory(Enum):
    TUMOR = "tumor"
    IMMUNE = "immune"
    CARDIOVASCULAR = "cardiovascular"
    NERVOUS = "nervous"
    CYTOKINE = "cytokine"
    BONE = "bone"
    BACTERIAL = "bacterial"
    VIRAL = "viral"
    UNKNOWN = "unknown"


class DiseaseCategory(Enum):
    CANCER = "cancer"
    AUTOIMMUNE = "autoimmune"
    CARDIOVASCULAR = "cardiovascular"
    NEUROLOGICAL = "neurological"
    METABOLIC = "metabolic"
    INFECTIOUS = "infectious"
    BONE = "bone"
    OTHER = "other"


# Target infixes from WHO INN naming convention
TARGET_INFIXES = {
    # Tumor targets - for cancer
    r"tu": TargetCategory.TUMOR,
    r"tum": TargetCategory.TUMOR,

    # Immune/lymphocyte targets - for autoimmune
    r"li": TargetCategory.IMMUNE,
    r"lim": TargetCategory.IMMUNE,

    # Circulatory targets - for cardiovascular
    r"ci": TargetCategory.CARDIOVASCULAR,
    r"cir": TargetCategory.CARDIOVASCULAR,

    # Nervous system targets - for neurological
    r"ne": TargetCategory.NERVOUS,
    r"ner": TargetCategory.NERVOUS,

    # Cytokine targets - for inflammatory/autoimmune
    r"ki": TargetCategory.CYTOKINE,
    r"kin": TargetCategory.CYTOKINE,

    # Bone targets - for osteoporosis
    r"os": TargetCategory.BONE,
    r"so": TargetCategory.BONE,  # sclerostin

    # Bacterial targets - for infections
    r"ba": TargetCategory.BACTERIAL,

    # Viral targets - for viral infections
    r"vi": TargetCategory.VIRAL,
}

# Known drug-target mappings (override naming convention when wrong)
KNOWN_TARGETS = {
    # MS drugs (immune targets)
    "natalizumab": (TargetCategory.IMMUNE, "α4-integrin", ["multiple sclerosis", "crohn"]),
    "rituximab": (TargetCategory.IMMUNE, "CD20", ["multiple sclerosis", "lymphoma", "leukemia", "rheumatoid"]),
    "ocrelizumab": (TargetCategory.IMMUNE, "CD20", ["multiple sclerosis"]),
    "alemtuzumab": (TargetCategory.IMMUNE, "CD52", ["multiple sclerosis", "leukemia"]),
    "ofatumumab": (TargetCategory.IMMUNE, "CD20", ["multiple sclerosis", "leukemia"]),

    # Cancer drugs (tumor targets)
    "volociximab": (TargetCategory.TUMOR, "α5β1-integrin", ["cancer"]),
    "adecatumumab": (TargetCategory.TUMOR, "EpCAM", ["cancer"]),
    "trastuzumab": (TargetCategory.TUMOR, "HER2", ["breast cancer", "gastric cancer"]),
    "bevacizumab": (TargetCategory.TUMOR, "VEGF", ["cancer"]),
    "cetuximab": (TargetCategory.TUMOR, "EGFR", ["colorectal cancer", "head and neck cancer"]),
    "atezolizumab": (TargetCategory.TUMOR, "PD-L1", ["cancer"]),
    "pembrolizumab": (TargetCategory.TUMOR, "PD-1", ["cancer"]),
    "nivolumab": (TargetCategory.TUMOR, "PD-1", ["cancer"]),

    # Bone drugs
    "romosozumab": (TargetCategory.BONE, "sclerostin", ["osteoporosis"]),
    "denosumab": (TargetCategory.BONE, "RANKL", ["osteoporosis", "bone metastases"]),

    # Migraine drugs (nervous system)
    "eptinezumab": (TargetCategory.NERVOUS, "CGRP", ["migraine"]),
    "erenumab": (TargetCategory.NERVOUS, "CGRP receptor", ["migraine"]),
    "fremanezumab": (TargetCategory.NERVOUS, "CGRP", ["migraine"]),
    "galcanezumab": (TargetCategory.NERVOUS, "CGRP", ["migraine"]),

    # Autoimmune drugs (immune/cytokine targets)
    "belimumab": (TargetCategory.IMMUNE, "BLyS", ["lupus"]),
    "eculizumab": (TargetCategory.IMMUNE, "C5 complement", ["PNH", "aHUS", "NMOSD"]),
    "adalimumab": (TargetCategory.CYTOKINE, "TNF-α", ["rheumatoid", "crohn", "psoriasis"]),
    "infliximab": (TargetCategory.CYTOKINE, "TNF-α", ["rheumatoid", "crohn", "psoriasis"]),
    "ustekinumab": (TargetCategory.CYTOKINE, "IL-12/23", ["psoriasis", "crohn"]),
    "secukinumab": (TargetCategory.CYTOKINE, "IL-17A", ["psoriasis", "ankylosing"]),
    "tocilizumab": (TargetCategory.CYTOKINE, "IL-6R", ["rheumatoid"]),

    # Asthma drugs
    "omalizumab": (TargetCategory.IMMUNE, "IgE", ["asthma"]),
    "mepolizumab": (TargetCategory.CYTOKINE, "IL-5", ["asthma", "eosinophilic"]),
    "benralizumab": (TargetCategory.CYTOKINE, "IL-5R", ["asthma"]),
    "tezepelumab": (TargetCategory.CYTOKINE, "TSLP", ["asthma"]),
    "dupilumab": (TargetCategory.CYTOKINE, "IL-4Rα", ["atopic dermatitis", "asthma"]),

    # Other drugs often confused
    "fontolizumab": (TargetCategory.CYTOKINE, "IFN-γ", ["crohn"]),  # NOT MS
    "enokizumab": (TargetCategory.CYTOKINE, "IL-9", ["asthma"]),  # NOT MS
}

# Disease category patterns
AUTOIMMUNE_PATTERNS = [
    r"multiple sclerosis", r"\bms\b", r"rheumatoid", r"lupus", r"psoriasis",
    r"crohn", r"ulcerative colitis", r"ankylosing spondylitis", r"sjögren",
    r"myasthenia", r"autoimmune", r"inflammatory bowel",
]

CANCER_PATTERNS = [
    r"cancer", r"carcinoma", r"melanoma", r"leukemia", r"lymphoma",
    r"sarcoma", r"tumor", r"myeloma", r"neoplasm", r"malignant",
]

CARDIOVASCULAR_PATTERNS = [
    r"heart failure", r"hypertension", r"coronary", r"myocardial",
    r"arrhythmia", r"atrial fibrillation", r"cardiomyopathy",
]

NEUROLOGICAL_PATTERNS = [
    r"parkinson", r"alzheimer", r"dementia", r"epilepsy", r"migraine",
    r"huntington", r"als\b", r"amyotrophic", r"neurodegenerat",
]

BONE_PATTERNS = [
    r"osteoporosis", r"osteopenia", r"paget", r"bone loss",
]


@dataclass
class BiologicValidation:
    drug: str
    disease: str
    drug_target: Optional[TargetCategory]
    drug_target_name: Optional[str]
    disease_category: DiseaseCategory
    is_compatible: bool
    confidence: str  # "known", "inferred", "uncertain"
    reason: str


def categorize_disease(disease: str) -> DiseaseCategory:
    """Categorize a disease based on patterns."""
    disease_lower = disease.lower()

    for pattern in CANCER_PATTERNS:
        if re.search(pattern, disease_lower):
            return DiseaseCategory.CANCER

    for pattern in AUTOIMMUNE_PATTERNS:
        if re.search(pattern, disease_lower):
            return DiseaseCategory.AUTOIMMUNE

    for pattern in CARDIOVASCULAR_PATTERNS:
        if re.search(pattern, disease_lower):
            return DiseaseCategory.CARDIOVASCULAR

    for pattern in NEUROLOGICAL_PATTERNS:
        if re.search(pattern, disease_lower):
            return DiseaseCategory.NEUROLOGICAL

    for pattern in BONE_PATTERNS:
        if re.search(pattern, disease_lower):
            return DiseaseCategory.BONE

    return DiseaseCategory.OTHER


def extract_target_from_name(drug: str) -> Optional[TargetCategory]:
    """Extract target category from -mab naming convention."""
    drug_lower = drug.lower()

    # Must be a monoclonal antibody
    if not drug_lower.endswith("mab"):
        return None

    # Remove the source suffix (-omab, -ximab, -zumab, -umab)
    stem = drug_lower[:-3]  # Remove 'mab'

    # Check for infixes
    for infix, target in TARGET_INFIXES.items():
        # Look for infix near the end (before source suffix)
        if infix in stem[-6:]:
            return target

    return TargetCategory.UNKNOWN


def is_target_compatible(target: TargetCategory, disease_cat: DiseaseCategory) -> bool:
    """Check if a drug target is compatible with a disease category."""
    compatibility = {
        TargetCategory.TUMOR: [DiseaseCategory.CANCER],
        TargetCategory.IMMUNE: [DiseaseCategory.AUTOIMMUNE, DiseaseCategory.CANCER],
        TargetCategory.CYTOKINE: [DiseaseCategory.AUTOIMMUNE],
        TargetCategory.CARDIOVASCULAR: [DiseaseCategory.CARDIOVASCULAR],
        TargetCategory.NERVOUS: [DiseaseCategory.NEUROLOGICAL],
        TargetCategory.BONE: [DiseaseCategory.BONE],
        TargetCategory.BACTERIAL: [DiseaseCategory.INFECTIOUS],
        TargetCategory.VIRAL: [DiseaseCategory.INFECTIOUS],
        TargetCategory.UNKNOWN: [],  # Unknown targets need manual review
    }

    return disease_cat in compatibility.get(target, [])


def validate_biologic(drug: str, disease: str) -> BiologicValidation:
    """Validate a biologic drug prediction against disease category."""
    drug_lower = drug.lower()
    disease_cat = categorize_disease(disease)

    # Check if we have known target information
    if drug_lower in KNOWN_TARGETS:
        target, target_name, indications = KNOWN_TARGETS[drug_lower]

        # Check if disease is in known indications
        disease_lower = disease.lower()
        is_known_indication = any(ind in disease_lower for ind in indications)

        if is_known_indication:
            return BiologicValidation(
                drug=drug,
                disease=disease,
                drug_target=target,
                drug_target_name=target_name,
                disease_category=disease_cat,
                is_compatible=True,
                confidence="known",
                reason=f"Known indication: {drug} targets {target_name}"
            )
        else:
            # Check if target category is at least compatible
            is_compat = is_target_compatible(target, disease_cat)
            return BiologicValidation(
                drug=drug,
                disease=disease,
                drug_target=target,
                drug_target_name=target_name,
                disease_category=disease_cat,
                is_compatible=is_compat,
                confidence="known",
                reason=f"{drug} targets {target_name} ({target.value}), " +
                       (f"compatible with {disease_cat.value}" if is_compat else
                        f"NOT compatible with {disease_cat.value}")
            )

    # Infer from naming convention
    target = extract_target_from_name(drug)

    if target is None:
        return BiologicValidation(
            drug=drug,
            disease=disease,
            drug_target=None,
            drug_target_name=None,
            disease_category=disease_cat,
            is_compatible=True,  # Not a -mab, pass through
            confidence="uncertain",
            reason="Not a monoclonal antibody (-mab)"
        )

    if target == TargetCategory.UNKNOWN:
        return BiologicValidation(
            drug=drug,
            disease=disease,
            drug_target=target,
            drug_target_name=None,
            disease_category=disease_cat,
            is_compatible=False,  # Uncertain, flag for review
            confidence="uncertain",
            reason=f"Unknown target from naming convention, needs manual review"
        )

    is_compat = is_target_compatible(target, disease_cat)
    return BiologicValidation(
        drug=drug,
        disease=disease,
        drug_target=target,
        drug_target_name=None,
        disease_category=disease_cat,
        is_compatible=is_compat,
        confidence="inferred",
        reason=f"Inferred {target.value} target, " +
               (f"compatible with {disease_cat.value}" if is_compat else
                f"NOT compatible with {disease_cat.value}")
    )


if __name__ == "__main__":
    # Test with MS biologic predictions
    test_cases = [
        # True positives (should pass)
        ("Natalizumab", "multiple sclerosis"),
        ("Rituximab", "multiple sclerosis"),
        ("Eculizumab", "neuromyelitis optica"),
        ("Adalimumab", "rheumatoid arthritis"),
        ("Trastuzumab", "breast cancer"),

        # False positives (should fail)
        ("Volociximab", "multiple sclerosis"),  # Cancer drug
        ("Adecatumumab", "multiple sclerosis"),  # Cancer drug
        ("Romosozumab", "multiple sclerosis"),  # Bone drug
        ("Eptinezumab", "multiple sclerosis"),  # Migraine drug
        ("Fontolizumab", "multiple sclerosis"),  # Crohn's drug
    ]

    print("=" * 70)
    print("BIOLOGIC TARGET VALIDATION TEST")
    print("=" * 70)

    passed = 0
    failed = 0

    for drug, disease in test_cases:
        result = validate_biologic(drug, disease)
        status = "✅ PASS" if result.is_compatible else "❌ FAIL"

        print(f"\n{drug} → {disease}")
        print(f"  Target: {result.drug_target_name or result.drug_target}")
        print(f"  Disease category: {result.disease_category.value}")
        print(f"  {status} - {result.reason}")
        print(f"  Confidence: {result.confidence}")

        if result.is_compatible:
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)
