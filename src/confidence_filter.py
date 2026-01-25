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
"""

import re
from typing import Dict, List, Optional, Tuple
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

# Failed Phase III drugs for specific indications (don't repurpose)
FAILED_PHASE3_COMBINATIONS = [
    # (drug_pattern, disease_pattern, reason)
    (r"linsitinib", r"breast.*cancer", "IGF-1R inhibitors failed Phase III breast cancer trials"),
]

# Metabolic disease names
METABOLIC_DISEASES = [
    "diabetes", "type 2 diabetes", "type 1 diabetes", "metabolic syndrome",
    "obesity", "hyperglycemia", "hypoglycemia", "insulin resistance",
    "diabetic", "glycemic",
]

# Cardiac conditions where alpha blockers are harmful
CARDIAC_CONDITIONS = [
    "heart failure", "congestive heart failure", "cardiac failure",
    "cardiomyopathy", "left ventricular dysfunction",
]


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
