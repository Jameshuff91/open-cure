#!/usr/bin/env python3
"""
Confounding Pattern Detector for Drug Repurposing Predictions.

Identifies false positive patterns discovered through deep dive analysis:
1. Cardiac-Metabolic Comorbidity - HF drugs appearing connected to T2D
2. Inverse Indication - Drug causes disease but prescribed for other benefits
3. Polypharmacy DDI - Drug interaction studies miscounted as treatment trials

Based on deep dives of Digoxin→T2D and Simvastatin→T2D false positives.
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from enum import Enum


class ConfoundingType(Enum):
    NONE = "none"
    CARDIAC_METABOLIC = "cardiac_metabolic"  # HF drugs + T2D
    INVERSE_INDICATION = "inverse_indication"  # Drug causes disease
    POLYPHARMACY = "polypharmacy"  # DDI studies, not treatment
    COMORBIDITY = "comorbidity"  # General comorbidity confounding
    MECHANISM_MISMATCH = "mechanism_mismatch"  # Drug mechanism worsens disease


@dataclass
class ConfoundingResult:
    drug: str
    disease: str
    confounding_type: ConfoundingType
    confidence: float  # 0-1, how confident we are this is confounded
    reason: str
    evidence: Optional[str] = None


# =============================================================================
# CARDIAC DRUGS (HF, arrhythmia, etc.)
# =============================================================================
CARDIAC_GLYCOSIDES = [
    "digoxin", "digitoxin", "ouabain", "digitalis",
]

LOOP_DIURETICS = [
    "furosemide", "bumetanide", "torsemide", "ethacrynic",
]

THIAZIDE_DIURETICS = [
    "hydrochlorothiazide", "chlorthalidone", "indapamide", "metolazone",
]

POTASSIUM_SPARING = [
    "spironolactone", "eplerenone", "amiloride", "triamterene",
]

BETA_BLOCKERS = [
    "metoprolol", "carvedilol", "bisoprolol", "atenolol", "propranolol",
    "labetalol", "nebivolol", "nadolol", "timolol", "sotalol",
]

# NOTE: ACE inhibitors and ARBs are EXCLUDED from cardiac-metabolic confounding
# because they actually PREVENT T2D (HOPE trial: 34% reduction in new diagnoses)
ACE_INHIBITORS = [
    "lisinopril", "enalapril", "ramipril", "captopril", "benazepril",
    "fosinopril", "quinapril", "perindopril", "trandolapril",
]

ARBS = [
    "losartan", "valsartan", "irbesartan", "olmesartan", "candesartan",
    "telmisartan", "azilsartan", "eprosartan",
]

# Drugs that PREVENT or TREAT diabetes (should NOT be flagged as confounded)
# - ACE inhibitors/ARBs: HOPE trial showed 34% reduction in new T2D diagnoses
# - Verapamil: RCT evidence for beta-cell preservation, HbA1c reduction (WMD -0.45)
DIABETES_PROTECTIVE_DRUGS = ACE_INHIBITORS + ARBS + ["verapamil"]

CCBs = [
    "amlodipine", "nifedipine", "diltiazem", "verapamil", "felodipine",
    "nicardipine", "isradipine", "nimodipine",
]

ANTIARRHYTHMICS = [
    "amiodarone", "dronedarone", "flecainide", "propafenone", "sotalol",
    "dofetilide", "ibutilide", "lidocaine", "mexiletine", "quinidine",
]

INOTROPES = [
    "dobutamine", "milrinone", "levosimendan", "dopamine",
]

# Cardiac drugs that MAY be confounded with metabolic diseases
# EXCLUDES ACE inhibitors and ARBs (they prevent T2D, so not confounded)
CARDIAC_DRUGS_CONFOUNDED = (
    CARDIAC_GLYCOSIDES + LOOP_DIURETICS + THIAZIDE_DIURETICS +
    POTASSIUM_SPARING + BETA_BLOCKERS + CCBs + ANTIARRHYTHMICS + INOTROPES
)

# All cardiac drugs (for reference)
ALL_CARDIAC_DRUGS = CARDIAC_DRUGS_CONFOUNDED + ACE_INHIBITORS + ARBS

# =============================================================================
# METABOLIC DISEASES
# =============================================================================
METABOLIC_DISEASES = [
    "type 2 diabetes", "type 1 diabetes", "diabetes mellitus",
    "t2d", "t1d", "metabolic syndrome", "insulin resistance",
    "hyperglycemia", "glucose intolerance", "prediabetes",
]

# =============================================================================
# INVERSE INDICATION PATTERNS
# Drug causes the disease but is prescribed for other benefits
# =============================================================================
INVERSE_INDICATION_PAIRS = [
    # Statins cause T2D but prescribed for CV protection
    ("simvastatin", "type 2 diabetes", "Statins INCREASE T2D risk (HR 1.12-1.44) but prescribed for CV protection"),
    ("atorvastatin", "type 2 diabetes", "Statins INCREASE T2D risk (HR 1.12-1.44) but prescribed for CV protection"),
    ("rosuvastatin", "type 2 diabetes", "Statins INCREASE T2D risk (HR 1.12-1.44) but prescribed for CV protection"),
    ("pravastatin", "type 2 diabetes", "Statins INCREASE T2D risk (HR 1.12-1.44) but prescribed for CV protection"),
    ("lovastatin", "type 2 diabetes", "Statins INCREASE T2D risk (HR 1.12-1.44) but prescribed for CV protection"),
    ("fluvastatin", "type 2 diabetes", "Statins INCREASE T2D risk (HR 1.12-1.44) but prescribed for CV protection"),
    ("pitavastatin", "type 2 diabetes", "Statins may increase T2D risk but prescribed for CV protection"),

    # Thiazides cause hyperglycemia
    ("hydrochlorothiazide", "type 2 diabetes", "Thiazides cause hyperglycemia but prescribed for hypertension"),
    ("chlorthalidone", "type 2 diabetes", "Thiazides cause hyperglycemia but prescribed for hypertension"),

    # Beta blockers can worsen glucose control
    ("metoprolol", "type 2 diabetes", "Beta blockers mask hypoglycemia and may worsen glucose control"),
    ("atenolol", "type 2 diabetes", "Beta blockers mask hypoglycemia and may worsen glucose control"),
    ("propranolol", "type 2 diabetes", "Beta blockers mask hypoglycemia and may worsen glucose control"),

    # Antipsychotics cause metabolic syndrome
    ("olanzapine", "type 2 diabetes", "Atypical antipsychotics CAUSE metabolic syndrome and T2D"),
    ("clozapine", "type 2 diabetes", "Atypical antipsychotics CAUSE metabolic syndrome and T2D"),
    ("quetiapine", "type 2 diabetes", "Atypical antipsychotics CAUSE metabolic syndrome and T2D"),
    ("risperidone", "type 2 diabetes", "Atypical antipsychotics CAUSE metabolic syndrome and T2D"),

    # Corticosteroids cause hyperglycemia
    ("prednisone", "type 2 diabetes", "Corticosteroids cause hyperglycemia"),
    ("prednisolone", "type 2 diabetes", "Corticosteroids cause hyperglycemia"),
    ("dexamethasone", "type 2 diabetes", "Corticosteroids cause hyperglycemia"),
    ("methylprednisolone", "type 2 diabetes", "Corticosteroids cause hyperglycemia"),
]

# =============================================================================
# MECHANISM MISMATCH - Drug mechanism worsens the disease
# =============================================================================
MECHANISM_MISMATCH_PAIRS = [
    # Digoxin worsens glucose via Na+/K+-ATPase inhibition
    ("digoxin", "type 2 diabetes", "Na+/K+-ATPase inhibition reduces GLUT4 translocation, worsening glucose"),
    ("digitoxin", "type 2 diabetes", "Na+/K+-ATPase inhibition reduces GLUT4 translocation, worsening glucose"),

    # NSAIDs worsen kidney disease
    ("ibuprofen", "chronic kidney disease", "NSAIDs reduce renal blood flow, worsening CKD"),
    ("naproxen", "chronic kidney disease", "NSAIDs reduce renal blood flow, worsening CKD"),
    ("diclofenac", "chronic kidney disease", "NSAIDs reduce renal blood flow, worsening CKD"),

    # Decongestants worsen hypertension
    ("pseudoephedrine", "hypertension", "Sympathomimetics raise blood pressure"),
    ("phenylephrine", "hypertension", "Sympathomimetics raise blood pressure"),

    # Checkpoint inhibitors CAUSE immune-related colitis (irAE)
    ("pembrolizumab", "ulcerative colitis", "Checkpoint inhibitors CAUSE immune-related colitis"),
    ("nivolumab", "ulcerative colitis", "Checkpoint inhibitors CAUSE immune-related colitis"),
    ("ipilimumab", "ulcerative colitis", "Checkpoint inhibitors CAUSE immune-related colitis"),
    ("atezolizumab", "ulcerative colitis", "Checkpoint inhibitors CAUSE immune-related colitis"),
    ("durvalumab", "ulcerative colitis", "Checkpoint inhibitors CAUSE immune-related colitis"),

    # Antipsychotics cause drug-induced parkinsonism
    ("haloperidol", "parkinson", "Antipsychotics cause drug-induced parkinsonism via D2 blockade"),
    ("chlorpromazine", "parkinson", "Antipsychotics cause drug-induced parkinsonism via D2 blockade"),
    ("risperidone", "parkinson", "Antipsychotics cause drug-induced parkinsonism via D2 blockade"),
    ("olanzapine", "parkinson", "Antipsychotics cause drug-induced parkinsonism via D2 blockade"),
    ("quetiapine", "parkinson", "Antipsychotics cause drug-induced parkinsonism via D2 blockade"),
]


def is_cardiac_drug(drug: str) -> bool:
    """Check if drug is in a cardiac category that may be confounded with metabolic diseases."""
    drug_lower = drug.lower()
    # Exclude ACE inhibitors and ARBs - they actually PREVENT T2D
    if any(d in drug_lower for d in DIABETES_PROTECTIVE_DRUGS):
        return False
    return any(d in drug_lower for d in CARDIAC_DRUGS_CONFOUNDED)


def is_metabolic_disease(disease: str) -> bool:
    """Check if disease is metabolic/diabetes-related."""
    disease_lower = disease.lower()
    return any(d in disease_lower for d in METABOLIC_DISEASES)


def check_inverse_indication(drug: str, disease: str) -> Optional[Tuple[str, float]]:
    """Check if this is a known inverse indication pattern."""
    drug_lower = drug.lower()
    disease_lower = disease.lower()

    for inv_drug, inv_disease, reason in INVERSE_INDICATION_PAIRS:
        if inv_drug in drug_lower and inv_disease in disease_lower:
            return (reason, 0.9)
    return None


def check_mechanism_mismatch(drug: str, disease: str) -> Optional[Tuple[str, float]]:
    """Check if drug mechanism is known to worsen the disease."""
    drug_lower = drug.lower()
    disease_lower = disease.lower()

    for mm_drug, mm_disease, reason in MECHANISM_MISMATCH_PAIRS:
        if mm_drug in drug_lower and mm_disease in disease_lower:
            return (reason, 0.85)
    return None


def detect_confounding(drug: str, disease: str, trial_count: int = 0) -> ConfoundingResult:
    """
    Detect confounding patterns in a drug-disease prediction.

    Args:
        drug: Drug name
        disease: Disease name
        trial_count: Number of clinical trials (high count with metabolic disease = suspicious)
        pub_count: Number of publications

    Returns:
        ConfoundingResult with detected pattern
    """
    # Check inverse indication first (highest confidence)
    inv_result = check_inverse_indication(drug, disease)
    if inv_result:
        reason, conf = inv_result
        return ConfoundingResult(
            drug=drug,
            disease=disease,
            confounding_type=ConfoundingType.INVERSE_INDICATION,
            confidence=conf,
            reason=reason,
            evidence=f"Drug is known to CAUSE or WORSEN this condition"
        )

    # Check mechanism mismatch
    mech_result = check_mechanism_mismatch(drug, disease)
    if mech_result:
        reason, conf = mech_result
        return ConfoundingResult(
            drug=drug,
            disease=disease,
            confounding_type=ConfoundingType.MECHANISM_MISMATCH,
            confidence=conf,
            reason=reason,
            evidence=f"Drug mechanism worsens disease pathophysiology"
        )

    # Check cardiac-metabolic comorbidity pattern
    if is_cardiac_drug(drug) and is_metabolic_disease(disease):
        # High trial count with cardiac drug + metabolic disease is suspicious
        # These are likely trials for CV protection IN diabetics, not treating T2D
        if trial_count >= 5:
            return ConfoundingResult(
                drug=drug,
                disease=disease,
                confounding_type=ConfoundingType.CARDIAC_METABOLIC,
                confidence=0.7,
                reason=f"Cardiac drug with {trial_count} metabolic disease trials - likely CV protection studies",
                evidence="HF and T2D have 75% comorbidity; trials likely studying CV outcomes in diabetics"
            )
        else:
            return ConfoundingResult(
                drug=drug,
                disease=disease,
                confounding_type=ConfoundingType.CARDIAC_METABOLIC,
                confidence=0.5,
                reason="Cardiac drug predicted for metabolic disease - comorbidity confounding possible",
                evidence="HF and T2D frequently co-occur, creating false association"
            )

    # No confounding detected
    return ConfoundingResult(
        drug=drug,
        disease=disease,
        confounding_type=ConfoundingType.NONE,
        confidence=0.0,
        reason="No known confounding pattern detected",
    )


def scan_validations(cache_path: str) -> Dict:
    """
    Scan validation cache for confounding patterns.

    Returns:
        Dictionary with confounded predictions and statistics
    """
    with open(cache_path) as f:
        cache = json.load(f)

    results = {
        "total": len(cache),
        "confounded": [],
        "by_type": {t.value: [] for t in ConfoundingType},
        "summary": {}
    }

    for key, val in cache.items():
        parts = key.split('|')
        if len(parts) != 2:
            continue

        drug, disease = parts
        trial_count = val.get('clinical_trials', {}).get('trial_count', 0)
        pub_count = val.get('pubmed', {}).get('publication_count', 0)

        result = detect_confounding(drug, disease, trial_count)

        if result.confounding_type != ConfoundingType.NONE:
            entry = {
                "drug": drug,
                "disease": disease,
                "type": result.confounding_type.value,
                "confidence": result.confidence,
                "reason": result.reason,
                "evidence": result.evidence,
                "trial_count": trial_count,
                "pub_count": pub_count,
            }
            results["confounded"].append(entry)
            results["by_type"][result.confounding_type.value].append(entry)

    # Summary
    results["summary"] = {
        "total_scanned": len(cache),
        "total_confounded": len(results["confounded"]),
        "confounded_pct": len(results["confounded"]) / len(cache) * 100 if cache else 0,
        "by_type_counts": {k: len(v) for k, v in results["by_type"].items()}
    }

    return results


def print_report(results: Dict) -> None:
    """Print a formatted report of confounding detection."""
    print("\n" + "=" * 70)
    print("CONFOUNDING PATTERN DETECTION REPORT")
    print("=" * 70)

    summary = results["summary"]
    print(f"\nTotal predictions scanned: {summary['total_scanned']}")
    print(f"Potentially confounded:    {summary['total_confounded']} ({summary['confounded_pct']:.1f}%)")

    print("\nBreakdown by type:")
    for ctype, count in summary["by_type_counts"].items():
        if count > 0:
            print(f"  {ctype}: {count}")

    # High confidence confounding
    high_conf = [c for c in results["confounded"] if c["confidence"] >= 0.7]
    if high_conf:
        print(f"\n{'=' * 70}")
        print(f"HIGH CONFIDENCE FALSE POSITIVES ({len(high_conf)})")
        print("=" * 70)
        for c in sorted(high_conf, key=lambda x: -x["confidence"]):
            print(f"\n  ❌ {c['drug']} → {c['disease']}")
            print(f"     Type: {c['type']}")
            print(f"     Confidence: {c['confidence']:.0%}")
            print(f"     Reason: {c['reason']}")
            if c.get('evidence'):
                print(f"     Evidence: {c['evidence']}")
            print(f"     Trials: {c['trial_count']}, Pubs: {c['pub_count']}")

    # Medium confidence
    med_conf = [c for c in results["confounded"] if 0.4 <= c["confidence"] < 0.7]
    if med_conf:
        print(f"\n{'=' * 70}")
        print(f"MEDIUM CONFIDENCE - NEEDS REVIEW ({len(med_conf)})")
        print("=" * 70)
        for c in sorted(med_conf, key=lambda x: -x["confidence"])[:20]:
            print(f"  ⚠️  {c['drug']} → {c['disease']} ({c['type']}, {c['confidence']:.0%})")


if __name__ == "__main__":
    cache_path = Path(__file__).parent.parent / "data" / "validation" / "validation_cache.json"

    if not cache_path.exists():
        print(f"Cache not found: {cache_path}")
        exit(1)

    results = scan_validations(str(cache_path))
    print_report(results)

    # Save results
    output_path = Path(__file__).parent.parent / "data" / "analysis" / "confounding_analysis.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
