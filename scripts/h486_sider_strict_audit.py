#!/usr/bin/env python3
"""
h486 (continued): Strict SIDER Adverse Effect Matching + Manual Audit

The original h486 script produced 1,462 false positives due to loose substring matching.
This script uses strict matching (exact or very close) and cross-checks against SIDER
indications to remove cases where the drug is known to TREAT the disease.

Matching rules:
1. EXACT match: disease_name == AE term (case-insensitive)
2. MEDICAL SYNONYM match: disease name contains the full AE term as a distinct medical concept
   (not a substring of a different word). Minimum AE term length >= 8 chars.
3. SIDER INDICATION EXCLUSION: If SIDER also lists the drug as indicated for that disease, SKIP.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from production_predictor import (
    DrugRepurposingPredictor,
    INVERSE_INDICATION_PAIRS,
)


# Known legitimate treatments that should NEVER be flagged as inverse indications
# These are cases where the drug causes a related but distinct condition, not the disease itself
KNOWN_TREATMENTS: Set[Tuple[str, str]] = {
    # Corticosteroids treat ulcerative colitis (AE is gastric ulcer, NOT UC)
    ("dexamethasone", "ulcerative colitis"),
    ("prednisone", "ulcerative colitis"),
    ("prednisolone", "ulcerative colitis"),
    ("methylprednisolone", "ulcerative colitis"),
    ("budesonide", "ulcerative colitis"),
    ("triamcinolone", "ulcerative colitis"),
    ("betamethasone", "ulcerative colitis"),
    ("hydrocortisone", "ulcerative colitis"),
    # Corticosteroids treat SLE
    ("dexamethasone", "systemic lupus erythematosus"),
    ("prednisone", "systemic lupus erythematosus"),
    ("prednisolone", "systemic lupus erythematosus"),
    ("methylprednisolone", "systemic lupus erythematosus"),
    ("triamcinolone", "systemic lupus erythematosus"),
    ("betamethasone", "systemic lupus erythematosus"),
    ("cortisone", "systemic lupus erythematosus"),
    # Corticosteroids treat asthma
    ("triamcinolone", "bronchial asthma"),
    ("budesonide", "bronchial asthma"),
    ("fluticasone", "bronchial asthma"),
    ("mometasone", "bronchial asthma"),
    # Antihypertensives treat hypertension (AE is hypotension, not hypertension)
    ("hydrochlorothiazide", "hypertension"),
    ("amiloride", "hypertension"),
    ("carvedilol", "hypertension"),
    ("prazosin", "hypertension"),
    ("lisinopril", "hypertension"),
    ("telmisartan", "hypertension"),
    ("verapamil", "hypertension"),
    # Antiarrhythmics treat arrhythmia (proarrhythmia is a distinct concern, handled by h484/h495)
    ("flecainide", "tachycardiac atrial fibrillation"),
    ("sotalol", "tachyarrhythmia"),
    ("propafenone", "tachyarrhythmia"),
    ("diltiazem", "tachyarrhythmia"),
    ("quinidine", "tachyarrhythmia"),
    ("adenosine", "tachyarrhythmia"),
    # Lipid-lowering drugs treat hypercholesterolemia
    ("ezetimibe", "familial hypercholesterolemia"),
    ("ezetimibe", "hypercholesterolemia"),
    # Diabetes drugs treat diabetes
    ("glimepiride", "type 1 diabetes mellitus"),
    ("glimepiride", "type 2 diabetes mellitus"),
    ("rosiglitazone", "type 1 diabetes mellitus"),
    ("rosiglitazone", "type 2 diabetes mellitus"),
    ("colesevelam", "type 2 diabetes mellitus"),
    ("colesevelam", "diabetes mellitus"),
    # Anti-obesity drugs treat obesity
    ("orlistat", "obesity"),
    # Antibiotics treat infections
    ("azithromycin", "streptococcal infections"),
    ("azithromycin", "pneumococcal infections"),
    ("doxycycline", "helicobacter pylori infection"),
    ("ciprofloxacin", "inhalational anthrax"),
    ("erythromycin", "streptococcal infections"),
    # Tacrolimus treats UC (different from colitis AE which is drug-induced colitis)
    ("tacrolimus", "ulcerative colitis"),
    # Azathioprine treats UC
    ("azathioprine", "ulcerative colitis"),
}


def load_sider_drug_names() -> Dict[str, str]:
    """Load SIDER CID → drug name mapping."""
    sider_dir = Path(__file__).parent.parent / "data" / "external" / "sider"
    result: Dict[str, str] = {}
    with open(sider_dir / "drug_names.tsv") as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                result[parts[0]] = parts[1].lower()
    return result


def load_sider_adverse_effects(drug_names: Dict[str, str]) -> Dict[str, Set[str]]:
    """Load SIDER adverse effects mapped to drug names."""
    sider_dir = Path(__file__).parent.parent / "data" / "external" / "sider"
    result: Dict[str, Set[str]] = defaultdict(set)
    with open(sider_dir / "meddra_all_se.tsv") as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 6:
                cid_flat = parts[0]
                drug_name = drug_names.get(cid_flat, "")
                if not drug_name:
                    continue
                se_name = parts[-1].lower()
                result[drug_name].add(se_name)
    return dict(result)


def load_sider_indications(drug_names: Dict[str, str]) -> Dict[str, Set[str]]:
    """Load SIDER indications (what drugs are used to TREAT)."""
    sider_dir = Path(__file__).parent.parent / "data" / "external" / "sider"
    result: Dict[str, Set[str]] = defaultdict(set)
    with open(sider_dir / "meddra_all_indications.tsv") as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 7:
                cid = parts[0]
                drug_name = drug_names.get(cid, "")
                if not drug_name:
                    continue
                indication = parts[-1].lower()
                result[drug_name].add(indication)
    return dict(result)


def normalize_name(name: str) -> str:
    """Normalize disease/AE name for comparison."""
    name = name.lower().strip()
    # Remove common suffixes/prefixes
    name = re.sub(r'\s+', ' ', name)
    return name


def strict_match(disease_name: str, adverse_effects: Set[str]) -> List[str]:
    """Strict matching: only exact or near-exact matches.

    Rules:
    1. Exact match (after normalization)
    2. AE is a complete phrase within disease name (word-boundary, min 8 chars)
    3. Disease name is a complete phrase within AE (word-boundary, min 8 chars)

    Does NOT match:
    - Partial word matches ("ulcer" in "ulcerative")
    - Short substring matches
    - Generic terms ("infection", "disorder", "pain")
    """
    disease_lower = normalize_name(disease_name)
    matches = []

    # Generic AE terms that match too broadly
    too_generic = {
        'infection', 'pain', 'disorder', 'disease', 'syndrome', 'failure',
        'inflammation', 'cancer', 'tumour', 'tumor', 'neoplasm', 'ulcer',
        'oedema', 'edema', 'nausea', 'fever', 'rash', 'anaemia', 'anemia',
        'arthritis', 'neuropathy', 'hepatitis', 'colitis', 'dermatitis',
        'pneumonia', 'meningitis', 'encephalitis', 'nephritis', 'vasculitis',
        'myocarditis', 'pancreatitis', 'enteritis', 'bronchitis', 'sinusitis',
        'urticaria', 'asthma', 'angioedema', 'hypertension', 'hypotension',
        'tachycardia', 'bradycardia', 'arrhythmia', 'seizure', 'tremor',
        'anxiety', 'depression', 'insomnia', 'fatigue', 'diarrhoea',
        'constipation', 'vomiting', 'headache', 'dizziness', 'pruritus',
        'alopecia', 'erythema', 'cough', 'dyspnoea', 'asthenia',
        'neuralgia', 'chorea', 'bursitis', 'gingivitis', 'blepharitis',
        'tension', 'cardiac arrest', 'cardiac disorder',
        'breast disorder', 'breast cancer', 'skin disorder',
        'withdrawal syndrome', 'encephalopathy', 'cytopenia',
        'thrombocytopenia', 'herpes simplex', 'herpes nos', 'herpes zoster',
        'neoplasm', 'lymphoma', 'leukaemia', 'leukemia',
        'renal cell carcinoma', 'ovarian cancer',
        'coronary artery occlusion', 'coronary artery disease',
        'acute coronary syndrome', 'optic neuritis',
        'neuropathy peripheral', 'diabetes mellitus',
        'heterozygous familial hypercholesterolemia',
    }

    for ae in adverse_effects:
        ae_lower = normalize_name(ae)

        # Skip too-generic terms
        if ae_lower in too_generic:
            continue

        # Rule 1: Exact match
        if ae_lower == disease_lower:
            matches.append(ae)
            continue

        # Rule 2: AE appears as complete phrase in disease name (word-boundary)
        # Must be >= 8 chars to avoid short-string false positives
        if len(ae_lower) >= 8:
            # Check word-boundary match (not substring of a different word)
            pattern = r'\b' + re.escape(ae_lower) + r'\b'
            if re.search(pattern, disease_lower):
                matches.append(ae)
                continue

        # Rule 3: Disease appears as complete phrase in AE
        if len(disease_lower) >= 8:
            pattern = r'\b' + re.escape(disease_lower) + r'\b'
            if re.search(pattern, ae_lower):
                matches.append(ae)
                continue

    return matches


def main() -> None:
    print("=" * 70)
    print("h486 (continued): Strict SIDER Adverse Effect Audit")
    print("=" * 70)

    # Load SIDER data
    print("\nLoading SIDER data...")
    sider_names = load_sider_drug_names()
    sider_aes = load_sider_adverse_effects(sider_names)
    sider_indications = load_sider_indications(sider_names)
    total_aes = sum(len(v) for v in sider_aes.values())
    total_inds = sum(len(v) for v in sider_indications.values())
    print(f"  Drugs with AEs: {len(sider_aes)} ({total_aes} total entries)")
    print(f"  Drugs with indications: {len(sider_indications)} ({total_inds} total entries)")

    # Load predictor
    print("\nLoading predictor...")
    predictor = DrugRepurposingPredictor()

    # Get existing inverse indications
    existing_inverse: Set[Tuple[str, str]] = set()
    for drug, diseases in INVERSE_INDICATION_PAIRS.items():
        for disease in diseases:
            existing_inverse.add((drug.lower(), disease.lower()))
    print(f"Existing inverse indications: {len(existing_inverse)} pairs")

    # Map our drug names to SIDER drug names
    our_to_sider: Dict[str, str] = {}
    for drug_id, drug_name in predictor.drug_id_to_name.items():
        drug_lower = drug_name.lower()
        if drug_lower in sider_aes:
            our_to_sider[drug_id] = drug_lower
    print(f"Our drugs with SIDER AE data: {len(our_to_sider)}")

    # Cross-reference with strict matching
    print("\nStrict cross-referencing predictions vs SIDER adverse effects...")
    all_diseases = [d for d in predictor.embeddings if d in predictor.disease_names]

    candidates: List[Dict] = []
    n_checked = 0
    n_sider_indication_excluded = 0
    n_known_treatment_excluded = 0

    for idx, disease_id in enumerate(all_diseases):
        disease_name = predictor.disease_names.get(disease_id, disease_id)

        try:
            result = predictor.predict(disease_name, top_n=30, include_filtered=True)
        except Exception:
            continue

        for pred in result.predictions:
            drug_id = pred.drug_id
            if drug_id not in our_to_sider:
                continue

            sider_drug = our_to_sider[drug_id]
            aes = sider_aes[sider_drug]

            n_checked += 1

            # Strict matching
            matches = strict_match(disease_name, aes)
            if not matches:
                continue

            # Check known treatment exclusions
            pair_key = (pred.drug_name.lower(), disease_name.lower())
            if pair_key in KNOWN_TREATMENTS:
                n_known_treatment_excluded += 1
                continue

            # Check SIDER indications - if drug is indicated for this disease, SKIP
            drug_indications = sider_indications.get(sider_drug, set())
            disease_lower = disease_name.lower()
            indication_match = False
            for ind in drug_indications:
                ind_lower = ind.lower()
                if ind_lower == disease_lower:
                    indication_match = True
                    break
                # Check if indication contains disease name or vice versa (word boundary)
                if len(disease_lower) >= 8:
                    if re.search(r'\b' + re.escape(disease_lower) + r'\b', ind_lower):
                        indication_match = True
                        break
                if len(ind_lower) >= 8:
                    if re.search(r'\b' + re.escape(ind_lower) + r'\b', disease_lower):
                        indication_match = True
                        break

            if indication_match:
                n_sider_indication_excluded += 1
                continue

            # Check if already known
            already_known = pair_key in existing_inverse

            candidates.append({
                'drug_name': pred.drug_name,
                'drug_id': drug_id,
                'disease_name': disease_name,
                'disease_id': disease_id,
                'confidence_tier': pred.confidence_tier.name,
                'tier_rule': pred.category_specific_tier or 'standard',
                'category': result.category,
                'rank': pred.rank,
                'ae_matches': matches[:5],
                'already_known': already_known,
            })

    new_candidates = [c for c in candidates if not c['already_known']]
    known_candidates = [c for c in candidates if c['already_known']]

    print(f"\nChecked {n_checked} drug-disease predictions with SIDER data")
    print(f"SIDER indication exclusions: {n_sider_indication_excluded}")
    print(f"Known treatment exclusions: {n_known_treatment_excluded}")
    print(f"Total candidates: {len(candidates)}")
    print(f"  Already known: {len(known_candidates)}")
    print(f"  NEW: {len(new_candidates)}")

    # Print results by tier for manual audit
    print("\n" + "=" * 70)
    print("NEW INVERSE INDICATION CANDIDATES (STRICT MATCHING)")
    print("=" * 70)

    by_tier: Dict[str, List] = defaultdict(list)
    for c in new_candidates:
        by_tier[c['confidence_tier']].append(c)

    for tier in ['GOLDEN', 'HIGH', 'MEDIUM', 'LOW', 'FILTER']:
        tier_cands = by_tier.get(tier, [])
        if not tier_cands:
            continue
        print(f"\n--- {tier} ({len(tier_cands)} candidates) ---")
        for c in tier_cands:
            print(f"  {c['drug_name']:<25} → {c['disease_name']:<45} (AE: {c['ae_matches'][:3]}) [{c['category']}]")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY FOR MANUAL AUDIT")
    print("=" * 70)
    print(f"\nTotal new candidates to audit: {len(new_candidates)}")
    print(f"By tier:")
    for tier in ['GOLDEN', 'HIGH', 'MEDIUM', 'LOW', 'FILTER']:
        print(f"  {tier}: {len(by_tier.get(tier, []))}")

    # Save results
    output_path = Path("data/analysis/h486_strict_audit.json")
    with open(output_path, "w") as f:
        json.dump({
            "total_checked": n_checked,
            "sider_indication_excluded": n_sider_indication_excluded,
            "known_treatment_excluded": n_known_treatment_excluded,
            "total_candidates": len(candidates),
            "new_candidates_count": len(new_candidates),
            "already_known_count": len(known_candidates),
            "new_by_tier": {t: len(cs) for t, cs in by_tier.items()},
            "candidates": new_candidates,
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
