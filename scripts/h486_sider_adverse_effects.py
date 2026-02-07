#!/usr/bin/env python3
"""
h486: Systematic Adverse Effect Mining from SIDER Database

Cross-reference our drug predictions with SIDER adverse effects to identify
'inverse indications' — predictions where the drug is known to CAUSE the disease.

Prior work (h480/h483): Manually found 29 inverse indications → FILTER.
This script automates the process using SIDER's 309K adverse effect entries.
"""

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from production_predictor import (
    DrugRepurposingPredictor,
    INVERSE_INDICATION_PAIRS,
)


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
    """Load SIDER adverse effects, mapped to drug names.

    Returns: drug_name_lower -> set of adverse effect terms (lowercased).
    """
    sider_dir = Path(__file__).parent.parent / "data" / "external" / "sider"
    result: Dict[str, Set[str]] = defaultdict(set)

    with open(sider_dir / "meddra_all_se.tsv") as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 6:
                cid_flat = parts[0]  # Flat CID
                # Get drug name
                drug_name = drug_names.get(cid_flat, "")
                if not drug_name:
                    continue

                # Side effect name is the last column
                se_name = parts[-1].lower()
                result[drug_name].add(se_name)

    return dict(result)


def match_disease_to_adverse_effects(
    disease_name: str,
    adverse_effects: Set[str],
) -> List[str]:
    """Check if any adverse effect matches a disease name.

    Uses substring matching with safety checks for short strings.
    Returns list of matching adverse effect terms.
    """
    disease_lower = disease_name.lower()
    matches = []

    for ae in adverse_effects:
        ae_lower = ae.lower()

        # Direct match
        if ae_lower == disease_lower:
            matches.append(ae)
            continue

        # Substring matching (with min length to avoid false positives)
        # Only match if the AE term is >= 5 chars AND appears as a substring
        if len(ae_lower) >= 5 and ae_lower in disease_lower:
            matches.append(ae)
            continue

        if len(disease_lower) >= 5 and disease_lower in ae_lower:
            matches.append(ae)
            continue

        # Key medical term matching
        # Split into words and check for significant overlapping terms
        ae_words = set(ae_lower.split())
        disease_words = set(disease_lower.split())

        # Remove common stop words
        stop_words = {'of', 'the', 'and', 'or', 'in', 'a', 'an', 'with', 'type', 'nos',
                      'unspecified', 'other', 'acute', 'chronic', 'severe', 'mild',
                      'moderate', 'disorder', 'disease', 'syndrome', 'condition'}
        ae_significant = ae_words - stop_words
        disease_significant = disease_words - stop_words

        # If most significant words overlap, it's a match
        if ae_significant and disease_significant:
            overlap = ae_significant & disease_significant
            if overlap and len(overlap) >= min(len(ae_significant), len(disease_significant)):
                matches.append(ae)

    return matches


def main() -> None:
    print("=" * 70)
    print("h486: SIDER Adverse Effect Mining for Inverse Indications")
    print("=" * 70)

    # Load SIDER data
    print("\nLoading SIDER data...")
    sider_names = load_sider_drug_names()
    print(f"  Drug names: {len(sider_names)}")

    sider_aes = load_sider_adverse_effects(sider_names)
    total_aes = sum(len(v) for v in sider_aes.values())
    print(f"  Drugs with AEs: {len(sider_aes)}")
    print(f"  Total AE entries: {total_aes}")

    # Load predictor
    print("\nLoading predictor...")
    predictor = DrugRepurposingPredictor()

    # Get existing inverse indications (dict: drug_name -> set of disease names)
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

    # Get all diseases with embeddings
    all_diseases = [d for d in predictor.embeddings if d in predictor.disease_names]

    # Cross-reference predictions with adverse effects
    print("\nCross-referencing predictions with SIDER adverse effects...")
    inverse_candidates: List[Dict] = []
    n_checked = 0

    for idx, disease_id in enumerate(all_diseases):
        disease_name = predictor.disease_names.get(disease_id, disease_id)

        try:
            result = predictor.predict(disease_name, top_n=30, include_filtered=True)
        except Exception:
            continue

        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx+1}/{len(all_diseases)} diseases, {len(inverse_candidates)} candidates so far...")

        for pred in result.predictions:
            drug_id = pred.drug_id
            if drug_id not in our_to_sider:
                continue

            sider_drug = our_to_sider[drug_id]
            aes = sider_aes[sider_drug]

            # Check if disease matches any adverse effect
            matches = match_disease_to_adverse_effects(disease_name, aes)
            if matches:
                # Check if already in existing inverse indications
                pair = (pred.drug_name.lower(), disease_name.lower())
                already_known = pair in existing_inverse

                inverse_candidates.append({
                    'drug_name': pred.drug_name,
                    'drug_id': drug_id,
                    'disease_name': disease_name,
                    'disease_id': disease_id,
                    'confidence_tier': pred.confidence_tier.name,
                    'tier_rule': pred.category_specific_tier or 'standard',
                    'category': result.category,
                    'rank': pred.rank,
                    'ae_matches': matches[:5],  # Top 5 matching AE terms
                    'already_known': already_known,
                })

            n_checked += 1

    print(f"\nChecked {n_checked} drug-disease predictions with SIDER data")
    print(f"Found {len(inverse_candidates)} inverse indication candidates")

    # Separate new vs already known
    new_candidates = [c for c in inverse_candidates if not c['already_known']]
    known_candidates = [c for c in inverse_candidates if c['already_known']]
    print(f"  Already known (in INVERSE_INDICATION_PAIRS): {len(known_candidates)}")
    print(f"  NEW candidates: {len(new_candidates)}")

    # Print results by tier
    print("\n" + "=" * 70)
    print("NEW INVERSE INDICATION CANDIDATES BY TIER")
    print("=" * 70)

    by_tier: Dict[str, List] = defaultdict(list)
    for c in new_candidates:
        by_tier[c['confidence_tier']].append(c)

    for tier in ['GOLDEN', 'HIGH', 'MEDIUM', 'LOW', 'FILTER']:
        candidates = by_tier.get(tier, [])
        if not candidates:
            continue
        print(f"\n--- {tier} ({len(candidates)} candidates) ---")
        # Group by drug
        by_drug: Dict[str, List] = defaultdict(list)
        for c in candidates:
            by_drug[c['drug_name']].append(c)

        for drug, preds in sorted(by_drug.items(), key=lambda x: -len(x[1]))[:20]:
            diseases = [p['disease_name'][:40] for p in preds[:3]]
            ae_sample = preds[0]['ae_matches'][:2]
            print(f"  {drug:<25} → {', '.join(diseases)} (AE: {ae_sample})")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Count by category
    by_cat: Dict[str, int] = defaultdict(int)
    for c in new_candidates:
        by_cat[c['category']] += 1

    print(f"\nNew candidates by category:")
    for cat, n in sorted(by_cat.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {n}")

    # Count drugs with multiple inverse indication predictions
    drug_counts: Dict[str, int] = defaultdict(int)
    for c in new_candidates:
        drug_counts[c['drug_name']] += 1

    multi_drugs = [(d, n) for d, n in drug_counts.items() if n >= 3]
    if multi_drugs:
        print(f"\nDrugs with 3+ inverse indication predictions:")
        for drug, n in sorted(multi_drugs, key=lambda x: -x[1])[:20]:
            print(f"  {drug}: {n} predictions")

    # Save results
    import json
    output_path = Path("data/analysis/h486_sider_inverse.json")
    with open(output_path, "w") as f:
        json.dump({
            "total_candidates": len(inverse_candidates),
            "new_candidates": len(new_candidates),
            "known_candidates": len(known_candidates),
            "new_by_tier": {t: len(cs) for t, cs in by_tier.items()},
            "candidates": new_candidates[:500],  # Save top 500
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
