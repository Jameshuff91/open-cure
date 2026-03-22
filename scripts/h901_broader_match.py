"""h901 Phase 2: Broader word-overlap matching for unresolved diseases.

Maps clinical-specific Every Cure disease names to broader DRKG MeSH terms
using word-overlap Jaccard similarity (threshold ≥ 0.6 to avoid false positives).

Example: "Advanced breast cancer" → "Breast cancer" (MESH:D001943)
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent.parent
REF_DIR = DATA_DIR / "data" / "reference"
ANALYSIS_DIR = DATA_DIR / "data" / "analysis"

JACCARD_THRESHOLD = 0.6
STOPWORDS = {
    'of', 'the', 'and', 'or', 'in', 'with', 'without', 'by', 'to',
    'a', 'an', 'for', 'not', 'on', 'at', 'is', 'as', 'from', 'due',
    'related', 'associated', 'induced', 'caused', 'other', 'nos',
    'unspecified', 'finding', 'disorder', 'condition', 'ctcae',
}

# Clinical qualifiers that are safe to strip for parent matching
STRIP_PREFIXES = [
    'acute ', 'chronic ', 'recurrent ', 'advanced ', 'metastatic ',
    'refractory ', 'relapsed ', 'persistent ', 'severe ', 'moderate ',
    'mild ', 'primary ', 'secondary ', 'progressive ', 'early ',
    'late ', 'localized ', 'locally advanced ', 'unresectable ',
    'newly diagnosed ', 'relapsed or refractory ', 'symptomatic ',
    'asymptomatic ', 'resistant ', 'sensitive ', 'complicated ',
    'uncomplicated ',
]

# Known abbreviation expansions not in the NLM resolver
EXTRA_ABBREVS = {
    'DVT': 'venous thrombosis',
    'PE': 'pulmonary embolism',
    'MI': 'myocardial infarction',
    'AF': 'atrial fibrillation',
    'CHF': 'congestive heart failure',
    'CKD': 'chronic kidney disease',
    'COPD': 'chronic obstructive pulmonary disease',
    'T2DM': 'type 2 diabetes mellitus',
    'T1DM': 'type 1 diabetes mellitus',
    'CAD': 'coronary artery disease',
    'OSA': 'obstructive sleep apnea',
    'IBD': 'inflammatory bowel disease',
    'SLE': 'systemic lupus erythematosus',
    'RA': 'rheumatoid arthritis',
    'MS': 'multiple sclerosis',
    'AML': 'acute myeloid leukemia',
    'ALL': 'acute lymphoblastic leukemia',
    'CML': 'chronic myeloid leukemia',
    'CLL': 'chronic lymphocytic leukemia',
    'NHL': 'non-hodgkin lymphoma',
    'NSCLC': 'non-small cell lung cancer',
    'SCLC': 'small cell lung cancer',
    'HCC': 'hepatocellular carcinoma',
    'RCC': 'renal cell carcinoma',
    'GBM': 'glioblastoma',
    'GERD': 'gastroesophageal reflux disease',
    'IBS': 'irritable bowel syndrome',
    'PTSD': 'post-traumatic stress disorder',
    'MDD': 'major depressive disorder',
    'OCD': 'obsessive-compulsive disorder',
    'GAD': 'generalized anxiety disorder',
    'BPH': 'benign prostatic hyperplasia',
    'UTI': 'urinary tract infection',
    'MRSA': 'staphylococcal infection',
    'VTE': 'venous thromboembolism',
    'PAH': 'pulmonary arterial hypertension',
    'IPF': 'idiopathic pulmonary fibrosis',
    'ITP': 'thrombocytopenic purpura',
    'MDS': 'myelodysplastic syndrome',
    'PNH': 'paroxysmal nocturnal hemoglobinuria',
    'GIST': 'gastrointestinal stromal tumor',
    'MCL': 'mantle cell lymphoma',
    'DLBCL': 'diffuse large B-cell lymphoma',
    'CTCL': 'cutaneous T-cell lymphoma',
    'GvHD': 'graft vs host disease',
    'GVHD': 'graft vs host disease',
    'CDI': 'clostridium difficile',
    'HAP': 'pneumonia',
    'VAP': 'pneumonia',
    'CAP': 'pneumonia',
    'NASH': 'fatty liver disease',
    'ADHF': 'heart failure',
    'HCM': 'hypertrophic cardiomyopathy',
}


def tokenize(name: str) -> set[str]:
    """Extract meaningful words from a disease name."""
    words = set(re.findall(r'[a-z]+', name.lower()))
    return words - STOPWORDS


def strip_qualifiers(name: str) -> str:
    """Strip clinical qualifiers to get base disease."""
    lower = name.lower().strip()
    for prefix in STRIP_PREFIXES:
        if lower.startswith(prefix):
            lower = lower[len(prefix):]
    # Also strip trailing qualifiers
    lower = re.sub(r'\s*\(.*?\)\s*$', '', lower)
    lower = re.sub(r',\s*(type|stage|grade|class|phase)\s+\S+$', '', lower, flags=re.I)
    lower = re.sub(r'\s+stage\s+\w+$', '', lower, flags=re.I)
    lower = re.sub(r'\s+grade\s+\w+$', '', lower, flags=re.I)
    return lower.strip()


def main() -> None:
    print("h901 Phase 2: Broader Word-Overlap Matching")
    print(f"Jaccard threshold: {JACCARD_THRESHOLD}")
    print("=" * 60)

    # Load DRKG MESH IDs
    entities = np.load(
        DATA_DIR / "data" / "embeddings" / "node2vec_256_entities.npy",
        allow_pickle=True
    )
    drkg_mesh = set()
    for e in entities:
        s = str(e)
        if 'Disease::MESH:' in s:
            drkg_mesh.add(s.split('Disease::MESH:')[1])

    # Build name → MESH ID index from existing mappings
    with open(REF_DIR / "mesh_mappings_from_agents.json") as f:
        mesh_data = json.load(f)
    name_to_mesh: dict[str, str] = {}
    for batch in mesh_data.values():
        if isinstance(batch, dict):
            for name, mid in batch.items():
                if mid and str(mid) in drkg_mesh:
                    name_to_mesh[name.lower()] = str(mid)

    # Also add from h901 resolved mappings
    h901_path = ANALYSIS_DIR / "h901_resolved_mappings.json"
    if h901_path.exists():
        with open(h901_path) as f:
            h901 = json.load(f)
        for disease, info in h901.get('mappings', {}).items():
            name_to_mesh[disease.lower()] = info['mesh_id']

    print(f"Known disease names with DRKG MESH: {len(name_to_mesh)}")

    # Build tokenized index for fast matching
    tokenized_index: list[tuple[str, str, set[str]]] = []
    for name, mesh_id in name_to_mesh.items():
        tokens = tokenize(name)
        if len(tokens) >= 1:
            tokenized_index.append((name, mesh_id, tokens))

    # Load unresolvable diseases
    with open(h901_path) as f:
        h901 = json.load(f)
    failures = h901['unresolvable']
    print(f"Unresolved diseases to match: {len(failures)}")

    # Match
    resolved: dict[str, dict] = {}
    still_failed: list[str] = []
    match_methods: dict[str, int] = defaultdict(int)

    for disease in failures:
        matched = False

        # Strategy 0: Abbreviation expansion
        disease_stripped = disease.strip()
        if disease_stripped in EXTRA_ABBREVS:
            expanded = EXTRA_ABBREVS[disease_stripped]
            exp_lower = expanded.lower()
            if exp_lower in name_to_mesh:
                resolved[disease] = {
                    'mesh_id': name_to_mesh[exp_lower],
                    'matched_name': expanded,
                    'confidence': 'medium',
                    'method': f'abbreviation ({disease_stripped} → {expanded})',
                    'jaccard': 1.0,
                }
                match_methods['abbreviation'] += 1
                continue

        # Strategy 1: Strip qualifiers then exact match
        stripped = strip_qualifiers(disease)
        if stripped.lower() in name_to_mesh and stripped.lower() != disease.lower():
            resolved[disease] = {
                'mesh_id': name_to_mesh[stripped.lower()],
                'matched_name': stripped,
                'confidence': 'medium',
                'method': 'qualifier_stripped_exact',
                'jaccard': 1.0,
            }
            match_methods['qualifier_stripped'] += 1
            continue

        # Strategy 2: Word-overlap Jaccard matching
        query_tokens = tokenize(disease)
        if len(query_tokens) < 1:
            still_failed.append(disease)
            continue

        # Also try with stripped version
        stripped_tokens = tokenize(stripped)
        best_match = None
        best_jaccard = 0.0

        for ref_name, mesh_id, ref_tokens in tokenized_index:
            # Try both original and stripped query tokens
            for q_tokens in [query_tokens, stripped_tokens]:
                if not q_tokens:
                    continue
                overlap = len(q_tokens & ref_tokens)
                union = len(q_tokens | ref_tokens)
                jaccard = overlap / union if union > 0 else 0

                if jaccard > best_jaccard and jaccard >= JACCARD_THRESHOLD:
                    best_jaccard = jaccard
                    best_match = (ref_name, mesh_id, jaccard)

        if best_match:
            resolved[disease] = {
                'mesh_id': best_match[1],
                'matched_name': best_match[0],
                'confidence': 'approximate' if best_jaccard < 0.8 else 'medium',
                'method': 'word_overlap',
                'jaccard': round(best_match[2], 3),
            }
            match_methods['word_overlap'] += 1
        else:
            still_failed.append(disease)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    print(f"Resolved: {len(resolved)} ({100*len(resolved)/len(failures):.1f}%)")
    print(f"Still failed: {len(still_failed)} ({100*len(still_failed)/len(failures):.1f}%)")
    print(f"\nBy method: {dict(match_methods)}")

    by_confidence: dict[str, int] = defaultdict(int)
    for r in resolved.values():
        by_confidence[r['confidence']] += 1
    print(f"By confidence: {dict(by_confidence)}")

    # Show examples
    print(f"\n--- Sample approximate matches ---")
    approx = [(d, r) for d, r in resolved.items() if r['confidence'] == 'approximate']
    for disease, info in approx[:20]:
        print(f"  {disease}")
        print(f"    → {info['matched_name']} (MESH:{info['mesh_id']}, J={info['jaccard']})")

    print(f"\n--- Sample still-failed ---")
    for d in still_failed[:20]:
        print(f"  {d}")

    # Save
    output = {
        'summary': {
            'input_failures': len(failures),
            'resolved': len(resolved),
            'still_failed': len(still_failed),
            'by_method': dict(match_methods),
            'by_confidence': dict(by_confidence),
        },
        'mappings': resolved,
        'unresolvable': still_failed,
    }
    output_path = ANALYSIS_DIR / "h901_broader_matches.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to: {output_path}")

    # Combined total
    phase1_resolved = len(h901.get('mappings', {}))
    total = 823 + phase1_resolved + len(resolved)
    print(f"\n{'=' * 60}")
    print(f"CUMULATIVE MAPPING COVERAGE")
    print(f"{'=' * 60}")
    print(f"Original mappings: 823")
    print(f"Phase 1 (NLM API): +{phase1_resolved}")
    print(f"Phase 2 (broader): +{len(resolved)}")
    print(f"TOTAL EVALUABLE: {total} diseases (was 823)")
    print(f"Improvement: {100*(total-823)/823:.0f}%")


if __name__ == '__main__':
    main()
