"""h901: Resolve unmapped Every Cure disease names to MeSH IDs via NLM API.

Uses the NLM MeSH Lookup API (https://id.nlm.nih.gov/mesh/lookup/descriptor)
to find MeSH descriptors for unmapped disease names. This is the authoritative
source — no hallucination risk.

Strategy:
1. Exact match first (fastest, highest confidence)
2. Contains match for partial names (broader, needs validation)
3. Cleaned/simplified names as fallback (abbreviation expansion, etc.)

Output: h901_resolved_mappings.json with confidence levels.
Safe to run alongside holdout evaluation (read-only on project data, writes only to analysis/).
"""

import json
import re
import time
import sys
from pathlib import Path
from typing import Optional

import requests
import numpy as np

DATA_DIR = Path(__file__).parent.parent
ANALYSIS_DIR = DATA_DIR / "data" / "analysis"
UNMAPPED_FILE = ANALYSIS_DIR / "h901_unmapped_diseases.txt"
OUTPUT_FILE = ANALYSIS_DIR / "h901_resolved_mappings.json"
CHECKPOINT_FILE = ANALYSIS_DIR / "h901_checkpoint.json"

NLM_API = "https://id.nlm.nih.gov/mesh/lookup/descriptor"
RATE_LIMIT_DELAY = 0.3  # seconds between requests to be polite

# Common abbreviations in Every Cure disease names
ABBREVIATIONS: dict[str, str] = {
    'ABSSSI': 'acute bacterial skin and skin structure infection',
    'ADHF': 'acute decompensated heart failure',
    'AIDS': 'acquired immunodeficiency syndrome',
    'ALS': 'amyotrophic lateral sclerosis',
    'ARDS': 'acute respiratory distress syndrome',
    'BPH': 'benign prostatic hyperplasia',
    'CABG': 'coronary artery bypass graft',
    'CAD': 'coronary artery disease',
    'CAP': 'community-acquired pneumonia',
    'CDI': 'clostridioides difficile infection',
    'CKD': 'chronic kidney disease',
    'CLL': 'chronic lymphocytic leukemia',
    'CML': 'chronic myeloid leukemia',
    'COPD': 'chronic obstructive pulmonary disease',
    'CRC': 'colorectal cancer',
    'CRPC': 'castration-resistant prostate cancer',
    'CTCL': 'cutaneous T-cell lymphoma',
    'CVD': 'cardiovascular disease',
    'DLBCL': 'diffuse large B-cell lymphoma',
    'DMD': 'Duchenne muscular dystrophy',
    'DME': 'diabetic macular edema',
    'DVT': 'deep vein thrombosis',
    'GAD': 'generalized anxiety disorder',
    'GBM': 'glioblastoma multiforme',
    'GERD': 'gastroesophageal reflux disease',
    'GIST': 'gastrointestinal stromal tumor',
    'GvHD': 'graft versus host disease',
    'GVHD': 'graft versus host disease',
    'HAP': 'hospital-acquired pneumonia',
    'HCC': 'hepatocellular carcinoma',
    'HCM': 'hypertrophic cardiomyopathy',
    'HER2': 'human epidermal growth factor receptor 2',
    'IBS': 'irritable bowel syndrome',
    'ICH': 'intracerebral hemorrhage',
    'ILD': 'interstitial lung disease',
    'IPF': 'idiopathic pulmonary fibrosis',
    'ITP': 'immune thrombocytopenic purpura',
    'MCL': 'mantle cell lymphoma',
    'MDD': 'major depressive disorder',
    'MDS': 'myelodysplastic syndrome',
    'MRSA': 'methicillin-resistant Staphylococcus aureus infection',
    'NASH': 'non-alcoholic steatohepatitis',
    'NHL': 'non-Hodgkin lymphoma',
    'NSCLC': 'non-small cell lung cancer',
    'OAB': 'overactive bladder',
    'OCD': 'obsessive-compulsive disorder',
    'PAD': 'peripheral arterial disease',
    'PAH': 'pulmonary arterial hypertension',
    'PBC': 'primary biliary cholangitis',
    'PE': 'pulmonary embolism',
    'PNH': 'paroxysmal nocturnal hemoglobinuria',
    'PSC': 'primary sclerosing cholangitis',
    'PTSD': 'post-traumatic stress disorder',
    'RA': 'rheumatoid arthritis',
    'RCC': 'renal cell carcinoma',
    'RLS': 'restless legs syndrome',
    'SCD': 'sickle cell disease',
    'SCLC': 'small cell lung cancer',
    'SLE': 'systemic lupus erythematosus',
    'SMA': 'spinal muscular atrophy',
    'SSc': 'systemic sclerosis',
    'T1D': 'type 1 diabetes mellitus',
    'T2D': 'type 2 diabetes mellitus',
    'TBI': 'traumatic brain injury',
    'UC': 'ulcerative colitis',
    'UTI': 'urinary tract infection',
    'VAP': 'ventilator-associated pneumonia',
    'VTE': 'venous thromboembolism',
    'aHUS': 'atypical hemolytic uremic syndrome',
    'cSCC': 'cutaneous squamous cell carcinoma',
    'mCRC': 'metastatic colorectal cancer',
    'mCRPC': 'metastatic castration-resistant prostate cancer',
    'rCDI': 'recurrent clostridioides difficile infection',
}


def load_drkg_mesh_ids() -> set[str]:
    """Load all MESH IDs that exist in DRKG embeddings."""
    entities = np.load(
        DATA_DIR / "data" / "embeddings" / "node2vec_256_entities.npy",
        allow_pickle=True
    )
    mesh_ids = set()
    for entity in entities:
        e = str(entity)
        if 'Disease::MESH:' in e:
            mesh_id = e.split('Disease::MESH:')[1]
            mesh_ids.add(mesh_id)
    return mesh_ids


def query_mesh_api(
    term: str,
    match: str = 'exact',
    limit: int = 5,
) -> list[dict[str, str]]:
    """Query NLM MeSH lookup API."""
    try:
        r = requests.get(
            NLM_API,
            params={'label': term, 'match': match, 'limit': limit},
            timeout=10,
        )
        if r.status_code == 200:
            return r.json()
        return []
    except (requests.RequestException, json.JSONDecodeError):
        return []


def extract_mesh_id(resource_url: str) -> str:
    """Extract MESH ID from NLM resource URL."""
    # http://id.nlm.nih.gov/mesh/D003920 → D003920
    parts = resource_url.rstrip('/').split('/')
    return parts[-1] if parts else ''


def clean_disease_name(name: str) -> str:
    """Clean disease name for better API matching."""
    name = name.strip()
    # Remove trailing qualifiers
    name = re.sub(r'\s*\(.*?\)\s*$', '', name)
    # Remove "recurrent", "advanced", "metastatic" prefixes for base disease
    # (we'll try with and without)
    return name


def get_base_disease(name: str) -> Optional[str]:
    """Strip clinical qualifiers to get base disease name."""
    prefixes = [
        'acute ', 'chronic ', 'recurrent ', 'advanced ', 'metastatic ',
        'refractory ', 'relapsed ', 'persistent ', 'severe ',
        'moderate ', 'mild ', 'primary ', 'secondary ', 'progressive ',
        'early ', 'late ', 'localized ', 'locally advanced ',
        'unresectable ', 'newly diagnosed ',
    ]
    lower = name.lower()
    for prefix in prefixes:
        if lower.startswith(prefix):
            return name[len(prefix):]
    return None


def resolve_disease(
    name: str,
    drkg_mesh_ids: set[str],
) -> Optional[dict]:
    """Try to resolve a disease name to a DRKG-valid MeSH ID.

    Returns dict with mesh_id, confidence, method, or None.
    """
    # Strategy 0: Check abbreviation expansion
    name_upper = name.strip()
    if name_upper in ABBREVIATIONS:
        expanded = ABBREVIATIONS[name_upper]
        result = resolve_disease(expanded, drkg_mesh_ids)
        if result:
            result['method'] = f'abbreviation_expansion ({name_upper} → {expanded})'
            result['original_query'] = name
            return result

    # Also check if the name starts with an abbreviation
    for abbrev, expansion in ABBREVIATIONS.items():
        if name_upper.startswith(abbrev + ' ') or name_upper.startswith(abbrev + ','):
            expanded = expansion + name_upper[len(abbrev):]
            result = resolve_disease(expanded, drkg_mesh_ids)
            if result:
                result['method'] = f'abbreviation_prefix ({abbrev} → {expansion})'
                result['original_query'] = name
                return result

    cleaned = clean_disease_name(name)

    # Strategy 1: Exact match
    results = query_mesh_api(cleaned, match='exact')
    for r in results:
        mesh_id = extract_mesh_id(r['resource'])
        if mesh_id in drkg_mesh_ids:
            return {
                'mesh_id': mesh_id,
                'mesh_label': r['label'],
                'confidence': 'high',
                'method': 'exact_match',
                'original_query': name,
            }

    time.sleep(RATE_LIMIT_DELAY)

    # Strategy 2: Contains match
    results = query_mesh_api(cleaned, match='contains', limit=10)
    for r in results:
        mesh_id = extract_mesh_id(r['resource'])
        if mesh_id in drkg_mesh_ids:
            # Check if the match label is reasonably close
            label_lower = r['label'].lower()
            query_lower = cleaned.lower()
            if label_lower == query_lower or query_lower in label_lower or label_lower in query_lower:
                return {
                    'mesh_id': mesh_id,
                    'mesh_label': r['label'],
                    'confidence': 'medium',
                    'method': 'contains_match',
                    'original_query': name,
                }

    time.sleep(RATE_LIMIT_DELAY)

    # Strategy 3: Try base disease (strip qualifiers)
    base = get_base_disease(cleaned)
    if base:
        results = query_mesh_api(base, match='exact')
        for r in results:
            mesh_id = extract_mesh_id(r['resource'])
            if mesh_id in drkg_mesh_ids:
                return {
                    'mesh_id': mesh_id,
                    'mesh_label': r['label'],
                    'confidence': 'medium',
                    'method': f'base_disease_exact ({cleaned} → {base})',
                    'original_query': name,
                }

        time.sleep(RATE_LIMIT_DELAY)

        results = query_mesh_api(base, match='contains', limit=10)
        for r in results:
            mesh_id = extract_mesh_id(r['resource'])
            if mesh_id in drkg_mesh_ids:
                label_lower = r['label'].lower()
                base_lower = base.lower()
                if label_lower == base_lower or base_lower in label_lower or label_lower in base_lower:
                    return {
                        'mesh_id': mesh_id,
                        'mesh_label': r['label'],
                        'confidence': 'low',
                        'method': f'base_disease_contains ({cleaned} → {base})',
                        'original_query': name,
                    }

    time.sleep(RATE_LIMIT_DELAY)

    # Strategy 4: Try without parenthetical content and with word reordering
    # e.g., "Cancer, Breast" → "Breast Cancer"
    if ',' in cleaned:
        parts = [p.strip() for p in cleaned.split(',', 1)]
        reordered = f"{parts[1]} {parts[0]}"
        results = query_mesh_api(reordered, match='exact')
        for r in results:
            mesh_id = extract_mesh_id(r['resource'])
            if mesh_id in drkg_mesh_ids:
                return {
                    'mesh_id': mesh_id,
                    'mesh_label': r['label'],
                    'confidence': 'medium',
                    'method': f'reordered ({cleaned} → {reordered})',
                    'original_query': name,
                }

    return None


def load_checkpoint() -> dict:
    """Load checkpoint if it exists."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {'resolved': {}, 'failed': [], 'processed': 0}


def save_checkpoint(state: dict) -> None:
    """Save checkpoint for resumability."""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(state, f, indent=2)


def main() -> None:
    print("h901: MeSH Mapping Resolution via NLM API")
    print("=" * 60)

    # Load unmapped diseases
    with open(UNMAPPED_FILE) as f:
        unmapped = [line.strip() for line in f if line.strip()]
    print(f"Unmapped diseases to resolve: {len(unmapped)}")

    # Load DRKG MESH IDs
    drkg_mesh_ids = load_drkg_mesh_ids()
    print(f"DRKG MESH IDs available: {len(drkg_mesh_ids)}")

    # Load checkpoint
    state = load_checkpoint()
    already_processed = state['processed']
    if already_processed > 0:
        print(f"Resuming from checkpoint: {already_processed} already processed")
        print(f"  Resolved so far: {len(state['resolved'])}")
        print(f"  Failed so far: {len(state['failed'])}")

    resolved = state['resolved']
    failed = state['failed']

    # Process remaining diseases
    total = len(unmapped)
    for i, disease in enumerate(unmapped):
        if i < already_processed:
            continue

        # Progress reporting every 50
        if (i + 1) % 50 == 0 or i == 0:
            pct = 100 * (i + 1) / total
            print(f"[{i+1}/{total}] ({pct:.1f}%) Resolved: {len(resolved)}, "
                  f"Failed: {len(failed)}")
            # Save checkpoint
            state = {
                'resolved': resolved,
                'failed': failed,
                'processed': i + 1,
            }
            save_checkpoint(state)

        result = resolve_disease(disease, drkg_mesh_ids)
        if result:
            resolved[disease] = result
        else:
            failed.append(disease)

    # Final save
    state = {
        'resolved': resolved,
        'failed': failed,
        'processed': total,
    }
    save_checkpoint(state)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"RESOLUTION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total processed: {total}")
    print(f"Resolved: {len(resolved)} ({100*len(resolved)/total:.1f}%)")
    print(f"Failed: {len(failed)} ({100*len(failed)/total:.1f}%)")

    by_confidence: dict[str, int] = {}
    by_method: dict[str, int] = {}
    for r in resolved.values():
        c = r['confidence']
        m = r['method'].split(' (')[0]  # strip details
        by_confidence[c] = by_confidence.get(c, 0) + 1
        by_method[m] = by_method.get(m, 0) + 1

    print(f"\nBy confidence: {json.dumps(by_confidence, indent=2)}")
    print(f"\nBy method: {json.dumps(by_method, indent=2)}")

    # Save final output
    output = {
        'summary': {
            'total_unmapped': total,
            'resolved': len(resolved),
            'failed': len(failed),
            'by_confidence': by_confidence,
            'by_method': by_method,
        },
        'mappings': resolved,
        'unresolvable': failed,
    }
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {OUTPUT_FILE}")

    # Sample resolved
    print(f"\nSample resolutions:")
    for name, info in list(resolved.items())[:15]:
        print(f"  {name} → MESH:{info['mesh_id']} ({info['mesh_label']}) "
              f"[{info['confidence']}, {info['method']}]")


if __name__ == '__main__':
    main()
