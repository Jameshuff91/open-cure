#!/usr/bin/env python3
"""
Batch fetch SMILES from PubChem for DrugBank drugs.

Uses PubChem's REST API with batching and caching for efficiency.
Saves progress incrementally to avoid data loss.
"""

import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

CACHE_DIR = Path('data/reference/chemical')
SMILES_CACHE = CACHE_DIR / 'drug_smiles.json'
DRUGBANK_LOOKUP = Path('data/reference/drugbank_lookup.json')

# PubChem API settings
PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
BATCH_SIZE = 10  # PubChem doesn't like too many at once via name lookup
SAVE_INTERVAL = 100  # Save cache every N drugs
MAX_WORKERS = 3  # Parallel workers (be nice to PubChem)


def load_cache() -> Dict[str, Optional[str]]:
    """Load existing SMILES cache."""
    if SMILES_CACHE.exists():
        with open(SMILES_CACHE) as f:
            return json.load(f)
    return {}


def save_cache(cache: Dict[str, Optional[str]]):
    """Save SMILES cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(SMILES_CACHE, 'w') as f:
        json.dump(cache, f, indent=2)


def fetch_smiles_single(drug_name: str, max_retries: int = 2) -> Optional[str]:
    """Fetch SMILES for a single drug from PubChem."""
    for attempt in range(max_retries):
        try:
            # Request multiple SMILES formats - PubChem returns different ones
            url = f"{PUBCHEM_BASE}/compound/name/{requests.utils.quote(drug_name)}/property/CanonicalSMILES,IsomericSMILES/JSON"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                props = data.get('PropertyTable', {}).get('Properties', [])
                if props:
                    # Try different SMILES field names (PubChem uses different ones)
                    p = props[0]
                    return (p.get('CanonicalSMILES') or
                            p.get('IsomericSMILES') or
                            p.get('SMILES') or
                            p.get('ConnectivitySMILES'))
            elif response.status_code == 404:
                return None  # Not found
            elif response.status_code == 503:
                time.sleep(1)
                continue

        except (requests.RequestException, json.JSONDecodeError):
            if attempt < max_retries - 1:
                time.sleep(0.5)

    return None


def fetch_batch(drug_names: List[str]) -> Dict[str, Optional[str]]:
    """Fetch SMILES for a batch of drugs using parallel requests."""
    results = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_name = {
            executor.submit(fetch_smiles_single, name): name
            for name in drug_names
        }

        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                smiles = future.result()
                results[name] = smiles
            except Exception:
                results[name] = None

    return results


def main():
    print("=" * 60)
    print("BATCH SMILES FETCHER")
    print("=" * 60)

    # Load DrugBank drugs
    print("\n1. Loading DrugBank drugs...")
    with open(DRUGBANK_LOOKUP) as f:
        drugbank = json.load(f)

    all_drug_names = list(drugbank.values())
    print(f"   Total drugs: {len(all_drug_names):,}")

    # Load existing cache
    print("\n2. Loading cache...")
    cache = load_cache()
    print(f"   Cached SMILES: {len(cache):,}")

    # Find drugs needing SMILES
    drugs_to_fetch = [
        name for name in all_drug_names
        if name not in cache
    ]
    print(f"   Need to fetch: {len(drugs_to_fetch):,}")

    if not drugs_to_fetch:
        print("\n✓ All drugs already have SMILES cached!")
        return

    # Fetch in batches
    print(f"\n3. Fetching SMILES (batch size: {BATCH_SIZE}, workers: {MAX_WORKERS})...")

    fetched = 0
    found = 0
    not_found = 0

    pbar = tqdm(total=len(drugs_to_fetch), desc="Fetching")

    for i in range(0, len(drugs_to_fetch), BATCH_SIZE):
        batch = drugs_to_fetch[i:i + BATCH_SIZE]

        # Fetch batch
        results = fetch_batch(batch)

        # Update cache
        for name, smiles in results.items():
            cache[name] = smiles
            fetched += 1
            if smiles:
                found += 1
            else:
                not_found += 1

        pbar.update(len(batch))

        # Save periodically
        if fetched % SAVE_INTERVAL < BATCH_SIZE:
            save_cache(cache)
            pbar.set_postfix({"found": found, "not_found": not_found, "saved": "✓"})

        # Rate limiting - be nice to PubChem
        time.sleep(0.2)

    pbar.close()

    # Final save
    save_cache(cache)

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total fetched: {fetched:,}")
    print(f"Found SMILES: {found:,} ({found/fetched:.1%})")
    print(f"Not found: {not_found:,} ({not_found/fetched:.1%})")

    # Count drugs with valid SMILES
    valid_smiles = sum(1 for v in cache.values() if v is not None)
    print(f"\nTotal drugs with SMILES: {valid_smiles:,}/{len(all_drug_names):,} ({valid_smiles/len(all_drug_names):.1%})")

    # Regenerate fingerprints reminder
    print("\n⚠ Remember to regenerate fingerprints:")
    print("   python -c \"from src.chemical_features import DrugFingerprinter; fp=DrugFingerprinter(); fp.regenerate_all_fingerprints()\"")


if __name__ == "__main__":
    main()
