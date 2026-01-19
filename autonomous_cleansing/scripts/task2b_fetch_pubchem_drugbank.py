#!/usr/bin/env python3
"""
Task 2b: Fetch DrugBank-PubChem Mappings

Downloads DrugBank ID to name mappings from PubChem's REST API.
PubChem has cross-references to DrugBank and provides drug names.

Uses batched requests to be efficient.
"""

import json
import re
import time
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime

# Project paths
PROJECT_ROOT = Path("/Users/jimhuff/github/open-cure")
DATA_DIR = PROJECT_ROOT / "data" / "processed"
REF_DIR = PROJECT_ROOT / "data" / "reference"


def get_drugbank_ids_needing_resolution():
    """Get list of DrugBank IDs that need name resolution."""
    import pandas as pd

    df = pd.read_csv(DATA_DIR / "unified_nodes_clean.csv", low_memory=False)
    drugs = df[df['type'] == 'Drug']

    db_pattern = re.compile(r'DB\d{5}')
    existing_lookup = {}

    # Load existing lookup
    lookup_file = REF_DIR / "drugbank_lookup.json"
    if lookup_file.exists():
        with open(lookup_file) as f:
            existing_lookup = json.load(f)

    # Find IDs that need resolution
    needed_ids = set()
    for _, row in drugs.iterrows():
        name = str(row['name'])
        node_id = str(row['id'])

        # Check if name is ID-only
        if name.startswith('Compound::') or db_pattern.match(name):
            match = db_pattern.search(node_id)
            if match:
                db_id = match.group()
                if db_id not in existing_lookup:
                    needed_ids.add(db_id)

    return sorted(needed_ids)


def fetch_pubchem_by_drugbank(db_ids, batch_size=100):
    """
    Fetch drug names from PubChem using DrugBank IDs.
    Uses PubChem's PUG REST API.
    """
    results = {}
    total = len(db_ids)

    print(f"Fetching {total} DrugBank IDs from PubChem...")

    # Process in batches
    for i in range(0, total, batch_size):
        batch = db_ids[i:i+batch_size]
        print(f"  Processing {i+1}-{min(i+batch_size, total)} of {total}...")

        for db_id in batch:
            try:
                # PubChem REST API to search by DrugBank synonym
                url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{db_id}/synonyms/JSON"
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

                with urllib.request.urlopen(req, timeout=10) as response:
                    data = json.loads(response.read().decode())

                    if 'InformationList' in data and 'Information' in data['InformationList']:
                        synonyms = data['InformationList']['Information'][0].get('Synonym', [])
                        if synonyms:
                            # First synonym is usually the best name
                            # Skip obvious ID-like names
                            for syn in synonyms:
                                if not syn.startswith('DB') and not syn.startswith('CHEMBL') \
                                   and not re.match(r'^[A-Z0-9\-]+$', syn):
                                    results[db_id] = syn
                                    break

            except urllib.error.HTTPError as e:
                if e.code == 404:
                    pass  # Not found in PubChem
                else:
                    print(f"    HTTP Error {e.code} for {db_id}")
            except Exception as e:
                print(f"    Error for {db_id}: {e}")

            # Rate limiting
            time.sleep(0.1)

    print(f"\nResolved {len(results)} additional drugs from PubChem")
    return results


def main():
    print("=" * 60)
    print("Task 2b: Fetch DrugBank-PubChem Mappings")
    print("=" * 60)

    # Get IDs needing resolution
    needed_ids = get_drugbank_ids_needing_resolution()
    print(f"\nDrugBank IDs needing resolution: {len(needed_ids)}")

    if not needed_ids:
        print("All drugs already have names!")
        return

    # Fetch all remaining (in batches, may take a while)
    sample_ids = needed_ids
    print(f"Fetching sample of {len(sample_ids)} IDs...")

    # Fetch from PubChem
    pubchem_results = fetch_pubchem_by_drugbank(sample_ids)

    # Load existing lookup
    lookup_file = REF_DIR / "drugbank_lookup.json"
    with open(lookup_file) as f:
        existing_lookup = json.load(f)

    # Merge results
    merged_lookup = {**existing_lookup, **pubchem_results}

    # Save merged lookup
    with open(lookup_file, 'w') as f:
        json.dump(merged_lookup, f, indent=2)

    print(f"\nUpdated lookup: {len(existing_lookup)} → {len(merged_lookup)} entries")
    print(f"Added {len(pubchem_results)} new mappings from PubChem")

    # Show sample new mappings
    if pubchem_results:
        print("\nSample new mappings:")
        for i, (db_id, name) in enumerate(list(pubchem_results.items())[:10]):
            print(f"  {db_id} → {name}")


if __name__ == "__main__":
    import sys
    # Only import pandas if we're running
    try:
        import pandas
    except ImportError:
        print("pandas not available. Skipping.")
        sys.exit(0)
    main()
