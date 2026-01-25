#!/usr/bin/env python3
"""
Chemical structure features for drug repurposing.

Uses molecular fingerprints to calculate structural similarity between drugs.
This helps identify repurposing candidates based on the principle that
structurally similar drugs may have similar therapeutic effects.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import pickle

import numpy as np
import requests
from tqdm import tqdm

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    from rdkit import RDLogger
    # Suppress RDKit warnings
    RDLogger.DisableLog('rdApp.*')
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. Install with: pip install rdkit")


# Cache file paths
CACHE_DIR = Path('data/reference/chemical')
SMILES_CACHE = CACHE_DIR / 'drug_smiles.json'
FINGERPRINT_CACHE = CACHE_DIR / 'drug_fingerprints.pkl'


def ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def fetch_smiles_from_pubchem(drug_name: str, max_retries: int = 3) -> Optional[str]:
    """
    Fetch SMILES string from PubChem by drug name.

    Uses PubChem's REST API to search for compound by name and retrieve
    the canonical SMILES representation.
    """
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

    for attempt in range(max_retries):
        try:
            # Search by name - request multiple SMILES formats
            url = f"{base_url}/compound/name/{requests.utils.quote(drug_name)}/property/CanonicalSMILES,IsomericSMILES/JSON"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                props = data.get('PropertyTable', {}).get('Properties', [])
                if props:
                    # Try different SMILES field names (PubChem uses different ones)
                    p = props[0]
                    smiles = (p.get('CanonicalSMILES') or
                              p.get('IsomericSMILES') or
                              p.get('SMILES') or
                              p.get('ConnectivitySMILES'))
                    return smiles
            elif response.status_code == 404:
                # Compound not found
                return None
            elif response.status_code == 503:
                # Rate limited, wait and retry
                time.sleep(1)
                continue
            else:
                return None

        except (requests.RequestException, json.JSONDecodeError):
            if attempt < max_retries - 1:
                time.sleep(0.5)
            continue

    return None


def load_smiles_cache() -> Dict[str, str]:
    """Load cached SMILES data."""
    if SMILES_CACHE.exists():
        with open(SMILES_CACHE) as f:
            return json.load(f)
    return {}


def save_smiles_cache(cache: Dict[str, str]):
    """Save SMILES cache to file."""
    ensure_cache_dir()
    with open(SMILES_CACHE, 'w') as f:
        json.dump(cache, f, indent=2)


def get_smiles_for_drugs(
    drug_names: List[str],
    use_cache: bool = True,
    fetch_missing: bool = True,
    batch_size: int = 100,
) -> Dict[str, str]:
    """
    Get SMILES for a list of drug names.

    Args:
        drug_names: List of drug names to look up
        use_cache: Whether to use cached SMILES data
        fetch_missing: Whether to fetch missing SMILES from PubChem
        batch_size: How many to fetch before saving cache

    Returns:
        Dict mapping drug name to SMILES string
    """
    smiles_map = load_smiles_cache() if use_cache else {}

    if fetch_missing:
        missing = [name for name in drug_names if name not in smiles_map]

        if missing:
            print(f"Fetching SMILES for {len(missing)} drugs from PubChem...")

            for i, name in enumerate(tqdm(missing)):
                smiles = fetch_smiles_from_pubchem(name)
                if smiles:
                    smiles_map[name] = smiles
                else:
                    smiles_map[name] = ""  # Mark as not found

                # Rate limiting
                if i % 5 == 0:
                    time.sleep(0.2)

                # Save periodically
                if (i + 1) % batch_size == 0:
                    save_smiles_cache(smiles_map)

            save_smiles_cache(smiles_map)

    return {name: smiles_map.get(name, "") for name in drug_names}


def smiles_to_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> Optional[np.ndarray]:
    """
    Convert SMILES to Morgan fingerprint.

    Args:
        smiles: SMILES string
        radius: Morgan fingerprint radius (default 2 = ECFP4 equivalent)
        n_bits: Number of bits in fingerprint

    Returns:
        Numpy array of fingerprint bits, or None if conversion fails
    """
    if not RDKIT_AVAILABLE:
        return None

    if not smiles:
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Generate Morgan fingerprint
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)

        # Convert to numpy array
        arr = np.zeros(n_bits, dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    except Exception:
        return None


def compute_tanimoto_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """
    Compute Tanimoto similarity between two fingerprints.

    Tanimoto = |A ∩ B| / |A ∪ B|
    """
    intersection = np.sum(fp1 & fp2)
    union = np.sum(fp1 | fp2)

    if union == 0:
        return 0.0

    return float(intersection) / float(union)


class DrugFingerprinter:
    """Manages drug fingerprints and similarity calculations."""

    def __init__(
        self,
        radius: int = 2,
        n_bits: int = 2048,
        use_cache: bool = True,
    ):
        self.radius = radius
        self.n_bits = n_bits
        self.use_cache = use_cache

        self.smiles_cache: Dict[str, str] = {}
        self.fingerprints: Dict[str, np.ndarray] = {}

        if use_cache:
            self._load_caches()

    def _load_caches(self):
        """Load cached data."""
        self.smiles_cache = load_smiles_cache()

        if FINGERPRINT_CACHE.exists():
            with open(FINGERPRINT_CACHE, 'rb') as f:
                self.fingerprints = pickle.load(f)

    def _save_caches(self):
        """Save cached data."""
        ensure_cache_dir()
        save_smiles_cache(self.smiles_cache)

        with open(FINGERPRINT_CACHE, 'wb') as f:
            pickle.dump(self.fingerprints, f)

    def get_fingerprint(self, drug_name: str, fetch_if_missing: bool = True) -> Optional[np.ndarray]:
        """
        Get fingerprint for a drug.

        Args:
            drug_name: Name of the drug
            fetch_if_missing: Whether to fetch from PubChem if not cached

        Returns:
            Fingerprint array or None if not available
        """
        # Check fingerprint cache first
        if drug_name in self.fingerprints:
            return self.fingerprints[drug_name]

        # Check SMILES cache
        smiles = self.smiles_cache.get(drug_name)

        if smiles is None and fetch_if_missing:
            smiles = fetch_smiles_from_pubchem(drug_name)
            self.smiles_cache[drug_name] = smiles if smiles else ""

        if not smiles:
            return None

        # Generate fingerprint
        fp = smiles_to_fingerprint(smiles, self.radius, self.n_bits)
        if fp is not None:
            self.fingerprints[drug_name] = fp

        return fp

    def compute_similarity(self, drug1: str, drug2: str) -> float:
        """Compute Tanimoto similarity between two drugs."""
        fp1 = self.get_fingerprint(drug1, fetch_if_missing=False)
        fp2 = self.get_fingerprint(drug2, fetch_if_missing=False)

        if fp1 is None or fp2 is None:
            return 0.0

        return compute_tanimoto_similarity(fp1, fp2)

    def get_max_similarity_to_set(
        self,
        drug_name: str,
        reference_drugs: List[str],
    ) -> Tuple[float, str]:
        """
        Find the maximum similarity between a drug and a set of reference drugs.

        Returns:
            Tuple of (max_similarity, most_similar_drug_name)
        """
        fp = self.get_fingerprint(drug_name, fetch_if_missing=False)
        if fp is None:
            return 0.0, ""

        max_sim = 0.0
        best_match = ""

        for ref_drug in reference_drugs:
            ref_fp = self.get_fingerprint(ref_drug, fetch_if_missing=False)
            if ref_fp is not None:
                sim = compute_tanimoto_similarity(fp, ref_fp)
                if sim > max_sim:
                    max_sim = sim
                    best_match = ref_drug

        return max_sim, best_match

    def precompute_fingerprints(
        self,
        drug_names: List[str],
        fetch_missing: bool = True,
    ) -> Dict[str, bool]:
        """
        Precompute fingerprints for a list of drugs.

        Returns:
            Dict mapping drug name to success status
        """
        results = {}

        # First, get all SMILES
        smiles_map = get_smiles_for_drugs(
            drug_names,
            use_cache=True,
            fetch_missing=fetch_missing,
        )
        self.smiles_cache.update(smiles_map)

        # Then compute fingerprints
        print("Computing fingerprints...")
        for name in tqdm(drug_names):
            smiles = smiles_map.get(name)
            if smiles:
                fp = smiles_to_fingerprint(smiles, self.radius, self.n_bits)
                if fp is not None:
                    self.fingerprints[name] = fp
                    results[name] = True
                else:
                    results[name] = False
            else:
                results[name] = False

        # Save caches
        self._save_caches()

        return results

    def get_coverage_stats(self, drug_names: List[str]) -> Dict:
        """Get coverage statistics for a list of drugs."""
        has_smiles = sum(1 for d in drug_names if self.smiles_cache.get(d))
        has_fp = sum(1 for d in drug_names if d in self.fingerprints)

        return {
            'total_drugs': len(drug_names),
            'with_smiles': has_smiles,
            'with_fingerprints': has_fp,
            'smiles_coverage': has_smiles / len(drug_names) if drug_names else 0,
            'fp_coverage': has_fp / len(drug_names) if drug_names else 0,
        }

    def regenerate_all_fingerprints(self) -> Dict[str, int]:
        """
        Regenerate fingerprints from all cached SMILES.

        Useful after batch SMILES fetching to update fingerprints.

        Returns:
            Dict with counts: generated, failed, total_smiles
        """
        # Reload SMILES cache to get latest
        self.smiles_cache = load_smiles_cache()

        # Filter to valid SMILES (non-empty, non-None)
        valid_smiles = {
            name: smiles for name, smiles in self.smiles_cache.items()
            if smiles and smiles.strip()
        }

        print(f"Regenerating fingerprints from {len(valid_smiles):,} SMILES...")

        generated = 0
        failed = 0

        for name, smiles in tqdm(valid_smiles.items(), desc="Generating"):
            fp = smiles_to_fingerprint(smiles, self.radius, self.n_bits)
            if fp is not None:
                self.fingerprints[name] = fp
                generated += 1
            else:
                failed += 1

        # Save fingerprints
        ensure_cache_dir()
        with open(FINGERPRINT_CACHE, 'wb') as f:
            pickle.dump(self.fingerprints, f)

        print(f"\n✓ Generated: {generated:,}")
        print(f"✗ Failed: {failed:,}")
        print(f"Total fingerprints: {len(self.fingerprints):,}")

        return {
            'generated': generated,
            'failed': failed,
            'total_smiles': len(valid_smiles),
            'total_fingerprints': len(self.fingerprints),
        }


def create_similarity_features(
    query_drugs: List[str],
    query_diseases: List[str],
    ground_truth: Dict[str, List[str]],
    drugbank_lookup: Dict[str, str],
    fingerprinter: DrugFingerprinter,
) -> np.ndarray:
    """
    Create chemical similarity features for drug-disease pairs.

    Features:
        1. max_sim_to_known: Max Tanimoto similarity to drugs known to treat this disease
        2. has_similar_drug: Binary, 1 if max_sim > 0.7
        3. sim_count_high: Count of known treatments with similarity > 0.7
        4. sim_count_moderate: Count of known treatments with similarity > 0.5

    Args:
        query_drugs: List of DrugBank IDs for query drugs
        query_diseases: List of MESH IDs for corresponding diseases
        ground_truth: Dict mapping MESH ID to list of DrugBank IDs (known treatments)
        drugbank_lookup: Dict mapping DrugBank ID to drug name
        fingerprinter: DrugFingerprinter instance with precomputed fingerprints

    Returns:
        Feature array of shape (n_pairs, 4)
    """
    n_pairs = len(query_drugs)
    features = np.zeros((n_pairs, 4), dtype=np.float32)

    for i, (drug_id, disease_id) in enumerate(zip(query_drugs, query_diseases)):
        # Get drug name
        drug_name = drugbank_lookup.get(drug_id, drug_id)

        # Get known treatments for this disease
        known_drug_ids = ground_truth.get(disease_id, [])
        known_drug_names = [drugbank_lookup.get(d, d) for d in known_drug_ids]

        if not known_drug_names:
            continue

        # Get query drug fingerprint
        query_fp = fingerprinter.get_fingerprint(drug_name, fetch_if_missing=False)
        if query_fp is None:
            continue

        # Compute similarities to all known treatments
        similarities = []
        for known_drug in known_drug_names:
            known_fp = fingerprinter.get_fingerprint(known_drug, fetch_if_missing=False)
            if known_fp is not None:
                sim = compute_tanimoto_similarity(query_fp, known_fp)
                similarities.append(sim)

        if similarities:
            features[i, 0] = max(similarities)  # max_sim_to_known
            features[i, 1] = 1.0 if max(similarities) > 0.7 else 0.0  # has_similar_drug
            features[i, 2] = sum(1 for s in similarities if s > 0.7)  # sim_count_high
            features[i, 3] = sum(1 for s in similarities if s > 0.5)  # sim_count_moderate

    return features


def get_chemical_feature_names() -> List[str]:
    """Get names for chemical similarity features."""
    return [
        'max_sim_to_known',
        'has_similar_drug',
        'sim_count_high',
        'sim_count_moderate',
    ]


if __name__ == '__main__':
    # Test the functionality
    print("Testing chemical features module...")

    if not RDKIT_AVAILABLE:
        print("ERROR: RDKit not available")
        exit(1)

    # Test with a few example drugs
    test_drugs = ['aspirin', 'ibuprofen', 'acetaminophen', 'metformin', 'atorvastatin']

    fingerprinter = DrugFingerprinter()

    print("\nFetching SMILES and computing fingerprints...")
    results = fingerprinter.precompute_fingerprints(test_drugs, fetch_missing=True)

    print("\nResults:")
    for drug, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {drug}")

    print("\nSimilarity matrix:")
    print("             ", end="")
    for d in test_drugs:
        print(f"{d[:8]:>10}", end="")
    print()

    for d1 in test_drugs:
        print(f"{d1[:12]:<12}", end="")
        for d2 in test_drugs:
            sim = fingerprinter.compute_similarity(d1, d2)
            print(f"{sim:>10.3f}", end="")
        print()

    # Print coverage stats
    stats = fingerprinter.get_coverage_stats(test_drugs)
    print(f"\nCoverage: {stats['fp_coverage']:.1%} ({stats['with_fingerprints']}/{stats['total_drugs']})")
