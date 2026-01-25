#!/usr/bin/env python3
"""
ATC (Anatomical Therapeutic Chemical) Code Features for Drug Repurposing.

Maps drugs to ATC therapeutic classes and creates features for the GB model.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

import pandas as pd
import numpy as np


# ATC Level 1 categories (14 main groups)
ATC_LEVEL1 = {
    'A': 'alimentary_tract_metabolism',
    'B': 'blood_blood_forming',
    'C': 'cardiovascular',
    'D': 'dermatologicals',
    'G': 'genito_urinary_sex_hormones',
    'H': 'systemic_hormonal',
    'J': 'antiinfectives_systemic',
    'L': 'antineoplastic_immunomodulating',
    'M': 'musculoskeletal',
    'N': 'nervous_system',
    'P': 'antiparasitic',
    'R': 'respiratory',
    'S': 'sensory_organs',
    'V': 'various',
}

# Disease-to-ATC relevance mapping (which ATC classes are relevant for which disease types)
DISEASE_ATC_RELEVANCE = {
    # Disease patterns -> relevant ATC level 1 codes
    'diabetes': ['A', 'C', 'H'],  # A10 antidiabetics, cardiovascular, hormones
    'hypertension': ['C'],  # Cardiovascular
    'heart failure': ['C', 'B'],  # Cardiovascular, blood
    'cancer': ['L'],  # Antineoplastic
    'breast cancer': ['L', 'H'],  # Antineoplastic, hormones
    'lung cancer': ['L'],
    'colorectal cancer': ['L'],
    'melanoma': ['L'],
    'multiple sclerosis': ['L', 'N'],  # Immunomodulating, nervous
    'rheumatoid arthritis': ['L', 'M', 'H'],  # Immunomodulating, musculoskeletal, hormones
    'asthma': ['R'],  # Respiratory
    'copd': ['R'],
    'alzheimer': ['N'],  # Nervous system
    'parkinson': ['N'],
    'depression': ['N'],
    'schizophrenia': ['N'],
    'epilepsy': ['N'],
    'infection': ['J', 'P'],  # Antiinfectives
    'psoriasis': ['D', 'L'],  # Dermatologicals, immunomodulating
    'ulcerative colitis': ['A', 'L'],  # GI, immunomodulating
    'crohn': ['A', 'L'],
}


class ATCMapper:
    """Maps drugs to ATC codes and creates therapeutic features."""

    # Common drug name synonyms (US name -> ATC name)
    SYNONYMS = {
        'acetaminophen': 'paracetamol',
        'epinephrine': 'adrenaline',
        'norepinephrine': 'noradrenaline',
        'albuterol': 'salbutamol',
        'furosemide': 'frusemide',
        'meclizine': 'meclozine',
        'cyclosporine': 'ciclosporin',
        'phenytoin': 'diphenylhydantoin',
        'levalbuterol': 'levosalbutamol',
        'amphetamine': 'amfetamine',
        'metformin': 'metformin',  # Should match directly
        'sulfasalazine': 'sulfasalazine',
        'simvastatin': 'simvastatin',
        'atorvastatin': 'atorvastatin',
        'lovastatin': 'lovastatin',
        'ibuprofen': 'ibuprofen',
        'aspirin': 'acetylsalicylic acid',
        'tylenol': 'paracetamol',
        'advil': 'ibuprofen',
    }

    def __init__(self, atc_file: str = 'data/external/atc/atc_codes_2024.csv'):
        self.atc_df = pd.read_csv(atc_file)
        self.atc_df['level'] = self.atc_df['atc_code'].str.len()

        # Build lookup tables
        self._build_name_to_atc()
        self._build_atc_hierarchy()

    def _normalize_name(self, name: str, apply_synonyms: bool = True) -> str:
        """Normalize drug name for matching."""
        name = name.lower().strip()
        # Remove common suffixes
        for suffix in [' hydrochloride', ' hcl', ' sodium', ' potassium',
                       ' acetate', ' sulfate', ' citrate', ' maleate',
                       ' mesylate', ' tartrate', ' phosphate', ' fumarate',
                       ' succinate', ' besylate', ' dihydrate', ' monohydrate']:
            name = name.replace(suffix, '')
        # Remove parenthetical content
        name = re.sub(r'\([^)]*\)', '', name)
        # Remove special characters
        name = re.sub(r'[^a-z0-9]', '', name)

        # Apply synonym mapping
        if apply_synonyms and name in self.SYNONYMS:
            synonym = self.SYNONYMS[name]
            name = re.sub(r'[^a-z0-9]', '', synonym.lower())

        return name

    def _build_name_to_atc(self) -> None:
        """Build mapping from drug names to ATC codes."""
        self.name_to_atc: Dict[str, List[str]] = defaultdict(list)

        # Only use level 5 (actual drugs)
        level5 = self.atc_df[self.atc_df['level'] == 7]

        for _, row in level5.iterrows():
            name = self._normalize_name(row['atc_name'])
            if name and name != 'combinations':
                self.name_to_atc[name].append(row['atc_code'])

    def _build_atc_hierarchy(self) -> None:
        """Build ATC hierarchy for level lookups."""
        self.atc_to_info: Dict[str, Dict] = {}

        for _, row in self.atc_df.iterrows():
            code = row['atc_code']
            self.atc_to_info[code] = {
                'name': row['atc_name'],
                'level': row['level'] if pd.notna(row['level']) else len(code),
                'level1': code[0] if len(code) >= 1 else None,
                'level2': code[:3] if len(code) >= 3 else None,
                'level3': code[:4] if len(code) >= 4 else None,
                'level4': code[:5] if len(code) >= 5 else None,
            }

    def get_atc_codes(self, drug_name: str) -> List[str]:
        """Get ATC codes for a drug name."""
        # Try with synonyms first
        normalized = self._normalize_name(drug_name, apply_synonyms=True)
        codes = self.name_to_atc.get(normalized, [])
        if codes:
            return codes

        # Try without synonyms
        normalized = self._normalize_name(drug_name, apply_synonyms=False)
        return self.name_to_atc.get(normalized, [])

    def get_atc_level1(self, drug_name: str) -> List[str]:
        """Get ATC level 1 codes (main therapeutic group) for a drug."""
        atc_codes = self.get_atc_codes(drug_name)
        return list(set(code[0] for code in atc_codes if code))

    def get_atc_level4(self, drug_name: str) -> List[str]:
        """Get ATC level 4 codes (chemical subgroup) for a drug."""
        atc_codes = self.get_atc_codes(drug_name)
        return list(set(code[:5] for code in atc_codes if len(code) >= 5))

    def is_relevant_for_disease(self, drug_name: str, disease_name: str) -> bool:
        """Check if drug's ATC class is relevant for the disease."""
        drug_atc_level1 = set(self.get_atc_level1(drug_name))
        if not drug_atc_level1:
            return False  # Unknown drug

        disease_lower = disease_name.lower()

        # Check each disease pattern
        for pattern, relevant_atc in DISEASE_ATC_RELEVANCE.items():
            if pattern in disease_lower:
                if drug_atc_level1 & set(relevant_atc):
                    return True

        return False

    def get_mechanism_score(self, drug_name: str, disease_name: str) -> float:
        """
        Get a mechanism relevance score (0-1) based on ATC classification.

        Returns:
            1.0 if drug's ATC class is highly relevant for disease
            0.5 if drug has known ATC but not specific to disease
            0.0 if drug has no ATC mapping
        """
        drug_atc_level1 = set(self.get_atc_level1(drug_name))
        if not drug_atc_level1:
            return 0.0

        if self.is_relevant_for_disease(drug_name, disease_name):
            return 1.0

        return 0.5  # Has ATC but not disease-specific


def create_atc_features(
    drugs: List[str],
    diseases: List[str],
    atc_mapper: Optional[ATCMapper] = None,
) -> np.ndarray:
    """
    Create ATC-based features for drug-disease pairs.

    Returns array with columns:
        - has_atc: 1 if drug has ATC mapping, 0 otherwise
        - mechanism_score: 0-1 relevance score
        - atc_A through atc_V: one-hot encoding of ATC level 1
    """
    if atc_mapper is None:
        atc_mapper = ATCMapper()

    n_pairs = len(drugs)

    # 1 (has_atc) + 1 (mechanism_score) + 14 (level1 one-hot)
    features = np.zeros((n_pairs, 16))

    atc_level1_list = list(ATC_LEVEL1.keys())

    for i, (drug, disease) in enumerate(zip(drugs, diseases)):
        atc_codes = atc_mapper.get_atc_level1(drug)

        if atc_codes:
            features[i, 0] = 1  # has_atc
            features[i, 1] = atc_mapper.get_mechanism_score(drug, disease)

            # One-hot for level 1
            for code in atc_codes:
                if code in atc_level1_list:
                    idx = atc_level1_list.index(code) + 2
                    features[i, idx] = 1

    return features


def get_atc_feature_names() -> List[str]:
    """Get names for ATC feature columns."""
    names = ['has_atc', 'mechanism_score']
    for code, name in ATC_LEVEL1.items():
        names.append(f'atc_{code}_{name}')
    return names


def analyze_atc_coverage(drugbank_file: str = 'data/reference/drugbank_lookup.json') -> Dict:
    """Analyze how many of our drugs have ATC mappings."""
    mapper = ATCMapper()

    with open(drugbank_file) as f:
        drugbank = json.load(f)

    total = 0
    mapped = 0
    by_level1: Dict[str, int] = defaultdict(int)
    unmapped_examples: List[str] = []

    for drug_id, drug_name in drugbank.items():
        # Handle both formats: direct string or dict with 'common_name'
        if isinstance(drug_name, dict):
            drug_name = drug_name.get('common_name', '')
        if not drug_name or not isinstance(drug_name, str):
            continue

        total += 1
        atc_codes = mapper.get_atc_level1(drug_name)

        if atc_codes:
            mapped += 1
            for code in atc_codes:
                by_level1[code] += 1
        else:
            if len(unmapped_examples) < 20:
                unmapped_examples.append(drug_name)

    return {
        'total_drugs': total,
        'mapped_drugs': mapped,
        'coverage': mapped / total if total > 0 else 0,
        'by_level1': dict(by_level1),
        'unmapped_examples': unmapped_examples,
    }


if __name__ == '__main__':
    # Analyze coverage
    print("Analyzing ATC coverage...")
    coverage = analyze_atc_coverage()

    print(f"\nTotal drugs: {coverage['total_drugs']}")
    print(f"Mapped to ATC: {coverage['mapped_drugs']}")
    print(f"Coverage: {coverage['coverage']:.1%}")

    print("\nBy ATC Level 1:")
    for code in sorted(coverage['by_level1'].keys()):
        count = coverage['by_level1'][code]
        name = ATC_LEVEL1.get(code, 'unknown')
        print(f"  {code} ({name}): {count}")

    print("\nUnmapped examples:")
    for drug in coverage['unmapped_examples'][:10]:
        print(f"  - {drug}")
