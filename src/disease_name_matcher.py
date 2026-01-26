#!/usr/bin/env python3
"""
Improved disease name matching with fuzzy matching and synonym handling.

This module fixes the critical data mapping issue where 77.9% of drug-disease
pairs were lost during training due to exact string matching failures.
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional, Set, Tuple
from collections import defaultdict

# Synonym groups - map variants to canonical name
DISEASE_SYNONYMS: Dict[str, str] = {
    # Atopic dermatitis variants - D003876 is the specific MESH code
    "atopic dermatitis": "atopic dermatitis",
    "atopic dermatitis ": "atopic dermatitis",  # trailing space
    "atopic eczema": "atopic dermatitis",
    # "eczema" alone is too generic, don't map it

    # Psoriasis variants
    "plaque psoriasis": "psoriasis",
    "chronic plaque psoriasis": "psoriasis",
    "moderate to severe plaque psoriasis": "psoriasis",
    "moderate-to-severe plaque psoriasis": "psoriasis",

    # Diabetes variants
    "type 2 diabetes": "type 2 diabetes mellitus",
    "type 2 diabetes mellitus": "type 2 diabetes mellitus",
    "type 2 diabetes mellitus ": "type 2 diabetes mellitus",  # trailing space
    "type ii diabetes": "type 2 diabetes mellitus",
    "type ii diabetes mellitus": "type 2 diabetes mellitus",
    "diabetes mellitus type 2": "type 2 diabetes mellitus",
    "t2dm": "type 2 diabetes mellitus",
    "diabetes mellitus": "type 2 diabetes mellitus",
    "noninsulin dependent diabetes mellitus type ii": "type 2 diabetes mellitus",
    "non-insulin dependent diabetes mellitus": "type 2 diabetes mellitus",

    # HIV variants
    "hiv1 infection": "hiv infection",
    "hiv-1 infection": "hiv infection",
    "hiv infection": "hiv infection",
    "human immunodeficiency virus infection": "hiv infection",

    # Parkinson's variants
    "parkinsons disease": "parkinson disease",
    "parkinson's disease": "parkinson disease",
    "parkinson disease": "parkinson disease",

    # Alzheimer's variants
    "alzheimers disease": "alzheimer disease",
    "alzheimer's disease": "alzheimer disease",
    "alzheimer disease": "alzheimer disease",

    # Lung cancer variants
    "non-small cell lung cancer": "lung cancer",
    "non small cell lung cancer": "lung cancer",
    "nonsmall cell lung cancer": "lung cancer",
    "nsclc": "lung cancer",
    "small cell lung cancer": "lung cancer",

    # Heart conditions
    "congestive heart failure": "heart failure",
    "chronic heart failure": "heart failure",
    "cardiac failure": "heart failure",

    # MS variants
    "relapsing multiple sclerosis": "multiple sclerosis",
    "relapsing-remitting multiple sclerosis": "multiple sclerosis",

    # Rheumatoid arthritis
    "ra": "rheumatoid arthritis",

    # Asthma variants
    "bronchial asthma": "asthma",
    "allergic asthma": "asthma",

    # COPD variants
    "chronic obstructive pulmonary disease": "copd",
    "chronic bronchitis": "copd",

    # Allergic rhinitis variants
    "seasonal allergic rhinitis": "allergic rhinitis",
    "perennial allergic rhinitis": "allergic rhinitis",
    "hay fever": "allergic rhinitis",

    # Hypertension variants
    "high blood pressure": "hypertension",
    "essential hypertension": "hypertension",
    "arterial hypertension": "hypertension",

    # Cancer variants (keep specific when MESH mapping exists)
    "breast neoplasm": "breast cancer",
    "breast neoplasms": "breast cancer",
    "mammary cancer": "breast cancer",
    "metastatic breast cancer": "breast cancer",

    # Hepatitis variants
    "chronic hepatitis c": "hepatitis c",
    "hepatitis c virus infection": "hepatitis c",
    "hcv infection": "hepatitis c",

    # Ankylosing spondylitis
    "ankylosing spondylitis": "ankylosing spondylitis",
    "as": "ankylosing spondylitis",

    # Multiple myeloma
    "multiple myeloma": "multiple myeloma",
    "mm": "multiple myeloma",

    # Inflammatory bowel disease
    "ulcerative colitis": "ulcerative colitis",
    "uc": "ulcerative colitis",
    "crohn disease": "crohn disease",
    "crohn's disease": "crohn disease",
    "crohns disease": "crohn disease",
    "inflammatory bowel disease": "inflammatory bowel disease",
    "ibd": "inflammatory bowel disease",
}


def normalize_disease_name(name: str) -> str:
    """Normalize a disease name for matching."""
    if not name:
        return ""

    # Lowercase and strip whitespace
    name = name.lower().strip()

    # Remove possessive apostrophes
    name = name.replace("'s", "s")
    name = name.replace("'", "")

    # Normalize hyphens and spaces
    name = re.sub(r'[-_]', ' ', name)
    name = re.sub(r'\s+', ' ', name)

    # Remove trailing punctuation
    name = name.rstrip('.,;:')

    return name


def get_canonical_name(disease_name: str) -> str:
    """Get the canonical disease name for mapping."""
    normalized = normalize_disease_name(disease_name)

    # Check exact match in synonyms
    if normalized in DISEASE_SYNONYMS:
        return DISEASE_SYNONYMS[normalized]

    # Only check for longer synonyms (>=6 chars) to avoid false substring matches
    # Short synonyms like "ra", "as", "mm" would match inside unrelated words
    for synonym, canonical in DISEASE_SYNONYMS.items():
        if len(synonym) >= 6:
            if synonym in normalized or normalized in synonym:
                return canonical

    return normalized


class DiseaseMatcher:
    """Match disease names to MESH IDs with fuzzy matching."""

    def __init__(self, mesh_mappings: Dict[str, str]):
        """
        Initialize with MESH mappings.

        Args:
            mesh_mappings: Dict mapping disease names to MESH IDs
        """
        self.mesh_mappings = mesh_mappings
        self.normalized_mappings: Dict[str, str] = {}
        self.canonical_to_mesh: Dict[str, str] = {}

        self._build_index()

    def _build_index(self):
        """Build normalized index for fast lookup."""
        for disease_name, mesh_id in self.mesh_mappings.items():
            normalized = normalize_disease_name(disease_name)
            canonical = get_canonical_name(disease_name)

            self.normalized_mappings[normalized] = mesh_id
            self.canonical_to_mesh[canonical] = mesh_id

    def get_mesh_id(self, disease_name: str) -> Optional[str]:
        """
        Get MESH ID for a disease name.

        Args:
            disease_name: The disease name to look up

        Returns:
            MESH ID if found, None otherwise
        """
        normalized = normalize_disease_name(disease_name)

        # 1. Try exact normalized match
        if normalized in self.normalized_mappings:
            return self.normalized_mappings[normalized]

        # 2. Try canonical name match
        canonical = get_canonical_name(disease_name)
        if canonical in self.canonical_to_mesh:
            return self.canonical_to_mesh[canonical]

        # 3. Try finding canonical in mappings
        for mapped_name, mesh_id in self.normalized_mappings.items():
            mapped_canonical = get_canonical_name(mapped_name)
            if canonical == mapped_canonical:
                return mesh_id

        # 4. Try substring matching for very long disease names (>20 chars)
        # and only if both strings are long enough to avoid false matches
        if len(normalized) > 20:
            for mapped_name, mesh_id in self.normalized_mappings.items():
                if len(mapped_name) > 10:  # Only match against longer disease names
                    if mapped_name in normalized or normalized in mapped_name:
                        return mesh_id

        return None

    def get_all_mappings(self, disease_names: Set[str]) -> Dict[str, str]:
        """
        Get MESH mappings for a set of disease names.

        Args:
            disease_names: Set of disease names to map

        Returns:
            Dict mapping disease names to MESH IDs
        """
        result = {}
        for name in disease_names:
            mesh_id = self.get_mesh_id(name)
            if mesh_id:
                result[name] = mesh_id
        return result

    def coverage_report(self, disease_names: Set[str]) -> Dict:
        """Generate coverage report for disease names."""
        mapped = 0
        unmapped = []

        for name in disease_names:
            if self.get_mesh_id(name):
                mapped += 1
            else:
                unmapped.append(name)

        return {
            "total": len(disease_names),
            "mapped": mapped,
            "unmapped": len(unmapped),
            "coverage": mapped / len(disease_names) if disease_names else 0,
            "unmapped_diseases": unmapped[:50],  # First 50 unmapped
        }


def load_mesh_mappings() -> Dict[str, str]:
    """Load MESH mappings from hardcoded + agent sources."""
    PROJECT_ROOT = Path(__file__).parent.parent
    REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"

    # Hardcoded mappings (from train_gb_enhanced.py)
    hardcoded = {
        "hiv infection": "drkg:Disease::MESH:D015658",
        "hepatitis c": "drkg:Disease::MESH:D006526",
        "tuberculosis": "drkg:Disease::MESH:D014376",
        "breast cancer": "drkg:Disease::MESH:D001943",
        "lung cancer": "drkg:Disease::MESH:D008175",
        "colorectal cancer": "drkg:Disease::MESH:D015179",
        "hypertension": "drkg:Disease::MESH:D006973",
        "heart failure": "drkg:Disease::MESH:D006333",
        "atrial fibrillation": "drkg:Disease::MESH:D001281",
        "epilepsy": "drkg:Disease::MESH:D004827",
        "parkinson disease": "drkg:Disease::MESH:D010300",
        "alzheimer disease": "drkg:Disease::MESH:D000544",
        "rheumatoid arthritis": "drkg:Disease::MESH:D001172",
        "multiple sclerosis": "drkg:Disease::MESH:D009103",
        "psoriasis": "drkg:Disease::MESH:D011565",
        "type 2 diabetes mellitus": "drkg:Disease::MESH:D003924",
        "obesity": "drkg:Disease::MESH:D009765",
        "asthma": "drkg:Disease::MESH:D001249",
        "copd": "drkg:Disease::MESH:D029424",
        "osteoporosis": "drkg:Disease::MESH:D010024",
        "myocardial infarction": "drkg:Disease::MESH:D009203",
        "stroke": "drkg:Disease::MESH:D020521",
        "ulcerative colitis": "drkg:Disease::MESH:D003093",
        "schizophrenia": "drkg:Disease::MESH:D012559",
        "major depressive disorder": "drkg:Disease::MESH:D003865",
        "osteoarthritis": "drkg:Disease::MESH:D010003",
        "atopic eczema": "drkg:Disease::MESH:D003876",  # Added for atopic dermatitis
        "atopic dermatitis": "drkg:Disease::MESH:D003876",  # Explicit mapping
        "ankylosing spondylitis": "drkg:Disease::MESH:D013167",
        "multiple myeloma": "drkg:Disease::MESH:D009101",
        "crohn disease": "drkg:Disease::MESH:D003424",
        "allergic rhinitis": "drkg:Disease::MESH:D065631",
        "acne vulgaris": "drkg:Disease::MESH:D000152",
    }

    # Agent mappings
    agent_path = REFERENCE_DIR / "mesh_mappings_from_agents.json"
    agent_mappings = {}

    if agent_path.exists():
        with open(agent_path) as f:
            agent_data = json.load(f)

        for batch_name, batch_data in agent_data.items():
            if batch_name == "metadata" or not isinstance(batch_data, dict):
                continue
            for disease_name, mesh_id in batch_data.items():
                if mesh_id is not None and mesh_id.startswith("D"):
                    agent_mappings[disease_name.lower()] = f"drkg:Disease::MESH:{mesh_id}"

    # Merge (agent takes priority)
    return {**hardcoded, **agent_mappings}


def test_matcher():
    """Test the disease matcher."""
    mesh_mappings = load_mesh_mappings()
    matcher = DiseaseMatcher(mesh_mappings)

    # Test cases
    test_diseases = [
        "atopic dermatitis",
        "atopic dermatitis ",  # trailing space
        "plaque psoriasis",
        "type 2 diabetes mellitus ",  # trailing space
        "hiv1 infection",
        "parkinsons disease",
        "Parkinson's Disease",  # mixed case
        "non-small cell lung cancer",
        "seasonal allergic rhinitis",
        "Rheumatoid Arthritis",
    ]

    print("=== Disease Matcher Test ===\n")
    for disease in test_diseases:
        mesh_id = matcher.get_mesh_id(disease)
        status = "✓" if mesh_id else "✗"
        print(f"{status} {disease:40} -> {mesh_id or 'NOT FOUND'}")

    # Coverage report
    import pandas as pd
    gt = pd.read_excel("data/reference/everycure/indicationList.xlsx")
    ec_diseases = set(gt['disease name'].unique())

    report = matcher.coverage_report(ec_diseases)
    print(f"\n=== Coverage Report ===")
    print(f"Total diseases: {report['total']}")
    print(f"Mapped: {report['mapped']} ({report['coverage']*100:.1f}%)")
    print(f"Unmapped: {report['unmapped']}")


if __name__ == "__main__":
    test_matcher()
