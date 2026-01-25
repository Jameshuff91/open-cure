#!/usr/bin/env python3
"""
Pathway enrichment features for drug repurposing.

Uses KEGG pathway data to identify drugs that target pathways relevant
to disease mechanisms. The hypothesis is that drugs affecting disease-
relevant pathways are more likely to be therapeutic.

Data sources:
- KEGG DRUG: Drug → Target → Pathway mappings
- KEGG DISEASE: Disease → Gene → Pathway mappings
- Drug targets from drug_targets.json (already have)
- Disease genes from disease_genes.json (already have)
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import pickle

import requests
from tqdm import tqdm

# Cache paths
CACHE_DIR = Path('data/reference/pathway')
GENE_PATHWAY_CACHE = CACHE_DIR / 'gene_pathways.json'
PATHWAY_INFO_CACHE = CACHE_DIR / 'pathway_info.json'
DRUG_PATHWAY_CACHE = CACHE_DIR / 'drug_pathways.json'
DISEASE_PATHWAY_CACHE = CACHE_DIR / 'disease_pathways.json'

# KEGG API settings
KEGG_BASE = "https://rest.kegg.jp"
RATE_LIMIT_DELAY = 0.35  # KEGG allows ~3 requests/sec


def ensure_cache_dir() -> None:
    """Create cache directory if needed."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def kegg_request(endpoint: str, max_retries: int = 3) -> Optional[str]:
    """Make a request to KEGG REST API with rate limiting."""
    url = f"{KEGG_BASE}/{endpoint}"

    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)
            time.sleep(RATE_LIMIT_DELAY)

            if response.status_code == 200:
                return response.text
            elif response.status_code == 404:
                return None
            elif response.status_code == 403:
                # Rate limited
                time.sleep(5)
                continue
            else:
                return None

        except requests.RequestException:
            if attempt < max_retries - 1:
                time.sleep(1)

    return None


def fetch_all_gene_pathways_bulk(organism: str = "hsa") -> Dict[str, Set[str]]:
    """
    Fetch ALL gene-pathway mappings for an organism in bulk.

    Uses KEGG's link API to download all mappings at once,
    which is much faster than per-gene queries.

    Args:
        organism: KEGG organism code (hsa = human)

    Returns:
        Dict mapping Entrez Gene ID to set of pathway IDs
    """
    # Check cache
    if GENE_PATHWAY_CACHE.exists():
        with open(GENE_PATHWAY_CACHE) as f:
            cache = json.load(f)
            if cache:  # Non-empty cache
                return {k: set(v) for k, v in cache.items()}

    print(f"Fetching all {organism} gene-pathway mappings from KEGG (bulk download)...")

    # Single request to get ALL gene-pathway links
    response = kegg_request(f"link/pathway/{organism}")

    if not response:
        print("Error: Failed to fetch gene-pathway mappings from KEGG")
        return {}

    # Parse response
    gene_pathways: Dict[str, Set[str]] = defaultdict(set)

    for line in response.strip().split('\n'):
        if '\t' in line:
            parts = line.split('\t')
            gene_id = parts[0].replace(f'{organism}:', '')  # e.g., "10327"
            pathway_id = parts[1].replace('path:', '')  # e.g., "hsa00010"
            gene_pathways[gene_id].add(pathway_id)

    print(f"Loaded {len(gene_pathways)} genes with pathway annotations")

    # Save cache
    ensure_cache_dir()
    with open(GENE_PATHWAY_CACHE, 'w') as f:
        json.dump({k: list(v) for k, v in gene_pathways.items()}, f)

    return gene_pathways


def fetch_gene_pathways(gene_ids: List[str], organism: str = "hsa") -> Dict[str, Set[str]]:
    """
    Get pathway memberships for a list of genes.

    Args:
        gene_ids: List of Entrez Gene IDs (numeric strings like "1022")
        organism: KEGG organism code (hsa = human)

    Returns:
        Dict mapping gene ID to set of pathway IDs
    """
    # Get all gene-pathway mappings (cached after first call)
    all_mappings = fetch_all_gene_pathways_bulk(organism)

    # Filter to requested genes
    results = {}
    for gene_id in gene_ids:
        results[gene_id] = all_mappings.get(gene_id, set())

    return results


def fetch_all_human_pathways() -> Dict[str, Dict]:
    """
    Fetch all human KEGG pathway information.

    Returns:
        Dict mapping pathway ID to info dict (name, category, genes)
    """
    if PATHWAY_INFO_CACHE.exists():
        with open(PATHWAY_INFO_CACHE) as f:
            return json.load(f)

    print("Fetching all human pathways from KEGG...")

    # Get list of all human pathways
    response = kegg_request("list/pathway/hsa")
    if not response:
        return {}

    pathways = {}

    for line in response.strip().split('\n'):
        parts = line.split('\t')
        if len(parts) >= 2:
            pathway_id = parts[0].replace('path:', '')
            pathway_name = parts[1].split(' - ')[0]  # Remove " - Homo sapiens"

            pathways[pathway_id] = {
                'name': pathway_name,
                'genes': [],
            }

    print(f"Found {len(pathways)} human pathways")

    # Fetch genes for each pathway (this is slow, do in batches)
    for pathway_id in tqdm(list(pathways.keys())[:100], desc="Pathway genes"):  # Limit for speed
        response = kegg_request(f"link/hsa/{pathway_id}")
        if response:
            genes = []
            for line in response.strip().split('\n'):
                if 'hsa:' in line:
                    gene_id = line.split('\t')[1].replace('hsa:', '')
                    genes.append(gene_id)
            pathways[pathway_id]['genes'] = genes

    # Save cache
    ensure_cache_dir()
    with open(PATHWAY_INFO_CACHE, 'w') as f:
        json.dump(pathways, f, indent=2)

    return pathways


def compute_drug_pathways(
    drug_targets: Dict[str, Set[str]],
    gene_pathways: Dict[str, Set[str]],
) -> Dict[str, Set[str]]:
    """
    Compute pathways affected by each drug based on its targets.

    Args:
        drug_targets: Dict mapping DrugBank ID to set of target gene symbols
        gene_pathways: Dict mapping gene symbol to set of pathway IDs

    Returns:
        Dict mapping DrugBank ID to set of pathway IDs
    """
    drug_pathways = {}

    for drug_id, targets in drug_targets.items():
        pathways = set()
        for gene in targets:
            if gene in gene_pathways:
                pathways.update(gene_pathways[gene])
        drug_pathways[drug_id] = pathways

    return drug_pathways


def compute_disease_pathways(
    disease_genes: Dict[str, Set[str]],
    gene_pathways: Dict[str, Set[str]],
) -> Dict[str, Set[str]]:
    """
    Compute pathways relevant to each disease based on associated genes.

    Args:
        disease_genes: Dict mapping MESH ID to set of gene symbols
        gene_pathways: Dict mapping gene symbol to set of pathway IDs

    Returns:
        Dict mapping MESH ID to set of pathway IDs
    """
    disease_pathways = {}

    for disease_id, genes in disease_genes.items():
        pathways = set()
        for gene in genes:
            if gene in gene_pathways:
                pathways.update(gene_pathways[gene])
        disease_pathways[disease_id] = pathways

    return disease_pathways


class PathwayEnrichment:
    """Compute pathway enrichment features for drug-disease pairs."""

    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        self.drug_pathways: Dict[str, Set[str]] = {}
        self.disease_pathways: Dict[str, Set[str]] = {}
        self.gene_pathways: Dict[str, Set[str]] = {}

        if use_cache:
            self._load_caches()

    def _load_caches(self) -> None:
        """Load cached pathway data."""
        if DRUG_PATHWAY_CACHE.exists():
            with open(DRUG_PATHWAY_CACHE) as f:
                data = json.load(f)
                self.drug_pathways = {k: set(v) for k, v in data.items()}

        if DISEASE_PATHWAY_CACHE.exists():
            with open(DISEASE_PATHWAY_CACHE) as f:
                data = json.load(f)
                self.disease_pathways = {k: set(v) for k, v in data.items()}

        if GENE_PATHWAY_CACHE.exists():
            with open(GENE_PATHWAY_CACHE) as f:
                data = json.load(f)
                self.gene_pathways = {k: set(v) for k, v in data.items()}

    def _save_caches(self) -> None:
        """Save pathway data to cache."""
        ensure_cache_dir()

        with open(DRUG_PATHWAY_CACHE, 'w') as f:
            json.dump({k: list(v) for k, v in self.drug_pathways.items()}, f)

        with open(DISEASE_PATHWAY_CACHE, 'w') as f:
            json.dump({k: list(v) for k, v in self.disease_pathways.items()}, f)

    def build_pathway_mappings(
        self,
        drug_targets_path: Path = Path('data/reference/drug_targets.json'),
        disease_genes_path: Path = Path('data/reference/disease_genes.json'),
    ) -> None:
        """
        Build drug and disease pathway mappings from gene associations.

        This is a one-time operation that fetches pathway data from KEGG
        for all genes in our drug targets and disease genes files.
        """
        print("Building pathway mappings...")

        # Load drug targets and disease genes
        with open(drug_targets_path) as f:
            drug_targets = {k: set(v) for k, v in json.load(f).items()}

        with open(disease_genes_path) as f:
            disease_genes = {k: set(v) for k, v in json.load(f).items()}

        # Get all unique genes
        all_genes = set()
        for genes in drug_targets.values():
            all_genes.update(genes)
        for genes in disease_genes.values():
            all_genes.update(genes)

        print(f"Total unique genes: {len(all_genes)}")

        # Fetch gene → pathway mappings
        self.gene_pathways = fetch_gene_pathways(list(all_genes))

        genes_with_pathways = sum(1 for g in self.gene_pathways.values() if g)
        print(f"Genes with pathway data: {genes_with_pathways}/{len(all_genes)}")

        # Compute drug → pathway mappings
        self.drug_pathways = compute_drug_pathways(drug_targets, self.gene_pathways)
        drugs_with_pathways = sum(1 for p in self.drug_pathways.values() if p)
        print(f"Drugs with pathway data: {drugs_with_pathways}/{len(drug_targets)}")

        # Compute disease → pathway mappings
        self.disease_pathways = compute_disease_pathways(disease_genes, self.gene_pathways)
        diseases_with_pathways = sum(1 for p in self.disease_pathways.values() if p)
        print(f"Diseases with pathway data: {diseases_with_pathways}/{len(disease_genes)}")

        # Save caches
        self._save_caches()

    def get_pathway_overlap(self, drug_id: str, disease_id: str) -> Tuple[int, float, Set[str]]:
        """
        Compute pathway overlap between a drug and disease.

        Args:
            drug_id: DrugBank ID (e.g., "DB00001")
            disease_id: MESH ID (e.g., "MESH:D000544")

        Returns:
            Tuple of (overlap_count, jaccard_similarity, shared_pathways)
        """
        drug_pathways = self.drug_pathways.get(drug_id, set())
        disease_pathways = self.disease_pathways.get(disease_id, set())

        if not drug_pathways or not disease_pathways:
            return 0, 0.0, set()

        shared = drug_pathways & disease_pathways
        union = drug_pathways | disease_pathways

        jaccard = len(shared) / len(union) if union else 0.0

        return len(shared), jaccard, shared

    def get_pathway_score(self, drug_id: str, disease_id: str) -> float:
        """
        Get a pathway-based score for a drug-disease pair.

        Higher scores indicate more pathway overlap.
        """
        overlap, jaccard, _ = self.get_pathway_overlap(drug_id, disease_id)

        # Combine overlap count and Jaccard
        # More overlap = higher score, capped at 10 for stability
        return min(overlap, 10) * 0.1 + jaccard * 0.5

    def get_coverage_stats(self) -> Dict:
        """Get statistics about pathway coverage."""
        return {
            'drugs_with_pathways': sum(1 for p in self.drug_pathways.values() if p),
            'total_drugs': len(self.drug_pathways),
            'diseases_with_pathways': sum(1 for p in self.disease_pathways.values() if p),
            'total_diseases': len(self.disease_pathways),
            'genes_with_pathways': sum(1 for p in self.gene_pathways.values() if p),
            'total_genes': len(self.gene_pathways),
        }


def main():
    """Test pathway enrichment functionality."""
    print("=" * 60)
    print("PATHWAY ENRICHMENT TEST")
    print("=" * 60)

    pe = PathwayEnrichment()

    # Check if we have cached data
    stats = pe.get_coverage_stats()

    if stats['total_genes'] == 0:
        print("\nNo pathway data cached. Building mappings...")
        pe.build_pathway_mappings()
        stats = pe.get_coverage_stats()

    print(f"\nPathway coverage:")
    print(f"  Genes: {stats['genes_with_pathways']}/{stats['total_genes']}")
    print(f"  Drugs: {stats['drugs_with_pathways']}/{stats['total_drugs']}")
    print(f"  Diseases: {stats['diseases_with_pathways']}/{stats['total_diseases']}")

    # Test a few drug-disease pairs
    test_pairs = [
        ("DB00945", "MESH:D003920"),  # Aspirin - Diabetes
        ("DB00571", "MESH:D006973"),  # Propranolol - Hypertension
        ("DB00619", "MESH:D001249"),  # Imatinib - Leukemia
    ]

    print("\nTest pairs:")
    for drug_id, disease_id in test_pairs:
        overlap, jaccard, shared = pe.get_pathway_overlap(drug_id, disease_id)
        score = pe.get_pathway_score(drug_id, disease_id)
        print(f"  {drug_id} + {disease_id}: overlap={overlap}, jaccard={jaccard:.3f}, score={score:.3f}")


if __name__ == "__main__":
    main()
