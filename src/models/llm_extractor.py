#!/usr/bin/env python3
"""
LLM-Powered Relationship Extraction from Biomedical Literature.

This module uses Large Language Models to:
1. Extract drug-disease relationships from papers not yet in knowledge graphs
2. Identify novel mechanisms of action
3. Find supporting evidence for predictions
4. Generate hypotheses for rare diseases with sparse data

The key insight is that LLMs can read papers that haven't been structured
into knowledge graphs yet, potentially finding relationships before
they appear in curated databases.
"""

import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")


@dataclass
class ExtractedRelationship:
    """A relationship extracted from literature."""

    drug: str
    drug_id: str | None
    disease: str
    disease_id: str | None
    relationship_type: str  # treats, may_treat, inhibits, targets, etc.
    mechanism: str | None
    confidence: float
    evidence: str
    source_paper: str | None
    pmid: str | None


@dataclass
class LiteratureChunk:
    """A chunk of literature text for processing."""

    text: str
    source: str
    pmid: str | None = None
    section: str | None = None


@dataclass
class ExtractionConfig:
    """Configuration for LLM extraction."""

    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    temperature: float = 0.1  # Low for factual extraction
    batch_size: int = 5
    max_retries: int = 3


class LLMRelationshipExtractor:
    """
    Extract drug-disease relationships from literature using LLMs.

    This addresses a key limitation of knowledge graphs: they only contain
    relationships that have been manually curated. LLMs can read raw text
    and extract relationships that haven't been formalized yet.
    """

    def __init__(self, config: ExtractionConfig | None = None):
        self.config = config or ExtractionConfig()
        self.client = None
        self._init_client()

    def _init_client(self):
        """Initialize the Anthropic client."""
        try:
            from anthropic import Anthropic

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                self.client = Anthropic(api_key=api_key)
            else:
                logger.warning("ANTHROPIC_API_KEY not set. LLM extraction disabled.")
        except ImportError:
            logger.warning("anthropic package not installed. LLM extraction disabled.")

    EXTRACTION_PROMPT = """You are a biomedical expert extracting drug-disease relationships from scientific literature.

Analyze the following text and extract any relationships between drugs/compounds and diseases/conditions.

For each relationship found, provide:
1. drug_name: The name of the drug or compound
2. disease_name: The name of the disease or condition
3. relationship_type: One of [treats, may_treat, prevents, alleviates, targets_pathway, inhibits, exacerbates, contraindicated]
4. mechanism: Brief description of the mechanism of action (if mentioned)
5. confidence: Your confidence in this extraction (0.0-1.0)
6. evidence: The exact quote from the text supporting this relationship

Return your response as a JSON array of objects. If no relationships are found, return an empty array [].

Important guidelines:
- Only extract relationships explicitly stated or strongly implied in the text
- Do not infer relationships not supported by the text
- Be conservative with confidence scores
- Include the exact evidence quote

Text to analyze:
---
{text}
---

Respond with only the JSON array, no other text."""

    RARE_DISEASE_PROMPT = """You are a rare disease specialist analyzing literature for potential drug repurposing opportunities.

Given this text about {disease_name}, identify any drugs or compounds that might be therapeutically relevant, even if the connection is indirect.

Consider:
1. Drugs mentioned as treating similar conditions
2. Compounds targeting the same biological pathways
3. Drugs used in related rare diseases
4. Off-label uses mentioned in case reports
5. Experimental treatments in clinical trials

For each potential drug-disease relationship, provide:
1. drug_name: The drug or compound
2. relationship_type: One of [established_treatment, potential_treatment, targets_pathway, case_report, experimental, related_disease_treatment]
3. mechanism: How it might work (if known)
4. confidence: Your confidence (0.0-1.0)
5. reasoning: Why this drug might be relevant
6. evidence: Supporting quote from text

Text:
---
{text}
---

Respond with only a JSON array."""

    def extract_from_text(
        self,
        text: str,
        source: str | None = None,
        pmid: str | None = None,
    ) -> list[ExtractedRelationship]:
        """Extract relationships from a single text chunk."""
        if not self.client:
            logger.warning("LLM client not initialized")
            return []

        try:
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[
                    {
                        "role": "user",
                        "content": self.EXTRACTION_PROMPT.format(text=text),
                    }
                ],
            )

            # Parse JSON response
            response_text = response.content[0].text.strip()

            # Handle potential markdown code blocks
            if response_text.startswith("```"):
                response_text = re.sub(r"```(?:json)?\n?", "", response_text)
                response_text = response_text.strip()

            relationships_data = json.loads(response_text)

            relationships = []
            for item in relationships_data:
                rel = ExtractedRelationship(
                    drug=item.get("drug_name", ""),
                    drug_id=None,  # Will be resolved later
                    disease=item.get("disease_name", ""),
                    disease_id=None,
                    relationship_type=item.get("relationship_type", "unknown"),
                    mechanism=item.get("mechanism"),
                    confidence=float(item.get("confidence", 0.5)),
                    evidence=item.get("evidence", ""),
                    source_paper=source,
                    pmid=pmid,
                )
                relationships.append(rel)

            return relationships

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            return []
        except Exception as e:
            logger.error(f"Error during extraction: {e}")
            return []

    def extract_for_rare_disease(
        self,
        disease_name: str,
        text: str,
        source: str | None = None,
    ) -> list[ExtractedRelationship]:
        """
        Extract relationships specifically for rare diseases.

        Uses a more exploratory prompt that looks for indirect connections,
        which is important for diseases with limited literature.
        """
        if not self.client:
            return []

        try:
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=0.2,  # Slightly higher for more exploration
                messages=[
                    {
                        "role": "user",
                        "content": self.RARE_DISEASE_PROMPT.format(
                            disease_name=disease_name,
                            text=text,
                        ),
                    }
                ],
            )

            response_text = response.content[0].text.strip()
            if response_text.startswith("```"):
                response_text = re.sub(r"```(?:json)?\n?", "", response_text)

            relationships_data = json.loads(response_text)

            relationships = []
            for item in relationships_data:
                rel = ExtractedRelationship(
                    drug=item.get("drug_name", ""),
                    drug_id=None,
                    disease=disease_name,
                    disease_id=None,
                    relationship_type=item.get("relationship_type", "potential_treatment"),
                    mechanism=item.get("mechanism"),
                    confidence=float(item.get("confidence", 0.3)),
                    evidence=item.get("evidence", item.get("reasoning", "")),
                    source_paper=source,
                    pmid=None,
                )
                relationships.append(rel)

            return relationships

        except Exception as e:
            logger.error(f"Error during rare disease extraction: {e}")
            return []

    async def extract_batch_async(
        self,
        chunks: list[LiteratureChunk],
    ) -> list[ExtractedRelationship]:
        """Extract relationships from multiple chunks asynchronously."""
        if not self.client:
            return []

        try:
            from anthropic import AsyncAnthropic

            async_client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

            async def process_chunk(chunk: LiteratureChunk) -> list[ExtractedRelationship]:
                try:
                    response = await async_client.messages.create(
                        model=self.config.model,
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                        messages=[
                            {
                                "role": "user",
                                "content": self.EXTRACTION_PROMPT.format(text=chunk.text),
                            }
                        ],
                    )

                    response_text = response.content[0].text.strip()
                    if response_text.startswith("```"):
                        response_text = re.sub(r"```(?:json)?\n?", "", response_text)

                    data = json.loads(response_text)
                    return [
                        ExtractedRelationship(
                            drug=item.get("drug_name", ""),
                            drug_id=None,
                            disease=item.get("disease_name", ""),
                            disease_id=None,
                            relationship_type=item.get("relationship_type", "unknown"),
                            mechanism=item.get("mechanism"),
                            confidence=float(item.get("confidence", 0.5)),
                            evidence=item.get("evidence", ""),
                            source_paper=chunk.source,
                            pmid=chunk.pmid,
                        )
                        for item in data
                    ]
                except Exception as e:
                    logger.error(f"Error processing chunk: {e}")
                    return []

            # Process in batches
            all_relationships = []
            for i in range(0, len(chunks), self.config.batch_size):
                batch = chunks[i : i + self.config.batch_size]
                tasks = [process_chunk(chunk) for chunk in batch]
                results = await asyncio.gather(*tasks)
                for rels in results:
                    all_relationships.extend(rels)

            return all_relationships

        except ImportError:
            # Fall back to synchronous
            all_rels = []
            for chunk in chunks:
                rels = self.extract_from_text(chunk.text, chunk.source, chunk.pmid)
                all_rels.extend(rels)
            return all_rels

    def extract_batch(
        self,
        chunks: list[LiteratureChunk],
    ) -> list[ExtractedRelationship]:
        """Synchronous wrapper for batch extraction."""
        return asyncio.run(self.extract_batch_async(chunks))


class PubMedFetcher:
    """Fetch abstracts from PubMed for relationship extraction."""

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self, email: str | None = None):
        self.email = email or os.environ.get("PUBMED_EMAIL", "")

    def search(
        self,
        query: str,
        max_results: int = 100,
    ) -> list[str]:
        """Search PubMed and return PMIDs."""
        import requests

        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "email": self.email,
        }

        response = requests.get(f"{self.BASE_URL}/esearch.fcgi", params=params)
        response.raise_for_status()

        data = response.json()
        return data.get("esearchresult", {}).get("idlist", [])

    def fetch_abstracts(
        self,
        pmids: list[str],
    ) -> list[LiteratureChunk]:
        """Fetch abstracts for given PMIDs."""
        import requests

        if not pmids:
            return []

        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "abstract",
            "retmode": "xml",
            "email": self.email,
        }

        response = requests.get(f"{self.BASE_URL}/efetch.fcgi", params=params)
        response.raise_for_status()

        # Parse XML (simplified - would use xml.etree in production)
        chunks = []
        import xml.etree.ElementTree as ET

        root = ET.fromstring(response.text)

        for article in root.findall(".//PubmedArticle"):
            pmid_elem = article.find(".//PMID")
            abstract_elem = article.find(".//AbstractText")
            title_elem = article.find(".//ArticleTitle")

            if abstract_elem is not None and abstract_elem.text:
                pmid = pmid_elem.text if pmid_elem is not None else None
                title = title_elem.text if title_elem is not None else ""

                chunks.append(
                    LiteratureChunk(
                        text=f"Title: {title}\n\nAbstract: {abstract_elem.text}",
                        source=f"PubMed:{pmid}",
                        pmid=pmid,
                    )
                )

        return chunks


class LiteratureKnowledgeEnricher:
    """
    Enrich knowledge graphs with relationships extracted from literature.

    This class coordinates:
    1. Fetching relevant papers from PubMed
    2. Extracting relationships using LLMs
    3. Resolving entities to knowledge graph IDs
    4. Adding new edges to the graph
    """

    def __init__(
        self,
        extractor: LLMRelationshipExtractor | None = None,
        fetcher: PubMedFetcher | None = None,
    ):
        self.extractor = extractor or LLMRelationshipExtractor()
        self.fetcher = fetcher or PubMedFetcher()
        self.entity_resolver: dict[str, str] = {}  # name -> ID mapping

    def load_entity_mappings(self, nodes_path: Path):
        """Load entity name to ID mappings from unified graph."""
        import pandas as pd

        nodes_df = pd.read_csv(nodes_path)
        for _, row in nodes_df.iterrows():
            name = str(row.get("name", "")).lower()
            if name:
                self.entity_resolver[name] = row["id"]

    def resolve_entity(self, name: str, entity_type: str) -> str | None:
        """Try to resolve an entity name to a knowledge graph ID."""
        name_lower = name.lower()

        # Direct match
        if name_lower in self.entity_resolver:
            return self.entity_resolver[name_lower]

        # Partial match (for drugs with brand/generic names)
        for known_name, entity_id in self.entity_resolver.items():
            if name_lower in known_name or known_name in name_lower:
                return entity_id

        return None

    def enrich_for_disease(
        self,
        disease_name: str,
        max_papers: int = 50,
    ) -> list[ExtractedRelationship]:
        """
        Enrich knowledge graph with literature about a specific disease.

        Workflow:
        1. Search PubMed for papers about the disease
        2. Fetch abstracts
        3. Extract relationships using LLM
        4. Resolve entities to KG IDs
        """
        logger.info(f"Enriching knowledge for: {disease_name}")

        # Search for relevant papers
        query = f'("{disease_name}"[Title/Abstract]) AND (treatment[Title/Abstract] OR drug[Title/Abstract] OR therapy[Title/Abstract])'
        pmids = self.fetcher.search(query, max_results=max_papers)

        logger.info(f"Found {len(pmids)} papers")

        if not pmids:
            return []

        # Fetch abstracts
        chunks = self.fetcher.fetch_abstracts(pmids)
        logger.info(f"Fetched {len(chunks)} abstracts")

        # Extract relationships
        relationships = self.extractor.extract_batch(chunks)
        logger.info(f"Extracted {len(relationships)} relationships")

        # Resolve entities
        for rel in relationships:
            rel.drug_id = self.resolve_entity(rel.drug, "Drug")
            rel.disease_id = self.resolve_entity(rel.disease, "Disease")

        return relationships

    def enrich_for_rare_disease(
        self,
        disease_name: str,
        max_papers: int = 100,
    ) -> list[ExtractedRelationship]:
        """
        Special enrichment for rare diseases with sparse data.

        Uses broader search and more exploratory extraction.
        """
        logger.info(f"Rare disease enrichment for: {disease_name}")

        # Broader search including related terms
        queries = [
            f'"{disease_name}"[Title/Abstract]',
            f'"{disease_name}" treatment',
            f'"{disease_name}" therapy',
            f'"{disease_name}" case report',
        ]

        all_pmids = set()
        for query in queries:
            pmids = self.fetcher.search(query, max_results=max_papers // len(queries))
            all_pmids.update(pmids)

        logger.info(f"Found {len(all_pmids)} papers for rare disease")

        if not all_pmids:
            return []

        # Fetch and extract
        chunks = self.fetcher.fetch_abstracts(list(all_pmids))

        # Use rare disease specific extraction
        relationships = []
        for chunk in chunks:
            rels = self.extractor.extract_for_rare_disease(
                disease_name, chunk.text, chunk.source
            )
            relationships.extend(rels)

        # Resolve entities
        for rel in relationships:
            rel.drug_id = self.resolve_entity(rel.drug, "Drug")

        return relationships
