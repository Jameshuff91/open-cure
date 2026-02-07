#!/usr/bin/env python3
"""
Literature Mining for Drug Repurposing Prediction Validation.

Automates literature evidence gathering using:
1. PubMed E-utilities (esearch + efetch) for publication counts and abstracts
2. ClinicalTrials.gov API v2 for trial evidence
3. Claude Haiku for abstract classification (optional LLM mode)

Reuses existing validation_cache.json (1,052 entries) and adds deeper
analysis with tiered queries and LLM-based abstract interpretation.

Evidence levels:
- STRONG_EVIDENCE: Phase 3+ trial OR 20+ pubs with clinical trial pubs OR LLM confirms treatment
- MODERATE_EVIDENCE: Phase 2 trial OR 5+ pubs with recent activity OR LLM confirms mechanism
- WEAK_EVIDENCE: 1-4 pubs, mechanism only
- ADVERSE_EFFECT: LLM detects drug causes/worsens disease
- NO_EVIDENCE: 0 publications found
- NOT_ASSESSED: Not yet processed
"""

import json
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

# Reuse existing API classes
from src.external_validation import PubMedAPI, ClinicalTrialsAPI

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VALIDATION_DIR = DATA_DIR / "validation"
VALIDATION_DIR.mkdir(exist_ok=True)

# Cache paths
LITERATURE_CACHE_PATH = VALIDATION_DIR / "literature_mining_cache.json"
EXISTING_CACHE_PATH = VALIDATION_DIR / "validation_cache.json"

# NCBI efetch for abstract retrieval
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


@dataclass
class LiteratureEvidence:
    """Structured evidence result for a drug-disease pair."""
    drug_name: str
    disease_name: str

    # PubMed evidence
    pubmed_total: int = 0
    pubmed_recent: int = 0  # Last 5 years
    pubmed_clinical_trial: int = 0
    pubmed_review: int = 0
    sample_pmids: list[str] = field(default_factory=list)

    # ClinicalTrials.gov evidence
    trial_count: int = 0
    trial_phases: list[str] = field(default_factory=list)
    trial_statuses: list[str] = field(default_factory=list)
    has_phase3_plus: bool = False

    # Tiered query results
    direct_hit_count: int = 0      # "drug" AND "disease" in title/abstract
    treatment_context: int = 0     # + treatment/therapy/efficacy
    drug_class_hit: int = 0        # Drug class name instead of specific drug

    # LLM classification (if use_llm=True)
    llm_classifications: list[str] = field(default_factory=list)
    llm_adverse_detected: bool = False
    llm_treatment_confirmed: bool = False
    llm_summary: str = ""

    # Computed evidence level
    evidence_level: str = "NOT_ASSESSED"
    evidence_score: float = 0.0

    # Metadata
    assessed_at: str = ""
    used_llm: bool = False
    from_existing_cache: bool = False

    def compute_evidence(self) -> None:
        """Compute evidence level and score from raw data."""
        score = 0.0

        # Trial evidence (strongest signal)
        if self.has_phase3_plus:
            score += 4.0
        elif self.trial_count > 0:
            # Phase 2 = moderate, Phase 1 = weak
            if any("PHASE3" in p or "PHASE4" in p or "Phase 3" in p or "Phase 4" in p
                   for p in self.trial_phases):
                score += 4.0
                self.has_phase3_plus = True
            elif any("PHASE2" in p or "Phase 2" in p for p in self.trial_phases):
                score += 2.5
            else:
                score += 1.0

        # Publication evidence
        if self.pubmed_total >= 20 and self.pubmed_clinical_trial > 0:
            score += 3.0
        elif self.pubmed_total >= 5:
            score += 1.5
            if self.pubmed_recent >= 3:
                score += 0.5  # Active research area
        elif self.pubmed_total >= 1:
            score += 0.5

        # Treatment-context publications bonus
        if self.treatment_context >= 5:
            score += 1.0
        elif self.treatment_context >= 1:
            score += 0.3

        # Clinical trial publications bonus
        if self.pubmed_clinical_trial >= 3:
            score += 1.0

        # LLM signals
        if self.llm_adverse_detected:
            self.evidence_level = "ADVERSE_EFFECT"
            self.evidence_score = score
            return

        if self.llm_treatment_confirmed:
            score += 2.0

        # Assign level based on score
        self.evidence_score = round(score, 2)

        if score >= 4.0:
            self.evidence_level = "STRONG_EVIDENCE"
        elif score >= 2.0:
            self.evidence_level = "MODERATE_EVIDENCE"
        elif score > 0:
            self.evidence_level = "WEAK_EVIDENCE"
        else:
            self.evidence_level = "NO_EVIDENCE"


class LiteratureMiner:
    """Automated literature mining for drug-disease pair validation."""

    def __init__(self, use_llm: bool = False, anthropic_model: str = "claude-haiku-4-5-20251001"):
        self.pubmed = PubMedAPI(rate_limit=0.34)
        self.ct_api = ClinicalTrialsAPI(rate_limit=0.5)
        self.use_llm = use_llm
        self.anthropic_model = anthropic_model
        self.anthropic_client = None

        # Load caches
        self.cache: dict[str, dict] = {}
        self._load_cache()

        self.existing_cache: dict[str, dict] = {}
        self._load_existing_cache()

        if use_llm:
            self._init_anthropic()

    def _init_anthropic(self) -> None:
        """Initialize Anthropic client for LLM classification."""
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("WARNING: ANTHROPIC_API_KEY not set. LLM classification disabled.")
            self.use_llm = False
            return
        try:
            import anthropic
            self.anthropic_client = anthropic.Anthropic(api_key=api_key)
            print(f"Anthropic client initialized (model: {self.anthropic_model})")
        except ImportError:
            print("WARNING: anthropic package not installed. LLM classification disabled.")
            self.use_llm = False

    def _load_cache(self) -> None:
        """Load literature mining cache."""
        if LITERATURE_CACHE_PATH.exists():
            try:
                with open(LITERATURE_CACHE_PATH) as f:
                    self.cache = json.load(f)
                print(f"Loaded {len(self.cache)} literature mining cache entries")
            except json.JSONDecodeError:
                self.cache = {}

    def _save_cache(self) -> None:
        """Save literature mining cache."""
        with open(LITERATURE_CACHE_PATH, "w") as f:
            json.dump(self.cache, f, indent=2)

    def _load_existing_cache(self) -> None:
        """Load existing validation_cache.json from prior runs."""
        if EXISTING_CACHE_PATH.exists():
            try:
                with open(EXISTING_CACHE_PATH) as f:
                    self.existing_cache = json.load(f)
                print(f"Loaded {len(self.existing_cache)} existing validation cache entries")
            except json.JSONDecodeError:
                self.existing_cache = {}

    @staticmethod
    def _cache_key(drug: str, disease: str) -> str:
        return f"{drug.lower()}|{disease.lower()}"

    def _check_existing_cache(self, drug: str, disease: str) -> Optional[dict]:
        """Check if drug-disease pair already has validation data."""
        key = self._cache_key(drug, disease)
        return self.existing_cache.get(key)

    def _fetch_abstracts(self, pmids: list[str], max_abstracts: int = 5) -> list[dict]:
        """Fetch abstracts from PubMed via efetch."""
        if not pmids:
            return []

        ids_to_fetch = pmids[:max_abstracts]
        time.sleep(0.34)  # NCBI rate limit

        params = {
            "db": "pubmed",
            "id": ",".join(ids_to_fetch),
            "rettype": "abstract",
            "retmode": "xml"
        }

        try:
            response = requests.get(EFETCH_URL, params=params, timeout=30)
            response.raise_for_status()
            root = ET.fromstring(response.text)

            abstracts = []
            for article in root.findall(".//PubmedArticle"):
                pmid_el = article.find(".//PMID")
                pmid = pmid_el.text if pmid_el is not None else ""

                title_el = article.find(".//ArticleTitle")
                title = title_el.text if title_el is not None else ""

                abstract_el = article.find(".//Abstract/AbstractText")
                abstract_text = ""
                if abstract_el is not None:
                    abstract_text = abstract_el.text or ""
                    # Handle structured abstracts
                    if not abstract_text:
                        parts = []
                        for part in article.findall(".//Abstract/AbstractText"):
                            label = part.get("Label", "")
                            text = part.text or ""
                            if label:
                                parts.append(f"{label}: {text}")
                            else:
                                parts.append(text)
                        abstract_text = " ".join(parts)

                abstracts.append({
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract_text[:2000]  # Truncate very long abstracts
                })

            return abstracts
        except (requests.RequestException, ET.ParseError) as e:
            print(f"  efetch error: {e}")
            return []

    def _classify_abstracts(self, drug: str, disease: str,
                            abstracts: list[dict]) -> dict:
        """Use Claude Haiku to classify abstracts."""
        if not self.anthropic_client or not abstracts:
            return {"classifications": [], "adverse_detected": False,
                    "treatment_confirmed": False, "summary": ""}

        abstracts_text = ""
        for i, ab in enumerate(abstracts[:5], 1):
            abstracts_text += f"\n--- Abstract {i} (PMID: {ab['pmid']}) ---\n"
            abstracts_text += f"Title: {ab['title']}\n"
            abstracts_text += f"{ab['abstract']}\n"

        prompt = f"""Classify the relationship between the drug "{drug}" and the disease "{disease}" based on these PubMed abstracts.

For each abstract, classify as ONE of:
- TREATS: Evidence that the drug treats or improves the disease
- CLINICAL_TRIAL: A clinical trial testing the drug for this disease
- MECHANISM: Mechanistic evidence (in vitro, animal model, pathway analysis)
- ADVERSE_EFFECT: Evidence that the drug CAUSES or WORSENS the disease
- NO_RELEVANCE: Abstract is not relevant to this drug-disease relationship

{abstracts_text}

Respond in this exact JSON format:
{{"classifications": ["TREATS", "MECHANISM", ...], "adverse_detected": false, "treatment_confirmed": true, "summary": "Brief 1-sentence summary of the evidence"}}

IMPORTANT: Set adverse_detected=true ONLY if an abstract clearly states the drug causes, induces, or worsens the disease. Set treatment_confirmed=true if any abstract provides clinical evidence of efficacy."""

        try:
            response = self.anthropic_client.messages.create(
                model=self.anthropic_model,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            text = response.content[0].text.strip()

            # Parse JSON response
            # Handle potential markdown wrapping
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            result = json.loads(text)
            return result
        except Exception as e:
            print(f"  LLM classification error: {e}")
            return {"classifications": [], "adverse_detected": False,
                    "treatment_confirmed": False, "summary": f"LLM error: {e}"}

    def _tiered_pubmed_search(self, drug: str, disease: str) -> dict:
        """Run tiered PubMed searches: direct → treatment-context → drug-class."""
        results = {
            "direct_hit_count": 0,
            "treatment_context": 0,
            "drug_class_hit": 0,
        }

        # Tier 1: Direct co-occurrence (already done by PubMedAPI.search)
        # Just get treatment context count
        treatment_query = (
            f'"{drug}"[Title/Abstract] AND "{disease}"[Title/Abstract] '
            f'AND (treatment[Title/Abstract] OR therapy[Title/Abstract] '
            f'OR efficacy[Title/Abstract] OR clinical trial[Publication Type])'
        )
        results["treatment_context"] = self.pubmed._count_results(treatment_query)

        return results

    def mine_pair(self, drug: str, disease: str,
                  force_refresh: bool = False) -> LiteratureEvidence:
        """Mine literature evidence for a single drug-disease pair."""
        key = self._cache_key(drug, disease)

        # Check our cache first (unless forcing refresh)
        if not force_refresh and key in self.cache:
            cached = self.cache[key]
            evidence = LiteratureEvidence(
                drug_name=drug,
                disease_name=disease,
                **{k: v for k, v in cached.items()
                   if k not in ("drug_name", "disease_name")}
            )
            return evidence

        evidence = LiteratureEvidence(
            drug_name=drug,
            disease_name=disease,
            assessed_at=datetime.now().isoformat(),
        )

        # Check existing validation cache for PubMed/CT data
        existing = self._check_existing_cache(drug, disease)
        if existing and not force_refresh:
            # Reuse existing PubMed data
            pm = existing.get("pubmed", {})
            evidence.pubmed_total = pm.get("publication_count", 0)
            evidence.pubmed_recent = pm.get("recent_count", 0)
            evidence.pubmed_clinical_trial = pm.get("clinical_trial_pubs", 0)
            evidence.pubmed_review = pm.get("review_count", 0)
            evidence.sample_pmids = pm.get("sample_pmids", [])

            # Reuse existing ClinicalTrials data
            ct = existing.get("clinical_trials", {})
            evidence.trial_count = ct.get("trial_count", 0)
            evidence.trial_phases = ct.get("phases", [])
            evidence.trial_statuses = ct.get("statuses", [])

            evidence.from_existing_cache = True
            evidence.direct_hit_count = evidence.pubmed_total
        else:
            # Fresh PubMed search
            pm_result = self.pubmed.search(drug, disease)
            evidence.pubmed_total = pm_result.publication_count
            evidence.pubmed_recent = pm_result.recent_count
            evidence.pubmed_clinical_trial = pm_result.clinical_trial_pubs
            evidence.pubmed_review = pm_result.review_count
            evidence.sample_pmids = pm_result.sample_pmids
            evidence.direct_hit_count = pm_result.publication_count

            # Fresh ClinicalTrials search
            ct_result = self.ct_api.search(drug, disease)
            evidence.trial_count = ct_result.trial_count
            evidence.trial_phases = ct_result.phases
            evidence.trial_statuses = ct_result.statuses

        # Check for Phase 3+
        evidence.has_phase3_plus = any(
            "PHASE3" in p or "PHASE4" in p or "Phase 3" in p or "Phase 4" in p
            for p in evidence.trial_phases
        )

        # Tiered PubMed search (treatment context) - only if we have direct hits
        if evidence.pubmed_total > 0 and not evidence.from_existing_cache:
            tiered = self._tiered_pubmed_search(drug, disease)
            evidence.treatment_context = tiered["treatment_context"]

        # LLM classification if enabled and we have abstracts to classify
        if self.use_llm and evidence.sample_pmids:
            evidence.used_llm = True
            abstracts = self._fetch_abstracts(evidence.sample_pmids)
            if abstracts:
                llm_result = self._classify_abstracts(drug, disease, abstracts)
                evidence.llm_classifications = llm_result.get("classifications", [])
                evidence.llm_adverse_detected = llm_result.get("adverse_detected", False)
                evidence.llm_treatment_confirmed = llm_result.get("treatment_confirmed", False)
                evidence.llm_summary = llm_result.get("summary", "")

        # Compute evidence level
        evidence.compute_evidence()

        # Cache result
        self.cache[key] = {
            "pubmed_total": evidence.pubmed_total,
            "pubmed_recent": evidence.pubmed_recent,
            "pubmed_clinical_trial": evidence.pubmed_clinical_trial,
            "pubmed_review": evidence.pubmed_review,
            "sample_pmids": evidence.sample_pmids,
            "trial_count": evidence.trial_count,
            "trial_phases": evidence.trial_phases,
            "trial_statuses": evidence.trial_statuses,
            "has_phase3_plus": evidence.has_phase3_plus,
            "direct_hit_count": evidence.direct_hit_count,
            "treatment_context": evidence.treatment_context,
            "drug_class_hit": evidence.drug_class_hit,
            "llm_classifications": evidence.llm_classifications,
            "llm_adverse_detected": evidence.llm_adverse_detected,
            "llm_treatment_confirmed": evidence.llm_treatment_confirmed,
            "llm_summary": evidence.llm_summary,
            "evidence_level": evidence.evidence_level,
            "evidence_score": evidence.evidence_score,
            "assessed_at": evidence.assessed_at,
            "used_llm": evidence.used_llm,
            "from_existing_cache": evidence.from_existing_cache,
        }

        return evidence

    def mine_batch(self, pairs: list[dict], save_every: int = 25,
                   force_refresh: bool = False) -> list[LiteratureEvidence]:
        """Mine literature evidence for a batch of drug-disease pairs.

        Args:
            pairs: List of dicts with 'drug_name' and 'disease_name' keys.
            save_every: Save cache every N pairs.
            force_refresh: Re-query even if cached.

        Returns:
            List of LiteratureEvidence results.
        """
        results = []
        cached_count = 0
        fresh_count = 0

        for i, pair in enumerate(tqdm(pairs, desc="Mining literature")):
            drug = pair["drug_name"]
            disease = pair["disease_name"]

            key = self._cache_key(drug, disease)
            was_cached = key in self.cache and not force_refresh

            evidence = self.mine_pair(drug, disease, force_refresh=force_refresh)
            results.append(evidence)

            if was_cached:
                cached_count += 1
            else:
                fresh_count += 1

            # Save periodically
            if (i + 1) % save_every == 0:
                self._save_cache()

        # Final save
        self._save_cache()

        print(f"\nMining complete: {len(results)} pairs "
              f"({cached_count} cached, {fresh_count} fresh queries)")

        return results

    def get_cached_evidence(self, drug: str, disease: str) -> Optional[LiteratureEvidence]:
        """Get cached evidence without querying APIs."""
        key = self._cache_key(drug, disease)
        if key not in self.cache:
            return None

        cached = self.cache[key]
        evidence = LiteratureEvidence(
            drug_name=drug,
            disease_name=disease,
        )
        for k, v in cached.items():
            if hasattr(evidence, k):
                setattr(evidence, k, v)
        return evidence

    def summary_stats(self) -> dict:
        """Generate summary statistics from cache."""
        if not self.cache:
            return {"total": 0}

        levels = {}
        for entry in self.cache.values():
            level = entry.get("evidence_level", "NOT_ASSESSED")
            levels[level] = levels.get(level, 0) + 1

        scores = [e.get("evidence_score", 0) for e in self.cache.values()]
        adverse = sum(1 for e in self.cache.values()
                      if e.get("llm_adverse_detected", False))
        with_llm = sum(1 for e in self.cache.values()
                       if e.get("used_llm", False))

        return {
            "total": len(self.cache),
            "by_level": levels,
            "mean_score": round(sum(scores) / len(scores), 2) if scores else 0,
            "adverse_effects_detected": adverse,
            "with_llm_classification": with_llm,
        }
