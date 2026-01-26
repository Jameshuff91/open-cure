#!/usr/bin/env python3
"""
External Validation Pipeline for Drug Repurposing Predictions.

Validates predictions using:
1. ClinicalTrials.gov - Active/completed trials for drug-disease pairs
2. PubMed - Literature evidence for mechanism support
3. DrugBank - Existing indications check

This provides real-world evidence scoring independent of the training data.
"""

import json
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REFERENCE_DIR = DATA_DIR / "reference"
VALIDATION_DIR = DATA_DIR / "validation"

# Ensure validation directory exists
VALIDATION_DIR.mkdir(exist_ok=True)


@dataclass
class ClinicalTrialEvidence:
    """Evidence from ClinicalTrials.gov."""
    has_trials: bool = False
    trial_count: int = 0
    phases: list[str] = field(default_factory=list)
    statuses: list[str] = field(default_factory=list)
    trial_ids: list[str] = field(default_factory=list)
    earliest_trial: Optional[str] = None
    latest_trial: Optional[str] = None


@dataclass
class PubMedEvidence:
    """Evidence from PubMed literature."""
    has_publications: bool = False
    publication_count: int = 0
    recent_count: int = 0  # Last 5 years
    review_count: int = 0
    clinical_trial_pubs: int = 0
    sample_pmids: list[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Combined validation result for a drug-disease prediction."""
    drug_name: str
    disease_name: str
    drugbank_id: str
    mesh_id: str
    model_score: float

    # External evidence
    clinical_trials: Optional[ClinicalTrialEvidence] = None
    pubmed: Optional[PubMedEvidence] = None

    # Computed scores
    trial_score: float = 0.0
    literature_score: float = 0.0
    validation_score: float = 0.0

    # Metadata
    validated_at: str = ""

    def compute_scores(self) -> None:
        """Compute validation scores from evidence."""
        # Trial score: weighted by phase and status
        if self.clinical_trials and self.clinical_trials.has_trials:
            phase_weights = {"Phase 4": 1.0, "Phase 3": 0.8, "Phase 2": 0.5, "Phase 1": 0.3}
            status_weights = {"COMPLETED": 1.0, "ACTIVE_NOT_RECRUITING": 0.8,
                            "RECRUITING": 0.7, "NOT_YET_RECRUITING": 0.5}

            phase_score = max((phase_weights.get(p, 0.2) for p in self.clinical_trials.phases), default=0)
            status_score = max((status_weights.get(s, 0.3) for s in self.clinical_trials.statuses), default=0)
            count_bonus = min(self.clinical_trials.trial_count / 10, 1.0)

            self.trial_score = (phase_score * 0.5 + status_score * 0.3 + count_bonus * 0.2)

        # Literature score: weighted by recency and type
        if self.pubmed and self.pubmed.has_publications:
            base_score = min(self.pubmed.publication_count / 50, 1.0)
            recency_bonus = min(self.pubmed.recent_count / 20, 0.5)
            review_bonus = min(self.pubmed.review_count / 5, 0.3)
            clinical_bonus = min(self.pubmed.clinical_trial_pubs / 5, 0.3)

            self.literature_score = base_score * 0.4 + recency_bonus + review_bonus + clinical_bonus

        # Combined validation score
        self.validation_score = (
            self.trial_score * 0.6 +  # Trials are stronger evidence
            self.literature_score * 0.4
        )


class ClinicalTrialsAPI:
    """Query ClinicalTrials.gov API v2."""

    BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

    def __init__(self, rate_limit: float = 0.5):
        self.rate_limit = rate_limit
        self.last_request = 0.0

    def _wait_for_rate_limit(self) -> None:
        """Respect rate limiting."""
        elapsed = time.time() - self.last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request = time.time()

    def search(self, drug: str, disease: str) -> ClinicalTrialEvidence:
        """Search for trials matching drug and disease."""
        self._wait_for_rate_limit()

        evidence = ClinicalTrialEvidence()

        # Build query - search in intervention and condition fields
        query = f'AREA[InterventionName]{drug} AND AREA[Condition]{disease}'

        params = {
            "query.term": query,
            "pageSize": 50,
            "fields": "NCTId,BriefTitle,Phase,OverallStatus,StartDate,CompletionDate"
        }

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            studies = data.get("studies", [])

            if studies:
                evidence.has_trials = True
                evidence.trial_count = len(studies)

                for study in studies:
                    protocol = study.get("protocolSection", {})

                    # NCT ID
                    nct_id = protocol.get("identificationModule", {}).get("nctId", "")
                    if nct_id:
                        evidence.trial_ids.append(nct_id)

                    # Phase
                    design = protocol.get("designModule", {})
                    phases = design.get("phases", [])
                    for phase in phases:
                        if phase not in evidence.phases:
                            evidence.phases.append(phase)

                    # Status
                    status = protocol.get("statusModule", {}).get("overallStatus", "")
                    if status and status not in evidence.statuses:
                        evidence.statuses.append(status)

                    # Dates
                    status_mod = protocol.get("statusModule", {})
                    start = status_mod.get("startDateStruct", {}).get("date", "")
                    if start:
                        if not evidence.earliest_trial or start < evidence.earliest_trial:
                            evidence.earliest_trial = start
                        if not evidence.latest_trial or start > evidence.latest_trial:
                            evidence.latest_trial = start

        except requests.RequestException as e:
            print(f"  ClinicalTrials API error: {e}")
        except json.JSONDecodeError:
            print("  ClinicalTrials API returned invalid JSON")

        return evidence


class PubMedAPI:
    """Query PubMed E-utilities API."""

    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

    def __init__(self, rate_limit: float = 0.34):  # NCBI allows 3 requests/sec
        self.rate_limit = rate_limit
        self.last_request = 0.0

    def _wait_for_rate_limit(self) -> None:
        elapsed = time.time() - self.last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request = time.time()

    def search(self, drug: str, disease: str) -> PubMedEvidence:
        """Search PubMed for drug-disease publications."""
        evidence = PubMedEvidence()

        # Base search
        base_query = f'"{drug}"[Title/Abstract] AND "{disease}"[Title/Abstract]'
        evidence.publication_count = self._count_results(base_query)
        evidence.has_publications = evidence.publication_count > 0

        if evidence.has_publications:
            # Recent publications (last 5 years)
            recent_query = f'{base_query} AND ("2021"[Date - Publication] : "2026"[Date - Publication])'
            evidence.recent_count = self._count_results(recent_query)

            # Reviews
            review_query = f'{base_query} AND "review"[Publication Type]'
            evidence.review_count = self._count_results(review_query)

            # Clinical trials in literature
            trial_query = f'{base_query} AND "clinical trial"[Publication Type]'
            evidence.clinical_trial_pubs = self._count_results(trial_query)

            # Get sample PMIDs
            evidence.sample_pmids = self._get_pmids(base_query, max_results=5)

        return evidence

    def _count_results(self, query: str) -> int:
        """Get count of search results."""
        self._wait_for_rate_limit()

        params = {
            "db": "pubmed",
            "term": query,
            "rettype": "count",
            "retmode": "json"
        }

        try:
            response = requests.get(self.ESEARCH_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            return int(data.get("esearchresult", {}).get("count", 0))
        except (requests.RequestException, json.JSONDecodeError, ValueError):
            return 0

    def _get_pmids(self, query: str, max_results: int = 5) -> list[str]:
        """Get PMIDs for a query."""
        self._wait_for_rate_limit()

        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance"
        }

        try:
            response = requests.get(self.ESEARCH_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get("esearchresult", {}).get("idlist", [])
        except (requests.RequestException, json.JSONDecodeError):
            return []


class ExternalValidator:
    """Main validation pipeline."""

    def __init__(self, cache_file: Optional[Path] = None):
        self.ct_api = ClinicalTrialsAPI()
        self.pubmed_api = PubMedAPI()
        self.cache_file = cache_file or VALIDATION_DIR / "validation_cache.json"
        self.cache: dict[str, dict] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cached validation results."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    self.cache = json.load(f)
                print(f"Loaded {len(self.cache)} cached validations")
            except json.JSONDecodeError:
                self.cache = {}

    def _save_cache(self) -> None:
        """Save validation cache."""
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f, indent=2)

    def _get_cache_key(self, drug: str, disease: str) -> str:
        """Generate cache key for drug-disease pair."""
        return f"{drug.lower()}|{disease.lower()}"

    def validate_prediction(
        self,
        drug_name: str,
        disease_name: str,
        drugbank_id: str,
        mesh_id: str,
        model_score: float,
        use_cache: bool = True
    ) -> ValidationResult:
        """Validate a single drug-disease prediction."""

        cache_key = self._get_cache_key(drug_name, disease_name)

        # Check cache
        if use_cache and cache_key in self.cache:
            cached = self.cache[cache_key]
            result = ValidationResult(
                drug_name=drug_name,
                disease_name=disease_name,
                drugbank_id=drugbank_id,
                mesh_id=mesh_id,
                model_score=model_score,
                clinical_trials=ClinicalTrialEvidence(**cached.get("clinical_trials", {})),
                pubmed=PubMedEvidence(**cached.get("pubmed", {})),
                validated_at=cached.get("validated_at", "")
            )
            result.compute_scores()
            return result

        # Query external sources
        result = ValidationResult(
            drug_name=drug_name,
            disease_name=disease_name,
            drugbank_id=drugbank_id,
            mesh_id=mesh_id,
            model_score=model_score,
            validated_at=datetime.now().isoformat()
        )

        # ClinicalTrials.gov
        result.clinical_trials = self.ct_api.search(drug_name, disease_name)

        # PubMed
        result.pubmed = self.pubmed_api.search(drug_name, disease_name)

        # Compute scores
        result.compute_scores()

        # Cache result
        self.cache[cache_key] = {
            "clinical_trials": asdict(result.clinical_trials) if result.clinical_trials else {},
            "pubmed": asdict(result.pubmed) if result.pubmed else {},
            "validated_at": result.validated_at
        }

        return result

    def validate_predictions(
        self,
        predictions: list[dict],
        max_predictions: Optional[int] = None,
        save_every: int = 10
    ) -> list[ValidationResult]:
        """Validate a list of predictions."""

        if max_predictions:
            predictions = predictions[:max_predictions]

        results = []

        for i, pred in enumerate(tqdm(predictions, desc="Validating")):
            result = self.validate_prediction(
                drug_name=pred["drug_name"],
                disease_name=pred["disease_name"],
                drugbank_id=pred.get("drugbank_id", ""),
                mesh_id=pred.get("mesh_id", ""),
                model_score=pred.get("score", pred.get("model_score", 0))
            )
            results.append(result)

            # Save cache periodically
            if (i + 1) % save_every == 0:
                self._save_cache()

        # Final save
        self._save_cache()

        return results

    def generate_report(self, results: list[ValidationResult]) -> dict:
        """Generate summary report from validation results."""

        # Sort by validation score
        sorted_results = sorted(results, key=lambda r: r.validation_score, reverse=True)

        # Categorize
        strong_evidence = [r for r in sorted_results if r.validation_score >= 0.5]
        moderate_evidence = [r for r in sorted_results if 0.2 <= r.validation_score < 0.5]
        weak_evidence = [r for r in sorted_results if 0 < r.validation_score < 0.2]
        no_evidence = [r for r in sorted_results if r.validation_score == 0]

        # Count trials
        with_trials = [r for r in results if r.clinical_trials and r.clinical_trials.has_trials]
        with_pubs = [r for r in results if r.pubmed and r.pubmed.has_publications]

        report = {
            "summary": {
                "total_validated": len(results),
                "strong_evidence": len(strong_evidence),
                "moderate_evidence": len(moderate_evidence),
                "weak_evidence": len(weak_evidence),
                "no_evidence": len(no_evidence),
                "with_clinical_trials": len(with_trials),
                "with_publications": len(with_pubs),
            },
            "top_validated": [
                {
                    "drug": r.drug_name,
                    "disease": r.disease_name,
                    "model_score": r.model_score,
                    "validation_score": round(r.validation_score, 3),
                    "trial_count": r.clinical_trials.trial_count if r.clinical_trials else 0,
                    "phases": r.clinical_trials.phases if r.clinical_trials else [],
                    "pub_count": r.pubmed.publication_count if r.pubmed else 0,
                    "recent_pubs": r.pubmed.recent_count if r.pubmed else 0,
                }
                for r in sorted_results[:50]
            ],
            "by_evidence_category": {
                "strong": [{"drug": r.drug_name, "disease": r.disease_name, "score": r.validation_score}
                          for r in strong_evidence[:20]],
                "moderate": [{"drug": r.drug_name, "disease": r.disease_name, "score": r.validation_score}
                            for r in moderate_evidence[:20]],
            },
            "generated_at": datetime.now().isoformat()
        }

        return report


def main():
    """Run validation pipeline on top predictions."""

    print("=" * 70)
    print("EXTERNAL VALIDATION PIPELINE")
    print("Validating predictions against ClinicalTrials.gov and PubMed")
    print("=" * 70)

    # Load predictions
    pred_file = DATA_DIR / "analysis" / "novel_predictions.json"
    print(f"\n1. Loading predictions from {pred_file}")

    with open(pred_file) as f:
        pred_data = json.load(f)

    predictions = pred_data.get("top_100", pred_data.get("predictions", []))
    print(f"   Found {len(predictions)} predictions")

    # Initialize validator
    print("\n2. Initializing validator...")
    validator = ExternalValidator()

    # Validate top predictions
    print("\n3. Validating predictions (this may take a while)...")
    print("   - Querying ClinicalTrials.gov")
    print("   - Querying PubMed")

    results = validator.validate_predictions(
        predictions,
        max_predictions=100,  # Start with top 100
        save_every=10
    )

    # Generate report
    print("\n4. Generating report...")
    report = validator.generate_report(results)

    # Save report
    report_file = VALIDATION_DIR / "external_validation_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"   Saved to {report_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    summary = report["summary"]
    print(f"\nTotal validated: {summary['total_validated']}")
    print(f"With clinical trials: {summary['with_clinical_trials']}")
    print(f"With publications: {summary['with_publications']}")
    print(f"\nEvidence categories:")
    print(f"  Strong (≥0.5):   {summary['strong_evidence']}")
    print(f"  Moderate (0.2-0.5): {summary['moderate_evidence']}")
    print(f"  Weak (<0.2):     {summary['weak_evidence']}")
    print(f"  None (0):        {summary['no_evidence']}")

    print("\n" + "-" * 70)
    print("TOP 10 VALIDATED PREDICTIONS")
    print("-" * 70)

    for i, pred in enumerate(report["top_validated"][:10], 1):
        print(f"\n{i}. {pred['drug']} → {pred['disease']}")
        print(f"   Model score: {pred['model_score']:.3f}")
        print(f"   Validation score: {pred['validation_score']:.3f}")
        print(f"   Clinical trials: {pred['trial_count']} (Phases: {', '.join(pred['phases']) or 'N/A'})")
        print(f"   Publications: {pred['pub_count']} total, {pred['recent_pubs']} recent")

    print("\n" + "=" * 70)
    print("Done! Full report saved to data/validation/external_validation_report.json")


if __name__ == "__main__":
    main()
