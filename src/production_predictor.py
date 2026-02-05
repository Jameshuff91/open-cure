#!/usr/bin/env python3
"""
Production Drug Repurposing Predictor

Unified pipeline integrating validated research findings:
- h39: kNN collaborative filtering with Node2Vec embeddings (best method)
- h135: Production tiered confidence system (GOLDEN/HIGH/MEDIUM/LOW/FILTER)
- h136: Category-specific filters for Tier 2/3 rescue

USAGE:
    # Get predictions for a disease
    from production_predictor import DrugRepurposingPredictor
    predictor = DrugRepurposingPredictor()
    results = predictor.predict("rheumatoid arthritis")

    # CLI usage
    python -m src.production_predictor "rheumatoid arthritis"
    python -m src.production_predictor --disease "type 2 diabetes" --top-k 30

TIER SYSTEM (h135 validated, 9.1x separation):
- GOLDEN (57.7%): Tier1 category + freq>=10 + mechanism
- HIGH (20.9%):   freq>=15 + mechanism OR rank<=5 + freq>=10 + mechanism
- MEDIUM (14.3%): freq>=5 + mechanism OR freq>=10
- LOW (6.4%):     All else passing filter
- FILTER (3.2%):  rank>20 OR no_targets OR (freq<=2 AND no_mechanism)
"""

import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class ConfidenceTier(Enum):
    """Confidence tiers from h135 (validated 9.1x precision separation)."""
    GOLDEN = "GOLDEN"    # ~57.7% precision
    HIGH = "HIGH"        # ~20.9% precision
    MEDIUM = "MEDIUM"    # ~14.3% precision
    LOW = "LOW"          # ~6.4% precision
    FILTER = "FILTER"    # ~3.2% precision (excluded)


@dataclass
class DrugPrediction:
    """A single drug prediction with confidence metadata."""
    drug_name: str
    drug_id: str
    rank: int
    knn_score: float
    norm_score: float
    confidence_tier: ConfidenceTier
    train_frequency: int
    mechanism_support: bool
    has_targets: bool
    category: str
    disease_tier: int

    # Category-specific rescue criteria (h136)
    category_rescue_applied: bool = False
    category_specific_tier: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            'drug': self.drug_name,
            'drug_id': self.drug_id,
            'rank': self.rank,
            'score': float(self.knn_score),
            'norm_score': float(self.norm_score),
            'confidence_tier': self.confidence_tier.value,
            'train_frequency': self.train_frequency,
            'mechanism_support': self.mechanism_support,
            'has_targets': self.has_targets,
            'category': self.category,
            'category_rescue_applied': self.category_rescue_applied,
        }


@dataclass
class PredictionResult:
    """Complete prediction result for a disease."""
    disease_name: str
    disease_id: Optional[str]
    category: str
    disease_tier: int
    predictions: List[DrugPrediction]
    neighbors_used: int
    coverage_warning: Optional[str] = None

    def get_by_tier(self, tier: ConfidenceTier) -> List[DrugPrediction]:
        """Get predictions filtered by confidence tier."""
        return [p for p in self.predictions if p.confidence_tier == tier]

    def summary(self) -> Dict:
        """Get summary statistics."""
        tier_counts = defaultdict(int)
        for p in self.predictions:
            tier_counts[p.confidence_tier.value] += 1
        return {
            'disease': self.disease_name,
            'category': self.category,
            'tier': self.disease_tier,
            'total_predictions': len(self.predictions),
            'by_tier': dict(tier_counts),
            'coverage_warning': self.coverage_warning,
        }


# Category definitions (from h71/h135)
TIER_1_CATEGORIES = {'autoimmune', 'dermatological', 'psychiatric', 'ophthalmic'}
TIER_2_CATEGORIES = {'cardiovascular', 'other', 'cancer'}

# h144: Statins achieve 60% precision for metabolic diseases (vs 6% baseline)
STATIN_DRUGS = {
    'atorvastatin', 'simvastatin', 'rosuvastatin', 'pravastatin', 'lovastatin',
    'fluvastatin', 'pitavastatin', 'cerivastatin',
}

CATEGORY_KEYWORDS = {
    'autoimmune': ['autoimmune', 'lupus', 'rheumatoid', 'arthritis', 'scleroderma', 'myasthenia',
                   'multiple sclerosis', 'crohn', 'colitis', 'psoriasis', 'sjögren'],
    'infectious': ['infection', 'bacterial', 'viral', 'fungal', 'hiv', 'aids', 'hepatitis',
                   'tuberculosis', 'malaria', 'pneumonia', 'sepsis', 'meningitis'],
    'cancer': ['cancer', 'carcinoma', 'tumor', 'leukemia', 'lymphoma', 'melanoma',
               'neoplasm', 'oncology', 'sarcoma', 'myeloma'],
    'cardiovascular': ['cardiac', 'heart', 'coronary', 'hypertension', 'arrhythmia',
                       'atherosclerosis', 'stroke', 'vascular', 'myocardial', 'angina'],
    'neurological': ['neurological', 'alzheimer', 'parkinson', 'epilepsy', 'neuropathy',
                     'dementia', 'huntington', 'brain'],
    'metabolic': ['diabetes', 'metabolic', 'obesity', 'thyroid', 'hyperlipidemia',
                  'hypercholesterolemia', 'gout', 'porphyria'],
    'psychiatric': ['depression', 'anxiety', 'bipolar', 'schizophrenia', 'psychiatric',
                    'ptsd', 'ocd', 'adhd', 'psychosis'],
    'respiratory': ['respiratory', 'asthma', 'copd', 'pulmonary', 'lung', 'bronchitis',
                    'pneumonitis', 'fibrosis'],
    'gastrointestinal': ['gastrointestinal', 'gastric', 'intestinal', 'bowel', 'liver',
                         'hepatic', 'cirrhosis', 'pancreatitis', 'celiac'],
    'dermatological': ['skin', 'dermatitis', 'eczema', 'dermatological',
                       'acne', 'urticaria', 'vitiligo'],
    'ophthalmic': ['eye', 'retinal', 'glaucoma', 'macular', 'ophthalmic', 'uveitis',
                   'conjunctivitis', 'keratitis'],
    'hematological': ['anemia', 'hemophilia', 'thrombocytopenia',
                      'neutropenia', 'hematological', 'myelodysplastic'],
}


class DrugRepurposingPredictor:
    """
    Production drug repurposing predictor using kNN collaborative filtering.

    Based on validated research:
    - h39: kNN with k=20 achieves 37.04% R@30 (best method)
    - h135: Tiered confidence with 9.1x precision separation
    - h136: Category-specific filters rescue Tier 2/3 categories
    """

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize the predictor by loading required data."""
        if data_dir is None:
            data_dir = Path(__file__).parent.parent

        self.data_dir = data_dir
        self.reference_dir = data_dir / "data" / "reference"
        self.embeddings_dir = data_dir / "data" / "embeddings"

        self._load_data()

    def _load_data(self) -> None:
        """Load all required data files."""
        # Load Node2Vec embeddings
        embeddings_path = self.embeddings_dir / "node2vec_256_named.csv"
        df = pd.read_csv(embeddings_path)
        dim_cols = [c for c in df.columns if c.startswith("dim_")]
        self.embeddings: Dict[str, np.ndarray] = {}
        for _, row in df.iterrows():
            entity = f"drkg:{row['entity']}"
            self.embeddings[entity] = row[dim_cols].values.astype(np.float32)

        # Load DrugBank lookup
        with open(self.reference_dir / "drugbank_lookup.json") as f:
            id_to_name = json.load(f)
        self.name_to_drug_id = {
            name.lower(): f"drkg:Compound::{db_id}"
            for db_id, name in id_to_name.items()
        }
        self.drug_id_to_name = {
            f"drkg:Compound::{db_id}": name
            for db_id, name in id_to_name.items()
        }

        # Load MESH mappings
        mesh_path = self.reference_dir / "mesh_mappings_from_agents.json"
        self.mesh_mappings: Dict[str, str] = {}
        if mesh_path.exists():
            with open(mesh_path) as f:
                mesh_data = json.load(f)
            for batch_data in mesh_data.values():
                if isinstance(batch_data, dict):
                    for disease_name, mesh_id in batch_data.items():
                        if mesh_id:
                            mesh_str = str(mesh_id)
                            if mesh_str.startswith("D") or mesh_str.startswith("C"):
                                self.mesh_mappings[disease_name.lower()] = f"drkg:Disease::MESH:{mesh_str}"

        # Load ground truth (for training drug frequencies)
        self._load_ground_truth()

        # Load drug targets
        self.drug_targets: Dict[str, Set[str]] = {}
        targets_path = self.reference_dir / "drug_targets.json"
        if targets_path.exists():
            with open(targets_path) as f:
                drug_targets = json.load(f)
            self.drug_targets = {
                f"drkg:Compound::{k}": set(v)
                for k, v in drug_targets.items()
            }

        # Load disease genes
        self.disease_genes: Dict[str, Set[str]] = {}
        genes_path = self.reference_dir / "disease_genes.json"
        if genes_path.exists():
            with open(genes_path) as f:
                disease_genes = json.load(f)
            for k, v in disease_genes.items():
                gene_set = set(v)
                self.disease_genes[k] = gene_set
                if k.startswith('MESH:'):
                    self.disease_genes[f"drkg:Disease::{k}"] = gene_set

        # Build disease lists for kNN
        self._build_knn_index()

    def _load_ground_truth(self) -> None:
        """Load ground truth for training drug frequencies."""
        # Import disease matcher
        sys.path.insert(0, str(self.data_dir / "src"))
        from disease_name_matcher import DiseaseMatcher, load_mesh_mappings

        df = pd.read_excel(self.reference_dir / "everycure" / "indicationList.xlsx")
        fuzzy_mappings = load_mesh_mappings()
        matcher = DiseaseMatcher(fuzzy_mappings)

        self.ground_truth: Dict[str, Set[str]] = defaultdict(set)
        self.disease_names: Dict[str, str] = {}

        for _, row in df.iterrows():
            disease = str(row.get("disease name", "")).strip()
            drug = str(row.get("final normalized drug label", "")).strip()
            if not disease or not drug:
                continue

            disease_id = matcher.get_mesh_id(disease)
            if not disease_id:
                disease_id = self.mesh_mappings.get(disease.lower())
            if not disease_id:
                continue

            self.disease_names[disease_id] = disease
            drug_id = self.name_to_drug_id.get(drug.lower())
            if drug_id:
                self.ground_truth[disease_id].add(drug_id)

        self.ground_truth = dict(self.ground_truth)

    def _build_knn_index(self) -> None:
        """Build index for kNN lookups."""
        # Training diseases (all diseases in ground truth)
        self.train_diseases = [d for d in self.ground_truth if d in self.embeddings]
        self.train_embeddings = np.array(
            [self.embeddings[d] for d in self.train_diseases],
            dtype=np.float32
        )

        # Drug training frequency
        self.drug_train_freq: Dict[str, int] = defaultdict(int)
        for disease_id, drugs in self.ground_truth.items():
            for drug_id in drugs:
                self.drug_train_freq[drug_id] += 1

    @staticmethod
    def categorize_disease(disease_name: str) -> str:
        """Categorize a disease by name."""
        name_lower = disease_name.lower()
        for category, keywords in CATEGORY_KEYWORDS.items():
            for kw in keywords:
                if kw in name_lower:
                    return category
        return 'other'

    @staticmethod
    def get_category_tier(category: str) -> int:
        """Get the tier (1-3) for a disease category."""
        if category in TIER_1_CATEGORIES:
            return 1
        elif category in TIER_2_CATEGORIES:
            return 2
        else:
            return 3

    def _compute_mechanism_support(self, drug_id: str, disease_id: str) -> bool:
        """Check if drug targets overlap with disease genes."""
        drug_genes = self.drug_targets.get(drug_id, set())
        dis_genes = self.disease_genes.get(disease_id, set())
        return len(drug_genes & dis_genes) > 0

    def _assign_confidence_tier(
        self,
        rank: int,
        train_frequency: int,
        mechanism_support: bool,
        has_targets: bool,
        disease_tier: int,
        category: str,
        drug_name: str = "",
    ) -> Tuple[ConfidenceTier, bool, Optional[str]]:
        """
        Assign confidence tier based on h135 criteria.
        Also applies h136/h144 category-specific rescue for Tier 2/3 diseases.

        Returns: (tier, rescue_applied, category_specific_tier)
        """
        # FILTER tier (h123 negative signals)
        if rank > 20:
            return ConfidenceTier.FILTER, False, None
        if not has_targets:
            return ConfidenceTier.FILTER, False, None
        if train_frequency <= 2 and not mechanism_support:
            return ConfidenceTier.FILTER, False, None

        # Apply h136/h144 category-specific rescue for Tier 2/3
        if disease_tier > 1:
            rescued_tier = self._apply_category_rescue(
                rank, train_frequency, mechanism_support, category, drug_name
            )
            if rescued_tier is not None:
                return rescued_tier, True, category

        # Standard h135 tier assignment for Tier 1
        # GOLDEN tier (Tier1 + freq>=10 + mechanism)
        if disease_tier == 1 and train_frequency >= 10 and mechanism_support:
            return ConfidenceTier.GOLDEN, False, None

        # HIGH tier
        if train_frequency >= 15 and mechanism_support:
            return ConfidenceTier.HIGH, False, None
        if rank <= 5 and train_frequency >= 10 and mechanism_support:
            return ConfidenceTier.HIGH, False, None

        # MEDIUM tier
        if train_frequency >= 5 and mechanism_support:
            return ConfidenceTier.MEDIUM, False, None
        if train_frequency >= 10:
            return ConfidenceTier.MEDIUM, False, None

        # LOW tier
        return ConfidenceTier.LOW, False, None

    def _apply_category_rescue(
        self,
        rank: int,
        train_frequency: int,
        mechanism_support: bool,
        category: str,
        drug_name: str = "",
    ) -> Optional[ConfidenceTier]:
        """
        Apply h136/h144 category-specific rescue filters.

        Returns the rescued tier or None if no rescue criteria met.

        h136 findings:
        - Infectious: rank<=10 + freq>=15 + mech = 55.6% precision (GOLDEN!)
        - Cardiovascular: rank<=5 + mech = 38.2% precision (HIGH)
        - Respiratory: rank<=10 + freq>=15 + mech = 35.0% precision (HIGH)

        h144 findings:
        - Metabolic + statin + rank<=10 = 60.0% precision (GOLDEN!)
        """
        if category == 'infectious':
            if rank <= 10 and train_frequency >= 15 and mechanism_support:
                return ConfidenceTier.GOLDEN  # 55.6% precision
            if rank <= 10 and train_frequency >= 10 and mechanism_support:
                return ConfidenceTier.HIGH

        elif category == 'cardiovascular':
            if rank <= 5 and mechanism_support:
                return ConfidenceTier.HIGH  # 38.2% precision

        elif category == 'respiratory':
            if rank <= 10 and train_frequency >= 15 and mechanism_support:
                return ConfidenceTier.HIGH  # 35.0% precision

        elif category == 'metabolic':
            # h144: Statin drugs achieve 60% precision for metabolic diseases
            drug_lower = drug_name.lower()
            if rank <= 10 and any(statin in drug_lower for statin in STATIN_DRUGS):
                return ConfidenceTier.GOLDEN  # 60.0% precision

        return None

    def find_disease_id(self, disease_name: str) -> Optional[str]:
        """Find the DRKG disease ID for a disease name."""
        # Try exact match first
        disease_lower = disease_name.lower()
        if disease_lower in self.mesh_mappings:
            return self.mesh_mappings[disease_lower]

        # Try fuzzy matching
        sys.path.insert(0, str(self.data_dir / "src"))
        from disease_name_matcher import DiseaseMatcher, load_mesh_mappings

        fuzzy_mappings = load_mesh_mappings()
        matcher = DiseaseMatcher(fuzzy_mappings)
        return matcher.get_mesh_id(disease_name)

    def predict(
        self,
        disease_name: str,
        k: int = 20,
        top_n: int = 30,
        include_filtered: bool = False,
    ) -> PredictionResult:
        """
        Generate drug predictions for a disease.

        Args:
            disease_name: Name of the disease
            k: Number of nearest neighbors for kNN (default 20 from h39)
            top_n: Number of top predictions to return
            include_filtered: If True, include FILTER tier predictions

        Returns:
            PredictionResult with ranked predictions and confidence tiers
        """
        # Find disease ID
        disease_id = self.find_disease_id(disease_name)

        # Categorize disease
        category = self.categorize_disease(disease_name)
        disease_tier = self.get_category_tier(category)

        # Check if disease is in embeddings
        coverage_warning = None
        if disease_id is None or disease_id not in self.embeddings:
            coverage_warning = f"Disease '{disease_name}' not found in DRKG. Using name-based matching only."
            # Fall back to finding similar diseases by name
            disease_id = None

        predictions = []

        if disease_id and disease_id in self.embeddings:
            # Run kNN (h39 method)
            test_emb = self.embeddings[disease_id].reshape(1, -1)
            sims = cosine_similarity(test_emb, self.train_embeddings)[0]
            top_k_idx = np.argsort(sims)[-k:]

            # Aggregate drug scores from neighbors
            drug_scores: Dict[str, float] = defaultdict(float)
            for idx in top_k_idx:
                neighbor_disease = self.train_diseases[idx]
                neighbor_sim = sims[idx]
                for drug_id in self.ground_truth[neighbor_disease]:
                    if drug_id in self.embeddings:
                        drug_scores[drug_id] += neighbor_sim

            if not drug_scores:
                coverage_warning = "No drugs found in kNN neighborhood."
            else:
                # Rank drugs
                sorted_drugs = sorted(drug_scores.items(), key=lambda x: x[1], reverse=True)
                max_score = sorted_drugs[0][1] if sorted_drugs else 1.0

                for rank, (drug_id, score) in enumerate(sorted_drugs[:top_n], 1):
                    norm_score = score / max_score if max_score > 0 else 0
                    train_freq = self.drug_train_freq.get(drug_id, 0)
                    mech_support = self._compute_mechanism_support(drug_id, disease_id)
                    has_targets = drug_id in self.drug_targets and len(self.drug_targets[drug_id]) > 0
                    drug_name = self.drug_id_to_name.get(drug_id, drug_id)

                    tier, rescue_applied, cat_specific = self._assign_confidence_tier(
                        rank, train_freq, mech_support, has_targets, disease_tier, category, drug_name
                    )

                    pred = DrugPrediction(
                        drug_name=drug_name,
                        drug_id=drug_id,
                        rank=rank,
                        knn_score=score,
                        norm_score=norm_score,
                        confidence_tier=tier,
                        train_frequency=train_freq,
                        mechanism_support=mech_support,
                        has_targets=has_targets,
                        category=category,
                        disease_tier=disease_tier,
                        category_rescue_applied=rescue_applied,
                        category_specific_tier=cat_specific,
                    )

                    if include_filtered or tier != ConfidenceTier.FILTER:
                        predictions.append(pred)

        return PredictionResult(
            disease_name=disease_name,
            disease_id=disease_id,
            category=category,
            disease_tier=disease_tier,
            predictions=predictions,
            neighbors_used=k,
            coverage_warning=coverage_warning,
        )

    def batch_predict(
        self,
        diseases: List[str],
        **kwargs,
    ) -> Dict[str, PredictionResult]:
        """Generate predictions for multiple diseases."""
        results = {}
        for disease in diseases:
            results[disease] = self.predict(disease, **kwargs)
        return results


def print_predictions(result: PredictionResult) -> None:
    """Pretty-print prediction results."""
    print("=" * 80)
    print(f"PREDICTIONS FOR: {result.disease_name}")
    print("=" * 80)
    print(f"Category: {result.category} (Tier {result.disease_tier})")
    print(f"Disease ID: {result.disease_id or 'Not found'}")
    if result.coverage_warning:
        print(f"⚠️  Warning: {result.coverage_warning}")
    print()

    # Print by tier
    tier_order = [ConfidenceTier.GOLDEN, ConfidenceTier.HIGH, ConfidenceTier.MEDIUM, ConfidenceTier.LOW]
    tier_emoji = {
        ConfidenceTier.GOLDEN: "🏆",
        ConfidenceTier.HIGH: "✓",
        ConfidenceTier.MEDIUM: "○",
        ConfidenceTier.LOW: "·",
    }
    tier_precision = {
        ConfidenceTier.GOLDEN: "~58%",
        ConfidenceTier.HIGH: "~21%",
        ConfidenceTier.MEDIUM: "~14%",
        ConfidenceTier.LOW: "~6%",
    }

    for tier in tier_order:
        preds = result.get_by_tier(tier)
        if preds:
            print(f"\n{tier_emoji[tier]} {tier.value} CONFIDENCE (expected precision: {tier_precision[tier]})")
            print("-" * 70)
            print(f"{'Rank':<6} {'Drug':<35} {'Score':<8} {'Freq':<6} {'Mech':<6}")
            for p in preds[:10]:  # Show top 10 per tier
                mech = "✓" if p.mechanism_support else "-"
                rescue = " [rescued]" if p.category_rescue_applied else ""
                print(f"{p.rank:<6} {p.drug_name[:33]:<35} {p.norm_score:<8.3f} {p.train_frequency:<6} {mech:<6}{rescue}")

    # Summary
    print("\n" + "=" * 80)
    summary = result.summary()
    print(f"Total predictions: {summary['total_predictions']}")
    print(f"By tier: {summary['by_tier']}")


def main():
    """CLI interface for the predictor."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Drug Repurposing Predictions using kNN Collaborative Filtering"
    )
    parser.add_argument("disease", nargs="?", help="Disease name to predict for")
    parser.add_argument("--disease", "-d", dest="disease_flag", help="Disease name (alternative)")
    parser.add_argument("--top-k", "-k", type=int, default=30, help="Number of predictions (default: 30)")
    parser.add_argument("--neighbors", "-n", type=int, default=20, help="kNN neighbors (default: 20)")
    parser.add_argument("--include-filtered", action="store_true", help="Include FILTER tier predictions")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    disease = args.disease or args.disease_flag
    if not disease:
        parser.print_help()
        print("\nExample: python -m src.production_predictor 'rheumatoid arthritis'")
        sys.exit(1)

    print("Loading predictor...")
    predictor = DrugRepurposingPredictor()

    print(f"Generating predictions for: {disease}")
    result = predictor.predict(
        disease,
        k=args.neighbors,
        top_n=args.top_k,
        include_filtered=args.include_filtered,
    )

    if args.json:
        output = {
            'summary': result.summary(),
            'predictions': [p.to_dict() for p in result.predictions],
        }
        print(json.dumps(output, indent=2))
    else:
        print_predictions(result)


if __name__ == "__main__":
    main()
