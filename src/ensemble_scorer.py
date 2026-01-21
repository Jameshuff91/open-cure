#!/usr/bin/env python3
"""
Ensemble Drug Repurposing Scorer
================================

Combines TransE/RotatE knowledge graph embeddings with RGCN graph neural network
predictions for more robust drug repurposing candidates.

Approach:
1. TransE captures global graph structure and relation patterns
2. RGCN captures local neighborhood information via message passing
3. Ensemble combines both for higher confidence predictions

Tiered Output:
- Tier 1: Both models agree (top candidates)
- Tier 2: Strong agreement from one model
- Tier 3: Moderate signal from either model
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


class EnsembleScorer:
    """Combines TransE and RGCN for drug repurposing predictions."""

    def __init__(
        self,
        transe_path: Optional[Path] = None,
        rgcn_path: Optional[Path] = None,
        use_cleaned_data: bool = True,
    ):
        self.device = self._get_device()
        self.use_cleaned_data = use_cleaned_data

        # Model paths
        self.transe_path = transe_path or MODELS_DIR / "transe.pt"
        self.rgcn_path = rgcn_path or MODELS_DIR / "drug_repurposing_rgcn.pt"

        # Data structures
        self.entity2id: dict[str, int] = {}
        self.id2entity: dict[int, str] = {}
        self.drug_ids: set[int] = set()
        self.disease_ids: set[int] = set()
        self.entity_names: dict[int, str] = {}

        # Models (loaded lazily)
        self.transe_embeddings: Optional[torch.Tensor] = None
        self.transe_relation_embs: Optional[torch.Tensor] = None
        self.transe_indication_rel_ids: list[int] = []
        self.rgcn_model = None
        self.rgcn_embeddings: Optional[torch.Tensor] = None
        self.rgcn_decoder = None  # MLP decoder for drug-disease scoring

        # Load entity mappings
        self._load_entities()

    def _get_device(self) -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_entities(self) -> None:
        """Load entity mappings from processed data."""
        logger.info("Loading entity mappings...")

        # Choose data file
        if self.use_cleaned_data:
            nodes_file = PROCESSED_DIR / "unified_nodes_clean.csv"
            if not nodes_file.exists():
                nodes_file = PROCESSED_DIR / "unified_nodes.csv"
        else:
            nodes_file = PROCESSED_DIR / "unified_nodes.csv"

        logger.info(f"Using: {nodes_file.name}")
        nodes_df = pd.read_csv(nodes_file, low_memory=False)

        for idx, row in tqdm(nodes_df.iterrows(), total=len(nodes_df), desc="Loading entities"):
            entity_id = row["id"]
            self.entity2id[entity_id] = idx
            self.id2entity[idx] = entity_id

            # Get name (use resolved name if available, otherwise ID)
            name = str(row.get("name", entity_id))
            if "::" in name and name == entity_id:
                # Still has ID-style name, try to extract meaningful part
                parts = name.split("::")
                name = parts[-1] if len(parts) > 1 else name
            self.entity_names[idx] = name

            # Track drugs and diseases
            node_type = str(row.get("type", "")).lower()
            if "drug" in node_type or "compound" in node_type:
                self.drug_ids.add(idx)
            elif "disease" in node_type:
                self.disease_ids.add(idx)

        logger.info(f"Loaded {len(self.entity2id):,} entities")
        logger.info(f"  Drugs: {len(self.drug_ids):,}")
        logger.info(f"  Diseases: {len(self.disease_ids):,}")

    def load_transe(self) -> None:
        """Load TransE embeddings and relation embeddings."""
        if not self.transe_path.exists():
            logger.warning(f"TransE model not found: {self.transe_path}")
            return

        logger.info(f"Loading TransE from {self.transe_path}")
        checkpoint = torch.load(self.transe_path, map_location=self.device)

        # Extract entity embeddings
        if "entity_embeddings" in checkpoint:
            self.transe_embeddings = checkpoint["entity_embeddings"]
        elif "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
            # Try common key patterns
            for key in ["entity_embeddings.weight", "ent_embeddings.weight", "embeddings.weight"]:
                if key in state:
                    self.transe_embeddings = state[key]
                    break

        # Extract relation embeddings for proper TransE scoring
        if "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
            if "relation_embeddings.weight" in state:
                self.transe_relation_embs = state["relation_embeddings.weight"]
                logger.info(f"TransE relation embeddings shape: {self.transe_relation_embs.shape}")

        # Find indication-related relation IDs
        relation2id = checkpoint.get("relation2id", {})
        indication_keywords = ["treats", "indication", "therapeutic", "palliates"]
        for rel_name, rel_id in relation2id.items():
            rel_lower = rel_name.lower()
            if any(kw in rel_lower for kw in indication_keywords):
                if "contraind" not in rel_lower:  # Exclude contraindication
                    self.transe_indication_rel_ids.append(rel_id)
                    logger.info(f"  Indication relation: {rel_name} (ID: {rel_id})")

        if self.transe_embeddings is not None:
            logger.info(f"TransE embeddings shape: {self.transe_embeddings.shape}")
        else:
            logger.warning("Could not extract TransE embeddings")

    def load_rgcn(self) -> None:
        """Load RGCN model embeddings and decoder."""
        if not self.rgcn_path.exists():
            logger.warning(f"RGCN model not found: {self.rgcn_path}")
            return

        logger.info(f"Loading RGCN from {self.rgcn_path}")
        checkpoint = torch.load(self.rgcn_path, map_location=self.device)

        # Extract embeddings (the trained entity representations)
        if "entity_embeddings" in checkpoint:
            self.rgcn_embeddings = checkpoint["entity_embeddings"]
        elif "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
            for key in ["entity_embeddings.weight", "embeddings.weight"]:
                if key in state:
                    self.rgcn_embeddings = state[key]
                    break

        # Try more key patterns for RGCN
        if self.rgcn_embeddings is None and "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
            for key in ["encoder.entity_embeddings.weight", "entity_embeddings.weight", "embeddings.weight"]:
                if key in state:
                    self.rgcn_embeddings = state[key]
                    break

        # Load the decoder MLP for drug-disease scoring
        if "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
            if "decoder.0.weight" in state:
                # Reconstruct decoder: Linear(256,128) -> ReLU -> Dropout -> Linear(128,64) -> ReLU -> Dropout -> Linear(64,1)
                import torch.nn as nn
                self.rgcn_decoder = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )
                # Load weights
                self.rgcn_decoder[0].weight.data = state["decoder.0.weight"]
                self.rgcn_decoder[0].bias.data = state["decoder.0.bias"]
                self.rgcn_decoder[3].weight.data = state["decoder.3.weight"]
                self.rgcn_decoder[3].bias.data = state["decoder.3.bias"]
                self.rgcn_decoder[6].weight.data = state["decoder.6.weight"]
                self.rgcn_decoder[6].bias.data = state["decoder.6.bias"]
                self.rgcn_decoder = self.rgcn_decoder.to(self.device)
                self.rgcn_decoder.eval()
                logger.info("RGCN decoder loaded (MLP for drug-disease scoring)")

        if self.rgcn_embeddings is not None:
            logger.info(f"RGCN embeddings shape: {self.rgcn_embeddings.shape}")
        else:
            logger.warning("Could not extract RGCN embeddings")

    def _score_transe(self, drug_idx: int, disease_idx: int) -> float:
        """Score drug-disease pair using proper TransE scoring: score = -||h + r - t||."""
        if self.transe_embeddings is None:
            return 0.0

        try:
            drug_emb = self.transe_embeddings[drug_idx]  # head (drug)
            disease_emb = self.transe_embeddings[disease_idx]  # tail (disease)

            # Use relation embedding if available, otherwise just use distance
            if self.transe_relation_embs is not None and self.transe_indication_rel_ids:
                # Average score across all indication relations
                scores = []
                for rel_id in self.transe_indication_rel_ids:
                    rel_emb = self.transe_relation_embs[rel_id]
                    # TransE: h + r should be close to t
                    # Score = -||h + r - t|| (higher is better when negated)
                    distance = torch.norm(drug_emb + rel_emb - disease_emb, p=2).item()
                    # Convert distance to score: lower distance = higher score
                    score = 1.0 / (1.0 + distance)
                    scores.append(score)
                return max(scores)  # Best matching relation
            else:
                # Fallback: just use direct distance
                distance = torch.norm(drug_emb - disease_emb, p=2).item()
                return 1.0 / (1.0 + distance)
        except (IndexError, RuntimeError):
            return 0.0

    def _score_rgcn(self, drug_idx: int, disease_idx: int) -> float:
        """Score drug-disease pair using RGCN decoder or embedding similarity."""
        if self.rgcn_embeddings is None:
            return 0.0

        try:
            drug_emb = self.rgcn_embeddings[drug_idx]
            disease_emb = self.rgcn_embeddings[disease_idx]

            # Use trained decoder if available
            if self.rgcn_decoder is not None:
                with torch.no_grad():
                    # Concatenate drug and disease embeddings (as the decoder expects)
                    pair_emb = torch.cat([drug_emb, disease_emb], dim=0).unsqueeze(0)
                    score = self.rgcn_decoder(pair_emb).item()
                    return score
            else:
                # Fallback: cosine similarity
                sim = torch.cosine_similarity(
                    drug_emb.unsqueeze(0), disease_emb.unsqueeze(0)
                ).item()
                return (sim + 1) / 2
        except (IndexError, RuntimeError):
            return 0.0

    def score_pair(self, drug_id: str, disease_id: str) -> dict:
        """Score a single drug-disease pair."""
        drug_idx = self.entity2id.get(drug_id)
        disease_idx = self.entity2id.get(disease_id)

        if drug_idx is None or disease_idx is None:
            return {"error": "Entity not found"}

        transe_score = self._score_transe(drug_idx, disease_idx)
        rgcn_score = self._score_rgcn(drug_idx, disease_idx)

        # Ensemble score (average)
        ensemble_score = (transe_score + rgcn_score) / 2

        # Determine tier based on agreement
        if transe_score > 0.6 and rgcn_score > 0.6:
            tier = 1  # Both models agree strongly
        elif transe_score > 0.5 or rgcn_score > 0.5:
            tier = 2  # At least one model shows signal
        else:
            tier = 3  # Weak signal

        return {
            "drug_id": drug_id,
            "disease_id": disease_id,
            "drug_name": self.entity_names.get(drug_idx, drug_id),
            "disease_name": self.entity_names.get(disease_idx, disease_id),
            "transe_score": transe_score,
            "rgcn_score": rgcn_score,
            "ensemble_score": ensemble_score,
            "tier": tier,
        }

    def find_drugs_for_disease(
        self,
        disease_id: str,
        top_k: int = 50,
        min_tier: int = 3,
    ) -> pd.DataFrame:
        """Find top drug candidates for a disease."""
        disease_idx = self.entity2id.get(disease_id)
        if disease_idx is None:
            logger.error(f"Disease not found: {disease_id}")
            return pd.DataFrame()

        disease_name = self.entity_names.get(disease_idx, disease_id)
        logger.info(f"Finding drugs for: {disease_name}")

        results = []
        for drug_idx in tqdm(self.drug_ids, desc="Scoring drugs"):
            drug_id = self.id2entity[drug_idx]

            transe_score = self._score_transe(drug_idx, disease_idx)
            rgcn_score = self._score_rgcn(drug_idx, disease_idx)
            ensemble_score = (transe_score + rgcn_score) / 2

            # Determine tier
            if transe_score > 0.6 and rgcn_score > 0.6:
                tier = 1
            elif transe_score > 0.5 or rgcn_score > 0.5:
                tier = 2
            else:
                tier = 3

            if tier <= min_tier:
                results.append({
                    "drug_id": drug_id,
                    "drug_name": self.entity_names.get(drug_idx, drug_id),
                    "transe_score": transe_score,
                    "rgcn_score": rgcn_score,
                    "ensemble_score": ensemble_score,
                    "tier": tier,
                    "agreement": "Both" if tier == 1 else ("Single" if tier == 2 else "Weak"),
                })

        df = pd.DataFrame(results)
        if len(df) > 0:
            df = df.sort_values("ensemble_score", ascending=False).head(top_k)

        return df

    def find_diseases_for_drug(
        self,
        drug_id: str,
        top_k: int = 50,
        min_tier: int = 3,
    ) -> pd.DataFrame:
        """Find potential diseases a drug might treat."""
        drug_idx = self.entity2id.get(drug_id)
        if drug_idx is None:
            logger.error(f"Drug not found: {drug_id}")
            return pd.DataFrame()

        drug_name = self.entity_names.get(drug_idx, drug_id)
        logger.info(f"Finding indications for: {drug_name}")

        results = []
        for disease_idx in tqdm(self.disease_ids, desc="Scoring diseases"):
            disease_id = self.id2entity[disease_idx]

            transe_score = self._score_transe(drug_idx, disease_idx)
            rgcn_score = self._score_rgcn(drug_idx, disease_idx)
            ensemble_score = (transe_score + rgcn_score) / 2

            # Determine tier
            if transe_score > 0.6 and rgcn_score > 0.6:
                tier = 1
            elif transe_score > 0.5 or rgcn_score > 0.5:
                tier = 2
            else:
                tier = 3

            if tier <= min_tier:
                results.append({
                    "disease_id": disease_id,
                    "disease_name": self.entity_names.get(disease_idx, disease_id),
                    "transe_score": transe_score,
                    "rgcn_score": rgcn_score,
                    "ensemble_score": ensemble_score,
                    "tier": tier,
                    "agreement": "Both" if tier == 1 else ("Single" if tier == 2 else "Weak"),
                })

        df = pd.DataFrame(results)
        if len(df) > 0:
            df = df.sort_values("ensemble_score", ascending=False).head(top_k)

        return df

    def search_entity(self, query: str, entity_type: str = "all") -> list[dict]:
        """Search for entities by name (fuzzy matching)."""
        query_lower = query.lower()
        results = []

        for idx, name in self.entity_names.items():
            if query_lower in name.lower():
                entity_id = self.id2entity[idx]
                is_drug = idx in self.drug_ids
                is_disease = idx in self.disease_ids

                if entity_type == "drug" and not is_drug:
                    continue
                if entity_type == "disease" and not is_disease:
                    continue

                results.append({
                    "id": entity_id,
                    "name": name,
                    "type": "Drug" if is_drug else ("Disease" if is_disease else "Other"),
                })

        return results[:20]  # Limit results

    def load_known_edges(self) -> None:
        """Load known drug-disease edges from knowledge graph for validation."""
        logger.info("Loading known drug-disease edges...")

        # Use cleaned or original edges
        if self.use_cleaned_data:
            edges_file = PROCESSED_DIR / "unified_edges_clean.csv"
            if not edges_file.exists():
                edges_file = PROCESSED_DIR / "unified_edges.csv"
        else:
            edges_file = PROCESSED_DIR / "unified_edges.csv"

        # Known positive relations
        indication_relations = {
            "INDICATED_FOR", "TREATS", "indication", "treats",
            "DRUGBANK::treats::Compound:Disease",
            "Hetionet::CtD::Compound:Disease",
            "Hetionet::CpD::Compound:Disease",
        }

        self.known_indications: set[tuple[int, int]] = set()

        edges_df = pd.read_csv(edges_file, low_memory=False)
        for _, row in tqdm(edges_df.iterrows(), total=len(edges_df), desc="Loading edges"):
            rel = str(row.get("relation", ""))
            if rel in indication_relations:
                src = row.get("source", "")
                tgt = row.get("target", "")
                src_idx = self.entity2id.get(src)
                tgt_idx = self.entity2id.get(tgt)
                if src_idx is not None and tgt_idx is not None:
                    if src_idx in self.drug_ids and tgt_idx in self.disease_ids:
                        self.known_indications.add((src_idx, tgt_idx))

        logger.info(f"Loaded {len(self.known_indications):,} known drug-disease indications")

    def validate_predictions(self, top_k: int = 100) -> dict:
        """Validate ensemble predictions against known drug-disease pairs."""
        if not hasattr(self, 'known_indications'):
            self.load_known_edges()

        logger.info("Validating predictions against known indications...")

        # Score all known indications
        true_scores = []
        for drug_idx, disease_idx in tqdm(list(self.known_indications)[:1000], desc="Scoring known"):
            transe_score = self._score_transe(drug_idx, disease_idx)
            rgcn_score = self._score_rgcn(drug_idx, disease_idx)
            ensemble_score = (transe_score + rgcn_score) / 2
            true_scores.append(ensemble_score)

        # Score random negative pairs for comparison
        random_scores = []
        drug_list = list(self.drug_ids)
        disease_list = list(self.disease_ids)
        for _ in range(min(1000, len(true_scores))):
            import random
            drug_idx = random.choice(drug_list)
            disease_idx = random.choice(disease_list)
            if (drug_idx, disease_idx) not in self.known_indications:
                transe_score = self._score_transe(drug_idx, disease_idx)
                rgcn_score = self._score_rgcn(drug_idx, disease_idx)
                ensemble_score = (transe_score + rgcn_score) / 2
                random_scores.append(ensemble_score)

        # Calculate metrics
        true_mean = np.mean(true_scores)
        random_mean = np.mean(random_scores)

        # AUC approximation
        from sklearn.metrics import roc_auc_score
        y_true = [1] * len(true_scores) + [0] * len(random_scores)
        y_scores = true_scores + random_scores
        auc = roc_auc_score(y_true, y_scores)

        results = {
            "known_mean_score": true_mean,
            "random_mean_score": random_mean,
            "lift": true_mean / random_mean if random_mean > 0 else float('inf'),
            "auc": auc,
            "num_known": len(true_scores),
            "num_random": len(random_scores),
        }

        logger.info(f"Validation Results:")
        logger.info(f"  Known indications mean score: {true_mean:.4f}")
        logger.info(f"  Random pairs mean score: {random_mean:.4f}")
        logger.info(f"  Lift (known/random): {results['lift']:.2f}x")
        logger.info(f"  AUC: {auc:.4f}")

        return results

    def get_drug_similar_drugs(self, drug_id: str, top_k: int = 10) -> list[dict]:
        """Find drugs similar to the given drug (for mechanism insight)."""
        drug_idx = self.entity2id.get(drug_id)
        if drug_idx is None:
            return []

        if self.transe_embeddings is None:
            return []

        drug_emb = self.transe_embeddings[drug_idx]

        similarities = []
        for other_idx in self.drug_ids:
            if other_idx == drug_idx:
                continue
            other_emb = self.transe_embeddings[other_idx]
            sim = torch.cosine_similarity(
                drug_emb.unsqueeze(0), other_emb.unsqueeze(0)
            ).item()
            similarities.append((other_idx, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, sim in similarities[:top_k]:
            results.append({
                "drug_id": self.id2entity[idx],
                "drug_name": self.entity_names.get(idx, ""),
                "similarity": sim,
            })

        return results


def main():
    """Demo: Find drug candidates for a disease."""
    import argparse

    parser = argparse.ArgumentParser(description="Ensemble Drug Repurposing Scorer")
    parser.add_argument("--disease", type=str, help="Disease ID or search term")
    parser.add_argument("--drug", type=str, help="Drug ID or search term")
    parser.add_argument("--search", type=str, help="Search for entity by name")
    parser.add_argument("--validate", action="store_true", help="Validate against known edges")
    parser.add_argument("--similar-drugs", type=str, help="Find drugs similar to given drug")
    parser.add_argument("--top-k", type=int, default=30, help="Number of results")
    args = parser.parse_args()

    # Initialize scorer
    scorer = EnsembleScorer(use_cleaned_data=True)

    # Load models
    logger.info("Loading models...")
    scorer.load_transe()
    scorer.load_rgcn()

    if args.search:
        print(f"\nSearching for: {args.search}")
        results = scorer.search_entity(args.search)
        for r in results:
            print(f"  [{r['type']}] {r['id']}: {r['name']}")

    elif args.disease:
        # Search for disease if not an ID
        if not args.disease.startswith("drkg:") and not args.disease.startswith("hetionet:"):
            results = scorer.search_entity(args.disease, "disease")
            if results:
                disease_id = results[0]["id"]
                print(f"Found disease: {results[0]['name']} ({disease_id})")
            else:
                print(f"Disease not found: {args.disease}")
                return
        else:
            disease_id = args.disease

        # Find drugs
        df = scorer.find_drugs_for_disease(disease_id, top_k=args.top_k)
        if len(df) > 0:
            print(f"\nTop {len(df)} drug candidates:")
            print("=" * 80)
            for tier in [1, 2, 3]:
                tier_df = df[df["tier"] == tier]
                if len(tier_df) > 0:
                    print(f"\n--- Tier {tier} ({len(tier_df)} drugs) ---")
                    for _, row in tier_df.iterrows():
                        print(f"  {row['drug_name'][:40]:<40} "
                              f"T:{row['transe_score']:.3f} R:{row['rgcn_score']:.3f} "
                              f"E:{row['ensemble_score']:.3f}")

    elif args.drug:
        # Search for drug if not an ID
        if not args.drug.startswith("drkg:") and not args.drug.startswith("hetionet:"):
            results = scorer.search_entity(args.drug, "drug")
            if results:
                drug_id = results[0]["id"]
                print(f"Found drug: {results[0]['name']} ({drug_id})")
            else:
                print(f"Drug not found: {args.drug}")
                return
        else:
            drug_id = args.drug

        # Find diseases
        df = scorer.find_diseases_for_drug(drug_id, top_k=args.top_k)
        if len(df) > 0:
            print(f"\nTop {len(df)} potential indications:")
            print("=" * 80)
            for tier in [1, 2, 3]:
                tier_df = df[df["tier"] == tier]
                if len(tier_df) > 0:
                    print(f"\n--- Tier {tier} ({len(tier_df)} diseases) ---")
                    for _, row in tier_df.iterrows():
                        print(f"  {row['disease_name'][:40]:<40} "
                              f"T:{row['transe_score']:.3f} R:{row['rgcn_score']:.3f} "
                              f"E:{row['ensemble_score']:.3f}")

    elif args.validate:
        print("\nValidating ensemble against known drug-disease pairs...")
        results = scorer.validate_predictions()
        print(f"\n{'='*60}")
        print(f"Validation Summary:")
        print(f"  AUC: {results['auc']:.4f}")
        print(f"  Known indications mean: {results['known_mean_score']:.4f}")
        print(f"  Random pairs mean: {results['random_mean_score']:.4f}")
        print(f"  Lift: {results['lift']:.2f}x")
        print(f"{'='*60}")

    elif args.similar_drugs:
        # Search for drug if not an ID
        if not args.similar_drugs.startswith("drkg:") and not args.similar_drugs.startswith("hetionet:"):
            results = scorer.search_entity(args.similar_drugs, "drug")
            if results:
                drug_id = results[0]["id"]
                print(f"Found drug: {results[0]['name']} ({drug_id})")
            else:
                print(f"Drug not found: {args.similar_drugs}")
                return
        else:
            drug_id = args.similar_drugs

        # Find similar drugs
        similar = scorer.get_drug_similar_drugs(drug_id, top_k=args.top_k)
        if similar:
            print(f"\nDrugs similar to {scorer.entity_names.get(scorer.entity2id.get(drug_id), drug_id)}:")
            print("=" * 60)
            for drug in similar:
                print(f"  {drug['drug_name'][:40]:<40} Similarity: {drug['similarity']:.4f}")

    else:
        print("Usage:")
        print("  python ensemble_scorer.py --disease 'eczema'")
        print("  python ensemble_scorer.py --drug 'metformin'")
        print("  python ensemble_scorer.py --search 'diabetes'")
        print("  python ensemble_scorer.py --validate")
        print("  python ensemble_scorer.py --similar-drugs 'aspirin'")


if __name__ == "__main__":
    main()
