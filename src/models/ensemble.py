#!/usr/bin/env python3
"""
Ensemble Voting System for Drug Repurposing Predictions.

Combines predictions from multiple models to improve accuracy and reduce
false positives. Key insight: different models capture different signals,
and consensus predictions are more reliable.

Ensemble methods:
1. Simple averaging
2. Weighted voting (based on model confidence/validation performance)
3. Stacking (meta-learner)
4. Rank aggregation (Borda count, reciprocal rank fusion)
"""

import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")


class AggregationMethod(Enum):
    """Methods for aggregating model predictions."""

    MEAN = "mean"
    WEIGHTED_MEAN = "weighted_mean"
    MAX = "max"
    MEDIAN = "median"
    BORDA_COUNT = "borda_count"
    RECIPROCAL_RANK = "reciprocal_rank"
    STACKING = "stacking"


@dataclass
class ModelPrediction:
    """Prediction from a single model."""

    model_name: str
    drug_id: str
    disease_id: str
    score: float
    rank: int
    confidence: float  # Model's self-reported confidence


@dataclass
class EnsemblePrediction:
    """Aggregated prediction from ensemble."""

    drug_id: str
    drug_name: str
    disease_id: str
    disease_name: str
    ensemble_score: float
    ensemble_rank: int
    agreement: float  # What fraction of models agree
    model_scores: dict[str, float]
    method: AggregationMethod


class EnsembleAggregator:
    """
    Aggregate predictions from multiple drug repurposing models.

    Each model may use different approaches:
    - Knowledge graph embeddings (TransE, RotatE, ComplEx)
    - Graph neural networks (GAT, GraphSAGE, GIN)
    - Multi-hop reasoning
    - Similarity-based transfer
    - LLM extraction

    The ensemble combines these signals for more robust predictions.
    """

    def __init__(
        self,
        model_weights: dict[str, float] | None = None,
        default_method: AggregationMethod = AggregationMethod.RECIPROCAL_RANK,
    ):
        """
        Initialize ensemble aggregator.

        Args:
            model_weights: Weights for each model (for weighted aggregation)
            default_method: Default aggregation method
        """
        self.model_weights = model_weights or {}
        self.default_method = default_method
        self.stacking_model: nn.Module | None = None

    def aggregate(
        self,
        predictions: dict[str, list[ModelPrediction]],
        method: AggregationMethod | None = None,
        top_k: int = 100,
    ) -> list[EnsemblePrediction]:
        """
        Aggregate predictions from multiple models.

        Args:
            predictions: Dict mapping model_name -> list of predictions
            method: Aggregation method (defaults to self.default_method)
            top_k: Number of top predictions to return

        Returns:
            Sorted list of ensemble predictions
        """
        method = method or self.default_method

        # Group predictions by (drug_id, disease_id)
        grouped: dict[tuple[str, str], dict[str, ModelPrediction]] = {}

        for model_name, model_preds in predictions.items():
            for pred in model_preds:
                key = (pred.drug_id, pred.disease_id)
                if key not in grouped:
                    grouped[key] = {}
                grouped[key][model_name] = pred

        # Aggregate each drug-disease pair
        ensemble_predictions = []
        num_models = len(predictions)

        for (drug_id, disease_id), model_preds in grouped.items():
            if method == AggregationMethod.MEAN:
                score = self._aggregate_mean(model_preds)
            elif method == AggregationMethod.WEIGHTED_MEAN:
                score = self._aggregate_weighted_mean(model_preds)
            elif method == AggregationMethod.MAX:
                score = self._aggregate_max(model_preds)
            elif method == AggregationMethod.MEDIAN:
                score = self._aggregate_median(model_preds)
            elif method == AggregationMethod.BORDA_COUNT:
                score = self._aggregate_borda(model_preds, predictions)
            elif method == AggregationMethod.RECIPROCAL_RANK:
                score = self._aggregate_reciprocal_rank(model_preds)
            elif method == AggregationMethod.STACKING:
                score = self._aggregate_stacking(model_preds)
            else:
                score = self._aggregate_mean(model_preds)

            # Calculate agreement (fraction of models that ranked this in top 100)
            agreement = len(model_preds) / num_models

            # Get names (from first prediction that has them)
            drug_name = drug_id
            disease_name = disease_id
            for pred in model_preds.values():
                if hasattr(pred, "drug_name") and pred.drug_name:  # type: ignore[union-attr]
                    drug_name = pred.drug_name  # type: ignore[union-attr]
                if hasattr(pred, "disease_name") and pred.disease_name:  # type: ignore[union-attr]
                    disease_name = pred.disease_name  # type: ignore[union-attr]
                break

            ensemble_predictions.append(
                EnsemblePrediction(
                    drug_id=drug_id,
                    drug_name=drug_name,
                    disease_id=disease_id,
                    disease_name=disease_name,
                    ensemble_score=score,
                    ensemble_rank=0,  # Will be set after sorting
                    agreement=agreement,
                    model_scores={name: pred.score for name, pred in model_preds.items()},
                    method=method,
                )
            )

        # Sort by score and assign ranks
        ensemble_predictions.sort(key=lambda x: x.ensemble_score, reverse=True)
        for i, pred in enumerate(ensemble_predictions):
            pred.ensemble_rank = i + 1

        return ensemble_predictions[:top_k]

    def _aggregate_mean(self, model_preds: dict[str, ModelPrediction]) -> float:
        """Simple mean of scores."""
        scores = [pred.score for pred in model_preds.values()]
        return float(np.mean(scores))

    def _aggregate_weighted_mean(
        self, model_preds: dict[str, ModelPrediction]
    ) -> float:
        """Weighted mean using model weights."""
        total_weight = 0.0
        weighted_sum = 0.0

        for model_name, pred in model_preds.items():
            weight = self.model_weights.get(model_name, 1.0)
            weighted_sum += weight * pred.score
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _aggregate_max(self, model_preds: dict[str, ModelPrediction]) -> float:
        """Maximum score across models."""
        return max(pred.score for pred in model_preds.values())

    def _aggregate_median(self, model_preds: dict[str, ModelPrediction]) -> float:
        """Median score across models."""
        scores = [pred.score for pred in model_preds.values()]
        return float(np.median(scores))

    def _aggregate_borda(
        self,
        model_preds: dict[str, ModelPrediction],
        all_predictions: dict[str, list[ModelPrediction]],
    ) -> float:
        """
        Borda count: sum of (N - rank) across models.

        Higher-ranked items get more points.
        """
        total_points = 0.0

        for model_name, pred in model_preds.items():
            # Get total number of predictions for this model
            n = len(all_predictions.get(model_name, []))
            # Borda points: N - rank + 1
            points = max(0, n - pred.rank + 1)
            total_points += points

        # Normalize by number of models
        return total_points / len(model_preds) if model_preds else 0.0

    def _aggregate_reciprocal_rank(
        self, model_preds: dict[str, ModelPrediction]
    ) -> float:
        """
        Reciprocal Rank Fusion (RRF).

        Score = sum(1 / (k + rank)) where k is a smoothing constant.
        This method is robust to different score scales across models.
        """
        k = 60  # Standard RRF constant
        total = 0.0

        for pred in model_preds.values():
            total += 1.0 / (k + pred.rank)

        return total

    def _aggregate_stacking(self, model_preds: dict[str, ModelPrediction]) -> float:
        """
        Stacking: use a meta-learner to combine model scores.

        Requires training the stacking model first.
        """
        if self.stacking_model is None:
            # Fall back to weighted mean
            return self._aggregate_weighted_mean(model_preds)

        # Prepare feature vector (scores from each model in fixed order)
        model_names = sorted(self.model_weights.keys())
        features = []
        for name in model_names:
            if name in model_preds:
                features.append(model_preds[name].score)
            else:
                features.append(0.0)

        with torch.no_grad():
            x = torch.tensor([features], dtype=torch.float32)
            score = self.stacking_model(x).item()

        return score

    def train_stacking_model(
        self,
        training_data: list[tuple[dict[str, float], float]],
        model_names: list[str],
    ):
        """
        Train a stacking meta-learner.

        Args:
            training_data: List of (model_scores_dict, true_label)
            model_names: Ordered list of model names
        """
        # Simple MLP for stacking
        self.stacking_model = nn.Sequential(
            nn.Linear(len(model_names), 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        # Prepare data
        X = []
        y = []
        for scores_dict, label in training_data:
            features = [scores_dict.get(name, 0.0) for name in model_names]
            X.append(features)
            y.append(label)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        # Train
        optimizer = torch.optim.Adam(self.stacking_model.parameters(), lr=0.001)
        criterion = nn.BCELoss()

        for epoch in range(100):
            self.stacking_model.train()
            optimizer.zero_grad()

            pred = self.stacking_model(X_tensor)
            loss = criterion(pred, y_tensor)

            loss.backward()
            optimizer.step()

        self.stacking_model.eval()


class ConsensusFilter:
    """
    Filter predictions based on model consensus.

    Predictions are more reliable when multiple models agree.
    This class provides various consensus criteria.
    """

    @staticmethod
    def filter_by_agreement(
        predictions: list[EnsemblePrediction],
        min_agreement: float = 0.5,
    ) -> list[EnsemblePrediction]:
        """Keep only predictions where >= min_agreement fraction of models agree."""
        return [p for p in predictions if p.agreement >= min_agreement]

    @staticmethod
    def filter_by_unanimous(
        predictions: list[EnsemblePrediction],
        num_models: int,
    ) -> list[EnsemblePrediction]:
        """Keep only predictions found by all models."""
        return [p for p in predictions if len(p.model_scores) == num_models]

    @staticmethod
    def filter_by_top_in_all(
        predictions: list[EnsemblePrediction],
        model_ranks: dict[str, dict[tuple[str, str], int]],
        max_rank: int = 100,
    ) -> list[EnsemblePrediction]:
        """Keep predictions ranked in top N by all models."""
        filtered = []
        for pred in predictions:
            key = (pred.drug_id, pred.disease_id)
            in_top_all = all(
                model_ranks.get(model, {}).get(key, float("inf")) <= max_rank
                for model in model_ranks
            )
            if in_top_all:
                filtered.append(pred)
        return filtered


class ModelCalibrator:
    """
    Calibrate model scores to be comparable across models.

    Different models may output scores on different scales.
    Calibration normalizes scores to a common range.
    """

    def __init__(self):
        self.calibration_params: dict[str, dict[str, float]] = {}

    def fit(
        self,
        predictions: dict[str, list[ModelPrediction]],
        method: str = "minmax",
    ):
        """
        Fit calibration parameters for each model.

        Args:
            predictions: Dict mapping model_name -> predictions
            method: Calibration method ('minmax', 'zscore', 'isotonic')
        """
        for model_name, preds in predictions.items():
            scores = [p.score for p in preds]

            if method == "minmax":
                min_score = min(scores)
                max_score = max(scores)
                self.calibration_params[model_name] = {
                    "method": "minmax",
                    "min": min_score,
                    "max": max_score,
                }
            elif method == "zscore":
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                self.calibration_params[model_name] = {
                    "method": "zscore",
                    "mean": mean_score,
                    "std": std_score if std_score > 0 else 1.0,
                }

    def transform(
        self, predictions: dict[str, list[ModelPrediction]]
    ) -> dict[str, list[ModelPrediction]]:
        """
        Apply calibration to model predictions.

        Returns new predictions with calibrated scores.
        """
        calibrated = {}

        for model_name, preds in predictions.items():
            if model_name not in self.calibration_params:
                calibrated[model_name] = preds
                continue

            params = self.calibration_params[model_name]
            new_preds = []

            for pred in preds:
                if params["method"] == "minmax":
                    min_s, max_s = params["min"], params["max"]
                    if max_s > min_s:
                        new_score = (pred.score - min_s) / (max_s - min_s)
                    else:
                        new_score = 0.5
                elif params["method"] == "zscore":
                    new_score = (pred.score - params["mean"]) / params["std"]
                    # Convert to 0-1 range using sigmoid
                    new_score = 1 / (1 + np.exp(-new_score))
                else:
                    new_score = pred.score

                new_preds.append(
                    ModelPrediction(
                        model_name=pred.model_name,
                        drug_id=pred.drug_id,
                        disease_id=pred.disease_id,
                        score=new_score,
                        rank=pred.rank,
                        confidence=pred.confidence,
                    )
                )

            calibrated[model_name] = new_preds

        return calibrated


class EnsemblePredictor:
    """
    High-level interface for ensemble drug repurposing predictions.

    Coordinates multiple models, calibration, and aggregation.
    """

    def __init__(self):
        self.models: dict[str, Any] = {}  # model_name -> model instance
        self.aggregator = EnsembleAggregator()
        self.calibrator = ModelCalibrator()

    def register_model(
        self,
        name: str,
        model: Any,
        weight: float = 1.0,
        predict_fn: Callable[[Any, str], list[ModelPrediction]] | None = None,
    ):
        """
        Register a model for the ensemble.

        Args:
            name: Unique name for the model
            model: The model instance
            weight: Weight for weighted aggregation
            predict_fn: Optional custom prediction function
        """
        self.models[name] = {
            "model": model,
            "weight": weight,
            "predict_fn": predict_fn,
        }
        self.aggregator.model_weights[name] = weight

    def predict(
        self,
        disease_id: str,
        method: AggregationMethod = AggregationMethod.RECIPROCAL_RANK,
        top_k: int = 100,
        calibrate: bool = True,
        min_agreement: float = 0.0,
    ) -> list[EnsemblePrediction]:
        """
        Generate ensemble predictions for a disease.

        Args:
            disease_id: Disease to find treatments for
            method: Aggregation method
            top_k: Number of predictions to return
            calibrate: Whether to calibrate scores
            min_agreement: Minimum model agreement required

        Returns:
            Ranked list of ensemble predictions
        """
        # Get predictions from all models
        all_predictions: dict[str, list[ModelPrediction]] = {}

        for name, model_info in self.models.items():
            model = model_info["model"]
            predict_fn = model_info["predict_fn"]

            if predict_fn:
                preds = predict_fn(model, disease_id)
            elif hasattr(model, "predict_all_drugs"):
                preds = model.predict_all_drugs(disease_id, top_k=500)
                # Convert to ModelPrediction format
                preds = [
                    ModelPrediction(
                        model_name=name,
                        drug_id=p.drug_id,
                        disease_id=p.disease_id,
                        score=p.score,
                        rank=p.rank,
                        confidence=getattr(p, "confidence", 0.5),
                    )
                    for p in preds
                ]
            else:
                logger.warning(f"Model {name} has no prediction method")
                continue

            # Assign ranks
            preds.sort(key=lambda x: x.score, reverse=True)
            for i, pred in enumerate(preds):
                pred.rank = i + 1

            all_predictions[name] = preds

        # Calibrate if requested
        if calibrate and all_predictions:
            self.calibrator.fit(all_predictions)
            all_predictions = self.calibrator.transform(all_predictions)

        # Aggregate
        ensemble_preds = self.aggregator.aggregate(
            all_predictions, method=method, top_k=top_k * 2
        )

        # Filter by agreement
        if min_agreement > 0:
            ensemble_preds = ConsensusFilter.filter_by_agreement(
                ensemble_preds, min_agreement
            )

        return ensemble_preds[:top_k]
