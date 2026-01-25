#!/usr/bin/env python3
"""
Confidence calibration model for drug repurposing predictions.

Trains a model to predict: "Given these features, what's the probability
this drug will be in the top-30 for this disease?"

This allows us to:
1. Prioritize high-confidence novel predictions
2. Flag uncertain predictions for human review
3. Know when to trust the model
"""

import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Set, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, precision_recall_curve, auc

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
MODELS_DIR = PROJECT_ROOT / "models"


@dataclass
class PredictionFeatures:
    """Features for a single drug-disease prediction."""
    base_score: float
    target_overlap: int
    atc_score: float
    chemical_sim: float
    pathway_overlap: int

    # Drug type flags
    is_biologic: bool
    is_kinase_inhibitor: bool
    is_antibiotic: bool

    # Disease category flags
    is_cancer: bool
    is_infectious: bool
    is_autoimmune: bool

    # Feature coverage flags
    has_fingerprint: bool
    has_targets: bool
    has_atc: bool

    # Final boosted score
    boosted_score: float

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        return np.array([
            self.base_score,
            self.target_overlap,
            self.atc_score,
            self.chemical_sim,
            self.pathway_overlap,
            float(self.is_biologic),
            float(self.is_kinase_inhibitor),
            float(self.is_antibiotic),
            float(self.is_cancer),
            float(self.is_infectious),
            float(self.is_autoimmune),
            float(self.has_fingerprint),
            float(self.has_targets),
            float(self.has_atc),
            self.boosted_score,
        ])

    @staticmethod
    def feature_names() -> List[str]:
        return [
            'base_score',
            'target_overlap',
            'atc_score',
            'chemical_sim',
            'pathway_overlap',
            'is_biologic',
            'is_kinase_inhibitor',
            'is_antibiotic',
            'is_cancer',
            'is_infectious',
            'is_autoimmune',
            'has_fingerprint',
            'has_targets',
            'has_atc',
            'boosted_score',
        ]


def classify_drug_type(drug_name: str) -> Dict[str, bool]:
    """Classify drug by name pattern."""
    name_lower = drug_name.lower()

    return {
        'is_biologic': (
            name_lower.endswith(('mab', 'umab', 'zumab', 'ximab')) or
            name_lower.endswith(('cept', 'ept')) or
            name_lower.endswith(('lin', 'sulin'))
        ),
        'is_kinase_inhibitor': name_lower.endswith(('ib', 'nib', 'tinib')),
        'is_antibiotic': (
            name_lower.endswith('cycline') or
            name_lower.endswith('cillin') or
            name_lower.endswith('mycin')
        ),
    }


def classify_disease_category(disease_name: str) -> Dict[str, bool]:
    """Classify disease by name pattern."""
    name_lower = disease_name.lower()

    return {
        'is_cancer': any(x in name_lower for x in [
            'cancer', 'carcinoma', 'tumor', 'leukemia', 'lymphoma', 'sarcoma', 'melanoma'
        ]),
        'is_infectious': any(x in name_lower for x in [
            'infection', 'bacterial', 'viral', 'hiv', 'hepatitis'
        ]),
        'is_autoimmune': any(x in name_lower for x in [
            'arthritis', 'lupus', 'autoimmune', 'crohn', 'colitis'
        ]),
    }


class ConfidenceCalibrator:
    """
    Calibrated confidence predictor for drug repurposing.

    Given features about a drug-disease prediction, predicts the probability
    that this drug will appear in the top-30 predictions for the disease.
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.model: Optional[LogisticRegression] = None
        self.feature_importances: Optional[Dict[str, float]] = None

        if model_path and model_path.exists():
            self.load(model_path)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        use_cross_val: bool = True,
    ) -> Dict[str, float]:
        """
        Train the confidence calibration model.

        Args:
            X: Feature matrix (N, 15)
            y: Binary labels (1 = in top-30, 0 = not in top-30)
            use_cross_val: Whether to use cross-validation for calibration

        Returns:
            Dict of evaluation metrics
        """
        # Use logistic regression for calibrated probabilities
        self.model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight='balanced',  # Handle class imbalance
        )

        if use_cross_val:
            # Get cross-validated predictions for calibration evaluation
            cv_probs = cross_val_predict(
                self.model, X, y,
                cv=5,
                method='predict_proba'
            )[:, 1]

            # Fit on full data for deployment
            self.model.fit(X, y)
            probs = cv_probs
        else:
            self.model.fit(X, y)
            probs = self.model.predict_proba(X)[:, 1]

        # Store feature importances (coefficients)
        self.feature_importances = dict(zip(
            PredictionFeatures.feature_names(),
            self.model.coef_[0]
        ))

        # Evaluate
        metrics = self._evaluate(y, probs)

        return metrics

    def _evaluate(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        """Evaluate calibration quality."""
        # Brier score (lower is better)
        brier = brier_score_loss(y_true, y_prob)

        # AUROC
        auroc = roc_auc_score(y_true, y_prob)

        # AUPRC
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        auprc = auc(recall, precision)

        # Calibration curve bins
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)

        # Expected calibration error (ECE)
        ece = np.mean(np.abs(prob_true - prob_pred))

        return {
            'brier_score': brier,
            'auroc': auroc,
            'auprc': auprc,
            'ece': ece,
            'calibration_true': prob_true.tolist(),
            'calibration_pred': prob_pred.tolist(),
        }

    def predict_confidence(self, features: PredictionFeatures) -> float:
        """
        Predict confidence for a single drug-disease pair.

        Returns probability that this drug will be in top-30.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        X = features.to_array().reshape(1, -1)
        return float(self.model.predict_proba(X)[0, 1])

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Predict confidence for batch of predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importances(self) -> Dict[str, float]:
        """Get feature importance ranking."""
        if self.feature_importances is None:
            raise ValueError("Model not trained.")

        # Sort by absolute importance
        return dict(sorted(
            self.feature_importances.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        ))

    def save(self, path: Path) -> None:
        """Save model to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_importances': self.feature_importances,
            }, f)

    def load(self, path: Path) -> None:
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.feature_importances = data['feature_importances']


def interpret_confidence(confidence: float) -> str:
    """Convert confidence score to human-readable category."""
    if confidence >= 0.8:
        return "Very High"
    elif confidence >= 0.6:
        return "High"
    elif confidence >= 0.4:
        return "Medium"
    elif confidence >= 0.2:
        return "Low"
    else:
        return "Very Low"
