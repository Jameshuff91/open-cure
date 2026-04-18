"""h907: Leakage-safe loader for expert-reviewed predictions.

Expert labels (e.g., Ryland Mortlock's blinded review at
data/reference/expert_labels_ryland.json) live OUTSIDE the DRKG/GT pipeline.
They are never merged into predictor.ground_truth or expanded_ground_truth.json
because doing so would contaminate drug_train_freq / drug_to_diseases / kNN
train embeddings — the exact structures used to produce the predictions Ryland
is judging.

Use ExpertLabels only for evaluation-side precision scoring. See
scripts/h907_eval_expert_labels.py for the parallel (DRKG-GT vs expert) split.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
REF_DIR = REPO_ROOT / "data" / "reference"
DEFAULT_RYLAND_PATH = REF_DIR / "expert_labels_ryland.json"

HIT_VERDICTS: FrozenSet[str] = frozenset({"plausible", "known"})
MISS_VERDICTS: FrozenSet[str] = frozenset({"implausible", "adverse"})
SKIP_VERDICTS: FrozenSet[str] = frozenset({"unsure"})


@dataclass(frozen=True)
class ExpertLabel:
    prediction_id: str
    drug_id: Optional[str]
    disease_id: Optional[str]
    verdict: str
    reviewer_confidence: Optional[int]
    reasoning: Optional[str]
    provenance: str


class ExpertLabels:
    """Lookup helper for (disease_id, drug_id) -> ExpertLabel."""

    def __init__(self, records: Iterable[ExpertLabel], source: str):
        self._by_pair: Dict[tuple, ExpertLabel] = {}
        self._by_pred_id: Dict[str, ExpertLabel] = {}
        self.source = source
        for rec in records:
            self._by_pred_id[rec.prediction_id] = rec
            if rec.disease_id and rec.drug_id:
                self._by_pair[(rec.disease_id, rec.drug_id)] = rec

    def __len__(self) -> int:
        return len(self._by_pred_id)

    def get(self, disease_id: str, drug_id: str) -> Optional[ExpertLabel]:
        return self._by_pair.get((disease_id, drug_id))

    def is_hit(
        self,
        disease_id: str,
        drug_id: str,
        min_confidence: int = 3,
    ) -> Optional[bool]:
        """Return True/False/None (None = skip from precision).

        None covers: no label, verdict == 'unsure', or reviewer_confidence
        below min_confidence. Callers must exclude these from the denominator.
        """
        rec = self._by_pair.get((disease_id, drug_id))
        if rec is None or rec.verdict in SKIP_VERDICTS:
            return None
        if rec.reviewer_confidence is not None and rec.reviewer_confidence < min_confidence:
            return None
        if rec.verdict in HIT_VERDICTS:
            return True
        if rec.verdict in MISS_VERDICTS:
            return False
        return None


def load_expert_labels(path: Path = DEFAULT_RYLAND_PATH) -> ExpertLabels:
    """Load expert labels from JSON produced by scripts/import_ryland_review.py.

    Missing file returns an empty ExpertLabels so downstream code degrades
    gracefully while we wait for the review to arrive.
    """
    if not path.exists():
        return ExpertLabels([], source=str(path))

    with open(path) as f:
        raw = json.load(f)

    records = []
    for pred_id, entry in raw.items():
        if entry.get("provenance") != "expert_ryland":
            # Defensive: never load records that were not tagged properly.
            continue
        records.append(ExpertLabel(
            prediction_id=pred_id,
            drug_id=entry.get("drug_id"),
            disease_id=entry.get("disease_id"),
            verdict=entry["verdict"],
            reviewer_confidence=entry.get("reviewer_confidence"),
            reasoning=entry.get("reasoning"),
            provenance=entry["provenance"],
        ))
    return ExpertLabels(records, source=str(path))
