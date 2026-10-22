"""Shared lightweight interfaces (no heavy framework dependency)."""
from dataclasses import dataclass
from typing import Protocol
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SampleMetadata:
    sample_id: str
    patient_id: str
    split: str
    label: int | None = None  # 0=normal, 1=anomaly; may be hidden for test


class AnomalyScorer(Protocol):
    """All methods return shape (batch,) with larger = more anomalous."""

    def score_samples(self, images: NDArray[np.float32]) -> NDArray[np.float64]: ...
