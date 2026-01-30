import numpy as np
import logging
from typing import Protocol, Dict, Any
from sklearn.ensemble import IsolationForest
from core.utils.exceptions import SimulationError

logger = logging.getLogger(__name__)


class DetectionStrategy(Protocol):
    """Protocol for threat detection strategies."""

    def detect(self, data: np.ndarray) -> np.ndarray: ...


class AnomalyDetectionStrategy:
    """Uses Isolation Forest for statistical outlier detection."""

    def __init__(self, contamination: float = 0.1):
        self.model = IsolationForest(contamination=contamination, random_state=42)

    def detect(self, data: np.ndarray) -> np.ndarray:
        if data.size == 0:
            raise SimulationError("Empty data provided for detection.")

        reshaped_data = data.reshape(-1, 1) if data.ndim == 1 else data
        # Returns -1 for outliers, 1 for inliers
        return self.model.fit_predict(reshaped_data)


class ThreatDetector:
    """High-level threat detection service."""

    def __init__(self, strategy: DetectionStrategy):
        self._strategy = strategy

    def run_assessment(self, data: np.ndarray) -> Dict[str, Any]:
        logger.info("Running adversarial threat assessment...")
        predictions = self._strategy.detect(data)

        poisoning_detected = np.any(predictions == -1)
        anomaly_ratio = np.count_nonzero(predictions == -1) / len(predictions)

        return {
            "is_compromised": poisoning_detected,
            "threat_level": "HIGH" if anomaly_ratio > 0.2 else "LOW",
            "anomaly_indices": np.where(predictions == -1)[0].tolist(),
            "resilience_score": 1.0 - anomaly_ratio
        }