# core/metrics/evaluator.py
import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import time
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import scipy.stats as stats

from config.constants import MetricCategory, ThreatLevel
from core.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics container."""

    # Core classification metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float

    # Detailed metrics
    confusion_matrix: np.ndarray
    classification_report: Dict[str, Any]

    # Confidence intervals
    accuracy_ci: Tuple[float, float]
    f1_ci: Tuple[float, float]

    # Performance metrics
    inference_latency_ms: float
    throughput: float
    memory_usage_mb: float

    # Threat detection
    threats_detected: List[str]
    threat_level: ThreatLevel

    # Feature analysis
    feature_importance: Dict[str, float]

    # Metadata
    n_samples: int
    timestamp: float


class MetricsEvaluator:
    """Evaluator for model performance and threat detection."""

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level

    def evaluate(
            self,
            model,
            X_test: torch.Tensor,
            y_test: torch.Tensor,
            parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Comprehensive model evaluation."""

        start_time = time.time()

        # 1. Move data to appropriate device
        X_test = X_test.to(model.device)
        # We keep y_test on CPU for sklearn later
        y_true = y_test.cpu().numpy().flatten()

        # 2. Generate predictions
        with torch.no_grad():
            model.eval()

            inference_start = time.time()
            # Use predict_proba which returns a NumPy array in your BaseModel
            y_pred_proba = model.predict_proba(X_test).flatten()
            inference_end = time.time()

            # 3. Convert to binary predictions (Now safe because it's NumPy)
            threshold = parameters.get('threshold', 0.5)
            y_pred = (y_pred_proba > threshold).astype(int)

        # 4. Calculate core metrics (using the NumPy arrays y_true and y_pred)
        accuracy = float(accuracy_score(y_true, y_pred))
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        # ROC AUC (handle single class case)
        try:
            roc_auc = roc_auc_score(y_true, y_pred_proba)
        except:
            roc_auc = 0.5  # Default for single class

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Classification report
        report = classification_report(
            y_true, y_pred,
            output_dict=True,
            zero_division=0
        )

        # Calculate confidence intervals
        accuracy_ci = self._calculate_confidence_interval(
            accuracy, len(y_true), self.confidence_level
        )

        f1_ci = self._calculate_confidence_interval(
            f1, len(y_true), self.confidence_level
        )

        # Calculate performance metrics
        inference_latency = (inference_end - inference_start) * 1000  # ms
        throughput = len(y_true) / (inference_end - inference_start)

        # Estimate memory usage
        memory_usage = self._estimate_memory_usage(model, X_test)

        # Detect threats
        threats_detected = self._detect_threats(
            accuracy, precision, recall, parameters
        )

        # Determine threat level
        threat_level = self._determine_threat_level(threats_detected, accuracy)

        # Feature importance
        feature_importance = model.get_feature_importance(X_test[:100])  # Sample

        # Compile results
        metrics = EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,

            confusion_matrix=cm,
            classification_report=report,

            accuracy_ci=accuracy_ci,
            f1_ci=f1_ci,

            inference_latency_ms=inference_latency,
            throughput=throughput,
            memory_usage_mb=memory_usage,

            threats_detected=threats_detected,
            threat_level=threat_level,

            feature_importance=feature_importance,

            n_samples=len(y_true),
            timestamp=start_time
        )

        logger.info(f"Evaluation completed in {time.time() - start_time:.2f}s")
        return self._metrics_to_dict(metrics)

    def _calculate_confidence_interval(
            self,
            proportion: float,
            n: int,
            confidence_level: float
    ) -> Tuple[float, float]:
        """Calculate Wilson score interval for proportion."""
        if n == 0:
            return (0.0, 0.0)

        z = stats.norm.ppf(1 - (1 - confidence_level) / 2)

        denominator = 1 + z ** 2 / n
        centre_adjusted_proportion = proportion + z ** 2 / (2 * n)
        adjusted_standard_deviation = np.sqrt(
            (proportion * (1 - proportion) + z ** 2 / (4 * n)) / n
        )

        lower_bound = (
                              centre_adjusted_proportion - z * adjusted_standard_deviation
                      ) / denominator

        upper_bound = (
                              centre_adjusted_proportion + z * adjusted_standard_deviation
                      ) / denominator

        return (max(0.0, lower_bound), min(1.0, upper_bound))

    def _estimate_memory_usage(self, model, X_batch: torch.Tensor) -> float:
        """Estimate memory usage in MB."""
        try:
            # Model parameters memory
            param_memory = sum(
                p.numel() * p.element_size()
                for p in model.parameters()
            ) / (1024 ** 2)  # MB

            # Activation memory (rough estimate)
            with torch.no_grad():
                _ = model(X_batch)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    activation_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
                else:
                    activation_memory = X_batch.numel() * X_batch.element_size() * 10 / (1024 ** 2)

            return param_memory + activation_memory
        except:
            return 0.0

    def _detect_threats(
            self,
            accuracy: float,
            precision: float,
            recall: float,
            parameters: Dict[str, Any]
    ) -> List[str]:
        """Detect potential threats based on metrics."""
        threats = []

        # Accuracy-based threats
        if accuracy < 0.7:
            threats.append("Low accuracy - potential misclassifications")

        if precision < 0.6:
            threats.append("High false positive rate")

        if recall < 0.6:
            threats.append("High false negative rate - threats may be missed")

        # Parameter-based threats
        poison_rate = parameters.get('poison_rate', 0.0)
        if poison_rate > 0.3:
            threats.append(f"High data poisoning rate ({poison_rate:.1%})")

        noise_level = parameters.get('noise_level', 0.0)
        if noise_level > 1.5:
            threats.append(f"High noise level ({noise_level:.1f})")

        return threats

    def _determine_threat_level(
            self,
            threats: List[str],
            accuracy: float
    ) -> ThreatLevel:
        """Determine overall threat level."""
        if accuracy < 0.6 or len(threats) > 3:
            return ThreatLevel.CRITICAL
        elif accuracy < 0.75 or len(threats) > 1:
            return ThreatLevel.HIGH
        elif accuracy < 0.85:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW

    def _metrics_to_dict(self, metrics: EvaluationMetrics) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            'accuracy': metrics.accuracy,
            'precision': metrics.precision,
            'recall': metrics.recall,
            'f1_score': metrics.f1_score,
            'roc_auc': metrics.roc_auc,

            'confusion_matrix': metrics.confusion_matrix.tolist(),
            'classification_report': metrics.classification_report,

            'confidence_intervals': {
                'accuracy': metrics.accuracy_ci,
                'f1_score': metrics.f1_ci
            },

            'performance': {
                'inference_latency_ms': metrics.inference_latency_ms,
                'throughput': metrics.throughput,
                'memory_usage_mb': metrics.memory_usage_mb
            },

            'threats_detected': metrics.threats_detected,
            'threat_level': {
                'level': metrics.threat_level.name,
                'icon': metrics.threat_level.icon,
                'description': metrics.threat_level.description
            },

            'feature_importance': metrics.feature_importance,

            'metadata': {
                'n_samples': metrics.n_samples,
                'timestamp': metrics.timestamp
            }
        }