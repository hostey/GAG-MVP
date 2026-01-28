# tests/test_metrics.py
import pytest
import numpy as np
import torch

from core.metrics.evaluator import MetricsEvaluator, EvaluationMetrics
from core.models.nn_models import ThreatDetectionModel, ModelConfig


class TestMetricsEvaluator:
    """Test suite for Metrics Evaluator."""

    @pytest.fixture
    def evaluator(self):
        return MetricsEvaluator(confidence_level=0.95)

    @pytest.fixture
    def mock_model(self):
        config = ModelConfig(input_dim=2)
        return ThreatDetectionModel(config)

    @pytest.fixture
    def test_data(self):
        # Create simple test data
        X_test = torch.randn(100, 2)
        y_test = torch.randint(0, 2, (100,)).float()
        return X_test, y_test

    def test_evaluation_metrics_dataclass(self):
        """Test EvaluationMetrics dataclass."""
        metrics = EvaluationMetrics(
            accuracy=0.85,
            precision=0.82,
            recall=0.88,
            f1_score=0.85,
            roc_auc=0.89,
            confusion_matrix=np.array([[45, 5], [8, 42]]),
            classification_report={'precision': 0.82, 'recall': 0.88},
            accuracy_ci=(0.76, 0.91),
            f1_ci=(0.78, 0.90),
            inference_latency_ms=15.5,
            throughput=6451.6,
            memory_usage_mb=25.3,
            threats_detected=['Low accuracy'],
            threat_level="MEDIUM",
            feature_importance={'feature_1': 0.6, 'feature_2': 0.4},
            n_samples=100,
            timestamp=1234567890.0
        )

        assert metrics.accuracy == 0.85
        assert metrics.threat_level == "MEDIUM"
        assert len(metrics.threats_detected) == 1

    def test_confidence_interval_calculation(self, evaluator):
        """Test confidence interval calculation."""
        proportion = 0.85
        n = 100
        ci = evaluator._calculate_confidence_interval(proportion, n, 0.95)

        assert len(ci) == 2
        assert ci[0] < ci[1]
        assert 0 <= ci[0] <= 1
        assert 0 <= ci[1] <= 1

    def test_threat_detection(self, evaluator):
        """Test threat detection logic."""
        threats = evaluator._detect_threats(
            accuracy=0.65,
            precision=0.55,
            recall=0.58,
            parameters={'poison_rate': 0.35}
        )

        assert len(threats) > 0
        assert any("Low accuracy" in t for t in threats)
        assert any("High data poisoning rate" in t for t in threats)

    def test_threat_level_determination(self, evaluator):
        """Test threat level determination."""
        # Test critical level
        threats = ["Threat 1", "Threat 2", "Threat 3", "Threat 4"]
        level = evaluator._determine_threat_level(threats, accuracy=0.55)
        assert level.name == "CRITICAL"

        # Test low level
        threats = []
        level = evaluator._determine_threat_level(threats, accuracy=0.90)
        assert level.name == "LOW"

    def test_memory_usage_estimation(self, evaluator, mock_model):
        """Test memory usage estimation."""
        X_batch = torch.randn(32, 2)

        # This is a rough estimate, so we just check it returns a number
        memory = evaluator._estimate_memory_usage(mock_model, X_batch)
        assert isinstance(memory, float)
        assert memory >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])