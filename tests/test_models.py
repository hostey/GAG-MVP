# tests/test_models.py
import pytest
import torch
import numpy as np

from core.models.nn_models import ThreatDetectionModel, ModelConfig


class TestThreatDetectionModel:
    """Test suite for ThreatDetectionModel."""

    def test_model_initialization(self):
        """Test model initialization with valid config."""
        config = ModelConfig(input_dim=4)
        model = ThreatDetectionModel(config)

        assert model is not None
        assert hasattr(model, 'id')
        assert len(model.id) > 0

    def test_forward_pass(self):
        """Test model forward pass produces correct shape."""
        config = ModelConfig(input_dim=4)
        model = ThreatDetectionModel(config)

        # Create test input
        x = torch.randn(10, 4)
        output = model(x)

        assert output.shape == (10, 1)
        assert torch.all(output >= 0) and torch.all(output <= 1)

    def test_predict_method(self):
        """Test binary prediction method."""
        config = ModelConfig(input_dim=2)
        model = ThreatDetectionModel(config)

        x = torch.randn(5, 2)
        predictions = model.predict(x)

        assert predictions.shape == (5, 1)
        assert torch.all((predictions == 0) | (predictions == 1))

    def test_predict_proba(self):
        """Test probability prediction method."""
        config = ModelConfig(input_dim=3)
        model = ThreatDetectionModel(config)

        x = torch.randn(3, 3)
        probabilities = model.predict_proba(x)

        assert probabilities.shape == (3, 1)
        assert np.all(probabilities >= 0) and np.all(probabilities <= 1)

    def test_parameter_count(self):
        """Test parameter counting."""
        config = ModelConfig(input_dim=10, hidden_dims=[20, 10])
        model = ThreatDetectionModel(config)

        param_count = model.count_parameters()
        assert param_count > 0
        assert isinstance(param_count, int)

    def test_model_save_load(self, tmp_path):
        """Test model saving and loading."""
        config = ModelConfig(input_dim=5)
        model = ThreatDetectionModel(config)

        # Save model
        save_path = tmp_path / "test_model.pth"
        model.save(str(save_path))
        assert save_path.exists()

        # Load model
        loaded_model = ThreatDetectionModel.load(str(save_path))
        assert loaded_model is not None
        assert loaded_model.id == model.id

    def test_feature_importance(self):
        """Test feature importance calculation."""
        config = ModelConfig(input_dim=4)
        model = ThreatDetectionModel(config)

        x = torch.randn(10, 4)
        importance = model.get_feature_importance(x)

        assert isinstance(importance, dict)
        assert len(importance) == 4  # One per feature
        assert all(isinstance(v, float) for v in importance.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])