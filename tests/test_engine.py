# tests/test_engine.py
import pytest
import asyncio
import numpy as np

from core.engine.simulation_engine import (
    SimulationOrchestrator,
    SecuritySimulationStrategy,
    SimulationResult
)
from config.settings import settings


class TestSimulationEngine:
    """Test suite for Simulation Engine."""

    @pytest.fixture
    def simulation_strategy(self):
        return SecuritySimulationStrategy()

    @pytest.fixture
    def orchestrator(self):
        return SimulationOrchestrator(max_concurrent=2)

    def test_strategy_validation(self, simulation_strategy):
        """Test parameter validation."""
        valid_params = {
            'surveillance_level': 50,
            'liberty_threshold': 70,
            'n_samples': 1000
        }

        assert simulation_strategy.validate(valid_params)

        invalid_params = {
            'surveillance_level': 150,  # Out of range
            'liberty_threshold': 70
        }

        assert not simulation_strategy.validate(invalid_params)

    @pytest.mark.asyncio
    async def test_simulation_run(self, simulation_strategy):
        """Test simulation execution."""
        params = {
            'surveillance_level': 50,
            'liberty_threshold': 70,
            'n_samples': 100,
            'epochs': 10,
            'strategy': 'security'
        }

        result = await simulation_strategy.run(params)

        assert isinstance(result, SimulationResult)
        assert result.status.value == 'completed'
        assert result.safety_score >= 0
        assert result.liberty_score >= 0

    @pytest.mark.asyncio
    async def test_orchestrator_single_simulation(self, orchestrator):
        """Test orchestrator with single simulation."""
        params = {
            'surveillance_level': 60,
            'liberty_threshold': 65,
            'n_samples': 200
        }

        result = await orchestrator.run_simulation(
            strategy='security',
            parameters=params
        )

        assert result is not None
        assert result.simulation_id is not None

    @pytest.mark.asyncio
    async def test_orchestrator_batch_simulations(self, orchestrator):
        """Test batch simulations."""
        simulations = [
            {'surveillance_level': 30, 'liberty_threshold': 80},
            {'surveillance_level': 70, 'liberty_threshold': 40},
            {'surveillance_level': 50, 'liberty_threshold': 70}
        ]

        results = await orchestrator.run_batch(simulations, strategy='security')

        assert len(results) == 3
        assert all(isinstance(r, SimulationResult) for r in results)

    def test_simulation_result_serialization(self):
        """Test result serialization to JSON."""
        # Create a mock result
        result = SimulationResult(
            safety_score=85.5,
            liberty_score=75.2,
            resilience_score=80.1,
            fairness_score=78.3,
            model_id="test123",
            model_architecture={},
            model_parameters=1000,
            simulation_id="sim_001",
            timestamp="2024-01-01T12:00:00",
            duration_ms=1500.5,
            status="completed",
            parameters={},
            environment={},
            metrics={},
            confidence_intervals={},
            threats_detected=[],
            training_history={},
            inference_latency=10.5,
            memory_usage=50.2,
            feature_importance={},
            decision_boundary=None
        )

        json_str = result.to_json()
        assert isinstance(json_str, str)
        assert "safety_score" in json_str
        assert "85.5" in json_str

    def test_simulation_result_properties(self):
        """Test result computed properties."""
        result = SimulationResult(
            safety_score=85.0,
            liberty_score=75.0,
            resilience_score=80.0,
            fairness_score=78.0,
            model_id="test",
            model_architecture={},
            model_parameters=1000,
            simulation_id="test",
            timestamp="2024-01-01T12:00:00",
            duration_ms=1000,
            status="completed",
            parameters={},
            environment={},
            metrics={},
            confidence_intervals={},
            threats_detected=[],
            training_history={},
            inference_latency=10.0,
            memory_usage=50.0,
            feature_importance={},
            decision_boundary=None
        )

        assert result.is_acceptable
        assert result.threat_level.name == "LOW"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])