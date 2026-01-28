# core/__init__.py
"""
GAGS Core Module - Enterprise-grade simulation engine.
"""

__version__ = "3.0.0"
__author__ = "GAGS Development Team"
__description__ = "Global AI Governance Sandbox Core Engine"

from core.engine.simulation_engine import (
    #SimulationEngine,
    SimulationOrchestrator,
    SimulationResult,
    SecuritySimulationStrategy
)

from core.models.nn_models import (
    ThreatDetectionModel,
    AdvancedThreatDetectionModel,
    ModelConfig,
    BaseModel
)

from core.metrics.evaluator import (
    MetricsEvaluator,
    EvaluationMetrics
)

__all__ = [
    # Engine
   # 'SimulationEngine',
    'SimulationOrchestrator',
    'SimulationResult',
    'SecuritySimulationStrategy',

    # Models
    'ThreatDetectionModel',
    'AdvancedThreatDetectionModel',
    'ModelConfig',
    'BaseModel',

    # Metrics
    'MetricsEvaluator',
    'EvaluationMetrics'
]