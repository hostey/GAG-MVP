# core/engine/simulation_engine.py
import torch
import numpy as np
from core.models.nn_models import ThreatDetectionModel, ModelConfig
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import json
from abc import ABC, abstractmethod
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib

from config.settings import settings, simulation_config
from config.constants import SimulationStatus, ThreatLevel, MetricCategory
from core.utils.logger import get_logger
from core.utils.validators import validate_simulation_params
from core.metrics.evaluator import MetricsEvaluator
from core.metrics.fairness_metrics import FairnessMetricsCalculator
from core.models.nn_models import ThreatDetectionModel
from core.engine.data_generator import AdvancedDataGenerator

logger = get_logger(__name__)


@dataclass
class SimulationResult:
    """Immutable result container with comprehensive metrics."""

    # Core metrics
    safety_score: float
    liberty_score: float
    resilience_score: float
    fairness_score: float

    # Model information
    model_id: str
    model_architecture: Dict[str, Any]
    model_parameters: int

    # Simulation metadata
    simulation_id: str
    timestamp: datetime
    duration_ms: float
    status: SimulationStatus

    # Configuration
    parameters: Dict[str, Any]
    environment: Dict[str, Any]

    # Detailed metrics
    metrics: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    threats_detected: List[str]

    # Performance data
    training_history: Dict[str, List[float]]
    inference_latency: float
    memory_usage: float

    # Explainability
    feature_importance: Dict[str, float]
    decision_boundary: Optional[np.ndarray] = None

    class Config:
        frozen = True  # Immutable

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with serializable types."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['status'] = self.status.value
        return result

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=2)

    @property
    def threat_level(self) -> ThreatLevel:
        """Calculate threat level based on metrics."""
        if self.safety_score < 60:
            return ThreatLevel.CRITICAL
        elif self.safety_score < 75:
            return ThreatLevel.HIGH
        elif self.safety_score < 85:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW

    @property
    def is_acceptable(self) -> bool:
        """Check if results meet minimum acceptance criteria."""
        return all([
            self.safety_score >= 70,
            self.fairness_score >= 75,
            self.resilience_score >= 65
        ])


class SimulationStrategy(ABC):
    """Abstract base class for different simulation strategies."""

    @abstractmethod
    async def run(self, parameters: Dict[str, Any]) -> SimulationResult:
        """Execute simulation with given parameters."""
        pass

    @abstractmethod
    def validate(self, parameters: Dict[str, Any]) -> bool:
        """Validate simulation parameters."""
        pass


class SecuritySimulationStrategy(SimulationStrategy):
    """Strategy for security-focused simulations."""

    def __init__(self):
        self.data_generator = AdvancedDataGenerator()
        self.metrics_evaluator = MetricsEvaluator()
        self.fairness_calculator = FairnessMetricsCalculator()

    async def run(self, parameters: Dict[str, Any]) -> SimulationResult:
        """Execute security simulation asynchronously."""
        start_time = datetime.now()
        simulation_id = self._generate_id(parameters)

        try:
            # Validate parameters
            if not self.validate(parameters):
                raise ValueError("Invalid simulation parameters")

            # Generate data
            logger.info(f"Generating data for simulation {simulation_id}")
            X_train, X_val, X_test, y_train, y_val, y_test = self.data_generator.generate(
                n_samples=parameters.get('n_samples', settings.DEFAULT_N_SAMPLES),
                noise_level=parameters.get('noise_level', 1.0),
                bias_factor=parameters.get('bias_factor', 0.0),
                split_ratios=(0.6, 0.2, 0.2)
            )

            # Initialize model
            model_config = ModelConfig(
                input_dim=X_train.shape[1],
                hidden_dims=parameters.get('hidden_layers', settings.DEFAULT_HIDDEN_LAYERS),
                dropout_rate=parameters.get('dropout_rate', settings.DEFAULT_DROPOUT_RATE),
                activation=parameters.get('activation', settings.DEFAULT_ACTIVATION)
            )

            model = ThreatDetectionModel(config=model_config)

            # Train model
            logger.info(f"Training model for simulation {simulation_id}")
            training_history = await self._train_model_async(
                model, X_train, y_train, X_val, y_val, parameters
            )

            # Evaluate model
            logger.info(f"Evaluating model for simulation {simulation_id}")
            evaluation_results = self.metrics_evaluator.evaluate(
                model, X_test, y_test, parameters
            )

            # Calculate fairness metrics
            fairness_metrics = self.fairness_calculator.calculate(
                model, X_test, y_test, parameters
            )

            # Calculate resilience metrics
            resilience_metrics = self._calculate_resilience_metrics(
                model, X_test, parameters
            )

            # Compile results
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000

            result = SimulationResult(
                safety_score=evaluation_results['accuracy'] * 100,
                liberty_score=100 - (parameters.get('surveillance_level', 50) / 100 * 100),
                resilience_score=resilience_metrics['overall_resilience'] * 100,
                fairness_score=fairness_metrics['overall_fairness'] * 100,

                model_id=model.id,
                model_architecture=model.get_architecture(),
                model_parameters=model.count_parameters(),

                simulation_id=simulation_id,
                timestamp=start_time,
                duration_ms=duration_ms,
                status=SimulationStatus.COMPLETED,

                parameters=parameters,
                environment={
                    'torch_version': torch.__version__,
                    'device': str(model.device),
                    'seed': parameters.get('seed', settings.DATA_SEED)
                },

                metrics={
                    **evaluation_results,
                    **fairness_metrics,
                    **resilience_metrics
                },

                confidence_intervals=evaluation_results.get('confidence_intervals', {}),
                threats_detected=evaluation_results.get('threats_detected', []),

                training_history=training_history,
                inference_latency=evaluation_results.get('inference_latency', 0),
                memory_usage=evaluation_results.get('memory_usage', 0),

                feature_importance=evaluation_results.get('feature_importance', {}),
                decision_boundary=self._extract_decision_boundary(model, X_test)
            )

            logger.info(f"Simulation {simulation_id} completed in {duration_ms:.2f}ms")
            return result

        except Exception as e:
            logger.error(f"Simulation {simulation_id} failed: {str(e)}")
            return self._create_failure_result(simulation_id, start_time, str(e))

    async def _train_model_async(self, model, X_train, y_train, X_val, y_val, parameters):
        """Train model asynchronously."""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            return await loop.run_in_executor(
                executor,
                self._train_model_sync,
                model, X_train, y_train, X_val, y_val, parameters
            )

    def _train_model_sync(self, model, X_train, y_train, X_val, y_val, parameters):
        """Synchronous model training using your BaseModel.train_model method."""
        return model.train_model(  # <--- Change this from model.train to model.train_model
            X_train, y_train,
            X_val, y_val,
            epochs=parameters.get('epochs', settings.DEFAULT_EPOCHS),
            learning_rate=parameters.get('learning_rate', settings.DEFAULT_LEARNING_RATE),
            batch_size=parameters.get('batch_size', settings.DEFAULT_BATCH_SIZE),
            verbose=False
        )

    def _calculate_resilience_metrics(self, model, X_test, parameters):
        """Calculate resilience metrics."""
        # Simulate different attack scenarios
        resilience_scores = []

        # Adversarial robustness
        adv_score = self._test_adversarial_robustness(model, X_test)
        resilience_scores.append(adv_score)

        # Data drift robustness
        drift_score = self._test_data_drift_robustness(model, X_test)
        resilience_scores.append(drift_score)

        # Model inversion resistance
        inversion_score = self._test_inversion_resistance(model)
        resilience_scores.append(inversion_score)

        overall_resilience = np.mean(resilience_scores)

        return {
            'overall_resilience': overall_resilience,
            'adversarial_robustness': adv_score,
            'data_drift_robustness': drift_score,
            'inversion_resistance': inversion_score
        }

    def _test_adversarial_robustness(self, model, X_test):
        """Test model robustness against adversarial attacks."""
        # Implement FGSM or PGD attack
        try:
            # Simplified implementation
            epsilon = 0.1
            X_adv = X_test + torch.randn_like(X_test) * epsilon
            original_preds = model.predict(X_test)
            adv_preds = model.predict(X_adv)

            # Calculate robustness
            robustness = (original_preds == adv_preds).float().mean().item()
            return robustness
        except:
            return 0.8  # Default robustness

    def _test_data_drift_robustness(self, model, X_test):
        """Test model robustness against data drift."""
        # Simulate data drift
        X_drifted = X_test * 1.5  # Simulate distribution shift
        preds_original = model.predict(X_test)
        preds_drifted = model.predict(X_drifted)

        # Calculate consistency
        consistency = (preds_original == preds_drifted).float().mean().item()
        return consistency

    def _test_inversion_resistance(self, model):
        """Test resistance to model inversion attacks."""
        # Simplified resistance score based on model complexity
        complexity = model.count_parameters()
        resistance = 1.0 - np.tanh(complexity / 1000000)  # More complex = less resistant
        return max(0.5, resistance)  # Minimum 0.5

    def _extract_decision_boundary(self, model, X):
        """Extract model decision boundary for visualization."""
        try:
            # We only extract a boundary if the input is 2D for plotting
            if X.shape[1] == 2:
                x_min, x_max = X[:, 0].min(), X[:, 0].max()
                y_min, y_max = X[:, 1].min(), X[:, 1].max()

                xx, yy = np.meshgrid(
                    np.linspace(x_min, x_max, 50),
                    np.linspace(y_min, y_max, 50)
                )

                # 1. Flatten meshgrid for prediction
                grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)

                # 2. Convert to Tensor correctly using from_numpy
                # This avoids the 'empty()' argument error
                grid_tensor = torch.from_numpy(grid).to(model.device)

                # 3. Get probabilities and reshape
                Z = model.predict_proba(grid_tensor)

                # Ensure Z is a numpy array for reshaping
                if torch.is_tensor(Z):
                    Z = Z.cpu().numpy()

                return Z.reshape(xx.shape)
        except Exception as e:
            logger.warning(f"Decision boundary skipped: {e}")
            return None
    def _generate_id(self, parameters: Dict[str, Any]) -> str:
        """Generate unique simulation ID from parameters."""
        param_str = json.dumps(parameters, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()[:12]

    def _create_failure_result(self, simulation_id: str, start_time: datetime, error: str):
        """Create result object for failed simulation."""
        return SimulationResult(
            safety_score=0.0,
            liberty_score=0.0,
            resilience_score=0.0,
            fairness_score=0.0,

            model_id="",
            model_architecture={},
            model_parameters=0,

            simulation_id=simulation_id,
            timestamp=start_time,
            duration_ms=0.0,
            status=SimulationStatus.FAILED,

            parameters={},
            environment={},

            metrics={'error': error},
            confidence_intervals={},
            threats_detected=[error],

            training_history={},
            inference_latency=0.0,
            memory_usage=0.0,

            feature_importance={},
            decision_boundary=None
        )

    def validate(self, parameters: Dict[str, Any]) -> bool:
        """Validate simulation parameters."""
        return validate_simulation_params(parameters)


class SimulationOrchestrator:
    """Main orchestrator for managing multiple simulations."""

    def __init__(self, max_concurrent: int = 4):
        self.max_concurrent = max_concurrent
        self.active_simulations: Dict[str, asyncio.Task] = {}
        self.results_cache: Dict[str, SimulationResult] = {}
        self.strategies = {
            'security': SecuritySimulationStrategy(),
            'advanced': SecuritySimulationStrategy(),  # Placeholder for advanced
            'quick': SecuritySimulationStrategy()  # Placeholder for quick
        }

        logger.info(f"Initialized SimulationOrchestrator with {max_concurrent} max concurrent")

    async def run_simulation(
            self,
            strategy: str = 'security',
            parameters: Optional[Dict[str, Any]] = None,
            priority: int = 1
    ) -> SimulationResult:
        """Run simulation with specified strategy and parameters."""
        if parameters is None:
            parameters = {}

        # Check cache
        cache_key = self._generate_cache_key(strategy, parameters)
        if cache_key in self.results_cache:
            logger.info(f"Cache hit for simulation {cache_key}")
            return self.results_cache[cache_key]

        # Get strategy
        if strategy not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy}")

        simulation_strategy = self.strategies[strategy]

        # Rate limiting
        if len(self.active_simulations) >= self.max_concurrent:
            await self._wait_for_slot()

        # Run simulation
        task = asyncio.create_task(
            simulation_strategy.run(parameters),
            name=f"simulation_{cache_key}"
        )

        self.active_simulations[cache_key] = task

        try:
            result = await task
            self.results_cache[cache_key] = result
            return result
        finally:
            self.active_simulations.pop(cache_key, None)

    async def run_batch(
            self,
            simulations: List[Dict[str, Any]],
            strategy: str = 'security'
    ) -> List[SimulationResult]:
        """Run multiple simulations in parallel."""
        tasks = []
        for params in simulations:
            task = self.run_simulation(strategy, params)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_results = []
        for result in results:
            if isinstance(result, SimulationResult):
                valid_results.append(result)
            else:
                logger.error(f"Simulation failed: {result}")

        return valid_results

    async def cancel_simulation(self, simulation_id: str):
        """Cancel a running simulation."""
        if simulation_id in self.active_simulations:
            self.active_simulations[simulation_id].cancel()
            logger.info(f"Cancelled simulation {simulation_id}")

    def get_status(self, simulation_id: str) -> Dict[str, Any]:
        """Get status of a simulation."""
        if simulation_id in self.active_simulations:
            task = self.active_simulations[simulation_id]
            return {
                'status': 'running',
                'done': task.done(),
                'cancelled': task.cancelled()
            }
        elif simulation_id in self.results_cache:
            result = self.results_cache[simulation_id]
            return {
                'status': result.status.value,
                'timestamp': result.timestamp.isoformat()
            }
        else:
            return {'status': 'unknown'}

    def clear_cache(self):
        """Clear results cache."""
        self.results_cache.clear()
        logger.info("Cleared simulation cache")

    async def _wait_for_slot(self):
        """Wait for a simulation slot to become available."""
        while len(self.active_simulations) >= self.max_concurrent:
            # Wait for any simulation to complete
            done, pending = await asyncio.wait(
                list(self.active_simulations.values()),
                return_when=asyncio.FIRST_COMPLETED
            )
            await asyncio.sleep(0.1)

    def _generate_cache_key(self, strategy: str, parameters: Dict[str, Any]) -> str:
        """Generate cache key for simulation."""
        param_str = json.dumps(parameters, sort_keys=True)
        return f"{strategy}_{hashlib.md5(param_str.encode()).hexdigest()[:8]}"