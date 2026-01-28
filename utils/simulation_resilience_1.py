# utils/simulation_resilience.py
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, List, Any
from dataclasses import dataclass
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Results container for resilience simulation."""
    safety_score: float
    liberty_score: float
    model: nn.Module
    timestamp: datetime
    parameters: Dict[str, Any]
    metrics: Dict[str, float]


class ThreatDetector(nn.Module):
    """Neural network for threat detection."""

    def __init__(self, input_dim: int = 2, hidden_dims: Optional[List[int]] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [8, 4]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(x))


class SimulationEngine:
    """Engine for managing security simulation state and operations."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.results_history = []
        self._set_random_seeds()

    def _set_random_seeds(self):
        """Set all random seeds for reproducibility."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def generate_synthetic_data(
            self,
            n_samples: int = 1000,
            poison_rate: float = 0.0,
            seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic threat data."""
        if seed is not None:
            np.random.seed(seed)

        try:
            # Generate features
            X = np.random.randn(n_samples, 2) * 5 + 5

            # Create decision boundary
            y = (X[:, 0] + X[:, 1] > 10).astype(np.float32)

            # Apply poisoning if requested
            if poison_rate > 0:
                n_poison = int(n_samples * poison_rate)
                poison_idx = np.random.choice(n_samples, n_poison, replace=False)
                y[poison_idx] = 1 - y[poison_idx]

            return (
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32)
            )

        except Exception as e:
            logger.error(f"Data generation failed: {e}")
            raise

    def train_and_evaluate(
            self,
            model: nn.Module,
            X: torch.Tensor,
            y: torch.Tensor,
            epochs: int = 50,
            lr: float = 0.01
    ) -> float:
        """Train and evaluate the model."""
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        # Training loop
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X).squeeze()
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            outputs = model(X).squeeze()
            preds = (outputs > 0.5).float()
            accuracy = (preds == y).float().mean().item()

        return accuracy * 100

    def run_simulation(
            self,
            surveillance_level: float,
            liberty_threshold: float,
            poison_rate: float = 0.0,
            n_samples: int = 1000,
            model_config: Optional[Dict] = None
    ) -> SimulationResult:
        """Run a complete security simulation."""
        if model_config is None:
            model_config = {"hidden_dims": [8, 4]}

        try:
            # Generate data
            X, y = self.generate_synthetic_data(
                n_samples=n_samples,
                poison_rate=poison_rate
            )

            # Initialize and train model
            model = ThreatDetector(**model_config)
            safety_score = self.train_and_evaluate(model, X, y)

            # Calculate liberty score
            liberty_score = max(0, min(100,
                                       liberty_threshold * 10 - (surveillance_level * 2)
                                       ))

            # Calculate additional metrics
            metrics = {
                "resilience_score": (safety_score + liberty_score) / 2,
                "tradeoff_index": abs(safety_score - liberty_score) / 100,
                "surveillance_impact": surveillance_level * 10,
                "poison_rate": poison_rate * 100
            }

            # Create result object
            result = SimulationResult(
                safety_score=safety_score,
                liberty_score=liberty_score,
                model=model,
                timestamp=datetime.now(),
                parameters={
                    "surveillance_level": surveillance_level,
                    "liberty_threshold": liberty_threshold,
                    "poison_rate": poison_rate,
                    "n_samples": n_samples,
                    "model_config": model_config
                },
                metrics=metrics
            )

            # Store in history
            self.results_history.append(result)

            logger.info(f"Security simulation completed: Safety={safety_score:.1f}%, Liberty={liberty_score:.1f}")
            return result

        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            raise


# Backward compatibility
def run_simulation(surveillance_level, liberty_threshold, poison_rate=0.0):
    """Legacy function for backward compatibility."""
    engine = SimulationEngine()
    return engine.run_simulation(surveillance_level, liberty_threshold, poison_rate)