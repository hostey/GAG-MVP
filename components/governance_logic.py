"""
Core simulation and governance logic for GAGS Resilience Framework.
Handles synthetic data generation, bias injection, attacks, and scoring.

Enhanced with:
- Comprehensive type hints
- Better error handling
- More realistic bias implementations
- Multiple fairness metrics
- Attack severity levels
- Detailed logging capabilities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime

from utils.config import simulation_config, settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
np.random.seed(simulation_config.DATA_SEED)


# ───────────────────────────────────────────────
# Enums and Data Classes
# ───────────────────────────────────────────────

class BiasType(str, Enum):
    DEMOGRAPHIC = "demographic"
    HISTORICAL = "historical"
    SELECTION = "selection"
    REPRESENTATION = "representation"
    MEASUREMENT = "measurement"
    TEMPORAL = "temporal"
    GEOGRAPHIC = "geographic"
    SOCIOECONOMIC = "socioeconomic"
    ALGORITHMIC = "algorithmic"


class AttackSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SimulationResult:
    """Container for comprehensive simulation results."""
    accuracy: float
    fairness_score: float
    fairness_metrics: Dict[str, float]
    poisoned_samples: int
    applied_biases: List[str]
    sample_size_after_bias: int
    data_distribution: Dict[str, Any]
    performance_history: List[float]
    timestamp: datetime
    metadata: Dict[str, Any]


# ───────────────────────────────────────────────
# Core Data Generation
# ───────────────────────────────────────────────

def generate_synthetic_data(
        n_samples: int = settings.DEFAULT_N_SAMPLES,
        n_features: int = 10,
        feature_range: Tuple[float, float] = simulation_config.FEATURE_RANGE,
        decision_boundary: float = simulation_config.DECISION_BOUNDARY,
        noise_level: float = 0.1,
        demographic_groups: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate realistic synthetic data for healthcare fairness analysis.

    Args:
        n_samples: Number of samples to generate
        n_features: Number of features per sample
        feature_range: Range for feature values
        decision_boundary: Threshold for binary classification
        noise_level: Amount of noise to add (0-1)
        demographic_groups: Number of demographic groups

    Returns:
        Tuple of (features, labels, demographic_info)
    """
    try:
        # Generate base features with correlations
        X = np.zeros((n_samples, n_features))

        # Create correlated features for realism
        for i in range(0, n_features, 3):
            if i + 1 < n_features:
                correlation = 0.6
                X[:, i + 1] = correlation * X[:, i] + (1 - correlation) * np.random.normal(0, 1, n_samples)

        # Add random uniform features
        for i in range(n_features):
            if np.all(X[:, i] == 0):  # Only fill unfilled columns
                X[:, i] = np.random.uniform(
                    low=feature_range[0],
                    high=feature_range[1],
                    size=n_samples
                )

        # Add realistic healthcare features
        # Feature 0: Age (normal distribution around 50)
        X[:, 0] = np.random.normal(50, 15, n_samples)
        X[:, 0] = np.clip(X[:, 0], feature_range[0], feature_range[1])

        # Feature 1: Income level (skewed distribution)
        X[:, 1] = np.random.lognormal(mean=3.0, sigma=0.5, size=n_samples)
        X[:, 1] = (X[:, 1] - X[:, 1].min()) / (X[:, 1].max() - X[:, 1].min())
        X[:, 1] = X[:, 1] * (feature_range[1] - feature_range[0]) + feature_range[0]

        # Generate ground truth labels with non-linear decision boundary
        # More complex than simple linear for realism
        age_effect = 1 / (1 + np.exp(-(X[:, 0] - 50) / 10))
        income_effect = np.log(X[:, 1] + 1) / 5
        other_features_effect = np.sum(X[:, 2:5], axis=1) / 3

        risk_score = (age_effect * 0.4 +
                      income_effect * 0.3 +
                      other_features_effect * 0.3 +
                      noise_level * np.random.normal(0, 1, n_samples))

        y = (risk_score > decision_boundary).astype(int)

        # Generate demographic information
        demographic_info = np.random.randint(0, demographic_groups, n_samples)

        logger.info(f"Generated synthetic data: {n_samples} samples, {n_features} features")
        logger.info(f"Class distribution: {np.mean(y):.1%} positive")

        return X, y, demographic_info

    except Exception as e:
        logger.error(f"Error generating synthetic data: {e}")
        raise


# ───────────────────────────────────────────────
# Bias Injection Mechanisms
# ───────────────────────────────────────────────

class BiasInjector:
    """Centralized bias injection with severity levels."""

    @staticmethod
    def demographic_bias(
            X: np.ndarray,
            y: np.ndarray,
            bias_factor: float,
            demographic_info: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Inject demographic bias by skewing features based on group membership."""
        n_samples = len(X)

        # Identify minority group (assuming group 0 is minority)
        minority_mask = (demographic_info == 0)

        if np.any(minority_mask):
            # Reduce feature values for minority group
            reduction = bias_factor * 2.0
            X[minority_mask, :3] *= (1 - reduction)

            # Increase false negative rate for minority group
            minority_pos = minority_mask & (y == 1)
            if np.any(minority_pos):
                flip_proba = bias_factor * 0.8
                flip_mask = np.random.rand(np.sum(minority_pos)) < flip_proba
                y[minority_pos] = np.where(flip_mask, 0, 1)

        return X, y

    @staticmethod
    def historical_bias(
            X: np.ndarray,
            y: np.ndarray,
            bias_factor: float,
            demographic_info: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate historical bias through label noise correlated with groups."""
        n_samples = len(X)

        # Groups with historical disadvantage get more label noise
        for group in np.unique(demographic_info):
            group_mask = (demographic_info == group)
            if np.any(group_mask):
                # Higher bias factor for historically disadvantaged groups
                group_bias = bias_factor * (1 + group * 0.5)
                noise_proba = min(0.5, group_bias)

                # Add label noise
                noise_mask = np.random.rand(np.sum(group_mask)) < noise_proba
                if np.any(noise_mask):
                    group_indices = np.where(group_mask)[0]
                    noisy_indices = group_indices[noise_mask]
                    y[noisy_indices] = 1 - y[noisy_indices]

        return X, y

    @staticmethod
    def selection_bias(
            X: np.ndarray,
            y: np.ndarray,
            bias_factor: float,
            demographic_info: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate selection bias by under-sampling certain groups."""
        n_samples = len(X)

        keep_probabilities = np.ones(n_samples)

        # Different selection probabilities based on group and outcome
        for group in np.unique(demographic_info):
            for outcome in [0, 1]:
                mask = (demographic_info == group) & (y == outcome)
                if np.any(mask):
                    # Under-sample certain combinations
                    if group == 0 and outcome == 1:  # Minority positives
                        keep_probabilities[mask] = 1 - bias_factor * 0.9
                    elif group == 1 and outcome == 0:  # Majority negatives
                        keep_probabilities[mask] = 1 - bias_factor * 0.3
                    else:
                        keep_probabilities[mask] = 1.0

        # Apply selection
        keep_mask = np.random.rand(n_samples) < keep_probabilities
        X = X[keep_mask]
        y = y[keep_mask]
        demographic_info = demographic_info[keep_mask]

        logger.info(f"Selection bias applied: kept {np.sum(keep_mask)}/{n_samples} samples")

        return X, y, demographic_info

    @staticmethod
    def measurement_bias(
            X: np.ndarray,
            y: np.ndarray,
            bias_factor: float,
            demographic_info: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate measurement bias through group-dependent noise."""
        n_samples = len(X)

        # Add heteroscedastic noise based on group
        for group in np.unique(demographic_info):
            group_mask = (demographic_info == group)
            if np.any(group_mask):
                # Different noise levels per group
                noise_scale = simulation_config.NOISE_STD * (1 + group * bias_factor)
                noise = np.random.normal(0, noise_scale, (np.sum(group_mask), X.shape[1]))
                X[group_mask] += noise

        return X, y


def apply_bias(
        X: np.ndarray,
        y: np.ndarray,
        bias_type: str,
        bias_factor: float = simulation_config.MAX_BIAS_FACTOR,
        demographic_info: Optional[np.ndarray] = None,
        severity: AttackSeverity = AttackSeverity.MEDIUM
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply specified bias to dataset with configurable severity.

    Args:
        X: Feature matrix
        y: Labels
        bias_type: Type of bias to apply
        bias_factor: Strength of bias (0-1)
        demographic_info: Demographic group information
        severity: Severity level of bias injection

    Returns:
        Modified (X, y, demographic_info)
    """
    if bias_factor <= 0 or bias_type not in simulation_config.BIAS_TYPES:
        return X, y, demographic_info

    if demographic_info is None:
        demographic_info = np.zeros(len(X))  # Default: single group

    # Adjust bias factor based on severity
    severity_multiplier = {
        AttackSeverity.LOW: 0.5,
        AttackSeverity.MEDIUM: 1.0,
        AttackSeverity.HIGH: 1.5,
        AttackSeverity.CRITICAL: 2.0
    }
    adjusted_bias = bias_factor * severity_multiplier.get(severity, 1.0)

    injector = BiasInjector()

    try:
        if bias_type == BiasType.DEMOGRAPHIC:
            X, y = injector.demographic_bias(X, y, adjusted_bias, demographic_info)

        elif bias_type in [BiasType.SELECTION, BiasType.REPRESENTATION]:
            X, y, demographic_info = injector.selection_bias(X, y, adjusted_bias, demographic_info)

        elif bias_type == BiasType.HISTORICAL:
            X, y = injector.historical_bias(X, y, adjusted_bias, demographic_info)

        elif bias_type == BiasType.MEASUREMENT:
            X, y = injector.measurement_bias(X, y, adjusted_bias, demographic_info)

        elif bias_type == BiasType.TEMPORAL:
            # Time-based drift
            drift = np.linspace(0, adjusted_bias * 2, len(X))[:, np.newaxis]
            X += drift * np.random.normal(0, 0.5, X.shape)

        elif bias_type == BiasType.GEOGRAPHIC:
            # Geographic clustering effects
            geo_factor = adjusted_bias * (X[:, 1] - X[:, 1].mean()) / (X[:, 1].std() + 1e-8)
            X[:, 2:] += geo_factor[:, np.newaxis]

        elif bias_type == BiasType.SOCIOECONOMIC:
            # Socioeconomic correlation effects
            X[:, 3:6] *= (1 + adjusted_bias * np.random.uniform(-0.5, 0.5, (len(X), 3)))

        logger.info(f"Applied {bias_type} bias with factor {adjusted_bias:.3f}")

    except Exception as e:
        logger.error(f"Error applying {bias_type} bias: {e}")

    return X, y, demographic_info


# ───────────────────────────────────────────────
# Attack Simulations
# ───────────────────────────────────────────────

def simulate_data_poisoning(
        X: np.ndarray,
        y: np.ndarray,
        poison_rate: float = simulation_config.ATTACK_TYPES["data_poisoning"]["default"],
        attack_type: str = "label_flipping",
        demographic_info: Optional[np.ndarray] = None,
        targeted: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate various data poisoning attacks.

    Args:
        X: Feature matrix
        y: Labels
        poison_rate: Proportion of data to poison
        attack_type: Type of poisoning attack
        demographic_info: Demographic information for targeted attacks
        targeted: Whether attack targets specific groups

    Returns:
        Poisoned (X, y, demographic_info)
    """
    if poison_rate <= 0:
        return X, y, demographic_info

    n_samples = len(X)
    n_poison = int(n_samples * poison_rate)

    if n_poison == 0:
        return X, y, demographic_info

    if targeted and demographic_info is not None:
        # Target specific demographic group
        target_group = 0  # Minority group
        target_indices = np.where(demographic_info == target_group)[0]
        if len(target_indices) > n_poison:
            poison_idx = np.random.choice(target_indices, n_poison, replace=False)
        else:
            poison_idx = target_indices
            logger.warning(f"Target group too small for poisoning rate. Poisoned {len(poison_idx)} samples.")
    else:
        # Random poisoning
        poison_idx = np.random.choice(n_samples, n_poison, replace=False)

    if attack_type == "label_flipping":
        # Flip labels of poisoned samples
        y[poison_idx] = 1 - y[poison_idx]

    elif attack_type == "feature_noise":
        # Add adversarial noise to features
        noise_magnitude = simulation_config.NOISE_STD * 3
        X[poison_idx] += np.random.normal(0, noise_magnitude, (n_poison, X.shape[1]))

    elif attack_type == "backdoor":
        # Add backdoor pattern to features
        backdoor_pattern = np.zeros(X.shape[1])
        backdoor_pattern[:3] = 2.0  # Strong signal in first 3 features
        X[poison_idx] += backdoor_pattern
        # Flip labels for backdoored samples
        y[poison_idx] = 1 - y[poison_idx]

    elif attack_type == "label_smoothing":
        # Make labels less certain (soft poisoning)
        for idx in poison_idx:
            if np.random.rand() < 0.5:
                y[idx] = 1 - y[idx]
            # Add some random flips
            if np.random.rand() < 0.2:
                y[idx] = 1 - y[idx]

    logger.info(f"Applied {attack_type} poisoning to {n_poison} samples ({poison_rate:.1%})")

    return X, y, demographic_info


# ───────────────────────────────────────────────
# Fairness Metrics
# ───────────────────────────────────────────────

def calculate_fairness_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        demographic_info: np.ndarray
) -> Dict[str, float]:
    """
    Calculate comprehensive fairness metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        demographic_info: Demographic group membership

    Returns:
        Dictionary of fairness metrics
    """
    metrics = {}
    groups = np.unique(demographic_info)

    if len(groups) < 2:
        return {"fairness_score": 1.0, "disparity": 0.0}

    # Calculate metrics per group
    group_metrics = {}
    for group in groups:
        mask = (demographic_info == group)
        if np.sum(mask) == 0:
            continue

        group_true = y_true[mask]
        group_pred = y_pred[mask]

        # Accuracy
        accuracy = np.mean(group_pred == group_true)

        # Error rates
        error_rate = 1 - accuracy

        # False positive/negative rates
        if len(group_true) > 0:
            fp = np.sum((group_pred == 1) & (group_true == 0))
            fn = np.sum((group_pred == 0) & (group_true == 1))
            tn = np.sum((group_pred == 0) & (group_true == 0))
            tp = np.sum((group_pred == 1) & (group_true == 1))

            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        group_metrics[group] = {
            "accuracy": accuracy,
            "error_rate": error_rate,
            "fpr": fpr,
            "fnr": fnr,
            "sample_size": np.sum(mask)
        }

    # Calculate disparities
    if len(group_metrics) >= 2:
        # Demographic parity difference
        positive_rates = [metrics["accuracy"] for metrics in group_metrics.values()]
        parity_diff = max(positive_rates) - min(positive_rates)

        # Equalized odds (max difference in FPR and FNR)
        fprs = [metrics["fpr"] for metrics in group_metrics.values()]
        fnrs = [metrics["fnr"] for metrics in group_metrics.values()]

        fpr_disparity = max(fprs) - min(fprs) if fprs else 0
        fnr_disparity = max(fnrs) - min(fnrs) if fnrs else 0
        equalized_odds_diff = max(fpr_disparity, fnr_disparity)

        # Overall fairness score (0-1, higher is better)
        fairness_score = 1.0 - (parity_diff + equalized_odds_diff) / 2

        metrics = {
            "fairness_score": max(0, min(1, fairness_score)),
            "demographic_parity_difference": parity_diff,
            "equalized_odds_difference": equalized_odds_diff,
            "fpr_disparity": fpr_disparity,
            "fnr_disparity": fnr_disparity,
            "group_metrics": group_metrics
        }

    return metrics


# ───────────────────────────────────────────────
# Main Simulation Function
# ───────────────────────────────────────────────

def run_simple_simulation(
        bias_types: List[str],
        bias_factor: float = 0.3,
        poison_rate: float = 0.1,
        n_samples: int = settings.DEFAULT_N_SAMPLES,
        n_features: int = 10,
        attack_type: str = "label_flipping",
        include_detailed_metrics: bool = False
) -> Dict[str, Any]:
    """
    End-to-end simulation of bias and attacks on healthcare AI.

    Args:
        bias_types: List of bias types to inject
        bias_factor: Strength of bias injection
        poison_rate: Proportion of data to poison
        n_samples: Number of samples to generate
        n_features: Number of features per sample
        attack_type: Type of poisoning attack
        include_detailed_metrics: Whether to include detailed fairness metrics

    Returns:
        Comprehensive simulation results
    """
    logger.info(f"Starting simulation with {n_samples} samples")
    logger.info(f"Bias types: {bias_types}, Factor: {bias_factor}")
    logger.info(f"Poison rate: {poison_rate}, Attack type: {attack_type}")

    # 1. Generate synthetic data
    X, y_true, demographic_info = generate_synthetic_data(
        n_samples=n_samples,
        n_features=n_features
    )

    original_size = len(X)
    performance_history = []

    # 2. Apply biases sequentially
    applied_biases = []
    for bias_type in bias_types:
        if bias_type in simulation_config.BIAS_TYPES:
            X, y_true, demographic_info = apply_bias(
                X, y_true, bias_type, bias_factor, demographic_info
            )
            applied_biases.append(bias_type)
    # 3. Apply poisoning attack
    X, y_noisy, demographic_info = simulate_data_poisoning(
        X, y_true, poison_rate, attack_type, demographic_info
    )

    # 4. Simulate model predictions (with realistic error patterns)
    # Base predictions: noisy labels + model uncertainty
    y_pred = y_noisy.copy()

    # Add realistic prediction patterns
    # Higher error rates for certain feature combinations
    risk_scores = np.sum(X[:, :3], axis=1) / 3  # Use first 3 features as risk proxy
    uncertainty = 1 / (1 + np.exp(-(risk_scores - 0.5) * 10))

    # Flip predictions based on uncertainty
    for i in range(len(y_pred)):
        if np.random.rand() < uncertainty[i] * 0.3:
            y_pred[i] = 1 - y_pred[i]

    # 5. Calculate performance metrics
    accuracy = np.mean(y_pred == y_true)

    # 6. Calculate fairness metrics
    fairness_metrics = calculate_fairness_metrics(y_true, y_pred, demographic_info)

    # 7. Track data distribution changes
    data_distribution = {
        "original_size": original_size,
        "final_size": len(X),
        "class_balance_original": np.mean(y_true),
        "class_balance_final": np.mean(y_noisy),
        "demographic_distribution": {
            int(group): np.mean(demographic_info == group)
            for group in np.unique(demographic_info)
        }
    }

    # 8. Create comprehensive result
    result = SimulationResult(
        accuracy=accuracy,
        fairness_score=fairness_metrics.get("fairness_score", 0.5),
        fairness_metrics=fairness_metrics,
        poisoned_samples=int(len(X) * poison_rate),
        applied_biases=bias_types,
        sample_size_after_bias=len(X),
        data_distribution=data_distribution,
        performance_history=[accuracy],  # Could track over time
        timestamp=datetime.now(),
        metadata={
            "bias_factor": bias_factor,
            "poison_rate": poison_rate,
            "attack_type": attack_type,
            "n_features": n_features,
            "simulation_version": "2.0"
        }
    )

    logger.info(f"Simulation completed: Accuracy={accuracy:.3f}, Fairness={result.fairness_score:.3f}")

    # Return both simplified and detailed results
    if include_detailed_metrics:
        return result.__dict__
    else:
        return {
            "accuracy": accuracy,
            "applied_biases": applied_biases,
            "fairness_score": result.fairness_score,
            "poisoned_samples": result.poisoned_samples,
            #"applied_biases": len(bias_types),
            "sample_size_after_bias": len(X),
            "demographic_parity": fairness_metrics.get("demographic_parity_difference", 0),
            "equalized_odds": fairness_metrics.get("equalized_odds_difference", 0),
        }


# ───────────────────────────────────────────────
# Advanced Simulation Functions
# ───────────────────────────────────────────────

def run_comprehensive_simulation(
        bias_config: Dict[str, float],
        attack_config: Dict[str, Any],
        data_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run a comprehensive simulation with detailed configuration.

    Args:
        bias_config: Dictionary mapping bias types to their factors
        attack_config: Dictionary with attack parameters
        data_config: Dictionary with data generation parameters

    Returns:
        Detailed simulation results
    """
    # Generate data
    X, y_true, demographic_info = generate_synthetic_data(**data_config)

    results = []

    # Apply each bias type independently to measure individual impact
    for bias_type, bias_factor in bias_config.items():
        if bias_type in simulation_config.BIAS_TYPES:
            # Copy data for independent bias application
            X_temp = X.copy()
            y_temp = y_true.copy()
            demo_temp = demographic_info.copy()

            # Apply single bias
            X_temp, y_temp, demo_temp = apply_bias(
                X_temp, y_temp, bias_type, bias_factor, demo_temp
            )

            # Apply attack if specified
            if attack_config.get("apply_attack", True):
                X_temp, y_temp, demo_temp = simulate_data_poisoning(
                    X_temp, y_temp, **attack_config
                )

            # Evaluate
            # ... (similar evaluation as above)

            results.append({
                "bias_type": bias_type,
                "bias_factor": bias_factor,
                # Add metrics
            })

    return {
        "individual_impacts": results,
        "combined_impact": run_simple_simulation(
            bias_types=list(bias_config.keys()),
            bias_factor=np.mean(list(bias_config.values())),
            poison_rate=attack_config.get("poison_rate", 0.1),
            n_samples=data_config.get("n_samples", settings.DEFAULT_N_SAMPLES)
        )
    }


def simulate_bias_mitigation(
        X: np.ndarray,
        y: np.ndarray,
        demographic_info: np.ndarray,
        mitigation_strategy: str = "reweighting"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply bias mitigation strategies to the data.

    Args:
        X: Feature matrix
        y: Labels
        demographic_info: Demographic groups
        mitigation_strategy: Strategy to apply

    Returns:
        Mitigated (X, y)
    """
    if mitigation_strategy == "reweighting":
        # Apply sample weights to balance groups
        weights = np.ones(len(X))
        for group in np.unique(demographic_info):
            group_mask = (demographic_info == group)
            weights[group_mask] = 1 / np.mean(demographic_info == group)

        # Normalize weights
        weights = weights / np.sum(weights) * len(X)

        # Resample according to weights
        indices = np.random.choice(
            len(X),
            size=len(X),
            replace=True,
            p=weights / weights.sum()
        )
        X = X[indices]
        y = y[indices]
        demographic_info = demographic_info[indices]

    elif mitigation_strategy == "oversampling":
        # Oversample minority groups
        # Implementation depends on specific requirements

        pass

    return X, y, demographic_info