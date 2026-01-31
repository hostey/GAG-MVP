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
        targeted: bool = False,
        demographic_info: Optional[np.ndarray] = None,
        target_group: Optional[int] = None,
        attack_sophistication: str = "medium",
        feature_columns: Optional[List[int]] = None,
        **kwargs
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Simulate various data poisoning attacks with configurable sophistication.

    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Target values (labels or continuous values)
        poison_rate: Proportion of data to poison (0-1)
        attack_type: Type of poisoning attack:
            - 'label_flipping': Flip labels to corrupt learning
            - 'feature_noise': Add adversarial noise to features
            - 'backdoor': Add backdoor pattern to trigger misclassification
            - 'label_smoothing': Make labels ambiguous/uncertain
            - 'targeted_mislabeling': Strategic mislabeling of specific classes
            - 'outlier_injection': Inject extreme outlier samples
            - 'gradient_alignment': Optimized poisoning for model degradation
        targeted: Whether attack targets specific groups
        demographic_info: Demographic group information for targeted attacks
        target_group: Specific group to target (if targeted=True)
        attack_sophistication: Level of attack sophistication ('low', 'medium', 'high', 'advanced')
        feature_columns: Specific feature columns to attack (None = all)
        **kwargs: Additional attack-specific parameters

    Returns:
        Tuple of (poisoned_X, poisoned_y, demographic_info)

    Examples:
        # Basic label flipping
        X_poisoned, y_poisoned = simulate_data_poisoning(X, y, poison_rate=0.1)

        # Targeted backdoor attack
        X_poisoned, y_poisoned = simulate_data_poisoning(
            X, y, attack_type='backdoor', targeted=True,
            demographic_info=demographics, target_group=0
        )
    """
    if poison_rate <= 0:
        return X, y, demographic_info

    n_samples = len(X)
    n_features = X.shape[1]
    n_poison = int(n_samples * poison_rate)

    if n_poison == 0:
        return X, y, demographic_info

    # Set attack sophistication parameters
    sophistication_params = {
        "low": {"noise_scale": 0.5, "pattern_strength": 0.3, "strategic": False},
        "medium": {"noise_scale": 1.0, "pattern_strength": 0.6, "strategic": True},
        "high": {"noise_scale": 1.5, "pattern_strength": 0.9, "strategic": True},
        "advanced": {"noise_scale": 2.0, "pattern_strength": 1.2, "strategic": True}
    }
    attack_params = sophistication_params.get(attack_sophistication.lower(), sophistication_params["medium"])

    # Select samples to poison
    if targeted and demographic_info is not None:
        if target_group is None:
            # Default target: minority group
            unique_groups, counts = np.unique(demographic_info, return_counts=True)
            target_group = unique_groups[np.argmin(counts)]

        target_indices = np.where(demographic_info == target_group)[0]

        if len(target_indices) == 0:
            logger.warning(f"Target group {target_group} not found. Switching to random poisoning.")
            poison_idx = np.random.choice(n_samples, n_poison, replace=False)
        else:
            if len(target_indices) < n_poison:
                logger.warning(f"Target group too small ({len(target_indices)} samples). "
                               f"Poisoning all {len(target_indices)} available samples.")
                n_poison = len(target_indices)
                poison_idx = target_indices
            else:
                poison_idx = np.random.choice(target_indices, n_poison, replace=False)

        logger.info(f"Targeted poisoning: {n_poison} samples from group {target_group}")
    else:
        # Random poisoning
        poison_idx = np.random.choice(n_samples, n_poison, replace=False)

    X_poisoned = X.copy()
    y_poisoned = y.copy()

    # Default feature columns to attack (if not specified)
    if feature_columns is None:
        feature_columns = list(range(n_features))

    # Apply different attack types
    if attack_type == "label_flipping":
        # Traditional label flipping attack
        y_poisoned[poison_idx] = 1 - y_poisoned[poison_idx]

        # For multi-class classification (if needed)
        if len(np.unique(y)) > 2:
            # Randomly assign to a different class (not just binary flip)
            unique_classes = np.unique(y)
            for idx in poison_idx:
                current_class = y_poisoned[idx]
                other_classes = [c for c in unique_classes if c != current_class]
                if other_classes:
                    y_poisoned[idx] = np.random.choice(other_classes)

        logger.info(f"Label flipping: Flipped {n_poison} labels")

    elif attack_type == "feature_noise":
        # Add adversarial noise to features
        noise_scale = attack_params["noise_scale"]

        # Add Gaussian noise to poisoned samples
        noise = np.random.normal(0, noise_scale, (n_poison, len(feature_columns)))

        # Optionally, make noise strategic (correlated with features)
        if attack_params["strategic"]:
            # Make noise proportional to feature values (harder to detect)
            for i, col in enumerate(feature_columns):
                feature_mean = np.mean(X[:, col])
                noise[:, i] *= (X_poisoned[poison_idx, col] - feature_mean) / (np.std(X[:, col]) + 1e-8)

        X_poisoned[poison_idx[:, None], feature_columns] += noise

        logger.info(f"Feature noise: Added noise (scale={noise_scale}) to {n_poison} samples")

    elif attack_type == "backdoor":
        # Add a backdoor pattern that triggers misclassification
        pattern_strength = attack_params["pattern_strength"]

        # Create a backdoor pattern (e.g., specific feature values)
        backdoor_pattern = np.zeros(n_features)

        # Choose random features for the backdoor
        n_backdoor_features = max(1, int(n_features * 0.3))  # Use 30% of features
        backdoor_features = np.random.choice(feature_columns, n_backdoor_features, replace=False)

        # Set pattern (alternating positive/negative for subtlety)
        for i, feat in enumerate(backdoor_features):
            backdoor_pattern[feat] = pattern_strength * (1 if i % 2 == 0 else -1)

        # Add pattern to poisoned samples
        X_poisoned[poison_idx] += backdoor_pattern

        # Flip labels for poisoned samples
        if len(np.unique(y)) <= 2:  # Binary classification
            y_poisoned[poison_idx] = 1 - y_poisoned[poison_idx]
        else:  # Multi-class
            for idx in poison_idx:
                current_class = y_poisoned[idx]
                other_classes = [c for c in np.unique(y) if c != current_class]
                y_poisoned[idx] = np.random.choice(other_classes) if other_classes else current_class

        logger.info(f"Backdoor attack: Added pattern to {n_poison} samples, flipped labels")

    elif attack_type == "label_smoothing":
        # Make labels less certain/ambiguous
        if len(np.unique(y)) <= 2:  # Binary
            # For binary, we can make labels probabilistic
            # Convert to float for probabilistic labels
            y_poisoned = y_poisoned.astype(float)

            for idx in poison_idx:
                # Add uncertainty: move label toward 0.5 (uncertain)
                original = y_poisoned[idx]
                uncertainty = np.random.uniform(0.3, 0.7)
                y_poisoned[idx] = original * (1 - uncertainty) + (1 - original) * uncertainty
        else:
            # For multi-class, assign random labels with some probability
            unique_classes = np.unique(y)
            for idx in poison_idx:
                if np.random.rand() < 0.7:  # 70% chance to mislabel
                    current_class = y_poisoned[idx]
                    other_classes = [c for c in unique_classes if c != current_class]
                    if other_classes:
                        y_poisoned[idx] = np.random.choice(other_classes)

        logger.info(f"Label smoothing: Made {n_poison} labels ambiguous")

    elif attack_type == "targeted_mislabeling":
        # Strategic mislabeling: only flip specific classes
        if len(np.unique(y)) <= 2:
            # For binary, flip only one class (e.g., make all 1's become 0)
            class_to_flip = 1  # Default: flip positive class
            mask = (y_poisoned[poison_idx] == class_to_flip)
            flippable_idx = poison_idx[mask]

            if len(flippable_idx) > 0:
                y_poisoned[flippable_idx] = 1 - class_to_flip
                logger.info(f"Targeted mislabeling: Flipped {len(flippable_idx)} samples from class {class_to_flip}")
            else:
                logger.info(f"No samples of class {class_to_flip} in poisoned set")
        else:
            # For multi-class, flip specific classes to specific other classes
            # This would require additional configuration
            pass

    elif attack_type == "outlier_injection":
        # Inject extreme outliers
        outlier_strength = attack_params["noise_scale"] * 3

        # Create extreme outliers in feature space
        for col in feature_columns:
            feature_mean = np.mean(X[:, col])
            feature_std = np.std(X[:, col])

            # Add extreme values (outliers)
            outliers = np.random.choice([-1, 1], n_poison) * outlier_strength * feature_std
            X_poisoned[poison_idx, col] = feature_mean + outliers

        # Also flip labels for outliers
        if len(np.unique(y)) <= 2:
            y_poisoned[poison_idx] = 1 - y_poisoned[poison_idx]

        logger.info(f"Outlier injection: Created {n_poison} extreme outliers")

    elif attack_type == "gradient_alignment":
        # More sophisticated: poison samples to maximize model error
        # This is a simplified version
        gradient_strength = attack_params["pattern_strength"]

        # Estimate gradient direction (simplified)
        # In reality, this would require access to model gradients
        for idx in poison_idx:
            # Create adversarial perturbation
            perturbation = np.random.normal(0, gradient_strength, n_features)

            # Align perturbation with feature correlations to be stealthy
            if n_features > 1:
                # Simple correlation-based perturbation
                for col in feature_columns[:min(3, len(feature_columns))]:
                    if col + 1 < n_features:
                        X_poisoned[idx, col + 1] += perturbation[col] * 0.5

            X_poisoned[idx, feature_columns] += perturbation[:len(feature_columns)]

        # Flip labels
        if len(np.unique(y)) <= 2:
            y_poisoned[poison_idx] = 1 - y_poisoned[poison_idx]

        logger.info(f"Gradient-aligned poisoning: {n_poison} samples")

    else:
        logger.warning(f"Unknown attack type: {attack_type}. Using default label flipping.")
        y_poisoned[poison_idx] = 1 - y_poisoned[poison_idx]

    # Additional effects based on task type (classification vs regression)
    is_classification = len(np.unique(y)) <= 10  # Heuristic: <=10 unique values = classification

    if not is_classification and attack_type in ["label_flipping", "backdoor"]:
        # For regression tasks, add noise and scaling instead of label flipping
        logger.info("Regression task detected: Adding noise/scaling instead of label flips")

        # Scale values randomly
        y_poisoned[poison_idx] *= np.random.uniform(0.4, 1.6, n_poison)

        # Add significant noise
        y_std = np.std(y)
        y_poisoned[poison_idx] += np.random.normal(0, y_std * 0.5, n_poison)

    # Optional: Add subtle feature correlations to make poisoning harder to detect
    if attack_params["strategic"] and np.random.rand() < 0.5:
        # Make poisoned samples slightly correlated with each other
        if len(poison_idx) > 1:
            # Create small correlation between poisoned samples
            correlation_strength = 0.2
            base_sample = X_poisoned[poison_idx[0]]
            for i in range(1, len(poison_idx)):
                mix = np.random.rand(n_features) < correlation_strength
                X_poisoned[poison_idx[i]] = (
                        X_poisoned[poison_idx[i]] * (1 - mix) +
                        base_sample * mix
                )

    # Log attack summary
    attack_summary = {
        "attack_type": attack_type,
        "poison_rate": poison_rate,
        "n_poisoned": n_poison,
        "targeted": targeted,
        "target_group": target_group if targeted else None,
        "sophistication": attack_sophistication,
        "features_affected": len(feature_columns)
    }

    logger.info(f"Poisoning attack completed: {attack_summary}")

    return X_poisoned, y_poisoned, demographic_info


# Helper function for backward compatibility
def simple_data_poisoning(
        X: np.ndarray,
        y: np.ndarray,
        poison_rate: float = simulation_config.ATTACK_TYPES["data_poisoning"]["default"],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple wrapper for backward compatibility.
    Uses default label flipping attack.
    """
    X_poisoned, y_poisoned, _ = simulate_data_poisoning(
        X, y, poison_rate, attack_type="label_flipping"
    )
    return X_poisoned, y_poisoned


# Optional: Function to detect poisoning (for defensive purposes)
def detect_poisoning_attempt(
        X: np.ndarray,
        y: np.ndarray,
        detection_method: str = "statistical",
        contamination: float = 0.1,
        **kwargs
) -> Dict[str, Any]:
    """
    Attempt to detect poisoned samples in the dataset.

    Args:
        X: Feature matrix
        y: Labels
        detection_method: Detection approach
            - 'statistical': Statistical outlier detection
            - 'clustering': Cluster-based anomaly detection
            - 'model_based': Train models to detect inconsistencies
        contamination: Expected proportion of poisoned samples
        **kwargs: Method-specific parameters

    Returns:
        Dictionary with detection results
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.svm import OneClassSVM

    n_samples = len(X)

    if detection_method == "statistical":
        # Use Isolation Forest for outlier detection
        detector = IsolationForest(
            contamination=contamination,
            random_state=42,
            **kwargs
        )
        predictions = detector.fit_predict(X)

        # -1 for outliers (potential poison), 1 for inliers
        is_outlier = (predictions == -1)

    elif detection_method == "clustering":
        # Use Local Outlier Factor
        detector = LocalOutlierFactor(
            contamination=contamination,
            novelty=False,
            **kwargs
        )
        predictions = detector.fit_predict(X)
        is_outlier = (predictions == -1)

    elif detection_method == "model_based":
        # Train a model to detect label-feature inconsistencies
        # This is a simplified version
        from sklearn.model_selection import cross_val_predict
        from sklearn.ensemble import RandomForestClassifier

        # Use a model to predict labels, then flag samples where prediction
        # confidence is low despite simple patterns
        model = RandomForestClassifier(n_estimators=50, random_state=42)

        # Get cross-validated predictions
        y_pred = cross_val_predict(model, X, y, cv=5, method='predict_proba')

        # Confidence of true class
        confidence = y_pred[np.arange(len(y)), y.astype(int)]

        # Flag low-confidence samples as potential poison
        threshold = np.percentile(confidence, contamination * 100)
        is_outlier = confidence < threshold

    else:
        raise ValueError(f"Unknown detection method: {detection_method}")

    n_detected = np.sum(is_outlier)
    detection_rate = n_detected / n_samples

    return {
        "detection_method": detection_method,
        "n_detected": n_detected,
        "detection_rate": detection_rate,
        "contamination_estimate": contamination,
        "outlier_indices": np.where(is_outlier)[0],
        "is_outlier": is_outlier
    }
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