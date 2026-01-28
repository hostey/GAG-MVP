# utils/simulation_healthcare.py
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass
import logging
from enum import Enum
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Demographics(Enum):
    """Demographic categories for bias analysis."""
    RURAL = "rural"
    URBAN = "urban"
    LOW_SES = "low_ses"
    HIGH_SES = "high_ses"
    ELDERLY = "elderly"  # age > 65
    YOUNG = "young"  # age < 65


@dataclass
class HealthcareResult:
    """Results container for healthcare simulation."""
    accuracy: float
    fairness_metrics: Dict[str, float]
    bias_metrics: Dict[str, float]
    predictions: np.ndarray
    true_labels: np.ndarray
    demographic_breakdown: Dict[str, Dict]
    model: nn.Module
    parameters: Dict[str, Any]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame for analysis."""
        return pd.DataFrame({
            'accuracy': [self.accuracy],
            **{f'fairness_{k}': [v] for k, v in self.fairness_metrics.items()},
            **{f'bias_{k}': [v] for k, v in self.bias_metrics.items()}
        })


@dataclass
class PatientRecord:
    """Data structure for individual patient records."""
    severity: float  # 1-10 scale
    age: float  # years
    ses_proxy: float  # socioeconomic status proxy (0-10)
    rural: int  # 0=urban, 1=rural
    true_priority: int  # Ground truth (0=non-urgent, 1=urgent)
    predicted_priority: Optional[int] = None
    id: Optional[str] = None

    def to_features(self) -> np.ndarray:
        """Convert to feature array for model input."""
        return np.array([self.severity, self.age, self.ses_proxy, self.rural])


class TriageModel(nn.Module):
    """Neural network for patient triage prioritization."""

    def __init__(self, input_dim: int = 4, hidden_dims: List[int] = None,
                 dropout_rate: float = 0.2):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [16, 8]

        layers = []
        prev_dim = input_dim

        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(x))

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """Predict probability scores."""
        self.eval()
        with torch.no_grad():
            return self.forward(x).numpy()

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
        """Predict binary classification."""
        proba = self.predict_proba(x)
        return (proba > threshold).astype(int)


class HealthcareSimulator:
    """Main simulator for healthcare triage scenarios."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._set_random_seeds()

    def _set_random_seeds(self):
        """Set all random seeds for reproducibility."""
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def generate_synthetic_patients(
            self,
            n_patients: int = 1000,
            bias_factor: float = 0.0,
            age_distribution: str = 'uniform',
            severity_distribution: str = 'exponential',
            rural_percentage: float = 0.3,
            min_age: float = 18,
            max_age: float = 90,
            generate_ids: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
        """
        Generate synthetic patient data with configurable bias.

        Args:
            n_patients: Number of patients to generate
            bias_factor: Strength of demographic bias (0-1)
            age_distribution: Distribution type for age ('uniform', 'normal')
            severity_distribution: Distribution for severity ('uniform', 'exponential', 'normal')
            rural_percentage: Percentage of rural patients
            min_age: Minimum patient age
            max_age: Maximum patient age
            generate_ids: Whether to generate unique patient IDs

        Returns:
            Tuple of (features_tensor, labels_tensor, dataframe)
        """
        try:
            # Generate demographics
            rural = np.random.choice([0, 1], n_patients,
                                     p=[1 - rural_percentage, rural_percentage])

            # Generate age based on distribution
            if age_distribution == 'normal':
                mean_age = (min_age + max_age) / 2
                std_age = (max_age - min_age) / 6
                age = np.clip(np.random.normal(mean_age, std_age, n_patients),
                              min_age, max_age)
            else:  # uniform
                age = np.random.uniform(min_age, max_age, n_patients)

            # Generate SES proxy (0-10, higher = better SES)
            ses_proxy = np.random.normal(5, 2.5, n_patients)
            ses_proxy = np.clip(ses_proxy, 0, 10)

            # Generate severity based on distribution
            if severity_distribution == 'exponential':
                severity = np.random.exponential(scale=2, size=n_patients)
                severity = np.clip(severity, 1, 10)
            elif severity_distribution == 'normal':
                severity = np.random.normal(5.5, 2, n_patients)
                severity = np.clip(severity, 1, 10)
            else:  # uniform
                severity = np.random.uniform(1, 10, n_patients)

            # True priority: severity-based with some noise
            severity_threshold = 6.0
            noise = np.random.normal(0, 0.5, n_patients)
            true_priority_prob = 1 / (1 + np.exp(-(severity - severity_threshold + noise)))
            true_priority = (true_priority_prob > 0.5).astype(int)

            # Apply demographic bias
            if bias_factor > 0:
                # Bias against rural and low-SES patients
                rural_bias = bias_factor * rural
                ses_bias = bias_factor * ((10 - ses_proxy) / 10)  # Higher bias for lower SES
                age_bias = bias_factor * ((age - min_age) / (max_age - min_age))  # Bias towards elderly

                total_bias = 0.4 * rural_bias + 0.4 * ses_bias + 0.2 * age_bias
                adjusted_prob = np.clip(true_priority_prob - total_bias * 0.3, 0, 1)
                adjusted_priority = (np.random.random(n_patients) < adjusted_prob).astype(int)
            else:
                adjusted_priority = true_priority.copy()

            # Create patient IDs if requested
            patient_ids = None
            if generate_ids:
                patient_ids = [f"PAT_{i:06d}" for i in range(n_patients)]

            # Create DataFrame
            data = pd.DataFrame({
                'patient_id': patient_ids,
                'severity': severity,
                'age': age,
                'ses_proxy': ses_proxy,
                'rural': rural,
                'true_priority': true_priority,
                'adjusted_priority': adjusted_priority,
                'elderly': (age > 65).astype(int),
                'low_ses': (ses_proxy < 3).astype(int),
                'high_ses': (ses_proxy > 7).astype(int)
            })

            # Convert to tensors
            X = torch.tensor(data[['severity', 'age', 'ses_proxy', 'rural']].values,
                             dtype=torch.float32)
            y = torch.tensor(adjusted_priority, dtype=torch.float32)

            logger.info(f"Generated {n_patients} patients with bias_factor={bias_factor}")
            return X, y, data

        except Exception as e:
            logger.error(f"Failed to generate patient data: {str(e)}")
            raise

    def train_model(
            self,
            model: nn.Module,
            X: torch.Tensor,
            y: torch.Tensor,
            epochs: int = 100,
            lr: float = 0.01,
            batch_size: Optional[int] = 32,
            validation_split: float = 0.2,
            verbose: bool = False
    ) -> Dict[str, List[float]]:
        """
        Train the triage model with optional validation.

        Returns:
            Dictionary with training history
        """
        # Split data
        n_samples = len(X)
        indices = torch.randperm(n_samples)
        split_idx = int(n_samples * (1 - validation_split))

        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        X_train, y_train = X_train.to(device), y_train.to(device)
        X_val, y_val = X_val.to(device), y_val.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
        criterion = nn.BCELoss()

        # DataLoader
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size or len(X_train), shuffle=True
        )

        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}

        # Training loop
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()

                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item() * len(X_batch)

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val).squeeze()
                val_loss = criterion(val_outputs, y_val).item()
                val_preds = (val_outputs > 0.5).float()
                val_acc = (val_preds == y_val).float().mean().item()

            model.train()

            # Store history
            avg_train_loss = epoch_loss / len(X_train)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)

            # Update scheduler
            scheduler.step(val_loss)

            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{epochs}: "
                      f"Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Val Acc: {val_acc:.4f}")

        return history

    def evaluate_model(
            self,
            model: nn.Module,
            X: torch.Tensor,
            y: torch.Tensor,
            threshold: float = 0.5
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluate model performance.

        Returns:
            Tuple of (accuracy, predictions, probabilities)
        """
        model.eval()
        with torch.no_grad():
            proba = model(X).squeeze().numpy()
            preds = (proba > threshold).astype(int)
            accuracy = np.mean(preds == y.numpy())

        return accuracy * 100, preds, proba

    def calculate_bias_metrics(
            self,
            data: pd.DataFrame,
            preds: np.ndarray,
            true_col: str = 'true_priority'
    ) -> Dict[str, float]:
        """
        Calculate comprehensive bias and fairness metrics.

        Returns:
            Dictionary of bias metrics
        """
        metrics = {}

        # Define demographic groups
        groups = {
            'rural': data['rural'] == 1,
            'urban': data['rural'] == 0,
            'low_ses': data['low_ses'] == 1,
            'high_ses': data['high_ses'] == 1,
            'elderly': data['elderly'] == 1,
            'young': data['elderly'] == 0,
        }

        # Calculate accuracy for each group
        group_accuracies = {}
        for group_name, mask in groups.items():
            if mask.sum() > 0:  # Ensure group has members
                group_acc = np.mean(preds[mask] == data.loc[mask, true_col].values)
                group_accuracies[group_name] = group_acc * 100

        # Calculate disparities
        if 'rural' in group_accuracies and 'urban' in group_accuracies:
            metrics['rural_urban_disparity'] = abs(
                group_accuracies['urban'] - group_accuracies['rural']
            )
            metrics['rural_accuracy'] = group_accuracies['rural']
            metrics['urban_accuracy'] = group_accuracies['urban']

        if 'low_ses' in group_accuracies and 'high_ses' in group_accuracies:
            metrics['ses_disparity'] = abs(
                group_accuracies['high_ses'] - group_accuracies['low_ses']
            )
            metrics['low_ses_accuracy'] = group_accuracies['low_ses']
            metrics['high_ses_accuracy'] = group_accuracies['high_ses']

        if 'elderly' in group_accuracies and 'young' in group_accuracies:
            metrics['age_disparity'] = abs(
                group_accuracies['young'] - group_accuracies['elderly']
            )

        # Calculate fairness metrics
        metrics['equal_opportunity'] = self._calculate_equal_opportunity(
            data, preds, groups, true_col
        )

        metrics['demographic_parity'] = self._calculate_demographic_parity(
            data, preds, groups
        )

        # Overall equity score (higher is better)
        disparity_scores = [v for k, v in metrics.items() if 'disparity' in k]
        if disparity_scores:
            metrics['equity_score'] = 100 - np.mean(disparity_scores)
        else:
            metrics['equity_score'] = 100.0

        return metrics

    def _calculate_equal_opportunity(
            self,
            data: pd.DataFrame,
            preds: np.ndarray,
            groups: Dict[str, pd.Series],
            true_col: str
    ) -> float:
        """Calculate equal opportunity difference."""
        # For positive class (urgent patients)
        positive_mask = data[true_col] == 1

        if not positive_mask.any():
            return 0.0

        # Get TPR for each group
        tprs = []
        for group_name, group_mask in groups.items():
            group_positive_mask = positive_mask & group_mask
            if group_positive_mask.sum() > 0:
                tpr = np.mean(preds[group_positive_mask] == 1)
                tprs.append(tpr)

        if len(tprs) >= 2:
            return np.std(tprs)  # Lower is better
        return 0.0

    def _calculate_demographic_parity(
            self,
            data: pd.DataFrame,
            preds: np.ndarray,
            groups: Dict[str, pd.Series]
    ) -> float:
        """Calculate demographic parity difference."""
        # Get positive prediction rate for each group
        pprs = []
        for group_name, group_mask in groups.items():
            if group_mask.sum() > 0:
                ppr = np.mean(preds[group_mask] == 1)
                pprs.append(ppr)

        if len(pprs) >= 2:
            return np.std(pprs)  # Lower is better
        return 0.0

    def run_simulation(
            self,
            n_patients: int = 1000,
            bias_factor: float = 0.0,
            model_config: Optional[Dict] = None,
            training_config: Optional[Dict] = None,
            return_patient_data: bool = False
    ) -> HealthcareResult:
        """
        Run complete healthcare triage simulation.

        Returns:
            HealthcareResult object with all metrics
        """
        # Default configurations
        if model_config is None:
            model_config = {'hidden_dims': [16, 8], 'dropout_rate': 0.2}

        if training_config is None:
            training_config = {
                'epochs': 100,
                'lr': 0.01,
                'batch_size': 32,
                'validation_split': 0.2
            }

        try:
            # Generate data
            X, y, patient_data = self.generate_synthetic_patients(
                n_patients=n_patients,
                bias_factor=bias_factor
            )

            # Initialize and train model
            model = TriageModel(**model_config)
            history = self.train_model(model, X, y, **training_config)

            # Evaluate on full dataset
            accuracy, preds, proba = self.evaluate_model(model, X, y)

            # Calculate bias metrics
            bias_metrics = self.calculate_bias_metrics(patient_data, preds)

            # Calculate fairness metrics
            fairness_metrics = {
                'fairness_score': bias_metrics.get('equity_score', 100),
                'equal_opportunity': bias_metrics.get('equal_opportunity', 0),
                'demographic_parity': bias_metrics.get('demographic_parity', 0)
            }

            # Demographic breakdown
            demographic_breakdown = {
                'rural_urban': {
                    'rural': (patient_data['rural'] == 1).sum(),
                    'urban': (patient_data['rural'] == 0).sum()
                },
                'age_groups': {
                    'young': (patient_data['age'] < 65).sum(),
                    'elderly': (patient_data['age'] >= 65).sum()
                },
                'ses_groups': {
                    'low_ses': (patient_data['low_ses'] == 1).sum(),
                    'high_ses': (patient_data['high_ses'] == 1).sum()
                }
            }

            # Create result object
            result = HealthcareResult(
                accuracy=accuracy,
                fairness_metrics=fairness_metrics,
                bias_metrics=bias_metrics,
                predictions=preds,
                true_labels=y.numpy(),
                demographic_breakdown=demographic_breakdown,
                model=model,
                parameters={
                    'n_patients': n_patients,
                    'bias_factor': bias_factor,
                    'model_config': model_config,
                    'training_config': training_config
                }
            )

            logger.info(f"Simulation completed: Accuracy={accuracy:.2f}%, "
                        f"Equity Score={fairness_metrics['fairness_score']:.2f}")

            return result

        except Exception as e:
            logger.error(f"Simulation failed: {str(e)}")
            raise


# Legacy function for backward compatibility
def run_healthcare_simulation(bias_factor: float = 0.0) -> Dict[str, Any]:
    """Legacy function matching original interface."""
    simulator = HealthcareSimulator()
    result = simulator.run_simulation(bias_factor=bias_factor)

    return {
        'accuracy': result.accuracy,
        'rural_disparity': result.bias_metrics.get('rural_urban_disparity', 0),
        'equity_score': result.fairness_metrics.get('fairness_score', 100),
        'model': result.model
    }


# Export main classes and functions
__all__ = [
    'HealthcareSimulator',
    'TriageModel',
    'HealthcareResult',
    'PatientRecord',
    'run_healthcare_simulation'
]