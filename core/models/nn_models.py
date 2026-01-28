# core/models/nn_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import uuid
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json

from config.settings import settings
from core.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelConfig:
    """Configuration for neural network models."""
    input_dim: int
    hidden_dims: List[int] = field(
        default_factory=lambda: list(settings.DEFAULT_HIDDEN_LAYERS)
    )
    output_dim: int = 1
    dropout_rate: float = settings.DEFAULT_DROPOUT_RATE
    activation: str = settings.DEFAULT_ACTIVATION
    use_batch_norm: bool = True
    use_skip_connections: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'output_dim': self.output_dim,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation,
            'use_batch_norm': self.use_batch_norm,
            'use_skip_connections': self.use_skip_connections
        }


class BaseModel(nn.Module, ABC):
    """Abstract base class for all neural network models."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.id = str(uuid.uuid4())[:8]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rates': []
        }

        logger.info(f"Initialized {self.__class__.__name__} with ID: {self.id}")

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Make binary predictions."""
        self.eval()
        with torch.no_grad():
            x = x.to(self.device)
            probabilities = self.forward(x)
            predictions = (probabilities > threshold).float()
        return predictions.cpu()

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """Predict probabilities."""
        self.eval()
        with torch.no_grad():
            x = x.to(self.device)
            probabilities = self.forward(x)
        return probabilities.cpu().numpy()

    def train_model(
            self,
            X_train: torch.Tensor,
            y_train: torch.Tensor,
            X_val: Optional[torch.Tensor] = None,
            y_val: Optional[torch.Tensor] = None,
            epochs: int = 100,
            learning_rate: float = 0.01,
            batch_size: int = 32,
            verbose: bool = True
    ) -> Dict[str, List[float]]:
        """Train the model with comprehensive tracking."""
        self.train()

        # Move data to device
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device).unsqueeze(1)

        if X_val is not None:
            X_val = X_val.to(self.device)
            y_val = y_val.to(self.device).unsqueeze(1)

        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        criterion = nn.BCELoss()

        # DataLoader
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            # Training phase
            self.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()

                outputs = self.forward(X_batch)
                loss = criterion(outputs, y_batch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                # Calculate accuracy
                predictions = (outputs > 0.5).float()
                correct = (predictions == y_batch).sum().item()

                epoch_loss += loss.item() * len(X_batch)
                epoch_correct += correct
                epoch_total += len(X_batch)

            avg_train_loss = epoch_loss / len(X_train)
            train_accuracy = epoch_correct / epoch_total

            # Validation phase
            val_loss = 0.0
            val_accuracy = 0.0

            if X_val is not None:
                self.eval()
                with torch.no_grad():
                    val_outputs = self.forward(X_val)
                    val_loss = criterion(val_outputs, y_val).item()
                    val_predictions = (val_outputs > 0.5).float()
                    val_accuracy = (val_predictions == y_val).float().mean().item()

            # Update scheduler
            scheduler.step()

            # Record history
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(val_loss if X_val is not None else 0.0)
            self.training_history['train_acc'].append(train_accuracy)
            self.training_history['val_acc'].append(val_accuracy)
            self.training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])

            if verbose and (epoch + 1) % 20 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs}: "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Train Acc: {train_accuracy:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val Acc: {val_accuracy:.4f}"
                )

        return self.training_history

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_architecture(self) -> Dict[str, Any]:
        """Get model architecture information."""
        layers = []
        for name, module in self.named_children():
            layers.append({
                'name': name,
                'type': module.__class__.__name__,
                'parameters': sum(p.numel() for p in module.parameters()),
                'shape': str(getattr(module, 'weight', None))
            })

        return {
            'model_id': self.id,
            'class': self.__class__.__name__,
            'config': self.config.to_dict(),
            'total_parameters': self.count_parameters(),
            'device': str(self.device),
            'layers': layers
        }

    def save(self, path: str):
        """Save model to disk."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config.to_dict(),
            'training_history': self.training_history,
            'id': self.id
        }, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str, device: Optional[str] = None):
        """Load model from disk."""
        checkpoint = torch.load(path, map_location=device)
        config = ModelConfig(**checkpoint['config'])
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.training_history = checkpoint['training_history']
        model.id = checkpoint['id']

        if device:
            model.device = torch.device(device)
            model.to(model.device)

        logger.info(f"Model loaded from {path}")
        return model


class ThreatDetectionModel(BaseModel):
    """Advanced threat detection model with multiple architectures."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        layers = []
        prev_dim = config.input_dim

        # Build hidden layers
        for i, hidden_dim in enumerate(config.hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Batch normalization
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # Activation
            if config.activation.lower() == 'relu':
                layers.append(nn.ReLU())
            elif config.activation.lower() == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.1))
            elif config.activation.lower() == 'selu':
                layers.append(nn.SELU())
            else:
                layers.append(nn.Tanh())

            # Dropout
            if config.dropout_rate > 0:
                layers.append(nn.Dropout(config.dropout_rate))

            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, config.output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(x))

    def get_feature_importance(self, X: torch.Tensor) -> Dict[str, float]:
        """Calculate feature importance using gradient-based method."""
        self.eval()
        X = X.to(self.device)
        X.requires_grad = True

        # Forward pass
        output = self.forward(X)

        # Backward pass to get gradients
        output.backward(torch.ones_like(output))

        # Calculate importance as absolute gradient mean
        gradients = X.grad.abs().mean(dim=0).cpu().numpy()

        # Create feature importance dictionary
        importance = {
            f'feature_{i}': float(gradients[i])
            for i in range(len(gradients))
        }

        return importance


class AdvancedThreatDetectionModel(ThreatDetectionModel):
    """Enhanced model with attention mechanism."""

    def __init__(self, config: ModelConfig, attention_dim: int = 32):
        self.attention_dim = attention_dim
        super().__init__(config)

        # Add attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(config.hidden_dims[-1], attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get features from base network
        features = self.network[:-1](x)  # Exclude output layer

        # Apply attention
        attention_weights = self.attention(features)
        weighted_features = (features * attention_weights).sum(dim=1, keepdim=True)

        # Output layer
        output = self.network[-1](weighted_features)

        return torch.sigmoid(output)