import torch
import torch.nn as nn
from typing import List
from config.settings import settings


class ResilientNetwork(nn.Module):
    """Production-grade MLP for simulation modeling."""

    def __init__(self, input_dim: int, output_dim: int = 1):
        super().__init__()

        layers = []
        current_dim = input_dim
        hidden_dims = settings.DEFAULT_HIDDEN_LAYERS

        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))  # Essential for production stability
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=settings.DEFAULT_DROPOUT_RATE))
            current_dim = h_dim

        layers.append(nn.Linear(current_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)