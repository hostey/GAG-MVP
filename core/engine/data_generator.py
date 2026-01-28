import torch
import numpy as np
from typing import Tuple


class AdvancedDataGenerator:
    """Generates synthetic security data for simulation."""

    def generate(
            self,
            n_samples: int = 1000,
            noise_level: float = 1.0,
            bias_factor: float = 0.0,
            split_ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. Create features (10-dimensional)
        X = np.random.randn(n_samples, 10).astype(np.float32)

        # 2. Create labels (1 for threat, 0 for clean)
        y = (np.sum(X[:, :3], axis=1) > 0).astype(np.float32)  # Use float32 here

        # 3. Apply Noise
        noise = np.random.normal(0, noise_level, X.shape).astype(np.float32)
        X = X + noise

        # 4. Convert to Tensors
        # BCELoss requires FloatTensor for both X and y
        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y)

        # 5. Split data
        train_idx = int(n_samples * split_ratios[0])
        val_idx = train_idx + int(n_samples * split_ratios[1])

        return (
            X_tensor[:train_idx], X_tensor[train_idx:val_idx], X_tensor[val_idx:],
            y_tensor[:train_idx], y_tensor[train_idx:val_idx], y_tensor[val_idx:]
        )