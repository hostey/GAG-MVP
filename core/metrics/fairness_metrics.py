import torch
import numpy as np
from typing import Dict, Any


class FairnessMetricsCalculator:
    """Calculates bias and fairness scores for the model."""

    def calculate(self, model: Any, X: torch.Tensor, y: torch.Tensor, parameters: Dict[str, Any]) -> Dict[str, float]:
        model.eval()
        with torch.no_grad():
            preds = model.predict(X)

        # Mocking a "protected attribute" (e.g., user group 0 or 1)
        # In a real app, this would be a specific column in your data
        protected_attr = (X[:, 0] > 0).int()

        # Calculate Demographic Parity (Is selection rate similar?)
        selection_rate_group1 = (preds[protected_attr == 1] == 1).float().mean().item()
        selection_rate_group0 = (preds[protected_attr == 0] == 1).float().mean().item()

        # Parity = 1.0 - absolute difference (closer to 1.0 is better)
        parity_score = 1.0 - abs(selection_rate_group1 - selection_rate_group0)

        # Factor in the 'liberty_threshold' from the dashboard to adjust the score
        liberty_weight = parameters.get('liberty_threshold', 70) / 100
        overall_fairness = (parity_score * 0.7) + (liberty_weight * 0.3)

        return {
            "overall_fairness": max(0, min(1, overall_fairness)),
            "demographic_parity": parity_score,
            "group_1_selection": selection_rate_group1,
            "group_0_selection": selection_rate_group0
        }