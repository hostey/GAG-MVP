# config/constants.py
from enum import Enum, StrEnum
from typing import Dict, Any, Final

# config/constants.py
PILLARS = ["Healthcare Equity", "National Security", "Sustainable Agrotech"]
ISO_STANDARDS = "ISO/IEC 42001:2023"
NIGERIA_EGOV_BILL_REF = "HB. 15.02.2026"

class SimulationStatus(StrEnum):
    """Simulation status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ThreatLevel(Enum):
    """Threat level classification."""
    LOW = (1, "🟢", "Low risk")
    MEDIUM = (2, "🟡", "Medium risk")
    HIGH = (3, "🟠", "High risk")
    CRITICAL = (4, "🔴", "Critical risk")

    def __init__(self, value: int, icon: str, description: str):
        self._value_ = value
        self.icon = icon
        self.description = description


class MetricCategory(StrEnum):
    """Metric categories for classification."""
    SAFETY = "safety"
    LIBERTY = "liberty"
    PERFORMANCE = "performance"
    FAIRNESS = "fairness"
    RESILIENCE = "resilience"
    EXPLAINABILITY = "explainability"


# Color schemes
COLOR_SCHEMES: Final[Dict[str, Dict]] = {
    "security": {
        "primary": "#1E88E5",
        "secondary": "#1565C0",
        "accent": "#0D47A1",
        "success": "#4CAF50",
        "warning": "#FFC107",
        "danger": "#F44336"
    },
    "healthcare": {
        "primary": "#1976D2",
        "secondary": "#0D47A1",
        "accent": "#64B5F6",
        "success": "#388E3C",
        "warning": "#FF9800",
        "danger": "#D32F2F"
    }
}

# Performance thresholds
THRESHOLDS: Final[Dict[str, Dict]] = {
    "accuracy": {"excellent": 0.95, "good": 0.85, "fair": 0.70, "poor": 0.60},
    "fairness": {"excellent": 0.95, "good": 0.85, "fair": 0.75, "poor": 0.65},
    "latency": {"excellent": 50, "good": 100, "fair": 200, "poor": 500},
    "reliability": {"excellent": 0.999, "good": 0.99, "fair": 0.95, "poor": 0.90}
}

# Simulation parameters
DEFAULT_PARAMETERS: Final[Dict[str, Any]] = {
    "surveillance_level": {"min": 0, "max": 100, "default": 50, "step": 1},
    "liberty_threshold": {"min": 0, "max": 100, "default": 70, "step": 1},
    "poison_rate": {"min": 0.0, "max": 0.5, "default": 0.1, "step": 0.01},
    "n_samples": {"min": 1000, "max": 100000, "default": 10000, "step": 1000},
    "noise_level": {"min": 0.0, "max": 2.0, "default": 1.0, "step": 0.1}
}