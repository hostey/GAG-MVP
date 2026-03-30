"""
utils/config.py — GAGS Resilience Framework v.0
Centralised simulation configuration and application settings.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class SimulationConfig:
    """All tunable simulation constants in one place."""

    # ── Random seed ────────────────────────────────────────────────────────────
    DATA_SEED: int = 42

    # ── Bias parameters ────────────────────────────────────────────────────────
    MAX_BIAS_FACTOR: float = 1.0
    NOISE_STD: float = 0.05
    DECISION_BOUNDARY: float = 0.5
    FEATURE_RANGE: tuple = (0.0, 1.0)

    # ── Bias type registry ─────────────────────────────────────────────────────
    BIAS_TYPES: List[str] = field(default_factory=lambda: [
        "demographic",
        "historical",
        "socioeconomic",
        "geographic",
        "gender",
        "selection",
        "representation",
        "measurement",
        "label",
        "linguistic",
        "temporal",
    ])

    # ── Attack / poisoning types ───────────────────────────────────────────────
    ATTACK_TYPES: Dict[str, Any] = field(default_factory=lambda: {
        "data_poisoning": {
            "default": 0.05,
            "max": 0.50,
            "step": 0.01,
        },
        "label_flipping": {
            "default": 0.05,
            "max": 0.40,
            "step": 0.01,
        },
        "feature_noise": {
            "default": 0.10,
            "max": 0.50,
            "step": 0.05,
        },
        "backdoor": {
            "default": 0.03,
            "max": 0.20,
            "step": 0.01,
        },
    })


@dataclass
class AppSettings:
    """Application-level UI and behaviour settings."""
    DEFAULT_N_SAMPLES: int = 5_000
    DEFAULT_N_RUNS: int = 3
    MAX_N_SAMPLES: int = 50_000
    MIN_N_SAMPLES: int = 500
    CACHE_TTL_SECONDS: int = 3_600
    MAX_UPLOAD_MB: int = 50
    DEFAULT_BIAS_INTENSITY: float = 0.30
    SHOW_ADVANCED: bool = False
    EXPORT_FORMATS: List[str] = field(default_factory=lambda: ["CSV", "JSON", "PDF"])


# ── Singletons (imported everywhere as `simulation_config` and `settings`) ─────
simulation_config = SimulationConfig()
settings = AppSettings()
