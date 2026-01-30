# utils/config.py
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    """Application settings with environment variable support."""

    # App Metadata
    APP_NAME: str = "GAGS Resilience Framework"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")

    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent  # Adjust for utils/ nesting
    DATA_DIR: Path = BASE_DIR / "data"
    MODELS_DIR: Path = BASE_DIR / "models"
    LOGS_DIR: Path = BASE_DIR / "logs"
    EXPORTS_DIR: Path = BASE_DIR / "exports"

    # Simulation Defaults
    DEFAULT_N_SAMPLES: int = Field(default=10000, ge=1000, le=100000)
    DEFAULT_EPOCHS: int = Field(default=100, ge=10, le=1000)
    DEFAULT_LEARNING_RATE: float = Field(default=0.01, ge=0.0001, le=0.1)
    DEFAULT_BATCH_SIZE: int = Field(default=32, ge=16, le=1024)

    # Model Architecture
    DEFAULT_HIDDEN_LAYERS: List[int] = Field(default=[32, 16, 8])
    DEFAULT_DROPOUT_RATE: float = Field(default=0.2, ge=0.0, le=0.5)
    DEFAULT_ACTIVATION: str = Field(default="relu")

    # Performance
    CACHE_ENABLED: bool = Field(default=True)
    CACHE_TTL: int = Field(default=3600)  # 1 hour
    MAX_WORKERS: int = Field(default=os.cpu_count() or 4)

    # Security
    ENCRYPTION_KEY: Optional[str] = Field(default=None, env="ENCRYPTION_KEY")
    ALLOWED_ORIGINS: List[str] = Field(default=["*"])

    # Monitoring
    ENABLE_TELEMETRY: bool = Field(default=False)
    LOG_LEVEL: str = Field(default="INFO")
    SENTRY_DSN: Optional[str] = Field(default=None, env="SENTRY_DSN")

    # Export Formats
    SUPPORTED_EXPORT_FORMATS: List[str] = Field(
        default=["json", "csv", "parquet", "pickle", "onnx"]
    )

    # Validation
    MAX_SIMULATION_RUNS: int = Field(default=1000, description="Maximum runs per session")
    MIN_SURVEILLANCE_LEVEL: int = 0
    MAX_SURVEILLANCE_LEVEL: int = 100
    MIN_LIBERTY_THRESHOLD: int = 0
    MAX_LIBERTY_THRESHOLD: int = 100

    # Governance References
    ISO_STANDARD: str = "ISO/IEC 42001:2023"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @field_validator("DATA_DIR", "MODELS_DIR", "LOGS_DIR", "EXPORTS_DIR")
    def create_directories(cls, v: Path) -> Path:
        """Ensure directories exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT == "development"

    @property
    def is_testing(self) -> bool:
        return self.ENVIRONMENT == "testing"


class SimulationConfig(BaseSettings):
    """Simulation-specific configuration."""

    # Data Generation
    DATA_SEED: int = 42
    NOISE_STD: float = 1.0
    FEATURE_RANGE: tuple = (-10.0, 10.0)
    DECISION_BOUNDARY: float = 0.0

    # Bias Configuration
    MAX_BIAS_FACTOR: float = 1.0
    BIAS_TYPES: List[str] = [
        # Core / Protected Attribute Biases (common across sectors)
        "demographic",  # e.g., race, gender, age, ethnicity – classic protected attributes
        "socioeconomic",  # income, education, occupation – often proxies for class/wealth
        "geographic",  # renamed from "location" for clarity; urban/rural, regional access
        "temporal",  # time-based drift, historical patterns, seasonal effects

        # Data & Representation Biases (key in training/sim data generation)
        "historical",  # inherits past societal inequalities (e.g., under-treatment in healthcare records)
        "selection",  # non-representative sampling (e.g., over-representation of urban patients)
        "representation",  # under-representation of groups in datasets
        "measurement",  # flawed proxies/labels (e.g., healthcare cost as proxy for need, ignoring access barriers)

        # Algorithm & Process Biases (during model training/inference)
        "algorithmic",  # emerges from model design, optimization, or feature weighting
        "confirmation",  # reinforces pre-existing assumptions/patterns in data
        "automation",  # over-reliance on AI outputs (human-in-loop bias in sims)

        # Sector-Relevant / Emerging Biases (tailored to GAGS pillars)
        "behavioral",  # user interaction patterns (e.g., surveillance evasion in security)
        "cultural",  # language, customs, norms (esp. relevant in Global South/agrotech contexts)
        "intersectional",  # compounded effects (e.g., Black women in healthcare diagnostics)
        "accessibility",  # digital divide, tech literacy (critical in low-resource agrotech/security)
    ]

    # Attack Configuration
    ATTACK_TYPES: Dict[str, Dict] = {
        "data_poisoning": {
            "min_rate": 0.0,
            "max_rate": 0.5,
            "default": 0.1
        },
        "evasion": {
            "epsilon": 0.1,
            "iterations": 10
        },
        "model_extraction": {
            "queries": 1000
        }
    }

    # Performance Targets
    ACCURACY_TARGET: float = 0.85
    FAIRNESS_TARGET: float = 0.90
    LATENCY_TARGET_MS: float = 100.0

    class Config:
        frozen = True  # Immutable configuration


# Singleton instances
settings = AppSettings()
simulation_config = SimulationConfig()