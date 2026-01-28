# core/utils/validators.py
import re
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pydantic import BaseModel, ValidationError, validator
from datetime import datetime

from config.settings import settings
from config.constants import DEFAULT_PARAMETERS
from core.utils.logger import get_logger

logger = get_logger(__name__)


class SimulationParameters(BaseModel):
    """Pydantic model for simulation parameter validation."""

    # Required parameters
    surveillance_level: float
    liberty_threshold: float

    # Optional parameters with defaults
    poison_rate: float = 0.0
    noise_level: float = 1.0
    n_samples: int = settings.DEFAULT_N_SAMPLES
    epochs: int = settings.DEFAULT_EPOCHS
    learning_rate: float = settings.DEFAULT_LEARNING_RATE
    batch_size: int = settings.DEFAULT_BATCH_SIZE

    # Model configuration
    hidden_layers: List[int] = settings.DEFAULT_HIDDEN_LAYERS
    dropout_rate: float = settings.DEFAULT_DROPOUT_RATE
    activation: str = settings.DEFAULT_ACTIVATION

    # Simulation metadata
    strategy: str = "security"
    seed: Optional[int] = None
    threshold: float = 0.5

    @validator('surveillance_level')
    def validate_surveillance_level(cls, v):
        if not (settings.MIN_SURVEILLANCE_LEVEL <= v <= settings.MAX_SURVEILLANCE_LEVEL):
            raise ValueError(
                f"Surveillance level must be between {settings.MIN_SURVEILLANCE_LEVEL} and {settings.MAX_SURVEILLANCE_LEVEL}")
        return v

    @validator('liberty_threshold')
    def validate_liberty_threshold(cls, v):
        if not (settings.MIN_LIBERTY_THRESHOLD <= v <= settings.MAX_LIBERTY_THRESHOLD):
            raise ValueError(
                f"Liberty threshold must be between {settings.MIN_LIBERTY_THRESHOLD} and {settings.MAX_LIBERTY_THRESHOLD}")
        return v

    @validator('poison_rate')
    def validate_poison_rate(cls, v):
        if v < 0 or v > 1:
            raise ValueError("Poison rate must be between 0 and 1")
        return v

    @validator('n_samples')
    def validate_n_samples(cls, v):
        if v < 100 or v > 1000000:
            raise ValueError("Sample size must be between 100 and 1,000,000")
        return v

    @validator('hidden_layers')
    def validate_hidden_layers(cls, v):
        if not v:
            raise ValueError("Hidden layers cannot be empty")
        if any(layer <= 0 for layer in v):
            raise ValueError("Hidden layer sizes must be positive")
        if len(v) > 10:
            raise ValueError("Maximum 10 hidden layers allowed")
        return v

    @validator('learning_rate')
    def validate_learning_rate(cls, v):
        if v <= 0 or v > 1:
            raise ValueError("Learning rate must be between 0 and 1")
        return v

    @validator('activation')
    def validate_activation(cls, v):
        valid_activations = ['relu', 'leaky_relu', 'tanh', 'sigmoid', 'selu']
        if v not in valid_activations:
            raise ValueError(f"Activation must be one of: {valid_activations}")
        return v

    @validator('strategy')
    def validate_strategy(cls, v):
        valid_strategies = ['security', 'advanced', 'quick', 'baseline']
        if v not in valid_strategies:
            raise ValueError(f"Strategy must be one of: {valid_strategies}")
        return v


def validate_simulation_params(params: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[Dict]]:
    """
    Validate simulation parameters.

    Returns:
        Tuple of (is_valid, error_message, validated_params)
    """
    try:
        validated = SimulationParameters(**params)
        return True, None, validated.dict()
    except ValidationError as e:
        error_msg = "; ".join([f"{err['loc'][0]}: {err['msg']}" for err in e.errors()])
        return False, error_msg, None


def sanitize_input(input_str: str) -> str:
    """Sanitize user input to prevent injection attacks."""
    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>"\']', '', input_str)
    # Limit length
    return sanitized[:1000]


def validate_file_upload(file_bytes: bytes, max_size_mb: int = 10) -> Tuple[bool, str]:
    """Validate uploaded file."""
    max_bytes = max_size_mb * 1024 * 1024

    if len(file_bytes) > max_bytes:
        return False, f"File size exceeds {max_size_mb}MB limit"

    # Check for null bytes (potential exploit)
    if b'\x00' in file_bytes:
        return False, "File contains null bytes"

    return True, "File is valid"


def validate_json_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate JSON data against a schema."""
    errors = []

    for key, expected_type in schema.items():
        if key not in data:
            errors.append(f"Missing required field: {key}")
        elif not isinstance(data[key], expected_type):
            errors.append(f"Field {key} must be of type {expected_type.__name__}")

    return len(errors) == 0, errors


class ParameterSanitizer:
    """Sanitizer for simulation parameters."""

    @staticmethod
    def sanitize_numeric(value: Any, default: float = 0.0, min_val: float = 0.0, max_val: float = 100.0) -> float:
        """Sanitize numeric parameter."""
        try:
            num = float(value)
            return max(min_val, min(max_val, num))
        except (ValueError, TypeError):
            return default

    @staticmethod
    def sanitize_list(value: Any, default: List = None) -> List:
        """Sanitize list parameter."""
        if default is None:
            default = []

        if isinstance(value, list):
            return value
        elif isinstance(value, (str, int, float)):
            return [value]
        else:
            return default

    @staticmethod
    def sanitize_dict(value: Any, default: Dict = None) -> Dict:
        """Sanitize dictionary parameter."""
        if default is None:
            default = {}

        if isinstance(value, dict):
            return value
        else:
            return default

    @classmethod
    def sanitize_all(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize all parameters."""
        sanitized = {}

        for key, value in params.items():
            if isinstance(value, (int, float)):
                # Apply appropriate bounds based on parameter type
                if key in ['surveillance_level', 'liberty_threshold']:
                    sanitized[key] = cls.sanitize_numeric(value, 0.0, 0.0, 100.0)
                elif key in ['poison_rate', 'dropout_rate', 'threshold']:
                    sanitized[key] = cls.sanitize_numeric(value, 0.0, 0.0, 1.0)
                elif key == 'learning_rate':
                    sanitized[key] = cls.sanitize_numeric(value, 0.01, 0.0001, 0.1)
                elif key == 'noise_level':
                    sanitized[key] = cls.sanitize_numeric(value, 1.0, 0.0, 10.0)
                else:
                    sanitized[key] = value
            elif isinstance(value, list):
                sanitized[key] = cls.sanitize_list(value)
            elif isinstance(value, dict):
                sanitized[key] = cls.sanitize_dict(value)
            else:
                sanitized[key] = value

        return sanitized