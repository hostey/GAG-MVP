# core/utils/logger.py
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
import json
from datetime import datetime
from typing import Optional, Dict, Any

from config.settings import settings


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_object = {
            'timestamp': datetime.now().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
            'environment': settings.ENVIRONMENT
        }

        if record.exc_info:
            log_object['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_object)


class ColoredFormatter(logging.Formatter):
    """Colorized console formatter."""

    COLORS = {
        'DEBUG': '\033[36m',  # Cyan
        'INFO': '\033[32m',  # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',  # Red
        'CRITICAL': '\033[41m',  # Red background
        'RESET': '\033[0m'
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])

        if settings.is_production:
            return super().format(record)
        else:
            return f"{color}[{record.levelname}] {record.name}: {record.getMessage()}{self.COLORS['RESET']}"


def setup_logging():
    """Setup comprehensive logging configuration."""
    # Ensure logs directory exists
    settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler for development
    if not settings.is_production:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(ColoredFormatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        root_logger.addHandler(console_handler)

    # File handler for all environments
    file_handler = RotatingFileHandler(
        settings.LOGS_DIR / 'gags.log',
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    root_logger.addHandler(file_handler)

    # JSON file handler for production
    if settings.is_production:
        json_handler = RotatingFileHandler(
            settings.LOGS_DIR / 'gags.json.log',
            maxBytes=10 * 1024 * 1024,
            backupCount=3
        )
        json_handler.setLevel(logging.INFO)
        json_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(json_handler)

    # Error file handler
    error_handler = RotatingFileHandler(
        settings.LOGS_DIR / 'gags.error.log',
        maxBytes=5 * 1024 * 1024,
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(error_handler)

    # Suppress noisy library logs
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('streamlit').setLevel(logging.WARNING)


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Get a configured logger instance."""
    logger = logging.getLogger(name)

    if not logger.handlers:  # Avoid duplicate handlers
        if level:
            logger.setLevel(getattr(logging, level.upper()))

    return logger


class SimulationLogger:
    """Specialized logger for simulation operations."""

    def __init__(self, simulation_id: str):
        self.simulation_id = simulation_id
        self.logger = get_logger(f"simulation.{simulation_id}")
        self.metrics: Dict[str, Any] = {}

    def start(self):
        """Log simulation start."""
        self.logger.info(f"Simulation {self.simulation_id} started")

    def progress(self, stage: str, progress: float):
        """Log simulation progress."""
        self.logger.debug(f"Stage {stage}: {progress:.1%} complete")

    def metric(self, name: str, value: Any):
        """Log a metric."""
        self.metrics[name] = value
        self.logger.info(f"Metric {name}: {value}")

    def warning(self, message: str):
        """Log a warning."""
        self.logger.warning(f"Warning: {message}")

    def error(self, message: str, exc_info: bool = False):
        """Log an error."""
        self.logger.error(f"Error: {message}", exc_info=exc_info)

    def complete(self, success: bool = True):
        """Log simulation completion."""
        status = "completed successfully" if success else "failed"
        self.logger.info(f"Simulation {self.simulation_id} {status}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get all logged metrics."""
        return self.metrics.copy()


# Initialize logging when module is imported
setup_logging()