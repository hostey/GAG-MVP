# interfaces/streamlit_components/__init__.py
"""
Streamlit UI Components for GAGS Dashboard.
"""

from .dashboard import SecurityDashboard
from .visualizations import (
    create_metric_cards,
    create_performance_radar,
    create_trend_chart,
    create_confusion_matrix,
    create_feature_importance
)
from .widgets import (
    create_parameter_slider,
    create_model_selector,
    create_strategy_selector,
    create_export_buttons
)

__all__ = [
    'SecurityDashboard',
    'create_metric_cards',
    'create_performance_radar',
    'create_trend_chart',
    'create_confusion_matrix',
    'create_feature_importance',
    'create_parameter_slider',
    'create_model_selector',
    'create_strategy_selector',
    'create_export_buttons'
]