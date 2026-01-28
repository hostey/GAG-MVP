# interfaces/__init__.py
"""
GAGS Interfaces Layer - UI components and API endpoints.
"""

from interfaces.streamlit_components.dashboard import SecurityDashboard
from interfaces.streamlit_components.visualizations import (
    create_metric_cards,
    create_performance_radar,
    create_trend_chart
)
from interfaces.streamlit_components.widgets import (
    create_parameter_slider,
    create_model_selector
)

__all__ = [
    'SecurityDashboard',
    'create_metric_cards',
    'create_performance_radar',
    'create_trend_chart',
    'create_parameter_slider',
    'create_model_selector'
]